# tests/test_retrieval_system.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.retrieval.retrieval_system import RetrievalSystem


@pytest.fixture
def mock_arxiv_dataset():
    """
    Create a mock object that emulates a Hugging Face Dataset with a FAISS index.
    We'll pretend it has a 'search' method returning fixed indices & scores.
    """
    mock_dataset = MagicMock()

    # Suppose the dataset has length 5, and for each index
    #   we mock 'abstract', 'keywords', 'date', 'cites', etc.
    data = {
        "abstract": ["paper abstract " + str(i) for i in range(5)],
        "keywords": [["kw1", "kw2"], ["kw1"], [], ["kw2"], ["kw3"]],
        "date": pd.to_datetime(
            ["2020-01-01", "2019-06-15", "2021-12-31", "2015-07-23", "2023-01-01"]
        ),
        "cites": [10, 100, 5, 500, 250],
        "other_columns": ["val" + str(i) for i in range(5)],
    }

    def mock_getitem(indices):
        # If indices is a list or np.ndarray, return subset
        if isinstance(indices, (list, np.ndarray)):
            return {k: [v[i] for i in indices] for k, v in data.items()}
        else:
            # single index
            return {k: data[k][indices] for k in data}

    # The dataset object should behave like a dictionary when sliced with a list of indices:
    mock_dataset.__getitem__.side_effect = mock_getitem

    # We'll also define a 'search' method that returns a result with .indices, .scores
    mock_search_result = MagicMock()
    # Suppose it returns 3 indices, with "scores" being distances
    mock_search_result.indices = np.array([0, 1, 3])
    mock_search_result.scores = np.array(
        [1.0, 2.0, 5.0]
    )  # Distances, smaller is better
    mock_dataset.search.return_value = mock_search_result

    # The add_faiss_index call can be a no-op
    mock_dataset.add_faiss_index = MagicMock()

    return mock_dataset


@pytest.fixture
def mock_cohere_client():
    """
    Mock a Cohere client that can do rerank. We'll pretend it returns a static ordering.
    """
    mock_cohere = MagicMock()
    mock_rerank_result = MagicMock()
    # Suppose top_n=2 => it returns these sorted indices
    mock_rerank_result.results = [
        MagicMock(index=1, relevance_score=0.9),
        MagicMock(index=0, relevance_score=0.7),
        MagicMock(index=2, relevance_score=0.6),
    ]
    mock_cohere.rerank.return_value = mock_rerank_result
    return mock_cohere


@pytest.fixture
def retrieval_system(mock_arxiv_dataset, mock_cohere_client):
    """
    Instantiate RetrievalSystem with the dataset and cohere client mocked.
    Also mock out the OpenAI/Azure calls in get_openai_embeddings and get_openai_chat_llm.
    """
    with patch(
        "src.retrieval.retrieval_system.load_dataset", return_value=mock_arxiv_dataset
    ):
        with patch(
            "src.retrieval.retrieval_system.get_openai_embeddings"
        ) as mock_embed_model:
            with patch(
                "src.retrieval.retrieval_system.get_openai_chat_llm"
            ) as mock_llm:
                # Mock the embedding model
                mock_embed_instance = MagicMock()
                mock_embed_instance.embed_query.side_effect = lambda text: np.ones(
                    768, dtype=np.float32
                )
                mock_embed_model.return_value = mock_embed_instance

                # Mock the LLM (for HyDE or other)
                mock_llm_instance = MagicMock()
                mock_llm_instance.predict.side_effect = (
                    lambda prompt: "This is a hypothetical doc"
                )
                mock_llm.return_value = mock_llm_instance

                # Create system
                system = RetrievalSystem()

                # Replace real cohere client with our mock
                system.cohere_client = mock_cohere_client

                yield system


def test_retrieve_basic(retrieval_system):
    """
    Test that retrieve() returns a DataFrame of top_k results.
    We'll do a basic check with no toggles, no HyDE, no re-rank.
    """
    df = retrieval_system.retrieve(
        query="test query", top_k=2, toggles=[], use_hyde=False, use_rerank=False
    )

    # We expect a 2-row DataFrame (head(2)).
    assert len(df) == 2, f"Expected top_k=2 results, got {len(df)}"

    # Check columns are present
    assert "abstract" in df.columns
    assert "final_score" in df.columns
    # By default, "final_score" is similarity * 1.0 for each weight
    # Just ensure that it's not None
    assert not df["final_score"].isnull().any()


def test_compute_keyword_weight_no_toggles(retrieval_system):
    """
    Test that if no toggles are set, kw_weight=1.0 for all rows.
    """
    # We'll directly call retrieve() with toggles=[] to avoid weighting
    df = retrieval_system.retrieve(
        query="some random query", top_k=3, toggles=[], use_hyde=False, use_rerank=False
    )
    # Verify the kw_weight (if present) is all 1.0
    if "kw_weight" in df.columns:
        assert all(
            df["kw_weight"] == 1.0
        ), "Keyword weighting is not 1.0 when toggles=[]"


def test_use_hyde_generates_embedding(retrieval_system):
    """
    Test that when use_hyde=True, we actually call the LLM and embed the hypothetical doc + query.
    """
    with patch.object(
        retrieval_system,
        "_get_hyde_embedding",
        wraps=retrieval_system._get_hyde_embedding,
    ) as mock_hyde:
        retrieval_system.retrieve(
            query="How do galaxies form?",
            top_k=2,
            toggles=[],
            use_hyde=True,
            hyde_temperature=0.7,
        )
        # Check the _get_hyde_embedding was called
        mock_hyde.assert_called_once()


def test_rerank_cohere(retrieval_system, mock_cohere_client):
    """
    Test that use_rerank=True calls the cohere rerank logic and
    returns a DataFrame sorted by cohere's results.
    """
    df = retrieval_system.retrieve(
        query="galaxy evolution",
        top_k=2,
        toggles=[],
        use_hyde=False,
        use_rerank=True,
        rerank_top_k=3,
    )
    # The mock_cohere_client returns results with sorted indices = [1, 0, 2].
    # So the final top-2 after rerank should come from original row indices [1, 0]
    # i.e. from the partial_df.head(3) we feed to cohere, we reorder by [1, 0, 2].
    # So row index=1 should appear at the top of our returned df.
    # Let's check that:
    final_indices = df["indices"].tolist()
    assert len(final_indices) == 2
    # The first index should be 1, second should be 0, matching the mock rerank ordering
    assert final_indices[0] == 1, "Cohere rerank not applied properly"
    assert final_indices[1] == 0, "Cohere rerank not applied properly"
