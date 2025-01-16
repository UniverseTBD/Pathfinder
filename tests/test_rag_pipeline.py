# tests/test_rag_pipeline.py

from unittest.mock import patch

import pandas as pd
import pytest

from src.pipeline.rag_pipeline import run_rag_qa


@pytest.fixture
def papers_df():
    """Provide a minimal DataFrame that matches the columns used by run_rag_qa."""
    return pd.DataFrame(
        {
            "title": ["Paper A", "Paper B"],
            "abstract": ["Abstract A text", "Abstract B text"],
            "ads_id": ["ads1234", "ads5678"],
        }
    )


@patch("langchain_core.runnables.base.RunnableParallel.invoke")
def test_run_rag_qa_basic(mock_chain_invoke, papers_df):
    """
    Test run_rag_qa with a normal 'papers_df' and no question_type => fallback to 'Regular'.
    We patch the final chain invocation to return a fake result.
    """
    # 1) Setup mock chain invocation
    mock_chain_invoke.return_value = {
        "answer": "Mock LLM answer from chain",
        "context": [],
    }

    # 2) Call the function under test
    result = run_rag_qa(query="Test query", papers_df=papers_df)

    # 3) Validate
    assert isinstance(result, dict), "Expected a dict"
    assert "answer" in result, "Should contain 'answer'"
    assert "chunks_used" in result, "Should contain 'chunks_used'"
    assert (
        result["answer"] == "Mock LLM answer from chain"
    ), "Used the mocked chain result"
    # Ensure we actually chunked: we expect some docs in 'chunks_used'
    assert len(result["chunks_used"]) > 0


@patch("langchain_core.runnables.base.RunnableParallel.invoke")
def test_run_rag_qa_bibliometric(mock_chain_invoke, papers_df):
    """
    If question_type='Bibliometric', check we still get a result from the chain.
    """
    mock_chain_invoke.return_value = {
        "answer": "Bibliometric LLM answer",
        "context": [],
    }
    result = run_rag_qa(
        query="How many references in NASA ADS?",
        papers_df=papers_df,
        question_type="Bibliometric",
    )
    assert result["answer"] == "Bibliometric LLM answer"
    assert len(result["chunks_used"]) > 0


@patch("langchain_core.runnables.base.RunnableParallel.invoke")
def test_run_rag_qa_empty_df(mock_chain_invoke):
    """
    If we pass an empty DataFrame, the pipeline should short-circuit.
    """
    empty_df = pd.DataFrame()
    result = run_rag_qa(query="No docs query", papers_df=empty_df)
    # Should not call the chain at all
    mock_chain_invoke.assert_not_called()
    assert result["answer"] == "No documents found for this query."
    assert result["chunks_used"] == []
