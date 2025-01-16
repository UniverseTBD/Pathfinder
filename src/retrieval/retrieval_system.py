# src/retrieval/retrieval_system.py
import os
from datetime import datetime
from typing import List, Optional

import cohere
import numpy as np
import pandas as pd
from datasets import load_dataset

# Local imports
from src.nlp_utils import get_keywords, load_nlp
from src.providers import get_openai_chat_llm, get_openai_embeddings

nlp = load_nlp()


class RetrievalSystem:
    """
    Encapsulate logic for retrieving documents from a HuggingFace dataset with a FAISS index
    and optionally re-ranking the results with Cohere or using a 'HyDE' embedding approach.

    Parameters
    ----------
    dataset_name : str, optional
        The HuggingFace dataset path or identifier to load (default: "kiyer/pathfinder_arxiv_data").
    split : str, optional
        The dataset split to load (default: "train").
    column_name : str, optional
        The name of the column that contains the precomputed embeddings for FAISS (default: "embed").
    """

    def __init__(
        self,
        dataset_name: str = "kiyer/pathfinder_arxiv_data",
        split: str = "train",
        column_name: str = "embed",
    ):
        """
        Initialize the RetrievalSystem by loading a FAISS-indexed dataset,
        setting up local weighting toggles, and preparing LLM/embedding clients.
        """
        # 1) Load the dataset (only once)
        self.dataset_name = dataset_name
        self.split = split
        self.column_name = column_name

        # Load the dataset from Hugging Face
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.dataset.add_faiss_index(column=self.column_name)
        print(
            f"Loaded dataset '{self.dataset_name}' (split='{self.split}') with FAISS index on column='{self.column_name}'."
        )

        # 2) Create optional external clients (Cohere, OpenAI, etc.)
        self.cohere_key = os.environ.get("cohere_key", "")
        self.cohere_client = cohere.Client(self.cohere_key) if self.cohere_key else None

        # 3) Load embedding model & chat LLM
        self.embedding_model = get_openai_embeddings()
        self.llm = get_openai_chat_llm()

        # 4) Weight toggles
        self.weight_keywords = False
        self.weight_date = False
        self.weight_citation = False

        # For storing user-provided keywords
        self.query_input_keywords: List[str] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        toggles: Optional[List[str]] = None,
        use_hyde: bool = False,
        use_rerank: bool = False,
        hyde_temperature: float = 0.5,
        rerank_top_k: int = 250,
    ) -> pd.DataFrame:
        """
        Retrieve documents relevant to `query` from the FAISS-indexed dataset.

        Steps:
        1. (Optional) Generate a hypothetical document embedding (HyDE).
        2. Perform FAISS search on the dataset to get `rerank_top_k` results.
        3. Apply weighting for keywords, date, citations (based on `toggles`).
        4. (Optional) Rerank these results with Cohere, returning top_k.

        Parameters
        ----------
        query : str
            The input query for retrieval.
        top_k : int, optional
            Final number of documents to return (default=10).
        toggles : List[str], optional
            Which weighting toggles to apply (e.g. ["Keywords", "Time", "Citations"]).
        use_hyde : bool, optional
            Whether to use hypothetical document generation + embedding (HyDE).
        use_rerank : bool, optional
            Whether to re-rank the retrieved subset with Cohere.
        hyde_temperature : float, optional
            Temperature to use if `use_hyde` is True (default=0.5).
        rerank_top_k : int, optional
            The number of documents to retrieve from FAISS before re-ranking (default=250).

        Returns
        -------
        pd.DataFrame
            A DataFrame of the top_k results, sorted by final scoring or re-ranking.
            Includes columns like "abstract", "keywords", "date", "cites", plus "final_score" and "indices".
        """
        # 1) Parse toggles
        toggles = toggles or []
        self.weight_keywords = "Keywords" in toggles
        self.weight_date = "Time" in toggles
        self.weight_citation = "Citations" in toggles

        # 2) Extract any additional user keywords
        self.query_input_keywords.extend(get_keywords(query, nlp=nlp))

        # 3) Generate the query embedding (HyDE or normal)
        if use_hyde:
            query_embedding = self._get_hyde_embedding(
                query, temperature=hyde_temperature
            )
        else:
            query_embedding = self.embedding_model.embed_query(query)

        # 4) Search the dataset's FAISS index
        search_results = self.dataset.search(
            self.column_name, np.array(query_embedding), k=rerank_top_k
        )
        indices, scores = search_results.indices, search_results.scores
        # Note: scores may be distances, so smaller is better. We'll invert them if needed.

        # 5) Postprocess & Weight, then optional Cohere reranking
        df = self._postprocess_and_weight(indices, scores, query, top_k, use_rerank)
        return df

    def _postprocess_and_weight(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        query: str,
        top_k: int,
        use_rerank: bool,
    ) -> pd.DataFrame:
        """
        Combine custom weighting (keywords, date, citations) + optional Cohere re-rank.
        Return the final top_k results as a DataFrame.

        Parameters
        ----------
        indices : np.ndarray
            Indices returned by the FAISS search.
        scores : np.ndarray
            Corresponding distances or scores from FAISS.
        query : str
            The query string (used for reranking or debugging).
        top_k : int
            The final number of results we want.
        use_rerank : bool
            Whether to re-rank the top subset with Cohere.

        Returns
        -------
        pd.DataFrame
            A DataFrame of the top_k results after weighting and optional reranking.
        """
        # Convert distance -> similarity if needed (example: 1 / (1 + distance))
        partial_df = pd.DataFrame(self.dataset[indices])
        partial_df["similarity"] = 1.0 / (1.0 + scores)  # example transform
        partial_df["indices"] = indices

        # Weight by toggles
        if self.weight_keywords:
            partial_df["kw_weight"] = self._compute_keyword_weight(partial_df)
        else:
            partial_df["kw_weight"] = 1.0

        if self.weight_date:
            partial_df["date_weight"] = self._compute_date_weight(partial_df)
        else:
            partial_df["date_weight"] = 1.0

        if self.weight_citation:
            partial_df["cite_weight"] = self._compute_citation_weight(partial_df)
        else:
            partial_df["cite_weight"] = 1.0

        # Combine into a final score
        partial_df["final_score"] = (
            partial_df["similarity"]
            * partial_df["kw_weight"]
            * partial_df["date_weight"]
            * partial_df["cite_weight"]
        )

        # Sort by final_score descending
        partial_df.sort_values("final_score", ascending=False, inplace=True)

        # If rerank with Cohere:
        if use_rerank and self.cohere_client:
            # We'll take the top N documents before rerank
            top_df = partial_df.head(250).copy()
            docs_for_rerank = top_df["abstract"].tolist()

            reranked = self.cohere_client.rerank(
                query=query,
                documents=docs_for_rerank,
                model="rerank-english-v3.0",
                top_n=top_k,
            )
            final_indices = [r.index for r in reranked.results]

            # Rebuild the DataFrame in the new order
            top_df = top_df.iloc[final_indices]
            # Possibly truncate to top_k again (some Cohere versions might return more).
            top_df = top_df.head(top_k)

            return top_df.reset_index(drop=True)

        # Otherwise, just return the top_k
        return partial_df.head(top_k).reset_index(drop=True)

    def _get_hyde_embedding(self, query: str, temperature: float) -> np.ndarray:
        """
        Generate a hypothetical document (HyDE) that might answer `query`,
        embed both the hypothetical doc and the original query,
        and return the average of those embeddings.

        Parameters
        ----------
        query : str
            The user query or question.
        temperature : float
            The LLM sampling temperature to encourage creativity in generating the doc.

        Returns
        -------
        np.ndarray
            An embedding vector representing the average of the hypothetical doc + query embeddings.
        """
        hyde_llm = get_openai_chat_llm(temperature=temperature)
        prompt = f"Generate a short abstract that might answer: {query}"
        doc = hyde_llm.predict(prompt)

        doc_embedding = self.embedding_model.embed_query(doc)
        query_embedding = self.embedding_model.embed_query(query)
        return np.mean([doc_embedding, query_embedding], axis=0)

    def _compute_keyword_weight(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute a keyword-based weight for each row in `df`,
        comparing the row's 'keywords' to self.query_input_keywords.

        Increments weight by 0.1 for each overlap.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing a 'keywords' column.

        Returns
        -------
        np.ndarray
            An array of per-row weights in [0,1].
        """
        weight = np.ones(len(df), dtype=float)
        for i, row in df.iterrows():
            row_keywords = row.get("keywords", [])
            overlap = len(set(row_keywords) & set(self.query_input_keywords))
            weight[i] += 0.1 * overlap
        weight /= max(weight.max(), 1e-8)
        return weight

    def _compute_date_weight(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute a recency-based weight for each row in `df`,
        preferring newer papers.

        weight[i] = 1 / (1 + age_in_years)
        Then normalize across all rows to [0, 1].

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing a 'date' column (datetime or easily parseable).

        Returns
        -------
        np.ndarray
            An array of per-row weights in [0,1].
        """
        now = datetime.utcnow()
        weight_vals = []
        for _, row in df.iterrows():
            paper_date = row.get("date", None)
            if paper_date is None:
                # If there's no date, we might default to a neutral weight of 0.5 or 1.0
                weight_vals.append(0.5)
                continue
            # Convert to datetime if needed
            # If it's already a datetime, just subtract
            if not isinstance(paper_date, datetime):
                paper_date = pd.to_datetime(paper_date)
            age_years = (now.date() - paper_date.date()).days / 365.0
            w = 1.0 / (1.0 + age_years)
            weight_vals.append(w)
        weight = np.array(weight_vals, dtype=float)
        weight /= weight.max() + 1e-8
        return weight

    def _compute_citation_weight(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute a citation-based weight for each row in `df`,
        saturating large citation counts around 300 using a logistic function.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing a 'cites' column with citation counts.

        Returns
        -------
        np.ndarray
            An array of per-row weights in [0,1].
        """
        weight_vals = []
        for _, row in df.iterrows():
            c = row.get("cites", 0)
            c = min(c, 300)  # saturate
            # logistic function around 50
            w = 1.0 / (1.0 + np.exp(-0.01 * (c - 50)))
            weight_vals.append(w)

        weight = np.array(weight_vals, dtype=float)
        weight /= weight.max() + 1e-8
        return weight
