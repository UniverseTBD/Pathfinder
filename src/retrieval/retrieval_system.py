# src/retrieval/retrieval_system.py

import os
from datetime import datetime
from typing import List, Optional

import cohere
import numpy as np
import pandas as pd
from datasets import load_dataset

# Local imports
from src.config import config
from src.nlp_utils import get_keywords, load_nlp
from src.prompts import hyde_prompt  # or however you name it
from src.providers import get_openai_chat_llm, get_openai_embeddings

nlp = load_nlp()


class RetrievalSystem:
    """
    Encapsulate logic for retrieving documents from a HuggingFace dataset with a FAISS index
    and optionally re-ranking the results with Cohere or using a 'HyDE' embedding approach
    that mimics the original code's logic.
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
        # 1) Load the dataset once
        self.dataset_name = dataset_name
        self.split = split
        self.column_name = column_name

        # Load from Hugging Face
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.dataset.add_faiss_index(column=self.column_name)
        print(
            f"Loaded dataset '{self.dataset_name}' (split='{self.split}') "
            f"with FAISS index on column='{self.column_name}'."
        )

        # 2) External clients (Cohere, OpenAI, etc.)
        # Get Cohere API key from config or environment variable as fallback
        self.cohere_key = config.get("cohere_api_key", os.environ.get("cohere_key", ""))
        
        # Initialize Cohere client if key is available
        if self.cohere_key:
            print("Initialized Cohere client for reranking")
            self.cohere_client = cohere.Client(self.cohere_key)
        else:
            print("No Cohere API key found, reranking will not be available")
            self.cohere_client = None

        # 3) Embeddings & Chat LLM
        self.embedding_model = get_openai_embeddings()
        self.llm = get_openai_chat_llm()

        # 4) Weight toggles
        self.weight_keywords = False
        self.weight_date = False
        self.weight_citation = False

        # For storing user-provided or additional keywords
        self.query_input_keywords: List[str] = []

        # HyDE prompt imported from prompts.py
        self.hyde_prompt = hyde_prompt

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        toggles: Optional[List[str]] = None,
        use_hyde: bool = False,
        use_rerank: bool = False,
        hyde_temperature: float = 0.5,
        rerank_top_k: int = 250,
        max_doclen: int = 250,
        generate_n: int = 1,
        return_scores: bool = False,
    ):
        """
        Retrieve documents relevant to `query` from the FAISS-indexed dataset,
        using original-like logic (distance->similarity, date/citation weighting, HyDE prompt, etc.).

        Steps:
          1. (Optional) Generate a hypothetical doc (HyDE) and embed it + the query.
          2. Perform FAISS search (up to `rerank_top_k` docs).
          3. Weight those docs by keywords/date/citations.
          4. (Optional) re-rank with Cohere -> final top_k results.
          5. (Optional) enforce a max token limit for the HyDE doc.

        Args:
            query: The user question or search query.
            top_k: Final number of documents to return (default 10).
            toggles: A list of weighting toggles, e.g. ["Keywords","Time","Citations"].
            use_hyde: If True, use HyDE generation for the query embedding.
            use_rerank: If True, re-rank top results with Cohere.
            hyde_temperature: LLM temperature for HyDE generation (default=0.5).
            rerank_top_k: Number of docs to retrieve from FAISS before re-ranking (default=250).
            max_doclen: The maximum token length for the hypothetical doc (default=250).
            generate_n: (Optional) how many hypothetical docs to generate for HyDE. (default=1)
            return_scores: If True, return a dictionary of scores along with the DataFrame.

        Returns:
            If return_scores is False, returns a DataFrame of top_k results.
            If return_scores is True, returns a tuple of (scores_dict, DataFrame).
        """
        toggles = toggles or []
        self.weight_keywords = "Keywords" in toggles
        self.weight_date = "Time" in toggles
        self.weight_citation = "Citations" in toggles

        # Possibly check for large doc request
        if max_doclen * generate_n > 8191:
            raise ValueError(
                f"Too many tokens: max_doclen({max_doclen}) * generate_n({generate_n}) > 8191. "
                "Please reduce max_doclen or generate_n."
            )

        self.query_input_keywords.extend(get_keywords(query, nlp=nlp))

        # 1) Possibly generate HyDE embedding
        if use_hyde:
            query_embedding = self._get_hyde_embedding(
                query=query,
                temperature=hyde_temperature,
                max_tokens=max_doclen,
                generate_n=generate_n,
            )
        else:
            query_embedding = self.embedding_model.embed_query(query)

        # 2) Perform FAISS search (returns distances)
        search_results = self.dataset.search(
            self.column_name, np.array(query_embedding), k=rerank_top_k
        )
        indices, distances = search_results.indices, search_results.scores

        # 3) Postprocess & Weight
        df, scores_dict = self._postprocess_and_weight(indices, distances, query, top_k, use_rerank)
        
        if return_scores:
            return scores_dict, df
        else:
            return df

    def _postprocess_and_weight(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        query: str,
        top_k: int,
        use_rerank: bool,
    ):
        """
        Convert FAISS distances to similarity, apply date/citation weighting (original style),
        apply optional Cohere re-rank, then return final top_k results.
        """
        # Original approach: similarity = 1 / distance
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = 1.0 / distances
            similarity[~np.isfinite(similarity)] = 0.0  # handle distance=0

        partial_df = pd.DataFrame(self.dataset[indices])
        partial_df["indices"] = indices
        partial_df["similarity"] = similarity

        # Weight toggles
        if self.weight_keywords:
            partial_df["kw_weight"] = self._compute_keyword_weight_original(
                partial_df, query
            )
        else:
            partial_df["kw_weight"] = 1.0

        if self.weight_date:
            partial_df["date_weight"] = self._compute_date_weight_original(partial_df)
        else:
            partial_df["date_weight"] = 1.0

        if self.weight_citation:
            partial_df["cite_weight"] = self._compute_citation_weight_original(
                partial_df
            )
        else:
            partial_df["cite_weight"] = 1.0

        partial_df["final_score"] = (
            partial_df["similarity"]
            * partial_df["kw_weight"]
            * partial_df["date_weight"]
            * partial_df["cite_weight"]
        )

        partial_df.sort_values("final_score", ascending=False, inplace=True)

        # Re-rank with Cohere if requested
        if use_rerank and self.cohere_client:
            top_df = partial_df.head(250).copy()
            docs_for_rerank = top_df["abstract"].tolist()

            reranked = self.cohere_client.rerank(
                query=query,
                documents=docs_for_rerank,
                model="rerank-english-v3.0",
                top_n=top_k,
            )
            final_indices = [r.index for r in reranked.results]
            top_df = top_df.iloc[final_indices]
            top_df = top_df.head(top_k)
            
            # Create scores dictionary for the reranked results
            scores_dict = {idx: score for idx, score in zip(top_df.indices, top_df.final_score)}
            
            return top_df.reset_index(drop=True), scores_dict

        # For non-reranked results
        top_df = partial_df.head(top_k).reset_index(drop=True)
        scores_dict = {idx: score for idx, score in zip(partial_df.head(top_k).indices, partial_df.head(top_k).final_score)}
        
        return top_df, scores_dict

    def _get_hyde_embedding(
        self,
        query: str,
        temperature: float,
        max_tokens: int,
        generate_n: int = 1,
    ) -> np.ndarray:
        """
        Generate up to `generate_n` hypothetical documents using our HyDE prompt,
        each restricted to `max_tokens` length. Then embed these docs + the original query,
        and return the average embedding. This matches the original style
        with a maximum token length in the prompt.
        """
        hyde_llm = get_openai_chat_llm(deployment_name="gpt-4o-mini")

        # We'll generate `generate_n` docs, then embed them
        # In the original, there's a single doc approach, but some code
        # uses multiple docs. We'll replicate that logic here.
        doc_embeddings = []
        for _ in range(generate_n):
            # Pass both the query and the max_tokens
            prompt_text = self.hyde_prompt.format(query=query, max_tokens=max_tokens)
            doc = hyde_llm.invoke(prompt_text).content
            doc_emb = self.embedding_model.embed_query(doc)
            doc_embeddings.append(doc_emb)

        # Also embed the original query itself
        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings.append(query_embedding)

        # Return the mean of all embeddings
        return np.mean(doc_embeddings, axis=0)

    def _compute_keyword_weight_original(
        self, df: pd.DataFrame, query: str
    ) -> np.ndarray:
        """
        Original logic:
         1. Start each row's weight at 0.1
         2. For each row's 'keywords', if they contain any of the combined query keywords,
            increment the row weight by 0.1 per match
         3. Then do a normalization by dividing by the max.
        """
        query_kws = get_keywords(query, nlp=nlp) + self.query_input_keywords
        query_kws = [k.lower() for k in query_kws if k]

        kw_weight = np.full(len(df), 0.1, dtype=float)

        for i, row in df.iterrows():
            row_keywords = row.get("keywords", [])
            for k in row_keywords:
                klower = k.lower()
                for qkw in query_kws:
                    if qkw in klower:
                        kw_weight[i] += 0.1

        max_val = kw_weight.max() if len(kw_weight) else 1e-8
        if max_val > 0:
            kw_weight /= max_val
        return kw_weight

    def _compute_date_weight_original(self, df: pd.DataFrame) -> np.ndarray:
        """
        Original date weighting logic:
            age_weight = (1 + exp(date_diff/0.7))^-1
        where date_diff = (today_date - row_date).days / 365.
        Then normalized by max.
        """
        now = datetime.now().date()
        out = []
        for _, row in df.iterrows():
            paper_date = row.get("date", None)
            if paper_date is None:
                out.append(1.0)
                continue

            if not isinstance(paper_date, datetime):
                paper_date = pd.to_datetime(paper_date)
            date_diff = (now - paper_date.date()).days / 365.0

            w = 1.0 / (1.0 + np.exp(date_diff / 0.7))
            out.append(w)

        arr = np.array(out, dtype=float)
        arr /= arr.max() + 1e-8
        return arr

    def _compute_citation_weight_original(self, df: pd.DataFrame) -> np.ndarray:
        """
        Original citation weighting logic:
            sub_cites = ...
            temp[sub_cites > 300] = 300.
            cite_weight = (1 + exp((300-temp)/42.0))^-1
            then normalized
        """
        cites = df.get("cites", [0] * len(df))
        cites_array = np.array(cites, dtype=float)
        cites_array[cites_array > 300] = 300

        cite_weight = 1.0 / (1.0 + np.exp((300 - cites_array) / 42.0))
        cite_weight /= cite_weight.max() + 1e-8
        return cite_weight
