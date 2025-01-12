"""
This module loads and indexes an arXiv corpus using Hugging Face datasets.
Includes retry logic and validation of the loaded dataset.
"""

import time

import numpy as np
from datasets import Dataset, load_dataset

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def load_arxiv_corpus(max_retries: int = MAX_RETRIES) -> Dataset:
    """
    Load the 'kiyer/pathfinder_arxiv_data' dataset and build a FAISS index in-memory.

    Args:
        max_retries: Maximum number of retries for loading the dataset

    Returns:
        Dataset: Loaded and indexed arXiv dataset

    Raises:
        ValueError: If dataset validation fails
    """
    for attempt in range(max_retries):
        try:
            arxiv_corpus = load_dataset("kiyer/pathfinder_arxiv_data", split="train")

            if len(arxiv_corpus) == 0:
                return arxiv_corpus

            sample_embed = arxiv_corpus[0]["embed"]
            if not isinstance(sample_embed, np.ndarray) or len(sample_embed) == 0:
                raise ValueError(
                    "Invalid embedding format: expected non-empty numpy array"
                )

            arxiv_corpus.add_faiss_index(column="embed")
            return arxiv_corpus

        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to load dataset after {max_retries} attempts"
                ) from e
            else:
                time.sleep(RETRY_DELAY)
