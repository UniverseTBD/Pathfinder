from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.pathfinder_dataset_loader import load_arxiv_corpus


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with valid embeddings"""
    dataset = MagicMock()
    # Use proper magic method mocking
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(
        return_value={
            "embed": np.random.rand(768),
            "text": "Sample text",
            "title": "Sample title",
        }
    )
    dataset.add_faiss_index = MagicMock()
    return dataset


@pytest.fixture
def empty_dataset():
    """Create an empty mock dataset"""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=0)
    return dataset


def test_successful_load(mock_dataset):
    """Test successful dataset loading"""
    with patch(
        "scripts.pathfinder_dataset_loader.load_dataset", return_value=mock_dataset
    ):
        dataset = load_arxiv_corpus()
        assert dataset is not None
        assert len(dataset) == 100
        mock_dataset.add_faiss_index.assert_called_once_with(column="embed")


def test_retry_mechanism(mock_dataset):
    """Test retry mechanism on temporary failures"""
    with patch(
        "scripts.pathfinder_dataset_loader.load_dataset",
        side_effect=[ValueError, ValueError, mock_dataset],
    ):
        dataset = load_arxiv_corpus()
        assert dataset is not None
        assert len(dataset) == 100


def test_empty_dataset(empty_dataset):
    """Test handling of empty dataset"""
    with patch(
        "scripts.pathfinder_dataset_loader.load_dataset", return_value=empty_dataset
    ):
        dataset = load_arxiv_corpus()
        assert dataset is not None
        assert len(dataset) == 0


def test_max_retries_exceeded():
    """Test exceeding maximum retry attempts"""
    with patch(
        "scripts.pathfinder_dataset_loader.load_dataset",
        side_effect=ValueError("Network error"),
    ):
        with pytest.raises(ValueError, match="Failed to load dataset after"):
            load_arxiv_corpus(max_retries=2)
