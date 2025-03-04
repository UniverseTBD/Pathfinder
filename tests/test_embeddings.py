import numpy as np
import pytest

from scripts.embeddings import EmbeddingService


@pytest.fixture
def embedding_service():
    return EmbeddingService()


def test_embed_text_returns_numpy_array(embedding_service):
    # Given
    test_text = "hello world"

    # When
    result = embedding_service.embed_text(test_text)

    # Then
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 1  # Should be 1D array


def test_embed_text_consistent_dimensions(embedding_service):
    # Given
    test_text1 = "hello world"
    test_text2 = "different text"

    # When
    result1 = embedding_service.embed_text(test_text1)
    result2 = embedding_service.embed_text(test_text2)

    # Then
    assert result1.shape == result2.shape


def test_embed_text_handles_empty_string(embedding_service):
    # Given
    test_text = ""

    # When/Then
    result = embedding_service.embed_text(test_text)
    assert isinstance(result, np.ndarray)
