from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.providers import get_openai_chat_llm, get_openai_embeddings


@pytest.fixture
def mock_config():
    return {
        "chat_base_url": "https://test.openai.azure.com",
        "chat_deployment_name": "test-gpt",
        "chat_api_version": "2024-02-01",
        "chat_api_key": "test-key",
        "chat_base_url_4omini": "https://test.openai.azure.com",
        "chat_deployment_name_4omini": "test-gpt-4o-mini",
        "chat_api_version_4omini": "2024-02-01",
        "chat_api_key_4omini": "test-key",
        "embedding_base_url": "https://test.openai.azure.com",
        "embedding_deployment_name": "test-ada",
        "embedding_api_key": "test-key",
        "embedding_api_version": "2024-02-01",
    }


@pytest.fixture(autouse=True)
def mock_azure_config(mock_config):
    with patch("src.providers.config", mock_config):
        yield


def test_get_openai_chat_llm():
    with patch("src.providers.AzureChatOpenAI") as mock_chat:
        # When
        llm = get_openai_chat_llm()

        # Then
        mock_chat.assert_called_once()
        assert llm is not None


def test_get_openai_chat_llm_with_custom_deployment():
    with patch("src.providers.AzureChatOpenAI") as mock_chat:
        # When
        llm = get_openai_chat_llm(deployment_name="custom-gpt")

        # Then
        mock_chat.assert_called_once()
        assert llm is not None


def test_get_openai_embeddings():
    with patch("src.providers.AzureOpenAIEmbeddings") as mock_embeddings:
        # Setup mock embedding response
        mock_instance = Mock()
        mock_instance.embed_query.return_value = np.zeros(
            1536
        )  # Standard embedding size
        mock_embeddings.return_value = mock_instance

        # When
        embeddings = get_openai_embeddings()
        result = embeddings.embed_query("test")

        # Then
        mock_embeddings.assert_called_once()
        assert isinstance(result, (list, np.ndarray))
        assert len(result) == 1536  # Standard OpenAI embedding dimension
