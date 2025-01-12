from unittest.mock import patch

from scripts.useful_api_calls import get_openai_embeddings


@patch(
    "scripts.useful_api_calls.config",
    {
        "embedding_base_url": "https://example.com/",
        "embedding_deployment_name": "demo-deployment",
        "embedding_api_key": "test-key",
        "embedding_api_version": "v1",
    },
)
def test_get_openai_embeddings():
    embeddings = get_openai_embeddings()
    assert embeddings is not None
