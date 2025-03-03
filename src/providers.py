from typing import Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from src.config import config

# Constants
DEFAULT_CHUNK_SIZE = 16
DEFAULT_TEMPERATURE = 0.0


def get_openai_chat_llm(
    deployment_name: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE
) -> AzureChatOpenAI:
    """Initialize Azure OpenAI Chat model.

    Args:
        deployment_name: Optional custom deployment name (ignored, always using gpt-4o-mini)
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        AzureChatOpenAI: Configured chat model instance with gpt-4o-mini

    Raises:
        KeyError: If required config values are missing
    """
    try:
        # Always use gpt-4o-mini
        llm = AzureChatOpenAI(
            azure_endpoint=config["chat_base_url_4omini"],
            azure_deployment=config["chat_deployment_name_4omini"],
            api_version=config["chat_api_version_4omini"],
            api_key=config["chat_api_key_4omini"],
            temperature=temperature,
        )
        print(f"Loaded OpenAI chat model: {config['chat_deployment_name_4omini']}")
        return llm
    except KeyError as e:
        raise KeyError(f"Missing required configuration: {e}")


def get_openai_embeddings(
    model_name: Optional[str] = None, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> AzureOpenAIEmbeddings:
    """Initialize Azure OpenAI Embeddings model.

    Args:
        model_name: Optional custom model name
        chunk_size: Batch size for processing

    Returns:
        AzureOpenAIEmbeddings: Configured embeddings instance

    Raises:
        KeyError: If required config values are missing
    """
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config["embedding_base_url"],
            deployment=model_name or config["embedding_deployment_name"],
            api_key=config["embedding_api_key"],
            api_version=config["embedding_api_version"],
            chunk_size=chunk_size,
        )
        print(
            f"Loaded OpenAI embeddings model: {model_name or config['embedding_deployment_name']}"
        )
        return embeddings
    except KeyError as e:
        raise KeyError(f"Missing required configuration: {e}")
