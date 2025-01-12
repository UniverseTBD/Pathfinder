"""API utilities such as Azure OpenAI API for chat and embeddings.

Configuration:
    This module requires a config.yml file in the parent directory with API credentials.

    1. Create a copy of config_template.yml and rename it to config.yml
    2. Update config.yml with your API (e.g. Azure OpenAI) credentials:
        - embedding_base_url: Your Azure OpenAI endpoint for embeddings
        - embedding_api_key: Your API key for embeddings
        - embedding_deployment_name: Your deployment name (e.g. "text-embedding-3-small")
        - embedding_api_version: API version (e.g. "2023-05-15")
        - chat_base_url: Your Azure OpenAI endpoint for chat
        - chat_api_key: Your API key for chat
        - chat_deployment_name: Your deployment name (e.g. "gpt-4")
        - chat_api_version: API version (e.g. "2023-05-15")
    3. Never commit config.yml to version control
"""

import os
from typing import Optional

import yaml
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Constants
DEFAULT_CHUNK_SIZE = 16
DEFAULT_TEMPERATURE = 0.0

try:
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yml"), "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "config.yml not found. Please copy config_template.yml to config.yml and update with your credentials & make sure the path to the config file is correct."
    )


def get_openai_chat_llm(
    deployment_name: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE
) -> AzureChatOpenAI:
    """Initialize Azure OpenAI Chat model.

    Args:
        deployment_name: Optional custom deployment name
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        AzureChatOpenAI: Configured chat model instance

    Raises:
        KeyError: If required config values are missing
    """
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=config["chat_base_url"],
            azure_deployment=deployment_name or config["chat_deployment_name"],
            api_version=config["chat_api_version"],
            api_key=config["chat_api_key"],
            temperature=temperature,
        )
        print(
            f"Loaded OpenAI chat model: {deployment_name or config['chat_deployment_name']}"
        )
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
