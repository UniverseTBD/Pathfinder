from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from src.config import config


def get_openai_chat_llm(deployment_name=None, temperature=0.0):
    llm = AzureChatOpenAI(
        azure_endpoint=config['chat_base_url'],
        azure_deployment=deployment_name or config["chat_deployment_name"],
        api_version=config["chat_api_version"],
        api_key=config["chat_api_key"],
        temperature=temperature,
    )
    print(f"Loaded OpenAI chat model: {config['chat_deployment_name']}")
    return llm

def get_openai_embeddings(model_name=None):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=config["embedding_base_url"],
        deployment=config["embedding_deployment_name"],
        api_key=config["embedding_api_key"],
        api_version=config["embedding_api_version"],
        chunk_size=16
    )
    print(f"Loaded OpenAI embeddings model: {config['embedding_deployment_name']}")
    return embeddings

if __name__ == "__main__":
    # Test the embeddings
    embeddings = get_openai_embeddings()
    result = embeddings.embed_query("test text")
    print(f"Successfully generated embedding of length: {len(result)}")
