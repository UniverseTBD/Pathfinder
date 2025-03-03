# Pathfinder

**A Semantic Framework for Literature Review and Knowledge Discovery in Astronomy**

Pathfinder is a modular, retrieval-augmented generation (RAG) tool for doing semantic search and question-answering on a large corpus of astronomy papers. It leverages modern Large Language Models (LLMs), vector embeddings, and optional re-ranking to deliver relevant papers and concise answers to astronomy-related queries.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
  - [1. Installation](#1-installation)
  - [2. Configuration](#2-configuration)
  - [3. Running the App](#3-running-the-app)
- [Detailed Explanation](#detailed-explanation)
  - [1. `config.yml` & `src/config.py`](#1-configyml--srcconfigpy)
  - [2. LLM Providers](#2-llm-providers)
  - [3. Embeddings](#3-embeddings)
  - [4. Retrieval System](#4-retrieval-system)
  - [5. Pipelines: RAG & ReAct](#5-pipelines-rag--react)
  - [6. App Layer](#6-app-layer)
  - [7. Scripts Folder](#7-scripts-folder)
  - [8. Testing](#8-testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

**Pathfinder** is a modular, retrieval-augmented generation (RAG) system designed for astronomy research. It enables natural-language queries over a large corpus of astronomy papers, retrieves semantically relevant documents, and uses LLMs to generate context-aware answers.

Key capabilities:

- **Semantic Search**: Find papers based on meaning, not just keywords
- **Advanced Retrieval**: HyDE (Hypothetical Document Embeddings) for improved search accuracy
- **Adaptive Weighting**: Weight results by keywords, recency, or citation count
- **Reranking**: Improve retrieval quality using Cohere's reranking model
- **Multiple RAG Modes**: Choose appropriate prompt types for different question types
- **Deep Research**: Break complex questions into sub-questions for comprehensive answers
- **Consensus Evaluation**: Assess agreement level among retrieved papers
- **Question Type Detection**: Automatically detect optimal processing for different questions
- **Multi-Model Support**: Flexible support for different models like GPT-4o and various Azure OpenAI offerings
- **Robust Error Handling**: Fallback mechanisms to ensure reliability when services fail

Pathfinder complements existing tools like NASA ADS or arXiv search by allowing free-form queries and advanced summarization of results. Its modular architecture makes it adaptable to other domains beyond astronomy.

---

## Features

- **Modular code**: Separated into retrieval, embeddings, pipeline, and UI layers.
- **Single config**: All credentials and environment variables kept in `config.yml`.
- **Flexible**: Swap in different LLMs, embedding models, or re-rankers.
- **Multiple frontends**: Streamlit, Gradio, Slack bots, etc., can hook into the same underlying library.
- **Extensible**: Add scripts to build or update FAISS indexes, advanced pipelines, or custom ranking logic.
- **Advanced retrieval methods**: HyDE (Hypothetical Document Embeddings), Cohere reranking, and weighted scoring by keywords, publication date, and citation count.

---

## Project Structure

A recommended layout:

```
pathfinder/
├── LICENSE
├── README.md
├── requirements.txt
├── config.yml
├── data/
│   ├── ...
│   └── (local data files, e.g. FAISS indexes)
├── local_files/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── providers.py
│   ├── embeddings.py
│   ├── nlp_utils.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retrieval_system.py
│   ├── prompts.py
│   └── pipeline/
│       ├── __init__.py
│       ├── rag_pipeline.py
│       └── react_pipeline.py
├── app/
│   ├── app_gradio.py
│   └── ...
├── scripts/
│   ├── build_faiss_index.py
│   ├── pathfinder_dataset_loader.py
│   ├── useful_api_calls.py
│   └── any_other_preprocessing.py
└── tests/
    ├── test_providers.py
    ├── test_embeddings.py
    └── ...
```

---

## Quickstart

### 1. Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/<username>/pathfinder.git
   cd pathfinder
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configuration

Copy `config.yml.template` to `config.yml` and edit it with your credentials:

```yaml
# config.yml
embedding_base_url: "https://your-azure-endpoint.openai.azure.com"
embedding_api_key: "your-api-key"
embedding_deployment: "text-embedding-3-small"
embedding_deployment_name: "text-embedding-3-small"
embedding_api_version: "2023-05-15"

chat_base_url: "https://your-azure-chat-endpoint.openai.azure.com"
chat_api_key: "your-chat-api-key"
chat_deployment_name: "o1-mini"  # Or your preferred model
chat_api_version: "2023-05-15"

# Optional: Add additional model configurations
chat_base_url_4omini: "https://your-gpt4o-azure-endpoint.openai.azure.com"
chat_api_key_4omini: "your-gpt4o-api-key"
chat_deployment_name_4omini: "gpt-4o-mini"
chat_api_version_4omini: "2025-01-01-preview"
```

### 3. Running the App

- **Streamlit version**:
  ```bash
  streamlit run app/app_streamlit.py
  ```
- **Gradio version**:
  ```bash
  python app/app_gradio.py
  ```

---

## Detailed Explanation

### 1. `config.yml` & `src/config.py`

The configuration system consists of:

- **`config.yml`**: Stores all secrets and environment variables
- **`src/config.py`**: Loads and exposes the configuration

Example `config.py`:

```python
import yaml, os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()
```

---

### 2. LLM Providers

`src/providers.py` centralizes LLM and embedding model initialization with support for multiple models:

```python
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import config

def get_openai_chat_llm(deployment_name=None, temperature=0.0):
    """Initialize Azure OpenAI Chat model with support for multiple deployments."""
    # Check if we're explicitly requesting a specific model like GPT-4o-mini
    if deployment_name == "gpt-4o-mini":
        llm = AzureChatOpenAI(
            azure_endpoint=config["chat_base_url_4omini"],
            azure_deployment=config["chat_deployment_name_4omini"],
            api_version=config["chat_api_version_4omini"],
            api_key=config["chat_api_key_4omini"],
            temperature=temperature,
        )
    else:
        # Use the default model configuration
        llm = AzureChatOpenAI(
            azure_endpoint=config["chat_base_url"],
            azure_deployment=deployment_name or config["chat_deployment_name"],
            api_version=config["chat_api_version"],
            api_key=config["chat_api_key"],
            temperature=temperature,
        )
    return llm

def get_openai_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=config["embedding_base_url"],
        deployment=config["embedding_deployment_name"],
        api_key=config["embedding_api_key"],
        api_version=config["embedding_api_version"],
    )
```

---

### 3. Embeddings

`src/embeddings.py` handles vector embeddings and FAISS operations:

```python
from src.llm_providers import get_openai_embeddings

class EmbeddingService:
    def __init__(self):
        self.embeddings = get_openai_embeddings()

    def embed_text(self, text: str):
        return self.embeddings.embed_query(text)
```

---

### 4. Retrieval System

The retrieval system (`src/retrieval/retrieval_system.py`) manages semantic search and document ranking:

```python
from src.retrieval.retrieval_system import RetrievalSystem

# Initialize the retrieval system
retrieval = RetrievalSystem()

# Basic retrieval
results, papers_df = retrieval.retrieve(
    query="How are exoplanets detected?", 
    top_k=10
)

# Advanced retrieval with HyDE and reranking
results, papers_df = retrieval.retrieve(
    query="What is the current understanding of dark energy?",
    top_k=10,
    toggles=["Keywords", "Time", "Citations"],  # Weight by keywords, recency, and citation count
    use_hyde=True,    # Use hypothetical document embeddings
    use_rerank=True,  # Use Cohere reranking
    hyde_temperature=0.5,  # Control HyDE generation creativity
    rerank_top_k=250,  # Number of candidates for reranking
    max_doclen=250,   # Max length of generated HyDE document 
    generate_n=1,     # Number of HyDE documents to generate
)
```

The RetrievalSystem supports multiple retrieval methods and weighting options:

1. **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical document that answers the query, and uses its embedding to find similar papers.
2. **Cohere Reranking**: Re-ranks initial retrieval results using Cohere's reranking model.
3. **Weighting Options**:
   - Keywords: Weights results by keyword match with query
   - Time: Weights results by recency
   - Citations: Weights results by citation count

---

### 5. Pipelines: RAG & Deep Research

The pipeline module contains the reasoning systems built on top of retrieval:

```python
# RAG Pipeline (simple question answering)
from src.pipeline.rag_pipeline import run_rag_qa

result = run_rag_qa(
    query="What is the Hubble constant?",
    papers_df=papers_dataframe,
    question_type="Multi-paper"  # or "Single-paper", "Bibliometric", "Broad but nuanced"
)
answer = result["answer"]

# Deep Research Pipeline (complex questions)
from src.pipeline.deep_research_pipeline import deep_research

papers_df, result = deep_research(
    question="How does dark matter affect galaxy formation and evolution?",
    top_k=10,  # papers per sub-question
    retrieval_system=retrieval_system
)
complex_answer = result["answer"]
```

The Deep Research pipeline breaks down complex questions into atomic sub-questions, researches each one, and then synthesizes a comprehensive answer.

---

### 6. Consensus Evaluation

The consensus evaluation module analyzes the agreement level between retrieved papers and the query:

```python
from src.consensus_evaluation import evaluate_overall_consensus

# Get consensus evaluation from abstracts
consensus = evaluate_overall_consensus(
    query="Are exoplanets common in our galaxy?",
    abstracts=["Abstract 1...", "Abstract 2...", "Abstract 3..."]
)

print(f"Consensus: {consensus.consensus}")
print(f"Explanation: {consensus.explanation}")
print(f"Relevance score: {consensus.relevance_score}")
```

### 7. Integrated Interface

The main run_pathfinder function combines all components:

```python
from src.run_pathfinder import run_pathfinder

# Run the complete Pathfinder system
result = run_pathfinder(
    query="What is the evidence for dark matter?",
    top_k=10,                           # Number of papers to retrieve
    extra_keywords="galaxy,rotation",   # Optional keywords to guide search
    toggles=["Keywords", "Time"],       # Weight by keywords and recency
    prompt_type="Auto",                # Auto-detect question type or choose specific type
    rag_type="Semantic + HyDE + CoHERE" # Retrieval method
)

# Access results
papers_df = result["papers"]         # Retrieved papers
answer = result["answer"]            # Generated answer
question_type = result["question_type"]  # Detected question type
consensus = result["consensus"]      # Consensus evaluation
```

### 8. App Layer

The app layer provides user interfaces using Streamlit or Gradio:

```python
# app/app_streamlit.py
import streamlit as st
from src.run_pathfinder import run_pathfinder

def main():
    st.title("Pathfinder Demo")
    query = st.text_input("Ask a question:")
    if query:
        result = run_pathfinder(query=query)
        st.write(result["answer"])
        st.dataframe(result["papers"])
        
        if result["consensus"]:
            st.write(result["consensus"])

if __name__ == "__main__":
    main()
```

---

### 9. Scripts Folder

Contains utility scripts for tasks like building FAISS indexes:

```bash
python scripts/build_faiss_index.py
```

---

### 10. Testing

Run tests with `pytest`:

```bash
pytest tests/
```

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Thanks to the UniverseTBD and JHU JSALT 2024 Evals LLMs teams for Astro teams for development support
- Thanks to all open-source libraries used in this project
- Special thanks to early testers and contributors

> **Disclaimer**: Pathfinder is meant to complement, not replace, services like arXiv or NASA ADS. Always validate LLM-generated text against primary sources.
