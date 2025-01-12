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

**Pathfinder** enables natural-language queries over a large corpus (e.g. 300k+ astronomy papers). It retrieves semantically relevant documents and then uses an LLM to generate context-aware answers. It can also:

- Perform semantic literature search (beyond simple keyword matching).
- Weight search results by recency, citation count, or custom logic.
- Rerank search results using external models (e.g. Cohere).
- Synthesize answers via **RAG** or more complex ReAct-based reasoning.

Pathfinder complements existing tools like NASA ADS or arXiv search by allowing free-form queries and advanced summarization of results.

---

## Features

- **Modular code**: Separated into retrieval, embeddings, pipeline, and UI layers.
- **Single config**: All credentials and environment variables kept in `config.yml`.
- **Flexible**: Swap in different LLMs, embedding models, or re-rankers.
- **Multiple frontends**: Streamlit, Gradio, Slack bots, etc., can hook into the same underlying library.
- **Extensible**: Add scripts to build or update FAISS indexes, advanced pipelines, or custom ranking logic.

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
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retrieve.py
│   │   ├── ranker.py
│   │   └── hyde.py
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
    ├── useful_api_calls.py
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
embedding_base_url: "your-azure-endpoint"
embedding_api_key: "your-api-key"
embedding_deployment: "your-deployment-name"
embedding_deployment_name: "your-model-name"
embedding_api_version: "2023-05-15"

chat_base_url: "your-azure-chat-endpoint"
chat_api_key: "your-chat-api-key"
chat_deployment_name: "your-chat-deployment"
chat_api_version: "2023-05-15"
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

`src/azure_providers.py` centralizes LLM and embedding model initialization:

```python
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import config

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

The retrieval system (`src/retrieval/retrieve.py`) manages semantic search and document ranking:

```python
class RetrievalSystem:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.embeddings = get_openai_embeddings()

    def retrieve(self, query):
        # Implement retrieval logic
        pass
```

---

### 5. Pipelines: RAG & ReAct

- **RAG pipeline (`rag_pipeline.py`)**: Combines retrieval with generation
- **ReAct pipeline (`react_pipeline.py`)**: Implements multi-step reasoning

---

### 6. App Layer

The app layer provides user interfaces using Streamlit or Gradio:

```python
# app/app_streamlit.py
import streamlit as st
from src.pipeline.rag_pipeline import RAGPipeline

def main():
    st.title("Pathfinder Demo")
    query = st.text_input("Ask a question:")
    if query:
        pipeline = RAGPipeline()
        answer = pipeline.run(query)
        st.write(answer)

if __name__ == "__main__":
    main()
```

---

### 7. Scripts Folder

Contains utility scripts for tasks like building FAISS indexes:

```bash
python scripts/build_faiss_index.py
```

---

### 8. Testing

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
