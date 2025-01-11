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
- [Citation](#citation)
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
  │   ├── llm_providers.py
  │   ├── embeddings.py
  │   ├── retrieval/
  │   │   ├── __init__.py
  │   │   ├── retrieve.py
  │   │   ├── ranker.py
  │   │   └── hyde.py
  │   ├── prompts/
  │   │   ├── __init__.py
  │   │   ├── single_paper_prompt.txt
  │   │   └── multi_paper_prompt.txt
  │   └── pipeline/
  │       ├── __init__.py
  │       ├── rag_pipeline.py
  │       └── react_pipeline.py
  ├── app/
  │   ├── app_streamlit.py
  │   ├── app_gradio.py
  │   └── ...
  ├── scripts/
  │   ├── build_faiss_index.py
  │   └── any_other_preprocessing.py
  └── tests/
      ├── test_retrieval.py
      ├── test_embeddings.py
      └── ...
