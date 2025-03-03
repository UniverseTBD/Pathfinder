# Pathfinder-Lite

**A Semantic Framework for Literature Review and Knowledge Discovery in Astronomy**

Pathfinder-Lite, a variation of our original Pathfinder, is a modular, retrieval-augmented generation (RAG) tool for doing semantic search and question-answering on a large corpus of astronomy papers. It leverages modern Large Language Models (LLMs), vector embeddings, and optional re-ranking to deliver relevant papers and concise answers to astronomy-related queries.

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
- [Using Custom Datasets](#using-custom-datasets)
  - [1. Prepare Your Dataset](#1-prepare-your-dataset)
  - [2. Build the FAISS Index](#2-build-the-faiss-index)
  - [3. Configure Pathfinder](#3-configure-pathfinder-to-use-your-index)
  - [4. Run Pathfinder](#4-run-pathfinder-with-your-dataset)
  - [5. Customizing the Dataset Loader](#5-customizing-the-dataset-loader-advanced)
- [Contributing](#contributing)
- [License](#license)
- [Recent Improvements](#recent-improvements)
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

- **Modular code**: Separated into retrieval, pipeline, and UI layers.
- **Single config**: All credentials and environment variables kept in `config.yml`.
- **Flexible**: Swap in different LLMs, embedding models, or re-rankers.
- **Multiple frontends**: Gradio, Slack bots, etc., can hook into the same underlying library.
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
│   ├── run_pathfinder.py
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
   git clone git@github.com:UniverseTBD/Pathfinder.git
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

# GPT-4o-mini Configuration
chat_base_url_4omini: "https://your-gpt4o-azure-endpoint.openai.azure.com"
chat_api_key_4omini: "your-gpt4o-api-key"
chat_deployment_name_4omini: "gpt-4o-mini"
chat_api_version_4omini: "2025-01-01-preview"
```

### 3. Running Pathfinder

#### Command Line Usage

You can use Pathfinder directly from the command line:

```bash
# Basic usage
python -m src.run_pathfinder "What is dark matter?"

# Specify number of papers to retrieve
python -m src.run_pathfinder "What is dark matter?" --top-k 5

# Specify additional keywords to guide the search
python -m src.run_pathfinder "What is dark matter?" --keywords "galaxy,rotation"

# Specify the prompt type
python -m src.run_pathfinder "What is dark matter?" --prompt-type "Multi-paper"

# Specify the retrieval method
python -m src.run_pathfinder "What is dark matter?" --rag-type "Semantic Search"

# Full example with all options
python -m src.run_pathfinder "What is dark matter?" --top-k 10 --keywords "galaxy,rotation" --toggles Keywords Time --prompt-type "Multi-paper" --rag-type "Semantic + HyDE + CoHERE"
```

#### Available Options

- `--top-k`: Number of papers to retrieve (default: 10)
- `--keywords`: Additional keywords to guide the search, comma-separated
- `--toggles`: Weight toggles, can be "Keywords", "Time", or "Citations"
- `--prompt-type`: Type of prompt to use
  - Options: "Auto", "Single-paper", "Multi-paper", "Bibliometric", "Broad but nuanced", "Deep Research (BETA)"
- `--rag-type`: Type of retrieval method to use
  - Options: "Semantic Search", "Semantic + HyDE", "Semantic + CoHERE", "Semantic + HyDE + CoHERE"

#### Web Interface

- **Gradio version**:
  ```bash
  python -m src.app.app_gradio
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

`src/providers.py` centralizes LLM and embedding model initialization, focusing on gpt-4o-mini for all LLM operations:

```python
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import config

def get_openai_chat_llm(deployment_name=None, temperature=0.0):
    """Initialize Azure OpenAI Chat model (always using gpt-4o-mini)."""
    # Always use gpt-4o-mini for all operations
    llm = AzureChatOpenAI(
        azure_endpoint=config["chat_base_url_4omini"],
        azure_deployment=config["chat_deployment_name_4omini"],
        api_version=config["chat_api_version_4omini"],
        api_key=config["chat_api_key_4omini"],
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
# app/app_gradio.py
import gradio as gr
import pandas as pd
from src.run_pathfinder import run_pathfinder

def process_query(query):
    result = run_pathfinder(query=query)
    consensus = result["consensus"] if result["consensus"] else ""
    return result["answer"], result["papers"].to_html(), consensus

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Pathfinder Demo")
        with gr.Row():
            query = gr.Textbox(label="Ask a question:")
        with gr.Row():
            submit_btn = gr.Button("Submit")
        with gr.Row():
            answer_box = gr.Markdown(label="Answer")
        with gr.Row():
            papers_html = gr.HTML(label="Retrieved Papers")
        with gr.Row():
            consensus_box = gr.Markdown(label="Consensus")
            
        submit_btn.click(process_query, inputs=query, outputs=[answer_box, papers_html, consensus_box])
    
    demo.launch()

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

The app layer provides a comprehensive test suite using pytest:

```python
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_retrieval_system.py

# Run with verbose output
pytest tests/ -v
```

Each component has dedicated test files that verify functionality and handle edge cases. The test suite includes:

- Unit tests for individual functions and classes
- Integration tests for component interactions
- Fixtures for consistent test environments

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

## Using Custom Datasets

Pathfinder can be adapted to work with your own dataset. Here's a guide to integrating a custom corpus:

### 1. Prepare Your Dataset

Your dataset should follow the structure expected by Hugging Face's dataset library, with the following fields:
- `title`: Document title
- `abstract`: Document content or summary
- `year`: Publication year (numeric)
- `authors`: List of authors 
- `citations`: Citation count (optional, for citation-based weighting)
- `embed`: Pre-computed embeddings (will be added in step 2)

You can start with a CSV or JSON file containing these fields (except for embeddings which will be computed later).

### 2. Create a Custom Dataset Script

Use the standalone script in the `scripts/` directory to process your custom dataset:

```python
# scripts/embeddings.py
import os
import yaml
import faiss
import numpy as np
import pandas as pd
from typing import List, Optional
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path
from numpy.typing import NDArray
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Load config for embedding model
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Get embedding model from config
def get_openai_embeddings():
    config = load_config()
    return AzureOpenAIEmbeddings(
        azure_endpoint=config["embedding_base_url"],
        deployment=config["embedding_deployment_name"],
        api_key=config["embedding_api_key"],
        api_version=config["embedding_api_version"],
    )

class EmbeddingService:
    EMBEDDING_DIM = 1536  # OpenAI's embedding dimension

    def __init__(self):
        self.embeddings = get_openai_embeddings()
        self.index: Optional[faiss.Index] = None

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text."""
        if not text:
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding, dtype=np.float32)

    def embed_text_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        embeddings = self.embeddings.embed_documents(texts)
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    def create_index(self, dimension: int = EMBEDDING_DIM) -> faiss.Index:
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(dimension)
        return self.index

# Functions for processing custom datasets
def load_custom_data(data_path):
    """Load data from CSV, JSON, or other format."""
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    return df

def process_dataset(df):
    """Process and validate the dataset."""
    required_columns = ['title', 'abstract', 'year', 'authors']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert to dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def create_embeddings(dataset, batch_size=32):
    """Create embeddings for all documents in the dataset."""
    embedding_service = EmbeddingService()
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        texts = [doc['abstract'] for doc in batch]
        # Generate embeddings for this batch
        embeddings = embedding_service.embed_text_batch(texts)
        all_embeddings.extend(embeddings)
    
    # Add embeddings to dataset
    dataset = dataset.add_column('embed', all_embeddings)
    return dataset

def build_and_save_index(dataset, output_dir):
    """Build FAISS index and save dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Add FAISS index
    dataset.add_faiss_index(column='embed')
    
    # Save dataset and metadata
    dataset.save_to_disk(os.path.join(output_dir, 'dataset'))
    
    # Create metadata for retrieval
    metadata = {
        'dataset_name': 'custom',
        'count': len(dataset),
        'created_date': pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        import pickle
        pickle.dump(metadata, f)
        
    print(f"Dataset and index saved to {output_dir}")
    return dataset

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build a custom dataset with FAISS index")
    parser.add_argument("--input", required=True, help="Path to input data file (CSV, JSON)")
    parser.add_argument("--output", required=True, help="Directory to save processed dataset and index")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    df = load_custom_data(args.input)
    
    print(f"Processing dataset with {len(df)} documents")
    dataset = process_dataset(df)
    
    print("Creating embeddings (this may take a while)...")
    dataset_with_embeds = create_embeddings(dataset, batch_size=args.batch_size)
    
    print("Building and saving FAISS index...")
    build_and_save_index(dataset_with_embeds, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
```

### 3. Use the Custom Dataset Script

Run the script to process your data and create the necessary FAISS index:

```bash
# Create a virtual environment if you haven't already
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Process your custom dataset
python scripts/custom_dataset_builder.py --input your_data.csv --output data/custom_dataset
```

This will:
1. Load your custom dataset
2. Generate embeddings for each document using the configured embedding model
3. Build a FAISS index for fast similarity search
4. Save the dataset and index to the specified output location

### 4. Modify the Retrieval System

Create a custom version of the retrieval system that works with your local dataset:

```python
# scripts/custom_retrieval.py
import os
import yaml
import numpy as np
import pandas as pd
from datasets import load_from_disk
import cohere

# Load config for API keys
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()

class CustomRetrievalSystem:
    """Retrieval system for custom datasets saved with the embeddings.py script."""
    
    def __init__(
        self,
        dataset_path,
        column_name="embed",
    ):
        """Initialize with a local dataset instead of from Hugging Face."""
        # Dataset info
        self.dataset_path = dataset_path
        self.column_name = column_name
        
        # Load dataset from disk
        self.dataset = load_from_disk(self.dataset_path)
        self.dataset.add_faiss_index(column=self.column_name)
        print(f"Loaded custom dataset from '{self.dataset_path}' with FAISS index.")
        
        # Set up cohere client for reranking if available
        self.cohere_key = config.get("cohere_api_key", os.environ.get("cohere_key", ""))
        if self.cohere_key:
            print("Initialized Cohere client for reranking")
            self.cohere_client = cohere.Client(self.cohere_key)
        else:
            print("No Cohere API key found, reranking will not be available")
            self.cohere_client = None
            
        # Get embedding model
        from scripts.embeddings import get_openai_embeddings
        self.embedding_model = get_openai_embeddings()
        
        # Weight toggles
        self.weight_keywords = False
        self.weight_date = False
        self.weight_citation = False
    
    def retrieve(self, query, top_k=10, use_rerank=False):
        """Retrieve documents relevant to the query."""
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Perform FAISS search
        scores, indices = self.dataset.get_nearest_examples(
            self.column_name, query_embedding, k=top_k
        )
        
        # Convert to dataframe for easier handling
        results = []
        for i, idx in enumerate(indices):
            item = self.dataset[idx]
            results.append({
                "title": item["title"],
                "abstract": item["abstract"],
                "year": item["year"],
                "authors": item["authors"],
                "score": scores[i],
                # Add other fields as available in your dataset
            })
        
        results_df = pd.DataFrame(results)
        
        # Optional reranking with Cohere
        if use_rerank and self.cohere_client:
            rerank_results = self.cohere_client.rerank(
                query=query,
                documents=results_df["abstract"].tolist(),
                top_n=top_k,
                model="rerank-english-v2.0"
            )
            
            # Update scores based on reranking
            reranked_indices = [r.index for r in rerank_results]
            reranked_df = results_df.iloc[reranked_indices].copy()
            reranked_df["score"] = [r.relevance_score for r in rerank_results]
            return reranked_df
        
        return results_df
```

### 5. Create a Custom Run Script

Create a simple script to run queries against your custom dataset:

```python
# scripts/run_custom_query.py
import argparse
import pandas as pd
from custom_retrieval import CustomRetrievalSystem

def main():
    parser = argparse.ArgumentParser(description="Query a custom Pathfinder dataset")
    parser.add_argument("query", help="The query to search for")
    parser.add_argument("--dataset", default="data/custom_dataset/dataset", 
                      help="Path to the dataset directory")
    parser.add_argument("--top-k", type=int, default=10, 
                      help="Number of results to return")
    parser.add_argument("--use-rerank", action="store_true", 
                      help="Use Cohere reranking")
    
    args = parser.parse_args()
    
    # Initialize retrieval system
    retrieval = CustomRetrievalSystem(dataset_path=args.dataset)
    
    # Retrieve results
    results = retrieval.retrieve(
        query=args.query,
        top_k=args.top_k,
        use_rerank=args.use_rerank
    )
    
    # Display results
    print(f"\nTop {len(results)} results for query: '{args.query}'\n")
    
    for i, row in results.iterrows():
        print(f"{i+1}. {row['title']} ({row['year']})")
        print(f"   Authors: {row['authors']}")
        print(f"   Score: {row['score']:.4f}")
        print(f"   Abstract: {row['abstract'][:200]}...")
        print()

if __name__ == "__main__":
    main()
```

### 6. Run Queries Against Your Custom Dataset

Now you can use your custom dataset with a simple command:

```bash
# Run a simple query
python scripts/run_custom_query.py "Your query here"

# Use more options
python scripts/run_custom_query.py "Your query here" --dataset data/your_custom_dataset/dataset --top-k 5 --use-rerank
```

### 7. Integration with Existing Pathfinder (Optional)

If you want to integrate your custom dataset with the existing Pathfinder system:

1. Add your custom dataset path to your `config.yml`:

```yaml
# Add to your config.yml
custom_dataset:
  path: "data/custom_dataset/dataset"  # Path to the saved dataset
  use_custom: true  # Flag to use custom dataset
```

2. Create a custom retrieval system in the src directory:

```python
# src/retrieval/custom_retrieval_system.py
from src.retrieval.retrieval_system import RetrievalSystem
from datasets import load_from_disk

class CustomRetrievalSystem(RetrievalSystem):
    def __init__(
        self,
        dataset_path,
        column_name="embed",
    ):
        """Initialize with a local dataset."""
        # Call parent __init__ but we'll override the dataset
        super().__init__()
        
        # Override dataset loading
        self.dataset = load_from_disk(dataset_path)
        self.dataset.add_faiss_index(column=column_name)
        print(f"Loaded custom dataset from '{dataset_path}'")
```

3. Modify the main run script:

```python
# In src/run_pathfinder.py

from src.retrieval.custom_retrieval_system import CustomRetrievalSystem

def run_pathfinder(query, use_custom_dataset=False, **kwargs):
    if use_custom_dataset:
        dataset_path = config.get("custom_dataset", {}).get("path")
        if dataset_path:
            retrieval_system = CustomRetrievalSystem(dataset_path=dataset_path)
        else:
            retrieval_system = RetrievalSystem()  # Fallback to default
    else:
        retrieval_system = RetrievalSystem()
        
    # Continue with existing code...
```

4. Run Pathfinder with your custom dataset:

```bash
python -m src.run_pathfinder "Your query here" --use-custom-dataset
```

## Recent Improvements

The latest version of Pathfinder includes several significant improvements:

1. **Streamlined Model Integration**: Now exclusively using gpt-4o-mini for all language model operations, providing optimal performance and consistency.

2. **Enhanced UI**: Completely redesigned Gradio interface with a modern dark theme for better readability and user experience.

3. **Improved Project Structure**: Reorganized repository structure with cleaner separation of concerns and more intuitive file layout.

4. **Enhanced Retrieval**: Refined Cohere reranking to improve search result quality and relevance.

5. **Structured Output Generation**: Implemented robust structured output with fallback mechanisms for consistent results.

6. **More Robust Error Handling**: Added comprehensive error handling throughout the system to gracefully recover from failures.

7. **Command-line Interface**: Streamlined command-line interface with clear documentation for all available options.

8. **Custom Dataset Support**: Added support for integrating and searching custom document collections.

## Acknowledgments

- Thanks to the UniverseTBD and JHU JSALT 2024 Evals LLMs teams for Astro teams for development support
- Thanks to all open-source libraries used in this project
- Special thanks to early testers and contributors

> **Disclaimer**: Pathfinder is meant to complement, not replace, services like arXiv or NASA ADS. Always validate LLM-generated text against primary sources.
