# src/pipeline/rag_pipeline.py

from typing import Optional

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Local imports
from src.prompts import (
    bibliometric_prompt,
    deep_knowledge_prompt,
    regular_prompt,
    single_paper_prompt,
)
from src.providers import get_openai_chat_llm, get_openai_embeddings


def run_rag_qa(
    query: str,
    papers_df: pd.DataFrame,
    question_type: Optional[str] = None,
    chunk_size: int = 150,
    chunk_overlap: int = 50,
    top_k_chunks: int = 6,
) -> dict:
    """
    Perform a RAG pipeline over the given DataFrame of papers, creating chunk-level documents,
    building a local Chroma vector store, and using a question_type-specific prompt.

    If 'question_type' is not among {"Bibliometric", "Single-paper", "Broad but nuanced"},
    we default to "Regular".

    Returns a dict with:
      "answer": The final LLM answer
      "chunks_used": The local splitted docs used for chunk-level retrieval
    """

    # 1) Fallback if question_type is unrecognized
    recognized_types = {"Bibliometric", "Single-paper", "Broad but nuanced"}
    if not question_type or question_type not in recognized_types:
        question_type = "Regular"

    # 2) If DataFrame is empty, short-circuit so we don't build Chroma
    if papers_df.empty:
        return {"answer": "No documents found for this query.", "chunks_used": []}

    # Convert each row to a Document, preserving e.g. ads_id
    docs = []
    for i, row in papers_df.iterrows():
        content = f"Paper {i+1}: {row.get('title', 'No Title')}\n{row.get('abstract', '')}\n\n"
        metadata = {"source": row.get("ads_id", f"doc-{i}")}
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)

    # 3) Split doc text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    splits = splitter.split_documents(docs)

    # If no chunks were generated, short-circuit as well
    if not splits:
        return {"answer": "No chunked data to build local store.", "chunks_used": []}

    # 4) Build a local Chroma store from the chunked docs
    embeddings = get_openai_embeddings()  # or mock in tests
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, collection_name="rag_local_chunks"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k_chunks},
    )

    # 5) Pick a prompt template
    if question_type == "Bibliometric":
        template = bibliometric_prompt
    elif question_type == "Single-paper":
        template = single_paper_prompt
    elif question_type == "Broad but nuanced":
        template = deep_knowledge_prompt
    else:
        template = regular_prompt

    prompt_template = PromptTemplate.from_template(template)
    llm = get_openai_chat_llm(deployment_name="gpt-4o-mini", temperature=0.0)

    # Helper to join chunk contents
    def format_docs(chunks):
        return "\n\n".join(chunk.page_content for chunk in chunks)

    # Build short chain: retrieve local chunks => prompt => LLM => parse
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt_template
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # 6) Execute
    result = rag_chain_with_source.invoke(query)
    vectorstore.delete_collection()

    return {"answer": result["answer"], "chunks_used": splits[:top_k_chunks]}
