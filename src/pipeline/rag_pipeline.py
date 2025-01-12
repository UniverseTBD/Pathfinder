# src/pipeline/rag_pipeline.py
import os
from typing import List, Literal

import openai
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI as openai_llm
from pydantic import BaseModel, Field

from src.embeddings import EmbeddingService
from src.prompts import (
    bibliometric_prompt,
    deep_knowledge_prompt,
    question_categorization_prompt,
    regular_prompt,
    single_paper_prompt,
)


# If you have a separate "gen_llm" or "consensus_client", define them here:
def get_gen_llm(temp=0.0):
    return openai_llm(
        temperature=temp,
        model_name="gpt-4o-mini",
        openai_api_key=os.environ["openai_key"],
    )


class OverallConsensusEvaluation(BaseModel):
    rewritten_statement: str = Field(...)
    consensus: Literal[
        "Strong Agreement Between Abstracts and Query",
        "Moderate Agreement Between Abstracts and Query",
        "Weak Agreement Between Abstracts and Query",
        "No Clear Agreement/Disagreement Between Abstracts and Query",
        "Weak Disagreement Between Abstracts and Query",
        "Moderate Disagreement Between Abstracts and Query",
        "Strong Disagreement Between Abstracts and Query",
    ] = Field(...)
    explanation: str = Field(...)
    relevance_score: float = Field(ge=0, le=1)


def run_rag_qa(query: str, df: pd.DataFrame, question_type: str):
    """
    RAG pipeline: chunk docs, build a local vectorstore, run a chain.
    df is the top-k papers from retrieval.
    """
    # convert df to Documents
    docs = []
    for i, row in df.iterrows():
        content = f"Paper {i}: {row['title']}\n{row['abstract']}\n\n"
        metadata = {"source": row["ads_id"]}  # or row.get("ads_id")
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embed_service = EmbeddingService()
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embed_service.embeddings, collection_name="retdoc4"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    if question_type == "Bibliometric":
        template = bibliometric_prompt
    elif question_type == "Single-paper":
        template = single_paper_prompt
    elif question_type == "Broad but nuanced":
        template = deep_knowledge_prompt
    else:
        template = regular_prompt

    prompt = PromptTemplate.from_template(template)
    llm = get_gen_llm(temp=0.0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    result = rag_chain_with_source.invoke(query)
    vectorstore.delete_collection()

    return result  # typically a dict: { 'answer': '...' }


def guess_question_type(query: str):
    """
    Use a 0-temp LLM to classify the question type.
    """
    categorizer = get_gen_llm(temp=0.0)
    system_prompt = question_categorization_prompt
    messages = [("system", system_prompt), ("user", query)]
    resp = categorizer.invoke(messages)
    return resp.content


def evaluate_overall_consensus(
    query: str, abstracts: List[str]
) -> OverallConsensusEvaluation:
    """
    Creates a prompt for a specialized LLM that returns an OverallConsensusEvaluation pydantic object.
    """
    prompt = f"""
    Query: {query}
    You will be provided with {len(abstracts)} scientific abstracts. Your task is to do the following:
    1. If the provided query is a question...
    ...
    Here are the abstracts:
    {' '.join([f"Abstract {i+1}: {abstract}" for i, abstract in enumerate(abstracts)])}
    Provide your evaluation in the structured format described above.
    """

    # If you have a special "consensus_client", do that, or just a normal openai call
    # Here we do a direct openai call with a response_model
    # Might require extra setup or a "patch" for instructor if using it
    # For brevity, let's do standard openai chat:
    # openai.api_key = os.environ["openai_key"]

    # We'll just return raw text for now:
    # (If you want pydantic validation, parse the raw text into OverallConsensusEvaluation)

    messages = [
        {
            "role": "system",
            "content": "You are an assistant with expertise in astrophysics...",
        },
        {"role": "user", "content": prompt},
    ]

    # Basic chat:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", messages=messages, temperature=0
    )
    # parse it however you want
    text_out = response["choices"][0]["message"]["content"]
    return text_out  # or parse into pydantic if your usage requires it
