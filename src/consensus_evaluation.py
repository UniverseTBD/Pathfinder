# src/evaluation/consensus_evaluation.py

import os
from typing import List, Literal

import instructor  # so we can patch OpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

# If you store your OpenAI key in an environment variable:
openai_key = os.environ.get("openai_key", "YOUR_OPENAI_KEY")

# Patch the OpenAI object with instructor
consensus_client = instructor.patch(OpenAI(api_key=openai_key))


class OverallConsensusEvaluation(BaseModel):
    """
    Data model that represents the overall consensus between a user query
    (rewritten as a statement) and a set of scientific abstracts.
    """

    rewritten_statement: str = Field(
        ...,
        description="The query rewritten as a statement if it was originally a question.",
    )
    consensus: Literal[
        "Strong Agreement Between Abstracts and Query",
        "Moderate Agreement Between Abstracts and Query",
        "Weak Agreement Between Abstracts and Query",
        "No Clear Agreement/Disagreement Between Abstracts and Query",
        "Weak Disagreement Between Abstracts and Query",
        "Moderate Disagreement Between Abstracts and Query",
        "Strong Disagreement Between Abstracts and Query",
    ] = Field(
        ...,
        description="The overall level of consensus between the rewritten statement and the abstracts.",
    )
    explanation: str = Field(
        ...,
        description="A brief explanation (max ~6 sentences) supporting the chosen consensus.",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A score in [0..1], indicating how relevant the abstracts are to the query.",
    )


def evaluate_overall_consensus(
    query: str, abstracts: List[str]
) -> OverallConsensusEvaluation:
    """
    Evaluate the overall consensus between a user query and a list of scientific abstracts,
    returning an OverallConsensusEvaluation object. Uses the 'consensus_client' (instructor-patched OpenAI)
    to generate the structured response.

    Steps:
    1. Rewrite the query as a statement (if it's originally a question).
    2. Determine the consensus level across the abstracts.
    3. Provide an explanation (up to six sentences).
    4. Provide a relevance score in [0..1].

    Args:
        query (str): The original user question or statement.
        abstracts (List[str]): A list of scientific abstracts (strings).

    Returns:
        OverallConsensusEvaluation: The final structured evaluation.

    Example usage:
        from src.evaluation.consensus_evaluation import evaluate_overall_consensus

        consensus = evaluate_overall_consensus(
            query="What is the role of dark matter in galaxy formation?",
            abstracts=["abstract1", "abstract2", ...]
        )
        print(consensus.consensus, consensus.explanation)
    """

    # Build a prompt to instruct the LLM how to respond
    # (Mirrors the style of the original code)
    prompt = f"""
    Query: {query}
    You will be provided with {len(abstracts)} scientific abstracts. Your task:
    1. If the query is a question, rewrite it as a statement. Output as 'Rewritten Statement:'.
    2. Choose one consensus level:
        - Strong/Moderate/Weak Agreement
        - No Clear Agreement/Disagreement
        - Strong/Moderate/Weak Disagreement
       Output as 'Consensus:'.
    3. Provide an explanation (up to six sentences). Output as 'Explanation:'.
    4. Assign a relevance score [0..1]. Output as 'Relevance Score:'.
    Here are the abstracts:
    {''.join(f"Abstract {i+1}: {abs_}\n" for i, abs_ in enumerate(abstracts))}

    Return your answer in the structured format:
      Rewritten Statement: ...
      Consensus: ...
      Explanation: ...
      Relevance Score: ...
    """

    # Use the instructor-patched OpenAI client
    # Note: model="gpt-4o-mini" or any model you prefer
    # 'response_model=OverallConsensusEvaluation' is how 'instructor' can parse it into pydantic
    response = consensus_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=OverallConsensusEvaluation,
        messages=[
            {
                "role": "system",
                "content": """You are an assistant with expertise in astrophysics for question-answering tasks.
                Evaluate the overall consensus among the provided abstracts regarding a given query.
                If uncertain, say you don't know. Keep your explanation concise.""",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response
