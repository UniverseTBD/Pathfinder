# src/evaluation/consensus_evaluation.py

import os
from typing import List, Literal

import instructor  # so we can patch OpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

# Import our config
from src.config import config

# Import the provider
from src.providers import get_openai_chat_llm

# Properly configured Instructor client for Azure OpenAI
def get_instructor_client():
    # For Azure OpenAI, we need to use the proper configuration format
    # Based on the Instructor documentation for Azure
    
    # Create a standard AzureOpenAI client
    from openai import AzureOpenAI
    
    # Use proper Azure OpenAI client configuration
    azure_client = AzureOpenAI(
        api_key=config["chat_api_key_4omini"],
        api_version="2024-02-01",  # Use the appropriate API version
        azure_endpoint=config["chat_base_url_4omini"],
    )
    
    # Use instructor.from_openai instead of patch
    patched_client = instructor.from_openai(azure_client)
    
    # Print details for debugging
    print(f"Configured instructor client with Azure OpenAI for {config['chat_deployment_name_4omini']}")
    
    return patched_client

# Create a function that uses instructor when it can, with fallback to regular LLM
def get_consensus(query, abstracts):
    try:
        # Try to use instructor with properly configured client
        instructor_client = get_instructor_client()
        
        # Build a prompt for the model
        prompt_text = f"""
        Query: {query}
        You will be provided with {len(abstracts)} scientific abstracts. Your task:
        1. If the query is a question, rewrite it as a statement.
        2. Choose one consensus level from the following options:
            - Strong Agreement Between Abstracts and Query
            - Moderate Agreement Between Abstracts and Query
            - Weak Agreement Between Abstracts and Query
            - No Clear Agreement/Disagreement Between Abstracts and Query
            - Weak Disagreement Between Abstracts and Query
            - Moderate Disagreement Between Abstracts and Query
            - Strong Disagreement Between Abstracts and Query
        3. Provide an explanation (up to six sentences).
        4. Assign a relevance score between 0.0 and 1.0.
        
        Here are the abstracts:
        {''.join(f"Abstract {i+1}: {abs_}\n" for i, abs_ in enumerate(abstracts))}
        """
        
        # Use instructor to get structured output for Azure OpenAI
        # Following the recommended pattern from the tutorial
        response = instructor_client.chat.completions.create(
            model=config["chat_deployment_name_4omini"],  # This is the deployment name
            response_model=OverallConsensusEvaluation,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ],
        )
        return response
        
    except Exception as e:
        # Fallback to regular LLM if instructor fails
        print(f"Instructor failed with error: {e}. Falling back to regular LLM approach.")
        llm = get_openai_chat_llm(deployment_name="gpt-4o-mini")
        
        # Build a prompt for regular LLM
        prompt = f"""
        Query: {query}
        You will be provided with {len(abstracts)} scientific abstracts. Your task:
        1. If the query is a question, rewrite it as a statement. Output as 'Rewritten Statement:'.
        2. Choose one consensus level from the following options. Output as 'Consensus:'.
            - Strong Agreement Between Abstracts and Query
            - Moderate Agreement Between Abstracts and Query
            - Weak Agreement Between Abstracts and Query
            - No Clear Agreement/Disagreement Between Abstracts and Query
            - Weak Disagreement Between Abstracts and Query
            - Moderate Disagreement Between Abstracts and Query
            - Strong Disagreement Between Abstracts and Query
        3. Provide an explanation (up to six sentences). Output as 'Explanation:'.
        4. Assign a relevance score between 0.0 and 1.0. Output as 'Relevance Score:'.
        
        Here are the abstracts:
        {''.join(f"Abstract {i+1}: {abs_}\n" for i, abs_ in enumerate(abstracts))}
        
        Format your response exactly like this:
        Rewritten Statement: [your rewritten statement]
        Consensus: [your consensus level choice]
        Explanation: [your explanation]
        Relevance Score: [your score between 0.0 and 1.0]
        """
        
        result = llm.invoke(prompt).content
        
        # Parse the result
        parts = result.split("\n")
        rewritten = next((p for p in parts if p.strip().startswith("Rewritten") or p.strip().startswith("Statement")), "")
        consensus = next((p for p in parts if p.strip().startswith("Consensus")), "")
        explanation = next((p for p in parts if p.strip().startswith("Explanation")), "")
        relevance = next((p for p in parts if p.strip().startswith("Relevance")), "")
        
        # Extract the score
        try:
            score = float(relevance.split(":")[-1].strip())
        except:
            score = 0.5  # Default
        
        # Create a result object like OverallConsensusEvaluation
        class ConsensusResult:
            def __init__(self, rewritten, consensus, explanation, relevance_score):
                self.rewritten_statement = rewritten
                self.consensus = consensus
                self.explanation = explanation
                self.relevance_score = relevance_score
        
        return ConsensusResult(
            rewritten.split(":", 1)[-1].strip() if ":" in rewritten else rewritten,
            consensus.split(":", 1)[-1].strip() if ":" in consensus else consensus,
            explanation.split(":", 1)[-1].strip() if ":" in explanation else explanation,
            score
        )


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
):
    """
    Evaluate the overall consensus between a user query and a list of scientific abstracts.

    Steps:
    1. Rewrite the query as a statement (if it's originally a question).
    2. Determine the consensus level across the abstracts.
    3. Provide an explanation (up to six sentences).
    4. Provide a relevance score in [0..1].

    Args:
        query (str): The original user question or statement.
        abstracts (List[str]): A list of scientific abstracts (strings).

    Returns:
        ConsensusResult: An object containing the consensus evaluation.

    Example usage:
        from src.evaluation.consensus_evaluation import evaluate_overall_consensus

        consensus = evaluate_overall_consensus(
            query="What is the role of dark matter in galaxy formation?",
            abstracts=["abstract1", "abstract2", ...]
        )
        print(consensus.consensus, consensus.explanation)
    """
    # Use our simpler implementation for now
    return get_consensus(query, abstracts)
