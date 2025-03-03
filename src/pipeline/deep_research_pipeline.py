# src/pipeline/deep_research_pipeline.py

from typing import List, Tuple

import pandas as pd

# Local imports
from src.pipeline.rag_pipeline import run_rag_qa
from src.providers import get_openai_chat_llm
from src.retrieval.retrieval_system import RetrievalSystem


def decompose_question(question: str) -> List[str]:
    """
    Decomposes a complex research question into atomic sub-questions.
    
    Args:
        question: The complex research question to decompose
        
    Returns:
        List of atomic questions
    """
    llm = get_openai_chat_llm(deployment_name="gpt-4o-mini", temperature=0)
    
    # Prompt to decompose questions
    prompt = """You are an expert researcher in astronomy and astrophysics. 
    Your task is to break down a complex research question into clear, atomic sub-questions
    that can be answered individually to form a comprehensive response.
    
    Complex question: {question}
    
    Provide 3-5 atomic questions that would help answer this complex question.
    Each question should be focused on a specific aspect of the complex question.
    Return ONLY the list of questions, one per line, without numbering or bullets.
    """
    
    messages = [
        ("system", prompt.format(question=question)),
        ("human", question),
    ]
    
    result = llm.invoke(messages).content
    
    # Parse the result into a list of questions
    atomic_questions = [q.strip() for q in result.split('\n') if q.strip()]
    return atomic_questions


def get_small_answer(question: str, papers_df: pd.DataFrame) -> str:
    """
    Generate a concise answer for a specific atomic question based on papers.
    
    Args:
        question: The atomic question to answer
        papers_df: DataFrame containing paper information
        
    Returns:
        Answer to the atomic question
    """
    # Use the RAG pipeline to get the answer
    rag_result = run_rag_qa(
        query=question,
        papers_df=papers_df,
        question_type="Single-paper",  # Using single-paper for atomic questions
    )
    
    return rag_result["answer"]


def compile_information(
    original_question: str, 
    atomic_questions: List[str], 
    atomic_answers: List[str]
) -> str:
    """
    Compile information from atomic question answers into a comprehensive response.
    
    Args:
        original_question: The original complex question
        atomic_questions: List of atomic questions
        atomic_answers: List of answers to atomic questions
        
    Returns:
        Compiled comprehensive answer
    """
    llm = get_openai_chat_llm(deployment_name="gpt-4o-mini", temperature=0)
    
    # Build the context from atomic questions and answers
    context = "Here is information gathered to answer your question:\n\n"
    for q, a in zip(atomic_questions, atomic_answers):
        context += f"Question: {q}\nAnswer: {a}\n\n"
    
    # Prompt to synthesize answers
    prompt = """You are an expert astronomer synthesizing information to answer a complex research question.
    Below are several sub-questions and their answers that relate to the main question.
    
    Main question: {question}
    
    {context}
    
    Based on the information above, provide a comprehensive, well-structured answer to the main question.
    Synthesize the information logically, highlight key findings, and acknowledge any uncertainties or areas where more research is needed.
    """
    
    messages = [
        ("system", prompt.format(question=original_question, context=context)),
        ("human", "Please synthesize a comprehensive answer based on the information provided."),
    ]
    
    result = llm.invoke(messages).content
    return result


def deep_research(
    question: str, 
    top_k: int = 10, 
    retrieval_system: RetrievalSystem = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Performs deep research on a complex question by breaking it down into atomic questions,
    retrieving relevant papers for each, generating answers, and synthesizing a comprehensive response.
    
    Args:
        question: The complex research question
        top_k: Number of papers to retrieve for each atomic question
        retrieval_system: RetrievalSystem instance (will create one if None)
        
    Returns:
        Tuple of (combined dataframe of papers, result dictionary with answer)
    """
    if retrieval_system is None:
        retrieval_system = RetrievalSystem()
    
    # Build full answer with the original question as header
    full_answer = f"## {question}\n\n"
    
    # Step 1: Decompose the question into atomic sub-questions
    atomic_questions = decompose_question(question)
    
    # Step 2: Research each atomic question
    atomic_dataframes = []
    atomic_answers = []
    
    for atomic_q in atomic_questions:
        # Retrieve papers relevant to this atomic question
        rs, small_df = retrieval_system.retrieve(
            query=atomic_q, 
            top_k=top_k, 
            return_scores=True
        )
        
        # Convert to DataFrame with proper formatting
        formatted_df = pd.DataFrame(small_df)
        # Add scores and other relevant info
        formatted_df.insert(0, "Relevance", [rs.get(idx, 0.0) for idx in small_df.indices], True)
        
        # Store the DataFrame
        atomic_dataframes.append(formatted_df)
        
        # Generate an answer for this atomic question
        answer = get_small_answer(atomic_q, formatted_df)
        atomic_answers.append(answer)
        
        # Add to the full answer
        full_answer += f"### {atomic_q}\n{answer}\n\n"
    
    # Step 3: Compile information into a comprehensive response
    final_summary = compile_information(question, atomic_questions, atomic_answers)
    full_answer += f"### Summary\n{final_summary}\n"
    
    # Combine all dataframes while avoiding duplicate papers
    if atomic_dataframes:
        combined_df = pd.concat(atomic_dataframes, ignore_index=True).drop_duplicates(subset=["title"])
        combined_df.reset_index(drop=True, inplace=True)
        combined_df.index = combined_df.index + 1  # 1-based indexing
    else:
        combined_df = pd.DataFrame()
    
    return combined_df, {"answer": full_answer}