# src/run_pathfinder.py

import os
from typing import List, Optional

import pandas as pd

from src.consensus_evaluation import evaluate_overall_consensus
from src.pipeline.deep_research_pipeline import deep_research
from src.pipeline.rag_pipeline import run_rag_qa
from src.providers import get_openai_chat_llm
from src.retrieval.retrieval_system import RetrievalSystem


def guess_question_type(query: str) -> str:
    """
    Guess the best prompt type for a given query.
    
    Args:
        query: The query to categorize
        
    Returns:
        Suggested question type
    """
    from src.prompts import question_categorization_prompt
    
    llm = get_openai_chat_llm(deployment_name="gpt-4o-mini", temperature=0)
    messages = [
        ("system", question_categorization_prompt),
        ("human", query),
    ]
    result = llm.invoke(messages).content
    
    # Extract categorization from the result
    if "<categorization>" in result:
        result = result.split("<categorization>")[1]
    if "</categorization>" in result:
        result = result.split("</categorization>")[0]
    
    # Convert categorization to prompt type
    category_to_prompt = {
        "Summarization": "Multi-paper",
        "Single-paper factual": "Single-paper",
        "Multi-paper factual": "Multi-paper",
        "Named entity recognition": "Multi-paper",
        "Jargon-specific questions / overloaded words": "Single-paper",
        "Time-sensitive": "Multi-paper",
        "Consensus evaluation": "Broad but nuanced",
        "What-ifs and counterfactuals": "Broad but nuanced",
        "Compositional": "Deep Research (BETA)"
    }
    
    # Try to extract category from result
    for category in category_to_prompt:
        if category in result:
            return category_to_prompt[category]
    
    # Default to Multi-paper if we can't determine
    return "Multi-paper"


def run_pathfinder(
    query: str,
    top_k: int = 10,
    extra_keywords: str = "",
    toggles: Optional[List[str]] = None,
    prompt_type: str = "Auto",
    rag_type: str = "Semantic + HyDE + CoHERE"
) -> dict:
    """
    Main function to run the Pathfinder system.
    
    Args:
        query: The query to research
        top_k: Number of papers to retrieve
        extra_keywords: Additional keywords to use in search
        toggles: Weight toggles ["Keywords", "Time", "Citations"]
        prompt_type: Prompt type to use
        rag_type: Retrieval method to use
        
    Returns:
        A dictionary with the results
    """
    toggles = toggles or ["Keywords"]
    
    # Initialize retrieval system
    retrieval_system = RetrievalSystem()
    
    # Extract keywords from the extra_keywords string
    input_keywords = [kw.strip() for kw in extra_keywords.split(",")] if extra_keywords else []
    retrieval_system.query_input_keywords = input_keywords
    
    # Configure retrieval settings
    use_hyde = "HyDE" in rag_type
    use_rerank = "CoHERE" in rag_type
    
    # Auto-detect prompt type if needed
    if prompt_type == "Auto":
        prompt_type = guess_question_type(query)
    
    # Deep research pipeline for complex questions
    if prompt_type == "Deep Research (BETA)":
        papers_df, rag_answer = deep_research(
            question=query,
            top_k=top_k,
            retrieval_system=retrieval_system
        )
        return {
            "papers": papers_df,
            "answer": rag_answer["answer"],
            "question_type": prompt_type,
            "consensus": None,
        }
    
    # Standard retrieval and RAG pipeline
    else:
        # Retrieve papers based on query
        rs, small_df = retrieval_system.retrieve(
            query=query,
            top_k=top_k,
            toggles=toggles,
            use_hyde=use_hyde,
            use_rerank=use_rerank,
            return_scores=True
        )
        
        # Format papers dataframe
        papers_df = pd.DataFrame(small_df)
        if not papers_df.empty:
            if "bibcode" in papers_df.columns:
                # Add ADS links if available
                links = [
                    f"[{i}](https://ui.adsabs.harvard.edu/abs/{i}/abstract)"
                    for i in papers_df["bibcode"]
                ]
                papers_df.insert(1, "ADS Link", links, True)
            
            # Add relevance scores
            scores = [rs.get(idx, 0.0) for idx in papers_df.indices] if hasattr(papers_df, 'indices') else [0.0] * len(papers_df)
            papers_df.insert(0, "Relevance", scores, True)
            
            # Reset index
            papers_df.reset_index(drop=True, inplace=True)
            papers_df.index = papers_df.index + 1  # 1-based indexing
        
        # Generate answer using RAG
        rag_result = run_rag_qa(
            query=query,
            papers_df=papers_df,
            question_type=prompt_type
        )
        
        # Generate consensus evaluation
        consensus = None
        if not papers_df.empty and "abstract" in papers_df.columns:
            abstracts = papers_df["abstract"].tolist()
            consensus_result = evaluate_overall_consensus(query, abstracts)
            consensus = f"## Consensus\n{consensus_result.consensus}\n\n{consensus_result.explanation}\n\n> Relevance of retrieved papers: {consensus_result.relevance_score:.1f}"
        
        return {
            "papers": papers_df,
            "answer": rag_result["answer"],
            "question_type": prompt_type,
            "consensus": consensus,
        }


if __name__ == "__main__":
    # Simple command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Pathfinder for astronomy research")
    parser.add_argument("query", help="The research query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of papers to retrieve")
    parser.add_argument("--keywords", default="", help="Additional keywords (comma-separated)")
    parser.add_argument("--toggles", nargs="*", default=["Keywords"], choices=["Keywords", "Time", "Citations"], 
                      help="Weight toggles")
    parser.add_argument("--prompt-type", default="Auto", 
                      choices=["Auto", "Single-paper", "Multi-paper", "Bibliometric", "Broad but nuanced", "Deep Research (BETA)"],
                      help="Prompt type")
    parser.add_argument("--rag-type", default="Semantic + HyDE + CoHERE",
                      choices=["Semantic Search", "Semantic + HyDE", "Semantic + CoHERE", "Semantic + HyDE + CoHERE"],
                      help="Retrieval method")
    
    args = parser.parse_args()
    
    result = run_pathfinder(
        query=args.query,
        top_k=args.top_k,
        extra_keywords=args.keywords,
        toggles=args.toggles,
        prompt_type=args.prompt_type,
        rag_type=args.rag_type
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"QUERY: {args.query}")
    print(f"QUESTION TYPE: {result['question_type']}")
    print("="*80 + "\n")
    
    print("ANSWER:")
    print(result["answer"])
    print("\n" + "="*80 + "\n")
    
    if result["consensus"]:
        print("CONSENSUS:")
        print(result["consensus"])
        print("\n" + "="*80 + "\n")
    
    print(f"PAPERS RETRIEVED: {len(result['papers'])}")
    if not result["papers"].empty:
        pd.set_option('display.max_colwidth', 80)
        print(result["papers"][["Relevance", "title"]].head(5))