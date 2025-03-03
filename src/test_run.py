# src/test_run.py
# Simple test script to verify the main functionality works

import os
from src.retrieval.retrieval_system import RetrievalSystem
from src.pipeline.rag_pipeline import run_rag_qa

# Set up environment variables if needed
# (Make sure you've set these in your actual environment)
if "openai_key" not in os.environ:
    print("Warning: 'openai_key' not found in environment. Using placeholder.")
    os.environ["openai_key"] = "YOUR_OPENAI_KEY_HERE"

if "cohere_key" not in os.environ:
    print("Warning: 'cohere_key' not found in environment. Using placeholder.")
    os.environ["cohere_key"] = "YOUR_COHERE_KEY_HERE"

# Simple test for the retrieval system
def test_retrieval():
    print("Testing retrieval system...")
    try:
        # Create retrieval system
        retrieval = RetrievalSystem()
        
        # Test basic retrieval (without HyDE or reranking for speed)
        print("Running basic retrieval...")
        query = "What is the Hubble constant?"
        results, papers_df = retrieval.retrieve(
            query=query, 
            top_k=3,      # Small number for quick test
            use_hyde=False,
            use_rerank=False
        )
        
        # Print results
        print(f"Retrieved {len(papers_df)} papers for query: '{query}'")
        if len(papers_df) > 0:
            print("\nSample paper title:")
            print(papers_df["title"][0])
            print("\nRetrieval test successful!")
        else:
            print("Warning: No papers retrieved. Check dataset connection.")
        
        return papers_df
        
    except Exception as e:
        print(f"ERROR in retrieval test: {str(e)}")
        return None

# Simple test for the RAG pipeline
def test_rag(papers_df):
    if papers_df is None or len(papers_df) == 0:
        print("Skipping RAG test - no papers available.")
        return
    
    print("\nTesting RAG pipeline...")
    try:
        # Run RAG on retrieved papers
        print("Running RAG pipeline...")
        query = "What is the Hubble constant?"
        rag_result = run_rag_qa(
            query=query,
            papers_df=papers_df,
            question_type="Single-paper"
        )
        
        # Print answer
        print(f"\nRAG Answer:")
        print(rag_result["answer"][:200] + "..." if len(rag_result["answer"]) > 200 else rag_result["answer"])
        print("\nRAG test successful!")
        
    except Exception as e:
        print(f"ERROR in RAG test: {str(e)}")

if __name__ == "__main__":
    print("=== PATHFINDER SYSTEM TEST ===")
    
    # Test retrieval component
    papers_df = test_retrieval()
    
    # Test RAG pipeline with retrieved papers
    test_rag(papers_df)
    
    print("\n=== TEST COMPLETE ===")