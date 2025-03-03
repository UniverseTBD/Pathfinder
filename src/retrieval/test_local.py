# src/retrieval/test_local.py
# Minimal test that can be run from the retrieval directory

import os
import sys

# Add parent directory to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.retrieval_system import RetrievalSystem

# Set up environment variables if needed
if "openai_key" not in os.environ:
    print("Warning: 'openai_key' not found in environment. Using placeholder.")
    os.environ["openai_key"] = "sk-..."  # Replace with your OpenAI key

if "cohere_key" not in os.environ:
    print("Warning: 'cohere_key' not found in environment. Using placeholder.")
    os.environ["cohere_key"] = "co-..."  # Replace with your Cohere key

def main():
    print("=== PATHFINDER BASIC TEST ===")
    
    try:
        # Try to initialize the retrieval system
        print("Initializing RetrievalSystem...")
        retrieval = RetrievalSystem()
        print("Successfully initialized RetrievalSystem!")
        
        # Test basic retrieval functionality
        print("\nTesting basic retrieval (this might take a moment)...")
        query = "What is dark matter?"
        results, papers_df = retrieval.retrieve(
            query=query,
            top_k=2,
            use_hyde=False,
            use_rerank=False
        )
        
        # Check results
        print(f"Retrieved {len(papers_df)} papers for query: '{query}'")
        if len(papers_df) > 0:
            print("\nFirst paper title:")
            print(papers_df["title"][0])
            print("\nBasic retrieval test PASSED!")
        else:
            print("\nWarning: No papers retrieved, but retrieval didn't fail with an error.")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()