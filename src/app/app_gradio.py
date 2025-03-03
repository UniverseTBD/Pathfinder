#!/usr/bin/env python3
# src/app/app_gradio.py

import os
import sys
import pandas as pd
import gradio as gr

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import run_pathfinder
from src.run_pathfinder import run_pathfinder

# Define the available prompt types and RAG types
PROMPT_TYPES = [
    "Auto", 
    "Single-paper", 
    "Multi-paper", 
    "Bibliometric", 
    "Broad but nuanced", 
    "Deep Research (BETA)"
]

RAG_TYPES = [
    "Semantic Search", 
    "Semantic + HyDE", 
    "Semantic + CoHERE", 
    "Semantic + HyDE + CoHERE"
]

TOGGLES = ["Keywords", "Time", "Citations"]

def format_paper_row(row):
    """Format a paper row for display in the UI"""
    title = row.get('title', 'No Title')
    authors = row.get('authors', 'Unknown Authors')
    year = row.get('date', 'Unknown Date')
    if hasattr(year, 'year'):
        year = year.year
    
    # Format with relevance score
    score = row.get('Relevance', 0.0)
    return f"**[{score:.2f}] {title}** ({year})\n{authors}\n\n"
    
def run_search(
    query, 
    top_k, 
    extra_keywords, 
    prompt_type, 
    rag_type, 
    use_keywords, 
    use_time, 
    use_citations
):
    """Run Pathfinder search with the given parameters"""
    if not query:
        return "Please enter a query", "", "", ""
    
    # Prepare toggles based on checkboxes
    toggles = []
    if use_keywords:
        toggles.append("Keywords")
    if use_time:
        toggles.append("Time")
    if use_citations:
        toggles.append("Citations")
    
    try:
        # Run Pathfinder
        result = run_pathfinder(
            query=query,
            top_k=top_k,
            extra_keywords=extra_keywords,
            toggles=toggles,
            prompt_type=prompt_type,
            rag_type=rag_type
        )
        
        # Format papers
        papers_text = ""
        if not result["papers"].empty:
            papers_text = "## Retrieved Papers\n\n"
            for _, row in result["papers"].iterrows():
                papers_text += format_paper_row(row)
        
        # Get answer and consensus
        answer = result["answer"]
        consensus = result["consensus"] if result["consensus"] else ""
        detected_type = f"**Detected Question Type:** {result['question_type']}"
        
        return answer, papers_text, consensus, detected_type
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

# Create Gradio interface
with gr.Blocks(title="Pathfinder - Astronomy Research Assistant") as demo:
    gr.Markdown("# =- Pathfinder")
    gr.Markdown("## Semantic Research Assistant for Astronomy")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Research Question",
                placeholder="What is the current evidence for water on Mars?",
                lines=3
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    top_k_slider = gr.Slider(
                        minimum=1, 
                        maximum=20, 
                        value=5, 
                        step=1, 
                        label="Number of papers"
                    )
                
                with gr.Column(scale=2):
                    keywords_input = gr.Textbox(
                        label="Additional Keywords (comma-separated)",
                        placeholder="e.g., atmosphere, methane, ice",
                        lines=1
                    )
            
            with gr.Row():
                with gr.Column():
                    prompt_type_dropdown = gr.Dropdown(
                        choices=PROMPT_TYPES,
                        value="Auto",
                        label="Question Type"
                    )
                    
                with gr.Column():
                    rag_type_dropdown = gr.Dropdown(
                        choices=RAG_TYPES,
                        value="Semantic + HyDE + CoHERE",
                        label="Retrieval Method"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Sorting Factors")
                    with gr.Row():
                        keywords_toggle = gr.Checkbox(label="Keywords", value=True)
                        time_toggle = gr.Checkbox(label="Recent papers", value=False)
                        citations_toggle = gr.Checkbox(label="High citations", value=False)
                
                with gr.Column():
                    search_button = gr.Button("= Research", variant="primary")
    
    gr.Markdown("---")
    
    with gr.Tab("Results"):
        detected_type_md = gr.Markdown()
        answer_text = gr.Markdown(label="Answer")
        consensus_text = gr.Markdown(label="Consensus")
        papers_markdown = gr.Markdown(label="Papers")
    
    # Connect the button to the search function
    search_button.click(
        fn=run_search,
        inputs=[
            query_input,
            top_k_slider,
            keywords_input,
            prompt_type_dropdown,
            rag_type_dropdown,
            keywords_toggle,
            time_toggle,
            citations_toggle
        ],
        outputs=[
            answer_text,
            papers_markdown,
            consensus_text,
            detected_type_md
        ]
    )
    
    # Add example queries
    gr.Examples(
        [
            ["What is dark matter?"],
            ["What is the evidence for exoplanet atmospheres?"],
            ["How do neutron stars form?"],
            ["Explain the difference between Type Ia and Type II supernovae."],
            ["What has the James Webb Space Telescope discovered so far?"]
        ],
        inputs=[query_input]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()