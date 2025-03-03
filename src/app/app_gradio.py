#!/usr/bin/env python3
# src/app/app_gradio.py - Pathfinder-Lite interface

import os
import sys
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

# Define some CSS for a dark theme with consistent fonts
custom_css = """
:root {
    --primary-color: #7c4dff;
    --primary-hover: #651fff;
    --bg-color: #121212;
    --card-bg: #1e1e1e;
    --text-color: #e0e0e0;
    --secondary-text: #a0a0a0;
    --border-color: #333333;
    --accent-color: #bb86fc;
}

body, .gradio-container {
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
}

.dark {
    color-scheme: dark;
}

h1, h2, h3, h4, p, label, button {
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

h1 {
    color: var(--text-color);
    margin-bottom: 5px;
    font-weight: 600;
    font-size: 2.2rem;
}

h2 {
    color: var(--text-color);
    margin-top: 0px;
    font-weight: 500;
    font-size: 1.5rem;
}

h3 {
    color: var(--text-color);
    font-weight: 500;
    font-size: 1.2rem;
}

.gradio-button.primary {
    background-color: var(--primary-color);
    border: none;
}

.gradio-button.primary:hover {
    background-color: var(--primary-hover);
}

.footer {
    text-align: center;
    font-size: 0.9rem;
    margin-top: 30px;
    color: var(--secondary-text);
    padding: 20px 0;
    border-top: 1px solid var(--border-color);
}

.powered-by {
    font-weight: 500;
    color: var(--accent-color);
}

#input-container, #results-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
    border: 1px solid var(--border-color);
}

.header-accent {
    color: var(--accent-color);
    font-weight: 600;
}

/* Tab styling */
.tabs {
    border-bottom-color: var(--border-color);
}

.tab-button {
    color: var(--secondary-text);
}

.tab-button.selected {
    color: var(--accent-color);
    border-color: var(--accent-color);
}

/* Input elements */
input, textarea, select {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

/* Make sure text is visible in all states */
::placeholder {
    color: var(--secondary-text) !important;
    opacity: 0.7;
}

/* Improve markdown readability */
.md p, .md li {
    color: var(--text-color);
    font-size: 1rem;
    line-height: 1.6;
}

.md a {
    color: var(--accent-color);
}

/* Checkbox and slider styling */
input[type="checkbox"] {
    accent-color: var(--accent-color);
}

/* Better visibility for paper listings */
#papers strong {
    color: var(--accent-color);
}

/* Enhanced search button */
#search-button {
    background-color: var(--primary-color);
    border: none;
    padding: 12px 24px;
    font-size: 1.1rem;
    font-weight: 500;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    margin-top: 10px;
}

#search-button:hover {
    background-color: var(--primary-hover);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

/* Better query input */
#query-input textarea {
    font-size: 1.1rem;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color) !important;
    transition: all 0.3s ease;
}

#query-input textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(124, 77, 255, 0.2);
}

/* Better spacing for the whole UI */
#input-container {
    margin-top: 20px;
}

#search-row {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

/* Make tabs more distinct */
#results-tabs .tab-button {
    font-size: 1.05rem;
    padding: 10px 20px;
    margin-right: 8px;
    border-radius: 8px 8px 0 0;
}

#results-tabs .tab-button.selected {
    background-color: rgba(187, 134, 252, 0.1);
}

/* Better markdown rendering in results */
#answer, #papers, #consensus {
    line-height: 1.6;
    font-size: 1.05rem;
}
"""

def format_paper_row(row):
    """Format a paper row for display in the UI"""
    title = row.get('title', 'No Title')
    authors = row.get('authors', 'Unknown Authors')
    year = row.get('date', 'Unknown Date')
    if hasattr(year, 'year'):
        year = year.year
    
    # Format with relevance score
    score = row.get('Relevance', 0.0)
    return f"### **[{score:.2f}] {title}**\n**Year:** {year}  \n**Authors:** {authors}\n\n---\n\n"
    
def run_search(query, top_k, extra_keywords, prompt_type, rag_type, use_keywords, use_time, use_citations):
    """Run Pathfinder search with the given parameters"""
    if not query:
        return "Please enter a query", "", "", ""
    
    # Set toggles based on user selection
    toggles = []
    if use_keywords:
        toggles.append("Keywords")
    if use_time:
        toggles.append("Time")
    if use_citations:
        toggles.append("Citations")
    
    # Default to Keywords if no toggles selected
    if not toggles:
        toggles = ["Keywords"]
    
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
            # Make sure papers are sorted by relevance score (descending)
            sorted_papers = result["papers"].copy()
            if "Relevance" in sorted_papers.columns:
                sorted_papers = sorted_papers.sort_values("Relevance", ascending=False)
            
            papers_text = "## Retrieved Papers (Sorted by Relevance)\n\n"
            for _, row in sorted_papers.iterrows():
                papers_text += format_paper_row(row)
        
        # Get answer and consensus
        answer = result["answer"]
        consensus = result["consensus"] if result["consensus"] else ""
        detected_type = f"**Detected Question Type:** {result['question_type']}"
        
        return answer, papers_text, consensus, detected_type
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

# Create a modern, dark theme interface
with gr.Blocks(title="Pathfinder-Lite by UniverseTBD", css=custom_css, theme="dark") as demo:
    # Clean, modern header
    with gr.Row(elem_id="header"):
        with gr.Column(scale=8):
            gr.Markdown("# Pathfinder<span class='header-accent'>-Lite</span> 🔭")
            gr.Markdown("## Semantic Research Assistant for Astronomy")
        with gr.Column(scale=2, elem_id="branding"):
            gr.Markdown("by **UniverseTBD**")
    
    # Input area container
    with gr.Group(elem_id="input-container"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Research Question",
                    placeholder="What is dark matter?",
                    lines=3,
                    elem_id="query-input",
                    scale=1
                )
        
        with gr.Accordion("Search Options", open=True):
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
                        placeholder="e.g., galaxy, rotation",
                        lines=1
                    )
            
            with gr.Row():
                with gr.Column():
                    prompt_type_dropdown = gr.Dropdown(
                        choices=PROMPT_TYPES,
                        value="Auto",
                        label="Question Type",
                        info="Use 'Deep Research (BETA)' for complex questions requiring multi-step analysis"
                    )
                
                with gr.Column():
                    rag_type_dropdown = gr.Dropdown(
                        choices=RAG_TYPES,
                        value="Semantic + HyDE + CoHERE",
                        label="Retrieval Method"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Ranking Factors")
                    with gr.Row():
                        keywords_toggle = gr.Checkbox(label="Keywords", value=True)
                        time_toggle = gr.Checkbox(label="Recent papers", value=False)
                        citations_toggle = gr.Checkbox(label="High citations", value=False)
        
        with gr.Row(elem_id="search-row"):
            search_button = gr.Button("🔍 Search", variant="primary", size="lg", elem_id="search-button")
    
    # Output area
    with gr.Group(elem_id="results-container"):
        with gr.Tabs(elem_id="results-tabs"):
            with gr.TabItem("Answer", elem_id="answer-tab"):
                detected_type_md = gr.Markdown(elem_id="detected_type")
                answer_text = gr.Markdown(elem_id="answer")
            
            with gr.TabItem("Papers", elem_id="papers-tab"):
                papers_markdown = gr.Markdown(elem_id="papers")
            
            with gr.TabItem("Consensus", elem_id="consensus-tab"):
                consensus_text = gr.Markdown(elem_id="consensus")
    
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
    
    # Deep Research Example
    with gr.Accordion("Try Deep Research", open=False):
        gr.Markdown("""
        ### Deep Research Examples
        
        Try complex research questions that benefit from multi-step analysis:
        
        - How do stellar winds affect exoplanet atmospheres and their habitability?
        - What's the relationship between black hole mass and galaxy evolution?
        - How do planetary migration theories explain the diversity of exoplanet systems?
        
        For these questions, select "Deep Research (BETA)" from the Question Type dropdown.
        """)
    
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
    
    # Clean, minimalist footer
    gr.Markdown(
        """<div class="footer">
        <p><span class="powered-by">Powered by UniverseTBD</span></p>
        <p>A lightweight interface for semantic research in astronomy</p>
        </div>""",
        elem_id="footer"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()