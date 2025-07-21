"""
Agentic Web Summarizer - A Streamlit application that uses AI to summarize web content.

This application demonstrates the concept of AI agents by:
1. Fetching content from a web page
2. Using an LLM to generate a concise summary
3. Storing the summary in a vector database for potential future retrieval

The app showcases how to combine different AI capabilities (LLMs, embeddings)
with web scraping to create a useful tool.
"""

import sys
import os

# Workaround for Streamlit + PyTorch import bug
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Disable PyTorch warnings and fix torch._classes.__path__ issue
import warnings
warnings.filterwarnings("ignore")

# Fix torch._classes.__path__ issue before importing any other modules
try:
    import torch
    
    # Create a dummy __path__ attribute
    class DummyPath:
        _path = []
    
    # Replace the problematic module
    if hasattr(torch, "_classes"):
        torch._classes.__path__ = DummyPath()
        
        # Also replace __getattr__ to prevent any other issues
        original_getattr = getattr(torch._classes, "__getattr__", None)
        
        def safe_getattr(name):
            if name == "__path__":
                return DummyPath()
            if original_getattr:
                return original_getattr(name)
            raise AttributeError(f"module 'torch._classes' has no attribute '{name}'")
        
        torch._classes.__getattr__ = safe_getattr
        
    print("PyTorch patched successfully")
except ImportError:
    print("PyTorch not found, no need to patch")
except Exception as e:
    print(f"Failed to patch PyTorch: {e}")

# Fix for asyncio error in Streamlit
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import logging
from datetime import datetime
from typing import List, Tuple
import warnings

# Suppress PyTorch warnings that might interfere with Streamlit
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch._classes")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*torch._classes.*")

# Fix for asyncio error in Streamlit
import asyncio

# Always create a new event loop to avoid issues with PyTorch
asyncio.set_event_loop(asyncio.new_event_loop())

# Import our custom modules
from llm.llm_provider import get_llm
from tools.fetcher import fetch_url_content
from tools.summarizer import summarize_text
from tools.vectorstore import get_embeddings, create_faiss_store, store_summary, save_vectorstore

# Import agent modules
from agents.orchestrator_agent import OrchestratorAgent
from agents.summarization_agent import SummarizationAgent
from agents.research_agent import ResearchAgent
from agents.critique_agent import CritiqueAgent

# Import default keys configuration
from default_keys import USE_DEFAULT_KEYS

# Import database session and model for logging
from config.database import SessionLocal, init_db
from models.agent_decisions import AgentDecision
from models.agent_log_utils import log_agent_decision, log_agent_communication
from models.agent_message import AgentMessage, get_messages_for_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Agentic Web Summarizer",
    page_icon="🔍",
    layout="wide"
)

# Application title and description
st.title("🔍 Agentic Web Summarizer")
st.markdown("""
This tool demonstrates how AI agents can help process web content:
1. Enter a URL to fetch the content
2. The AI will generate a concise summary
3. The summary is stored in a vector database for future reference
""")

# Sidebar for configuration options
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys section
    with st.expander("API Keys", expanded=False):
        if USE_DEFAULT_KEYS:
            st.success("Default API keys are available. You can use your own keys or leave fields empty to use default keys.")
        else:
            st.info("API keys are stored in session state and not saved to disk")
        
        # Use default keys option
        use_default = st.checkbox(
            "Use default API keys when available", 
            value=USE_DEFAULT_KEYS,
            help="When checked, the application will use built-in API keys if you don't provide your own"
        )
        st.info(" If you are not selecting the default keys Choose any one LLM model and pass the API keys to It.")
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key for accessing GPT models"
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not use_default:
            st.warning("No OpenAI API key provided. Required for OpenAI models and embeddings.")
            
        # Groq API Key
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Your Groq API key for accessing Groq models"
        )
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not use_default:
            st.warning("No Groq API key provided. Required for Groq models.")
            
        # LangChain API Key for tracing
        langchain_api_key = st.text_input(
            "LangChain API Key",
            type="password",
            value=os.getenv("LANGCHAIN_API_KEY", ""),
            help="Your LangChain API key for tracing and monitoring"
        )
        if langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    
    # LLM provider selection
    provider = st.selectbox(
        "Choose LLM Provider",
        options=["openai", "groq"],
        help="Select which AI model provider to use for summarization"
    )
    
    # Embedding model selection
    embedder = st.selectbox(
        "Choose Embedding Model",
        options=["openai", "huggingface"],
        help="Select which embedding model to use for vector storage"
    )
    
    # Summary style selection
    summary_style = st.radio(
        "Summary Style",
        options=["default", "bullet"],
        help="Choose how the summary should be formatted"
    )
    
    # Advanced options in an expandable section
    with st.expander("Advanced Options"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        max_chars = st.number_input(
            "Max Input Characters",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=1000,
            help="Maximum number of characters to process from the webpage"
        )
        
        save_to_disk = st.checkbox(
            "Save Summaries to Disk",
            value=False,
            help="Store summaries in a local vector database for future reference"
        )
        
        # Model options based on provider
        if provider == "groq":
            groq_model = st.selectbox(
                "Groq Model",
                options=["deepseek-r1-distill-llama-70b", "llama3-70b-8192", "mixtral-8x7b-32768","llama-3.3-70b-versatile"],
                help="Select which Groq model to use"
            )
            model_kwargs = {"model": groq_model}
        elif provider == "openai":
            openai_model = st.selectbox(
                "OpenAI Model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                help="Select which OpenAI model to use"
            )
            model_kwargs = {"model": openai_model}
        else:
            model_kwargs = {}

# Function to generate reflection and reasoning chain (for logging)
def generate_summary_reflection(raw_text: str, summary: str) -> Tuple[(str, List[str])]:
    reflection_parts = []
    if len(raw_text) < 1000:
        reflection_parts.append("Source content was brief.")
    elif len(raw_text) > 10000:
        reflection_parts.append("Source content was long.")
    if len(summary) / len(raw_text) < 0.1:
        reflection_parts.append("Summary is concise.")
    elif len(summary) / len(raw_text) > 0.3:
        reflection_parts.append("Summary might be verbose.")
    if "error" in summary.lower() or "not available" in summary.lower():
        reflection_parts.append("Summary may be incomplete.")
    else:
        reflection_parts.append("Summary successfully generated.")
    reflection_parts.append("Recommendation: Review summarization if accuracy is critical.")
    reasoning_chain = [
        "Step 1: Fetched web content.",
        f"Step 2: Extracted {len(raw_text)} characters.",
        "Step 3: Generated summary using LLM.",
        f"Step 4: Reflection generated: {' '.join(reflection_parts)}"
    ]
    return " ".join(reflection_parts), reasoning_chain

# Main content area
url = st.text_input(
    "Enter a web page URL:",
    placeholder="https://example.com",
    help="Enter the full URL including https://"
)

# Initialize database tables
try:
    init_db()
    logger.info("Database tables initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {e}")
    st.error(f"Database initialization error: {e}. Some features may not work properly.")
    # Continue anyway to allow the app to run with limited functionality

# Add a tab for agent mode selection
tab1, tab2 = st.tabs(["Single Agent Mode", "Multi-Agent Mode"])

with tab1:
    st.markdown("### Single Agent Summarization")
    st.markdown("This mode uses a single agent to fetch and summarize web content.")
    
    # Process the URL when the button is clicked
    if st.button("Summarize (Single Agent)", type="primary", key="single_agent_button"):
        if not url:
            st.error("Please enter a valid URL")
        else:
            try:
                # Show a spinner while processing
                with st.spinner("Fetching and analyzing the web page..."):
                    # Step 1: Initialize the LLM and embedding model
                    llm = get_llm(provider, temperature=temperature, model_kwargs=model_kwargs)
                    embed_model = get_embeddings(embedder)
                    vectorstore = create_faiss_store(embed_model)
                    
                    # Step 2: Fetch the content from the URL
                    raw_text = fetch_url_content(url)
                    
                    # Step 3: Generate a summary using the LLM
                    summary = summarize_text(
                        text=raw_text,
                        llm=llm,
                        style=summary_style,
                        max_chars=max_chars
                    )

                    # Generate reflection and reasoning chain for logging
                    reflection, reasoning_chain = generate_summary_reflection(raw_text, summary)

                    # Log the agent decision (reflection and reasoning) to the database silently
                    log_agent_decision(
                        agent_name="summarization_agent",
                        input_data=url,
                        output_summary=summary,
                        reflection=reflection,
                        reasoning_chain=reasoning_chain
                    )
                    
                    # Step 4: Store the summary in the vector database
                    metadata = {
                        "source_url": url,
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider,
                        "style": summary_style
                    }
                    store_summary(vectorstore, summary, metadata)
                    
                    # Step 5: Save to disk if requested
                    if save_to_disk:
                        save_dir = os.path.join("data", "summaries")
                        os.makedirs(save_dir, exist_ok=True)
                        save_vectorstore(vectorstore, save_dir)
                        st.success(f"Summary saved to disk in {save_dir}")
                
                # Display the results
                st.subheader("📝 Summary")
                st.write(summary)
                
                # Show some stats
                st.info(f"Processed {len(raw_text)} characters from the webpage")
                
            except ValueError as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error processing URL {url}: {e}")
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.exception(f"Unexpected error processing URL {url}")

with tab2:
    st.markdown("### Multi-Agent Collaborative Summarization")
    st.markdown("This mode uses multiple specialized agents that work together to produce an enhanced summary.")
    
    # Add a checkbox to show agent interactions
    show_agent_interactions = st.checkbox("Show agent interactions", value=False)
    
    # Process the URL with multiple agents when the button is clicked
    if st.button("Summarize (Multi-Agent)", type="primary", key="multi_agent_button"):
        if not url:
            st.error("Please enter a valid URL")
        else:
            try:
                # Show a spinner while processing
                with st.spinner("Orchestrating multi-agent workflow..."):
                    # Initialize the LLM
                    llm = get_llm(provider, temperature=temperature, model_kwargs=model_kwargs)
                    
                    # Create the orchestrator agent
                    orchestrator = OrchestratorAgent(llm, embedder, verbose=False)
                    
                    # Process the URL through the multi-agent workflow
                    results = orchestrator.process_url(
                        url=url,
                        summary_style=summary_style,
                        max_chars=max_chars
                    )
                    
                    # Save to disk if requested
                    if save_to_disk:
                        save_dir = os.path.join("data", "summaries")
                        os.makedirs(save_dir, exist_ok=True)
                        save_vectorstore(orchestrator.vectorstore, save_dir)
                        st.success(f"Summary saved to disk in {save_dir}")
                
                # Display the results
                st.subheader("📝 Final Summary")
                st.write(results["final_summary"])
                
                # Show evaluation if available
                if "evaluation" in results and "scores" in results["evaluation"] and results["evaluation"]["scores"]:
                    st.subheader("📊 Summary Evaluation")
                    scores = results["evaluation"]["scores"]
                    
                    # Create columns for scores only if we have scores
                    if scores and len(scores) > 0:
                        cols = st.columns(len(scores))
                        for i, (criterion, score) in enumerate(scores.items()):
                            with cols[i]:
                                st.metric(criterion.capitalize(), f"{score}/10")
                    else:
                        st.info("No evaluation scores available")
                    
                    # Show feedback
                    if "feedback" in results["evaluation"] and results["evaluation"]["feedback"]:
                        st.info(f"Feedback: {results['evaluation']['feedback']}")
                
                # Show research results if available
                if "research" in results and "topics" in results["research"]:
                    st.subheader("🔍 Research Insights")
                    topics = results["research"].get("topics", [])
                    research_data = results["research"].get("research", {})
                    
                    for topic in topics:
                        if topic in research_data:
                            with st.expander(f"Topic: {topic}"):
                                st.write(research_data[topic])
                
                # Show agent interactions if requested
                if show_agent_interactions:
                    st.subheader("🤖 Agent Interactions")
                    
                    # Get recent messages
                    with SessionLocal() as session:
                        messages = session.query(AgentMessage).order_by(
                            AgentMessage.timestamp.desc()
                        ).limit(10).all()
                    
                    # Display messages
                    for msg in messages:
                        with st.expander(f"{msg.sender} → {msg.recipient} ({msg.message_type})"):
                            st.text(f"Time: {msg.timestamp}")
                            st.text(f"Status: {msg.status}")
                            if msg.in_reply_to:
                                st.text(f"In reply to: {msg.in_reply_to}")
                            st.json(msg.content)
                
                # Show some stats
                st.info(f"Processed {len(results['raw_text'])} characters from the webpage")
                
            except ValueError as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error processing URL {url}: {e}")
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.exception(f"Unexpected error processing URL {url}")

# Footer
st.markdown("---")
st.markdown(
    "Made By QAID Software Pvt Ltd. using Streamlit and LangChain"
)
