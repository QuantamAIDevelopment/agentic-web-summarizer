# cli.py
"""
Command-line interface for the Agentic Web Summarizer.

This script provides a simple CLI for summarizing web pages without
needing to run the full Streamlit application.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

from langsmith import expect
from config.database import init_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from llm.llm_provider import get_llm
from tools.fetcher import fetch_url_content
from tools.summarizer import summarize_text
from tools.vectorstore import get_embeddings, create_faiss_store, store_summary, save_vectorstore
from agents.summarization_agent import SummarizationAgent

def main():
    """Main entry point for the CLI application."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Summarize web pages from the command line")
    parser.add_argument("url", help="URL of the web page to summarize")
    parser.add_argument(
        "--provider", 
        choices=["openai", "groq"], 
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--style", 
        choices=["default", "bullet"], 
        default="default",
        help="Summary style (default: default)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="LLM temperature (0.0-1.0, default: 0.0)"
    )
    parser.add_argument(
        "--save", 
        action="store_true",
        help="Save the summary to disk"
    )
    parser.add_argument(
        "--agent", 
        action="store_true",
        help="Use the agent-based approach"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed processing information"
    )
    
    args = parser.parse_args()
    
    try:
        # Check for required API keys
        if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is required")
            sys.exit(1)
        elif args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
            print("Error: GROQ_API_KEY environment variable is required")
            sys.exit(1)
        
        # Initialize the LLM
        llm = get_llm(args.provider, temperature=args.temperature)
        
        if args.agent:
            # Use the agent-based approach
            print(f"Using {args.provider} agent to summarize: {args.url}")
            agent = SummarizationAgent(llm, verbose=args.verbose)
            result = agent.run(args.url, {"style": args.style})
            summary = result["summary"]
        else:
            # Use the direct approach
            print(f"Fetching content from: {args.url}")
            raw_text = fetch_url_content(args.url)
            
            print(f"Generating summary using {args.provider}...")
            summary = summarize_text(raw_text, llm, style=args.style)
            
            # Store in vector database if saving
            if args.save:
                embed_model = get_embeddings("openai")
                vectorstore = create_faiss_store(embed_model)
                metadata = {
                    "source_url": args.url,
                    "timestamp": datetime.now().isoformat(),
                    "provider": args.provider,
                    "style": args.style
                }
                store_summary(vectorstore, summary, metadata)
                
                # Save to disk
                save_dir = os.path.join("data", "summaries")
                os.makedirs(save_dir, exist_ok=True)
                save_vectorstore(vectorstore, save_dir)
                print(f"Summary saved to: {save_dir}")
        
        # Print the summary
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(summary)
        print("=" * 40)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        init_db()
        logger.info("Database Initialized successfully")
    except Exception as e:
        logger.error("Database initialization failed: {e}")
        raise
    main()