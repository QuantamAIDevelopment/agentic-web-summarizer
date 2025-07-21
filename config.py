# config.py
"""
Configuration settings for the Agentic Web Summarizer application.

This module centralizes configuration parameters to make the application
more maintainable and easier to customize.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Directory for storing data
DATA_DIR = os.path.join(BASE_DIR, "data")
SUMMARIES_DIR = os.path.join(DATA_DIR, "summaries")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# LLM Configuration
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-3.5-turbo",
        "env_var": "OPENAI_API_KEY"
    },
    "groq": {
        "name": "Groq",
        "default_model": "deepseek-r1-distill-llama-70b",
        "env_var": "GROQ_API_KEY"
    }
}

# Embedding Configuration
EMBEDDING_PROVIDERS = {
    "openai": {
        "name": "OpenAI Embeddings",
        "env_var": "OPENAI_API_KEY"
    },
    "huggingface": {
        "name": "HuggingFace Embeddings",
        "default_model": "all-MiniLM-L6-v2"
    }
}

# Summarization Configuration
SUMMARY_STYLES = {
    "default": "Standard paragraph summary",
    "bullet": "Bullet point summary",
}

# Default parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_CHARS = 4000