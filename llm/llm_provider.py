# llm/llm_provider.py
"""
This module provides access to different Large Language Models (LLMs) through LangChain.
LangChain serves as the abstraction layer that standardizes interactions with various LLM providers,
allowing the application to work with any supported model through a consistent interface.
"""

import os
import logging
from typing import Optional, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseLanguageModel
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Import default keys
from default_keys import (
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_LANGCHAIN_API_KEY,
    USE_DEFAULT_KEYS
)

logger = logging.getLogger(__name__)

def setup_tracing(project_name: str = "agentic-web-summarizer") -> Optional[CallbackManager]:
    """
    Sets up LangChain tracing if LANGCHAIN_API_KEY is available.
    
    Args:
        project_name (str): Name of the project for LangChain tracing
        
    Returns:
        Optional[CallbackManager]: Callback manager with tracers if tracing is enabled, None otherwise
    """
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    
    if not langchain_api_key:
        # Use default key if enabled and available
        if USE_DEFAULT_KEYS and DEFAULT_LANGCHAIN_API_KEY:
            logger.info("Using default LangChain API key")
            os.environ["LANGCHAIN_API_KEY"] = DEFAULT_LANGCHAIN_API_KEY
            langchain_api_key = DEFAULT_LANGCHAIN_API_KEY
        else:
            logger.info("LangChain API key not found. Tracing disabled.")
            return None
    
    # Skip tracing if the key is empty or placeholder
    if langchain_api_key in ["your_langchain_api_key_here", ""]:
        logger.info("LangChain API key is empty or placeholder. Tracing disabled.")
        return None
    
    try:
        # Set up tracing with LangChain
        tracer = LangChainTracer(
            project_name=project_name,
        )
        
        # Add console handler for local debugging
        console_handler = ConsoleCallbackHandler()
        
        # Create callback manager with both tracers
        callback_manager = CallbackManager([tracer, console_handler])
        
        logger.info(f"LangChain tracing enabled for project: {project_name}")
        return callback_manager
    except Exception as e:
        logger.warning(f"Failed to set up LangChain tracing: {e}")
        # Return a callback manager with just the console handler
        return CallbackManager([ConsoleCallbackHandler()])

def get_llm(
    provider: str = 'openai', 
    temperature: float = 0.0,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> BaseLanguageModel:
    """
    Returns a LangChain language model instance based on the specified provider.
    
    LangChain provides a unified interface to work with different LLM backends,
    abstracting away the specific implementation details of each provider.
    
    Args:
        provider (str): The LLM backend to use through LangChain ('openai' or 'groq')
        temperature (float): Controls randomness in the model's output
                            (0.0 = deterministic, 1.0 = creative)
        model_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the model
    
    Returns:
        A LangChain language model instance with a standardized interface
    
    Raises:
        ValueError: If an unsupported provider is specified
        EnvironmentError: If required API keys are missing
    """
    logger.info(f"Initializing LangChain LLM with backend: {provider}")
    
    # Set up tracing if LangChain API key is available
    callback_manager = setup_tracing()
    
    # Default model kwargs if none provided
    if model_kwargs is None:
        model_kwargs = {}
    
    # Common kwargs for all models
    common_kwargs = {
        "temperature": temperature,
        "callbacks": callback_manager,
    }
    
    # Merge with any custom model kwargs
    kwargs = {**common_kwargs, **model_kwargs}
    
    if provider == 'groq':
        # Check for Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            # Use default key if enabled and available
            if USE_DEFAULT_KEYS and DEFAULT_GROQ_API_KEY:
                logger.info("Using default Groq API key")
                os.environ["GROQ_API_KEY"] = DEFAULT_GROQ_API_KEY
                kwargs["api_key"] = DEFAULT_GROQ_API_KEY
            else:
                raise EnvironmentError(
                    "GROQ_API_KEY environment variable is required for LangChain's Groq integration"
                )
        
        # Set default model if not provided in model_kwargs
        if 'model' not in kwargs:
            kwargs['model'] = "llama-3.3-70b-versatile"
            
        # LangChain's wrapper for Groq models
        return ChatGroq(**kwargs)
    
    elif provider == 'openai':
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Use default key if enabled and available
            if USE_DEFAULT_KEYS and DEFAULT_OPENAI_API_KEY:
                logger.info("Using default OpenAI API key")
                os.environ["OPENAI_API_KEY"] = DEFAULT_OPENAI_API_KEY
                kwargs["api_key"] = DEFAULT_OPENAI_API_KEY
            else:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is required for LangChain's OpenAI integration"
                )
        
        # Set default model if not provided in model_kwargs
        if 'model' not in kwargs:
            kwargs['model'] = "gpt-3.5-turbo"
            
        # LangChain's wrapper for OpenAI models
        return ChatOpenAI(**kwargs)
    
    else:
        raise ValueError(f"Unsupported LangChain LLM backend: {provider}")
