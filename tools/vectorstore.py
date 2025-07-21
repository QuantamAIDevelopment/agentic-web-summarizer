# tools/vectorstore.py

import os
import logging
from typing import List, Dict, Optional, Union

# LangChain imports for vector stores and embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

# Import default keys
try:
    from ..default_keys import (
        DEFAULT_OPENAI_API_KEY,
        USE_DEFAULT_KEYS
    )
except ImportError:
    # Fallback if import fails
    DEFAULT_OPENAI_API_KEY = ""
    USE_DEFAULT_KEYS = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_embeddings(provider: str = "openai", model_name: Optional[str] = None) -> Embeddings:
    """
    Returns a LangChain embedding model based on the specified provider.
    
    LangChain provides a unified interface for working with different embedding models,
    abstracting away the specific implementation details.

    Args:
        provider (str): "openai" or "huggingface".
        model_name (str): Optional custom HuggingFace model name.

    Returns:
        LangChain Embeddings object with standardized interface.

    Raises:
        ValueError: If an unsupported provider is given.
    """
    if provider == "openai":
        logger.info("Using LangChain's OpenAI embeddings integration")
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Use default key if enabled and available
            if USE_DEFAULT_KEYS and DEFAULT_OPENAI_API_KEY:
                logger.info("Using default OpenAI API key for embeddings")
                os.environ["OPENAI_API_KEY"] = DEFAULT_OPENAI_API_KEY
                return OpenAIEmbeddings(openai_api_key=DEFAULT_OPENAI_API_KEY)
            else:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is required for OpenAI embeddings"
                )
        return OpenAIEmbeddings()
    elif provider == "huggingface":
        model = model_name or "all-MiniLM-L6-v2"
        logger.info(f"Using LangChain's HuggingFace embeddings integration with model '{model}'")
        return HuggingFaceEmbeddings(model_name=model)
    else:
        raise ValueError(f"Unsupported LangChain embedding provider: {provider}")


def create_faiss_store(embedding_model: Embeddings) -> FAISS:
    """
    Creates a LangChain FAISS vector store using the embedding model.
    
    LangChain provides a standardized interface to various vector stores,
    including FAISS, Chroma, and others.

    Args:
        embedding_model: A LangChain embedding model instance.

    Returns:
        LangChain FAISS vector store.
    """
    try:
        logger.info("Creating LangChain vector store with FAISS backend")
        # In newer versions of LangChain, we need to use FAISS.from_texts instead
        # If we have no texts yet, we'll create an empty list
        return FAISS.from_texts(texts=["Initial empty document"], embedding=embedding_model)
    except Exception as e:
        logger.error(f"Failed to create LangChain vector store: {e}")
        raise


def store_summary(
    vectorstore: FAISS,
    summary: Union[str, List[str]],
    metadata: Union[Dict, List[Dict]]
):
    """
    Stores one or more summaries in the LangChain vector store with associated metadata.
    
    Uses LangChain's standardized interface for adding documents to vector stores.

    Args:
        vectorstore (FAISS): The LangChain vector store.
        summary (str | List[str]): One or more summaries.
        metadata (dict | List[dict]): Corresponding metadata entries.

    Raises:
        ValueError: If input types do not align.
    """
    if isinstance(summary, str):
        summary = [summary]
    if isinstance(metadata, dict):
        metadata = [metadata]

    if len(summary) != len(metadata):
        raise ValueError("Length of summaries and metadata must match")

    logger.info(f"Storing {len(summary)} summary embeddings to LangChain vector store.")
    # In newer versions of LangChain, we need to pass metadatas as a list of dictionaries
    vectorstore.add_texts(texts=summary, metadatas=metadata)
    
    # Log the storage operation to the database
    try:
        from models.agent_log_utils import log_agent_decision
        for i, (summ, meta) in enumerate(zip(summary, metadata)):
            log_agent_decision(
                agent_name="vectorstore_tool",
                input_data=summ[:100] + "..." if len(summ) > 100 else summ,
                output_summary=f"Stored summary with metadata: {meta}",
                reflection=f"Successfully stored summary in vector database",
                reasoning_chain=[f"Received summary with {len(summ)} characters", f"Added to vector store with metadata"]
            )
    except Exception as e:
        logger.error(f"Failed to log vector store operation: {e}")


def save_vectorstore(vectorstore: FAISS, path: str):
    """
    Saves a LangChain vector store to disk using LangChain's persistence methods.

    Args:
        vectorstore (FAISS): The LangChain vector store instance.
        path (str): Path to save the index to.
    """
    logger.info(f"Saving LangChain vector store to {path}")
    # In newer versions of LangChain, we use the persist method
    try:
        vectorstore.save_local(folder_path=path)
    except (TypeError, AttributeError):
        # Fall back to newer API if available
        try:
            vectorstore.persist(persist_path=path)
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise


def load_vectorstore(path: str, embedding_model: Embeddings) -> FAISS:
    """
    Loads a LangChain vector store from disk using LangChain's persistence methods.

    Args:
        path (str): Path to the saved index.
        embedding_model: The LangChain embedding model used during saving.

    Returns:
        FAISS: The loaded LangChain vector store.
    """
    logger.info(f"Loading LangChain vector store from {path}")
    try:
        # Try the newer API first
        return FAISS.load_local(folder_path=path, embeddings=embedding_model)
    except (TypeError, AttributeError):
        # Fall back to older API
        try:
            return FAISS.load_local(path, embedding_model)
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
