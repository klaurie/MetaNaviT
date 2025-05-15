"""
LLM Module

Provides factory functions to create language model and embedding model instances
based on environment configuration.
"""

import os
import logging
from functools import lru_cache
from typing import Dict, Any, Optional

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_llm(model: Optional[str] = None, **kwargs) -> Any:
    """
    Returns a language model instance based on environment configuration.
    
    Args:
        model: Optional model name override
        **kwargs: Additional parameters to pass to the model constructor
        
    Returns:
        A configured LLM instance
    """
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    model_name = model or os.getenv("MODEL", "gpt-4o-mini-2024-07-18")
    
    # Get configuration from environment variables or use defaults
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    logger.info(f"Initializing {provider} LLM with model: {model_name}")
    
    # Configure specific parameters for the model
    model_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    model_kwargs.update(kwargs)
    
    if provider == "openai":
        # Initialize OpenAI model
        return OpenAI(
            model=model_name,
            **model_kwargs
        )
    
    elif provider == "ollama":
        # For Ollama models
        from llama_index.llms.ollama import Ollama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return Ollama(
            model=model_name,
            base_url=base_url,
            **model_kwargs
        )
    
    else:
        # Default to OpenAI if provider not recognized
        logger.warning(f"Unrecognized provider '{provider}', falling back to OpenAI")
        return OpenAI(
            model=model_name,
            **model_kwargs
        )

@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """
    Returns an embedding model instance based on environment configuration.
    
    Returns:
        A configured embedding model instance
    """
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embed_dim = int(os.getenv("EMBEDDING_DIM", "3068"))
    
    logger.info(f"Initializing embedding model: {model_name} with dimension: {embed_dim}")
    
    # Initialize OpenAI embedding model
    return OpenAIEmbedding(
        model=model_name,
        embed_batch_size=10,  # Process 10 documents at a time
        dimensions=embed_dim
    )

def initialize_settings():
    """Initialize the global LlamaIndex settings with our models."""
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
    
    logger.info("LlamaIndex settings initialized with default models")
    return Settings 