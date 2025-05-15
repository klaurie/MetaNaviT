"""
Vector Store Module

Provides a singleton accessor for the vector store manager.
Acts as a facade/adapter for the VectorStoreManager implementation.
"""

import os
import logging
from functools import lru_cache

from .vector_store_manager import VectorStoreManager

logger = logging.getLogger("uvicorn")

@lru_cache(maxsize=1)
def get_vector_store_manager() -> VectorStoreManager:
    """
    Returns a singleton instance of the VectorStoreManager.
    
    Uses lru_cache to ensure only one instance is created during the application lifecycle.
    
    Returns:
        VectorStoreManager: A configured vector store manager instance
    """
    # Get connection string from environment
    conn_string = os.getenv("PG_CONNECTION_STRING")
    
    if not conn_string:
        logger.warning("PG_CONNECTION_STRING environment variable not set")
        logger.warning("Vector store operations will likely fail")
    
    logger.info("Creating VectorStoreManager singleton instance")
    return VectorStoreManager(conn_string=conn_string)