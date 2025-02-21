import logging
from typing import Any, Optional

from llama_index.core.indices import VectorStoreIndex
from pydantic import BaseModel

from app.database.vector_store_manager import get_vector_store


logger = logging.getLogger("uvicorn")


class IndexConfig(BaseModel):
    """
    Configuration for index creation and management.
    
    The callback_manager field allows registering handlers for events like:
    - retrieve: When documents are retrieved
    - llm: When the language model is queried
    - embedding: When text is converted to vectors
    - error: When operations fail
    - tokens: Track token usage
    """
    callback_manager: Optional[Any] = None
   
    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "properties": {
                "callback_manager": {
                    "description": "Callback manager for the index"
                }
            }
        }


def get_index(config: IndexConfig = None):
    """
    Factory function for creating or retrieving cached vector store index
    
    Args:
        config: Optional index configuration
        
    Returns:
        Configured VectorStoreIndex instance
    """
    if config is None:
        config = IndexConfig()
    logger.info("Connecting vector store...")
    store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        store, callback_manager=config.callback_manager
    )
    # logger.info(f"Finished loading index from {store}")
    return index
