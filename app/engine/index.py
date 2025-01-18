import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional


from cachetools import TTLCache, cached  # type: ignore
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.indices import VectorStoreIndex
from pydantic import BaseModel

from app.engine.vectordb import get_vector_store


logger = logging.getLogger("uvicorn")




class IndexConfig(BaseModel):
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
    if config is None:
        config = IndexConfig()
    logger.info("Connecting vector store...")
    store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        store, callback_manager=config.callback_manager
    )
    logger.info(f"Finished loading index from {store}")
    return index