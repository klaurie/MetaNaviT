"""
Document Loader Configuration & Orchestration

This module coordinates loading documents from multiple sources:
- File system documents (PDFs, TXT, etc)
- Web content (URLs, APIs)
- Database records

Note: The only configuration currently set up is for file system documents.

Configuration Example (loaders.yaml):
    file:
      path: "data/documents"
    db:
      - connection: "postgresql://..."
        query: "SELECT * FROM docs"
"""

import logging
from typing import Any, Dict, List

import yaml  # type: ignore
from app.database.index_manager import IndexManager
from app.engine.loaders.db import DBLoaderConfig, get_db_documents
from app.engine.loaders.file_system import get_files
from llama_index.core import Document

logger = logging.getLogger(__name__)

# Initialize global index manager
index_manager: IndexManager = IndexManager()

def load_configs() -> Dict[str, Any]:
    """Loads loader configurations from YAML file"""
    with open("config/loaders.yaml") as f:
        configs = yaml.safe_load(f)
    return configs


def get_documents() -> List[Document]:
    """
    Main entry point for document loading.
    Orchestrates loading from all configured sources.
    
    Returns:
        List[Document]: Aggregated list of Document objects from all sources
        
    Raises:
        ValueError: If loader type is invalid
        yaml.YAMLError: If config file is invalid
        IOError: If config file cannot be read
    """
    documents = []
    config = load_configs()

    for loader_type, loader_config in config.items():
        logger.info(
            f"Loading documents from loader: {loader_type}, config: {loader_config}"
        )
        match loader_type:
            case "file_system":
                    documents.extend(get_files(loader_config["path"], index_manager))

            case "db":
                documents.extend(
                    get_db_documents(
                        configs=[DBLoaderConfig(**cfg) for cfg in loader_config]
                    )
                )
            case _:
                raise ValueError(f"Invalid loader type: {loader_type}")

    return documents
