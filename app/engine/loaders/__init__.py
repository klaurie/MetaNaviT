"""
Document Loader Configuration & Orchestration

This module coordinates loading documents from multiple sources:
- File system documents (PDFs, TXT, etc)
- Web content (URLs, APIs)
- Database records

Note:

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
from app.engine.loaders.db import DBLoaderConfig, get_db_documents
from app.engine.loaders.file import FileLoaderConfig, get_file_documents
from llama_index.core import Document

logger = logging.getLogger(__name__)


def load_configs() -> Dict[str, Any]:
    """Loads loader configurations from YAML file"""
    with open("config/loaders.yaml") as f:
        configs = yaml.safe_load(f)
    return configs


def get_documents() -> List[Document]:
    """
    Main entry point for document loading.
    Orchestrates loading from all configured sources.
    Returns aggregated list of Document objects.
    """
    documents = []

    """
    Configurations just get the arguments for each loader type
    """
    config = load_configs()
    for loader_type, loader_config in config.items():
        logger.info(
            f"Loading documents from loader: {loader_type}, config: {loader_config}"
        )
        match loader_type:
            case "file":
                document = get_file_documents(FileLoaderConfig(**loader_config))
            case "db":
                document = get_db_documents(
                    configs=[DBLoaderConfig(**cfg) for cfg in loader_config]
                )
            case _:
                raise ValueError(f"Invalid loader type: {loader_type}")
        documents.extend(document)

    return documents
