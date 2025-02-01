"""
Database Document Loader Module

Handles loading documents from SQL databases using LlamaIndex's DatabaseReader.
Supports multiple database connections and queries through configuration.

Honestly I haven't dug into the customization that can be applied with llamaindex's DatabaseReader, 
but it's easier to implement than doing it manually for now, so we might as well use it.

I may end up implementing this manually if I don't like it though. Or someone else can figure that out.
"""

import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DBLoaderConfig(BaseModel):
    """
    Configuration for database document loading
    
    Attributes:
        uri: SQLAlchemy connection string (e.g. postgresql://user:pass@host/db)
        queries: List of SQL queries to execute for document extraction
    """
    uri: str
    queries: List[str]


def get_db_documents(configs: list[DBLoaderConfig]):
    """
    Load documents from configured database sources
    
    Args:
        configs: List of database configurations to process
        
    Returns:
        List of loaded documents from all configured sources
        
    Raises:
        ImportError: If DatabaseReader dependency is not installed
        Exception: For database connection or query errors
    """
    try:
        from llama_index.readers.database import DatabaseReader
    except ImportError:
        logger.error(
            "Failed to import DatabaseReader. Make sure llama_index is installed."
        )
        raise

    docs = []
    # Process each database configuration
    for entry in configs:
        loader = DatabaseReader(uri=entry.uri)
        # Execute each query and collect documents
        for query in entry.queries:
            logger.info(f"Loading data from database with query: {query}")
            documents = loader.load_data(query=query)
            docs.extend(documents)

    return docs
