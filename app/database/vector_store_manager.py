"""
Vector Store Manager Module

Manages PostgreSQL vector operations using pgvector extension for document embeddings.
Provides connection pooling and vector store management through a singleton pattern.

Features:
    - Vector store initialization and management
    - Connection pooling via parent DatabaseManager
    - pgvector extension handling
    - Both sync (psycopg2) and async (asyncpg) support
    - Automatic table creation and dimension handling

Configuration (via environment variables):
    PGVECTOR_SCHEMA: Schema name for vector tables (default: "public")
    PGVECTOR_TABLE: Table name for embeddings (default: "llamaindex_embedding")
    EMBEDDING_DIM: Embedding dimension size (default: 1024)
    PG_CONNECTION_STRING: PostgreSQL connection string

Dependencies:
    - PostgreSQL with pgvector extension
    - llama-index for vector store operations
    - psycopg2 for database connections
    - asyncpg for async operations
"""
import os
import logging
from llama_index.vector_stores.postgres import PGVectorStore
from urllib.parse import urlparse
from psycopg2 import sql
from dotenv import load_dotenv

from .db_base_manager import DatabaseManager

load_dotenv()
logger = logging.getLogger("uvicorn")

class VectorStoreManager(DatabaseManager):
    """Manages vector store operations using pgvector extension."""
    
    def __init__(self, conn_string: str = None):
        """Initialize vector store manager with connection pooling."""
        # Initialize parent DatabaseManager
        super().__init__(conn_string)
        
        self.schema_name = os.getenv("PGVECTOR_SCHEMA", "public")
        self.table_name = os.getenv("PGVECTOR_TABLE", "llamaindex_embedding")
        self.embed_dim = int(os.getenv("EMBEDDING_DIM", 1024))
        self.vector_store = None
        
        # Ensure vector extension exists
        self._ensure_vector_extension()
        
    def _ensure_vector_extension(self) -> None:
        """Create pgvector extension if it doesn't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                except Exception as e:
                    logger.error(f"Error creating vector extension: {e}")
                    raise

    def get_vector_store(self) -> PGVectorStore:
        """Get or create PGVectorStore instance."""
        if self.vector_store is None:
            # Get original connection string from environment
            conn_string = os.getenv("PG_CONNECTION_STRING")
            if not conn_string:
                raise ValueError("PG_CONNECTION_STRING environment variable not set")

            # Create connection strings for different drivers
            original_scheme = urlparse(conn_string).scheme + "://"
            psycopg2_conn = conn_string.replace(
                original_scheme, "postgresql+psycopg2://"
            )
            asyncpg_conn = conn_string.replace(
                original_scheme, "postgresql+asyncpg://"
            )

            # Initialize vector store
            self.vector_store = PGVectorStore(
                connection_string=psycopg2_conn,
                async_connection_string=asyncpg_conn,
                schema_name=self.schema_name,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
            )

        return self.vector_store

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self.vector_store:
            # Add any vector store cleanup here if needed
            self.vector_store = None
        super().close()  # Call parent close method

# Global instance
_vector_store_manager = None

def get_vector_store():
    """Get global vector store instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager.get_vector_store()
