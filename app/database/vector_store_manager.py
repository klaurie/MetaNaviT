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
    """Manages vector store operations using pgvector and pg_search extensions."""
    
    def __init__(self, conn_string: str = None):
        """Initialize vector store manager with connection pooling."""
        # Initialize parent DatabaseManager
        super().__init__(conn_string)
        
        self.schema_name = os.getenv("PGVECTOR_SCHEMA", "public")
        self.table_name = os.getenv("PGVECTOR_TABLE", "llamaindex_embedding")
        self.embed_dim = int(os.getenv("EMBEDDING_DIM", 1024))
        self.vector_store = None
        
        # Ensure required extensions exist
        self._ensure_vector_extension()
        self._ensure_pg_search_extension()

        # Ensure BM25 index exists (call after extensions are ensured)
        # This assumes get_vector_store() might implicitly create the table
        # or that the table exists from previous runs.
        self._create_bm25_index()
        
    def _ensure_vector_extension(self) -> None:
        """Create pgvector extension if it doesn't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    logger.info("Ensuring 'vector' extension exists...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    logger.info("'vector' extension check complete.")
                except Exception as e:
                    logger.error(f"Error creating vector extension: {e}")
                    # Rollback might not be needed if autocommit is true, but good practice
                    conn.rollback()
                    raise

    def _ensure_pg_search_extension(self) -> None:
        """Create pg_search extension if it doesn't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    logger.info("Ensuring 'pg_search' extension exists...")
                    # Assumes pg_search is installed in the PostgreSQL instance
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search SCHEMA public;")
                    logger.info("'pg_search' extension check complete.")
                except Exception as e:
                    # Catch specific error if extension is not installed vs. other errors
                    if "extension \"pg_search\" does not exist" in str(e):
                         logger.error("pg_search extension is not installed in PostgreSQL.")
                         logger.error("Please install it using the .deb package and restart PostgreSQL.")
                    else:
                        logger.error(f"Error creating pg_search extension: {e}")
                    conn.rollback()
                    raise

    def _create_bm25_index(self) -> None:
        """Create BM25 index on the vector store table if it doesn't exist."""
        # Use sql.Identifier for safe quoting of schema and table names
        table_identifier = sql.Identifier(self.schema_name, self.table_name)
        # Define index name dynamically
        index_name = sql.Identifier(f"{self.table_name}_bm25_idx")

        # IMPORTANT ASSUMPTION:
        # Assumes the text content is in metadata_ ->> 'text'
        # Assumes the primary key / node identifier is 'node_id' (UUID)
        # Adjust column names ('node_id', "metadata_ ->> 'text'") if your schema differs.
        create_index_sql = sql.SQL("""
            CREATE INDEX IF NOT EXISTS {index_name} ON {table}
            USING bm25 (node_id, metadata_ ->> 'text')
            WITH (key_field='node_id');
        """).format(
            index_name=index_name,
            table=table_identifier
        )

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    logger.info(f"Ensuring BM25 index '{index_name.strings[0]}' exists on table '{table_identifier.strings[0]}.{table_identifier.strings[1]}'...")
                    cur.execute(create_index_sql)
                    logger.info("BM25 index check complete.")
                except Exception as e:
                    logger.error(f"Error creating BM25 index: {e}")
                    # Check if the error is due to missing columns
                    if "column" in str(e) and "does not exist" in str(e):
                        logger.error("BM25 index creation failed: Check if 'node_id' and 'metadata_' columns exist in the table.")
                        logger.error("Ensure text content is stored under the 'text' key within the 'metadata_' JSONB column.")
                    conn.rollback()
                    # Decide if you want to raise or just log
                    # raise

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
            # This might implicitly create the table if it doesn't exist
            self.vector_store = PGVectorStore(
                connection_string=psycopg2_conn,
                async_connection_string=asyncpg_conn,
                schema_name=self.schema_name,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
            )
            logger.info(f"PGVectorStore initialized for table '{self.schema_name}.{self.table_name}'")

            # Optionally, ensure index exists *after* store is initialized
            # self._create_bm25_index() # Could be called here instead of __init__

        return self.vector_store

    def search_bm25(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform BM25 search using paradedb.match.

        Args:
            query: The search query string.
            limit: The maximum number of results to return.

        Returns:
            A list of dictionaries, each containing 'node_id', 'text', and 'score'.
            Returns an empty list if search fails or yields no results.

        Raises:
            ValueError: If query is empty.
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        table_identifier = sql.Identifier(self.schema_name, self.table_name)

        # IMPORTANT ASSUMPTION: Matches the index creation.
        # Selects 'node_id' and the text content from metadata_ ->> 'text'.
        # Uses paradedb.match on metadata_ ->> 'text' and paradedb.score on 'node_id'.
        # Adjust if your schema or index differs.
        search_sql = sql.SQL("""
            SELECT node_id, metadata_ ->> 'text' as text, paradedb.score(node_id) AS score
            FROM {table}
            WHERE node_id @@@ paradedb.match(metadata_ ->> 'text', %s)
            ORDER BY score DESC
            LIMIT %s;
        """).format(table=table_identifier)

        try:
            logger.debug(f"Executing BM25 search for query: '{query}' with limit: {limit}")
            results = self.execute_query(
                query=search_sql,
                params=(query, limit),
                fetch=True
            )

            if results:
                # Convert list of tuples [(node_id, text, score), ...] to list of dicts
                result_list = [
                    {"node_id": row[0], "text": row[1], "score": row[2]}
                    for row in results
                ]
                logger.info(f"BM25 search returned {len(result_list)} results.")
                return result_list
            else:
                logger.info("BM25 search returned no results.")
                return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            # Depending on desired behavior, you might want to raise the exception
            # or just return an empty list.
            # raise
            return []

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self.vector_store:
            # Add any vector store cleanup here if needed
            self.vector_store = None
        super().close()  # Call parent close method

# Global instance - Consider if singleton is truly needed or if explicit instantiation is better
_vector_store_manager = None

def get_vector_store_manager() -> VectorStoreManager: # Renamed for clarity
    """Get global vector store manager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        logger.info("Initializing global VectorStoreManager...")
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager

# Keep original function for compatibility if needed, but point to manager
def get_vector_store() -> PGVectorStore:
    """Get global vector store instance via the manager."""
    manager = get_vector_store_manager()
    return manager.get_vector_store()

# --- New Function: Expose BM25 Search ---
def search_bm25(query: str, limit: int = 5) -> list[dict]:
    """Perform BM25 search using the global VectorStoreManager."""
    manager = get_vector_store_manager()
    return manager.search_bm25(query=query, limit=limit)
