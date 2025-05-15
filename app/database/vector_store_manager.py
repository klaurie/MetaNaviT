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
from llama_index.core.schema import TextNode
import uuid
import time
import psycopg2

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
        
        # First, detect the actual vector dimensions in the database
        # before setting self.embed_dim
        detected_dim = self._detect_vector_dimensions()
        if detected_dim is not None:
            logger.info(f"Detected existing vector dimension in database: {detected_dim}")
            self.embed_dim = detected_dim
        else:
            # If no table exists yet, use the environment variable
            self.embed_dim = int(os.getenv("EMBEDDING_DIM", 3068))  # Default to 768 instead of 1024
            logger.info(f"Using vector dimension from environment: {self.embed_dim}")
        
        self.vector_store = None
        
        # Ensure required extensions exist
        self._ensure_vector_extension()
        self._ensure_pg_search_extension()
        
        # ==> Explicitly create the LlamaIndex table structure BEFORE PGVectorStore init <==
        self._create_llamaindex_table_if_not_exists() 
        
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
                    logger.info("Ensuring 'pg_search' extension (in 'paradedb' schema) exists...")
                    # pg_search is expected to be in the 'paradedb' schema as per ParadeDB's setup
                    # and confirmed by your \dx output.
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search SCHEMA paradedb;")
                    logger.info("'pg_search' extension check complete (expected in 'paradedb' schema).")
                except Exception as e:
                    # Catch specific error if extension is not installed vs. other errors
                    if "extension \"pg_search\" does not exist" in str(e) or \
                       "could not open extension control file" in str(e): # More general check
                         logger.error("pg_search extension is not installed in PostgreSQL or control file is missing.")
                         logger.error("Please ensure it is correctly installed (e.g., via .deb package) and PostgreSQL is restarted.")
                    else:
                        logger.error(f"Error creating/ensuring pg_search extension: {e}")
                    conn.rollback()
                    raise

    def _create_bm25_index(self) -> None:
        """Create BM25 index on the 'text' column if it doesn't exist."""
        # Name for the BM25 index
        index_name = f"{self.table_name}_bm25_idx_on_text"
        table_name = f"{self.schema_name}.{self.table_name}"
        
        # Use the exact syntax from ParadeDB documentation:
        # CREATE INDEX ON table USING bm25 (id, data) WITH (key_field = 'id');
        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}
            USING bm25 (node_id, text)
            WITH (key_field = 'node_id');
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    logger.info(f"Creating BM25 index '{index_name}' using ParadeDB official syntax")
                    
                    # Set the search path to include paradedb schema first, then public
                    cur.execute("SET search_path TO paradedb, public;")
                    
                    # Now execute the CREATE INDEX with the exact ParadeDB syntax
                    logger.info(f"Executing SQL: {create_index_sql}")
                    cur.execute(create_index_sql)
                    
            logger.info(f"BM25 index created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating BM25 index: {e}")
            
            # For troubleshooting, try another variant without quotes around node_id
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SET search_path TO paradedb, public;")
                        
                        alt_sql = f"""
                            CREATE INDEX IF NOT EXISTS {index_name}_alt ON {table_name}
                            USING bm25 (node_id, text)
                            WITH (key_field = node_id);
                        """
                        logger.info(f"Trying alternative syntax: {alt_sql}")
                        cur.execute(alt_sql)
                        logger.info("Alternative syntax worked!")
                        return True
            except Exception as alt_e:
                logger.error(f"Alternative syntax also failed: {alt_e}")
        
        logger.warning("BM25 index creation failed. The application will continue without BM25 search capability.")
        return False

    def _create_llamaindex_table_if_not_exists(self) -> None:
        """
        Explicitly creates the table structure expected by PGVectorStore.
        This helps ensure the table exists and is committed before PGVectorStore tries to use it.
        """
        table_identifier = sql.Identifier(self.schema_name, self.table_name)
        # This SQL is based on the typical structure PGVectorStore creates.
        # It includes: id, node_id, text, metadata_ (jsonb), embedding (vector)
        # and an index on node_id.
        # Ensure 'embedding vector({self.embed_dim})' matches your dimension.
        create_table_sql = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                node_id VARCHAR UNIQUE NOT NULL,
                text TEXT,
                metadata_ JSONB,
                embedding vector({embed_dim})
            );
        """).format(table=table_identifier, embed_dim=sql.Literal(self.embed_dim))
        
        # PGVectorStore also often creates an index on node_id.
        create_node_id_index_sql = sql.SQL("""
            CREATE INDEX IF NOT EXISTS {idx_name} ON {table} (node_id);
        """).format(
            idx_name=sql.Identifier(f"idx_{self.table_name}_node_id"), # Example index name
            table=table_identifier
        )

        try:
            with self.get_connection() as conn: # Use your reliable DatabaseManager connection
                with conn.cursor() as cur:
                    logger.info(f"Attempting to explicitly create LlamaIndex table '{self.schema_name}.{self.table_name}' if it doesn't exist...")
                    cur.execute(create_table_sql)
                    logger.info(f"LlamaIndex table '{self.schema_name}.{self.table_name}' creation command executed.")
                    
                    logger.info(f"Attempting to explicitly create index on node_id for LlamaIndex table '{self.schema_name}.{self.table_name}'...")
                    cur.execute(create_node_id_index_sql)
                    logger.info(f"Index on node_id for LlamaIndex table creation command executed.")
                conn.commit() # Crucially, commit these changes
            logger.info(f"Explicit creation/check of LlamaIndex table '{self.schema_name}.{self.table_name}' and node_id index complete and committed.")
        except Exception as e:
            logger.error(f"Error during explicit creation of LlamaIndex table '{self.schema_name}.{self.table_name}': {e}", exc_info=True)
            raise # If this fails, something is seriously wrong with DB access

    def get_vector_store(self) -> PGVectorStore:
        """Get or create PGVectorStore instance."""
        if self.vector_store is None:
            conn_string_env = os.getenv("PG_CONNECTION_STRING")
            if not conn_string_env:
                raise ValueError("PG_CONNECTION_STRING environment variable not set")

            original_scheme = urlparse(conn_string_env).scheme + "://"
            psycopg2_conn_str = conn_string_env.replace(
                original_scheme, "postgresql+psycopg2://"
            )
            asyncpg_conn_str = conn_string_env.replace(
                original_scheme, "postgresql+asyncpg://"
            )
            logger.info(f"PGVectorStore will use psycopg2_conn_str: {psycopg2_conn_str}")
            logger.info(f"PGVectorStore will use asyncpg_conn_str: {asyncpg_conn_str}")
            
            # Log the database name expected by DatabaseManager's pool
            # Assuming self.DATABASE_NAME is set in DatabaseManager's __init__
            # from the same PG_CONNECTION_STRING
            if hasattr(self, 'DATABASE_NAME'):
                logger.info(f"DatabaseManager's pool is configured for database: '{self.DATABASE_NAME}' (derived from PG_CONNECTION_STRING)")
            else: # Fallback if DATABASE_NAME isn't on self for some reason
                parsed_main_conn = urlparse(self.conn_string if self.conn_string else conn_string_env)
                logger.info(f"DatabaseManager's pool is configured for database: '{parsed_main_conn.path.lstrip('/')}' (derived from PG_CONNECTION_STRING)")


            logger.info(f"Attempting to initialize PGVectorStore for table '{self.schema_name}.{self.table_name}' (expected to exist)...")
            self.vector_store = PGVectorStore(
                connection_string=psycopg2_conn_str,
                async_connection_string=asyncpg_conn_str,
                schema_name=self.schema_name,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
            )
            logger.info(f"PGVectorStore Python object initialized for table '{self.schema_name}.{self.table_name}'")

            # Probe (Optional but good for sanity check - can be simplified now)
            dummy_node_id_for_probe = f"test_node_{uuid.uuid4()}"
            try:
                logger.info(f"Probe: Attempting to ADD dummy node '{dummy_node_id_for_probe}' to pre-created table...")
                logger.info(f"Using embedding dimension: {self.embed_dim}")
                
                # Create a zero vector with the CORRECT dimension
                zero_embedding = [0.0] * self.embed_dim
                
                dummy_node = TextNode(
                    id_=dummy_node_id_for_probe,
                    text="dummy_text_content_for_probe_in_precreated_table",
                    embedding=zero_embedding,  # This will now have the correct dimensions
                    metadata={"text": "dummy_metadata_text_for_probe_in_precreated_table"}
                )
                self.vector_store.add([dummy_node])
                logger.info(f"Probe: Successfully EXECUTED add for dummy node '{dummy_node_id_for_probe}'.")
                # Now, try to delete it immediately using PGVectorStore to ensure it can write and delete
                self.vector_store.delete(ref_doc_id=dummy_node_id_for_probe)
                logger.info(f"Probe: Successfully deleted dummy node '{dummy_node_id_for_probe}' using PGVectorStore.")
            except Exception as e:
                logger.error(f"Probe FAILED during PGVectorStore.add/delete on pre-created table: {e}", exc_info=True)
                # Provide more helpful error for dimension mismatch
                if "expected" in str(e) and "dimensions" in str(e):
                    actual_dim = None
                    try:
                        # Try to extract the expected dimension from the error message
                        import re
                        match = re.search(r'expected (\d+) dimensions', str(e))
                        if match:
                            actual_dim = int(match.group(1))
                            logger.error(f"DIMENSION MISMATCH ERROR: Database expects {actual_dim} dimensions, but code is using {self.embed_dim}")
                            logger.error(f"To fix: Set EMBEDDING_DIM={actual_dim} in your environment variables")
                    except:
                        pass
                
                raise RuntimeError(
                    f"PGVectorStore failed .add/delete on pre-created table '{self.schema_name}.{self.table_name}'."
                ) from e

            # BM25 Index Creation - attempt it but don't fail if it doesn't work
            logger.info(f"Proceeding to ensure BM25 index exists for {self.schema_name}.{self.table_name}...")
            try:
                bm25_created = self._create_bm25_index()
                if bm25_created:
                    logger.info("BM25 index creation/verification successful")
                else:
                    logger.warning("BM25 index creation failed, but continuing anyway")
            except Exception as e:
                logger.warning(f"BM25 index creation failed: {e}", exc_info=True)
                logger.warning("Continuing without BM25 search capability")
                # We'll continue without BM25 functionality

        return self.vector_store

    def search_bm25(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform BM25 search using the 'text' column.
        
        If BM25 search fails, falls back to a basic text search.
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        results = []
        
        # Try BM25 search with correct ParadeDB syntax
        try:
            table_identifier = sql.Identifier(self.schema_name, self.table_name)
            
            # BM25 search SQL using the ParadeDB syntax
            # The @@@ operator checks if text matches the query
            # The paradedb.score() function returns the relevance score
            search_sql = sql.SQL("""
                SELECT node_id, text, paradedb.score(node_id) AS score
                FROM {table}
                WHERE text @@@ {query}
                ORDER BY score DESC
                LIMIT %s;
            """).format(
                table=table_identifier,
                query=sql.Literal(query)  # ParadeDB docs show using a literal not parameter binding
            )
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Set search_path
                    cur.execute("SET search_path TO paradedb, public;")
                    # Execute search
                    cur.execute(search_sql, (limit,))  # Only limit is a parameter
                    results = cur.fetchall()
            
            if results:
                logger.info(f"BM25 search successful, found {len(results)} results")
                return [{"node_id": row[0], "text": row[1], "score": row[2]} for row in results]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            # Fall back to basic search
        
        # If BM25 search failed or returned no results, try basic text search fallback
        try:
            fallback_sql = sql.SQL("""
                SELECT node_id, text, 1.0 AS score
                FROM {table}
                WHERE text ILIKE %s
                LIMIT %s;
            """).format(table=sql.Identifier(self.schema_name, self.table_name))
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(fallback_sql, (f'%{query}%', limit))
                    results = cur.fetchall()
            
            if results:
                logger.info(f"Fallback text search successful, found {len(results)} results")
                return [{"node_id": row[0], "text": row[1], "score": row[2]} for row in results]
            else:
                logger.info("No results found in fallback search")
                return []
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self.vector_store:
            # Add any vector store cleanup here if needed
            self.vector_store = None
        super().close()  # Call parent close method

    def _detect_vector_dimensions(self) -> int:
        """Detect the vector dimensions in the existing table, if any."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Try to query the table definition to get vector dimensions
                    cur.execute(f"""
                        SELECT a.atttypmod - 4  -- Subtracting 4 gets the actual dimensions
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = '{self.schema_name}'
                        AND c.relname = '{self.table_name}'
                        AND a.attname = 'embedding'
                        AND a.atttypid = (SELECT oid FROM pg_type WHERE typname = 'vector');
                    """)
                    result = cur.fetchone()
                    if result and result[0] > 0:
                        return result[0]  # Return the detected dimension
                    
                    # If that fails, let's try another approach
                    logger.info("Trying alternate method to detect vector dimensions...")
                    cur.execute(f"""
                        SELECT description 
                        FROM pg_description 
                        JOIN pg_class ON pg_description.objoid = pg_class.oid
                        JOIN pg_namespace ON pg_class.relnamespace = pg_namespace.oid
                        WHERE pg_namespace.nspname = '{self.schema_name}'
                        AND pg_class.relname = '{self.table_name}';
                    """)
                    # If no result or can't parse dimension, we'll return None
                    
                    return None
        except Exception as e:
            logger.warning(f"Error detecting vector dimensions: {e}")
            return None

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