"""
Database Manager Module

Handles PostgreSQL connections and operations with connection pooling
and proper resource management.

Classes:
    DatabaseManager: Main database connection and operation handler

Environment Variables:
    PG_CONNECTION_STRING: Primary connection string
    PSYCOPG2_CONNECTION_STRING: Alternative connection string
    DB_NAME: Database name (default: metanavit)
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional, Generator
from urllib.parse import urlparse, urlunparse

import psycopg2
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations using connection pooling."""
    
    def __init__(self, conn_string: Optional[str] = None):
        """
        Initialize database manager with connection pooling.
        
        Args:
            conn_string: Optional database connection string.
                        Falls back to environment variable if not provided.
        
        Raises:
            ValueError: If no connection string is available
        """
        self.DATABASE_NAME = os.getenv("DB_NAME", "metanavit")
        self.conn_string = conn_string or os.getenv("PSYCOPG2_CONNECTION_STRING")
        if not self.conn_string:
            raise ValueError("Database connection string not provided")
            
        self.pool = None
        # Create database first, then initialize connection pool
        self._ensure_database_exists()
        self._initialize_connection_pool()

    @contextmanager     # This decorator allows using 'with' statements
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a database connection from the pool using context manager.
        
        Yields:
            psycopg2.extensions.connection: Database connection
            
        Example:
            ```python
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM table")
            ```
        """
        conn = self.pool.getconn()
        try:
            yield conn  # Pass connection to the 'with' block
        except Exception as e:
            conn.rollback()   # Roll back transaction on error
            logger.error(f"Database operation failed: {e}")
            raise
        else:
            conn.commit()   # Commit transaction if no errors
        finally:
            # Always return connection to pool, even if there was an error
            self.pool.putconn(conn)

    def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist."""
        parsed = urlparse(self._get_admin_conn_string())
        
        # Connect to default 'postgres' database to create our database
        with psycopg2.connect(
            database="postgres",   # Default PostgreSQL database
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port
        ) as conn:
            # autocommit required for database creation
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if database already exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.DATABASE_NAME,)
                )
                if not cur.fetchone():
                    # Use sql.SQL and Identifier to safely quote database name
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(self.DATABASE_NAME)
                        )
                    )
                    logger.info(f"Database '{self.DATABASE_NAME}' created")

    def _initialize_connection_pool(self, minconn: int = 1, maxconn: int = 10) -> None:
        """Initialize the connection pool."""
        self.pool = SimpleConnectionPool(
            minconn,
            maxconn,
            self.conn_string
        )
        logger.info("Connection pool initialized")

    def _get_admin_conn_string(self) -> str:
        """Get admin connection string for database creation."""
        conn_string = os.getenv("PG_CONNECTION_STRING")
        if not conn_string:
            raise ValueError("PG_CONNECTION_STRING environment variable not set")
        
        parsed = urlparse(conn_string)
        return urlunparse(parsed._replace(path="/postgres"))

    def close(self) -> None:
        """Close all database connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("All database connections closed")

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True   # Set to False for INSERT/UPDATE/DELETE queries
    ) -> Optional[list]:
        """
        Execute a SQL query and optionally return results.
        
        Args:
            query: SQL query string
            params: Query parameters as tuple
            fetch: Whether to return results (SELECT) or not (INSERT/UPDATE)
        
        Returns:
            Optional[list]: Query results if fetch=True, None otherwise
            
        Raises:
            psycopg2.Error: Database errors
            ValueError: Invalid query or parameters
            ```
        """
        if not query:
            raise ValueError("Query string cannot be empty")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # params are passed separately to prevent SQL injection
                    cur.execute(query, params)
                    if fetch:
                        # fetchall() returns list of tuples, each tuple is a row
                        results = cur.fetchall()
                        logger.debug(f"Query returned {len(results)} rows")
                        return results
                    # For non-SELECT queries, return number of affected rows
                    affected = cur.rowcount
                    logger.debug(f"Query affected {affected} rows")
                    return None
                    
                except psycopg2.Error as e:
                    logger.error(f"Query execution failed: {e}")
                    raise


    def create_bm25_index(self, table: str = "documents", fields: str = "title, content", key_field: str = "id") -> None:
        """Create a BM25 index on the given table and fields."""
        index_sql = f"""
        CREATE INDEX IF NOT EXISTS {table}_bm25_idx
        ON {table}
        USING bm25 (id, {fields})
        WITH (key_field = '{key_field}');
        """
        self.execute_query(index_sql, fetch=False)
        logger.info(f"BM25 index created on {table}({fields})")

    def run_bm25_search(self, table: str, search_column: str, query_text: str, limit: int = 10) -> list:
        """Perform a BM25-ranked search on the given table."""
        search_sql = f"""
        SELECT *, paradedb.score(id) AS score
        FROM {table}
        WHERE id @@@ paradedb.match('{search_column}', %s)
        ORDER BY score DESC
        LIMIT {limit};
        """
        return self.execute_query(search_sql, params=(query_text,), fetch=True)

    def create_documents_table(self):
        """Create a basic documents table for BM25 search."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            content TEXT
        );
        """
        self.execute_query(create_table_sql, fetch=False)
        logger.info("Table 'documents' created or already exists")

    def insert_document(self, title: str, content: str):
        """Insert a document into the documents table."""
        insert_sql = """
        INSERT INTO documents (title, content)
        VALUES (%s, %s);
        """
        self.execute_query(insert_sql, params=(title, content), fetch=False)
        logger.info("Document inserted")