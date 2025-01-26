from dotenv import load_dotenv

load_dotenv()

import os
import logging
from llama_index.vector_stores.postgres import PGVectorStore
from urllib.parse import urlparse, urlunparse
import psycopg2
from psycopg2 import sql

logger = logging.getLogger("uvicorn")

PGVECTOR_SCHEMA = "public"
PGVECTOR_TABLE = "llamaindex_embedding"
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM", 1024)
DATABASE_NAME = os.getenv("DB_NAME", "metanavit")

vector_store: PGVectorStore = None

def create_database():
    original_conn_string = os.getenv("PG_CONNECTION_STRING")
    if original_conn_string is None or original_conn_string == "":
        raise ValueError("PG_CONNECTION_STRING environment variable is not set.")

    # Remove the database name from the connection string to connect to the default database
    parsed_url = urlparse(original_conn_string)
    conn_string_without_db = urlunparse(parsed_url._replace(path="/postgres"))

    conn = psycopg2.connect(conn_string_without_db)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DATABASE_NAME}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DATABASE_NAME)))
            logger.info(f"Database '{DATABASE_NAME}' created successfully.")

            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Vector extension created or already exists.")
            except Exception as e:
                logger.error(f"Error creating vector extension: {e}")
        else:
            logger.info(f"Database '{DATABASE_NAME}' already exists.")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
    finally:
        cursor.close()
        conn.close()

def get_vector_store():
    global vector_store

    if vector_store is None:
        create_database()

        original_conn_string = os.environ.get("PG_CONNECTION_STRING")
        if original_conn_string is None or original_conn_string == "":
            raise ValueError("PG_CONNECTION_STRING environment variable is not set.")

        # The PGVectorStore requires both two connection strings, one for psycopg2 and one for asyncpg
        # Update the configured scheme with the psycopg2 and asyncpg schemes
        original_scheme = urlparse(original_conn_string).scheme + "://"
        conn_string = original_conn_string.replace(
            original_scheme, "postgresql+psycopg2://"
        )
        async_conn_string = original_conn_string.replace(
            original_scheme, "postgresql+asyncpg://"
        )

        logger.info(conn_string)

        vector_store = PGVectorStore(
            connection_string=conn_string,
            async_connection_string=async_conn_string,
            schema_name=PGVECTOR_SCHEMA,
            table_name=PGVECTOR_TABLE,
            embed_dim=int(EMBEDDING_DIM),
        )
        logger.info(f"info: {vars(vector_store)}")

    return vector_store


def create_vector_extension():
    conn_string = os.environ.get("PG_CONNECTION_STRING")
    if conn_string is None or conn_string == "":
        raise ValueError("PG_CONNECTION_STRING environment variable is not set.")
    
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("Vector extension created or already exists.")
    except Exception as e:
        logger.error(f"Error creating vector extension: {e}")
    finally:
        cursor.close()
        conn.close()