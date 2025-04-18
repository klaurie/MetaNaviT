import os
import logging
from llama_index.vector_stores.postgres import PGVectorStore
from urllib.parse import urlparse
import psycopg2

logger = logging.getLogger("uvicorn")

PGVECTOR_SCHEMA = "public"
PGVECTOR_TABLE = "llamaindex_embedding"
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM", 1024)

vector_store: PGVectorStore = None



def get_vector_store():
    global vector_store

    if vector_store is None:
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

        vector_store = PGVectorStore(
            connection_string=conn_string,
            async_connection_string=async_conn_string,
            schema_name=PGVECTOR_SCHEMA,
            table_name=PGVECTOR_TABLE,
            embed_dim=int(EMBEDDING_DIM),
        )
        logger.info(f"info: {vars(vector_store)}")

    return vector_store


def match_vector_dim(conn_string):
    p = urlparse(conn_string)

    pg_connection_dict = {
        'dbname': p.path[1:],
        'user': p.username,
        'password': p.password,
        'port': p.port,
        'host': p.hostname
    }
    conn = psycopg2.connect(**pg_connection_dict)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{PGVECTOR_SCHEMA}' AND table_name = '{PGVECTOR_TABLE}' AND column_name = 'embedding';
        """)
        result = cursor.fetchone()
        if result:
            current_dim = int(result[1].split('(')[1].strip(')'))
            if current_dim != int(EMBEDDING_DIM):
                logger.info(f"Current embedding dimension ({current_dim}) does not match expected dimension ({EMBEDDING_DIM}). Updating...")
                cursor.execute(f"""
                    ALTER TABLE {PGVECTOR_SCHEMA}.{PGVECTOR_TABLE}
                    DROP COLUMN embedding;
                """)
                cursor.execute(f"""
                    ALTER TABLE {PGVECTOR_SCHEMA}.{PGVECTOR_TABLE}
                    ADD COLUMN embedding VECTOR({EMBEDDING_DIM});
                """)
                logger.info(f"Updated embedding dimension to {EMBEDDING_DIM}.")
            else:
                logger.info(f"Embedding dimension is already correct: {current_dim}.")
        else:
            logger.error("Embedding column not found in the table.")
    except Exception as e:
        logger.error(f"Error ensuring correct vector dimension: {e}")
    finally:
        cursor.close()
        conn.close()