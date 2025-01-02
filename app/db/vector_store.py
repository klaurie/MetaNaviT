import logging
import asyncpg
from typing import List, Dict, Any, Optional, Union
import json
import numpy as np
from llama_index.vector_stores.postgres import PGVectorStore as LlamaIndexPGStore
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
import uuid
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataProcessor:
    @staticmethod
    def process_file(file_path: str, chunk_index: int) -> Dict[str, Any]:
        """Process file metadata for a given chunk"""
        try:
            file_stat = os.stat(file_path)
            return {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": file_stat.st_size,
                "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "chunk_index": chunk_index,
                "directory": os.path.dirname(file_path),
                "file_extension": os.path.splitext(file_path)[1].lower(),
                "node_id": str(uuid.uuid4())
            }
        except Exception as e:
            logger.error(f"Error processing metadata for {file_path}: {str(e)}")
            return {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "chunk_index": chunk_index,
                "error": str(e),
                "node_id": str(uuid.uuid4())
            }
class PGVectorStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.logger = logging.getLogger(__name__)
        self.llama_store = None
        self.metadata_processor = MetadataProcessor()
        logger.info(f"Initializing PGVectorStore with URL: {database_url}")

    async def initialize(self):
        """Initialize connections and create tables"""
        if self.pool is not None:
            logger.warning("Pool already initialized")
            return

        try:
            # Create asyncpg pool for custom queries
            self.pool = await asyncpg.create_pool(
                dsn=self.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )

            # Parse database URL components
            db_params = {
                "host": self.database_url.split('@')[1].split(':')[0],
                "port": 5432,
                "database": "postgres",
                "user": "postgres",
                "password": "postgres",
            }

            # Initialize LlamaIndex PGVectorStore
            self.llama_store = LlamaIndexPGStore.from_params(
                **db_params,
                table_name="vector_store",
                embed_dim=768,  # For nomic-embed-text
                text_search_config="english"
            )

            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.llama_store
            )

            await self.init_table()
            logger.info("Vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

    async def add_chunk(
        self,
        chunk: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        chunk_index: int
    ) -> None:
        """Add a chunk using LlamaIndex nodes"""
        try:
            # Process metadata if file_path is provided
            if "file_path" in metadata:
                processed_metadata = self.metadata_processor.process_file_metadata(
                    metadata["file_path"], 
                    chunk_index
                )
                metadata.update(processed_metadata)

            # Create a TextNode with relationships
            node = TextNode(
                text=chunk,
                metadata=metadata,
                embedding=embedding,
                node_id=metadata.get("node_id", str(uuid.uuid4()))
            )

            # Set relationships if available
            if chunk_index > 0:
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=f"{metadata.get('file_path')}_{chunk_index-1}"
                )
            
            if "total_chunks" in metadata and chunk_index < metadata["total_chunks"] - 1:
                node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=f"{metadata.get('file_path')}_{chunk_index+1}"
                )

            # Add to vector store
            if self.llama_store:
                self.llama_store.add(node)
            else:
                raise ValueError("LlamaIndex store not initialized")

        except Exception as e:
            logger.error(f"Error adding chunk: {str(e)}")
            raise

    async def get_similar_chunks(
        self,
        embedding: Union[List[float], np.ndarray],
        limit: int = 5,
        file_pattern: str = None,
        file_path: str = None,
        directory: str = None
    ) -> List[Dict[str, Any]]:
        """Get similar chunks with enhanced metadata and relationships"""
        try:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Query using LlamaIndex
            query_embedding = np.array(embedding)
            results = self.llama_store.similarity_search(
                query_embedding,
                similarity_top_k=limit,
                filters = {
                    "file_pattern": file_pattern,
                    "file_path": file_path,
                    "directory": directory
                } if any([file_pattern, file_path, directory]) else None
            )

            # Process results
            processed_results = []
            for node in results:
                try:
                    result = {
                        'id': node.node_id,
                        'metadata': node.metadata,
                        'snippet': node.text,
                        'similarity': node.score if hasattr(node, 'score') else 0.0,
                        'relationships': {
                            'previous': node.relationships.get(NodeRelationship.PREVIOUS),
                            'next': node.relationships.get(NodeRelationship.NEXT)
                        }
                    }
                    processed_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    continue

            return processed_results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise

    async def init_table(self):
        """Initialize database schema"""
        async with self.pool.acquire() as conn:
            try:
                # Enable extensions
                await conn.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                """)

                # Drop existing table if it exists
                await conn.execute("""
                    DROP TABLE IF EXISTS vector_store CASCADE;
                """)

                # Create table with all required columns
                await conn.execute("""
                    CREATE TABLE vector_store (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        node_id UUID NOT NULL UNIQUE,  -- Added UNIQUE constraint
                        metadata JSONB,
                        snippet TEXT,
                        embedding vector(768),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        previous_node_id UUID,
                        next_node_id UUID
                    );
                """)

                # Add foreign key constraints
                await conn.execute("""
                    ALTER TABLE vector_store
                    ADD CONSTRAINT vector_store_previous_node_id_fkey
                    FOREIGN KEY (previous_node_id) 
                    REFERENCES vector_store(node_id) DEFERRABLE INITIALLY DEFERRED;

                    ALTER TABLE vector_store
                    ADD CONSTRAINT vector_store_next_node_id_fkey
                    FOREIGN KEY (next_node_id) 
                    REFERENCES vector_store(node_id) DEFERRABLE INITIALLY DEFERRED;
                """)

                # Create indices
                await conn.execute("""
                    CREATE INDEX vector_store_embedding_idx 
                    ON vector_store USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);

                    CREATE INDEX vector_store_metadata_idx 
                    ON vector_store USING GIN (metadata);

                    CREATE INDEX vector_store_node_id_idx 
                    ON vector_store(node_id);

                    CREATE INDEX vector_store_relationship_idx 
                    ON vector_store(previous_node_id, next_node_id);
                """)

                logger.info("Vector store table and indices created successfully")

            except Exception as e:
                logger.error(f"Error in table initialization: {str(e)}")
                raise

    async def add(self, embedding: List[float], metadata: Dict[str, Any], text: str) -> None:
        """Add a vector to the store"""
        try:
            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            # Insert into database using the correct table name
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO vector_store (embedding, metadata, snippet, node_id)
                    VALUES ($1::vector, $2, $3, $4)
                    """,
                    embedding_str,  # Cast to vector type
                    json.dumps(metadata),
                    text,
                    metadata.get('node_id')
                )
        except Exception as e:
            self.logger.error(f"Error adding vector: {e}")
            raise

# Create global instance
pg_storage = PGVectorStore(database_url="postgresql://postgres:postgres@db:5432/postgres")