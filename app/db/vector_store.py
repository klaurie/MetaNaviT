import logging
import asyncpg
import asyncio
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PGVectorStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        logger.info(f"Initializing PGVectorStore with URL: {database_url}")

    async def initialize(self):
        """Initialize the connection pool"""
        if self.pool is not None:
            logger.warning("Pool already initialized")
            return

        try:
            logger.info("Creating database connection pool...")
            logger.info(f"Attempting to connect with URL: {self.database_url}")
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.pool = await asyncpg.create_pool(
                        dsn=self.database_url,
                        min_size=1,
                        max_size=10,
                        command_timeout=60
                    )
                    logger.info("Database connection pool created successfully")
                    
                    # Test the connection and log user info
                    async with self.pool.acquire() as conn:
                        version = await conn.fetchval('SELECT version()')
                        current_user = await conn.fetchval('SELECT current_user')
                        logger.info(f"Connected to PostgreSQL: {version}")
                        logger.info(f"Connected as user: {current_user}")
                    return
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Connection attempt {retry_count} failed: {str(e)}")
                    if retry_count < max_retries:
                        await asyncio.sleep(5)
                    else:
                        raise
        except Exception as e:
            logger.error(f"Failed to create database pool: {str(e)}")
            raise

    async def add_chunk(self, snippet: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Add a chunk to the vector store"""
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding list to string
                embedding_str = json.dumps(embedding)
                # Convert metadata to string if it's not already
                metadata_str = json.dumps(metadata) if isinstance(metadata, dict) else metadata
                
                await conn.execute(
                    """
                    INSERT INTO vector_store (snippet, embedding, metadata)
                    VALUES ($1, $2, $3)
                    """,
                    snippet, embedding_str, metadata_str
                )
        except Exception as e:
            logger.error(f"Error adding chunk to vector store: {e}")
            raise

    async def get_similar_chunks(
        self, 
        embedding: List[float], 
        limit: int = 5,
        file_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get similar chunks from the database with balanced representation from all matching files"""
        if not self.pool:
            await self.initialize()
            
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            # First, get the list of matching files
            files_query = """
            SELECT DISTINCT metadata->>'file_name' as filename
            FROM vector_store
            WHERE metadata->>'file_name' IS NOT NULL
            """
            if file_pattern:
                pattern = f'%{file_pattern}%' if not file_pattern.startswith('%') else file_pattern
                files_query += f" AND metadata->>'file_name' LIKE '{pattern}'"
            
            async with self.pool.acquire() as conn:
                matching_files = await conn.fetch(files_query)
                logger.info(f"Found matching files: {[r['filename'] for r in matching_files]}")
                
                processed_results = []
                
                # Get chunks from each file separately
                for file_row in matching_files:
                    file_name = file_row['filename']
                    
                    # Query for this specific file
                    file_query = """
                    WITH RankedChunks AS (
                        SELECT DISTINCT ON (snippet)
                            id,
                            metadata,
                            snippet,
                            embedding <-> $1::vector as similarity
                        FROM vector_store
                        WHERE metadata->>'file_name' = $2
                        ORDER BY snippet, (embedding <-> $1::vector)
                    )
                    SELECT *
                    FROM RankedChunks
                    ORDER BY similarity ASC
                    LIMIT $3;
                    """
                    
                    # Get top 3 chunks from this file
                    file_results = await conn.fetch(file_query, embedding_str, file_name, 3)
                    logger.info(f"Found {len(file_results)} chunks from {file_name}")
                    
                    for row in file_results:
                        try:
                            metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'])
                            processed_results.append({
                                "id": str(row['id']),
                                "metadata": metadata,
                                "snippet": row['snippet'],
                                "similarity": float(row['similarity'])
                            })
                        except Exception as e:
                            logger.error(f"Error processing row from {file_name}: {e}")
                            continue
                
                # Sort all results by similarity
                processed_results.sort(key=lambda x: x['similarity'])
                
                # Log summary
                files_found = set(r['metadata'].get('file_name') for r in processed_results)
                logger.info(f"Successfully processed {len(processed_results)} chunks from files: {files_found}")
                
                return processed_results
                
        except Exception as e:
            logger.error(f"Error getting similar chunks: {e}")
            raise

    async def get_similar_chunks_for_file(
        self, 
        embedding: List[float], 
        file_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar chunks for a specific file with distinct snippets"""
        if not self.pool:
            await self.initialize()
            
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            async with self.pool.acquire() as conn:
                query = """
                WITH RankedChunks AS (
                    SELECT DISTINCT ON (snippet)
                        id,
                        metadata,
                        snippet,
                        embedding <-> $1::vector as similarity
                    FROM vector_store
                    WHERE metadata->>'file_name' = $2
                    ORDER BY snippet, similarity ASC
                )
                SELECT * FROM RankedChunks
                ORDER BY similarity ASC
                LIMIT $3
                """
                
                results = await conn.fetch(query, embedding_str, file_name, limit)
                return [
                    {
                        "id": str(row['id']),
                        "metadata": row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata']),
                        "snippet": row['snippet'],
                        "similarity": float(row['similarity'])
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error getting similar chunks for file: {e}")
            raise

    async def get_similar_chunks_for_directory(
        self, 
        embedding: List[float], 
        directory_path: str,
        file_pattern: str = "*",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar chunks from files in a directory"""
        if not self.pool:
            await self.initialize()
            
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            async with self.pool.acquire() as conn:
                query = """
                WITH RankedChunks AS (
                    SELECT DISTINCT ON (snippet)
                        id,
                        metadata,
                        snippet,
                        embedding <-> $1::vector as similarity
                    FROM vector_store
                    WHERE metadata->>'file_path' LIKE $3 || '%'
                      AND metadata->>'file_name' LIKE $4
                    ORDER BY snippet, (embedding <-> $1::vector)
                )
                SELECT id, metadata, snippet, similarity
                FROM RankedChunks
                ORDER BY similarity ASC
                LIMIT $2;
                """
                
                results = await conn.fetch(query, embedding_str, limit, directory_path, file_pattern)
                return [
                    {
                        "id": str(row['id']),
                        "metadata": row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata']),
                        "snippet": row['snippet'],
                        "similarity": float(row['similarity'])
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error getting similar chunks: {e}")
            raise

    async def similarity_search(self, embedding: List[float], limit: int = 4) -> List[Dict[str, Any]]:
        """Perform similarity search using embeddings with distinct results"""
        if not self.pool:
            await self.initialize()
            
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            # Modified query to ensure distinct results
            query = """
            WITH RankedDocs AS (
                SELECT DISTINCT ON (snippet)  -- This ensures unique snippets
                    id,
                    metadata,
                    snippet,
                    embedding <-> $1::vector as similarity
                FROM vector_store
                ORDER BY snippet, (embedding <-> $1::vector)
            )
            SELECT *
            FROM RankedDocs
            ORDER BY similarity ASC
            LIMIT $2
            """
            
            async with self.pool.acquire() as conn:
                results = await conn.fetch(query, embedding_str, limit)
                
                # Additional deduplication in Python
                seen_snippets = set()
                unique_results = []
                
                for row in results:
                    snippet = row['snippet']
                    if snippet not in seen_snippets:
                        seen_snippets.add(snippet)
                        metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'])
                        unique_results.append({
                            "id": str(row['id']),
                            "metadata": metadata,
                            "snippet": snippet,
                            "similarity": float(row['similarity'])
                        })
                
                logger.info(f"Found {len(unique_results)} unique results")
                return unique_results
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    async def store_query_history(
        self,
        query_id: str,
        query: str,
        response: str,
        context: str
    ) -> None:
        """Store query history in the database"""
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                # First, check if the query_history table exists
                check_table = """
                CREATE TABLE IF NOT EXISTS query_history (
                    query_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                await conn.execute(check_table)
                
                # Insert the query history
                insert_query = """
                INSERT INTO query_history (query_id, query, response, context)
                VALUES ($1, $2, $3, $4)
                """
                await conn.execute(
                    insert_query,
                    query_id,
                    query,
                    response,
                    context
                )
                
                logger.info(f"Stored query history with ID: {query_id}")
                
        except Exception as e:
            logger.error(f"Error storing query history: {e}")
            raise

    async def get_query_history(self, query_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve query history from the database"""
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                if query_id:
                    # Get specific query
                    query = """
                    SELECT 
                        query_id,
                        query,
                        response,
                        context,
                        timestamp
                    FROM query_history
                    WHERE query_id = $1
                    """
                    result = await conn.fetch(query, query_id)
                else:
                    # Get all queries, most recent first
                    query = """
                    SELECT 
                        query_id,
                        query,
                        response,
                        context,
                        timestamp
                    FROM query_history
                    ORDER BY timestamp DESC
                    LIMIT 100
                    """
                    result = await conn.fetch(query)
                    
                return [
                    {
                        "query_id": str(row['query_id']),
                        "query": row['query'],
                        "response": row['response'],
                        "context": row['context'],
                        "timestamp": str(row['timestamp'])
                    }
                    for row in result
                ]
                
        except Exception as e:
            logger.error(f"Error retrieving query history: {e}")
            raise

    async def add_vector(
        self,
        document_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        snippet: str
    ) -> None:
        """Add a vector to the database"""
        if not self.pool:
            await self.initialize()
            
        try:
            # Convert embedding to string format for PostgreSQL
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            async with self.pool.acquire() as conn:
                # First ensure the table exists
                create_table = """
                CREATE TABLE IF NOT EXISTS vector_store (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    snippet TEXT,
                    embedding vector(768)
                );
                """
                await conn.execute(create_table)
                
                # Insert the vector
                query = """
                INSERT INTO vector_store (id, metadata, snippet, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (id) DO UPDATE
                SET 
                    metadata = EXCLUDED.metadata,
                    snippet = EXCLUDED.snippet,
                    embedding = EXCLUDED.embedding
                """
                
                await conn.execute(
                    query,
                    document_id,
                    json.dumps(metadata),
                    snippet,
                    embedding_str
                )
                
                logger.info(f"Added vector with ID: {document_id}")
                
        except Exception as e:
            logger.error(f"Error adding vector: {e}")
            raise

    async def init_table(self):
        """Initialize the database tables"""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            # Enable required extensions
            await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            """)
            
            # Create the vector store table
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_store (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                metadata JSONB,
                snippet TEXT,
                embedding vector(768)
            );
            
            -- Create an index for vector similarity search if it doesn't exist
            CREATE INDEX IF NOT EXISTS vector_store_embedding_idx ON vector_store 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            
            # Create a table for document relationships
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_relationships (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_doc TEXT,
                target_doc TEXT,
                relationship_type TEXT,
                context TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_source_doc ON document_relationships(source_doc);
            CREATE INDEX IF NOT EXISTS idx_target_doc ON document_relationships(target_doc);
            """)
            
            # Create query history table
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                query_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                query TEXT NOT NULL,
                response TEXT,
                context TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)

    async def store_relationships(self, relationships_data: Dict[str, Any]):
        """Store document relationships"""
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                for rel in relationships_data.get("relationships", []):
                    await conn.execute("""
                    INSERT INTO document_relationships 
                    (source_doc, target_doc, relationship_type, context)
                    VALUES ($1, $2, $3, $4)
                    """,
                    relationships_data["resource_id"],
                    rel["target"],
                    rel["type"],
                    rel.get("context", "")
                    )
        except Exception as e:
            logger.error(f"Error storing relationships: {e}")
            raise

    async def get_related_documents(self, doc_id: str, max_depth: int = 2):
        """Get related documents up to a certain depth"""
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                query = """
                WITH RECURSIVE related_docs AS (
                    -- Base case
                    SELECT 
                        source_doc,
                        target_doc,
                        relationship_type,
                        context,
                        1 as depth
                    FROM document_relationships
                    WHERE source_doc = $1
                    
                    UNION
                    
                    -- Recursive case
                    SELECT 
                        r.source_doc,
                        r.target_doc,
                        r.relationship_type,
                        r.context,
                        rd.depth + 1
                    FROM document_relationships r
                    INNER JOIN related_docs rd ON r.source_doc = rd.target_doc
                    WHERE rd.depth < $2
                )
                SELECT * FROM related_docs;
                """
                
                results = await conn.fetch(query, doc_id, max_depth)
                return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Error getting related documents: {e}")
            raise

# Create a single instance with explicit postgres credentials
pg_storage = PGVectorStore(database_url="postgresql://postgres:postgres@db:5432/postgres")