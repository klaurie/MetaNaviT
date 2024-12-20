import logging
import asyncpg
import asyncio
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class PGVectorStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.logger = logging.getLogger(__name__)
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

    async def add_chunk(
        self,
        chunk: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        chunk_index: int
    ) -> None:
        """Add a chunk to the vector store with proper formatting"""
        try:
            await self.ensure_connection()
            
            # Ensure metadata is a dict and properly formatted
            cleaned_metadata = {
                "directory": metadata.get("directory", ""),
                "file_name": metadata.get("file_name", ""),
                "chunk_index": chunk_index,
                "content_type": metadata.get("content_type", ""),
                "start_char": metadata.get("start_char", 0),  # Add position tracking
                "end_char": metadata.get("end_char", 0),
                "page_number": metadata.get("page_number", 1)  # Add page number for PDFs
            }
            
            async with self.pool.acquire() as conn:
                # Check for duplicate content
                existing = await conn.fetchval(
                    """
                    SELECT COUNT(*) 
                    FROM vector_store 
                    WHERE 
                        metadata->>'file_name' = $1 
                        AND metadata->>'chunk_index' = $2
                        AND snippet = $3
                    """,
                    cleaned_metadata["file_name"],
                    str(cleaned_metadata["chunk_index"]),
                    chunk
                )
                
                if existing > 0:
                    self.logger.warning(f"Duplicate chunk detected for {cleaned_metadata['file_name']}, chunk {chunk_index}")
                    return
                
                # Insert new chunk with properly formatted metadata
                await conn.execute(
                    """
                    INSERT INTO vector_store (metadata, snippet, embedding)
                    VALUES ($1, $2, $3::vector)
                    """,
                    json.dumps(cleaned_metadata),  # Store as JSON
                    chunk,
                    f"[{','.join(str(x) for x in embedding)}]"
                )
                
        except Exception as e:
            self.logger.error(f"Error adding chunk: {e}")
            raise

    async def similar_chunks(
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
            
            # Handle file pattern matching
            if file_pattern:
                # Convert glob pattern to SQL LIKE pattern
                sql_pattern = file_pattern.replace('*', '%')
                if not sql_pattern.startswith('%'):
                    sql_pattern = '%' + sql_pattern
                if not sql_pattern.endswith('%'):
                    sql_pattern = sql_pattern + '%'
                files_query += " AND metadata->>'file_name' LIKE $1"
                params = [sql_pattern]
            else:
                params = []
            
            async with self.pool.acquire() as conn:
                matching_files = await conn.fetch(files_query, *params)
                logger.info(f"Found matching files: {[r['filename'] for r in matching_files]}")
                
                processed_results = []
                
                # Get chunks from each file separately
                for file_row in matching_files:
                    file_name = file_row['filename']
                    
                    # Query for this specific file
                    file_query = """
                    WITH RankedChunks AS (
                        SELECT 
                            id,
                            metadata,
                            snippet,
                            embedding <-> $1::vector as distance
                        FROM vector_store
                        WHERE metadata->>'file_name' = $2
                    )
                    SELECT 
                        id,
                        metadata,
                        snippet,
                        distance as similarity
                    FROM RankedChunks
                    ORDER BY distance ASC
                    LIMIT $3;
                    """
                    
                    # Get top chunks from this file
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
        embedding: Union[List[float], np.ndarray],
        file_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar chunks with proper metadata handling"""
        try:
            if not self.pool:
                await self.initialize()
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            async with self.pool.acquire() as conn:
                # Modified query to ensure unique chunks and proper ordering
                chunks = await conn.fetch(
                    """
                    WITH RankedChunks AS (
                        SELECT 
                            id,
                            metadata,
                            snippet,
                            1 - (embedding <=> $1::vector) as similarity,
                            (metadata->>'chunk_index')::int as chunk_index,
                            (metadata->>'page_number')::int as page_number
                        FROM vector_store
                        WHERE metadata->>'file_name' = $2
                    )
                    SELECT DISTINCT ON (snippet) 
                        id,
                        metadata,
                        snippet,
                        similarity
                    FROM RankedChunks
                    ORDER BY snippet, similarity DESC, page_number ASC, chunk_index ASC
                    LIMIT $3
                    """,
                    embedding_str,
                    file_name,
                    limit
                )
                
                # Format the response with parsed metadata
                return [
                    {
                        'id': str(chunk['id']),
                        'metadata': json.loads(chunk['metadata']),  # Parse JSON to dict
                        'snippet': chunk['snippet'],
                        'similarity': float(chunk['similarity'])
                    }
                    for chunk in chunks
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting similar chunks for file {file_name}: {str(e)}")
            raise

    async def get_similar_chunks_for_directory(
        self,
        embedding: Union[List[float], np.ndarray],
        directory_pattern: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar chunks across all files in a directory"""
        try:
            if not self.pool:
                await self.initialize()
            
            # Convert numpy array to list if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Convert to PostgreSQL vector format
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            async with self.pool.acquire() as conn:
                # Query with DISTINCT ON to avoid duplicates
                if directory_pattern:
                    chunks = await conn.fetch(
                        """
                        SELECT DISTINCT ON (snippet)
                            id,
                            metadata,
                            snippet,
                            1 - (embedding <=> $1::vector) as similarity
                        FROM vector_store
                        WHERE metadata->>'file_name' LIKE $2
                        ORDER BY snippet, similarity DESC
                        LIMIT $3
                        """,
                        embedding_str,
                        f"%{directory_pattern}%",
                        limit
                    )
                else:
                    chunks = await conn.fetch(
                        """
                        SELECT DISTINCT ON (snippet)
                            id,
                            metadata,
                            snippet,
                            1 - (embedding <=> $1::vector) as similarity
                        FROM vector_store
                        ORDER BY snippet, similarity DESC
                        LIMIT $2
                        """,
                        embedding_str,
                        limit
                    )
                
                self.logger.info(f"Found {len(chunks)} unique chunks in directory search")
                
                return [
                    {
                        'id': str(chunk['id']),
                        'metadata': chunk['metadata'],
                        'snippet': chunk['snippet'],
                        'similarity': float(chunk['similarity'])
                    }
                    for chunk in chunks
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting similar chunks for directory: {str(e)}")
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

    async def add_vector(self, embedding: List[float], metadata: Dict[str, Any], snippet: str) -> str:
        """Add a vector to the store"""
        try:
            if not self.pool:
                await self.initialize()

            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)

            async with self.pool.acquire() as conn:
                # Insert the vector and return the generated ID
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO vector_store (embedding, metadata, snippet)
                    VALUES ($1::vector, $2::jsonb, $3)
                    RETURNING id::text
                    """,
                    embedding_str, metadata_json, snippet
                )
                return doc_id
        except Exception as e:
            logger.error(f"Error adding vector to store: {e}")
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

    async def get_similar_chunks(
        self,
        embedding: Union[List[float], np.ndarray],
        limit: int = 5,
        file_pattern: str = None,
        file_path: str = None,
        directory: str = None
    ) -> List[Dict[str, Any]]:
        """Get similar chunks across all files or matching pattern/path/directory"""
        try:
            if not self.pool:
                await self.initialize()
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            async with self.pool.acquire() as conn:
                query = """
                    WITH SimilarChunks AS (
                        SELECT DISTINCT ON (snippet)
                            id,
                            metadata,
                            snippet,
                            1 - (embedding <=> $1::vector) as similarity
                        FROM vector_store
                        WHERE 1=1
                """
                
                params = [embedding_str]
                
                # Handle different types of filtering
                if directory:
                    # Directory-based filtering
                    query += " AND metadata->>'directory' = $2"
                    params.append(directory)
                    self.logger.info(f"Filtering by directory: {directory}")
                elif file_path:
                    # File path-based filtering
                    dir_path = os.path.dirname(file_path)
                    filename = os.path.basename(file_path)
                    
                    query += """
                        AND metadata->>'directory' = $2 
                        AND metadata->>'file_name' = $3
                    """
                    params.extend([dir_path, filename])
                    self.logger.info(f"Filtering by directory: {dir_path} and file: {filename}")
                elif file_pattern:
                    # Pattern-based filtering
                    sql_pattern = file_pattern.replace('*', '%')
                    query += " AND metadata->>'file_name' LIKE $2"
                    params.append(sql_pattern)
                    self.logger.info(f"Filtering by pattern: {sql_pattern}")
                
                query += """
                        ORDER BY snippet, similarity DESC
                    )
                    SELECT * FROM SimilarChunks
                    ORDER BY similarity DESC
                    LIMIT $""" + str(len(params) + 1)
                
                params.append(limit)
                
                # Debug: Log the query and parameters
                self.logger.info(f"Executing query: {query}")
                self.logger.info(f"With parameters: {params}")
                
                chunks = await conn.fetch(query, *params)
                self.logger.info(f"Found {len(chunks)} chunks")
                
                # Process chunks with proper JSON parsing
                processed_chunks = []
                for chunk in chunks:
                    try:
                        metadata = json.loads(chunk['metadata']) if isinstance(chunk['metadata'], str) else chunk['metadata']
                        
                        chunk_data = {
                            'id': str(chunk['id']),
                            'file_name': metadata.get('file_name', 'Unknown'),
                            'directory': metadata.get('directory', 'Unknown'),
                            'chunk_index': int(metadata.get('chunk_index', 0)),
                            'content_type': metadata.get('content_type', 'Unknown'),
                            'snippet': chunk['snippet'],
                            'similarity': float(chunk['similarity']),
                            'snippet_preview': chunk['snippet'][:200] + "..." if len(chunk['snippet']) > 200 else chunk['snippet']
                        }
                        
                        processed_chunks.append(chunk_data)
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing metadata JSON: {e}")
                        self.logger.error(f"Raw metadata: {chunk['metadata']}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing chunk: {e}")
                        self.logger.error(f"Problematic chunk data: {chunk}")
                        continue
                
                if not processed_chunks:
                    # Debug: show available files and their locations
                    files = await conn.fetch(
                        """
                        SELECT DISTINCT 
                            metadata->>'file_name' as name,
                            metadata->>'directory' as directory,
                            COUNT(*) as chunk_count
                        FROM vector_store
                        GROUP BY metadata->>'file_name', metadata->>'directory'
                        ORDER BY metadata->>'directory', metadata->>'file_name'
                        """
                    )
                    self.logger.warning("No chunks found. Available files:")
                    for file in files:
                        self.logger.warning(f"  - {file['directory']}/{file['name']} ({file['chunk_count']} chunks)")
                
                return processed_chunks
                
        except Exception as e:
            self.logger.error(f"Error getting similar chunks: {str(e)}")
            self.logger.error(f"Full error details: ", exc_info=True)
            raise

# First create the global instance
pg_storage = PGVectorStore(database_url="postgresql://postgres:postgres@db:5432/postgres")

# Then define the function that uses it
async def get_pg_storage():
    """Get the PGVectorStore instance and ensure it's initialized"""
    global pg_storage  # Add this line to explicitly use the global variable
    if pg_storage.pool is None:
        await pg_storage.initialize()
    return pg_storage
