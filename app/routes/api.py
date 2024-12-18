from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends, BackgroundTasks, Path, Body
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4
import os
import logging
import json
import gc
from app.utils.helpers import *
from app.models.api_models import *
from app.db.vector_store import pg_storage
import numpy as np
from app.metadata.metadata_processor import MetadataProcessor
import fnmatch
from typing import Optional, Dict, Any
from fastapi import HTTPException
from uuid import UUID
from app.utils.relationship_extractor import RelationshipExtractor
from pydantic import BaseModel
from datetime import datetime
from llama_index.core.node_parser import SentenceSplitter
from app.utils.embeddings import NomicEmbeddings, SDPMChunker
from app.config import UPLOAD_DIR, OLLAMA_HOST
from app.utils.helpers import get_files_recursive
from pathlib import Path as PathLib

from llama_index.readers.file.docs import PDFReader
import fitz  # PyMuPDF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import io
import base64
from functools import lru_cache
import asyncio
from asyncio import TimeoutError
from typing import AsyncGenerator
import httpx


router = APIRouter()
logger = logging.getLogger(__name__)

class DirectoryInput(BaseModel):
    directory: str

# Constants for optimization
MAX_CONCURRENT_REQUESTS = 1  # Sequential processing
CHUNK_LIMIT = 5  # Reduced chunk limit
SNIPPET_LENGTH = 500  # Optimized snippet length
FILE_QUERY_LIMIT = 1  # Process one file at a time
MAX_TOKENS = 256  # Reduced token limit
TEMPERATURE = 0.3
OLLAMA_TIMEOUT = 60  # Reduced timeout
BATCH_TIMEOUT = 120  # Reduced batch timeout
MODEL_LOAD_WAIT = 5  # Reduced model load wait
VRAM_CLEANUP_DELAY = 2  # Reduced cleanup delay

# CUDA environment variables
os.environ["GGML_CUDA_NO_PINNED"] = "1"  # Disable pinned memory
os.environ["GGML_CUDA_FORCE_CUBLAS"] = "1"  # Force cuBLAS
os.environ["GGML_CUDA_FORCE_FLAT"] = "1"  # Force flat memory

# Semaphore for concurrent processing
ollama_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Helper function to get pg_storage from app state
async def get_pg_storage(request: Request):
    if not hasattr(request.app.state, 'pg_storage'):
        raise HTTPException(status_code=500, detail="Database not initialized")
    return request.app.state.pg_storage



@lru_cache(maxsize=100)
async def get_cached_embedding(text: str) -> List[float]:
    """Cache embeddings to avoid redundant calls"""
    embedding = await get_ollama_embedding(text)
    return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

async def create_ollama_response(query: str, context: str) -> str:
    """Create a new response coroutine for each request"""
    try:
        # Get client for this request
        client = await get_ollama_client()
        
        logger.debug(f"Sending request to Ollama with context length: {len(context)}")
        
        # Create request with minimal parameters and longer timeout
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": f"Context:\n{context}\n\nQuestion: {query}",
                "temperature": 0.7,
                "stream": True,  # Enable streaming
                "num_predict": 256,
                "stop": ["</response>", "\n\n"],
                "context_window": 1024
            },
            timeout=httpx.Timeout(300.0)
        )
        response.raise_for_status()
        
        # Handle streaming response
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
                
            try:
                chunk = json.loads(line)
                if chunk.get("done", False):
                    break
                    
                response_part = chunk.get("response", "")
                full_response += response_part
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chunk: {line}")
                continue
                
        if not full_response:
            logger.error("Empty response from Ollama")
            raise ValueError("Empty response from Ollama")
            
        return full_response.strip()
            
    except (httpx.ReadTimeout, httpx.ConnectTimeout):
        logger.error("Ollama request timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error in create_ollama_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def summarize_chunk(chunk: str) -> str:
    """Summarize a chunk of text while preserving key information"""
    try:
        # Truncate chunk if too long before summarization
        if len(chunk) > 1000:
            chunk = chunk[:1000]
            
        async with ollama_semaphore:
            client = await get_ollama_client()
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": f"Summarize this text in 1-2 short sentences, keeping only the most important information:\n\n{chunk}",
                    "temperature": 0.1,
                    "num_predict": 100,  # Very short summary
                    "stop": ["\n", "</response>"],
                },
                timeout=httpx.Timeout(300.0)  # Very short timeout for summaries
            )
            summary = response.json().get("response", "").strip()
            await client.aclose()
            return summary if summary else chunk[:150]  # Even shorter fallback
    except:
        return chunk[:150]  # Short fallback on error

async def process_file(
    file_name: str,
    embedding: List[float],
    conn,
    query: str
) -> Dict[str, Any]:
    """Process a single file with optimized context building"""
    try:
        # Get chunks for this specific file with reduced limit
        similar_chunks = await pg_storage.get_similar_chunks_for_file(
            embedding=embedding,
            file_name=file_name,
            limit=CHUNK_LIMIT  # Only get the single most relevant chunk
        )
        
        if not similar_chunks:
            return {
                "file_name": file_name,
                "status": "no_relevant_content"
            }
        
        # Build context with single summarized chunk
        chunk = similar_chunks[0]
        summarized_snippet = await summarize_chunk(chunk['snippet'].strip())
        
        # Create minimal context
        context = f"From {file_name}:\n{summarized_snippet}"
        
        try:
            async with ollama_semaphore:
                response = await create_ollama_response(query=query, context=context)
                
                if not response:
                    return {
                        "file_name": file_name,
                        "status": "error",
                        "error": "Empty response from Ollama"
                    }
                
                return {
                    "file_name": file_name,
                    "reasoning": response,
                    "context_used": [chunk],
                    "status": "success"
                }
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            return {
                "file_name": file_name,
                "status": "error",
                "error": str(e)
            }
    except Exception as e:
        logger.error(f"Error getting similar chunks: {str(e)}")
        return {
            "file_name": file_name,
            "status": "error",
            "error": str(e)
        }

@router.post("/ollama/reasoning")
async def get_reasoning(
    query: ReasoningQuery,
    pg_storage=Depends(get_pg_storage)
) -> Dict[str, Any]:
    """Get reasoning based on document context"""
    try:
        # Get embedding with caching
        embedding = await get_ollama_embedding(query.query)
        
        if query.analysis_type == AnalysisType.INDIVIDUAL:
            try:
                async def process_batch():
                    async with pg_storage.pool.acquire() as conn:
                        # Construct file pattern for SQL
                        if query.file_pattern:
                            # Convert glob pattern to SQL LIKE pattern
                            sql_pattern = query.file_pattern.replace('*', '%')
                            if not sql_pattern.startswith('%'):
                                sql_pattern = '%' + sql_pattern
                            if not sql_pattern.endswith('%'):
                                sql_pattern = sql_pattern + '%'
                            files_query = """
                            SELECT DISTINCT metadata->>'file_name' as file_name
                            FROM vector_store
                            WHERE metadata->>'file_name' LIKE $1
                            """
                            params = [sql_pattern]
                        else:
                            files_query = """
                            SELECT DISTINCT metadata->>'file_name' as file_name
                            FROM vector_store
                            WHERE metadata->>'file_name' IS NOT NULL
                            """
                            params = []
                            
                        files = await conn.fetch(files_query, *params)
                        
                        if not files:
                            return {
                                "analysis_type": "individual",
                                "query": query.query,
                                "results": [],
                                "message": "No matching files found"
                            }
                        
                        results = []
                        for file_row in files:
                            result = await process_file(
                                file_row['file_name'],
                                embedding,
                                conn,
                                query.query
                            )
                            results.append(result)
                            await asyncio.sleep(VRAM_CLEANUP_DELAY)
                        
                        successful_results = [
                            r for r in results
                            if r.get("status") == "success"
                        ]
                        
                        # Store the results in RAG history
                        await conn.execute(
                            """
                            INSERT INTO query_history 
                            (query, response, context)
                            VALUES ($1, $2, $3)
                            """,
                            query.query,
                            json.dumps(results),  # Store full results
                            json.dumps({
                                "analysis_type": "individual",
                                "file_pattern": query.file_pattern,
                                "total_files": len(files),
                                "successful_analyses": len(successful_results)
                            })
                        )
                        
                        return {
                            "analysis_type": "individual",
                            "query": query.query,
                            "results": results,
                            "total_files_processed": len(files),
                            "successful_analyses": len(successful_results)
                        }
                
                return await asyncio.wait_for(process_batch(), timeout=BATCH_TIMEOUT)
                
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="Batch processing timed out"
                )
                
        else:  # Aggregate analysis
            try:
                # Get top 5 chunks with optimized limits
                similar_chunks = await pg_storage.get_similar_chunks(
                    embedding=embedding,
                    limit=CHUNK_LIMIT,
                    file_pattern=query.file_pattern
                )
                
                if not similar_chunks:
                    return {
                        "analysis_type": "aggregate",
                        "query": query.query,
                        "message": "No relevant content found"
                    }
                
                # Build context efficiently
                context_parts = []
                chunk_details = []  # New list to store chunk details
                
                for chunk in similar_chunks:
                    file_name = chunk.get('metadata', {}).get('file_name', 'Unknown')
                    snippet = chunk['snippet']
                    chunk_id = str(chunk.get('id', 'Unknown'))  # Get chunk ID
                    similarity = float(chunk.get('similarity', 0.0))  # Get similarity score
                    
                    # Add to context
                    if len(snippet) > SNIPPET_LENGTH:
                        words = snippet.split()
                        truncated_words = words[:SNIPPET_LENGTH//10]
                        snippet = ' '.join(truncated_words) + "..."
                    context_parts.append(f"\nFrom {file_name} (Chunk {chunk_id}):\n{snippet}")
                    
                    # Store chunk details
                    chunk_details.append({
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "similarity_score": round(similarity, 3),
                        "snippet_preview": snippet[:200] + "..." if len(snippet) > 200 else snippet
                    })
                
                context = "\nDocument Excerpts:\n" + "\n".join(context_parts)
                
                try:
                    # Create new response coroutine for this request
                    response = await create_ollama_response(query=query.query, context=context)
                    
                    if not response:
                        raise HTTPException(
                            status_code=500,
                            detail="Empty response from Ollama"
                        )
                    
                    # Store in RAG history
                    async with pg_storage.pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO query_history 
                            (query, response, context)
                            VALUES ($1, $2, $3)
                            """,
                            query.query,
                            response,
                            json.dumps({
                                "analysis_type": "aggregate",
                                "chunks_used": similar_chunks,
                                "file_pattern": query.file_pattern
                            })
                        )
                    
                    return {
                        "analysis_type": "aggregate",
                        "reasoning_result": response,
                        "query": query.query,
                        "chunks_used": chunk_details,  # Include detailed chunk information
                        "total_chunks_found": len(similar_chunks),
                        "status": "success"
                    }
                    
                except Exception as e:
                    logger.error(f"Error in aggregate analysis: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=str(e)
                    )
                    
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="Ollama request timed out"
                )
                
    except Exception as e:
        logger.error(f"Ollama reasoning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ollama/status")
async def ollama_status():
    """Check Ollama service status"""
    try:
        status = await check_ollama_health()
        return status
    except Exception as e:
        logger.error(f"Ollama status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/rag/check-accuracy")
async def check_accuracy(query_id: str):
    """Check the accuracy of a RAG response"""
    try:
        # Validate UUID format
        try:
            UUID(query_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid query ID format")
            
        # Get the query history
        history = await pg_storage.get_query_history(query_id)
        if not history or len(history) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No history found for query ID: {query_id}"
            )
            
        query_record = history[0]
        
        # Get the original sources used
        context_chunks = query_record['context'].split('\n\n')
        
        # Calculate basic metrics
        metrics = {
            "source_count": len(context_chunks),
            "response_length": len(query_record['response']),
            "context_coverage": calculate_coverage(
                query_record['response'],
                query_record['context']
            ),
            "query_relevance": calculate_relevance(
                query_record['query'],
                query_record['response']
            )
        }
        
        return {
            "query_id": query_id,
            "original_query": query_record['query'],
            "response": query_record['response'],
            "metrics": metrics,
            "sources_used": context_chunks,
            "timestamp": query_record['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Error checking accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for metrics
def calculate_coverage(response: str, context: str) -> float:
    """Calculate how much of the response is supported by the context"""
    try:
        # Simple word overlap metric
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        overlap = len(response_words.intersection(context_words))
        coverage = overlap / len(response_words) if response_words else 0
        return round(coverage * 100, 2)  # Return as percentage
    except Exception as e:
        logger.error(f"Error calculating coverage: {e}")
        return 0.0

def calculate_relevance(query: str, response: str) -> float:
    """Calculate query-response relevance"""
    try:
        # Simple word overlap metric between query and response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance = overlap / len(query_words) if query_words else 0
        return round(relevance * 100, 2)  # Return as percentage
    except Exception as e:
        logger.error(f"Error calculating relevance: {e}")
        return 0.0
    

@router.get("/rag/history")
async def get_history(
    query_id: Optional[str] = None,
    pg_storage=Depends(get_pg_storage)
):
    """Get RAG query history"""
    try:
        async with pg_storage.pool.acquire() as conn:
            if query_id:
                query = """
                SELECT query_id, query, response, context, created_at
                FROM query_history
                WHERE id = $1
                """
                result = await conn.fetch(query, query_id)
            else:
                query = """
                SELECT query_id, query, response, context, created_at
                FROM query_history
                ORDER BY created_at DESC
                LIMIT 10
                """
                result = await conn.fetch(query)
            
            history = []
            for row in result:
                history.append({
                    "id": str(row["query_id"]),
                    "query": row['query'],
                    "response": row['response'],
                    "context": row['context'],
                    "timestamp": row['created_at'].isoformat()
                })
            
            return {"history": history}
            
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-relationships")
async def extract_relationships(content: Dict[str, str]):
    """Extract relationships from text content"""
    try:
        if not content.get("content"):
            return {"relationships": []}
            
        extractor = RelationshipExtractor()
        relationships = await extractor.extract_relationships(text=content["content"])
        return {"relationships": relationships}
    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        return {"relationships": []}

@router.post("/upload_batch")
async def upload_batch(
    request: DirectoryInput,
    pg_storage=Depends(get_pg_storage)
):
    """Process and store documents from a directory with semantic chunking."""
    try:
        directory_path = os.path.abspath(request.directory)
        logger.info(f"Processing directory: {directory_path}")

        # Check if directory exists
        if not os.path.exists(directory_path):
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {directory_path}"
            )
            
        # Check if directory is empty
        if not any(os.scandir(directory_path)):
            raise HTTPException(
                status_code=400,
                detail=f"Directory is empty: {directory_path}"
            )
            
        # Check if it's a file or directory
        is_file = os.path.isfile(directory_path)
        if is_file:
            files = [directory_path]
        else:
            files = get_files_recursive(directory_path)
            
        unique_files = list(set(files))
        logger.info(f"Found {len(unique_files)} unique files in {directory_path}: {unique_files}")
        logger.info(f"Processing {len(unique_files)} documents")

        # Initialize embeddings and chunkers
        embeddings = NomicEmbeddings()
        semantic_chunker = SDPMChunker(
            embeddings=embeddings,
            similarity_threshold=0.5,
            max_chunk_size=1000,
            min_chunk_size=100,
            skip_window=2
        )
        
        # Initial sentence splitter for PDF preprocessing
        sentence_splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;]+[,.;]",
        )

        results = []
        
        for file_path in unique_files:
            try:
                file_info = {
                    "directory": os.path.dirname(file_path),
                    "file_name": os.path.basename(file_path),
                    "content_type": "application/pdf" if file_path.lower().endswith('.pdf') else "text/plain",
                    "content_size": os.path.getsize(file_path),
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    "chunks": []
                }

                # Process based on file type
                if file_path.lower().endswith('.pdf'):
                    # Use PyMuPDF for better PDF extraction
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                else:
                    # For non-PDF files, use regular file reading
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                # Clean the text
                text = text.replace('\x00', '')
                text = ' '.join(text.split())  # Normalize whitespace

                # First pass: Split into initial chunks using sentence splitter
                initial_chunks = sentence_splitter.split_text(text)
                
                # Second pass: Apply semantic chunking
                semantic_chunks = []
                for chunk in initial_chunks:
                    if not chunk.strip():
                        continue
                    # Process chunk with semantic chunker
                    chunk_result = await semantic_chunker.chunk_text(chunk)
                    semantic_chunks.extend(chunk_result)
                
                # Process final chunks
                for i, chunk in enumerate(semantic_chunks):
                    try:
                        # Store in vector database
                        doc_id = await pg_storage.add_vector(
                            embedding=chunk.embedding.tolist(),
                            metadata={
                                "directory": file_info["directory"],
                                "file_name": file_info["file_name"],
                                "chunk_index": i,
                                "content_type": file_info["content_type"]
                            },
                            snippet=chunk.text
                        )
                        
                        # Add chunk info to results
                        chunk_info = {
                            "chunk_index": i,
                            "token_size": chunk.token_count,
                            "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                        }
                        
                        # Add similarity with next chunk if available
                        if i < len(semantic_chunks) - 1:
                            similarity = semantic_chunker._cosine_similarity(
                                chunk.embedding,
                                semantic_chunks[i + 1].embedding
                            )
                            chunk_info["similarity_with_next"] = float(similarity)
                        
                        file_info["chunks"].append(chunk_info)
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {i} from {file_path}: {str(chunk_error)}")
                        continue
                
                logger.info(f"Successfully processed {len(file_info['chunks'])} semantic chunks from {file_path}")
                results.append(file_info)
                
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "error": str(e)
                })
                continue

        return {
            "message": "Batch upload completed",
            "path": directory_path,
            "type": "file" if is_file else "directory",
            "total_files": len(unique_files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def save_plot_to_file(plot_data: str, filename: str) -> str:
    """Save a base64 encoded plot to a file"""
    try:
        plot_dir = os.path.join(UPLOAD_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_path = os.path.join(plot_dir, filename)
        with open(plot_path, "wb") as f:
            f.write(base64.b64decode(plot_data))
        return plot_path
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        raise

@router.post("/compare")
async def compare_content(
    source1: str = Body(..., description="Path to first source relative to UPLOAD_DIR"),
    source2: str = Body(..., description="Path to second source relative to UPLOAD_DIR")
):
    """Compare similarities between any combination of files and directories"""
    try:
        # Ensure database pool is initialized
        if not pg_storage.pool:
            await pg_storage.initialize()
            
        async def get_chunks_for_source(path: str):
            """Get chunks for a file or directory"""
            # Clean up path
            path = path.strip('/')
            if path.startswith('uploaded_files/'):
                path = path[len('uploaded_files/'):]
            
            # Construct the full directory path as stored in metadata
            db_dir = f"/app/app/uploaded_files"
            if '/' in path:
                db_dir = f"{db_dir}/{os.path.dirname(path)}"
            
            # Get filename if it's a file
            filename = os.path.basename(path) if '.' in path else None
            
            logger.info(f"Searching for chunks with directory: {db_dir}, filename: {filename}")

            # Query database for chunks
            async with pg_storage.pool.acquire() as conn:
                if filename:  # If it's a file
                    query = """
                    SELECT id, metadata::text, snippet, embedding
                    FROM vector_store
                    WHERE metadata->>'directory' = $1
                    AND metadata->>'file_name' = $2
                    """
                    chunks = await conn.fetch(query, db_dir, filename)
                else:  # If it's a directory
                    query = """
                    SELECT id, metadata::text, snippet, embedding
                    FROM vector_store
                    WHERE metadata->>'directory' LIKE $1 || '%'
                    """
                    chunks = await conn.fetch(query, db_dir)
                
                # Parse metadata JSON strings
                processed_chunks = []
                for chunk in chunks:
                    try:
                        metadata = json.loads(chunk['metadata']) if isinstance(chunk['metadata'], str) else chunk['metadata']
                        processed_chunks.append({
                            'id': chunk['id'],
                            'metadata': metadata,
                            'snippet': chunk['snippet'],
                            'embedding': chunk['embedding']
                        })
                    except Exception as e:
                        logger.error(f"Error processing chunk metadata: {e}")
                        continue
                
                logger.info(f"Found {len(processed_chunks)} chunks for path: {path}")
                return processed_chunks, filename is None

        # Get chunks for both sources
        chunks1, is_dir1 = await get_chunks_for_source(source1)
        chunks2, is_dir2 = await get_chunks_for_source(source2)

        if not chunks1 or not chunks2:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for one or both sources. Source1: {len(chunks1)} chunks, Source2: {len(chunks2)} chunks"
            )

        # Create similarity matrix
        matrix_size1 = len(chunks1)
        matrix_size2 = len(chunks2)
        similarity_matrix = np.zeros((matrix_size1, matrix_size2))
        
        # Calculate full similarity matrix using database vector operations
        async with pg_storage.pool.acquire() as conn:
            for i, chunk1 in enumerate(chunks1):
                query = """
                SELECT 
                    c1.embedding <-> c2.embedding as similarity
                FROM vector_store c2
                CROSS JOIN (SELECT $1::vector as embedding) c1
                WHERE c2.id = ANY($2)
                ORDER BY c2.id
                """
                chunk2_ids = [str(c['id']) for c in chunks2]
                similarities = await conn.fetch(query, chunk1['embedding'], chunk2_ids)
                
                for j, sim in enumerate(similarities):
                    similarity_matrix[i][j] = float(sim['similarity'])

        # Find similar chunks (similarity < 0.5)
        similar_chunks = []
        for i in range(matrix_size1):
            for j in range(matrix_size2):
                similarity = similarity_matrix[i][j]
                if similarity < 0.5:  # Lower score means more similar for distance metric
                    chunk1 = chunks1[i]
                    chunk2 = chunks2[j]
                    similarity_info = {
                        "source1": source1,
                        "source2": source2,
                        "chunk1_text": chunk1['snippet'][:200] + "..." if len(chunk1['snippet']) > 200 else chunk1['snippet'],
                        "chunk2_text": chunk2['snippet'][:200] + "..." if len(chunk2['snippet']) > 200 else chunk2['snippet'],
                        "similarity": round(float(similarity), 3),
                        "file1": chunk1['metadata'].get('file_name', ''),
                        "file2": chunk2['metadata'].get('file_name', '')
                    }
                    similar_chunks.append(similarity_info)

        # Sort by similarity
        similar_chunks.sort(key=lambda x: x["similarity"])

        # Create source labels for matrix
        source1_labels = [
            f"{c['metadata'].get('file_name', '')}:{i+1}" 
            for i, c in enumerate(chunks1)
        ]
        source2_labels = [
            f"{c['metadata'].get('file_name', '')}:{i+1}" 
            for i, c in enumerate(chunks2)
        ]

        # Calculate statistics
        stats = {
            "source1_type": "directory" if is_dir1 else "file",
            "source2_type": "directory" if is_dir2 else "file",
            "total_chunks_1": matrix_size1,
            "total_chunks_2": matrix_size2,
            "similar_chunk_pairs": len(similar_chunks),
            "avg_similarity": round(float(np.mean(similarity_matrix)), 3),
            "max_similarity": round(float(np.min(similarity_matrix)), 3),  # Min distance = max similarity
            "min_similarity": round(float(np.max(similarity_matrix)), 3)   # Max distance = min similarity
        }

        # Add clustering analysis
        # Combine all embeddings for clustering
        all_embeddings = []
        all_metadata = []
        for chunk in chunks1:
            # Parse embedding if it's a string
            if isinstance(chunk['embedding'], str):
                # Remove brackets and split by comma
                embedding_str = chunk['embedding'].strip('[]')
                embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                embedding = np.array(embedding_values)
            else:
                embedding = np.array(chunk['embedding'])
            all_embeddings.append(embedding)
            all_metadata.append({
                'source': 'source1',
                'file': chunk['metadata'].get('file_name', ''),
                'snippet': chunk['snippet'][:100]
            })
        for chunk in chunks2:
            # Parse embedding if it's a string
            if isinstance(chunk['embedding'], str):
                # Remove brackets and split by comma
                embedding_str = chunk['embedding'].strip('[]')
                embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                embedding = np.array(embedding_values)
            else:
                embedding = np.array(chunk['embedding'])
            all_embeddings.append(embedding)
            all_metadata.append({
                'source': 'source2',
                'file': chunk['metadata'].get('file_name', ''),
                'snippet': chunk['snippet'][:100]
            })

        # Convert embeddings to numpy array
        embeddings_array = np.array(all_embeddings)

        # Perform k-means clustering
        n_clusters = min(10, len(all_embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Find closest vectors to centroids
        closest_to_centroids = []
        for i in range(n_clusters):
            # Get distances to centroid for all points in cluster
            cluster_mask = cluster_labels == i
            cluster_points = embeddings_array[cluster_mask]
            if len(cluster_points) == 0:
                continue
                
            # Calculate distances to centroid
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            
            # Get index of closest point
            closest_idx = np.argmin(distances)
            # Map back to original index
            original_idx = np.where(cluster_mask)[0][closest_idx]
            
            closest_to_centroids.append({
                'cluster_id': i,
                'distance_to_centroid': float(distances[closest_idx]),
                'metadata': all_metadata[original_idx],
                'full_text': chunks1[original_idx]['snippet'] if original_idx < len(chunks1) 
                            else chunks2[original_idx - len(chunks1)]['snippet']
            })
            
        # Sort by cluster ID
        closest_to_centroids.sort(key=lambda x: x['cluster_id'])

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)

        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='tab10',
                            alpha=0.6)
        plt.title('Data Clustering')
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                           title="Clusters")
        plt.gca().add_artist(legend1)

        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Calculate cluster information
        cluster_info = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_info.append({
                'cluster_id': i,
                'size': int(np.sum(cluster_mask)),
                'source1_docs': sum(1 for j, mask in enumerate(cluster_mask) 
                                  if mask and all_metadata[j]['source'] == 'source1'),
                'source2_docs': sum(1 for j, mask in enumerate(cluster_mask) 
                                  if mask and all_metadata[j]['source'] == 'source2'),
                'sample_docs': [
                    {
                        'source': all_metadata[j]['source'],
                        'file': all_metadata[j]['file'],
                        'snippet': all_metadata[j]['snippet']
                    }
                    for j in np.where(cluster_mask)[0][:3]  # Get up to 3 samples per cluster
                ]
            })

        # Save plot to file
        plot_filename = f"cluster_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = save_plot_to_file(plot_data, plot_filename)
        logger.info(f"Plot saved to: {plot_path}")

        # Combine all results
        return {
            "source1": source1,
            "source2": source2,
            "source1_type": "directory" if is_dir1 else "file",
            "source2_type": "directory" if is_dir2 else "file",
            "statistics": stats,
            "similarity_matrix": {
                "data": similarity_matrix.tolist(),
                "source1_labels": source1_labels,
                "source2_labels": source2_labels
            },
            "similar_chunks": similar_chunks,
            "clustering_analysis": {
                "n_clusters": n_clusters,
                "plot": plot_data,
                "plot_path": plot_path,
                "clusters": cluster_info,
                "closest_to_centroids": closest_to_centroids
            }
        }

    except Exception as e:
        logger.error(f"Error comparing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

