from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends
from typing import Dict, Any, Optional, List
from uuid import uuid4
import os
import logging
import json
from app.utils.helpers import (
    get_ollama_embedding,
    extract_relationships_from_text,
    get_client,
    retry_async
)
from app.models.api_models import (
    DirectoryInput,
    DatabaseInit,
    VectorInsert,
    DocumentChunk,
    EmbeddingGenerate,
    SimilarityQuery,
    OllamaQuery,
    RAGQuery,
    QueryResponse,
    QueryHistory,
    DocumentMetadata,
    AnalysisType,
    ReasoningQuery
)
from app.db.vector_store import pg_storage
import numpy as np
from app.text_splitter import TextSplitter
from app.metadata.metadata_processor import MetadataProcessor
import fnmatch
from typing import Optional, Dict, Any
from fastapi import HTTPException
from uuid import UUID
from llama_index.core import SimpleDirectoryReader, Document
from app.utils.relationship_extractor import RelationshipExtractor
from pydantic import BaseModel
from datetime import datetime
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

router = APIRouter()
logger = logging.getLogger(__name__)

class DirectoryInput(BaseModel):
    directory: str

# Helper function to get pg_storage from app state
async def get_pg_storage(request: Request):
    if not hasattr(request.app.state, 'pg_storage'):
        raise HTTPException(status_code=500, detail="Database not initialized")
    return request.app.state.pg_storage

@router.post("/database/init")
async def init_database():
    """Initialize the database and required tables"""
    try:
        await pg_storage.init_table()
        return {"message": "Database initialized successfully"}
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database/insert-vector")
async def insert_vector(vector_data: VectorInsert):
    """Insert a document vector into the database"""
    try:
        embedding = await get_ollama_embedding(vector_data.document_chunk)
        document_id = str(uuid4())
        
        await pg_storage.add_vector(
            document_id=document_id,
            embedding=embedding,
            metadata=json.loads(vector_data.metadata),
            snippet=vector_data.document_chunk[:300]
        )
        
        return {
            "message": "Vector inserted successfully",
            "document_id": document_id
        }
    except Exception as e:
        logger.error(f"Vector insertion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/document/chunk")
async def create_document_chunks(doc_data: DocumentChunk):
    """Create chunks from a document using LlamaIndex"""
    try:
        file_path = os.path.join("/app/uploaded_files", doc_data.document)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
            
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            num_workers=1  # Single file, so 1 worker is sufficient
        )
        
        documents = reader.load_data()
        chunk_ids = []
        
        for doc in documents:
            chunk_id = str(uuid4())
            embedding = await get_ollama_embedding(doc.text)
            
            await pg_storage.add_vector(
                document_id=chunk_id,
                embedding=embedding,
                metadata={"source_document": doc_data.document, **doc.metadata},
                snippet=doc.text[:300]
            )
            
            chunk_ids.append(chunk_id)
        
        return {
            "message": f"Document {doc_data.document} processed successfully",
            "chunks": len(documents),
            "chunk_ids": chunk_ids
        }
    except Exception as e:
        logger.error(f"Document chunking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/document/metadata")
async def get_document_metadata(documentId: str):
    """Get metadata for a document"""
    try:
        metadata = await pg_storage.get_metadata(documentId)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"document_id": documentId, "metadata": metadata}
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/generate")
async def generate_embeddings(embedding_data: EmbeddingGenerate):
    """Generate embeddings for a document chunk"""
    try:
        embedding = await get_ollama_embedding(embedding_data.document_chunk)
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding),
            "document_chunk": embedding_data.document_chunk[:100] + "..."
        }
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/similarity-search")
async def similarity_search(query_data: SimilarityQuery):
    """Perform similarity search"""
    try:
        query_embedding = await get_ollama_embedding(query_data.query)
        results = await pg_storage.similarity_search(
            query_embedding=query_embedding,
            directory=query_data.directory_scope,
            limit=5
        )
        
        return {
            "results": results,
            "query": query_data.query,
            "directory": query_data.directory_scope
        }
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ollama/reasoning")
async def get_reasoning(query: ReasoningQuery) -> Dict[str, Any]:
    try:
        # Get embedding for the query
        embedding = await get_ollama_embedding(query.query)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        if query.analysis_type == AnalysisType.INDIVIDUAL:
            # Process each file individually
            results = []
            async with pg_storage.pool.acquire() as conn:
                # Get distinct file names from metadata
                files_query = """
                SELECT DISTINCT metadata->>'file_name' as file_name
                FROM vector_store
                WHERE metadata->>'file_name' IS NOT NULL
                """
                if query.file_pattern:
                    files_query += f" AND metadata->>'file_name' LIKE '{query.file_pattern}'"
                
                files = await conn.fetch(files_query)
                
                for file_row in files:
                    file_name = file_row['file_name']
                    # Get chunks for this specific file
                    similar_chunks = await pg_storage.get_similar_chunks_for_file(
                        embedding=embedding,
                        file_name=file_name,
                        limit=6
                    )
                    
                    if similar_chunks:
                        context = f"\nDocument Excerpts from {file_name}:\n"
                        for chunk in similar_chunks:
                            context += f"\n{chunk['snippet']}\n"
                        
                        reasoning = await get_ollama_response(query.query, context)
                        
                        results.append({
                            "file_name": file_name,
                            "reasoning": reasoning,
                            "context_used": similar_chunks
                        })
            
            return {
                "analysis_type": "individual",
                "query": query.query,
                "results": results
            }
            
        else:
            # Aggregate analysis
            similar_chunks = await pg_storage.get_similar_chunks(
                embedding=embedding,
                limit=6,
                file_pattern=query.file_pattern
            )
            
            if not similar_chunks:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant content found in the database"
                )
            
            context = "\nDocument Excerpts:\n"
            for chunk in similar_chunks:
                metadata = chunk.get('metadata', {})
                doc_name = metadata.get('file_name', 'Unknown Document')
                context += f"\nFrom {doc_name}:\n{chunk['snippet']}\n"
            
            reasoning = await get_ollama_response(query.query, context)
            
            return {
                "analysis_type": "aggregate",
                "reasoning_result": reasoning,
                "context_used": similar_chunks,
                "query": query.query
            }
            
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

@router.post("/rag/retrieve-and-generate")
async def retrieve_and_generate(query_data: RAGQuery):
    """RAG pipeline implementation"""
    try:
        # Generate embedding for query
        query_embedding = await get_ollama_embedding(query_data.query)
        
        # Retrieve relevant documents
        relevant_docs = await pg_storage.similarity_search(query_embedding, limit=3)
        
        # Generate response using retrieved context
        context = "\n\n".join([doc["snippet"] for doc in relevant_docs])
        response = await get_ollama_response(query_data.query, context)
        
        # Store query history
        query_id = str(uuid4())
        await pg_storage.store_query_history(
            query_id=query_id,
            query=query_data.query,
            response=response,
            context=context
        )
        
        return {
            "query_id": query_id,
            "answer": response,
            "sources": relevant_docs
        }
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
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

@router.post("/upload/batch")
async def upload_batch(request: DirectoryInput, pg_storage=Depends(get_pg_storage)):
    """Process and store documents from a directory"""
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

        # Track processed files to prevent duplicates
        processed_paths = set()
        
        # Initialize document reader with recursive exploration
        try:
            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=True,
                exclude_hidden=True,
                required_exts=[".txt", ".pdf", ".doc", ".docx"],
                num_files_limit=None
            )
            documents = reader.load_data()
            
            # Filter out duplicate documents
            unique_documents = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                abs_path = os.path.abspath(file_path)
                if abs_path not in processed_paths:
                    processed_paths.add(abs_path)
                    unique_documents.append(doc)
            
            logger.info(f"Found {len(unique_documents)} unique files: {[doc.metadata.get('file_path', '') for doc in unique_documents]}")
            documents = unique_documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading documents: {str(e)}"
            )

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents found in directory"
            )
            
        logger.info(f"Processing {len(documents)} documents")

        successful = 0
        failed = 0
        file_details = []

        # Initialize text splitter once
        splitter = TextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=100,  # Increased overlap for better context
        )

        for doc in documents:
            try:
                # Validate document has content
                if not isinstance(doc.text, str) or not doc.text.strip():
                    logger.warning(f"Empty document found: {doc.metadata.get('file_path', 'unknown')}")
                    continue

                file_path = doc.metadata.get("file_path", "")
                if not file_path or not os.path.exists(file_path):
                    logger.warning(f"Invalid file path: {file_path}")
                    continue

                abs_path = os.path.abspath(file_path)
                file_stat = os.stat(file_path)
                
                # Get file metadata
                file_info = {
                    "directory": os.path.dirname(abs_path),
                    "file_name": os.path.basename(file_path),
                    "content_type": doc.metadata.get("content_type", "text/plain"),
                    "content_size": file_stat.st_size,
                    "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "chunks": []
                }

                try:
                    # Split text into chunks with better handling of PDFs
                    text = doc.text.replace('\x00', '')  # Remove null bytes
                    text = ' '.join(text.split())  # Normalize whitespace
                    
                    # Split into initial chunks
                    chunks = splitter.split_text(text)
                    
                    # Additional cleanup and filtering of chunks
                    valid_chunks = []
                    for chunk in chunks:
                        chunk = chunk.strip()
                        # Only keep chunks that are meaningful (not too short and contain actual content)
                        if chunk and len(chunk) >= 50 and not chunk.isspace():
                            valid_chunks.append(chunk)
                    
                    logger.info(f"Split document {file_path} into {len(valid_chunks)} chunks")
                    chunks = valid_chunks
                    
                except Exception as e:
                    logger.error(f"Error splitting document {file_path}: {str(e)}")
                    continue
                
                # Validate chunks
                valid_chunks = [c for c in chunks if c and c.strip()]
                if not valid_chunks:
                    logger.warning(f"No valid chunks found in document: {file_path}")
                    continue
                
                logger.info(f"Processing {len(valid_chunks)} valid chunks for {file_path}")
                chunk_errors = 0
                
                # Process each chunk
                for i, chunk in enumerate(valid_chunks):
                    try:
                        # Get embedding
                        embedding = await get_ollama_embedding(chunk)
                        if embedding is None or (isinstance(embedding, np.ndarray) and embedding.size == 0):
                            logger.warning(f"Failed to get embedding for chunk {i} in {file_path}")
                            chunk_errors += 1
                            continue

                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()

                        # Calculate token size (approximate)
                        token_size = len(chunk.split())

                        # Store chunk info
                        chunk_info = {
                            "chunk_index": i,
                            "token_size": token_size,
                            "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                        }
                        file_info["chunks"].append(chunk_info)

                        # Store in database with metadata
                        doc_id = await pg_storage.add_vector(
                            embedding=embedding,
                            metadata={
                                "file_name": os.path.basename(file_path),
                                "directory": os.path.dirname(abs_path),
                                "content_type": doc.metadata.get("content_type", "text/plain"),
                                "content_size": file_stat.st_size,
                                "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                                "chunk_index": i,
                                "token_size": token_size,
                                "total_chunks": len(valid_chunks)
                            },
                            snippet=chunk
                        )
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {i} in {file_path}: {str(chunk_error)}")
                        chunk_errors += 1
                        continue

                if file_info["chunks"]:
                    logger.info(f"Successfully processed {len(file_info['chunks'])} chunks for {file_path} with {chunk_errors} errors")
                    file_details.append(file_info)
                    successful += 1
                else:
                    logger.error(f"Failed to process any chunks for {file_path}")
                    failed += 1
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                failed += 1

        if not file_details:
            raise HTTPException(
                status_code=400,
                detail="No files were successfully processed"
            )

        return {
            "message": f"Directory {directory_path} processed",
            "summary": {
                "total_files": len(documents),
                "successful": successful,
                "failed": failed,
                "total_directories": len(set(info["directory"] for info in file_details))
            },
            "processed_files": file_details
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/upload/batch_extract")
async def batch_extract_relationships(request: DirectoryInput, pg_storage=Depends(get_pg_storage)):
    """Extract relationships from previously processed documents"""
    try:
        directory_path = os.path.abspath(request.directory)
        logger.info(f"Extracting relationships from directory: {directory_path}")

        # Check if directory exists
        if not os.path.exists(directory_path):
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {directory_path}"
            )

        # Initialize metadata processor
        metadata_processor = MetadataProcessor(pg_storage=pg_storage)
        
        # Process relationships
        relationship_results = await metadata_processor.process_directory(directory_path)
        
        # Track failed extractions
        failed_extractions = []
        for result in relationship_results:
            if not result.get("relationships", {}).get("relationships"):
                failed_extractions.append(result.get("file_name"))

        return {
            "message": f"Directory {directory_path} processed",
            "debug_info": {
                "relationship_results": relationship_results,
                "failed_extractions": failed_extractions
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in batch relationship extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/chunks")
async def get_debug_chunks():
    """Debug endpoint to check stored chunks"""
    try:
        async with pg_storage.pool.acquire() as conn:
            query = "SELECT COUNT(*) as count FROM vector_store"
            result = await conn.fetchval(query)
            
            # Get a sample of stored chunks
            sample_query = """
            SELECT id, metadata, substring(snippet, 1, 100) as snippet_preview 
            FROM vector_store 
            LIMIT 5
            """
            samples = await conn.fetch(sample_query)
            
            return {
                "total_chunks": result,
                "sample_chunks": [
                    {
                        "id": str(row['id']),
                        "metadata": row['metadata'],
                        "snippet_preview": row['snippet_preview']
                    }
                    for row in samples
                ]
            }
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/files")
async def get_stored_files():
    """Get list of files stored in the database"""
    try:
        async with pg_storage.pool.acquire() as conn:
            query = """
            SELECT DISTINCT metadata->>'file_name' as file_name,
                   COUNT(*) as chunk_count
            FROM vector_store
            WHERE metadata->>'file_name' IS NOT NULL
            GROUP BY metadata->>'file_name'
            """
            results = await conn.fetch(query)
            
            return {
                "files": [
                    {
                        "file_name": row['file_name'],
                        "chunk_count": row['chunk_count']
                    }
                    for row in results
                ]
            }
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/metadata")
async def check_metadata_format():
    """Check the format of metadata in the database"""
    try:
        async with pg_storage.pool.acquire() as conn:
            query = """
            SELECT id, metadata
            FROM vector_store
            LIMIT 5
            """
            results = await conn.fetch(query)
            
            return {
                "samples": [
                    {
                        "id": str(row['id']),
                        "metadata_type": str(type(row['metadata'])),
                        "metadata": row['metadata']
                    }
                    for row in results
                ]
            }
    except Exception as e:
        logger.error(f"Metadata check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/chunks-per-file")
async def check_chunks_distribution():
    """Check how many chunks we have per file"""
    try:
        async with pg_storage.pool.acquire() as conn:
            query = """
            SELECT 
                metadata->>'file_name' as filename,
                COUNT(*) as chunk_count
            FROM vector_store
            WHERE metadata->>'file_name' IS NOT NULL
            GROUP BY metadata->>'file_name'
            """
            results = await conn.fetch(query)
            
            return {
                "distribution": [
                    {
                        "file_name": row['filename'],
                        "chunk_count": row['chunk_count']
                    }
                    for row in results
                ]
            }
    except Exception as e:
        logger.error(f"Distribution check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/history")
async def get_history(query_id: Optional[str] = None):
    """Get RAG query history"""
    try:
        history = await pg_storage.get_query_history(query_id)
        return {
            "history": history
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/history/{query_id}")
async def get_specific_history(query_id: str):
    """Get specific RAG query history by ID"""
    try:
        history = await pg_storage.get_query_history(query_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"Query history not found for ID: {query_id}")
        return history[0]  # Return the specific query history
    except Exception as e:
        logger.error(f"Error retrieving history for {query_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/vectors")
async def get_vectors(limit: int = 10):
    """Get stored vectors from the database"""
    try:
        async with pg_storage.pool.acquire() as conn:
            query = """
            SELECT 
                id,
                metadata,
                snippet,
                embedding
            FROM vector_store
            LIMIT $1
            """
            results = await conn.fetch(query, limit)
            
            return {
                "vectors": [
                    {
                        "id": str(row['id']),
                        "metadata": json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                        "snippet": row['snippet'],
                        "embedding_size": len(row['embedding'])
                    }
                    for row in results
                ]
            }
    except Exception as e:
        logger.error(f"Error getting vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/relationships")
async def get_stored_relationships(
    pg_storage = Depends(get_pg_storage)
):
    """Get all stored relationships"""
    try:
        async with pg_storage.pool.acquire() as conn:
            # Check document_relationships table
            relationships = await conn.fetch("""
                SELECT * FROM document_relationships
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            # Check vector_store metadata
            vectors = await conn.fetch("""
                SELECT id, metadata->>'relationships' as relationships
                FROM vector_store
                WHERE metadata->>'relationships' IS NOT NULL
                LIMIT 10
            """)
            
            return {
                "relationships_table": [dict(r) for r in relationships],
                "vector_metadata": [dict(v) for v in vectors]
            }
    except Exception as e:
        logger.error(f"Error fetching relationships: {e}")
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

