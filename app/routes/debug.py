from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import numpy as np
from app.embeddings.ollama import get_ollama_embedding
from app.db.vector_store import get_pg_storage, PGVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])

# Pydantic models for request validation
class VectorInsertRequest(BaseModel):
    document_chunk: str
    metadata: Dict[str, Any]

class SimilaritySearchRequest(BaseModel):
    query: str
    directory_scope: Optional[str] = None

class ReasoningRequest(BaseModel):
    query: str
    file_pattern: Optional[str] = None
    chunk_limit: int = 5
    chunk_size: int = 500
    context_window: int = 1024
    num_predict: int = 256
    timeout: float = 30.0

@router.post("/database/init")
async def init_database(db: PGVectorStore = Depends(get_pg_storage)):
    """Initialize database connection"""
    try:
        await db.initialize()
        return {"status": "success", "message": "Database initialized"}
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database/insert-vector")
async def insert_vector(
    request: VectorInsertRequest,
    db: PGVectorStore = Depends(get_pg_storage)
):
    """Insert a vector into the database"""
    try:
        # Get embedding and convert to list if it's a numpy array
        embedding = await get_ollama_embedding(request.document_chunk)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Add to database
        await db.add_chunk(request.document_chunk, embedding, request.metadata)
        return {"status": "success", "message": "Vector inserted"}
    except Exception as e:
        logger.error(f"Vector insertion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/similarity-search")
async def similarity_search(
    request: SimilaritySearchRequest,
    db: PGVectorStore = Depends(get_pg_storage)
):
    """Perform similarity search"""
    try:
        embedding = await get_ollama_embedding(request.query)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        results = await db.similar_chunks(
            embedding=embedding,
            file_pattern=request.directory_scope
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning")
async def debug_reasoning(
    request: ReasoningRequest,
    db: PGVectorStore = Depends(get_pg_storage)
):
    """Debug reasoning endpoint"""
    try:
        query_embedding = await get_ollama_embedding(request.query)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        chunks = await db.similar_chunks(
            embedding=query_embedding,
            limit=request.chunk_limit,
            file_pattern=request.file_pattern
        )
        
        return {
            "status": "success",
            "chunks": chunks,
            "parameters": {
                "chunk_limit": request.chunk_limit,
                "chunk_size": request.chunk_size,
                "context_window": request.context_window,
                "num_predict": request.num_predict,
                "timeout": request.timeout
            }
        }
    except Exception as e:
        logger.error(f"Reasoning debug failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))