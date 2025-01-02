from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends, BackgroundTasks, Path, Body
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4
import os
import logging
import json
import gc
from app.metadata.metadata_processor import *
from app.utils.helpers import (
    get_ollama_embedding,
    extract_relationships_from_text,
    get_client,
    retry_async,
    get_ollama_response,
    get_ollama_client
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
from app.utils.embeddings import NomicEmbeddings, SDPMChunker
from app.config import UPLOAD_DIR, OLLAMA_HOST
from app.utils.helpers import get_files_recursive
from pathlib import Path as PathLib
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
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
CHUNK_LIMIT = 2  # Reduced chunk limit
SNIPPET_LENGTH = 500  # Optimized snippet length
FILE_QUERY_LIMIT = 1  # Process one file at a time
MAX_TOKENS = 256  # Reduced token limit
TEMPERATURE = 0.7
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

@router.post("/database/init")
async def init_database():
    """Initialize the database and required tables"""
    try:
        await pg_storage.init_table()
        return {"message": "Database initialized successfully"}
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/upload_batch")
async def upload_batch(
    request: DirectoryInput,
    pg_storage=Depends(get_pg_storage)
):
    try:
        directory_path = os.path.abspath(request.directory)
        logger.info(f"Processing directory: {directory_path}")

        # Initialize embeddings and chunkers
        embeddings = NomicEmbeddings()
        semantic_chunker = SDPMChunker(
            embeddings=embeddings,
            similarity_threshold=0.5,
            max_chunk_size=1000,
            min_chunk_size=100,
            skip_window=2
        )

        # Get files
        files = get_files_recursive(directory_path)
        
        results = []
        for file_path in files:
            try:
                # Create metadata processor instance
                metadata_processor = MetadataProcessor()
                
                # Extract metadata and content
                base_metadata, content = await metadata_processor.process_file(file_path)
                
                # Generate chunks
                chunks = await semantic_chunker.chunk_text(content)
                
                # Process chunks into nodes with relationships
                nodes = await metadata_processor.process_chunks(
                    chunks=[c.text for c in chunks],
                    base_metadata=base_metadata
                )
                
                # Store nodes and collect node metadata
                node_metadata = []
                for i, node in enumerate(nodes):
                    # Get embedding for the node
                    embedding = chunks[i].embedding
                    
                    # Prepare relationship metadata
                    relationship_metadata = {
                        "previous_id": node.relationships.get(NodeRelationship.PREVIOUS).node_id if NodeRelationship.PREVIOUS in node.relationships else None,
                        "next_id": node.relationships.get(NodeRelationship.NEXT).node_id if NodeRelationship.NEXT in node.relationships else None
                    }
                    
                    full_metadata = {
                        **node.metadata,
                        "relationships": relationship_metadata,
                        "node_id": node.node_id
                    }
                    
                    await pg_storage.add(
                        embedding=embedding.tolist(),
                        metadata=full_metadata,
                        text=node.get_content(metadata_mode=MetadataMode.ALL)
                    )
                    
                    node_metadata.append(full_metadata)
                
                results.append({
                    "file": file_path,
                    "nodes": len(nodes),
                    "metadata": base_metadata,
                    "node_metadata": node_metadata  # Include detailed node metadata
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                results.append({
                    "file": file_path,
                    "error": str(e)
                })
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

