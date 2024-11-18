from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from pydantic import validator

class AnalysisType(str, Enum):
    AGGREGATE = "aggregate"
    INDIVIDUAL = "individual"

class DirectoryInput(BaseModel):
    """Input model for directory processing"""
    directory: str

class DatabaseInit(BaseModel):
    """Empty model for database initialization"""
    pass

class VectorInsert(BaseModel):
    """Input model for vector insertion"""
    document_chunk: str
    metadata: str

    @validator('metadata')
    def validate_metadata(cls, v):
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError:
            raise ValueError('metadata must be a valid JSON string')

class DocumentChunk(BaseModel):
    """Input model for document chunking"""
    document: str

class EmbeddingGenerate(BaseModel):
    """Input model for embedding generation"""
    document_chunk: str

class SimilarityQuery(BaseModel):
    """Input model for similarity search"""
    query: str
    directory_scope: str

class OllamaQuery(BaseModel):
    """Input model for Ollama queries"""
    query: str

class RAGQuery(BaseModel):
    """Input model for RAG queries"""
    query: str

class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    sources: List[Dict[str, Any]]

class QueryHistory(BaseModel):
    """Model for query history"""
    query_id: str
    query: str
    response: str
    context: str
    timestamp: Optional[str] = None

class AccuracyMetrics(BaseModel):
    """Model for accuracy metrics"""
    relevance: float
    coherence: float
    factual: float

class DocumentMetadata(BaseModel):
    """Model for document metadata"""
    source_document: str
    chunk_index: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None

class ReasoningQuery(BaseModel):
    query: str
    analysis_type: AnalysisType = AnalysisType.AGGREGATE
    file_pattern: Optional[str] = None
    file_name: Optional[str] = None