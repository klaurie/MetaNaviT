"""
Knowledge Base Query Router

Provides endpoints for:
- Direct document querying
- Knowledge base search
- Non-chat information retrieval
"""

import logging
from fastapi import APIRouter
from app.engine.index import IndexConfig, get_index
from llama_index.core.base.base_query_engine import BaseQueryEngine

# Initialize router for query endpoints
query_router = r = APIRouter()

logger = logging.getLogger("uvicorn")

def get_query_engine() -> BaseQueryEngine:
    """
    Initialize query engine with default settings.
    index is currently using PGSVectorStore
    """
    index_config = IndexConfig(**{})
    index = get_index(index_config)
    return index.as_query_engine()

@r.get(
    "/",
    summary="Get information from the knowledge base",
    description="Retrieves relevant information from the knowledge base based on the provided search query. Returns a text response containing the matched information.",
)
async def query_request(
    query: str,
) -> str:
    """
    Process knowledge base queries.
    
    Args:
        query: Search string to match against documents
        
    Returns:
        Relevant text from knowledge base
    """
    query_engine = get_query_engine()
    response = await query_engine.aquery(query)
    return response.response
