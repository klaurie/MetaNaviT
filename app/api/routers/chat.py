"""
Chat Router Module

Provides FastAPI endpoints for a streaming style chat response

Features:
- Message history management
- Document filtering
- Event handling
- Error tracking
"""
import os
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from llama_index.core.llms import MessageRole

from app.api.routers.events import EventCallbackHandler
from app.api.routers.models import (
    ChatData,
    Message,
    Result,
    SourceNodes,
)
from app.api.routers.vercel_response import VercelStreamResponse
from app.engine.engine import get_chat_engine, get_engine_factory
from app.engine.query_filter import generate_filters
from fastapi.responses import JSONResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from app.database.vector_store import get_vector_store_manager
from app.engine.llm import get_llm, get_embedding_model


# Initialize router - will be mounted in main app
chat_router = r = APIRouter()

logger = logging.getLogger("uvicorn")

@r.post("")
async def chat(
    request: Request,
    data: ChatData,
    background_tasks: BackgroundTasks,
):
    """
    Streaming chat endpoint.
 
    Flow:
    1. Extract last message and history
    2. Apply document filters
    3. Initialize chat engine
    4. Stream response chunks
    """
    try:
        # Get latest message and conversation history
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()

        # Set up document filtering
        doc_ids = data.get_chat_document_ids()

        # The only filter generated at the moment is public/private file type
        filters = generate_filters(doc_ids)
        params = data.data or {}
        
        # Initialize chat engine with:
        # - filters: Control document visibility (public/private)
        # - params: Custom configuration like temperature/tokens
        # - event_handlers: Track operations and stream chunks
        event_handler = EventCallbackHandler()

        # Chat engine is the agent runner
        chat_engine = get_chat_engine(
            filters=filters,
            params=params,
            event_handlers=[event_handler],
        )
        

            
        # Stream response using Vercel's streaming response
        # (Returns response chunks incrementally)
        response = await chat_engine.achat(last_message_content, messages)

        #logger.info(f"Streaming response: {response.response} and {response.source_nodes}\nType: {type(response)}")
        response.is_dummy_stream = True
        return VercelStreamResponse(
            request=request,
            event_handler=event_handler,
            response=response,
            chat_data=data,
            background_tasks=background_tasks,
        )
        
    except Exception as e:
        logger.exception("Error in chat engine", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat engine: {e}",
        ) from e


# non-streaming endpoint for testing
@r.post("/request")
async def chat_request(
    data: ChatData,
) -> Result:
    try:
        logger.info("Starting chat request processing")
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()
        logger.info(f"Processing chat message: '{last_message_content[:100]}...'")

        doc_ids = data.get_chat_document_ids()
        filters = generate_filters(doc_ids)
        params = data.data or {}
        logger.info(f"Creating chat engine with filters: {str(filters)}")

        # Add step-by-step debug logs
        try:
            logger.info("Step 1: Getting vector store manager")
            vector_store_manager = get_vector_store_manager()
            
            logger.info("Step 2: Getting vector store")
            vector_store = vector_store_manager.get_vector_store()
            
            logger.info("Step 3: Creating VectorStoreIndex")
            from llama_index.core.indices.vector_store import VectorStoreIndex
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            logger.info("Step 4: Creating retriever")
            retriever = index.as_retriever(similarity_top_k=5)
            
            logger.info("Step 5: Getting engine factory")
            factory = get_engine_factory()
            
            logger.info("Step 6: Creating chat engine")
            chat_engine = factory.create_chat_engine(
                retriever=retriever,
                filters=filters,
                params=params,
            )
            
            logger.info("Step 7: Processing chat request")
            response = await chat_engine.achat(last_message_content, messages)
            
            logger.info(f"Chat response generated: {len(response.response)} chars")
            return Result(
                result=Message(role=MessageRole.ASSISTANT, content=response.response),
                nodes=[SourceNodes.from_source_nodes(response.source_nodes)],
            )
            
        except Exception as e:
            # Log detailed exception info including traceback
            logger.exception(f"Error in chat processing: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat processing error: {str(e)}"
            )
            
    except Exception as outer_e:
        logger.exception(f"Outer exception in chat_request: {outer_e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat request error: {str(outer_e)}"
        )

def get_vector_store_chat_engine(filters=None, params=None):
    """Create and return a chat engine instance with vector store retrieval."""
    # Add debugging
    logger.info("Creating chat engine with retriever enabled")
    
    if not filters:
        logger.info("No filters provided, using all documents")
    
    # Ensure vector store is properly connected
    vector_store_manager = get_vector_store_manager()
    vector_store = vector_store_manager.get_vector_store()
    
    # Enable verbose logging for retrieval
    from llama_index.core.settings import Settings
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()
    Settings.debug = True  # Enable debug mode
    
    # Create a VectorStoreIndex around the PostgreSQL vector store
    from llama_index.core.indices.vector_store import VectorStoreIndex
    
    # Create the index with the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Now we can create a retriever from the index
    retriever = index.as_retriever(similarity_top_k=5)
    logger.info(f"Created retriever with top_k=5")
    
    # Create chat engine with retriever
    chat_engine = get_engine_factory().create_chat_engine(
        retriever=retriever,
        filters=filters,
        params=params,
    )
    
    return chat_engine
