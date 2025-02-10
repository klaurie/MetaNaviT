"""
Chat Router Module

Provides FastAPI endpoints for a streaming style chat response

Features:
- Message history management
- Document filtering
- Event handling
- Error tracking
"""

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
from app.engine.engine import get_chat_engine
from app.engine.query_filter import generate_filters

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
        
        logger.info(f"Creating chat engine with filters: {str(filters)}")
        
        # Initialize chat engine with:
        # - filters: Control document visibility (public/private)
        # - params: Custom configuration like temperature/tokens
        # - event_handlers: Track operations and stream chunks
        event_handler = EventCallbackHandler()
        chat_engine = get_chat_engine(
            filters=filters,
            params=params,
            event_handlers=[event_handler]
        )
        
        # Stream response using Vercel's streaming response
        # (Returns response chunks incrementally)
        response = chat_engine.astream_chat(last_message_content, messages)
        return VercelStreamResponse(
            request, event_handler, response, data, background_tasks
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
    last_message_content = data.get_last_message_content()
    messages = data.get_history_messages()

    doc_ids = data.get_chat_document_ids()
    filters = generate_filters(doc_ids)
    params = data.data or {}
    logger.info(
        f"Creating chat engine with filters: {str(filters)}",
    )

    chat_engine = get_chat_engine(filters=filters, params=params)

    response = await chat_engine.achat(last_message_content, messages)
    return Result(
        result=Message(role=MessageRole.ASSISTANT, content=response.response),
        nodes=SourceNodes.from_source_nodes(response.source_nodes),
    )
