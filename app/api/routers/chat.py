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
from app.engine.engine import get_chat_engine
from app.engine.query_filter import generate_filters
from fastapi.responses import JSONResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse


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
        tools=response.sources
    )