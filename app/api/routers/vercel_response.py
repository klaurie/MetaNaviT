"""
Vercel Stream Response Handler

Formats LLM output into Vercel's streaming format:
1. Text Stream (0:): LLM response chunks
2. Data Stream (8:): Events, sources, suggestions
3. Error Stream (3:): Error messages

Flow:
1. Initialize response handlers
2. Merge chat and event streams
3. Process response chunks
4. Add metadata (sources)
"""

import json
import logging
from typing import Awaitable

from aiostream import stream
from fastapi import BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from app.api.routers.events import EventCallbackHandler
from app.api.routers.models import ChatData, SourceNodes

logger = logging.getLogger("uvicorn")


class VercelStreamResponse(StreamingResponse):
    """Formats LLM response for Vercel's streaming protocol"""

    # Stream type prefixes
    TEXT_PREFIX = "0:"   # Raw LLM output
    DATA_PREFIX = "8:"   # Metadata/events
    ERROR_PREFIX = "3:"  # Error messages

    def __init__(
        self,
        request: Request,
        event_handler: EventCallbackHandler,
        response: Awaitable[StreamingAgentChatResponse],
        chat_data: ChatData,
        background_tasks: BackgroundTasks,
    ):
        """Initialize streaming response with handlers"""
        content = self.content_generator(
            request, event_handler, response, chat_data, background_tasks
        )
        super().__init__(content=content)

    @classmethod
    async def content_generator(cls, request, event_handler, response, chat_data, background_tasks):
        """
        Merge and stream chat response with events.
        Handles disconnections and errors.
        """
        chat_response_generator = cls._chat_response_generator(
            response, background_tasks, event_handler, chat_data
        )
        event_generator = cls._event_generator(event_handler)

        # Merge the chat response generator and the event generator
        combine = stream.merge(chat_response_generator, event_generator)
        is_stream_started = False
        try:
            async with combine.stream() as streamer:
                async for output in streamer:
                    if await request.is_disconnected():
                        break

                    if not is_stream_started:
                        is_stream_started = True
                        # Stream a blank message to start displaying the response in the UI
                        yield cls.convert_text("")

                    yield output
        except Exception:
            logger.exception("Error in stream response")
            yield cls.convert_error(
                "An unexpected error occurred while processing your request, preventing the creation of a final answer. Please try again."
            )
        finally:
            # Ensure event handler is marked as done even if connection breaks
            event_handler.is_done = True

    @classmethod
    async def _event_generator(cls, event_handler: EventCallbackHandler):
        """Stream events from callback handler"""
        async for event in event_handler.async_event_gen():
            event_response = event.to_response()
            if event_response is not None:
                yield cls.convert_data(event_response)

    @classmethod
    async def _chat_response_generator(cls, response, background_tasks, event_handler, chat_data):
        """
        Stream chat response with:
        - Source documents
        - Text chunks
        """
        # Wait for the response from the chat engine
        result = await response

        # Yield the source nodes
        yield cls.convert_data(
            {
                "type": "sources",
                "data": {
                    "nodes": [
                        SourceNodes.from_source_node(node).model_dump()
                        for node in result.source_nodes
                    ]
                },
            }
        )

        final_response = ""
        async for token in result.async_response_gen():
            final_response += token
            yield cls.convert_text(token)

        # the text_generator is the leading stream, once it's finished, also finish the event stream
        event_handler.is_done = True

    @classmethod
    def convert_text(cls, token: str):
        """Format text chunks for streaming"""
        # Escape newlines and double quotes to avoid breaking the stream
        token = json.dumps(token)
        return f"{cls.TEXT_PREFIX}{token}\n"

    @classmethod
    def convert_data(cls, data: dict):
        """Format metadata for streaming"""
        data_str = json.dumps(data)
        return f"{cls.DATA_PREFIX}[{data_str}]\n"

    @classmethod
    def convert_error(cls, error: str):
        """Format error messages for streaming"""
        error_str = json.dumps(error)
        return f"{cls.ERROR_PREFIX}{error_str}\n"
