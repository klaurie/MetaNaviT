"""
LLM Event Handler Module

Handles events from LLM operations:
- Document retrieval events
- Tool execution events
- Response streaming events
- Error tracking

Flow:
1. LLM Operation (retrieve/tool/response) -> Triggers event
2. EventCallbackHandler -> Processes event
3. CallbackEvent -> Formats message
4. AsyncQueue -> Buffers events
5. Frontend -> Displays events
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.tools.types import ToolOutput
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CallbackEvent(BaseModel):
    """
    Models events from LLM operations for frontend display.
    
    Fields:
    - event_type: Type of LLM operation (retrieve, llm, tool, etc)
    - payload: Event-specific data
    - event_id: Unique identifier for event tracking
    """
    event_type: CBEventType
    payload: Optional[Dict[str, Any]] = None
    event_id: str = ""

    def get_retrieval_message(self) -> dict | None:
        """
        Format document retrieval events for frontend.
        Shows number of sources or query context.
        """
        if self.payload:
            # Extract nodes from payload (contains retrieved documents)
            nodes = self.payload.get("nodes")
            if nodes:
                # If nodes found, show count of retrieved sources
                msg = f"Retrieved {len(nodes)} sources to use as context for the query"
            else:
                # If no nodes, show query being processed
                msg = f"Retrieving context for query: '{self.payload.get('query_str')}'"
            # Return formatted event message
            return {
                "type": "events",
                "data": {"title": msg},
            }
        return None

    def get_tool_message(self) -> dict | None:
        """
        Format tool execution events for frontend.
        Shows tool name and input parameters.
        """
        # Skip if no payload data
        if self.payload is None:
            return None
            
        # Extract function call arguments and tool info
        func_call_args = self.payload.get("function_call")
        if func_call_args is not None and "tool" in self.payload:
            # Get tool metadata
            tool = self.payload.get("tool")
            if tool is None:
                return None
                
            # Format tool execution message
            return {
                "type": "events",
                "data": {
                    "title": f"Calling tool: {tool.name} with inputs: {func_call_args}",
                },
            }
        return None

    def _is_output_serializable(self, output: Any) -> bool:
        """
        Check if tool output can be JSON serialized.
        Used to validate responses before sending to frontend.
        """
        try:
            # Attempt JSON serialization
            json.dumps(output)
            return True
        except TypeError:
            # Return False if output can't be serialized
            return False

    def get_agent_tool_response(self) -> dict | None:
        """Process tool execution response for frontend display"""
        # Skip if no payload
        if self.payload is None:
            return None
        # Extract response data
        response = self.payload.get("response")
        if response is not None:
            sources = response.sources
            for source in sources:
                # Return the tool response here to include the toolCall information
                if isinstance(source, ToolOutput):
                    if self._is_output_serializable(source.raw_output):
                        output = source.raw_output
                    else:
                        output = source.content

                    return {
                        "type": "tools",
                        "data": {
                            "toolOutput": {
                                "output": output,
                                "isError": source.is_error,
                            },
                            "toolCall": {
                                "id": None,  # There is no tool id in the ToolOutput
                                "name": source.tool_name,
                                "input": source.raw_input,
                            },
                        },
                    }
        return None

    def to_response(self):
        try:
            match self.event_type:
                case "retrieve":
                    return self.get_retrieval_message()
                case "function_call":
                    return self.get_tool_message()
                case "agent_step":
                    return self.get_agent_tool_response()
                case _:
                    return None
        except Exception as e:
            logger.error(f"Error in converting event to response: {e}")
            return None


class EventCallbackHandler(BaseCallbackHandler):
    """
    Manages LLM operation events and streams them to frontend.
    Uses async queue for event buffering and streaming.
    """
    
    # Async queue for buffering events
    _aqueue: asyncio.Queue
    # Flag to indicate when event stream is complete
    is_done: bool = False

    def __init__(self):
        """
        Initialize event handler with ignored event types.
        Sets up async queue for event streaming.
        """
        # Skip processing for these internal events
        ignored_events = [
            CBEventType.CHUNKING,      # Document chunking
            CBEventType.NODE_PARSING,  # Text parsing
            CBEventType.EMBEDDING,     # Vector embedding
            CBEventType.LLM,          # Raw LLM calls
            CBEventType.TEMPLATING,    # Template processing
        ]
        super().__init__(ignored_events, ignored_events)
        self._aqueue = asyncio.Queue()

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Process start of LLM operation.
        Queues event if it has frontend-relevant data.
        """
        event = CallbackEvent(event_id=event_id, event_type=event_type, payload=payload)
        if event.to_response() is not None:
            self._aqueue.put_nowait(event)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Process end of LLM operation.
        Queues completion event if relevant.
        """
        event = CallbackEvent(event_id=event_id, event_type=event_type, payload=payload)
        if event.to_response() is not None:
            self._aqueue.put_nowait(event)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Trace start marker - currently unused."""
        # TODO: We need to be able to track multi-step LLM operations.
        # Things like monitoring the document processing pipeling or tool execution.
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Trace end marker - currently unused."""
        pass

    async def async_event_gen(self) -> AsyncGenerator[CallbackEvent, None]:
        """
        Generate stream of events for frontend.
        
        Flow:
        1. Check queue while events pending
        2. Yield events with 0.1s timeout
        3. Continue until queue empty and done
        """
        while not self._aqueue.empty() or not self.is_done:
            try:
                yield await asyncio.wait_for(self._aqueue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
