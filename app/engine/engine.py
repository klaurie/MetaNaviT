"""
Chat Engine Factory Module

Provides centralized configuration and initialization of chat engines
with support for:
- Tool registration and management
- Index-based query capabilities
- Event handling and callbacks
- Environment-based configuration
"""

import os
from typing import List, Optional, Dict, Any

from llama_index.core.agent import AgentRunner
from llama_index.core.callbacks import CallbackManager
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool

from app.engine.index import IndexConfig, get_index
from app.engine.tools import ToolFactory
from app.engine.tools.query_engine import get_query_engine_tool


def get_chat_engine(
    params: Optional[Dict[str, Any]] = None,
    event_handlers: Optional[List[Any]] = None,
    **kwargs
) -> AgentRunner:
    """
    Creates and configures a chat engine with tools and settings.
    
    Args:
        params: Index configuration parameters
        event_handlers: Callback handlers for events
        **kwargs: Additional configuration options
        
    Returns:
        Configured AgentRunner instance
    """
    # Get system prompt from environment
    system_prompt = os.getenv("SYSTEM_PROMPT", "I am an AI assistant.")
    tools: List[BaseTool] = []
    
    # Initialize callback manager
    callback_manager = CallbackManager(handlers=event_handlers or [])

    # Configure and add query tool if index exists
    index_config = IndexConfig(
        callback_manager=callback_manager,
        **(params or {})
    )
    index = get_index(index_config)
    if index is not None:
        query_engine_tool = get_query_engine_tool(
            index=index,
            **kwargs
        )
        tools.append(query_engine_tool)

    # Add tools configured in environment
    configured_tools: List[BaseTool] = ToolFactory.from_env()
    tools.extend(configured_tools)

    # Create and return configured agent
    return AgentRunner.from_llm(
        llm=Settings.llm,
        tools=tools,
        system_prompt=system_prompt,
        callback_manager=callback_manager,
        verbose=True,
        max_iterations=10,
    )
