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
import logging
from typing import List, Optional, Dict, Any, Union

from llama_index.core.agent import AgentRunner
from llama_index.core.callbacks import CallbackManager
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool

from app.engine.index import IndexConfig, get_index
from app.engine.tools import ToolFactory
from app.engine.tools.query_engine import get_query_engine_tool
from app.engine.agents.multi_agent_runner import MultiAgentRunner

logger = logging.getLogger(__name__)

def get_chat_engine(
    filters: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    event_handlers: Optional[List[Any]] = None,
    use_multi_agent: bool = False,
    **kwargs
):
    """
    Create and configure a chat engine with tools and settings.
    
    Args:
        filters: Optional document filters
        params: Optional configuration parameters
        event_handlers: Callback handlers for events
        use_multi_agent: Whether to use multi-agent workflow instead of single agent
        **kwargs: Additional configuration options
        
    Returns:
        Either a dict containing configured agent(s) or an AgentWorkflow instance
    """
    # Initialize system prompt from environment or default
    system_prompt = os.getenv("SYSTEM_PROMPT", "I am an AI assistant.")
    tools: List[BaseTool] = []
    
    # Initialize callback manager
    callback_manager = CallbackManager(handlers=event_handlers or [])

    # If using multi-agent workflow
    if use_multi_agent:
        # Create multi-agent runner that supports both workflow handoffs
        # and the familiar AgentRunner interface
        multi_agent = MultiAgentRunner.create_default_workflow(
            callback_manager=callback_manager,
            verbose=True
        )
        return multi_agent
    
    # Otherwise, create the regular agent(s)
    # Configure and add query tool if index exists
    index_config = IndexConfig(
        callback_manager=callback_manager,
        **(params or {})
    )
    index = get_index(index_config)

    if index is not None:
        # logger.info(f"Adding query engine tool to chat engine with filters: {filters}")
        query_engine_tool = get_query_engine_tool(
            index=index,
            **kwargs
        )
        tools.append(query_engine_tool)

    # Add tools configured in environment
    configured_tools: List[BaseTool] = ToolFactory.from_env()
    tools.extend(configured_tools)
       
    return AgentRunner.from_llm(
        llm=Settings.llm,
        tools=tools,
        system_prompt=system_prompt,
        callback_manager=callback_manager,
        verbose=True,
        max_iterations=15,
    )
