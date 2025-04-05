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
    event_handlers: Optional[List[Any]] = None
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

    # Create multi-agent runner that supports both workflow handoffs
    return MultiAgentRunner.create_default_workflow(
        callback_manager=callback_manager,
        verbose=True
    )

