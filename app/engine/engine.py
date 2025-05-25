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
from typing import List, Optional, Dict, Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.tools import BaseTool

from app.engine.agents.file_access_agent import create_file_access_agent
from app.engine.agents.python_exec_agent import create_python_exec_agent
from app.engine.agents.multi_agent_workflow import AgentWorkflow

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.settings import Settings

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

    # Check debug mode for callback behavior
    debug_mode = (params or {}).get("debug", False) or os.getenv("DEBUG_MODE", "false").lower() == "true"
    verbose = os.getenv("VERBOSE_AGENT", "false").lower() == "true"

    # Initialize callback manager
    # callback_manager = CallbackManager(handlers=event_handlers or [])
    callback_manager = CallbackManager(handlers=event_handlers if debug_mode else [])
    
    agents = []

    if not params or params.get("use_file_access", True):
        agents.append(
            create_file_access_agent(
                callback_manager=callback_manager
            )
        )
    
    if params and params.get('use_python_exec', False):
        agents.append(
            create_python_exec_agent(
                callback_manager=callback_manager
            )
        )
    
    if not agents:
        raise ValueError("No agents created. Ensure at least one agent is configured.")

    return AgentWorkflow(
        agents=agents,
        root_agent=agents[0].name,
        verbose=verbose)