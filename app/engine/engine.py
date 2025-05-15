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

from app.engine.llm import get_llm, get_embedding_model
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import ContextChatEngine

logger = logging.getLogger(__name__)

class EngineFactory:
    """Factory class for creating various engines."""
    
    def __init__(self):
        """Initialize the engine factory with default settings."""
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
    
    def create_chat_engine(self, retriever=None, filters=None, params=None):
        """Create a chat engine with optional retriever for context."""
        if retriever is None:
            logger.warning("Creating chat engine without retriever - no sources will be used")
            # Simple QA engine without retrieval - use a different approach
            from llama_index.core.chat_engine import SimpleChatEngine
            return SimpleChatEngine.from_defaults(llm=self.llm)
        
        # Create a query engine using the retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self.llm
        )
        
        # Create a chat engine with the query engine for context
        chat_engine = ContextChatEngine.from_defaults(
            llm=self.llm,
            retriever=retriever,
            query_engine=query_engine,
            verbose=True,
        )
        
        return chat_engine

def get_engine_factory():
    """Get a configured engine factory instance."""
    return EngineFactory()

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
    
    agents = []

    agents.append(
        create_file_access_agent(
            callback_manager=callback_manager
        )
    )
    agents.append(
        create_python_exec_agent(
            callback_manager=callback_manager
        )
    )

    return AgentWorkflow(
        agents=agents,
        root_agent=agents[0].name,
        verbose=True)