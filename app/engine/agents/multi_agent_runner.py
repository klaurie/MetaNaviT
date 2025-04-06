"""
Multi-Agent Runner Module

A hybrid implementation that combines AgentRunner's streaming API with 
AgentWorkflow's ability to hand off between multiple specialized agents.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
import json

from llama_index.core.callbacks import CallbackManager
from llama_index.core.agent import AgentRunner
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from app.engine.agents.file_access_agent import create_file_access_agent, FILE_ACCESS_PROMPT
from app.engine.agents.python_exec_agent import create_python_exec_agent, PYTHON_CODE_PROMPT

logger = logging.getLogger(__name__)

class MultiAgentRunner:
    """
    Hybrid agent runner that supports both chat streaming and agent handoffs.
    
    Provides the same interface as AgentRunner (chat, stream_chat, etc.) but 
    with the ability to hand off between specialized agents like AgentWorkflow.
    """
    
    def __init__(
        self,
        workflow: AgentWorkflow,
        callback_manager: Optional[CallbackManager] = None,
        memory: Optional[ChatMemoryBuffer] = None,
        verbose: bool = False
    ):
        """
        Initialize multi-agent runner with a workflow and callbacks.
        
        Args:
            workflow: The underlying AgentWorkflow for coordinating agents
            callback_manager: Optional callback manager for events
            memory: Optional chat memory buffer
            verbose: Whether to print verbose logs
        """
        self.workflow = workflow
        self.callback_manager = callback_manager or CallbackManager()
        self.memory = memory or ChatMemoryBuffer.from_defaults()
        self.verbose = verbose
        
        # Session state
        self.current_agent_name = None
        self.source_nodes = []
    
    @classmethod
    def from_agents(
        cls,
        agents_dict: Dict[str, Any],
        root_agent_name: str = None,
        callback_manager: Optional[CallbackManager] = None,
        memory: Optional[ChatMemoryBuffer] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> "MultiAgentRunner":
        """
        Create a MultiAgentRunner from a dictionary of agents.

        Note: I'm including this as an option because in the future we could initialize the workflow from functions
        
        Args:
            agents_dict: Dictionary mapping agent names to agent objects
            root_agent_name: Name of the starting agent (defaults to first agent if None)
            callback_manager: Optional callback manager for events
            memory: Optional chat memory buffer
            initial_state: Optional initial state dictionary
            verbose: Whether to print verbose logs
            
        Returns:
            Configured MultiAgentRunner instance
        """
        # Create workflow from agents
        workflow = AgentWorkflow(
            agents=list(agents_dict.values()),
            root_agent=root_agent_name or list(agents_dict.keys())[0],
            initial_state=initial_state or {},
            verbose=verbose
        )
        
        return cls(
            workflow=workflow,
            callback_manager=callback_manager,
            memory=memory,
            verbose=verbose
        )
    
    @classmethod
    def create_default_workflow(
        cls,
        callback_manager: Optional[CallbackManager] = None,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False
    ) -> "MultiAgentRunner":
        """
        Create a default multi-agent workflow with file access and Python execution agents.
        
        Args:
            callback_manager: Optional callback manager for events
            tools: Optional list of additional tools to include
            llm: Optional LLM to use (defaults to Settings.llm)
            verbose: Whether to print verbose logs
            
        Returns:
            MultiAgentRunner with default file and Python agents
        """
        file_agent = create_file_access_agent(callback_manager=callback_manager)
        python_agent = create_python_exec_agent(callback_manager=callback_manager)
        
        agents = {
            "FileAccessAgent": file_agent,
            "PythonCodeAgent": python_agent
        }
        
        initial_state = {
            "file_content": None,
            "code_executed": False,
            "execution_result": None
        }
        
        return cls.from_agents(
            agents_dict=agents,
            root_agent_name="FileAccessAgent",
            callback_manager=callback_manager,
            initial_state=initial_state,
            verbose=verbose
        )
    
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Any:
        """
        Run a non-streaming chat conversation.
        
        Args:
            message: User message
            chat_history: Optional conversation history
            
        Returns:
            Agent response
        """
        # Use async function with sync wrapper
        return asyncio.run(self.achat(message, chat_history))
    
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Any:
        """
        Run a non-streaming chat conversation asynchronously.
        
        Args:
            message: User message
            chat_history: Optional conversation history
        """
        pass

    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[StreamingAgentChatResponse, None]:
        """
        Stream chat responses.
        
        Args:
            message: User message
            chat_history: Optional conversation history
            
        Returns:
            Asynchronous generator of streaming responses
        """
        # Use async function with sync wrapper
        return asyncio.run(self.astream_chat(message, chat_history))
    
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat responses asynchronously as plain text.
        
        Args:
            message: User message
            chat_history: Optional conversation history
            
        Returns:
            Asynchronous generator of plain text chunks
        """
        # Start the workflow
        handler = self.workflow.run(
            user_msg=message,
            chat_history=chat_history or []
        )

        current_agent = None
        handoff_detected = False

        async for event in handler.stream_events():
            # Log events for debugging
            logger.debug(f"Event type: {type(event).__name__}")
            
            # Agent switch detection
            if (
                hasattr(event, "current_agent_name") 
                and event.current_agent_name != current_agent
            ):
                current_agent = event.current_agent_name
                logger.info(f"Switching to agent: {current_agent}")
                handoff_detected = True
                
                # Optionally yield agent switch information to frontend
                # yield f"\n[Agent switch to {current_agent}]\n"
                
            # Handle text streaming
            elif isinstance(event, AgentStream) and event.delta:
                logger.debug(f"Streaming delta: {event.delta}")
                yield event.delta
                
            # Handle agent outputs
            elif isinstance(event, AgentOutput) and event.response and hasattr(event.response, 'content'):
                logger.info(f"Agent output: {event.response.content}")
                # Only yield content if not already streamed by AgentStream
                if not handoff_detected and event.response.content:
                    yield event.response.content
                
            # Handle tool calls (no direct yield to frontend)
            elif isinstance(event, ToolCall):
                logger.info(f"Tool call: {event.tool_name}")
                
            # Handle tool results (no direct yield to frontend)
            elif isinstance(event, ToolCallResult):
                logger.info(f"Tool result: {event.tool_name}")
                
        # Get final response if nothing substantial was yielded
        # This is often needed after handoffs
        final_response = await handler
        if final_response and hasattr(final_response, 'content') and final_response.content:
            yield final_response.content
