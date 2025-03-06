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
        python_agent = create_python_code_agent(callback_manager=callback_manager)
        
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
        Run an async non-streaming chat conversation.
        
        Args:
            message: User message
            chat_history: Optional conversation history
            
        Returns:
            Agent response
        """
        # Create context for this run
        ctx = Context(self.workflow)
        
        # Run workflow
        handler = self.workflow.run(
            user_msg=message,
            chat_history=chat_history,
            ctx=ctx
        )
        
        # Await final response
        response = await handler
        return response
    
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Any:
        """
        Run a streaming chat conversation.
        
        Args:
            message: User message
            chat_history: Optional conversation history
            
        Returns:
            Streaming agent response
        """
        # Use async function with sync wrapper
        async def run_stream():
            async for response in self.astream_chat(message, chat_history):
                yield response
                
        return run_stream()
    
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]] | List[ChatMessage]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run an async streaming chat conversation compatible with AgentRunner.
        
        Args:
            message: User message
            chat_history: Optional conversation history
            
        Yields:
            Streaming chunks of agent response
        """
        # Reset source nodes
        self.source_nodes = []
        
        # Convert chat history if it's in dict format
        if chat_history and isinstance(chat_history[0], dict):
            converted_history = []
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                converted_history.append(ChatMessage(role=role, content=content))
            chat_history = converted_history
        
        # Create context for this run
        ctx = Context(self.workflow)
        
        # Run workflow and stream events
        handler = self.workflow.run(
            user_msg=message,
            chat_history=chat_history,
            ctx=ctx
        )
        
        # Stream events like AgentRunner's streaming
        current_response = ""
        
        # Track which agent is currently active
        self.current_agent_name = None
        
        # Create response object compatible with AgentRunner
        async for event in handler.stream_events():
            # Handle agent switch
            if hasattr(event, "current_agent_name") and event.current_agent_name:
                self.current_agent_name = event.current_agent_name
                if self.verbose:
                    logger.info(f"Agent switched to: {self.current_agent_name}")
            
            # Handle streaming text generation
            if isinstance(event, AgentStream):
                if event.delta:
                    current_response += event.delta
                    yield {
                        "delta": event.delta,
                        "response": current_response,
                        "sources": self.source_nodes
                    }
                    
            # Handle tool call results
            elif isinstance(event, ToolCallResult):
                # Check for source nodes in tool output
                if hasattr(event.tool_output, "source_nodes") and event.tool_output.source_nodes:
                    self.source_nodes.extend(event.tool_output.source_nodes)
            
            # Handle agent output containing full response
            elif isinstance(event, AgentOutput):
                if event.response and event.response.content:
                    current_response = event.response.content
                    yield {
                        "delta": current_response,
                        "response": current_response,
                        "sources": self.source_nodes
                    }
        
        # Final response with complete output
        if current_response:
            yield {
                "delta": "",  # Empty delta for final chunk
                "response": current_response,
                "sources": self.source_nodes,
                "is_completion": True
            }
    
    def reset(self) -> None:
        """Reset the runner state."""
        self.memory.reset()
        self.source_nodes = []