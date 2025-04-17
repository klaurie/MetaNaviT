"""
Basic Chat Agent Module

A general purpose conversational agent that handles standard chat interactions
and can hand off to specialized agents when needed.
"""

import os
import logging
from typing import Optional, Dict, Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings

from app.engine.tools import ToolFactory

logger = logging.getLogger(__name__)

# Basic chat agent prompt
BASIC_CHAT_PROMPT = os.getenv("BASIC_CHAT_PROMPT", """
You are a helpful assistant specializing in general conversations.

Your capabilities:
1. You can answer general knowledge questions
2. You can engage in casual conversation
3. You can provide explanations on various topics
4. You can recognize when specialized help is needed                      

Your workflow:
1. First understand what the user is asking for
2. If it's a general question or conversation, respond helpfully
3. If the user needs specialized assistance:
   - For file access or document analysis, hand off to the FileAccessAgent
   - For code or programming tasks, hand off to the PythonCodeAgent

Be friendly, concise, and helpful in your responses.
""")


def create_basic_chat_agent(
        callback_manager=None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
) -> FunctionAgent:
    """
    Create a basic conversational agent with simple tools.
    
    Args:
        callback_manager: Callback manager for events
        params: Optional parameters for configuration
        **kwargs: Additional keyword arguments
        
    Returns:
        FunctionAgent configured for general conversation
    """
    # Get basic conversational tools
    tool_factory = ToolFactory.from_env()
    
    # Filter for only general conversation tools
    # You can customize this list based on your needs
    general_tools = [
        tool for tool in tool_factory 
        if tool.metadata.name in ["web_search", "calculator", "current_date"]
    ]
    
    # Create basic chat agent
    chat_agent = FunctionAgent(
        name="ChatAgent",
        description="Handles general conversation and basic information retrieval",
        system_prompt=BASIC_CHAT_PROMPT,
        llm=Settings.llm,
        tools=general_tools,
        can_handoff_to=["FileAccessAgent", "PythonCodeAgent"],
    )
    
    logger.info(f"Basic chat agent created with {len(general_tools)} tools")
    return chat_agent