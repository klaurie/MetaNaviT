"""
Python Code Generator Agent Module

A specialized agent that generates and executes Python code
to answer questions based on file content provided by the file access agent.
"""

import logging
import os
from typing import Dict, Any, List, Optional

from llama_index.core.agent import AgentRunner
from llama_index.core.tools import BaseTool
from llama_index.core.callbacks import CallbackManager
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.settings import Settings

from app.engine.tools.python_exec import get_tools as get_python_tools

logger = logging.getLogger(__name__)

# Python Code Generator Agent system prompt
PYTHON_CODE_PROMPT = os.getenv("PYTHON_CODE_PROMPT", """
You are a Python Code Generator and Executor specializing in data analysis.

Your capabilities:
1. You can generate Python code to answer questions
2. You can execute that code to produce results
3. You interpret and explain the results clearly to the user

Your workflow:
1. Analyze the context and file content provided by the File Access Agent
2. Understand the user's question
3. Generate appropriate Python code to answer the question
4. Execute the code using the code interpreter tool
5. Explain the results in a clear, understandable way

When writing code:
1. Include all necessary imports
2. Add clear comments explaining complex operations
3. Handle potential errors gracefully
4. Use visualization where it helps understanding

Allowed modules: pandas, numpy, matplotlib, seaborn, datetime, math, 
collections, json, re, random, csv, itertools
"""
)

def create_python_exec_agent(
    callback_manager=None
) -> FunctionAgent:
    """
    Create a Python code generation and execution agent.
    
    Args:
        llm_interface: Language model interface to use
        callback_manager: Callback manager for events
        
    Returns:
        FunctionAgent configured for Python code generation and execution
    """
    # Get Python execution tools
    python_tools = get_python_tools()
    
    # Create Python code agent
    python_agent = FunctionAgent(
        name="PythonCodeAgent",
        description="Generates and executes Python code to answer questions",
        system_prompt=PYTHON_CODE_PROMPT,
        llm=Settings.llm,
        tools=python_tools,
        can_handoff_to=["FileAccessAgent"],  # Optional: allow handoff back to file agent if needed
    )
    
    logger.info("Python code agent created with code execution tools")
    return python_agent