"""
File Access Agent Module

A specialized agent that provides access to files from the document index
and can perform all operations except code execution.
"""

import os
import logging

from llama_index.core.agent.workflow import FunctionAgent

from app.config.settings import Settings
from app.engine.tools.tool_manager import ToolFactory
from app.engine.tools.index_file_access import get_tools as get_file_tools

logger = logging.getLogger(__name__)

# Updated prompt to reflect expanded capabilities
FILE_ACCESS_PROMPT = os.getenv("FILE_ACCESS_PROMPT", """
You are a File Access and Analysis Assistant that can access the document index and use various tools.

Your capabilities:
1. You can list and retrieve files from the document index
3. You can analyze and provide insights about files and data
4. You can provide file contents and analysis as context to the Python Code Agent

Your workflow:
1. First understand what the user is asking for
2. If they need file access, retrieve relevant files from the index
3. Use your other tools to process, analyze or transform the data as needed
4. If the user needs code execution or complex data analysis that requires programming:
   - Provide the necessary context and data
   - Hand off to the Python Code Agent for code generation and execution

Remember that you CANNOT execute Python code directly - this is reserved for the Python Code Agent.
"""
)


def create_file_access_agent(
    callback_manager=None
) -> FunctionAgent:
    """
    Create a file access agent with all tools except Python execution.
    
    Args:
        llm_interface: Language model interface to use
        callback_manager: Callback manager for events
        
    Returns:
        FunctionAgent configured with all tools except Python execution
    """
    # Get all tools from the tool factory
    all_tools = ToolFactory.from_env()
    
    # Get file access tools
    file_tools = get_file_tools()
    
    # Combine tools, filtering out python_exec tool
    combined_tools = []
    combined_tools.extend(file_tools)
    
    # Add all other tools except python_exec
    for tool in all_tools:
        if tool.metadata.get("name") != "python_exec" and not any(t.metadata.get("name") == tool.metadata.get("name") for t in combined_tools):
            combined_tools.append(tool)
    
    # Create file access agent with all non-python-exec tools
    file_agent = FunctionAgent(
        name="FileAccessAgent",
        description="Retrieves file contents and uses all tools except code execution",
        system_prompt=FILE_ACCESS_PROMPT,
        llm=Settings.llm,
        tools=combined_tools,
        can_handoff_to=["PythonCodeAgent"],
    )
    
    logger.info(f"File access agent created with {len(combined_tools)} tools (excluding python_exec)")
    return file_agent