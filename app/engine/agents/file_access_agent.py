"""
File Access Agent Module

A specialized agent that provides access to files from the document index
and can perform all operations except code execution.
"""

import os  # Add the missing import
import logging
from typing import List, Optional, Dict, Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import BaseTool
from llama_index.core.settings import Settings

from app.engine.tools import ToolFactory
from app.engine.tools.query_engine import get_query_engine_tool
from app.engine.index import IndexConfig, get_index

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
        callback_manager=None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
) -> FunctionAgent:
    """
    Create a file access agent with all tools except Python execution.
    
    Args:
        callback_manager: Callback manager for events
        params: Optional parameters for index configuration
        **kwargs: Additional keyword arguments
        
    Returns:
        FunctionAgent configured with all tools except Python execution
    """
    # Get all tools from the tool factory - just once
    all_tools = ToolFactory.from_env()
    
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
        all_tools.append(query_engine_tool)
    
    # Filter out python_exec tool and handle duplicates
    combined_tools = []
    seen_tool_names = set()
    
    for tool in all_tools:
        tool_name = tool.metadata.get("name")
        if tool_name != "python_exec" and tool_name not in seen_tool_names:
            combined_tools.append(tool)
            seen_tool_names.add(tool_name)
    
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