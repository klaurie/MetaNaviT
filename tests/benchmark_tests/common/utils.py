    
"""
Benchmark Test Utilities Module

This module provides common utility functions for the benchmark testing framework,
particularly focused on test case setup, tool call handling, and evaluation support.

The utilities here support various benchmark test categories including:
- Data transformation tests
- Organization tests
- Reasoning tests
- Tool use evaluation
"""
import logging
import json
from typing import List, Dict, Any
from deepeval.test_case import ToolCall
from app.engine.tools import get_tool_call_registry, clear_tool_calls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def convert_registry_to_tool_call():
    """
    Convert a tool call dictionary from the registry into a DeepEval ToolCall object.
    
    Args:
        tool_data: Dictionary containing tool call information from the registry
            Expects keys like "name", "description", "output", and "input_parameters"
            
    Returns:
        ToolCall object formatted for DeepEval testing
    """

    tool_registry = get_tool_call_registry()
    tool_calls = []

    for tool_data in tool_registry:
        tool_calls.append(ToolCall(
        name=tool_data.get("name", "unknown_tool"),
        description=tool_data.get("description"),
        output=tool_data.get("output"),
        ))
    
    logger.info(f"called: {tool_calls}")
    # clear registry for future queries
    clear_tool_calls()

    # If there were no tool calls, append an empty one, so we can still run the test
    if len(tool_calls) < 1:
        tool_calls.append(ToolCall(
            name="none",
            description="no tool was used",
            output="",
            ))

    return tool_calls 
    



def convert_test_case_tool_calls(tool_params: List[Dict]) -> List[ToolCall]:
    """
    Create tool calls for test case
    
    These tools measure these key aspects:
    1. Data Extraction: Can the LLM correctly extract structured data from unstructured text
    2. Schema Understanding: Can the LLM correctly apply the provided schema
    3. Format Transformation: Can the LLM transform the data into the required format (CSV)
    """
    tool_calls = []
    
    for tool_param in tool_params:
        tool_calls.append(
            ToolCall(
                name=tool_param.get("name", ""),
                description=tool_param.get("description", ""),
                output=tool_param.get("output", [])
            )
        )
        
    return tool_calls

def load_test_cases(test_cases_path: str) -> List[Dict]:
    """Load test cases from JSON file and return raw data"""
    with open(test_cases_path, 'r') as f:
        data = json.load(f)
        return data["test_cases"]