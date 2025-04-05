    
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
import json
from typing import List, Dict
from deepeval.test_case import ToolCall, LLMTestCase

def get_tool_calls(self, tool_params: List[Dict]) -> List[ToolCall]:
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
                input_parameters=tool_param.get("input_parameters", {}),
                output=tool_param.get("output", [])
            )
        )
        
    return tool_calls

def load_test_cases(self) -> List[LLMTestCase]:
    """Load test cases from JSON file and return raw data"""
    with open(self.test_cases_path, 'r') as f:
        data = json.load(f)
        return data["test_cases"]