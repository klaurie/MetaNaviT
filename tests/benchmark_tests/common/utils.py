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

def convert_registry_to_tool_call(registry):
    """
    Convert a tool call dictionary from the registry into a DeepEval ToolCall object.
    
    Args:
        tool_data: Dictionary containing tool call information from the registry
            Expects keys like "name", "description", "output", and "input_parameters"
            
    Returns:
        ToolCall object formatted for DeepEval testing
    """

    tool_calls = []
    
    for tool_data in registry:
        print(f"Processing tool_data: {tool_data}") # For debugging

        # Initialize with default values
        name = "unknown_tool"
        description = ""
        output = ""

        # Check if tool_data is a dictionary before accessing keys
        if isinstance(tool_data, dict):
            # Safely get 'tool_name'
            name_val = tool_data.get("tool_name")
            if name_val is not None:
                name = name_val
            else:
                print(f"Warning: 'tool_name' missing or None in tool_data: {tool_data}")

            # Safely get nested 'description'
            raw_input_val = tool_data.get("raw_input")
            if isinstance(raw_input_val, dict):
                description_val = raw_input_val.get("input")
                if description_val is not None:
                    description = description_val
                else:
                    print(f"Warning: 'input' key missing or None in 'raw_input' dict: {raw_input_val}")
            else:
                print(f"Warning: 'raw_input' is not a dictionary or missing in tool_data: {tool_data}")

            # Safely get 'output'
            output_val = tool_data.get("content")
            if output_val is not None:
                output = output_val
            else:
                print(f"Warning: 'content' missing or None in tool_data: {tool_data}")
        else:
            print(f"Warning: Expected a dictionary for tool_data, but got {type(tool_data)}: {tool_data}")

        # Append ToolCall with safely retrieved (or default) values, ensuring they are strings
        tool_calls.append(ToolCall(
            name=str(name),
            description=str(description),
            output=str(output),
        ))
    

    # clear registry for future queries
    clear_tool_calls()

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
    
def write_results_to_csv(results: Dict, filename: str="benchmark_results.csv"):
    """Write results to a CSV file"""
    import csv
    import os
    from pathlib import Path
    # Define a directory for results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True) # Ensure the directory exists

    full_path = results_dir / filename
    with open(full_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['test_case_input', 'metric_name', 'score', 'reason', 'app_response', 'tools_called', 'expected_tools']

        # Check if the file is empty to write the header
        if os.stat(full_path).st_size == 0:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow(results)
        logger.info(f"Results written to {full_path}")

