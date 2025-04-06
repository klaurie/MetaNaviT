"""
Tool Tracing Demo for Addition

This script demonstrates how tool tracing works in practice by:
1. Running a simple addition operation
2. Capturing tool calls and their inputs/outputs
3. Displaying the traced information

Usage:
    python test.py
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.settings import Settings
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool
import os

from app.engine.tools import create_tool_callback, tool_call_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define simple system prompt
SYSTEM_PROMPT = """You are a helpful assistant that can perform calculations.
Please use the addition tool when asked to add numbers together. The tool is named add_numbers. and your first arg is 'a' which is the first number and the second arg is 'b' which is the second number. when you use that too just send back the result in the response
"""

# Simple calculator function
def add_numbers(a: float, b: float) -> Dict[str, Any]:
    """
    Add two numbers together and return the result.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        Dictionary containing the result and a formatted equation
    """
    logger.info(f"Performing addition: {a} + {b}")
    
    result = a + b
    
    # Return a structured result
    return {
        "result": result,
        "equation": f"{a} + {b} = {result}",
        "operation": "addition"
    }

def get_calculator_tools():
    """Create and return a simple addition calculator tool"""
    addition_tool = FunctionTool.from_defaults(
        fn=add_numbers,
        name="add_numbers",
        description="Add two numbers together and return the result",
        callback=create_tool_callback("add_numbers", "Add two numbers together and return the result")  # This enables tracing of tool calls
    )
    
    return [addition_tool]

async def run_demo():
    """Run the tool tracing demonstration with addition"""
    logger.info("Starting tool tracing demo with addition")
    
    # Configure LLM - use Ollama if available, otherwise use an environment variable
    try:
        from llama_index.llms.ollama import Ollama
        Settings.llm = Ollama(
            model="llama3.2:3b",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7,
        )
    except ImportError:
        logger.warning("Ollama not available, using default LLM from Settings")
    
    # Get calculator tools
    tools = get_calculator_tools()
    logger.info(f"Loaded {len(tools)} calculator tools")
    
    # Create an agent runner with the tools
    logger.info("Creating agent runner...")
    agent = AgentRunner.from_llm(
        llm=Settings.llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        verbose=True
    )
    
    # Simple conversation history
    chat_history = [
        ChatMessage(role=MessageRole.USER, content="I need help with some math."),
        ChatMessage(role=MessageRole.ASSISTANT, content="I'd be happy to help with math problems! What would you like to calculate?")
    ]
    
    # Send a message that will trigger the addition tool
    user_message = "What is 42 + 17?"
    logger.info(f"Sending user message: {user_message}")
    
    # Run the conversation and capture the response
    try:
        response = await agent.achat(user_message, chat_history)
        logger.info(f"Received response: {response.response}")
        
        # Display the tool calls that were traced
        logger.info(f"Number of tool calls traced: {len(tool_call_registry)}")
        
        for i, call in enumerate(tool_call_registry):
            logger.info(f"\n--- Tool Call {i+1} ---")
            logger.info(f"Tool: {call.get('name')}")
            logger.info(f"Description: {call.get('description')}")
            
            # Show input parameters if available
            if 'input_parameters' in call and call['input_parameters']:
                logger.info(f"Input parameters: {json.dumps(call['input_parameters'], indent=2)}")
            
            # Show reasoning if available
            if 'reasoning' in call and call['reasoning']:
                logger.info(f"Reasoning: {call['reasoning']}")
            
            # Show abbreviated output
            if 'output' in call:
                # For large outputs, show a summary
                output_str = str(call['output'])
                if len(output_str) > 500:
                    logger.info(f"Output: {output_str[:500]}... (truncated)")
                else:
                    logger.info(f"Output: {output_str}")
        
        # Save the traced calls to a file for inspection
        output_file = "tool_calls_traced.json"
        with open(output_file, "w") as f:
            json.dump(tool_call_registry, f, indent=2, default=str)
        
        logger.info(f"Tool calls saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_demo())