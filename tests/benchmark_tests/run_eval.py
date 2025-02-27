import os
import aiohttp
import asyncio
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.integrations import trace_llama_index
from deepeval.auto_evaluate import auto_evaluate

from common.eval_llm import EvalLLM_4Bit

"""
LLM Response Evaluation Script

This script is intended to act as an example starting point for people to start writing their benchmarks.
Hopefully this helps connect using our app with deepeval!

Evaluates LLM responses against test cases and saves metrics to CSV.
Uses deepeval for metrics calculation and aiohttp for async API calls.
Results are stored in timestamped CSV files under /results directory.
"""

# Initialize tracing
trace_llama_index(auto_eval=True)

# Test cases define expected inputs/outputs for evaluation
TEST_CASES = [
    {
        "input": "What standards for letters exist?",
        "expected_output": "There are several ISO standards...",
        "context": "ISO standards define various letter formats..."
    }
]

async def get_chat_response(input_text: str):
    """Makes async HTTP request to chat endpoint
    
    Args:
        input_text: Query string to send to chat API
        
    Returns:
        JSON response from API containing chat result
    """
    url = 'http://localhost:8000/api/chat/request'
    headers = {'Content-Type': 'application/json'}
    data = {
        "messages": [
            {
                "role": "user",
                "content": input_text
            }
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            return await response.json() 

async def evaluate_response(eval_llm):
    """Evaluates chat responses using deepeval metrics
    
    Args:
        eval_llm: Language model for evaluation
        
    Prints:
        Metric scores and reasoning for each test case
    """
    # Initialize relevancy metric with evaluation model
    answer_relevancy_metric = AnswerRelevancyMetric(model=eval_llm)

    for test in TEST_CASES:
        # Get response from chat API
        response = await get_chat_response(test["input"])
        
        if response is not None:
            # Extract response content and source nodes
            actual_output = response['result']['content']
            
            # Get source nodes if available
            if len(response['nodes']) > 0:
                nodes_used = [node['text'] for node in response.get("nodes", [])]
            else:
                nodes_used = None
        else:
            actual_output = ""
            nodes_used = None
        
        # Create test case with actual output and context
        eval_case = LLMTestCase(
            input=test["input"],
            actual_output=actual_output,
            retrieval_context=nodes_used
        )
        
        # It is possible to add more metrics, and also run them at the same time
        answer_relevancy_metric.measure(eval_case)
        print(f"Score: {answer_relevancy_metric.score}")
        print(f"Reason: {answer_relevancy_metric.reason}")

if __name__ == "__main__":
    eval_llm = EvalLLM_4Bit()
    asyncio.run(evaluate_response(eval_llm))
