import os
import json
import aiohttp
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval, TaskCompletionMetric, ToolCorrectnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams

from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
from tests.benchmark_tests.common.utils import load_test_cases, convert_registry_to_tool_call, convert_test_case_tool_calls
from tests.benchmark_tests.run_eval import get_chat_response

class SearchTestRunner:
    """Handles search test execution"""
    
    def __init__(self, test_cases_path: Path):
        self.test_cases_path = test_cases_path
        self.eval_llm = EvalLLM_4Bit()
        
    def load_test_cases(self) -> List[LLMTestCase]:
        """Load test cases from JSON file and return raw data"""
        with open(self.test_cases_path, 'r') as f:
            data = json.load(f)
            return data["test_cases"]

    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""


        return [
            AnswerRelevancyMetric(model=self.eval_llm, include_reason=True),
            ContextualRelevancyMetric(model=self.eval_llm, include_reason=True),
            TaskCompletionMetric(threshold=0.7, model=self.eval_llm),
            ToolCorrectnessMetric(verbose_mode=True, should_consider_ordering=True)
        ]

    async def evaluate_response(self) -> None:
        """
        Run evaluation for all file system organization test cases.

        This method:
        1. Initializes evaluation metrics (relevancy, task completion)
        2. Loads test cases from JSON configuration
        3. Makes async requests to chat API for each test
        4. Processes responses and extracts relevant information
        5. Creates evaluation cases with context and tool calls
        6. Measures performance using configured metrics
        
        Args:
            context (TestContext): Test execution context containing:
                - eval_dir: Directory to evaluate
                - expected_dir_format: Expected directory structure
                - expected_linux_commands: Expected shell commands
        
        Flow:
            1. Initialize metrics → Load cases → For each case:
                a. Get LLM response
                b. Extract content and context
                c. Create evaluation case
                d. Apply metrics and log results
        """
        # Initialize metrics and load test cases
        metrics = self.init_metrics()
        test_cases = load_test_cases(self.test_cases_path)
        
        # Process each test case
        for test in test_cases:
            # Get LLM response through chat API
            response = await get_chat_response(test["input"])
            
            # Extract response content and relevant nodes
            if response is not None:
                actual_output = response['result']['content']

                # Get source nodes if available
                if len(response['nodes']) > 0:
                    nodes_used = [node['text'] for node in response.get("nodes", [])]
                else:
                    nodes_used = None
            else:
                # Handle failed responses
                actual_output = ""
                nodes_used = None

            
            eval_case = LLMTestCase(
                input=test["input"],
                actual_output=actual_output,
                retrieval_context=nodes_used,
                tools_called = convert_registry_to_tool_call(),
                expected_tools=convert_test_case_tool_calls(test["tool_params"])
            )
            
            # Apply each metric and log results
            for metric in metrics:
                metric.measure(eval_case)
                # Log metric results for analysis
                print(f"{metric.__class__.__name__} Score: {metric.score}")
                print(f"Reason: {metric.reason}")

async def main():
    """Main entry point for test execution"""    
    runner = SearchTestRunner(
        Path(__file__).parent / "test_cases.json"
    )
    await runner.evaluate_response()

if __name__ == "__main__":
    asyncio.run(main())