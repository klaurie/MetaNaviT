"""
File System Organization Test Module

Evaluates LLM's ability to organize file systems by:
1. Loading test cases from JSON
2. Running test cases through chat API
3. Evaluating responses using metrics
4. Tracking tool usage and commands
"""

import os
import json
import aiohttp
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval, TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams
from deepeval.integrations import trace_llama_index
from deepeval.auto_evaluate import auto_evaluate

from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
from tests.benchmark_tests.common.utils import load_test_cases, convert_registry_to_tool_call, convert_test_case_tool_calls
from tests.benchmark_tests.run_eval import get_chat_response

@dataclass
class TestContext:
    """Test execution context and configuration"""
    eval_dir: str
    expected_dir_format: str
    expected_linux_commands: List[str]

class FileSystemTestRunner:
    """Handles file system organization test execution"""
    
    def __init__(self, test_cases_path: Path):
        self.test_cases_path = test_cases_path
        self.eval_llm = EvalLLM_4Bit()
        trace_llama_index(auto_eval=True)
        
    def load_test_cases(self) -> List[LLMTestCase]:
        """Load test cases from JSON file and return raw data"""
        with open(self.test_cases_path, 'r') as f:
            data = json.load(f)
            return data["test_cases"]

    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""

        structure_metric = GEval(
            name="Structure Understanding",
            evaluation_steps=[
                "Does the output use the actual user's files from? If it's just an example of potential organization styles that is not a good output."
                "Is the file system generated output structured in a readable format in 'actual output'",
                "Does the folder hierarchy look like it has excessive nesting?",
                "Are there consistent naming conventions for files and folders?",
                "Are duplicate or redundant files minimized?",
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.eval_llm,
            verbose_mode=True
        )

        search_retrieval_metric = GEval(
            name="Search & Retrieval Efficiency",
            evaluation_steps=[
                "How quickly can a user find a specific file based on its name or metadata?",
                "Are files tagged with metadata or keywords to improve searchability?",
                "Is there a versioning system to track file changes?"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.eval_llm,
            verbose_mode=True
        )

        return [
            structure_metric,
            search_retrieval_metric,
            TaskCompletionMetric(threshold=0.7, model=self.eval_llm),
            ToolCorrectnessMetric(verbose_mode=True, should_consider_ordering=True)
        ]

    async def evaluate_response(self, context: TestContext) -> None:
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
            else:
                # Handle failed responses
                actual_output = ""

            
            eval_case = LLMTestCase(
                input=test["input"],
                actual_output=actual_output,
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
    test_context = TestContext(
        eval_dir="",
        expected_dir_format="",
        expected_linux_commands=[""]
    )
    
    runner = FileSystemTestRunner(
        Path(__file__).parent / "test_cases.json"
    )
    await runner.evaluate_response(test_context)

if __name__ == "__main__":
    asyncio.run(main())