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
from deepeval.metrics import AnswerRelevancyMetric, GEval, TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams
from deepeval.integrations import trace_llama_index
from deepeval.auto_evaluate import auto_evaluate

from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
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

    def get_tool_calls(self, tool_params) -> List[ToolCall]:
        """
        Create tool calls for test case
        
        These tools measure two key aspects:
        1. Structure Understanding: Can the LLM correctly interpret and represent
        hierarchical file organization
        2. Command Generation: Can the LLM generate correct Linux commands to
        achieve the desired organization
        """
        return [
            ToolCall(
                name="Classify Files",
                description="Uses AI to classify files based on content, type, and metadata.",
                input_parameters={"file_paths": tool_params["file_structure"]},
                output=[tool_params["file_categories"]]
            ),
            ToolCall(
                name="Generate Organization Plan",
                description="Generates the expected folder structure and file placement.",
                input_parameters={
                    "current_structure": tool_params["file_structure"],
                    "classification": tool_params["file_categories"]
                },
                output=[tool_params["expected_dir_format"]]
            ),
            ToolCall(
                name="Get Linux Commands",
                description="Generates Linux commands to move, rename, or delete files as needed.",
                input_parameters={
                    "current_dir": tool_params["eval_dir"],
                    "expected_format": tool_params["expected_dir_format"]
                },
                output=[tool_params["expected_linux_commands"]]
            ),
            ToolCall(
                name="Execute Commands",
                description="Executes the generated Linux commands to apply the file organization.",
                input_parameters={"commands": tool_params["expected_linux_commands"]},
                output=[tool_params["expected_exe_log"]]
            )
        ]

    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""

        structure_metric = GEval(
            name="Structure Understanding",
            evaluation_steps=[
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
            TaskCompletionMetric(threshold=0.7, model=self.eval_llm)
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
        test_cases = self.load_test_cases()
        
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
                tools_called = self.get_tool_calls(test["tool_params"])
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