"""
Time-Based Query Test Module

Evaluates LLM's ability to process and reason about time-based queries by:
1. Loading test cases from JSON
2. Running test cases through chat API
3. Evaluating responses using metrics
4. Tracking tool usage and commands
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import csv 
from datetime import datetime

from deepeval.metrics import AnswerRelevancyMetric, GEval, TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams

from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
from tests.benchmark_tests.run_eval import get_chat_response
from tests.benchmark_tests.common.utils import write_results_to_csv, convert_registry_to_tool_call

@dataclass
class TestContext:
    """Test execution context and configuration"""
    dataset_path: str

class TimeQueryTestRunner:
    """Handles time-based query test execution"""
    
    def __init__(self, test_cases_path: Path):
        self.test_cases_path = test_cases_path
        self.eval_llm = EvalLLM_4Bit()

    def load_test_cases(self) -> List[LLMTestCase]:
        """Load test cases from JSON file and return raw data"""
        with open(self.test_cases_path, 'r') as f:
            data = json.load(f)
            return data["test_cases"]

    def get_tool_calls(self, tool_params) -> List[ToolCall]:
        """
        Define tool calls for evaluating time-based reasoning.
        
        These tools measure key aspects:
        1. Time Understanding: Can the LLM correctly interpret file timestamps
        2. Chronological Relationship: Can the LLM determine temporal relationships
        3. File Filtering: Can the LLM filter files based on time constraints
        """
        tool_calls = [
            ToolCall(
                name="Time-Based File Query",
                description="Evaluates the model's ability to answer time-based queries about files.",
                input_parameters={
                    "file_structure": tool_params.get("file_structure", []),
                    "file_metadata": tool_params.get("file_metadata", {})
                },
                output=[tool_params["expected_dir_format"]]
            ),
            ToolCall(
                name="Chronological Analysis",
                description="Evaluates the model's understanding of file timestamp relationships.",
                input_parameters={
                    "file_structure": tool_params.get("file_structure", []),
                    "file_metadata": tool_params.get("file_metadata", {})
                },
                output=["Correct chronological ordering and time relationships"]
            )
        ]
        
        # Only add command evaluation if there are expected Linux commands
        if "expected_linux_commands" in tool_params and tool_params["expected_linux_commands"]:
            tool_calls.append(
                ToolCall(
                    name="Time-Based Command Generation",
                    description="Evaluates the model's ability to generate timestamp-oriented commands.",
                    input_parameters={"query": "Generate commands for time-based operations"},
                    output=[tool_params["expected_linux_commands"]]
                )
            )
            
        return tool_calls
    
    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""
        relevancy_metric = AnswerRelevancyMetric(model=self.eval_llm)
        completion_metric = TaskCompletionMetric(threshold=0.7, model=self.eval_llm)
        time_understanding_metric = GEval(
            name="Time Understanding",
            criteria="The response correctly interprets file timestamps and temporal relationships",
            evaluation_params={
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            },
            model=self.eval_llm
        )
        
        return [relevancy_metric, completion_metric, time_understanding_metric]
    
    async def evaluate_response(self, context: TestContext) -> None:
        """Evaluate test cases for time-based queries"""
        test_cases = self.load_test_cases()
        metrics = self.init_metrics()
        
        for test_case in test_cases:
            print(f"Test Case: {test_case['input']}")
            input_query = test_case["input"]
            expected_output = test_case["expected_response"]
            tool_params = test_case.get("tool_params", {})
            
            # Get response from the chat API
            response = await get_chat_response(input_query)
            tool_registry = response.get('tools', [])
            tools_called = convert_registry_to_tool_call(tool_registry)

            # Create test case for evaluation
            llm_test_case = LLMTestCase(
                input=input_query,
                actual_output=response,
                expected_output=expected_output,
                tools_called=tools_called
            )
            
            # TODO: John let me know if these changes are ok, it had an error at runtime
            # Run evaluation with metrics
            for metric in metrics:
                metric.measure(llm_test_case)
                print(f"{metric.__class__.__name__} Score: {metric.score}")
                print(f"Reason: {metric.reason}")

                # Write row to CSV
                write_results_to_csv({
                    'test_case_input': test_case["input"],
                    'metric_name': metric.__class__.__name__,
                    'score': metric.score,
                    'reason': metric.reason,
                    'app_response': response,
                    'tools_called': tools_called,
                    'expected_tools': self.get_tool_calls(tool_params),
                })


async def main():
    """Main entry point for test execution"""
    test_context = TestContext(
        dataset_path=""
    )
    
    runner = TimeQueryTestRunner(
        Path(__file__).parent / "test_cases.json"
    )
    await runner.evaluate_response(test_context)

if __name__ == "__main__":
    asyncio.run(main())
