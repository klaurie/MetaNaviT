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
from typing import List, Dict, Any

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
    dataset_path: str

class TimeQueryTestRunner:
    """Handles time-based query test execution"""
    
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
        Define tool calls for evaluating time-based reasoning.
        """
        return [
            ToolCall(
                name="Time-Based Query",
                description="Evaluates the model's ability to answer time-based queries from documents.",
                input_parameters={"query": tool_params["query"]},
                output=[tool_params["expected_response"]]
            )
        ]
    
    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""
        
        relevancy_metric = AnswerRelevancyMetric()
        
        temporal_accuracy_metric = GEval(
            name="Temporal Accuracy",
            evaluation_steps=[
                "Does the response correctly reflect the actual creation or modification times?",
                "Is the time-based reasoning logical and coherent?",
                "Are the relations between different timestamps correctly inferred?"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.eval_llm,
            verbose_mode=True
        )
        
        return [
            relevancy_metric,
            temporal_accuracy_metric,
            TaskCompletionMetric(threshold=0.7, model=self.eval_llm)
        ]

    async def evaluate_response(self, context: TestContext) -> None:
        """Run evaluation for time-based query test cases."""
        metrics = self.init_metrics()
        test_cases = self.load_test_cases()
        
        for test in test_cases:
            response = await get_chat_response(test["query"])
            actual_output = response['result']['content'] if response else ""
            
            eval_case = LLMTestCase(
                input=test["query"],
                actual_output=actual_output,
                tools_called=self.get_tool_calls(test)
            )
            
            for metric in metrics:
                metric.measure(eval_case)
                print(f"{metric.__class__.__name__} Score: {metric.score}")
                print(f"Reason: {metric.reason}")

async def main():
    """Main entry point for test execution"""
    test_context = TestContext(dataset_path="test_data/time_queries.json")
    
    runner = TimeQueryTestRunner(
        Path(__file__).parent / "test_cases.json"
    )
    await runner.evaluate_response(test_context)

if __name__ == "__main__":
    asyncio.run(main())
