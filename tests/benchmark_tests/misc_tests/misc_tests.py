import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any
import csv
from datetime import datetime

from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase


from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
from tests.benchmark_tests.common.utils import write_results_to_csv, convert_registry_to_tool_call, get_tool_call_registry
from tests.benchmark_tests.run_eval import get_chat_response

from . import test_cases

class MiscTestRunner:
    """Handles miscellaneous test execution"""

    def __init__(self, test_cases: List[Golden]):
        self.test_cases = test_cases
        self.eval_llm = EvalLLM_4Bit()
    
    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""
        return [
            AnswerRelevancyMetric(model=self.eval_llm, include_reason=True),
            ContextualRelevancyMetric(model=self.eval_llm, include_reason=True),
        ]

    async def evaluate_response(self) -> None:
        """Evaluate responses for all test cases"""
        metrics = self.init_metrics()

        for test in self.test_cases:

            # Get LLM response through chat API
            response = await get_chat_response(test.input)
            
            # Extract response content and relevant nodes
            if response is not None:
                actual_output = response['result']['content']
                # Get source nodes if available
                if len(response['nodes']) > 0:
                    nodes_used = [node['text'] for node in response.get("nodes", [])]
                else:
                    nodes_used = ["None"]
            else:
                # Handle failed responses
                actual_output = ""
                nodes_used = ["None"]
            
            eval_case = LLMTestCase(
                input=test.input,
                actual_output=actual_output,
                expected_output=test.expected_output,
                context=test.context,
                retrieval_context=nodes_used
            )
            
            tool_registry = response['tools']

            # Convert tool call registry to a list of tool calls
            tool_calls = convert_registry_to_tool_call(tool_registry)

            # Apply each metric and log results
            for metric in metrics:
                metric.measure(eval_case)
                # Log metric results for analysis
                print(f"{metric.__class__.__name__} Score: {metric.score}")
                print(f"Reason: {metric.reason}")

                # Write row to CSV
                write_results_to_csv({
                    'test_case_input': test.input,
                    'metric_name': metric.__class__.__name__,
                    'score': metric.score,
                    'reason': metric.reason,
                    'app_response': actual_output,
                    'tools_called': tool_calls,
                    'expected_tools': "N/A"
                })
                print(f"Wrote results to CSV for test case: {test.input}")
async def main():

    runner = MiscTestRunner(test_cases)

    await runner.evaluate_response()


if __name__ == "__main__":
    asyncio.run(main())

