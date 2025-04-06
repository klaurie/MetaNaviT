"""
Data Transformation Test Module

Evaluates LLM's ability to extract and transform unstructured data by:
1. Loading test cases from JSON
2. Running test cases through chat API
3. Evaluating responses using metrics
4. Tracking tool usage and data quality
"""

import json
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
    extraction_schema: Dict
    expected_format: str
    expected_fields: List[str]


class DataTransformationTestRunner:
    """Handles data transformation test execution"""
    
    def __init__(self, test_cases_path: Path):
        self.test_cases_path = test_cases_path
        self.eval_llm = EvalLLM_4Bit()
        trace_llama_index(auto_eval=True)
        
    
    def init_metrics(self) -> List[Any]:
        """Initialize evaluation metrics"""
        
        extraction_quality_metric = GEval(
            name="Extraction Quality",
            evaluation_steps=[
                "Does the extraction correctly identify all bird observations in the text?",
                "Are all required fields (Date, Location, Species, Count, Behavior) extracted for each observation?",
                "Are the values correctly extracted based on the context?",
                "Is there any hallucination or invented data not present in the source?"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.eval_llm,
            verbose_mode=True
        )
        
        format_quality_metric = GEval(
            name="Format Quality",
            evaluation_steps=[
                "Is the CSV format valid and well-structured?",
                "Are the column headers properly named according to the schema?",
                "Are field values properly formatted (dates, numbers, text)?",
                "Would the CSV be machine-readable by a standard CSV parser?"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.eval_llm,
            verbose_mode=True
        )

        return [
            extraction_quality_metric,
            format_quality_metric,
            TaskCompletionMetric(threshold=0.7, model=self.eval_llm),
            ToolCorrectnessMetric(verbose_mode=True, should_consider_ordering=True)
        ]
    
    async def evaluate_response(self) -> None:
        """
        Run evaluation for data transformation test cases.

        This method:
        1. Initializes evaluation metrics (extraction quality, format quality, task completion)
        2. Loads test cases from JSON configuration
        3. Makes async requests to chat API for each test
        4. Processes responses and extracts relevant information
        5. Creates evaluation cases with context and tool calls
        6. Measures performance using configured metrics
        
        Args:
            context (TestContext): Test execution context containing:
                - extraction_schema: Schema for data extraction
                - expected_format: Expected output format (CSV)
                - expected_fields: Required fields in the output
        """
        # Initialize metrics and load test cases
        metrics = self.init_metrics()
        test_case_data = load_test_cases(self.test_cases_path)
        
        # Process the test case
        # Get LLM response through chat API
        for test_case in test_case_data:
            response = await get_chat_response(test_case["input"])
            
            # Extract response content
            if response is not None:
                actual_output = response['result']['content']
            else:
                # Handle failed responses
                actual_output = ""
            
            tools_called = convert_registry_to_tool_call()

            # Create evaluation case
            eval_case = LLMTestCase(
                input=test_case["input"],
                actual_output=actual_output,
                tools_called=tools_called,
                expected_tools=convert_test_case_tool_calls(test_case["tool_params"])
            )
            
            # Apply each metric and log results
            for metric in metrics:
                metric.measure(eval_case)
                # Log metric results for analysis
                print(f"{metric.__class__.__name__} Score: {metric.score}")
                print(f"Reason: {metric.reason}")
            


async def main():
    """Main entry point for test execution"""
    # Get the schema from the test case
    test_case_path = Path(__file__).parent / "test_cases.json" 
    
    # Run tests
    runner = DataTransformationTestRunner(test_case_path)
    await runner.evaluate_response()


if __name__ == "__main__":
    asyncio.run(main())