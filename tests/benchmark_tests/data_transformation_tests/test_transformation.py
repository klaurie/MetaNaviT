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
from deepeval.metrics import AnswerRelevancyMetric, GEval, TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams
from deepeval.integrations import trace_llama_index
from deepeval.auto_evaluate import auto_evaluate

from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
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
        
    def load_test_cases(self) -> Dict[str, Any]:
        """Load test cases from JSON file and return raw data"""
        with open(self.test_cases_path, 'r') as f:
            data = json.load(f)
            return data
    
    def get_tool_calls(self, tool_params: List[Dict]) -> List[ToolCall]:
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
                    input_parameters=tool_param.get("input_parameters", {}),
                    output=tool_param.get("output", [])
                )
            )
            
        return tool_calls
    
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
            TaskCompletionMetric(threshold=0.7, model=self.eval_llm)
        ]
    
    async def evaluate_response(self, context: TestContext) -> None:
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
        test_case_data = self.load_test_cases()
        
        # Process the test case
        # Get LLM response through chat API
        response = await get_chat_response(test_case_data["input"])
        
        # Extract response content
        if response is not None:
            actual_output = response['result']['content']
        else:
            # Handle failed responses
            actual_output = ""
        
        # Create evaluation case
        eval_case = LLMTestCase(
            input=test_case_data["input"],
            actual_output=actual_output,
            expected_output=test_case_data.get("expected_output", ""),
            tools_called=self.get_tool_calls(test_case_data["tool_params"])
        )
        
        # Apply each metric and log results
        for metric in metrics:
            metric.measure(eval_case)
            # Log metric results for analysis
            print(f"{metric.__class__.__name__} Score: {metric.score}")
            print(f"Reason: {metric.reason}")
            
        # Perform auto-evaluation if enabled
        auto_evaluate(eval_case)


async def main():
    """Main entry point for test execution"""
    # Get the schema from the test case
    test_case_path = Path(__file__).parent / "test_cases.json"
    with open(test_case_path, 'r') as f:
        test_data = json.load(f)
        
    # Extract schema from test case
    schema = {}
    for tool_param in test_data["tool_params"]:
        if tool_param["name"] == "Extract Bird Observations" and "schema" in tool_param["input_parameters"]:
            schema = tool_param["input_parameters"]["schema"]
            break
    
    # Create test context
    test_context = TestContext(
        extraction_schema=schema,
        expected_format="CSV",
        expected_fields=list(schema.keys()) if schema else []
    )
    
    # Run tests
    runner = DataTransformationTestRunner(test_case_path)
    await runner.evaluate_response(test_context)


if __name__ == "__main__":
    asyncio.run(main())