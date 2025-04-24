"""
Summarization Test Module

Evaluates an LLM's ability to process documents (from JSON, DOCX, PDF, and TXT)
and generate factually correct summaries by:
1. Loading test cases from a JSON file.
2. Feeding document texts into a summarization API (our EvalLLM_4Bit model in eval__llm.py).
3. Evaluating responses against expected outputs using the SummarizationMetric from deepeval.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import csv
from datetime import datetime

from pydantic import BaseModel


from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

from tests.benchmark_tests.common.eval_llm import EvalLLM_4Bit
from tests.benchmark_tests.run_eval import get_chat_response
from tests.benchmark_tests.common.utils import write_results_to_csv

@dataclass
class TestContext:
    """Test execution context and configuration"""
    dataset_path: str

# Define a simple JSON schema for summarization outputs.
class SummarizationOutput(BaseModel):
    summary: str

"""Handles summarization test execution"""
class SummarizationTestRunner:
   

    def __init__(self, test_cases_path: Path):
        self.test_cases_path = test_cases_path
        self.eval_llm = EvalLLM_4Bit()

    
    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from a JSON file and return raw data."""
        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("test_cases", [])
    
    def init_metrics(self) -> List[Any]:
        """Initialize the SummarizationMetric."""
        metric = SummarizationMetric(
            threshold=0.5,
            model=self.eval_llm,  # Use the evaluation model as the judge.
            assessment_questions=[
                "Does the summary capture the critical details of the document?",
                "Is the summary factually correct?",
                "Is the summary concise and comprehensive?"
            ]
        )
        return [metric]
    
    async def evaluate_response(self, context: TestContext) -> None:
        """Evaluate test cases for summarization."""
        test_cases = self.load_test_cases()
        metrics = self.init_metrics()
    
        for test_case in test_cases:
            print(f"Test Case: {test_case['input']}")
            input_text = test_case["input"]

            expected_output = test_case["expected_output"]
            response = await get_chat_response(input_text)
            
            # The generate() method returns an instance of SummarizationOutput.
            result = self.eval_llm.generate(input_text, SummarizationOutput)
            generated_summary = result.summary
            
            print("Generated Summary:", generated_summary)
            print("-" * 50)
            
            # Create an LLMTestCase.
            llm_test_case = LLMTestCase(
                input=input_text,
                actual_output=generated_summary,
                expected_output=expected_output
            )
            
            # Evaluate each test case with all initialized metrics.
            for metric in metrics:
                metric.measure(llm_test_case)
                print(f"{metric.__class__.__name__} Score: {metric.score}")
                print(f"Reason: {metric.reason}")

                # Write row to CSV
                write_results_to_csv({
                    'test_case_input': input_text,
                    'metric_name': metric.__class__.__name__,
                    'score': metric.score,
                    'reason': metric.reason,
                    'app_response': generated_summary,
                    'tools_called': "N/A",
                    'expected_tools': "N/A",
                })

async def main():
    """Main entry point for test execution."""
    test_context = TestContext(dataset_path="")
    runner = SummarizationTestRunner(
        Path(__file__).parent / "test_cases.json")
    await runner.evaluate_response(test_context)

if __name__ == "__main__":
    asyncio.run(main())









