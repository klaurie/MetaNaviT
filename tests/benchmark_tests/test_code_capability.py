"""Tests for LLM-based code capability using synthetic dataset"""

import os
import json
import subprocess
from pathlib import Path
import pytest
from scripts.generate_code_data import main, TEST_DATA_DIR
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.benchmark_tests.common.utils import write_results_to_csv

@pytest.fixture(scope="module")
def dataset():
    """Generate and return the dataset before running testsd."""
    return main()


def test_generate_code_dataset_structure(dataset):
    """Test if the generated dataset has the correct structure."""
    assert isinstance(dataset, list), "Dataset should be a list"

    for entry in dataset:
        assert isinstance(entry, dict), "Each dataset entry should be a dictionary"
        assert "filename" in entry, "Entry should contain 'filename'"
        assert "code" in entry, "Entry should contain 'code'"
        assert "summary" in entry, "Entry should contain 'summary'"
        assert isinstance(entry["filename"], str) and entry["filename"].strip(), "Filename should be a non-empty string"
        assert isinstance(entry["code"], str) and entry["code"].strip(), "Code should be a non-empty string"
        assert isinstance(entry["summary"], str), "Summary should be a string (even if empty)"


def test_filenames_have_valid_extensions(dataset):
    """Ensure filenames have valid extensions (Python, Java, JS)."""
    valid_extensions = {".py", ".java", ".js"}

    for entry in dataset:
        file_path = Path(entry["filename"])
        assert file_path.suffix in valid_extensions, f"Invalid file extension: {file_path.suffix}"


def test_summary_files_exist(dataset):
    """Check that each code file has a corresponding summary file."""
    for entry in dataset:
        summary_path = Path(entry["filename"]).with_suffix('.summary.json')
        assert summary_path.exists(), f"Missing summary file for {entry['filename']}"

        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            assert "summary" in summary_data, f"Summary file missing 'summary' key: {summary_path}"
            assert isinstance(summary_data["summary"], str), f"Summary should be a string: {summary_path}"


def test_code_execution_feasibility(dataset):
    """Check if Python and JavaScript code can be executed without syntax errors."""
    for entry in dataset:
        file_path = Path(entry["filename"])
        if file_path.suffix == ".py":
            try:
                subprocess.run(["python", "-m", "py_compile", file_path], check=True)
            except subprocess.CalledProcessError:
                assert False, f"Python file {entry['filename']} contains syntax errors"

        elif file_path.suffix == ".js":
            try:
                subprocess.run(["node", "--check", file_path], check=True)
            except subprocess.CalledProcessError:
                assert False, f"JavaScript file {entry['filename']} contains syntax errors"


def test_dataset_diversity(dataset):
    """Ensure dataset contains a mix of documented and undocumented code."""
    documented_count = sum(1 for entry in dataset if entry["summary"].strip())
    undocumented_count = sum(1 for entry in dataset if not entry["summary"].strip())

    assert documented_count > 0, "Dataset should have documented code"
    assert undocumented_count > 0, "Dataset should include undocumented code as well"


def test_code_summary_relevance(dataset):
    """Ensure that generated summaries are relevant to the code using LLM-based evaluation."""
    for entry in dataset:
        query = f"Summarize the following code:\n{entry['code']}"
        expected_response = entry["summary"]
        
        metric = AnswerRelevancyMetric(include_reason=True)
        test_case = LLMTestCase(query=query, expected_response=expected_response)


        assert_test(expected_response, test_case, metric=metric)



if __name__ == "__main__":
    pytest.main()
