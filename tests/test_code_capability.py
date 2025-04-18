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
from deepeval.metrics import FaithfulnessMetric
from langdetect import detect
from collections import defaultdict

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
        
        metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(query=query, expected_response=expected_response)

        assert_test(expected_response, test_case, metric=metric)


def test_no_duplicate_filenames(dataset):
    """Ensure all filenames are unique."""
    filenames = [entry["filename"] for entry in dataset]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames found in dataset"
    


def test_cross_language_consistency(dataset):
    """Ensure summaries are consistent across languages for similar functionality."""
    # Group entries by summary content (approximate matching)
    summary_to_entries: Dict[str, List[dict]] = {}
    for entry in dataset:
        summary = entry["summary"].strip()
        if summary:
            summary_to_entries.setdefault(summary, []).append(entry)

    for summary, entries in summary_to_entries.items():
        if len(entries) > 1:
            extensions = {Path(entry["filename"]).suffix for entry in entries}
            assert len(extensions) > 1, (
                f"Summary '{summary}' used for multiple files but only in one language: {extensions}"
            )
            
            
def test_code_length_bounds(dataset, min_lines=3, max_lines=300):
    """Ensure code samples aren't too short or too long."""
    for entry in dataset:
        line_count = len(entry["code"].splitlines())
        assert min_lines <= line_count <= max_lines, f"Code length out of bounds: {entry['filename']}"



def test_summary_faithfulness(dataset):
    """Test that summaries are faithful to code logic."""
    metric = FaithfulnessMetric(threshold=0.7)

    for entry in dataset:
        query = f"Does the following summary correctly describe the code?\n\nCode:\n{entry['code']}\n\nSummary:\n{entry['summary']}"
        test_case = LLMTestCase(query=query, expected_response="Yes")  # expecting LLM to confirm the correctness

        assert_test("Yes", test_case, metric=metric)


def test_code_contains_comments(dataset):
    """Check that some code files contain comments to ensure realistic code samples."""
    comment_patterns = {
        ".py": r"#.*?$|('''.*?'''|\"\"\".*?\"\"\")",
        ".java": r"//.*?$|/\*.*?\*/",
        ".js": r"//.*?$|/\*.*?\*/"
    }
    has_comments = False

    for entry in dataset:
        file_path = Path(entry["filename"])
        pattern = comment_patterns.get(file_path.suffix)
        if pattern:
            with open(file_path, 'r') as f:
                code = f.read()
                if re.search(pattern, code, re.MULTILINE | re.DOTALL):
                    has_comments = True
                    break

    assert has_comments, "No code files contain comments, which is unrealistic"
    
    
def test_summary_language_is_english(dataset):
    """Ensure all summaries are written in English."""
    for entry in dataset:
        if entry["summary"].strip():
            lang = detect(entry["summary"])
            assert lang == "en", f"Non-English summary detected in file: {entry['filename']}"

def test_summary_is_not_too_short_or_too_long(dataset, min_words=3, max_words=100):
    """Ensure summaries are within a reasonable word range."""
    for entry in dataset:
        summary = entry["summary"].strip()
        if summary:
            word_count = len(summary.split())
            assert min_words <= word_count <= max_words, f"Summary word count out of bounds in {entry['filename']}"


def test_code_uniqueness(dataset):
    """Ensure all code samples are unique."""
    seen = set()
    for entry in dataset:
        code = entry["code"].strip()
        assert code not in seen, f"Duplicate code detected in: {entry['filename']}"
        seen.add(code)


if __name__ == "__main__":
    pytest.main()
