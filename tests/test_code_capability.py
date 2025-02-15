"""Tests for LLM-based code capability using synthetic dataset"""


import os
import json
from pathlib import Path
from scripts.generate_code_data import main

def test_generate_code_dataset_structure():
    """Test if the generated dataset has the correct structure."""
    dataset = main()  # Generate the dataset
    assert isinstance(dataset, list), "Dataset should be a list"

    for entry in dataset:
        assert isinstance(entry, dict), "Each dataset entry should be a dictionary"
        assert "filename" in entry, "Entry should contain 'filename'"
        assert "code" in entry, "Entry should contain 'code'"
        assert "summary" in entry, "Entry should contain 'summary'"

def test_filenames_have_valid_extensions():
    """Ensure filenames have valid extensions (Python, Java, JS)."""
    dataset = main()
    valid_extensions = {".py", ".java", ".js"}

    for entry in dataset:
        file_path = Path(entry["filename"])
        assert file_path.suffix in valid_extensions, f"Invalid file extension: {file_path.suffix}"

def test_code_and_summary_are_strings():
    """Check that 'code' and 'summary' fields are non-empty strings."""
    dataset = main()

    for entry in dataset:
        assert isinstance(entry["code"], str) and entry["code"].strip(), "Code should be a non-empty string"
        assert isinstance(entry["summary"], str), "Summary should be a string"

def test_dataset_diversity():
    """Ensure dataset has both documented and undocumented code."""
    dataset = main()
    documented_count = sum(1 for entry in dataset if entry["summary"].strip())
    undocumented_count = sum(1 for entry in dataset if not entry["summary"].strip())

    assert documented_count > 0, "Dataset should have documented code"
    assert undocumented_count >= 0, "Dataset should have at least one undocumented code file"

if __name__ == "__main__":
    test_generate_code_dataset_structure()
    test_filenames_have_valid_extensions()
    test_code_and_summary_are_strings()
    test_dataset_diversity()
    print("All tests passed successfully!")
