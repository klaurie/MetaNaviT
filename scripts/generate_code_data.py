"""
Code-Based Test Data Generator

Generates synthetic datasets to test MetaNaviT’s ability to summarize code files. 
Summarization should provide concise explanations of functionality, key components, 
and dependencies to improve readability and retrieval.

Usage:
    python -m tests.data.generate_code_data

Dataset Structure:
    /data/code_test_data/
        /python/        # Python scripts with functions and classes
            - calculator.py
            - data_processor.py
            - analysis.py
            
        /java/          # Java programs with basic structures
            - HelloWorld.java
            - UserManager.java
            - SortingAlgorithms.java
            
        /javascript/    # JavaScript modules for browser and backend use
            - app.js
            - utilities.js
            - data_fetcher.js
"""
import os
import json
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for test data
TEST_DATA_DIR = Path("data/code_test_data")

def setup_directory(base_dir: Path) -> None:
    """Create or clean the test data directory."""
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)

def create_file(filepath: Path, content: str, summary: str) -> None:
    """
    Create a code file with content and an associated summary file.
    
    Args:
        filepath: Path to create file
        content: Code content
        summary: Human-generated summary
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(content)

    summary_path = filepath.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump({"summary": summary}, f, indent=2)

    logger.info(f"Created {filepath} and {summary_path}")

def generate_python_files(base_dir: Path):
    """Generate Python test files."""
    py_dir = base_dir / "python"

    create_file(
        py_dir / "calculator.py",
        """def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n""",
        "A simple calculator module with addition and subtraction functions."
    )

def generate_java_files(base_dir: Path):
    """Generate Java test files."""
    java_dir = base_dir / "java"

    create_file(
        java_dir / "HelloWorld.java",
        """public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}""",
        "A simple Java program that prints 'Hello, World!' to the console."
    )

def generate_js_files(base_dir: Path):
    """Generate JavaScript test files."""
    js_dir = base_dir / "javascript"

    create_file(
        js_dir / "app.js",
        """function greet(name) {\n    return `Hello, ${name}!`;\n}\nconsole.log(greet(\"World\"));\n""",
        "A JavaScript function that greets a given name and logs it to the console."
    )

def generate_readme(base_dir: Path):
    """Generate a README explaining dataset structure."""
    readme_content = """# Code-Based Test Dataset

This dataset is designed to test MetaNaviT’s ability to summarize code files.

## Directory Structure

- **python/**: Python scripts with functions and classes.
- **java/**: Java programs with basic structures.
- **javascript/**: JavaScript modules for browser and backend use.

## Summarization Goals
- Generate docstring-style summaries for functions and classes.
- Extract key logic and dependencies.
- Summarize at both file-level and function-level.

## Expected Output Format
Each code file is accompanied by a .summary.json file containing:
{
  "summary": "Concise explanation of the file's functionality."
}
"""
    create_file(base_dir / "README.md", readme_content, "Dataset overview and structure.")

def generate_code_dataset():
    """Generate the complete dataset and return a list of file metadata."""
    setup_directory(TEST_DATA_DIR)
    generate_python_files(TEST_DATA_DIR)
    generate_java_files(TEST_DATA_DIR)
    generate_js_files(TEST_DATA_DIR)
    generate_readme(TEST_DATA_DIR)

    dataset = []
    for root, _, files in os.walk(TEST_DATA_DIR):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in {".py", ".java", ".js"}:
                with open(file_path, 'r') as f:
                    code_content = f.read()
                summary_path = file_path.with_suffix('.summary.json')
                summary = ""
                if summary_path.exists():
                    with open(summary_path, 'r') as sf:
                        summary_data = json.load(sf)
                        summary = summary_data.get("summary", "")

                dataset.append({
                    "filename": str(file_path),
                    "code": code_content,
                    "summary": summary
                })
    
    logger.info(f"Generated {len(dataset)} code files with summaries.")
    return dataset

def main():
    """Generate the dataset and return it."""
    return generate_code_dataset()

if __name__ == "__main__":
    main()
