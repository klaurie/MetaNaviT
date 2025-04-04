# Code-Based Test Dataset

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
