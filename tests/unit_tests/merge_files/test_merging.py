################################################
# Test script for merging content files (Kantaro Nakanishi)
# This script is used to test the merging of content files and the generation of a summary
# using a language model (LLM) in MetaNaviT.
# The script reads content files and a reference summary, merges the content files, and
# generates a summary using the LLM. The generated summary is then evaluated against the
# reference summary using semantic similarity metrics.
# The script is intended to be used as a unit test for the merging and summarization functionality
# in MetaNaviT.
############################################

import os
import glob
import deepeval
from deepeval.metrics import SemanticSimilarityMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Define the dataset path
data_path = "MetaNaviT/data/merge-files/email-meeting/"

# Identify summary and content files
summary_file = None
content_files = []

for file in glob.glob(os.path.join(data_path, "*.txt")):
    if "summary" in file.lower():
        summary_file = file
    else:
        content_files.append(file)

# Read the reference summary
if summary_file:
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            reference_summary = f.read().strip()
    except IOError as e:
        raise FileNotFoundError(f"Error reading reference summary file: {e}")
else:
    raise FileNotFoundError("Reference summary file not found!")

# Read and merge content files
def merge_content(files):
    if not files:
        raise FileNotFoundError("No content files found for merging.")
    
    content = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content.append(f.read().strip())
        except IOError as e:
            print(f"Error reading content file {file}: {e}")
    return "\n".join(content)

# Simulate LLM-generated summary (replace with actual LLM call in MetaNaviT)
# Placeholder: Truncate merged content to simulate LLM summary
generated_summary = merge_content(content_files)[:500]

# Define a test case for Deepeval
test_case = LLMTestCase(
    input_data={"content": content_files},
    actual_output=generated_summary,
    expected_output=reference_summary,
    metrics=[SemanticSimilarityMetric()],
    tags=["merge", "summary", "MetaNaviT"]
)

dataset = EvaluationDataset([test_case])

# Run the evaluation
deepeval.evaluate(dataset)