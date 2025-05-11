#!/bin/bash

# Ensure we're in the conda environment
conda activate metanavit

# Run synthetic data generation for datasets that need it
#python -m scripts.generate_time_data
#python -m scripts.generate_code_data

# Generate index 


# Run benchmarks
python3 -m tests.benchmark_tests.misc_tests.misc_tests
python3 -m tests.benchmark_tests.search_tests.test_search
python -m tests.benchmark_tests.file_system_tests.test_organization
python -m tests.benchmark_tests.time_tests.test_time_queries
python -m tests.benchmark_tests.test_code_capability