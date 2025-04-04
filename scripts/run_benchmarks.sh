#!/bin/bash

# Ensure we're in the conda environment
conda activate metanavit

# Run synthetic data generation for datasets that need it
python -m scripts.generate_time_data
python -m scripts.generate_code_data

# Generate index 
python -m app.engine.generate

# Run benchmarks
python -m tests.benchmark_tests.file_system_tests.test_organization
python -m tests.benchmark_tests.time_tests.test_time_queries
python -m tests.benchmark_tests.test_code_capability