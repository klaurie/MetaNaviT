#!/bin/bash

# Create the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yml

# activate the conda environment
conda activate metanavit

# Install manual dependencies
pip install -U deepeval
pip install python-dotenv

pip install lm-format-enforcer
