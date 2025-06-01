#!/bin/bash

# Create the conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Check if environment exists, if not create it
if ! conda env list | grep -q "metanavit"; then
    conda create -n metanavit python=3.11 -y
fi

# Activate the conda environment
conda activate metanavit

# Install system dependencies
sudo apt-get update
sudo apt-get install -y dos2unix

# Install conda packages
conda install -c conda-forge -y \
    python-dotenv \
    uvicorn \
    fastapi \
    psycopg2 \
    sqlalchemy \
    numpy \
    pandas

# Install manual dependencies
pip install -U deepeval
pip install python-dotenv
pip install lm-format-enforcer

# Install from requirements.txt if it exists
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "Setup completed! Environment 'metanavit' is ready."
echo "To activate: conda activate metanavit"
