#!/bin/bash

# Ensure we're in the conda environment
eval "$(conda shell.bash hook)"
conda activate metanavit

case $1 in
  "generate")
    python -m app.engine.generate
    ;;
  "dev")
    python run.py dev
    ;;
  "setup")
    python setup.py
    ;;
  "build")
    python run.py build
    ;;
  *)
    echo "Usage: ./run.sh [generate|dev|setup|build]"
    exit 1
    ;;
esac