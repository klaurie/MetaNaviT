#!/bin/bash
set -e  # Exit on error

echo "Setting up development environment..."

# Install system dependencies


# Set up Python environment
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
poetry install

# Set up Node.js environment
if [ -d ".frontend" ]; then
    cd .frontend
    npm install
    cd ..
fi

# Set up environment variables
echo "MODEL_PROVIDER=ollama" >> .env
echo "MODEL=llama3.2:1b" >> .env
echo "EMBEDDING_MODEL=BAAI/bge-base-en-v1.5" >> .env
echo "PG_CONNECTION_STRING=postgresql://postgres:password@localhost:5432/metanavit" >> .env
echo "APP_HOST=0.0.0.0" >> .env
echo "APP_PORT=8000" >> .env

# Initialize database
if [ -f "setup.py" ]; then
    python setup.py
fi

echo "Environment setup complete!"
