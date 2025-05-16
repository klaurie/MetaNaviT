# MetaNaviT
![alt text](https://github.com/klaurie/MetaNaviT/blob/main/.frontend/public/metanavit.jpeg?raw=true)

An AI-powered resource management tool designed to help users organize, extract, and interact with digital content across local directories, cloud platforms, and web sources. It enhances search and transformation workflows using Large Language Models (LLMs) and metadata-aware retrieval strategies.

### Built With

[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-5D3FD3?style=for-the-badge)](https://www.llamaindex.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge)](https://www.postgresql.org/)

## Value Proposition


### Problem Statement

### Core Feature and Benefits

## Prerequisites

1. Install Ollama (required for both dev container and manual setup):
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download/windows
```

2. Start the Ollama service:
```bash
ollama serve
```

3. Pull the required model:
```bash
ollama pull llama3.2:1b
```

4. Download PostgreSQL 
    https://www.postgresql.org/download/

```bash
#If you are using MacOS, you might want to try this:
brew install postgresql@14

# To start postgresql@14 now and restart at login:
brew services start postgresql@14

# Or, if you don't want/need a background service you can just run:
/opt/homebrew/opt/postgresql@14/bin/postgres -D /opt/homebrew/var/postgresql@14
```
If you need to install Homebrew: https://brew.sh/

5. Download pgvector
    https://github.com/pgvector/pgvector 

6. Make sure Poetry package manager is installed
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

7. MacOS user might want to check NVM installed version. An installed version can be old.
```bash
node -v
# If your version is not 20, follow this:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

nvm install 20

nvm use 20

nvm alias default 20
# Check the version again.
node -v
```

## Quick Start with Dev Container (Recommended)

The easiest way to get started is using the dev container, which handles all dependencies except Ollama and Database dependencies:

1. Install [Docker](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this project in VS Code
4. Click the green button in the bottom-left corner (or press `F1` and select "Dev Containers: Reopen in Container")
5. Wait for the container to build - all dependencies will be automatically installed!

## Manual Setup (Alternative)

If you prefer not to use the dev container, you'll need to additionally:

1. Install [Miniconda] (https://www.anaconda.com/docs/getting-started/miniconda/install) 

2. Setup the environment with Miniconda:

```bash
conda create --name metanavit python=3.11 #create env
conda activate metanavit #use env
pip install -r requirements.txt #install dependencies
```

## Configuration

Check the parameters that have been pre-configured in the `.env` file in this directory:
```env
MODEL_PROVIDER=ollama
MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

If you are using any tools or data sources, you can update their config files in the `config` folder.

## Running the Application

1. Generate the embeddings of the documents in the `./data` directory:
```bash
./scripts/run.sh generate
```

2. Run the app:
```bash
./scripts/run.sh dev
```

Open [http://localhost:8000](http://localhost:8000) with your browser to start the app.

The example provides two different API endpoints:

1. `/api/chat` - a streaming chat endpoint
2. `/api/chat/request` - a non-streaming chat endpoint

You can test the streaming endpoint with the following curl request:

```bash
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

And for the non-streaming endpoint run:

```bash
curl --location 'localhost:8000/api/chat/request' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

You can start editing the API endpoints by modifying `app/api/routers/chat.py`. The endpoints auto-update as you save the file. You can delete the endpoint you're not using.

## Deployments

For production deployments, check the [DEPLOY.md](DEPLOY.md) file.

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex.

You can check out [the LlamaIndex GitHub repository](https://github.com/run-llama/llama_index) - your feedback and contributions are welcome!
