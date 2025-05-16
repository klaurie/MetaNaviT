# MetaNaviT
![alt text](https://github.com/klaurie/MetaNaviT/blob/main/.frontend/public/metanavit.jpeg?raw=true)

An AI-powered resource management tool designed to help users organize, extract, and interact with digital content across local directories, cloud platforms, and web sources. It enhances search and transformation workflows using Large Language Models (LLMs) and metadata-aware retrieval strategies.

### Built With

[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-5D3FD3?style=for-the-badge)](https://www.llamaindex.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge)](https://www.postgresql.org/)


## 📌 Project Identity

### Active Team & Roles
- **Deepti R.** — Benchmark Testing & Documentation  
- **Carlana S.** — Benchmark Testing & Backend  
- **Kaitlyn L.** — DevOps / CI & Benchmark Infrastructure  
- **John T.** — Front-end / UX & Execution Sandbox  
- **Kantaro N.** — Retrieval & Indexing  
- **Jose S.** — Retrieval & Data APIs  

### Timeline / Status
- **Feb – Apr 2025:** Core FastAPI + LlamaIndex pipeline  
- **May 2025:** Benchmark harness + docs in progress ✅ *(current)*  
- **Jun 2025:** Usability tests → production deploy 🗓️  

## Value Proposition


### Problem Statement:
Traditional code retrieval and assistance tools often overlook contextual metadata, focusing solely on source code without accounting for surrounding documentation, comments, or user-specific context. This lack of context-aware processing limits the accuracy and depth of the insights these tools can provide, reducing their usefulness in complex development workflows. While current alternatives such as Aider, GitHub Copilot, and Cursor offer intelligent code suggestions and rely on tools like Tree-sitter for syntax parsing, they remain code-centric and do not effectively integrate broader contextual information. This gap highlights the need for more holistic, context-aware solutions in code understanding and retrieval.

### Target Audience
Our primary audience include researchers, content managers, data analysts, software engineers, and students, each with distinct needs:

Researcher: Search through academic papers and retrieve documents related to specific topics or keywords, even when file names do not contain those terms, using semantic search capabilities.

Content Managers: Manage large collections of media files, where images, videos, and documents are automatically categorized based on metadata such as tags, dates, and formats, improving organization.

Data Analyst: Automate the compilation of multiple CSV files from different sources into a single, cohesive dataset, applying transformations such as column standardization and data merging.

Software Engineer: Utilize advanced indexing, retrieval, and code generation capabilities to efficiently manage project files. MetaNaviT scans and indexes code files, documentation, and related resources, allowing for quick searches without manually navigating directories. Additionally, the tool can generate code based on retrieved context, assisting with automation, debugging, and project development.

Student: Organize and retrieve academic resources, including lecture notes, research papers, and textbooks, using metadata-based and contextual queries for faster access.


### Core Feature and Benefits

## Access and Usage 📘

### Prerequisites

1. Install Ollama https://ollama.com/download <img src="https://ollama.com/public/assets/c889cc0d-cb83-4c46-a98e-0d0e273151b9/42f6b28d-9117-48cd-ac0d-44baaf5c178e.png" alt="Ollama Logo" width="20" style="vertical-align: middle;" />

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

### Quick Start with Dev Container (Recommended)

The easiest way to get started is using the dev container, which handles all dependencies except Ollama and Database dependencies:

1. Install [Docker](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this project in VS Code
4. Click the green button in the bottom-left corner (or press `F1` and select "Dev Containers: Reopen in Container")
5. Wait for the container to build - all dependencies will be automatically installed!

### Manual Setup (Alternative)

If you prefer not to use the dev container, you'll need to additionally:

1. Install [Miniconda] (https://www.anaconda.com/docs/getting-started/miniconda/install) 

2. Setup the environment with Miniconda:

```bash
conda create --name metanavit python=3.11 #create env
conda activate metanavit #use env
pip install -r requirements.txt #install dependencies
```

### Configuration

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
