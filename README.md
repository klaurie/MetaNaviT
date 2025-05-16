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
- **Jan – Apr 2025:** Core FastAPI + LlamaIndex pipeline  
- **May 2025:** Benchmark harness + documentation in progress ✅ *(current)*  
- **Jun 2025:** Usability tests → production deploy 🗓️  

## 🌟 Value Proposition


### Problem Statement:
Traditional code retrieval and assistance tools often overlook contextual metadata, focusing solely on source code without accounting for surrounding documentation, comments, or user-specific context. This lack of context-aware processing limits the accuracy and depth of the insights these tools can provide, reducing their usefulness in complex development workflows. While current alternatives such as Aider, GitHub Copilot, and Cursor offer intelligent code suggestions and rely on tools like Tree-sitter for syntax parsing, they remain code-centric and do not effectively integrate broader contextual information. This gap highlights the need for more holistic, context-aware solutions in code understanding and retrieval.

### Target Audience:

Our primary audience include researchers, content managers, data analysts, software engineers, and students, each with distinct needs:

| Audience Type  |  Needs   |
| --- | --- |
| Researcher     |  Semantic search of academic paper by topic or keywords  |
| Content Manager | Auto-categorize media files using metadata(tag, data, formats) |
| Data Analyst    |  Merge and standardize CSVs from multiple sources    |
| Software Engineer  | Auto-index code/doc for fast, context search and snippet generation|
|   Student       | Organize and retrieve academic materials using metadata and contextual queries|




### Core Feature and Benefits:



#### 1. Intelligent File Indexing & Retrieval
The system automatically scans and indexes files using advanced techniques like BM25 and semantic embeddings. This enables fast, context-aware searching even when queries don’t match exact keywords, improving access to relevant information.

#### 2. Specialized Agents
Multiple dedicated agents—including File Reader, Dependency Identifier, Python Code Executor, and Task Router—work together to handle specific subtasks such as metadata extraction, context clustering, and execution of system commands. 

#### 3. Efficient Index Management
By tracking file modification times, the system avoids unnecessary reindexing. This optimization preserves accuracy while significantly reducing processing time, especially helpful in dynamic environments with frequent updates.

#### 4. Safe File Change Simulation
File modifications are safely tested in a sandboxed environment before being applied. This allows users to preview and approve or reject changes, providing a layer of human oversight for automated workflows and ensuring system safety.

## 📘 Access and Usage

### Prerequisites

1. &nbsp;<img src="https://ollama.com/public/assets/c889cc0d-cb83-4c46-a98e-0d0e273151b9/42f6b28d-9117-48cd-ac0d-44baaf5c178e.png" alt="Ollama Logo" width="20" style="vertical-align: middle;" />&nbsp; Install Ollama https://ollama.com/download

2. Start the Ollama service:
```bash
ollama serve
```

3. Pull the required model:
```bash
ollama pull llama3.2:1b
```

4. &nbsp;<img src="https://www.postgresql.org/media/img/about/press/elephant.png" alt="PostgreSQL Logo" width="20" style="vertical-align: middle;" />&nbsp; Install PostgreSQL https://www.postgresql.org/download/ & pgvector https://github.com/pgvector/pgvector 

### Environment Setup

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


## Contact Information
Our Email Addresses:
- Deepti R: ravidatd@oregonstate.edu
- Kaitlyn L: lauriek@oregonstate.edu
- Kantaro N: nakanika@oregonstate.edu
- Carlana S: soma@oregonstate.edu
- Jose G: sanchej7@oregonstate.edu
- John T: tranj8@oregonstate.edu
