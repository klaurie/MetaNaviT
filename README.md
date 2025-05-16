# MetaNaviT
<p align="center">
  <img src="https://github.com/klaurie/MetaNaviT/blob/main/.frontend/public/metanavit.jpeg?raw=true" alt="MetaNaviT UI Preview" width="600"/>
</p>

An AI-powered resource management tool designed to help users organize, extract, and interact with digital content across local directories, cloud platforms, and web sources. It enhances search and transformation workflows using Large Language Models (LLMs) and metadata-aware retrieval strategies.



### Built With

[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-5D3FD3?style=for-the-badge)](https://www.llamaindex.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge)](https://www.postgresql.org/)

## 📑 Table of Contents

- [Project Identity](#project-identity)  
  - [Active Team & Roles](#active-team--roles)  
  - [Timeline / Status](#timeline--status)  
- [Value Proposition](#value-proposition)  
  - [Problem Statement](#problem-statement)  
  - [Target Audience](#target-audience)  
- [Core Feature and Benefits](#core-feature-and-benefits)  
- [Access and Usage](#access-and-usage)  
  - [Prerequisites](#prerequisites)  
  - [Environment Setup](#environment-setup)  
  - [Configuration](#configuration)  
  - [Running the Application](#running-the-application)  
- [Automated Setup Script (WIP)](#-automated-setup-script-wip)  
- [Technical Roadmap](#%EF%B8%8F-technical-roadmap)
- [Architecture](#architecture)  
  - [Frontend](#frontend)  
  - [Backend](#backend)  
  - [Index Manager](#index-manager)  
- [Scripts](#scripts)  
- [Development Challenges and Solutions](#development-challenges-and-solutions)  
- [Patch Notes & Upcoming Fixes](#-patch-notes--upcoming-fixes) 
- [Contact Information](#contact-information)  

---



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

1. &nbsp;<img src="https://img.icons8.com/fluent/512/anaconda--v2.png" alt="Anaconda Logo" width="20" style="vertical-align: middle;" />&nbsp; Install Miniconda https://www.anaconda.com/download/success

2. Setup the environment with Miniconda:

```bash
conda create --name metanavit python=3.11
conda activate metanavit
pip install -r requirements.txt
```

3. Start PostgreSQL service (Depends on installation method)

### Configuration

Check the parameters that have been pre-configured in the `.env` file in this directory:
```env
MODEL_PROVIDER=ollama
MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

If you are using any tools or data sources, you can update their config files in the `config` folder.

### Running the Application

1. Generate the embeddings of the documents in the `./data` directory:
```bash
./scripts/run.sh generate
```

2. Run the app:
```bash
./scripts/run.sh dev
```

Open http://localhost:8000 with your browser to start the app.

## 🔧 Automated Setup Script (WIP)

> _Coming soon:_ a single command that asks for your model name and API key, writes `~/.metanavit/config.toml`, and then starts the app.

### 🗺️ Technical Roadmap

| Milestone                                 | Date          | Status                |
|-------------------------------------------|---------------|-----------------------|
| Setup Script                              | May 18, 2025  | 🚧 Prototype (WIP)    |
| Public Beta Release                       | May 25, 2025  | 🚀 Planned            |
| Multimodal Integration                    | May 28, 2025  | 🗓️ Planned            |
| Production Deployment for OSU students    | Jun 1, 2025   | 🗓️ Planned            |
| First User Experience Update (Batch)      | Jun 7, 2025   | 🗓️ Planned            |

## Architecture

MetaNaviT employs a client-server architecture.

<p align="center">
  <img src="https://github.com/klaurie/MetaNaviT/blob/main/doc/workflow.png?raw=true" alt="MetaNaviT Workflow" width="600"/>
</p>
<p align="center"><em>Workflow</em></p>

### Frontend
A [Next.js](https://nextjs.org/) application located in the [`.frontend/`](.frontend/) directory. It provides the user interface and interacts with the backend API.

### Backend
A [Python](https://www.python.org/) [FastAPI](https://fastapi.tiangolo.com/) application located in the [`app/`](app/) directory. It exposes API endpoints for the frontend and handles the core logic.
*   **API Layer**: Defined in [`app/api/routers/`](app/api/routers/), with specific endpoints like chat functionalities in [`app/api/routers/chat.py`](app/api/routers/chat.py).
*   **Engine**: The core processing unit resides in [`app/engine/`](app/engine/). It leverages [LlamaIndex](https://www.llamaindex.ai/) for AI-powered indexing, retrieval, and agentic workflows (see [`app/engine/agents/ReadME.md`](app/engine/agents/ReadME.md)). This includes:
    *   Specialized tools like the [`CodeGeneratorTool`](app/engine/tools/artifact.py) for functionalities such as code artifact generation (defined in [`app/engine/tools/artifact.py`](app/engine/tools/artifact.py)).
    *   Mechanisms for data indexing ([`app/engine/index.py`](app/engine/index.py)) and querying.
*   **Database**: [PostgreSQL](https://www.postgresql.org/) with the [pgvector](https://github.com/pgvector/pgvector) extension is used for storing data and vector embeddings. Database interactions are managed by modules in [`app/database/`](app/database/).
### Index Manager

The Index Manager, defined in [`app/database/index_manager.py`](app/database/index_manager.py), plays a crucial role in optimizing the data ingestion and indexing process.

**Purpose:**
Its primary purpose is to efficiently manage the state of files and directories that have been processed or indexed. This prevents redundant re-indexing of unmodified files, saving significant processing time and resources, especially when dealing with large datasets or frequent updates.

**How it Works:**
1.  **Tracking File Metadata:** The Index Manager maintains database tables (e.g., `indexed_files` and `directory_processing_results` as seen in its `_create_tables` method) to store metadata about files and directories. This metadata includes file paths, modification times (`mtime`), and details about the indexing process (e.g., `process_name`, `process_version`).
2.  **Efficient Updates:** When new data is processed (e.g., via `crawl_file_system` in [`app/engine/loaders/file_system.py`](app/engine/loaders/file_system.py)), the Index Manager checks against the stored metadata. For instance, the `batch_insert_indexed_files` method uses an `ON CONFLICT` SQL clause to update existing records if a file is reprocessed, ensuring that only changed or new files trigger full re-indexing.
3.  **Path Filtering:** It includes logic like `is_path_blocked` to exclude specified system directories or hidden files from the indexing process, further refining efficiency.
4.  **Integration with Loaders:** Data loaders, such as the file system crawler, utilize the Index Manager to determine which files need to be read and processed, ensuring that only new or modified content is passed on for embedding and storage in the vector database.


### Scripts
Utility scripts in the [`scripts/`](scripts/) directory, such as [`run.sh`](scripts/run.sh), facilitate tasks like generating embeddings and running the application.

### Development Challenges and Solutions
MetaNaviT’s development posed several technical challenges tied to its ambitious goals of cross-resource metadata extraction, intelligent resource mapping, and integration of large language models (LLMs). Key challenges included:

File Search Complexity: Designing an efficient file search algorithm based on metadata and semantic content proved more difficult than anticipated. This component was initially prioritized but had to be deprioritized due to implementation hurdles.
Solution: The team redirected focus toward components with clearer development paths—such as file processing, embedding generation, and RAG pipeline setup—to maintain project momentum.

Integration Overhead: Combining tools like Tree-sitter, Ollama, pgvector, PostgreSQL, and LlamaIndex introduced compatibility and orchestration challenges.
Solution: Responsibilities were modularized among team members (e.g., embedding with pgvector, file processing, and image metadata extraction), reducing coupling and easing integration.

Resource Mapping & Metadata Standardization: Creating coherent, adaptable resource maps across varied data types (text, code, images) required resolving discrepancies in metadata structures and formats.
Solution: The use of structured metadata, predefined parsing strategies, and vector-based indexing helped normalize diverse resource inputs into a unified representation.

Performance Under Scale: Ensuring the system could handle large directories and data volumes without lag required optimizations in both frontend rendering and backend indexing.
Solution: Lazy loading, caching, and parallel indexing techniques were applied to improve system responsiveness.

Testing with Uncertain LLM Outputs: Evaluating LLM-driven features like summarization and transformation required new testing strategies due to the non-deterministic nature of generative models.
Solution: The team used pytest for core components and the DeepEval library to benchmark LLM outputs on metrics like answer relevancy and task completion.

### Configuration
Application settings are managed through an [`.env`](.env) file

Configuration files are located in [`config/`](config/) folder.  [`loaders.yaml`](config/loaders.yaml) is used to modify parameters for generating metanavit's index. 

[`tools.yaml`](config/tools.yaml) manage which tools the app should use and how they should be configured. Allows for certain tools to be ignored, if they are not desired.

## 📝 Patch Notes & Upcoming Fixes

Below is our current backlog of improvements, tied to GitHub issues and rollout status:

| Feature / Fix                         | GitHub Issue                                   | Status         | Details                                                                                  |
|---------------------------------------|------------------------------------------------|----------------|------------------------------------------------------------------------------------------|
| **Human-in-the-Loop Approval**        | [#74](https://github.com/klaurie/MetaNaviT/issues/74) | 🚧 In Progress | - Preview diffs before applying changes<br>- Approve/reject in the UI<br>- Audit log of all actions |
| **Expanded Coding Benchmarks**        | [#60](https://github.com/klaurie/MetaNaviT/issues/60) | 🚧 In Progress | - New CPU/GPU/memory throughput tests<br>- Language-specific reports (Python, JS, Go)<br>- Integrated test harness |
| **Enhanced Ranking Capabilities**     | [#58](https://github.com/klaurie/MetaNaviT/issues/58) | 🗓️ Planned     | - Hybrid BM25 + embeddings<br>- Dynamic weight tuning via query params<br>- “Boost by recency” toggle |
| **Real-Time File Change Detection**   | [#54](https://github.com/klaurie/MetaNaviT/issues/54) | 🗓️ Planned     | - File-watcher (watchdog/inotify)<br>- WebSocket update notifications<br>- Batching to prevent thrashing |
| **Index Manager Integration**         | [#53](https://github.com/klaurie/MetaNaviT/issues/53) | 🗓️ Planned     | - Incremental upserts with `ON CONFLICT`<br>- Schema migrations for metadata<br>- Retry logic on failures |
| **Audio I/O Functionality**           | [#23](https://github.com/klaurie/MetaNaviT/issues/23) | 🗓️ Planned     | - Mic recording & upload endpoint<br>- Server-side transcription (e.g. Whisper)<br>- TTS playback UI |
| **Better Document Chunking**          | [#22](https://github.com/klaurie/MetaNaviT/issues/22) | 🗓️ Planned     | - Semantic boundary detection<br>- Configurable chunk size/overlap<br>- Heuristics to avoid splitting code |

> _We’ll update this list as each milestone is completed. Stay tuned!_

## Contact Information
Our Email Addresses:
- Deepti R: ravidatd@oregonstate.edu
- Kaitlyn L: lauriek@oregonstate.edu
- Kantaro N: nakanika@oregonstate.edu
- Carlana S: soma@oregonstate.edu
- Jose G: sanchej7@oregonstate.edu
- John T: tranj8@oregonstate.edu
