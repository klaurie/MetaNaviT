This is a [LlamaIndex](https://www.llamaindex.ai/) project using [FastAPI](https://fastapi.tiangolo.com/) bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama).

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

4. Install PosgreSQL version 16
    Follow instructions here: https://www.postgresql.org/download/
   
6. Install pgvector
    Follow Instructions here: https://github.com/pgvector/pgvector

    In .env you may need to change the connection string based on your own configuration. The default user is postgres
    and I just set my password to 'password' for ease of use right now.


## Quick Start with Dev Container

The easiest way to get started is using the dev container, which handles all dependencies except Ollama and Database dependencies:

1. Install [Docker](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this project in VS Code
4. Click the green button in the bottom-left corner (or press `F1` and select "Dev Containers: Reopen in Container")
5. Wait for the container to build - all dependencies will be automatically installed!

## Manual Setup

If you prefer not to use the dev container, you'll need to:

1. Setup the environment with poetry:
```bash
poetry install
poetry env use python  # Create and activate a new virtual environment
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
poetry run generate
```

2. Run the app:
```bash
poetry run dev
```

Open [http://localhost:8000](http://localhost:8000) with your browser to start the app.

The example provides two different API endpoints:

1. `/api/chat` - a streaming chat endpoint
2. `/api/chat/request` - a non-streaming chat endpoint

You can test the streaming endpoint with the following curl request:

```
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

And for the non-streaming endpoint run:

```
curl --location 'localhost:8000/api/chat/request' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

You can start editing the API endpoints by modifying `app/api/routers/chat.py`. The endpoints auto-update as you save the file. You can delete the endpoint you're not using.

To start the app optimized for **production**, run:

```
poetry run prod
```

## Deployments

For production deployments, check the [DEPLOY.md](DEPLOY.md) file.

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex.

You can check out [the LlamaIndex GitHub repository](https://github.com/run-llama/llama_index) - your feedback and contributions are welcome!
