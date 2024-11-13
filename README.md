# Current project structure:
- **docker-compose.yml**: Defines the services needed for the project.
- **metanavit_db.sql**: Dump file used to initialize PostgreSQL database.
- **LLMS Folder**: Contains Models for the Ollama service.
    
# Start software with docker compose (run container):

```bash
docker-compose up -d
```
- this will set up PostgresSQL database (metanavit_db).
- Set up the Ollama service to pull/run different models.

# Connecting to Database:

```bash
docker exec -it metanavit_db psql -U postgres -d metanavit
```
- this will allow you to connect to the db to see data or make changes.

**dataProcessing.py:**
- takes in user path to set a root directory to scan everything recursively inside it and stores it in the database for embedding.
- Collects metadata for each file, file name, type, size, last modifed date, and path, and stores in the **file_contents** table of database.
- Creates references to each file in the **embeddings** table.
  
**embedding.py:**
- this converts file content into embeddings using Ollama model **mxbai-embed-large** and stores it as a vector back in the database for the embedding's table.
- Queries the **embeddings** table to find records where the embedding field is **NULL**.
- script formats the metadata into texts, sends it to ollama's embedding API, and stores the genereated embedding vector back in the embeddings table.
- **Install dependencies**:

**Run Script:**
- First download these packages:
```bash
pip install psycopg2 requests
```
- Run dataProcessing.py:
```bash
python3 data_processing.py
```
- Run embedding.py:
```bash
python embedding.py
```

## Setting up Docker:

Download and install Docker Desktop for a consistent development environment, similar to production.

- [Docker Desktop Download](https://www.docker.com/products/docker-desktop)

## Docker Compose Ollama Configuration:

1. Open `docker-compose.yml`.
2. The configuration is pre-set for CPU usage and can be modified to support GPU access if needed. 
   
   For more information on the Ollama image and GPU support, visit the [Ollama Docker Hub Page](https://hub.docker.com/r/ollama/ollama).

3. You can add additional services and applications to the `docker-compose.yml` file as needed for your project.
