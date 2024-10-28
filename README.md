# Ollama Docker Setup

This guide provides instructions for setting up and running the Ollama Docker container to work with large language models.

## Download Docker Desktop

Download and install Docker Desktop for a consistent development environment, similar to production.

- [Docker Desktop Download](https://www.docker.com/products/docker-desktop)

## Docker Compose Configuration

1. Open `docker-compose.yml`.
2. The configuration is pre-set for CPU usage and can be modified to support GPU access if needed. 
   
   For more information on the Ollama image and GPU support, visit the [Ollama Docker Hub Page](https://hub.docker.com/r/ollama/ollama).

3. You can add additional services and applications to the `docker-compose.yml` file as needed for your project.

## Running The Container

To start the container, use the following command:

```bash
docker-compose up -d
```

## Running The Container

```bash
docker exec -it ollama ollama run llama3
```
