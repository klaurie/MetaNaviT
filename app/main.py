from fastapi import FastAPI
import httpx
import asyncio
import logging
from app.routes.api import router
from app.db.vector_store import PGVectorStore

app = FastAPI()
app.include_router(router)

logger = logging.getLogger(__name__)
pg_storage = PGVectorStore()

async def pull_model(model: str, max_retries: int = 5):
    """Pull a specific model with retries"""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://ollama:11434/api/pull",
                    json={"name": model},
                    timeout=600.0
                )
                response.raise_for_status()
                logger.info(f"Successfully pulled {model} model")
                return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to pull {model} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(10 * (attempt + 1))
            else:
                raise

async def check_model_exists(client: httpx.AsyncClient, model: str) -> bool:
    """Check if model is already downloaded"""
    try:
        response = await client.get("http://ollama:11434/api/tags")
        models = response.json().get("models", [])
        return any(m["name"] == model for m in models)
    except Exception:
        return False

async def pull_models():
    """Pull models only if they don't exist"""
    models = ["nomic-embed-text", "llama2:7b-chat"]
    
    async with httpx.AsyncClient() as client:
        for model in models:
            try:
                if not await check_model_exists(client, model):
                    logger.info(f"Pulling {model}...")
                    await pull_model(model)
                else:
                    logger.info(f"Model {model} already exists")
            except Exception as e:
                logger.error(f"Failed to handle {model}: {e}")
                raise

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize database pool
        await pg_storage.initialize()
        
        # Pull models if needed
        await pull_models()
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close the database pool on shutdown"""
    if pg_storage.pool:
        await pg_storage.pool.close()