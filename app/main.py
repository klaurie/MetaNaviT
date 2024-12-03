from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import logging
from app.routes.api import router
from app.db.vector_store import PGVectorStore
from app.config import DATABASE_URL, OLLAMA_HOST

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Initialize the vector store with correct credentials
pg_storage = PGVectorStore(database_url=DATABASE_URL)

@app.on_event("startup")
async def startup_event():
    """Initialize connections and pull models on startup"""
    try:
        # Initialize database connection
        logger.info("Starting database initialization...")
        await pg_storage.initialize()
        
        # Verify connection is working
        async with pg_storage.pool.acquire() as conn:
            await conn.execute('SELECT 1')
        logger.info("Database connection verified")
        
        # Store pg_storage in app state
        app.state.pg_storage = pg_storage
        logger.info("PGVectorStore stored in app state")

        # Check if models are available
        logger.info("Checking model availability...")
        from app.utils.helpers import get_client
        client = await get_client()
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        models = response.json().get("models", [])
        
        required_models = ["llama2:7b-chat", "nomic-embed-text"]
        for model in required_models:
            if not any(m.get("name") == model for m in models):
                logger.info(f"Pulling model {model}...")
                await client.post(
                    f"{OLLAMA_HOST}/api/pull",
                    json={"name": model},
                    timeout=600.0
                )
                logger.info(f"Model {model} pulled successfully")
            else:
                logger.info(f"Model {model} is already available")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup connections on shutdown"""
    try:
        if hasattr(app.state, 'pg_storage'):
            await app.state.pg_storage.close()
            logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Include routers
app.include_router(router)