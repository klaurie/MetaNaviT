from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import logging
from app.routes.api import router
from app.db.vector_store import PGVectorStore
from app.config import DATABASE_URL, OLLAMA_HOST
from httpx import AsyncClient
from app.utils.helpers import cleanup_clients, get_ollama_client
from app.routes import api, debug  # Import debug router

logger = logging.getLogger(__name__)

app = FastAPI()

# Include both routers
app.include_router(api.router)
app.include_router(debug.router)  # Add this line

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

        # Check model availability using shared client
        logger.info("Checking model availability...")
        client = await get_ollama_client()
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]
        
        # Pull nomic-embed-text for embeddings if needed
        if "nomic-embed-text" in model_names:
            logger.info("Model nomic-embed-text is already available")
        else:
            logger.info("Pulling model nomic-embed-text...")
            response = await client.post(f"{OLLAMA_HOST}/api/pull", json={"name": "nomic-embed-text"})
            if response.status_code == 200:
                logger.info("Model nomic-embed-text pulled successfully")
            else:
                logger.error(f"Failed to pull model: {response.text}")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup connections on shutdown"""
    try:
        # Close database pool
        if hasattr(app.state, "pg_storage") and app.state.pg_storage.pool:
            await app.state.pg_storage.pool.close()
            logger.info("Database pool closed")
            
        # Cleanup HTTP clients
        await cleanup_clients()
        logger.info("HTTP clients cleaned up")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")
        raise

# Include API routes
app.include_router(router)