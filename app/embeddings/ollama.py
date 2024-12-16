import os
import numpy as np
import logging
import httpx
from app.config import OLLAMA_HOST

logger = logging.getLogger("embeddings")
logger.setLevel(logging.ERROR)  # Reduce logging noise

# Use the same client configuration as helpers.py
TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=60.0,
    write=30.0
)

LIMITS = httpx.Limits(max_keepalive_connections=1, max_connections=2)

async def get_client():
    """Get configured async client"""
    return httpx.AsyncClient(
        timeout=TIMEOUT,
        limits=LIMITS,
        http2=False,
        verify=True,
        transport=httpx.AsyncHTTPTransport(
            retries=2,
            verify=True,
            http2=False
        )
    )

async def get_ollama_embedding(text):
    """Get embedding using async client with optimized settings"""
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
            
        # Clean and truncate text
        text = ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())
        text = text[:2048].strip()
        
        async with await get_client() as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text,
                    "options": {
                        "num_gpu": 1,
                        "num_thread": 1
                    }
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Ollama API error: {response.status_code}")
                
            data = response.json()
            embedding = data.get('embedding') or (data.get('embeddings', [None])[0])
            
            if not embedding or not isinstance(embedding, list):
                raise ValueError("Invalid embedding format")
                
            return np.array(embedding, dtype=np.float32)
            
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise
