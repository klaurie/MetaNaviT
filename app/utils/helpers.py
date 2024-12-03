import json
import logging
import httpx
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from app.config import OLLAMA_HOST
import traceback

logger = logging.getLogger(__name__)

OLLAMA_API_BASE = OLLAMA_HOST

RELATIONSHIP_PROMPT = """Analyze this text and list any relationships between entities (people, companies, technologies, concepts).
For each relationship, write one line in this format:
[source] -> [type] -> [target]

For example:
Paul Graham -> founded -> Y Combinator
Y Combinator -> invested in -> Dropbox
Dropbox -> uses -> Python

If no relationships are found, write "No relationships found."

Text to analyze:
{content}"""

# Configure httpx client limits
LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)
TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=120.0,  # Increased for model loading
    write=30.0,
    pool=60.0
)

# Create a shared client session
_client = None

async def get_client():
    """Get or create shared client session"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=TIMEOUT,
            limits=LIMITS,
            http2=False,
            verify=True,
            transport=httpx.AsyncHTTPTransport(
                retries=3,
                verify=True,
                http2=False
            )
        )
    return _client

async def retry_async(func, max_attempts=3, initial_delay=1):
    """Custom retry logic for async functions"""
    last_error = None
    delay = initial_delay

    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                logger.warning(f"Retry attempt {attempt + 1}/{max_attempts}")
            return await func()
        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed: {e.__class__.__name__}: {str(e)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
                delay *= 2
            continue
    
    raise last_error

def chunk_text(text: str, chunk_size: int = 4096) -> List[str]:
    """Split text into chunks of roughly equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def parse_relationships(text: str) -> List[Dict[str, Any]]:
    """Parse relationships from text output into structured format."""
    relationships = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line == "No relationships found.":
            continue
            
        # Try to parse "source -> type -> target" format
        parts = [p.strip() for p in line.split('->')]
        if len(parts) == 3:
            relationships.append({
                "source": parts[0],
                "target": parts[2],
                "type": parts[1],
                "strength": 1.0,
                "description": line
            })
    
    return relationships

# Semaphore to limit concurrent model calls
_model_semaphore = None

async def get_model_semaphore():
    """Get or create the model semaphore"""
    global _model_semaphore
    if _model_semaphore is None:
        _model_semaphore = asyncio.Semaphore(1)  # Only allow 1 concurrent model call
    return _model_semaphore

async def extract_relationships_from_text(content: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    # Split content into manageable chunks
    chunks = chunk_text(content, chunk_size=4096)
    all_relationships = []
    semaphore = await get_model_semaphore()
    
    for chunk_idx, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
        
        async def _make_request() -> dict:
            async with semaphore:  # Limit concurrent requests
                client = await get_client()
                response = await client.post(
                    f"{OLLAMA_API_BASE}/api/generate",
                    json={
                        "model": "llama2",
                        "prompt": RELATIONSHIP_PROMPT.format(content=chunk),
                        "stream": False,
                        "raw": True,
                        "context_length": 2048,  # Reduced context length
                        "num_predict": 512,      # Reduced prediction length
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                )
                response.raise_for_status()
                return response.json()

        for attempt in range(max_retries):
            try:
                result = await _make_request()
                response_text = result.get("response", "").strip()
                
                # Log the raw response for debugging
                logger.debug(f"Raw response from model: {response_text}")
                
                # Parse relationships from the text response
                chunk_relationships = parse_relationships(response_text)
                if chunk_relationships:
                    all_relationships.extend(chunk_relationships)
                    logger.debug(f"Found {len(chunk_relationships)} relationships in chunk {chunk_idx + 1}")
                break
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to process chunk after {max_retries} attempts")
                # Exponential backoff between retries
                await asyncio.sleep(2 ** attempt)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_relationships = []
    for rel in all_relationships:
        rel_key = (rel.get("source"), rel.get("target"), rel.get("type"))
        if all(rel_key) and rel_key not in seen:  # Ensure all key components exist
            seen.add(rel_key)
            unique_relationships.append(rel)
    
    logger.info(f"Extracted {len(unique_relationships)} unique relationships from {len(chunks)} chunks")
    return unique_relationships

async def get_ollama_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama"""
    
    async def _make_request():
        try:
            logger.debug(f"Getting embedding for text of length: {len(text)}")
            client = await get_client()
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                content=json.dumps({
                    "model": "nomic-embed-text",
                    "prompt": text
                }),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            logger.debug(f"Successfully got embedding of length: {len(embedding)}")
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {e.__class__.__name__}: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    return await retry_async(_make_request)

async def check_ollama_health() -> Dict[str, Any]:
    """Check if Ollama service is available"""
    try:
        client = await get_client()
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            return {"status": "available", "models": response.json().get("models", [])}
        return {"status": "unavailable", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error checking Ollama health: {e.__class__.__name__}: {str(e)}")
        return {"status": "unavailable", "error": str(e)}

async def get_ollama_response(query: str, context: str) -> str:
    """Get response from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
            
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                content=json.dumps({
                    "model": "llama2:7b-chat",
                    "prompt": prompt,
                    "stream": False
                }),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Error getting Ollama response: {str(e)}")
        raise
