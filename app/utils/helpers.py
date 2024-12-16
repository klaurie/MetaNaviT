import json
import logging
import httpx
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from app.config import OLLAMA_HOST
import traceback
import os

# Set environment variables to reduce Ollama logging
os.environ["OLLAMA_LOG_LEVEL"] = "error"
os.environ["OLLAMA_DEBUG"] = "false"
os.environ["GGML_LOG_LEVEL"] = "0"  # Suppress GGML logs
os.environ["LLAMA_LOG_LEVEL"] = "0"  # Suppress LLAMA logs
os.environ["LLAMA_PRINT_META"] = "0"  # Suppress model metadata printing
os.environ["CUDA_LOG_LEVEL"] = "3"  # Suppress CUDA logs (3=ERROR only)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_CACHE_DISABLE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
# Disable verbose logging
logging.getLogger('app.db.vector_store').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('ollama').setLevel(logging.ERROR)

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

# Configure httpx client limits and timeouts
LIMITS = httpx.Limits(max_keepalive_connections=1, max_connections=2)  # Minimal connections
TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=300.0,
    write=30.0,
    pool=180.0
)

# Create a shared client session
_client = None
_ollama_client = None
_model_loaded = False
_model_lock = asyncio.Lock()

async def get_client():
    """Get or create shared client session"""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=TIMEOUT,
            limits=LIMITS,
            http2=False,
            verify=True,
            transport=httpx.AsyncHTTPTransport(
                retries=2,  # Increased retries
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
    """Get or create the model semaphore with optimized concurrency"""
    global _model_semaphore
    if _model_semaphore is None:
        # Allow 2 concurrent model calls but with managed scheduling
        _model_semaphore = asyncio.Semaphore(2)
    return _model_semaphore

# Add model queue management
_model_queue = asyncio.Queue(maxsize=10)
_processing = False

async def queue_model_request(func):
    """Queue model requests to prevent overload"""
    try:
        # Add to queue with timeout
        await asyncio.wait_for(
            _model_queue.put(func),
            timeout=30.0
        )
        
        # Start processing if not already running
        global _processing
        if not _processing:
            _processing = True
            asyncio.create_task(_process_model_queue())
            
        # Wait for result with timeout
        return await asyncio.wait_for(
            func(),
            timeout=180.0
        )
    except asyncio.TimeoutError:
        raise TimeoutError("Model request timed out")
    except Exception as e:
        logger.error(f"Error in model queue: {e}")
        raise

async def _process_model_queue():
    """Process queued model requests"""
    global _processing
    try:
        while not _model_queue.empty():
            func = await _model_queue.get()
            try:
                semaphore = await get_model_semaphore()
                async with semaphore:
                    await func()
            except Exception as e:
                logger.error(f"Error processing queued request: {e}")
            finally:
                _model_queue.task_done()
    finally:
        _processing = False

async def ensure_model_loaded(client: httpx.AsyncClient, model_name: str) -> bool:
    """Ensure model is loaded before making calls"""
    try:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m.get("name") == model_name for m in models)
        return False
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return False

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
    """Get embedding from Ollama with optimized settings"""
    from app.embeddings.ollama import get_ollama_embedding as get_embedding
    return await get_embedding(text)

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
    """Get response from Ollama using shared client and semaphore"""
    semaphore = await get_model_semaphore()
    client = await get_ollama_client()
    
    async def _make_request():
        try:
            # Acquire semaphore to ensure only one model call at a time
            async with semaphore:
                # Simplified model name
                model_name = "llama2"
                
                # Truncate context if too long
                max_context_length = 2048
                if len(context) > max_context_length:
                    context_words = context.split()
                    truncated_context = " ".join(context_words[:max_context_length])
                else:
                    truncated_context = context
                
                prompt = f"Context:\n{truncated_context}\n\nQuestion: {query}"
                
                # Simplified request with minimal parameters
                response = await client.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "temperature": 0.7
                    },
                    timeout=httpx.Timeout(timeout=120.0)  # 2 minutes timeout
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                if not result:
                    raise ValueError("Empty response from Ollama")
                return result
                
        except Exception as e:
            logger.error(f"Error getting Ollama response: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
    
    # Retry with shorter delays
    return await retry_async(_make_request, max_attempts=2, initial_delay=2)

def get_files_recursive(directory: str) -> List[str]:
    """
    Recursively get all files in a directory and its subdirectories.
    
    Args:
        directory (str): The directory to search in
        
    Returns:
        List[str]: List of file paths relative to the input directory
    """
    all_files = []
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Get full path
            file_path = os.path.join(root, file)
            # Convert to relative path
            rel_path = os.path.relpath(file_path, directory)
            # Convert Windows paths to Unix-style for consistency
            rel_path = rel_path.replace('\\', '/')
            # Add to list
            all_files.append(os.path.join(directory, rel_path))
            
    return all_files

async def get_ollama_client():
    """Get or create shared Ollama client with optimized model persistence"""
    global _ollama_client, _model_loaded
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.AsyncClient(
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
        
        if not _model_loaded:
            async with _model_lock:
                if not _model_loaded:
                    try:
                        await _ensure_models_loaded(_ollama_client)
                        _model_loaded = True
                    except Exception as e:
                        logger.error(f"Model initialization error: {e}")
                        _ollama_client = None  # Reset client on error
                        raise
    return _ollama_client

async def _ensure_models_loaded(client: httpx.AsyncClient) -> None:
    """Ensure required models are loaded and cached"""
    try:
        # Check currently loaded models
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code != 200:
            raise Exception("Failed to get model list")
            
        loaded_models = [m["name"] for m in response.json().get("models", [])]
        required_models = {"llama2", "nomic-embed-text"}
        missing_models = required_models - set(loaded_models)
        
        if not missing_models:
            logger.info("All required models already loaded")
            return
            
        for model in missing_models:
            logger.info(f"Loading missing model {model}...")
            response = await client.post(
                f"{OLLAMA_HOST}/api/pull",
                json={
                    "name": model,
                    "insecure": True
                },
                timeout=httpx.Timeout(600.0)
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to pull model {model}")
            
            logger.info(f"Model {model} loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

async def cleanup_clients():
    """Cleanup client connections and reset model state"""
    global _ollama_client, _model_loaded
    
    if _ollama_client is not None and not _ollama_client.is_closed:
        await _ollama_client.aclose()
        _ollama_client = None
    
    _model_loaded = False

async def create_ollama_response(query: str, context: str) -> str:
    """Create response using persistent model with optimized settings"""
    try:
        client = await get_ollama_client()
        if client is None:
            raise ValueError("Failed to initialize Ollama client")
            
        # Truncate context if too long
        max_context = 512
        if len(context) > max_context:
            context = context[:max_context]
        
        # Optimized request with minimal settings
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "llama2",
                "prompt": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
                "options": {
                    "num_gpu": 1,
                    "num_thread": 1,
                    "num_ctx": 512,
                    "num_predict": 64,
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_last_n": 64,
                    "repeat_penalty": 1.1,
                    "mirostat": 0,
                    "num_batch": 8,
                    "num_keep": -1
                }
            },
            timeout=httpx.Timeout(90.0)
        )
        
        if response.status_code != 200:
            raise ValueError(f"Ollama API error: {response.status_code}")
            
        result = response.json().get("response", "")
        if not result:
            raise ValueError("Empty response from Ollama")
            
        return result.strip()
        
    except httpx.TimeoutError:
        logger.error("Ollama request timed out")
        raise TimeoutError("Request to Ollama timed out")
    except Exception as e:
        logger.error(f"Error in create_ollama_response: {str(e)}")
        raise
