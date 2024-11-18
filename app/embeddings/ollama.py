import os
import requests
import numpy as np
import logging
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger("embeddings")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = "nomic-embed-text"

session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('http://', HTTPAdapter(max_retries=retries))

def pull_model():
    try:
        logger.info(f"Pulling model {OLLAMA_MODEL}...")
        response = session.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": OLLAMA_MODEL},
            timeout=300
        )
        if response.status_code != 200:
            logger.error(f"Failed to pull model: {response.status_code} - {response.text}")
            raise ValueError(f"Failed to pull model: {response.status_code}")
        logger.info(f"Successfully pulled model {OLLAMA_MODEL}")
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        raise

def get_ollama_embedding(text):
    try:
        # Ensure text is valid
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
            
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        
        # Truncate if needed
        max_length = 2048
        text = text[:max_length].strip()
            
        logger.info(f"Requesting embedding for text (length: {len(text)})")
        
        # Make the embedding request
        response = session.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": OLLAMA_MODEL,
                "prompt": text,
                "options": {
                    "temperature": 0
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise ValueError(f"Ollama API error: {response.status_code}")
            
        response_data = response.json()
        logger.debug(f"Ollama response: {response_data}")
        
        # Extract embedding from response
        if 'embedding' in response_data:
            embedding = response_data['embedding']
        elif 'embeddings' in response_data and response_data['embeddings']:
            embedding = response_data['embeddings'][0]
        else:
            logger.error(f"Unexpected response format: {response_data}")
            raise ValueError(f"Unexpected response format from Ollama: {response_data}")
            
        if not embedding or not isinstance(embedding, list):
            raise ValueError(f"Invalid embedding format: {embedding}")
            
        embedding_array = np.array(embedding, dtype=np.float32)
        logger.info(f"Successfully generated embedding of shape: {embedding_array.shape}")
        
        return embedding_array
        
    except Exception as e:
        logger.error(f"Error in get_ollama_embedding: {str(e)}")
        logger.exception("Full traceback:")
        raise
