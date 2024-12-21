import pytest
from fastapi.testclient import TestClient
import logging
import json
from unittest.mock import AsyncMock, patch
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add a stream handler to print to console
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Import after logging config
from app.main import app
from app.db.vector_store import init_db

@pytest.fixture(scope="session")
async def setup_test_db():
    logger.info("Initializing test database...")
    await init_db()
    yield
    logger.info("Cleaning up test database...")

@pytest.mark.asyncio
async def test_debug_reasoning_integration(setup_test_db):
    client = TestClient(app)
    
    # Test payload
    payload = {
        "query": "what main themes are presented here",
        "file_pattern": "*.pdf",
        "chunk_limit": 3,
        "chunk_size": 770,
        "context_window": 308,
        "num_predict": 64,
        "timeout": 500.0
    }
    
    logger.info(f"Sending request with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = client.post("/debug/reasoning", json=payload)
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")
        else:
            data = response.json()
            logger.info(f"Response data: {json.dumps(data, indent=2)}")
            
            # Log specific response parts
            logger.debug(f"Chunks used: {data.get('chunks_used', 'N/A')}")
            logger.debug(f"Total context length: {data.get('total_context_length', 'N/A')}")
            
            if 'chunks' in data:
                for i, chunk in enumerate(data['chunks'], 1):
                    logger.debug(f"Chunk {i} - File: {chunk.get('file_name')}")
                    logger.debug(f"Chunk {i} - Similarity: {chunk.get('similarity')}")
                    logger.debug(f"Chunk {i} - Snippet preview: {chunk.get('snippet')[:100]}...")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert "chunks" in data
        assert len(data["chunks"]) <= payload["chunk_limit"]
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    pytest.main(["-v", "--log-cli-level=DEBUG"])