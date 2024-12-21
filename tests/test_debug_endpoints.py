import pytest
import httpx
import json
import asyncio

# Base URL of your running server
BASE_URL = "http://localhost:8001"  # or your server URL

# Configure timeouts
TIMEOUT = httpx.Timeout(
    connect=5.0,    # connection timeout
    read=30.0,      # read timeout
    write=5.0,      # write timeout
    pool=1.0        # pool timeout
)

@pytest.mark.asyncio
async def test_database_init():
    """Test database initialization"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{BASE_URL}/debug/database/init")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_insert_vector():
    """Test vector insertion"""
    test_data = {
        "document_chunk": "This is a test document",
        "metadata": {
            "source": "test",
            "type": "debug"
        }
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/debug/database/insert-vector",
            json=test_data
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_similarity_search():
    """Test similarity search"""
    test_data = {
        "query": "test query",
        "directory_scope": "test_directory"
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/debug/query/similarity-search",
            json=test_data
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_debug_reasoning():
    """Test reasoning endpoint"""
    test_data = {
        "query": "What is the meaning of life?",
        "file_pattern": "*.txt",
        "chunk_limit": 2,
        "chunk_size": 500,
        "context_window": 1024,
        "num_predict": 256,
        "timeout": 30.0
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/debug/reasoning",
            json=test_data
        )
        assert response.status_code == 200

if __name__ == "__main__":
    asyncio.run(pytest.main(["-v"]))