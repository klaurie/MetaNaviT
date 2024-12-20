import pytest
import aiohttp
import asyncio
import json
from datetime import datetime

pytestmark = pytest.mark.asyncio

async def make_request(endpoint, method="POST", payload=None):
    """Make async HTTP request and return response with detailed output"""
    url = f"http://localhost:8001{endpoint}"
    
    print(f"\n{'='*80}")
    print(f"Testing: {endpoint}")
    print(f"Method: {method}")
    if payload:
        print("\nPayload:")
        print(json.dumps(payload, indent=2))

    async with aiohttp.ClientSession() as session:
        try:
            if method == "POST":
                async with session.post(url, json=payload) as response:
                    status = response.status
                    response_json = await response.json()
            else:
                async with session.get(url) as response:
                    status = response.status
                    response_json = await response.json()

            print(f"\nResponse Status: {status}")
            print("Response Body:")
            print(json.dumps(response_json, indent=2))
            
            return status, response_json
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

async def test_rag_endpoints():
    """Test all RAG endpoints with detailed output"""
    
    print(f"\nStarting tests at {datetime.now()}")

    # Test 1: Compare documents (aggregate)
    print("\nTest 1: Compare Documents (Aggregate)")
    payload1 = {
        "query": "Compare these documents",
        "files": [
            "/app/app/uploaded_files/paul_graham/paul_graham_essay.txt",
            "/app/app/uploaded_files/Free Tuition Required Fees and Textbooks.pdf",
            "/app/app/uploaded_files/paul_graham/Final-Prep.docx"
        ],
        "analysis_type": "aggregate"
    }
    status1, response1 = await make_request("/ollama/reasoning", payload=payload1)
    assert status1 == 200

    # Test 2: Compare documents (individual)
    print("\nTest 2: Compare Documents (Individual)")
    payload2 = {
        "query": "Compare these documents",
        "files": [
            "/app/app/uploaded_files/paul_graham/paul_graham_essay.txt",
            "/app/app/uploaded_files/Free Tuition Required Fees and Textbooks.pdf",
            "/app/app/uploaded_files/paul_graham/Final-Prep.docx"
        ],
        "analysis_type": "individual"
    }
    status2, response2 = await make_request("/ollama/reasoning", payload=payload2)
    assert status2 == 200

    # Test 3: Summarize PDF
    print("\nTest 3: Summarize PDF")
    payload3 = {
        "query": "summerize the PDF content",
        "file_pattern": "*.pdf",
        "analysis_type": "aggregate"
    }
    status3, response3 = await make_request("/ollama/reasoning", payload=payload3)
    assert status3 == 200

    # Test 4: AI MetaNavit relation
    print("\nTest 4: AI MetaNavit Relation")
    payload4 = {
        "query": "how is ai related with MetaNavit",
        "path": "/app/app/uploaded_files/paul_graham",
        "analysis_type": "individual"
    }
    status4, response4 = await make_request("/ollama/reasoning", payload=payload4)
    assert status4 == 200

    # Test 5: RAG History
    print("\nTest 5: RAG History")
    status5, response5 = await make_request("/rag/history", method="GET")
    assert status5 == 200

    print(f"\nAll tests completed at {datetime.now()}")

def pytest_configure(config):
    """Configure pytest-asyncio to use function scope for event loops"""
    config.option.asyncio_default_fixture_loop_scope = "function"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])