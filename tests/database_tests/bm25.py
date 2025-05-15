#!/usr/bin/env python3
"""
BM25 API Test - Updated to match actual API endpoints from Swagger docs

This script tests the search functionality through the actual API endpoints
as defined in the project's Swagger documentation.
"""

import requests
import json
import time
import logging
import os
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bm25_api_test")

class BM25APITester:
    """Test the BM25 search functionality via the API."""
    
    def __init__(self, base_url: str = None):
        """Initialize with the base URL of the API."""
        if base_url is None:
            # Use environment variables for port if available
            port = os.getenv("APP_PORT", "8000")
            host = os.getenv("APP_HOST", "localhost")
            base_url = f"http://{host}:{port}"
            
        self.base_url = base_url
        logger.info(f"Initializing API tester with base URL: {base_url}")
        
        # Common test queries
        self.test_queries = [
            {"query": "Python programming language", "description": "Exact match query"},
            {"query": "database management systems", "description": "Technical term query"},
            {"query": "machine learning algorithms", "description": "Academic query"},
            {"query": "climate change technologies", "description": "Environmental topic query"},
            {"query": "smartphone with good camera", "description": "Feature-based query"}
        ]
        
    def verify_server_up(self):
        """Verify that the server is up and running."""
        try:
            response = requests.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                logger.info("✅ Server is up and running with Swagger docs available")
                return True
            else:
                # Try the root endpoint which should redirect to docs
                response = requests.get(self.base_url)
                if response.status_code == 200:
                    logger.info("✅ Server is up and running (root endpoint)")
                    return True
                else:
                    logger.error(f"❌ Server response: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to server: {e}")
            return False
            
    def test_search(self):
        """Test search using the /api/query/ endpoint."""
        logger.info("\n=== Testing Search with /api/query/ endpoint ===")
        
        results = []
        
        for test_case in self.test_queries:
            query = test_case["query"]
            description = test_case["description"]
            
            logger.info(f"Testing query: '{query}' - {description}")
            
            # The actual endpoint is /api/query/ according to Swagger
            try:
                url = f"{self.base_url}/api/query/"
                logger.info(f"Sending GET request to: {url}?query={query}")
                
                start_time = time.time()
                response = requests.get(url, params={"query": query})
                elapsed_time = time.time() - start_time
                
                logger.info(f"Response status: {response.status_code}")
                if response.status_code == 200:
                    try:
                        # The response should be a string (text content)
                        result_text = response.text
                        logger.info("✅ Success! Got search results")
                        logger.info(f"Response length: {len(result_text)} characters")
                        logger.info(f"Response preview: {result_text[:150]}...")
                        
                        results.append({
                            "query": query,
                            "description": description,
                            "status": "success",
                            "time": elapsed_time,
                            "response_length": len(result_text),
                            "response_preview": result_text[:150]
                        })
                    except Exception as e:
                        logger.warning(f"Error processing response: {e}")
                        results.append({
                            "query": query,
                            "description": description,
                            "status": "error_processing",
                            "time": elapsed_time,
                            "error": str(e)
                        })
                else:
                    logger.error(f"❌ Error: {response.status_code} - {response.text}")
                    results.append({
                        "query": query,
                        "description": description,
                        "status": "error",
                        "status_code": response.status_code,
                        "error_message": response.text
                    })
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                results.append({
                    "query": query,
                    "description": description,
                    "status": "exception",
                    "error": str(e)
                })
        
        return results
    
    def test_chat(self):
        """Test chat using the /api/chat/request endpoint (non-streaming)."""
        logger.info("\n=== Testing Chat with /api/chat/request endpoint ===")
        
        # Prepare chat queries that would trigger search
        chat_queries = [
            "Find information about Python programming",
            "What can you tell me about database management?",
            "Compare different machine learning algorithms",
            "Explain the impact of climate change on technology",
            "What are the latest smartphone camera technologies?"
        ]
        
        results = []
        
        for query in chat_queries:
            logger.info(f"Testing chat message: '{query}'")
            
            # Format exactly like the Swagger example
            chat_data = {
                "messages": [
                    {
                        "content": query,
                        "role": "user"
                    }
                ]
            }
            
            try:
                # Use the non-streaming endpoint for simpler testing
                url = f"{self.base_url}/api/chat/request"
                logger.info(f"Sending POST request to: {url}")
                
                start_time = time.time()
                response = requests.post(url, json=chat_data)
                elapsed_time = time.time() - start_time
                
                logger.info(f"Response status: {response.status_code}")
                if response.status_code == 200:
                    try:
                        result = response.json()
                        content = result.get("result", {}).get("content", "")
                        
                        logger.info("✅ Success! Got chat response")
                        logger.info(f"Response length: {len(content)} characters")
                        logger.info(f"Response preview: {content[:150]}...")
                        
                        # Check if there are source nodes (indicating search was used)
                        nodes = result.get("nodes", [])
                        logger.info(f"Source nodes: {len(nodes)}")
                        
                        results.append({
                            "query": query,
                            "status": "success",
                            "time": elapsed_time,
                            "response_length": len(content),
                            "response_preview": content[:150],
                            "has_sources": len(nodes) > 0,
                            "source_count": len(nodes)
                        })
                        
                        # Log first source node if available
                        if nodes:
                            logger.info(f"First source: {nodes[0].get('text', '')[:100]}...")
                            
                    except Exception as e:
                        logger.warning(f"Error processing response: {e}")
                        results.append({
                            "query": query,
                            "status": "error_processing",
                            "time": elapsed_time,
                            "error": str(e)
                        })
                else:
                    logger.error(f"❌ Error: {response.status_code} - {response.text}")
                    results.append({
                        "query": query,
                        "status": "error",
                        "status_code": response.status_code,
                        "error_message": response.text
                    })
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                results.append({
                    "query": query,
                    "status": "exception",
                    "error": str(e)
                })
        
        return results
    
    def print_summary(self, search_results, chat_results):
        """Print a summary of the test results."""
        logger.info("\n" + "="*80)
        logger.info("BM25 API TEST SUMMARY")
        logger.info("="*80)
        
        # Search API Summary
        logger.info("\nSearch API Results:")
        successful_searches = sum(1 for r in search_results if r["status"] == "success")
        logger.info(f"Total search queries: {len(search_results)}")
        logger.info(f"Successful: {successful_searches} ({successful_searches/len(search_results)*100:.1f}%)")
        
        if successful_searches > 0:
            avg_search_time = sum(r["time"] for r in search_results if r["status"] == "success") / successful_searches
            avg_response_length = sum(r["response_length"] for r in search_results if r["status"] == "success") / successful_searches
            logger.info(f"Average response time: {avg_search_time:.2f} seconds")
            logger.info(f"Average response length: {avg_response_length:.1f} characters")
        
        # Chat API Summary
        logger.info("\nChat API Results:")
        successful_chats = sum(1 for r in chat_results if r["status"] == "success")
        logger.info(f"Total chat queries: {len(chat_results)}")
        logger.info(f"Successful: {successful_chats} ({successful_chats/len(chat_results)*100:.1f}%)")
        
        if successful_chats > 0:
            avg_chat_time = sum(r["time"] for r in chat_results if r["status"] == "success") / successful_chats
            avg_chat_length = sum(r["response_length"] for r in chat_results if r["status"] == "success") / successful_chats
            logger.info(f"Average response time: {avg_chat_time:.2f} seconds")
            logger.info(f"Average response length: {avg_chat_length:.1f} characters")
            
            # Source utilization
            chats_with_sources = sum(1 for r in chat_results if r.get("has_sources", False))
            avg_sources = sum(r.get("source_count", 0) for r in chat_results) / len(chat_results)
            logger.info(f"Chats with sources: {chats_with_sources}/{len(chat_results)} ({chats_with_sources/len(chat_results)*100:.1f}%)")
            logger.info(f"Average sources per chat: {avg_sources:.1f}")
        
        # Identify most responsive queries
        if successful_searches > 0:
            logger.info("\nMost responsive search queries:")
            sorted_searches = sorted(search_results, key=lambda r: r.get("response_length", 0) if r["status"] == "success" else 0, reverse=True)
            for i, result in enumerate(sorted_searches[:3]):
                if result["status"] == "success":
                    logger.info(f"  {i+1}. '{result['query']}' - {result['response_length']} chars in {result['time']:.2f}s")
        
        if successful_chats > 0:
            logger.info("\nMost informative chat queries:")
            sorted_chats = sorted(chat_results, key=lambda r: r.get("source_count", 0) if r["status"] == "success" else 0, reverse=True)
            for i, result in enumerate(sorted_chats[:3]):
                if result["status"] == "success":
                    logger.info(f"  {i+1}. '{result['query']}' - {result.get('source_count', 0)} sources in {result['time']:.2f}s")
        
        # Overall impression
        logger.info("\nOverall Assessment:")
        if successful_searches == len(search_results) and successful_chats == len(chat_results):
            logger.info("✅ All tests passed successfully!")
        else:
            logger.info("⚠️ Some tests failed. Review the logs for details.")


def main():
    """Run the BM25 API test suite."""
    logger.info("Starting BM25 API test suite")
    
    # Check if PG_CONNECTION_STRING is set
    if not os.getenv("PG_CONNECTION_STRING"):
        logger.warning("PG_CONNECTION_STRING environment variable not set. Database connection may fail.")
    
    try:
        # Initialize test suite
        test_suite = BM25APITester()
        
        # Verify server is running
        if not test_suite.verify_server_up():
            logger.error("Cannot proceed with tests - server is not responding")
            return
        
        # Run search tests
        search_results = test_suite.test_search()
        
        # Run chat tests
        chat_results = test_suite.test_chat()
        
        # Print summary
        test_suite.print_summary(search_results, chat_results)
        
        logger.info("Test suite completed successfully")
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()