import requests
import json
import time
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8001"

def test_upload_batch() -> Dict[str, Any]:
    """
    Test the upload_batch endpoint
    """
    endpoint = f"{BASE_URL}/upload_batch/"
    payload = {
        "directory": "./app/uploaded_files"
    }
    
    logger.info("Testing upload_batch endpoint...")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Upload batch successful!")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response time: {response.elapsed.total_seconds():.2f} seconds")
        logger.info(f"Summary: {json.dumps(result.get('summary', {}), indent=2)}")
        logger.info(f"Full response: {json.dumps(result, indent=2)}")
        
        # Print processed files details
        processed_files = result.get('processed_files', [])
        logger.info(f"\nProcessed {len(processed_files)} files:")
        for file_info in processed_files:
            logger.info(f"\nFile: {file_info.get('file_name')}")
            logger.info(f"Directory: {file_info.get('directory')}")
            logger.info(f"Content type: {file_info.get('content_type')}")
            logger.info(f"Chunks processed: {len(file_info.get('chunks', []))}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error testing upload_batch endpoint: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Error response: {e.response.text}")
        raise

def test_compare() -> Dict[str, Any]:
    """
    Test the compare endpoint
    """
    endpoint = f"{BASE_URL}/compare"
    payload = {
        "source1": "paul_graham/paul_graham_essay.txt",
        "source2": "Free Tuition Required Fees and Textbooks.pdf"
    }
    
    logger.info("\nTesting compare endpoint...")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Compare analysis successful!")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response time: {response.elapsed.total_seconds():.2f} seconds")
        
        # Print statistics
        stats = result.get('statistics', {})
        logger.info("\nStatistics:")
        logger.info(f"Source1 type: {stats.get('source1_type')}")
        logger.info(f"Source2 type: {stats.get('source2_type')}")
        logger.info(f"Average similarity: {stats.get('avg_similarity')}")
        logger.info(f"Similar chunk pairs: {stats.get('similar_chunk_pairs')}")
        
        # Print similar chunks
        similar_chunks = result.get('similar_chunks', [])
        logger.info(f"\nFound {len(similar_chunks)} similar chunk pairs:")
        for i, chunk in enumerate(similar_chunks[:3], 1):  # Show first 3 chunks
            logger.info(f"\nSimilar Chunk Pair {i}:")
            logger.info(f"Similarity score: {chunk.get('similarity')}")
            logger.info(f"Source1 text: {chunk.get('chunk1_text')[:100]}...")
            logger.info(f"Source2 text: {chunk.get('chunk2_text')[:100]}...")
        
        # Print clustering info
        clustering = result.get('clustering_analysis', {})
        logger.info(f"\nClustering Analysis:")
        logger.info(f"Number of clusters: {clustering.get('n_clusters')}")
        logger.info(f"Plot saved to: {clustering.get('plot_path')}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error testing compare endpoint: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Error response: {e.response.text}")
        raise

def main():
    """
    Run all tests
    """
    try:
        # Test upload_batch first
        logger.info("Starting endpoint tests...")
        upload_result = test_upload_batch()
        
        # Wait a bit for processing to complete
        logger.info("\nWaiting 5 seconds before running compare test...")
        time.sleep(5)
        
        # Then test compare
        compare_result = test_compare()
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 