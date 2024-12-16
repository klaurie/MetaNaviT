#!/bin/bash

# Base URL for the API
BASE_URL="http://localhost:8001"

# Function to check API health
check_health() {
    echo "Checking API health..."
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL)
    if [ "$RESPONSE" -eq 200 ]; then
        echo "API is running successfully!"
    else
        echo "Failed to connect to API. HTTP Status Code: $RESPONSE"
        exit 1
    fi
}

# Function to index a document
index_document() {
    echo "Indexing a sample document..."
    RESPONSE=$(curl -s -X POST $BASE_URL/index/ \
        -H "Content-Type: application/json" \
        -d '{
            "text": "This is a test document for indexing.",
            "directory": "test_dir"
        }')
    echo "Response from /index/: $RESPONSE"
}

# Function to query the index
query_index() {
    echo "Querying the index for 'test document'..."
    RESPONSE=$(curl -s -X POST $BASE_URL/query/ \
        -H "Content-Type: application/json" \
        -d '{
            "query": "test document",
            "directory": "test_dir"
        }')
    echo "Response from /query/: $RESPONSE"
}

# Run the tests
check_health
index_document
query_index
