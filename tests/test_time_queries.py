"""Tests for LLM-based time queries using synthetic dataset"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from scripts.generate_time_data import TimeBasedDataGenerator, TEST_DATA_DIR
from llama_index import VectorStoreIndex
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

class TestTimeBasedQueries:
    """Test suite for time-based LLM queries"""
    
    def setup_method(self):
        """Ensure test data exists before each test"""
        if not TEST_DATA_DIR.exists():
            generator = TimeBasedDataGenerator(TEST_DATA_DIR)
            generator.generate_all()
    
    def test_creation_time_query(self):
        """Test querying file creation times"""
        index = VectorStoreIndex.from_documents(TEST_DATA_DIR.glob("*.txt"))
        query = "When was requirements_v1.txt created?"
        
        response = index.query(query)
        
        print(f"Query: {query}")
        print(f"Response: {response.response}")
        
        expected_response = "requirements_v1.txt was created on March 1st, 2024"
        metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(query=query, expected_response=expected_response)
        assert_test(response.response, test_case, metric=metric)

    def test_time_range_query(self):
        """Test querying files within a time range"""
        index = VectorStoreIndex.from_documents(TEST_DATA_DIR.glob("*.txt"))
        query = "What files were modified between March 1st and March 3rd?"
        
        response = index.query(query)
        
        print(f"Query: {query}")
        print(f"Response: {response.response}")
        
        expected_response = "Files modified between March 1-3: requirements_v1.txt, requirements_v2.txt"
        metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(query=query, expected_response=expected_response)
        assert_test(response.response, test_case, metric=metric)

    def test_temporal_relationship_query(self):
        """Test querying temporal relationships between files"""
        index = VectorStoreIndex.from_documents(TEST_DATA_DIR.glob("*.txt"))
        query = "What is the time relationship between requirements_v1.txt and requirements_v2.txt?"
        
        response = index.query(query)
        
        print(f"Query: {query}")
        print(f"Response: {response.response}")
        
        expected_response = "requirements_v2.txt was created 2 days after requirements_v1.txt"
        metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(query=query, expected_response=expected_response)
        assert_test(response.response, test_case, metric=metric)

    def test_latest_file_query(self):
        """Test querying for most recent files"""
        index = VectorStoreIndex.from_documents(TEST_DATA_DIR.glob("*.txt"))
        query = "What is the most recent meeting notes file?"
        
        response = index.query(query)
        
        print(f"Query: {query}")
        print(f"Response: {response.response}")
        
        expected_response = "The most recent meeting notes are from meeting_2024Q1_02.txt"
        metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(query=query, expected_response=expected_response)
        assert_test(response.response, test_case, metric=metric)

    def test_file_sequence_query(self):
        """Test querying sequential file relationships"""
        index = VectorStoreIndex.from_documents(TEST_DATA_DIR.glob("*.txt"))
        query = "What is the chronological order of project files?"
        
        response = index.query(query)
        
        print(f"Query: {query}")
        print(f"Response: {response.response}")
        
        expected_response = (
            "The project files were created in this order: requirements_v1.txt, "
            "requirements_v2.txt, design_doc.txt, implementation_notes.txt, review_feedback.txt"
        )
        metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(query=query, expected_response=expected_response)
        assert_test(response.response, test_case, metric=metric)