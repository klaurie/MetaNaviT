"""
PPTX Document Loader Module

Handles loading and processing of PPTX files. Extracts text content, speaker notes, 
and metadata from the slides. 

Carlana is on this
"""

import os
import logging
from typing import List, Dict, Any
from unstructured.partition.pptx import partition_pptx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PPTXLoaderConfig(BaseModel):
    """
    Configuration model for PPTX loading parameters.
    Can be extended in the future if needed.
    """
    directory: str  # Directory containing PPTX files to load

def extract_pptx_content(file_path: str) -> Dict[str, Any]:
    """
    Extracts text, speaker notes, and metadata from a PPTX file.

    """
    try:
        # Use unstructured to extract content and normalize it to JSON
        elements = partition_pptx(file_path)
        
        # Combine elements into a JSON-like structure
        content = {
            "file_name": os.path.basename(file_path),
            "slides": [
                {
                    "text": element.text,
                    "metadata": element.metadata.to_dict() if element.metadata else {}
                }
                for element in elements
            ]
        }

        logger.info(f"Successfully extracted content from {file_path}")
        return content

    except Exception as e:
        logger.error(f"Failed to extract content from {file_path}: {e}")
        raise

def load_pptx_documents(config: PPTXLoaderConfig) -> List[Dict[str, Any]]:
    """
    Loads all PPTX files in a given directory and extracts their content.

    A
    """
    documents = []

    try:
        for root, _, files in os.walk(config.directory):
            for file in files:
                if file.endswith(".pptx"):
                    file_path = os.path.join(root, file)
                    documents.append(extract_pptx_content(file_path))

        logger.info(f"Loaded {len(documents)} PPTX documents from {config.directory}")
        return documents

    except Exception as e:
        logger.error(f"Error loading PPTX documents: {e}")
        raise
