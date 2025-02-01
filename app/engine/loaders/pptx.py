"""
PPTX Document Loader Module

Handles loading and processing of PPTX files. Extracts text content, speaker notes, 
and metadata from the slides. 

Carlana is on this.
"""

import os
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import partition_pptx
try:
    from unstructured.partition.pptx import partition_pptx
except ImportError:
    logger.error("Failed to import partition_pptx. Ensure 'unstructured' is installed.")
    raise


class PPTXLoaderConfig(BaseModel):
    """
    Configuration model for PPTX loading parameters.
    Can be extended in the future if needed.
    """
    directory: str  # Directory containing PPTX files to load


def extract_pptx_content(file_path: str) -> Dict[str, Any]:
    """
    Extracts text, speaker notes, and metadata from a PPTX file.
    Ensures a structured JSON format.
    """
    try:
        elements = partition_pptx(file_path)

        slides = []
        slide_index = 0  # Track slide numbers manually (if needed)

        for element in elements:
            slide_data = {
                "slide_number": slide_index + 1,  # Assign slide numbers
                "text": element.text.strip() if element.text else "",
                "metadata": element.metadata.to_dict() if element.metadata else {},
                "speaker_notes": None  # Default value
            }

            # Check if the element contains speaker notes
            if hasattr(element, "category") and element.category == "Slide Notes":
                slide_data["speaker_notes"] = element.text.strip()

            slides.append(slide_data)
            slide_index += 1  # Increment slide count

        content = {
            "file_name": os.path.basename(file_path),
            "slides": slides,
        }

        logger.info(f"Successfully extracted content from {file_path}")
        return content

    except Exception as e:
        logger.error(f"Failed to extract content from {file_path}: {e}")
        raise


def load_pptx_documents(config: PPTXLoaderConfig) -> List[Dict[str, Any]]:
    """
    Loads all PPTX files in a given directory and extracts their content.
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
