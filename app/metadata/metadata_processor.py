import os
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
from app.utils.helpers import get_ollama_embedding
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
import numpy as np

import asyncio
from app.utils.relationship_extractor import RelationshipExtractor

logger = logging.getLogger(__name__)

class MetadataProcessor:
    def __init__(self, pg_storage=None):
        self.processed_files = set()
        self.pg_storage = pg_storage
        self.relationship_extractor = RelationshipExtractor()

    async def _read_file_content(self, file_path: str) -> str:
        """Read content from a file"""
        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            docs = reader.load_data()
            return " ".join(doc.text for doc in docs)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and extract its metadata"""
        try:
            # Read file content
            content = await self._read_file_content(file_path)
            if not content:
                raise ValueError(f"No content found in file {file_path}")

            # Extract relationships using the RelationshipExtractor
            relationships = await self.relationship_extractor.extract_relationships(text=content)

            # Create metadata object
            metadata = {
                "resource_id": os.path.basename(file_path),
                "relationships": relationships,
                "key_concepts": [],
                "summary": ""
            }

            return {
                "file_name": os.path.basename(file_path),
                "relationships": metadata,
                "content_preview": content[:100] if content else ""
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file_name": os.path.basename(file_path),
                "relationships": {
                    "resource_id": os.path.basename(file_path),
                    "relationships": [],
                    "key_concepts": [],
                    "summary": ""
                },
                "content_preview": ""
            }

    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all files in a directory"""
        results = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path not in self.processed_files:
                        result = await self.process_file(file_path)
                        results.append(result)
                        self.processed_files.add(file_path)
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
        return results

    def extract_metadata(self, directory_path: str) -> List[Dict[str, Any]]:
        """Extract metadata for all processed files in a directory"""
        metadata_list = []
        for file_path in self.processed_files:
            if file_path.startswith(directory_path):
                metadata_list.append({
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path
                })
        return metadata_list
