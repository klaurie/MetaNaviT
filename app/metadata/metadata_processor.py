import os
from datetime import datetime
import logging
from typing import List, Dict, Any
from app.utils.helpers import get_ollama_embedding  # Import your existing embedding function

logger = logging.getLogger(__name__)

class MetadataProcessor:
    def __init__(self, text_splitter, vector_store):
        self.text_splitter = text_splitter
        self.vector_store = vector_store

    def extract_metadata(self, directory_path: str) -> List[Dict[str, Any]]:
        """Extract metadata from all files in a directory"""
        parsed_files = []
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                metadata = {
                    "file_name": filename,
                    "file_type": os.path.splitext(filename)[1],
                    "file_size": os.path.getsize(file_path),
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    "file_path": file_path
                }
                parsed_files.append(metadata)
        return parsed_files

    async def process_directory(self, directory_path: str) -> None:
        """Process all supported files in a directory"""
        supported_extensions = {'.txt', '.pdf', '.md', '.doc', '.docx'}
        
        # First extract metadata for all files
        metadata_list = self.extract_metadata(directory_path)
        
        # Then process each supported file
        for metadata in metadata_list:
            if any(metadata["file_type"].lower() == ext for ext in supported_extensions):
                try:
                    with open(metadata["file_path"], 'r', encoding='utf-8') as f:
                        content = f.read()
                    await self.process_document(metadata, content)
                except Exception as e:
                    logger.error(f"Error processing {metadata['file_path']}: {e}")

    async def process_document(self, metadata: Dict[str, Any], content: str) -> None:
        """Process a document and store its chunks with metadata"""
        chunks = self.text_splitter.split_text(content)
        
        # Enhance metadata with processing timestamp
        metadata.update({
            "processed_at": datetime.now().isoformat(),
        })
        
        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            
            await self.vector_store.add_chunk(
                snippet=chunk,
                embedding=embedding,
                metadata=metadata
            )

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama"""
        return await get_ollama_embedding(text)
