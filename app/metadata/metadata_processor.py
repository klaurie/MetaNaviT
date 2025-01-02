from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor, 
    TitleExtractor,
    KeywordExtractor
)
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, MetadataMode
from typing import List, Dict, Any, Optional
from llama_index.llms.ollama import Ollama
import logging

import os
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class CustomTextNode(TextNode):
    """Custom node class with enhanced metadata handling"""
    
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Enhanced content representation including metadata"""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        base_content = self.text
        
        if metadata_mode != MetadataMode.NONE and metadata_str:
            return f"{base_content}\nMetadata: {metadata_str}"
        return base_content

class MetadataProcessor:
    def __init__(self):
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                model="llama3.2:1b",  # or any other model you have pulled
                request_timeout=120.0,
                temperature=0.1,
                base_url="http://ollama:11434"
            )
        # Initialize extractors with Ollama LLM
            self.extractors = [
                TitleExtractor(nodes=3, llm=self.llm),
                KeywordExtractor(keywords=10, llm=self.llm),
                SummaryExtractor(summaries=["self"], llm=self.llm),
                QuestionsAnsweredExtractor(questions=3, llm=self.llm)
            ]
        except Exception as e:
            logger.warning(f"Failed to initialize LLM extractors: {e}")
            # Initialize without extractors if LLM fails
            self.extractors = []
    async def process_file(self, file_path: str) -> tuple[Dict[str, Any], str]:
        """Process a file and extract metadata"""
        try:
            # Use SimpleDirectoryReader for loading
            reader = SimpleDirectoryReader(input_files=[file_path])
            docs = reader.load_data()
            
            # Extract base metadata
            base_metadata = {
                "file_name": os.path.basename(file_path),
                "directory": os.path.dirname(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "file_extension": os.path.splitext(file_path)[1].lower(),
                "content_type": "application/pdf" if file_path.lower().endswith('.pdf') else "text/plain",
                "doc_id": str(uuid.uuid4())
            }
            
              # Extract additional metadata using extractors
            for doc in docs:
                extracted = {}
                for extractor in self.extractors:
                    try:
                        metadata = extractor.extract([doc])
                        if metadata and metadata[0]:
                            extractor_name = extractor.__class__.__name__.replace('Extractor', '').lower()
                            extracted[extractor_name] = metadata[0]
                            logger.info(f"Extracted {extractor_name}: {metadata[0]}")
                    except Exception as e:
                        logger.warning(f"Extractor {extractor.__class__.__name__} failed: {e}")
                
                base_metadata["extracted_metadata"] = extracted
            
            return base_metadata, docs[0].text

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    async def process_chunks(self, chunks: List[str], base_metadata: Dict[str, Any]) -> List[CustomTextNode]:
        """Process chunks into nodes with relationships"""
        nodes = []
        
        for i, chunk in enumerate(chunks):
            # Create node with enhanced metadata
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            node = CustomTextNode(
                text=chunk,
                node_id=str(uuid.uuid4()),
                metadata=chunk_metadata
            )
            
            # Set relationships with previous and next nodes
            if i > 0:
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=nodes[i-1].node_id
                )
                nodes[i-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=node.node_id
                )
                
            nodes.append(node)
            
        return nodes