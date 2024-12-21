from typing import Dict, List, Any
import logging
from app.utils.helpers import get_ollama_embedding, extract_relationships_from_text
import asyncio
from itertools import islice
from dataclasses import dataclass
from enum import Enum
import json
from app.config import OLLAMA_HOST
import random

logger = logging.getLogger(__name__)

class RelationType(Enum): 
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    USES = "uses"
    REFERENCES = "references"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"

@dataclass
class Relationship:
    source: str
    target: str
    type: RelationType
    strength: float  # 0.0 to 1.0
    description: str = ""

class RelationshipExtractor:
    def __init__(self, chunk_size: int = 800, max_concurrent: int = 3):
        self.chunk_size = chunk_size
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _split_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = ""
        sentences = text.split(". ")
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    async def _process_chunk(self, chunk: str, chunk_id: int) -> List[Dict[str, Any]]:
        async with self._semaphore:
            try:
                return await extract_relationships_from_text(chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                return []

    async def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        chunks = self._split_text(text)
        tasks = []
        
        for i, chunk in enumerate(chunks):
            task = asyncio.create_task(self._process_chunk(chunk, i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        relationships = []
        
        for result in results:
            if isinstance(result, list):
                relationships.extend(result)
        
        return relationships

async def analyze_relationships(
    content: str,
    chunk_id: str = None,
    resource_type: str = None
) -> Dict[str, Any]:
    """Process a single chunk of content"""
    try:
        # Preprocess content
        content_truncated = content[:2000] if len(content) > 2000 else content
        
        # Add context based on resource type
        context = f"Resource type: {resource_type}\n" if resource_type else ""
        
        response = await get_ollama_relationships(content_truncated, chunk_id)
        
        # Enhanced JSON parsing with validation
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
            else:
                parsed = response
            
            # Validate and normalize relationships
            relationships = []
            for rel in parsed.get("relationships", []):
                if all(k in rel for k in ["source", "target", "type"]):
                    # Normalize relationship type
                    rel_type = rel["type"].lower()
                    if rel_type in RelationType.__members__:
                        relationships.append({
                            "source": rel["source"],
                            "target": rel["target"],
                            "type": rel_type,
                            "strength": float(rel.get("strength", 0.8)),
                            "description": rel.get("description", "")
                        })
            
            return {
                "chunk_id": chunk_id,
                "relationships": relationships,
                "key_concepts": list(set(parsed.get("key_concepts", []))),
                "summary": parsed.get("summary", "").strip(),
                "metadata": {
                    "resource_type": resource_type,
                    "relationship_count": len(relationships)
                }
            }
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for chunk {chunk_id}")
            return create_empty_result(chunk_id, resource_type)
            
    except Exception as e:
        logger.error(f"Error in relationship analysis: {str(e)}")
        return create_empty_result(chunk_id, resource_type)

def create_empty_result(chunk_id: str, resource_type: str) -> Dict[str, Any]:
    """Create empty result structure"""
    return {
        "chunk_id": chunk_id,
        "relationships": [],
        "key_concepts": [],
        "summary": "",
        "metadata": {
            "resource_type": resource_type,
            "relationship_count": 0,
            "error": True
        }
    }

async def batch_process_relationships(
    chunks: List[Dict[str, str]],
    batch_size: int = 3
) -> List[Dict[str, Any]]:
    """Process multiple chunks efficiently"""
    results = []
    semaphore = asyncio.Semaphore(batch_size)
    
    async def process_with_limit(chunk):
        async with semaphore:
            result = await analyze_relationships(
                chunk["content"],
                chunk.get("id"),
                chunk.get("type")
            )
            await asyncio.sleep(0.5)  # Small delay between requests
            return result
    
    tasks = [process_with_limit(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)