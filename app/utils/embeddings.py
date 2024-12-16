from typing import List, Optional, Union
import numpy as np
from httpx import AsyncClient
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseEmbeddings(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        pass

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts"""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in a text"""
        pass

class NomicEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "nomic-embed-text", batch_size: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self._dim = 768  # nomic-embed-text dimension
        self.client = AsyncClient(timeout=120.0)  # Increased timeout

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text using Ollama"""
        try:
            embeddings = await self.embed_batch([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts using Ollama"""
        try:
            # Process in smaller batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                # Process each text individually
                for text in batch:
                    try:
                        # Clean and truncate text
                        text = text.replace('\n', ' ').strip()
                        if len(text) > 2048:  # Limit text length
                            text = text[:2048]
                            
                        response = await self.client.post(
                            "http://ollama:11434/api/embeddings",
                            json={
                                "model": self.model_name,
                                "prompt": text
                            },
                            timeout=120.0  # Increased timeout
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        if "embedding" in data:
                            all_embeddings.append(np.array(data["embedding"]))
                        else:
                            logger.error(f"No embedding in response: {data}")
                            raise ValueError("No embedding in response")
                            
                    except Exception as e:
                        logger.error(f"Error processing text in batch: {str(e)}")
                        raise
                        
            return all_embeddings
        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            raise

    async def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text.split())  # Simple approximation

class SemanticChunk:
    def __init__(self, text: str, embedding: Optional[np.ndarray] = None):
        self.text = text
        self.embedding = embedding
        self.token_count = len(text.split())  # Simple approximation

class SDPMChunker:
    def __init__(
        self,
        embeddings: NomicEmbeddings,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        skip_window: int = 1
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.skip_window = skip_window

    async def chunk_text(self, text: str) -> List[SemanticChunk]:
        """Split text into semantic chunks using double-pass merging"""
        # Initial split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # First pass: Create initial chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(SemanticChunk(chunk_text))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(SemanticChunk(' '.join(current_chunk)))

        # Second pass: Merge similar chunks within skip window
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await self.embeddings.embed_batch(chunk_texts)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        merged_chunks = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            best_merge = None
            best_similarity = -1

            # Look ahead within skip window
            for j in range(i + 1, min(i + 1 + self.skip_window, len(chunks))):
                similarity = self._cosine_similarity(current.embedding, chunks[j].embedding)
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    if len(current.text.split()) + len(chunks[j].text.split()) <= self.max_chunk_size:
                        best_merge = j
                        best_similarity = similarity

            if best_merge is not None:
                # Merge chunks
                merged_text = current.text + " " + chunks[best_merge].text
                merged_embedding = await self.embeddings.embed(merged_text)
                current = SemanticChunk(merged_text, merged_embedding)
                i = best_merge + 1
            else:
                merged_chunks.append(current)
                i += 1

        return merged_chunks

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def chunk_batch(self, texts: List[str]) -> List[List[SemanticChunk]]:
        """Process multiple texts in batch"""
        results = []
        for text in texts:
            chunks = await self.chunk_text(text)
            results.append(chunks)
        return results 