from typing import Optional, List, Sequence, Any, Union
from chonkie import SDPMChunker
from pydantic import BaseModel, Field
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.settings import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.ollama import OllamaEmbedding
import logging

logger = logging.getLogger(__name__)

class EmbeddingWrapper:
    """Wrapper to make LlamaIndex embeddings compatible with chonkie"""
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.embedding_dim = 768  # NomicEmbed dimension
        self.max_seq_length = 1000
        
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Convert texts to embeddings using the wrapped model"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embed_model.get_text_embedding_batch(texts)
        return np.array(embeddings)
    
    def __str__(self):
        return "EmbeddingWrapper(NomicEmbed)"

class SemanticChunkerConfig(BaseModel):
    chunk_size: int = 1000
    min_sentences: int = 1
    skip_window: int = 1
    threshold: float = 0.5

class OllamaEmbeddingWrapper:
    """Wrapper to make Ollama embeddings compatible with sentence-transformers interface"""
    def __init__(self):
        self.ollama = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        self.max_seq_length = 1000
        self.embedding_dim = 768  # Nomic embed dimension
        
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.ollama.get_text_embedding_batch(texts)
        return np.array(embeddings)

class SDPMChunkerComponent(SimpleNodeParser):
    """Wrapper for SDPMChunker to make it compatible with LlamaIndex"""
    
    # Define both fields in the class
    chunker: SDPMChunker = Field(default_factory=lambda: SDPMChunker(
        embedding_model="all-MiniLM-L6-v2",
        threshold=0.5,
        chunk_size=1000,
        min_sentences=1,
        skip_window=1
    ))
    
    # Add embedding_model as a proper field
    embedding_model: Any = Field(default_factory=lambda: Settings.embed_model)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunker.use_spacy = False
    
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any
    ) -> List[BaseNode]:
        """Transform sequence of nodes by chunking their text content."""
        nodes = []
        for doc in documents:
            chunks = self.chunker.chunk(doc.text)
            for i, chunk in enumerate(chunks):
                # Generate embedding for this chunk
                embedding = Settings.embed_model.get_text_embedding(chunk.text)
                
                # Create new node for each chunk while preserving metadata
                new_node = Document(
                    text=chunk.text,
                    metadata=doc.metadata.copy(),
                    relationships=doc.relationships.copy(),
                    id_=f"{doc.doc_id}_{i}",  # Add explicit ID
                    embedding=embedding  # Set the embedding directly
                )
                nodes.append(new_node)
                
                # Log for debugging
                logger.info(f"Created node {new_node.id_} with embedding shape: {len(embedding)}")
                
        logger.info(f"Created total of {len(nodes)} nodes with embeddings")
        return nodes