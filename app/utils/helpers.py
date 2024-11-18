import numpy as np
from typing import List, Dict, Any
import logging
import os
import httpx
from PyPDF2 import PdfReader
import json
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

async def get_ollama_embedding(text: str) -> np.ndarray:
    """Get embeddings from Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=30.0
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding)
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def process_file_content(file_path: str) -> str:
    """Process file content based on file type"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
    return chunks

async def check_ollama_health() -> Dict[str, Any]:
    """Check Ollama service health"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            return {"status": "operational", "models": response.json()}
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {"status": "unavailable", "error": str(e)}

async def get_ollama_response(query: str, context: str = "") -> str:
    """Get response from Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
                    "stream": False
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Error getting Ollama response: {str(e)}")
        raise

def calculate_accuracy_metrics(query_history: Dict) -> Dict[str, float]:
    """Calculate accuracy metrics for a RAG response"""
    try:
        # Basic metrics calculation
        metrics = {
            "relevance": 0.0,
            "coherence": 0.0,
            "factual": 0.0
        }
        
        if not query_history.get("response") or not query_history.get("context"):
            return metrics
            
        # Calculate relevance based on context overlap
        context_words = set(query_history["context"].lower().split())
        response_words = set(query_history["response"].lower().split())
        overlap = len(context_words.intersection(response_words))
        metrics["relevance"] = min(1.0, overlap / len(response_words) if response_words else 0)
        
        # Simple coherence score based on response length and structure
        response_length = len(query_history["response"].split())
        metrics["coherence"] = min(1.0, response_length / 100)  # Normalize to max 1.0
        
        # Factual accuracy requires ground truth, set to neutral value
        metrics["factual"] = 0.5
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating accuracy metrics: {e}")
        return {"relevance": 0.0, "coherence": 0.0, "factual": 0.0}

def extract_metadata(directory_path: str) -> List[Dict[str, str]]:
    """Extract metadata from files in directory"""
    parsed_files = []
    supported_extensions = {'.txt', '.pdf', '.md', '.doc', '.docx'}
    
    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    file_path = os.path.join(root, file)
                    parsed_files.append({
                        "file_name": file,
                        "file_path": file_path,
                        "relative_path": os.path.relpath(file_path, directory_path)
                    })
        return parsed_files
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        raise

async def generate_ollama_response(prompt: str, context: str = "", model: str = "llama2") -> str:
    """Generate response using Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:",
                    "stream": False
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["response"]
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
