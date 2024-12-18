import asyncio
import os
from app.utils.embeddings import NomicEmbeddings, SDPMChunker
from app.utils.helpers import get_files_recursive
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_chunking():
    """Test the semantic chunking implementation"""
    # Initialize embeddings and chunker
    embeddings = NomicEmbeddings()
    chunker = SDPMChunker(
        embeddings=embeddings,
        similarity_threshold=0.5,
        max_chunk_size=1000,
        min_chunk_size=100,
        skip_window=2
    )
    
    # Test text
    test_text = """
    Artificial Intelligence (AI) has revolutionized many industries. Machine learning, 
    a subset of AI, enables computers to learn from data. Deep learning, which is a 
    type of machine learning, uses neural networks to process complex patterns.
    
    Neural networks are inspired by the human brain. They consist of layers of 
    interconnected nodes. Each node processes information and passes it to the next layer.
    
    Python is a popular programming language for AI development. Libraries like 
    TensorFlow and PyTorch make it easier to implement neural networks. Many 
    companies use these tools for their AI projects.
    """
    
    try:
        # Test chunking
        chunks = await chunker.chunk_text(test_text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Print each chunk and its embedding shape
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"\nChunk {i}:")
            logger.info(f"Text: {chunk.text}")
            logger.info(f"Embedding shape: {chunk.embedding.shape}")
            logger.info(f"Token count: {chunk.token_count}")
            
            # If there are more chunks, show similarity with next chunk
            if i < len(chunks):
                similarity = chunker._cosine_similarity(chunk.embedding, chunks[i].embedding)
                logger.info(f"Similarity with next chunk: {similarity:.3f}")
    
    except Exception as e:
        logger.error(f"Error in chunking test: {str(e)}")
        raise

async def test_file_processing():
    """Test processing actual files from the upload directory"""
    from app.config import UPLOAD_DIR
    
    # Get list of files
    files = get_files_recursive(UPLOAD_DIR)
    logger.info(f"Found {len(files)} files: {files}")
    
    # Process first file as a test
    if files:
        test_file = files[0]
        logger.info(f"\nTesting with file: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize chunker
        embeddings = NomicEmbeddings()
        chunker = SDPMChunker(
            embeddings=embeddings,
            similarity_threshold=0.5,
            max_chunk_size=1000,
            min_chunk_size=100,
            skip_window=2
        )
        
        # Process file
        chunks = await chunker.chunk_text(content)
        logger.info(f"Split file into {len(chunks)} chunks")
        
        # Show first chunk as sample
        if chunks:
            logger.info(f"\nFirst chunk sample:")
            logger.info(f"Text: {chunks[0].text[:200]}...")
            logger.info(f"Embedding shape: {chunks[0].embedding.shape}")
            logger.info(f"Token count: {chunks[0].token_count}")

if __name__ == "__main__":
    # Run both tests
    asyncio.run(test_chunking())
    asyncio.run(test_file_processing()) 