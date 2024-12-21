import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.metadata.metadata_processor import MetadataProcessor
from app.text_splitter import TextSplitter  # Your text splitter implementation
from app.db.vector_store import VectorStore  # Your vector store implementation

async def main():
    # Initialize dependencies
    text_splitter = TextSplitter()
    vector_store = await VectorStore.initialize()  # Assuming you have an initialize method
    
    # Create metadata processor
    processor = MetadataProcessor(text_splitter, vector_store)
    
    # Define the directory to process (you can make this configurable)
    docs_directory = "path/to/your/docs"
    
    # Process the directory
    try:
        await processor.process_directory(docs_directory)
        print(f"Successfully processed documents in {docs_directory}")
    except Exception as e:
        print(f"Error processing documents: {e}")
    finally:
        # Clean up connections if needed
        await vector_store.close()

if __name__ == "__main__":
    asyncio.run(main())