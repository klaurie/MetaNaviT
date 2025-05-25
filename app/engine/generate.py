"""
Document Ingestion Pipeline Module

Handles the process of:
1. Loading documents from sources
2. Setting up document storage
3. Creating embedding pipeline
4. Processing documents into vector store

Key Components:
- Document store initialization
- Vector store setup
- Ingestion pipeline configuration
- Environment-based settings
"""
# flake8: noqa: E402
from dotenv import load_dotenv

# VSCode caches .env files somewhere, so we need to override the environment variables 
load_dotenv(override=True)

import logging
import os
import re

from app.engine.loaders import get_documents
from app.database.vector_store_manager import get_vector_store
from llama_index.core.node_parser.interface import TextSplitter

from app.settings import init_settings

from llama_index.core.ingestion import  IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import TransformComponent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")

class NullByteRemover(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"\0", "", node.text)
        return nodes

def get_doc_store():
    """
    Initialize document store from storage directory or memory.
    
    Returns:
        SimpleDocumentStore: Initialized document store
        
    Note:
        Falls back to in-memory store if storage directory doesn't exist
    """
    # If the storage directory is there, load the document store from it.
    # If not, set up an in-memory document store since we can't load from a directory that doesn't exist.
    if os.path.exists(STORAGE_DIR):
        return SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    else:
        return SimpleDocumentStore()


def run_pipeline(docstore, vector_store, documents):
    """
    Run document ingestion pipeline with configured transformations.
    
    Args:
        docstore: Document storage backend
        vector_store: Vector database connection
        documents: List of documents to process
        
    Returns:
        List of processed document nodes
        
    Steps:
        1. Split documents into chunks
        2. Generate embeddings
        3. Store in vector database
    """
    #logger.info(f"embedding model {Settings.embed_model}\n\n\n")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
            ),
            NullByteRemover(),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )

    # Run the ingestion pipeline and store the results
    nodes = pipeline.run(show_progress=True, documents=documents)

    return 
    

def persist_storage(docstore, vector_store):
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )
    storage_context.persist(STORAGE_DIR)


def generate_datasource():
    init_settings()
    logger.info("Generate index for the provided data")

    # Get the stores and documents or create new ones
    documents = get_documents()
    
    if len(documents) > 0:
        # Set private=false to mark the document as public (required for filtering)
        for doc in documents:
            doc.metadata["private"] = "false"

        docstore = get_doc_store()
        vector_store = get_vector_store()

        #logger.info(f"Document store: {docstore}")
        #logger.info(f"Vector store: {vector_store}")

        # Run the ingestion pipeline
        _ = run_pipeline(docstore, vector_store, documents)

        # Build the index and persist storage
        persist_storage(docstore, vector_store)

        logger.info("Finished generating the index")
    else:
        logger.info("No new documents to index")



def print_all_env_variables():
    for key, value in os.environ.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    generate_datasource()