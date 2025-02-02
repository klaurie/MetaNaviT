import base64
import logging
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.readers.file.base import (
    _try_loading_included_file_formats as get_file_loaders_map,
)
from llama_index.core.schema import Document
from llama_index.readers.file import FlatReader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PRIVATE_STORE_PATH = str(Path("output", "uploaded"))
TOOL_STORE_PATH = str(Path("output", "tools"))
LLAMA_CLOUD_STORE_PATH = str(Path("output", "llamacloud"))


class DocumentFile(BaseModel):
    id: str
    name: str  # Stored file name
    type: str = None
    size: int = None
    url: str = None
    path: Optional[str] = Field(
        None,
        description="The stored file path. Used internally in the server.",
        exclude=True,
    )
    refs: Optional[List[str]] = Field(
        None, description="The document ids in the index."
    )


class FileService:
    """
    Service class for managing file operations and document indexing.
    
    Key responsibilities:
    - Handle file uploads and storage in the designated directories
    - Process different file types (txt, csv, etc.)
    - Manage document indexing for search and retrieval
    - Generate and track file metadata using DocumentFile model
    - Integrate with vector store and LlamaCloud indexing systems
    
    Dependencies:
    - LlamaIndex for document indexing and embedding
    - DocumentFile model for file metadata
    - Local filesystem for storage
    - Environment variables for URL configuration
    """

    @classmethod
    def process_private_file(
        cls,
        file_name: str,
        base64_content: str,
        params: Optional[dict] = None,
    ) -> DocumentFile:
        """
        Process uploaded files by saving them and optionally indexing their contents.
        This is the main entry point for file handling in the application.

        Flow:
        1. Configure indexing system
        2. Save file to storage
        3. Index content (skip for CSV files)
        4. Return file metadata
        """
        # Import indexing components - these handle document storage and retrieval
        try:
            from app.engine.index import IndexConfig, get_index
        except ImportError as e:
            raise ValueError("IndexConfig or get_index is not found") from e

        # Initialize optional indexing parameters
        if params is None:
            params = {}

        # Set up the indexing system with provided configuration
        index_config = IndexConfig(**params)
        index = get_index(index_config)

        # Decode the base64 file content and identify file type
        file_data, extension = cls._preprocess_base64_file(base64_content)

        # Save file to private storage area and get metadata
        document_file = cls.save_file(
            file_data,
            file_name=file_name,
            save_dir=PRIVATE_STORE_PATH,
        )

        # TODO: It might be a good idea to have specific handlers for other types as well
        # CSV files are handled separately by tools, so skip indexing
        if extension == "csv":
            return document_file
        
        # Load file content into documents and add to index
        # TODO: Implement with postgresql databse
        documents = cls._load_file_to_documents(document_file)
        cls._add_documents_to_vector_store_index(documents, index)
        document_file.refs = [doc.doc_id for doc in documents]

        return document_file

    @classmethod
    def save_file(cls, content: bytes | str, file_name: str, save_dir: Optional[str] = None) -> DocumentFile:
        """
        Saves uploaded file content to disk and creates metadata record.
        
        Args:
            content: Raw file content as bytes or string
            file_name: Original name of uploaded file
            save_dir: Custom storage directory (defaults to output/uploaded)

        Returns:
            DocumentFile with metadata including generated URL
        """
        # Set default storage directory if none provided
        if save_dir is None:
            save_dir = os.path.join("output", "uploaded")

        # Generate unique file name to prevent collisions
        file_id = str(uuid.uuid4())
        name, extension = os.path.splitext(file_name)
        extension = extension.lstrip(".")
        sanitized_name = _sanitize_file_name(name)  # Remove invalid characters
        if extension == "":
            raise ValueError("File is not supported!")
        new_file_name = f"{sanitized_name}_{file_id}.{extension}"

        # Construct full file path for storage
        file_path = os.path.join(save_dir, new_file_name)

        # Ensure content is in bytes format for writing
        if isinstance(content, str):
            content = content.encode()

        # Write file with error handling for common issues
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create dirs if needed
            with open(file_path, "wb") as file:
                file.write(content)
        except PermissionError as e:
            logger.error(f"Permission denied when writing to file {file_path}: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"IO error occurred when writing to file {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when writing to file {file_path}: {str(e)}")
            raise

        logger.info(f"Saved file to {file_path}")

        # Construct public URL for file access
        file_url_prefix = os.getenv("FILESERVER_URL_PREFIX")
        if file_url_prefix is None:
            logger.warning(
                "FILESERVER_URL_PREFIX is not set, fallback to http://localhost:8000/api/files"
            )
            file_url_prefix = "http://localhost:8000/api/files"
        file_size = os.path.getsize(file_path)

        # Build complete URL path
        file_url = os.path.join(
            file_url_prefix,
            save_dir,
            new_file_name,
        )

        # Return metadata object with all file details
        return DocumentFile(
            id=file_id,
            name=new_file_name,
            type=extension,
            size=file_size,
            path=file_path,
            url=file_url,
            refs=None,  # References to indexed documents, populated later
        )

    @staticmethod
    def _preprocess_base64_file(base64_content: str) -> Tuple[bytes, str | None]:
        """Decodes base64 content and determines file type"""
        header, data = base64_content.split(",", 1)
        mime_type = header.split(";")[0].split(":", 1)[1]
        extension = mimetypes.guess_extension(mime_type).lstrip(".")
        # File data as bytes
        return base64.b64decode(data), extension

    @staticmethod
    def _load_file_to_documents(file: DocumentFile) -> List[Document]:
        """Converts stored file into document objects for indexing"""
        _, extension = os.path.splitext(file.name)
        extension = extension.lstrip(".")

        # TODO: Properly handle different file types that have support on our end even if its not supported on llamaindex
        reader_cls = _default_file_loaders_map().get(f".{extension}")
        if reader_cls is None:
            raise ValueError(f"File extension {extension} is not supported")
        reader = reader_cls()
        
        if file.path is None:
            raise ValueError("Document file path is not set")
        documents = reader.load_data(Path(file.path))
        # Add custom metadata
        for doc in documents:
            doc.metadata["file_name"] = file.name
            doc.metadata["private"] = "true"
        return documents

    @staticmethod
    def _add_documents_to_vector_store_index(documents: List[Document], index: VectorStoreIndex) -> None:
        """Adds documents to vector store index using ingestion pipeline"""
        # TODO: Implement with postgresql databse
        pass


def _sanitize_file_name(file_name: str) -> str:
    """Cleans file names by replacing invalid characters"""
    sanitized_name = re.sub(r"[^a-zA-Z0-9.]", "_", file_name)
    return sanitized_name


def _default_file_loaders_map():
    """Provides mapping of file extensions to appropriate document loaders"""
    default_loaders = get_file_loaders_map()
    default_loaders[".txt"] = FlatReader
    default_loaders[".csv"] = FlatReader
    return default_loaders
