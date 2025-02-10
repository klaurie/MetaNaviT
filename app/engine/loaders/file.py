"""
File Document Loader Module

Handles loading documents from local filesystem using LlamaIndex's SimpleDirectoryReader.
Provides error handling for common issues like empty directories.

Note:
- This will most likely be moved around or obsolete with the addtion of the IndexManager I'm (Kaitlyn) working on.
"""

import logging
from pydantic import BaseModel

# Currently DATA_DIR is only /data for testing
# TODO: integrate with index manager
from app.config import DATA_DIR

logger = logging.getLogger(__name__)


class FileLoaderConfig(BaseModel):
    """Configuration model for file loading parameters"""
    pass  # Currently unused but we might want configuration options in the future


def get_file_documents(config: FileLoaderConfig):
    """
    Load documents from configured data directory.
    
    Uses SimpleDirectoryReader with settings:
    - recursive: True (process subdirectories)
    - filename_as_id: True (use filenames as document IDs)
    - raise_on_error: True (fail on invalid files)
    
    Returns:
        List of loaded documents, or empty list if directory is empty
    
    Raises:
        Exception: For any errors except empty directory
    """
    from llama_index.core.readers import SimpleDirectoryReader

    try:
        # TODO: customize Document() metadata fields **default excludes needed metadata (check vector store)
        reader = SimpleDirectoryReader(
            DATA_DIR,
            recursive=True,
            filename_as_id=True,
            raise_on_error=True
        )
        return reader.load_data()
    except Exception as e:
        import sys
        import traceback

        # Catch if the data directory is empty
        # and return as empty document list
        _, _, exc_traceback = sys.exc_info()
        function_name = traceback.extract_tb(exc_traceback)[-1].name
        if function_name == "_add_files":
            logger.warning(
                f"Failed to load file documents, error message: {e} . Return as empty document list."
            )
            return []
        else:
            # Raise the error if it is not the case of empty data dir
            raise e
