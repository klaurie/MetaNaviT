"""
Index File Access Tool

Provides capabilities for accessing file contents from the document index
rather than directly from the file system.
"""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool, BaseTool

from app.engine.index import get_index
from app.database.index_manager import IndexManager
from app.engine.tools import create_tool_callback

logger = logging.getLogger(__name__)

class ListFileSchema(BaseModel):
    pattern: str = Field(
        default="*",
        description="Filter pattern to match file paths (e.g., *.py for Python files)"
    )

class GetFileSchema(BaseModel):
    file_path: str = Field(
        description="Path to the file that should be retrieved from the index"
    )

def get_file_by_path(
    file_path: str = Field(
        description="Path to the file that should be retrieved from the index"
    )
) -> str:
    """
    Get file content from the index based on file path.
    
    This tool searches the document index for a document with the specified file path
    in its metadata and returns the content.
    
    Args:
        file_path: Path of the file to retrieve
        
    Returns:
        Content of the file as a string
    """
    # Get the index
    index = get_index()
    if not index:
        return "Error: Document index not available"
    
    # Query for documents with matching file_path in metadata
    query_engine = index.as_query_engine(
        filters={"file_path": {"$eq": file_path}},
        similarity_top_k=1
    )
    
    # Get the document
    try:
        response = query_engine.query("")  # Empty query to just get the document
        
        if not response.source_nodes:
            return f"Error: No document found with path {file_path}"
        
        # Return the content of the first matching document
        return response.source_nodes[0].get_content()
        
    except Exception as e:
        logger.error(f"Error retrieving file from index: {str(e)}")
        return f"Error: {str(e)}"

def list_available_files(
    pattern: str = Field(
        default="*",
        description="Filter pattern to match file paths (e.g., *.py for Python files)"
    )
) -> List[str]:
    """
    List available files in the index with optional pattern matching.
    
    Args:
        pattern: Filter pattern for file paths
        
    Returns:
        List of available file paths matching the pattern
    """
    logger.info("Listing available files in the index with pattern: %s", pattern)
    # Use IndexManager to query the database for indexed files
    index_manager = IndexManager()
    
    # Query for file paths from the indexed_files table
    with index_manager.get_connection() as conn:
        with conn.cursor() as cur:
            # Use LIKE for pattern matching
            sql_pattern = pattern.replace("*", "%").replace("?", "_")
            
            cur.execute("""
                SELECT DISTINCT file_path 
                FROM indexed_files 
                WHERE file_path LIKE %s
                ORDER BY file_path
                LIMIT 100
            """, (sql_pattern,))
            
            results = cur.fetchall()
            
    # Extract file paths from results
    file_paths = [row[0] for row in results]
    
    return file_paths

def get_tools() -> List[BaseTool]:
    """
    Create and return index-based file access tools.
    
    Returns:
        List of file access function tools
    """
    file_tool_name = "get_file_content"
    file_tool_description = "Get file content from the index based on file path"
    file_get_tool = FunctionTool.from_defaults(
        fn=get_file_by_path,
        name=file_tool_name,
        description=file_tool_description,
        fn_schema=GetFileSchema,
        callback=create_tool_callback(file_tool_name, file_tool_description)
    )
    
    file_list_tool_name ="list_available_files"
    file_list_tool_description = "List available files in the index with optional pattern matching"
    file_list_tool = FunctionTool.from_defaults(
        fn=list_available_files,
        name=file_list_tool_name,
        description=file_list_tool_description,
        fn_schema=ListFileSchema,
        callback=create_tool_callback(file_list_tool_name, file_list_tool_description)
    )
    
    return [file_get_tool, file_list_tool]