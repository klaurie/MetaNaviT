"""
File System Index Manager Module

This module provides functionality for:
1. Crawling file systems efficiently using parallel processing
2. Maintaining a database index of files and their metadata
3. Tracking directory processing status
4. Handling large file systems through batched operations

Dependencies:
    - PostgreSQL with connection pooling
    - ThreadPoolExecutor for parallel processing
    - Path from pathlib for cross-platform path handling
"""

import logging
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
from typing import List, Dict, Any, Optional

from .db_base_manager import DatabaseManager

logger = logging.getLogger(__name__)

class IndexManager(DatabaseManager):
    """
    Manages file system indexing and database tracking.
    
    Key Features:
        - Parallel directory crawling
        - Batched database operations
        - Path blocking for system directories
        - Connection pooling from parent DatabaseManager
        
    Tables Created:
        - indexed_files: Stores file metadata and processing status
        - directory_processing_results: Tracks directory processing state
    """
    
    def __init__(self, conn_string: Optional[str] = None):
        """
        Initialize IndexManager with database connection and default settings.
        
        Args:
            conn_string: Database connection string (optional)
                        Falls back to environment variables if not provided
        
        Note:
            Blocked directories are system paths that should not be indexed
            Hidden files (starting with '.') are blocked by default
        """
        # Initialize parent DatabaseManager
        super().__init__(conn_string)
        
        # Default system paths to skip crawling
        self.blocked_dirs = {
            '/proc', '/sys', '/run', '/dev', '/tmp',
            '/var/cache', '/var/tmp', '/anaconda3'
        }
        self.block_hidden_files = True
        
        # Create required tables
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Create database tables for file and directory tracking.
        
        Tables:
            indexed_files:
                - file_path: Full path to the file
                - process_name: Name of the indexing process
                - process_version: Version of the indexing process
                - mtime: File modification time as Unix timestamp
                - data: Optional binary data associated with the file
                
            directory_processing_results:
                - dir_path: Full path to the directory
                - process_name: Name of the processing operation
                - process_version: Version of the processor
                - is_applicable: Whether directory should be processed
                - mtime: Directory modification time
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create indexed_files table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS indexed_files (
                        file_path TEXT NOT NULL,
                        process_name TEXT NOT NULL,
                        process_version TEXT NOT NULL,
                        mtime BIGINT NOT NULL,
                        data BYTEA,
                        PRIMARY KEY (file_path, process_name, process_version)
                    );
                """)
                
                logger.info("Database tables created successfully")

    def is_path_blocked(self, path: str) -> bool:
        """
        Check if a path should be excluded from processing.
        
        Args:
            path: File or directory path to check
            
        Returns:
            bool: True if path should be blocked, False otherwise
            
        Rules:
            1. Hidden files/dirs (starting with '.') if block_hidden_files is True
            2. System directories listed in self.blocked_dirs
            3. Paths matching blocked patterns with wildcards
        """
        path_obj = Path(path).resolve()
        
        # Check 1: Hidden files/directories
        if self.block_hidden_files and path_obj.name.startswith('.'):
            logger.debug(f"Blocking hidden path: {path_obj}")
            return True
        
        # Check 2: Blocked directories and patterns
        for pattern in self.blocked_dirs:
            pattern_obj = Path(pattern)
            if str(pattern_obj).endswith('/*'):   # Wildcard pattern
                pattern_base = pattern_obj.parent
                if path_obj == pattern_base or pattern_base in path_obj.parents:
                    return True
            elif str(pattern_obj) in str(path_obj):   # Direct match
                return True
        
        return False

    def batch_insert_indexed_files(self, batch: List[Dict[str, Any]]) -> None:
        """
        Insert or update multiple files in the database.
        
        Args:
            batch: List of file information dictionaries containing:
                - pathname: Full path to the file
                - process_name: (optional) Name of the indexing process
                - process_version: (optional) Version of the process
                - modified_time: File modification timestamp
                - data: (optional) Additional file metadata
                
        Note:
            Uses UPSERT (INSERT ... ON CONFLICT) to handle duplicates
            Commits transaction automatically through DatabaseManager
        """
        values = [(
            file['pathname'],
            file.get('process_name', 'default'),
            file.get('process_version', '1.0'),
            int(file['modified_time']),
            file.get('data', None)
        ) for file in batch]
        
        self.execute_many_query(
            """
            INSERT INTO indexed_files
            (file_path, process_name, process_version, mtime, data)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (file_path, process_name, process_version)
            DO UPDATE SET
                mtime = EXCLUDED.mtime,
                data = EXCLUDED.data;
            """,
            params_list=values,
            fetch=False
        )
        logger.info(f"Inserted batch of {len(batch)} files")

    def check_processing_status(self):
        # TODOL: check and remove any files that we dont have processing capabilities for
        pass
    
    def is_file_modified(self, file_path: str) -> bool:
        """
        Check if a file has been modified since it was last indexed.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file has been modified or was never indexed, False otherwise
            
        Note:
            Returns True if file doesn't exist in the database (never indexed)
            Returns True if file exists on disk but with a newer mtime
            Returns False if file exists in database with same or newer mtime
        """
        # Get current file mtime if file exists
        path_obj = Path(file_path)
        if not path_obj.exists() or not path_obj.is_file():
            logger.debug(f"File does not exist: {file_path}")
            return False
            
        current_mtime = int(path_obj.stat().st_mtime)
        
        # Query the database for the stored mtime
        query = """
            SELECT mtime 
            FROM indexed_files 
            WHERE file_path = %s
        """
        results = self.execute_query(query, (file_path,))
        
        # If file not in database, consider it modified (needs indexing)
        if not results:
            logger.debug(f"File not previously indexed: {file_path}")
            return True
            
        stored_mtime = results[0][0]  # First row, first column contains the mtime
        
        # Compare mtimes
        is_modified = current_mtime != stored_mtime
        if is_modified:
            logger.debug(f"File modified since last indexing: {file_path}")
        
        return is_modified