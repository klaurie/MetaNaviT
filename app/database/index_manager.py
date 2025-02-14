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
from typing import Generator, List, Dict, Any, Optional

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
                
                # Create directory_processing_results table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS directory_processing_results (
                        dir_path TEXT NOT NULL,
                        process_name TEXT NOT NULL,
                        process_version TEXT NOT NULL,
                        is_applicable BOOLEAN NOT NULL,
                        mtime BIGINT NOT NULL,
                        PRIMARY KEY (dir_path, process_name, process_version)
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
        
        self.execute_query(
            """
            INSERT INTO indexed_files
            (file_path, process_name, process_version, mtime, data)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (file_path, process_name, process_version)
            DO UPDATE SET
                mtime = EXCLUDED.mtime,
                data = EXCLUDED.data;
            """,
            params=values,
            fetch=False
        )
        logger.info(f"Inserted batch of {len(batch)} files")

    def crawl_file_system(
        self, 
        dir_path: str, 
        max_workers: int = 4, 
        batch_size: int = 1000
    ) -> Generator[List[Dict], None, None]:
        """
        Crawl filesystem in parallel and yield batches of file information.
        
        Args:
            dir_path: Starting directory path
            max_workers: Number of parallel processing threads
            batch_size: Number of files to process before yielding
            
        Yields:
            List[Dict]: Batches of file information dictionaries
            
        Implementation:
            1. Uses ThreadPoolExecutor for parallel processing
            2. Manages thread-safe batching with Lock
            3. Handles directory access errors gracefully
            4. Yields results as soon as batch_size is reached
            
        Example:
            ```python
            index_manager = IndexManager()
            for batch in index_manager.crawl_file_system("/data"):
                index_manager.batch_insert_indexed_files(batch)
            ```
        """
        # Thread-safe queue for passing batches between workers and main thread
        result_queue: Queue = Queue()
        # Lock for synchronizing access to the current batch
        batch_lock = Lock()
        # Accumulator for files until we reach batch_size
        current_batch: List[Dict] = []

        def add_to_batch(item: Dict) -> Optional[List[Dict]]:
            """Thread-safe batch accumulator with backpressure."""
            nonlocal current_batch
            with batch_lock:
                # Implement backpressure if batch is full
                while len(current_batch) >= batch_size:
                    # Release lock temporarily to allow other threads to process
                    batch_lock.release()
                    time.sleep(0.1)  # Prevent CPU spinning
                    batch_lock.acquire()

                current_batch.append(item)
                # When batch is full, return a copy and clear original
                if len(current_batch) >= batch_size:
                    batch_to_yield = current_batch.copy()
                    current_batch.clear()
                    return batch_to_yield
            return None

        def process_directory(directory: str) -> None:
            """Worker function that runs in separate threads."""
            try:
                # os.walk yields (root, dirs, files) for each directory
                for root, dirs, files in os.walk(directory):
                    # Filter out blocked directories in-place
                    # Note: modifying dirs[:] affects which subdirs os.walk visits
                    dirs[:] = [d for d in dirs if not self._is_path_blocked(os.path.join(root, d))]
                    
                    for name in files:
                        file_path = os.path.join(root, name)
                        if self._is_path_blocked(file_path):
                            continue
                            
                        try:
                            # Collect file metadata
                            file_info = {
                                "pathname": file_path,
                                "modified_time": os.path.getmtime(file_path),
                                "file_type": os.path.splitext(name)[1]
                            }
                            # If batch is full, it's added to result queue
                            if batch := add_to_batch(file_info):
                                result_queue.put(batch)
                        except OSError as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                            
            except Exception as e:
                logger.error(f"Error processing directory {directory}: {e}")

        # Main parallel processing loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                # Get list of subdirectories to process in parallel
                subdirs = [
                    os.path.join(dir_path, d) for d in os.listdir(dir_path)
                    if os.path.isdir(os.path.join(dir_path, d)) 
                    and not self._is_path_blocked(os.path.join(dir_path, d))
                ]
            except OSError as e:
                logger.error(f"Error accessing directory {dir_path}: {e}")
                return

            # Submit all directories for processing
            futures = [executor.submit(process_directory, d) for d in subdirs]
            # Also process the root directory
            futures.append(executor.submit(process_directory, dir_path))
            
            # Main event loop: yield results as they become available
            while futures or not result_queue.empty():
                # Yield completed batches from queue
                while not result_queue.empty():
                    yield result_queue.get_nowait()
                
                # Check for completed futures
                done, futures = futures[:], []
                for future in as_completed(done):
                    if future.exception():
                        raise future.exception()

        # Don't forget partial batches at the end
        with batch_lock:
            if current_batch:
                yield current_batch