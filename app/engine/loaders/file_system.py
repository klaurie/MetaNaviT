import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
from typing import List, Dict, Optional, Generator
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Document

from app.database.index_manager import IndexManager

logger = logging.getLogger(__name__)


def crawl_file_system(
    index_manager: IndexManager,
    dir_path: str, 
    max_workers: int = 4, 
    batch_size: int = 1000,
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
                dirs[:] = [d for d in dirs if not index_manager.is_path_blocked(os.path.join(root, d))]
                
                for name in files:
                    file_path = os.path.join(root, name)
                    if index_manager.is_path_blocked(file_path):
                        # File is path is blocked from being indexed
                        continue
                    if not index_manager.is_file_modified(file_path):
                        # File has not been modified since last indexes
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
                and not index_manager.is_path_blocked(os.path.join(dir_path, d))
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

def get_files(path: str, index_manager: IndexManager(), max_workers: int = 4, batch_size: int = 1000):
    """ This function runs the pipeline to get all files from a directory,
        add them to the database, process if applicable, remove if not applicable,
        and return the list of documents.
        
        Note: I am unsure if this is the fastest way to get things done and I may need advice.
              There are a lot of steps to be taken, to improve the quality of our index and 
              also allow for index tracking capabilities. The first run would be slow, but
              in future runs (once implemented) the app will not have to reprocess the files         
    """
    file_paths = []

    # Just crawl and get batches of files
    for batch in crawl_file_system(
        index_manager,
        path,
        max_workers=max_workers,
        batch_size=batch_size
    ):
        logger.info(f"batch {batch}\n")
        # add files information to index manager database
        index_manager.batch_insert_indexed_files(batch)

        # Add pathnames to list for loading
        file_paths.extend([item["pathname"] for item in batch])
    
    documents = None

    # len(file_paths) = 0 when there are no new files to be indexed
    if len(file_paths) > 0:
        # Configure directory reader and load documents
        reader = SimpleDirectoryReader(
            input_files=file_paths,
            )
        documents = reader.load_data()
    return documents