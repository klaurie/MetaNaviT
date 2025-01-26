import logging
import psycopg2
from psycopg2 import sql
import os
import time
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
from typing import List, Dict

DATABASE_NAME = os.getenv("DB_NAME", "metanavit")

logger = logging.getLogger("uvicorn")

class IndexManager:
    def __init__(self, conn_string=None):
        """
        Initializes the database manager with a connection string.
        
        Args:
            conn_string: The connection string for the database.
        """
        # Default system paths to skip crawling
        self.blocked_dirs = {
            '/proc',
            '/sys',
            '/run',
            '/dev',
            '/tmp',
            '/var/cache',
            '/var/tmp',
            '/anaconda3'
        }

        # during testing processing hidden takes forever so im going to block it for now
        self.block_hidden_files = True

        if conn_string is None:
            conn_string = os.getenv("PSYCOPG2_CONNECTION_STRING")
            if conn_string is None:
                raise ValueError("Connection string is not provided.")
        
        self.conn_string = conn_string
        self.conn = None

        # check database is created and create if it does not exist
        self._create_database()

        self._connect()

        # Create tables if they do not exist
        self._create_indexed_files_table()
        self._create_directory_processing_results_table()

    def _connect(self):
        """ Connects to the database using psycopg2 """
        try:
            self.conn = psycopg2.connect(self.conn_string)
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def _close_connection(self):
        """ Closes the database connection """
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def _create_database(self):
        original_conn_string = os.getenv("PG_CONNECTION_STRING")
        if original_conn_string is None or original_conn_string == "":
            raise ValueError("PG_CONNECTION_STRING environment variable is not set.")

        # Remove the database name from the connection string to connect to the default database
        parsed_url = urlparse(original_conn_string)
        conn_string_without_db = urlunparse(parsed_url._replace(path="/postgres"))

        conn = psycopg2.connect(conn_string_without_db)
        conn.autocommit = True
        cursor = conn.cursor()

        try:
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = 'metanavit'")
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("metanavit")))
                print("Database 'metanavit' created successfully.")
            else:
                print("Database 'metanavit' already exists.")
        except Exception as e:
            print(f"Error creating database: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def _create_indexed_files_table(self):
        """ Creates the indexed_files table if it does not exist """
        try:
            with self.conn.cursor() as cur:
                query = """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT NOT NULL,
                    process_name TEXT NOT NULL,
                    process_version TEXT NOT NULL,
                    mtime BIGINT NOT NULL,
                    data BYTEA,
                    PRIMARY KEY (file_path, process_name, process_version)
                );
                """
                cur.execute(query)
                self.conn.commit()
                print("indexed_files table created successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating indexed_files table: {e}")

    def _create_directory_processing_results_table(self):
        """ Creates the directory_processing_results table if it does not exist """
        try:
            with self.conn.cursor() as cur:
                query = """
                CREATE TABLE IF NOT EXISTS directory_processing_results (
                    dir_path TEXT NOT NULL,
                    process_name TEXT NOT NULL,
                    process_version TEXT NOT NULL,
                    is_applicable BOOLEAN NOT NULL,
                    mtime BIGINT NOT NULL,
                    PRIMARY KEY (dir_path, process_name, process_version)
                );
                """
                cur.execute(query)
                self.conn.commit()
                print("directory_processing_results table created successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating directory_processing_results table: {e}")

    def _is_path_blocked(self, path: str) -> bool:
            """
        Check if a path should be blocked from processing.
        Uses pathlib for cross-platform compatibility (at least it should).
            
            Args:
                path: Path to check
            
            Returns:
                bool: True if path should be blocked
            """
            # Conver string to Path object for better handling
            path_obj = Path(path).resolve() #  Resolve eliminates symlinks

            # Check if path is hidden and block if feature is enabled
            if self.block_hidden_files and path_obj.name.startswith('.'):
                logger.debug(f"Blocking hidden path: {path_obj}")
                return True
            
            # Check blocked patterns
            for pattern in self.blocked_dirs:
                pattern_obj = Path(pattern) # COnvert to Path for consistent comparison
                
                # Handle wildcard patterns
                if str(pattern_obj).endswith('/*'):
                    pattern_base = pattern_obj.parent
                    if path_obj == pattern_base or pattern_base in path_obj.parents:
                        logger.debug(f"Blocking matched pattern {pattern}: {path_obj}")
                        return True
                # Handle contains and ends with matches
                elif str(pattern_obj) in str(path_obj) or str(path_obj).endswith(str(pattern_obj)):
                    logger.debug(f"Blocking pattern match {pattern}: {path_obj}")
                    return True
            
            return False

    def update_indexed_file(self, file_path, process_name, process_version, new_data):
        """
        Updates the processed data for a specific file in the indexed_files table.
        
        Args:
            file_path: Path of the file to update.
            process_name: Processing strategy name.
            process_version: Processing strategy version.
            new_data: New binary data to update the file with.
        """
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL("""
                    UPDATE indexed_files
                    SET data = %s, mtime = EXTRACT(EPOCH FROM NOW())::BIGINT
                    WHERE file_path = %s
                    AND process_name = %s
                    AND process_version = %s;
                """)
                cur.execute(query, (new_data, file_path, process_name, process_version))
                self.conn.commit()
                print("File data updated successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating file data: {e}")

    def update_directory_processing_result(self, dir_path, process_name, process_version, new_is_applicable):
        """
        Updates the 'is_applicable' field for a specific directory in the directory_processing_results table.
        
        Args:
            dir_path: Path of the directory to update.
            process_name: Processing strategy name.
            process_version: Processing strategy version.
            new_is_applicable: New value for the 'is_applicable' field (boolean).
        """
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL("""
                    UPDATE directory_processing_results
                    SET is_applicable = %s, mtime = EXTRACT(EPOCH FROM NOW())::BIGINT
                    WHERE dir_path = %s
                    AND process_name = %s
                    AND process_version = %s;
                """)
                cur.execute(query, (new_is_applicable, dir_path, process_name, process_version))
                self.conn.commit()
                print("Directory processing result updated successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating directory processing result: {e}")

    def insert_indexed_file(self, file_path, process_name, process_version, mtime, data):
        """ Inserts a new record into the indexed_files table """
        try:
            self._connect()
            # Ensure the connection is open
            if self.conn.closed:
                raise ValueError("Database connection is closed.")
            
            with self.conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO indexed_files (file_path, process_name, process_version, mtime, data)
                    VALUES (%s, %s, %s, %s, %s);
                """)
                cur.execute(query, (file_path, process_name, process_version, mtime, data))
                self.conn.commit()
                print("Indexed file inserted successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting indexed file: {e}")

    def insert_directory_processing_result(self, dir_path, process_name, process_version, is_applicable, mtime):
        """ Inserts a new record into the directory_processing_results table """
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO directory_processing_results (dir_path, process_name, process_version, is_applicable, mtime)
                    VALUES (%s, %s, %s, %s, %s);
                """)
                cur.execute(query, (dir_path, process_name, process_version, is_applicable, mtime))
                self.conn.commit()
                print("Directory processing result inserted successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting directory processing result: {e}")

    def delete_indexed_file(self, file_path, process_name, process_version):
        """ Deletes a record from the indexed_files table """
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL("""
                    DELETE FROM indexed_files
                    WHERE file_path = %s
                    AND process_name = %s
                    AND process_version = %s;
                """)
                cur.execute(query, (file_path, process_name, process_version))
                self.conn.commit()
                print("Indexed file deleted successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting indexed file: {e}")

    def delete_directory_processing_result(self, dir_path, process_name, process_version):
        """ Deletes a record from the directory_processing_results table """
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL("""
                    DELETE FROM directory_processing_results
                    WHERE dir_path = %s
                    AND process_name = %s
                    AND process_version = %s;
                """)
                cur.execute(query, (dir_path, process_name, process_version))
                self.conn.commit()
                print("Directory processing result deleted successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting directory processing result: {e}")

    def update_filesystem_info(self, dir_path='/'):
        """
        Crawls filesystem starting from the given directory and updates the database.
        
        Args:
            directory: The root directory to start crawling from. Default is root directory.
        
        Returns:

        """
        pass

    def crawl_file_system(self, dir_path, max_workers=4, batch_size=1000):
        """
        Crawls a directory recursively using multiple threads, yielding results in batches.

        Note: Threads are only created for first level directories, so right now that feature is useless
              if there are no subdirectories in the root folder.
        
        Args:
            dir_path: The root directory to start crawling from.
            max_workers: Maximum number of threads to use.
            batch_size: Number of files to accumulate before yielding.
        
        Yields:
            Lists of dictionaries containing file information:
            - pathname: str
            - modified_time: float
            - file_type: str
        Raises:
            OSError: If directory access fails
        """
        result_queue = Queue()
        batch_lock = Lock()
        current_batch =[]

        def add_to_batch(item):
            """current_batch is not thread-safe, so we need to use a lock to protect it"""
            nonlocal current_batch
            with batch_lock:
                # If the batch is full, wait for it to be processed
                while len(current_batch) >= batch_size:
                    # this should be fine since we are only adding one item at a time
                    batch_lock.release()  # Release lock temporarily to avoid blocking other threads
                    time.sleep(0.1)  # Sleep a short time before retrying
                    batch_lock.acquire()  # Reacquire the lock

                current_batch.append(item)

                # if batch is full yeild results to be processed
                if len(current_batch) >= batch_size:
                    batch_to_yield = current_batch
                    current_batch = []
                    return batch_to_yield
            return None
        
        def process_directory(directory: str):
            try:
                for root, dirs, files in os.walk(directory):
                    # Filter out blocked directories
                    dirs[:] = [d for d in dirs if not self._is_path_blocked(os.path.join(root, d))]
                
                    for name in files:
                        file_path = os.path.join(root, name)
                        
                        # Skip blocked files
                        if self._is_path_blocked(file_path):
                            logger.debug(f"Skipping blocked file: {file_path}")
                            continue
                        try:
                            modified_time = os.path.getmtime(file_path)
                            file_type = os.path.splitext(name)[1]
                            
                            file_info = {
                                "pathname": file_path,
                                "modified_time": modified_time,
                                "file_type": file_type
                            }
                            # Thread-safe addition to batch
                            if batch := add_to_batch(file_info):
                                result_queue.put(batch)
                        except OSError as e:
                            logger.error(f"Error processing file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error processing directory {directory}: {e}")

        # Get immediate subdirectories
        try:
            # Get immediate subdirectories, filtering blocked ones
            subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) 
                    if os.path.isdir(os.path.join(dir_path, d)) 
                    and not self._is_path_blocked(os.path.join(dir_path, d))]
            print(subdirs)
            logger.info(f"Found {subdirs} in {dir_path}")
        except OSError as e:
            logger.error(f"Error accessing directory {dir_path}: {e}")
            return

        # Process directories in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            """
             Execute process_directory for each subdirectory and current directory
             The Future objects allow us to check task status and get results
             This is nice because the object is created even when it's not immediately able to run
            """
            futures = [executor.submit(process_directory, d) for d in subdirs]
            futures.append(executor.submit(process_directory, dir_path))
            
            # Process and yield batches as they complete
            while futures or not result_queue.empty():
                while not result_queue.empty():
                    yield result_queue.get_nowait()
                
                done, futures = futures[:], []
                for future in as_completed(done):
                    # check for any errors that occurred during processing
                    if future.exception():
                        raise future.exception()

        # Yield any remaining files
        with batch_lock:
            if current_batch:
                yield current_batch

    def close(self):
        """ Close the connection to the database """
        self._close_connection()


if __name__=='__main__':
    from pathlib import Path
    index_manager = IndexManager()
    home_dir = os.path.expanduser('~')
    logger.info(f"Starting filesystem crawl from: {home_dir}")
    for batch in index_manager.crawl_file_system(home_dir):
        for file in batch:
            print(file)
    index_manager.close()