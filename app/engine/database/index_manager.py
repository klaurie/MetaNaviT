import logging
import psycopg2
from psycopg2 import sql
import os
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

    def update_indexed_file(self, file_path, process_name, process_version, new_data):
        """
        Updates the processed data for a specific file in the indexed_files table.
        
        Args:
            file_path: Path of the file to update.
            process_name: Processing strategy name.
            process_version: Processing strategy version.
            new_data: New binary data to update the file with.
        """
        pass

    def update_directory_processing_result(self, dir_path, process_name, process_version, new_is_applicable):
        """
        Updates the 'is_applicable' field for a specific directory in the directory_processing_results table.
        
        Args:
            dir_path: Path of the directory to update.
            process_name: Processing strategy name.
            process_version: Processing strategy version.
            new_is_applicable: New value for the 'is_applicable' field (boolean).
        """
        pass

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

        """
        pass


    def close(self):
        """ Close the connection to the database """
        self._close_connection()


if __name__=='__main__':
    index_manager = IndexManager()
    index_manager.close()