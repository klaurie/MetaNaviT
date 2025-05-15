#!/usr/bin/env python3
import os
import sys
import psycopg2
from psycopg2 import sql
from pathlib import Path

# First, try to load environment from .env file
def load_dotenv(env_file='.env'):
    """Load environment variables from .env file"""
    try:
        # Find the .env file - search in current directory and parent directories
        env_path = Path(env_file)
        if not env_path.exists():
            # Try to find it in the project root (parent directories)
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:  # Stop at filesystem root
                potential_path = current_dir / env_file
                if potential_path.exists():
                    env_path = potential_path
                    break
                current_dir = current_dir.parent
        
        # If we found the .env file, load it
        if env_path.exists():
            print(f"Loading environment from {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse variable assignments
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip()
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # Set environment variable
                        os.environ[key.strip()] = value
            return True
        else:
            print(f"Warning: .env file not found")
            return False
    except Exception as e:
        print(f"Error loading .env file: {e}")
        return False

# Try to load the environment file
load_dotenv()

def test_connection():
    """Test database connection and create database if needed."""
    
    # Get the database URL from environment
    db_url = os.getenv("PG_CONNECTION_STRING")
    if not db_url:
        print("ERROR: PG_CONNECTION_STRING environment variable is not set")
        sys.exit(1)
    
    print(f"Using connection string: {db_url}")
    
    try:
        # First, try connecting to postgres directly to create db if needed
        conn_parts = db_url.split('/')
        base_conn = '/'.join(conn_parts[:-1]) + '/postgres'
        
        print(f"Connecting to base PostgreSQL instance...")
        conn = psycopg2.connect(base_conn)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Get database name from URL
            db_name = conn_parts[-1].split('?')[0]
            print(f"Checking if database '{db_name}' exists...")
            
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone() is None:
                print(f"Database '{db_name}' does not exist. Creating it...")
                # Create safe SQL identifier
                db_name_sql = sql.Identifier(db_name).as_string(conn)
                # Create the database
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                print(f"Database '{db_name}' created successfully")
            else:
                print(f"Database '{db_name}' already exists")
        
        conn.close()
        
        # Now try connecting to the actual database
        print(f"Connecting to application database...")
        conn = psycopg2.connect(db_url)
        print("Connection successful!")
        
        # Test executing a simple query
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()
            print(f"PostgreSQL version: {version[0]}")
            
            # Check if pg_vector extension is installed
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone() is None:
                print("WARNING: 'vector' extension is not installed")
                print("Run: CREATE EXTENSION vector;")
            else:
                print("✅ 'vector' extension is installed")
            
            # Check if the llamaindex_embedding table exists
            cur.execute("""
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'llamaindex_embedding'
            """)
            if cur.fetchone() is None:
                print("WARNING: 'llamaindex_embedding' table does not exist")
                print("You need to create the necessary tables")
            else:
                print("✅ 'llamaindex_embedding' table exists")
                
                # Count documents
                cur.execute("SELECT COUNT(*) FROM public.llamaindex_embedding")
                count = cur.fetchone()[0]
                print(f"Found {count} documents in database")
                
                if count > 0:
                    # Sample document
                    cur.execute("SELECT text FROM public.llamaindex_embedding LIMIT 1")
                    sample = cur.fetchone()
                    print(f"Sample document: {sample[0][:100]}...")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()