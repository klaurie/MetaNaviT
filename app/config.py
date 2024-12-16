import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to WARNING to reduce noise
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('app.utils.helpers').setLevel(logging.INFO)

load_dotenv()

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@db:5432/postgres'
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploaded_files")

os.makedirs(UPLOAD_DIR, exist_ok=True)
