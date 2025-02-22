#!/usr/bin/env python3
"""
MetaNaviT Setup Script

This script handles:
1. Environment setup and validation
2. Database initialization
3. Model downloads
4. Configuration validation
5. Directory structure creation

Uses conda for environment management and requirements.txt for dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging
import psycopg2
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_DIRS = [
    "datasets",
    "datasets/file_organization",
    "datasets/file_organization/Easy",
    "storage",
    "output",
    "config"
]

REQUIRED_ENV_VARS = [
    "MODEL_PROVIDER",
    "MODEL",
    "EMBEDDING_MODEL",
    "PG_CONNECTION_STRING",
    "APP_HOST",
    "APP_PORT",
    "FRONTEND_DIR",
    "STATIC_DIR",
    "DATA_DIR",
    "STORAGE_DIR"
]

def create_env_file():
    """Create .env file if it doesn't exist with default values."""
    env_path = Path(".env")
    if not env_path.exists():
        logger.info("Creating default .env file")
        with open(env_path, "w") as f:
            f.write("""# MetaNaviT Environment Configuration
MODEL_PROVIDER=ollama
MODEL=llama2:7b
EMBEDDING_MODEL=all-MiniLM-L6-v2
PG_CONNECTION_STRING=postgresql://postgres:postgres@localhost:5432/metanavit
APP_HOST=localhost
APP_PORT=8000
FRONTEND_DIR=.frontend
STATIC_DIR=static
DATA_DIR=datasets
STORAGE_DIR=storage
""")
        logger.info("Created default .env file. Please update with your settings.")

def check_system_dependencies() -> List[str]:
    """Check if required system tools are installed."""
    missing = []
    dependencies = {
        "conda": "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh",
        "ollama": "curl -fsSL https://ollama.com/install.sh | sh",
        "psql": "sudo apt-get install postgresql postgresql-contrib"
    }
    
    for cmd in dependencies.keys():
        if not shutil.which(cmd):
            missing.append(cmd)
            logger.warning(f"Missing dependency: {cmd}")
            logger.info(f"Install with: {dependencies[cmd]}")
            
    return missing

def setup_conda_environment() -> None:
    """Set up conda environment with requirements."""
    try:
        # Create conda environment
        subprocess.run(
            ["conda", "create", "-n", "metanavit", "python=3.11", "-y"],
            check=True
        )
        
        # Activate the environment and install dependencies
        subprocess.run(
            ["conda", "run", "--name", "metanavit", "python", "-m", "pip", "install", "--upgrade", "pip"],
            check=True
        )

        subprocess.run(
            ["conda", "activate", "--name", "metanavit"],
            shell=True,
            check=True
        )

        subprocess.run("pip install -r requirements.txt",
        shell=True,
        check=True)

        subprocess.run("pip install -U deepeval",
        shell=True,
        check=True)
        
        logger.info("Conda environment setup complete")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conda environment setup failed: {e}")
        raise

def setup_directories() -> None:
    """Create required directory structure."""
    for dir_path in REQUIRED_DIRS:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def check_environment() -> bool:
    """Validate environment variables."""
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        return False
    return True


def main():
    """Main setup routine."""
    logger.info("Starting MetaNaviT setup...")
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    # Reload environment variables
    load_dotenv()
    
    # Check system dependencies
    missing_deps = check_system_dependencies()
    if missing_deps:
        logger.error("Please install missing dependencies before continuing")
        sys.exit(1)
    
    # Validate environment
    if not check_environment():
        logger.error("Please set required environment variables")
        sys.exit(1)
    
    # Create directory structure
    setup_directories()
    
    # Setup conda environment
    try:
        setup_conda_environment()
    except Exception as e:
        logger.error(f"Conda environment setup failed: {e}")
        sys.exit(1)
    
    logger.info("Setup completed successfully!")
    logger.info("To activate the environment, run: conda activate metanavit")

if __name__ == "__main__":
    main()
    sys.exit(0)
