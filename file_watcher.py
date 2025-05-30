import logging
import time
import yaml
from pathlib import Path
import subprocess
from typing import Set, Dict, Any
import threading


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

from app.engine.generate import generate_datasource

logger = logging.getLogger(__name__)

# Global observer reference to maintain access to it
_observer = None
_observer_thread = None

class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events and triggers embedding generation."""
    
    def __init__(self):
        config = load_config()
        self.watch_path = config.get("path", "./")

        
    def on_modified(self, event):
        generate_datasource()
    
    def on_created(self, event):
        generate_datasource()
            


def load_config() -> Dict[str, Any]:
    """Load configuration from loaders.yaml"""
    config_path = Path("config/loaders.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config


def _watchdog_thread_func():
    """Function that runs in the watchdog thread"""
    global _observer
    
    try:
        # Create event handler
        event_handler = FileChangeHandler()
        
        # Create and start the observer
        _observer = Observer()
        _observer.schedule(event_handler, event_handler.watch_path, recursive=True)
        _observer.start()
        
        logger.info("File watcher started successfully")
        
        # Keep the thread alive
        try:
            while _observer.is_alive():
                _observer.join(1)
        except KeyboardInterrupt:
            pass
    except Exception as e:
        logger.error(f"Error in file watcher thread: {e}")


def start_file_watcher():
    """Start the file watcher in a background thread"""
    global _observer_thread
    
    if _observer_thread is not None and _observer_thread.is_alive():
        logger.warning("File watcher is already running")
        return
        
    logger.info("Starting file watcher service...")
    _observer_thread = threading.Thread(
        target=_watchdog_thread_func, 
        daemon=True  # This ensures the thread will exit when the main program exits
    )
    _observer_thread.start()


def stop_file_watcher():
    """Stop the file watcher"""
    global _observer, _observer_thread
    
    if _observer is not None:
        logger.info("Stopping file watcher...")
        _observer.stop()
        _observer.join()
        _observer = None
        
    if _observer_thread is not None:
        _observer_thread = None
        
    logger.info("File watcher stopped")

def main():
    """Main entry point for the file watcher"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Starting file watcher...")
    try:
        # Create event handler
        event_handler = FileChangeHandler()
        
        # Create and start the observer
        observer = Observer()
        observer.schedule(event_handler, event_handler.watch_path, recursive=True)
        observer.start()
        
        # Keep the thread alive and check for pending files periodically
        try:
            while observer.is_alive():
                observer.join(1)
                time.sleep(3)
        except KeyboardInterrupt:
            logger.info("File watcher stopping...")
        finally:
            observer.stop()
            observer.join()
            logger.info("File watcher stopped")
            
    except Exception as e:
        logger.error(f"Error in file watcher: {e}")


if __name__ == "__main__":
    main()