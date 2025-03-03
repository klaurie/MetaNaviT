"""
Time-based Test Data Generator

Generates synthetic datasets with controlled timestamps for testing MetaNaviT's
temporal indexing capabilities. Creates files with known creation, modification,
and access times to validate time-based querying and organization.

Usage:
    python -m scripts.generate_time_data
"""

import os
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for test data
TEST_DATA_DIR = Path("data/time_test_data")

class TimeBasedDataGenerator:
    """Generates test data with controlled timestamps"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.start_date = datetime(2024, 3, 1)  # Base date for timestamps
        
    def setup_directory(self) -> None:
        """Create or clean the test data directory"""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.base_dir.mkdir(parents=True)
        
    def create_file(
        self,
        filepath: Path,
        content: str,
        created_delta: Optional[timedelta] = None,
        modified_delta: Optional[timedelta] = None,
        absolute_created_date: Optional[datetime] = None,
        absolute_modified_date: Optional[datetime] = None
    ) -> None:
        """
        Create a file with specified content and timestamps
        
        Args:
            filepath: Path to create file
            content: File content
            created_delta: Timedelta from start date for creation time
            modified_delta: Timedelta from start date for modification time
            absolute_created_date: Specific datetime for creation time (overrides delta)
            absolute_modified_date: Specific datetime for modification time (overrides delta)
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        # Set timestamps - prefer absolute dates if provided
        created_time = None
        modified_time = None
        
        if absolute_created_date:
            created_time = absolute_created_date.timestamp()
        elif created_delta:
            created_time = (self.start_date + created_delta).timestamp()
            
        if absolute_modified_date:
            modified_time = absolute_modified_date.timestamp()
        elif modified_delta:
            modified_time = (self.start_date + modified_delta).timestamp()
            
        if created_time and modified_time:
            os.utime(filepath, (created_time, modified_time))
        elif created_time:
            os.utime(filepath, (created_time, created_time))
            
        logger.info(f"Created {filepath} (created: {absolute_created_date or (self.start_date + created_delta) if created_delta else None}, " +
                   f"modified: {absolute_modified_date or (self.start_date + modified_delta) if modified_delta else None})")

    def generate_test_files(self) -> None:
        """Generate all test files in a flat structure to match test_cases.json"""
        
        # Project timeline files
        self.create_file(
            self.base_dir / "requirements_v1.txt",
            "Initial requirements draft\n- Feature A\n- Feature B",
            absolute_created_date=datetime(2024, 3, 1),
            absolute_modified_date=datetime(2024, 3, 1)
        )
        
        self.create_file(
            self.base_dir / "requirements_v2.txt",
            "Updated requirements\n- Feature A+\n- Feature B\n- Feature C",
            absolute_created_date=datetime(2024, 3, 3),
            absolute_modified_date=datetime(2024, 3, 3)
        )
        
        self.create_file(
            self.base_dir / "design_doc.txt",
            "System architecture and component design...",
            absolute_created_date=datetime(2024, 3, 5),
            absolute_modified_date=datetime(2024, 3, 5)
        )
        
        self.create_file(
            self.base_dir / "implementation_notes.txt",
            "Development progress and technical decisions...",
            absolute_created_date=datetime(2024, 3, 7),
            absolute_modified_date=datetime(2024, 3, 7)
        )
        
        self.create_file(
            self.base_dir / "review_feedback.txt",
            "Code review comments and suggestions...",
            absolute_created_date=datetime(2024, 3, 9),
            absolute_modified_date=datetime(2024, 3, 9)
        )
        
        # Meeting record files
        self.create_file(
            self.base_dir / "meeting_2024Q1_01.txt",
            "Meeting notes from 2024-01-15\n",
            absolute_created_date=datetime(2024, 1, 15),
            absolute_modified_date=datetime(2024, 1, 15)
        )
        
        self.create_file(
            self.base_dir / "meeting_2024Q1_02.txt",
            "Meeting notes from 2024-02-10\n",
            absolute_created_date=datetime(2024, 2, 10),
            absolute_modified_date=datetime(2024, 2, 10)
        )

    def generate_all(self) -> None:
        """Generate complete test dataset"""
        self.setup_directory()
        self.generate_test_files()
        
        # Create README
        readme_content = """# Time-Based Test Dataset

This dataset is designed to test MetaNaviT's temporal indexing capabilities.

## Timestamp Patterns

- Creation times reflect when documents were initially created
- Modification times show updates and revisions
- Files have meaningful temporal relationships

## File Organization

All files are placed in a flat structure to match the test_cases.json expectations.
"""

        self.create_file(
            self.base_dir / "README.md",
            readme_content
        )

def main():
    """Generate the test dataset"""
    generator = TimeBasedDataGenerator(TEST_DATA_DIR)
    generator.generate_all()
    logger.info(f"Test data generated in {TEST_DATA_DIR}")

if __name__ == "__main__":
    main()