"""
OUTDATED NOT NEEDED (KEEPING JUST IN CASE)

Time-based Test Data Generator

Generates synthetic datasets with controlled timestamps for testing MetaNaviT's
temporal indexing capabilities. Creates files with known creation, modification,
and access times to validate time-based querying and organization.

Usage:
    python -m tests.data.generate_time_data

Dataset Structure:
    /test_data/
        /project_timeline/      # Project progression example
            - requirements_v1.txt
            - requirements_v2.txt
            - design_doc.txt
            - implementation_notes.txt
            - review_feedback.txt
            
        /system_logs/          # System monitoring example
            - system_20240301.log
            - system_20240302.log
            - backup_20240301.json
            - backup_20240302.json
            
        /meeting_records/      # Sequential meeting documents
            - meeting_2024Q1_01.txt
            - meeting_2024Q1_02.txt
            - summary_2024Q1.txt
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
        modified_delta: Optional[timedelta] = None
    ) -> None:
        """
        Create a file with specified content and timestamps
        
        Args:
            filepath: Path to create file
            content: File content
            created_delta: Timedelta from start date for creation time
            modified_delta: Timedelta from start date for modification time
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        # Set timestamps if specified
        if created_delta:
            created_time = (self.start_date + created_delta).timestamp()
            os.utime(filepath, (created_time, created_time))
            
        if modified_delta:
            modified_time = (self.start_date + modified_delta).timestamp()
            os.utime(filepath, (modified_time, modified_time))
            
        logger.info(f"Created {filepath}")

    def generate_project_timeline(self) -> None:
        """Generate project development timeline files"""
        project_dir = self.base_dir / "project_timeline"
        
        # Requirements versions
        self.create_file(
            project_dir / "requirements_v1.txt",
            "Initial requirements draft\n- Feature A\n- Feature B",
            created_delta=timedelta(days=0),
            modified_delta=timedelta(days=0)
        )
        
        self.create_file(
            project_dir / "requirements_v2.txt",
            "Updated requirements\n- Feature A+\n- Feature B\n- Feature C",
            created_delta=timedelta(days=2),
            modified_delta=timedelta(days=2)
        )
        
        # Design document
        self.create_file(
            project_dir / "design_doc.txt",
            "System architecture and component design...",
            created_delta=timedelta(days=3),
            modified_delta=timedelta(days=4)
        )
        
        # Implementation notes
        self.create_file(
            project_dir / "implementation_notes.txt",
            "Development progress and technical decisions...",
            created_delta=timedelta(days=5),
            modified_delta=timedelta(days=7)
        )
        
        # Review feedback
        self.create_file(
            project_dir / "review_feedback.txt",
            "Code review comments and suggestions...",
            created_delta=timedelta(days=8),
            modified_delta=timedelta(days=8)
        )

    def generate_system_logs(self) -> None:
        """Generate system monitoring data"""
        logs_dir = self.base_dir / "system_logs"
        
        # System logs
        for day in range(2):
            # Log file
            log_date = self.start_date + timedelta(days=day)
            log_content = f"[{log_date.isoformat()}] System events...\n"
            
            self.create_file(
                logs_dir / f"system_{log_date.strftime('%Y%m%d')}.log",
                log_content,
                created_delta=timedelta(days=day),
                modified_delta=timedelta(days=day)
            )
            
            # Backup data
            backup_data = {
                "timestamp": log_date.isoformat(),
                "system_state": "healthy",
                "metrics": {"cpu": 45, "memory": 60}
            }
            
            self.create_file(
                logs_dir / f"backup_{log_date.strftime('%Y%m%d')}.json",
                json.dumps(backup_data, indent=2),
                created_delta=timedelta(days=day),
                modified_delta=timedelta(days=day)
            )

    def generate_meeting_records(self) -> None:
        """Generate meeting documentation"""
        meetings_dir = self.base_dir / "meeting_records"
        
        # Individual meeting notes
        for meeting_num in range(1, 3):
            meeting_date = self.start_date + timedelta(days=meeting_num * 7)
            
            self.create_file(
                meetings_dir / f"meeting_2024Q1_{meeting_num:02d}.txt",
                f"Meeting notes from {meeting_date.strftime('%Y-%m-%d')}\n",
                created_delta=timedelta(days=meeting_num * 7),
                modified_delta=timedelta(days=meeting_num * 7)
            )
        
        # Quarterly summary
        self.create_file(
            meetings_dir / "summary_2024Q1.txt",
            "Q1 2024 Meeting Summaries and Action Items\n",
            created_delta=timedelta(days=14),
            modified_delta=timedelta(days=14)
        )

    def generate_all(self) -> None:
        """Generate complete test dataset"""
        self.setup_directory()
        self.generate_project_timeline()
        self.generate_system_logs()
        self.generate_meeting_records()
        
        # Create README
        readme_content = """# Time-Based Test Dataset

This dataset is designed to test MetaNaviT's temporal indexing capabilities.

## Directory Structure

### /project_timeline/
Sequential project documentation showing development progression.
Files are timestamped to reflect the natural evolution of a project.

### /system_logs/
System monitoring data with daily logs and backups.
Demonstrates handling of regularly generated technical data.

### /meeting_records/
Meeting documentation organized by date.
Shows relationships between periodic meetings and summary documents.

## Timestamp Patterns

- Creation times reflect when documents were initially created
- Modification times show updates and revisions
- Files within each category have meaningful temporal relationships
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