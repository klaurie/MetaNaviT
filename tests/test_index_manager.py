import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from app.database.file_system_manager import FileSystemManager

# app/engine/database/test_index_manager.py


@pytest.fixture
def mock_env_vars():
    """Setup environment variables for testing"""
    with patch.dict(os.environ, {
        'PSYCOPG2_CONNECTION_STRING': 'postgresql://test:test@localhost:5432/test',
        'PG_CONNECTION_STRING': 'postgresql://test:test@localhost:5432/postgres'
    }):
        yield

@pytest.fixture
def mock_db_connection():
    """Mock database connection and cursor"""
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn
        yield mock_connect, mock_conn, mock_cur

@pytest.fixture
def index_manager(mock_env_vars, mock_db_connection):
    """Create FileSystemManager instance with mocked connections"""
    manager = FileSystemManager()
    yield manager
    manager.close()

def test_initialization(mock_env_vars, mock_db_connection):
    """Test FileSystemManager initialization"""
    manager = FileSystemManager()
    assert manager.block_hidden_files == True
    assert '/proc' in manager.blocked_dirs
    assert manager.conn is not None

def test_is_path_blocked(index_manager):
    """Test path blocking functionality"""
    # Test hidden file blocking
    assert index_manager._is_path_blocked('.hidden_file') == True
    
    # Test system directory blocking
    assert index_manager._is_path_blocked('/proc/test') == True
    assert index_manager._is_path_blocked('/home/user/test') == False

def test_insert_indexed_file(index_manager, mock_db_connection):
    """Test inserting a file into the database"""
    _, _, mock_cur = mock_db_connection
    
    index_manager.insert_indexed_file(
        file_path='/test/file.txt',
        process_name='test_process',
        process_version='1.0',
        mtime=123456789,
        data=b'test_data'
    )
    
    assert mock_cur.execute.called
    mock_cur.execute.assert_called_with(
        """
                    INSERT INTO indexed_files (file_path, process_name, process_version, mtime, data)
                    VALUES (%s, %s, %s, %s, %s);
                """,
        ('/test/file.txt', 'test_process', '1.0', 123456789, b'test_data')
    )

@patch('os.walk')
@patch('os.path.getmtime')
def test_crawl_file_system(mock_getmtime, mock_walk, index_manager):
    """Test filesystem crawling functionality"""
    mock_walk.return_value = [
        ('/test', ['dir1'], ['file1.txt', 'file2.txt']),
        ('/test/dir1', [], ['file3.txt'])
    ]
    mock_getmtime.return_value = 123456789
    
    batches = list(index_manager.crawl_file_system('/test', batch_size=2))
    
    assert len(batches) == 2
    assert len(batches[0]) == 2  # First batch with 2 files
    assert len(batches[1]) == 1  # Second batch with 1 file
    
    # Verify file info structure
    file_info = batches[0][0]
    assert 'pathname' in file_info
    assert 'modified_time' in file_info
    assert 'file_type' in file_info

def test_batch_insert_indexed_files(index_manager, mock_db_connection):
    """Test batch insertion of files"""
    _, _, mock_cur = mock_db_connection
    
    batch = [
        {
            'pathname': '/test/file1.txt',
            'modified_time': 123456789,
            'process_name': 'test_process',
            'process_version': '1.0',
            'data': b'test_data1'
        },
        {
            'pathname': '/test/file2.txt',
            'modified_time': 123456790,
            'process_name': 'test_process',
            'process_version': '1.0',
            'data': b'test_data2'
        }
    ]
    
    index_manager.batch_insert_indexed_files(batch)
    
    assert mock_cur.executemany.called

@pytest.mark.parametrize("conn_string", [
    None,
    "postgresql://test:test@localhost:5432/test"
])
def test_initialization_with_connection_string(conn_string, mock_env_vars):
    """Test initialization with different connection string scenarios"""
    if conn_string is None and 'PSYCOPG2_CONNECTION_STRING' not in os.environ:
        with pytest.raises(ValueError, match="Connection string is not provided."):
            FileSystemManager(conn_string)
    else:
        manager = FileSystemManager(conn_string)
        assert manager.conn_string is not None

def test_database_error_handling(mock_env_vars):
    """Test database connection error handling"""
    with patch('psycopg2.connect', side_effect=Exception("Connection failed")):
        with pytest.raises(Exception, match="Connection failed"):
            FileSystemManager()

def test_close_connection(index_manager, mock_db_connection):
    """Test database connection closing"""
    _, mock_conn, _ = mock_db_connection
    
    index_manager.close()
    assert mock_conn.close.called