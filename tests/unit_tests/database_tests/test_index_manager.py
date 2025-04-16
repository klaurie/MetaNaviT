import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from app.database.index_manager import IndexManager

@pytest.fixture
def mock_db_connection():
    """Fixture to mock database connection initialization."""
    with patch('app.database.db_base_manager.DatabaseManager.__init__') as mock_init:
        mock_init.return_value = None
        yield mock_init

@pytest.fixture
def index_manager(mock_db_connection):
    """Fixture to create IndexManager instance with mocked dependencies."""
    with patch('app.database.index_manager.IndexManager._create_tables'):
        manager = IndexManager("postgresql://test:test@localhost:5432/test")
        return manager

def test_init(mock_db_connection):
    """
    Test IndexManager initialization.
    
    Verifies:
    1. Default settings are correctly set
    2. Required tables are created
    3. SQL creation queries are properly formatted
    """
    # Arrange
    mock_cursor = MagicMock()
    mock_connection = MagicMock()
    mock_connection.__enter__.return_value = mock_connection
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    with patch('app.database.index_manager.IndexManager.get_connection') as mock_get_conn:
        mock_get_conn.return_value = mock_connection
        
        # Act
        manager = IndexManager("postgresql://test:test@localhost:5432/test")
        
        # Assert
        assert manager.block_hidden_files is True
        assert '/proc' in manager.blocked_dirs
        assert len(manager.blocked_dirs) > 0
        
        # Verify table creation
        create_table_calls = mock_cursor.execute.call_args_list
        assert len(create_table_calls) == 2
        assert "CREATE TABLE IF NOT EXISTS indexed_files" in create_table_calls[0][0][0]
        assert "CREATE TABLE IF NOT EXISTS directory_processing_results" in create_table_calls[1][0][0]

def test_is_path_blocked_hidden_file(index_manager):
    """
    Test hidden file detection in path blocking.
    
    Verifies:
    1. Files starting with '.' are blocked
    2. Normal files are not blocked
    """
    # Test hidden file blocking
    assert index_manager.is_path_blocked('.hidden_file') is True
    assert index_manager.is_path_blocked('normal_file') is False

def test_is_path_blocked_system_dirs(index_manager):
    """
    Test system directory blocking functionality.
    
    Verifies:
    1. System paths (/proc, /sys) are blocked
    2. Regular user paths are not blocked
    """
    # Test system directory blocking
    assert index_manager.is_path_blocked('/proc/test') is True
    assert index_manager.is_path_blocked('/sys/test') is True
    assert index_manager.is_path_blocked('/home/user/test') is False

def test_is_path_blocked_wildcards(index_manager):
    """
    Test wildcard pattern matching in path blocking.
    
    Verifies:
    1. Paths matching wildcard patterns are blocked
    2. Parent paths of wildcards are not blocked
    3. Unrelated paths are not blocked
    """
    # Add a wildcard pattern
    index_manager.blocked_dirs.add('/test/*')
    
    # Test wildcard blocking
    assert index_manager.is_path_blocked('/test/subdir/file') is True
    assert index_manager.is_path_blocked('/test') is False
    assert index_manager.is_path_blocked('/other/test/file') is False

def test_batch_insert_indexed_files(index_manager):
    """
    Test batch insertion of file metadata.
    
    Verifies:
    1. SQL query formatting is correct
    2. Default values are properly handled
    3. Batch parameters are correctly structured
    4. UPSERT functionality is included
    """
    # Arrange
    mock_cursor = MagicMock()
    mock_connection = MagicMock()
    mock_connection.__enter__.return_value = mock_connection
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    
    with patch('app.database.index_manager.IndexManager.execute_query') as mock_execute:
        # Act
        batch = [
            {
                'pathname': '/path/file1',
                'process_name': 'test_process',
                'process_version': '1.0',
                'modified_time': 1234567890,
                'data': b'test_data'
            },
            {
                'pathname': '/path/file2',
                'modified_time': 1234567891
            }
        ]
        
        index_manager.batch_insert_indexed_files(batch)
        
        # Assert
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[0]
        
        # Check SQL query
        assert "INSERT INTO indexed_files" in call_args[0]
        assert "ON CONFLICT" in call_args[0]
        
        # Check parameters
        params = call_args[1]
        assert len(params) == 2
        assert params[0] == ('/path/file1', 'test_process', '1.0', 1234567890, b'test_data')
        assert params[1] == ('/path/file2', 'default', '1.0', 1234567891, None)

def test_is_path_blocked_empty_path(index_manager):
    """
    Test empty path handling in path blocking.
    
    Verifies:
    1. Empty paths raise ValueError
    2. Error handling is properly implemented
    """
    # Test empty path handling
    with pytest.raises(ValueError):
        index_manager.is_path_blocked('')

def test_batch_insert_empty_batch(index_manager):
    """
    Test handling of empty batches in file insertion.
    
    Verifies:
    1. Empty batches are handled gracefully
    2. No database queries are executed
    """
    # Test empty batch handling
    with patch('app.database.index_manager.IndexManager.execute_query') as mock_execute:
        index_manager.batch_insert_indexed_files([])
        mock_execute.assert_not_called()

def test_batch_insert_invalid_data(index_manager):
    """
    Test handling of invalid data in batch insertion.
    
    Verifies:
    1. Invalid data raises KeyError
    2. Missing required fields are caught
    3. Data validation is enforced
    """
    # Test invalid batch data
    with pytest.raises(KeyError):
        index_manager.batch_insert_indexed_files([{'invalid': 'data'}])