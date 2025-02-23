# tests/test_database.py

import pytest
from unittest.mock import MagicMock
from app.engine.loaders.db import get_db_documents, DBLoaderConfig


def test_get_db_documents_success(mocker):
    # Arrange
    # Mock the import itself
    mock_DatabaseReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_DatabaseReader.return_value = mock_reader_instance
    
    # Mock the import statement in the function
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.readers': MagicMock(),
        'llama_index.readers.database': MagicMock(DatabaseReader=mock_DatabaseReader)
    })
    
    # Set up the mock responses for each query
    mock_reader_instance.load_data.side_effect = [
        ["doc1_db1", "doc2_db1"],  # First query result
        ["doc1_db2"]               # Second query result
    ]
    
    config1 = DBLoaderConfig(uri="postgresql://user:pass@localhost/db1", queries=["SELECT * FROM table1"])
    config2 = DBLoaderConfig(uri="postgresql://user:pass@localhost/db2", queries=["SELECT * FROM table2"])
    configs = [config1, config2]
    
    # Act
    docs = get_db_documents(configs)
    
    # Assert
    assert docs == ["doc1_db1", "doc2_db1", "doc1_db2"]
    
    # Verify the DatabaseReader was called with correct URIs
    mock_DatabaseReader.assert_any_call(uri="postgresql://user:pass@localhost/db1")
    mock_DatabaseReader.assert_any_call(uri="postgresql://user:pass@localhost/db2")
    
    # Verify load_data was called with correct queries
    calls = mock_reader_instance.load_data.call_args_list
    assert calls[0][1]['query'] == "SELECT * FROM table1"
    assert calls[1][1]['query'] == "SELECT * FROM table2"


def test_get_db_documents_import_error(mocker):
    # Arrange
    mocker.patch('llama_index.readers.database.DatabaseReader', side_effect=ImportError)
    
    config = DBLoaderConfig(uri="postgresql://user:pass@localhost/db", queries=["SELECT * FROM table"])
    
    # Act & Assert
    with pytest.raises(ImportError):
        get_db_documents([config])


def test_get_db_documents_query_error(mocker):
    # Arrange
    mock_DatabaseReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_DatabaseReader.return_value = mock_reader_instance
    
    # Mock the import statement in the function
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.readers': MagicMock(),
        'llama_index.readers.database': MagicMock(DatabaseReader=mock_DatabaseReader)
    })
    
    mock_reader_instance.load_data.side_effect = Exception("Invalid query")
    
    config = DBLoaderConfig(uri="postgresql://user:pass@localhost/db", queries=["SELECT * FROM invalid_table"])
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        get_db_documents([config])
    assert str(exc_info.value) == "Invalid query"


def test_get_db_documents_empty_results(mocker):
    # Arrange
    mock_DatabaseReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_DatabaseReader.return_value = mock_reader_instance
    
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.readers': MagicMock(),
        'llama_index.readers.database': MagicMock(DatabaseReader=mock_DatabaseReader)
    })
    
    mock_reader_instance.load_data.return_value = []
    
    config = DBLoaderConfig(uri="postgresql://user:pass@localhost/db", queries=["SELECT * FROM table"])
    
    # Act
    docs = get_db_documents([config])
    
    # Assert
    assert docs == []


def test_get_db_documents_multiple_queries_per_db(mocker):
    # Arrange
    mock_DatabaseReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_DatabaseReader.return_value = mock_reader_instance
    
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.readers': MagicMock(),
        'llama_index.readers.database': MagicMock(DatabaseReader=mock_DatabaseReader)
    })
    
    mock_reader_instance.load_data.side_effect = [
        ["doc1"],
        ["doc2"],
        ["doc3"]
    ]
    
    config = DBLoaderConfig(
        uri="postgresql://user:pass@localhost/db",
        queries=[
            "SELECT * FROM table1",
            "SELECT * FROM table2",
            "SELECT * FROM table3"
        ]
    )
    
    # Act
    docs = get_db_documents([config])
    
    # Assert
    assert docs == ["doc1", "doc2", "doc3"]
    assert mock_reader_instance.load_data.call_count == 3
    
    calls = mock_reader_instance.load_data.call_args_list
    assert calls[0][1]['query'] == "SELECT * FROM table1"
    assert calls[1][1]['query'] == "SELECT * FROM table2"
    assert calls[2][1]['query'] == "SELECT * FROM table3"
