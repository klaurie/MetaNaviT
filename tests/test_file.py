# tests/test_file.py

import pytest
from unittest.mock import MagicMock
from app.engine.loaders.file import get_file_documents, FileLoaderConfig
from app.config import DATA_DIR


def test_get_file_documents_success(mocker):
    # Arrange
    mock_SimpleDirectoryReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_SimpleDirectoryReader.return_value = mock_reader_instance
    
    # Mock the import statement in the function
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.core': MagicMock(),
        'llama_index.core.readers': MagicMock(SimpleDirectoryReader=mock_SimpleDirectoryReader)
    })
    
    mock_reader_instance.load_data.return_value = ["doc1", "doc2"]
    
    config = FileLoaderConfig()
    
    # Act
    documents = get_file_documents(config)
    
    # Assert
    mock_SimpleDirectoryReader.assert_called_once_with(
        DATA_DIR,
        recursive=True,
        filename_as_id=True,
        raise_on_error=True
    )
    assert documents == ["doc1", "doc2"]


def test_get_file_documents_empty_directory(mocker):
    # Arrange
    mock_SimpleDirectoryReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_SimpleDirectoryReader.return_value = mock_reader_instance
    
    # Mock the import statement in the function
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.core': MagicMock(),
        'llama_index.core.readers': MagicMock(SimpleDirectoryReader=mock_SimpleDirectoryReader)
    })
    
    # Mock traceback.extract_tb directly
    mock_traceback = mocker.patch('traceback.extract_tb')
    mock_frame = MagicMock()
    mock_frame.name = "_add_files"
    mock_traceback.return_value = [mock_frame]
    
    # Set up the exception
    mock_reader_instance.load_data.side_effect = Exception("_add_files")
    mock_logger = mocker.patch('app.engine.loaders.file.logger')
    
    config = FileLoaderConfig()
    
    # Act
    documents = get_file_documents(config)
    
    # Assert
    assert documents == []
    mock_logger.warning.assert_called_once_with(
        "Failed to load file documents, error message: _add_files . Return as empty document list."
    )


def test_get_file_documents_other_exception(mocker):
    # Arrange
    # Mock the import itself
    mock_module = MagicMock()
    mock_SimpleDirectoryReader = MagicMock()
    mock_module.SimpleDirectoryReader = mock_SimpleDirectoryReader
    mocker.patch.dict('sys.modules', {'llama_index.core.readers': mock_module})
    
    mock_reader_instance = mock_SimpleDirectoryReader.return_value
    mock_reader_instance.load_data.side_effect = Exception("Other error")
    
    config = FileLoaderConfig()
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        _ = get_file_documents(config)
    assert "Other error" in str(exc_info.value)


def test_get_file_documents_with_subfolders(mocker):
    # Arrange
    mock_SimpleDirectoryReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_SimpleDirectoryReader.return_value = mock_reader_instance
    
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.core': MagicMock(),
        'llama_index.core.readers': MagicMock(SimpleDirectoryReader=mock_SimpleDirectoryReader)
    })
    
    mock_reader_instance.load_data.return_value = ["doc1", "doc2", "doc3"]
    
    config = FileLoaderConfig()
    
    # Act
    documents = get_file_documents(config)
    
    # Assert
    mock_SimpleDirectoryReader.assert_called_once_with(
        DATA_DIR,
        recursive=True,  # Verify recursive is True
        filename_as_id=True,
        raise_on_error=True
    )
    assert documents == ["doc1", "doc2", "doc3"]


def test_get_file_documents_missing_directory(mocker):
    # Arrange
    mock_SimpleDirectoryReader = MagicMock()
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.core': MagicMock(),
        'llama_index.core.readers': MagicMock(SimpleDirectoryReader=mock_SimpleDirectoryReader)
    })
    
    mock_SimpleDirectoryReader.side_effect = FileNotFoundError("Directory not found")
    
    config = FileLoaderConfig()
    
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        get_file_documents(config)


def test_get_file_documents_invalid_format(mocker):
    # Arrange
    mock_SimpleDirectoryReader = MagicMock()
    mock_reader_instance = MagicMock()
    mock_SimpleDirectoryReader.return_value = mock_reader_instance
    
    mocker.patch.dict('sys.modules', {
        'llama_index': MagicMock(),
        'llama_index.core': MagicMock(),
        'llama_index.core.readers': MagicMock(SimpleDirectoryReader=mock_SimpleDirectoryReader)
    })
    
    mock_reader_instance.load_data.side_effect = ValueError("Unsupported file format")
    
    config = FileLoaderConfig()
    
    # Act & Assert
    with pytest.raises(ValueError):
        get_file_documents(config)
