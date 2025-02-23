# tests/test_file.py

from unittest.mock import MagicMock
from app.engine.generate import get_doc_store, run_pipeline, generate_datasource


def test_get_doc_store_no_storage_dir(mocker):
    # Arrange
    mocker.patch.dict('os.environ', {}, clear=True)  # Clear STORAGE_DIR
    mocker.patch('os.path.exists', return_value=False)  # Mock path.exists to return False
    mock_simple_store = mocker.patch('app.engine.generate.SimpleDocumentStore')
    
    # Act
    _ = get_doc_store()
    
    # Assert
    mock_simple_store.assert_called_once()


def test_get_doc_store_with_storage_dir(mocker):
    # Arrange
    mocker.patch('app.engine.generate.os.path.exists', return_value=True)
    mock_simple_store = mocker.patch('app.engine.generate.SimpleDocumentStore')
    
    # Act
    _ = get_doc_store()
    
    # Assert
    mock_simple_store.from_persist_dir.assert_called_once()


def test_run_pipeline_success(mocker):
    # Arrange
    # Mock the imports and Settings
    mocker.patch('app.engine.generate.Settings.embed_model', MagicMock())
    mocker.patch('app.engine.generate.Settings.chunk_size', 1000)
    mocker.patch('app.engine.generate.Settings.chunk_overlap', 200)
    
    # Mock the pipeline
    mock_pipeline = mocker.patch('app.engine.generate.IngestionPipeline')
    mock_pipeline_instance = mock_pipeline.return_value
    mock_pipeline_instance.run.return_value = ["processed_doc1", "processed_doc2"]
    
    docstore = MagicMock()
    vector_store = MagicMock()
    documents = ["doc1", "doc2"]
    
    # Act
    _ = run_pipeline(docstore, vector_store, documents)
    
    # Assert
    mock_pipeline.assert_called_once()
    mock_pipeline_instance.run.assert_called_once_with(
        show_progress=True,
        documents=documents
    )


def test_generate_datasource_success(mocker):
    # Arrange
    # Mock init_settings to prevent Ollama import error
    mocker.patch('app.engine.generate.init_settings')
    
    # Create mock documents with proper metadata dictionaries
    mock_doc1 = MagicMock()
    mock_doc2 = MagicMock()
    mock_doc1.metadata = {}
    mock_doc2.metadata = {}
    mock_docs = [mock_doc1, mock_doc2]
    
    # Set up all the mocks
    mock_get_documents = mocker.patch('app.engine.generate.get_documents', return_value=mock_docs)
    mock_get_doc_store = mocker.patch('app.engine.generate.get_doc_store')
    mock_get_vector_store = mocker.patch('app.engine.generate.get_vector_store')
    mock_run_pipeline = mocker.patch('app.engine.generate.run_pipeline')
    mock_persist_storage = mocker.patch('app.engine.generate.persist_storage')
    
    # Act
    generate_datasource()
    
    # Assert
    mock_get_documents.assert_called_once()
    mock_get_doc_store.assert_called_once()
    mock_get_vector_store.assert_called_once()
    mock_run_pipeline.assert_called_once()
    mock_persist_storage.assert_called_once()
    
    # Verify metadata was set correctly
    assert mock_doc1.metadata["private"] == "false"
    assert mock_doc2.metadata["private"] == "false"
