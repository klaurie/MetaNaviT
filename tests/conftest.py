import sys
import os
import pytest
from unittest.mock import MagicMock

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock all required llama_index modules
mock_modules = {
    'psycopg2': MagicMock(),
    'psycopg2.sql': MagicMock(),
    'psycopg2.pool': MagicMock(),
    'llama_index': MagicMock(),
    'llama_index.core': MagicMock(),
    'llama_index.core.readers': MagicMock(),
    'llama_index.core.readers.base': MagicMock(),
    'llama_index.core.readers.database': MagicMock(),
    'llama_index.core.indices': MagicMock(),
    'llama_index.core.indices.base': MagicMock(),
    'llama_index.core.indices.composability': MagicMock(),
    'llama_index.core.indices.composability.graph': MagicMock(),
    'llama_index.core.ingestion': MagicMock(),
    'llama_index.core.ingestion.pipeline': MagicMock(),
    'llama_index.core.multi_modal_llms': MagicMock(),
    'llama_index.core.settings': MagicMock(),
    'llama_index.vector_stores': MagicMock(),
    'llama_index.core.storage': MagicMock(),
    'llama_index.core.storage.docstore': MagicMock(),

    'llama_index.core.node_parser': MagicMock(),
    'llama_index.vector_stores.postgres': MagicMock(),
}

# Create MultiModalLLM mock
mock_multi_modal_llm = MagicMock()
mock_modules['llama_index.core.multi_modal_llms'].MultiModalLLM = mock_multi_modal_llm

# Apply all mocks
for mod_name, mock in mock_modules.items():
    sys.modules[mod_name] = mock


@pytest.fixture
def mock_vector_store(mocker):
    """Fixture to mock the vector store."""
    mock_store = MagicMock()
    mocker.patch('app.database.vector_store.get_vector_store', return_value=mock_store)
    return mock_store


@pytest.fixture
def mock_database_reader(mocker):
    """Fixture to mock the DatabaseReader."""
    mock_reader = MagicMock()
    mocker.patch('llama_index.readers.database.DatabaseReader', return_value=mock_reader)
    return mock_reader


@pytest.fixture(autouse=True)
def mock_env_vars(mocker):
    """Mock environment variables."""
    mocker.patch.dict('os.environ', {
        'PG_CONNECTION_STRING': 'postgresql://user:pass@localhost:5432/test',
        'STORAGE_DIR': 'test_storage',
        'DATA_DIR': '/test_data'
    })


@pytest.fixture(autouse=True)
def mock_llama_index(mocker):
    """Mock llama_index components."""
    mocker.patch('llama_index.core.indices.VectorStoreIndex')
    mocker.patch('llama_index.core.storage.StorageContext')
    mocker.patch('llama_index.core.Document', return_value=MagicMock())
    mocker.patch('llama_index.core.settings.Settings')
    mocker.patch('llama_index.core.multi_modal_llms.MultiModalLLM')