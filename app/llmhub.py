import logging
import os
from typing import Dict

from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class TSIEmbedding(OpenAIEmbedding):
    """Custom embedding class that uses the model name for both query and text engines"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._query_engine = self._text_engine = self.model_name


def llm_config_from_env() -> Dict:
    """
    Retrieves LLM configuration from environment variables.
    Returns a dictionary with model settings including:
    - model name
    - API key and base URL
    - temperature
    - max tokens
    """
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("MODEL", DEFAULT_MODEL)
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")
    api_key = os.getenv("T_SYSTEMS_LLMHUB_API_KEY")
    api_base = os.getenv("T_SYSTEMS_LLMHUB_BASE_URL")

    config = {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
    return config


def embedding_config_from_env() -> Dict:
    """
    Retrieves embedding configuration from environment variables.
    Returns a dictionary with embedding settings including:
    - model name
    - dimension
    - API credentials
    """
    from llama_index.core.constants import DEFAULT_EMBEDDING_DIM

    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    dimension = os.getenv("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    api_key = os.getenv("T_SYSTEMS_LLMHUB_API_KEY")
    api_base = os.getenv("T_SYSTEMS_LLMHUB_BASE_URL")

    config = {
        "model_name": model,
        "dimension": int(dimension) if dimension is not None else None,
        "api_key": api_key,
        "api_base": api_base,
    }
    return config


def init_llmhub():
    """
    Initializes the LLM environment by:
    1. Importing required OpenAILike class
    2. Getting configurations for LLM and embeddings
    3. Setting up global Settings with the configured models
    """
    try:
        from llama_index.llms.openai_like import OpenAILike
    except ImportError:
        logger.error("Failed to import OpenAILike. Make sure llama_index is installed.")
        raise

    llm_configs = llm_config_from_env()
    embedding_configs = embedding_config_from_env()

    Settings.embed_model = TSIEmbedding(**embedding_configs)

    # OpenAILike is a flexible adapter class that allows:
    # - Using OpenAI-compatible APIs from different providers
    # - Supporting both hosted and self-hosted LLM services
    # - Maintaining OpenAI-style interface while switching backends
    # - Easy integration with services that implement OpenAI's API spec
    Settings.llm = OpenAILike(
        **llm_configs,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=4096,
    )
