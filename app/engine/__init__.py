"""
Engine Module

Provides document processing and index generation capabilities.
Exposes key functions for document ingestion and storage management.
"""

from app.engine.generate import generate_datasource

__all__ = ["generate_datasource"]