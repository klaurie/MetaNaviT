"""
Query Filter Module

Handles document access control by managing public/private document filters
for vector store queries. Ensures proper document visibility based on 
specified document IDs.
"""
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters


def generate_filters(doc_ids):
    """
    Generate public/private document filters based on the doc_ids.
    
    Example Cases:
    1. generate_filters([])
       -> Returns only public documents
    2. generate_filters(["doc1", "doc2"])
       -> Returns public documents + documents with IDs "doc1" and "doc2"
    
    Args:
        doc_ids: List of document IDs to include in results
        
    Returns:
        MetadataFilters combining public access and specific document filters
    """
    public_doc_filter = MetadataFilter(
        key="private",
        value="true",
        operator="!=",  # type: ignore
    )
    selected_doc_filter = MetadataFilter(
        key="doc_id",
        value=doc_ids,
        operator="in",  # type: ignore
    )
    if len(doc_ids) > 0:
        # If doc_ids are provided, we will select both public and selected documents
        filters = MetadataFilters(
            filters=[
                public_doc_filter,
                selected_doc_filter,
            ],
            condition="or",  # type: ignore
        )
    else:
        filters = MetadataFilters(
            filters=[
                public_doc_filter,
            ]
        )

    return filters
