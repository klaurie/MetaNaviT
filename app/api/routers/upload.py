"""
File Upload Router Module
"""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.routers.models import DocumentFile
from app.engine.loaders.file import FileService

file_upload_router = r = APIRouter()

logger = logging.getLogger("uvicorn")


class FileUploadRequest(BaseModel):
    """
    File upload request structure:
    - base64: Encoded file content
    - name: Original filename
    - params: Optional processing parameters
    """
    base64: str
    name: str
    params: Any = None


@r.post("")
def upload_file(request: FileUploadRequest) -> DocumentFile:
    """
    Process private file uploads from chat UI.

    Raises:
        HTTPException: For processing errors
    """
    try:
        logger.info(f"Processing file: {request.name}")
        return FileService.process_private_file(
            request.name, request.base64, request.params
        )
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing file")
