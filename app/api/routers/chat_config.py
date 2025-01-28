"""
Chat Configuration Router Module

Provides API endpoints for retrieving chat interface configuration:
- Conversation starters/suggested questions
- Chat interface settings
- Environment-based customization
"""

import logging
import os

from fastapi import APIRouter, HTTPException
from app.api.routers.models import ChatConfig

# Initialize router with prefix handled by main app
config_router = r = APIRouter()

# Configure logging
logger = logging.getLogger("uvicorn")

@r.get("")
async def chat_config() -> ChatConfig:
    """
    Get chat interface configuration.
    
    Reads CONVERSATION_STARTERS from environment:
    - Each line becomes a starter question
    - Empty/missing env var means no starters
    - Whitespace is stripped from questions
    
    Returns:
        ChatConfig: Configuration including starter questions
    """
    # Initialize starters as None (optional field)
    starter_questions = None
    
    # Get starters from environment if configured
    conversation_starters = os.getenv("CONVERSATION_STARTERS")
    if conversation_starters and conversation_starters.strip():
        # Split on newlines to create question list
        starter_questions = conversation_starters.strip().split("\n")
        
    return ChatConfig(starter_questions=starter_questions)
