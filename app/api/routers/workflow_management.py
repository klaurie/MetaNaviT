"""
Workflow Management Router

Provides endpoints for managing workflow state and human-in-the-loop interaction
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from llama_index.core.agent.workflow import Context, JsonSerializer

from app.api.routers.chat import chat_workflows, workflow_contexts

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/workflow/state")
async def save_workflow_state(
    session_id: str,
) -> Dict[str, Any]:
    """
    Serialize and return the current workflow state
    
    This enables pausing and resuming workflows for human-in-the-loop interactions
    """
    try:
        if session_id not in chat_workflows or session_id not in workflow_contexts:
            raise HTTPException(
                status_code=404, 
                detail=f"No active workflow found for session {session_id}"
            )
            
        # Get workflow and context
        workflow = chat_workflows[session_id]
        ctx = workflow_contexts[session_id]
        
        # Serialize context
        serialized_ctx = ctx.to_dict(serializer=JsonSerializer())
        
        return {
            "session_id": session_id,
            "state": serialized_ctx
        }
    
    except Exception as e:
        logger.exception(f"Error saving workflow state: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save workflow state: {str(e)}"
        )

@router.post("/api/workflow/respond")
async def respond_to_workflow(
    session_id: str,
    event_name: str,
    response: str,
    user_name: Optional[str] = None
):
    """
    Send a human response to a workflow
    
    Used for human-in-the-loop interactions where the workflow is waiting for input
    """
    try:
        if session_id not in chat_workflows or session_id not in workflow_contexts:
            raise HTTPException(
                status_code=404, 
                detail=f"No active workflow found for session {session_id}"
            )
        
        # Get context
        ctx = workflow_contexts[session_id]
        
        # Determine which event type to send
        if event_name == "human_response":
            from llama_index.core.agent.workflow import HumanResponseEvent
            
            # Send human response event
            ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=user_name or "user"
                )
            )
            
        # Add more event types as needed
        
        return {"status": "Response sent successfully"}
    
    except Exception as e:
        logger.exception(f"Error sending workflow response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send workflow response: {str(e)}"
        )