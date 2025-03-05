"""
Agent Workflow Module

Manages the multi-agent workflow between the file access agent
and the Python code generator agent.
"""

import logging
from typing import Optional

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.callbacks import CallbackManager

from app.config.settings import Settings
from app.engine.agents.file_access_agent import create_file_access_agent
from app.engine.agents.python_code_agent import create_python_code_agent

logger = logging.getLogger(__name__)

def create_agent_workflow(
    llm_interface=None,
    callback_manager: Optional[CallbackManager] = None,
) -> AgentWorkflow:
    """
    Create a multi-agent workflow that manages file access and code execution.
    
    Args:
        llm_interface: Language model interface to use
        callback_manager: Callback manager for events
        
    Returns:
        AgentWorkflow configured with file access and Python code agents
    """
    # Create both agents
    file_agent = create_file_access_agent(
        callback_manager=callback_manager
    )
    
    python_agent = create_python_code_agent(
        callback_manager=callback_manager
    )
    
    # Create workflow
    workflow = AgentWorkflow(
        agents=[file_agent, python_agent],
        root_agent=file_agent.name,  # Start with file access agent
        initial_state={
            "file_content": None,
            "code_executed": False,
            "execution_result": None,
        },
        verbose=True
    )
    
    logger.info("Created multi-agent workflow with file access and Python code agents")
    return workflow