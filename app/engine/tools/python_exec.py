"""
Python Execution Tool

Provides the python engine tool with the capability of executing Python code.
This is used to complete specific tasks like analyzing data and solving problems.
Dynamically generated python code can be better than an LLM models capabilities. 
"""

# Standard library imports for file, system and text operations
import logging
import os
import io 
import contextlib 
from pathlib import Path  
from typing import Dict, Any, List, Optional  
from llama_index.core.tools import FunctionTool

# Initialize logger to record errors and information
logger = logging.getLogger(__name__)


class PythonExecTool:
    """Tool for executing Python code safely with constraints"""
    
    @classmethod
    def setup_workspace(cls, workspace_dir: Optional[str] = None) -> Path:
        """
        Set up and return the workspace directory
        
        This creates a directory where execution artifacts (saved code files,
        generated figures) will be stored.
        
        Args:
            workspace_dir: Custom directory path, or None to use default
            
        Returns:
            Path object pointing to the workspace directory
        """
        # Use provided directory or default to env variable or /tmp
        workspace_path = workspace_dir or os.path.join(os.getenv("STORAGE_DIR", "/tmp"), "dev_workspace")
        workspace_dir_path = Path(workspace_path)
        # Create directory if it doesn't exist
        workspace_dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PythonExecTool initialized with workspace at {workspace_dir_path}")
        return workspace_dir_path
    
    
    @classmethod
    def execute_code(cls, 
                     code: str,
                     workspace_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute Python code safely and return results.
        
        This is the main method that runs user code in a controlled environment,
        captures all output, and handles errors gracefully.
        
        Args:
            code: Python code to execute
            workspace_dir: Directory for saving execution artifacts
            
        Returns:
            Dictionary containing execution results with these keys:
            - output: Standard output captured during execution
            - error: Error message if execution failed
            - result: Value of 'result' variable if defined in the code
            - figure_path: Path to saved matplotlib figure if created
        """
        logger.info('Executing Python code')
        # Set up workspace directory to store files
        workspace_dir_path = cls.setup_workspace(workspace_dir)
        
        # Set up output capturing using StringIO objects
        # These act like files but store text in memory
        stdout_capture = io.StringIO()  # Captures print() output
        stderr_capture = io.StringIO()  # Captures error messages
        result = {"output": "", "error": "", "result": None}  # Initialize result dict
        
        try:
            # Save code to file for debugging purposes
            # Using id(code) to create a unique filename
            temp_file = workspace_dir_path / f"exec_{id(code)}.py"
            with open(temp_file, "w") as f:
                f.write(code)
                
            # Create a namespace for code execution
            # This is like a fresh environment where the code will run
            namespace = {}
            
            # Execute code with output redirection
            # The context managers temporarily redirect stdout and stderr
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code, namespace)  # Execute the code in the namespace
            
            # Collect the standard output
            result["output"] = stdout_capture.getvalue()
            
            # If the code defined a variable named "result", include it in the output
            if "result" in namespace:
                result["result"] = str(namespace["result"])
                
            # Check if the code created a matplotlib figure
            # If so, save it and include the path in the result
            if "plt" in namespace:
                try:
                    fig_file = workspace_dir_path / f"figure_{id(code)}.png"
                    namespace["plt"].savefig(fig_file)
                    result["figure_path"] = str(fig_file)
                except Exception as fig_error:
                    # If figure saving fails, log but don't crash
                    logger.warning(f"Could not save figure: {fig_error}")
                
            return result
            
        except Exception as e:
            # If anything goes wrong during execution, capture the error
            error_text = f"Error: {str(e)}\n{stderr_capture.getvalue()}"
            result["error"] = error_text
            logger.error(f"Python execution error: {error_text}")
            return result


def get_tools(**kwargs):
    """
    Create and return the Python execution tool.
    
    This function is the entry point called by the tool factory system.
    It creates and returns a FunctionTool that the agent can use.
        
    Returns:
        List containing the Python execution FunctionTool
    """

    # Return the tool in a list, as required by the tool factory system
    return [FunctionTool.from_defaults(PythonExecTool.execute_code)]
