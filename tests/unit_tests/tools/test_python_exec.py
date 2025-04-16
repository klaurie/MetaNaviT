import pytest
from unittest.mock import patch, mock_open, MagicMock, call
from pathlib import Path
import io
import os
from app.engine.tools.python_exec import PythonExecTool, get_tools

# filepath: app/engine/tools/test_python_exec.py


class TestPythonExecTool:

    def test_setup_workspace_default(self, mocker):
        # Arrange
        mock_path = mocker.patch('app.engine.tools.python_exec.Path')
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mocker.patch('app.engine.tools.python_exec.os.path.join', 
                    return_value='/tmp/dev_workspace')
        mocker.patch('app.engine.tools.python_exec.os.getenv', 
                    return_value=None)
        
        # Act
        result = PythonExecTool.setup_workspace()
        
        # Assert
        mock_path.assert_called_once_with('/tmp/dev_workspace')
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result == mock_path_instance

    def test_setup_workspace_custom_dir(self, mocker):
        # Arrange
        mock_path = mocker.patch('app.engine.tools.python_exec.Path')
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        
        # Act
        result = PythonExecTool.setup_workspace('/custom/workspace')
        
        # Assert
        mock_path.assert_called_once_with('/custom/workspace')
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result == mock_path_instance

    def test_execute_code_basic(self, mocker):
        # Arrange
        mocker.patch('app.engine.tools.python_exec.PythonExecTool.setup_workspace',
                    return_value=Path('/fake/workspace'))
        mocker.patch('builtins.open', mock_open())
        test_code = 'print("Hello, world!")'
        
        # Act
        result = PythonExecTool.execute_code(test_code)
        
        # Assert
        assert result["output"] == "Hello, world!\n"
        assert result["error"] == ""
        assert result["result"] is None

    def test_execute_code_with_result(self, mocker):
        # Arrange
        mocker.patch('app.engine.tools.python_exec.PythonExecTool.setup_workspace',
                    return_value=Path('/fake/workspace'))
        mocker.patch('builtins.open', mock_open())
        test_code = 'result = 42'
        
        # Act
        result = PythonExecTool.execute_code(test_code)
        
        # Assert
        assert result["output"] == ""
        assert result["error"] == ""
        assert result["result"] == "42"

    def test_execute_code_with_error(self, mocker):
        # Arrange
        mocker.patch('app.engine.tools.python_exec.PythonExecTool.setup_workspace',
                    return_value=Path('/fake/workspace'))
        mocker.patch('builtins.open', mock_open())
        test_code = 'x = 1/0'
        
        # Act
        result = PythonExecTool.execute_code(test_code)
        
        # Assert
        assert result["output"] == ""
        assert "division by zero" in result["error"]
        assert "Error:" in result["error"]

    def test_execute_code_with_matplotlib(self, mocker):
        # Arrange
        workspace_path = Path('/fake/workspace')
        mocker.patch('app.engine.tools.python_exec.PythonExecTool.setup_workspace',
                    return_value=workspace_path)
        mocker.patch('builtins.open', mock_open())
        
        # Mock exec to create a plt object in namespace
        def mock_exec(code, namespace):
            namespace['plt'] = MagicMock()
        
        mocker.patch('builtins.exec', side_effect=mock_exec)
        test_code = 'import matplotlib.pyplot as plt'
        
        # Act
        result = PythonExecTool.execute_code(test_code)
        
        # Assert
        assert "figure_path" in result
        assert str(workspace_path) in result["figure_path"]

    def test_execute_code_with_matplotlib_error(self, mocker):
        # Arrange
        mocker.patch('app.engine.tools.python_exec.PythonExecTool.setup_workspace',
                    return_value=Path('/fake/workspace'))
        mocker.patch('builtins.open', mock_open())
        
        # Create mock plt that raises exception on savefig
        mock_plt = MagicMock()
        mock_plt.savefig.side_effect = Exception("Figure error")
        
        def mock_exec(code, namespace):
            namespace['plt'] = mock_plt
        
        mocker.patch('builtins.exec', side_effect=mock_exec)
        test_code = 'import matplotlib.pyplot as plt'
        mock_logger = mocker.patch('app.engine.tools.python_exec.logger')
        
        # Act
        result = PythonExecTool.execute_code(test_code)
        
        # Assert
        assert "figure_path" not in result
        mock_logger.warning.assert_called_once()
        assert "Figure error" in mock_logger.warning.call_args[0][0]

    def test_get_tools(self, mocker):
        # Arrange
        mock_function_tool = MagicMock()
        mocker.patch('app.engine.tools.python_exec.FunctionTool.from_defaults', 
                    return_value=mock_function_tool)
        
        # Act
        tools = get_tools()
        
        # Assert
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0] == mock_function_tool