import pytest
from unittest.mock import MagicMock, patch
from app.engine.agents.file_access_agent import create_file_access_agent, FILE_ACCESS_PROMPT

# filepath: app/engine/agents/test_file_access_agent.py



class TestFileAccessAgent:
    
    def test_create_file_access_agent_filters_python_exec(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        
        # Create mock tools
        python_exec_tool = MagicMock()
        python_exec_tool.metadata = {"name": "python_exec"}
        
        other_tool = MagicMock()
        other_tool.metadata = {"name": "other_tool"}
        
        # Configure mocks
        mock_tool_factory.from_env.return_value = [python_exec_tool, other_tool]
        mock_get_file_tools.return_value = []
        
        # Act
        create_file_access_agent()
        
        # Assert
        # Verify the python_exec tool wasn't included in the tools passed to FunctionAgent
        tools_arg = mock_function_agent.call_args[1]['tools']
        tool_names = [tool.metadata.get("name") for tool in tools_arg]
        assert "python_exec" not in tool_names
        assert "other_tool" in tool_names
    
    def test_create_file_access_agent_combines_tools(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        
        # Create mock tools
        file_tool = MagicMock()
        file_tool.metadata = {"name": "file_tool"}
        
        other_tool = MagicMock()
        other_tool.metadata = {"name": "other_tool"}
        
        # Configure mocks
        mock_tool_factory.from_env.return_value = [other_tool]
        mock_get_file_tools.return_value = [file_tool]
        
        # Act
        create_file_access_agent()
        
        # Assert
        # Verify both tools were included
        tools_arg = mock_function_agent.call_args[1]['tools']
        tool_names = [tool.metadata.get("name") for tool in tools_arg]
        assert "file_tool" in tool_names
        assert "other_tool" in tool_names
        assert len(tools_arg) == 2
    
    def test_create_file_access_agent_prevents_duplicate_tools(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        
        # Create mock tools with same name
        file_tool = MagicMock()
        file_tool.metadata = {"name": "duplicate_tool"}
        
        other_tool = MagicMock()
        other_tool.metadata = {"name": "duplicate_tool"}
        
        # Configure mocks
        mock_tool_factory.from_env.return_value = [other_tool]
        mock_get_file_tools.return_value = [file_tool]
        
        # Act
        create_file_access_agent()
        
        # Assert
        # Verify only one instance of the tool was included
        tools_arg = mock_function_agent.call_args[1]['tools']
        tool_names = [tool.metadata.get("name") for tool in tools_arg]
        assert tool_names.count("duplicate_tool") == 1
        assert len(tools_arg) == 1
    
    def test_create_file_access_agent_with_custom_llm(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        mock_settings = mocker.patch('app.engine.agents.file_access_agent.Settings')
        
        mock_tool_factory.from_env.return_value = []
        mock_get_file_tools.return_value = []
        
        custom_llm = MagicMock()
        
        # Act
        create_file_access_agent(llm_interface=custom_llm)
        
        # Assert
        # Verify the custom LLM was used
        assert mock_function_agent.call_args[1]['llm'] == custom_llm
        assert mock_settings.llm != mock_function_agent.call_args[1]['llm']
    
    def test_create_file_access_agent_uses_default_llm(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        mock_settings = mocker.patch('app.engine.agents.file_access_agent.Settings')
        
        mock_tool_factory.from_env.return_value = []
        mock_get_file_tools.return_value = []
        
        # Act
        create_file_access_agent()
        
        # Assert
        # Verify the default LLM from settings was used
        assert mock_function_agent.call_args[1]['llm'] == mock_settings.llm
    
    def test_create_file_access_agent_correct_parameters(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        
        mock_tool_factory.from_env.return_value = []
        mock_get_file_tools.return_value = []
        
        # Act
        create_file_access_agent()
        
        # Assert
        # Verify the agent was created with correct parameters
        assert mock_function_agent.call_args[1]['name'] == "FileAccessAgent"
        assert mock_function_agent.call_args[1]['description'] == "Retrieves file contents and uses all tools except code execution"
        assert mock_function_agent.call_args[1]['system_prompt'] == FILE_ACCESS_PROMPT
        assert mock_function_agent.call_args[1]['can_handoff_to'] == ["PythonCodeAgent"]
    
    def test_create_file_access_agent_logging(self, mocker):
        # Arrange
        mock_tool_factory = mocker.patch('app.engine.agents.file_access_agent.ToolFactory')
        mock_get_file_tools = mocker.patch('app.engine.agents.file_access_agent.get_file_tools')
        mock_function_agent = mocker.patch('app.engine.agents.file_access_agent.FunctionAgent')
        mock_logger = mocker.patch('app.engine.agents.file_access_agent.logger')
        
        tool1 = MagicMock()
        tool1.metadata = {"name": "tool1"}
        tool2 = MagicMock()
        tool2.metadata = {"name": "tool2"}
        
        mock_tool_factory.from_env.return_value = [tool1]
        mock_get_file_tools.return_value = [tool2]
        
        # Act
        create_file_access_agent()
        
        # Assert
        # Verify logging occurred with correct tool count
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "2 tools" in log_message