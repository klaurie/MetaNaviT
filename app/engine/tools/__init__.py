"""
Tool Factory Module

Manages loading and configuration of LLM tools from:
1. Local project tools (app.engine.tools)
2. LlamaHub tools (llama_index.tools)

Configuration (tools.yaml)
"""

import importlib
import os
from typing import Dict, List, Union

import yaml
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ToolType:
    """Define tool source locations"""
    # Could potentially use llama_index.tools as well
    LOCAL = "local"        # Local project tools


class ToolFactory:
    """Factory for loading and configuring LLM tools"""

    # Map tool types to their package paths
    TOOL_SOURCE_PACKAGE_MAP = {
        ToolType.LOCAL: "app.engine.tools"
    }

    @staticmethod
    def load_tools(
        tool_type: str,     # Source type (llamahub/local)
        tool_name: str,     # Tool module/class name
        config: dict        # Tool configuration
    ) -> List[FunctionTool]:
        """
        Load and configure tools from specified source.

        Note: ToolSpec classes define structured tools that can be registered with an LLM.
        
        Handles:
        1. ToolSpec classes (e.g. OpenAIToolSpec)
        2. Tool modules with get_tools()
        
        Raises:
            ValueError: Import or configuration errors
        """
        source_package = ToolFactory.TOOL_SOURCE_PACKAGE_MAP[tool_type]
        try:
            # Handle ToolSpec class loading
            if "ToolSpec" in tool_name:
                tool_package, tool_cls_name = tool_name.split(".")
                module_name = f"{source_package}.{tool_package}"
                module = importlib.import_module(module_name)
                tool_class = getattr(module, tool_cls_name)
                tool_spec: BaseToolSpec = tool_class(**config)
                return tool_spec.to_tool_list()
                
            # Handle tool module loading
            else:
                module = importlib.import_module(f"{source_package}.{tool_name}")
                tools = module.get_tools(**config)
                if not all(isinstance(tool, FunctionTool) for tool in tools):
                    raise ValueError(f"Invalid tools in {module}")
                return tools
                
        except ImportError as e:
            raise ValueError(f"Import failed for {tool_name}: {e}")
        except AttributeError as e:
            raise ValueError(f"Configuration failed for {tool_name}: {e}")

    @staticmethod
    def from_env(
        map_result: bool = False,  # Return dict instead of list
    ) -> Union[Dict[str, List[FunctionTool]], List[FunctionTool]]:
        """
        Load tools from YAML configuration.
        
        Flow:
        1. Read tools.yaml
        2. Process each tool config
        3. Load and configure tools
        4. Return as list or dict
        """
        # Initialize return structure based on map_result
        tools: Union[Dict[str, FunctionTool], List[FunctionTool]] = (
            {} if map_result else []
        )

        # Load and process tool configurations
        if os.path.exists("config/tools.yaml"):
            with open("config/tools.yaml", "r") as f:
                tool_configs = yaml.safe_load(f)
                for tool_type, config_entries in tool_configs.items():
                    for tool_name, config in config_entries.items():
                        loaded_tools = ToolFactory.load_tools(
                            tool_type, tool_name, config
                        )
                        # Add to dict or list based on map_result
                        if map_result:
                            tools.update(  # type: ignore
                                {tool.metadata.name: tool for tool in loaded_tools}
                            )
                        else:
                            tools.extend(loaded_tools)  # type: ignore

        return tools
