# AgentWorkflow Class Overview

The `AgentWorkflow` class manages a multi-agent system that enables specialized agents to collaborate and hand off tasks to one another. It extends the base `LlamaIndex` Workflow system with streaming support and human interaction capabilities.

---

## Core Functions

| Function              | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `__init__`            | Initializes the workflow with agents, prompts, and configuration. Validates agent setup and configures handoff capabilities. |
| `run`                 | Executes the workflow with user message and chat history, returning a handler that manages the workflow execution. |
| `from_tools_or_functions` | Creates a workflow with a single agent using provided tools or functions. Automatically selects between `FunctionAgent` or `ReActAgent` based on LLM capabilities. |

---

## Prompt Management Functions

| Function              | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `_get_prompts`        | Returns the prompt templates used by the workflow (handoff, state, etc.). |
| `_get_prompt_modules` | Returns the agents as prompt modules for prompt management.             |
| `_update_prompts`     | Updates the workflow prompts when prompt templates change.              |

---

## Tool Management Functions

| Function              | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `_ensure_tools_are_async` | Converts any synchronous tools to asynchronous versions for consistent handling. |
| `_get_handoff_tool`   | Creates a specialized handoff tool for agents to transfer control to other agents. |
| `get_tools`           | Gets all applicable tools for a specific agent, including agent-specific tools, retrieved tools, and handoff tools. |
| `_call_tool`          | Executes a tool with given input, handling context requirements and exceptions. |

---

## Context Management Functions

| Function              | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `_init_context`       | Sets up the workflow context with memory, agent information, state, and configuration. |

---

## Workflow Step Functions (Execution Pipeline)

| Function              | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `init_run`            | Sets up the workflow run, processes user input and chat history, and initializes the first agent. |
| `setup_agent`         | Prepares the current agent for execution with appropriate system prompt and input. |
| `run_agent_step`      | Executes the current agent with available tools and captures the output. |
| `parse_agent_output`  | Processes the agent's response to either return a final answer or execute tool calls. |
| `call_tool`           | Calls a specific tool with the given parameters and handles the result. |
| `aggregate_tool_results` | Processes tool outputs, handles agent handoffs, and determines the next step in the workflow. |

---

## External Helper Function

| Function              | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `handoff`             | An async function that manages the transfer of control between agents, validating the handoff is allowed and formatting the handoff message. |

---