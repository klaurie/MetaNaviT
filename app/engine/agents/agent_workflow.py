import logging
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event

from typing import Any, List

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.agent.workflow.workflow_events import (
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentSetup,
    AgentOutput,
)

from llama_index.core.agent.workflow import FunctionAgent

# TODO:find out the actual values
DEFAULT_AGENT_NAME=""
DEFAULT_AGENT_DESCRIPTION=""

logger = logging.getLogger(__name__)

# ------------------ Define Event Classes ------------------
class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput

class LoopEvent(Event):
    loop_output: str

# ------------------ Configure Workflow and Steps ------------------
class MyWorkflow(Workflow):
    def __init__(
            self, 
            agents: List[FunctionAgent],
            *workflow_kwargs: Any
            ):
        super().__init__(timeout=timeout, **workflow_kwargs)
        if not agents:
            raise ValueError("At least one agent must be provided")

        # Raise an error if any agent has no name or no description
        if len(agents) > 1 and any(
            agent.name == DEFAULT_AGENT_NAME for agent in agents
        ):
            raise ValueError("All agents must have a name in a multi-agent workflow")

        if len(agents) > 1 and any(
            agent.description == DEFAULT_AGENT_DESCRIPTION for agent in agents
        ):
            raise ValueError(
                "All agents must have a description in a multi-agent workflow"
            )

        # Stor agents in a dictionary for lookup by name
        self.agents = {cfg.name: cfg for cfg in agents}

        # Determine starting agent 
        if len(agents) == 1:
            # Only one agent --> automatically root agent
            root_agent = agents[0].name
        elif root_agent is None:
            # If multiple, a root must be specified
            raise ValueError("Exactly one root agent must be provided")
        else:
            root_agent = root_agent

        # Ensure the root agent is actually in the list of provided agents
        if root_agent not in self.agents:
            raise ValueError(f"Root agent {root_agent} not found in provided agents")

        self.root_agent = root_agent
        self.initial_state = initial_state or {}

        # Configure handoff prompt
        handoff_prompt = handoff_prompt or DEFAULT_HANDOFF_PROMPT
        if isinstance(handoff_prompt, str):
            handoff_prompt = PromptTemplate(handoff_prompt)

            # Validate that the prompt contains the required {agent_info} placeholder
            if "{agent_info}" not in handoff_prompt.get_template():
                raise ValueError("Handoff prompt must contain {agent_info}")
        self.handoff_prompt = handoff_prompt

        # Configure handoff prompt
        handoff_output_prompt = handoff_output_prompt or DEFAULT_HANDOFF_OUTPUT_PROMPT
        if isinstance(handoff_output_prompt, str):
            handoff_output_prompt = PromptTemplate(handoff_output_prompt)
            # Validate that the prompt contains the required {to_agent} and {reason} placeholders
            if (
                "{to_agent}" not in handoff_output_prompt.get_template()
                or "{reason}" not in handoff_output_prompt.get_template()
            ):
                raise ValueError(
                    "Handoff output prompt must contain {to_agent} and {reason}"
                )
        self.handoff_output_prompt = handoff_output_prompt

        # Configure the state prompt
        state_prompt = state_prompt or DEFAULT_STATE_PROMPT
        if isinstance(state_prompt, str):
            state_prompt = PromptTemplate(state_prompt)
            # Validate that the prompt contains the required {state} and {msg} placeholders
            if (
                "{state}" not in state_prompt.get_template()
                or "{msg}" not in state_prompt.get_template()
            ):
                raise ValueError("State prompt must contain {state} and {msg}")
        self.state_prompt = state_prompt

    @step
    async def start_step(self,
                        ctx: Context,
                        ev: StartEvent) -> StopEvent:
        # prepare the context
        ctx.set("sources", [])  # clear sources

        # check if memory is setup
        memory = await ctx.get("memory", default=None)

        if not memory:
            memory = ChatMemoryBuffer(llm=self.llm)

        return StopEvent(result="Hello from start")
    
    @step
    async def run_agent_step(
        self,
        ctx: Context,
        ev: AgentSetup) -> AgentOutput
    )

async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())