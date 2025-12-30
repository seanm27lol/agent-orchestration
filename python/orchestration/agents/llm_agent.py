"""
LLM-Powered Agent

An agent implementation that uses language models for reasoning and decision-making.
Supports tool use, multi-turn conversations, and streaming responses.
"""

import json
import re
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field

from .base import (
    Agent,
    AgentBuilder,
    AgentCapability,
    AgentStatus,
    AgentObservation,
    AgentAction,
    ToolCall,
)
from ..bridge.model_serving import (
    ModelServer,
    InferenceRequest,
    InferenceResponse,
    global_model_server,
)
from ..tools.base import Tool, ToolContext, ToolResult


# ============================================================================
# Tool Definitions for LLM
# ============================================================================

@dataclass
class ToolDefinitionForLLM:
    """Tool definition formatted for LLM consumption."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_prompt_format(self) -> str:
        """Format tool for inclusion in system prompt."""
        params_str = json.dumps(self.parameters, indent=2)
        return f"""## {self.name}
{self.description}

Parameters:
```json
{params_str}
```"""


# ============================================================================
# LLM Agent
# ============================================================================

class LLMAgent(Agent):
    """
    Agent powered by a language model.

    Features:
    - Multi-turn conversation with memory
    - Tool use with automatic parsing and execution
    - Configurable system prompts
    - Support for multiple model providers
    """

    def __init__(
        self,
        agent_id: str,
        capabilities: Optional[List[AgentCapability]] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_server: Optional[ModelServer] = None,
        **kwargs,
    ):
        default_capabilities = [
            AgentCapability.TEXT_GENERATION,
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
        ]
        super().__init__(
            agent_id=agent_id,
            capabilities=capabilities or default_capabilities,
            name=name,
            config=config,
            **kwargs,
        )

        self.model_server = model_server or global_model_server
        self.model_id = (config or {}).get("model_id", None)
        self.provider = (config or {}).get("provider", None)
        self.system_prompt = (config or {}).get(
            "system_prompt",
            "You are a helpful AI assistant. You can use tools to accomplish tasks."
        )
        self.temperature = (config or {}).get("temperature", 0.7)
        self.max_tokens = (config or {}).get("max_tokens", 2048)

        # Tool registry for this agent
        self._tools: Dict[str, Tool] = {}
        self._tool_definitions: List[ToolDefinitionForLLM] = []

        # Conversation history
        self._messages: List[Dict[str, str]] = []

        # Max iterations to prevent infinite loops
        self._max_tool_iterations = (config or {}).get("max_tool_iterations", 10)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self._tools[tool.name] = tool

        # Create LLM-friendly definition
        params_schema = {
            "type": "object",
            "properties": {
                p.name: {"type": p.type, "description": p.description}
                for p in tool.parameters
            },
            "required": [p.name for p in tool.parameters if p.required],
        }

        self._tool_definitions.append(ToolDefinitionForLLM(
            name=tool.name,
            description=tool.description,
            parameters=params_schema,
        ))

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including tool definitions."""
        if not self._tool_definitions:
            return self.system_prompt

        tools_section = "\n\n".join(
            t.to_prompt_format() for t in self._tool_definitions
        )

        return f"""{self.system_prompt}

# Available Tools

You can use the following tools by responding with a JSON block in this format:
```tool
{{"tool": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
```

{tools_section}

When you need to use a tool, respond ONLY with the tool call JSON block.
After receiving the tool result, continue your response."""

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        tool_calls = []

        # Look for ```tool blocks
        tool_pattern = r'```tool\s*\n?(.*?)\n?```'
        matches = re.findall(tool_pattern, content, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if "tool" in parsed:
                    tool_calls.append(parsed)
            except json.JSONDecodeError:
                continue

        # Also try to parse standalone JSON objects
        if not tool_calls:
            json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
            matches = re.findall(json_pattern, content)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if "tool" in parsed:
                        tool_calls.append(parsed)
                except json.JSONDecodeError:
                    continue

        return tool_calls

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool and return the result."""
        if tool_name not in self._tools:
            from ..tools.base import ToolResult as TR
            return TR(
                call_id="error",
                tool_name=tool_name,
                success=False,
                error={"code": "TOOL_NOT_FOUND", "message": f"Tool '{tool_name}' not found"},
            )

        tool = self._tools[tool_name]
        context = ToolContext(
            agent_id=self.agent_id,
            execution_id=None,
            workflow_id=None,
        )

        return await tool.invoke(context, **arguments)

    async def initialize(self) -> AgentObservation:
        """Initialize the agent."""
        self._status = AgentStatus.RUNNING
        self._messages = [{"role": "system", "content": self._build_system_prompt()}]

        return AgentObservation(
            source="system",
            data={
                "status": "initialized",
                "model": self.model_id,
                "tools": [t.name for t in self._tool_definitions],
            },
            observation_type="state_update",
        )

    async def step(self, observation: AgentObservation) -> AgentAction:
        """Process an observation and produce an action."""
        self._step_count += 1

        # Add observation to messages
        if observation.observation_type == "tool_result":
            # Tool result - add as assistant message
            self._messages.append({
                "role": "assistant",
                "content": f"Tool result: {json.dumps(observation.data)}",
            })
        else:
            # Regular message
            content = observation.data
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)

            self._messages.append({
                "role": "user",
                "content": content,
            })

        # Run inference loop with tool use
        iteration = 0
        while iteration < self._max_tool_iterations:
            iteration += 1

            # Call model
            request = InferenceRequest(
                model_id=self.model_id,
                messages=self._messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            response = await self.model_server.infer(request, provider=self.provider)

            if not response.success:
                return AgentAction(
                    action_type="error",
                    data={"error": response.error},
                    reasoning="Model inference failed",
                )

            content = response.content or ""

            # Check for tool calls
            tool_calls = self._parse_tool_calls(content)

            if not tool_calls:
                # No tool calls - return the response
                self._messages.append({"role": "assistant", "content": content})

                return AgentAction(
                    action_type="message",
                    data=content,
                    reasoning=f"Generated response after {iteration} iteration(s)",
                    metadata={
                        "tokens": response.total_tokens,
                        "latency_ms": response.latency_ms,
                    },
                )

            # Execute tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool", "")
                arguments = tool_call.get("arguments", {})

                result = await self._execute_tool(tool_name, arguments)

                # Add tool call and result to messages
                self._messages.append({
                    "role": "assistant",
                    "content": f"```tool\n{json.dumps(tool_call)}\n```",
                })
                self._messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' result: {json.dumps(result.result if result.success else result.error)}",
                })

                # Record in state
                self.state.tool_history.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result.result if result.success else None,
                    "error": result.error if not result.success else None,
                })

        # Max iterations reached
        return AgentAction(
            action_type="error",
            data={"error": "Max tool iterations reached"},
            reasoning=f"Stopped after {self._max_tool_iterations} tool calls",
        )

    async def finalize(self) -> Dict[str, Any]:
        """Finalize and return metrics."""
        self._status = AgentStatus.COMPLETED

        return {
            **self.get_metrics(),
            "conversation_length": len(self._messages),
            "tools_used": [h["tool"] for h in self.state.tool_history],
            "tool_calls_count": len(self.state.tool_history),
        }


# ============================================================================
# LLM Agent Builder
# ============================================================================

class LLMAgentBuilder(AgentBuilder):
    """
    Builder for LLM agents with fluent configuration.

    Example:
        agent = await (
            LLMAgentBuilder()
            .with_model("claude-3-sonnet-20240229")
            .with_provider("anthropic")
            .with_system_prompt("You are a helpful coding assistant.")
            .with_tool(calculator_tool)
            .with_tool(search_tool)
            .build()
        )
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._tools: List[Tool] = []
        self._capabilities: List[AgentCapability] = [
            AgentCapability.TEXT_GENERATION,
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
        ]
        self._model_server: Optional[ModelServer] = None

    def with_model(self, model_id: str) -> "LLMAgentBuilder":
        """Set the model to use."""
        self._config["model_id"] = model_id
        return self

    def with_provider(self, provider: str) -> "LLMAgentBuilder":
        """Set the model provider."""
        self._config["provider"] = provider
        return self

    def with_system_prompt(self, prompt: str) -> "LLMAgentBuilder":
        """Set the system prompt."""
        self._config["system_prompt"] = prompt
        return self

    def with_temperature(self, temperature: float) -> "LLMAgentBuilder":
        """Set the sampling temperature."""
        self._config["temperature"] = temperature
        return self

    def with_max_tokens(self, max_tokens: int) -> "LLMAgentBuilder":
        """Set the maximum tokens per response."""
        self._config["max_tokens"] = max_tokens
        return self

    def with_tool(self, tool: Tool) -> "LLMAgentBuilder":
        """Add a tool for the agent to use."""
        self._tools.append(tool)
        return self

    def with_tools(self, tools: List[Tool]) -> "LLMAgentBuilder":
        """Add multiple tools."""
        self._tools.extend(tools)
        return self

    def with_capability(self, capability: AgentCapability) -> "LLMAgentBuilder":
        """Add a capability."""
        if capability not in self._capabilities:
            self._capabilities.append(capability)
        return self

    def with_model_server(self, server: ModelServer) -> "LLMAgentBuilder":
        """Set a custom model server."""
        self._model_server = server
        return self

    def with_config(self, key: str, value: Any) -> "LLMAgentBuilder":
        """Set a custom config value."""
        self._config[key] = value
        return self

    async def build(self) -> LLMAgent:
        """Build the LLM agent."""
        import uuid

        agent = LLMAgent(
            agent_id=str(uuid.uuid4()),
            capabilities=self._capabilities,
            config=self._config,
            model_server=self._model_server,
        )

        # Register tools
        for tool in self._tools:
            agent.register_tool(tool)

        return agent

    def get_capabilities(self) -> List[AgentCapability]:
        return self._capabilities

    def get_agent_type(self) -> str:
        return "llm"


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_llm_agent(
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    **kwargs,
) -> LLMAgent:
    """
    Quick factory function for creating LLM agents.

    Example:
        agent = await create_llm_agent(
            model_id="claude-3-sonnet-20240229",
            provider="anthropic",
            system_prompt="You are a helpful assistant.",
            tools=[calculator_tool, search_tool],
        )
    """
    builder = LLMAgentBuilder()

    if model_id:
        builder.with_model(model_id)
    if provider:
        builder.with_provider(provider)
    if system_prompt:
        builder.with_system_prompt(system_prompt)
    if tools:
        builder.with_tools(tools)

    for key, value in kwargs.items():
        builder.with_config(key, value)

    return await builder.build()
