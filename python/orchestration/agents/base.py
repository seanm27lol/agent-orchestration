"""
Base Agent Abstractions

This module defines the core agent abstractions following patterns from
tinker-cookbook's Env and EnvGroupBuilder for consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeAlias, Any, Optional, List, Dict
from enum import Enum
from datetime import datetime
import uuid


# Type aliases
AgentId: TypeAlias = str
Message: TypeAlias = Dict[str, Any]
ToolResult: TypeAlias = Dict[str, Any]


class AgentCapability(Enum):
    """Capabilities that an agent can possess."""
    TEXT_GENERATION = "text_generation"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    TRADING = "trading"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    IMAGE_GENERATION = "image_generation"
    EMBEDDING = "embedding"


@dataclass
class AgentState:
    """Persistent state for an agent across interactions."""
    memory: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    tool_history: List[ToolResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_to_memory(self, message: Message) -> None:
        """Add a message to the agent's memory."""
        self.memory.append(message)

    def update_context(self, key: str, value: Any) -> None:
        """Update a context value."""
        self.context[key] = value

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        self.metrics[name] = value


@dataclass
class AgentObservation:
    """Observation received by an agent from the environment or other agents."""
    source: AgentId
    content: Any
    observation_type: str  # "message", "tool_result", "state_update", "task_input"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentAction:
    """Action produced by an agent in response to observations."""
    action_type: str  # "message", "tool_call", "delegate", "complete", "error"
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolCall:
    """A request to execute a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DelegationRequest:
    """A request to delegate work to another agent."""
    target_agent_type: str
    target_capabilities: Optional[List[AgentCapability]] = None
    task: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: Optional[int] = None


class Agent(ABC):
    """
    Base agent class following the tinker-cookbook Env pattern.

    Agents are single-use per task: they are initialized, stepped through
    observations, and finalized when complete.
    """

    def __init__(
        self,
        agent_id: AgentId,
        capabilities: List[AgentCapability],
        name: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.name = name or f"agent-{agent_id[:8]}"
        self.state = AgentState()
        self._step_count = 0
        self._created_at = datetime.utcnow()

    @abstractmethod
    async def initialize(self) -> AgentObservation:
        """
        Initialize the agent and return the initial observation.

        This is called once when the agent is spawned, before any steps.
        """
        pass

    @abstractmethod
    async def step(self, observation: AgentObservation) -> AgentAction:
        """
        Process an observation and produce an action.

        This is the main agent loop - receive observation, think, act.
        """
        pass

    @abstractmethod
    async def finalize(self) -> Dict[str, Any]:
        """
        Cleanup and return final metrics/results.

        Called when the agent's task is complete or on termination.
        """
        pass

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if the agent has a specific capability."""
        return capability in self.capabilities

    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "step_count": self._step_count,
            "created_at": self._created_at.isoformat(),
            "memory_size": len(self.state.memory),
            **self.state.metrics,
        }


class AgentBuilder(ABC):
    """
    Builder pattern for constructing agents.

    Follows the EnvGroupBuilder pattern from tinker-cookbook for
    consistent agent construction.
    """

    @abstractmethod
    async def build(self) -> Agent:
        """Build and return a new agent instance."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the capabilities this builder produces."""
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the type identifier for agents from this builder."""
        pass


class SimpleAgent(Agent):
    """
    A simple agent implementation for testing and basic use cases.

    This agent echoes back observations and can be extended for
    simple rule-based behaviors.
    """

    def __init__(
        self,
        agent_id: AgentId,
        capabilities: Optional[List[AgentCapability]] = None,
        name: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        super().__init__(
            agent_id=agent_id,
            capabilities=capabilities or [AgentCapability.TEXT_GENERATION],
            name=name,
        )
        self.system_prompt = system_prompt

    async def initialize(self) -> AgentObservation:
        """Initialize with the system prompt."""
        return AgentObservation(
            source="system",
            content={"status": "initialized", "system_prompt": self.system_prompt},
            observation_type="state_update",
        )

    async def step(self, observation: AgentObservation) -> AgentAction:
        """Process observation and respond."""
        self._step_count += 1
        self.state.add_to_memory({
            "role": "user" if observation.source != "system" else "system",
            "content": str(observation.content),
        })

        # Simple echo behavior - override in subclasses for real logic
        response = f"Received: {observation.content}"

        self.state.add_to_memory({
            "role": "assistant",
            "content": response,
        })

        return AgentAction(
            action_type="message",
            content=response,
        )

    async def finalize(self) -> Dict[str, Any]:
        """Return final metrics."""
        return self.get_metrics()


class SimpleAgentBuilder(AgentBuilder):
    """Builder for SimpleAgent instances."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        capabilities: Optional[List[AgentCapability]] = None,
    ):
        self.system_prompt = system_prompt
        self.capabilities = capabilities or [AgentCapability.TEXT_GENERATION]

    async def build(self) -> SimpleAgent:
        """Build a new SimpleAgent."""
        agent_id = str(uuid.uuid4())
        return SimpleAgent(
            agent_id=agent_id,
            capabilities=self.capabilities,
            system_prompt=self.system_prompt,
        )

    def get_capabilities(self) -> List[AgentCapability]:
        return self.capabilities

    def get_agent_type(self) -> str:
        return "simple"
