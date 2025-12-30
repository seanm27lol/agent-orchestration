"""
ML Bridge Server

FastAPI server that exposes Python agents and ML models to the TypeScript
orchestration layer. Provides REST endpoints for agent lifecycle management
and WebSocket for streaming agent interactions.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..agents.base import Agent, AgentStatus, AgentObservation, AgentAction


# ============================================================================
# Request/Response Models
# ============================================================================

class SpawnAgentRequest(BaseModel):
    """Request to spawn a new agent instance."""
    agent_type: str = Field(..., description="Type of agent to spawn")
    agent_id: Optional[str] = Field(None, description="Custom agent ID (auto-generated if not provided)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Agent metadata")


class SpawnAgentResponse(BaseModel):
    """Response from spawning an agent."""
    agent_id: str
    agent_type: str
    status: str
    created_at: str


class AgentStepRequest(BaseModel):
    """Request to execute one agent step."""
    observation: Dict[str, Any] = Field(..., description="Observation to process")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AgentStepResponse(BaseModel):
    """Response from agent step execution."""
    agent_id: str
    action_type: str
    action_data: Dict[str, Any]
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    execution_time_ms: float


class AgentInfoResponse(BaseModel):
    """Information about an agent."""
    agent_id: str
    agent_type: str
    status: str
    capabilities: List[str]
    created_at: str
    last_active: Optional[str] = None
    step_count: int
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    active_agents: int


# ============================================================================
# Agent Manager
# ============================================================================

class AgentInstance:
    """Wrapper for a running agent instance."""

    def __init__(self, agent: Agent, agent_type: str):
        self.agent = agent
        self.agent_type = agent_type
        self.created_at = datetime.utcnow()
        self.last_active = self.created_at
        self.step_count = 0

    def update_activity(self):
        self.last_active = datetime.utcnow()
        self.step_count += 1


class AgentManager:
    """
    Manages agent lifecycle and execution.

    Maintains a registry of active agents and provides methods for
    spawning, stepping, and terminating agents.
    """

    def __init__(self):
        self._agents: Dict[str, AgentInstance] = {}
        self._agent_factories: Dict[str, type] = {}
        self._lock = asyncio.Lock()

    def register_agent_type(self, agent_type: str, factory: type):
        """Register an agent factory for a given type."""
        self._agent_factories[agent_type] = factory

    def get_registered_types(self) -> List[str]:
        """Get list of registered agent types."""
        return list(self._agent_factories.keys())

    async def spawn(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Spawn a new agent instance.

        Args:
            agent_type: Type of agent to spawn
            agent_id: Optional custom ID
            config: Agent configuration
            metadata: Agent metadata

        Returns:
            The agent ID

        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in self._agent_factories:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self._agent_factories.keys())}")

        agent_id = agent_id or str(uuid.uuid4())

        async with self._lock:
            if agent_id in self._agents:
                raise ValueError(f"Agent {agent_id} already exists")

            # Create agent instance
            factory = self._agent_factories[agent_type]
            agent = factory(
                agent_id=agent_id,
                config=config or {},
                **(metadata or {})
            )

            # Initialize agent
            await agent.initialize()

            # Store instance
            self._agents[agent_id] = AgentInstance(agent, agent_type)

        return agent_id

    async def step(
        self,
        agent_id: str,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentAction:
        """
        Execute one step of an agent.

        Args:
            agent_id: Agent to step
            observation: Observation data
            context: Additional context

        Returns:
            The agent's action
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not found")

            instance = self._agents[agent_id]

        # Create observation object
        obs = AgentObservation(
            source="bridge",
            data=observation,
            context=context or {},
        )

        # Execute step
        action = await instance.agent.step(obs)
        instance.update_activity()

        return action

    async def terminate(self, agent_id: str) -> Dict[str, Any]:
        """
        Terminate an agent and get final state.

        Args:
            agent_id: Agent to terminate

        Returns:
            Final agent state
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not found")

            instance = self._agents.pop(agent_id)

        # Finalize agent
        return await instance.agent.finalize()

    def get_info(self, agent_id: str) -> AgentInfoResponse:
        """Get information about an agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")

        instance = self._agents[agent_id]
        agent = instance.agent

        return AgentInfoResponse(
            agent_id=agent.agent_id,
            agent_type=instance.agent_type,
            status=agent.status.value,
            capabilities=[c.value for c in agent.capabilities],
            created_at=instance.created_at.isoformat(),
            last_active=instance.last_active.isoformat() if instance.last_active else None,
            step_count=instance.step_count,
            metadata=agent.metadata,
        )

    def list_agents(self) -> List[AgentInfoResponse]:
        """List all active agents."""
        return [self.get_info(aid) for aid in self._agents.keys()]

    @property
    def active_count(self) -> int:
        """Number of active agents."""
        return len(self._agents)


# ============================================================================
# FastAPI Application
# ============================================================================

# Global state
agent_manager = AgentManager()
start_time = datetime.utcnow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("ML Bridge starting up...")

    # Register default agent types here
    # agent_manager.register_agent_type("llm", LLMAgent)

    yield

    # Shutdown
    print("ML Bridge shutting down...")
    # Terminate all agents
    for agent_id in list(agent_manager._agents.keys()):
        try:
            await agent_manager.terminate(agent_id)
        except Exception as e:
            print(f"Error terminating agent {agent_id}: {e}")


app = FastAPI(
    title="Agent Orchestration ML Bridge",
    description="FastAPI server for Python agent operations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for TypeScript client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime,
        active_agents=agent_manager.active_count,
    )


@app.get("/agents/types")
async def list_agent_types():
    """List available agent types."""
    return {"types": agent_manager.get_registered_types()}


@app.post("/agents/spawn", response_model=SpawnAgentResponse)
async def spawn_agent(request: SpawnAgentRequest):
    """Spawn a new agent instance."""
    try:
        agent_id = await agent_manager.spawn(
            agent_type=request.agent_type,
            agent_id=request.agent_id,
            config=request.config,
            metadata=request.metadata,
        )

        info = agent_manager.get_info(agent_id)

        return SpawnAgentResponse(
            agent_id=agent_id,
            agent_type=request.agent_type,
            status=info.status,
            created_at=info.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/agents", response_model=List[AgentInfoResponse])
async def list_agents():
    """List all active agents."""
    return agent_manager.list_agents()


@app.get("/agents/{agent_id}", response_model=AgentInfoResponse)
async def get_agent(agent_id: str):
    """Get information about a specific agent."""
    try:
        return agent_manager.get_info(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agents/{agent_id}/step", response_model=AgentStepResponse)
async def step_agent(agent_id: str, request: AgentStepRequest):
    """Execute one step of an agent."""
    try:
        start = datetime.utcnow()

        action = await agent_manager.step(
            agent_id=agent_id,
            observation=request.observation,
            context=request.context,
        )

        duration_ms = (datetime.utcnow() - start).total_seconds() * 1000

        return AgentStepResponse(
            agent_id=agent_id,
            action_type=action.action_type,
            action_data=action.data,
            reasoning=action.reasoning,
            confidence=action.confidence,
            execution_time_ms=duration_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/agents/{agent_id}")
async def terminate_agent(agent_id: str):
    """Terminate an agent and get final state."""
    try:
        final_state = await agent_manager.terminate(agent_id)
        return {"agent_id": agent_id, "final_state": final_state}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# WebSocket for Streaming
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for agent streaming."""

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, agent_id: str, websocket: WebSocket):
        """Connect a client to an agent's stream."""
        await websocket.accept()
        if agent_id not in self._connections:
            self._connections[agent_id] = []
        self._connections[agent_id].append(websocket)

    def disconnect(self, agent_id: str, websocket: WebSocket):
        """Disconnect a client from an agent's stream."""
        if agent_id in self._connections:
            self._connections[agent_id].remove(websocket)
            if not self._connections[agent_id]:
                del self._connections[agent_id]

    async def broadcast(self, agent_id: str, message: Dict[str, Any]):
        """Broadcast a message to all clients watching an agent."""
        if agent_id in self._connections:
            for connection in self._connections[agent_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # Client disconnected


connection_manager = ConnectionManager()


@app.websocket("/agents/{agent_id}/stream")
async def agent_stream(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for streaming agent interactions.

    Messages from client:
    - {"type": "step", "observation": {...}}
    - {"type": "ping"}

    Messages to client:
    - {"type": "action", "action": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}
    """
    # Verify agent exists
    try:
        agent_manager.get_info(agent_id)
    except ValueError:
        await websocket.close(code=4004, reason="Agent not found")
        return

    await connection_manager.connect(agent_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "step":
                try:
                    observation = data.get("observation", {})
                    context = data.get("context", {})

                    action = await agent_manager.step(
                        agent_id=agent_id,
                        observation=observation,
                        context=context,
                    )

                    await websocket.send_json({
                        "type": "action",
                        "action": {
                            "action_type": action.action_type,
                            "data": action.data,
                            "reasoning": action.reasoning,
                            "confidence": action.confidence,
                        }
                    })

                    # Broadcast to other watchers
                    await connection_manager.broadcast(agent_id, {
                        "type": "step_complete",
                        "agent_id": agent_id,
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        connection_manager.disconnect(agent_id, websocket)


# ============================================================================
# Batch Operations
# ============================================================================

class BatchStepRequest(BaseModel):
    """Request to step multiple agents at once."""
    steps: List[Dict[str, Any]] = Field(
        ...,
        description="List of {agent_id, observation, context} objects"
    )


@app.post("/agents/batch/step")
async def batch_step(request: BatchStepRequest):
    """
    Execute steps for multiple agents in parallel.

    Useful for synchronized multi-agent workflows.
    """
    async def step_one(step_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = step_data.get("agent_id")
        observation = step_data.get("observation", {})
        context = step_data.get("context", {})

        try:
            action = await agent_manager.step(
                agent_id=agent_id,
                observation=observation,
                context=context,
            )
            return {
                "agent_id": agent_id,
                "success": True,
                "action": {
                    "action_type": action.action_type,
                    "data": action.data,
                }
            }
        except Exception as e:
            return {
                "agent_id": agent_id,
                "success": False,
                "error": str(e),
            }

    # Execute all steps in parallel
    results = await asyncio.gather(*[step_one(s) for s in request.steps])

    return {"results": results}


# ============================================================================
# Entry Point
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the ML Bridge server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
