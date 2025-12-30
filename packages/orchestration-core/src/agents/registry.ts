/**
 * Agent Registry
 *
 * Manages agent registration, lifecycle, and discovery.
 * Supports agent pools for scaling and load balancing.
 */

import { v4 as uuid } from 'uuid';
import type {
  AgentId,
  AgentRegistration,
  AgentStatus,
  AgentCapability,
  AgentSpawnConfig,
  AgentSpawnResponse,
  AgentPoolConfig,
  AgentLifecycleEvent,
} from './types.js';
import { AgentEventBus } from '../communication/event-bus.js';

/**
 * Agent pool state
 */
interface AgentPool {
  config: AgentPoolConfig;
  agents: Set<AgentId>;
  pendingSpawns: number;
}

/**
 * Agent Registry
 *
 * Central registry for all agents in the system. Handles:
 * - Agent registration and discovery
 * - Capability-based agent matching
 * - Pool management and scaling
 * - Health monitoring via heartbeats
 */
export class AgentRegistry {
  private agents: Map<AgentId, AgentRegistration> = new Map();
  private pools: Map<string, AgentPool> = new Map();
  private eventBus: AgentEventBus;
  private heartbeatIntervalMs: number;
  private heartbeatTimeoutMs: number;
  private heartbeatTimer?: ReturnType<typeof setInterval>;

  constructor(
    eventBus: AgentEventBus,
    options: {
      heartbeatIntervalMs?: number;
      heartbeatTimeoutMs?: number;
    } = {}
  ) {
    this.eventBus = eventBus;
    this.heartbeatIntervalMs = options.heartbeatIntervalMs ?? 30000;
    this.heartbeatTimeoutMs = options.heartbeatTimeoutMs ?? 90000;
  }

  /**
   * Start the registry (begins heartbeat monitoring)
   */
  start(): void {
    this.heartbeatTimer = setInterval(() => {
      this.checkHeartbeats();
    }, this.heartbeatIntervalMs);
  }

  /**
   * Stop the registry
   */
  stop(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
  }

  /**
   * Register a new agent
   */
  async register(
    registration: Omit<AgentRegistration, 'id' | 'registeredAt' | 'lastHeartbeat' | 'status'>
  ): Promise<AgentRegistration> {
    const id = uuid();
    const now = new Date();

    const agent: AgentRegistration = {
      ...registration,
      id,
      status: 'available' as AgentStatus,
      registeredAt: now,
      lastHeartbeat: now,
    };

    this.agents.set(id, agent);

    // Add to pool if exists
    const pool = this.pools.get(agent.type);
    if (pool) {
      pool.agents.add(id);
    }

    // Emit lifecycle event
    this.emitLifecycleEvent({
      type: 'registered',
      agentId: id,
      timestamp: now,
      data: { type: agent.type, capabilities: agent.capabilities },
    });

    return agent;
  }

  /**
   * Unregister an agent
   */
  async unregister(agentId: AgentId): Promise<boolean> {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    this.agents.delete(agentId);

    // Remove from pool
    const pool = this.pools.get(agent.type);
    if (pool) {
      pool.agents.delete(agentId);
    }

    this.emitLifecycleEvent({
      type: 'terminated',
      agentId,
      timestamp: new Date(),
    });

    return true;
  }

  /**
   * Get an agent by ID
   */
  get(agentId: AgentId): AgentRegistration | undefined {
    return this.agents.get(agentId);
  }

  /**
   * List all agents
   */
  listAll(): AgentRegistration[] {
    return Array.from(this.agents.values());
  }

  /**
   * Find agents by type
   */
  findByType(type: string): AgentRegistration[] {
    return this.listAll().filter((a) => a.type === type);
  }

  /**
   * Find agents by capabilities
   */
  findByCapabilities(
    capabilities: AgentCapability[],
    options: { status?: AgentStatus; type?: string } = {}
  ): AgentRegistration[] {
    return this.listAll().filter((agent) => {
      // Check type filter
      if (options.type && agent.type !== options.type) return false;

      // Check status filter
      if (options.status && agent.status !== options.status) return false;

      // Check all required capabilities
      return capabilities.every((cap) => agent.capabilities.includes(cap));
    });
  }

  /**
   * Acquire an available agent for a task
   *
   * Marks the agent as busy and returns it.
   * Returns null if no suitable agent is available.
   */
  async acquire(
    agentType: string,
    capabilities?: AgentCapability[]
  ): Promise<AgentRegistration | null> {
    const candidates = this.findByCapabilities(capabilities ?? [], {
      type: agentType,
      status: 'available' as AgentStatus,
    });

    if (candidates.length === 0) {
      // Check if we can scale up the pool
      const pool = this.pools.get(agentType);
      if (pool && pool.agents.size + pool.pendingSpawns < pool.config.maxInstances) {
        // Could trigger auto-scaling here
        return null;
      }
      return null;
    }

    // Pick the first available agent
    const agent = candidates[0];
    agent.status = 'busy' as AgentStatus;
    return agent;
  }

  /**
   * Release an agent back to available status
   */
  async release(agentId: AgentId): Promise<void> {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.status = 'available' as AgentStatus;
    }
  }

  /**
   * Update agent status
   */
  updateStatus(agentId: AgentId, status: AgentStatus): boolean {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    agent.status = status;
    return true;
  }

  /**
   * Record a heartbeat from an agent
   */
  heartbeat(agentId: AgentId): boolean {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    agent.lastHeartbeat = new Date();

    // If agent was offline, bring it back
    if (agent.status === ('offline' as AgentStatus)) {
      agent.status = 'available' as AgentStatus;
    }

    this.emitLifecycleEvent({
      type: 'heartbeat',
      agentId,
      timestamp: agent.lastHeartbeat,
    });

    return true;
  }

  /**
   * Configure an agent pool for a type
   */
  configurePool(config: AgentPoolConfig): void {
    this.pools.set(config.type, {
      config,
      agents: new Set(),
      pendingSpawns: 0,
    });
  }

  /**
   * Get pool status
   */
  getPoolStatus(type: string): {
    config: AgentPoolConfig;
    currentInstances: number;
    availableInstances: number;
    busyInstances: number;
  } | null {
    const pool = this.pools.get(type);
    if (!pool) return null;

    const agents = Array.from(pool.agents)
      .map((id) => this.agents.get(id))
      .filter((a): a is AgentRegistration => a !== undefined);

    return {
      config: pool.config,
      currentInstances: agents.length,
      availableInstances: agents.filter((a) => a.status === 'available').length,
      busyInstances: agents.filter((a) => a.status === 'busy').length,
    };
  }

  /**
   * Check for stale agents and mark them offline
   */
  private checkHeartbeats(): void {
    const now = Date.now();
    const timeout = this.heartbeatTimeoutMs;

    for (const agent of this.agents.values()) {
      const elapsed = now - agent.lastHeartbeat.getTime();
      if (elapsed > timeout && agent.status !== ('offline' as AgentStatus)) {
        agent.status = 'offline' as AgentStatus;
        this.emitLifecycleEvent({
          type: 'error',
          agentId: agent.id,
          timestamp: new Date(),
          data: { reason: 'heartbeat_timeout', elapsed },
        });
      }
    }
  }

  /**
   * Emit a lifecycle event
   */
  private emitLifecycleEvent(event: AgentLifecycleEvent): void {
    this.eventBus.publish({
      source: 'system',
      target: 'broadcast',
      channel: 'agent.lifecycle',
      payload: event,
    });
  }

  /**
   * Get registry stats
   */
  getStats(): {
    totalAgents: number;
    byStatus: Record<string, number>;
    byType: Record<string, number>;
  } {
    const agents = this.listAll();
    const byStatus: Record<string, number> = {};
    const byType: Record<string, number> = {};

    for (const agent of agents) {
      byStatus[agent.status] = (byStatus[agent.status] ?? 0) + 1;
      byType[agent.type] = (byType[agent.type] ?? 0) + 1;
    }

    return {
      totalAgents: agents.length,
      byStatus,
      byType,
    };
  }

  /**
   * List all unique agent types currently registered
   */
  listAgentTypes(): string[] {
    const types = new Set<string>();
    for (const agent of this.agents.values()) {
      types.add(agent.type);
    }
    return Array.from(types);
  }
}

// Export singleton instance with default event bus
import { globalEventBus } from '../communication/event-bus.js';
export const globalAgentRegistry = new AgentRegistry(globalEventBus);
