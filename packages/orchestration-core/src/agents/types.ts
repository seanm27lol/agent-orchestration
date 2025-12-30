/**
 * Core Agent Types for the Orchestration Framework
 *
 * These types define the fundamental abstractions for agents, messages,
 * and their capabilities within the orchestration system.
 */

// Type aliases for clarity
export type AgentId = string;
export type TaskId = string;
export type WorkflowId = string;
export type ChannelId = string;

/**
 * Capabilities that an agent can possess.
 * Used for routing tasks to appropriate agents.
 */
export enum AgentCapability {
  TEXT_GENERATION = 'text_generation',
  CODE_EXECUTION = 'code_execution',
  WEB_SEARCH = 'web_search',
  DATA_ANALYSIS = 'data_analysis',
  TRADING = 'trading',
  REASONING = 'reasoning',
  TOOL_USE = 'tool_use',
  IMAGE_GENERATION = 'image_generation',
  EMBEDDING = 'embedding',
}

/**
 * Current operational status of an agent
 */
export enum AgentStatus {
  AVAILABLE = 'available',
  BUSY = 'busy',
  OFFLINE = 'offline',
  ERROR = 'error',
}

/**
 * Registration information for an agent in the system
 */
export interface AgentRegistration {
  id: AgentId;
  type: string;
  name: string;
  capabilities: AgentCapability[];
  endpoint: string;  // URL for the Python ML bridge
  status: AgentStatus;
  metadata: Record<string, unknown>;
  registeredAt: Date;
  lastHeartbeat: Date;
}

/**
 * Configuration for spawning a new agent
 */
export interface AgentSpawnConfig {
  type: string;
  name?: string;
  capabilities?: AgentCapability[];
  config?: Record<string, unknown>;
}

/**
 * Response from spawning an agent
 */
export interface AgentSpawnResponse {
  agentId: AgentId;
  capabilities: AgentCapability[];
  status: AgentStatus;
}

/**
 * Message passed between agents or between orchestrator and agents
 */
export interface AgentMessage {
  id: string;
  timestamp: Date;
  source: AgentId | 'orchestrator' | 'system';
  target: AgentId | 'broadcast';
  channel: ChannelId;
  correlationId?: string;  // For request-response patterns
  replyTo?: string;        // Message ID to reply to
  payload: unknown;
  metadata?: Record<string, unknown>;
}

/**
 * Observation received by an agent
 */
export interface AgentObservation {
  source: AgentId | 'system';
  content: unknown;
  observationType: 'message' | 'tool_result' | 'state_update' | 'task_input';
  metadata?: Record<string, unknown>;
}

/**
 * Action produced by an agent
 */
export interface AgentAction {
  actionType: 'message' | 'tool_call' | 'delegate' | 'complete' | 'error';
  content: unknown;
  metadata?: Record<string, unknown>;
}

/**
 * Tool call request from an agent
 */
export interface ToolCall {
  toolName: string;
  arguments: Record<string, unknown>;
  callId: string;
}

/**
 * Tool execution result
 */
export interface ToolResult {
  callId: string;
  success: boolean;
  result?: unknown;
  error?: string;
}

/**
 * Delegation request from one agent to another
 */
export interface DelegationRequest {
  targetAgentType: string;
  targetCapabilities?: AgentCapability[];
  task: string;
  context: Record<string, unknown>;
  timeout?: number;
}

/**
 * Pool configuration for managing multiple agents of the same type
 */
export interface AgentPoolConfig {
  type: string;
  minInstances: number;
  maxInstances: number;
  idleTimeoutMs: number;
  scaleUpThreshold: number;   // Queue depth to trigger scale up
  scaleDownThreshold: number; // Idle time to trigger scale down
}

/**
 * Metrics collected from agent operations
 */
export interface AgentMetrics {
  agentId: AgentId;
  totalSteps: number;
  totalTokensUsed: number;
  averageStepDurationMs: number;
  errorCount: number;
  lastActiveAt: Date;
}

/**
 * State that persists across agent interactions
 */
export interface AgentState {
  memory: AgentMessage[];
  context: Record<string, unknown>;
  toolHistory: ToolResult[];
  metrics: Record<string, number>;
}

/**
 * Event emitted during agent lifecycle
 */
export interface AgentLifecycleEvent {
  type: 'registered' | 'spawned' | 'terminated' | 'error' | 'heartbeat';
  agentId: AgentId;
  timestamp: Date;
  data?: Record<string, unknown>;
}
