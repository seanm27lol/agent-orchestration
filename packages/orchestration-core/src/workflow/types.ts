/**
 * Workflow Types
 *
 * Defines the structure of workflow DAGs (Directed Acyclic Graphs)
 * for orchestrating multi-agent tasks.
 */

import type { AgentCapability, TaskId, WorkflowId } from '../agents/types.js';

/**
 * Input source for a task
 */
export interface TaskInput {
  type: 'static' | 'task_output' | 'workflow_input';
  value?: unknown;           // For static inputs
  sourceTaskId?: TaskId;     // For task_output
  outputKey?: string;        // Key in source task's output
  inputKey?: string;         // Key in workflow inputs
}

/**
 * Retry policy for failed tasks
 */
export interface RetryPolicy {
  maxAttempts: number;
  backoffMs: number;
  backoffMultiplier: number;
  retryableErrors?: string[];
}

/**
 * A node (task) in the workflow DAG
 */
export interface TaskNode {
  id: TaskId;
  name: string;
  agentType: string;
  requiredCapabilities?: AgentCapability[];
  inputs: Record<string, TaskInput>;
  outputs: string[];
  config: Record<string, unknown>;
  retryPolicy?: RetryPolicy;
  timeoutMs?: number;
}

/**
 * Condition for conditional branching
 */
export interface ConditionalBranch {
  type: 'success' | 'failure' | 'output_match' | 'always';
  outputPattern?: Record<string, unknown>;
  customMatcher?: string; // Serialized function or expression
}

/**
 * An edge connecting tasks in the workflow
 */
export interface TaskEdge {
  from: TaskId;
  to: TaskId;
  condition?: ConditionalBranch;
}

/**
 * Input definition for the workflow
 */
export interface WorkflowInputDef {
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required: boolean;
  default?: unknown;
  description?: string;
}

/**
 * Output definition for the workflow
 */
export interface WorkflowOutputDef {
  sourceTaskId: TaskId;
  outputKey: string;
  description?: string;
}

/**
 * Complete workflow DAG definition
 */
export interface WorkflowDAG {
  id: WorkflowId;
  name: string;
  version: string;
  description?: string;
  nodes: TaskNode[];
  edges: TaskEdge[];
  inputs: Record<string, WorkflowInputDef>;
  outputs: Record<string, WorkflowOutputDef>;
  defaultRetryPolicy?: RetryPolicy;
  defaultTimeoutMs?: number;
}

/**
 * Execution status for tasks
 */
export enum TaskExecutionStatus {
  PENDING = 'pending',
  WAITING = 'waiting',    // Waiting for dependencies
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  SKIPPED = 'skipped',
  CANCELLED = 'cancelled',
}

/**
 * Execution status for workflows
 */
export enum WorkflowExecutionStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Runtime state of a task execution
 */
export interface TaskExecutionState {
  taskId: TaskId;
  agentId?: string;
  status: TaskExecutionStatus;
  inputs: Record<string, unknown>;
  outputs?: unknown;
  error?: string;
  retryCount: number;
  startedAt?: Date;
  completedAt?: Date;
  durationMs?: number;
}

/**
 * Runtime state of a workflow execution
 */
export interface WorkflowExecutionState {
  id: string;
  workflowId: WorkflowId;
  status: WorkflowExecutionStatus;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  tasks: Map<TaskId, TaskExecutionState>;
  startedAt: Date;
  completedAt?: Date;
  error?: string;
}

/**
 * Event emitted during workflow execution
 */
export interface WorkflowEvent {
  type: 'workflow_started' | 'workflow_completed' | 'workflow_failed' |
        'task_started' | 'task_completed' | 'task_failed' | 'task_retry';
  executionId: string;
  workflowId: WorkflowId;
  taskId?: TaskId;
  timestamp: Date;
  data?: Record<string, unknown>;
}
