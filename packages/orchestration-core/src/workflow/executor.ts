/**
 * Workflow Executor
 *
 * Executes workflow DAGs with support for:
 * - Parallel task execution
 * - Dependency resolution
 * - Retry with exponential backoff
 * - Real-time progress updates
 * - Error handling and rollback
 */

import { v4 as uuid } from 'uuid';
import { globalEventBus } from '../communication/event-bus.js';
import {
  type WorkflowDAG,
  type TaskNode,
  type TaskEdge,
  type TaskInput,
  TaskExecutionStatus,
  WorkflowExecutionStatus,
} from './types.js';

// Type aliases for cleaner code
type TaskStatus = TaskExecutionStatus;
type WorkflowStatus = WorkflowExecutionStatus;

// ============================================================================
// Execution Types
// ============================================================================

export interface ExecutionConfig {
  maxRetries: number;
  retryDelayMs: number;
  retryBackoffMultiplier: number;
  taskTimeoutMs: number;
  parallelLimit: number;
  pythonBridgeUrl: string;
}

export const defaultExecutionConfig: ExecutionConfig = {
  maxRetries: 3,
  retryDelayMs: 1000,
  retryBackoffMultiplier: 2,
  taskTimeoutMs: 300000, // 5 minutes
  parallelLimit: 10,
  pythonBridgeUrl: 'http://localhost:8000',
};

export interface TaskExecution {
  taskId: string;
  taskNode: TaskNode;
  status: TaskStatus;
  startedAt?: Date;
  completedAt?: Date;
  result?: unknown;
  error?: string;
  retryCount: number;
  agentId?: string;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  workflow: WorkflowDAG;
  status: WorkflowStatus;
  startedAt: Date;
  completedAt?: Date;
  tasks: Map<string, TaskExecution>;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  error?: string;
}

export interface ExecutionProgress {
  executionId: string;
  workflowId: string;
  status: WorkflowStatus;
  completedTasks: number;
  totalTasks: number;
  currentTasks: string[];
  errors: Array<{ taskId: string; error: string }>;
}

// ============================================================================
// Task Handler
// ============================================================================

export type TaskHandler = (
  task: TaskNode,
  inputs: Record<string, unknown>,
  execution: WorkflowExecution
) => Promise<unknown>;

// Default handler that calls Python bridge
async function defaultTaskHandler(
  task: TaskNode,
  inputs: Record<string, unknown>,
  execution: WorkflowExecution,
  config: ExecutionConfig
): Promise<unknown> {
  // Spawn agent if needed
  const spawnResponse = await fetch(`${config.pythonBridgeUrl}/agents/spawn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      agent_type: task.agentType,
      config: task.config,
    }),
  });

  if (!spawnResponse.ok) {
    const error = await spawnResponse.json();
    throw new Error(`Failed to spawn agent: ${JSON.stringify(error)}`);
  }

  const spawnData = await spawnResponse.json() as { agent_id: string };
  const agentId = spawnData.agent_id;

  try {
    // Step agent with inputs
    const stepResponse = await fetch(`${config.pythonBridgeUrl}/agents/${agentId}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        observation: {
          task_id: task.id,
          inputs,
          workflow_id: execution.workflowId,
        },
        context: {
          execution_id: execution.id,
        },
      }),
    });

    if (!stepResponse.ok) {
      const error = await stepResponse.json();
      throw new Error(`Agent step failed: ${JSON.stringify(error)}`);
    }

    const result = await stepResponse.json() as { action_data: unknown };

    // Terminate agent
    await fetch(`${config.pythonBridgeUrl}/agents/${agentId}`, {
      method: 'DELETE',
    });

    return result.action_data;
  } catch (error) {
    // Cleanup agent on error
    await fetch(`${config.pythonBridgeUrl}/agents/${agentId}`, {
      method: 'DELETE',
    }).catch(() => {});

    throw error;
  }
}

// ============================================================================
// Workflow Executor
// ============================================================================

export class WorkflowExecutor {
  private config: ExecutionConfig;
  private customHandlers: Map<string, TaskHandler> = new Map();
  private activeExecutions: Map<string, WorkflowExecution> = new Map();

  constructor(config: Partial<ExecutionConfig> = {}) {
    this.config = { ...defaultExecutionConfig, ...config };
  }

  /**
   * Register a custom task handler for a specific agent type.
   */
  registerHandler(agentType: string, handler: TaskHandler): void {
    this.customHandlers.set(agentType, handler);
  }

  /**
   * Execute a workflow with given inputs.
   */
  async execute(
    workflow: WorkflowDAG,
    inputs: Record<string, unknown> = {}
  ): Promise<WorkflowExecution> {
    const execution: WorkflowExecution = {
      id: uuid(),
      workflowId: workflow.id,
      workflow,
      status: WorkflowExecutionStatus.PENDING,
      startedAt: new Date(),
      tasks: new Map(),
      inputs,
      outputs: {},
    };

    // Initialize task executions
    for (const task of workflow.nodes) {
      execution.tasks.set(task.id, {
        taskId: task.id,
        taskNode: task,
        status: TaskExecutionStatus.PENDING,
        retryCount: 0,
      });
    }

    this.activeExecutions.set(execution.id, execution);

    // Publish start event
    globalEventBus.publish({
      source: 'workflow-executor',
      target: 'broadcast',
      channel: 'workflow.execution_started',
      payload: {
        executionId: execution.id,
        workflowId: workflow.id,
        workflowName: workflow.name,
      },
    });

    try {
      execution.status = WorkflowExecutionStatus.RUNNING;
      await this.executeDAG(execution);

      // Check if any task failed (executeDAG may have set status to FAILED)
      const currentStatus = execution.status as WorkflowExecutionStatus;
      if (currentStatus !== WorkflowExecutionStatus.FAILED) {
        execution.status = WorkflowExecutionStatus.COMPLETED;
        execution.completedAt = new Date();
      }
    } catch (error) {
      execution.status = WorkflowExecutionStatus.FAILED;
      execution.error = String(error);
      execution.completedAt = new Date();
    }

    // Publish completion event
    globalEventBus.publish({
      source: 'workflow-executor',
      target: 'broadcast',
      channel: 'workflow.execution_completed',
      payload: {
        executionId: execution.id,
        workflowId: workflow.id,
        status: execution.status,
        outputs: execution.outputs,
        error: execution.error,
      },
    });

    this.activeExecutions.delete(execution.id);
    return execution;
  }

  /**
   * Get current progress of an execution.
   */
  getProgress(executionId: string): ExecutionProgress | undefined {
    const execution = this.activeExecutions.get(executionId);
    if (!execution) return undefined;

    const tasks = Array.from(execution.tasks.values());
    const completed = tasks.filter((t) => t.status === TaskExecutionStatus.COMPLETED).length;
    const current = tasks.filter((t) => t.status === TaskExecutionStatus.RUNNING).map((t) => t.taskId);
    const errors = tasks
      .filter((t) => t.status === TaskExecutionStatus.FAILED)
      .map((t) => ({ taskId: t.taskId, error: t.error || 'Unknown error' }));

    return {
      executionId: execution.id,
      workflowId: execution.workflowId,
      status: execution.status,
      completedTasks: completed,
      totalTasks: tasks.length,
      currentTasks: current,
      errors,
    };
  }

  /**
   * Cancel an active execution.
   */
  async cancel(executionId: string): Promise<boolean> {
    const execution = this.activeExecutions.get(executionId);
    if (!execution) return false;

    execution.status = WorkflowExecutionStatus.CANCELLED;
    execution.error = 'Execution cancelled';
    execution.completedAt = new Date();

    // Mark all pending/running tasks as cancelled
    for (const task of execution.tasks.values()) {
      if (task.status === TaskExecutionStatus.PENDING || task.status === TaskExecutionStatus.RUNNING) {
        task.status = TaskExecutionStatus.CANCELLED;
        task.error = 'Execution cancelled';
      }
    }

    return true;
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  private async executeDAG(execution: WorkflowExecution): Promise<void> {
    const { workflow } = execution;

    // Build dependency graph
    const dependencyCount = new Map<string, number>();
    const dependents = new Map<string, string[]>();

    for (const task of workflow.nodes) {
      dependencyCount.set(task.id, 0);
      dependents.set(task.id, []);
    }

    for (const edge of workflow.edges) {
      const count = dependencyCount.get(edge.to) || 0;
      dependencyCount.set(edge.to, count + 1);

      const deps = dependents.get(edge.from) || [];
      deps.push(edge.to);
      dependents.set(edge.from, deps);
    }

    // Find initial tasks (no dependencies)
    const readyTasks: string[] = [];
    for (const [taskId, count] of dependencyCount) {
      if (count === 0) {
        readyTasks.push(taskId);
      }
    }

    // Execute tasks in parallel waves
    while (readyTasks.length > 0) {
      // Take up to parallelLimit tasks
      const batch = readyTasks.splice(0, this.config.parallelLimit);

      // Execute batch in parallel
      const results = await Promise.allSettled(
        batch.map((taskId) => this.executeTask(taskId, execution))
      );

      // Process results and find newly ready tasks
      for (let i = 0; i < batch.length; i++) {
        const taskId = batch[i];
        const result = results[i];

        if (result.status === 'rejected') {
          const taskExec = execution.tasks.get(taskId)!;
          if (taskExec.status !== TaskExecutionStatus.COMPLETED) {
            taskExec.status = TaskExecutionStatus.FAILED;
            taskExec.error = String(result.reason);
            execution.status = WorkflowExecutionStatus.FAILED;
          }
        }

        // Update dependents
        const deps = dependents.get(taskId) || [];
        for (const depId of deps) {
          const count = dependencyCount.get(depId)! - 1;
          dependencyCount.set(depId, count);

          if (count === 0 && execution.status !== WorkflowExecutionStatus.FAILED) {
            readyTasks.push(depId);
          }
        }
      }

      // Stop if workflow failed
      if (execution.status === WorkflowExecutionStatus.FAILED) {
        break;
      }
    }
  }

  private async executeTask(
    taskId: string,
    execution: WorkflowExecution
  ): Promise<void> {
    const taskExec = execution.tasks.get(taskId)!;
    const task = taskExec.taskNode;

    taskExec.status = TaskExecutionStatus.RUNNING;
    taskExec.startedAt = new Date();

    // Publish task started event
    globalEventBus.publish({
      source: 'workflow-executor',
      target: 'broadcast',
      channel: 'workflow.task_started',
      payload: {
        executionId: execution.id,
        taskId,
        taskName: task.name,
        agentType: task.agentType,
      },
    });

    // Resolve inputs from dependencies and workflow inputs
    const inputs = this.resolveInputs(task, execution);

    // Execute with retry logic
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        // Use custom handler if registered, otherwise default
        const handler = this.customHandlers.get(task.agentType);
        let result: unknown;

        if (handler) {
          result = await this.withTimeout(
            handler(task, inputs, execution),
            this.config.taskTimeoutMs
          );
        } else {
          result = await this.withTimeout(
            defaultTaskHandler(task, inputs, execution, this.config),
            this.config.taskTimeoutMs
          );
        }

        // Success
        taskExec.status = TaskExecutionStatus.COMPLETED;
        taskExec.result = result;
        taskExec.completedAt = new Date();

        // Store output using task outputs array or task id
        if (task.outputs && task.outputs.length > 0) {
          // If the task has named outputs, try to extract them
          const resultObj = result as Record<string, unknown>;
          for (const outputKey of task.outputs) {
            execution.outputs[`${taskId}.${outputKey}`] = resultObj?.[outputKey] ?? result;
          }
        }
        // Always store the full result under the task id
        execution.outputs[taskId] = result;

        // Publish success event
        globalEventBus.publish({
          source: 'workflow-executor',
          target: 'broadcast',
          channel: 'workflow.task_completed',
          payload: {
            executionId: execution.id,
            taskId,
            result,
          },
        });

        return;
      } catch (error) {
        lastError = error as Error;
        taskExec.retryCount = attempt + 1;

        if (attempt < this.config.maxRetries) {
          // Wait before retry with exponential backoff
          const delay =
            this.config.retryDelayMs *
            Math.pow(this.config.retryBackoffMultiplier, attempt);
          await this.sleep(delay);
        }
      }
    }

    // All retries failed
    taskExec.status = TaskExecutionStatus.FAILED;
    taskExec.error = lastError?.message || 'Unknown error';
    taskExec.completedAt = new Date();

    // Publish failure event
    globalEventBus.publish({
      source: 'workflow-executor',
      target: 'broadcast',
      channel: 'workflow.task_failed',
      payload: {
        executionId: execution.id,
        taskId,
        error: taskExec.error,
        retryCount: taskExec.retryCount,
      },
    });

    throw lastError;
  }

  private resolveInputs(
    task: TaskNode,
    execution: WorkflowExecution
  ): Record<string, unknown> {
    const inputs: Record<string, unknown> = {};

    if (!task.inputs) {
      return inputs;
    }

    for (const [key, inputDef] of Object.entries(task.inputs)) {
      if (inputDef.type === 'workflow_input') {
        // Get from workflow inputs
        inputs[key] = execution.inputs[inputDef.inputKey || key];
      } else if (inputDef.type === 'task_output') {
        // Get from another task's output
        const sourceTaskId = inputDef.sourceTaskId!;
        const sourceTask = execution.tasks.get(sourceTaskId);

        if (sourceTask?.result) {
          const result = sourceTask.result as Record<string, unknown>;
          inputs[key] = inputDef.outputKey ? result[inputDef.outputKey] : result;
        }
      } else if (inputDef.type === 'static') {
        inputs[key] = inputDef.value;
      }
    }

    return inputs;
  }

  private async withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Task timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      promise
        .then((result) => {
          clearTimeout(timer);
          resolve(result);
        })
        .catch((error) => {
          clearTimeout(timer);
          reject(error);
        });
    });
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Global Executor
// ============================================================================

export const globalWorkflowExecutor = new WorkflowExecutor();

/**
 * Convenience function to execute a workflow.
 */
export async function executeWorkflow(
  workflow: WorkflowDAG,
  inputs: Record<string, unknown> = {}
): Promise<WorkflowExecution> {
  return globalWorkflowExecutor.execute(workflow, inputs);
}
