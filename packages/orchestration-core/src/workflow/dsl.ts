/**
 * Workflow DSL Builder
 *
 * Provides a fluent API for building workflow DAGs.
 * Inspired by pipeline builders in modern orchestration tools.
 */

import { v4 as uuid } from 'uuid';
import type { AgentCapability } from '../agents/types.js';
import type {
  WorkflowDAG,
  TaskNode,
  TaskEdge,
  TaskInput,
  RetryPolicy,
  ConditionalBranch,
  WorkflowInputDef,
  WorkflowOutputDef,
} from './types.js';

/**
 * Task configuration for the builder
 */
export interface TaskConfig {
  name?: string;
  capabilities?: AgentCapability[];
  inputs?: Record<string, unknown>;
  config?: Record<string, unknown>;
  timeout?: number;
  retry?: Partial<RetryPolicy>;
}

/**
 * Workflow Builder
 *
 * Fluent API for constructing workflow DAGs.
 *
 * @example
 * ```typescript
 * const workflow = new WorkflowBuilder()
 *   .input('query', { type: 'string', required: true })
 *   .task('analyze', 'AnalystAgent', { inputs: { data: '$input.query' } })
 *   .parallel(
 *     b => b.task('research', 'ResearchAgent'),
 *     b => b.task('compute', 'ComputeAgent')
 *   )
 *   .task('synthesize', 'SynthesisAgent')
 *   .output('result', 'synthesize', 'summary')
 *   .build('my-workflow', 'Analysis Pipeline');
 * ```
 */
export class WorkflowBuilder {
  private nodes: TaskNode[] = [];
  private edges: TaskEdge[] = [];
  private inputs: Record<string, WorkflowInputDef> = {};
  private outputs: Record<string, WorkflowOutputDef> = {};
  private currentNodeIds: string[] = [];
  private defaultRetry?: RetryPolicy;
  private defaultTimeout?: number;
  private description?: string;

  /**
   * Set workflow description
   */
  describe(description: string): this {
    this.description = description;
    return this;
  }

  /**
   * Define a workflow input
   */
  input(
    name: string,
    def: Omit<WorkflowInputDef, 'type'> & { type?: WorkflowInputDef['type'] }
  ): this {
    this.inputs[name] = {
      type: def.type ?? 'string',
      required: def.required,
      default: def.default,
      description: def.description,
    };
    return this;
  }

  /**
   * Define a workflow output
   */
  output(name: string, sourceTaskId: string, outputKey: string, description?: string): this {
    this.outputs[name] = {
      sourceTaskId,
      outputKey,
      description,
    };
    return this;
  }

  /**
   * Set default retry policy for all tasks
   */
  defaultRetryPolicy(policy: RetryPolicy): this {
    this.defaultRetry = policy;
    return this;
  }

  /**
   * Set default timeout for all tasks
   */
  defaultTimeoutMs(timeout: number): this {
    this.defaultTimeout = timeout;
    return this;
  }

  /**
   * Add a task to the workflow
   */
  task(id: string, agentType: string, config: TaskConfig = {}): this {
    const node: TaskNode = {
      id,
      name: config.name ?? id,
      agentType,
      requiredCapabilities: config.capabilities,
      inputs: this.parseInputs(config.inputs ?? {}),
      outputs: [],
      config: config.config ?? {},
      timeoutMs: config.timeout ?? this.defaultTimeout,
      retryPolicy: config.retry
        ? { ...this.getDefaultRetry(), ...config.retry }
        : this.defaultRetry,
    };

    this.nodes.push(node);

    // Connect to previous nodes
    for (const prevId of this.currentNodeIds) {
      this.edges.push({ from: prevId, to: id });
    }

    // Update current nodes
    this.currentNodeIds = [id];

    return this;
  }

  /**
   * Execute multiple tasks in parallel
   *
   * All parallel branches receive the same inputs from previous tasks
   * and their outputs are available to subsequent tasks.
   */
  parallel(...builders: ((b: WorkflowBuilder) => WorkflowBuilder)[]): this {
    const parallelEndNodes: string[] = [];
    const startNodes = [...this.currentNodeIds];

    for (const builder of builders) {
      const subBuilder = new WorkflowBuilder();
      subBuilder.defaultRetry = this.defaultRetry;
      subBuilder.defaultTimeout = this.defaultTimeout;

      builder(subBuilder);

      // Add all nodes and edges from sub-builder
      this.nodes.push(...subBuilder.nodes);
      this.edges.push(...subBuilder.edges);

      // Connect start nodes to first node in sub-builder
      if (subBuilder.nodes.length > 0) {
        const firstSubNode = subBuilder.nodes[0];
        for (const startId of startNodes) {
          this.edges.push({ from: startId, to: firstSubNode.id });
        }
        parallelEndNodes.push(...subBuilder.currentNodeIds);
      }
    }

    // Set current nodes to all parallel end nodes
    this.currentNodeIds = parallelEndNodes;

    return this;
  }

  /**
   * Add conditional branching
   */
  conditional(
    condition: ConditionalBranch,
    ifTrue: (b: WorkflowBuilder) => WorkflowBuilder,
    ifFalse?: (b: WorkflowBuilder) => WorkflowBuilder
  ): this {
    const startNodes = [...this.currentNodeIds];
    const endNodes: string[] = [];

    // True branch
    const trueBuilder = new WorkflowBuilder();
    trueBuilder.defaultRetry = this.defaultRetry;
    trueBuilder.defaultTimeout = this.defaultTimeout;
    ifTrue(trueBuilder);

    this.nodes.push(...trueBuilder.nodes);
    this.edges.push(...trueBuilder.edges);

    if (trueBuilder.nodes.length > 0) {
      const firstTrueNode = trueBuilder.nodes[0];
      for (const startId of startNodes) {
        this.edges.push({
          from: startId,
          to: firstTrueNode.id,
          condition: { ...condition },
        });
      }
      endNodes.push(...trueBuilder.currentNodeIds);
    }

    // False branch (optional)
    if (ifFalse) {
      const falseBuilder = new WorkflowBuilder();
      falseBuilder.defaultRetry = this.defaultRetry;
      falseBuilder.defaultTimeout = this.defaultTimeout;
      ifFalse(falseBuilder);

      this.nodes.push(...falseBuilder.nodes);
      this.edges.push(...falseBuilder.edges);

      if (falseBuilder.nodes.length > 0) {
        const firstFalseNode = falseBuilder.nodes[0];
        const inverseCondition: ConditionalBranch = {
          type: condition.type === 'success' ? 'failure' : 'success',
        };
        for (const startId of startNodes) {
          this.edges.push({
            from: startId,
            to: firstFalseNode.id,
            condition: inverseCondition,
          });
        }
        endNodes.push(...falseBuilder.currentNodeIds);
      }
    }

    this.currentNodeIds = endNodes.length > 0 ? endNodes : startNodes;

    return this;
  }

  /**
   * Join parallel branches back together
   */
  join(id: string, agentType: string, config: TaskConfig = {}): this {
    return this.task(id, agentType, config);
  }

  /**
   * Add a retry wrapper around a task or group
   */
  withRetry(policy: Partial<RetryPolicy>, builder: (b: WorkflowBuilder) => WorkflowBuilder): this {
    const subBuilder = new WorkflowBuilder();
    subBuilder.defaultRetry = { ...this.getDefaultRetry(), ...policy };
    subBuilder.defaultTimeout = this.defaultTimeout;
    subBuilder.currentNodeIds = [...this.currentNodeIds];

    builder(subBuilder);

    // Merge sub-builder
    this.nodes.push(...subBuilder.nodes);
    this.edges.push(...subBuilder.edges);
    this.currentNodeIds = subBuilder.currentNodeIds;

    return this;
  }

  /**
   * Build the final workflow DAG
   */
  build(id: string, name: string, version: string = '1.0.0'): WorkflowDAG {
    // Validate the DAG
    this.validate();

    return {
      id: id || uuid(),
      name,
      version,
      description: this.description,
      nodes: this.nodes,
      edges: this.edges,
      inputs: this.inputs,
      outputs: this.outputs,
      defaultRetryPolicy: this.defaultRetry,
      defaultTimeoutMs: this.defaultTimeout,
    };
  }

  /**
   * Parse input references (e.g., "$input.query" or "$task.analyze.result")
   */
  private parseInputs(inputs: Record<string, unknown>): Record<string, TaskInput> {
    const result: Record<string, TaskInput> = {};

    for (const [key, value] of Object.entries(inputs)) {
      if (typeof value === 'string' && value.startsWith('$')) {
        const ref = value.substring(1);
        const parts = ref.split('.');

        if (parts[0] === 'input') {
          result[key] = {
            type: 'workflow_input',
            inputKey: parts.slice(1).join('.'),
          };
        } else if (parts[0] === 'task') {
          result[key] = {
            type: 'task_output',
            sourceTaskId: parts[1],
            outputKey: parts.slice(2).join('.') || 'result',
          };
        } else {
          result[key] = { type: 'static', value };
        }
      } else {
        result[key] = { type: 'static', value };
      }
    }

    return result;
  }

  /**
   * Get default retry policy
   */
  private getDefaultRetry(): RetryPolicy {
    return this.defaultRetry ?? {
      maxAttempts: 3,
      backoffMs: 1000,
      backoffMultiplier: 2,
    };
  }

  /**
   * Validate the workflow DAG
   */
  private validate(): void {
    // Check for duplicate node IDs
    const nodeIds = new Set<string>();
    for (const node of this.nodes) {
      if (nodeIds.has(node.id)) {
        throw new Error(`Duplicate task ID: ${node.id}`);
      }
      nodeIds.add(node.id);
    }

    // Check edge references
    for (const edge of this.edges) {
      if (!nodeIds.has(edge.from)) {
        throw new Error(`Edge references unknown task: ${edge.from}`);
      }
      if (!nodeIds.has(edge.to)) {
        throw new Error(`Edge references unknown task: ${edge.to}`);
      }
    }

    // Check for cycles (simple DFS)
    const visited = new Set<string>();
    const recStack = new Set<string>();

    const hasCycle = (nodeId: string): boolean => {
      visited.add(nodeId);
      recStack.add(nodeId);

      const outEdges = this.edges.filter((e) => e.from === nodeId);
      for (const edge of outEdges) {
        if (!visited.has(edge.to)) {
          if (hasCycle(edge.to)) return true;
        } else if (recStack.has(edge.to)) {
          return true;
        }
      }

      recStack.delete(nodeId);
      return false;
    };

    for (const node of this.nodes) {
      if (!visited.has(node.id)) {
        if (hasCycle(node.id)) {
          throw new Error('Workflow contains a cycle');
        }
      }
    }

    // Check output references
    for (const [name, output] of Object.entries(this.outputs)) {
      if (!nodeIds.has(output.sourceTaskId)) {
        throw new Error(`Output "${name}" references unknown task: ${output.sourceTaskId}`);
      }
    }
  }
}

/**
 * Create a new workflow builder
 */
export function workflow(): WorkflowBuilder {
  return new WorkflowBuilder();
}
