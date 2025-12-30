/**
 * Tool Registry and Executor
 *
 * Manages tool registration, discovery, and execution with
 * validation, timeout handling, and error management.
 */

import { v4 as uuid } from 'uuid';
import type {
  Tool,
  ToolDefinition,
  ToolInvocation,
  ToolResult,
  ToolContext,
  ToolExecutionOptions,
  ToolHandler,
  ToolParameter,
} from './types.js';
import { ToolErrorCode } from './types.js';

/**
 * Validation error for tool inputs
 */
export class ToolValidationError extends Error {
  constructor(
    public readonly toolName: string,
    public readonly parameter: string,
    message: string
  ) {
    super(`Tool "${toolName}" parameter "${parameter}": ${message}`);
    this.name = 'ToolValidationError';
  }
}

/**
 * Tool execution error
 */
export class ToolExecutionError extends Error {
  constructor(
    public readonly toolName: string,
    public readonly code: ToolErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(`Tool "${toolName}" execution failed: ${message}`);
    this.name = 'ToolExecutionError';
  }
}

/**
 * Tool Registry
 *
 * Central registry for all available tools. Handles:
 * - Tool registration and discovery
 * - Input validation
 * - Execution with timeout
 * - Result formatting
 */
export class ToolRegistry {
  private tools: Map<string, Tool> = new Map();
  private executionHistory: ToolResult[] = [];
  private maxHistorySize: number;

  constructor(options: { maxHistorySize?: number } = {}) {
    this.maxHistorySize = options.maxHistorySize ?? 1000;
  }

  /**
   * Register a tool
   */
  register<TInput, TOutput>(tool: Tool<TInput, TOutput>): void {
    const name = tool.definition.name;
    if (this.tools.has(name)) {
      throw new Error(`Tool "${name}" is already registered`);
    }
    this.tools.set(name, tool as Tool);
  }

  /**
   * Unregister a tool
   */
  unregister(name: string): boolean {
    return this.tools.delete(name);
  }

  /**
   * Get a tool by name
   */
  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  /**
   * Get tool definition (safe to share with agents)
   */
  getDefinition(name: string): ToolDefinition | undefined {
    return this.tools.get(name)?.definition;
  }

  /**
   * List all registered tools
   */
  listAll(): ToolDefinition[] {
    return Array.from(this.tools.values()).map((t) => t.definition);
  }

  /**
   * List tools by category
   */
  listByCategory(category: string): ToolDefinition[] {
    return this.listAll().filter((t) => t.category === category);
  }

  /**
   * Search tools by name or description
   */
  search(query: string): ToolDefinition[] {
    const lowerQuery = query.toLowerCase();
    return this.listAll().filter(
      (t) =>
        t.name.toLowerCase().includes(lowerQuery) ||
        t.description.toLowerCase().includes(lowerQuery)
    );
  }

  /**
   * Execute a tool
   */
  async execute(invocation: ToolInvocation): Promise<ToolResult> {
    const startTime = Date.now();
    const { toolName, arguments: args, callId, context } = invocation;

    const tool = this.tools.get(toolName);
    if (!tool) {
      return this.createErrorResult(callId, toolName, startTime, {
        code: ToolErrorCode.NOT_FOUND,
        message: `Tool "${toolName}" not found`,
      });
    }

    // Validate input if enabled
    if (tool.options?.validateInput !== false) {
      try {
        this.validateInput(tool.definition, args);
      } catch (error) {
        return this.createErrorResult(callId, toolName, startTime, {
          code: ToolErrorCode.INVALID_INPUT,
          message: error instanceof Error ? error.message : 'Validation failed',
        });
      }
    }

    // Execute with timeout
    const timeout = tool.options?.timeout ?? 30000;

    try {
      const result = await this.executeWithTimeout(
        tool.handler(args, context),
        timeout
      );

      const toolResult: ToolResult = {
        callId,
        toolName,
        success: true,
        result,
        duration: Date.now() - startTime,
        timestamp: new Date(),
      };

      this.addToHistory(toolResult);
      return toolResult;
    } catch (error) {
      const isTimeout = error instanceof Error && error.message === 'TIMEOUT';

      return this.createErrorResult(callId, toolName, startTime, {
        code: isTimeout ? ToolErrorCode.TIMEOUT : ToolErrorCode.EXECUTION_ERROR,
        message: error instanceof Error ? error.message : 'Execution failed',
        details: error,
      });
    }
  }

  /**
   * Execute multiple tools in parallel
   */
  async executeParallel(invocations: ToolInvocation[]): Promise<ToolResult[]> {
    return Promise.all(invocations.map((inv) => this.execute(inv)));
  }

  /**
   * Create an invocation object
   */
  createInvocation(
    toolName: string,
    args: Record<string, unknown>,
    context: ToolContext
  ): ToolInvocation {
    return {
      toolName,
      arguments: args,
      callId: uuid(),
      context,
      timestamp: new Date(),
    };
  }

  /**
   * Validate tool input against parameter definitions
   */
  private validateInput(
    definition: ToolDefinition,
    input: Record<string, unknown>
  ): void {
    for (const param of definition.parameters) {
      const value = input[param.name];

      // Check required parameters
      if (param.required && value === undefined) {
        throw new ToolValidationError(
          definition.name,
          param.name,
          'Required parameter is missing'
        );
      }

      // Skip validation for undefined optional parameters
      if (value === undefined) continue;

      // Type validation
      if (!this.validateType(value, param.type)) {
        throw new ToolValidationError(
          definition.name,
          param.name,
          `Expected type "${param.type}", got "${typeof value}"`
        );
      }
    }
  }

  /**
   * Validate value against expected type
   */
  private validateType(value: unknown, expectedType: string): boolean {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'object':
        return typeof value === 'object' && value !== null && !Array.isArray(value);
      case 'array':
        return Array.isArray(value);
      case 'null':
        return value === null;
      default:
        return true;
    }
  }

  /**
   * Execute with timeout
   */
  private async executeWithTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('TIMEOUT'));
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

  /**
   * Create error result
   */
  private createErrorResult(
    callId: string,
    toolName: string,
    startTime: number,
    error: { code: ToolErrorCode; message: string; details?: unknown }
  ): ToolResult {
    const result: ToolResult = {
      callId,
      toolName,
      success: false,
      error,
      duration: Date.now() - startTime,
      timestamp: new Date(),
    };

    this.addToHistory(result);
    return result;
  }

  /**
   * Add result to history
   */
  private addToHistory(result: ToolResult): void {
    this.executionHistory.push(result);
    if (this.executionHistory.length > this.maxHistorySize) {
      this.executionHistory.shift();
    }
  }

  /**
   * Get execution history
   */
  getHistory(options: {
    toolName?: string;
    success?: boolean;
    limit?: number;
  } = {}): ToolResult[] {
    let results = [...this.executionHistory];

    if (options.toolName) {
      results = results.filter((r) => r.toolName === options.toolName);
    }
    if (options.success !== undefined) {
      results = results.filter((r) => r.success === options.success);
    }
    if (options.limit) {
      results = results.slice(-options.limit);
    }

    return results;
  }

  /**
   * Get registry stats
   */
  getStats(): {
    totalTools: number;
    byCategory: Record<string, number>;
    executionCount: number;
    successRate: number;
  } {
    const tools = this.listAll();
    const byCategory: Record<string, number> = {};

    for (const tool of tools) {
      byCategory[tool.category] = (byCategory[tool.category] ?? 0) + 1;
    }

    const totalExecutions = this.executionHistory.length;
    const successfulExecutions = this.executionHistory.filter((r) => r.success).length;

    return {
      totalTools: tools.length,
      byCategory,
      executionCount: totalExecutions,
      successRate: totalExecutions > 0 ? successfulExecutions / totalExecutions : 1,
    };
  }
}

// Export singleton instance
export const globalToolRegistry = new ToolRegistry();
