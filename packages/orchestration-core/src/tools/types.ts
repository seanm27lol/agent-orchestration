/**
 * Tool System Types
 *
 * Defines the structure for creating, registering, and executing tools
 * that agents can use to perform actions.
 */

import type { AgentId } from '../agents/types.js';

/**
 * JSON Schema type for parameter validation
 */
export interface JSONSchema {
  type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'null';
  description?: string;
  properties?: Record<string, JSONSchema>;
  items?: JSONSchema;
  required?: string[];
  enum?: unknown[];
  default?: unknown;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
}

/**
 * Parameter definition for a tool
 */
export interface ToolParameter {
  name: string;
  type: JSONSchema['type'];
  description: string;
  required: boolean;
  default?: unknown;
  schema?: JSONSchema;
}

/**
 * Tool definition - describes what the tool does
 */
export interface ToolDefinition {
  name: string;
  description: string;
  category: string;
  parameters: ToolParameter[];
  returns: {
    type: JSONSchema['type'];
    description: string;
    schema?: JSONSchema;
  };
  examples?: Array<{
    description: string;
    input: Record<string, unknown>;
    output: unknown;
  }>;
  metadata?: Record<string, unknown>;
}

/**
 * Tool execution context - provides info about the calling agent
 */
export interface ToolContext {
  agentId: AgentId;
  executionId?: string;
  workflowId?: string;
  conversationId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Tool invocation request
 */
export interface ToolInvocation {
  toolName: string;
  arguments: Record<string, unknown>;
  callId: string;
  context: ToolContext;
  timestamp: Date;
}

/**
 * Tool execution result
 */
export interface ToolResult {
  callId: string;
  toolName: string;
  success: boolean;
  result?: unknown;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  duration: number;
  timestamp: Date;
}

/**
 * Tool execution options
 */
export interface ToolExecutionOptions {
  timeout?: number;
  retries?: number;
  validateInput?: boolean;
  validateOutput?: boolean;
}

/**
 * Tool handler function type
 */
export type ToolHandler<TInput = Record<string, unknown>, TOutput = unknown> = (
  input: TInput,
  context: ToolContext
) => Promise<TOutput>;

/**
 * Complete tool configuration
 */
export interface Tool<TInput = Record<string, unknown>, TOutput = unknown> {
  definition: ToolDefinition;
  handler: ToolHandler<TInput, TOutput>;
  options?: ToolExecutionOptions;
}

/**
 * Tool builder for fluent tool creation
 */
export interface ToolBuilder<TInput = Record<string, unknown>, TOutput = unknown> {
  name(name: string): ToolBuilder<TInput, TOutput>;
  description(desc: string): ToolBuilder<TInput, TOutput>;
  category(cat: string): ToolBuilder<TInput, TOutput>;
  parameter<K extends string>(
    name: K,
    type: JSONSchema['type'],
    description: string,
    options?: { required?: boolean; default?: unknown }
  ): ToolBuilder<TInput & Record<K, unknown>, TOutput>;
  returns(type: JSONSchema['type'], description: string): ToolBuilder<TInput, TOutput>;
  example(description: string, input: TInput, output: TOutput): ToolBuilder<TInput, TOutput>;
  handler(fn: ToolHandler<TInput, TOutput>): ToolBuilder<TInput, TOutput>;
  options(opts: ToolExecutionOptions): ToolBuilder<TInput, TOutput>;
  build(): Tool<TInput, TOutput>;
}

/**
 * Tool category for organization
 */
export enum ToolCategory {
  WEB = 'web',
  CODE = 'code',
  DATA = 'data',
  FILE = 'file',
  COMMUNICATION = 'communication',
  SEARCH = 'search',
  UTILITY = 'utility',
  CUSTOM = 'custom',
}

/**
 * Common error codes for tool execution
 */
export enum ToolErrorCode {
  INVALID_INPUT = 'INVALID_INPUT',
  EXECUTION_ERROR = 'EXECUTION_ERROR',
  TIMEOUT = 'TIMEOUT',
  NOT_FOUND = 'NOT_FOUND',
  PERMISSION_DENIED = 'PERMISSION_DENIED',
  RATE_LIMITED = 'RATE_LIMITED',
  EXTERNAL_ERROR = 'EXTERNAL_ERROR',
}
