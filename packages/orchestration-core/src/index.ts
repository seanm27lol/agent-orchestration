/**
 * Agent Orchestration Framework
 *
 * A hybrid TypeScript/Python framework for orchestrating
 * multiple AI agents working together on complex tasks.
 *
 * @packageDocumentation
 */

// Agent types and abstractions
export type {
  AgentId,
  TaskId,
  WorkflowId,
  ChannelId,
  AgentRegistration,
  AgentMessage,
  AgentSpawnConfig,
  AgentSpawnResponse,
  AgentPoolConfig,
  AgentLifecycleEvent,
  AgentObservation,
  AgentAction,
  ToolCall,
  DelegationRequest,
  AgentMetrics,
  AgentState,
} from './agents/types.js';
export {
  AgentCapability,
  AgentStatus,
} from './agents/types.js';
// Note: ToolResult from agents/types conflicts with tools/types - use tools version
export { AgentRegistry, globalAgentRegistry } from './agents/registry.js';

// Communication
export {
  AgentEventBus,
  globalEventBus,
  type SharedStateUpdate,
  type Subscription,
  type MessageHandler,
  type StateHandler,
} from './communication/event-bus.js';

// Workflow types
export type {
  TaskInput,
  RetryPolicy,
  TaskNode,
  ConditionalBranch,
  TaskEdge,
  WorkflowInputDef,
  WorkflowOutputDef,
  WorkflowDAG,
  TaskExecutionState,
  WorkflowExecutionState,
  WorkflowEvent,
} from './workflow/types.js';
export {
  TaskExecutionStatus,
  WorkflowExecutionStatus,
} from './workflow/types.js';
export { WorkflowBuilder, workflow } from './workflow/dsl.js';
export {
  WorkflowExecutor,
  globalWorkflowExecutor,
  executeWorkflow,
  type ExecutionConfig,
  type TaskExecution,
  type WorkflowExecution,
  type ExecutionProgress,
  type TaskHandler,
} from './workflow/executor.js';

// Tools
export type {
  ToolDefinition,
  ToolParameter,
  ToolInvocation,
  ToolResult,
  ToolContext,
  ToolExecutionOptions,
  Tool,
  ToolHandler,
} from './tools/types.js';
export { ToolCategory, ToolErrorCode } from './tools/types.js';
export {
  ToolRegistry,
  globalToolRegistry,
  ToolValidationError,
  ToolExecutionError,
} from './tools/registry.js';
export { ToolBuilderImpl, createTool, simpleTool } from './tools/builder.js';
export {
  calculatorTool,
  jsonParserTool,
  textTransformTool,
  dateTimeTool,
  randomTool,
  httpFetchTool,
  exampleTools,
  registerExampleTools,
} from './tools/examples.js';

// API Server
export {
  createServer,
  startServer,
  defaultConfig,
  type ServerConfig,
  type ServerContext,
} from './api/index.js';

// Version
export const VERSION = '0.1.0';
