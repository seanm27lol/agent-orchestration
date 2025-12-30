/**
 * Agent Orchestration Framework
 *
 * A hybrid TypeScript/Python framework for orchestrating
 * multiple AI agents working together on complex tasks.
 *
 * @packageDocumentation
 */

// Agent types and abstractions
export * from './agents/types.js';
export { AgentRegistry } from './agents/registry.js';

// Communication
export {
  AgentEventBus,
  globalEventBus,
  type SharedStateUpdate,
  type Subscription,
  type MessageHandler,
  type StateHandler,
} from './communication/event-bus.js';

// Workflow
export * from './workflow/types.js';
export { WorkflowBuilder, workflow } from './workflow/dsl.js';

// Version
export const VERSION = '0.1.0';
