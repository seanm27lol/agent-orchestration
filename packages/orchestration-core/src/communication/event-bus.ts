/**
 * Event Bus for Agent Communication
 *
 * Provides pub/sub messaging, shared state management, and
 * supports hierarchical, peer-to-peer, and broadcast patterns.
 */

import { EventEmitter } from 'events';
import { v4 as uuid } from 'uuid';
import type { AgentId, AgentMessage, ChannelId } from '../agents/types.js';

/**
 * Shared state update event
 */
export interface SharedStateUpdate {
  key: string;
  value: unknown;
  version: number;
  updatedBy: AgentId | 'system';
  timestamp: Date;
  namespace: string;
}

/**
 * Subscription handle for cleanup
 */
export interface Subscription {
  unsubscribe: () => void;
}

/**
 * Message handler callback
 */
export type MessageHandler = (message: AgentMessage) => void | Promise<void>;

/**
 * State update handler callback
 */
export type StateHandler = (update: SharedStateUpdate) => void | Promise<void>;

/**
 * Agent Event Bus
 *
 * Central communication hub supporting:
 * - Direct agent-to-agent messaging
 * - Broadcast messaging
 * - Channel-based pub/sub
 * - Shared state with versioning
 */
export class AgentEventBus {
  private emitter: EventEmitter;
  private sharedState: Map<string, SharedStateUpdate>;
  private messageHistory: AgentMessage[];
  private maxHistorySize: number;

  constructor(options: { maxHistorySize?: number } = {}) {
    this.emitter = new EventEmitter();
    this.emitter.setMaxListeners(100); // Support many agents
    this.sharedState = new Map();
    this.messageHistory = [];
    this.maxHistorySize = options.maxHistorySize ?? 1000;
  }

  /**
   * Publish a message to target agent(s)
   */
  publish(message: Omit<AgentMessage, 'id' | 'timestamp'>): AgentMessage {
    const fullMessage: AgentMessage = {
      ...message,
      id: uuid(),
      timestamp: new Date(),
    };

    // Store in history
    this.messageHistory.push(fullMessage);
    if (this.messageHistory.length > this.maxHistorySize) {
      this.messageHistory.shift();
    }

    // Route message
    if (fullMessage.target === 'broadcast') {
      this.emitter.emit('broadcast', fullMessage);
    } else {
      this.emitter.emit(`agent:${fullMessage.target}`, fullMessage);
    }

    // Always emit on channel
    this.emitter.emit(`channel:${fullMessage.channel}`, fullMessage);

    // Emit general message event for logging/monitoring
    this.emitter.emit('message', fullMessage);

    return fullMessage;
  }

  /**
   * Subscribe to messages for a specific agent
   */
  subscribeAgent(agentId: AgentId, handler: MessageHandler): Subscription {
    const wrappedHandler = (msg: AgentMessage) => handler(msg);

    // Subscribe to direct messages and broadcasts
    this.emitter.on(`agent:${agentId}`, wrappedHandler);
    this.emitter.on('broadcast', wrappedHandler);

    return {
      unsubscribe: () => {
        this.emitter.off(`agent:${agentId}`, wrappedHandler);
        this.emitter.off('broadcast', wrappedHandler);
      },
    };
  }

  /**
   * Subscribe to a specific channel
   */
  subscribeChannel(channel: ChannelId, handler: MessageHandler): Subscription {
    const wrappedHandler = (msg: AgentMessage) => handler(msg);
    this.emitter.on(`channel:${channel}`, wrappedHandler);

    return {
      unsubscribe: () => {
        this.emitter.off(`channel:${channel}`, wrappedHandler);
      },
    };
  }

  /**
   * Subscribe to all messages (for monitoring/logging)
   */
  subscribeAll(handler: MessageHandler): Subscription {
    const wrappedHandler = (msg: AgentMessage) => handler(msg);
    this.emitter.on('message', wrappedHandler);

    return {
      unsubscribe: () => {
        this.emitter.off('message', wrappedHandler);
      },
    };
  }

  /**
   * Request-response pattern: send message and wait for reply
   */
  async request(
    message: Omit<AgentMessage, 'id' | 'timestamp' | 'correlationId'>,
    timeoutMs: number = 30000
  ): Promise<AgentMessage> {
    const correlationId = uuid();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.emitter.off(`reply:${correlationId}`, handler);
        reject(new Error(`Request timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      const handler = (reply: AgentMessage) => {
        clearTimeout(timeout);
        resolve(reply);
      };

      this.emitter.once(`reply:${correlationId}`, handler);

      this.publish({
        ...message,
        correlationId,
      });
    });
  }

  /**
   * Reply to a message with a correlation ID
   */
  reply(originalMessage: AgentMessage, replyPayload: unknown): AgentMessage {
    if (!originalMessage.correlationId) {
      throw new Error('Cannot reply to message without correlationId');
    }

    const reply = this.publish({
      source: originalMessage.target === 'broadcast' ? 'system' : originalMessage.target,
      target: originalMessage.source,
      channel: originalMessage.channel,
      replyTo: originalMessage.id,
      correlationId: originalMessage.correlationId,
      payload: replyPayload,
    });

    // Emit on reply channel for request-response pattern
    this.emitter.emit(`reply:${originalMessage.correlationId}`, reply);

    return reply;
  }

  /**
   * Update shared state with optimistic locking
   */
  updateState(
    namespace: string,
    key: string,
    value: unknown,
    updatedBy: AgentId | 'system',
    expectedVersion?: number
  ): SharedStateUpdate {
    const stateKey = `${namespace}:${key}`;
    const current = this.sharedState.get(stateKey);

    // Optimistic locking check
    if (expectedVersion !== undefined && current && current.version !== expectedVersion) {
      throw new Error(
        `State update conflict: expected version ${expectedVersion}, got ${current.version}`
      );
    }

    const update: SharedStateUpdate = {
      namespace,
      key,
      value,
      version: (current?.version ?? 0) + 1,
      updatedBy,
      timestamp: new Date(),
    };

    this.sharedState.set(stateKey, update);
    this.emitter.emit('state:update', update);
    this.emitter.emit(`state:${namespace}`, update);
    this.emitter.emit(`state:${stateKey}`, update);

    return update;
  }

  /**
   * Get current state value
   */
  getState(namespace: string, key: string): unknown {
    const stateKey = `${namespace}:${key}`;
    return this.sharedState.get(stateKey)?.value;
  }

  /**
   * Get state with version info
   */
  getStateWithVersion(namespace: string, key: string): SharedStateUpdate | undefined {
    const stateKey = `${namespace}:${key}`;
    return this.sharedState.get(stateKey);
  }

  /**
   * Get all state in a namespace
   */
  getNamespaceState(namespace: string): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, update] of this.sharedState) {
      if (key.startsWith(`${namespace}:`)) {
        const shortKey = key.substring(namespace.length + 1);
        result[shortKey] = update.value;
      }
    }
    return result;
  }

  /**
   * Subscribe to state updates
   */
  subscribeState(handler: StateHandler): Subscription;
  subscribeState(namespace: string, handler: StateHandler): Subscription;
  subscribeState(
    namespaceOrHandler: string | StateHandler,
    maybeHandler?: StateHandler
  ): Subscription {
    if (typeof namespaceOrHandler === 'function') {
      // Subscribe to all state updates
      const handler = namespaceOrHandler;
      this.emitter.on('state:update', handler);
      return {
        unsubscribe: () => this.emitter.off('state:update', handler),
      };
    } else {
      // Subscribe to namespace-specific updates
      const namespace = namespaceOrHandler;
      const handler = maybeHandler!;
      this.emitter.on(`state:${namespace}`, handler);
      return {
        unsubscribe: () => this.emitter.off(`state:${namespace}`, handler),
      };
    }
  }

  /**
   * Get message history (for debugging/replay)
   */
  getHistory(options: {
    channel?: ChannelId;
    source?: AgentId;
    target?: AgentId;
    limit?: number;
  } = {}): AgentMessage[] {
    let messages = [...this.messageHistory];

    if (options.channel) {
      messages = messages.filter((m) => m.channel === options.channel);
    }
    if (options.source) {
      messages = messages.filter((m) => m.source === options.source);
    }
    if (options.target) {
      messages = messages.filter((m) => m.target === options.target);
    }
    if (options.limit) {
      messages = messages.slice(-options.limit);
    }

    return messages;
  }

  /**
   * Clear all state and history (for testing)
   */
  clear(): void {
    this.sharedState.clear();
    this.messageHistory = [];
  }

  /**
   * Get stats about the event bus
   */
  getStats(): {
    messageCount: number;
    stateKeyCount: number;
    listenerCount: number;
  } {
    return {
      messageCount: this.messageHistory.length,
      stateKeyCount: this.sharedState.size,
      listenerCount: this.emitter.listenerCount('message'),
    };
  }
}

// Export singleton instance for convenience
export const globalEventBus = new AgentEventBus();
