/**
 * API Server
 *
 * Fastify-based REST + WebSocket server for the orchestration framework.
 * Provides endpoints for workflow management, agent operations, and real-time updates.
 */

import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import fastifyCors from '@fastify/cors';
import { globalEventBus } from '../communication/event-bus.js';
import { AgentRegistry, globalAgentRegistry } from '../agents/registry.js';
import { globalToolRegistry, ToolRegistry } from '../tools/registry.js';

// ============================================================================
// Server Configuration
// ============================================================================

export interface ServerConfig {
  host: string;
  port: number;
  cors: {
    origin: string | string[] | boolean;
    credentials?: boolean;
  };
  logger: boolean;
  pythonBridgeUrl?: string;
}

export const defaultConfig: ServerConfig = {
  host: '0.0.0.0',
  port: 3000,
  cors: {
    origin: true,
    credentials: true,
  },
  logger: true,
  pythonBridgeUrl: 'http://localhost:8000',
};

// ============================================================================
// Server Context
// ============================================================================

export interface ServerContext {
  agentRegistry: AgentRegistry;
  toolRegistry: ToolRegistry;
  eventBus: typeof globalEventBus;
  pythonBridgeUrl: string;
}

// Extend FastifyInstance to include our context
declare module 'fastify' {
  interface FastifyInstance {
    ctx: ServerContext;
  }
}

// ============================================================================
// Server Factory
// ============================================================================

export async function createServer(
  config: Partial<ServerConfig> = {}
): Promise<FastifyInstance> {
  const fullConfig = { ...defaultConfig, ...config };

  const fastify = Fastify({
    logger: fullConfig.logger,
  });

  // Register context
  fastify.decorate('ctx', {
    agentRegistry: globalAgentRegistry,
    toolRegistry: globalToolRegistry,
    eventBus: globalEventBus,
    pythonBridgeUrl: fullConfig.pythonBridgeUrl || 'http://localhost:8000',
  });

  // Register CORS
  await fastify.register(fastifyCors, {
    origin: fullConfig.cors.origin,
    credentials: fullConfig.cors.credentials,
  });

  // Register WebSocket support
  await fastify.register(fastifyWebsocket);

  // Register routes
  await registerRoutes(fastify);

  return fastify;
}

// ============================================================================
// Route Registration
// ============================================================================

async function registerRoutes(fastify: FastifyInstance): Promise<void> {
  // Health check
  fastify.get('/health', async () => {
    return {
      status: 'healthy',
      version: '0.1.0',
      timestamp: new Date().toISOString(),
      agents: fastify.ctx.agentRegistry.listAll().length,
      tools: fastify.ctx.toolRegistry.listAll().length,
    };
  });

  // ========================================================================
  // Agent Routes
  // ========================================================================

  // List registered agent types
  fastify.get('/agents/types', async () => {
    const types = fastify.ctx.agentRegistry.listAgentTypes();
    return { types };
  });

  // List active agents
  fastify.get('/agents', async () => {
    const agents = fastify.ctx.agentRegistry.listAll();
    return { agents };
  });

  // Get agent details
  fastify.get<{ Params: { id: string } }>('/agents/:id', async (request, reply) => {
    const agent = fastify.ctx.agentRegistry.get(request.params.id);
    if (!agent) {
      reply.status(404);
      return { error: 'Agent not found' };
    }
    return agent;
  });

  // Spawn agent via Python bridge
  fastify.post<{
    Body: { agentType: string; agentId?: string; config?: Record<string, unknown> };
  }>('/agents/spawn', async (request, reply) => {
    const { agentType, agentId, config } = request.body;

    try {
      const response = await fetch(`${fastify.ctx.pythonBridgeUrl}/agents/spawn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_type: agentType,
          agent_id: agentId,
          config: config || {},
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        reply.status(response.status);
        return error;
      }

      return await response.json();
    } catch (error) {
      reply.status(503);
      return { error: 'Python bridge unavailable', details: String(error) };
    }
  });

  // Step agent via Python bridge
  fastify.post<{
    Params: { id: string };
    Body: { observation: Record<string, unknown>; context?: Record<string, unknown> };
  }>('/agents/:id/step', async (request, reply) => {
    const { id } = request.params;
    const { observation, context } = request.body;

    try {
      const response = await fetch(`${fastify.ctx.pythonBridgeUrl}/agents/${id}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ observation, context }),
      });

      if (!response.ok) {
        const error = await response.json();
        reply.status(response.status);
        return error;
      }

      return await response.json();
    } catch (error) {
      reply.status(503);
      return { error: 'Python bridge unavailable', details: String(error) };
    }
  });

  // Terminate agent via Python bridge
  fastify.delete<{ Params: { id: string } }>('/agents/:id', async (request, reply) => {
    const { id } = request.params;

    try {
      const response = await fetch(`${fastify.ctx.pythonBridgeUrl}/agents/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        reply.status(response.status);
        return error;
      }

      return await response.json();
    } catch (error) {
      reply.status(503);
      return { error: 'Python bridge unavailable', details: String(error) };
    }
  });

  // ========================================================================
  // Tool Routes
  // ========================================================================

  // List all tools
  fastify.get('/tools', async () => {
    const tools = fastify.ctx.toolRegistry.listAll();
    return { tools };
  });

  // Get tool definition
  fastify.get<{ Params: { name: string } }>('/tools/:name', async (request, reply) => {
    const tool = fastify.ctx.toolRegistry.get(request.params.name);
    if (!tool) {
      reply.status(404);
      return { error: 'Tool not found' };
    }
    return tool.definition;
  });

  // Execute tool
  fastify.post<{
    Params: { name: string };
    Body: { arguments: Record<string, unknown>; agentId?: string };
  }>('/tools/:name/execute', async (request, reply) => {
    const { name } = request.params;
    const { arguments: args, agentId } = request.body;

    try {
      const result = await fastify.ctx.toolRegistry.execute({
        toolName: name,
        arguments: args,
        callId: `call-${Date.now()}`,
        context: {
          agentId: agentId || 'api-caller',
        },
        timestamp: new Date(),
      });

      return result;
    } catch (error) {
      reply.status(400);
      return { error: String(error) };
    }
  });

  // ========================================================================
  // Event Bus Routes
  // ========================================================================

  // Publish message to event bus
  fastify.post<{
    Body: {
      source: string;
      target?: string;
      channel: string;
      payload: unknown;
    };
  }>('/events/publish', async (request) => {
    const { source, target, channel, payload } = request.body;

    const message = fastify.ctx.eventBus.publish({
      source,
      target: target || 'broadcast',
      channel,
      payload,
    });

    return { messageId: message.id, timestamp: message.timestamp };
  });

  // Get shared state
  fastify.get<{
    Params: { namespace: string };
    Querystring: { key?: string };
  }>('/state/:namespace', async (request) => {
    const { namespace } = request.params;
    const { key } = request.query;

    if (key) {
      const value = fastify.ctx.eventBus.getState(namespace, key);
      return { namespace, key, value };
    }

    // Return all state in namespace
    const state = fastify.ctx.eventBus.getNamespaceState(namespace);
    return { namespace, state };
  });

  // Update shared state
  fastify.put<{
    Params: { namespace: string; key: string };
    Body: { value: unknown; updatedBy: string };
  }>('/state/:namespace/:key', async (request) => {
    const { namespace, key } = request.params;
    const { value, updatedBy } = request.body;

    const update = fastify.ctx.eventBus.updateState(namespace, key, value, updatedBy);

    return update;
  });

  // ========================================================================
  // WebSocket Routes
  // ========================================================================

  // Real-time event stream
  fastify.get('/ws/events', { websocket: true }, (socket, request) => {
    const subscriptions: Array<{ unsubscribe: () => void }> = [];

    socket.on('message', (rawMessage: Buffer) => {
      try {
        const message = JSON.parse(rawMessage.toString());

        switch (message.type) {
          case 'subscribe_channel': {
            const sub = fastify.ctx.eventBus.subscribeChannel(
              message.channel,
              (msg) => {
                socket.send(JSON.stringify({ type: 'message', message: msg }));
              }
            );
            subscriptions.push(sub);
            socket.send(JSON.stringify({ type: 'subscribed', channel: message.channel }));
            break;
          }

          case 'subscribe_state': {
            const sub = fastify.ctx.eventBus.subscribeState(
              message.namespace,
              (update) => {
                socket.send(JSON.stringify({ type: 'state_update', update }));
              }
            );
            subscriptions.push(sub);
            socket.send(JSON.stringify({ type: 'subscribed_state', namespace: message.namespace }));
            break;
          }

          case 'publish': {
            const pubMsg = fastify.ctx.eventBus.publish({
              source: message.source || 'websocket',
              target: message.target || 'broadcast',
              channel: message.channel,
              payload: message.payload,
            });
            socket.send(JSON.stringify({ type: 'published', messageId: pubMsg.id }));
            break;
          }

          case 'ping':
            socket.send(JSON.stringify({ type: 'pong' }));
            break;
        }
      } catch (error) {
        socket.send(JSON.stringify({ type: 'error', message: String(error) }));
      }
    });

    socket.on('close', () => {
      for (const sub of subscriptions) {
        sub.unsubscribe();
      }
    });
  });

  // Agent-specific WebSocket stream (proxies to Python bridge)
  fastify.get<{ Params: { id: string } }>('/ws/agents/:id', { websocket: true }, async (socket, request) => {
    const { id } = request.params;

    socket.on('message', async (rawMessage: Buffer) => {
      try {
        const message = JSON.parse(rawMessage.toString());

        if (message.type === 'step') {
          // Forward to Python bridge
          const response = await fetch(`${fastify.ctx.pythonBridgeUrl}/agents/${id}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              observation: message.observation,
              context: message.context,
            }),
          });

          const result = await response.json() as Record<string, unknown>;
          socket.send(JSON.stringify({ type: 'action', ...result }));
        } else if (message.type === 'ping') {
          socket.send(JSON.stringify({ type: 'pong' }));
        }
      } catch (error) {
        socket.send(JSON.stringify({ type: 'error', message: String(error) }));
      }
    });
  });
}

// ============================================================================
// Server Runner
// ============================================================================

export async function startServer(config: Partial<ServerConfig> = {}): Promise<FastifyInstance> {
  const fullConfig = { ...defaultConfig, ...config };
  const server = await createServer(config);

  try {
    await server.listen({ host: fullConfig.host, port: fullConfig.port });
    console.log(`Server listening on http://${fullConfig.host}:${fullConfig.port}`);
    return server;
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Export for direct execution
export { FastifyInstance };
