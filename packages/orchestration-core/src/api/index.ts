/**
 * API Module
 *
 * Exports the Fastify-based REST + WebSocket server.
 */

export {
  createServer,
  startServer,
  defaultConfig,
  type ServerConfig,
  type ServerContext,
  type FastifyInstance,
} from './server.js';
