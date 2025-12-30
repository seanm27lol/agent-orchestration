/**
 * Hello World Example
 *
 * Demonstrates a simple 2-agent workflow where:
 * 1. A "Greeter" agent generates a greeting
 * 2. A "Responder" agent responds to the greeting
 *
 * This example runs entirely in TypeScript without the Python bridge,
 * using mock agents for demonstration purposes.
 */

import {
  AgentEventBus,
  AgentRegistry,
  WorkflowBuilder,
  AgentCapability,
  AgentStatus,
  type AgentMessage,
  type AgentRegistration,
  type WorkflowDAG,
} from '../../packages/orchestration-core/src/index.js';

// ============================================================================
// Mock Agent Implementations
// ============================================================================

interface MockAgent {
  registration: AgentRegistration;
  process: (input: unknown) => Promise<unknown>;
}

/**
 * Create a mock Greeter agent
 */
function createGreeterAgent(eventBus: AgentEventBus): MockAgent {
  const registration: AgentRegistration = {
    id: 'greeter-001',
    type: 'greeter',
    name: 'Greeter Agent',
    capabilities: [AgentCapability.TEXT_GENERATION],
    endpoint: 'mock://greeter',
    status: AgentStatus.AVAILABLE,
    metadata: {},
    registeredAt: new Date(),
    lastHeartbeat: new Date(),
  };

  const process = async (input: unknown): Promise<unknown> => {
    const name = (input as { name?: string })?.name ?? 'World';

    // Simulate processing
    await sleep(100);

    const greeting = `Hello, ${name}! Welcome to the Agent Orchestration Framework.`;

    // Publish message on event bus
    eventBus.publish({
      source: registration.id,
      target: 'broadcast',
      channel: 'greetings',
      payload: { greeting, generatedAt: new Date().toISOString() },
    });

    return { greeting };
  };

  return { registration, process };
}

/**
 * Create a mock Responder agent
 */
function createResponderAgent(eventBus: AgentEventBus): MockAgent {
  const registration: AgentRegistration = {
    id: 'responder-001',
    type: 'responder',
    name: 'Responder Agent',
    capabilities: [AgentCapability.TEXT_GENERATION, AgentCapability.REASONING],
    endpoint: 'mock://responder',
    status: AgentStatus.AVAILABLE,
    metadata: {},
    registeredAt: new Date(),
    lastHeartbeat: new Date(),
  };

  const process = async (input: unknown): Promise<unknown> => {
    const greeting = (input as { greeting?: string })?.greeting ?? '';

    // Simulate processing
    await sleep(150);

    const response = `Thank you for the warm welcome! I'm excited to collaborate. "${greeting}" - that's a great way to start!`;

    // Publish response
    eventBus.publish({
      source: registration.id,
      target: 'broadcast',
      channel: 'responses',
      payload: { response, respondedAt: new Date().toISOString() },
    });

    return { response };
  };

  return { registration, process };
}

// ============================================================================
// Simple Workflow Executor
// ============================================================================

interface TaskResult {
  taskId: string;
  output: unknown;
  duration: number;
}

async function executeWorkflow(
  workflow: WorkflowDAG,
  inputs: Record<string, unknown>,
  agents: Map<string, MockAgent>
): Promise<Map<string, TaskResult>> {
  console.log(`\n🚀 Executing workflow: ${workflow.name} (v${workflow.version})`);
  console.log(`   Inputs: ${JSON.stringify(inputs)}\n`);

  const results = new Map<string, TaskResult>();
  const executed = new Set<string>();

  // Find tasks with no dependencies (start nodes)
  const getReadyTasks = (): string[] => {
    return workflow.nodes
      .filter((node) => {
        if (executed.has(node.id)) return false;

        // Check if all dependencies are complete
        const incomingEdges = workflow.edges.filter((e) => e.to === node.id);
        return incomingEdges.every((e) => executed.has(e.from));
      })
      .map((n) => n.id);
  };

  // Execute tasks in topological order
  while (executed.size < workflow.nodes.length) {
    const readyTasks = getReadyTasks();

    if (readyTasks.length === 0 && executed.size < workflow.nodes.length) {
      throw new Error('Workflow has unresolvable dependencies');
    }

    // Execute ready tasks (could be parallel in production)
    for (const taskId of readyTasks) {
      const node = workflow.nodes.find((n) => n.id === taskId)!;
      const agent = agents.get(node.agentType);

      if (!agent) {
        throw new Error(`No agent found for type: ${node.agentType}`);
      }

      console.log(`   📋 Task "${node.name}" starting (agent: ${node.agentType})`);

      // Resolve inputs
      const taskInputs: Record<string, unknown> = {};
      for (const [key, inputDef] of Object.entries(node.inputs)) {
        if (inputDef.type === 'workflow_input') {
          taskInputs[key] = inputs[inputDef.inputKey!];
        } else if (inputDef.type === 'task_output') {
          const sourceResult = results.get(inputDef.sourceTaskId!);
          // Extract specific key from output if specified
          const outputObj = sourceResult?.output as Record<string, unknown>;
          taskInputs[key] = inputDef.outputKey && outputObj
            ? outputObj[inputDef.outputKey]
            : outputObj;
        } else {
          taskInputs[key] = inputDef.value;
        }
      }

      const startTime = Date.now();
      const output = await agent.process(taskInputs);
      const duration = Date.now() - startTime;

      results.set(taskId, { taskId, output, duration });
      executed.add(taskId);

      console.log(`   ✅ Task "${node.name}" completed (${duration}ms)`);
      console.log(`      Output: ${JSON.stringify(output)}\n`);
    }
  }

  return results;
}

// ============================================================================
// Utility Functions
// ============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  Agent Orchestration Framework - Hello World Example');
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Create event bus
  const eventBus = new AgentEventBus();

  // Create agent registry
  const registry = new AgentRegistry(eventBus);
  registry.start();

  // Create mock agents
  const greeterAgent = createGreeterAgent(eventBus);
  const responderAgent = createResponderAgent(eventBus);

  // Register agents
  await registry.register(greeterAgent.registration);
  await registry.register(responderAgent.registration);

  console.log('📡 Registered agents:');
  for (const agent of registry.listAll()) {
    console.log(`   - ${agent.name} (${agent.type}) [${agent.capabilities.join(', ')}]`);
  }

  // Create agent map for executor
  const agents = new Map<string, MockAgent>([
    ['greeter', greeterAgent],
    ['responder', responderAgent],
  ]);

  // Subscribe to event bus for monitoring
  eventBus.subscribeAll((message: AgentMessage) => {
    console.log(`   📨 [${message.channel}] ${message.source} → ${message.target}`);
  });

  // Define workflow using the DSL
  const workflow = new WorkflowBuilder()
    .describe('A simple greeting workflow with two agents')
    .input('name', { type: 'string', required: true, description: 'Name to greet' })
    .task('greet', 'greeter', {
      name: 'Generate Greeting',
      inputs: { name: '$input.name' },
    })
    .task('respond', 'responder', {
      name: 'Generate Response',
      inputs: { greeting: '$task.greet.greeting' },
    })
    .output('finalResponse', 'respond', 'response')
    .build('hello-world', 'Hello World Workflow', '1.0.0');

  console.log('\n📝 Workflow Definition:');
  console.log(`   Name: ${workflow.name}`);
  console.log(`   Version: ${workflow.version}`);
  console.log(`   Tasks: ${workflow.nodes.map((n) => n.name).join(' → ')}`);

  // Execute workflow
  const results = await executeWorkflow(workflow, { name: 'Developer' }, agents);

  // Get final output
  const finalResult = results.get('respond');

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  Workflow Complete!');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`\n🎉 Final Response: ${(finalResult?.output as { response: string })?.response}\n`);

  // Show stats
  console.log('📊 Statistics:');
  const stats = registry.getStats();
  console.log(`   Total Agents: ${stats.totalAgents}`);
  console.log(`   By Status: ${JSON.stringify(stats.byStatus)}`);
  console.log(`   Event Bus: ${JSON.stringify(eventBus.getStats())}`);

  // Cleanup
  registry.stop();
}

// Run the example
main().catch(console.error);
