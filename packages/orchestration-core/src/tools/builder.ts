/**
 * Tool Builder
 *
 * Fluent API for creating tools with type safety and validation.
 */

import type {
  Tool,
  ToolDefinition,
  ToolParameter,
  ToolHandler,
  ToolExecutionOptions,
  JSONSchema,
  ToolContext,
} from './types.js';
import { ToolCategory } from './types.js';

/**
 * Fluent Tool Builder
 *
 * @example
 * ```typescript
 * const searchTool = createTool()
 *   .name('web_search')
 *   .description('Search the web for information')
 *   .category(ToolCategory.SEARCH)
 *   .parameter('query', 'string', 'Search query', { required: true })
 *   .parameter('maxResults', 'number', 'Maximum results', { default: 10 })
 *   .returns('array', 'Search results')
 *   .handler(async ({ query, maxResults }) => {
 *     // Implementation
 *   })
 *   .build();
 * ```
 */
export class ToolBuilderImpl<TInput = Record<string, unknown>, TOutput = unknown> {
  private _name: string = '';
  private _description: string = '';
  private _category: string = ToolCategory.CUSTOM;
  private _parameters: ToolParameter[] = [];
  private _returns: { type: JSONSchema['type']; description: string } = {
    type: 'object',
    description: 'Result',
  };
  private _examples: Array<{ description: string; input: TInput; output: TOutput }> = [];
  private _handler?: ToolHandler<TInput, TOutput>;
  private _options: ToolExecutionOptions = {};
  private _metadata: Record<string, unknown> = {};

  /**
   * Set the tool name (must be unique)
   */
  name(name: string): this {
    this._name = name;
    return this;
  }

  /**
   * Set the tool description
   */
  description(desc: string): this {
    this._description = desc;
    return this;
  }

  /**
   * Set the tool category
   */
  category(cat: string | ToolCategory): this {
    this._category = cat;
    return this;
  }

  /**
   * Add a parameter
   */
  parameter(
    name: string,
    type: JSONSchema['type'],
    description: string,
    options: { required?: boolean; default?: unknown; schema?: JSONSchema } = {}
  ): this {
    this._parameters.push({
      name,
      type,
      description,
      required: options.required ?? false,
      default: options.default,
      schema: options.schema,
    });
    return this;
  }

  /**
   * Add a required parameter
   */
  requiredParam(
    name: string,
    type: JSONSchema['type'],
    description: string,
    schema?: JSONSchema
  ): this {
    return this.parameter(name, type, description, { required: true, schema });
  }

  /**
   * Add an optional parameter with default
   */
  optionalParam(
    name: string,
    type: JSONSchema['type'],
    description: string,
    defaultValue: unknown,
    schema?: JSONSchema
  ): this {
    return this.parameter(name, type, description, { required: false, default: defaultValue, schema });
  }

  /**
   * Define the return type
   */
  returns(type: JSONSchema['type'], description: string): this {
    this._returns = { type, description };
    return this;
  }

  /**
   * Add an example usage
   */
  example(description: string, input: TInput, output: TOutput): this {
    this._examples.push({ description, input, output });
    return this;
  }

  /**
   * Set the handler function
   */
  handler(fn: ToolHandler<TInput, TOutput>): this {
    this._handler = fn;
    return this;
  }

  /**
   * Set execution options
   */
  options(opts: ToolExecutionOptions): this {
    this._options = { ...this._options, ...opts };
    return this;
  }

  /**
   * Set timeout
   */
  timeout(ms: number): this {
    this._options.timeout = ms;
    return this;
  }

  /**
   * Set retry count
   */
  retries(count: number): this {
    this._options.retries = count;
    return this;
  }

  /**
   * Add metadata
   */
  meta(key: string, value: unknown): this {
    this._metadata[key] = value;
    return this;
  }

  /**
   * Build the tool
   */
  build(): Tool<TInput, TOutput> {
    if (!this._name) {
      throw new Error('Tool name is required');
    }
    if (!this._description) {
      throw new Error('Tool description is required');
    }
    if (!this._handler) {
      throw new Error('Tool handler is required');
    }

    const definition: ToolDefinition = {
      name: this._name,
      description: this._description,
      category: this._category,
      parameters: this._parameters,
      returns: this._returns,
      examples: this._examples.length > 0
        ? this._examples.map(e => ({
            description: e.description,
            input: e.input as Record<string, unknown>,
            output: e.output as unknown,
          }))
        : undefined,
      metadata: Object.keys(this._metadata).length > 0 ? this._metadata : undefined,
    };

    return {
      definition,
      handler: this._handler,
      options: this._options,
    };
  }
}

/**
 * Create a new tool builder
 */
export function createTool<TInput = Record<string, unknown>, TOutput = unknown>(): ToolBuilderImpl<TInput, TOutput> {
  return new ToolBuilderImpl<TInput, TOutput>();
}

/**
 * Helper to create a simple tool from a function
 */
export function simpleTool<TInput extends Record<string, unknown>, TOutput>(
  name: string,
  description: string,
  handler: ToolHandler<TInput, TOutput>,
  options?: {
    category?: string;
    parameters?: ToolParameter[];
    timeout?: number;
  }
): Tool<TInput, TOutput> {
  return {
    definition: {
      name,
      description,
      category: options?.category ?? ToolCategory.UTILITY,
      parameters: options?.parameters ?? [],
      returns: { type: 'object', description: 'Result' },
    },
    handler,
    options: options?.timeout ? { timeout: options.timeout } : undefined,
  };
}
