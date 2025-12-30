/**
 * Example Tool Implementations
 *
 * Ready-to-use tools that can be registered with the ToolRegistry.
 */

import { createTool } from './builder.js';
import { ToolCategory } from './types.js';
import type { Tool, ToolContext } from './types.js';

// ============================================================================
// Calculator Tool
// ============================================================================

interface CalculatorInput {
  expression: string;
}

export const calculatorTool = createTool<CalculatorInput, number>()
  .name('calculator')
  .description('Evaluate a mathematical expression')
  .category(ToolCategory.UTILITY)
  .requiredParam('expression', 'string', 'Mathematical expression to evaluate (e.g., "2 + 2 * 3")')
  .returns('number', 'Result of the calculation')
  .example('Simple addition', { expression: '2 + 2' }, 4)
  .example('Complex expression', { expression: '(10 + 5) * 2 / 3' }, 10)
  .timeout(5000)
  .handler(async ({ expression }) => {
    // Safe math evaluation (no eval!)
    const sanitized = expression.replace(/[^0-9+\-*/().%\s]/g, '');
    if (sanitized !== expression) {
      throw new Error('Expression contains invalid characters');
    }

    // Use Function constructor for safer evaluation
    const result = new Function(`return ${sanitized}`)();
    if (typeof result !== 'number' || isNaN(result)) {
      throw new Error('Expression did not evaluate to a number');
    }

    return result;
  })
  .build();

// ============================================================================
// JSON Parser Tool
// ============================================================================

interface JsonParseInput {
  text: string;
  path?: string;
}

export const jsonParserTool = createTool<JsonParseInput, unknown>()
  .name('json_parser')
  .description('Parse JSON text and optionally extract a value by path')
  .category(ToolCategory.DATA)
  .requiredParam('text', 'string', 'JSON text to parse')
  .optionalParam('path', 'string', 'Dot-notation path to extract (e.g., "user.name")', undefined)
  .returns('object', 'Parsed JSON or extracted value')
  .timeout(5000)
  .handler(async ({ text, path }) => {
    const parsed = JSON.parse(text);

    if (!path) return parsed;

    // Extract value by path
    const parts = path.split('.');
    let value: unknown = parsed;
    for (const part of parts) {
      if (value === null || value === undefined) break;
      value = (value as Record<string, unknown>)[part];
    }

    return value;
  })
  .build();

// ============================================================================
// Text Transform Tool
// ============================================================================

interface TextTransformInput {
  text: string;
  operation: 'uppercase' | 'lowercase' | 'capitalize' | 'reverse' | 'trim' | 'slugify';
}

export const textTransformTool = createTool<TextTransformInput, string>()
  .name('text_transform')
  .description('Transform text using various operations')
  .category(ToolCategory.UTILITY)
  .requiredParam('text', 'string', 'Text to transform')
  .requiredParam('operation', 'string', 'Operation: uppercase, lowercase, capitalize, reverse, trim, slugify')
  .returns('string', 'Transformed text')
  .handler(async ({ text, operation }) => {
    switch (operation) {
      case 'uppercase':
        return text.toUpperCase();
      case 'lowercase':
        return text.toLowerCase();
      case 'capitalize':
        return text.replace(/\b\w/g, (c) => c.toUpperCase());
      case 'reverse':
        return text.split('').reverse().join('');
      case 'trim':
        return text.trim();
      case 'slugify':
        return text
          .toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-');
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
  })
  .build();

// ============================================================================
// Date/Time Tool
// ============================================================================

interface DateTimeInput {
  operation: 'now' | 'parse' | 'format' | 'diff';
  date?: string;
  date2?: string;
  format?: string;
  timezone?: string;
}

interface DateTimeOutput {
  iso: string;
  unix: number;
  formatted?: string;
  diff?: {
    days: number;
    hours: number;
    minutes: number;
    seconds: number;
  };
}

export const dateTimeTool = createTool<DateTimeInput, DateTimeOutput>()
  .name('datetime')
  .description('Work with dates and times')
  .category(ToolCategory.UTILITY)
  .requiredParam('operation', 'string', 'Operation: now, parse, format, diff')
  .optionalParam('date', 'string', 'Date string to parse or format', undefined)
  .optionalParam('date2', 'string', 'Second date for diff operation', undefined)
  .optionalParam('format', 'string', 'Format string for output', undefined)
  .optionalParam('timezone', 'string', 'Timezone (e.g., "America/New_York")', undefined)
  .returns('object', 'Date information')
  .handler(async ({ operation, date, date2 }) => {
    let d: Date;

    switch (operation) {
      case 'now':
        d = new Date();
        break;
      case 'parse':
        if (!date) throw new Error('date parameter required for parse');
        d = new Date(date);
        break;
      case 'format':
        if (!date) throw new Error('date parameter required for format');
        d = new Date(date);
        break;
      case 'diff':
        if (!date || !date2) throw new Error('date and date2 required for diff');
        const d1 = new Date(date);
        const d2 = new Date(date2);
        const diffMs = Math.abs(d2.getTime() - d1.getTime());
        return {
          iso: d1.toISOString(),
          unix: d1.getTime(),
          diff: {
            days: Math.floor(diffMs / (1000 * 60 * 60 * 24)),
            hours: Math.floor((diffMs / (1000 * 60 * 60)) % 24),
            minutes: Math.floor((diffMs / (1000 * 60)) % 60),
            seconds: Math.floor((diffMs / 1000) % 60),
          },
        };
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }

    if (isNaN(d.getTime())) {
      throw new Error('Invalid date');
    }

    return {
      iso: d.toISOString(),
      unix: d.getTime(),
      formatted: d.toLocaleString(),
    };
  })
  .build();

// ============================================================================
// Random Generator Tool
// ============================================================================

interface RandomInput {
  type: 'number' | 'string' | 'uuid' | 'choice';
  min?: number;
  max?: number;
  length?: number;
  choices?: string[];
  charset?: 'alphanumeric' | 'alphabetic' | 'numeric' | 'hex';
}

export const randomTool = createTool<RandomInput, string | number>()
  .name('random')
  .description('Generate random values')
  .category(ToolCategory.UTILITY)
  .requiredParam('type', 'string', 'Type: number, string, uuid, choice')
  .optionalParam('min', 'number', 'Minimum value for number', 0)
  .optionalParam('max', 'number', 'Maximum value for number', 100)
  .optionalParam('length', 'number', 'Length for string', 16)
  .optionalParam('choices', 'array', 'Array of choices for choice type', undefined)
  .optionalParam('charset', 'string', 'Character set for string', 'alphanumeric')
  .returns('string', 'Generated random value')
  .handler(async ({ type, min = 0, max = 100, length = 16, choices, charset = 'alphanumeric' }) => {
    switch (type) {
      case 'number':
        return Math.floor(Math.random() * (max - min + 1)) + min;

      case 'uuid':
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
          const r = (Math.random() * 16) | 0;
          const v = c === 'x' ? r : (r & 0x3) | 0x8;
          return v.toString(16);
        });

      case 'string':
        const charsets: Record<string, string> = {
          alphanumeric: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
          alphabetic: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
          numeric: '0123456789',
          hex: '0123456789abcdef',
        };
        const chars = charsets[charset] || charsets.alphanumeric;
        let result = '';
        for (let i = 0; i < length; i++) {
          result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;

      case 'choice':
        if (!choices || choices.length === 0) {
          throw new Error('choices array required for choice type');
        }
        return choices[Math.floor(Math.random() * choices.length)];

      default:
        throw new Error(`Unknown type: ${type}`);
    }
  })
  .build();

// ============================================================================
// HTTP Fetch Tool (Placeholder - would need actual implementation)
// ============================================================================

interface HttpFetchInput {
  url: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: unknown;
}

interface HttpFetchOutput {
  status: number;
  headers: Record<string, string>;
  body: unknown;
}

export const httpFetchTool = createTool<HttpFetchInput, HttpFetchOutput>()
  .name('http_fetch')
  .description('Make HTTP requests to external APIs')
  .category(ToolCategory.WEB)
  .requiredParam('url', 'string', 'URL to fetch')
  .optionalParam('method', 'string', 'HTTP method', 'GET')
  .optionalParam('headers', 'object', 'Request headers', {})
  .optionalParam('body', 'object', 'Request body for POST/PUT', undefined)
  .returns('object', 'HTTP response with status, headers, and body')
  .timeout(30000)
  .handler(async ({ url, method = 'GET', headers = {}, body }) => {
    // Note: This is a placeholder. In a real implementation,
    // you would use fetch() or a similar HTTP client.
    // For now, we'll simulate a response.

    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });

    let responseBody: unknown;
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      responseBody = await response.json();
    } else {
      responseBody = await response.text();
    }

    return {
      status: response.status,
      headers: responseHeaders,
      body: responseBody,
    };
  })
  .build();

// ============================================================================
// Export all example tools
// ============================================================================

export const exampleTools: Tool[] = [
  calculatorTool,
  jsonParserTool,
  textTransformTool,
  dateTimeTool,
  randomTool,
  httpFetchTool,
];

/**
 * Register all example tools with a registry
 */
export function registerExampleTools(registry: {
  register: (tool: Tool) => void;
}): void {
  for (const tool of exampleTools) {
    registry.register(tool);
  }
}
