import { z } from 'zod';

/**
 * Safely send JSON response with schema validation
 * Fails closed if data doesn't match schema - prevents API drift
 */
export function sendJson<T>(
  res: any,
  schema: z.ZodSchema<T>,
  data: unknown,
  statusCode: number = 200
): void {
  const parsed = schema.safeParse(data);

  if (!parsed.success) {
    console.error('Schema validation failed:', {
      endpoint: res.req?.originalUrl || 'unknown',
      issues: parsed.error.issues,
      data: JSON.stringify(data).substring(0, 500)
    });

    res.status(500).json({
      error: 'SchemaMismatch',
      message: 'Backend response does not match expected schema',
      issues: parsed.error.issues
    });
    return;
  }

  res.status(statusCode).json(parsed.data);
}

/**
 * Validate incoming request data against schema
 */
export function validateRequest<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; error: z.ZodError } {
  const parsed = schema.safeParse(data);
  return parsed.success
    ? { success: true, data: parsed.data }
    : { success: false, error: parsed.error };
}

/**
 * Helper to ensure array responses are consistent
 * Handles both direct arrays and {items: []} wrapped responses
 */
export function asArray<T>(data: unknown): T[] {
  if (Array.isArray(data)) {
    return data as T[];
  }
  if (data && typeof data === 'object' && 'items' in data && Array.isArray(data.items)) {
    return data.items as T[];
  }
  return [];
}
