import { z } from 'zod';

// Valid status values
const STATUS_VALUES = ['active', 'idle', 'stopped', 'error', 'paused', 'waiting', 'unknown'] as const;
export const StrategyStatus = z.enum(STATUS_VALUES);

// Define a schema for strategy performance metrics
export const StrategyPerformance = z.object({
  win_rate: z.number().min(0).max(1).nullable().optional().default(null),
  sharpe_ratio: z.number().nullable().optional().default(null),
  max_drawdown: z.number().nullable().optional().default(null),
  trades_count: z.number().int().nullable().optional().default(0),
}).default({});

// Define a schema for individual strategies with safer coercion
export const Strategy = z.object({
  id: z.string().or(z.number().transform(n => String(n))),
  name: z.string().default('—'),
  // Handle unknown status values safely
  status: z.string()
    .transform(s => {
      // Convert to lowercase for case-insensitive matching
      const normalized = s?.toLowerCase?.() || '';
      
      // Check if it's a known status
      if (STATUS_VALUES.includes(normalized as any)) {
        return normalized as z.infer<typeof StrategyStatus>;
      }
      
      // Map to the closest matching status or default to idle
      if (normalized === 'on' || normalized === 'running') return 'active';
      if (normalized === 'off' || normalized === 'disabled') return 'idle';
      if (normalized === 'failed' || normalized === 'crashed') return 'error';
      
      return 'idle'; // Default fallback
    }),
  asset_class: z.string().nullable().optional().default('unknown'),
  description: z.string().nullable().optional(),
  priority_score: z.number().min(0).max(1).default(0)
    .or(z.string().transform(s => {
      const n = parseFloat(s);
      return isNaN(n) ? 0 : Math.max(0, Math.min(1, n));
    })),
  performance: StrategyPerformance,
}).default({});

// Define a schema for strategy response formats
export const StrategyList = z.object({
  items: z.array(Strategy),
}).or(z.object({
  data: z.array(Strategy),
})).or(z.array(Strategy));

// Helper function to safely parse a strategy response
export function parseStrategyList(data: unknown): z.infer<typeof Strategy>[] {
  // Handle null/undefined
  if (data == null) {
    console.warn('Strategy data is null/undefined');
    return [];
  }
  
  try {
    // Try to parse with schema
    const result = StrategyList.safeParse(data);
    
    if (result.success) {
      if (Array.isArray(result.data)) {
        return result.data;
      } else if ('items' in result.data) {
        return result.data.items;
      } else if ('data' in result.data) {
        return result.data.data;
      }
    } else {
      // Log the validation errors
      console.warn('Strategy schema validation failed:', result.error.format());
      
      // Try to extract array data anyway for resilience
      const rawData = data as any;
      if (Array.isArray(rawData)) {
        return rawData.map(item => Strategy.parse(item));
      } else if (Array.isArray(rawData?.items)) {
        return rawData.items.map(item => Strategy.parse(item));
      } else if (Array.isArray(rawData?.data)) {
        return rawData.data.map(item => Strategy.parse(item));
      }
    }
  } catch (error) {
    console.error('Error parsing strategy data:', error);
  }
  
  // Fallback to empty array if parsing fails
  return [];
}
