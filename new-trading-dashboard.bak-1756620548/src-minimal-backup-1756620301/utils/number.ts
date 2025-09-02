// Safe number handling utilities

/**
 * Convert any value to a number, returning undefined if not a valid number
 */
export const asNum = (v: unknown): number | undefined => {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : undefined;
};

/**
 * Format a number with fixed decimal places, with fallback for invalid numbers
 */
export const fmtNum = (v: unknown, digits = 2, fallback = '—'): string => {
  const n = asNum(v);
  return n === undefined ? fallback : n.toFixed(digits);
};

/**
 * Safely convert any value to a number, returning defaultValue if not a valid number
 */
export const asNumber = (v: unknown, defaultValue = 0): number => {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : defaultValue;
};

/**
 * Format a number as a percentage
 */
export const fmtPercent = (v: unknown, digits = 1, fallback = '—'): string => {
  const n = asNum(v);
  return n === undefined ? fallback : `${n.toFixed(digits)}%`;
};
