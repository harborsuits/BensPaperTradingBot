/**
 * Collection of safety utilities for handling potentially undefined/null values
 */

/**
 * Safely converts any value to an array
 */
export const toArray = <T>(v: unknown): T[] => (Array.isArray(v) ? v as T[] : []);

/**
 * Safely converts any value to a string
 */
export const toString = (v: unknown, fallback = ""): string => 
  (v === undefined || v === null) ? fallback : String(v);

/**
 * Safely splits a string with fallback for null/undefined
 */
export const safeSplit = (v: unknown, sep: string | RegExp, limit?: number): string[] =>
  typeof v === "string" ? v.split(sep, limit) : [];

/**
 * Safely gets the length of an array or string with fallback for null/undefined
 */
export const len = (v: unknown): number => {
  if (v === null || v === undefined) return 0;
  if (Array.isArray(v)) return v.length;
  if (typeof v === 'string') return v.length;
  return 0;
};

/**
 * Safely gets tokens from a string by splitting with fallback
 */
export const getTokens = (s?: string, sep = ' '): string[] => 
  s?.split(sep).filter(Boolean) ?? [];

/**
 * Safely access an array element with bounds checking
 */
export const safeGet = <T>(arr: T[] | null | undefined, index: number): T | undefined => {
  if (!arr || !Array.isArray(arr)) return undefined;
  return (index >= 0 && index < arr.length) ? arr[index] : undefined;
};

/**
 * Safely convert string to lowercase with fallback
 */
export const lowerCase = (s: unknown): string => 
  typeof s === 'string' ? s.toLowerCase() : '';
