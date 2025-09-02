// Safe string and character handling utilities

/**
 * Safely convert any value to a string
 */
export const asText = (v: unknown): string => (v ?? '').toString();

/**
 * Get the first letter/character of a string (uppercase), or fallback
 */
export const initial = (v: unknown, fallback = 'S'): string => {
  const s = asText(v).trim();
  return s ? s.slice(0, 1).toUpperCase() : fallback;
};

/**
 * Get a numeric hash seed from a string's first character
 * Uses codePointAt which is safer than charCodeAt for emojis and special characters
 */
export const seed = (v: unknown): number => {
  const s = asText(v);
  return s ? s.codePointAt(0)! : 0; 
};

/**
 * Capitalize the first letter of a string
 */
export const capitalize = (v: unknown, fallback = '—'): string => {
  const s = asText(v).trim();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : fallback;
};
