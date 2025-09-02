import { describe, it, expect } from 'vitest';
import { normalizeCandidatesResponse } from './candidates';

const row = { symbol: "pltr", score: 1.23 };

describe('normalizeCandidatesResponse', () => {
  it('handles array format', () => {
    expect(normalizeCandidatesResponse([row])).toEqual([{ symbol: "PLTR", score: 1.23 }]);
  });
  
  it('handles {items} format', () => {
    expect(normalizeCandidatesResponse({ items: [row] })).toHaveLength(1);
  });
  
  it('handles {candidates} format', () => {
    expect(normalizeCandidatesResponse({ candidates: [row] })).toHaveLength(1);
  });
  
  it('handles junk data', () => {
    expect(normalizeCandidatesResponse({ foo: 1 })).toHaveLength(0);
  });
  
  it('salvages nested array', () => {
    expect(normalizeCandidatesResponse({ data: [row] })).toHaveLength(1);
  });
  
  it('dedupes by symbol', () => {
    expect(normalizeCandidatesResponse([row, { symbol: "PLTR" }])).toHaveLength(1);
  });
  
  it('handles null/undefined', () => {
    expect(normalizeCandidatesResponse(null)).toHaveLength(0);
    expect(normalizeCandidatesResponse(undefined)).toHaveLength(0);
  });
  
  it('transforms symbol to uppercase', () => {
    const result = normalizeCandidatesResponse([{ symbol: "aapl" }]);
    expect(result[0].symbol).toBe("AAPL");
  });
  
  it('preserves unknown fields', () => {
    const result = normalizeCandidatesResponse([{ symbol: "MSFT", customField: "value" }]);
    expect(result[0].customField).toBe("value");
  });
});