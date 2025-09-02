import { describe, it, expect } from 'vitest';
import { safeSplit, toArray, len, toString, getTokens, safeGet, lowerCase } from './safe';

describe('Safe utilities', () => {
  describe('safeSplit', () => {
    it('handles normal string splits', () => {
      expect(safeSplit('a,b,c', ',')).toEqual(['a', 'b', 'c']);
    });
    
    it('handles undefined values', () => {
      expect(safeSplit(undefined, ',')).toEqual([]);
    });
    
    it('handles null values', () => {
      expect(safeSplit(null, ',')).toEqual([]);
    });
    
    it('handles non-string values', () => {
      expect(safeSplit(42, ',')).toEqual([]);
    });
    
    it('handles regex patterns', () => {
      expect(safeSplit('a-b_c', /[-_]/)).toEqual(['a', 'b', 'c']);
    });
  });
  
  describe('toArray', () => {
    it('returns original array', () => {
      expect(toArray([1, 2, 3])).toEqual([1, 2, 3]);
    });
    
    it('handles undefined', () => {
      expect(toArray(undefined)).toEqual([]);
    });
    
    it('handles null', () => {
      expect(toArray(null)).toEqual([]);
    });
    
    it('handles primitive values', () => {
      expect(toArray(42)).toEqual([]);
      expect(toArray('string')).toEqual([]);
      expect(toArray(true)).toEqual([]);
    });
  });
  
  describe('len', () => {
    it('returns array length', () => {
      expect(len([1, 2, 3])).toBe(3);
    });
    
    it('returns string length', () => {
      expect(len('hello')).toBe(5);
    });
    
    it('handles undefined', () => {
      expect(len(undefined)).toBe(0);
    });
    
    it('handles null', () => {
      expect(len(null)).toBe(0);
    });
    
    it('handles non-array, non-string values', () => {
      expect(len(42)).toBe(0);
      expect(len({})).toBe(0);
      expect(len(true)).toBe(0);
    });
  });
  
  describe('toString', () => {
    it('returns string values', () => {
      expect(toString('hello')).toBe('hello');
    });
    
    it('converts numbers to strings', () => {
      expect(toString(42)).toBe('42');
    });
    
    it('handles undefined with fallback', () => {
      expect(toString(undefined)).toBe('');
      expect(toString(undefined, 'N/A')).toBe('N/A');
    });
    
    it('handles null with fallback', () => {
      expect(toString(null)).toBe('');
      expect(toString(null, 'N/A')).toBe('N/A');
    });
  });
  
  describe('getTokens', () => {
    it('splits strings into tokens', () => {
      expect(getTokens('hello world')).toEqual(['hello', 'world']);
    });
    
    it('filters out empty tokens', () => {
      expect(getTokens('hello  world')).toEqual(['hello', 'world']);
    });
    
    it('handles custom separators', () => {
      expect(getTokens('hello,world', ',')).toEqual(['hello', 'world']);
    });
    
    it('handles undefined', () => {
      expect(getTokens(undefined)).toEqual([]);
    });
  });
  
  describe('safeGet', () => {
    it('safely gets array elements', () => {
      expect(safeGet(['a', 'b', 'c'], 1)).toBe('b');
    });
    
    it('returns undefined for out of bounds indexes', () => {
      expect(safeGet(['a', 'b', 'c'], 5)).toBeUndefined();
      expect(safeGet(['a', 'b', 'c'], -1)).toBeUndefined();
    });
    
    it('handles null/undefined arrays', () => {
      expect(safeGet(null, 0)).toBeUndefined();
      expect(safeGet(undefined, 0)).toBeUndefined();
    });
  });
  
  describe('lowerCase', () => {
    it('converts strings to lowercase', () => {
      expect(lowerCase('HELLO')).toBe('hello');
    });
    
    it('handles non-string values', () => {
      expect(lowerCase(null)).toBe('');
      expect(lowerCase(undefined)).toBe('');
      expect(lowerCase(42)).toBe('');
    });
  });
});
