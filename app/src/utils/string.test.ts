import { initial, seed, capitalize } from './string';

describe('string utils', () => {
  test('initial handles null/empty/emoji', () => {
    expect(initial(undefined)).toBe('S'); // default fallback
    expect(initial('', '—')).toBe('—');
    expect(initial('equities')).toBe('E');
    expect(initial('💹', '—')).toBe('💹'); // emoji safe
  });

  test('seed is stable and safe', () => {
    expect(seed(undefined)).toBe(0);
    expect(seed('A')).toBe('A'.codePointAt(0)!);
    expect(seed('💹')).toBe('💹'.codePointAt(0)!);
  });

  test('capitalize is defensive', () => {
    expect(capitalize(undefined)).toBe('—');
    expect(capitalize('')).toBe('—');
    expect(capitalize(' bonds')).toBe('Bonds');
    expect(capitalize('💹')).toBe('💹');
  });
});
