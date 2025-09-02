import { asNum, fmtNum, fmtPercent } from './number';

describe('number utils', () => {
  test('asNum accepts numbers and numeric strings', () => {
    expect(asNum(1.23)).toBe(1.23);
    expect(asNum('1.23')).toBe(1.23);
    expect(asNum('abc')).toBeUndefined();
    expect(asNum(undefined)).toBeUndefined();
    expect(asNum(null)).toBeUndefined();
    expect(asNum(NaN)).toBeUndefined();
  });

  test('fmtNum guards toFixed', () => {
    expect(fmtNum(1.2345)).toBe('1.23');
    expect(fmtNum('2.5', 1)).toBe('2.5');
    expect(fmtNum('abc', 2, '—')).toBe('—');
    expect(fmtNum(undefined)).toBe('—');
    expect(fmtNum(null)).toBe('—');
    expect(fmtNum(NaN)).toBe('—');
  });

  test('fmtPercent formats percentages', () => {
    expect(fmtPercent(0.1234)).toBe('12.3%');
    expect(fmtPercent('0.5', 0)).toBe('50%');
    expect(fmtPercent(undefined)).toBe('—');
    expect(fmtPercent(null)).toBe('—');
    expect(fmtPercent('invalid')).toBe('—');
  });
});
