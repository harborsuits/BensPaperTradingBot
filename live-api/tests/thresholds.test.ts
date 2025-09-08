/**
 * Unit tests for dynamic threshold calculations
 */

import { routeThreshold, planThreshold, gatesThreshold } from '../src/logic/thresholds';

describe('Dynamic Thresholds', () => {
  test('raises routeThreshold in bear market', () => {
    const normal = routeThreshold({
      vix: 18, regime: 'neutral', ddPct: 0, slippageBps: 20
    });
    const bear = routeThreshold({
      vix: 18, regime: 'bear', ddPct: 0, slippageBps: 20
    });
    expect(bear).toBeGreaterThan(normal);
    expect(bear - normal).toBe(0.4);
  });

  test('raises routeThreshold with high VIX', () => {
    const normalVix = routeThreshold({
      vix: 18, regime: 'neutral', ddPct: 0, slippageBps: 20
    });
    const highVix = routeThreshold({
      vix: 28, regime: 'neutral', ddPct: 0, slippageBps: 20
    });
    expect(highVix).toBeGreaterThan(normalVix);
    expect(highVix - normalVix).toBe(0.5); // 10pts * 0.05
  });

  test('raises routeThreshold in drawdown', () => {
    const noDD = routeThreshold({
      vix: 18, regime: 'neutral', ddPct: 0, slippageBps: 20
    });
    const withDD = routeThreshold({
      vix: 18, regime: 'neutral', ddPct: 6, slippageBps: 20
    });
    expect(withDD).toBeGreaterThan(noDD);
    expect(withDD - noDD).toBe(0.3); // 6% * 0.05 = 0.3, capped appropriately
  });

  test('raises routeThreshold with slippage', () => {
    const lowSlippage = routeThreshold({
      vix: 18, regime: 'neutral', ddPct: 0, slippageBps: 20
    });
    const highSlippage = routeThreshold({
      vix: 18, regime: 'neutral', ddPct: 0, slippageBps: 70
    });
    expect(highSlippage).toBeGreaterThan(lowSlippage);
    expect(highSlippage - lowSlippage).toBe(0.1); // 50bps * 0.002 = 0.1
  });

  test('routeThreshold caps at 9.9', () => {
    const extreme = routeThreshold({
      vix: 40, regime: 'bear', ddPct: 10, slippageBps: 100
    });
    expect(extreme).toBeLessThanOrEqual(9.9);
  });

  test('planThreshold adjusts for regime', () => {
    const bull = planThreshold({ vix: 18, regime: 'bull' });
    const bear = planThreshold({ vix: 18, regime: 'bear' });
    expect(bull).toBeLessThan(bear);
  });

  test('gatesThreshold adjusts for regime', () => {
    const bull = gatesThreshold({ vix: 18, regime: 'bull' });
    const bear = gatesThreshold({ vix: 18, regime: 'bear' });
    expect(bull).toBeLessThan(bear);
  });
});
