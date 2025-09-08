/**
 * Unit tests for pipeline stage management with hysteresis
 */

import { StageManager, PipelineStage } from '../src/logic/staging';

describe('Stage Manager with Hysteresis', () => {
  let manager: StageManager;

  beforeEach(() => {
    manager = new StageManager();
  });

  test('promotes to ROUTE when score meets threshold', () => {
    const result = manager.nextStage('AAPL', 'PLAN', 9.5, 9.3);
    expect(result.stage).toBe('ROUTE');
    expect(result.shouldTransition).toBe(true);
    expect(result.reason).toContain('Promoting to ROUTE');
  });

  test('holds ROUTE with hysteresis', () => {
    // First promote to ROUTE
    manager.nextStage('AAPL', 'PLAN', 9.5, 9.3);
    expect(manager.getLastTransition('AAPL')?.to).toBe('ROUTE');

    // Should hold ROUTE even when score drops a bit (hysteresis)
    const result = manager.nextStage('AAPL', 'ROUTE', 8.9, 9.3);
    expect(result.stage).toBe('ROUTE');
    expect(result.shouldTransition).toBe(false);
    expect(result.reason).toContain('Holding ROUTE');
  });

  test('demotes from ROUTE when score drops significantly', () => {
    // Promote to ROUTE first
    manager.nextStage('AAPL', 'PLAN', 9.5, 9.3);

    // Drop score below hysteresis threshold
    const result = manager.nextStage('AAPL', 'ROUTE', 8.6, 9.3);
    expect(result.stage).toBe('PLAN');
    expect(result.shouldTransition).toBe(true);
    expect(result.reason).toContain('Demoting to PLAN');
  });

  test('respects minimum hold time', () => {
    // This test would need mocking of Date.now() or time manipulation
    // For now, we'll test the basic logic
    const result1 = manager.nextStage('AAPL', 'PLAN', 9.5, 9.3);
    expect(result1.stage).toBe('ROUTE');

    // Immediate second call should respect hold time
    const result2 = manager.nextStage('AAPL', 'ROUTE', 8.0, 9.3);
    expect(result2.shouldTransition).toBe(false);
    expect(result2.reason).toContain('min hold time');
  });

  test('handles multiple symbols independently', () => {
    // AAPL promotes
    manager.nextStage('AAPL', 'PLAN', 9.5, 9.3);
    expect(manager.getLastTransition('AAPL')?.to).toBe('ROUTE');

    // MSFT stays in PLAN
    manager.nextStage('MSFT', 'PLAN', 7.0, 9.3);
    expect(manager.getLastTransition('MSFT')?.to).toBe('PLAN');

    // NVDA promotes
    manager.nextStage('NVDA', 'GATES', 9.5, 9.3);
    expect(manager.getLastTransition('NVDA')?.to).toBe('ROUTE');

    // Verify all are tracked separately
    const transitions = manager.getAllTransitions();
    expect(transitions.length).toBe(3);
  });

  test('handles edge cases', () => {
    // Very low score should stay in lower stages
    const result = manager.nextStage('UNKNOWN', 'CONTEXT', 2.0, 9.3);
    expect(result.stage).toBe('CONTEXT');
    expect(result.shouldTransition).toBe(false);
  });
});
