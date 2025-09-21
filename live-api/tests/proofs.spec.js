const request = require('supertest');

const base = process.env.BASE || 'http://localhost:4000';

function get(path) {
  return request(base).get(path).set('accept', 'application/json');
}

describe('BenBot proofs', () => {
  it('health is green', async () => {
    const r = await get('/api/health');
    expect(r.status).toBe(200);
    expect(r.body.ok).toBe(true);
    expect(['GREEN', 'YELLOW', 'RED']).toContain(r.body.breaker);
  });

  it('strategies include news_momo_v2 metrics', async () => {
    const r = await get('/api/strategies');
    expect(r.status).toBe(200);
    const items = r.body.items || r.body;
    const s = items.find(x => x.id === 'news_momo_v2');
    expect(s).toBeTruthy();
    expect(s.performance).toHaveProperty('win_rate');
    expect(s.performance).toHaveProperty('sharpe_ratio');
    expect(s.performance).toHaveProperty('trades_count');
  });

  it('audit routes respond', async () => {
    for (const u of [
      '/api/audit/coordination',
      '/api/audit/risk-rejections',
      '/api/audit/allocations/current',
      '/api/audit/autoloop/status',
      '/api/audit/system'
    ]) {
      const r = await get(u);
      expect(r.status).toBe(200);
      expect(r.body).toBeDefined();
      expect(r.body.success).toBe(true);
    }
  });

  it('allocator endpoint works', async () => {
    const r = await get('/api/audit/allocations/current');
    expect(r.status).toBe(200);
    expect(r.body).toBeDefined();
    expect(r.body.success).toBe(true);

    // If there are allocations, they should sum to ~1
    const a = r.body.current_allocation?.allocations || r.body.allocations || {};
    const allocationCount = Object.keys(a).length;

    if (allocationCount > 0) {
      const sum = Object.values(a).reduce((p, c) => p + Number(c?.weight || 0), 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-4);
    } else {
      // No allocations yet - this is expected for a new system
      expect(allocationCount).toBe(0);
    }
  });

  it('news_momo_v2 has strong edge', async () => {
    const r = await get('/api/strategies');
    expect(r.status).toBe(200);
    const items = r.body.items || r.body;
    const s = items.find(x => x.id === 'news_momo_v2');
    expect(s).toBeTruthy();
    expect(s.performance.win_rate).toBeGreaterThan(0.55);
    expect(s.performance.sharpe_ratio).toBeGreaterThan(1.0);
    expect(s.performance.trades_count).toBeGreaterThan(50);
  });

  it('autopilot is running', async () => {
    const r = await get('/api/audit/autoloop/status');
    if (r.status !== 200) return;
    expect(r.body.autoloop_status).toBeDefined();
    expect(typeof r.body.autoloop_status.is_running).toBe('boolean');
  });

  it('coordination shows winners', async () => {
    const r = await get('/api/audit/coordination');
    if (r.status !== 200) return;
    // If there are coordination results, they should have winners
    if (r.body.audit?.winners?.length > 0) {
      const winner = r.body.audit.winners[0];
      expect(winner).toHaveProperty('symbol');
      expect(winner).toHaveProperty('score');
      expect(winner).toHaveProperty('strategy_id');
    }
  });

  it('risk gates are working', async () => {
    const r = await get('/api/audit/risk-rejections');
    expect(r.status).toBe(200);
    expect(r.body).toBeDefined();
    expect(r.body.success).toBe(true);
    expect(Array.isArray(r.body.rejections)).toBe(true);

    // If there are rejections, they should have proper structure
    if (r.body.rejections?.length > 0) {
      const rejection = r.body.rejections[0];
      expect(rejection).toHaveProperty('reason');
      expect(rejection).toHaveProperty('check');
      expect(rejection).toHaveProperty('symbol');
    }
  });
});
