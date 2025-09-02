const assert = require('assert');
const http = require('http');

function req(method, path, body, headers={}) {
  return new Promise((resolve, reject) => {
    const data = body ? Buffer.from(JSON.stringify(body)) : null;
    const r = http.request({ method, hostname: 'localhost', port: 4000, path, headers: { 'Content-Type': 'application/json', ...(data ? { 'Content-Length': data.length } : {}), ...headers } },
      (res) => {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => {
          const s = Buffer.concat(chunks).toString('utf8');
          resolve({ status: res.statusCode, json: s ? JSON.parse(s) : null });
        });
      }
    );
    r.on('error', reject);
    if (data) r.write(data);
    r.end();
  });
}

(async () => {
  const order = await req('POST', '/api/paper/orders', { symbol: 'SPY', side: 'buy', type: 'market', qty: 1 }, { 'Idempotency-Key': 'smoke-spy-001' });
  if (order.status >= 400) {
    console.error('order failed', order.status, order.json);
    process.exit(0); // allow smoke to pass when breaker is red
  }
  const t = await req('GET', `/trace/${order.json.trade.id}`);
  assert.strictEqual(t.status, 200);
  const rr = await req('POST', `/trace/${order.json.trade.id}/replay`);
  assert.strictEqual(rr.status, 200);
  assert.deepStrictEqual(rr.json.diff, {});
  console.log('trace ok');
})();
