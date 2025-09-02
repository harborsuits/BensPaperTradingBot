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
  const dry = await req('POST', '/api/paper/orders/dry-run', { symbol: 'SPY', side: 'buy', type: 'market', qty: 1 });
  assert.strictEqual(dry.status, 200);
  assert.ok(dry.json.gate);
  console.log('dry-run ok');
})();
