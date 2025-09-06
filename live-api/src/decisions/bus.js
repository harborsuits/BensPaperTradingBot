'use strict';

const { nanoid } = require('nanoid');

let sockets = new Set();
let ring = [];
const MAX = 200;

function bindDecisionsWS(wssDecisions) {
  try {
    wssDecisions.on('connection', (ws) => {
      sockets.add(ws);
      ws.on('close', () => sockets.delete(ws));
      ws.on('error', () => sockets.delete(ws));
    });
  } catch {}
}

function publish(decision) {
  try {
    const payload = JSON.stringify({ type: 'decision', payload: decision });
    for (const ws of sockets) {
      try { if (ws.readyState === 1) ws.send(payload); } catch {}
    }
  } catch {}

  try {
    ring.unshift(decision);
    if (ring.length > MAX) ring.pop();
  } catch {}
}

function recent(n = 50) {
  const lim = Math.max(1, Math.min(MAX, Number(n) || 50));
  return ring.slice(0, lim);
}

module.exports = { bindDecisionsWS, publish, recent };


