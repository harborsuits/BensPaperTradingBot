const { nanoid } = require('nanoid');
const store = require('./store');

let sockets = new Set();

function bindWebSocketServer(wss) {
  try {
    wss.on('connection', (ws) => {
      sockets.add(ws);
      ws.on('close', () => sockets.delete(ws));
      ws.on('error', () => sockets.delete(ws));
    });
  } catch {}
}

function broadcastAlert(alert) {
  const payload = JSON.stringify({ type: 'alert', payload: alert });
  for (const ws of sockets) {
    try { if (ws.readyState === 1) ws.send(payload); } catch {}
  }
}

function createAlert({ severity, source, message, trace_id = null }) {
  const alert = {
    id: nanoid(12),
    severity,
    source,
    message,
    trace_id,
    acknowledged: 0,
    ts: new Date().toISOString(),
  };
  try { store.insert(alert); } catch {}
  try { broadcastAlert(alert); } catch {}
  return alert;
}

module.exports = { bindWebSocketServer, createAlert };


