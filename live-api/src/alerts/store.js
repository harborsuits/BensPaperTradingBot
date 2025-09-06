const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

// Ensure data directory exists
const dataDir = path.join(__dirname, '../data');
try { if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true }); } catch {}

const db = new Database(path.join(dataDir, 'alerts.db'));
db.pragma('journal_mode = WAL');
db.exec(`
CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY,
  severity TEXT NOT NULL,
  source TEXT NOT NULL,
  message TEXT NOT NULL,
  trace_id TEXT,
  acknowledged INTEGER NOT NULL DEFAULT 0,
  ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts DESC);
`);

function list(limit = 10) {
  const lim = Math.max(1, Math.min(100, Number(limit) || 10));
  return db.prepare(`SELECT * FROM alerts ORDER BY ts DESC LIMIT ?`).all(lim);
}

function ack(id) {
  db.prepare(`UPDATE alerts SET acknowledged=1 WHERE id=?`).run(id);
  return db.prepare(`SELECT * FROM alerts WHERE id=?`).get(id);
}

function insert(a) {
  db.prepare(`
    INSERT INTO alerts (id,severity,source,message,trace_id,acknowledged,ts)
    VALUES (@id,@severity,@source,@message,@trace_id,@acknowledged,@ts)
  `).run(a);
}

module.exports = { list, ack, insert };


