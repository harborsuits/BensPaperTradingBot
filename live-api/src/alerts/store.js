const fs = require('fs');
const path = require('path');
// const Database = require('better-sqlite3');

// Ensure data directory exists
const dataDir = path.join(__dirname, '../data');
try { if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true }); } catch {}

// const db = new Database(path.join(dataDir, 'alerts.db'));
// db.pragma('journal_mode = WAL');
// db.exec(`
// CREATE TABLE IF NOT EXISTS alerts (
//   id TEXT PRIMARY KEY,
//   severity TEXT NOT NULL,
//   source TEXT NOT NULL,
//   message TEXT NOT NULL,
//   trace_id TEXT,
//   acknowledged INTEGER NOT NULL DEFAULT 0,
//   ts TEXT NOT NULL
// );
// CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts DESC);
// `);

function list(limit = 10) {
  // Mock empty list for debug
  return [];
}

function ack(id) {
  // Mock for debug
  return {};
}

function insert(a) {
  // Mock for debug
}

module.exports = { list, ack, insert };


