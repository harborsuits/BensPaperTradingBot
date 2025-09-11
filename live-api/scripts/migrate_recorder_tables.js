/**
 * Database migration script for MarketRecorder tables
 * Adds event-sourced logging tables for safety-critical data
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');

async function migrateRecorderTables() {
  const dbPath = path.join(__dirname, '../data/evotester.db');
  const db = new sqlite3.Database(dbPath);

  console.log('Starting recorder tables migration...');

  try {
    // Enable foreign keys
    await run(db, 'PRAGMA foreign_keys = ON');

    // Create recorder tables
    const tables = [
      `CREATE TABLE IF NOT EXISTS quotes_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        bid REAL,
        ask REAL,
        mid REAL,
        ts_feed TEXT NOT NULL,
        ts_recv TEXT NOT NULL,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      `CREATE TABLE IF NOT EXISTS chains_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        expiry TEXT NOT NULL,
        strike REAL NOT NULL,
        bid REAL,
        ask REAL,
        oi INTEGER,
        vol INTEGER,
        delta REAL,
        gamma REAL,
        theta REAL,
        vega REAL,
        rho REAL,
        iv REAL,
        ts_feed TEXT NOT NULL,
        ts_recv TEXT NOT NULL,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      `CREATE TABLE IF NOT EXISTS orders_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id TEXT NOT NULL,
        route TEXT NOT NULL,
        ladders TEXT NOT NULL,
        planned_max_slip REAL NOT NULL,
        ts_created TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      `CREATE TABLE IF NOT EXISTS fills_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id TEXT NOT NULL,
        side TEXT NOT NULL,
        price REAL NOT NULL,
        qty REAL NOT NULL,
        fees REAL DEFAULT 0,
        ts_fill TEXT NOT NULL,
        broker_attestation TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (plan_id) REFERENCES orders_snapshot (plan_id)
      )`,

      `CREATE TABLE IF NOT EXISTS ledger_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cash_before REAL NOT NULL,
        cash_after REAL NOT NULL,
        change_reason TEXT NOT NULL,
        plan_id TEXT,
        ts_change TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      // Indexes for efficient querying
      `CREATE INDEX IF NOT EXISTS idx_quotes_symbol_ts ON quotes_snapshot(symbol, ts_recv)`,
      `CREATE INDEX IF NOT EXISTS idx_chains_symbol_expiry ON chains_snapshot(symbol, expiry)`,
      `CREATE INDEX IF NOT EXISTS idx_orders_plan_id ON orders_snapshot(plan_id)`,
      `CREATE INDEX IF NOT EXISTS idx_fills_plan_id ON fills_snapshot(plan_id)`,
      `CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger_snapshot(ts_change)`
    ];

    for (const sql of tables) {
      console.log(`Creating table/index: ${sql.split('(')[0].trim()}`);
      await run(db, sql);
    }

    // Verify tables were created
    const tablesResult = await all(db, "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_snapshot'");
    console.log('Created tables:', tablesResult.map(r => r.name));

    console.log('✅ Recorder tables migration completed successfully');

  } catch (error) {
    console.error('❌ Migration failed:', error);
    throw error;
  } finally {
    db.close();
  }
}

function run(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function(err) {
      if (err) reject(err);
      else resolve({ id: this.lastID, changes: this.changes });
    });
  });
}

function all(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

// Run migration if called directly
if (require.main === module) {
  migrateRecorderTables()
    .then(() => {
      console.log('Migration completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Migration failed:', error);
      process.exit(1);
    });
}

module.exports = { migrateRecorderTables };
