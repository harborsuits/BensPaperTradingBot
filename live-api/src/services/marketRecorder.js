/**
 * MarketRecorder - Event-sourced logging for safety-critical market data
 * Records all quotes, chains, orders, fills, and ledger changes for proof validation
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const { runtime } = require('./runtimeConfig');

class MarketRecorder {
  constructor(dbPath = './data/evotester.db') {
    this.dbPath = dbPath;
    this.db = null;
    this.initialized = false;
  }

  /**
   * Initialize database connection and create tables
   */
  async initialize() {
    if (this.initialized) return;

    this.db = new sqlite3.Database(this.dbPath);

    await this.createTables();
    this.initialized = true;

    console.log('MarketRecorder initialized with event-sourced tables');
  }

  /**
   * Create all recorder tables
   */
  async createTables() {
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
      await this.run(sql);
    }
  }

  /**
   * Record quote snapshot with WORM audit trail
   */
  async recordQuote(symbol, quote, ts_feed, ts_recv, source = 'tradier') {
    await this.initialize();

    const auditStamp = runtime.generateAuditStamp(ts_feed, ts_recv);

    const sql = `
      INSERT INTO quotes_snapshot (symbol, bid, ask, mid, ts_feed, ts_recv, source)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;

    const mid = quote.bid && quote.ask ? (quote.bid + quote.ask) / 2 : null;

    try {
      const result = await this.run(sql, [symbol, quote.bid, quote.ask, mid, ts_feed, ts_recv, source]);

      // Record audit trail (WORM mode)
      await this.recordAuditTrail('quote', result.id, auditStamp);

      return result;
    } catch (error) {
      console.error(`Failed to record quote with audit trail:`, error);
      throw error;
    }
  }

  /**
   * Record option chain snapshot
   */
  async recordChain(symbol, chain, ts_feed, ts_recv, source = 'tradier') {
    await this.initialize();

    const sql = `
      INSERT INTO chains_snapshot (symbol, expiry, strike, bid, ask, oi, vol, delta, gamma, theta, vega, rho, iv, ts_feed, ts_recv, source)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    for (const option of chain.options || []) {
      await this.run(sql, [
        symbol,
        chain.expiry,
        option.strike,
        option.bid,
        option.ask,
        option.oi,
        option.vol,
        option.greeks?.delta,
        option.greeks?.gamma,
        option.greeks?.theta,
        option.greeks?.vega,
        option.greeks?.rho,
        option.impliedVolatility,
        ts_feed,
        ts_recv,
        source
      ]);
    }
  }

  /**
   * Record order plan snapshot
   */
  async recordOrder(plan_id, route, ladders, planned_max_slip) {
    await this.initialize();

    const sql = `
      INSERT INTO orders_snapshot (plan_id, route, ladders, planned_max_slip, ts_created)
      VALUES (?, ?, ?, ?, ?)
    `;

    await this.run(sql, [
      plan_id,
      route,
      JSON.stringify(ladders),
      planned_max_slip,
      new Date().toISOString()
    ]);
  }

  /**
   * Record fill execution
   */
  async recordFill(plan_id, side, price, qty, fees = 0, ts_fill, broker_attestation = null) {
    await this.initialize();

    const sql = `
      INSERT INTO fills_snapshot (plan_id, side, price, qty, fees, ts_fill, broker_attestation)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;

    await this.run(sql, [plan_id, side, price, qty, fees, ts_fill, broker_attestation]);
  }

  /**
   * Record ledger change
   */
  async recordLedgerChange(cash_before, cash_after, change_reason, plan_id = null) {
    await this.initialize();

    const sql = `
      INSERT INTO ledger_snapshot (cash_before, cash_after, change_reason, plan_id, ts_change)
      VALUES (?, ?, ?, ?, ?)
    `;

    await this.run(sql, [
      cash_before,
      cash_after,
      change_reason,
      plan_id,
      new Date().toISOString()
    ]);
  }

  /**
   * Query helpers for proofs
   */

  /**
   * Get latest fresh quote for symbol
   */
  async getLatestFreshQuote(symbol, maxAgeMs = 5000) {
    await this.initialize();

    const sql = `
      SELECT * FROM quotes_snapshot
      WHERE symbol = ?
        AND (strftime('%s', 'now') * 1000 - strftime('%s', ts_recv) * 1000) <= ?
      ORDER BY ts_recv DESC
      LIMIT 1
    `;

    return await this.get(sql, [symbol, maxAgeMs]);
  }

  /**
   * Get NBBO mid at specific timestamp
   */
  async getNBBOAtTime(symbol, timestamp, toleranceMs = 1000) {
    await this.initialize();

    const sql = `
      SELECT mid FROM quotes_snapshot
      WHERE symbol = ?
        AND ABS(strftime('%s', ts_recv) * 1000 - strftime('%s', ?) * 1000) <= ?
      ORDER BY ABS(strftime('%s', ts_recv) * 1000 - strftime('%s', ?) * 1000)
      LIMIT 1
    `;

    return await this.get(sql, [symbol, timestamp, toleranceMs, timestamp]);
  }

  /**
   * Get fills for plan_id
   */
  async getFillsForPlan(plan_id) {
    await this.initialize();

    const sql = `SELECT * FROM fills_snapshot WHERE plan_id = ? ORDER BY ts_fill`;
    return await this.all(sql, [plan_id]);
  }

  /**
   * Get ledger changes in time window
   */
  async getLedgerChanges(startTime, endTime) {
    await this.initialize();

    const sql = `
      SELECT * FROM ledger_snapshot
      WHERE ts_change >= ? AND ts_change <= ?
      ORDER BY ts_change
    `;

    return await this.all(sql, [startTime, endTime]);
  }

  /**
   * Get 24h friction statistics
   */
  async get24hFrictionStats(windowStart) {
    await this.initialize();

    const sql = `
      SELECT
        COUNT(*) as total_fills,
        AVG(fees / (price * qty)) as avg_friction,
        SUM(CASE WHEN (fees / (price * qty)) <= 0.20 THEN 1 ELSE 0 END) as friction_20_count,
        SUM(CASE WHEN (fees / (price * qty)) <= 0.25 THEN 1 ELSE 0 END) as friction_25_count
      FROM fills_snapshot
      WHERE ts_fill >= ?
    `;

    return await this.get(sql, [windowStart]);
  }

  /**
   * Record audit trail for WORM compliance
   */
  async recordAuditTrail(recordType, recordId, auditStamp) {
    await this.initialize();

    // Create audit trail table if it doesn't exist
    const createAuditTable = `
      CREATE TABLE IF NOT EXISTS audit_trail (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_type TEXT NOT NULL,
        record_id INTEGER NOT NULL,
        ts_feed TEXT,
        ts_recv TEXT,
        server_ts TEXT NOT NULL,
        commit_hash TEXT,
        policy_hash TEXT,
        environment TEXT,
        worm_mode BOOLEAN DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )
    `;

    await this.run(createAuditTable);

    const sql = `
      INSERT INTO audit_trail
      (record_type, record_id, ts_feed, ts_recv, server_ts, commit_hash, policy_hash, environment, worm_mode)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    await this.run(sql, [
      recordType,
      recordId,
      auditStamp.ts_feed,
      auditStamp.ts_recv,
      auditStamp.server_ts,
      auditStamp.commit_hash,
      auditStamp.policy_hash,
      auditStamp.environment,
      auditStamp.worm_mode
    ]);
  }

  /**
   * SQLite helper methods
   */
  run(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve({ id: this.lastID, changes: this.changes });
      });
    });
  }

  get(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  all(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  close() {
    if (this.db) {
      this.db.close();
      this.initialized = false;
    }
  }
}

// Singleton instance
const recorder = new MarketRecorder();

module.exports = { MarketRecorder, recorder };
