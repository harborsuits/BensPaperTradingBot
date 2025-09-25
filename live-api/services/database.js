/**
 * Database Service
 * 
 * Provides persistent storage for trading data using SQLite.
 * Handles trades, decisions, strategies, and system state.
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

class DatabaseService {
  constructor(config = {}) {
    this.config = {
      dbPath: config.dbPath || path.join(__dirname, '../data/trading.db'),
      wal: config.wal !== false, // Write-Ahead Logging for better concurrency
      busyTimeout: config.busyTimeout || 5000,
      ...config
    };
    
    // Ensure data directory exists
    const dbDir = path.dirname(this.config.dbPath);
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }
    
    this.db = null;
    this.isInitialized = false;
  }
  
  /**
   * Initialize database connection and create tables
   */
  async initialize() {
    return new Promise((resolve, reject) => {
      this.db = new sqlite3.Database(this.config.dbPath, (err) => {
        if (err) {
          console.error('[Database] Failed to open database:', err);
          reject(err);
          return;
        }
        
        console.log('[Database] Connected to SQLite database');
        
        // Configure database
        this.db.serialize(() => {
          // Enable WAL mode for better concurrency
          if (this.config.wal) {
            this.db.run('PRAGMA journal_mode=WAL');
          }
          
          // Set busy timeout
          this.db.run(`PRAGMA busy_timeout=${this.config.busyTimeout}`);
          
          // Create tables
          this.createTables()
            .then(() => {
              this.isInitialized = true;
              console.log('[Database] All tables created successfully');
              resolve();
            })
            .catch(reject);
        });
      });
    });
  }
  
  /**
   * Create all required tables
   */
  async createTables() {
    const tables = [
      // Trades table
      `CREATE TABLE IF NOT EXISTS trades (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        strategy_id TEXT,
        decision_id TEXT,
        order_id TEXT,
        status TEXT DEFAULT 'pending',
        pnl REAL,
        pnl_percent REAL,
        fees REAL,
        slippage REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        executed_at TIMESTAMP,
        metadata TEXT
      )`,
      
      // Decisions table
      `CREATE TABLE IF NOT EXISTS decisions (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        strategy_id TEXT,
        confidence REAL,
        brain_score REAL,
        stage TEXT,
        reason TEXT,
        analysis TEXT,
        market_context TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        executed BOOLEAN DEFAULT 0,
        trade_id TEXT,
        metadata TEXT
      )`,
      
      // Strategies table
      `CREATE TABLE IF NOT EXISTS strategies (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        family TEXT,
        status TEXT DEFAULT 'paper',
        round TEXT DEFAULT 'R1',
        capital_allocated REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        promoted_at TIMESTAMP,
        demoted_at TIMESTAMP,
        parameters TEXT,
        performance TEXT,
        metadata TEXT
      )`,
      
      // Orders table
      `CREATE TABLE IF NOT EXISTS orders (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL NOT NULL,
        order_type TEXT NOT NULL,
        limit_price REAL,
        stop_price REAL,
        status TEXT DEFAULT 'pending',
        broker_order_id TEXT,
        filled_quantity REAL DEFAULT 0,
        avg_fill_price REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        submitted_at TIMESTAMP,
        filled_at TIMESTAMP,
        cancelled_at TIMESTAMP,
        metadata TEXT
      )`,
      
      // Performance metrics table
      `CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id TEXT,
        metric_type TEXT NOT NULL,
        value REAL NOT NULL,
        period TEXT,
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
      )`,
      
      // System state table
      `CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Alerts table
      `CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity TEXT NOT NULL,
        source TEXT NOT NULL,
        message TEXT NOT NULL,
        context TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        acknowledged BOOLEAN DEFAULT 0,
        acknowledged_at TIMESTAMP
      )`,
      
      // Bot competitions table
      `CREATE TABLE IF NOT EXISTS bot_competitions (
        id TEXT PRIMARY KEY,
        status TEXT DEFAULT 'active',
        trigger TEXT,
        total_pool REAL,
        bot_count INTEGER,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMP,
        winner_bot_id TEXT,
        results TEXT
      )`,
      
      // Evolution history table
      `CREATE TABLE IF NOT EXISTS evolution_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation INTEGER,
        family TEXT,
        phenotype_id TEXT,
        fitness REAL,
        parameters TEXT,
        backtest_results TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )`
    ];
    
    // Create indexes
    const indexes = [
      'CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)',
      'CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at)',
      'CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id)',
      'CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol)',
      'CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at)',
      'CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status)',
      'CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)',
      'CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)'
    ];
    
    // Execute all CREATE TABLE statements
    for (const sql of tables) {
      await this.run(sql);
    }
    
    // Create indexes
    for (const sql of indexes) {
      await this.run(sql);
    }
  }
  
  /**
   * Promisified run method
   */
  run(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve({ lastID: this.lastID, changes: this.changes });
      });
    });
  }
  
  /**
   * Promisified get method
   */
  get(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }
  
  /**
   * Promisified all method
   */
  all(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
  
  /**
   * Record a trade
   */
  async recordTrade(trade) {
    const sql = `
      INSERT INTO trades (
        id, symbol, side, quantity, price, strategy_id, decision_id,
        order_id, status, pnl, pnl_percent, fees, slippage, 
        executed_at, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      trade.id || `trade_${Date.now()}_${Math.random().toString(36).substring(7)}`,
      trade.symbol,
      trade.side,
      trade.quantity,
      trade.price,
      trade.strategy_id || null,
      trade.decision_id || null,
      trade.order_id || null,
      trade.status || 'executed',
      trade.pnl || null,
      trade.pnl_percent || null,
      trade.fees || 0,
      trade.slippage || 0,
      trade.executed_at || new Date().toISOString(),
      JSON.stringify(trade.metadata || {})
    ];
    
    await this.run(sql, params);
    return params[0]; // Return trade ID
  }
  
  /**
   * Record a decision
   */
  async recordDecision(decision) {
    const sql = `
      INSERT INTO decisions (
        id, symbol, action, strategy_id, confidence, brain_score,
        stage, reason, analysis, market_context, executed, trade_id, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      decision.id || `decision_${Date.now()}_${Math.random().toString(36).substring(7)}`,
      decision.symbol,
      decision.action,
      decision.strategy_id || null,
      decision.confidence || 0,
      decision.brain_score || null,
      decision.stage || 'proposed',
      decision.reason || null,
      JSON.stringify(decision.analysis || {}),
      JSON.stringify(decision.market_context || {}),
      decision.executed ? 1 : 0,
      decision.trade_id || null,
      JSON.stringify(decision.metadata || {})
    ];
    
    await this.run(sql, params);
    return params[0]; // Return decision ID
  }
  
  /**
   * Get recent trades
   */
  async getRecentTrades(limit = 100, symbol = null) {
    let sql = 'SELECT * FROM trades WHERE 1=1';
    const params = [];
    
    if (symbol) {
      sql += ' AND symbol = ?';
      params.push(symbol);
    }
    
    sql += ' ORDER BY created_at DESC LIMIT ?';
    params.push(limit);
    
    const rows = await this.all(sql, params);
    
    // Parse JSON fields
    return rows.map(row => ({
      ...row,
      metadata: row.metadata ? JSON.parse(row.metadata) : {}
    }));
  }
  
  /**
   * Get recent decisions
   */
  async getRecentDecisions(limit = 100, symbol = null) {
    let sql = 'SELECT * FROM decisions WHERE 1=1';
    const params = [];
    
    if (symbol) {
      sql += ' AND symbol = ?';
      params.push(symbol);
    }
    
    sql += ' ORDER BY created_at DESC LIMIT ?';
    params.push(limit);
    
    const rows = await this.all(sql, params);
    
    // Parse JSON fields
    return rows.map(row => ({
      ...row,
      analysis: row.analysis ? JSON.parse(row.analysis) : {},
      market_context: row.market_context ? JSON.parse(row.market_context) : {},
      metadata: row.metadata ? JSON.parse(row.metadata) : {}
    }));
  }
  
  /**
   * Update strategy performance
   */
  async updateStrategyPerformance(strategyId, performance) {
    const sql = `
      UPDATE strategies 
      SET performance = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `;
    
    await this.run(sql, [JSON.stringify(performance), strategyId]);
  }
  
  /**
   * Save system state
   */
  async saveSystemState(key, value) {
    const sql = `
      INSERT OR REPLACE INTO system_state (key, value, updated_at)
      VALUES (?, ?, CURRENT_TIMESTAMP)
    `;
    
    await this.run(sql, [key, JSON.stringify(value)]);
  }
  
  /**
   * Get system state
   */
  async getSystemState(key) {
    const row = await this.get('SELECT value FROM system_state WHERE key = ?', [key]);
    return row ? JSON.parse(row.value) : null;
  }
  
  /**
   * Create alert
   */
  async createAlert(alert) {
    const sql = `
      INSERT INTO alerts (severity, source, message, context)
      VALUES (?, ?, ?, ?)
    `;
    
    await this.run(sql, [
      alert.severity,
      alert.source,
      alert.message,
      JSON.stringify(alert.context || {})
    ]);
  }
  
  /**
   * Get performance stats for a strategy
   */
  async getStrategyStats(strategyId, days = 30) {
    const sql = `
      SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(pnl) as total_pnl,
        AVG(pnl_percent) as avg_return,
        MAX(pnl) as best_trade,
        MIN(pnl) as worst_trade
      FROM trades
      WHERE strategy_id = ?
        AND created_at >= datetime('now', '-' || ? || ' days')
    `;
    
    return await this.get(sql, [strategyId, days]);
  }
  
  /**
   * Backup database
   */
  async backup() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = path.join(
      path.dirname(this.config.dbPath),
      `backup_${timestamp}.db`
    );
    
    return new Promise((resolve, reject) => {
      const backup = new sqlite3.Database(backupPath);
      this.db.backup(backup, (err) => {
        backup.close();
        if (err) reject(err);
        else {
          console.log(`[Database] Backup created: ${backupPath}`);
          resolve(backupPath);
        }
      });
    });
  }
  
  /**
   * Close database connection
   */
  async close() {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        resolve();
        return;
      }
      
      this.db.close((err) => {
        if (err) reject(err);
        else {
          console.log('[Database] Connection closed');
          resolve();
        }
      });
    });
  }
}

// Singleton instance
let instance = null;

module.exports = {
  getInstance: (config) => {
    if (!instance) {
      instance = new DatabaseService(config);
    }
    return instance;
  },
  DatabaseService
};
