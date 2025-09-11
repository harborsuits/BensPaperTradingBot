/**
 * Crypto MarketRecorder - Event-sourced logging for crypto market data
 * Extends base MarketRecorder with crypto-specific data structures
 */

const { MarketRecorder } = require('./marketRecorder');

class CryptoMarketRecorder extends MarketRecorder {
  constructor(dbPath) {
    super(dbPath);
  }

  /**
   * Initialize crypto-specific tables
   */
  async initialize() {
    await super.initialize();

    // Create crypto-specific tables
    const cryptoTables = [
      `CREATE TABLE IF NOT EXISTS crypto_quotes_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        price REAL NOT NULL,
        bid REAL,
        ask REAL,
        volume24h REAL,
        spreadBps REAL,
        exchange TEXT NOT NULL,
        ts_feed TEXT NOT NULL,
        ts_recv TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      `CREATE TABLE IF NOT EXISTS crypto_orderbook_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        bids TEXT NOT NULL,  -- JSON array of [price, size]
        asks TEXT NOT NULL,  -- JSON array of [price, size]
        depth INTEGER DEFAULT 20,
        ts_feed TEXT NOT NULL,
        ts_recv TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      `CREATE TABLE IF NOT EXISTS crypto_balances_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        free REAL NOT NULL,
        locked REAL NOT NULL,
        total REAL NOT NULL,
        exchange TEXT NOT NULL,
        ts_snapshot TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      )`,

      // Indexes for crypto tables
      `CREATE INDEX IF NOT EXISTS idx_crypto_quotes_symbol_ts ON crypto_quotes_snapshot(symbol, ts_recv)`,
      `CREATE INDEX IF NOT EXISTS idx_crypto_quotes_exchange ON crypto_quotes_snapshot(exchange)`,
      `CREATE INDEX IF NOT EXISTS idx_crypto_orderbook_symbol ON crypto_orderbook_snapshot(symbol, ts_recv)`,
      `CREATE INDEX IF NOT EXISTS idx_crypto_balances_asset ON crypto_balances_snapshot(asset, ts_snapshot)`
    ];

    for (const sql of cryptoTables) {
      await this.run(sql);
    }

    console.log('CryptoMarketRecorder initialized with crypto-specific tables');
  }

  /**
   * Record crypto quote snapshot
   */
  async recordCryptoQuote(symbol, quote, ts_feed, ts_recv, exchange = 'binance') {
    await this.initialize();

    const sql = `
      INSERT INTO crypto_quotes_snapshot
      (symbol, price, bid, ask, volume24h, spreadBps, exchange, ts_feed, ts_recv)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    await this.run(sql, [
      symbol,
      quote.price,
      quote.bid,
      quote.ask,
      quote.volume24h,
      quote.spreadBps,
      exchange,
      ts_feed,
      ts_recv
    ]);
  }

  /**
   * Record crypto order book snapshot
   */
  async recordCryptoOrderBook(symbol, orderBook, ts_feed, ts_recv, exchange = 'binance') {
    await this.initialize();

    const sql = `
      INSERT INTO crypto_orderbook_snapshot
      (symbol, exchange, bids, asks, depth, ts_feed, ts_recv)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;

    await this.run(sql, [
      symbol,
      exchange,
      JSON.stringify(orderBook.bids),
      JSON.stringify(orderBook.asks),
      orderBook.bids.length,
      ts_feed,
      ts_recv
    ]);
  }

  /**
   * Record crypto balances snapshot
   */
  async recordCryptoBalances(balances, ts_snapshot, exchange = 'binance') {
    await this.initialize();

    const sql = `
      INSERT INTO crypto_balances_snapshot
      (asset, free, locked, total, exchange, ts_snapshot)
      VALUES (?, ?, ?, ?, ?, ?)
    `;

    for (const [asset, balance] of Object.entries(balances)) {
      await this.run(sql, [
        asset,
        balance.free,
        balance.locked,
        balance.total,
        exchange,
        ts_snapshot
      ]);
    }
  }

  /**
   * Get latest crypto quote
   */
  async getLatestCryptoQuote(symbol, exchange = 'binance', maxAgeMs = 5000) {
    await this.initialize();

    const sql = `
      SELECT * FROM crypto_quotes_snapshot
      WHERE symbol = ? AND exchange = ?
        AND (strftime('%s', 'now') * 1000 - strftime('%s', ts_recv) * 1000) <= ?
      ORDER BY ts_recv DESC
      LIMIT 1
    `;

    return await this.get(sql, [symbol, exchange, maxAgeMs]);
  }

  /**
   * Get crypto order book at time
   */
  async getCryptoOrderBookAtTime(symbol, timestamp, exchange = 'binance', toleranceMs = 1000) {
    await this.initialize();

    const sql = `
      SELECT * FROM crypto_orderbook_snapshot
      WHERE symbol = ? AND exchange = ?
        AND ABS(strftime('%s', ts_recv) * 1000 - strftime('%s', ?) * 1000) <= ?
      ORDER BY ABS(strftime('%s', ts_recv) * 1000 - strftime('%s', ?) * 1000)
      LIMIT 1
    `;

    const result = await this.get(sql, [symbol, exchange, timestamp, toleranceMs, timestamp]);

    if (result) {
      // Parse JSON bids/asks
      result.bids = JSON.parse(result.bids);
      result.asks = JSON.parse(result.asks);
    }

    return result;
  }

  /**
   * Get crypto balances at time
   */
  async getCryptoBalancesAtTime(asset, timestamp, exchange = 'binance', toleranceMs = 1000) {
    await this.initialize();

    const sql = `
      SELECT * FROM crypto_balances_snapshot
      WHERE asset = ? AND exchange = ?
        AND ABS(strftime('%s', ts_snapshot) * 1000 - strftime('%s', ?) * 1000) <= ?
      ORDER BY ABS(strftime('%s', ts_snapshot) * 1000 - strftime('%s', ?) * 1000)
    `;

    const results = await this.all(sql, [asset, exchange, timestamp, toleranceMs, timestamp]);
    return results;
  }

  /**
   * Get 24h crypto friction statistics
   */
  async get24hCryptoFrictionStats(exchange = 'binance') {
    await this.initialize();

    // For crypto, friction is primarily exchange fees + slippage
    // This is a simplified implementation
    const sql = `
      SELECT
        COUNT(*) as total_quotes,
        AVG(spreadBps) as avg_spread_bps,
        MIN(spreadBps) as min_spread_bps,
        MAX(spreadBps) as max_spread_bps
      FROM crypto_quotes_snapshot
      WHERE exchange = ? AND ts_recv >= datetime('now', '-1 day')
    `;

    return await this.get(sql, [exchange]);
  }
}

module.exports = { CryptoMarketRecorder };
