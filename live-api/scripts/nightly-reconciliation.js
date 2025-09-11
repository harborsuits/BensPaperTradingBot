/**
 * Nightly Reconciliation Script
 * Compares ledger balances with broker positions to detect drift
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');

class NightlyReconciliation {
  constructor(dbPath) {
    this.dbPath = dbPath;
    this.db = null;
    this.thresholds = {
      cashDriftBps: 5,      // 5bps max cash difference
      positionDriftPct: 0.1, // 0.1% max position difference
      alertThresholdBps: 10  // Alert at 10bps
    };
  }

  async initialize() {
    this.db = new sqlite3.Database(this.dbPath);
    console.log('🔍 Starting nightly reconciliation...');
  }

  /**
   * Run full reconciliation
   */
  async runReconciliation() {
    await this.initialize();

    const results = {
      timestamp: new Date().toISOString(),
      cashReconciliation: await this.reconcileCash(),
      positionReconciliation: await this.reconcilePositions(),
      summary: {},
      alerts: []
    };

    // Generate summary
    results.summary = this.generateSummary(results);

    // Check for alerts
    results.alerts = this.checkAlerts(results);

    // Log results
    console.log('📊 RECONCILIATION RESULTS:');
    console.log(JSON.stringify(results, null, 2));

    // Store reconciliation record
    await this.storeReconciliationRecord(results);

    // Send alerts if needed
    if (results.alerts.length > 0) {
      await this.sendAlerts(results.alerts);
    }

    this.db.close();
    return results;
  }

  /**
   * Reconcile cash balances
   */
  async reconcileCash() {
    console.log('💰 Reconciling cash balances...');

    // Get latest ledger cash balance
    const ledgerCash = await this.query(`
      SELECT cash_after as balance, ts_change
      FROM ledger_snapshot
      ORDER BY ts_change DESC
      LIMIT 1
    `);

    // Simulate broker cash balance (in real implementation, call broker API)
    const brokerCash = await this.simulateBrokerCashBalance();

    const difference = Math.abs(ledgerCash.balance - brokerCash.balance);
    const differenceBps = (difference / ledgerCash.balance) * 10000;

    return {
      ledger_balance: ledgerCash.balance,
      broker_balance: brokerCash.balance,
      difference: difference,
      difference_bps: Number(differenceBps.toFixed(2)),
      within_threshold: differenceBps <= this.thresholds.cashDriftBps,
      ledger_timestamp: ledgerCash.ts_change,
      broker_timestamp: brokerCash.timestamp
    };
  }

  /**
   * Reconcile positions
   */
  async reconcilePositions() {
    console.log('📈 Reconciling positions...');

    // Get current positions from fills
    const ledgerPositions = await this.queryAll(`
      SELECT
        symbol,
        SUM(CASE WHEN side = 'BUY_TO_OPEN' THEN qty WHEN side = 'SELL_TO_CLOSE' THEN -qty ELSE 0 END) as quantity,
        AVG(price) as avg_cost
      FROM fills_snapshot
      GROUP BY symbol
      HAVING quantity != 0
    `);

    // Simulate broker positions (in real implementation, call broker API)
    const brokerPositions = await this.simulateBrokerPositions();

    const reconciliation = [];

    for (const ledgerPos of ledgerPositions) {
      const brokerPos = brokerPositions.find(p => p.symbol === ledgerPos.symbol) || { quantity: 0, avg_cost: 0 };

      const quantityDiff = Math.abs(ledgerPos.quantity - brokerPos.quantity);
      const quantityDiffPct = ledgerPos.quantity !== 0 ? (quantityDiff / Math.abs(ledgerPos.quantity)) * 100 : 0;

      const costDiff = Math.abs(ledgerPos.avg_cost - brokerPos.avg_cost);
      const costDiffPct = ledgerPos.avg_cost !== 0 ? (costDiff / ledgerPos.avg_cost) * 100 : 0;

      reconciliation.push({
        symbol: ledgerPos.symbol,
        ledger_quantity: ledgerPos.quantity,
        broker_quantity: brokerPos.quantity,
        quantity_difference: quantityDiff,
        quantity_difference_pct: Number(quantityDiffPct.toFixed(4)),
        ledger_avg_cost: ledgerPos.avg_cost,
        broker_avg_cost: brokerPos.avg_cost,
        cost_difference: costDiff,
        cost_difference_pct: Number(costDiffPct.toFixed(4)),
        within_threshold: quantityDiffPct <= this.thresholds.positionDriftPct
      });
    }

    return reconciliation;
  }

  /**
   * Generate summary
   */
  generateSummary(results) {
    const cashDriftBps = results.cashReconciliation.difference_bps;
    const positionDrifts = results.positionReconciliation.map(p => p.quantity_difference_pct);
    const maxPositionDrift = Math.max(...positionDrifts, 0);

    return {
      overall_status: (cashDriftBps <= this.thresholds.cashDriftBps &&
                      maxPositionDrift <= this.thresholds.positionDriftPct) ? 'healthy' : 'degraded',
      cash_drift_bps: cashDriftBps,
      max_position_drift_pct: maxPositionDrift,
      positions_reconciled: results.positionReconciliation.length,
      positions_with_drift: results.positionReconciliation.filter(p => !p.within_threshold).length
    };
  }

  /**
   * Check for alerts
   */
  checkAlerts(results) {
    const alerts = [];

    if (results.cashReconciliation.difference_bps > this.thresholds.alertThresholdBps) {
      alerts.push({
        type: 'cash_drift',
        severity: 'high',
        message: `Cash drift ${results.cashReconciliation.difference_bps}bps exceeds threshold ${this.thresholds.alertThresholdBps}bps`,
        details: results.cashReconciliation
      });
    }

    const severePositionDrifts = results.positionReconciliation.filter(
      p => p.quantity_difference_pct > this.thresholds.positionDriftPct * 2
    );

    if (severePositionDrifts.length > 0) {
      alerts.push({
        type: 'position_drift',
        severity: 'high',
        message: `${severePositionDrifts.length} positions have severe drift`,
        details: severePositionDrifts
      });
    }

    return alerts;
  }

  /**
   * Store reconciliation record
   */
  async storeReconciliationRecord(results) {
    const sql = `
      INSERT INTO reconciliation_log
      (timestamp, cash_drift_bps, max_position_drift_pct, status, alerts_count)
      VALUES (?, ?, ?, ?, ?)
    `;

    await this.run(sql, [
      results.timestamp,
      results.summary.cash_drift_bps,
      results.summary.max_position_drift_pct,
      results.summary.overall_status,
      results.alerts.length
    ]);

    console.log('✅ Reconciliation record stored');
  }

  /**
   * Send alerts (placeholder - implement based on your alerting system)
   */
  async sendAlerts(alerts) {
    console.log('🚨 SENDING ALERTS:');
    alerts.forEach(alert => {
      console.log(`  ${alert.severity.toUpperCase()}: ${alert.message}`);

      // Here you would integrate with your alerting system:
      // - Email notifications
      // - Slack webhooks
      // - PagerDuty
      // - Internal dashboards
    });
  }

  /**
   * Simulate broker cash balance (replace with real broker API call)
   */
  async simulateBrokerCashBalance() {
    // In production, this would call your broker's API
    const ledgerCash = await this.query(`
      SELECT cash_after as balance FROM ledger_snapshot
      ORDER BY ts_change DESC LIMIT 1
    `);

    // Simulate small drift for testing
    const drift = (Math.random() - 0.5) * 0.001; // ±0.1% drift
    const brokerBalance = ledgerCash.balance * (1 + drift);

    return {
      balance: brokerBalance,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Simulate broker positions (replace with real broker API call)
   */
  async simulateBrokerPositions() {
    // In production, this would call your broker's API for current positions
    const ledgerPositions = await this.queryAll(`
      SELECT
        symbol,
        SUM(CASE WHEN side = 'BUY_TO_OPEN' THEN qty WHEN side = 'SELL_TO_CLOSE' THEN -qty ELSE 0 END) as quantity,
        AVG(price) as avg_cost
      FROM fills_snapshot
      GROUP BY symbol
      HAVING quantity != 0
    `);

    // Simulate small position drift for testing
    return ledgerPositions.map(pos => ({
      symbol: pos.symbol,
      quantity: pos.quantity * (1 + (Math.random() - 0.5) * 0.001), // ±0.1% drift
      avg_cost: pos.avg_cost
    }));
  }

  /**
   * Helper methods
   */
  query(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  queryAll(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  run(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve({ id: this.lastID, changes: this.changes });
      });
    });
  }
}

// Create reconciliation table if it doesn't exist
async function ensureReconciliationTable(dbPath) {
  const db = new sqlite3.Database(dbPath);
  const sql = `
    CREATE TABLE IF NOT EXISTS reconciliation_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      cash_drift_bps REAL,
      max_position_drift_pct REAL,
      status TEXT,
      alerts_count INTEGER,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `;

  return new Promise((resolve, reject) => {
    db.run(sql, (err) => {
      db.close();
      if (err) reject(err);
      else resolve();
    });
  });
}

// Run reconciliation if called directly
if (require.main === module) {
  const dbPath = path.join(__dirname, '../data/evotester.db');

  ensureReconciliationTable(dbPath)
    .then(() => {
      const reconciler = new NightlyReconciliation(dbPath);
      return reconciler.runReconciliation();
    })
    .then((results) => {
      console.log('\n🎉 Nightly reconciliation completed successfully!');
      console.log(`Status: ${results.summary.overall_status}`);
      console.log(`Alerts: ${results.alerts.length}`);

      // Exit with error code if reconciliation failed
      if (results.summary.overall_status !== 'healthy') {
        process.exit(1);
      }
    })
    .catch((error) => {
      console.error('❌ Nightly reconciliation failed:', error);
      process.exit(1);
    });
}

module.exports = { NightlyReconciliation };
