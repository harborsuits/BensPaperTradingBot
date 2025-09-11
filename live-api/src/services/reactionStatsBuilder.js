/**
 * Reaction Stats Builder - Offline computation of empirical market reactions
 * Builds statistical profiles for event types with validation gates
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');

class ReactionStatsBuilder {
  constructor(dbPath = null) {
    this.dbPath = dbPath || path.join(__dirname, '../../data/evotester.db');
    this.stats = new Map(); // eventType -> sector -> stats
  }

  /**
   * Build reaction statistics from historical data
   * @param {number} lookbackDays - Days of historical data to analyze
   * @param {number} minSamples - Minimum samples required for validation
   */
  async buildReactionStats(lookbackDays = 365, minSamples = 100) {
    console.log(`🔬 Building reaction stats from ${lookbackDays} days of data...`);

    const db = new sqlite3.Database(this.dbPath);
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - lookbackDays);

    try {
      // Get news events with market reactions
      const events = await this._getHistoricalEvents(db, cutoffDate);

      console.log(`📊 Processing ${events.length} historical events...`);

      // Group by event type and sector
      const eventGroups = this._groupEventsByTypeAndSector(events);

      // Calculate reaction statistics for each group
      for (const [eventType, sectorGroups] of eventGroups) {
        for (const [sector, events] of sectorGroups) {
          if (events.length >= minSamples) {
            const stats = await this._calculateReactionStats(db, events, eventType, sector);
            if (stats && this._passesValidationGates(stats)) {
              this._storeStats(eventType, sector, stats);
            }
          }
        }
      }

      console.log(`✅ Built reaction stats for ${this.stats.size} event-sector combinations`);

      return this.exportStats();

    } finally {
      db.close();
    }
  }

  /**
   * Get historical events with market data
   */
  async _getHistoricalEvents(db, cutoffDate) {
    return new Promise((resolve, reject) => {
      const query = `
        SELECT
          n.*,
          q.symbol,
          q.bid,
          q.ask,
          q.timestamp as quote_timestamp,
          (q.bid + q.ask) / 2 as mid_price
        FROM news_snapshot n
        LEFT JOIN quotes_snapshot q ON q.symbol = n.symbol
          AND q.timestamp >= n.timestamp
          AND q.timestamp <= datetime(n.timestamp, '+1 hour')
        WHERE n.timestamp >= ?
        ORDER BY n.timestamp, q.timestamp
      `;

      db.all(query, [cutoffDate.toISOString()], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Group events by type and sector
   */
  _groupEventsByTypeAndSector(events) {
    const groups = new Map();

    for (const event of events) {
      // Classify event (simplified for now - in practice would use EventClassifier)
      const eventType = this._classifyHistoricalEvent(event);
      const sector = this._getSectorForSymbol(event.symbol);

      if (!groups.has(eventType)) {
        groups.set(eventType, new Map());
      }

      if (!groups.get(eventType).has(sector)) {
        groups.get(eventType).set(sector, []);
      }

      groups.get(eventType).get(sector).push(event);
    }

    return groups;
  }

  /**
   * Classify historical event (simplified version)
   */
  _classifyHistoricalEvent(event) {
    const headline = event.headline.toLowerCase();

    if (headline.includes('earnings') || headline.includes('eps')) {
      if (headline.includes('beat') || headline.includes('surprise') || headline.includes('raise')) {
        return 'earnings.surprise.up';
      }
      if (headline.includes('miss') || headline.includes('cut') || headline.includes('lower')) {
        return 'earnings.surprise.down';
      }
    }

    if (headline.includes('fed') || headline.includes('rate')) {
      if (headline.includes('hike') || headline.includes('hawkish') || headline.includes('higher')) {
        return 'macro.fed.hawkish';
      }
      if (headline.includes('cut') || headline.includes('dovish') || headline.includes('lower')) {
        return 'macro.fed.dovish';
      }
    }

    if (headline.includes('merger') || headline.includes('acquire') || headline.includes('deal')) {
      return 'corporate.mna.deal';
    }

    return 'other';
  }

  /**
   * Get sector for symbol (simplified mapping)
   */
  _getSectorForSymbol(symbol) {
    // Simplified sector mapping - in practice would use a proper mapping service
    const sectorMap = {
      'AAPL': 'technology',
      'MSFT': 'technology',
      'GOOGL': 'technology',
      'AMZN': 'consumer_discretionary',
      'TSLA': 'consumer_discretionary',
      'NVDA': 'technology',
      'META': 'technology',
      'NFLX': 'consumer_discretionary',
      'JPM': 'financials',
      'BAC': 'financials',
      'WFC': 'financials',
      'JNJ': 'healthcare',
      'PFE': 'healthcare',
      'XOM': 'energy',
      'CVX': 'energy'
    };

    return sectorMap[symbol] || 'other';
  }

  /**
   * Calculate reaction statistics for a group of events
   */
  async _calculateReactionStats(db, events, eventType, sector) {
    const reactions = [];

    for (const event of events) {
      const reaction = await this._calculateSingleReaction(db, event);
      if (reaction) {
        reactions.push(reaction);
      }
    }

    if (reactions.length < 30) return null; // Need minimum sample size

    // Calculate statistical measures
    const returns5m = reactions.map(r => r.return5m).filter(r => r !== null);
    const returns30m = reactions.map(r => r.return30m).filter(r => r !== null);
    const returns1d = reactions.map(r => r.return1d).filter(r => r !== null);

    if (returns5m.length < 10) return null;

    const stats = {
      eventType,
      sector,
      sampleSize: reactions.length,
      sampleSize5m: returns5m.length,
      sampleSize30m: returns30m.length,
      sampleSize1d: returns1d.length,

      // 5-minute reactions
      avgReturn5m: this._mean(returns5m),
      stdReturn5m: this._std(returns5m),
      medianReturn5m: this._median(returns5m),
      skewReturn5m: this._skewness(returns5m),

      // 30-minute reactions
      avgReturn30m: returns30m.length > 0 ? this._mean(returns30m) : null,
      stdReturn30m: returns30m.length > 0 ? this._std(returns30m) : null,

      // 1-day reactions
      avgReturn1d: returns1d.length > 0 ? this._mean(returns1d) : null,
      stdReturn1d: returns1d.length > 0 ? this._std(returns1d) : null,

      // Hit rate (percentage of positive reactions)
      hitRate5m: returns5m.filter(r => r > 0).length / returns5m.length,
      hitRate30m: returns30m.filter(r => r > 0).length / returns30m.length,
      hitRate1d: returns1d.filter(r => r > 0).length / returns1d.length,

      // Effect size (Cohen's d relative to market)
      effectSize5m: this._calculateEffectSize(returns5m),
      effectSize30m: returns30m.length > 0 ? this._calculateEffectSize(returns30m) : null,
      effectSize1d: returns1d.length > 0 ? this._calculateEffectSize(returns1d) : null,

      // Last updated
      lastUpdated: new Date().toISOString(),

      // Validation flags
      passesValidation: false, // Will be set by validation
      orthogonalityScore: null, // Correlation with other factors
      minSampleThreshold: returns5m.length >= 100,
      last12mThreshold: true, // Would check date range in practice
      effectThreshold: Math.abs(this._calculateEffectSize(returns5m)) >= 0.2
    };

    return stats;
  }

  /**
   * Calculate market reaction for a single event
   */
  async _calculateSingleReaction(db, event) {
    return new Promise((resolve) => {
      const eventTime = new Date(event.timestamp);

      // Get pre-event price (5 minutes before)
      const preTime = new Date(eventTime.getTime() - 5 * 60 * 1000);

      // Get post-event prices at different horizons
      const post5m = new Date(eventTime.getTime() + 5 * 60 * 1000);
      const post30m = new Date(eventTime.getTime() + 30 * 60 * 1000);
      const post1d = new Date(eventTime.getTime() + 24 * 60 * 60 * 1000);

      // Query for price data
      const queries = [
        this._getPriceAtTime(db, event.symbol, preTime),
        this._getPriceAtTime(db, event.symbol, post5m),
        this._getPriceAtTime(db, event.symbol, post30m),
        this._getPriceAtTime(db, event.symbol, post1d)
      ];

      Promise.all(queries).then(([prePrice, price5m, price30m, price1d]) => {
        if (!prePrice) {
          resolve(null);
          return;
        }

        const reaction = {
          eventId: event.id,
          symbol: event.symbol,
          prePrice,
          price5m,
          price30m,
          price1d,
          return5m: price5m ? (price5m - prePrice) / prePrice : null,
          return30m: price30m ? (price30m - prePrice) / prePrice : null,
          return1d: price1d ? (price1d - prePrice) / prePrice : null
        };

        resolve(reaction);
      }).catch(() => resolve(null));
    });
  }

  /**
   * Get price at specific time (finds closest quote)
   */
  _getPriceAtTime(db, symbol, targetTime) {
    return new Promise((resolve) => {
      const query = `
        SELECT (bid + ask) / 2 as mid_price
        FROM quotes_snapshot
        WHERE symbol = ?
          AND timestamp >= datetime(?, '-5 minutes')
          AND timestamp <= datetime(?, '+5 minutes')
        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?)) ASC
        LIMIT 1
      `;

      db.get(query, [symbol, targetTime.toISOString(), targetTime.toISOString(), targetTime.toISOString()], (err, row) => {
        if (err || !row) {
          resolve(null);
        } else {
          resolve(row.mid_price);
        }
      });
    });
  }

  /**
   * Statistical helper functions
   */
  _mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  _std(arr) {
    const mean = this._mean(arr);
    const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
  }

  _median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  _skewness(arr) {
    const mean = this._mean(arr);
    const std = this._std(arr);
    const n = arr.length;

    if (std === 0) return 0;

    const skewness = arr.reduce((a, b) => a + Math.pow((b - mean) / std, 3), 0) / n;
    return skewness;
  }

  _calculateEffectSize(returns) {
    const mean = this._mean(returns);
    const std = this._std(returns);

    // Simplified effect size (Cohen's d relative to zero)
    return mean / (std + 0.0001); // Avoid division by zero
  }

  /**
   * Check if stats pass validation gates
   */
  _passesValidationGates(stats) {
    return (
      stats.minSampleThreshold &&
      stats.last12mThreshold &&
      stats.effectThreshold &&
      Math.abs(stats.effectSize5m) >= 0.2 && // Minimum effect size
      stats.sampleSize5m >= 100 // Minimum sample size
    );
  }

  /**
   * Store validated statistics
   */
  _storeStats(eventType, sector, stats) {
    if (!this.stats.has(eventType)) {
      this.stats.set(eventType, new Map());
    }

    stats.passesValidation = true;
    this.stats.get(eventType).set(sector, stats);
  }

  /**
   * Export statistics as JSON
   */
  exportStats() {
    const result = {};

    for (const [eventType, sectorStats] of this.stats) {
      result[eventType] = {};

      for (const [sector, stats] of sectorStats) {
        result[eventType][sector] = stats;
      }
    }

    return result;
  }

  /**
   * Load pre-computed statistics from file
   */
  async loadStatsFromFile(filePath) {
    try {
      const fs = require('fs').promises;
      const data = await fs.readFile(filePath, 'utf8');
      const stats = JSON.parse(data);

      // Convert back to Map structure
      this.stats = new Map();
      for (const [eventType, sectorStats] of Object.entries(stats)) {
        const sectorMap = new Map();
        for (const [sector, statData] of Object.entries(sectorStats)) {
          sectorMap.set(sector, statData);
        }
        this.stats.set(eventType, sectorMap);
      }

      console.log(`📊 Loaded reaction stats for ${this.stats.size} event types`);
      return true;
    } catch (error) {
      console.error('Failed to load reaction stats:', error);
      return false;
    }
  }

  /**
   * Get reaction stats for specific event type and sector
   */
  getReactionStats(eventType, sector) {
    const eventStats = this.stats.get(eventType);
    if (!eventStats) return null;

    return eventStats.get(sector) || eventStats.get('all') || null;
  }

  /**
   * Get all validated event types
   */
  getValidatedEventTypes() {
    const validated = [];

    for (const [eventType, sectorStats] of this.stats) {
      for (const [sector, stats] of sectorStats) {
        if (stats.passesValidation) {
          validated.push({
            eventType,
            sector,
            effectSize: stats.effectSize5m,
            sampleSize: stats.sampleSize5m,
            hitRate: stats.hitRate5m
          });
        }
      }
    }

    return validated;
  }
}

module.exports = { ReactionStatsBuilder };
