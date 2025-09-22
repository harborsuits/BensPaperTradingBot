/**
 * Bot Competition Service
 * Manages AI bot competitions with micro-capital allocations
 */

const EventEmitter = require('events');

class BotCompetitionService extends EventEmitter {
  constructor() {
    super();
    this.competitions = new Map();
    this.bots = new Map();
    this.leaderboard = new Map();
    this.reallocationInterval = null;
  }

  /**
   * Start a new competition
   */
  startCompetition(config = {}) {
    const competitionId = `comp-${Date.now()}`;
    const competition = {
      id: competitionId,
      status: 'active',
      startTime: new Date().toISOString(),
      endTime: null,
      config: {
        durationDays: config.durationDays || 7,
        initialCapitalMin: config.initialCapitalMin || 50,  // Everyone gets $50
        initialCapitalMax: config.initialCapitalMax || 50,  // Fixed amount for fairness
        winnerBonus: config.winnerBonus || 0.2, // 20% more capital
        loserPenalty: config.loserPenalty || 0.5, // 50% less capital
        reallocationIntervalHours: config.reallocationIntervalHours || 1,
        totalPoolCapital: config.totalPoolCapital || 2000, // 10% of 20k = $2000 total pool
        ...config
      },
      stats: {
        totalTrades: 0,
        totalReturn: 0,
        activeBots: 0
      }
    };

    this.competitions.set(competitionId, competition);
    this.startReallocationCycle(competitionId);
    
    return competition;
  }

  /**
   * Add a bot to the competition
   */
  addBot(competitionId, strategyData) {
    const competition = this.competitions.get(competitionId);
    if (!competition || competition.status !== 'active') {
      throw new Error('Competition not active');
    }

    const botId = `bot-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    const initialCapital = this.randomBetween(
      competition.config.initialCapitalMin,
      competition.config.initialCapitalMax
    );

    const bot = {
      id: botId,
      competitionId,
      strategy: {
        name: strategyData.name || 'Unknown Strategy',
        type: strategyData.type || 'momentum',
        symbol: strategyData.symbol || 'SPY',
        generation: strategyData.generation || 1,
        ...strategyData
      },
      capital: {
        initial: initialCapital,
        current: initialCapital,
        peak: initialCapital,
        allocations: [{
          timestamp: new Date().toISOString(),
          amount: initialCapital,
          reason: 'initial'
        }]
      },
      performance: {
        totalReturn: 0,
        totalReturnPct: 0,
        trades: 0,
        wins: 0,
        losses: 0,
        lastUpdate: new Date().toISOString()
      },
      status: 'active'
    };

    this.bots.set(botId, bot);
    this.updateLeaderboard(competitionId);
    competition.stats.activeBots++;

    return bot;
  }

  /**
   * Record a trade for a bot
   */
  recordTrade(botId, trade) {
    const bot = this.bots.get(botId);
    if (!bot || bot.status !== 'active') return null;

    const competition = this.competitions.get(bot.competitionId);
    if (!competition) return null;

    // Update bot capital based on trade P&L
    const pnl = trade.pnl || 0;
    bot.capital.current += pnl;
    bot.capital.peak = Math.max(bot.capital.peak, bot.capital.current);

    // Update performance metrics
    bot.performance.trades++;
    if (pnl > 0) {
      bot.performance.wins++;
    } else if (pnl < 0) {
      bot.performance.losses++;
    }

    bot.performance.totalReturn = bot.capital.current - bot.capital.initial;
    bot.performance.totalReturnPct = (bot.performance.totalReturn / bot.capital.initial) * 100;
    bot.performance.lastUpdate = new Date().toISOString();

    // Update competition stats
    competition.stats.totalTrades++;
    
    this.updateLeaderboard(bot.competitionId);
    return bot;
  }

  /**
   * Reallocate capital based on performance
   */
  reallocateCapital(competitionId) {
    const competition = this.competitions.get(competitionId);
    if (!competition || competition.status !== 'active') return;

    const competitionBots = Array.from(this.bots.values())
      .filter(bot => bot.competitionId === competitionId && bot.status === 'active')
      .sort((a, b) => b.performance.totalReturnPct - a.performance.totalReturnPct);

    if (competitionBots.length === 0) return;

    // Determine winners and losers
    const topThird = Math.ceil(competitionBots.length / 3);
    const bottomThird = Math.floor(competitionBots.length / 3);

    competitionBots.forEach((bot, index) => {
      let adjustment = 0;
      let reason = 'hold';

      if (index < topThird) {
        // Winners get bonus capital
        adjustment = bot.capital.current * competition.config.winnerBonus;
        reason = 'winner_bonus';
      } else if (index >= competitionBots.length - bottomThird) {
        // Losers lose capital
        adjustment = -bot.capital.current * competition.config.loserPenalty;
        reason = 'loser_penalty';
      }

      if (adjustment !== 0) {
        bot.capital.current += adjustment;
        bot.capital.allocations.push({
          timestamp: new Date().toISOString(),
          amount: adjustment,
          reason,
          newTotal: bot.capital.current
        });
      }
    });

    this.updateLeaderboard(competitionId);
    return competitionBots;
  }

  /**
   * Update leaderboard
   */
  updateLeaderboard(competitionId) {
    const competitionBots = Array.from(this.bots.values())
      .filter(bot => bot.competitionId === competitionId)
      .sort((a, b) => b.performance.totalReturnPct - a.performance.totalReturnPct)
      .slice(0, 10); // Top 10

    this.leaderboard.set(competitionId, competitionBots);

    // Calculate total return for the competition
    const competition = this.competitions.get(competitionId);
    if (competition) {
      const totalReturn = competitionBots.reduce((sum, bot) => 
        sum + bot.performance.totalReturn, 0
      );
      const totalInitial = competitionBots.reduce((sum, bot) => 
        sum + bot.capital.initial, 0
      );
      competition.stats.totalReturn = totalInitial > 0 ? 
        (totalReturn / totalInitial) * 100 : 0;
    }

    return competitionBots;
  }

  /**
   * Get competition status
   */
  getCompetitionStatus(competitionId) {
    const competition = this.competitions.get(competitionId);
    if (!competition) return null;

    const leaderboard = this.leaderboard.get(competitionId) || [];
    const bots = Array.from(this.bots.values())
      .filter(bot => bot.competitionId === competitionId);

    const now = new Date();
    const start = new Date(competition.startTime);
    const daysElapsed = (now - start) / (1000 * 60 * 60 * 24);
    const daysLeft = Math.max(0, competition.config.durationDays - daysElapsed);

    // Calculate total capital in play
    const totalCapitalInPlay = bots.reduce((sum, bot) => sum + bot.capital.current, 0);

    return {
      ...competition,
      stats: {
        ...competition.stats,
        activeBots: bots.filter(b => b.status === 'active').length,
        daysElapsed: Math.floor(daysElapsed),
        daysLeft: Math.ceil(daysLeft),
        hoursToNextReallocation: this.getHoursToNextReallocation(competitionId),
        totalCapitalInPlay,
        poolUtilization: competition.config.totalPoolCapital ? 
          (totalCapitalInPlay / competition.config.totalPoolCapital * 100).toFixed(1) : 0
      },
      leaderboard: leaderboard.map((bot, index) => ({
        rank: index + 1,
        id: bot.id,
        strategy: bot.strategy.name,
        symbol: bot.strategy.symbol,
        generation: bot.strategy.generation,
        returnPct: bot.performance.totalReturnPct,
        returnDollar: bot.performance.totalReturn,
        currentCapital: bot.capital.current,
        trades: bot.performance.trades,
        winRate: bot.performance.trades > 0 ? 
          (bot.performance.wins / bot.performance.trades) * 100 : 0
      }))
    };
  }

  /**
   * End competition
   */
  endCompetition(competitionId) {
    const competition = this.competitions.get(competitionId);
    if (!competition) return null;

    competition.status = 'completed';
    competition.endTime = new Date().toISOString();

    // Stop reallocation
    if (this.reallocationInterval) {
      clearInterval(this.reallocationInterval);
      this.reallocationInterval = null;
    }

    // Deactivate all bots
    const bots = Array.from(this.bots.values())
      .filter(bot => bot.competitionId === competitionId);
    
    bots.forEach(bot => {
      bot.status = 'completed';
    });

    // Emit competition complete event
    this.emit('competition_complete', {
      competitionId: competition.id,
      competition: competition,
      bots: bots,
      timestamp: new Date().toISOString()
    });

    return this.getCompetitionStatus(competitionId);
  }

  /**
   * Start automatic reallocation cycle
   */
  startReallocationCycle(competitionId) {
    const competition = this.competitions.get(competitionId);
    if (!competition) return;

    const intervalMs = competition.config.reallocationIntervalHours * 60 * 60 * 1000;
    
    this.reallocationInterval = setInterval(() => {
      if (competition.status === 'active') {
        console.log(`[BotCompetition] Reallocating capital for ${competitionId}`);
        this.reallocateCapital(competitionId);
      }
    }, intervalMs);
  }

  /**
   * Get hours until next reallocation
   */
  getHoursToNextReallocation(competitionId) {
    const competition = this.competitions.get(competitionId);
    if (!competition || competition.status !== 'active') return null;

    // Simple calculation - assumes reallocations happen on the hour
    const now = new Date();
    const nextHour = new Date(now);
    nextHour.setHours(nextHour.getHours() + 1, 0, 0, 0);
    
    const hoursUntil = (nextHour - now) / (1000 * 60 * 60);
    return Math.max(0, hoursUntil);
  }

  /**
   * Get active competitions
   */
  getActiveCompetitions() {
    return Array.from(this.competitions.values())
      .filter(comp => comp.status === 'active')
      .map(comp => this.getCompetitionStatus(comp.id));
  }

  /**
   * Simulate a trade for a bot based on a decision
   */
  simulateBotTrade(botId, decision) {
    const bot = this.bots.get(botId);
    if (!bot || bot.status !== 'active') return null;

    // Calculate trade size based on bot's capital and Kelly fraction
    // Use 25% per position for small accounts to allow meaningful trades
    const positionPct = bot.capital.current < 100 ? 0.25 : 0.1;
    const positionSize = Math.floor(bot.capital.current * positionPct / decision.price);
    if (positionSize < 1) return null; // Not enough capital

    // Simulate entry
    const entryPrice = decision.price * (1 + (Math.random() - 0.5) * 0.002); // Add slippage
    
    // Simulate exit after random holding period (1-60 minutes)
    const holdingMinutes = Math.floor(Math.random() * 60) + 1;
    const priceChange = (Math.random() - 0.48) * 0.01 * decision.confidence; // Slight edge based on confidence
    const exitPrice = entryPrice * (1 + priceChange);
    
    // Calculate P&L
    const grossPnL = (exitPrice - entryPrice) * positionSize * (decision.side === 'buy' ? 1 : -1);
    const commission = positionSize * 0.01; // $0.01 per share
    const netPnL = grossPnL - commission;

    const trade = {
      botId,
      symbol: decision.symbol,
      side: decision.side,
      entryPrice,
      exitPrice,
      shares: positionSize,
      grossPnL,
      commission,
      pnl: netPnL,
      holdingMinutes,
      timestamp: new Date().toISOString()
    };

    // Record the trade
    this.recordTrade(botId, trade);
    
    return trade;
  }

  /**
   * Get bot by strategy type and symbol
   */
  getBotByStrategy(competitionId, strategyType, symbol) {
    const bots = Array.from(this.bots.values())
      .filter(bot => 
        bot.competitionId === competitionId && 
        bot.status === 'active' &&
        bot.strategy.type === strategyType &&
        (bot.strategy.symbol === symbol || bot.strategy.symbol === 'ALL')
      );
    
    return bots.length > 0 ? bots[0] : null;
  }

  /**
   * Utility: Random number between min and max
   */
  randomBetween(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }
}

module.exports = BotCompetitionService;
