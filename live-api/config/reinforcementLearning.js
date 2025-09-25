
/**
 * Reinforcement Learning Configuration
 * Q-Learning for strategy selection
 */

module.exports = {
  enabled: true,
  
  // Q-Learning parameters
  alpha: 0.1,        // Learning rate
  gamma: 0.95,       // Discount factor
  epsilon: 0.1,      // Exploration rate
  
  // State space
  states: {
    marketRegime: ['trending', 'choppy', 'volatile'],
    timeOfDay: ['morning', 'midday', 'afternoon'],
    volatility: ['low', 'medium', 'high'],
    newsImpact: ['none', 'low', 'high']
  },
  
  // Action space (strategies)
  actions: [
    'rsi_reversion',
    'ma_crossover',
    'vwap_reversion',
    'momentum_advanced',
    'news_momentum',
    'breakout',
    'options_covered_calls',
    'options_cash_puts'
  ],
  
  // Reward function
  rewardFunction: (trade) => {
    const pnl = trade.pnl || 0;
    const riskAdjusted = pnl / (trade.riskAmount || 1);
    const timeDecay = Math.exp(-trade.holdingTime / (24 * 60 * 60 * 1000)); // Decay over 24h
    
    return riskAdjusted * timeDecay;
  }
};
