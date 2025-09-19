const { tradingState } = require('../lib/tradingState');

function requireTradingOn(req, res, next) {
  if (tradingState.isPaused()) {
    return res.status(423).json({ error: 'trading_paused' });
  }
  next();
}

module.exports = { requireTradingOn };


