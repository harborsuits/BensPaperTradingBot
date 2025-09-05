const express = require('express');
const router = express.Router();

/**
 * @route GET /api/live/status
 * @desc Get live status of WebSocket connections and configuration
 * @access Public
 */
router.get('/status', (req, res) => {
  const { wss, wssDecisions, wssPrices } = req.app.locals;
  
  res.json({
    prices_ws_clients: wssPrices?.clients?.size || 0,
    decisions_ws_clients: wssDecisions?.clients?.size || 0,
    quotes_refresh_ms: Number(process.env.QUOTES_REFRESH_MS || 5000),
    autorefresh: process.env.AUTOREFRESH_ENABLED === '1',
    live_quotes: process.env.QUOTES_PROVIDER !== 'synthetic' && !!process.env.TRADIER_TOKEN
  });
});

module.exports = router;
