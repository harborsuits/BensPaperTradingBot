const express = require('express');
const router = express.Router();
const { getQuotesCache, getQuotesStatus } = require('../src/services/quotesService');

/**
 * @route GET /api/quotes
 * @desc Get quotes for all symbols or filtered by provided symbols
 * @access Public
 */
router.get('/', (req, res) => {
  const { symbols } = req.query;
  const { quotes, asOf } = getQuotesCache();
  
  if (symbols) {
    const symbolList = symbols.split(',').map(s => s.trim().toUpperCase());
    const filtered = {};
    symbolList.forEach(sym => {
      if (quotes[sym]) filtered[sym] = quotes[sym];
    });
    return res.json({ quotes: filtered, asOf });
  }
  
  res.json({ quotes, asOf });
});

/**
 * @route GET /api/quotes/status
 * @desc Get status of quotes service
 * @access Public
 */
router.get('/status', (req, res) => {
  res.json(getQuotesStatus());
});

module.exports = router;
