const express = require('express');
const router = express.Router();

// Basic metrics endpoint
router.get('/live', async (req, res) => {
  const strat = req.query.strategy || "ma_crossover_v1";

  // Mock response for now - replace with actual metrics calculation
  const mockResponse = {
    strategy: String(strat),
    oos: {
      sharpe: 0.8,
      pf: 1.2,
      dd: 0.15,
      trades: 150,
      avg_spread_bps: 2.5,
      realized_slippage_bps: 1.8,
      breach_rate: 0.05
    },
    live_paper_20d: {
      sharpe: 0.6,
      pf: 1.1,
      dd: 0.08,
      trades: 45,
      avg_spread_bps: 3.2,
      realized_slippage_bps: 2.1,
      breach_rate: 0.03
    },
    live_paper_60d: {
      sharpe: 0.7,
      pf: 1.15,
      dd: 0.12,
      trades: 120,
      avg_spread_bps: 2.8,
      realized_slippage_bps: 1.9,
      breach_rate: 0.04
    },
    regime_breakdown: {
      trend: {
        sharpe: 0.9,
        pf: 1.3,
        dd: 0.1,
        trades: 40
      },
      chop: {
        sharpe: 0.4,
        pf: 0.9,
        dd: 0.18,
        trades: 35
      },
      high_vol: {
        sharpe: 0.6,
        pf: 1.1,
        dd: 0.15,
        trades: 45
      }
    }
  };

  res.json(mockResponse);
});

module.exports = router;
