#!/usr/bin/env node

// This script adds missing endpoints to minimal_server.js

const fs = require('fs');
const path = require('path');

const serverFile = path.join(__dirname, 'minimal_server.js');
const content = fs.readFileSync(serverFile, 'utf8');

// Find the spot to insert - right before server.listen
const insertPoint = content.indexOf('// Removed duplicate poolStatus endpoint');

if (insertPoint === -1) {
  console.error('Could not find insertion point');
  process.exit(1);
}

// The endpoints to add
const endpointsToAdd = `
// ========== EVOLUTION & DISCOVERY ENDPOINTS ==========

// Evolution strategy hypotheses
app.get('/api/evo/strategy-hypotheses', (req, res) => {
  res.json({
    hypotheses: [
      {
        id: 'hyp_001',
        name: 'Sentiment-Driven Momentum',
        description: 'Strategies that combine news sentiment with price momentum',
        status: 'active',
        performance: { win_rate: 0.62, sharpe: 1.2 }
      },
      {
        id: 'hyp_002',
        name: 'Volatility Arbitrage',
        description: 'Exploit volatility disparities across correlated assets',
        status: 'testing',
        performance: { win_rate: 0.58, sharpe: 0.9 }
      }
    ],
    asOf: new Date().toISOString()
  });
});

// Pipeline stages for evolution
app.get('/api/pipeline/stages', (req, res) => {
  res.json({
    stages: [
      { id: 'research', name: 'Research', active: 10, completed: 45 },
      { id: 'backtest', name: 'Backtest', active: 8, completed: 37 },
      { id: 'paper', name: 'Paper Trading', active: 20, completed: 25 },
      { id: 'live', name: 'Live Trading', active: 0, completed: 0 }
    ],
    asOf: new Date().toISOString()
  });
});

// Evolution deployment metrics
app.get('/api/evo/deployment-metrics', (req, res) => {
  res.json({
    metrics: {
      total_strategies: 20,
      active_paper: 20,
      active_live: 0,
      total_trades: 100,
      win_rate: 0.58,
      avg_sharpe: 0.95,
      evolution_cycles: 2
    },
    asOf: new Date().toISOString()
  });
});

// Evolution trigger rules
app.get('/api/evo/trigger-rules', (req, res) => {
  res.json({
    rules: [
      {
        id: 'trade_count',
        name: 'Trade Count Trigger',
        condition: 'trades >= 50',
        current: 100,
        target: 50,
        triggered: true
      },
      {
        id: 'performance',
        name: 'Performance Trigger',
        condition: 'sharpe > 1.0',
        current: 0.95,
        target: 1.0,
        triggered: false
      }
    ],
    next_evolution: 'Ready - awaiting trigger',
    asOf: new Date().toISOString()
  });
});

// Fundamentals endpoint
app.get('/api/fundamentals', (req, res) => {
  const symbols = (req.query.symbols || 'SPY,QQQ,AAPL').split(',');
  const fundamentals = {};
  
  symbols.forEach(symbol => {
    fundamentals[symbol] = {
      pe_ratio: 15 + Math.random() * 20,
      market_cap: Math.floor(Math.random() * 1000) + 100,
      dividend_yield: Math.random() * 3,
      eps: Math.random() * 10 + 1,
      revenue_growth: (Math.random() - 0.3) * 0.5
    };
  });
  
  res.json({
    data: fundamentals,
    asOf: new Date().toISOString()
  });
});

// Market discovery endpoint
app.get('/api/discovery/market', (req, res) => {
  res.json({
    discoveries: [
      {
        id: 'disc_001',
        type: 'trend',
        description: 'Tech sector showing strong momentum',
        symbols: ['QQQ', 'NVDA', 'MSFT'],
        confidence: 0.75,
        discovered_at: new Date().toISOString()
      },
      {
        id: 'disc_002',
        type: 'anomaly',
        description: 'Unusual options activity in SPY',
        symbols: ['SPY'],
        confidence: 0.82,
        discovered_at: new Date().toISOString()
      }
    ],
    market_regime: 'neutral',
    asOf: new Date().toISOString()
  });
});

`;

// Insert the new endpoints
const newContent = content.slice(0, insertPoint) + endpointsToAdd + '\n' + content.slice(insertPoint);

// Write back
fs.writeFileSync(serverFile, newContent);

console.log('✅ Added missing evolution & discovery endpoints to minimal_server.js');
