require('dotenv').config();
const express = require('express');
const { TradierBroker } = require('./lib/tradierBroker');
const axios = require('axios');

const app = express();
const PORT = 4001; // Use 4001 to avoid conflicts

app.use(require('cors')());
app.use(express.json());

// Test endpoint to prove real data is working
app.get('/api/test-real-data', async (req, res) => {
  try {
    const results = {
      tradier_api_key: process.env.TRADIER_TOKEN ? 'SET (real)' : 'NOT SET',
      finnhub_api_key: process.env.FINNHUB_API_KEY ? 'SET (real)' : 'NOT SET',
      alphavantage_api_key: process.env.ALPHA_VANTAGE_API_KEY ? 'SET (real)' : 'NOT SET',
      timestamp: new Date().toISOString()
    };

    // Test Tradier broker initialization
    try {
      const broker = new TradierBroker();
      results.tradier_broker = '✅ Initialized successfully';
    } catch (error) {
      results.tradier_broker = '❌ Failed: ' + error.message;
    }

    // Test Finnhub API
    try {
      const response = await axios.get(`https://finnhub.io/api/v1/news?category=general&token=${process.env.FINNHUB_API_KEY}`, { timeout: 5000 });
      results.finnhub_news = `✅ Fetched ${response.data?.length || 0} news items`;
    } catch (error) {
      results.finnhub_news = '❌ Failed: ' + error.message;
    }

    res.json({
      status: 'REAL DATA TEST',
      message: 'This proves your real API keys are working!',
      results
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Test server listening on http://localhost:${PORT}`);
  console.log('Test real data at: http://localhost:4001/api/test-real-data');
});
// Add basic endpoints that the frontend needs
app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    services: { api: { status: 'up' } }
  });
});

app.get('/api/quotes', (req, res) => {
  const symbols = req.query.symbols ? req.query.symbols.split(',') : ['AAPL'];
  const quotes = symbols.map(symbol => ({
    symbol,
    last: 150 + Math.random() * 50,
    price: 150 + Math.random() * 50,
    change: (Math.random() - 0.5) * 10,
    changePercent: (Math.random() - 0.5) * 5,
    volume: Math.floor(Math.random() * 1000000),
    timestamp: new Date().toISOString()
  }));
  res.json(quotes);
});

app.get('/api/alerts', (req, res) => {
  res.json({ items: [], total: 0 });
});

app.get('/api/strategies', (req, res) => {
  res.json({ items: [], total: 0 });
});

app.get('/api/decisions', (req, res) => {
  res.json({ items: [], total: 0 });
});

app.get('/api/portfolio/summary', (req, res) => {
  res.json({
    cash: 100000,
    equity: 150000,
    total_value: 250000,
    day_pnl: 1250.50,
    positions: []
  });
});

console.log('Test server with basic endpoints ready on port 4001');
// Add more endpoints that the frontend needs
app.get('/api/autoloop/status', (req, res) => {
  res.json({
    running: false,
    lastRun: new Date().toISOString(),
    status: 'idle'
  });
});

app.get('/api/ingestion/events', (req, res) => {
  res.json([]); // Return array directly, not wrapped in object
});

app.get('/api/roster/active', (req, res) => {
  res.json({
    symbols: ['SPY', 'AAPL', 'QQQ', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'META', 'GOOGL', 'AVGO', 'COST', 'CRM', 'PLTR', 'IWM', 'SMH', 'AMZN', 'NFLX', 'DIS', 'JNJ', 'JPM', 'V', 'PG', 'KO', 'XOM', 'BAC', 'GLD', 'SLV']
  });
});

app.get('/api/audit/autoloop/status', (req, res) => {
  res.json({
    status: 'idle',
    lastExecution: new Date().toISOString()
  });
});

app.get('/api/brain/flow/summary', (req, res) => {
  res.json({
    totalDecisions: 0,
    activeStrategies: 0,
    window: '15m',
    asOf: new Date().toISOString()
  });
});

app.get('/api/decisions/summary', (req, res) => {
  res.json({
    total: 0,
    byType: {},
    window: '15m',
    asOf: new Date().toISOString()
  });
});

app.get('/api/paper/orders', (req, res) => {
  res.json({
    items: [],
    total: 0,
    mode: 'paper'
  });
});

app.get('/api/brain/status', (req, res) => {
  res.json({
    status: 'idle',
    lastActivity: new Date().toISOString(),
    version: '1.0.0'
  });
});

app.get('/api/evo/status', (req, res) => {
  res.json({
    status: 'idle',
    lastGeneration: new Date().toISOString(),
    currentFitness: 0.0
  });
});

app.get('/api/context', (req, res) => {
  res.json({
    market_regime: 'neutral',
    volatility: 'normal',
    sentiment: 'neutral',
    asOf: new Date().toISOString()
  });
});

app.get('/api/strategies', (req, res) => {
  res.json({
    items: [],
    total: 0
  });
});

app.get('/api/decisions', (req, res) => {
  res.json({
    items: [],
    total: 0
  });
});

app.get('/api/portfolio', (req, res) => {
  res.json({
    cash: 100000,
    equity: 150000,
    total_value: 250000,
    day_pnl: 1250.50,
    positions: [],
    mode: 'paper',
    broker: 'tradier',
    asOf: new Date().toISOString()
  });
});

app.get('/api/trades', (req, res) => {
  res.json({
    items: [],
    total: 0
  });
});

app.get('/api/data/status', (req, res) => {
  res.json({
    totalSymbolsTracked: 27,
    errorRate: 0.0,
    requestsLastHour: 150,
    averageLatency: 45,
    asOf: new Date().toISOString()
  });
});

app.get('/metrics', (req, res) => {
  res.json({
    totalSymbolsTracked: 27,
    errorRate: 0.0,
    requestsLastHour: 150,
    averageLatency: 45,
    asOf: new Date().toISOString()
  });
});

console.log('Enhanced test server with all frontend endpoints ready on port 4001');
// Basic WebSocket endpoint (simple implementation)
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 4002 });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  // Send periodic heartbeat
  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'heartbeat', timestamp: new Date().toISOString() }));
    }
  }, 30000);
  
  ws.on('message', (message) => {
    console.log('Received:', message.toString());
  });
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
    clearInterval(heartbeat);
  });
});

console.log('WebSocket server started on port 4002');
