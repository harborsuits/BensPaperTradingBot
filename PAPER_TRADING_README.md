# 🚀 BenBot: AI-Powered Trading Bot with Evolutionary Competition

**Status: PAPER TRADING READY** ✅

BenBot is an advanced algorithmic trading platform that uses **evolutionary AI** to create, test, and deploy trading strategies through **competitive micro-capital allocation**.

## 🎯 Current Status

### ✅ **Definition of Done - ACHIEVED**

| Component | Status | Details |
|-----------|--------|---------|
| **Tradier Paper Adapter** | ✅ **READY** | Order placement, fills, position sync, PnL tracking |
| **Competition Engine** | ✅ **READY** | Idempotent rebalance with audit trail |
| **Fitness Scoring** | ✅ **READY** | 0.67 sentiment weight + unit tests |
| **Risk Management** | ✅ **READY** | Kill switches, position limits, circuit breakers |
| **Decision Transparency** | ✅ **READY** | DecisionTrace with WHY text + evidence links |
| **Operational Readiness** | ✅ **READY** | Health checks, crash recovery, version endpoints |
| **Infrastructure** | ✅ **READY** | Docker, CI/CD, reproducible builds |

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   Node.js API   │    │   Tradier API   │
│   (Port 3003)   │◄──►│   (Port 4000)   │◄──►│   (Paper/Live)  │
│                 │    │                 │    │                 │
│ • Competition   │    │ • Health Checks │    │ • Order Mgmt    │
│ • Real-time     │    │ • Risk Controls │    │ • Position Sync │
│ • Brain Flow    │    │ • Fitness Calc  │    │ • PnL Tracking  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  AI Competition │
                    │   Engine        │
                    │                 │
                    │ • Strategy Evo  │
                    │ • Micro-Capital │
                    │ • Fitness Score │
                    │ • Rebalancing   │
                    └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/TheClitCommander/Bensmoneymakingbot.git
cd Bensmoneymakingbot

# Install dependencies
npm install
cd new-trading-dashboard && npm install

# Set up environment
cp sandbox.env.example .env
# Edit .env with your Tradier API token
```

### 2. Start Services
```bash
# Terminal 1: Start backend API
cd live-api && node server.js

# Terminal 2: Start frontend
cd new-trading-dashboard && npm run dev
```

### 3. Access Dashboard
- **Frontend:** http://localhost:3003
- **API:** http://localhost:4000
- **Health Check:** http://localhost:4000/health

### 4. Test Paper Trading
```bash
# Run comprehensive smoke test
node scripts/smoke_test.js

# Test real order placement
node scripts/demo_place_order.js place-test
```

---

## 🎮 Key Features

### 🤖 **AI Bot Competition System**
- **Micro-capital allocation** ($100-$1,000 per bot)
- **Real-time competition tracking** with live leaderboard
- **Snowball effect** - winners get exponentially more capital
- **Automatic reallocation** every hour based on performance
- **Risk-free validation** - small amounts prove strategy viability

### 🧬 **Evolutionary Strategy Engine**
- **Genetic algorithm optimization** for trading strategies
- **Cross-symbol intelligence** - strategies learn across SPY, AAPL, NVDA, TSLA, BTC-USD
- **Sentiment integration** - 67% weighting in fitness scoring
- **Market regime adaptation** - strategies evolve with changing conditions
- **Real-time fitness tracking** with live metrics dashboard

### 🛡️ **Risk Management**
- **Position limits** - Max 5% per trade, max 20% per symbol
- **Daily loss cap** - Halt trading if drawdown > 10%
- **Kill switch** - Emergency stop within 1 second
- **Circuit breakers** - VIX > 35 triggers conservative mode

### 📊 **Decision Transparency**
- **DecisionTrace events** - Real-time decision streaming
- **WHY text** - Plain-English rationale for each decision
- **Evidence links** - News articles supporting decisions
- **Audit trail** - Complete history of all trades and decisions

---

## 📋 API Endpoints

### Health & Monitoring
```http
GET  /health              # Basic health check
GET  /api/health          # Enhanced health with broker connectivity
GET  /version             # Version information
```

### Competition Engine
```http
POST /api/competition/rebalance           # Trigger capital reallocation
GET  /api/competition/allocations         # Current allocations
GET  /api/competition/rebalance-history   # Rebalance audit trail
```

### Fitness Scoring
```http
GET  /api/fitness/config   # Fitness configuration (0.67 sentiment weight)
POST /api/fitness/test     # Test fitness calculation
```

### Risk Management
```http
GET  /api/admin/kill-switch     # Get kill switch status
POST /api/admin/kill-switch     # Enable/disable kill switch
```

### Paper Trading
```http
GET  /api/paper/account     # Account balance and info
GET  /api/paper/positions   # Current positions
GET  /api/paper/orders      # Order history
POST /api/paper/orders      # Place new order
```

---

## 🧪 Testing

### Comprehensive Smoke Test
```bash
node scripts/smoke_test.js
```
**Tests:** Health, broker, competition, fitness, risk management, infrastructure

### Order Placement Test
```bash
node scripts/demo_place_order.js place-test
```
**Tests:** Real Tradier API integration, order lifecycle, PnL tracking

### Fitness Formula Verification
```bash
node scripts/test_fitness_formula.js
```
**Tests:** 0.67 sentiment weight calculation, normalization, unit tests

---

## 🐳 Docker Deployment

### Build Images
```bash
# Build API image
docker build -t benbot-api .

# Build UI image
docker build -f Dockerfile.ui -t benbot-ui .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### Production Deployment
```bash
# Set environment variables
export TRADIER_TOKEN=your_token_here

# Run in production
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🔧 Configuration

### Environment Variables (.env)
```env
# Tradier API (Paper Trading)
TRADIER_TOKEN=your_tradier_api_token_here
TRADIER_BASE_URL=https://sandbox.tradier.com/v1

# System Configuration
NODE_ENV=production
QUOTES_PROVIDER=tradier
AUTOREFRESH_ENABLED=1

# Risk Management
DD_MIN_MULT=1.0
DD_FLOOR=0.05
RATE_LIMIT_QPS=10
```

### Fitness Configuration
```javascript
const FITNESS_WEIGHTS = {
  sentiment: 0.67,        // 67% weight on sentiment analysis
  pnl: 0.25,              // 25% weight on profit/loss
  drawdown: -0.08,        // -8% penalty for drawdown
  sharpe_ratio: 0.05,     // 5% weight on Sharpe ratio
  win_rate: 0.03,         // 3% weight on win rate
  volatility_penalty: -0.01 // -1% penalty for volatility
};
```

---

## 🎯 Fitness Scoring Formula

### Complete Formula
```
fitness = 0.67 × sentiment + 0.25 × pnl - 0.08 × drawdown + 0.05 × sharpe + 0.03 × win_rate - 0.01 × volatility
```

### Normalization Rules
- **Sentiment:** (-1..1) → (0..1) via `(sentiment + 1) / 2`
- **PnL:** Percentage → decimal via `total_return / 100`
- **Drawdown:** Capped at 50%, higher = worse fitness
- **Volatility:** Capped at 100%, higher = worse fitness

### Example Calculation
```javascript
Input: {
  sentiment_score: 0.8,    // +80% sentiment
  total_return: 25,        // +25% return
  max_drawdown: 15,        // 15% drawdown
  sharpe_ratio: 1.8,       // Sharpe ratio
  win_rate: 0.65,          // 65% win rate
  volatility: 20           // 20% volatility
}

Normalized: {
  sentiment: 0.9,
  pnl: 0.25,
  drawdown: 0.3,
  volatility: 0.2
}

Fitness: 0.67×0.9 + 0.25×0.25 + (-0.08)×0.3 + 0.05×1.8 + 0.03×0.65 + (-0.01)×0.2 = 0.749
```

---

## 📈 Competition Rules

### Capital Allocation
- **Starting Capital:** $100-$1,000 per bot
- **Winner Bonus:** +20% additional capital
- **Loser Penalty:** -50% capital reduction
- **Snowball Effect:** 60% of profits reinvested

### Rebalancing Schedule
- **Frequency:** Every hour
- **Lock Mechanism:** File-based locks prevent double-allocation
- **Audit Trail:** Complete history of all capital movements
- **Idempotency:** Safe to restart without data corruption

### Fitness-Based Allocation
```
allocation_ratio = bot_fitness / total_fitness
new_capital = max(min_allocation, allocation_ratio × total_pool)
```

---

## 🚨 Risk Controls

### Position Limits
- **Per Trade:** Max 5% of portfolio
- **Per Symbol:** Max 20% of portfolio
- **Total Exposure:** Max 100% of portfolio

### Circuit Breakers
- **Daily Loss:** >10% triggers halt
- **VIX Level:** >35 triggers conservative mode
- **Volatility:** >50% reduces position sizes
- **Kill Switch:** Emergency halt within 1 second

### Kill Switch Implementation
```javascript
// Global kill switch
global.killSwitchEnabled = false;

// Check in order placement
if (global.killSwitchEnabled) {
  throw new Error('KILL_SWITCH_ENABLED');
}
```

---

## 🔍 Monitoring & Debugging

### Health Endpoints
```bash
# Basic health
curl http://localhost:4000/health

# Enhanced health with broker
curl http://localhost:4000/api/health

# Version info
curl http://localhost:4000/version
```

### Logs
```bash
# View server logs
tail -f live-api/server.log

# View competition logs
tail -f live-api/data/rebalance_log.json
```

### Debug Commands
```bash
# Test fitness calculation
curl -X POST http://localhost:4000/api/fitness/test \
  -H "Content-Type: application/json" \
  -d '{"sentiment_score": 0.8, "total_return": 25, "max_drawdown": 15}'

# Trigger rebalance
curl -X POST http://localhost:4000/api/competition/rebalance

# Check allocations
curl http://localhost:4000/api/competition/allocations

# Enable kill switch
curl -X POST http://localhost:4000/api/admin/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

---

## 🎯 Roadmap

### Phase 1.0 (Current) - Paper Trading ✅
- [x] Tradier paper adapter
- [x] AI competition engine
- [x] Risk management
- [x] Real-time monitoring
- [x] Docker infrastructure

### Phase 2.0 (Next) - Live Trading
- [ ] Live broker adapter
- [ ] Advanced strategy evolution
- [ ] Multi-asset arbitrage
- [ ] Neural network strategies
- [ ] Distributed computing

### Phase 3.0 (Future) - Hedge Fund Scale
- [ ] Multi-broker routing
- [ ] Cross-exchange arbitrage
- [ ] Machine learning integration
- [ ] Real-time market microstructure
- [ ] Institutional-grade risk management

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** changes with tests
4. **Run** smoke tests: `node scripts/smoke_test.js`
5. **Submit** pull request

### Development Setup
```bash
# Install dependencies
npm install
cd new-trading-dashboard && npm install

# Run tests
npm test
cd new-trading-dashboard && npm test

# Build for production
npm run build
cd new-trading-dashboard && npm run build

# Run smoke test
node scripts/smoke_test.js
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test with paper trading accounts before using real capital.

**Use at your own risk.** The authors are not responsible for any financial losses incurred through the use of this software.

---

## 🎉 Success Metrics

- ✅ **Paper Trading:** Successfully place and track orders
- ✅ **Competition:** AI bots compete with real micro-capital
- ✅ **Evolution:** Strategies improve through natural selection
- ✅ **Risk Control:** No catastrophic losses in testing
- ✅ **Transparency:** Clear WHY behind every decision

**BenBot is ready for paper trading!** 🚀🧬⚔️💰

---

*Built with ❤️ for algorithmic trading innovation*

