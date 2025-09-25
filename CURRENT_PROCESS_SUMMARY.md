# BenBot Current Process Summary 📋

## 🔄 How Your Bot Currently Works

### 1. **Every 30 Seconds** (Main Trading Loop)
```
1. Health Check → Is market open? Data fresh? Capital available?
2. Find Candidates → Scanner (watchlists) + News ("diamonds") + Dynamic discovery
3. Test Strategies → Each candidate tested against RSI & MA strategies
4. Calculate Value → Expected profit - costs = Expected Value
5. Risk Check → Position limits, capital limits, correlation check
6. Size Position → 1-2% risk per trade, max 10% position
7. Execute Trade → Paper broker simulates the trade
8. Record Results → Database + performance tracking
```

### 2. **Every 15 Minutes** (AI Orchestrator)
```
1. Check Market → Bull/Bear? High/Low volatility?
2. Review Strategies → Which are winning? Which are losing?
3. Tournament Update → Promote winners, demote losers
4. Spawn New → Create evolved strategies for testing
```

### 3. **Every Morning at 9:00 AM** (Pre-market)
```
1. Fetch News → What happened overnight?
2. Market Context → What's the market mood?
3. Prepare Watchlists → Update symbols to monitor
```

## 📊 Current Strategy Logic

### RSI Strategy (Oversold/Overbought)
- **Buy Signal**: RSI < 30 (oversold)
- **Sell Signal**: RSI > 70 (overbought)
- **Exit**: When RSI normalizes or stop loss hit

### MA Crossover Strategy
- **Buy Signal**: 5-period MA crosses above 20-period MA
- **Sell Signal**: 5-period MA crosses below 20-period MA
- **Exit**: Opposite crossover or stop loss

## 💰 Position Management

### Entry Rules
- Must have fresh data (< 5 seconds old)
- Must have positive expected value
- Must pass risk gates
- Cannot exceed position limits

### Exit Rules
- Stop loss: 2% default
- Take profit: 5% default
- Quick exit for news trades (< 1 hour)
- Daily cleanup of stale positions

## 🎯 Current Performance Factors

### What's Working Well
1. **Safety Systems**: Circuit breakers prevent major losses
2. **Data Validation**: Won't trade on bad data
3. **News Response**: Quick reaction to breaking news
4. **Learning System**: Tracks what works/doesn't work

### What Needs Improvement
1. **Limited Strategies**: Only 2 active (RSI, MA)
2. **Basic Exits**: No trailing stops or profit targets
3. **Symbol Discovery**: Limited to same ~20 symbols
4. **Position Sizing**: Fixed percentage, not adaptive

## 📈 Key Metrics Explained

### Win Rate (37%)
- **Current**: 21 wins out of 57 trades
- **Target**: 45%+ 
- **Why Low**: Basic strategies, no trend following

### Daily Return (0.026%)
- **Current**: $26/day on $100k
- **Target**: $100-200/day (0.1-0.2%)
- **Why Low**: Conservative position sizing, limited strategies

### Best Performers
- **BB (+600%)**: Caught a major move
- **F (+223%)**: Good timing on reversal
- **NOK (+32%)**: Steady trend capture

## 🔍 Data Flow

```
Market Data (Tradier) → Quote Service → AutoLoop
                           ↓
                      Strategies Test
                           ↓
                      Risk Validation
                           ↓
                      Paper Broker
                           ↓
                      Database/Learning
```

## 🚦 Decision Example

```
Symbol: AAPL
Price: $150
RSI: 28 (oversold)
MA: No signal
News: None
---
Decision: BUY
Confidence: 72%
Size: $1,500 (1% risk)
Stop: $147 (2%)
Target: $157.50 (5%)
```

## 💡 Quick Improvement Ideas

1. **Add VWAP Strategy**: Buy below VWAP, sell above
2. **Trailing Stops**: Lock in profits on winners
3. **News Sentiment**: Score news positive/negative
4. **Volume Confirmation**: Only trade with volume
5. **Time Stops**: Exit if no movement in 24h

## 📱 How to Monitor

1. **Dashboard**: Shows positions, P&L, recent trades
2. **Logs**: `live-api/server-latest.log` for details
3. **Database**: SQLite has all historical data
4. **WebSocket**: Real-time updates in UI

## 🔧 Configuration Files

- `ai_policy.yaml`: Capital limits, risk parameters
- `watchlists.json`: Symbols to monitor
- `tradingThresholds.js`: Entry/exit thresholds
- `.env`: API keys and settings

This is your bot's current state - a solid foundation ready for enhancements! 🚀
