# BenBot System Flow Diagram

## 📊 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA LAYER                              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│ Tradier API     │ News Aggregator │ Market Scanner  │ Quote Service   │
│ (Prices/Orders) │ (AlphaVantage)  │ (Candidates)    │ (Real-time)     │
└────────┬────────┴────────┬────────┴────────┬────────┴────────┬────────┘
         │                 │                  │                  │
         ▼                 ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DECISION LAYER (30s cycle)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  AutoLoop                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ 1. Health   │→ │ 2. Discovery │→ │ 3. Signals  │→ │ 4. Risk Gate │ │
│  │    Check    │  │   Candidates │  │  Generation │  │  Validation  │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
│         │                 │                  │                  │        │
│         ▼                 ▼                  ▼                  ▼        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Circuit     │  │ Scanner +    │  │ Strategy    │  │ Expected     │ │
│  │ Breakers    │  │ News Filter  │  │ Manager     │  │ Value Calc   │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Position    │  │ Paper Broker │  │ Order       │  │ Capital      │ │
│  │ Sizing      │→ │ (Simulation) │→ │ Management  │→ │ Tracker      │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEARNING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Performance │  │ Trade        │  │ Pattern     │  │ Strategy     │ │
│  │ Recorder    │→ │ Analytics    │→ │ Learning    │→ │ Evolution    │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
│         │                 │                  │                  │        │
│         ▼                 ▼                  ▼                  ▼        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SQLite Database                               │   │
│  │  trades | decisions | strategies | performance | learning_data   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY EVOLUTION (15min cycle)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  AI Orchestrator                                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Market      │  │ Tournament   │  │ Evolution   │  │ Strategy     │ │
│  │ Context     │→ │ Controller   │→ │ Bridge      │→ │ Spawning     │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
│                                                                         │
│  Tournament Stages: R1 (Paper) → R2 (Small $) → R3 (Medium $) → Live   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Key Process Flows

### 1. **Trade Decision Flow** (every 30 seconds)
```
Market Data → Candidate Discovery → Strategy Analysis → Risk Validation 
    → Position Sizing → Order Execution → Performance Recording
```

### 2. **Learning Feedback Loop**
```
Trade Results → Performance Analysis → Pattern Recognition 
    → Confidence Adjustment → Strategy Weight Update → Next Trade
```

### 3. **Strategy Evolution Flow** (every 15 minutes)
```
Performance Metrics → Tournament Evaluation → Promotion/Demotion 
    → Genetic Breeding → New Strategy Creation → R1 Testing
```

### 4. **Risk Management Chain**
```
Pre-Trade Gates → Position Limits → Capital Constraints 
    → Circuit Breakers → Emergency Halt
```

## 📊 Data Storage

### Real-time Memory
- Active positions
- Pending orders  
- Recent trades
- Market context
- Strategy states

### Persistent Storage (SQLite)
- Trade history
- Decision log
- Strategy parameters
- Performance metrics
- Learning observations

### Configuration Files
- strategies.json
- watchlists.json
- ai_policy.yaml
- tradingThresholds.js

## 🎯 System States

1. **IDLE**: No signals, waiting for opportunities
2. **DISCOVERING**: Finding trading candidates
3. **ANALYZING**: Strategies evaluating candidates
4. **COORDINATING**: Resolving conflicting signals
5. **VALIDATING**: Risk gate checks
6. **EXECUTING**: Placing orders
7. **MONITORING**: Watching open positions
8. **LEARNING**: Recording and analyzing results
9. **HALTED**: Circuit breaker activated

## 🔐 Safety Mechanisms

1. **Data Validation**: Rejects stale quotes (>5 seconds)
2. **Capital Limits**: Max $20k deployed, min $1k cash buffer
3. **Position Limits**: Max 10 concurrent positions
4. **Daily Limits**: Max 50 trades per day
5. **Drawdown Protection**: Halts at 5% daily loss
6. **Circuit Breakers**: Auto-stop on system failures
7. **Network Resilience**: Pauses on disconnection
