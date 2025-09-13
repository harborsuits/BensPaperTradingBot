# 🤖 Autonomous AI Orchestrator System

## Overview

The AI Orchestrator is a fully autonomous trading system that manages the entire lifecycle of trading strategies end-to-end, from initial seeding through live deployment, without human intervention. It continuously monitors market conditions, evaluates strategy performance, and makes intelligent decisions about when to spawn, promote, demote, or retire strategies.

## 🎯 Key Features

### Autonomous Operation
- **15-minute decision cycles** during market hours
- **Zero-touch strategy management** - completely automated
- **Self-adapting to market regimes** - trend, mean-reversion, breakout strategies
- **Capacity-aware spawning** - respects budget and slot constraints
- **Performance-based lifecycle** - automatic promotion/demotion

### Intelligent Triggers
- **Regime Trigger**: Spawns trend/meanrev/breakout families based on market conditions
- **Capacity Trigger**: Top-ups roster when budget >30% free or slots <80% utilized
- **Decay Trigger**: Demotes strategies with Sharpe <0.8 or edge Δ < -25%
- **Event Trigger**: Spawns short-horizon strategies for earnings/FOMC/CPI events
- **Drift Trigger**: Reseeds families when feature/alpha drift detected
- **Novelty Trigger**: Maintains 10% exploration quota for new strategy discovery

### Safety & Quality Gates
- **Pre-registration filters**: Liquidity, borrow/SSR, earnings blackout
- **Execution gates**: Spread, quote-age, participation limits
- **Promotion gates**: R1→R2→R3→Live with strict criteria
- **Live demotion**: 2+ failures trigger automatic removal

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Brain      │    │  Policy Engine   │    │  Seed Generator │
│                 │    │                  │    │                 │
│ • Market Context│    │ • Trigger Logic  │    │ • Gene Bounds   │
│ • Decision Loop │    │ • Capacity Mgmt  │    │ • Family Select │
│ • Orchestration │    │ • Safety Gates   │    │ • Phenotype Gen │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────────────┐
                    │ Tournament System  │
                    │                    │
                    │ • R1 Micro Capital │
                    │ • R2 Growth Capital│
                    │ • R3 Pre-Live      │
                    │ • Live Probation   │
                    └────────────────────┘
```

## 📊 Tournament Rounds

| Round | Capital | Duration | Criteria | Max Slots |
|-------|---------|----------|----------|-----------|
| **R1** | $100-200 | 7 days | Sharpe≥0.9, PF≥1.15, DD≤8% | 50 |
| **R2** | $500-800 | 10 days | Sharpe≥1.1, PF≥1.25, DD≤7% | 20 |
| **R3** | $1500-2000 | 15 days | Sharpe≥1.2, PF≥1.3, DD≤6% | 8 |
| **Live** | $2500-5000 | Ongoing | Probation monitoring | ∞ |

## 🎛️ Strategy Families

### Trend Following
- **Parameters**: MA fast/slow, ADX min, Pullback ATR, Stop ATR, TP Ratio
- **Regimes**: Trending markets, moderate volatility
- **Gene Bounds**: `ma_fast:[8,20], ma_slow:[40,200], adx_min:[18,30]`

### Mean Reversion
- **Parameters**: RSI length, RSI buy/sell levels, BB deviation, Hold time
- **Regimes**: Choppy markets, low volatility
- **Gene Bounds**: `rsi_len:[9,21], rsi_buy:[20,35], bb_dev:[1.6,2.4]`

### Breakout
- **Parameters**: Range minutes, Min volume multiple, Slippage max, Stop ATR
- **Regimes**: Volatile markets, event-driven
- **Gene Bounds**: `range_mins:[5,15], min_vol_x:[1.2,2.0]`

## 🚀 Quick Start

1. **Start the system**:
   ```bash
   ./start_ai_orchestrator.sh
   ```

2. **Monitor autonomous operation**:
   - Dashboard: http://localhost:3003
   - AI Status Card shows real-time decisions
   - API Status: http://localhost:4000/api/ai/status

3. **Manual intervention** (if needed):
   ```bash
   curl -X POST http://localhost:4000/api/ai/trigger-cycle
   ```

## 📡 API Endpoints

### Core AI Endpoints
- `GET /api/ai/status` - Current orchestrator status and recent decisions
- `GET /api/ai/context` - Market context, roster metrics, capacity status
- `GET /api/ai/policy` - Current AI policy configuration
- `POST /api/ai/trigger-cycle` - Manually trigger orchestration cycle

### EvoTester Integration
- `POST /api/ai/evo/seed` - Request phenotype generation
- `POST /api/ai/evo/feedback` - Send promotion/demotion feedback

### Strategy Management
- `GET /api/live/strategies` - All strategies with tournament stage
- `GET /api/live/tournament` - Tournament ladder and pass rates
- `POST /api/live/strategies/:id/promote` - Manual promotion override

## 🎨 UI Components

### AI Orchestrator Status Card
- Real-time orchestration status (Active/Inactive)
- Cycle count and last run timestamp
- Current market regime and VIX level
- Tournament stage utilization bars
- Recent AI decisions feed
- Manual trigger button

### Tournament Ladder
- Visual representation of R1→R2→R3→Live progression
- Pass rates and capacity indicators
- Strategy counts per stage
- Performance metrics aggregation

### Decision Feed
- Real-time AI decision announcements
- Trigger explanations ("Regime=Chop → spawning MeanRev")
- Strategy lifecycle events
- Performance-based actions

## ⚙️ Configuration

### AI Policy (ai_policy.yaml)
```yaml
ai_policy:
  paper_cap_max: 20000          # Global paper budget
  exploration_quota: 0.10        # 10% for novel strategies
  rounds:
    R1: { max_slots: 50, cap_total: 5000 }
    R2: { max_slots: 20, cap_total: 8000 }
    R3: { max_slots: 8,  cap_total: 7000 }

families:
  trend: { weight: 0.5, genes: { ma_fast: [8,20], ... } }
  meanrev: { weight: 0.3, genes: { rsi_len: [9,21], ... } }
  breakout: { weight: 0.2, genes: { range_mins: [5,15], ... } }

triggers:
  capacity_management:
    paper_budget_threshold: 0.3    # 30% free triggers spawn
    roster_gap_threshold: 0.2      # 20% below target
  decay_detection:
    sharpe_decay_threshold: -0.25
    edge_decay_threshold: -0.25
```

## 🔧 Technical Implementation

### AI Orchestrator Service
- **15-minute intervals** during market hours (9:30 AM - 4:00 PM ET)
- **Trigger evaluation** using Policy Engine
- **Decision execution** through Tournament Controller
- **Feedback loop** to EvoTester for learning

### Policy Engine
- **Trigger evaluation**: Capacity, Decay, Regime, Event, Drift, Novelty
- **Family selection**: Context-aware with roster balancing
- **Gene bounds**: Dynamic adaptation by market regime
- **Safety gates**: Pre/post-trade validation

### Seed Generator
- **Family-aware generation**: Trend/MeanRev/Breakout phenotypes
- **Parameter validation**: Statistical bounds and constraints
- **Regime adaptation**: Gene bounds shift by market conditions
- **Quality assurance**: Parameter range validation

## 📈 Monitoring & Transparency

### Real-time Metrics
- **Orchestrator status**: Active cycles, trigger firings, decision counts
- **Market context**: Regime, volatility, VIX, calendar events
- **Roster health**: Strategy counts, performance distribution, underperformers
- **Capacity utilization**: Budget used, slot availability, stage distribution

### Decision Transparency
- **Trigger explanations**: Why each decision was made
- **Performance metrics**: Before/after stats for promotions/demotions
- **Regime adaptation**: How gene bounds changed with market conditions
- **Capacity reasoning**: Budget/slot utilization triggers

### Alert System
- **Critical failures**: Tournament stalls, capacity exhaustion
- **Performance alerts**: Mass demotions, spawn failures
- **System health**: Service availability, integration status

## 🛡️ Safety & Risk Management

### Budget Controls
- **Paper cap**: $20k global limit across all strategies
- **Per-round caps**: R1≤$5k, R2≤$8k, R3≤$7k
- **Concurrency limits**: Max 50 R1, 20 R2, 8 R3 active

### Execution Safety
- **Pre-trade filters**: Liquidity, borrow availability, earnings blackout
- **Execution gates**: Spread limits, quote age, participation caps
- **Position sizing**: Risk-based allocation per strategy
- **Kill switches**: Emergency halt capabilities

### Live Probation
- **Strict criteria**: Sharpe≥1.2, PF≥1.3, DD≤6%
- **Demotion triggers**: 2+ failures (Sharpe<0.8, DD>6%, 3 breaches/24h)
- **Graduated sizing**: Start small, scale with performance
- **Continuous monitoring**: Real-time risk assessment

## 🔄 Evolutionary Learning

### Feedback Integration
- **Promotion feedback**: Successful parameter combinations
- **Demotion feedback**: Failed parameter combinations
- **Regime adaptation**: Market condition learning
- **Performance weighting**: Better strategies get higher influence

### Continuous Improvement
- **Parameter optimization**: Genetic algorithm refinement
- **Family balancing**: Dynamic weight adjustment
- **Regime specialization**: Context-specific parameter tuning
- **Exploration vs Exploitation**: Novelty injection for discovery

## 🚀 Future Enhancements

- **Multi-asset support**: Extend beyond equities
- **Options integration**: Complex strategy types
- **Sentiment analysis**: News-driven decision making
- **Portfolio optimization**: Cross-strategy correlation management
- **Machine learning**: Deep learning for pattern recognition
- **High-frequency adaptation**: Sub-minute regime detection

---

## 🎯 Summary

The AI Orchestrator represents a fully autonomous trading system that can:

1. **Continuously monitor** market conditions and strategy performance
2. **Automatically spawn** new strategies when capacity allows
3. **Progress strategies** through tournament rounds based on merit
4. **Adapt to changing** market regimes with appropriate families
5. **Maintain portfolio health** through performance-based lifecycle management
6. **Learn and improve** through evolutionary feedback loops

This system eliminates manual strategy management while maintaining strict risk controls and providing complete transparency into all autonomous decisions.
