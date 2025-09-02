# Decision Router System

This document explains the policy-driven decision router system that helps choose between ETF, Crypto, and Options trading opportunities consistently, legally, and efficiently.

## Overview

The Decision Router is a policy-driven system that:

1. Applies strict compliance gates to trading opportunities
2. Scores opportunities based on multiple factors
3. Routes approved opportunities to instrument-specific playbooks
4. Maintains an audit trail of all decisions (including rejections)
5. Enforces risk limits and compliance rules

## Key Components

### 1. Policy Framework

- **Policy Types**: Defines the structure of trading policies
- **Default Policy**: Provides baseline settings
- **Policy Service**: Manages policy loading, validation, and hot reloading

### 2. Compliance Gates

- **Jurisdiction Checks**: Ensures trading only in allowed jurisdictions
- **Venue Whitelists**: Restricts trading to approved venues
- **Public Disclosure Rules**: Enforces lagged use of public disclosures
- **Instrument-Specific Gates**: Applies specialized rules for each instrument type

### 3. Scoring Engine

- **Multi-Factor Scoring**: Combines alpha, regime alignment, sentiment, costs, and risk
- **Explainable Results**: Provides detailed reasons for each score
- **Configurable Weights**: Allows tuning the importance of each factor

### 4. Instrument Playbooks

- **ETF Trend**: Conservative limit orders with vol-budget sizing
- **Crypto Burst**: Quick IOC orders with tight time constraints
- **Options Events**: Defined-risk spreads for event trading

### 5. Audit System

- **Decision Trail**: Records all decisions with timestamps and reasons
- **Gate Failures**: Tracks compliance gate failures
- **Statistics**: Provides aggregated decision statistics

### 6. Decision Engine

- **Opportunity Processing**: Handles batches of trading opportunities
- **Policy Integration**: Applies current policy to decisions
- **Risk Limits**: Enforces daily loss limits and exposure caps
- **Callback System**: Notifies subscribers of trading decisions

### 7. API Integration

- **FastAPI Router**: Exposes decision engine functionality via REST API
- **Opportunity Submission**: Accepts opportunities for processing
- **Decision Retrieval**: Provides access to recent decisions
- **Policy Management**: Allows viewing and updating the current policy

## How the System Chooses Between Instruments

| Instrument | Gate (must pass) | Why it wins | Size/Timing defaults |
|------------|-----------------|-------------|----------------------|
| ETF (trend/carry/defensive) | Liquidity (ADV, spread), market hours, regime ≠ chaos | Low friction, scalable, good risk aggregation | Vol-budget sizing, linearly add, trail exits |
| Crypto (burst/flow) | Fresh sentiment/news anomaly, venue whitelisted, tight slippage, max dwell | Latency edge in short windows | IOC limit, max slippage bps, auto-cancel in seconds |
| Options (event spreads) | Defined-risk only, IV pctile/DTE caps, greeks caps | Event convexity or premium harvesting under policy | Debit/credit spreads only, forced post-event exit |

## Guardrails and Safety Measures

### Hard Stops (Veto)

- Daily loss > policy → PAUSE engine
- Venue not whitelisted → veto
- Options violating IV/DTE/greeks → veto
- Data stale → veto
- Public disclosure signal not lagged → veto

### Soft Throttles

- Vol spike → reduce size budget multiplier
- Drawdown streak → cool-off timer
- Data quality degradation → increase estimated cost, lowering scores

## Usage

### 1. Process Opportunities

```python
from trading_bot.policy.service import PolicyService
from trading_bot.engine.decision_engine import DecisionEngine

# Initialize services
policy_service = PolicyService()
engine = DecisionEngine(policy_service)

# Process opportunities
opportunities = [
    {
        "instrument": "ETF",
        "symbol": "SPY",
        "alpha": 0.7,
        "sentiment_boost": 0.5,
        "est_cost_bps": 1.5,
        "risk_penalty": 0.1,
        "price": 450.25,
        "volume": 1000000,
        "size_budget_usd": 10000,
        "meta": {
            "venue": "NYSE",
            "jurisdiction": "US",
            "adv_usd": 50000000,
            "spread_bps": 1.2
        }
    },
    # More opportunities...
]

routed_opps = engine.process_opportunities(opportunities)
```

### 2. Register Order Callback

```python
def handle_order(opportunity):
    print(f"Executing orders for {opportunity['symbol']}")
    for order in opportunity['orders']:
        print(f"  {order['side']} {order['qty']} @ {order.get('limit', 'MARKET')}")

engine.register_order_callback(handle_order)
```

### 3. Update Policy

```python
new_policy = policy_service.get_policy()
new_policy["risk"]["max_daily_loss_pct"] = 1.5
policy_service.update_policy(new_policy)
```

### 4. View Recent Decisions

```python
from trading_bot.engine.audit import recent_decisions

decisions = recent_decisions(limit=10)
for decision in decisions:
    print(f"{decision['action']} {decision['instrument']} {decision['symbol']}")
    print(f"  Reasons: {', '.join(decision['reasons'])}")
```

## API Endpoints

- `POST /decisions/opportunities`: Process trading opportunities
- `GET /decisions/recent`: Get recent trading decisions
- `GET /decisions/stats`: Get decision statistics
- `GET /decisions/engine/status`: Get decision engine status
- `POST /decisions/engine/enable`: Enable or disable the decision engine
- `POST /decisions/engine/update-loss`: Update daily loss percentage
- `GET /decisions/policy`: Get the current trading policy
- `POST /decisions/policy`: Update the trading policy

## Example

Run the example script to see the decision router in action:

```bash
python examples/decision_router_example.py
```

## Integration with Existing Systems

The Decision Router is designed to integrate with:

1. **Market Regime Classifier**: Provides regime context for scoring
2. **Sentiment Analysis**: Boosts opportunities based on sentiment
3. **News Anomaly Detection**: Enables crypto burst playbook
4. **Risk Management System**: Updates daily loss and risk metrics
5. **Order Execution System**: Receives routed opportunities

## Compliance and Legal Considerations

- The system is designed to enforce strict compliance with legal requirements
- Public disclosure signals must be lagged by at least 24 hours
- Insider trading sources are strictly forbidden
- All decisions are audited with timestamps and reasons

## Next Steps

1. **Paper Testing**: Run with full gates and verify audit trails
2. **Shadow Trading**: Calculate decisions without sending orders
3. **Small Live Testing**: Start with tiny budgets and active monitoring
4. **Weekly Review**: Analyze decisions, gate failures, and performance
5. **Gradual Tuning**: Adjust weights and caps based on performance
