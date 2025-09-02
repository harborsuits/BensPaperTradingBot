# Straddle/Strangle Strategy Monthly Review Process

## Purpose
This document outlines the structured review process for continuously monitoring, evaluating, and tuning the Straddle/Strangle options strategy. The goal is to ensure the strategy remains effective across changing market conditions and to guide decisions about parameter adjustments or strategy retirement.

## Review Schedule
- **Frequency**: Monthly
- **Timing**: First business day of each month
- **Duration**: 2-3 hours
- **Participants**: Trading Strategy Developer, Portfolio Manager

## Data Collection Requirements

### Performance Metrics
| Metric | Calculation | Target | Warning Threshold |
|--------|-------------|--------|-------------------|
| Win Rate | `successful_trades / total_trades` | >50% | <40% |
| Average P&L | `total_profit_loss / total_trades` | >3× transaction costs | <2× transaction costs |
| Sharpe Ratio | `(return - risk_free) / std_dev` | >1.0 | <0.7 |
| Max Drawdown | `max(cum_returns - running_max)` | <15% | >20% |
| Volatility of Returns | Standard deviation of returns | <15% | >20% |

### Market Environment Data
- Average VIX level during period
- Highest and lowest VIX during period 
- Major market events (earnings, Fed decisions, etc.)
- Market regime classification (high/low volatility, bullish/bearish)

### Trade Data
- Complete list of all trades executed
- Entry/exit reasons for each trade
- Expiration dates selected vs. predicted events
- Slippage analysis (expected vs. actual execution prices)

## Review Process

### 1. Performance Comparison

```
# Template Table: Backtest vs. Live Performance
| Metric | Backtest Expectation | Live Performance | Deviation | Action Required |
|--------|----------------------|-----------------|-----------|-----------------|
| Win Rate | % | % | % | Yes/No |
| Avg P&L | $ | $ | $ | Yes/No |
| Sharpe | # | # | # | Yes/No |
| Max DD | % | % | % | Yes/No |
```

- Flag any metric where live deviation exceeds ±10% from backtest expectations
- Calculate confidence intervals on metrics given sample size

### 2. Market Regime Analysis

1. **Regime Classification**
   - Determine which regime the past month represents (high/low vol, bullish/bearish)
   - Compare to pre-defined regime parameters from grid search
   - Calculate how well actual parameters fit the regime

2. **Parameter Effectiveness**
   - Evaluate if current parameters were optimal for the regime
   - Project what performance would have been with regime-optimal parameters

### 3. Trade Post-Mortem

1. **Winning Trades Analysis**
   - Common characteristics of profitable trades
   - Which exit conditions were most effective
   
2. **Losing Trades Analysis**
   - Common failure patterns
   - Potential improvements to entry criteria
   - Alternative exit criteria that would have improved outcomes

3. **Missed Opportunities**
   - Signals that weren't taken (due to capital constraints, etc.)
   - Potential trades that the system missed but fit the strategy

### 4. Parameter Adjustment Decision

1. **Criteria for Parameter Changes**
   - Consistent deviation from expected metrics (>10% for 2+ months)
   - Clear pattern in losing trades addressable by parameter adjustments
   - Market regime shift requiring pre-determined parameter changes

2. **Parameter Adjustment Implementation**
   ```python
   # Example parameter adjustment record
   {
       "date": "2025-06-01",
       "reason": "Persistent underperformance in low-vol environment",
       "changes": {
           "profit_target_pct": [40, 35],  # [old, new]
           "exit_iv_drop_pct": [20, 15]    # [old, new]
       },
       "expected_impact": "Faster profit taking in low-vol periods; expect +5% win rate, -10% avg profit"
   }
   ```

3. **Tracking Change Impact**
   - Create specific tracking metrics for each parameter change
   - Set review period for evaluating effectiveness (typically 50 trades or 2 months)

### 5. Strategy Continuation Decision

1. **Criteria for Strategy Suspension**
   - Review performance against abandonment thresholds (see below)
   - Document market conditions that are definitively unfavorable
   - Calculate opportunity cost versus other strategies

2. **Abandonment Threshold Review**

   ```
   # Abandonment Thresholds
   - 3+ consecutive months of net losses
   - Win rate < 40% over 20+ trades
   - Max drawdown > 20% 
   - VIX < 15 for > 30 days
   - Other strategies outperforming by > 25% ROI
   ```

3. **Partial vs. Complete Abandonment**
   - Consider reducing allocation rather than complete discontinuation
   - Evaluate performance on specific symbols vs. all symbols
   - Consider strategy variant adjustments (straddle vs. strangle)

## Documentation Requirements

### Monthly Review Report Template

```markdown
# Straddle/Strangle Strategy Review - [MONTH YEAR]

## Executive Summary
- Overall performance assessment: [GOOD/ADEQUATE/POOR]
- Key metrics: Win rate: %, Avg P&L: $, Sharpe: #, Max DD: %
- Market regime: [HIGH_VOL/LOW_VOL/BULLISH/BEARISH]
- Recommended actions: [MAINTAIN/ADJUST/SUSPEND]

## Detailed Performance Analysis
[Insert performance comparison table]

## Trade Analysis
- Total trades: #
- Winners: # (%)
- Losers: # (%)
- Top performing symbols: 
- Underperforming symbols:

## Parameter Effectiveness
- Current parameters fit for regime: [YES/NO]
- Parameter adjustment needed: [YES/NO]
- Specific adjustments recommended:
  - [Parameter]: [Old] → [New]
  - [Parameter]: [Old] → [New]

## Forward Outlook
- Expected market regime next month:
- Recommended allocation:
- Specific symbols to focus on/avoid:

## Additional Notes
[Any other observations or concerns]
```

### Parameter Change Log

Maintain a version-controlled JSON file at `config/straddle_config_history.json`:

```json
{
  "parameter_history": [
    {
      "date": "2025-05-01",
      "version": "1.0.0",
      "parameters": {
        "strategy_variant": "straddle",
        "profit_target_pct": 40,
        "stop_loss_pct": 60,
        "max_dte": 30,
        "exit_dte": 7,
        "exit_iv_drop_pct": 20,
        "max_drawdown_threshold": 8
      },
      "reason": "Initial implementation",
      "performance": {
        "win_rate": 0,
        "avg_pnl": 0,
        "sharpe": 0
      }
    },
    {
      "date": "2025-06-01",
      "version": "1.0.1",
      "parameters": {
        "strategy_variant": "straddle",
        "profit_target_pct": 35,
        "stop_loss_pct": 60,
        "max_dte": 30,
        "exit_dte": 7,
        "exit_iv_drop_pct": 15,
        "max_drawdown_threshold": 10
      },
      "reason": "Adjustment for low volatility environments",
      "performance": {
        "win_rate": 45,
        "avg_pnl": 120,
        "sharpe": 0.8
      }
    }
  ]
}
```

## Implementation Checklist

- [ ] Set up automated data collection for monthly review
- [ ] Create Jupyter notebook template for analysis
- [ ] Implement parameter versioning system in strategy code
- [ ] Configure alerts for hitting abandonment thresholds
- [ ] Schedule monthly review meeting
- [ ] Set up dashboard for ongoing monitoring between reviews

## Additional Resources

- [Strategy Optimization Grid Search Scripts](../scripts/straddle_grid_search.py)
- [Backtesting Results Archive](../reports/straddle_validation/)
- [Live Trading Logs](../logs/straddle_logs/)
