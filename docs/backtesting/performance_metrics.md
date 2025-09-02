# Performance Metrics in the Trading System

This document provides a comprehensive overview of the performance metrics calculated by the backtesting system and explains how to interpret them.

## Table of Contents

1. [Basic Return Metrics](#basic-return-metrics)
2. [Risk Metrics](#risk-metrics)
3. [Risk-Adjusted Performance Metrics](#risk-adjusted-performance-metrics)
4. [Trade Statistics](#trade-statistics)
5. [Drawdown Analysis](#drawdown-analysis)
6. [Rolling Metrics](#rolling-metrics)
7. [Strategy Correlation Analysis](#strategy-correlation-analysis)
8. [Interpreting Performance Reports](#interpreting-performance-reports)

## Basic Return Metrics

### Total Return
- **Description**: The cumulative percentage increase (or decrease) in portfolio value from the start to the end of the backtest.
- **Formula**: `(final_capital / initial_capital - 1) * 100`
- **Interpretation**: Higher values indicate better absolute performance. However, this metric should be viewed alongside risk metrics.

### Absolute Return
- **Description**: The absolute change in portfolio value in currency units.
- **Formula**: `final_capital - initial_capital`

### Annualized Return
- **Description**: The average annual growth rate of the portfolio, which allows for comparing strategies tested over different time periods.
- **Formula**: `(final_capital / initial_capital) ^ (1 / years) - 1`
- **Interpretation**: This normalizes performance over time, making it useful for comparing strategies with different durations.

## Risk Metrics

### Volatility
- **Description**: The annualized standard deviation of daily returns, representing the portfolio's risk or fluctuation.
- **Formula**: `standard_deviation(daily_returns) * sqrt(252)`
- **Interpretation**: Lower volatility generally indicates a more stable strategy, though some strategies intentionally accept higher volatility for higher returns.

### Maximum Drawdown
- **Description**: The largest percentage decline in portfolio value from a previous peak.
- **Formula**: `min((portfolio_value - running_max) / running_max) * 100`
- **Interpretation**: A critical risk measure indicating how much an investor could have lost if they invested at the worst time. Smaller absolute values (less negative) are preferable.

### Value at Risk (VaR)
- **Description**: The expected maximum loss at a specific confidence level (e.g., 95% VaR).
- **Interpretation**: A 95% VaR of -2% means there's a 5% chance of losing more than 2% in a single day.

### Conditional VaR (CVaR)
- **Description**: The expected loss given that the loss exceeds the VaR threshold.
- **Interpretation**: Provides insight into the severity of tail losses. Lower absolute values (less negative) are better.

## Risk-Adjusted Performance Metrics

### Sharpe Ratio
- **Description**: Measures excess return per unit of total risk.
- **Formula**: `(annual_return - risk_free_rate) / volatility`
- **Interpretation**: 
  - < 0: Underperforming the risk-free asset
  - 0-1: Not good on a risk-adjusted basis
  - 1-2: Good
  - 2-3: Very good
  - > 3: Excellent

### Sortino Ratio
- **Description**: Similar to Sharpe but only considers downside risk.
- **Formula**: `(annual_return - risk_free_rate) / downside_deviation`
- **Interpretation**: Like Sharpe ratio, but more relevant for strategies where upside volatility is desired. Higher is better.

### Calmar Ratio
- **Description**: Measures excess return per unit of maximum drawdown.
- **Formula**: `annual_return / abs(max_drawdown)`
- **Interpretation**: Higher values indicate better risk-adjusted returns relative to worst-case losses. Values above 0.5 are generally considered good.

## Trade Statistics

### Win Rate
- **Description**: The percentage of trades that were profitable.
- **Formula**: `profitable_trades / total_trades * 100`
- **Interpretation**: A high win rate is generally good, but must be viewed alongside the win/loss ratio. Some profitable strategies have lower win rates but larger winning trades.

### Profit Factor
- **Description**: The ratio of gross profits to gross losses.
- **Formula**: `sum(profitable_trades) / abs(sum(losing_trades))`
- **Interpretation**: 
  - < 1: Losing strategy
  - 1-1.5: Marginally profitable
  - 1.5-2: Good
  - > 2: Excellent

### Win/Loss Ratio
- **Description**: The ratio of the average win to the average loss.
- **Formula**: `average_win / abs(average_loss)`
- **Interpretation**: Should be viewed alongside win rate. A strategy with a low win rate might still be profitable with a high win/loss ratio.

### Maximum Consecutive Wins/Losses
- **Description**: The longest streak of winning or losing trades.
- **Interpretation**: Helps assess strategy consistency and potential psychological challenges.

## Drawdown Analysis

### Major Drawdowns
- **Description**: Details of significant drawdown periods including:
  - Start and end dates
  - Duration (in days)
  - Depth (as a percentage)
  - Recovery time

- **Interpretation**: Helps understand the worst periods the strategy went through and how long recovery took.

### Drawdown Distribution
- **Description**: The frequency and severity of drawdowns of different magnitudes.
- **Interpretation**: Helps understand the risk profile and recovery patterns of a strategy.

## Rolling Metrics

### Rolling Returns
- **Description**: Annualized returns calculated over a rolling window (e.g., 20 days).
- **Interpretation**: Shows how returns evolve over time and during different market conditions.

### Rolling Volatility
- **Description**: Annualized volatility calculated over a rolling window.
- **Interpretation**: Shows how risk evolves over time and helps identify periods of market stress.

### Rolling Sharpe Ratio
- **Description**: Sharpe ratio calculated over a rolling window.
- **Interpretation**: Shows how risk-adjusted performance evolves over time.

## Strategy Correlation Analysis

- **Description**: Correlation matrix between returns of different strategies.
- **Interpretation**: 
  - Low or negative correlations between strategies (< 0.3) indicate good diversification potential.
  - High correlations (> 0.7) suggest redundancy and limited diversification benefit.
  - Values around 0 indicate no linear relationship.

## Interpreting Performance Reports

The system generates a structured performance report with sections for overall summary, risk metrics, risk-adjusted metrics, trade metrics, and drawdowns. When evaluating strategies, consider:

1. **Risk-Return Balance**: Higher returns with lower risk are ideal, but there's usually a tradeoff.

2. **Consistency**: Check rolling metrics to ensure performance is relatively consistent.

3. **Worst-Case Scenarios**: Pay special attention to maximum drawdown and recovery periods.

4. **Strategy Correlations**: For multi-strategy portfolios, low correlations between strategies are desirable.

5. **Transaction Costs**: Make sure trading costs don't significantly erode returns.

### Example Analysis

```python
# Example of analyzing a performance report
if metrics['sharpe_ratio'] > 1.0 and metrics['max_drawdown_pct'] > -20.0:
    print("Strategy has acceptable risk-adjusted returns and manageable drawdowns")
else:
    print("Strategy either has poor risk-adjusted returns or excessive drawdowns")
    
if metrics['win_rate_pct'] < 50 and metrics['win_loss_ratio'] < 2.0:
    print("Strategy has both a low win rate and insufficient average profit per trade")
```

By examining these metrics collectively, you can make better-informed decisions about strategy selection, optimization, and risk management. 