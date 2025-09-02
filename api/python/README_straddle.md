# Straddle Trading Strategy

## Overview
The Straddle Trading Strategy is designed to capitalize on significant price movements around high-volatility events, regardless of direction. It involves simultaneously purchasing both a call and a put option at the same strike price, creating a position that profits when the underlying asset makes a large move in either direction.

## Key Components

### 1. Market Universe & Event Selection
- **Trading Universe**: Liquid, high IV stocks and ETFs (SPY, QQQ, AAPL, etc.)
- **Catalyst Events**: Earnings reports, FDA announcements, product launches, economic data releases
- **Event Classification**: Events are categorized by importance and historical volatility impact

### 2. Option & Strike Selection
- **ATM Strategy**: Primarily uses at-the-money (ATM) strikes for balanced delta exposure
- **Expiration Selection**: Typically targets 5-10 days after the catalyst event
- **IV Evaluation**: Prioritizes events with relatively low IV compared to historical event impact

### 3. Greeks & Risk Management
- **Vega-Focused**: Primary profit driver is volatility expansion (positive vega)
- **Delta-Neutral**: Near-zero delta at initiation for direction neutrality
- **Theta Management**: Manages time decay risk by precise event timing

### 4. Entry Criteria
- **IV Rank/Percentile**: Preferably enters when IV is below historical average for similar events
- **Timing**: Typically 3-8 days before the catalyst event
- **Liquidity Requirements**: Minimum option chain liquidity metrics are enforced

### 5. Position Sizing & Risk Controls
- **Account Risk**: Limited to 1-3% of portfolio per trade
- **Correlation Risk**: Manages exposure across correlated assets
- **Max Positions**: Limits concurrent positions based on market conditions

### 6. Exit Rules
- **Profit Taking**: Targets 40-65% of the straddle cost as profit
- **Pre-Event Exit**: May exit 1 day prior to event if IV expansion has already occurred
- **Post-Event Management**: Strategies for managing winning/losing positions after event

### 7. Order Execution
- **Leg Entry**: Order types and execution approach for entering both legs
- **Mid-Price Targets**: Aims for execution between bid/ask to manage costs
- **Cancellation Parameters**: Rules for abandoning trades with poor fills

## Implementation in Trading Bot

The strategy is implemented in three main components:

1. **Strategy Class** (`trading_bot/strategies/options_spreads/straddle_trading.py`): Contains the core logic and signal generation
2. **Configuration** (`trading_bot/config/straddle_trading_config.py`): Defines parameters and settings
3. **Utilities** (`trading_bot/utils/straddle_utils.py`): Specialized functions for straddle analysis and pricing

## Key Features

- **Event-Driven**: Focus on high-impact market events
- **IV Analysis**: Sophisticated volatility modeling
- **Directionally Neutral**: Profits from magnitude, not direction of movement
- **Risk-Controlled**: Multi-layered risk management approach

## Optimization Parameters

The strategy can be optimized along several dimensions:

- Event type selection
- Days before/after event
- Strike selection
- Position sizing
- IV rank thresholds
- Profit taking levels

## Performance Metrics

Performance is tracked using:

- Win/loss ratio
- Average P&L per trade
- P&L by event type
- Vega and theta contribution
- Maximum drawdown
- Sharpe and Sortino ratios

## Advanced Usage

Advanced traders can utilize additional features:

- IV skew analysis for strike selection
- Delta-weighted position adjustments
- Gamma scalping opportunities
- Backtesting against historical events
- Custom event importance scoring 