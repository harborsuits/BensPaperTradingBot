# Strategy Organization

This document outlines the organization and classification of trading strategies within the system.

## Strategy Classification System

Each strategy is classified along multiple dimensions:

### 1. Asset Class
- **Forex**: Currency pair trading strategies
- **Stocks**: Equities trading strategies
- **Options**: Options trading strategies
- **Crypto**: Cryptocurrency trading strategies
- **Multi-Asset**: Strategies that trade across multiple asset classes

### 2. Strategy Type
- **Trend Following**: Strategies that follow established price trends
- **Mean Reversion**: Strategies that look for price reversals to the mean
- **Breakout**: Strategies that trade breakouts from established ranges or patterns
- **Range/Oscillation**: Strategies that trade within established price ranges
- **Momentum**: Strategies that follow momentum indicators
- **Carry**: Strategies that exploit interest rate differentials
- **Arbitrage**: Strategies that exploit price discrepancies
- **Machine Learning**: Strategies using ML/AI techniques

### 3. Time Frame
- **Scalping**: Very short-term (seconds to minutes)
- **Day Trading**: Intraday (minutes to hours) 
- **Swing Trading**: Short to medium-term (days to weeks)
- **Position Trading**: Long-term (weeks to months)

### 4. Market Regime Compatibility
- **Trending Markets**: Works best in clearly trending markets
- **Ranging Markets**: Works best in sideways/ranging markets
- **Volatile Markets**: Works best in high-volatility environments
- **Low Volatility**: Works best in low-volatility environments
- **All Weather**: Functions across different market regimes

## Directory Structure

The `strategies` directory is organized hierarchically:

1. First level: Asset class (forex, stocks, options, crypto)
2. Second level: Strategy type (trend, range, breakout, etc.)
3. Within each strategy type: Individual strategy implementations

## Naming Conventions

Strategies follow a consistent naming pattern:

```
[AssetClass][StrategyType][Timeframe]Strategy.py
```

Examples:
- `ForexTrendFollowingStrategy.py`
- `StockMeanReversionDailyStrategy.py`
- `CryptoBreakoutScalpingStrategy.py`

## Strategy Registration

All strategies must be registered in the appropriate `__init__.py` file to be discoverable by the strategy factory.

Example:
```python
from .trend_following_strategy import ForexTrendFollowingStrategy

__all__ = ['ForexTrendFollowingStrategy']
```

## Strategy Implementation Pattern

All strategies must:

1. Inherit from the appropriate base class (e.g., `ForexBaseStrategy`, `StockBaseStrategy`)
2. Implement required methods (get_signals, calculate_indicators, etc.)
3. Include metadata that describes the strategy characteristics
4. Follow the standard code structure pattern

## Strategy Factory

The `StrategyFactory` is responsible for creating strategy instances based on configuration and market conditions. It uses the registry to discover available strategies.
