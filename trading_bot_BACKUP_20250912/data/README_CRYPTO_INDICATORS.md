# Cryptocurrency Technical Indicators Suite

A specialized suite of technical indicators optimized for cryptocurrency markets, which can also be applied to forex and futures trading.

## Features

- Volume-based indicators (OBV, MFI, Chaikin Money Flow)
- Momentum indicators with optimized parameters for crypto volatility
- Volatility indicators (ATR, Bollinger Bands, Historical Volatility)
- Crypto-specific indicators (Heikin-Ashi, Trading Range, Volatility Regime)
- Trend indicators (Moving Averages, SuperTrend)
- Funding rate analysis for perpetual futures
- Built-in signal generation strategies

## Quick Start

```python
from trading_bot.data.crypto_indicators import CryptoIndicatorSuite, IndicatorConfig

# Initialize with custom config
config = {
    'default_length': 14,
    'fast_rsi_window': 8,  # More responsive for crypto
    'donchian_window': 20,
    'bollinger_dev': 2.5,  # Wider bands for crypto volatility
}
indicator_suite = CryptoIndicatorSuite(config)

# Calculate indicators on a DataFrame with OHLCV data
df_with_indicators = indicator_suite.add_all_indicators(df)

# Generate trading signals
df_with_signals = indicator_suite.generate_signals(df_with_indicators, strategy='trend_following')
```

## Available Strategies

- `trend_following`: Uses MACD crossovers and SuperTrend direction
- `mean_reversion`: Uses RSI oversold/overbought conditions and Bollinger Band extremes
- `volatility_breakout`: Identifies breakouts after periods of low volatility
- `combined`: Prioritized combination of all three strategies

## Funding Rate Analysis for Futures

For perpetual futures, you can add funding rate indicators:

```python
# Assuming you have a list of funding_data dictionaries with 'timestamp' and 'rate' keys
df_with_funding = indicator_suite.add_funding_indicators(df, funding_data)
```

## Extending to Other Assets

While optimized for cryptocurrencies, this suite works well for other volatile markets:

### Forex Adaptation
- Adjust timeframes (typically use H1, H4, D1 for forex)
- Consider lower volatility settings (e.g., `bollinger_dev: 2.0`)
- Pay attention to session-specific behavior

### Futures Adaptation
- Include volume profile analysis for price levels
- Adjust for contract multipliers and specifications
- Consider adding term structure indicators

## Metadata and Documentation

Each indicator includes detailed metadata:

```python
# Get metadata for a specific indicator
rsi_metadata = indicator_suite.get_indicator_metadata('rsi')

# Get all available metadata
all_metadata = indicator_suite.get_indicator_metadata()
```

## Example

See `trading_bot/examples/crypto_indicators_example.py` for a complete usage example. 