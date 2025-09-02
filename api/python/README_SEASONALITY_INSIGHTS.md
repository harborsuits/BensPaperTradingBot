# Seasonality Insights Framework Integration Guide

This guide explains how to use the seasonality insights framework feature that has been integrated into the trading bot's macro guidance system.

## Overview

The seasonality insights framework allows the trading bot to make intelligent decisions based on historical seasonal patterns and recurring market events. The framework provides detailed guidance for:

- Monthly seasonal patterns with expected biases and trading strategies
- Recurring market patterns (Fed meetings, options expirations, etc.)
- Sector-specific seasonal performance data
- Recommended equity and options strategies for different time periods
- Composite seasonality scores to quantify current seasonal strength

## Key Seasonal Patterns

The framework captures two main types of seasonal patterns:

1. **Monthly Patterns** - Calendar-based seasonality that repeats annually
   - January Effect, Summer Rally, September Weakness, Santa Rally, etc.
   - Each month has specific sectors that tend to outperform/underperform
   - Historical reliability metrics for each pattern

2. **Recurring Patterns** - Event-based patterns that occur regularly
   - Quad Witching Weeks (4 times per year)
   - Options Expiration Friday (monthly)
   - Fed Meeting Weeks (8 times per year)
   - End-of-Month Flows (monthly)
   - Jobs Report Friday (monthly)
   - CPI Release Week (monthly)

## Using the API Endpoints

The framework provides several API endpoints for interacting with the seasonality insights system:

### Get Seasonality Guidance

```
GET /macro-guidance/seasonality?month=January&ticker=AAPL
```

Parameters:
- `month` (optional): Specific month to get guidance for (defaults to current month)
- `ticker` (optional): Ticker symbol to get specific guidance for

Returns seasonality guidance for the current or specified month, including trading strategies and ticker-specific recommendations.

### Get Active Seasonality Patterns

```
GET /macro-guidance/active-patterns
```

Returns currently active seasonal patterns based on the current date, including the monthly pattern and any active recurring patterns.

### Update Seasonality Framework

```
POST /macro-guidance/seasonality-update
```

Updates the seasonality insights framework with new data. The request body should contain the complete framework data as a JSON object with a top-level `seasonality_insights` key.

### Get Seasonality-Enhanced Recommendation

```
GET /macro-guidance/seasonality-enhanced-recommendation?ticker=AAPL
```

Parameters:
- `ticker` (required): Ticker symbol to get recommendation for
- `account_value` (optional): Account value for position sizing
- Various `market_*` parameters for current market conditions

Returns a trading recommendation enhanced with seasonality insights, integrated with other macro guidance factors.

## Using the Update Script

You can update the seasonality insights framework using the provided `update_seasonality_insights.py` script:

```bash
python trading_bot/update_seasonality_insights.py --file path/to/framework.json
```

By default, the script will send the data to `http://localhost:5000/macro-guidance/seasonality-update`. You can specify a different endpoint with the `--url` parameter.

## Framework Integration

The seasonality insights framework is integrated into the trading bot's decision-making process in several ways:

1. **Position Sizing Adjustments**: Positions receive larger or smaller allocations based on seasonal biases and sector seasonal performance.

2. **Strategy Selection**: The trading bot recommends specific equity and options strategies based on the current seasonal patterns.

3. **Risk Management**: Stop-loss and take-profit levels are adjusted based on expected seasonal volatility.

4. **Macro-Enhanced Trading Decisions**: All trading decisions are enhanced with seasonality insights alongside other macro factors.

## Configuration

The seasonality insights framework is configured in `config.yaml` under the `macro_guidance` section:

```yaml
macro_guidance:
  enabled: true
  seasonality_insights_path: "configs/seasonality_insights_framework.json"
  # Other settings...
  seasonality_adjustments:
    apply_monthly_bias: true
    apply_recurring_patterns: true
    composite_score_thresholds:
      strong_bullish: 80
      bullish: 60
      neutral: 40
      bearish: 20
    position_sizing_adjustments:
      strong_bullish: 1.2
      bullish: 1.1
      neutral: 1.0
      bearish: 0.9
      strong_bearish: 0.8
```

The `seasonality_insights_path` setting specifies the path to the seasonality insights framework data file, while the `seasonality_adjustments` section controls how seasonality factors influence trading decisions.

## Framework Data Structure

The seasonality insights framework data is structured as a JSON object with the following top-level structure:

```json
{
  "seasonality_insights": {
    "framework_version": "3.0.0",
    "last_updated": "2025-04-11",
    "monthly_patterns": [
      {
        "month": "January",
        "name": "January Effect",
        "primary_asset_classes": [...],
        "trading_strategies": {...},
        ...
      },
      ...
    ],
    "recurring_patterns": [
      {
        "pattern": "Quad Witching Weeks",
        "frequency": "4 times per year",
        "primary_asset_classes": [...],
        "trading_strategies": {...},
        ...
      },
      ...
    ],
    "seasonality_framework": {...},
    "meta_data": {...}
  }
}
```

Each monthly and recurring pattern contains detailed information about asset class performance, specific patterns, trading strategies, and implementation guidance.

## Example Use Cases

1. **Seasonal Allocation Decisions**: Determine which sectors to overweight based on the current month's seasonal patterns.

2. **Event-Based Trading**: Implement specific strategies around recurring events like Fed meetings or options expiration.

3. **Position Sizing**: Adjust position sizes based on seasonal bias strength.

4. **Strategy Timing**: Optimize entry and exit timing based on seasonal patterns.

5. **Risk Management**: Adjust risk parameters based on expected seasonal volatility.

## Troubleshooting

If you encounter issues with the seasonality insights framework:

1. Check the logs for error messages
2. Verify that the framework data file exists and is correctly formatted
3. Ensure the macro guidance module is enabled in the configuration
4. Check if the seasonality data path is correct in the configuration

## Further Development

The seasonality insights framework can be extended with:

1. More granular timeframe analysis (weekly/daily patterns)
2. Integration with specific company earnings seasonality
3. Custom pattern detection for specific market regimes
4. Machine learning for more accurate seasonal pattern detection 