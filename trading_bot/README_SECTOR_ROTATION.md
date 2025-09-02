# Sector Rotation Framework Integration Guide

This guide explains how to use the sector rotation framework feature that has been integrated into the trading bot's macro guidance system.

## Overview

The sector rotation framework allows the trading bot to make intelligent decisions about sector allocation and trading strategies based on the current economic cycle phase. The framework provides detailed guidance for:

- Primary and secondary sectors to focus on in each economic cycle phase
- Sectors to avoid
- Recommended equity and options strategies for each phase
- Implementation guidance and risk management parameters
- Historical performance data for reference

## Economic Cycle Phases

The framework supports the following economic cycle phases:

1. **Early Expansion** - Initial phase of economic growth following a recession
2. **Mid Cycle** - Period of stable economic growth with moderate inflation
3. **Late Cycle** - Final phase of economic expansion with rising inflation pressures
4. **Recession** - Period of economic contraction
5. **Recovery** - Initial phase of economic improvement following a recession

## Using the API Endpoints

The framework provides several API endpoints for interacting with the sector rotation system:

### Get Current Economic Cycle

```
GET /macro-guidance/current-economic-cycle
```

Returns the current economic cycle phase determination with confidence scores and supporting indicators.

### Get Sector Rotation Guidance

```
GET /macro-guidance/sector-rotation?cycle_phase=early_expansion&ticker=AAPL
```

Parameters:
- `cycle_phase` (optional): Specific cycle phase to get guidance for
- `ticker` (optional): Ticker symbol to get specific guidance for

Returns sector rotation guidance for the current or specified economic cycle phase, including favored sectors, strategies, and implementation guidance.

### Update Sector Rotation Framework

```
POST /macro-guidance/sector-rotation-update
```

Updates the sector rotation framework with new data. The request body should contain the complete framework data as a JSON object.

## Using the Update Script

You can update the sector rotation framework using the provided `update_sector_rotation.py` script:

```bash
python trading_bot/update_sector_rotation.py --file path/to/framework.json
```

By default, the script will send the data to `http://localhost:5000/macro-guidance/sector-rotation-update`. You can specify a different endpoint with the `--url` parameter.

## Framework Integration

The sector rotation framework is integrated into the trading bot's decision-making process in several ways:

1. **Position Sizing Adjustments**: Positions in favored sectors receive larger allocations, while positions in sectors to avoid are reduced.

2. **Strategy Selection**: The trading bot can recommend specific equity and options strategies based on the current economic cycle and sector classification.

3. **Risk Management**: Stop-loss and take-profit levels are adjusted based on sector rotation guidance.

4. **Macro-Enhanced Trading Decisions**: All trading decisions are enhanced with sector rotation insights.

## Configuration

The sector rotation framework is configured in `config.yaml` under the `macro_guidance` section:

```yaml
macro_guidance:
  enabled: true
  sector_rotation_path: "configs/sector_rotation_framework.json"
  # Other settings...
```

The `sector_rotation_path` setting specifies the path to the sector rotation framework data file.

## Framework Data Structure

The sector rotation framework data is structured as a JSON object with the following top-level keys:

- `framework_version`: Version of the framework
- `last_updated`: Date when the framework was last updated
- `early_expansion`: Guidance for the early expansion phase
- `mid_cycle`: Guidance for the mid-cycle phase
- `late_cycle`: Guidance for the late-cycle phase
- `recession`: Guidance for the recession phase
- `recovery`: Guidance for the recovery phase
- Additional sections like `advanced_identification_framework` for more sophisticated analysis

Each cycle phase contains detailed guidance on sectors, strategies, and implementation.

## Example Use Cases

1. **Sector Allocation Decisions**: Determine which sectors to overweight, underweight, or avoid based on the current economic cycle.

2. **Strategy Selection**: Choose appropriate trading strategies based on sector rotation guidance.

3. **Position Sizing**: Adjust position sizes based on sector classification and economic cycle.

4. **Risk Management**: Customize stop-loss and take-profit levels based on sector rotation insights.

## Troubleshooting

If you encounter issues with the sector rotation framework:

1. Check the logs for error messages
2. Verify that the framework data file exists and is correctly formatted
3. Ensure the macro guidance module is enabled in the configuration
4. Check if the economic cycle determination is accurate

## Further Development

The sector rotation framework can be extended with:

1. Custom cycle phase definitions
2. Additional strategy templates
3. Integration with external economic data sources
4. Machine learning for more accurate cycle identification 