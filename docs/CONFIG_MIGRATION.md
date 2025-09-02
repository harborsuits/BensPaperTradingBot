# BensBot Configuration Migration Guide

This document outlines how to migrate from legacy BensBot configuration formats to the new unified YAML-based configuration system. The new system consolidates multiple configuration files into a single YAML file, making setup easier and more maintainable.

## Table of Contents
- [Overview of Configuration Changes](#overview-of-configuration-changes)
- [Key Differences](#key-differences)
- [Field Mapping Reference](#field-mapping-reference)
- [Using the Configuration Converter](#using-the-configuration-converter)
- [Environment Variables](#environment-variables)
- [Manual Migration Steps](#manual-migration-steps)
- [Example Configurations](#example-configurations)
- [Troubleshooting](#troubleshooting)

## Overview of Configuration Changes

BensBot has transitioned from using multiple configuration files (often in JSON or INI format) to a single consolidated YAML configuration file. This change:

- Simplifies setup with a single entry point
- Provides better validation through schema checking
- Supports hierarchical organization of settings
- Allows for environment variable integration using Pydantic's settings capabilities
- Ensures consistent configuration across development and deployment environments

## Key Differences

| Legacy Approach | New Approach |
|-----------------|--------------|
| Multiple config files (broker_config.json, strategy_config.json, etc.) | Single unified config.yaml file |
| Mixed formats (JSON, Python files, INI, etc.) | YAML format for better readability and hierarchical data |
| Hard-coded defaults in code | Defaults defined in schema with clear documentation |
| Direct environment variable access in code | Structured environment variable mapping |
| Manual validation | Automated schema validation with detailed error messages |
| Limited typing | Strong typing with Pydantic |

## Field Mapping Reference

Below is a reference mapping of legacy configuration fields to their new locations in the YAML structure.

### Account & Risk Settings

| Old Field | New Location | Notes |
|-----------|--------------|-------|
| `initial_balance` | `account.initial_balance` | Account starting balance |
| `risk_per_trade` | `risk.max_risk_per_trade` | Maximum risk percentage per trade |
| `max_trades` | `risk.max_concurrent_trades` | Maximum concurrent open trades |
| `trailing_stop` | `risk.use_trailing_stop` | Whether to use trailing stops |
| `trailing_stop_pct` | `risk.trailing_stop_percent` | Trailing stop percentage |
| `take_profit_pct` | `risk.take_profit_percent` | Take profit percentage |
| `paper_trading` | `account.paper_trading` | Whether to use paper trading |

### Broker Settings

| Old Field | New Location | Notes |
|-----------|--------------|-------|
| `api_key` | `brokers.credentials.<broker>.api_key` | broker-specific API key |
| `api_secret` | `brokers.credentials.<broker>.api_secret` | Broker-specific API secret |
| `account_number` | `brokers.credentials.<broker>.account_number` | Broker account number |
| `broker_api_url` | `brokers.settings.<broker>.api_url` | Broker API URL |
| `tradier_account_id` | `brokers.credentials.tradier.account_id` | Tradier account ID |
| `tradier_token` | `brokers.credentials.tradier.token` | Tradier API token |
| `alpaca_key_id` | `brokers.credentials.alpaca.api_key` | Alpaca API key |
| `alpaca_secret_key` | `brokers.credentials.alpaca.api_secret` | Alpaca API secret |
| `alpaca_paper` | `brokers.settings.alpaca.paper` | Use Alpaca paper trading |

### Strategy Settings

| Old Field | New Location | Notes |
|-----------|--------------|-------|
| `strategy_name` | `strategy.name` | Strategy name |
| `strategy_class` | `strategy.class` | Fully qualified strategy class name |
| `strategy_params` | `strategy.parameters` | Strategy-specific parameters |
| `symbols` | `data.symbols` | Trading symbols list |
| `timeframe` | `data.timeframe` | Data timeframe |

### Backtest Settings

| Old Field | New Location | Notes |
|-----------|--------------|-------|
| `start_date` | `backtest.start_date` | Backtest start date |
| `end_date` | `backtest.end_date` | Backtest end date |
| `data_source` | `data.source` | Data source for backtesting |
| `data_file` | `data.file_path` | Path to historical data file |
| `commission` | `backtest.commission_rate` | Commission rate for backtesting |
| `slippage` | `backtest.slippage_rate` | Slippage rate for backtesting |

### Logging Settings

| Old Field | New Location | Notes |
|-----------|--------------|-------|
| `log_level` | `logging.level` | Logging level (INFO, DEBUG, etc.) |
| `log_file` | `logging.file_path` | Log file path |

## Using the Configuration Converter

We've provided a configuration conversion utility to help migrate your existing configurations to the new format. The tool can be found at `utils/convert_config.py`.

### Basic Usage

```bash
# Convert a single config file
python utils/convert_config.py --input old_config.json --output new_config.yaml

# Convert a directory of config files
python utils/convert_config.py --input config_dir/ --output new_config.yaml

# Include environment variables in the conversion
python utils/convert_config.py --input old_config.json --include-env --output new_config.yaml

# Force overwrite of existing output file
python utils/convert_config.py --input old_config.json --output new_config.yaml --force
```

### Tips for Using the Converter

1. Always review the generated YAML file to ensure all settings were correctly migrated
2. The converter will provide warnings about missing required sections or potential issues
3. For complex configurations, use the `--include-env` flag to incorporate environment variables
4. Keep your old configuration files until you've verified the new one works correctly

## Environment Variables

The new configuration system supports environment variables integration via Pydantic's BaseSettings. This means you can still use environment variables for sensitive information like API keys.

### Supported Environment Variables

| Environment Variable | Config Path | Notes |
|---------------------|-------------|-------|
| `TRADIER_TOKEN` | `brokers.credentials.tradier.token` | Tradier API token |
| `TRADIER_ACCOUNT_ID` | `brokers.credentials.tradier.account_id` | Tradier account ID |
| `ALPACA_API_KEY` | `brokers.credentials.alpaca.api_key` | Alpaca API key |
| `ALPACA_SECRET_KEY` | `brokers.credentials.alpaca.api_secret` | Alpaca API secret |
| `ETRADE_CONSUMER_KEY` | `brokers.credentials.etrade.consumer_key` | E*Trade consumer key |
| `ETRADE_CONSUMER_SECRET` | `brokers.credentials.etrade.consumer_secret` | E*Trade consumer secret |
| `API_KEY` | `brokers.credentials.api_key` | Generic API key |
| `API_SECRET` | `brokers.credentials.api_secret` | Generic API secret |
| `MONGODB_URI` | `database.mongodb_uri` | MongoDB connection URI |
| `REDIS_URI` | `database.redis_uri` | Redis connection URI |
| `JWT_SECRET` | `security.jwt_secret` | JWT encryption secret |
| `DASHBOARD_ADMIN_PASSWORD` | `dashboard.admin_password` | Dashboard admin password |
| `LOG_LEVEL` | `logging.level` | Logging level |

### Using Environment Variables with Docker

If you're using Docker, you can pass environment variables to the container:

```yaml
# Example in docker-compose.yml
services:
  trading-bot:
    image: bensbot:latest
    env_file: .env
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/bensbot
      - REDIS_URI=redis://redis:6379/0
```

## Manual Migration Steps

If you prefer to manually migrate your configuration, follow these steps:

1. Create a new `config.yaml` file following the new structure
2. Refer to the [Field Mapping Reference](#field-mapping-reference) to move settings to their new locations
3. Check for any broker-specific settings that need to be reorganized
4. Update strategy parameters to follow the new hierarchical structure
5. Move logging configuration to the `logging` section
6. Validate your config with the new schema validation tool:
   ```bash
   python -m trading_bot.config.schema_validator --config your_new_config.yaml
   ```

## Example Configurations

### Example: Simple Configuration

```yaml
# config.yaml
metadata:
  version: "1.0.0"
  description: "Basic BensBot configuration"

account:
  initial_balance: 10000.0
  paper_trading: true

brokers:
  enabled:
    - alpaca
  credentials:
    alpaca:
      api_key: ${ALPACA_API_KEY}
      api_secret: ${ALPACA_SECRET_KEY}
  settings:
    alpaca:
      paper: true

strategy:
  name: "momentum_breakout"
  class: "trading_bot.strategies.momentum.BreakoutStrategy"
  parameters:
    lookback_period: 20
    volatility_factor: 1.5
    profit_target_pct: 5.0
    stop_loss_pct: 2.0

data:
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN"]
  timeframe: "1h"
  source: "alpaca"

risk:
  max_risk_per_trade: 1.0
  max_concurrent_trades: 5
  max_drawdown_pct: 15.0
  use_trailing_stop: true
  trailing_stop_percent: 1.5

logging:
  level: "INFO"
  file_path: "logs/trading.log"
  console: true
```

### Example: Multi-Broker Configuration

```yaml
# config.yaml
metadata:
  version: "1.0.0"
  description: "Multi-broker configuration"

account:
  initial_balance: 10000.0
  paper_trading: false

brokers:
  enabled:
    - alpaca
    - tradier
  primary: "alpaca"
  failover: true
  asset_routing:
    equities: "alpaca"
    options: "tradier"
    default: "alpaca"
  credentials:
    alpaca:
      api_key: ${ALPACA_API_KEY}
      api_secret: ${ALPACA_SECRET_KEY}
    tradier:
      token: ${TRADIER_TOKEN}
      account_id: ${TRADIER_ACCOUNT_ID}
  settings:
    alpaca:
      paper: false
      api_url: "https://api.alpaca.markets"
    tradier:
      sandbox: false

# ... other sections as in the simple example
```

## Troubleshooting

### Common Issues

#### Missing Required Sections

If you're seeing errors about missing required sections, ensure your configuration includes all the necessary top-level sections (brokers, strategy, risk, logging, etc.).

#### Credential Errors

Errors related to credentials might mean:
- Environment variables aren't set correctly
- Credential format has changed in the new structure
- A broker is enabled but doesn't have credentials configured

#### Strategy Import Errors

If you see strategy import errors:
- Ensure the strategy class path is correct
- Check that the strategy parameters match the expected format
- Verify the strategy is compatible with the new BensBot architecture

### Getting Help

If you encounter issues with the migration process:
1. Check the logs for detailed error messages
2. Consult the [Field Mapping Reference](#field-mapping-reference)
3. Try running the configuration validator for detailed feedback:
   ```bash
   python -m trading_bot.config.schema_validator --config your_config.yaml
   ```
4. Refer to the example configurations as templates
