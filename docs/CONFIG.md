# BensBot Configuration Guide

This document explains how to configure BensBot using the configuration system.

## Configuration Files

BensBot uses the following configuration files:

- `config/system_config.json` - Main system configuration
- `config/market_regime_config.json` - Market regime detection configuration
- `config/market_data_config.json` - Market data sources and settings

## Schema Validation

All configuration files are validated against JSON Schema definitions. This ensures that:

- Required fields are present
- Values are of the correct type
- Values meet specified constraints (min/max values, allowed patterns, etc.)
- Configuration is consistent and complete

## Environment Variable Overrides

Any configuration value can be overridden using environment variables. This is useful for:

- Development vs. production settings
- Sensitive values that shouldn't be committed to git
- Containerized deployments
- CI/CD pipelines
- Quick testing without modifying config files

### How to Use Environment Variables

BensBot uses the prefix `BENBOT_` for all environment variables. The variable name corresponds to the configuration path with dots replaced by underscores.

#### Examples:

```bash
# Simple values
export BENBOT_INITIAL_CAPITAL=20000
export BENBOT_RISK_PER_TRADE=0.05
export BENBOT_LOG_LEVEL=DEBUG

# Nested values
export BENBOT_TRADING_HOURS_START=10:00
export BENBOT_TRADING_HOURS_END=16:30
export BENBOT_SYSTEM_SAFEGUARDS_CIRCUIT_BREAKERS_MAX_DRAWDOWN_PERCENT=15

# Lists (comma-separated)
export BENBOT_WATCHED_SYMBOLS=SPY,QQQ,AAPL,MSFT
```

### Type Conversion

Environment variables are converted to the appropriate type based on the config schema:

- **Booleans**: `true`, `yes`, `1`, `y` (case insensitive) are converted to `true`
- **Numbers**: Converted to integers or floats as appropriate
- **Lists**: Comma-separated values are converted to lists
- **Strings**: Used as-is

### Validation

Environment variable overrides are validated just like regular config values. If an invalid value is provided, BensBot will fail to start and display a clear error message.

## Validating Configuration

You can validate your configuration without starting BensBot using the validation tool:

```bash
python3 config/validate_config.py
```

This will:

1. Check if your configuration file is valid JSON
2. Validate it against the schema
3. Check for any environment variable overrides
4. Verify that referenced files exist

Use the `--show-env` flag to display information about environment variable usage:

```bash
python3 config/validate_config.py --show-env
```

## Hot Reload

BensBot supports hot reloading of configuration files. This means you can change the configuration while BensBot is running, and the changes will be applied without requiring a restart.

To enable hot reloading, set the following in your configuration:

```json
{
  "enable_config_hot_reload": true,
  "config_reload_interval_seconds": 60
}
```

With hot reloading enabled, BensBot will check for changes to the configuration files at the specified interval. If changes are detected, the new configuration will be validated and applied.

### Limitations

Some configuration changes may require a restart to take effect:
- Changes to broker connection settings
- Changes that affect already-running strategies
- Changes to system-level settings like logging configuration

## Common Configuration Examples

### Paper Trading Mode

```bash
export BENBOT_ENABLE_PAPER_TRADING=true
export BENBOT_INITIAL_CAPITAL=10000
```

### Aggressive Risk Profile

```bash
export BENBOT_RISK_PER_TRADE=0.05
export BENBOT_MAX_OPEN_POSITIONS=10
```

### Conservative Risk Profile

```bash
export BENBOT_RISK_PER_TRADE=0.01
export BENBOT_SYSTEM_SAFEGUARDS_CIRCUIT_BREAKERS_MAX_DRAWDOWN_PERCENT=5
export BENBOT_SYSTEM_SAFEGUARDS_CIRCUIT_BREAKERS_MAX_DAILY_LOSS_PERCENT=1
```

### Debug Mode

```bash
export BENBOT_LOG_LEVEL=DEBUG
export BENBOT_ENABLE_PERFORMANCE_LOGGING=true
```
