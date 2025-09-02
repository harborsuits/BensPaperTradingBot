# BensBot Typed Settings Configuration

This document describes the typed settings system used to configure all aspects of the BensBot trading system.

## Overview

The trading bot uses a Pydantic-based typed settings system that provides:

- **Type validation** - prevents configuration errors that could lead to runtime failures
- **Default values** - sensible defaults reduce configuration burden
- **Environment variable integration** - sensitive data can be set via environment variables
- **Documentation** - field descriptions and validation rules are part of the code

## Configuration Sources (Priority Order)

1. **Environment variables** - Highest priority, override file settings
2. **YAML/JSON config files** - Main configuration source
3. **Default values** - Used when neither of the above are provided

## File-based Configuration

Configuration can be specified in YAML or JSON format. YAML is recommended for human readability.

### Basic Example (config.yaml)

```yaml
broker: 
  name: "tradier"
  api_key: "YOUR_API_KEY"  # Consider using env vars instead
  account_id: "YOUR_ACCOUNT_ID"
  sandbox: true

risk:
  max_position_pct: 0.05
  max_risk_pct: 0.01
  max_portfolio_risk: 0.20
  max_open_trades: 5

backtest:
  default_symbols: ["AAPL", "MSFT", "GOOG"]
  initial_capital: 100000.0
  data_source: "alpha_vantage"

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
```

## API Keys Configuration

BensBot supports multiple API integrations for market data, news, and AI capabilities:

### Market Data APIs

```yaml
data:
  provider: "tradier"  # Primary data source
  historical_source: "alpha_vantage"
  api_keys:
    alpha_vantage: "YOUR_ALPHA_VANTAGE_KEY"
    finnhub: "YOUR_FINNHUB_KEY"
    alpaca: "YOUR_ALPACA_KEY"
```

### News APIs

```yaml
api:
  api_keys:
    marketaux: "YOUR_MARKETAUX_KEY"
    newsdata: "YOUR_NEWSDATA_KEY"
    gnews: "YOUR_GNEWS_KEY"
    mediastack: "YOUR_MEDIASTACK_KEY"
    currents: "YOUR_CURRENTS_KEY"
    nytimes: "YOUR_NYTIMES_KEY"
```

### AI Model APIs

```yaml
api:
  api_keys:
    huggingface: "YOUR_HUGGINGFACE_KEY"
    openai_primary: "YOUR_OPENAI_KEY"
    openai_secondary: "YOUR_BACKUP_OPENAI_KEY"
    claude: "YOUR_ANTHROPIC_KEY"
    mistral: "YOUR_MISTRAL_KEY"
    cohere: "YOUR_COHERE_KEY"
    gemini: "YOUR_GEMINI_KEY"
```

### Notification Configuration

```yaml
notifications:
  enable_notifications: true
  telegram_token: "YOUR_TELEGRAM_BOT_TOKEN"
  telegram_chat_id: "YOUR_TELEGRAM_CHAT_ID"
  slack_webhook: "YOUR_SLACK_WEBHOOK_URL"
  notification_levels: ["critical", "error", "warning", "info"]
```

## Environment Variables

Environment variables can be used to override configuration from files. This is particularly useful for:

1. **API keys** and sensitive data
2. **Environment-specific settings** (development/testing/production)
3. **Server-specific settings** (port, host, etc.)

### Common Environment Variables

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| Broker API Key | `TRADIER_API_KEY` | Tradier API Key |
| Broker Account ID | `TRADIER_ACCOUNT_ID` | Tradier Account ID |
| Broker Sandbox Mode | `TRADIER_SANDBOX` | "true" or "false" |
| Max Risk Per Trade | `MAX_RISK_PCT` | Float (0-1) |
| Max Open Trades | `MAX_OPEN_TRADES` | Integer |
| API Port | `API_PORT` | Integer (1024-65535) |
| Telegram Bot Token | `TELEGRAM_TOKEN` | Telegram Bot Token |
| Telegram Chat ID | `TELEGRAM_CHAT_ID` | Telegram Chat ID |
| Alpha Vantage API Key | `ALPHA_VANTAGE_KEY` | Alpha Vantage API Key |
| OpenAI API Key | `OPENAI_API_KEY` | OpenAI API Key |

## Validation Rules

The typed settings system enforces validation rules to prevent configuration errors:

### Broker Settings
- `api_key`: Required, cannot be empty
- `name`: Must be one of the supported brokers ("tradier", "alpaca")

### Risk Settings
- `max_position_pct`: Float between 0 and 1
- `max_risk_pct`: Float between 0 and 1
- `max_portfolio_risk`: Float between 0 and 1
- `max_sector_allocation`: Float between 0 and 1
- `max_open_trades`: Positive integer

### API Settings
- `port`: Integer between 1024 and 65535
- `rate_limit_requests`: Positive integer
- `rate_limit_period_seconds`: Positive integer

### Backtest Settings
- `default_start_date`, `default_end_date`: Must be in "YYYY-MM-DD" format
- `initial_capital`: Positive float
- `slippage_pct`: Float between 0 and 1

## Loading Configuration in Code

```python
from trading_bot.config.typed_settings import load_config, TradingBotSettings

# Load from default locations (environment variables + config files)
settings = load_config()

# Or specify a config file path
settings = load_config("/path/to/config.yaml")

# Access settings in a type-safe way
api_key = settings.broker.api_key
max_risk = settings.risk.max_risk_pct

# Validate settings programmatically
if settings.risk.max_position_pct > 0.1:
    print("Warning: Max position percentage is high!")
```

## Advanced: Combined Configuration

```python
from trading_bot.config.typed_settings import load_config, RiskSettings

# Load base config
settings = load_config("/path/to/config.yaml")

# Create risk settings override
risk_override = RiskSettings(
    max_position_pct=0.02,  # More conservative than default
    max_risk_pct=0.005,     # More conservative than default
)

# Combine settings
combined_settings = TradingBotSettings(
    **{**settings.dict(), "risk": risk_override.dict()}
)
```

## Migration from Legacy Config

If migrating from legacy configuration:

1. Use the provided `scripts/migrate_config.py` script to convert your old config
2. Review and adjust the generated YAML file
3. Add any missing settings that are required by the new system

```bash
python scripts/migrate_config.py old_config.json config.yaml
```
