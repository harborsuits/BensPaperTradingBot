# Environment Variables

The BensBot Trading System uses environment variables for secure configuration of sensitive information like API keys, account credentials, and risk parameters.

## Why Use Environment Variables

Environment variables provide several advantages:

1. **Security**: Credentials are never stored in code or config files
2. **Environment-specific settings**: Easily switch between development, testing, and production setups
3. **Container compatibility**: Simplifies deployment in Docker and cloud environments
4. **Separation of concerns**: Keeps sensitive data separate from application code

## Core Environment Variables

### Broker Credentials

| Variable | Description | Default |
|----------|-------------|---------|
| `TRADIER_API_KEY` | Tradier API key | (Required) |
| `TRADIER_ACCOUNT_ID` | Tradier account ID | (Required) |
| `TRADIER_SANDBOX` | Use Tradier sandbox | `"true"` |
| `ALPACA_API_KEY` | Alpaca API key | None |
| `ALPACA_SECRET_KEY` | Alpaca API secret | None |

### Risk Parameters

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_RISK_PCT` | Maximum risk per trade | `0.01` (1%) |
| `MAX_POSITION_PCT` | Maximum position size | `0.05` (5%) |
| `MAX_PORTFOLIO_RISK` | Maximum total portfolio risk | `0.25` (25%) |
| `MAX_OPEN_TRADES` | Maximum concurrent positions | `5` |
| `PORTFOLIO_STOP_LOSS_PCT` | Portfolio stop-loss threshold | `0.05` (5%) |
| `INITIAL_CAPITAL` | Starting capital for backtests | `100000.0` |

### API Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Host for the trading bot API | `"0.0.0.0"` |
| `API_PORT` | Port for the trading bot API | `8000` |
| `ENABLE_API` | Enable/disable the API | `"true"` |
| `API_CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `"*"` |
| `API_RATE_LIMIT` | API rate limit (requests per minute) | `100` |

### Notifications

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_TOKEN` | Telegram bot token | None |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | None |
| `ENABLE_NOTIFICATIONS` | Enable/disable notifications | `"true"` |
| `NOTIFICATION_LEVEL` | Minimum level to notify | `"info"` |

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_CONNECTION_STRING` | Database connection URL | `"sqlite:///trading_bot.db"` |
| `DB_ENABLE_LOGGING` | Enable DB query logging | `"false"` |

## Market Data API Keys

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHA_VANTAGE_KEY` | Alpha Vantage API key | None |
| `FINNHUB_KEY` | Finnhub API key | None |
| `POLYGON_KEY` | Polygon.io API key | None |

## News API Keys

| Variable | Description | Default |
|----------|-------------|---------|
| `MARKETAUX_KEY` | Marketaux API key | None |
| `NEWSDATA_KEY` | NewsData.io API key | None |
| `GNEWS_KEY` | GNews API key | None |
| `MEDIASTACK_KEY` | MediaStack API key | None |
| `CURRENTS_KEY` | Currents API key | None |
| `NYTIMES_KEY` | NY Times API key | None |

## AI Model Keys

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI primary API key | None |
| `OPENAI_SECONDARY_KEY` | OpenAI backup API key | None |
| `ANTHROPIC_KEY` | Anthropic Claude API key | None |
| `MISTRAL_KEY` | Mistral AI API key | None |
| `COHERE_KEY` | Cohere API key | None |
| `GEMINI_KEY` | Google Gemini API key | None |
| `HUGGINGFACE_KEY` | HuggingFace API key | None |

## Logging and Debugging

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging verbosity | `"INFO"` |
| `LOG_FILE` | Log file path | `"trading_bot.log"` |
| `DEBUG_MODE` | Enable extended debugging | `"false"` |

## Using Environment Variables

### In Local Development

Create a `.env` file in the project root (and add to .gitignore):

```bash
# .env
TRADIER_API_KEY=your_api_key
TRADIER_ACCOUNT_ID=your_account_id
MAX_RISK_PCT=0.01
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

Then load using python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
```

### In Docker

Pass environment variables when running the container:

```bash
docker run -e TRADIER_API_KEY=your_key -e MAX_RISK_PCT=0.01 bensbot:latest
```

Or use a docker-compose.yml file:

```yaml
version: '3'
services:
  trading_bot:
    image: bensbot:latest
    env_file:
      - .env
```

### In Production

For cloud deployments, use the platform's secure environment variable storage:

- **AWS**: Use Parameter Store or Secrets Manager
- **GCP**: Use Secret Manager
- **Azure**: Use Key Vault
- **Heroku**: Use Config Variables

## Accessing Environment Variables in Code

The typed settings system loads environment variables automatically when creating configuration objects:

```python
from trading_bot.config.typed_settings import load_config

# This will load values from environment variables
settings = load_config()

# Access the loaded values
api_key = settings.broker.api_key  # Loaded from TRADIER_API_KEY
max_risk = settings.risk.max_risk_pct  # Loaded from MAX_RISK_PCT
```
