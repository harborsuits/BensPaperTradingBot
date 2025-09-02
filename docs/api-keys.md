# API Keys

The BensBot Trading System integrates with multiple external services for market data, news feeds, and AI-powered analysis. This page documents the API keys required and how to configure them.

## Supported API Services

The trading system supports the following API services:

### Market Data APIs

| Provider | Purpose | Documentation |
|----------|---------|---------------|
| Tradier | Primary market data and trading execution | [Tradier API Docs](https://documentation.tradier.com/) |
| Alpha Vantage | Historical data and financial indicators | [Alpha Vantage Docs](https://www.alphavantage.co/documentation/) |
| Finnhub | Real-time market data and sentiment | [Finnhub Docs](https://finnhub.io/docs/api) |
| Alpaca | Alternative broker and market data | [Alpaca Docs](https://alpaca.markets/docs/api-documentation/) |

### News APIs

| Provider | Purpose | Documentation |
|----------|---------|---------------|
| Marketaux | Financial news with sentiment analysis | [Marketaux Docs](https://www.marketaux.com/documentation) |
| NewsData.io | Comprehensive news coverage | [NewsData.io Docs](https://newsdata.io/documentation) |
| GNews | Google News aggregator | [GNews Docs](https://gnews.io/docs/v4) |
| MediaStack | Global news data | [MediaStack Docs](https://mediastack.com/documentation) |
| Currents | Trending financial news | [Currents API Docs](https://currentsapi.services/en/docs) |
| NYTimes | New York Times articles | [NYTimes API](https://developer.nytimes.com/apis) |

### AI Model APIs

| Provider | Purpose | Documentation |
|----------|---------|---------------|
| OpenAI (primary) | Market analysis and text generation | [OpenAI API Docs](https://platform.openai.com/docs/api-reference) |
| OpenAI (secondary) | Backup for primary OpenAI services | [OpenAI API Docs](https://platform.openai.com/docs/api-reference) |
| Claude (Anthropic) | Alternative LLM for analysis | [Anthropic API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api) |
| Mistral | Efficient analysis and summarization | [Mistral API Docs](https://docs.mistral.ai/) |
| Cohere | Embeddings and semantic analysis | [Cohere API Docs](https://docs.cohere.com/reference/about) |
| Gemini (Google) | Alternative model for analysis | [Gemini API Docs](https://ai.google.dev/tutorials/rest_quickstart) |
| HuggingFace | Specialized machine learning models | [HuggingFace API Docs](https://huggingface.co/docs/api-inference/index) |

## Configuration Methods

API keys can be configured using two methods (in order of precedence):

1. **Environment Variables** (Recommended for security)
2. **Configuration Files** (YAML or JSON)

## Environment Variables

For the most secure setup, configure API keys using environment variables:

```bash
# Broker credentials
export TRADIER_API_KEY="your-tradier-api-key"
export TRADIER_ACCOUNT_ID="your-tradier-account-id"
export TRADIER_SANDBOX="true"  # Use sandbox environment

# Market data providers
export ALPHA_VANTAGE_KEY="your-alphavantage-key"
export FINNHUB_KEY="your-finnhub-key"
export ALPACA_KEY="your-alpaca-key"
export ALPACA_SECRET="your-alpaca-secret"

# News APIs
export MARKETAUX_KEY="your-marketaux-key"
export NEWSDATA_KEY="your-newsdata-key"
export GNEWS_KEY="your-gnews-key"
export MEDIASTACK_KEY="your-mediastack-key"
export CURRENTS_KEY="your-currents-key"
export NYTIMES_KEY="your-nytimes-key"

# AI model providers
export OPENAI_API_KEY="your-openai-key"
export OPENAI_SECONDARY_KEY="your-backup-openai-key" 
export ANTHROPIC_KEY="your-claude-key"
export MISTRAL_KEY="your-mistral-key"
export COHERE_KEY="your-cohere-key"
export GEMINI_KEY="your-gemini-key"
export HUGGINGFACE_KEY="your-huggingface-key"

# Notification services
export TELEGRAM_TOKEN="your-telegram-bot-token"
export TELEGRAM_CHAT_ID="your-telegram-chat-id"
```

## Configuration File

Alternatively, you can use a YAML configuration file:

```yaml
broker:
  name: "tradier"
  api_key: "YOUR_TRADIER_API_KEY"
  account_id: "YOUR_TRADIER_ACCOUNT_ID"
  sandbox: true

data:
  provider: "tradier"
  historical_source: "alpha_vantage"

api:
  api_keys:
    # Market data
    alpha_vantage: "YOUR_ALPHA_VANTAGE_KEY"
    finnhub: "YOUR_FINNHUB_KEY"
    alpaca: "YOUR_ALPACA_KEY"
    
    # News providers
    marketaux: "YOUR_MARKETAUX_KEY"
    newsdata: "YOUR_NEWSDATA_KEY"
    gnews: "YOUR_GNEWS_KEY"
    mediastack: "YOUR_MEDIASTACK_KEY"
    currents: "YOUR_CURRENTS_KEY"
    nytimes: "YOUR_NYTIMES_KEY"
    
    # AI models
    openai_primary: "YOUR_OPENAI_KEY"
    openai_secondary: "YOUR_BACKUP_OPENAI_KEY"
    claude: "YOUR_ANTHROPIC_KEY"
    mistral: "YOUR_MISTRAL_KEY"
    cohere: "YOUR_COHERE_KEY"
    gemini: "YOUR_GEMINI_KEY"
    huggingface: "YOUR_HUGGINGFACE_KEY"

notifications:
  enable_notifications: true
  telegram_token: "YOUR_TELEGRAM_BOT_TOKEN"
  telegram_chat_id: "YOUR_TELEGRAM_CHAT_ID"
```

!!! warning "Security Best Practices"
    For production usage, **always** use environment variables for API keys instead of storing them in configuration files. This prevents accidental exposure of credentials in version control systems.

## API Key Management

For managing multiple API keys across different environments:

1. Create a `.env` file for local development (add to `.gitignore`)
2. Use environment variables in CI/CD pipelines
3. For production servers, use a secure environment variable management system

## Fallback Mechanism

The system implements a fallback mechanism for news and market data APIs:

1. If the primary API is unavailable, the system will try the next API
2. If all APIs fail, the system will log an error and use cached data if available

Example fallback chain for news:
```
NewsData.io → Marketaux → GNews → MediaStack → Currents → NYTimes
```

## API Key Setup Instructions

### Tradier Account Setup

1. Sign up for a [Tradier Brokerage account](https://brokerage.tradier.com/) or [Developer Sandbox](https://developer.tradier.com/user/sign_up)
2. Navigate to Settings → API Access
3. Generate a new API key
4. Note your account ID and API key

### Other API Services

Follow these general steps for each service:

1. Register on the provider's website
2. Navigate to the API or Developer section
3. Generate an API key/token
4. Add the key to your environment variables or configuration file
