# Enhanced Trading Dashboard

A comprehensive dashboard for monitoring and controlling your automated trading system with real-time statistics, interactive visualizations, and intelligent recommendations.

## Features

- **Real-time Monitoring**: Track open positions, account metrics, and trading bot status
- **Interactive Charts**: Visualize equity curves, win rates, strategy performance, and more
- **AI Recommendations**: Receive intelligent suggestions for improving trading performance
- **Strategy Management**: View and control strategy allocations and performance
- **Trade Analysis**: Analyze historical trades with advanced filtering and pattern detection
- **Notifications**: Get alerts for important events via desktop, email, or Slack
- **Customizable**: Configure settings to match your trading style and preferences

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r trading_bot/requirements.txt
   ```

3. Initialize the dashboard configuration:
   ```bash
   python -m trading_bot.launch_dashboard --init
   ```

## Configuration

The dashboard configuration is stored in `~/.trading_bot/dashboard_config.json`. You can modify it directly or use the dashboard settings panel.

Key configuration options:
- API connection settings
- Refresh intervals
- Chart preferences
- Notification settings
- UI customization

## Running the Dashboard

Start the dashboard with:

```bash
python -m trading_bot.launch_dashboard
```

Or use command-line arguments to override configuration:

```bash
python -m trading_bot.launch_dashboard --api-url http://your-api-server:5000 --refresh 15
```

Available command-line options:
- `--config`: Path to custom configuration file
- `--api-url`: Override API URL
- `--refresh`: Override refresh interval (seconds)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--init`: Initialize dashboard directories and configuration
- `--reset-config`: Reset configuration to defaults

## API Server Setup

The dashboard connects to your trading system via API endpoints. Ensure your Flask API server is running:

```bash
python -m trading_bot.webhook_receiver
```

### API Authentication

By default, API authentication is disabled. To enable token-based authentication:

1. Set environment variable:
   ```bash
   export API_AUTH_ENABLED=true
   ```

2. Provide tokens in environment or file:
   ```bash
   # Option 1: Environment variable
   export API_TOKENS="token1:user1,token2:user2"
   
   # Option 2: Create tokens file
   echo '{"token1": "user1", "token2": "user2"}' > api_tokens.json
   export API_TOKENS_FILE=api_tokens.json
   ```

## Dashboard Components

### Enhanced Dashboard (Main UI)
The main dashboard interface with multiple panels for different views

### API Client
Robust communication with the trading bot API, including error handling

### Dashboard Charts
Interactive Plotly visualizations for trading performance

### Notification Manager
Advanced notification system with multiple delivery channels

### Config Manager
Configuration handling and persistence

## Usage Guide

### Navigation
- Use tab/arrow keys to navigate between panels
- Press `?` for help menu with keyboard shortcuts
- Press `q` to quit

### Dashboard Sections
- **Summary Panel**: Overview of account and key metrics
- **Positions Panel**: Current open positions and their status
- **Trades Panel**: Recent trade history with filtering
- **Charts Panel**: Interactive visualizations of performance
- **Recommendations**: AI-generated suggestions
- **Settings**: Configure dashboard preferences

## Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

### Running Tests
```bash
pytest trading_bot/tests/
```

## Troubleshooting

### Common Issues
- **API Connection Failed**: Check API URL and API server status
- **Authentication Error**: Verify API tokens are correctly set
- **Missing Data**: Ensure trading journal has sufficient data
- **Display Issues**: Try adjusting your terminal size or font

For more help, check the log file at `~/.trading_bot/logs/dashboard.log`

## License

MIT License

# TradingView Webhook to Tradier Trading Integration

This module provides a complete bridge between TradingView alerts and the Tradier brokerage API. It receives webhook signals from TradingView, validates them against market context and risk parameters, and executes trades through Tradier.

## Key Features

- **TradingView Webhook Receiver**: Process alerts from TradingView with custom JSON payloads
- **Market Context Analysis**: Analyze current market regime (bullish, bearish, sideways, volatile) to make smarter trading decisions
- **Smart Position Sizing**: Adjust position sizes based on market conditions, VIX levels, and strategy performance
- **Risk Management**: Apply limits on trade sizes, stop losses, and daily risk
- **Tradier Integration**: Seamless trade execution through Tradier's API
- **Webhook Security**: Optional signature verification for secure webhooks

## Setup Instructions

### 1. Configure Your Environment

Create a configuration file at `~/.trading_bot/config.json` based on `example_config.json`. At minimum, set:

```json
{
  "tradier": {
    "api_key": "YOUR_TRADIER_API_KEY",
    "account_id": "YOUR_TRADIER_ACCOUNT_ID",
    "use_sandbox": true
  }
}
```

### 2. Start the Webhook Server

Run the Flask application:

```bash
python -m trading_bot.flask_app
```

This starts a server at `http://localhost:5000` with an endpoint at `/api/webhook/tradingview`.

### 3. Make it Accessible (for Production)

For TradingView to send webhook alerts, your server needs to be accessible from the internet. Options:
- Deploy to a cloud provider (AWS, GCP, etc.)
- Use a service like ngrok: `ngrok http 5000`

### 4. Create TradingView Alerts

1. Open TradingView and create a new alert
2. Set your alert conditions
3. Select "Webhook URL" as the alert action
4. Paste your webhook URL (e.g. `https://your-server.com/api/webhook/tradingview`)
5. Set the webhook message body to match one of the templates in `example_config.json`:

```json
{
  "symbol": "{{ticker}}",
  "strategy": "rsi_oversold",
  "signal": "buy",
  "timeframe": "{{interval}}",
  "price": "{{close}}",
  "stop_loss": "{{strategy.order.stop_price}}",
  "take_profit": "{{strategy.order.limit_price}}",
  "risk": 1.0
}
```

## Example Webhook Payload

A minimal valid webhook payload requires:

```json
{
  "symbol": "AAPL",
  "strategy": "rsi_oversold",
  "signal": "buy"
}
```

Enhanced payload with stop loss and take profit:

```json
{
  "symbol": "AAPL",
  "strategy": "macd_crossover",
  "signal": "buy",
  "timeframe": "1h",
  "price": 150.25,
  "stop_loss": 148.50,
  "take_profit": 155.00,
  "risk": 1.0,
  "entry_type": "market"
}
```

## Signal Types

The system recognizes the following signal types:

- `buy`, `long`: Open a long position
- `sell`, `short`: Open a short position
- `close_long`: Close an existing long position
- `close_short`: Cover an existing short position

## API Endpoints

- `POST /api/webhook/tradingview`: Receive webhook signals from TradingView
- `GET /api/signals`: View signal history
- `GET /api/stats`: Get processing statistics and counts
- `GET /api/market-context`: View current market context and regime
- `GET /healthcheck`: Health check endpoint

## Creating a TradingView Strategy with Alerts

Here's a simple Pine Script example that creates a basic RSI strategy and sends alerts:

```pine
//@version=4
strategy("RSI Strategy with Alerts", overlay=true)

// RSI settings
rsiLength = input(14, "RSI Length")
rsiOverbought = input(70, "RSI Overbought Level")
rsiOversold = input(30, "RSI Oversold Level")

// Calculate RSI
rsiValue = rsi(close, rsiLength)

// Entry conditions
longCondition = crossover(rsiValue, rsiOversold)
shortCondition = crossunder(rsiValue, rsiOverbought)

// Exit conditions
exitLongCondition = crossover(rsiValue, rsiOverbought)
exitShortCondition = crossunder(rsiValue, rsiOversold)

// Strategy execution
if (longCondition)
    strategy.entry("Long", strategy.long, comment="rsi_oversold")
    
if (shortCondition)
    strategy.entry("Short", strategy.short, comment="rsi_overbought")
    
if (exitLongCondition)
    strategy.close("Long")
    
if (exitShortCondition)
    strategy.close("Short")

// Alert conditions
alertcondition(longCondition, title="RSI Oversold Entry", message="RSI Oversold Buy Signal")
alertcondition(shortCondition, title="RSI Overbought Entry", message="RSI Overbought Short Signal")
```

## Security Considerations

For production use, it's recommended to:

1. Enable webhook signature verification by setting `webhook.verify_webhook_signatures` to `true` and configuring a secret
2. Use HTTPS for your webhook endpoint
3. Deploy behind a proper reverse proxy like Nginx
4. Set appropriate access controls and rate limits

## Troubleshooting

- Check the logs for detailed error messages
- Ensure your Tradier API key has trading permissions
- Verify the market is open when testing market orders
- Test with the Tradier sandbox environment first

# BenBot Trading Assistant

## Overview

BenBot is an AI-powered trading assistant that helps users manage their trading portfolio, analyze market data, and make informed trading decisions. The system consists of multiple components that work together to provide a comprehensive trading experience.

## Core Components

### Portfolio State Manager (`portfolio_state.py`)

The Portfolio State Manager maintains a comprehensive record of the trading system's state, including:

- Portfolio holdings and valuation
- Trading strategies and their allocations
- Performance metrics
- Recent trades and signals
- System and learning status

It serves as the central data repository that other components can query to get the current state of the trading system.

### Assistant Context (`assistant_context.py`)

The Assistant Context acts as a bridge between the Portfolio State Manager and the BenBot assistant. It:

- Formats portfolio state data into context that can be used by the assistant
- Analyzes user queries to provide relevant context
- Structures information in a way that makes it easy for the assistant to generate helpful responses

The context provided is dynamically adjusted based on the content of the user's query, ensuring that the assistant has the most relevant information to provide an accurate and helpful response.

## Integration

The integration between these components enables the BenBot assistant to:

1. Access real-time portfolio data
2. Provide context-aware responses to user queries
3. Offer insights based on current portfolio performance
4. Alert users to important events or changes in their portfolio

## Usage Example

A demonstration of how to use these components together can be found in `examples/benbot_context_demo.py`. This script:

1. Initializes a Portfolio State Manager with simulated data
2. Creates an Assistant Context linked to the Portfolio State Manager
3. Simulates portfolio updates over time
4. Demonstrates how different types of user queries result in different context being provided to the assistant

To run the demo:

```bash
python -m trading_bot.examples.benbot_context_demo
```

## Query Types

The Assistant Context can handle various types of queries, including:

- Portfolio information (holdings, total value, allocation)
- Strategy performance and allocation
- Recent trading activity
- Performance metrics (returns, Sharpe ratio, drawdown)
- System status and learning progress

Each query type triggers different context to be provided to the assistant, ensuring that responses are tailored to the user's needs.

## Extending the System

The system is designed to be extensible. You can:

- Add new metrics to the Portfolio State Manager
- Implement new formatting methods in the Assistant Context
- Create new query types and corresponding context providers

## Requirements

- Python 3.7+
- Required packages listed in requirements.txt 