# Command Center for Trading Bot

The Command Center provides a comprehensive CLI interface for managing all aspects of the trading system, including trade execution, market regime analysis, strategy management, and monitoring.

## Features

- **Trade Execution**: Manually execute trades or use the market-aware trader for automated trading
- **Market Regime Analysis**: Analyze current market conditions and determine the market regime
- **Risk Management**: Change risk modes and manage account settings
- **Market Data Access**: Retrieve price data and market statistics
- **System Management**: Reload strategies and get system status

## Installation

Make sure you have installed all the required dependencies:

```bash
pip install pandas numpy requests pyyaml tabulate
```

## Usage

### Basic Usage

```bash
# Show system status
python command_center.py

# Show system status explicitly
python command_center.py --status

# Use a specific config file
python command_center.py --config my_config.yaml
```

### Trading Commands

```bash
# Execute a trade (symbol, strategy, direction, quantity)
python command_center.py --trade AAPL trend_following buy 10

# Execute a trade with default direction (buy) and quantity
python command_center.py --trade SPY breakout

# Exit a trade
python command_center.py --exit trade_123456
```

### Risk Management

```bash
# Change risk mode
python command_center.py --risk-mode conservative
python command_center.py --risk-mode balanced
python command_center.py --risk-mode aggressive

# Set account balance
python command_center.py --balance 10000
```

### Market Regime Analysis

```bash
# Analyze current market regime
python command_center.py --analyze-regime

# Run market-aware trader
python command_center.py --run-trader

# Run market-aware trader on specific symbols
python command_center.py --run-trader AAPL MSFT GOOG
```

### Market Data

```bash
# Show current market statistics
python command_center.py --market-stats

# Get price for a symbol
python command_center.py --price AAPL
```

### System Commands

```bash
# Reload strategies
python command_center.py --reload
```

## Integration with Existing Components

The Command Center seamlessly integrates with:

1. **TradeExecutor**: For executing trades with proper risk management
2. **MarketAwareTrader**: For market regime analysis and regime-based trading
3. **TradeJournalSystem**: For comprehensive trade journaling and analysis (via the journaled executor)

## Architecture

The Command Center consists of several key components:

- **CommandCenter**: Main class that coordinates all operations
- **MarketDataManager**: Retrieves and manages market data
- **NotificationManager**: Sends notifications about trades and other events

## Configuration

Create a `config.yaml` file with the following structure:

```yaml
initial_balance: 10000.0
default_risk_mode: balanced
market_data_source: mock
data_directory: data
default_universe:
  - SPY
  - QQQ
  - AAPL
  - MSFT
  - AMZN
max_positions: 5
max_trades_per_day: 10
enable_notifications: false
``` 