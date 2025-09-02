# Live Trading Dashboard

A real-time monitoring dashboard for trading systems, providing visualizations of portfolio performance, market data, regime detection, and trading decisions.

## Features

- **Real-time Portfolio Monitoring**: Track portfolio value, allocation, and performance metrics
- **Market Data Visualization**: View price charts and market data for monitored symbols
- **Market Regime Analysis**: Visualize detected market regimes and performance by regime
- **Trading Activity Tracking**: Monitor recent trades and strategy allocations
- **Multiple Data Sources**: Connect to Alpaca, Interactive Brokers, or use mock data for demonstration

## Screenshots

The dashboard includes:

- Portfolio performance charts with drawdown analysis
- Real-time price charts for tracked symbols
- Market regime timeline and statistics
- Trading activity visualization
- Strategy allocation tracking

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_dashboard.txt
```

2. Install Streamlit if not already installed:

```bash
pip install streamlit
```

## Usage

### Running the Dashboard

Use the provided launcher script:

```bash
python trading_bot/run_dashboard.py
```

Or run directly with Streamlit:

```bash
streamlit run trading_bot/visualization/live_trading_dashboard.py
```

### Command Line Options

The launcher script supports the following options:

```
--port PORT           Port to run the dashboard on (default: 8501)
--alpaca-key KEY      Alpaca API key (can also be set via ALPACA_API_KEY env var)
--alpaca-secret SECRET Alpaca API secret (can also be set via ALPACA_API_SECRET env var)
--mock-data           Use mock data instead of connecting to real data sources
```

Example:

```bash
python trading_bot/run_dashboard.py --port 8888 --mock-data
```

### Using with Real Data

To use with Alpaca:

1. Set your API credentials either as environment variables:

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"
```

2. Or pass them directly to the launcher:

```bash
python trading_bot/run_dashboard.py --alpaca-key "your-api-key" --alpaca-secret "your-api-secret"
```

To use with Interactive Brokers:

1. Make sure TWS or IB Gateway is running
2. Configure connection settings in the dashboard interface

## Dashboard Interface

The dashboard consists of several main sections:

### Configuration Sidebar

- Select data source (Alpaca, Interactive Brokers, or Mock Data)
- Configure symbols to monitor
- Set update frequency and time range
- Connect/disconnect from data sources

### Portfolio Performance

- Real-time portfolio value chart
- Allocation breakdown
- Performance metrics (returns, volatility, Sharpe ratio, drawdown)

### Market Data

- Real-time price charts for each monitored symbol
- OHLCV data visualization

### Market Regime Analysis

- Current market regime display
- Regime timeline and transition history
- Performance statistics by regime

### Trading Activity

- Recent trade history
- Strategy allocation charts
- Trading volume by symbol

## Integration

The dashboard integrates with the following components:

- `RealTimeDataManager` for market data processing
- `AdvancedMarketRegimeDetector` for regime detection
- Trading system components for portfolio and trade data

## Customization

To customize the dashboard:

1. Modify the color schemes in `REGIME_COLORS` variable
2. Add additional visualization components in the respective display functions
3. Extend the data processing capabilities by modifying the callbacks

## Troubleshooting

If you encounter issues:

- Check that you have all required dependencies installed
- Verify your API credentials if using real data sources
- Examine the console logs for error messages
- Make sure required ports are open if accessing remotely

## License

This dashboard is released under the same license as the main project. 