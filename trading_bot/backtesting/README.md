# Autonomous ML Backtesting System

This package provides a complete end-to-end backtesting system that uses machine learning to autonomously generate, test, and improve trading strategies based on market data, news sentiment, and technical analysis.

## Key Features

- 🧠 **Autonomous Strategy Generation**: ML autonomously creates trading strategies tailored to current market conditions
- 📈 **Comprehensive Data Integration**: Combines historical price data with real-time news sentiment and technical indicators
- 🔍 **Transparent ML Reasoning**: Explains strategy decisions across political, social, and economic dimensions
- 📊 **Advanced Performance Analytics**: Detailed analysis of winning and losing strategies across different market conditions
- 🧪 **Continuous Learning**: System learns from backtest results to improve future strategy generation

## Architecture

The system is composed of four main components:

### 1. Data Integration Layer

Integrates multiple data sources:
- Historical price data
- News sentiment across political, social, and economic dimensions
- Technical indicators
- Volume and liquidity metrics

### 2. Strategy Generator

Creates trading strategies based on ML analysis:
- Uses ML to select best strategy templates for current market conditions
- Optimizes strategy parameters
- Applies risk management rules
- Provides detailed reasoning for why each strategy was selected

### 3. Autonomous Backtester

Runs and analyzes backtests:
- Tests strategies across different timeframes
- Simulates trades with realistic execution
- Calculates comprehensive performance metrics
- Categorizes strategies as winning or losing

### 4. ML Learning Component

Learns from backtest results:
- Updates ML model based on successful and unsuccessful strategies
- Identifies patterns in winning and losing strategies
- Suggests improvements to existing strategies
- Optimizes parameters for different market conditions

## Usage

```python
# Initialize components
from trading_bot.backtesting import (
    initialize_ml_backtesting,
    register_ml_backtest_endpoints
)

# Initialize the ML backtesting system with your news fetcher
initialize_ml_backtesting(news_fetcher)

# Register API endpoints for the frontend
register_ml_backtest_endpoints(app)

# Run an autonomous backtest cycle
@app.route('/api/autonomous-backtest', methods=['POST'])
def run_autonomous_backtest():
    params = request.json
    
    # Run the full autonomous cycle
    results = backtester.run_full_autonomous_cycle(
        tickers=params.get('tickers'),
        timeframes=params.get('timeframes'),
        sectors=params.get('sectors')
    )
    
    # Learn from the results
    learning_metrics = ml_optimizer.learn_from_results(results)
    
    # Return results
    return jsonify({
        'success': True,
        'results': results,
        'learning_metrics': learning_metrics
    })
```

## API Endpoints

The system provides the following API endpoints:

- `POST /api/autonomous-backtest`: Run a full autonomous backtesting cycle
- `GET /api/ml-strategy-suggestions`: Get ML-suggested strategies for current market conditions
- `POST /api/ml-improve-strategy`: Suggest improvements for an existing strategy
- `GET /api/ml-model-insights`: Get insights from the ML model

## Frontend Integration

The system includes a React component (`BacktestTab.js`) that provides a complete user interface for the autonomous ML backtesting system. It displays:

- ML Learning Status with data sources and insights
- Top ML-generated winning and losing strategies
- User collaborative strategies with ML assistance
- Detailed ML reasoning for strategy decisions
- Performance metrics and visualizations

## Customization

The system is designed to be modular and extensible:

- Add new strategy templates in `StrategyTemplateLibrary`
- Customize risk management rules in `RiskManager`
- Extend sentiment analysis in `SentimentAnalyzer`
- Add new technical indicators in the data integration layer

## Requirements

- Python 3.8+
- Flask
- Pandas, NumPy
- React (for frontend) 