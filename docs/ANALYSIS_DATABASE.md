# Analysis Database Documentation

## Overview

The Analysis Database is a SQLite-based solution for persistently storing analysis results and selection history in the trading bot system. It provides structured storage for various types of analysis data, including technical/fundamental analysis results, sentiment analysis, stock and strategy selections, and market regime classifications.

## Key Features

- **Persistent Storage**: All analysis results and selections are stored in a SQLite database for long-term persistence.
- **Structured Data**: Provides a well-defined schema for different types of analysis data.
- **JSON Serialization**: Complex data structures are stored as JSON in text fields for flexibility.
- **Historical Tracking**: Maintains a timestamp-based history of all analysis and selections.
- **Queryable**: Supports retrieving data by symbol, type, and other criteria.

## Database Schema

The database consists of the following tables:

### 1. `stock_analysis`
Stores analysis results for individual stocks.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| symbol | TEXT | Stock symbol |
| timestamp | TEXT | ISO format timestamp |
| analysis_type | TEXT | Type of analysis (technical, fundamental, ml, etc.) |
| score | REAL | Analysis score (normalized value) |
| rank | INTEGER | Rank among analyzed stocks |
| recommendation | TEXT | Trading recommendation (buy, sell, hold) |
| metrics | TEXT | JSON of analysis metrics |
| analysis_details | TEXT | JSON of detailed analysis information |
| model_version | TEXT | Version of the analysis model used |

### 2. `stock_selection_history`
Tracks history of stock selections.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TEXT | ISO format timestamp |
| selection_criteria | TEXT | Criteria used for selection |
| symbols | TEXT | JSON array of selected stock symbols |
| weights | TEXT | JSON mapping of symbols to allocation weights |
| market_regime | TEXT | Current market regime |
| reasoning | TEXT | Explanation for the selection |
| performance_snapshot | TEXT | JSON of portfolio performance metrics |

### 3. `strategy_selection_history`
Tracks history of strategy selections.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TEXT | ISO format timestamp |
| selected_strategy | TEXT | Name of the selected strategy |
| market_regime | TEXT | Current market regime |
| confidence_score | REAL | Confidence score for the selection |
| reasoning | TEXT | Explanation for the selection |
| parameters | TEXT | JSON of strategy parameters |

### 4. `market_regime_history`
Stores market regime analysis history.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TEXT | ISO format timestamp |
| regime | TEXT | Identified market regime |
| confidence | REAL | Confidence score |
| indicators | TEXT | JSON of indicator values used |
| description | TEXT | Description of the market regime |

### 5. `sentiment_analysis`
Stores sentiment analysis results.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| symbol | TEXT | Stock symbol |
| timestamp | TEXT | ISO format timestamp |
| source | TEXT | Source of sentiment data (twitter, news, etc.) |
| sentiment_score | REAL | Normalized sentiment score |
| sentiment_label | TEXT | Sentiment classification label |
| volume | INTEGER | Volume of sentiment data points |
| key_phrases | TEXT | JSON array of key phrases or topics |
| source_details | TEXT | JSON of additional details about the source |

## Usage Examples

### Initializing the Database

```python
from trading_bot.data import AnalysisDatabase

# Initialize with default path
db = AnalysisDatabase()

# Or specify a custom path
db = AnalysisDatabase("/path/to/custom/analysis_results.db")
```

### Saving Stock Analysis

```python
db.save_stock_analysis(
    symbol="AAPL",
    analysis_type="technical",
    score=0.85,
    rank=1,
    recommendation="buy",
    metrics={
        "rsi": 65.4,
        "macd": 1.2,
        "bollinger_bands": 0.5
    },
    analysis_details={
        "signal_strength": "strong",
        "trend": "bullish",
        "support_level": 145.30,
        "resistance_level": 152.80
    },
    model_version="1.2.0"
)
```

### Retrieving Stock Analysis

```python
# Get the latest technical analysis for a symbol
results = db.get_latest_stock_analysis("AAPL", "technical")

# Get all latest analyses for a symbol
all_analyses = db.get_latest_stock_analysis("AAPL")
```

### Saving Stock Selection

```python
db.save_stock_selection(
    selection_criteria="momentum_strategy",
    symbols=["AAPL", "MSFT", "AMZN"],
    weights={"AAPL": 0.4, "MSFT": 0.3, "AMZN": 0.3},
    market_regime="bullish",
    reasoning="Selected top momentum stocks in tech sector",
    performance_snapshot={
        "expected_return": 0.12,
        "expected_volatility": 0.18,
        "sharpe_ratio": 1.8
    }
)
```

### Retrieving Selection History

```python
# Get recent selection history
selections = db.get_stock_selection_history(limit=5)

# Get specific selection criteria history
momentum_selections = db.get_stock_selection_history(
    limit=10, 
    selection_criteria="momentum_strategy"
)
```

### Saving and Retrieving Market Regime Analysis

```python
# Save market regime analysis
db.save_market_regime(
    regime="bullish",
    confidence=0.85,
    indicators={
        "vix": 15.2,
        "atr": 2.3,
        "market_breadth": 0.8
    },
    description="Strong bullish trend with low volatility"
)

# Get market regime history
regime_history = db.get_market_regime_history(limit=5)
```

### Data Management

```python
# Purge old data (keep only recent data)
db.purge_old_data(days_to_keep=90)

# Get all symbols with analysis data
symbols = db.get_symbols_with_analysis()

# Get symbols with specific analysis type since a date
symbols = db.get_symbols_with_analysis(
    analysis_type="fundamental", 
    since="2023-01-01T00:00:00"
)
```

## Integration with Other Components

The Analysis Database is designed to integrate with other components of the trading system:

1. **Analysis Modules**: Analysis modules can store their results directly in the database.
2. **Strategy Selector**: The strategy selector can use historical analysis data to make decisions.
3. **Dashboard**: Visualization components can query the database to display analysis results.
4. **Backtesters**: Backtesting modules can retrieve historical selections for evaluation.

## Performance Considerations

- The database includes indexes on commonly queried fields to improve query performance.
- Consider using `purge_old_data()` periodically to prevent the database from growing too large.
- For very high-frequency operations, consider batch inserts rather than individual inserts.

## Future Enhancements

- Support for additional database backends (PostgreSQL, MySQL)
- Integration with time-series databases for high-frequency data
- Query optimization for large datasets
- Advanced filtering and aggregation methods 