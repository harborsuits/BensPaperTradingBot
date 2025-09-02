# Broker Historical Performance Tracking

This document describes the broker historical performance tracking system that records, analyzes, and visualizes broker performance metrics over time.

## Overview

The historical tracking system continuously records broker metrics and provides tools for trend analysis, anomaly detection, and performance comparison. This helps identify performance degradation over time, detect operational issues early, and make informed decisions about broker selection and routing.

## Key Components

1. **Storage Layer**
   - `SQLiteTimeSeriesStore`: Efficient SQLite-based storage for time series data
   - `CSVTimeSeriesStore`: Simple CSV-based storage with file-per-day organization

2. **Recording System**
   - `BrokerPerformanceTracker`: Records metrics at regular intervals
   - Integrates with the event system to capture metrics in real-time
   - Automatically prunes old data according to retention policy

3. **Analysis Tools**
   - `BrokerPerformanceAnalyzer`: Provides trend analysis, anomaly detection, and forecasting
   - Supports moving averages, z-score anomaly detection, and seasonality analysis
   - Can forecast future performance based on historical trends

4. **Visualization Dashboard**
   - Interactive dashboard for exploring historical performance
   - Time series trends and pattern analysis
   - Anomaly detection with configurable thresholds
   - Side-by-side broker comparison and ranking

## Setup and Configuration

### Starting the Historical Tracker

Run the setup script to initialize the historical tracker:

```bash
python -m trading_bot.scripts.setup_historical_tracker
```

Options:
- `--storage-type`: Storage type ('sqlite' or 'csv'), default: 'sqlite'
- `--storage-path`: Path to storage location, default: 'data/broker_performance'
- `--sampling-interval`: Interval in seconds between recordings, default: 300 (5 minutes)
- `--retention-days`: Days to retain historical data, default: 90
- `--no-recording`: Flag to prevent immediate recording start

### Accessing Historical Data in Code

```python
from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker
from trading_bot.event_system.event_bus import EventBus

# Create tracker
event_bus = EventBus()
tracker = BrokerPerformanceTracker(
    event_bus=event_bus,
    storage_type='sqlite',
    storage_path='data/broker_performance',
    sampling_interval=300,
    retention_days=90
)

# Start recording
tracker.start_recording()

# Access analyzer for data analysis
analyzer = tracker.get_analyzer()

# Get data as DataFrame for specific broker
df = analyzer.store.get_as_dataframe(
    broker_id='broker_a',
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

# Calculate moving averages
ma_df = analyzer.calculate_moving_averages(
    broker_id='broker_a',
    metric_name='latency_mean_ms',
    windows=[5, 20, 50]
)

# Detect anomalies
anomalies_df = analyzer.detect_anomalies(
    broker_id='broker_a',
    metric_name='latency_mean_ms',
    zscore_threshold=3.0
)
```

## Dashboard Usage

The historical performance dashboard is accessible through the main BensBot Trading Dashboard under the "Broker & Positions" section, in the "Historical Performance" tab.

### Dashboard Features

1. **Time Series Trends**
   - View performance metrics over time
   - Display moving averages to identify trends
   - Analyze seasonality by hour, day, or week

2. **Anomaly Detection**
   - Identify unusual performance patterns
   - Configurable detection sensitivity
   - Detailed view of detected anomalies

3. **Broker Comparison**
   - Side-by-side metric comparison
   - Radar chart for multi-dimensional analysis
   - Performance ranking across metrics

4. **Data Export**
   - Download raw data as CSV for external analysis
   - View data table with pagination

## Metrics Tracked

The system tracks the following broker performance metrics:

- **Latency**: Average response time in milliseconds
- **Reliability**: Availability percentage and error counts
- **Execution Quality**: Average slippage percentage
- **Cost**: Average commission per trade
- **Overall Performance**: Combined score across all metrics

## Implementation Notes

### Data Storage

Historical data is stored in either SQLite database format or CSV files. The SQLite format is more efficient for querying but requires the SQLite library, while CSV is simpler but less efficient for large datasets.

### Sampling Frequency

The default sampling interval is 5 minutes. This balances data granularity with storage requirements. For high-frequency trading systems, consider reducing this interval to 1 minute or less.

### Data Retention

By default, data is retained for 90 days. This provides enough history for seasonal analysis while managing storage requirements. Adjust the retention period based on your storage capacity and analysis needs.

### Database Size Management

For long-term deployments, consider:
- Implementing data aggregation (e.g., hourly averages for data older than 7 days)
- Setting up database maintenance tasks (vacuum, indexing)
- Configuring automated backups

## Future Enhancements

Planned enhancements for the historical tracking system include:

1. **Machine Learning Models**
   - Predictive broker failure detection
   - Automated threshold adjustment
   - Pattern recognition for failure precursors

2. **Enhanced Visualization**
   - Heatmaps for time-of-day performance patterns
   - Correlation analysis between metrics
   - Statistical significance testing

3. **Alert Integration**
   - Configure alerts based on historical patterns
   - Predict potential issues before they occur
   - Integration with notification systems
