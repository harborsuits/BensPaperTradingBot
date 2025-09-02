# Data Processing Architecture

This document outlines the separation of data cleaning and data quality assurance in the trading system.

## Architecture Overview

The data processing system follows a clear separation of concerns with two distinct components:

### 1. Data Cleaning (Transformation)
The `DataCleaningProcessor` focuses exclusively on:
- Data normalization and standardization
- Timestamp alignment and adjustment
- Split/dividend adjustments
- Smoothing and filtering operations
- Format conversion and preparation

### 2. Data Quality Assurance (Validation)
The `DataQualityProcessor` focuses exclusively on:
- Detecting anomalies and data integrity issues
- Repairing problematic data when possible
- Monitoring and reporting quality metrics
- Alerting on critical data quality issues
- Tracking data quality over time

### 3. Unified Pipeline
The `DataPipeline` integrates both components in a configurable way:
- Sequential processing (cleaning → quality)
- Comprehensive metrics and reporting
- Event-driven monitoring
- Configurable behavior based on data quality thresholds

## Using the Data Pipeline

### Basic Usage

```python
from trading_bot.data.data_pipeline import create_data_pipeline
from trading_bot.core.event_system import EventBus

# Create event bus
event_bus = EventBus()

# Create data pipeline
pipeline = create_data_pipeline(
    config={
        'cleaning': {
            'standardize_timestamps': True,
            'normalize_volume': True
        },
        'quality': {
            'auto_repair': True,
            'quality_threshold': 80
        }
    },
    event_bus=event_bus
)

# Process data
processed_data, metadata = pipeline.process(
    data=your_dataframe,
    symbol="EURUSD",
    source="yahoo"
)

# Check processing results
print(f"Quality score: {metadata['quality_score']}")
print(f"Issues detected: {metadata['issues_detected']}")
print(f"Issues fixed: {metadata['issues_fixed']}")
```

### Processing Multiple Symbols

```python
# Create dictionary with multiple dataframes
data_dict = {
    "EURUSD": eurusd_df,
    "GBPUSD": gbpusd_df,
    "USDJPY": usdjpy_df
}

# Process all dataframes
processed_dict, metadata = pipeline.process(data_dict, source="yahoo")

# Access results by symbol
eurusd_processed = processed_dict["EURUSD"]
```

### Configuration Options

#### Cleaning Configuration
```python
cleaning_config = {
    'standardize_timestamps': True,  # Standardize timestamp format and timezone
    'normalize_volume': False,       # Calculate relative volume
    'adjust_for_splits': True,       # Apply split adjustments
    'apply_smoothing': False,        # Apply smoothing to price data
    'smoothing_method': 'ewm',       # Smoothing method (ewm, sma, gaussian)
    'smoothing_window': 5,           # Window size for smoothing
}
```

#### Quality Configuration
```python
quality_config = {
    'auto_repair': True,              # Automatically repair data issues when possible
    'quality_threshold': 80,          # Minimum acceptable quality score
    'check_duplicate_data': True,     # Check for and remove duplicate rows
    'check_missing_values': True,     # Check for and fix missing values
    'check_price_outliers': True,     # Check for and fix price outliers
    'check_ohlc_integrity': True,     # Check for and fix OHLC integrity violations
    'check_data_gaps': True,          # Check for gaps in time series
    'check_timestamp_irregularities': True,  # Check for timestamp issues
    'check_stale_data': True,         # Check for stale/unchanged data
}
```

## Migration from Legacy Processors

To ensure backward compatibility while migrating to the new architecture, follow these steps:

1. First, use the new unified pipeline for new code:
```python
from trading_bot.data.data_pipeline import create_data_pipeline

pipeline = create_data_pipeline(config=your_config)
processed_data, metadata = pipeline.process(your_data, symbol, source)
```

2. For existing code using the legacy processors, use this compatibility approach:
```python
# If using the old DataCleaningProcessor directly
from trading_bot.data.data_pipeline import create_data_pipeline

pipeline = create_data_pipeline(config={'skip_quality': True})
cleaned_data, _ = pipeline.process(your_data, symbol, source)
```

This architecture maintains a clear separation between data transformation (cleaning) and validation (quality), while providing a unified interface for both operations.

## Event Integration

The data quality system integrates with the event system to provide real-time monitoring:

1. `EventType.DATA_QUALITY_WARNING`: Minor quality issues
2. `EventType.DATA_QUALITY_CRITICAL`: Critical quality issues requiring attention

To handle these events:
```python
def on_quality_warning(event):
    print(f"WARNING: Data quality issue with {event.data['symbol']}")
    
def on_quality_critical(event):
    send_alert(f"CRITICAL: Data issue with {event.data['symbol']}")
    
# Register handlers
event_bus.register(EventType.DATA_QUALITY_WARNING, on_quality_warning)
event_bus.register(EventType.DATA_QUALITY_CRITICAL, on_quality_critical)
```

## Data Quality Dashboard

A real-time data quality dashboard is available at:
- URL: `http://localhost:8050/data-quality`
- Update frequency: Every 5 minutes
- Historical view: Last 7 days of data quality metrics
