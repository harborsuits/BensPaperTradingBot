# Data Quality Assurance System

## Overview

The Data Quality Assurance system provides comprehensive monitoring, validation, and automatic repair of market data throughout the trading platform. It integrates seamlessly with the existing event-driven architecture to ensure high-quality data for strategy execution.

This system addresses the previously identified gaps in data quality handling:
- Comprehensive duplicate detection and handling
- Missing data detection with smart interpolation
- Price and volume outlier detection
- OHLC integrity validation
- Time series gap detection and filling
- Stale data identification
- Timestamp irregularity detection

## Architecture

The system consists of four main components:

1. **DataQualityManager**: Core component responsible for coordinating quality checks and reporting
2. **DataQualityChecks**: Collection of individual quality validation functions
3. **DataQualityMetrics**: Tracking and aggregation of quality metrics across the system
4. **DataQualityProcessor**: Integration with the existing data processing pipeline

## Integration with Event System

The quality system publishes events when data quality issues are detected:

- `EventType.DATA_QUALITY_CRITICAL`: Severe quality issues requiring immediate attention
- `EventType.DATA_QUALITY_WARNING`: Notable quality issues that should be monitored

These events can trigger alerts, logging, or automated recovery actions.

## Integration with Persistence Layer

When connected to the persistence layer, the system stores:

- Quality metrics for each data check in `data_quality_metrics` collection
- Detailed reports for critical issues in `data_quality_issues` collection
- Historical quality trends for ongoing analysis

## Usage Example

### Basic Setup

```python
from trading_bot.core.event_system import EventBus
from trading_bot.data.persistence import PersistenceManager
from trading_bot.data.quality.integration import setup_data_quality_system

# Create dependencies
event_bus = EventBus()
persistence = PersistenceManager(config.get('mongodb_uri'), config.get('mongodb_database'))

# Configure quality system
quality_config = {
    'duplicate_threshold': 0.01,        # 1% duplicates tolerated
    'missing_data_threshold': 0.05,     # 5% missing data tolerated
    'outlier_threshold': 0.02,          # 2% outliers tolerated
    'critical_quality_threshold': 60,   # Min acceptable quality score
    'auto_repair': True                 # Automatically fix issues
}

# Initialize the quality system
quality_processor = setup_data_quality_system(
    event_bus=event_bus,
    persistence=persistence,
    config=quality_config
)
```

### Integrating into Data Pipeline

```python
from trading_bot.data.pipeline import DataPipeline
from trading_bot.data.processors.data_cleaning_processor import DataCleaningProcessor
from trading_bot.data.quality.integration import DataQualityProcessor

# Create data pipeline with quality processor
pipeline = DataPipeline(name="MarketDataPipeline")

# Add processors in order
pipeline.add_processor(DataCleaningProcessor())
pipeline.add_processor(DataQualityProcessor(
    event_bus=event_bus, 
    persistence=persistence
))

# Process data through the pipeline
cleaned_df = pipeline.process(raw_data, symbol="EURUSD", source="alpha_vantage")
```

### Manual Quality Check

```python
from trading_bot.data.quality.data_quality_manager import DataQualityManager

# Create quality manager
quality_manager = DataQualityManager(config=quality_config, event_bus=event_bus)

# Run quality check on a DataFrame
cleaned_df, quality_report = quality_manager.check_data_quality(
    df=market_data, 
    symbol="EURUSD", 
    source="alpha_vantage"
)

# Inspect quality report
print(f"Quality score: {quality_report['quality_score']}")
print(f"Issues found: {len(quality_report['issues'])}")
print(f"Issues fixed: {len(quality_report['fixed_issues'])}")
```

### Generating Quality Reports

```python
# Generate a JSON quality report
json_report = quality_processor.generate_quality_report(output_format="json")

# Generate an HTML quality report
html_report = quality_processor.generate_quality_report(output_format="html")
```

## Integration with Dashboard

The data quality system can be monitored through the trading dashboard by adding a new tab or section:

```python
# In dashboard code
def data_quality_page():
    st.header("Data Quality Monitor")
    
    # Fetch quality metrics from API
    quality_metrics = get_data_quality_metrics()
    
    # Display overall quality score
    st.metric("System Quality Score", f"{quality_metrics['average_quality_score']:.1f}/100")
    
    # Display metrics by data source
    st.subheader("Quality by Data Source")
    source_df = pd.DataFrame(quality_metrics['sources'])
    st.dataframe(source_df)
    
    # Display recent issues
    st.subheader("Recent Issues")
    issues_df = pd.DataFrame(quality_metrics['recent_issues'])
    st.dataframe(issues_df)
```

## Benefits Over Previous Implementation

This system builds upon and enhances the existing `DataCleaningProcessor` with these improvements:

1. **Comprehensive Checks**: More thorough validation including time series gaps and stale data detection
2. **Event Integration**: Publishes quality events for system-wide monitoring
3. **Quality Metrics**: Tracks and reports on data quality over time
4. **Persistence**: Stores quality metrics for historical analysis
5. **Auto-repair**: Intelligent fixing of detected issues where appropriate
6. **Reporting**: Generates detailed quality reports for operational oversight

The implementation follows the event-driven pattern successfully used in other components of the system, ensuring consistency and maintainability.
