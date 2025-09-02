"""
Broker Performance Historical Tracker - Part 2

Contains the main BrokerPerformanceTracker class and trend analysis functionality.
This complements the historical_tracker.py file with the storage implementations.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from threading import Lock, Thread
from time import sleep

from trading_bot.event_system.event_bus import EventBus
from trading_bot.event_system.event_types import EventType
from trading_bot.brokers.metrics.base import MetricType, MetricOperation
from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceRecord, SQLiteTimeSeriesStore, CSVTimeSeriesStore


logger = logging.getLogger(__name__)


class BrokerPerformanceAnalyzer:
    """Analyze broker performance data for trends and anomalies"""
    
    def __init__(self, time_series_store: Union[SQLiteTimeSeriesStore, CSVTimeSeriesStore]):
        """
        Initialize analyzer
        
        Args:
            time_series_store: Storage for time series data
        """
        self.store = time_series_store
    
    def calculate_moving_averages(
        self,
        broker_id: str,
        metric_name: str,
        windows: List[int] = [5, 20, 50],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate moving averages for a metric
        
        Args:
            broker_id: Broker ID
            metric_name: Name of metric to analyze
            windows: List of window sizes for moving averages
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            DataFrame with original metric and moving averages
        """
        # Get data as DataFrame
        df = self.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty or metric_name not in df.columns:
            logger.warning(f"No data found for metric {metric_name}")
            return pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Calculate moving averages
        for window in windows:
            df[f'{metric_name}_ma{window}'] = df[metric_name].rolling(window=window).mean()
        
        return df
    
    def detect_anomalies(
        self,
        broker_id: str,
        metric_name: str,
        zscore_threshold: float = 3.0,
        lookback_periods: int = 50,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies in broker metrics using z-score method
        
        Args:
            broker_id: Broker ID
            metric_name: Name of metric to analyze
            zscore_threshold: Z-score threshold for anomaly detection
            lookback_periods: Periods to use for calculating mean and std
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            DataFrame with original metric and anomaly flags
        """
        # Get data as DataFrame
        df = self.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty or metric_name not in df.columns:
            logger.warning(f"No data found for metric {metric_name}")
            return pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Calculate rolling mean and std
        df[f'{metric_name}_mean'] = df[metric_name].rolling(window=lookback_periods).mean()
        df[f'{metric_name}_std'] = df[metric_name].rolling(window=lookback_periods).std()
        
        # Calculate z-scores
        df[f'{metric_name}_zscore'] = (df[metric_name] - df[f'{metric_name}_mean']) / df[f'{metric_name}_std']
        
        # Flag anomalies
        df[f'{metric_name}_anomaly'] = np.abs(df[f'{metric_name}_zscore']) > zscore_threshold
        
        return df
    
    def identify_seasonality(
        self,
        broker_id: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        freq: str = 'H'  # 'H' for hourly, 'D' for daily, 'W' for weekly
    ) -> pd.DataFrame:
        """
        Identify seasonality patterns in broker metrics
        
        Args:
            broker_id: Broker ID
            metric_name: Name of metric to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            freq: Frequency for seasonality analysis
            
        Returns:
            DataFrame with seasonality patterns
        """
        # Get data as DataFrame
        df = self.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty or metric_name not in df.columns:
            logger.warning(f"No data found for metric {metric_name}")
            return pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Extract seasonality components
        if freq == 'H':
            # Hourly patterns
            df['hour'] = df.index.hour
            seasonal = df.groupby('hour')[metric_name].mean()
            return pd.DataFrame({
                'hour': seasonal.index,
                f'{metric_name}_avg': seasonal.values
            })
            
        elif freq == 'D':
            # Daily patterns
            df['day'] = df.index.day_name()
            seasonal = df.groupby('day')[metric_name].mean()
            
            # Reorder days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            seasonal = seasonal.reindex(days_order)
            
            return pd.DataFrame({
                'day': seasonal.index,
                f'{metric_name}_avg': seasonal.values
            })
            
        elif freq == 'W':
            # Weekly patterns
            df['week'] = df.index.isocalendar().week
            seasonal = df.groupby('week')[metric_name].mean()
            return pd.DataFrame({
                'week': seasonal.index,
                f'{metric_name}_avg': seasonal.values
            })
        
        return pd.DataFrame()
    
    def forecast_trend(
        self,
        broker_id: str,
        metric_name: str,
        forecast_periods: int = 10,
        lookback_periods: int = 50,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple forecasting of broker metrics using linear trend
        
        Args:
            broker_id: Broker ID
            metric_name: Name of metric to analyze
            forecast_periods: Number of periods to forecast
            lookback_periods: Number of periods to use for forecasting
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Tuple with (historical data, forecast data)
        """
        # Get data as DataFrame
        df = self.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty or metric_name not in df.columns:
            logger.warning(f"No data found for metric {metric_name}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Get the most recent data points for forecasting
        recent_data = df.iloc[-lookback_periods:] if len(df) >= lookback_periods else df
        
        # Simple linear regression for forecasting
        import statsmodels.api as sm
        
        # Add constant for intercept
        X = sm.add_constant(np.arange(len(recent_data)))
        y = recent_data[metric_name].values
        
        # Fit linear model
        model = sm.OLS(y, X).fit()
        
        # Generate future X values
        X_future = sm.add_constant(np.arange(len(recent_data), len(recent_data) + forecast_periods))
        
        # Predict
        y_future = model.predict(X_future)
        
        # Create forecast DataFrame
        last_time = recent_data.index[-1]
        future_index = pd.date_range(
            start=last_time + pd.Timedelta(seconds=1),
            periods=forecast_periods,
            freq=pd.infer_freq(recent_data.index)
        )
        
        forecast_df = pd.DataFrame({
            metric_name: y_future
        }, index=future_index)
        
        return recent_data, forecast_df
    
    def get_performance_summary(
        self,
        broker_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary statistics
        
        Args:
            broker_id: Broker ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Dictionary with summary statistics
        """
        # Get data as DataFrame
        df = self.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty:
            logger.warning(f"No data found for broker {broker_id}")
            return {}
        
        # Identify metric columns
        metric_cols = [col for col in df.columns if col not in ['broker_id', 'asset_class', 'operation_type']]
        
        # Calculate summary statistics
        summary = {}
        for col in metric_cols:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                summary[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'latest': df[col].iloc[-1] if not df[col].empty else None,
                    'trend': 'improving' if df[col].iloc[-1] > df[col].mean() else 'degrading'
                }
        
        return summary


class BrokerPerformanceTracker:
    """
    Track and analyze historical broker performance
    
    Records performance metrics at regular intervals and
    provides analysis tools for trend detection and visualization.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        storage_type: str = 'sqlite',
        storage_path: str = 'data/broker_performance',
        sampling_interval: int = 300,  # 5 minutes
        retention_days: int = 90
    ):
        """
        Initialize the performance tracker
        
        Args:
            event_bus: Event bus for subscribing to broker events
            storage_type: Storage type ('sqlite' or 'csv')
            storage_path: Path to storage location
            sampling_interval: Interval in seconds between recordings
            retention_days: Days to retain historical data
        """
        self.event_bus = event_bus
        self.sampling_interval = sampling_interval
        self.retention_days = retention_days
        
        # Set up storage
        if storage_type == 'sqlite':
            if not storage_path.endswith('.db'):
                storage_path = os.path.join(storage_path, 'broker_performance.db')
            self.storage = SQLiteTimeSeriesStore(storage_path)
        else:
            self.storage = CSVTimeSeriesStore(storage_path)
        
        # Create analyzer
        self.analyzer = BrokerPerformanceAnalyzer(self.storage)
        
        # Recording state
        self.recording_thread = None
        self.stop_recording = False
        self.lock = Lock()
        
        # Metrics cache
        self.metrics_cache = {}
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info(f"BrokerPerformanceTracker initialized with {storage_type} storage at {storage_path}")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant broker events"""
        # Subscribe to broker metrics events
        self.event_bus.subscribe(
            event_type=EventType.BROKER_METRICS,
            handler=self._handle_broker_metrics_event
        )
        
        # Subscribe to broker intelligence events
        self.event_bus.subscribe(
            event_type=EventType.BROKER_INTELLIGENCE,
            handler=self._handle_broker_intelligence_event
        )
    
    def _handle_broker_metrics_event(self, event: Dict[str, Any]):
        """Handle broker metrics event"""
        broker_id = event.get("broker_id")
        if not broker_id:
            return
        
        # Update metrics cache
        with self.lock:
            if broker_id not in self.metrics_cache:
                self.metrics_cache[broker_id] = {}
            
            # Update metrics
            metrics = event.get("metrics", {})
            asset_class = event.get("asset_class")
            operation_type = event.get("operation_type")
            
            # Create cache key
            cache_key = (asset_class, operation_type)
            
            # Update cache
            if cache_key not in self.metrics_cache[broker_id]:
                self.metrics_cache[broker_id][cache_key] = {}
            
            self.metrics_cache[broker_id][cache_key].update(metrics)
    
    def _handle_broker_intelligence_event(self, event: Dict[str, Any]):
        """Handle broker intelligence event"""
        event_subtype = event.get("event_subtype")
        broker_id = event.get("broker_id")
        
        if not broker_id:
            return
        
        # Record performance scores
        if event_subtype == "broker_performance_scored":
            scores = event.get("scores", {})
            asset_class = event.get("asset_class")
            operation_type = event.get("operation_type")
            
            # Store scores with next record
            with self.lock:
                if broker_id not in self.metrics_cache:
                    self.metrics_cache[broker_id] = {}
                
                cache_key = (asset_class, operation_type)
                
                if cache_key not in self.metrics_cache[broker_id]:
                    self.metrics_cache[broker_id][cache_key] = {}
                
                if "scores" not in self.metrics_cache[broker_id][cache_key]:
                    self.metrics_cache[broker_id][cache_key]["scores"] = {}
                
                self.metrics_cache[broker_id][cache_key]["scores"].update(scores)
    
    def start_recording(self):
        """Start periodic recording of broker performance"""
        if self.recording_thread is not None:
            logger.warning("Recording thread already running")
            return
        
        self.stop_recording = False
        self.recording_thread = Thread(
            target=self._recording_loop,
            daemon=True,
            name="BrokerPerformanceRecordingThread"
        )
        self.recording_thread.start()
        
        logger.info(f"Started broker performance recording every {self.sampling_interval} seconds")
    
    def stop_recording(self):
        """Stop periodic recording"""
        if self.recording_thread is None:
            logger.warning("Recording thread not running")
            return
        
        self.stop_recording = True
        self.recording_thread.join(timeout=5.0)
        self.recording_thread = None
        
        logger.info("Stopped broker performance recording")
    
    def _recording_loop(self):
        """Background thread for periodic recording"""
        while not self.stop_recording:
            try:
                # Record current metrics
                self._record_current_metrics()
                
                # Periodically prune old records
                self._prune_old_records()
                
                # Sleep until next recording
                sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in performance recording loop: {str(e)}")
                sleep(30)  # Sleep briefly on error
    
    def _record_current_metrics(self):
        """Record current metrics from cache"""
        with self.lock:
            # Make a copy of the cache to avoid modification during iteration
            metrics_snapshot = self.metrics_cache.copy()
        
        # Current timestamp
        now = datetime.now()
        
        # Record metrics for each broker and context
        for broker_id, contexts in metrics_snapshot.items():
            for (asset_class, operation_type), metrics in contexts.items():
                # Get scores if available
                scores = metrics.pop("scores", None)
                
                # Create record
                record = BrokerPerformanceRecord(
                    broker_id=broker_id,
                    timestamp=now,
                    metrics=metrics,
                    asset_class=asset_class,
                    operation_type=operation_type,
                    scores=scores
                )
                
                # Store record
                self.storage.store_record(record)
    
    def _prune_old_records(self):
        """Prune old records to maintain retention policy"""
        if hasattr(self.storage, 'prune_old_records'):
            count = self.storage.prune_old_records(self.retention_days)
            if count > 0:
                logger.info(f"Pruned {count} old performance records")
    
    def record_metrics_snapshot(
        self,
        broker_id: str,
        metrics: Dict[str, Any],
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None
    ):
        """
        Manually record a metrics snapshot
        
        Args:
            broker_id: Broker ID
            metrics: Metrics data
            asset_class: Optional asset class context
            operation_type: Optional operation type context
            scores: Optional performance scores
        """
        # Create record
        record = BrokerPerformanceRecord(
            broker_id=broker_id,
            timestamp=datetime.now(),
            metrics=metrics,
            asset_class=asset_class,
            operation_type=operation_type,
            scores=scores
        )
        
        # Store record
        self.storage.store_record(record)
    
    def get_analyzer(self) -> BrokerPerformanceAnalyzer:
        """Get the performance analyzer"""
        return self.analyzer
    
    def close(self):
        """Close the tracker and storage"""
        # Stop recording
        self.stop_recording = True
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        
        # Close storage
        if hasattr(self.storage, 'close'):
            self.storage.close()
        
        logger.info("Broker performance tracker closed")
