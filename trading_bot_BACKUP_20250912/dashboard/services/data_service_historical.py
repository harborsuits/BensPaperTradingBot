"""
Data Service Extensions for Historical Broker Performance

This module extends the DataService with methods to access
historical broker performance data for visualization components.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker


logger = logging.getLogger(__name__)


class BrokerHistoricalDataService:
    """
    Data service for historical broker performance data
    
    This class can be used standalone or as a mixin for the main DataService
    """
    
    def __init__(
        self,
        performance_tracker: Optional['BrokerPerformanceTracker'] = None,
        historical_data_path: str = "data/broker_performance",
        use_demo_data: bool = False
    ):
        """
        Initialize the historical data service
        
        Args:
            performance_tracker: Optional BrokerPerformanceTracker for real data
            historical_data_path: Path to historical data files
            use_demo_data: Whether to use generated demo data
        """
        self.performance_tracker = performance_tracker
        self.historical_data_path = historical_data_path
        self.use_demo_data = use_demo_data
        
        # Cache of broker information
        self._broker_cache = {}
    
    def get_broker_historical_performance(
        self,
        broker_id: str,
        start_time: datetime,
        end_time: datetime,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical performance data for a broker
        
        Args:
            broker_id: Broker ID
            start_time: Start time for data
            end_time: End time for data
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            metrics: Optional list of specific metrics to return
            
        Returns:
            List of performance data records
        """
        try:
            # If we have a performance tracker, use it
            if self.performance_tracker and not self.use_demo_data:
                # Get data from tracker
                analyzer = self.performance_tracker.get_analyzer()
                df = analyzer.store.get_as_dataframe(
                    broker_id=broker_id,
                    start_time=start_time,
                    end_time=end_time,
                    asset_class=asset_class,
                    operation_type=operation_type
                )
                
                # Filter to requested metrics if specified
                if metrics and df.shape[0] > 0:
                    # Include broker_id, timestamp, asset_class, operation_type
                    base_cols = [col for col in df.columns if col in ['broker_id', 'asset_class', 'operation_type']]
                    metric_cols = [col for col in metrics if col in df.columns]
                    df = df[base_cols + metric_cols]
                
                # Convert to records
                if df.shape[0] > 0:
                    # Reset index to make timestamp a column
                    df = df.reset_index()
                    return df.to_dict('records')
                else:
                    return []
            
            # Otherwise, generate demo data
            return self._generate_demo_historical_data(
                broker_id=broker_id,
                start_time=start_time,
                end_time=end_time,
                asset_class=asset_class,
                operation_type=operation_type,
                metrics=metrics
            )
                
        except Exception as e:
            logger.error(f"Error getting historical performance data: {str(e)}")
            return []
    
    def _generate_demo_historical_data(
        self,
        broker_id: str,
        start_time: datetime,
        end_time: datetime,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate demo historical data for development and testing
        
        Args:
            broker_id: Broker ID
            start_time: Start time for data
            end_time: End time for data
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            metrics: Optional list of specific metrics to return
            
        Returns:
            List of generated performance data records
        """
        # Determine time interval based on duration
        duration = end_time - start_time
        
        if duration.total_seconds() < 3600:  # Less than 1 hour
            interval = timedelta(minutes=1)
        elif duration.total_seconds() < 86400:  # Less than 1 day
            interval = timedelta(minutes=5)
        elif duration.total_seconds() < 604800:  # Less than 7 days
            interval = timedelta(minutes=30)
        else:
            interval = timedelta(hours=1)
        
        # Create timestamp range
        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += interval
        
        # Generate data
        data = []
        
        # Define base characteristics for this broker
        # This ensures consistent broker personality across calls
        import hashlib
        broker_seed = int(hashlib.md5(broker_id.encode()).hexdigest(), 16) % 10000
        np.random.seed(broker_seed)
        
        # Base parameters that define this broker's characteristics
        base_latency = 100 + np.random.randint(-50, 150)  # 50-250ms
        base_reliability = 98 + np.random.random() * 2  # 98-100%
        base_slippage = 0.05 + np.random.random() * 0.15  # 0.05-0.2%
        base_commission = 1 + np.random.random() * 4  # $1-5
        
        # Variability factors
        latency_variability = 0.2 + np.random.random() * 0.3  # 20-50%
        reliability_variability = 0.01 + np.random.random() * 0.03  # 1-4%
        slippage_variability = 0.3 + np.random.random() * 0.3  # 30-60%
        commission_variability = 0.05 + np.random.random() * 0.15  # 5-20%
        
        # Reset seed for time series generation
        np.random.seed()
        
        # Asset class adjustments
        if asset_class == "forex":
            base_latency *= 0.8  # Forex typically faster
            base_slippage *= 1.2  # But with higher slippage
        elif asset_class == "options":
            base_latency *= 1.2  # Options typically slower
            base_commission *= 1.5  # And more expensive
        
        # Generate time series with realistic patterns
        for i, timestamp in enumerate(timestamps):
            # Sine wave component for time-of-day effects (24h cycle)
            hour_of_day = timestamp.hour + timestamp.minute / 60.0
            time_factor = np.sin(hour_of_day * 2 * np.pi / 24) * 0.2 + 1.0
            
            # Day of week factor (weekends worse)
            day_of_week = timestamp.weekday()
            weekend_factor = 1.2 if day_of_week >= 5 else 1.0
            
            # Gradual trend component (improve or degrade over time)
            trend_direction = 1 if broker_seed % 2 == 0 else -1
            trend_factor = 1.0 + (i / len(timestamps)) * 0.1 * trend_direction
            
            # Random noise component
            noise = np.random.normal(1.0, 0.1)
            
            # Occasional spikes/anomalies
            spike = 1.0
            if np.random.random() < 0.02:  # 2% chance of spike
                spike = np.random.choice([1.5, 2.0, 3.0])
            
            # Calculate metrics with all factors
            combined_factor = time_factor * weekend_factor * trend_factor * noise * spike
            
            latency = base_latency * (1 + (combined_factor - 1) * latency_variability)
            reliability = max(0, min(100, base_reliability - (combined_factor - 1) * 100 * reliability_variability))
            slippage = base_slippage * (1 + (combined_factor - 1) * slippage_variability)
            commission = base_commission * (1 + (combined_factor - 1) * commission_variability)
            
            # Create record
            record = {
                "timestamp": timestamp,
                "broker_id": broker_id,
                "latency_mean_ms": latency,
                "reliability_availability": reliability,
                "reliability_errors": max(0, min(10, 10 - reliability / 10)),
                "execution_quality_avg_slippage_pct": slippage,
                "cost_avg_commission": commission,
                "score_overall": max(0, min(100, 100 - (latency / 1000 * 20 + (100 - reliability) * 2 + slippage * 100 + commission * 2))),
                "score_latency": max(0, min(100, 100 - latency / 5)),
                "score_reliability": reliability,
                "score_execution_quality": max(0, min(100, 100 - slippage * 200)),
                "score_cost": max(0, min(100, 100 - commission * 5))
            }
            
            # Add asset class and operation type if provided
            if asset_class:
                record["asset_class"] = asset_class
            
            if operation_type:
                record["operation_type"] = operation_type
            
            data.append(record)
        
        return data
    
    def get_historical_metrics_list(self) -> List[str]:
        """
        Get list of available historical metrics
        
        Returns:
            List of metric names
        """
        return [
            "latency_mean_ms",
            "reliability_availability",
            "reliability_errors",
            "execution_quality_avg_slippage_pct",
            "cost_avg_commission",
            "score_overall",
            "score_latency",
            "score_reliability",
            "score_execution_quality",
            "score_cost"
        ]
    
    def get_broker_performance_summary(
        self,
        broker_id: str,
        lookback_days: int = 7,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary for a broker
        
        Args:
            broker_id: Broker ID
            lookback_days: Number of days to look back
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Dict with summary statistics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        try:
            # Get historical data
            history = self.get_broker_historical_performance(
                broker_id=broker_id,
                start_time=start_time,
                end_time=end_time,
                asset_class=asset_class,
                operation_type=operation_type
            )
            
            if not history:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            
            # Calculate summary statistics
            summary = {
                "broker_id": broker_id,
                "metrics": {},
                "scores": {},
                "trends": {}
            }
            
            # Process metrics
            metrics = [
                "latency_mean_ms",
                "reliability_availability",
                "reliability_errors",
                "execution_quality_avg_slippage_pct",
                "cost_avg_commission"
            ]
            
            for metric in metrics:
                if metric in df.columns:
                    summary["metrics"][metric] = {
                        "mean": float(df[metric].mean()),
                        "median": float(df[metric].median()),
                        "min": float(df[metric].min()),
                        "max": float(df[metric].max()),
                        "std": float(df[metric].std()),
                        "latest": float(df[metric].iloc[-1]) if len(df) > 0 else None
                    }
            
            # Process scores
            scores = [
                "score_overall",
                "score_latency",
                "score_reliability",
                "score_execution_quality",
                "score_cost"
            ]
            
            for score in scores:
                if score in df.columns:
                    summary["scores"][score] = {
                        "current": float(df[score].iloc[-1]) if len(df) > 0 else None,
                        "mean": float(df[score].mean()),
                        "min": float(df[score].min()),
                        "max": float(df[score].max())
                    }
            
            # Calculate trends (simple linear regression)
            for metric in metrics + scores:
                if metric in df.columns and len(df) > 5:
                    # Simple linear regression
                    x = np.arange(len(df))
                    y = df[metric].values
                    
                    # Calculate trend line
                    z = np.polyfit(x, y, 1)
                    slope = z[0]
                    
                    # Normalize to percentage
                    if df[metric].mean() != 0:
                        pct_change = (slope * len(df) / df[metric].mean()) * 100
                    else:
                        pct_change = 0
                    
                    # Determine if improving or degrading
                    # For some metrics (like latency, errors), decreasing is better
                    # For others (like availability, scores), increasing is better
                    if metric in ["latency_mean_ms", "reliability_errors", "execution_quality_avg_slippage_pct", "cost_avg_commission"]:
                        improving = slope < 0
                    else:
                        improving = slope > 0
                    
                    summary["trends"][metric] = {
                        "slope": float(slope),
                        "pct_change": float(pct_change),
                        "improving": improving
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}


# Example of extending the main DataService
def extend_data_service_with_historical(DataService):
    """
    Extend the DataService class with historical methods
    
    Args:
        DataService: The DataService class to extend
        
    Returns:
        Extended DataService class
    """
    
    class ExtendedDataService(DataService, BrokerHistoricalDataService):
        """DataService extended with historical broker performance methods"""
        
        def __init__(self, *args, **kwargs):
            """Initialize both parent classes"""
            DataService.__init__(self, *args, **kwargs)
            
            # Get performance_tracker from kwargs or create dummy
            performance_tracker = kwargs.get('performance_tracker')
            historical_data_path = kwargs.get('historical_data_path', "data/broker_performance")
            use_demo_data = kwargs.get('use_demo_data', True)
            
            BrokerHistoricalDataService.__init__(
                self,
                performance_tracker=performance_tracker,
                historical_data_path=historical_data_path,
                use_demo_data=use_demo_data
            )
    
    return ExtendedDataService
