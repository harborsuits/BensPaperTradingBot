"""
Regime Performance Tracker

This module tracks the performance of trading strategies across different market regimes,
enabling the system to identify which strategies excel in specific market conditions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
import json
import os

# Import local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeType

logger = logging.getLogger(__name__)

class RegimePerformanceTracker:
    """
    Tracks and analyzes strategy performance across different market regimes.
    
    Maintains a performance database that correlates trading results with market regimes,
    enabling the system to:
    - Identify which strategies perform best in each regime
    - Detect strategy drift or degradation
    - Inform capital allocation decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance tracker.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Directory to store performance data
        self.performance_dir = self.config.get("performance_dir", "data/regime_performance")
        
        # Performance database
        # Structure: {strategy_id: {regime_type: {metric_name: [historical values]}}}
        self.performance_history: Dict[str, Dict[MarketRegimeType, Dict[str, List[float]]]] = {}
        
        # Time-series performance data
        self.time_series_data: Dict[str, Dict[MarketRegimeType, List[Dict[str, Any]]]] = {}
        
        # Performance metrics to track
        self.tracked_metrics = self.config.get("tracked_metrics", [
            "win_rate", "profit_factor", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "avg_profit", "avg_loss", "expectancy"
        ])
        
        # Statistical analysis cache
        self.performance_stats: Dict[str, Dict[MarketRegimeType, Dict[str, float]]] = {}
        
        # Recent performance by timestamp
        self.recent_performance: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._load_performance_data()
        
        logger.info("Regime Performance Tracker initialized")
    
    def _load_performance_data(self) -> None:
        """Load performance data from disk."""
        try:
            # Create performance directory if it doesn't exist
            os.makedirs(self.performance_dir, exist_ok=True)
            
            # Check for performance files
            for strategy_file in os.listdir(self.performance_dir):
                if strategy_file.endswith(".json") and strategy_file.startswith("performance_"):
                    strategy_id = strategy_file[12:-5]  # Remove "performance_" prefix and ".json" suffix
                    
                    file_path = os.path.join(self.performance_dir, strategy_file)
                    with open(file_path, 'r') as f:
                        performance_data = json.load(f)
                    
                    # Convert string keys to MarketRegimeType
                    regime_performance = {}
                    for regime_str, metrics in performance_data.get('metrics', {}).items():
                        try:
                            regime_type = MarketRegimeType(regime_str)
                            regime_performance[regime_type] = metrics
                        except ValueError:
                            logger.warning(f"Unknown regime type in performance file: {regime_str}")
                    
                    # Load historical time series data
                    time_series = {}
                    for regime_str, series in performance_data.get('time_series', {}).items():
                        try:
                            regime_type = MarketRegimeType(regime_str)
                            # Convert string timestamps to datetime
                            for entry in series:
                                if 'timestamp' in entry:
                                    entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                            time_series[regime_type] = series
                        except ValueError:
                            logger.warning(f"Unknown regime type in time series: {regime_str}")
                    
                    self.performance_history[strategy_id] = regime_performance
                    self.time_series_data[strategy_id] = time_series
            
            loaded_count = len(self.performance_history)
            if loaded_count > 0:
                logger.info(f"Loaded performance data for {loaded_count} strategies")
            else:
                logger.info("No performance data found")
                
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
    
    def _save_performance_data(self, strategy_id: str) -> None:
        """
        Save performance data to disk.
        
        Args:
            strategy_id: Strategy to save data for
        """
        try:
            # Check if we have data to save
            if strategy_id not in self.performance_history and strategy_id not in self.time_series_data:
                return
            
            # Create performance directory if it doesn't exist
            os.makedirs(self.performance_dir, exist_ok=True)
            
            file_path = os.path.join(self.performance_dir, f"performance_{strategy_id}.json")
            
            # Prepare data for serialization
            performance_data = {'metrics': {}, 'time_series': {}}
            
            # Convert MarketRegimeType to strings for metrics
            if strategy_id in self.performance_history:
                for regime, metrics in self.performance_history[strategy_id].items():
                    performance_data['metrics'][regime.value] = metrics
            
            # Convert MarketRegimeType to strings for time series
            if strategy_id in self.time_series_data:
                for regime, series in self.time_series_data[strategy_id].items():
                    # Convert datetime objects to iso format strings
                    serializable_series = []
                    for entry in series:
                        serializable_entry = entry.copy()
                        if 'timestamp' in serializable_entry:
                            if isinstance(serializable_entry['timestamp'], datetime):
                                serializable_entry['timestamp'] = serializable_entry['timestamp'].isoformat()
                        serializable_series.append(serializable_entry)
                    
                    performance_data['time_series'][regime.value] = serializable_series
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logger.debug(f"Saved performance data for strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
    
    def update_performance(
        self, strategy_id: str, regime_type: MarketRegimeType,
        performance_metrics: Dict[str, Any], symbol: str, timeframe: str
    ) -> None:
        """
        Update performance metrics for a strategy in a specific market regime.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Market regime type
            performance_metrics: Performance metrics
            symbol: Symbol being traded
            timeframe: Timeframe being traded
        """
        with self._lock:
            try:
                # Initialize if needed
                if strategy_id not in self.performance_history:
                    self.performance_history[strategy_id] = {}
                
                if strategy_id not in self.time_series_data:
                    self.time_series_data[strategy_id] = {}
                
                if regime_type not in self.performance_history[strategy_id]:
                    self.performance_history[strategy_id][regime_type] = {}
                
                if regime_type not in self.time_series_data[strategy_id]:
                    self.time_series_data[strategy_id][regime_type] = []
                
                # Update metrics history
                for metric_name, metric_value in performance_metrics.items():
                    if metric_name in self.tracked_metrics:
                        if metric_name not in self.performance_history[strategy_id][regime_type]:
                            self.performance_history[strategy_id][regime_type][metric_name] = []
                        
                        # Convert to float and add to history
                        try:
                            float_value = float(metric_value)
                            self.performance_history[strategy_id][regime_type][metric_name].append(float_value)
                        except (ValueError, TypeError):
                            # Skip non-numeric metrics
                            pass
                
                # Limit history size
                max_history = self.config.get("max_metric_history", 100)
                for metric_name in self.performance_history[strategy_id][regime_type]:
                    if len(self.performance_history[strategy_id][regime_type][metric_name]) > max_history:
                        self.performance_history[strategy_id][regime_type][metric_name] = \
                            self.performance_history[strategy_id][regime_type][metric_name][-max_history:]
                
                # Add time series entry
                time_entry = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'metrics': {k: v for k, v in performance_metrics.items() if k in self.tracked_metrics}
                }
                
                self.time_series_data[strategy_id][regime_type].append(time_entry)
                
                # Limit time series size
                max_time_series = self.config.get("max_time_series", 500)
                if len(self.time_series_data[strategy_id][regime_type]) > max_time_series:
                    self.time_series_data[strategy_id][regime_type] = \
                        self.time_series_data[strategy_id][regime_type][-max_time_series:]
                
                # Update recent performance
                if strategy_id not in self.recent_performance:
                    self.recent_performance[strategy_id] = {}
                
                recent_key = f"{symbol}_{timeframe}"
                if recent_key not in self.recent_performance[strategy_id]:
                    self.recent_performance[strategy_id][recent_key] = []
                
                recent_entry = {
                    'timestamp': datetime.now(),
                    'regime_type': regime_type,
                    'metrics': {k: v for k, v in performance_metrics.items() if k in self.tracked_metrics}
                }
                
                self.recent_performance[strategy_id][recent_key].append(recent_entry)
                
                # Limit recent performance size
                max_recent = self.config.get("max_recent_entries", 20)
                if len(self.recent_performance[strategy_id][recent_key]) > max_recent:
                    self.recent_performance[strategy_id][recent_key] = \
                        self.recent_performance[strategy_id][recent_key][-max_recent:]
                
                # Clear stats cache since data has changed
                if strategy_id in self.performance_stats and regime_type in self.performance_stats[strategy_id]:
                    del self.performance_stats[strategy_id][regime_type]
                
                # Save to disk
                self._save_performance_data(strategy_id)
                
            except Exception as e:
                logger.error(f"Error updating performance metrics: {str(e)}")
    
    def get_performance_by_regime(self, strategy_id: str) -> Dict[MarketRegimeType, Dict[str, Any]]:
        """
        Get performance statistics by regime for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict mapping regime types to performance statistics
        """
        if strategy_id not in self.performance_history:
            return {}
        
        # Calculate statistics for each regime
        result = {}
        
        for regime_type, metrics in self.performance_history[strategy_id].items():
            # Check if we have cached stats
            if strategy_id in self.performance_stats and regime_type in self.performance_stats[strategy_id]:
                result[regime_type] = self.performance_stats[strategy_id].copy()
                continue
            
            # Calculate statistics
            regime_stats = {}
            
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    regime_stats[f"{metric_name}_mean"] = float(np.mean(values))
                    regime_stats[f"{metric_name}_median"] = float(np.median(values))
                    regime_stats[f"{metric_name}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
                    regime_stats[f"{metric_name}_min"] = float(np.min(values))
                    regime_stats[f"{metric_name}_max"] = float(np.max(values))
                    regime_stats[f"{metric_name}_latest"] = float(values[-1])
                    
                    # Calculate trend (slope of linear regression)
                    if len(values) > 2:
                        x = np.arange(len(values))
                        slope, _ = np.polyfit(x, values, 1)
                        regime_stats[f"{metric_name}_trend"] = float(slope)
                    else:
                        regime_stats[f"{metric_name}_trend"] = 0.0
            
            # Add sample size
            sample_sizes = [len(values) for values in metrics.values()]
            regime_stats["sample_size"] = min(sample_sizes) if sample_sizes else 0
            
            # Add to result
            result[regime_type] = regime_stats
            
            # Cache statistics
            if strategy_id not in self.performance_stats:
                self.performance_stats[strategy_id] = {}
            self.performance_stats[strategy_id][regime_type] = regime_stats
        
        return result
    
    def get_best_strategies_for_regime(
        self, regime_type: MarketRegimeType, metric_name: str = "profit_factor", 
        min_sample_size: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get best performing strategies for a specific regime.
        
        Args:
            regime_type: Market regime type
            metric_name: Metric to rank by
            min_sample_size: Minimum sample size required
            
        Returns:
            List of (strategy_id, metric_value) tuples sorted by performance
        """
        results = []
        
        for strategy_id in self.performance_history:
            if regime_type in self.performance_history[strategy_id]:
                metrics = self.performance_history[strategy_id][regime_type]
                
                # Check if metric exists and has enough samples
                if metric_name in metrics and len(metrics[metric_name]) >= min_sample_size:
                    # Use average of recent values
                    recent_values = metrics[metric_name][-min_sample_size:]
                    avg_value = float(np.mean(recent_values))
                    
                    results.append((strategy_id, avg_value))
        
        # Sort by metric value (descending)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_strategy_regime_ranking(self, strategy_id: str, metric_name: str = "profit_factor") -> List[Tuple[MarketRegimeType, float]]:
        """
        Get ranking of regimes for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            metric_name: Metric to rank by
            
        Returns:
            List of (regime_type, metric_value) tuples sorted by performance
        """
        if strategy_id not in self.performance_history:
            return []
        
        results = []
        
        for regime_type, metrics in self.performance_history[strategy_id].items():
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                # Use average of all values
                avg_value = float(np.mean(metrics[metric_name]))
                results.append((regime_type, avg_value))
        
        # Sort by metric value (descending)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_performance_time_series(
        self, strategy_id: str, regime_type: MarketRegimeType,
        metric_name: str, limit: Optional[int] = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get time series of a performance metric.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Market regime type
            metric_name: Metric to get time series for
            limit: Optional limit on number of points
            
        Returns:
            List of (timestamp, value) tuples
        """
        if strategy_id not in self.time_series_data or \
           regime_type not in self.time_series_data[strategy_id]:
            return []
        
        series = []
        
        for entry in self.time_series_data[strategy_id][regime_type]:
            if 'timestamp' in entry and 'metrics' in entry and metric_name in entry['metrics']:
                try:
                    value = float(entry['metrics'][metric_name])
                    timestamp = entry['timestamp']
                    series.append((timestamp, value))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    pass
        
        # Sort by timestamp
        series.sort(key=lambda x: x[0])
        
        # Apply limit if specified
        if limit is not None and len(series) > limit:
            return series[-limit:]
        
        return series
    
    def get_regime_transition_performance(
        self, strategy_id: str, from_regime: MarketRegimeType, to_regime: MarketRegimeType,
        days_window: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze strategy performance during regime transitions.
        
        Args:
            strategy_id: Strategy identifier
            from_regime: Starting regime type
            to_regime: Ending regime type
            days_window: Number of days to analyze around transition
            
        Returns:
            Dict with performance statistics during transition
        """
        if strategy_id not in self.time_series_data:
            return {}
        
        # Find transitions in recent performance data
        transitions = []
        
        for key, entries in self.recent_performance.get(strategy_id, {}).items():
            # Need at least 2 entries to detect a transition
            if len(entries) < 2:
                continue
            
            # Check for transitions
            for i in range(1, len(entries)):
                prev_entry = entries[i-1]
                curr_entry = entries[i]
                
                if prev_entry['regime_type'] == from_regime and curr_entry['regime_type'] == to_regime:
                    # Found a transition
                    transition_time = curr_entry['timestamp']
                    
                    # Find nearby performance entries
                    window_start = transition_time - timedelta(days=days_window)
                    window_end = transition_time + timedelta(days=days_window)
                    
                    # Get all entries in all regimes that fall within the window
                    window_entries = []
                    for regime, series in self.time_series_data[strategy_id].items():
                        for entry in series:
                            if window_start <= entry['timestamp'] <= window_end:
                                window_entries.append({
                                    'timestamp': entry['timestamp'],
                                    'regime': regime,
                                    'metrics': entry['metrics'],
                                    'symbol': entry.get('symbol', ''),
                                    'timeframe': entry.get('timeframe', '')
                                })
                    
                    # Sort by timestamp
                    window_entries.sort(key=lambda x: x['timestamp'])
                    
                    transitions.append({
                        'transition_time': transition_time,
                        'from_regime': from_regime,
                        'to_regime': to_regime,
                        'window_entries': window_entries
                    })
        
        # Analyze transitions
        if not transitions:
            return {}
        
        # Aggregate performance metrics
        before_metrics = {}
        after_metrics = {}
        
        for transition in transitions:
            transition_time = transition['transition_time']
            
            for entry in transition['window_entries']:
                # Determine if before or after transition
                if entry['timestamp'] < transition_time:
                    period = 'before'
                    metrics_dict = before_metrics
                else:
                    period = 'after'
                    metrics_dict = after_metrics
                
                # Add metrics to appropriate period
                for metric_name, value in entry['metrics'].items():
                    try:
                        float_value = float(value)
                        
                        if metric_name not in metrics_dict:
                            metrics_dict[metric_name] = []
                        
                        metrics_dict[metric_name].append(float_value)
                    except (ValueError, TypeError):
                        # Skip non-numeric metrics
                        pass
        
        # Calculate statistics
        result = {
            'transition_count': len(transitions),
            'from_regime': from_regime.value,
            'to_regime': to_regime.value,
            'days_window': days_window,
            'before': {},
            'after': {},
            'change': {}
        }
        
        # Calculate before stats
        for metric_name, values in before_metrics.items():
            if len(values) > 0:
                result['before'][f"{metric_name}_mean"] = float(np.mean(values))
                result['before'][f"{metric_name}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        
        # Calculate after stats
        for metric_name, values in after_metrics.items():
            if len(values) > 0:
                result['after'][f"{metric_name}_mean"] = float(np.mean(values))
                result['after'][f"{metric_name}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        
        # Calculate changes
        for metric_name in set(before_metrics.keys()).intersection(after_metrics.keys()):
            if len(before_metrics[metric_name]) > 0 and len(after_metrics[metric_name]) > 0:
                before_mean = np.mean(before_metrics[metric_name])
                after_mean = np.mean(after_metrics[metric_name])
                
                if before_mean != 0:
                    percent_change = (after_mean - before_mean) / abs(before_mean) * 100
                    result['change'][f"{metric_name}_pct"] = float(percent_change)
                
                result['change'][f"{metric_name}_abs"] = float(after_mean - before_mean)
        
        return result
    
    def detect_strategy_drift(
        self, strategy_id: str, regime_type: MarketRegimeType,
        metric_name: str = "profit_factor", window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect if a strategy's performance is drifting over time.
        
        Args:
            strategy_id: Strategy identifier
            regime_type: Market regime type
            metric_name: Metric to analyze for drift
            window_size: Size of comparison windows
            
        Returns:
            Dict with drift detection results
        """
        if strategy_id not in self.performance_history or \
           regime_type not in self.performance_history[strategy_id] or \
           metric_name not in self.performance_history[strategy_id][regime_type]:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        # Get metric values
        values = self.performance_history[strategy_id][regime_type][metric_name]
        
        # Need at least 2 windows of data
        if len(values) < window_size * 2:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        # Divide into early and recent windows
        early_window = values[-2*window_size:-window_size]
        recent_window = values[-window_size:]
        
        # Calculate statistics
        early_mean = np.mean(early_window)
        recent_mean = np.mean(recent_window)
        
        early_std = np.std(early_window)
        recent_std = np.std(recent_window)
        
        # Calculate z-score of difference
        pooled_std = np.sqrt((early_std**2 + recent_std**2) / 2)
        z_score = abs(recent_mean - early_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate percent change
        percent_change = (recent_mean - early_mean) / abs(early_mean) * 100 if early_mean != 0 else 0
        
        # Detect drift (z-score > 2 or percent change > 30%)
        drift_detected = z_score > 2 or abs(percent_change) > 30
        
        # Determine direction
        if recent_mean > early_mean:
            direction = 'improving'
        elif recent_mean < early_mean:
            direction = 'degrading'
        else:
            direction = 'stable'
        
        # Prepare result
        result = {
            'drift_detected': drift_detected,
            'direction': direction,
            'early_mean': float(early_mean),
            'recent_mean': float(recent_mean),
            'z_score': float(z_score),
            'percent_change': float(percent_change),
            'metric_name': metric_name,
            'window_size': window_size
        }
        
        return result
    
    def analyze_correlation(
        self, strategy_ids: List[str], regime_type: MarketRegimeType,
        metric_name: str = "returns"
    ) -> Dict[str, Any]:
        """
        Analyze correlation between strategies in a specific regime.
        
        Args:
            strategy_ids: List of strategy identifiers
            regime_type: Market regime type
            metric_name: Metric to analyze correlation for
            
        Returns:
            Dict with correlation analysis results
        """
        if not strategy_ids:
            return {}
        
        # Collect time series data for each strategy
        strategy_data = {}
        
        for strategy_id in strategy_ids:
            if strategy_id in self.time_series_data and regime_type in self.time_series_data[strategy_id]:
                series = []
                
                for entry in self.time_series_data[strategy_id][regime_type]:
                    if 'timestamp' in entry and 'metrics' in entry and metric_name in entry['metrics']:
                        try:
                            value = float(entry['metrics'][metric_name])
                            timestamp = entry['timestamp']
                            series.append((timestamp, value))
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            pass
                
                if series:
                    strategy_data[strategy_id] = series
        
        # Need at least 2 strategies with data
        if len(strategy_data) < 2:
            return {'error': 'Insufficient data'}
        
        # Create a pandas DataFrame with aligned timestamps
        all_timestamps = set()
        
        for series in strategy_data.values():
            all_timestamps.update(ts for ts, _ in series)
        
        # Sort timestamps
        all_timestamps = sorted(all_timestamps)
        
        # Create DataFrame
        df = pd.DataFrame(index=all_timestamps)
        
        for strategy_id, series in strategy_data.items():
            # Create a dictionary mapping timestamps to values
            ts_dict = {ts: val for ts, val in series}
            
            # Add to DataFrame, with NaN for missing timestamps
            df[strategy_id] = df.index.map(lambda ts: ts_dict.get(ts, np.nan))
        
        # Calculate correlation matrix
        correlation_matrix = df.corr().fillna(0).to_dict()
        
        # Calculate average correlation for each strategy
        avg_correlations = {}
        
        for strategy_id in strategy_ids:
            if strategy_id in correlation_matrix:
                correlations = [
                    value for other_id, value in correlation_matrix[strategy_id].items()
                    if other_id != strategy_id  # Exclude self-correlation
                ]
                
                if correlations:
                    avg_correlations[strategy_id] = float(np.mean(correlations))
        
        return {
            'correlation_matrix': correlation_matrix,
            'avg_correlations': avg_correlations,
            'sample_size': len(all_timestamps),
            'regime_type': regime_type.value
        }
    
    def get_regime_performance_summary(self, strategy_id: str) -> Dict[MarketRegimeType, Dict[str, Any]]:
        """
        Get a performance summary for all regimes.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict mapping regime types to performance summaries
        """
        performance_by_regime = self.get_performance_by_regime(strategy_id)
        
        # Enhance with additional metrics
        result = {}
        
        for regime_type, metrics in performance_by_regime.items():
            summary = {
                'sample_size': metrics.get('sample_size', 0),
                'metrics': {}
            }
            
            # Extract key metrics
            for base_metric in self.tracked_metrics:
                if f"{base_metric}_mean" in metrics:
                    summary['metrics'][base_metric] = {
                        'mean': metrics.get(f"{base_metric}_mean"),
                        'latest': metrics.get(f"{base_metric}_latest"),
                        'trend': metrics.get(f"{base_metric}_trend", 0)
                    }
            
            # Add summary stats
            if 'profit_factor_mean' in metrics:
                summary['profitable'] = metrics.get('profit_factor_mean', 1.0) > 1.0
            
            if 'win_rate_mean' in metrics:
                summary['win_rate'] = metrics.get('win_rate_mean')
            
            if 'expectancy_mean' in metrics:
                summary['expectancy'] = metrics.get('expectancy_mean')
            
            # Detect drift
            drift = self.detect_strategy_drift(strategy_id, regime_type)
            summary['drift'] = drift
            
            result[regime_type] = summary
        
        return result
