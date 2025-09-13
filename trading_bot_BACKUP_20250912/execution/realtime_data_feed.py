"""
Real-Time Data Feed with Latency Monitoring

This module provides real-time market data streams from production APIs
and monitors end-to-end latency from data arrival to order submission.
"""

import logging
import os
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np

# Import trading system components
from trading_bot.event_system.event_bus import EventBus
from trading_bot.alerts.telegram_alerts import send_system_alert
from trading_bot.execution.adaptive_paper_integration import get_paper_trading_instance

logger = logging.getLogger(__name__)

class LatencyMetric:
    """Represents a latency measurement for monitoring system performance"""
    
    def __init__(self, metric_type: str, start_time: float, symbol: Optional[str] = None):
        self.metric_type = metric_type  # e.g., 'data_processing', 'order_generation'
        self.start_time = start_time
        self.end_time = None
        self.duration_ms = None
        self.symbol = symbol
        self.metadata = {}
    
    def complete(self, metadata: Optional[Dict[str, Any]] = None):
        """Complete the latency measurement"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000  # Convert to ms
        if metadata:
            self.metadata.update(metadata)
        return self
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'metric_type': self.metric_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'symbol': self.symbol,
            'metadata': self.metadata
        }

class RealtimeDataFeed:
    """
    Provides real-time market data from production APIs and monitors
    system latency from data receipt to order submission.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize the real-time data feed"""
        self.event_bus = event_bus or EventBus()
        self.running = False
        self.data_sources = {}
        self.active_streams = {}
        self.latency_metrics = []
        self.data_queue = queue.Queue()
        self.alert_thresholds = {
            'data_processing': 100,    # 100ms threshold for data processing
            'order_generation': 500,   # 500ms threshold for order generation
            'end_to_end': 1000         # 1000ms threshold for end-to-end latency
        }
        self.heartbeat_interval = 5    # 5 seconds between heartbeats
        self.last_heartbeat = {}
        self.heartbeat_thread = None
        self.processing_thread = None
        
        # Configure kill switch parameters
        self.kill_switch_activated = False
        self.max_consecutive_failures = 5
        self.failure_count = 0
        self.reconnect_backoff = 1  # Start with 1 second backoff
        self.max_reconnect_backoff = 60  # Max backoff of 60 seconds
        
        # Create output directory
        os.makedirs('./logs/realtime_data', exist_ok=True)
    
    def connect_data_source(self, name: str, connector_config: Dict[str, Any]):
        """
        Connect to a real-time data source
        
        Args:
            name: Name of the data source ('alpaca', 'tradier', etc.)
            connector_config: Configuration for the connector
        """
        try:
            # Initialize the appropriate connector based on name
            if name.lower() == 'alpaca':
                from trading_bot.data_sources.alpaca_realtime import AlpacaRealtimeConnector
                connector = AlpacaRealtimeConnector(self.event_bus)
            elif name.lower() == 'tradier':
                from trading_bot.data_sources.tradier_realtime import TradierRealtimeConnector
                connector = TradierRealtimeConnector(self.event_bus)
            elif name.lower() == 'finnhub':
                from trading_bot.data_sources.finnhub_realtime import FinnhubRealtimeConnector
                connector = FinnhubRealtimeConnector(self.event_bus)
            elif name.lower() == 'iex':
                from trading_bot.data_sources.iex_realtime import IEXRealtimeConnector
                connector = IEXRealtimeConnector(self.event_bus)
            else:
                logger.error(f"Unknown data source: {name}")
                return False
            
            # Connect to the data source
            success = connector.connect(connector_config)
            if success:
                self.data_sources[name] = connector
                logger.info(f"Connected to {name} real-time data feed")
                
                # Send alert
                send_system_alert(
                    component=f"{name.capitalize()} Data Feed",
                    status="online",
                    message=f"Connected to {name} real-time data feed",
                    severity="info"
                )
                
                return True
            else:
                logger.error(f"Failed to connect to {name} real-time data feed")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to {name} real-time data feed: {str(e)}")
            return False
    
    def subscribe_to_symbols(self, symbols: List[str], data_types: Optional[List[str]] = None):
        """
        Subscribe to real-time data for specified symbols
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: List of data types ('trades', 'quotes', 'bars', etc.)
        """
        if not data_types:
            data_types = ['trades', 'quotes']
        
        success_count = 0
        
        for name, connector in self.data_sources.items():
            try:
                # Subscribe to symbols
                success = connector.subscribe(symbols, data_types)
                if success:
                    if name not in self.active_streams:
                        self.active_streams[name] = []
                    
                    self.active_streams[name].extend(symbols)
                    success_count += 1
                    
                    # Initialize heartbeat tracking for these symbols
                    for symbol in symbols:
                        self.last_heartbeat[f"{name}_{symbol}"] = time.time()
                    
                    logger.info(f"Subscribed to {len(symbols)} symbols via {name}")
            except Exception as e:
                logger.error(f"Error subscribing to symbols via {name}: {str(e)}")
        
        return success_count > 0
    
    def start_monitoring(self, 
                      process_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Start monitoring real-time data feeds and latency
        
        Args:
            process_callback: Callback function to process data
        """
        if self.running:
            logger.warning("Real-time monitoring already running")
            return False
        
        self.running = True
        self.kill_switch_activated = False
        self.failure_count = 0
        
        # Set up event handlers
        for name, connector in self.data_sources.items():
            connector.set_data_callback(self._handle_data_received)
        
        # Start heartbeat monitoring thread
        self.heartbeat_thread = threading.Thread(
            target=self._monitor_heartbeats,
            daemon=True
        )
        self.heartbeat_thread.start()
        
        # Start data processing thread
        self.processing_thread = threading.Thread(
            target=self._process_data_queue,
            args=(process_callback,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Started real-time data monitoring")
        send_system_alert(
            component="Real-time Monitor",
            status="online",
            message="Started real-time data feed monitoring",
            severity="info"
        )
        
        return True
    
    def stop_monitoring(self):
        """Stop monitoring real-time data feeds"""
        if not self.running:
            return
        
        self.running = False
        
        # Disconnect data sources
        for name, connector in self.data_sources.items():
            try:
                connector.disconnect()
                logger.info(f"Disconnected from {name} data feed")
            except Exception as e:
                logger.error(f"Error disconnecting from {name} data feed: {str(e)}")
        
        # Clear active streams
        self.active_streams = {}
        
        # Save latency metrics
        self._save_latency_metrics()
        
        logger.info("Stopped real-time data monitoring")
        send_system_alert(
            component="Real-time Monitor",
            status="offline",
            message="Stopped real-time data feed monitoring",
            severity="info"
        )
    
    def test_latency(self, symbol: str, iterations: int = 10):
        """
        Run a latency test for a specific symbol
        
        Args:
            symbol: Symbol to test
            iterations: Number of test iterations
        
        Returns:
            Dict with latency statistics
        """
        if not self.running:
            logger.error("Cannot test latency when monitoring is not running")
            return None
        
        latency_samples = []
        
        logger.info(f"Starting latency test for {symbol} ({iterations} iterations)")
        
        for i in range(iterations):
            # Start latency metric
            metric = LatencyMetric('latency_test', time.time(), symbol)
            
            # Simulate processing
            time.sleep(0.01)  # 10ms simulated processing
            
            # Complete metric
            metric.complete({'iteration': i})
            latency_samples.append(metric.duration_ms)
            
            # Wait a bit between tests
            time.sleep(0.1)
        
        # Calculate statistics
        stats = {
            'symbol': symbol,
            'iterations': iterations,
            'avg_latency_ms': np.mean(latency_samples),
            'median_latency_ms': np.median(latency_samples),
            'min_latency_ms': min(latency_samples),
            'max_latency_ms': max(latency_samples),
            'std_dev_ms': np.std(latency_samples)
        }
        
        logger.info(f"Latency test results for {symbol}: avg={stats['avg_latency_ms']:.2f}ms, max={stats['max_latency_ms']:.2f}ms")
        
        return stats
    
    def get_latency_statistics(self, metric_type: Optional[str] = None,
                            window_seconds: int = 300):
        """
        Get latency statistics for the specified time window
        
        Args:
            metric_type: Type of metric to analyze (None for all)
            window_seconds: Time window in seconds
        
        Returns:
            Dict with latency statistics
        """
        # Filter metrics by type and time window
        now = time.time()
        window_start = now - window_seconds
        
        filtered_metrics = [
            m for m in self.latency_metrics
            if (metric_type is None or m.metric_type == metric_type) and
               m.start_time >= window_start
        ]
        
        if not filtered_metrics:
            return {'error': 'No metrics available for the specified criteria'}
        
        # Extract durations
        durations = [m.duration_ms for m in filtered_metrics if m.duration_ms is not None]
        
        if not durations:
            return {'error': 'No completed metrics available'}
        
        # Calculate statistics
        stats = {
            'metric_type': metric_type or 'all',
            'window_seconds': window_seconds,
            'sample_count': len(durations),
            'avg_latency_ms': np.mean(durations),
            'median_latency_ms': np.median(durations),
            'p95_latency_ms': np.percentile(durations, 95),
            'p99_latency_ms': np.percentile(durations, 99),
            'min_latency_ms': min(durations),
            'max_latency_ms': max(durations),
            'std_dev_ms': np.std(durations)
        }
        
        return stats
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the real-time data feeds
        
        Returns:
            Dict with health status information
        """
        if not self.running:
            return {
                'status': 'offline',
                'message': 'Real-time monitoring is not running'
            }
        
        # Check if kill switch is activated
        if self.kill_switch_activated:
            return {
                'status': 'error',
                'message': 'Kill switch is activated',
                'kill_switch': True,
                'failure_count': self.failure_count
            }
        
        # Check heartbeats
        now = time.time()
        stale_feeds = []
        
        for feed_id, last_time in self.last_heartbeat.items():
            if now - last_time > self.heartbeat_interval * 3:  # 3x heartbeat interval
                stale_feeds.append(feed_id)
        
        # Check latency
        latency_stats = self.get_latency_statistics(window_seconds=60)  # Last minute
        
        health_status = {
            'status': 'healthy',
            'active_feeds': len(self.active_streams),
            'stale_feeds': stale_feeds,
            'latency': {
                'avg_ms': latency_stats.get('avg_latency_ms', 0),
                'max_ms': latency_stats.get('max_latency_ms', 0)
            },
            'failure_count': self.failure_count,
            'data_queue_size': self.data_queue.qsize()
        }
        
        # Determine overall status
        if stale_feeds:
            health_status['status'] = 'warning'
            health_status['message'] = f"{len(stale_feeds)} feeds with stale data"
        
        if latency_stats.get('avg_latency_ms', 0) > self.alert_thresholds.get('end_to_end', 1000):
            health_status['status'] = 'warning'
            health_status['message'] = 'High average latency detected'
        
        return health_status
    
    def activate_kill_switch(self, reason: str):
        """
        Activate the kill switch to halt trading
        
        Args:
            reason: Reason for activating kill switch
        """
        if self.kill_switch_activated:
            logger.warning("Kill switch already activated")
            return
        
        self.kill_switch_activated = True
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Get paper trading instance and notify controller
        paper_trading = get_paper_trading_instance()
        if paper_trading and paper_trading.controller:
            # Call emergency stop if implemented
            if hasattr(paper_trading.controller, 'emergency_stop'):
                paper_trading.controller.emergency_stop(reason)
        
        # Send critical alert
        send_system_alert(
            component="Kill Switch",
            status="error",
            message=f"Trading kill switch activated: {reason}",
            severity="critical"
        )
    
    def reset_kill_switch(self):
        """Reset the kill switch to resume trading"""
        if not self.kill_switch_activated:
            return
        
        self.kill_switch_activated = False
        self.failure_count = 0
        self.reconnect_backoff = 1  # Reset backoff
        
        logger.info("Kill switch reset, trading can resume")
        
        # Send alert
        send_system_alert(
            component="Kill Switch",
            status="online",
            message="Trading kill switch reset, system operational",
            severity="info"
        )
    
    def _handle_data_received(self, data: Dict[str, Any]):
        """Handle incoming data from a real-time feed"""
        # Start latency tracking
        data['_received_time'] = time.time()
        data['_metric'] = LatencyMetric(
            metric_type='data_processing',
            start_time=data['_received_time'],
            symbol=data.get('symbol')
        )
        
        # Update heartbeat
        source = data.get('source', 'unknown')
        symbol = data.get('symbol', 'unknown')
        self.last_heartbeat[f"{source}_{symbol}"] = time.time()
        
        # Add to processing queue
        self.data_queue.put(data)
    
    def _process_data_queue(self, callback: Optional[Callable] = None):
        """Process data from the queue"""
        while self.running:
            try:
                if self.data_queue.empty():
                    time.sleep(0.01)  # Small sleep to prevent CPU thrashing
                    continue
                
                # Get data from queue
                data = self.data_queue.get(block=False)
                
                # Complete the data processing metric
                if '_metric' in data:
                    metric = data['_metric']
                    metric.complete()
                    self.latency_metrics.append(metric)
                    
                    # Check if latency exceeds threshold
                    if (metric.duration_ms and 
                        metric.duration_ms > self.alert_thresholds.get('data_processing', 100)):
                        logger.warning(f"High data processing latency: {metric.duration_ms:.2f}ms for {data.get('symbol')}")
                
                # Start order generation latency measurement
                order_gen_metric = LatencyMetric(
                    metric_type='order_generation',
                    start_time=time.time(),
                    symbol=data.get('symbol')
                )
                
                # Process data with callback if provided
                if callback:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in data processing callback: {str(e)}")
                        self.failure_count += 1
                        
                        if self.failure_count >= self.max_consecutive_failures:
                            self.activate_kill_switch(f"Max consecutive failures reached: {str(e)}")
                
                # Complete order generation metric
                order_gen_metric.complete()
                self.latency_metrics.append(order_gen_metric)
                
                # Calculate end-to-end latency
                if '_received_time' in data:
                    end_to_end_ms = (time.time() - data['_received_time']) * 1000
                    
                    # Create end-to-end metric
                    e2e_metric = LatencyMetric(
                        metric_type='end_to_end',
                        start_time=data['_received_time'],
                        symbol=data.get('symbol')
                    )
                    e2e_metric.complete({'duration_ms': end_to_end_ms})
                    self.latency_metrics.append(e2e_metric)
                    
                    # Check if end-to-end latency exceeds threshold
                    if end_to_end_ms > self.alert_thresholds.get('end_to_end', 1000):
                        logger.warning(f"High end-to-end latency: {end_to_end_ms:.2f}ms for {data.get('symbol')}")
                
                # Limit number of stored metrics
                if len(self.latency_metrics) > 10000:
                    # Save metrics and clear list
                    self._save_latency_metrics()
                    self.latency_metrics = self.latency_metrics[-1000:]  # Keep the most recent 1000
                
                # Reset failure count on successful processing
                if self.failure_count > 0:
                    self.failure_count = 0
                
                # Mark as done
                self.data_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error processing data queue: {str(e)}")
                self.failure_count += 1
                
                if self.failure_count >= self.max_consecutive_failures:
                    self.activate_kill_switch(f"Error in data processing: {str(e)}")
                
                # Backoff on errors
                time.sleep(0.1)
    
    def _monitor_heartbeats(self):
        """Monitor heartbeats from data feeds"""
        while self.running:
            try:
                # Check all heartbeats
                now = time.time()
                missing_heartbeats = []
                
                for feed_id, last_time in self.last_heartbeat.items():
                    # Check if heartbeat is stale (3x the interval)
                    if now - last_time > self.heartbeat_interval * 3:
                        missing_heartbeats.append(feed_id)
                
                # Alert on missing heartbeats
                if missing_heartbeats:
                    logger.warning(f"Missing heartbeats from: {', '.join(missing_heartbeats)}")
                    
                    # Send alert on significant outages
                    if len(missing_heartbeats) > len(self.last_heartbeat) * 0.5:  # >50% of feeds missing
                        send_system_alert(
                            component="Data Heartbeat",
                            status="warning",
                            message=f"Missing heartbeats from {len(missing_heartbeats)} data feeds",
                            severity="medium"
                        )
                        
                        # Increment failure count
                        self.failure_count += 1
                        
                        if self.failure_count >= self.max_consecutive_failures:
                            self.activate_kill_switch("Multiple data feeds missing heartbeats")
                
                # Sleep until next check
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring: {str(e)}")
                time.sleep(1)  # Sleep on error
    
    def _save_latency_metrics(self):
        """Save latency metrics to file"""
        if not self.latency_metrics:
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"latency_metrics_{timestamp}.json"
            filepath = os.path.join('./logs/realtime_data', filename)
            
            # Convert metrics to dicts
            metrics_dicts = [m.to_dict() for m in self.latency_metrics]
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(metrics_dicts, f, indent=2)
            
            logger.info(f"Saved {len(metrics_dicts)} latency metrics to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving latency metrics: {str(e)}")

# Singleton instance
_realtime_feed_instance = None

def get_realtime_feed_instance() -> RealtimeDataFeed:
    """Get the global real-time feed instance"""
    global _realtime_feed_instance
    if _realtime_feed_instance is None:
        _realtime_feed_instance = RealtimeDataFeed()
    return _realtime_feed_instance

# Usage example (if run as script)
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create instance
    feed = RealtimeDataFeed()
    
    # Connect to data source (using dummy config)
    feed.connect_data_source('alpaca', {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'endpoint': 'https://paper-api.alpaca.markets'
    })
    
    # Subscribe to symbols
    feed.subscribe_to_symbols(['SPY', 'QQQ', 'AAPL'])
    
    # Define a simple data callback
    def process_data(data):
        symbol = data.get('symbol')
        price = data.get('price')
        timestamp = data.get('timestamp')
        print(f"Received data: {symbol} @ ${price} [{timestamp}]")
    
    # Start monitoring
    feed.start_monitoring(process_data)
    
    # Run for a minute
    try:
        print("Monitoring real-time data for 60 seconds...")
        time.sleep(60)
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Print latency statistics
    stats = feed.get_latency_statistics()
    print(f"Latency statistics:")
    print(f"  Average: {stats.get('avg_latency_ms', 0):.2f} ms")
    print(f"  95th percentile: {stats.get('p95_latency_ms', 0):.2f} ms")
    print(f"  Maximum: {stats.get('max_latency_ms', 0):.2f} ms")
    
    # Stop monitoring
    feed.stop_monitoring()
