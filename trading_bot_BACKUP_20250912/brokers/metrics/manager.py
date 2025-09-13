#!/usr/bin/env python3
"""
Broker Metrics Manager

Provides centralized management of broker performance metrics collection,
analysis, and reporting.
"""

import os
import time
import json
import logging
import threading
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set

from trading_bot.brokers.metrics.base import (
    MetricType, MetricOperation, MetricPeriod, MetricValue,
    MetricsStore, InMemoryMetricsStore, FileMetricsStore
)

from trading_bot.brokers.metrics.collectors import (
    LatencyTracker, ReliabilityMonitor, 
    ExecutionQualityAnalyzer, CostTracker
)

# Configure logging
logger = logging.getLogger(__name__)

class BrokerMetricsManager:
    """
    Central manager for broker performance metrics collection and analysis
    
    Integrates with the Multi-Broker Manager to provide comprehensive 
    monitoring of all broker operations.
    """
    
    def __init__(
        self,
        metrics_dir: str = "data/metrics/brokers",
        use_file_storage: bool = True,
        retention_days: int = 30,
        health_check_interval: int = 300  # 5 minutes
    ):
        """
        Initialize broker metrics manager
        
        Args:
            metrics_dir: Directory for storing metrics
            use_file_storage: Whether to persist metrics to files
            retention_days: Number of days to retain metrics
            health_check_interval: Interval between health checks (seconds)
        """
        self.metrics_dir = metrics_dir
        self.use_file_storage = use_file_storage
        self.retention_days = retention_days
        self.health_check_interval = health_check_interval
        
        # Create metrics directory if needed
        if use_file_storage:
            os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics store
        if use_file_storage:
            metrics_file = os.path.join(self.metrics_dir, "broker_metrics.jsonl")
            self.metrics_store = FileMetricsStore(metrics_file)
        else:
            self.metrics_store = InMemoryMetricsStore(max_retention_days=retention_days)
        
        # Initialize collectors
        self.latency_tracker = LatencyTracker(self.metrics_store)
        self.reliability_monitor = ReliabilityMonitor(self.metrics_store)
        self.execution_analyzer = ExecutionQualityAnalyzer(self.metrics_store)
        self.cost_tracker = CostTracker(self.metrics_store)
        
        # Broker tracking
        self.active_brokers = {}
        self.broker_types = {}
        self.broker_lock = threading.RLock()
        
        # Health check thread
        self.health_check_thread = None
        self.stop_health_check = threading.Event()
        
        logger.info(f"Initialized BrokerMetricsManager (storage={'file' if use_file_storage else 'memory'})")
    
    def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.health_check_thread is None or not self.health_check_thread.is_alive():
            self.stop_health_check.clear()
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_check_thread.start()
            logger.info(f"Started broker health check thread (interval={self.health_check_interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.stop_health_check.set()
            self.health_check_thread.join(timeout=10)
            logger.info("Stopped broker health check thread")
    
    def register_broker(self, broker_id: str, broker_type: str, broker_instance: Any):
        """
        Register a broker for monitoring
        
        Args:
            broker_id: Broker identifier
            broker_type: Type of broker
            broker_instance: Broker instance
        """
        with self.broker_lock:
            self.active_brokers[broker_id] = broker_instance
            self.broker_types[broker_id] = broker_type
            logger.info(f"Registered broker for monitoring: {broker_id} ({broker_type})")
    
    def unregister_broker(self, broker_id: str):
        """
        Unregister a broker from monitoring
        
        Args:
            broker_id: Broker identifier
        """
        with self.broker_lock:
            if broker_id in self.active_brokers:
                del self.active_brokers[broker_id]
                del self.broker_types[broker_id]
                logger.info(f"Unregistered broker from monitoring: {broker_id}")
    
    def register_broker_connection(self, broker_id: str, connected: bool, metadata: Optional[Dict[str, Any]] = None):
        """
        Register broker connection state change
        
        Args:
            broker_id: Broker identifier
            connected: Whether broker is connected
            metadata: Additional connection metadata
        """
        self.reliability_monitor.record_connection_state(broker_id, connected, metadata)
        logger.debug(f"Broker {broker_id} connection state: {'connected' if connected else 'disconnected'}")
    
    def register_broker_error(
        self,
        broker_id: str,
        error_type: str,
        error_message: str,
        operation: Optional[MetricOperation] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a broker error
        
        Args:
            broker_id: Broker identifier
            error_type: Type of error
            error_message: Error message
            operation: Operation that caused the error
            metadata: Additional error metadata
        """
        self.reliability_monitor.record_error(broker_id, error_type, error_message, operation, metadata)
        logger.debug(f"Broker {broker_id} error: {error_type} - {error_message}")
    
    def track_operation_latency(self, operation: MetricOperation):
        """
        Decorator to track operation latency
        
        Args:
            operation: Operation to track
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(broker_instance, *args, **kwargs):
                # Get broker ID
                broker_id = self._get_broker_id(broker_instance)
                if not broker_id:
                    # If broker not registered, just call the function
                    return func(broker_instance, *args, **kwargs)
                
                # Start tracking
                op_id = self.latency_tracker.start_operation(
                    operation=operation,
                    broker_id=broker_id,
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
                
                try:
                    # Call the function
                    result = func(broker_instance, *args, **kwargs)
                    
                    # Record successful completion
                    self.latency_tracker.end_operation(
                        operation_id=op_id,
                        success=True,
                        additional_metadata={"result": str(result) if not isinstance(result, dict) else "dict_result"}
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    self.latency_tracker.end_operation(
                        operation_id=op_id,
                        success=False,
                        additional_metadata={"error": str(e)}
                    )
                    
                    # Register error for reliability metrics
                    self.register_broker_error(
                        broker_id=broker_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        operation=operation
                    )
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        
        return decorator
    
    def record_order_execution(
        self,
        broker_id: str,
        order_id: str,
        symbol: str,
        order_type: str,
        expected_price: float,
        executed_price: float,
        quantity: float,
        side: str,
        time_in_force: str,
        submission_time: float,
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record order execution details
        
        Args:
            broker_id: Broker identifier
            order_id: Order identifier
            symbol: Instrument symbol
            order_type: Type of order
            expected_price: Expected execution price
            executed_price: Actual execution price
            quantity: Order quantity
            side: Order side (buy/sell)
            time_in_force: Time in force
            submission_time: Time order was submitted
            execution_time: Time order was executed
            metadata: Additional contextual data
        """
        self.execution_analyzer.record_order_execution(
            broker_id, order_id, symbol, order_type, expected_price, executed_price,
            quantity, side, time_in_force, submission_time, execution_time, metadata
        )
    
    def record_order_fill(
        self,
        broker_id: str,
        order_id: str,
        symbol: str,
        order_type: str,
        requested_quantity: float,
        filled_quantity: float,
        side: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record order fill rate
        
        Args:
            broker_id: Broker identifier
            order_id: Order identifier
            symbol: Instrument symbol
            order_type: Type of order
            requested_quantity: Requested order quantity
            filled_quantity: Filled order quantity
            side: Order side (buy/sell)
            metadata: Additional contextual data
        """
        self.execution_analyzer.record_fill_rate(
            broker_id, order_id, symbol, order_type, 
            requested_quantity, filled_quantity, side, metadata
        )
    
    def record_commission(
        self,
        broker_id: str,
        order_id: str,
        symbol: str,
        commission: float,
        exchange_fee: float = 0.0,
        regulatory_fee: float = 0.0,
        other_fees: float = 0.0,
        trade_value: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record commission and fees
        
        Args:
            broker_id: Broker identifier
            order_id: Order identifier
            symbol: Instrument symbol
            commission: Broker commission
            exchange_fee: Exchange fee
            regulatory_fee: Regulatory fee
            other_fees: Other fees
            trade_value: Total value of the trade
            metadata: Additional contextual data
        """
        self.cost_tracker.record_commission(
            broker_id, order_id, symbol, commission, exchange_fee,
            regulatory_fee, other_fees, trade_value, metadata
        )
    
    def get_broker_metrics(
        self,
        broker_id: str,
        period: MetricPeriod = MetricPeriod.DAY
    ) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a broker
        
        Args:
            broker_id: Broker identifier
            period: Time period for metrics
            
        Returns:
            Dictionary of metrics
        """
        # Get latency metrics
        latency_stats = self.metrics_store.get_stats(
            metric_type=MetricType.LATENCY,
            broker_id=broker_id,
            period=period
        )
        
        # Get reliability metrics
        reliability_metrics = self.metrics_store.query(
            metric_type=MetricType.RELIABILITY,
            broker_id=broker_id,
            period=period,
            limit=10000
        )
        
        # Calculate reliability statistics
        connection_events = [m for m in reliability_metrics 
                            if m.operation in [MetricOperation.CONNECT, MetricOperation.DISCONNECT]]
        reconnect_events = [m for m in reliability_metrics if m.operation == MetricOperation.RECONNECT]
        error_events = [m for m in reliability_metrics if m.operation == MetricOperation.ERROR]
        
        # Get health check metrics
        health_metrics = self.metrics_store.query(
            metric_type=MetricType.HEALTH,
            operation=MetricOperation.HEALTH_CHECK,
            broker_id=broker_id,
            period=period,
            limit=10000
        )
        
        # Calculate health statistics
        health_count = len(health_metrics)
        health_success = sum(1 for m in health_metrics if m.value == 1)
        availability = (health_success / health_count * 100) if health_count > 0 else 0
        
        # Get execution quality metrics
        avg_slippage = self.execution_analyzer.get_avg_slippage(broker_id, period)
        
        # Get cost metrics
        cost_breakdown = self.cost_tracker.get_total_costs(broker_id, period)
        avg_commission = self.cost_tracker.get_avg_commission(broker_id, period)
        
        # Compile overall metrics
        return {
            "broker_id": broker_id,
            "broker_type": self.broker_types.get(broker_id, "unknown"),
            "period": period.value,
            "latency": {
                "mean_ms": latency_stats.get("mean", 0),
                "median_ms": latency_stats.get("median", 0),
                "min_ms": latency_stats.get("min", 0),
                "max_ms": latency_stats.get("max", 0),
                "stddev_ms": latency_stats.get("stddev", 0)
            },
            "reliability": {
                "connection_changes": len(connection_events),
                "reconnects": len(reconnect_events),
                "errors": len(error_events),
                "availability": availability
            },
            "execution_quality": {
                "avg_slippage_pct": avg_slippage
            },
            "costs": {
                "avg_commission": avg_commission,
                "total_commission": cost_breakdown["total_commission"],
                "total_exchange_fee": cost_breakdown["total_exchange_fee"],
                "total_regulatory_fee": cost_breakdown["total_regulatory_fee"],
                "total_other_fees": cost_breakdown["total_other_fees"],
                "overall_total": cost_breakdown["overall_total"]
            }
        }
    
    def get_metrics_report(
        self,
        period: MetricPeriod = MetricPeriod.DAY,
        format: str = "json"
    ) -> str:
        """
        Generate a metrics report for all brokers
        
        Args:
            period: Time period for report
            format: Output format (json/text)
            
        Returns:
            Report string
        """
        # Get metrics for all brokers
        broker_metrics = {}
        for broker_id in self.active_brokers:
            broker_metrics[broker_id] = self.get_broker_metrics(broker_id, period)
        
        # Generate report
        if format.lower() == "json":
            return json.dumps(broker_metrics, indent=2)
        else:
            # Text format
            report = [f"=== Broker Metrics Report ({period.value}) ===\n"]
            
            for broker_id, metrics in broker_metrics.items():
                report.append(f"Broker: {broker_id} ({metrics['broker_type']})")
                
                # Latency
                latency = metrics["latency"]
                report.append(f"  Latency:")
                report.append(f"    Mean: {latency['mean_ms']:.2f} ms")
                report.append(f"    Median: {latency['median_ms']:.2f} ms")
                report.append(f"    Min/Max: {latency['min_ms']:.2f}/{latency['max_ms']:.2f} ms")
                
                # Reliability
                rel = metrics["reliability"]
                report.append(f"  Reliability:")
                report.append(f"    Availability: {rel['availability']:.2f}%")
                report.append(f"    Reconnects: {rel['reconnects']}")
                report.append(f"    Errors: {rel['errors']}")
                
                # Execution quality
                exec_qual = metrics["execution_quality"]
                report.append(f"  Execution Quality:")
                report.append(f"    Avg Slippage: {exec_qual['avg_slippage_pct']:.4f}%")
                
                # Costs
                costs = metrics["costs"]
                report.append(f"  Costs:")
                report.append(f"    Avg Commission: ${costs['avg_commission']:.4f}")
                report.append(f"    Total Costs: ${costs['overall_total']:.2f}")
                
                report.append("")
            
            return "\n".join(report)
    
    def _get_broker_id(self, broker_instance: Any) -> Optional[str]:
        """Get broker ID from instance"""
        with self.broker_lock:
            for broker_id, instance in self.active_brokers.items():
                if instance is broker_instance:
                    return broker_id
            return None
    
    def _health_check_loop(self):
        """Background thread for periodic health checks"""
        while not self.stop_health_check.is_set():
            try:
                self._check_all_brokers()
            except Exception as e:
                logger.error(f"Error in health check: {str(e)}")
            
            # Wait for next check
            self.stop_health_check.wait(self.health_check_interval)
    
    def _check_all_brokers(self):
        """Perform health check on all registered brokers"""
        with self.broker_lock:
            for broker_id, broker in self.active_brokers.items():
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Check connection
                    connected = False
                    if hasattr(broker, "is_connected"):
                        connected = broker.is_connected()
                    
                    # End timing
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Record health check
                    self.reliability_monitor.record_health_check(
                        broker_id=broker_id,
                        success=connected,
                        response_time_ms=elapsed_ms
                    )
                    
                    logger.debug(f"Health check for {broker_id}: {'✅' if connected else '❌'} ({elapsed_ms:.2f} ms)")
                    
                except Exception as e:
                    logger.error(f"Error checking broker {broker_id}: {str(e)}")
                    
                    # Record failed health check
                    self.reliability_monitor.record_health_check(
                        broker_id=broker_id,
                        success=False,
                        response_time_ms=0,
                        metadata={"error": str(e)}
                    )

# Integration functions for broker manager

def integrate_with_multi_broker_manager(broker_manager, metrics_manager):
    """
    Integrate metrics manager with multi-broker manager
    
    This patches the broker manager's methods to collect metrics automatically.
    
    Args:
        broker_manager: MultiBrokerManager instance
        metrics_manager: BrokerMetricsManager instance
    """
    # Register existing brokers
    for broker_id, broker in broker_manager.brokers.items():
        broker_type = getattr(broker, "broker_type", "unknown")
        metrics_manager.register_broker(broker_id, broker_type, broker)
    
    # Patch add_broker method
    original_add_broker = broker_manager.add_broker
    
    @functools.wraps(original_add_broker)
    def patched_add_broker(broker_id, broker, *args, **kwargs):
        result = original_add_broker(broker_id, broker, *args, **kwargs)
        broker_type = getattr(broker, "broker_type", "unknown")
        metrics_manager.register_broker(broker_id, broker_type, broker)
        return result
    
    broker_manager.add_broker = patched_add_broker
    
    # Patch remove_broker method
    original_remove_broker = broker_manager.remove_broker
    
    @functools.wraps(original_remove_broker)
    def patched_remove_broker(broker_id, *args, **kwargs):
        result = original_remove_broker(broker_id, *args, **kwargs)
        metrics_manager.unregister_broker(broker_id)
        return result
    
    broker_manager.remove_broker = patched_remove_broker
    
    # Add metrics manager as attribute
    broker_manager.metrics_manager = metrics_manager
    
    # Start monitoring
    metrics_manager.start_monitoring()
