#!/usr/bin/env python3
"""
Broker Metrics System - Metric Collectors

Implements specialized collectors for different types of broker metrics:
- Latency (API response times)
- Reliability (connection stability, errors)
- Execution Quality (slippage, fill rates)
- Cost (commissions, fees)
"""

import time
import logging
import threading
import functools
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from trading_bot.brokers.metrics.base import (
    MetricType, MetricOperation, MetricPeriod, MetricValue,
    MetricsStore, InMemoryMetricsStore, FileMetricsStore
)

# Configure logging
logger = logging.getLogger(__name__)

class LatencyTracker:
    """Tracks API response times and operation latency"""
    
    def __init__(self, metrics_store: MetricsStore):
        """
        Initialize latency tracker
        
        Args:
            metrics_store: Storage for collected metrics
        """
        self.metrics_store = metrics_store
        self.pending_operations = {}
        self.lock = threading.RLock()
    
    def start_operation(
        self, 
        operation: MetricOperation,
        broker_id: str,
        operation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking an operation
        
        Args:
            operation: Type of operation
            broker_id: Broker identifier
            operation_id: Optional unique ID for this operation
            metadata: Additional contextual data
            
        Returns:
            Operation ID for later reference
        """
        with self.lock:
            # Generate an operation ID if not provided
            if operation_id is None:
                operation_id = f"{broker_id}_{operation.value}_{time.time()}"
            
            # Store start time
            self.pending_operations[operation_id] = {
                "start_time": time.time(),
                "operation": operation,
                "broker_id": broker_id,
                "metadata": metadata or {}
            }
            
            return operation_id
    
    def end_operation(
        self, 
        operation_id: str,
        success: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        End tracking an operation and record latency
        
        Args:
            operation_id: ID returned from start_operation
            success: Whether operation was successful
            additional_metadata: Additional data to add to metadata
            
        Returns:
            Elapsed time in milliseconds
        """
        with self.lock:
            if operation_id not in self.pending_operations:
                logger.warning(f"Operation ID not found: {operation_id}")
                return 0
            
            # Calculate elapsed time
            op_data = self.pending_operations.pop(operation_id)
            start_time = op_data["start_time"]
            elapsed_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Combine metadata
            metadata = op_data["metadata"].copy()
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Add success flag
            metadata["success"] = success
            
            # Store metric
            metric = MetricValue(
                value=elapsed_ms,
                metric_type=MetricType.LATENCY,
                operation=op_data["operation"],
                broker_id=op_data["broker_id"],
                metadata=metadata
            )
            
            self.metrics_store.store(metric)
            return elapsed_ms
    
    def track_operation(
        self,
        operation: MetricOperation,
        broker_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to track latency of a function call
        
        Args:
            operation: Type of operation
            broker_id: Broker identifier
            metadata: Additional contextual data
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Start tracking
                op_id = self.start_operation(operation, broker_id, metadata=metadata)
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # End tracking (success)
                    self.end_operation(op_id, success=True)
                    
                    return result
                    
                except Exception as e:
                    # End tracking (failure)
                    self.end_operation(op_id, success=False, 
                                       additional_metadata={"error": str(e)})
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        
        return decorator

class ReliabilityMonitor:
    """Monitors broker reliability, connection stability, and errors"""
    
    def __init__(self, metrics_store: MetricsStore):
        """
        Initialize reliability monitor
        
        Args:
            metrics_store: Storage for collected metrics
        """
        self.metrics_store = metrics_store
        self.connection_statuses = {}
        self.error_counts = {}
        self.lock = threading.RLock()
    
    def record_connection_state(
        self,
        broker_id: str,
        connected: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record connection state change
        
        Args:
            broker_id: Broker identifier
            connected: Whether broker is connected
            metadata: Additional contextual data
        """
        with self.lock:
            # Determine if this is a change in state
            prev_state = self.connection_statuses.get(broker_id)
            self.connection_statuses[broker_id] = connected
            
            # Create metadata if None
            if metadata is None:
                metadata = {}
            
            # Add previous state
            metadata["previous_state"] = prev_state
            
            # Record connection state change
            metric = MetricValue(
                value=1 if connected else 0,
                metric_type=MetricType.RELIABILITY,
                operation=MetricOperation.CONNECT if connected else MetricOperation.DISCONNECT,
                broker_id=broker_id,
                metadata=metadata
            )
            
            self.metrics_store.store(metric)
            
            # If reconnecting, record reconnect event
            if connected and prev_state is False:
                reconnect_metric = MetricValue(
                    value=1,
                    metric_type=MetricType.RELIABILITY,
                    operation=MetricOperation.RECONNECT,
                    broker_id=broker_id,
                    metadata=metadata
                )
                
                self.metrics_store.store(reconnect_metric)
    
    def record_error(
        self,
        broker_id: str,
        error_type: str,
        error_message: str,
        operation: Optional[MetricOperation] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a broker error
        
        Args:
            broker_id: Broker identifier
            error_type: Type of error
            error_message: Error message
            operation: Operation that caused the error
            metadata: Additional contextual data
        """
        with self.lock:
            # Update error count
            error_key = f"{broker_id}_{error_type}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Create metadata if None
            if metadata is None:
                metadata = {}
            
            # Add error details
            metadata["error_type"] = error_type
            metadata["error_message"] = error_message
            metadata["error_count"] = self.error_counts[error_key]
            
            # Record error
            metric = MetricValue(
                value=1,
                metric_type=MetricType.RELIABILITY,
                operation=operation or MetricOperation.ERROR,
                broker_id=broker_id,
                metadata=metadata
            )
            
            self.metrics_store.store(metric)
    
    def record_health_check(
        self,
        broker_id: str,
        success: bool,
        response_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record broker health check result
        
        Args:
            broker_id: Broker identifier
            success: Whether health check was successful
            response_time_ms: Response time in milliseconds
            metadata: Additional contextual data
        """
        with self.lock:
            # Create metadata if None
            if metadata is None:
                metadata = {}
            
            # Add health check details
            metadata["success"] = success
            metadata["response_time_ms"] = response_time_ms
            
            # Record health check
            metric = MetricValue(
                value=1 if success else 0,
                metric_type=MetricType.HEALTH,
                operation=MetricOperation.HEALTH_CHECK,
                broker_id=broker_id,
                metadata=metadata
            )
            
            self.metrics_store.store(metric)
    
    def get_availability(
        self,
        broker_id: str,
        period: MetricPeriod = MetricPeriod.DAY
    ) -> float:
        """
        Calculate broker availability percentage
        
        Args:
            broker_id: Broker identifier
            period: Time period for calculation
            
        Returns:
            Availability percentage (0-100)
        """
        # Get health check metrics
        health_metrics = self.metrics_store.query(
            metric_type=MetricType.HEALTH,
            operation=MetricOperation.HEALTH_CHECK,
            broker_id=broker_id,
            period=period,
            limit=10000
        )
        
        if not health_metrics:
            return 0.0
        
        # Calculate availability
        success_count = sum(1 for m in health_metrics if m.value == 1)
        availability = (success_count / len(health_metrics)) * 100
        
        return availability

class ExecutionQualityAnalyzer:
    """Analyzes trade execution quality, including slippage and fill rates"""
    
    def __init__(self, metrics_store: MetricsStore):
        """
        Initialize execution quality analyzer
        
        Args:
            metrics_store: Storage for collected metrics
        """
        self.metrics_store = metrics_store
    
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
            order_type: Type of order (market, limit, etc.)
            expected_price: Expected execution price
            executed_price: Actual execution price
            quantity: Order quantity
            side: Order side (buy/sell)
            time_in_force: Time in force
            submission_time: Time order was submitted
            execution_time: Time order was executed
            metadata: Additional contextual data
        """
        # Calculate execution time (ms)
        execution_time_ms = (execution_time - submission_time) * 1000
        
        # Calculate slippage
        is_buy = side.lower() == "buy"
        # For buys, positive slippage means executed price is lower than expected
        # For sells, positive slippage means executed price is higher than expected
        slippage = (expected_price - executed_price) if is_buy else (executed_price - expected_price)
        
        # Calculate slippage percentage
        slippage_pct = (slippage / expected_price) * 100 if expected_price > 0 else 0
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "order_id": order_id,
            "symbol": symbol,
            "order_type": order_type,
            "expected_price": expected_price,
            "executed_price": executed_price,
            "quantity": quantity,
            "side": side,
            "time_in_force": time_in_force,
            "submission_time": submission_time,
            "execution_time": execution_time,
            "execution_time_ms": execution_time_ms
        })
        
        # Record execution time
        time_metric = MetricValue(
            value=execution_time_ms,
            metric_type=MetricType.EXECUTION_QUALITY,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            metadata=metadata
        )
        
        self.metrics_store.store(time_metric)
        
        # Record slippage
        slippage_metric = MetricValue(
            value=slippage_pct,
            metric_type=MetricType.EXECUTION_QUALITY,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            metadata=metadata
        )
        
        self.metrics_store.store(slippage_metric)
    
    def record_fill_rate(
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
        # Calculate fill rate percentage
        fill_rate = (filled_quantity / requested_quantity) * 100 if requested_quantity > 0 else 0
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "order_id": order_id,
            "symbol": symbol,
            "order_type": order_type,
            "requested_quantity": requested_quantity,
            "filled_quantity": filled_quantity,
            "side": side
        })
        
        # Record fill rate
        metric = MetricValue(
            value=fill_rate,
            metric_type=MetricType.EXECUTION_QUALITY,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            metadata=metadata
        )
        
        self.metrics_store.store(metric)
    
    def get_avg_slippage(
        self,
        broker_id: str,
        period: MetricPeriod = MetricPeriod.DAY,
        symbol: Optional[str] = None
    ) -> float:
        """
        Get average slippage percentage
        
        Args:
            broker_id: Broker identifier
            period: Time period for calculation
            symbol: Optional symbol for filtering
            
        Returns:
            Average slippage percentage
        """
        # Query slippage metrics
        metrics = self.metrics_store.query(
            metric_type=MetricType.EXECUTION_QUALITY,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            period=period,
            limit=10000
        )
        
        # Filter by symbol if specified
        if symbol:
            metrics = [m for m in metrics if m.metadata.get("symbol") == symbol]
        
        # Filter out metrics that aren't slippage (those with execution_time_ms in metadata)
        slippage_metrics = [m for m in metrics if "execution_time_ms" not in m.metadata]
        
        if not slippage_metrics:
            return 0.0
        
        # Calculate average
        total = sum(m.value for m in slippage_metrics)
        return total / len(slippage_metrics)

class CostTracker:
    """Tracks trading costs, including commissions and fees"""
    
    def __init__(self, metrics_store: MetricsStore):
        """
        Initialize cost tracker
        
        Args:
            metrics_store: Storage for collected metrics
        """
        self.metrics_store = metrics_store
    
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
        Record commission and fees for a trade
        
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
        # Calculate total cost
        total_cost = commission + exchange_fee + regulatory_fee + other_fees
        
        # Calculate cost percentage if trade value provided
        cost_percentage = (total_cost / trade_value) * 100 if trade_value > 0 else 0
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "order_id": order_id,
            "symbol": symbol,
            "commission": commission,
            "exchange_fee": exchange_fee,
            "regulatory_fee": regulatory_fee,
            "other_fees": other_fees,
            "trade_value": trade_value,
            "total_cost": total_cost,
            "cost_percentage": cost_percentage
        })
        
        # Record commission
        metric = MetricValue(
            value=total_cost,
            metric_type=MetricType.COST,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            metadata=metadata
        )
        
        self.metrics_store.store(metric)
    
    def get_avg_commission(
        self,
        broker_id: str,
        period: MetricPeriod = MetricPeriod.DAY,
        symbol: Optional[str] = None
    ) -> float:
        """
        Get average commission cost
        
        Args:
            broker_id: Broker identifier
            period: Time period for calculation
            symbol: Optional symbol for filtering
            
        Returns:
            Average commission cost
        """
        # Query cost metrics
        metrics = self.metrics_store.query(
            metric_type=MetricType.COST,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            period=period,
            limit=10000
        )
        
        # Filter by symbol if specified
        if symbol:
            metrics = [m for m in metrics if m.metadata.get("symbol") == symbol]
        
        if not metrics:
            return 0.0
        
        # Calculate average
        total = sum(m.value for m in metrics)
        return total / len(metrics)
    
    def get_total_costs(
        self,
        broker_id: str,
        period: MetricPeriod = MetricPeriod.DAY
    ) -> Dict[str, float]:
        """
        Get total trading costs breakdown
        
        Args:
            broker_id: Broker identifier
            period: Time period for calculation
            
        Returns:
            Dictionary of cost breakdowns
        """
        # Query cost metrics
        metrics = self.metrics_store.query(
            metric_type=MetricType.COST,
            operation=MetricOperation.PLACE_ORDER,
            broker_id=broker_id,
            period=period,
            limit=10000
        )
        
        if not metrics:
            return {
                "total_commission": 0.0,
                "total_exchange_fee": 0.0, 
                "total_regulatory_fee": 0.0,
                "total_other_fees": 0.0,
                "overall_total": 0.0
            }
        
        # Calculate totals
        total_commission = sum(m.metadata.get("commission", 0) for m in metrics)
        total_exchange_fee = sum(m.metadata.get("exchange_fee", 0) for m in metrics)
        total_regulatory_fee = sum(m.metadata.get("regulatory_fee", 0) for m in metrics)
        total_other_fees = sum(m.metadata.get("other_fees", 0) for m in metrics)
        
        return {
            "total_commission": total_commission,
            "total_exchange_fee": total_exchange_fee,
            "total_regulatory_fee": total_regulatory_fee,
            "total_other_fees": total_other_fees,
            "overall_total": total_commission + total_exchange_fee + total_regulatory_fee + total_other_fees
        }
