#!/usr/bin/env python3
"""
TWAP (Time-Weighted Average Price) Order Executor for BensBot

This module implements the TWAP algorithm for order execution,
dividing orders into equal-sized chunks executed at regular time intervals.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import math
import random

from trading_bot.core.advanced_orders import (
    TWAPOrder, AlgorithmicChunkPlaced, AlgorithmicOrderProgress
)
from trading_bot.core.algo_execution.base_executor import (
    AlgorithmicExecutionBase, AlgorithmicOrderTracker
)
from trading_bot.event_system.event_bus import EventBus


class TWAPExecutor(AlgorithmicExecutionBase):
    """
    TWAP (Time-Weighted Average Price) Order Executor
    
    Implements the TWAP algorithm, dividing orders into equal-sized chunks
    executed at regular time intervals.
    """
    
    def __init__(self, broker_manager, event_bus: EventBus, market_data_provider=None):
        """
        Initialize the TWAP executor
        
        Args:
            broker_manager: Multi-broker manager for order routing
            event_bus: EventBus for order events
            market_data_provider: Provider for market data (optional)
        """
        super().__init__(broker_manager, event_bus)
        self.market_data_provider = market_data_provider
        self.logger = logging.getLogger(__name__)
    
    def _create_order_tracker(self, order: TWAPOrder) -> 'TWAPOrderTracker':
        """
        Create a tracker for a TWAP order
        
        Args:
            order: TWAP order
            
        Returns:
            TWAP order tracker
        """
        return TWAPOrderTracker(order, self.event_bus)
    
    def _execute_algorithm(self, order_id: str):
        """
        Execute a TWAP order
        
        This method runs in a separate thread and manages the execution
        of the TWAP algorithm, including chunk scheduling and placement.
        
        Args:
            order_id: Order ID
        """
        with self.lock:
            tracker = self.active_orders.get(order_id)
            
            if not tracker:
                self.logger.error(f"TWAP order {order_id} not found")
                return
        
        order = tracker.order
        
        try:
            # Generate execution schedule
            self._generate_execution_schedule(tracker)
            
            # Execute chunks according to schedule
            self._execute_chunks_according_to_schedule(tracker)
            
        except Exception as e:
            self.logger.error(f"Error executing TWAP order {order_id}: {str(e)}")
            
            # Fail the order
            remaining = tracker.get_remaining_quantity()
            self._fail_order(order_id, f"Error executing TWAP: {str(e)}", remaining)
        
        # Clean up thread
        with self.lock:
            if order_id in self.executor_threads:
                del self.executor_threads[order_id]
    
    def _generate_execution_schedule(self, tracker: 'TWAPOrderTracker'):
        """
        Generate execution schedule for a TWAP order
        
        Args:
            tracker: TWAP order tracker
        """
        order = tracker.order
        
        # Get order parameters
        total_quantity = order.total_quantity
        num_chunks = order.num_chunks
        
        # If num_chunks is not specified, calculate based on duration and interval
        if not num_chunks or num_chunks <= 0:
            if order.duration_seconds and order.interval_seconds:
                num_chunks = max(1, math.floor(order.duration_seconds / order.interval_seconds))
            else:
                num_chunks = 10  # Default to 10 chunks
        
        # Calculate chunk size
        base_chunk_size = total_quantity / num_chunks
        
        # Generate schedule
        schedule = []
        start_time = datetime.now()
        
        for i in range(num_chunks):
            # Apply randomization if enabled
            if order.randomize_times and order.randomize_percent > 0:
                rand_factor = random.uniform(
                    -order.randomize_percent / 100, 
                    order.randomize_percent / 100
                )
                rand_offset_seconds = order.interval_seconds * rand_factor
            else:
                rand_offset_seconds = 0
            
            # Calculate scheduled time
            seconds_offset = i * order.interval_seconds + rand_offset_seconds
            scheduled_time = start_time + timedelta(seconds=seconds_offset)
            
            # Apply size randomization if enabled
            if order.randomize_sizes and order.randomize_percent > 0:
                rand_factor = random.uniform(
                    1 - (order.randomize_percent / 100), 
                    1 + (order.randomize_percent / 100)
                )
                chunk_size = base_chunk_size * rand_factor
            else:
                chunk_size = base_chunk_size
            
            # Ensure we don't exceed total quantity with randomization
            remaining = total_quantity - sum(chunk.get("quantity", 0) for chunk in schedule)
            if i == num_chunks - 1:  # Last chunk
                chunk_size = remaining
            else:
                chunk_size = min(chunk_size, remaining * 0.95)  # Leave some for the last chunk
            
            # Create chunk entry
            chunk = {
                "chunk_index": i,
                "quantity": chunk_size,
                "scheduled_time": scheduled_time,
                "status": "scheduled"
            }
            
            schedule.append(chunk)
        
        # Save schedule to tracker
        tracker.set_schedule(schedule)
        
        self.logger.info(
            f"Generated TWAP schedule for {order_id}: {num_chunks} chunks "
            f"over {order.duration_seconds} seconds"
        )
    
    def _execute_chunks_according_to_schedule(self, tracker: 'TWAPOrderTracker'):
        """
        Execute chunks according to the schedule
        
        Args:
            tracker: TWAP order tracker
        """
        order = tracker.order
        schedule = tracker.get_schedule()
        
        # Get broker ID
        broker_id = order.broker_id or self.broker_manager.get_preferred_broker(
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side
        )
        
        # Execute each chunk at its scheduled time
        for chunk in schedule:
            # Check if order was cancelled or failed
            with self.lock:
                if order.order_id not in self.active_orders:
                    self.logger.info(f"TWAP order {order.order_id} was cancelled or completed")
                    break
            
            # Wait until scheduled time
            now = datetime.now()
            scheduled_time = datetime.fromisoformat(chunk["scheduled_time"]) if isinstance(chunk["scheduled_time"], str) else chunk["scheduled_time"]
            
            if now < scheduled_time:
                wait_seconds = (scheduled_time - now).total_seconds()
                
                # Wait with periodic checks for cancellation
                wait_start = time.time()
                while time.time() - wait_start < wait_seconds:
                    # Check for cancellation every second
                    with self.lock:
                        if order.order_id not in self.active_orders:
                            break
                    
                    # Wait up to 1 second
                    time.sleep(min(1.0, wait_seconds - (time.time() - wait_start)))
                    
                    # Check if we've waited long enough
                    if time.time() - wait_start >= wait_seconds:
                        break
            
            # Check again if order was cancelled
            with self.lock:
                if order.order_id not in self.active_orders:
                    break
            
            # Place chunk order
            chunk_index = chunk["chunk_index"]
            quantity = chunk["quantity"]
            
            self.logger.debug(
                f"Placing TWAP chunk {chunk_index} for order {order.order_id}, "
                f"quantity: {quantity}, scheduled: {scheduled_time.isoformat()}"
            )
            
            # Apply market adaptation if enabled
            adapted_params = self._adapt_to_market_conditions(
                tracker, chunk_index, quantity, order.price
            )
            
            quantity = adapted_params.get("quantity", quantity)
            price = adapted_params.get("price", order.price)
            
            # Generate chunk order ID
            chunk_order_id = f"{order.order_id}_chunk_{chunk_index}"
            
            try:
                # Place the order with broker manager
                result = self.broker_manager.place_order(
                    broker_id=broker_id,
                    symbol=order.symbol,
                    quantity=quantity,
                    side=order.side,
                    order_type=order.order_type,
                    price=price,
                    time_in_force=order.time_in_force,
                    metadata={
                        "algo_order_id": order.order_id,
                        "chunk_index": chunk_index,
                        "algorithm": "TWAP"
                    }
                )
                
                # Get assigned order ID
                assigned_order_id = result.get("order_id")
                
                if not assigned_order_id:
                    error_msg = f"Failed to place TWAP chunk {chunk_index}: {result.get('message', 'No order ID returned')}"
                    self.logger.error(error_msg)
                    
                    # Add retry logic here if needed
                    retry_count = tracker.increment_retry_count(chunk_index)
                    
                    if retry_count >= order.max_retries:
                        remaining = tracker.get_remaining_quantity()
                        self._fail_order(order.order_id, error_msg, remaining)
                        break
                    
                    # Wait before retry
                    time.sleep(1.0)
                    continue
                
                # Update tracker with chunk order
                with self.lock:
                    # Map chunk to parent order
                    self.chunk_map[assigned_order_id] = order.order_id
                    
                    # Record chunk
                    tracker.add_chunk(
                        chunk_index=chunk_index,
                        chunk_order_id=assigned_order_id,
                        quantity=quantity,
                        scheduled_time=scheduled_time
                    )
                
                # Emit chunk placed event
                self.event_bus.emit(AlgorithmicChunkPlaced(
                    order_id=order.order_id,
                    chunk_index=chunk_index,
                    quantity=quantity,
                    remaining_quantity=tracker.get_remaining_quantity(),
                    chunk_order_id=assigned_order_id,
                    algorithm="TWAP"
                ))
                
                # Calculate drift from schedule
                actual_time = datetime.now()
                drift_seconds = (actual_time - scheduled_time).total_seconds()
                
                tracker.update_performance_metric(f"chunk_{chunk_index}_drift_seconds", drift_seconds)
                
                # Emit progress event
                self._emit_progress_event(tracker)
                
            except Exception as e:
                self.logger.error(f"Error executing TWAP chunk {chunk_index}: {str(e)}")
                
                # Add retry logic if needed
                retry_count = tracker.increment_retry_count(chunk_index)
                
                if retry_count >= order.max_retries:
                    remaining = tracker.get_remaining_quantity()
                    self._fail_order(order.order_id, f"Error executing TWAP chunk: {str(e)}", remaining)
                    break
                
                # Wait before retry
                time.sleep(1.0)
    
    def _adapt_to_market_conditions(self, tracker: 'TWAPOrderTracker', 
                                   chunk_index: int, quantity: float, 
                                   price: Optional[float]) -> Dict[str, Any]:
        """
        Adapt order parameters to current market conditions
        
        Args:
            tracker: TWAP order tracker
            chunk_index: Chunk index
            quantity: Original quantity
            price: Original price
            
        Returns:
            Dictionary with adapted parameters
        """
        order = tracker.order
        
        # If adaptation is disabled, return original params
        if not order.adapt_to_market:
            return {"quantity": quantity, "price": price}
        
        # Get market data if available
        if self.market_data_provider:
            try:
                # Get current market conditions
                market_data = self.market_data_provider.get_market_data(order.symbol)
                
                # Analyze volatility
                volatility = market_data.get("volatility", 0.0)
                
                # Adapt quantity based on volatility
                if volatility > order.high_volatility_threshold:
                    # Reduce quantity during high volatility
                    quantity = quantity * (1 - order.volatility_adjustment_factor)
                elif volatility < order.low_volatility_threshold:
                    # Increase quantity during low volatility
                    quantity = quantity * (1 + order.volatility_adjustment_factor)
                
                # Adapt price if limit order
                if price and order.order_type.lower() in ["limit", "stop_limit"]:
                    # Get current price
                    current_price = market_data.get("last_price")
                    
                    if current_price:
                        if order.side.lower() == "buy":
                            # For buy orders, adjust price to match market movement
                            price = min(price, current_price * (1 + order.price_buffer_percent/100))
                        else:
                            # For sell orders, adjust price to match market movement
                            price = max(price, current_price * (1 - order.price_buffer_percent/100))
                
                # Track market conditions
                tracker.update_performance_metric(f"chunk_{chunk_index}_volatility", volatility)
                tracker.update_performance_metric(f"chunk_{chunk_index}_market_price", market_data.get("last_price"))
                
            except Exception as e:
                self.logger.warning(f"Error adapting to market conditions: {str(e)}")
        
        return {"quantity": quantity, "price": price}
    
    def _get_performance_metrics(self, tracker: 'TWAPOrderTracker') -> Dict[str, Any]:
        """
        Get performance metrics for a TWAP order
        
        Args:
            tracker: TWAP order tracker
            
        Returns:
            Performance metrics
        """
        metrics = super()._get_performance_metrics(tracker)
        
        # Add TWAP-specific performance metrics
        
        # Time drift metrics
        chunk_drifts = [
            value for key, value in tracker.get_performance_metrics().items()
            if key.startswith("chunk_") and key.endswith("_drift_seconds")
        ]
        
        if chunk_drifts:
            metrics["avg_time_drift_seconds"] = sum(chunk_drifts) / len(chunk_drifts)
            metrics["max_time_drift_seconds"] = max(chunk_drifts)
        
        # Calculate performance vs benchmark (if market data available)
        if self.market_data_provider:
            try:
                # Get TWAP benchmark for the order's time window
                start_time = tracker.start_time
                end_time = datetime.now()
                
                benchmark_twap = self.market_data_provider.get_twap(
                    symbol=tracker.order.symbol,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if benchmark_twap:
                    # Calculate our TWAP
                    our_twap = tracker.get_average_fill_price()
                    
                    # Performance vs benchmark
                    if our_twap and benchmark_twap:
                        if tracker.order.side.lower() == "buy":
                            # For buys, lower is better
                            metrics["vs_benchmark_bps"] = (benchmark_twap - our_twap) / benchmark_twap * 10000
                        else:
                            # For sells, higher is better
                            metrics["vs_benchmark_bps"] = (our_twap - benchmark_twap) / benchmark_twap * 10000
                        
                        metrics["benchmark_twap"] = benchmark_twap
                        metrics["our_twap"] = our_twap
            
            except Exception as e:
                self.logger.warning(f"Error calculating TWAP benchmark metrics: {str(e)}")
        
        return metrics


class TWAPOrderTracker(AlgorithmicOrderTracker):
    """
    Tracks the status of a TWAP order
    """
    
    def __init__(self, order: TWAPOrder, event_bus: EventBus):
        """
        Initialize the TWAP order tracker
        
        Args:
            order: TWAP order
            event_bus: EventBus
        """
        super().__init__(order, event_bus)
        
        # Additional TWAP-specific tracking
        self.drifts = []  # Time drifts from schedule
