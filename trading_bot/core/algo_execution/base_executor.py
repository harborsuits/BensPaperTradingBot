#!/usr/bin/env python3
"""
Base Algorithmic Order Executor for BensBot

This module implements the base class for algorithmic order execution
strategies, providing common functionality for TWAP, VWAP and other
execution algorithms.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import asyncio
from uuid import uuid4

from trading_bot.core.advanced_orders import (
    AlgorithmicOrder, AlgorithmicOrderStarted, 
    AlgorithmicChunkPlaced, AlgorithmicChunkFilled,
    AlgorithmicOrderCompleted, AlgorithmicOrderFailed,
    AlgorithmicOrderProgress, AlgorithmicOrderStatus
)
from trading_bot.event_system.event_bus import EventBus


class AlgorithmicExecutionBase:
    """
    Base class for algorithmic order execution strategies
    
    Implements common functionality for TWAP, VWAP and other execution
    algorithms, including chunk tracking, progress reporting, and
    order lifecycle management.
    """
    
    def __init__(self, broker_manager, event_bus: EventBus):
        """
        Initialize the algorithmic order executor
        
        Args:
            broker_manager: Multi-broker manager for order routing
            event_bus: EventBus for order events
        """
        self.broker_manager = broker_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Track active algorithmic orders
        self.active_orders = {}  # order_id -> OrderTracker
        
        # Track chunks to parent orders
        self.chunk_map = {}  # chunk_order_id -> algorithmic_order_id
        
        # Thread pools
        self.executor_threads = {}  # order_id -> thread
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Flag for thread shutdown
        self.shutdown = False
        
        # Subscribe to order events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to order fill events"""
        # Subscribe to order fill events to track chunk fills
        from trading_bot.core.events import OrderFilled, OrderPartialFill, OrderRejected
        
        self.event_bus.on(OrderFilled, self._on_order_filled)
        self.event_bus.on(OrderPartialFill, self._on_order_partial_fill)
        self.event_bus.on(OrderRejected, self._on_order_rejected)
    
    def execute_order(self, order: AlgorithmicOrder) -> Dict[str, Any]:
        """
        Execute an algorithmic order
        
        Args:
            order: Algorithmic order to execute
            
        Returns:
            Dictionary with execution result
        """
        self.logger.info(
            f"Executing algorithmic order {order.order_id} for {order.symbol}, "
            f"algorithm: {order.algorithm_type}, quantity: {order.total_quantity}"
        )
        
        # Create tracker for this order
        tracker = self._create_order_tracker(order)
        
        with self.lock:
            self.active_orders[order.order_id] = tracker
        
        # Emit order started event
        self.event_bus.emit(AlgorithmicOrderStarted(
            order_id=order.order_id,
            symbol=order.symbol,
            algorithm=order.algorithm_type,
            total_quantity=order.total_quantity,
            start_time=datetime.now().isoformat(),
            end_time=(datetime.now() + timedelta(seconds=order.duration_seconds)).isoformat() 
                     if order.duration_seconds else None,
            metadata=order.user_metadata
        ))
        
        # Start executor thread
        thread = threading.Thread(
            target=self._execute_algorithm,
            args=(order.order_id,),
            daemon=True
        )
        
        with self.lock:
            self.executor_threads[order.order_id] = thread
        
        thread.start()
        
        # Return immediate result
        return {
            "order_id": order.order_id,
            "status": "in_progress",
            "message": f"Algorithmic order execution started: {order.algorithm_type}"
        }
    
    def _create_order_tracker(self, order: AlgorithmicOrder) -> 'AlgorithmicOrderTracker':
        """
        Create a tracker for an algorithmic order
        
        Args:
            order: Algorithmic order
            
        Returns:
            Order tracker
        """
        # This is a base implementation; subclasses may override
        return AlgorithmicOrderTracker(order, self.event_bus)
    
    def _execute_algorithm(self, order_id: str):
        """
        Execute an algorithmic order
        
        Args:
            order_id: Order ID
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _execute_algorithm")
    
    def _on_order_filled(self, event):
        """
        Handle order filled event
        
        Args:
            event: OrderFilled event
        """
        order_id = event.order_id
        
        # Check if this is a chunk of an algorithmic order
        algo_order_id = self._get_parent_order_for_chunk(order_id)
        if not algo_order_id:
            return
        
        with self.lock:
            tracker = self.active_orders.get(algo_order_id)
            
            if not tracker:
                # Clean up chunk mapping
                if order_id in self.chunk_map:
                    del self.chunk_map[order_id]
                return
            
            # Update chunk fill
            tracker.set_chunk_filled(
                chunk_order_id=order_id,
                fill_quantity=event.total_qty,
                fill_price=event.avg_fill_price
            )
            
            # Get chunk info
            chunk_info = tracker.get_chunk_by_order_id(order_id)
            chunk_index = chunk_info.get("chunk_index", 0)
            
            # Emit chunk filled event
            self.event_bus.emit(AlgorithmicChunkFilled(
                order_id=algo_order_id,
                chunk_index=chunk_index,
                quantity=event.total_qty,
                price=event.avg_fill_price,
                remaining_quantity=tracker.get_remaining_quantity(),
                algorithm=tracker.order.algorithm_type
            ))
            
            # Check if all chunks are filled and order is complete
            if tracker.is_complete():
                self._complete_order(algo_order_id)
            else:
                # Emit progress event
                self._emit_progress_event(tracker)
    
    def _on_order_partial_fill(self, event):
        """
        Handle order partial fill event
        
        Args:
            event: OrderPartialFill event
        """
        order_id = event.order_id
        
        # Check if this is a chunk of an algorithmic order
        algo_order_id = self._get_parent_order_for_chunk(order_id)
        if not algo_order_id:
            return
        
        with self.lock:
            tracker = self.active_orders.get(algo_order_id)
            
            if not tracker:
                return
            
            # Update chunk partial fill
            tracker.update_chunk_partial_fill(
                chunk_order_id=order_id,
                filled_quantity=event.filled_qty,
                fill_price=event.fill_price,
                remaining_quantity=event.remaining_qty
            )
            
            # Emit progress event
            self._emit_progress_event(tracker)
    
    def _on_order_rejected(self, event):
        """
        Handle order rejected event
        
        Args:
            event: OrderRejected event
        """
        order_id = event.order_id
        
        # Check if this is a chunk of an algorithmic order
        algo_order_id = self._get_parent_order_for_chunk(order_id)
        if not algo_order_id:
            return
        
        with self.lock:
            tracker = self.active_orders.get(algo_order_id)
            
            if not tracker:
                # Clean up chunk mapping
                if order_id in self.chunk_map:
                    del self.chunk_map[order_id]
                return
            
            # Set chunk rejected
            tracker.set_chunk_rejected(
                chunk_order_id=order_id,
                reason=event.reason
            )
            
            # The executor thread will handle retries
    
    def _get_parent_order_for_chunk(self, chunk_order_id: str) -> Optional[str]:
        """
        Get parent order ID for a chunk order ID
        
        Args:
            chunk_order_id: Chunk order ID
            
        Returns:
            Parent order ID or None
        """
        with self.lock:
            return self.chunk_map.get(chunk_order_id)
    
    def _fail_order(self, order_id: str, reason: str, remaining_quantity: float):
        """
        Fail an algorithmic order
        
        Args:
            order_id: Order ID
            reason: Failure reason
            remaining_quantity: Remaining quantity
        """
        with self.lock:
            tracker = self.active_orders.get(order_id)
            
            if not tracker:
                return
            
            # Calculate filled quantity
            total_quantity = tracker.order.total_quantity
            filled_quantity = total_quantity - remaining_quantity
            
            # Get performance metrics
            performance_metrics = self._get_performance_metrics(tracker)
            
            # Emit failure event
            self.event_bus.emit(AlgorithmicOrderFailed(
                order_id=order_id,
                reason=reason,
                filled_quantity=filled_quantity,
                remaining_quantity=remaining_quantity,
                algorithm=tracker.order.algorithm_type,
                performance_metrics=performance_metrics
            ))
            
            # Clean up
            del self.active_orders[order_id]
            
            # Clean up chunk mappings
            chunk_order_ids = tracker.get_all_chunk_order_ids()
            for chunk_id in chunk_order_ids:
                if chunk_id in self.chunk_map:
                    del self.chunk_map[chunk_id]
    
    def _complete_order(self, order_id: str):
        """
        Complete an algorithmic order
        
        Args:
            order_id: Order ID
        """
        with self.lock:
            tracker = self.active_orders.get(order_id)
            
            if not tracker:
                return
            
            # Calculate execution metrics
            total_quantity = tracker.order.total_quantity
            total_filled = tracker.get_total_filled_quantity()
            avg_price = tracker.get_average_fill_price()
            
            start_time = tracker.start_time
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Get performance metrics
            performance_metrics = self._get_performance_metrics(tracker)
            
            # Emit completion event
            self.event_bus.emit(AlgorithmicOrderCompleted(
                order_id=order_id,
                total_filled=total_filled,
                avg_price=avg_price,
                execution_time_ms=execution_time_ms,
                chunks_count=tracker.get_chunk_count(),
                algorithm=tracker.order.algorithm_type,
                performance_metrics=performance_metrics
            ))
            
            # Clean up
            del self.active_orders[order_id]
            
            # Clean up chunk mappings
            chunk_order_ids = tracker.get_all_chunk_order_ids()
            for chunk_id in chunk_order_ids:
                if chunk_id in self.chunk_map:
                    del self.chunk_map[chunk_id]
    
    def _get_performance_metrics(self, tracker) -> Dict[str, Any]:
        """
        Get performance metrics for an algorithmic order
        
        Args:
            tracker: Order tracker
            
        Returns:
            Performance metrics
        """
        # This method should be implemented by subclasses
        # Base implementation returns basic metrics
        return {
            "avg_price": tracker.get_average_fill_price(),
            "execution_time_ms": int((datetime.now() - tracker.start_time).total_seconds() * 1000),
            "chunks_completed": tracker.get_filled_chunk_count(),
            "chunks_total": tracker.get_chunk_count()
        }
    
    def _emit_progress_event(self, tracker):
        """
        Emit a progress event for an algorithmic order
        
        Args:
            tracker: Order tracker
        """
        # Calculate progress metrics
        filled_quantity = tracker.get_total_filled_quantity()
        remaining_quantity = tracker.get_remaining_quantity()
        total_quantity = tracker.order.total_quantity
        
        progress_pct = 0
        if total_quantity > 0:
            progress_pct = int((filled_quantity / total_quantity) * 100)
        
        avg_price = tracker.get_average_fill_price()
        
        # Get time metrics
        start_time = tracker.start_time
        current_time = datetime.now()
        elapsed_ms = int((current_time - start_time).total_seconds() * 1000)
        
        # Estimate remaining time based on progress
        remaining_ms = 0
        if progress_pct > 0:
            remaining_ms = int((elapsed_ms / progress_pct) * (100 - progress_pct))
        
        # Emit progress event
        self.event_bus.emit(AlgorithmicOrderProgress(
            order_id=tracker.order.order_id,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            progress_percent=progress_pct,
            avg_price=avg_price,
            elapsed_ms=elapsed_ms,
            estimated_remaining_ms=remaining_ms,
            algorithm=tracker.order.algorithm_type
        ))
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an algorithmic order
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation result
        """
        with self.lock:
            tracker = self.active_orders.get(order_id)
            
            if not tracker:
                return {"status": "error", "message": f"Algorithmic order {order_id} not found"}
            
            # Cancel all unfilled chunks
            active_chunks = tracker.get_active_chunks()
            cancelled_chunks = []
            failed_chunks = []
            
            for chunk_info in active_chunks:
                chunk_order_id = chunk_info.get("order_id")
                
                if chunk_order_id:
                    # Get broker ID from the order
                    broker_id = tracker.order.broker_id
                    
                    if not broker_id:
                        # Use default broker selection
                        broker_id = self.broker_manager.get_preferred_broker(
                            symbol=tracker.order.symbol,
                            order_type=tracker.order.order_type,
                            side=tracker.order.side
                        )
                    
                    try:
                        # Cancel order
                        result = self.broker_manager.cancel_order(broker_id, chunk_order_id)
                        
                        if result.get("status") == "ok":
                            cancelled_chunks.append(chunk_order_id)
                        else:
                            failed_chunks.append(chunk_order_id)
                            
                    except Exception as e:
                        self.logger.error(f"Error cancelling chunk {chunk_order_id}: {str(e)}")
                        failed_chunks.append(chunk_order_id)
            
            # Fail the order
            remaining = tracker.get_remaining_quantity()
            self._fail_order(order_id, "Cancelled by user", remaining)
            
            return {
                "status": "ok",
                "cancelled_chunks": cancelled_chunks,
                "failed_chunks": failed_chunks,
                "message": "Algorithmic order cancelled"
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an algorithmic order
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status information
        """
        with self.lock:
            tracker = self.active_orders.get(order_id)
            
            if not tracker:
                return {"status": "unknown", "message": f"Algorithmic order {order_id} not found"}
            
            return tracker.get_status()
    
    def shutdown_executor(self):
        """Shut down the executor"""
        self.shutdown = True
        
        # Wait for all threads to exit
        with self.lock:
            threads = list(self.executor_threads.values())
        
        for thread in threads:
            thread.join(timeout=1.0)


class AlgorithmicOrderTracker:
    """
    Tracks the status of an algorithmic order's chunks
    """
    
    def __init__(self, order: AlgorithmicOrder, event_bus: EventBus):
        """
        Initialize the algorithmic order tracker
        
        Args:
            order: Algorithmic order
            event_bus: EventBus
        """
        self.order = order
        self.event_bus = event_bus
        self.start_time = datetime.now()
        
        # Track chunks
        self.chunks = []  # List of chunk info dictionaries
        
        # Track execution schedule
        self.schedule = []  # List of scheduled chunks with timestamps
        
        # Track retry counts
        self.retry_counts = {}  # chunk_index -> retry_count
        
        # Track errors
        self.errors = []  # List of error messages
        
        # Performance metrics
        self.performance_metrics = {}
    
    def add_chunk(self, chunk_index: int, chunk_order_id: str, quantity: float, 
                 scheduled_time: datetime = None):
        """
        Add a chunk
        
        Args:
            chunk_index: Chunk index
            chunk_order_id: Chunk order ID
            quantity: Chunk quantity
            scheduled_time: Scheduled execution time
        """
        chunk_info = {
            "chunk_index": chunk_index,
            "order_id": chunk_order_id,
            "quantity": quantity,
            "filled_quantity": 0.0,
            "avg_price": 0.0,
            "status": "pending",
            "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,
            "created_at": datetime.now().isoformat()
        }
        
        self.chunks.append(chunk_info)
    
    def set_chunk_filled(self, chunk_order_id: str, fill_quantity: float, fill_price: float):
        """
        Set a chunk as filled
        
        Args:
            chunk_order_id: Chunk order ID
            fill_quantity: Fill quantity
            fill_price: Fill price
        """
        for chunk in self.chunks:
            if chunk.get("order_id") == chunk_order_id:
                chunk["filled_quantity"] = fill_quantity
                chunk["avg_price"] = fill_price
                chunk["status"] = "filled"
                chunk["filled_at"] = datetime.now().isoformat()
                break
    
    def update_chunk_partial_fill(self, chunk_order_id: str, filled_quantity: float, 
                                fill_price: float, remaining_quantity: float):
        """
        Update a chunk with partial fill information
        
        Args:
            chunk_order_id: Chunk order ID
            filled_quantity: Filled quantity
            fill_price: Fill price
            remaining_quantity: Remaining quantity
        """
        for chunk in self.chunks:
            if chunk.get("order_id") == chunk_order_id:
                # If status is already filled, don't update
                if chunk.get("status") == "filled":
                    return
                
                chunk["filled_quantity"] = filled_quantity
                chunk["avg_price"] = fill_price
                chunk["status"] = "partial"
                chunk["remaining_quantity"] = remaining_quantity
                chunk["last_update"] = datetime.now().isoformat()
                break
    
    def set_chunk_rejected(self, chunk_order_id: str, reason: str):
        """
        Set a chunk as rejected
        
        Args:
            chunk_order_id: Chunk order ID
            reason: Rejection reason
        """
        for chunk in self.chunks:
            if chunk.get("order_id") == chunk_order_id:
                chunk["status"] = "rejected"
                chunk["error"] = reason
                chunk["rejected_at"] = datetime.now().isoformat()
                break
        
        # Add to errors
        self.errors.append(f"Chunk {chunk_order_id} rejected: {reason}")
    
    def increment_retry_count(self, chunk_index: int) -> int:
        """
        Increment retry count for a chunk
        
        Args:
            chunk_index: Chunk index
            
        Returns:
            New retry count
        """
        current = self.retry_counts.get(chunk_index, 0)
        new_count = current + 1
        self.retry_counts[chunk_index] = new_count
        return new_count
    
    def get_chunk_by_order_id(self, chunk_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk info by order ID
        
        Args:
            chunk_order_id: Chunk order ID
            
        Returns:
            Chunk info or None
        """
        for chunk in self.chunks:
            if chunk.get("order_id") == chunk_order_id:
                return chunk
        return None
    
    def get_active_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all active (non-filled, non-rejected) chunks
        
        Returns:
            List of active chunk info
        """
        return [
            chunk for chunk in self.chunks
            if chunk.get("status") not in ["filled", "rejected"]
        ]
    
    def get_all_chunk_order_ids(self) -> List[str]:
        """
        Get all chunk order IDs
        
        Returns:
            List of chunk order IDs
        """
        return [chunk.get("order_id") for chunk in self.chunks if chunk.get("order_id")]
    
    def get_chunk_count(self) -> int:
        """
        Get the number of chunks
        
        Returns:
            Chunk count
        """
        return len(self.chunks)
    
    def get_filled_chunk_count(self) -> int:
        """
        Get the number of filled chunks
        
        Returns:
            Filled chunk count
        """
        return sum(1 for chunk in self.chunks if chunk.get("status") == "filled")
    
    def get_total_filled_quantity(self) -> float:
        """
        Get total filled quantity across all chunks
        
        Returns:
            Total filled quantity
        """
        return sum(chunk.get("filled_quantity", 0.0) for chunk in self.chunks)
    
    def get_remaining_quantity(self) -> float:
        """
        Get remaining quantity to be filled
        
        Returns:
            Remaining quantity
        """
        total_filled = self.get_total_filled_quantity()
        return self.order.total_quantity - total_filled
    
    def get_average_fill_price(self) -> float:
        """
        Get weighted average fill price across all chunks
        
        Returns:
            Average fill price
        """
        total_filled = 0.0
        total_value = 0.0
        
        for chunk in self.chunks:
            filled_qty = chunk.get("filled_quantity", 0.0)
            if filled_qty > 0:
                avg_price = chunk.get("avg_price", 0.0)
                total_filled += filled_qty
                total_value += filled_qty * avg_price
        
        if total_filled > 0:
            return total_value / total_filled
        else:
            return 0.0
    
    def is_complete(self) -> bool:
        """
        Check if the order is complete (all chunks filled)
        
        Returns:
            True if complete
        """
        # Check if total filled quantity matches total order quantity
        total_filled = self.get_total_filled_quantity()
        return abs(total_filled - self.order.total_quantity) < 0.0001
    
    def has_errors(self) -> bool:
        """
        Check if the order has errors
        
        Returns:
            True if errors exist
        """
        return len(self.errors) > 0
    
    def get_error_reason(self) -> str:
        """
        Get combined error reason
        
        Returns:
            Error reason string
        """
        if not self.errors:
            return ""
        
        if len(self.errors) == 1:
            return self.errors[0]
        else:
            return f"Multiple errors: {', '.join(self.errors[:3])}" + (
                f" and {len(self.errors) - 3} more" if len(self.errors) > 3 else ""
            )
    
    def update_performance_metric(self, key: str, value: Any):
        """
        Update a performance metric
        
        Args:
            key: Metric key
            value: Metric value
        """
        self.performance_metrics[key] = value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Performance metrics
        """
        return self.performance_metrics
    
    def set_schedule(self, schedule: List[Dict[str, Any]]):
        """
        Set the execution schedule
        
        Args:
            schedule: List of scheduled chunks
        """
        self.schedule = schedule
    
    def get_schedule(self) -> List[Dict[str, Any]]:
        """
        Get the execution schedule
        
        Returns:
            Execution schedule
        """
        return self.schedule
    
    def get_next_scheduled_chunk(self) -> Optional[Dict[str, Any]]:
        """
        Get the next scheduled chunk
        
        Returns:
            Next scheduled chunk or None
        """
        # Filter to pending chunks and order by scheduled time
        pending_chunks = [
            chunk for chunk in self.schedule
            if chunk.get("status") not in ["filled", "rejected"]
        ]
        
        if not pending_chunks:
            return None
        
        # Sort by scheduled time
        pending_chunks.sort(key=lambda x: x.get("scheduled_time"))
        
        return pending_chunks[0]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get order status
        
        Returns:
            Status information
        """
        # Determine overall status
        total_filled = self.get_total_filled_quantity()
        remaining = self.order.total_quantity - total_filled
        
        if self.is_complete():
            status = "filled"
        elif self.has_errors() and len(self.errors) > self.order.max_retries:
            status = "failed"
        elif total_filled > 0:
            status = "partial"
        else:
            status = "pending"
        
        # Create chunk info
        chunks_info = []
        for i, chunk in enumerate(self.chunks):
            chunks_info.append({
                "chunk_index": i,
                "order_id": chunk.get("order_id"),
                "quantity": chunk.get("quantity"),
                "filled_quantity": chunk.get("filled_quantity", 0.0),
                "avg_price": chunk.get("avg_price", 0.0),
                "status": chunk.get("status", "unknown"),
                "scheduled_time": chunk.get("scheduled_time"),
                "filled_at": chunk.get("filled_at")
            })
        
        return {
            "order_id": self.order.order_id,
            "symbol": self.order.symbol,
            "side": self.order.side,
            "status": status,
            "total_quantity": self.order.total_quantity,
            "filled_quantity": total_filled,
            "remaining_quantity": remaining,
            "avg_price": self.get_average_fill_price(),
            "chunks_count": len(self.chunks),
            "chunks_completed": self.get_filled_chunk_count(),
            "algorithm": self.order.algorithm_type,
            "error_reason": self.get_error_reason() if self.has_errors() else None,
            "start_time": self.start_time.isoformat(),
            "elapsed_ms": int((datetime.now() - self.start_time).total_seconds() * 1000),
            "performance_metrics": self.get_performance_metrics()
        }
