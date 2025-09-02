#!/usr/bin/env python3
"""
Iceberg Order Executor for BensBot

This module implements the logic for executing iceberg orders.
Breaks large orders into smaller pieces to minimize market impact,
while maintaining order tracking and completion management.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from uuid import uuid4

from trading_bot.core.advanced_orders import (
    IcebergOrder, IcebergOrderStarted, IcebergChunkPlaced,
    IcebergChunkFilled, IcebergOrderCompleted, IcebergOrderFailed
)
from trading_bot.event_system.event_bus import EventBus


class IcebergOrderExecutor:
    """
    Executor for iceberg orders
    
    Handles the execution of large orders by breaking them into smaller chunks
    and managing the execution of those chunks.
    """
    
    def __init__(self, broker_manager, event_bus: EventBus):
        """
        Initialize the iceberg order executor
        
        Args:
            broker_manager: Multi-broker manager for order routing
            event_bus: EventBus for order events
        """
        self.broker_manager = broker_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Track active iceberg orders
        self.active_icebergs = {}  # order_id -> IcebergOrderTracker
        
        # Track chunks to parent icebergs
        self.chunk_map = {}  # chunk_order_id -> iceberg_order_id
        
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
    
    def execute_iceberg(self, order: IcebergOrder) -> Dict[str, Any]:
        """
        Execute an iceberg order
        
        Args:
            order: Iceberg order to execute
            
        Returns:
            Dictionary with execution result
        """
        self.logger.info(
            f"Executing iceberg order {order.order_id} for {order.symbol}, "
            f"total: {order.total_quantity}, display: {order.display_quantity}"
        )
        
        # Create tracker for this iceberg
        tracker = IcebergOrderTracker(order, self.event_bus)
        
        with self.lock:
            self.active_icebergs[order.order_id] = tracker
        
        # Emit iceberg order started event
        self.event_bus.emit(IcebergOrderStarted(
            order_id=order.order_id,
            symbol=order.symbol,
            total_quantity=order.total_quantity,
            display_quantity=order.display_quantity,
            side=order.side,
            metadata=order.user_metadata
        ))
        
        # Start executor thread
        thread = threading.Thread(
            target=self._execute_chunks,
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
            "message": "Iceberg order execution started"
        }
    
    def _execute_chunks(self, order_id: str):
        """
        Execute chunks of an iceberg order
        
        Args:
            order_id: Iceberg order ID
        """
        with self.lock:
            tracker = self.active_icebergs.get(order_id)
            
            if not tracker:
                self.logger.error(f"Iceberg order {order_id} not found")
                return
        
        order = tracker.order
        remaining = order.total_quantity
        chunk_index = 0
        
        # Keep track of the broker for all chunks
        broker_id = order.broker_id or self.broker_manager.get_preferred_broker(
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side
        )
        
        while remaining > 0 and not self.shutdown:
            try:
                with self.lock:
                    # Check if order was cancelled
                    if order_id not in self.active_icebergs:
                        self.logger.info(f"Iceberg order {order_id} was cancelled or completed")
                        break
                    
                    # Get current tracker
                    tracker = self.active_icebergs.get(order_id)
                    
                    if not tracker:
                        break
                    
                    # Check for errors
                    if tracker.has_errors():
                        error_reason = tracker.get_error_reason()
                        self._fail_iceberg(order_id, error_reason, remaining)
                        break
                
                # Determine chunk size
                chunk_size = min(order.display_quantity, remaining)
                
                # Place chunk order
                self.logger.debug(f"Placing chunk {chunk_index} for iceberg {order_id}, size: {chunk_size}")
                
                chunk_order_id = f"{order_id}_chunk_{chunk_index}"
                
                # Determine if this is the final chunk
                is_final_chunk = (remaining <= order.display_quantity)
                
                # Place the order with broker manager
                result = self.broker_manager.place_order(
                    broker_id=broker_id,
                    symbol=order.symbol,
                    quantity=chunk_size,
                    side=order.side,
                    order_type=order.order_type,
                    price=order.price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    metadata={
                        "iceberg_order_id": order_id,
                        "chunk_index": chunk_index,
                        "is_final_chunk": is_final_chunk
                    }
                )
                
                # Get assigned order ID
                assigned_order_id = result.get("order_id")
                
                if not assigned_order_id:
                    error_msg = f"Failed to place chunk {chunk_index}: {result.get('message', 'No order ID returned')}"
                    self.logger.error(error_msg)
                    
                    # Add retry logic here
                    retry_count = tracker.increment_retry_count(chunk_index)
                    
                    if retry_count >= order.max_retries:
                        self._fail_iceberg(order_id, error_msg, remaining)
                        break
                    
                    # Wait before retry
                    time.sleep(1.0)
                    continue
                
                # Update tracker with chunk order
                with self.lock:
                    # Map chunk to parent iceberg
                    self.chunk_map[assigned_order_id] = order_id
                    
                    # Record chunk
                    tracker.add_chunk(
                        chunk_index=chunk_index,
                        chunk_order_id=assigned_order_id,
                        quantity=chunk_size
                    )
                
                # Emit chunk placed event
                self.event_bus.emit(IcebergChunkPlaced(
                    order_id=order_id,
                    chunk_index=chunk_index,
                    quantity=chunk_size,
                    remaining_quantity=remaining - chunk_size,
                    chunk_order_id=assigned_order_id
                ))
                
                # Wait for chunk to fill before placing next chunk
                filled = self._wait_for_chunk_fill(assigned_order_id, tracker, order.max_retries)
                
                if not filled:
                    # Check if we failed the whole iceberg
                    with self.lock:
                        if order_id not in self.active_icebergs:
                            break
                
                # Update remaining quantity
                remaining -= chunk_size
                chunk_index += 1
                
                # Wait between chunks
                if not is_final_chunk and order.child_delay_ms > 0:
                    time.sleep(order.child_delay_ms / 1000)
                
            except Exception as e:
                self.logger.error(f"Error executing chunk for iceberg {order_id}: {str(e)}")
                
                # Add retry logic here
                retry_count = tracker.increment_retry_count(chunk_index)
                
                if retry_count >= order.max_retries:
                    self._fail_iceberg(order_id, f"Error executing chunks: {str(e)}", remaining)
                    break
                
                # Wait before retry
                time.sleep(1.0)
        
        # Clean up thread
        with self.lock:
            if order_id in self.executor_threads:
                del self.executor_threads[order_id]
    
    def _wait_for_chunk_fill(self, chunk_order_id: str, tracker, max_retries: int) -> bool:
        """
        Wait for a chunk to fill
        
        Args:
            chunk_order_id: Chunk order ID
            tracker: Iceberg order tracker
            max_retries: Maximum retries
            
        Returns:
            True if filled, False if failed
        """
        # Check initial status
        chunk_info = tracker.get_chunk_by_order_id(chunk_order_id)
        
        if not chunk_info:
            return False
        
        # If already filled, return immediately
        if chunk_info.get("status") == "filled":
            return True
        
        # Wait for fill with timeout
        retry_count = 0
        max_wait_time = 30  # 30 seconds total wait time
        interval = 0.5  # Check every 0.5 seconds
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # Check if filled
            chunk_info = tracker.get_chunk_by_order_id(chunk_order_id)
            
            if chunk_info.get("status") == "filled":
                return True
            
            # Check if rejected or error
            if chunk_info.get("status") in ["rejected", "error"]:
                # Try to recover
                retry_count += 1
                
                if retry_count > max_retries:
                    return False
                
                # Could retry placing the chunk here
                
                # For now, just return false
                return False
            
            # Wait for next check
            time.sleep(interval)
        
        # Timeout - could implement a cancel and replace strategy here
        self.logger.warning(f"Timeout waiting for chunk {chunk_order_id} to fill")
        
        # For now, assume it's still working and return true
        return True
    
    def _on_order_filled(self, event):
        """
        Handle order filled event
        
        Args:
            event: OrderFilled event
        """
        order_id = event.order_id
        
        # Check if this is a chunk of an iceberg
        iceberg_id = self._get_iceberg_for_chunk(order_id)
        if not iceberg_id:
            return
        
        with self.lock:
            tracker = self.active_icebergs.get(iceberg_id)
            
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
            self.event_bus.emit(IcebergChunkFilled(
                order_id=iceberg_id,
                chunk_index=chunk_index,
                quantity=event.total_qty,
                price=event.avg_fill_price,
                remaining_quantity=tracker.get_remaining_quantity()
            ))
            
            # Check if all chunks are filled and iceberg is complete
            if tracker.is_complete():
                self._complete_iceberg(iceberg_id)
    
    def _on_order_partial_fill(self, event):
        """
        Handle order partial fill event
        
        Args:
            event: OrderPartialFill event
        """
        # For iceberg, we generally just wait for full fills of each chunk
        # But we could track partial fills for more granular updates
        order_id = event.order_id
        
        # Check if this is a chunk of an iceberg
        iceberg_id = self._get_iceberg_for_chunk(order_id)
        if not iceberg_id:
            return
        
        with self.lock:
            tracker = self.active_icebergs.get(iceberg_id)
            
            if not tracker:
                return
            
            # Update chunk partial fill
            tracker.update_chunk_partial_fill(
                chunk_order_id=order_id,
                filled_quantity=event.filled_qty,
                fill_price=event.fill_price,
                remaining_quantity=event.remaining_qty
            )
    
    def _on_order_rejected(self, event):
        """
        Handle order rejected event
        
        Args:
            event: OrderRejected event
        """
        order_id = event.order_id
        
        # Check if this is a chunk of an iceberg
        iceberg_id = self._get_iceberg_for_chunk(order_id)
        if not iceberg_id:
            return
        
        with self.lock:
            tracker = self.active_icebergs.get(iceberg_id)
            
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
    
    def _get_iceberg_for_chunk(self, chunk_order_id: str) -> Optional[str]:
        """
        Get iceberg ID for a chunk order ID
        
        Args:
            chunk_order_id: Chunk order ID
            
        Returns:
            Iceberg order ID or None
        """
        with self.lock:
            return self.chunk_map.get(chunk_order_id)
    
    def _fail_iceberg(self, order_id: str, reason: str, remaining_quantity: float):
        """
        Fail an iceberg order
        
        Args:
            order_id: Iceberg order ID
            reason: Failure reason
            remaining_quantity: Remaining quantity
        """
        with self.lock:
            tracker = self.active_icebergs.get(order_id)
            
            if not tracker:
                return
            
            # Calculate filled quantity
            total_quantity = tracker.order.total_quantity
            filled_quantity = total_quantity - remaining_quantity
            
            # Emit failure event
            self.event_bus.emit(IcebergOrderFailed(
                order_id=order_id,
                reason=reason,
                filled_quantity=filled_quantity,
                remaining_quantity=remaining_quantity
            ))
            
            # Clean up
            del self.active_icebergs[order_id]
            
            # Clean up chunk mappings
            chunk_order_ids = tracker.get_all_chunk_order_ids()
            for chunk_id in chunk_order_ids:
                if chunk_id in self.chunk_map:
                    del self.chunk_map[chunk_id]
    
    def _complete_iceberg(self, order_id: str):
        """
        Complete an iceberg order
        
        Args:
            order_id: Iceberg order ID
        """
        with self.lock:
            tracker = self.active_icebergs.get(order_id)
            
            if not tracker:
                return
            
            # Calculate weighted average price and execution time
            total_quantity = tracker.order.total_quantity
            total_filled = tracker.get_total_filled_quantity()
            avg_price = tracker.get_average_fill_price()
            
            start_time = tracker.start_time
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Emit completion event
            self.event_bus.emit(IcebergOrderCompleted(
                order_id=order_id,
                total_filled=total_filled,
                avg_price=avg_price,
                execution_time_ms=execution_time_ms,
                chunks_count=tracker.get_chunk_count()
            ))
            
            # Clean up
            del self.active_icebergs[order_id]
            
            # Clean up chunk mappings
            chunk_order_ids = tracker.get_all_chunk_order_ids()
            for chunk_id in chunk_order_ids:
                if chunk_id in self.chunk_map:
                    del self.chunk_map[chunk_id]
    
    def cancel_iceberg(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an iceberg order
        
        Args:
            order_id: Iceberg order ID
            
        Returns:
            Cancellation result
        """
        with self.lock:
            tracker = self.active_icebergs.get(order_id)
            
            if not tracker:
                return {"status": "error", "message": f"Iceberg order {order_id} not found"}
            
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
            
            # Fail the iceberg
            remaining = tracker.get_remaining_quantity()
            self._fail_iceberg(order_id, "Cancelled by user", remaining)
            
            return {
                "status": "ok",
                "cancelled_chunks": cancelled_chunks,
                "failed_chunks": failed_chunks,
                "message": "Iceberg order cancelled"
            }
    
    def get_iceberg_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an iceberg order
        
        Args:
            order_id: Iceberg order ID
            
        Returns:
            Iceberg status information
        """
        with self.lock:
            tracker = self.active_icebergs.get(order_id)
            
            if not tracker:
                return {"status": "unknown", "message": f"Iceberg order {order_id} not found"}
            
            return tracker.get_status()
    
    def shutdown_executor(self):
        """Shut down the executor"""
        self.shutdown = True
        
        # Wait for all threads to exit
        with self.lock:
            threads = list(self.executor_threads.values())
        
        for thread in threads:
            thread.join(timeout=1.0)


class IcebergOrderTracker:
    """
    Tracks the status of an iceberg order's chunks
    """
    
    def __init__(self, order: IcebergOrder, event_bus: EventBus):
        """
        Initialize the iceberg order tracker
        
        Args:
            order: Iceberg order
            event_bus: EventBus
        """
        self.order = order
        self.event_bus = event_bus
        self.start_time = datetime.now()
        
        # Track chunks
        self.chunks = []  # List of chunk info dictionaries
        
        # Track retry counts
        self.retry_counts = {}  # chunk_index -> retry_count
        
        # Track errors
        self.errors = []  # List of error messages
    
    def add_chunk(self, chunk_index: int, chunk_order_id: str, quantity: float):
        """
        Add a chunk
        
        Args:
            chunk_index: Chunk index
            chunk_order_id: Chunk order ID
            quantity: Chunk quantity
        """
        chunk_info = {
            "chunk_index": chunk_index,
            "order_id": chunk_order_id,
            "quantity": quantity,
            "filled_quantity": 0.0,
            "avg_price": 0.0,
            "status": "pending",
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
        Check if the iceberg is complete (all chunks filled)
        
        Returns:
            True if complete
        """
        # Check if total filled quantity matches total order quantity
        total_filled = self.get_total_filled_quantity()
        return abs(total_filled - self.order.total_quantity) < 0.0001
    
    def has_errors(self) -> bool:
        """
        Check if the iceberg has errors
        
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
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get iceberg status
        
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
                "status": chunk.get("status", "unknown")
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
            "chunks": chunks_info,
            "has_errors": self.has_errors(),
            "error_reason": self.get_error_reason() if self.has_errors() else None,
            "start_time": self.start_time.isoformat(),
            "elapsed_ms": int((datetime.now() - self.start_time).total_seconds() * 1000)
        }
