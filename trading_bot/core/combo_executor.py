#!/usr/bin/env python3
"""
Combo Order Executor for BensBot

This module implements the logic for executing combo/multi-leg orders.
Handles routing to appropriate brokers, monitoring fills, and ensuring
all legs execute according to the order strategy.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import asyncio
from asyncio import Event as AsyncEvent
from uuid import uuid4

from trading_bot.core.advanced_orders import (
    ComboOrder, ComboLeg, 
    ComboOrderPlaced, ComboLegFilled, 
    ComboOrderCompleted, ComboOrderFailed
)
from trading_bot.event_system.event_bus import EventBus


class ComboOrderExecutor:
    """
    Executor for combo/multi-leg orders
    
    Handles the execution of multi-leg orders, ensuring proper routing
    and fill tracking across all legs.
    """
    
    def __init__(self, broker_manager, event_bus: EventBus):
        """
        Initialize the combo order executor
        
        Args:
            broker_manager: Multi-broker manager for order routing
            event_bus: EventBus for order events
        """
        self.broker_manager = broker_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Track active combo orders
        self.active_combos = {}  # combo_id -> ComboOrderTracker
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Subscribe to order events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to order fill events"""
        # Subscribe to order fill events to track leg fills
        from trading_bot.core.events import OrderFilled, OrderPartialFill, OrderRejected
        
        self.event_bus.on(OrderFilled, self._on_order_filled)
        self.event_bus.on(OrderPartialFill, self._on_order_partial_fill)
        self.event_bus.on(OrderRejected, self._on_order_rejected)
    
    def execute_combo(self, combo: ComboOrder) -> Dict[str, Any]:
        """
        Execute a combo order
        
        Args:
            combo: Combo order to execute
            
        Returns:
            Dictionary with execution result
        """
        self.logger.info(f"Executing combo order {combo.combo_id} with {len(combo.legs)} legs")
        
        # Create tracker for this combo
        tracker = ComboOrderTracker(combo, self.event_bus)
        
        with self.lock:
            self.active_combos[combo.combo_id] = tracker
        
        # Emit combo order placed event
        self.event_bus.emit(ComboOrderPlaced(
            combo_id=combo.combo_id,
            legs=combo.legs,
            order_strategy=combo.order_strategy,
            metadata=combo.user_metadata
        ))
        
        # Execute each leg
        for leg in combo.legs:
            try:
                # Determine broker for this leg
                broker_id = self._get_broker_for_leg(leg, combo)
                
                # Format leg order parameters
                order_params = self._format_leg_order(leg, combo)
                
                # Place the order with broker manager
                order_result = self.broker_manager.place_order(
                    broker_id=broker_id,
                    symbol=leg.symbol,
                    quantity=leg.quantity,
                    side=leg.side,
                    order_type=leg.order_type,
                    price=leg.price,
                    stop_price=leg.stop_price,
                    time_in_force=combo.time_in_force,
                    metadata={
                        "combo_id": combo.combo_id,
                        "leg_id": leg.leg_id,
                        "order_strategy": combo.order_strategy
                    }
                )
                
                # Update tracker with order ID
                tracker.add_leg_order(leg.leg_id, order_result.get("order_id"), broker_id)
                
                self.logger.debug(f"Placed leg order for {leg.symbol}, leg_id: {leg.leg_id}, " 
                                 f"order_id: {order_result.get('order_id')}")
                
            except Exception as e:
                self.logger.error(f"Error placing leg order for {leg.symbol}: {str(e)}")
                tracker.set_leg_error(leg.leg_id, str(e))
                
                # Check if combo should fail
                if self._should_fail_combo(tracker, combo):
                    self._fail_combo(combo.combo_id, f"Leg order failed: {str(e)}")
                    break
        
        # Return immediate result
        return {
            "combo_id": combo.combo_id,
            "status": "in_progress",
            "legs_count": len(combo.legs),
            "message": "Combo order execution started"
        }
    
    def _get_broker_for_leg(self, leg: ComboLeg, combo: ComboOrder) -> str:
        """
        Determine the appropriate broker for a leg
        
        Args:
            leg: Combo leg
            combo: Parent combo order
            
        Returns:
            Broker ID to use
        """
        # Check if leg has specific broker override
        if leg.broker_id:
            return leg.broker_id
        
        # Check routing instructions
        if leg.asset_class and leg.asset_class in combo.routing_instructions:
            return combo.routing_instructions[leg.asset_class]
        
        # Fall back to broker manager's routing logic
        return self.broker_manager.get_preferred_broker(
            symbol=leg.symbol,
            order_type=leg.order_type,
            side=leg.side
        )
    
    def _format_leg_order(self, leg: ComboLeg, combo: ComboOrder) -> Dict[str, Any]:
        """
        Format parameters for leg order
        
        Args:
            leg: Combo leg
            combo: Parent combo order
            
        Returns:
            Formatted order parameters
        """
        return {
            "symbol": leg.symbol,
            "quantity": leg.quantity,
            "side": leg.side,
            "order_type": leg.order_type,
            "price": leg.price,
            "stop_price": leg.stop_price,
            "time_in_force": combo.time_in_force,
            "metadata": {
                "combo_id": combo.combo_id,
                "leg_id": leg.leg_id,
                "strategy": combo.order_strategy
            }
        }
    
    def _on_order_filled(self, event):
        """
        Handle order filled event
        
        Args:
            event: OrderFilled event
        """
        # Check if this is a leg of a combo
        order_id = event.order_id
        metadata = getattr(event, 'metadata', {})
        combo_id = metadata.get('combo_id')
        
        if not combo_id:
            return
        
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                self.logger.warning(f"Received fill for unknown combo: {combo_id}, order: {order_id}")
                return
            
            # Update leg fill
            leg_id = tracker.get_leg_id_for_order(order_id)
            if not leg_id:
                self.logger.warning(f"Order {order_id} not associated with any leg in combo {combo_id}")
                return
            
            # Record fill
            tracker.set_leg_filled(
                leg_id=leg_id,
                fill_quantity=event.total_qty,
                fill_price=event.avg_fill_price
            )
            
            # Emit leg filled event
            leg_info = tracker.get_leg_info(leg_id)
            self.event_bus.emit(ComboLegFilled(
                combo_id=combo_id,
                leg_id=leg_id,
                symbol=event.symbol,
                quantity=event.total_qty,
                price=event.avg_fill_price,
                side=event.side
            ))
            
            # Check if all legs are filled
            if tracker.is_complete():
                self._complete_combo(combo_id)
    
    def _on_order_partial_fill(self, event):
        """
        Handle order partial fill event
        
        Args:
            event: OrderPartialFill event
        """
        # Similar to _on_order_filled but for partial fills
        # Track progress but don't complete the combo
        order_id = event.order_id
        metadata = getattr(event, 'metadata', {})
        combo_id = metadata.get('combo_id')
        
        if not combo_id:
            return
        
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                return
            
            # Update leg partial fill
            leg_id = tracker.get_leg_id_for_order(order_id)
            if not leg_id:
                return
            
            # Record partial fill
            tracker.update_leg_partial_fill(
                leg_id=leg_id,
                filled_quantity=event.filled_qty,
                fill_price=event.fill_price
            )
    
    def _on_order_rejected(self, event):
        """
        Handle order rejected event
        
        Args:
            event: OrderRejected event
        """
        order_id = event.order_id
        metadata = getattr(event, 'metadata', {})
        combo_id = metadata.get('combo_id')
        
        if not combo_id:
            return
        
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                return
            
            # Update leg rejection
            leg_id = tracker.get_leg_id_for_order(order_id)
            if not leg_id:
                return
            
            # Record rejection
            tracker.set_leg_rejected(
                leg_id=leg_id, 
                reason=event.reason
            )
            
            # Check if combo should fail
            if self._should_fail_combo(tracker, tracker.combo):
                self._fail_combo(combo_id, f"Leg order rejected: {event.reason}")
    
    def _should_fail_combo(self, tracker, combo: ComboOrder) -> bool:
        """
        Determine if a combo should fail based on current state
        
        Args:
            tracker: Combo order tracker
            combo: Original combo order
            
        Returns:
            True if combo should fail
        """
        # Check for critical leg failures
        for leg_id, status in tracker.leg_statuses.items():
            if status == "rejected" or status == "error":
                return True
        
        # Add more sophisticated logic here based on combo.order_strategy
        # For example, some strategies might allow partial executions
        
        return False
    
    def _fail_combo(self, combo_id: str, reason: str):
        """
        Fail a combo order
        
        Args:
            combo_id: Combo order ID
            reason: Failure reason
        """
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                return
            
            # Get failed and filled legs
            failed_legs = []
            partial_fills = {}
            
            for leg_id, status in tracker.leg_statuses.items():
                if status == "rejected" or status == "error":
                    failed_legs.append(leg_id)
                elif status == "partial" or status == "filled":
                    fill_info = tracker.leg_fills.get(leg_id, {})
                    partial_fills[leg_id] = fill_info
            
            # Emit failure event
            self.event_bus.emit(ComboOrderFailed(
                combo_id=combo_id,
                reason=reason,
                failed_legs=failed_legs,
                partial_fills=partial_fills
            ))
            
            # Clean up
            del self.active_combos[combo_id]
    
    def _complete_combo(self, combo_id: str):
        """
        Complete a combo order
        
        Args:
            combo_id: Combo order ID
        """
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                return
            
            # Calculate execution time
            start_time = tracker.start_time
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Get fill details
            fill_details = {
                leg_id: info 
                for leg_id, info in tracker.leg_fills.items()
            }
            
            # Emit completion event
            self.event_bus.emit(ComboOrderCompleted(
                combo_id=combo_id,
                fill_details=fill_details,
                execution_time_ms=execution_time_ms
            ))
            
            # Clean up
            del self.active_combos[combo_id]
    
    def cancel_combo(self, combo_id: str) -> Dict[str, Any]:
        """
        Cancel a combo order
        
        Args:
            combo_id: Combo order ID
            
        Returns:
            Cancellation result
        """
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                return {"status": "error", "message": f"Combo order {combo_id} not found"}
            
            # Cancel all unfilled leg orders
            cancelled_legs = []
            failed_legs = []
            
            for leg_id, status in tracker.leg_statuses.items():
                if status not in ["filled", "rejected", "error"]:
                    order_id = tracker.leg_orders.get(leg_id)
                    broker_id = tracker.leg_brokers.get(leg_id)
                    
                    if order_id and broker_id:
                        try:
                            # Cancel order
                            result = self.broker_manager.cancel_order(broker_id, order_id)
                            
                            if result.get("status") == "ok":
                                cancelled_legs.append(leg_id)
                            else:
                                failed_legs.append(leg_id)
                                
                        except Exception as e:
                            self.logger.error(f"Error cancelling leg {leg_id}: {str(e)}")
                            failed_legs.append(leg_id)
            
            # Fail the combo
            self._fail_combo(combo_id, "Cancelled by user")
            
            return {
                "status": "ok",
                "cancelled_legs": cancelled_legs,
                "failed_legs": failed_legs
            }
    
    def get_combo_status(self, combo_id: str) -> Dict[str, Any]:
        """
        Get status of a combo order
        
        Args:
            combo_id: Combo order ID
            
        Returns:
            Combo status information
        """
        with self.lock:
            tracker = self.active_combos.get(combo_id)
            
            if not tracker:
                return {"status": "unknown", "message": f"Combo order {combo_id} not found"}
            
            return tracker.get_status()


class ComboOrderTracker:
    """
    Tracks the status of a combo order's legs
    """
    
    def __init__(self, combo: ComboOrder, event_bus: EventBus):
        """
        Initialize the combo order tracker
        
        Args:
            combo: Combo order
            event_bus: EventBus
        """
        self.combo = combo
        self.event_bus = event_bus
        self.start_time = datetime.now()
        
        # Track leg orders and statuses
        self.leg_orders = {}  # leg_id -> order_id
        self.leg_brokers = {}  # leg_id -> broker_id
        self.leg_statuses = {}  # leg_id -> status (new, partial, filled, rejected, error)
        self.leg_fills = {}  # leg_id -> fill info
        self.leg_errors = {}  # leg_id -> error message
        
        # Initialize statuses
        for leg in combo.legs:
            self.leg_statuses[leg.leg_id] = "new"
            self.leg_fills[leg.leg_id] = {
                "filled_quantity": 0.0,
                "avg_price": 0.0,
                "status": "new"
            }
    
    def add_leg_order(self, leg_id: str, order_id: str, broker_id: str):
        """
        Add an order ID for a leg
        
        Args:
            leg_id: Leg ID
            order_id: Order ID
            broker_id: Broker ID
        """
        self.leg_orders[leg_id] = order_id
        self.leg_brokers[leg_id] = broker_id
        self.leg_statuses[leg_id] = "pending"
    
    def get_leg_id_for_order(self, order_id: str) -> Optional[str]:
        """
        Get leg ID for an order ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Leg ID or None if not found
        """
        for leg_id, leg_order_id in self.leg_orders.items():
            if leg_order_id == order_id:
                return leg_id
        return None
    
    def set_leg_filled(self, leg_id: str, fill_quantity: float, fill_price: float):
        """
        Set a leg as filled
        
        Args:
            leg_id: Leg ID
            fill_quantity: Fill quantity
            fill_price: Fill price
        """
        self.leg_statuses[leg_id] = "filled"
        self.leg_fills[leg_id] = {
            "filled_quantity": fill_quantity,
            "avg_price": fill_price,
            "status": "filled",
            "fill_time": datetime.now().isoformat()
        }
    
    def update_leg_partial_fill(self, leg_id: str, filled_quantity: float, fill_price: float):
        """
        Update a leg with partial fill information
        
        Args:
            leg_id: Leg ID
            filled_quantity: Filled quantity
            fill_price: Fill price
        """
        self.leg_statuses[leg_id] = "partial"
        
        # Update fill info
        fill_info = self.leg_fills.get(leg_id, {})
        
        # Calculate new average price
        old_qty = fill_info.get("filled_quantity", 0.0)
        old_price = fill_info.get("avg_price", 0.0)
        
        # If old_qty is 0, use fill_price directly
        if old_qty == 0:
            new_avg_price = fill_price
        else:
            # Calculate weighted average
            new_avg_price = ((old_qty * old_price) + (filled_quantity * fill_price)) / filled_quantity
        
        self.leg_fills[leg_id] = {
            "filled_quantity": filled_quantity,
            "avg_price": new_avg_price,
            "status": "partial",
            "last_update": datetime.now().isoformat()
        }
    
    def set_leg_rejected(self, leg_id: str, reason: str):
        """
        Set a leg as rejected
        
        Args:
            leg_id: Leg ID
            reason: Rejection reason
        """
        self.leg_statuses[leg_id] = "rejected"
        self.leg_errors[leg_id] = reason
    
    def set_leg_error(self, leg_id: str, error: str):
        """
        Set a leg error
        
        Args:
            leg_id: Leg ID
            error: Error message
        """
        self.leg_statuses[leg_id] = "error"
        self.leg_errors[leg_id] = error
    
    def get_leg_info(self, leg_id: str) -> Dict[str, Any]:
        """
        Get information for a leg
        
        Args:
            leg_id: Leg ID
            
        Returns:
            Leg information
        """
        return {
            "leg_id": leg_id,
            "order_id": self.leg_orders.get(leg_id),
            "broker_id": self.leg_brokers.get(leg_id),
            "status": self.leg_statuses.get(leg_id),
            "fill": self.leg_fills.get(leg_id, {}),
            "error": self.leg_errors.get(leg_id)
        }
    
    def is_complete(self) -> bool:
        """
        Check if the combo is complete (all legs filled)
        
        Returns:
            True if complete
        """
        return all(status == "filled" for status in self.leg_statuses.values())
    
    def has_errors(self) -> bool:
        """
        Check if the combo has errors
        
        Returns:
            True if errors exist
        """
        return any(status in ["rejected", "error"] for status in self.leg_statuses.values())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get combo status
        
        Returns:
            Status information
        """
        leg_info = {
            leg_id: {
                "status": self.leg_statuses.get(leg_id),
                "order_id": self.leg_orders.get(leg_id),
                "broker_id": self.leg_brokers.get(leg_id),
                "fill": self.leg_fills.get(leg_id, {}),
                "error": self.leg_errors.get(leg_id)
            }
            for leg_id in self.leg_statuses.keys()
        }
        
        # Determine overall status
        if self.is_complete():
            status = "filled"
        elif self.has_errors():
            status = "error"
        elif any(s == "partial" for s in self.leg_statuses.values()):
            status = "partial"
        else:
            status = "pending"
        
        return {
            "combo_id": self.combo.combo_id,
            "status": status,
            "legs": leg_info,
            "strategy": self.combo.order_strategy,
            "start_time": self.start_time.isoformat(),
            "elapsed_ms": int((datetime.now() - self.start_time).total_seconds() * 1000)
        }
