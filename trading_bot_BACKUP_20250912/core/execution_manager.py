#!/usr/bin/env python3
"""
Advanced Order Execution Manager for BensBot

This module implements the main execution manager for advanced order types,
serving as the central entry point for executing combo orders, iceberg orders,
and algorithmic orders (TWAP/VWAP).
"""

import logging
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

from trading_bot.core.advanced_orders import (
    ComboOrder, IcebergOrder, TWAPOrder, VWAPOrder, 
    AlgorithmicOrder
)
from trading_bot.core.combo_executor import ComboOrderExecutor
from trading_bot.core.iceberg_executor import IcebergOrderExecutor
from trading_bot.core.algo_execution.twap_executor import TWAPExecutor
from trading_bot.core.algo_execution.vwap_executor import VWAPExecutor
from trading_bot.event_system.event_bus import EventBus


class AdvancedOrderExecutionManager:
    """
    Advanced Order Execution Manager
    
    Serves as the central entry point for executing all types of advanced orders,
    including combo orders, iceberg orders, and algorithmic orders (TWAP/VWAP).
    """
    
    def __init__(self, broker_manager, event_bus: EventBus, market_data_provider=None):
        """
        Initialize the advanced order execution manager
        
        Args:
            broker_manager: Multi-broker manager for order routing
            event_bus: EventBus for order events
            market_data_provider: Provider for market data (optional)
        """
        self.broker_manager = broker_manager
        self.event_bus = event_bus
        self.market_data_provider = market_data_provider
        self.logger = logging.getLogger(__name__)
        
        # Initialize executors
        self.combo_executor = ComboOrderExecutor(broker_manager, event_bus)
        self.iceberg_executor = IcebergOrderExecutor(broker_manager, event_bus)
        self.twap_executor = TWAPExecutor(broker_manager, event_bus, market_data_provider)
        self.vwap_executor = VWAPExecutor(broker_manager, event_bus, market_data_provider)
        
        # Track active orders
        self.active_orders = {}  # order_id -> order details
    
    def execute_combo_order(self, order: ComboOrder) -> Dict[str, Any]:
        """
        Execute a combo order
        
        Args:
            order: Combo order to execute
            
        Returns:
            Dictionary with execution result
        """
        self.logger.info(f"Executing combo order {order.combo_id} with {len(order.legs)} legs")
        
        # Track order
        self.active_orders[order.combo_id] = {
            "order_id": order.combo_id,
            "type": "combo",
            "status": "in_progress"
        }
        
        # Execute with combo executor
        result = self.combo_executor.execute_combo(order)
        
        return result
    
    def execute_iceberg_order(self, order: IcebergOrder) -> Dict[str, Any]:
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
        
        # Track order
        self.active_orders[order.order_id] = {
            "order_id": order.order_id,
            "type": "iceberg",
            "status": "in_progress"
        }
        
        # Execute with iceberg executor
        result = self.iceberg_executor.execute_iceberg(order)
        
        return result
    
    def execute_twap_order(self, order: TWAPOrder) -> Dict[str, Any]:
        """
        Execute a TWAP order
        
        Args:
            order: TWAP order to execute
            
        Returns:
            Dictionary with execution result
        """
        self.logger.info(
            f"Executing TWAP order {order.order_id} for {order.symbol}, "
            f"quantity: {order.total_quantity}, duration: {order.duration_seconds}s"
        )
        
        # Track order
        self.active_orders[order.order_id] = {
            "order_id": order.order_id,
            "type": "twap",
            "status": "in_progress"
        }
        
        # Execute with TWAP executor
        result = self.twap_executor.execute_order(order)
        
        return result
    
    def execute_vwap_order(self, order: VWAPOrder) -> Dict[str, Any]:
        """
        Execute a VWAP order
        
        Args:
            order: VWAP order to execute
            
        Returns:
            Dictionary with execution result
        """
        self.logger.info(
            f"Executing VWAP order {order.order_id} for {order.symbol}, "
            f"quantity: {order.total_quantity}, duration: {order.duration_seconds}s"
        )
        
        # Track order
        self.active_orders[order.order_id] = {
            "order_id": order.order_id,
            "type": "vwap",
            "status": "in_progress"
        }
        
        # Execute with VWAP executor
        result = self.twap_executor.execute_order(order)
        
        return result
    
    def execute_algorithmic_order(self, order: AlgorithmicOrder) -> Dict[str, Any]:
        """
        Execute an algorithmic order of any type
        
        Args:
            order: Algorithmic order to execute
            
        Returns:
            Dictionary with execution result
        """
        algorithm_type = order.algorithm_type.lower()
        
        if algorithm_type == "twap":
            return self.execute_twap_order(order)
        elif algorithm_type == "vwap":
            return self.execute_vwap_order(order)
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
    
    def execute_order(self, order: Union[ComboOrder, IcebergOrder, AlgorithmicOrder]) -> Dict[str, Any]:
        """
        Execute any type of advanced order
        
        Args:
            order: Order to execute
            
        Returns:
            Dictionary with execution result
        """
        # Determine order type and route to appropriate executor
        if isinstance(order, ComboOrder):
            return self.execute_combo_order(order)
        elif isinstance(order, IcebergOrder):
            return self.execute_iceberg_order(order)
        elif isinstance(order, TWAPOrder):
            return self.execute_twap_order(order)
        elif isinstance(order, VWAPOrder):
            return self.execute_vwap_order(order)
        elif isinstance(order, AlgorithmicOrder):
            return self.execute_algorithmic_order(order)
        else:
            raise ValueError(f"Unsupported order type: {type(order).__name__}")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an advanced order
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation result
        """
        # Look up order type
        order_info = self.active_orders.get(order_id)
        
        if not order_info:
            return {"status": "error", "message": f"Order {order_id} not found"}
        
        order_type = order_info.get("type", "unknown")
        
        # Route to appropriate executor
        if order_type == "combo":
            result = self.combo_executor.cancel_combo(order_id)
        elif order_type == "iceberg":
            result = self.iceberg_executor.cancel_iceberg(order_id)
        elif order_type in ["twap", "vwap"]:
            # Both algorithmic order types use the same cancellation method
            result = self.twap_executor.cancel_order(order_id)
        else:
            return {"status": "error", "message": f"Unknown order type: {order_type}"}
        
        # Update tracking
        if result.get("status") == "ok":
            if order_id in self.active_orders:
                self.active_orders[order_id]["status"] = "cancelled"
        
        return result
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an advanced order
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status information
        """
        # Look up order type
        order_info = self.active_orders.get(order_id)
        
        if not order_info:
            return {"status": "unknown", "message": f"Order {order_id} not found"}
        
        order_type = order_info.get("type", "unknown")
        
        # Route to appropriate executor
        if order_type == "combo":
            return self.combo_executor.get_combo_status(order_id)
        elif order_type == "iceberg":
            return self.iceberg_executor.get_iceberg_status(order_id)
        elif order_type == "twap":
            return self.twap_executor.get_order_status(order_id)
        elif order_type == "vwap":
            return self.vwap_executor.get_order_status(order_id)
        else:
            return {"status": "unknown", "message": f"Unknown order type: {order_type}"}
    
    def get_all_active_orders(self) -> List[Dict[str, Any]]:
        """
        Get all active advanced orders
        
        Returns:
            List of order information
        """
        results = []
        
        for order_id, order_info in self.active_orders.items():
            # Get detailed status
            status = self.get_order_status(order_id)
            
            # Include order type
            status["order_type"] = order_info.get("type", "unknown")
            
            results.append(status)
        
        return results
    
    def shutdown(self):
        """Shutdown all executors"""
        self.iceberg_executor.shutdown_executor()
        self.twap_executor.shutdown_executor()
        self.vwap_executor.shutdown_executor()
        
        self.logger.info("Advanced order execution manager shut down")
