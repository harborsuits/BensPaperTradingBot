#!/usr/bin/env python3
"""
Integration tests for advanced order execution

Tests the functionality of combo, iceberg, TWAP, and VWAP 
order execution components.
"""

import unittest
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import threading
import time
import uuid

from trading_bot.core.advanced_orders import (
    ComboOrder, ComboLeg, IcebergOrder, TWAPOrder, VWAPOrder
)
from trading_bot.core.combo_executor import ComboOrderExecutor
from trading_bot.core.iceberg_executor import IcebergOrderExecutor
from trading_bot.core.algo_execution.twap_executor import TWAPExecutor
from trading_bot.core.algo_execution.vwap_executor import VWAPExecutor
from trading_bot.core.execution_manager import AdvancedOrderExecutionManager
from trading_bot.event_system.event_bus import EventBus
from trading_bot.core.events import OrderFilled


class TestAdvancedOrderExecution(unittest.TestCase):
    """Tests for the advanced order execution system"""

    def setUp(self):
        """Set up test fixtures for each test"""
        # Set up logging to a low level to avoid console spam during tests
        logging.basicConfig(level=logging.CRITICAL)
        
        # Create event bus
        self.event_bus = EventBus()
        
        # Create mock broker manager
        self.broker_manager = MagicMock()
        
        # Create mock market data provider
        self.market_data_provider = MagicMock()
        
        # Configure the broker_manager.place_order to return a valid order_id
        self.broker_manager.place_order.return_value = {
            "status": "ok",
            "order_id": "mock_order_123",
            "message": "Order placed successfully"
        }
        
        # Configure the broker_manager.cancel_order to return success
        self.broker_manager.cancel_order.return_value = {
            "status": "ok",
            "message": "Order cancelled successfully"
        }
        
        # Configure market data provider
        self.market_data_provider.get_market_data.return_value = {
            "symbol": "AAPL",
            "last_price": 150.0,
            "bid": 149.95,
            "ask": 150.05,
            "volume": 10000,
            "volatility": 0.15
        }
        
        self.market_data_provider.get_twap.return_value = 150.0
        self.market_data_provider.get_vwap.return_value = 150.0
        self.market_data_provider.get_volume_profile.return_value = [0.1, 0.1, 0.2, 0.3, 0.2, 0.1]
        
        # Create executors and manager
        self.execution_manager = AdvancedOrderExecutionManager(
            self.broker_manager, 
            self.event_bus,
            self.market_data_provider
        )
        
        # Store events for verification
        self.captured_events = []
        
        # Subscribe to all relevant events
        event_types = [
            "ComboOrderPlaced", "ComboLegFilled", "ComboOrderCompleted", "ComboOrderFailed",
            "IcebergOrderStarted", "IcebergChunkPlaced", "IcebergChunkFilled", "IcebergOrderCompleted",
            "AlgorithmicOrderStarted", "AlgorithmicChunkPlaced", "AlgorithmicChunkFilled", 
            "AlgorithmicOrderProgress", "AlgorithmicOrderCompleted"
        ]
        
        for event_type in event_types:
            self.event_bus.on_any(event_type, self._capture_event)
    
    def _capture_event(self, event):
        """Capture events for verification"""
        self.captured_events.append(event)
    
    def test_combo_order_execution(self):
        """Test basic combo order execution flow"""
        # Create a combo order
        combo = ComboOrder(
            combo_id=f"combo_{uuid.uuid4()}",
            order_strategy="vertical_spread",
            time_in_force="day",
            routing_instructions={"option": "tradier"},
            user_metadata={"strategy_id": "test_strategy"},
            legs=[
                ComboLeg(
                    leg_id="leg1",
                    symbol="AAPL_C_150_20230721",
                    quantity=1,
                    side="buy",
                    order_type="limit",
                    price=5.0,
                    asset_class="option"
                ),
                ComboLeg(
                    leg_id="leg2",
                    symbol="AAPL_C_160_20230721",
                    quantity=1,
                    side="sell",
                    order_type="limit",
                    price=2.0,
                    asset_class="option"
                )
            ]
        )
        
        # Execute the combo order
        result = self.execution_manager.execute_combo_order(combo)
        
        # Verify the result
        self.assertEqual(result["combo_id"], combo.combo_id)
        self.assertEqual(result["status"], "in_progress")
        
        # Verify that place_order was called for each leg
        self.assertEqual(self.broker_manager.place_order.call_count, 2)
        
        # Verify that events were emitted
        self.assertTrue(any(e.__class__.__name__ == "ComboOrderPlaced" for e in self.captured_events))
        
        # Simulate fills for each leg
        for i, leg in enumerate(combo.legs):
            # Get the captured order_id from the broker_manager calls
            args, kwargs = self.broker_manager.place_order.call_args_list[i]
            metadata = kwargs.get('metadata', {})
            
            # Create and emit a fill event
            fill_event = OrderFilled(
                order_id="mock_order_123",
                symbol=leg.symbol,
                side=leg.side,
                total_qty=leg.quantity,
                avg_fill_price=leg.price,
                metadata=metadata
            )
            self.event_bus.emit(fill_event)
        
        # Allow events to be processed
        time.sleep(0.1)
        
        # Verify that completion events were emitted
        self.assertTrue(any(e.__class__.__name__ == "ComboOrderCompleted" for e in self.captured_events))
    
    def test_iceberg_order_execution(self):
        """Test basic iceberg order execution flow"""
        # Create an iceberg order
        iceberg = IcebergOrder(
            order_id=f"iceberg_{uuid.uuid4()}",
            symbol="AAPL",
            side="buy",
            total_quantity=1000,
            display_quantity=200,
            order_type="limit",
            price=150.0,
            time_in_force="day",
            max_retries=3,
            child_delay_ms=100,
            user_metadata={"strategy_id": "test_strategy"}
        )
        
        # Execute the iceberg order
        result = self.execution_manager.execute_iceberg_order(iceberg)
        
        # Verify the result
        self.assertEqual(result["order_id"], iceberg.order_id)
        self.assertEqual(result["status"], "in_progress")
        
        # Verify that events were emitted
        self.assertTrue(any(e.__class__.__name__ == "IcebergOrderStarted" for e in self.captured_events))
        
        # Allow first chunk to be placed
        time.sleep(0.2)
        
        # Verify that place_order was called
        self.broker_manager.place_order.assert_called()
        
        # Simulate a fill for the first chunk
        args, kwargs = self.broker_manager.place_order.call_args
        metadata = kwargs.get('metadata', {})
        
        # Create and emit a fill event
        fill_event = OrderFilled(
            order_id="mock_order_123",
            symbol=iceberg.symbol,
            side=iceberg.side,
            total_qty=iceberg.display_quantity,
            avg_fill_price=iceberg.price,
            metadata=metadata
        )
        self.event_bus.emit(fill_event)
        
        # Allow events to be processed
        time.sleep(0.1)
        
        # Verify that chunk filled events were emitted
        self.assertTrue(any(e.__class__.__name__ == "IcebergChunkFilled" for e in self.captured_events))
    
    def test_twap_order_execution(self):
        """Test basic TWAP order execution flow"""
        # Create a TWAP order with short duration for testing
        twap = TWAPOrder(
            order_id=f"twap_{uuid.uuid4()}",
            symbol="AAPL",
            side="buy",
            total_quantity=1000,
            order_type="limit",
            price=150.0,
            time_in_force="day",
            duration_seconds=5,  # Short duration for test
            interval_seconds=1,
            num_chunks=5,
            max_retries=3,
            randomize_times=False,
            randomize_sizes=False,
            adapt_to_market=True,
            user_metadata={"strategy_id": "test_strategy"}
        )
        
        # Execute the TWAP order
        result = self.execution_manager.execute_twap_order(twap)
        
        # Verify the result
        self.assertEqual(result["order_id"], twap.order_id)
        self.assertEqual(result["status"], "in_progress")
        
        # Verify that events were emitted
        self.assertTrue(any(e.__class__.__name__ == "AlgorithmicOrderStarted" for e in self.captured_events))
        
        # Allow first chunk to be placed
        time.sleep(1.5)
        
        # Verify that place_order was called
        self.broker_manager.place_order.assert_called()
        
        # Simulate a fill for the first chunk
        args, kwargs = self.broker_manager.place_order.call_args
        metadata = kwargs.get('metadata', {})
        
        # Create and emit a fill event
        fill_event = OrderFilled(
            order_id="mock_order_123",
            symbol=twap.symbol,
            side=twap.side,
            total_qty=twap.total_quantity / twap.num_chunks,
            avg_fill_price=twap.price,
            metadata=metadata
        )
        self.event_bus.emit(fill_event)
        
        # Allow events to be processed
        time.sleep(0.1)
        
        # Verify that chunk filled events were emitted
        self.assertTrue(any(e.__class__.__name__ == "AlgorithmicChunkFilled" for e in self.captured_events))
        
        # Verify that progress events were emitted
        self.assertTrue(any(e.__class__.__name__ == "AlgorithmicOrderProgress" for e in self.captured_events))
    
    def test_vwap_order_execution(self):
        """Test basic VWAP order execution flow"""
        # Create a VWAP order with short duration for testing
        vwap = VWAPOrder(
            order_id=f"vwap_{uuid.uuid4()}",
            symbol="AAPL",
            side="buy",
            total_quantity=1000,
            order_type="limit",
            price=150.0,
            time_in_force="day",
            duration_seconds=5,  # Short duration for test
            num_chunks=5,
            max_retries=3,
            asset_class="equity",
            adapt_to_market=True,
            price_buffer_percent=0.1,
            user_metadata={"strategy_id": "test_strategy"}
        )
        
        # Execute the VWAP order
        result = self.execution_manager.execute_vwap_order(vwap)
        
        # Verify the result
        self.assertEqual(result["order_id"], vwap.order_id)
        self.assertEqual(result["status"], "in_progress")
        
        # Verify that events were emitted
        self.assertTrue(any(e.__class__.__name__ == "AlgorithmicOrderStarted" for e in self.captured_events))
        
        # Allow first chunk to be placed
        time.sleep(1.5)
        
        # Verify that place_order was called
        self.broker_manager.place_order.assert_called()
        
        # Simulate a fill for the first chunk
        args, kwargs = self.broker_manager.place_order.call_args
        metadata = kwargs.get('metadata', {})
        
        # Create and emit a fill event
        fill_event = OrderFilled(
            order_id="mock_order_123",
            symbol=vwap.symbol,
            side=vwap.side,
            total_qty=vwap.total_quantity / vwap.num_chunks,
            avg_fill_price=vwap.price,
            metadata=metadata
        )
        self.event_bus.emit(fill_event)
        
        # Allow events to be processed
        time.sleep(0.1)
        
        # Verify that chunk filled events were emitted
        self.assertTrue(any(e.__class__.__name__ == "AlgorithmicChunkFilled" for e in self.captured_events))
        
        # Verify that progress events were emitted
        self.assertTrue(any(e.__class__.__name__ == "AlgorithmicOrderProgress" for e in self.captured_events))
    
    def test_cancel_order(self):
        """Test cancelling an advanced order"""
        # Create and execute a combo order
        combo = ComboOrder(
            combo_id=f"combo_{uuid.uuid4()}",
            order_strategy="vertical_spread",
            time_in_force="day",
            routing_instructions={"option": "tradier"},
            user_metadata={"strategy_id": "test_strategy"},
            legs=[
                ComboLeg(
                    leg_id="leg1",
                    symbol="AAPL_C_150_20230721",
                    quantity=1,
                    side="buy",
                    order_type="limit",
                    price=5.0,
                    asset_class="option"
                ),
                ComboLeg(
                    leg_id="leg2",
                    symbol="AAPL_C_160_20230721",
                    quantity=1,
                    side="sell",
                    order_type="limit",
                    price=2.0,
                    asset_class="option"
                )
            ]
        )
        
        # Execute the combo order
        result = self.execution_manager.execute_combo_order(combo)
        
        # Cancel the order
        cancel_result = self.execution_manager.cancel_order(combo.combo_id)
        
        # Verify the result
        self.assertEqual(cancel_result["status"], "ok")
        
        # Verify that cancel_order was called
        self.broker_manager.cancel_order.assert_called()
    
    def test_get_order_status(self):
        """Test getting order status"""
        # Create and execute a combo order
        combo = ComboOrder(
            combo_id=f"combo_{uuid.uuid4()}",
            order_strategy="vertical_spread",
            time_in_force="day",
            routing_instructions={"option": "tradier"},
            user_metadata={"strategy_id": "test_strategy"},
            legs=[
                ComboLeg(
                    leg_id="leg1",
                    symbol="AAPL_C_150_20230721",
                    quantity=1,
                    side="buy",
                    order_type="limit",
                    price=5.0,
                    asset_class="option"
                )
            ]
        )
        
        # Execute the combo order
        result = self.execution_manager.execute_combo_order(combo)
        
        # Get order status
        status = self.execution_manager.get_order_status(combo.combo_id)
        
        # Verify the status has the combo_id
        self.assertEqual(status["combo_id"], combo.combo_id)
    
    def test_get_all_active_orders(self):
        """Test getting all active orders"""
        # Create and execute multiple orders
        combo = ComboOrder(
            combo_id=f"combo_{uuid.uuid4()}",
            order_strategy="vertical_spread",
            time_in_force="day",
            routing_instructions={"option": "tradier"},
            legs=[
                ComboLeg(
                    leg_id="leg1",
                    symbol="AAPL_C_150_20230721",
                    quantity=1,
                    side="buy",
                    order_type="limit",
                    price=5.0,
                    asset_class="option"
                )
            ]
        )
        
        iceberg = IcebergOrder(
            order_id=f"iceberg_{uuid.uuid4()}",
            symbol="AAPL",
            side="buy",
            total_quantity=1000,
            display_quantity=200,
            order_type="limit",
            price=150.0,
            time_in_force="day"
        )
        
        # Execute the orders
        self.execution_manager.execute_combo_order(combo)
        self.execution_manager.execute_iceberg_order(iceberg)
        
        # Get all active orders
        active_orders = self.execution_manager.get_all_active_orders()
        
        # Verify we have 2 active orders
        self.assertEqual(len(active_orders), 2)
        
        # Verify the order types
        order_types = [order.get("order_type") for order in active_orders]
        self.assertIn("combo", order_types)
        self.assertIn("iceberg", order_types)
    
    def tearDown(self):
        """Clean up resources"""
        # Shutdown execution manager
        self.execution_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
