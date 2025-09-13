#!/usr/bin/env python3
"""
Tradier broker adapter implementation with order lifecycle tracking.

This adapter implements the BrokerInterface for the Tradier brokerage,
including order lifecycle events for accurate fill tracking and slippage measurement.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.brokers.lifecycle_tracking import OrderLifecycleTracker
from trading_bot.event_system.event_bus import EventBus

# Tradier implementations
from trading_bot.brokers.tradier.client import TradierClient, TradierOrderStatus
from trading_bot.brokers.tradier.margin_extension import TradierMarginExtension


class TradierAdapter(BrokerInterface, TradierMarginExtension):
    """Adapter for Tradier Brokerage API"""
    
    def __init__(
        self, 
        api_key: str, 
        account_id: str, 
        sandbox: bool = False,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the Tradier adapter
        
        Args:
            api_key: Tradier API key
            account_id: Tradier account ID
            sandbox: Whether to use sandbox environment
            event_bus: Event bus for emitting events
        """
        super().__init__(event_bus)
        self.broker_id = "tradier"
        self.api_key = api_key
        self.account_id = account_id
        self.sandbox = sandbox
        
        # Initialize Tradier client
        self.client = TradierClient(
            api_key=api_key,
            account_id=account_id,
            sandbox=sandbox
        )
        
        # Initialize order lifecycle tracker
        self.lifecycle_tracker = OrderLifecycleTracker(
            broker_id=self.broker_id,
            event_bus=event_bus
        )
        
        # Track orders being polled
        self.polling_orders = {}  # order_id -> dict with status info
        self.polling_thread = None
        self.polling_active = False
        self.polling_interval = 2.0  # seconds
        self.polling_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def is_market_open(self) -> bool:
        """Check if the market is open"""
        try:
            return self.client.is_market_open()
        except Exception as e:
            self.logger.error(f"Error checking if market is open: {str(e)}")
            return False
    
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours"""
        try:
            return self.client.get_market_hours()
        except Exception as e:
            self.logger.error(f"Error getting market hours: {str(e)}")
            return {}
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            return self.client.get_account_info()
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_account_balance(self) -> float:
        """Get account cash balance"""
        try:
            account_info = self.client.get_account_balances()
            return float(account_info.get('cash', 0.0))
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            return 0.0
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            return self.client.get_positions()
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders
        
        Args:
            status: Filter by status ('open', 'filled', 'canceled', etc.)
            
        Returns:
            List of order information
        """
        try:
            return self.client.get_orders(status=status)
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_order_by_id(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Order information
        """
        try:
            return self.client.get_order_by_id(order_id)
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {str(e)}")
            return {}
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for symbols
        
        Args:
            symbols: List of symbols to get quotes for
            
        Returns:
            Dict mapping symbols to quote information
        """
        try:
            return self.client.get_quotes(symbols)
        except Exception as e:
            self.logger.error(f"Error getting quotes for {symbols}: {str(e)}")
            return {}
    
    def place_equity_order(
        self, 
        symbol: str, 
        quantity: int, 
        side: str, 
        order_type: str, 
        time_in_force: str = 'day', 
        limit_price: float = None, 
        stop_price: float = None,
        expected_price: float = None
    ) -> Dict[str, Any]:
        """
        Place an equity order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Order duration ('day', 'gtc')
            limit_price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            expected_price: Expected execution price for slippage calculation
            
        Returns:
            Order information
        """
        try:
            # Map to Tradier's order parameters
            tradier_params = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type,
                'duration': time_in_force
            }
            
            if limit_price is not None:
                tradier_params['price'] = limit_price
            
            if stop_price is not None:
                tradier_params['stop'] = stop_price
            
            # Place the order
            order_result = self.client.place_equity_order(**tradier_params)
            
            if not order_result or 'id' not in order_result:
                # Order was rejected
                self.lifecycle_tracker.emit_order_rejected(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    reason="Order rejected by Tradier API",
                    limit_price=limit_price
                )
                return {'status': 'rejected', 'message': 'Order rejected by Tradier API'}
            
            order_id = str(order_result['id'])
            
            # Record expected price for slippage calculation
            if expected_price is not None:
                self.lifecycle_tracker.record_expected_price(order_id, expected_price)
            
            # Emit order acknowledged event
            self.lifecycle_tracker.emit_order_acknowledged(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price
            )
            
            # Start polling for fills if not already polling
            self._add_order_to_polling(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            return {
                'order_id': order_id,
                'status': 'acknowledged',
                **order_result
            }
            
        except Exception as e:
            self.logger.error(f"Error placing equity order for {symbol}: {str(e)}")
            
            # Emit order rejected event
            self.lifecycle_tracker.emit_order_rejected(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                reason=f"Exception: {str(e)}",
                limit_price=limit_price
            )
            
            return {'status': 'error', 'message': str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation result information
        """
        try:
            result = self.client.cancel_order(order_id)
            
            if result.get('status') == 'ok':
                # Emit order cancelled event
                self.lifecycle_tracker.emit_order_cancelled(
                    order_id=order_id,
                    reason="Cancelled by user request"
                )
                
                # Remove from polling
                self._remove_order_from_polling(order_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _add_order_to_polling(
        self, 
        order_id: str, 
        symbol: str, 
        side: str, 
        quantity: float
    ):
        """
        Add an order to the fill polling loop
        
        Args:
            order_id: Order ID
            symbol: Asset symbol
            side: Order side
            quantity: Order quantity
        """
        with self.polling_lock:
            self.polling_orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'last_checked': datetime.now(),
                'last_filled': 0.0,
                'is_acknowledged': True,
                'check_count': 0
            }
            
            # Start polling thread if not already running
            if not self.polling_active:
                self._start_polling_thread()
    
    def _remove_order_from_polling(self, order_id: str):
        """
        Remove an order from the fill polling loop
        
        Args:
            order_id: Order ID
        """
        with self.polling_lock:
            if order_id in self.polling_orders:
                del self.polling_orders[order_id]
    
    def _start_polling_thread(self):
        """Start the order status polling thread"""
        if self.polling_active:
            return
        
        self.polling_active = True
        self.polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True
        )
        self.polling_thread.start()
    
    def _stop_polling_thread(self):
        """Stop the order status polling thread"""
        self.polling_active = False
        if self.polling_thread:
            self.polling_thread.join(timeout=5.0)
            self.polling_thread = None
    
    def _polling_loop(self):
        """Background loop to poll for order status updates"""
        while self.polling_active:
            try:
                # Copy the current orders to avoid holding the lock
                with self.polling_lock:
                    orders_to_check = dict(self.polling_orders)
                
                # If no orders to check, sleep and continue
                if not orders_to_check:
                    time.sleep(self.polling_interval)
                    continue
                
                # Check each order
                for order_id, order_info in orders_to_check.items():
                    try:
                        # Get current status from API
                        order_status = self.client.get_order_status(order_id)
                        
                        # Check if order is acknowledged
                        is_acknowledged = order_info.get('is_acknowledged', False)
                        if not is_acknowledged and order_status.id:
                            self.lifecycle_tracker.emit_order_acknowledged(
                                order_id=order_id,
                                symbol=order_info['symbol'],
                                side=order_info['side'],
                                quantity=order_info['quantity'],
                                order_type=order_status.order_type
                            )
                            
                            # Update tracking
                            with self.polling_lock:
                                if order_id in self.polling_orders:
                                    self.polling_orders[order_id]['is_acknowledged'] = True
                        
                        # Check for partial fills
                        last_filled = order_info.get('last_filled', 0.0)
                        current_filled = order_status.filled_quantity
                        
                        if current_filled > last_filled:
                            # Get current fill price
                            fill_price = order_status.avg_fill_price
                            remaining_qty = order_status.quantity - current_filled
                            
                            # Emit partial fill event
                            self.lifecycle_tracker.emit_order_partial_fill(
                                order_id=order_id,
                                symbol=order_info['symbol'],
                                side=order_info['side'],
                                filled_qty=current_filled,
                                remaining_qty=remaining_qty,
                                fill_price=fill_price
                            )
                            
                            # Update tracking
                            with self.polling_lock:
                                if order_id in self.polling_orders:
                                    self.polling_orders[order_id]['last_filled'] = current_filled
                        
                        # Check for complete fills
                        if order_status.status == 'filled':
                            # Emit filled event
                            self.lifecycle_tracker.emit_order_filled(
                                order_id=order_id,
                                symbol=order_info['symbol'],
                                side=order_info['side'],
                                total_qty=order_status.quantity,
                                avg_fill_price=order_status.avg_fill_price
                            )
                            
                            # Remove from polling
                            self._remove_order_from_polling(order_id)
                        
                        # Check for cancelled orders
                        elif order_status.status == 'canceled':
                            # Emit cancelled event
                            self.lifecycle_tracker.emit_order_cancelled(
                                order_id=order_id,
                                reason="Order cancelled"
                            )
                            
                            # Remove from polling
                            self._remove_order_from_polling(order_id)
                        
                        # Check for rejected orders
                        elif order_status.status == 'rejected':
                            # Emit rejected event
                            self.lifecycle_tracker.emit_order_rejected(
                                order_id=order_id,
                                symbol=order_info['symbol'],
                                side=order_info['side'],
                                quantity=order_info['quantity'],
                                order_type=order_status.order_type,
                                reason=order_status.get('reason', 'Unknown rejection reason')
                            )
                            
                            # Remove from polling
                            self._remove_order_from_polling(order_id)
                        
                        # Update check count
                        with self.polling_lock:
                            if order_id in self.polling_orders:
                                self.polling_orders[order_id]['check_count'] += 1
                                self.polling_orders[order_id]['last_checked'] = datetime.now()
                        
                    except Exception as e:
                        self.logger.error(f"Error checking order {order_id}: {str(e)}")
                
                # Sleep before next check
                time.sleep(self.polling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in polling loop: {str(e)}")
                time.sleep(self.polling_interval)
                
        self.logger.info("Order polling thread stopped")
    
    def update_order_tracking(self):
        """Manually trigger an update of order tracking (for testing)"""
        try:
            # Copy the current orders to avoid holding the lock
            with self.polling_lock:
                orders_to_check = dict(self.polling_orders)
            
            # Check each order
            for order_id, order_info in orders_to_check.items():
                # Get current status from API
                order_status = self.client.get_order_status(order_id)
                
                # Process status (same as in _polling_loop)
                # This is intentionally duplicated for testability
                
                # Check for partial fills
                last_filled = order_info.get('last_filled', 0.0)
                current_filled = order_status.filled_quantity
                
                if current_filled > last_filled:
                    # Update tracking
                    with self.polling_lock:
                        if order_id in self.polling_orders:
                            self.polling_orders[order_id]['last_filled'] = current_filled
                
                # Update check count
                with self.polling_lock:
                    if order_id in self.polling_orders:
                        self.polling_orders[order_id]['check_count'] += 1
                        self.polling_orders[order_id]['last_checked'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating order tracking: {str(e)}")
            return False
    
    def close(self):
        """Close the adapter and clean up resources"""
        self._stop_polling_thread()
        self.client.close()
