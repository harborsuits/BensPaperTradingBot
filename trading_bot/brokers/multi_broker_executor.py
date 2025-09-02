"""
Multi-Broker Trade Executor with Failover

This module provides a trade executor that can use multiple brokers
with automatic failover for enhanced trading system reliability.
"""

import logging
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.brokers.broker_interface import BrokerInterface, MarketSession
from trading_bot.brokers.tradier_client import TradierClient
try:
    from trading_bot.brokers.ig_adapter import IGAdapter
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiBrokerExecutor:
    """
    Trade executor that can use multiple brokers with automatic failover
    
    This executor enhances system reliability by supporting multiple brokers
    and automatically switching between them if one experiences issues.
    """
    
    def __init__(self, 
                brokers: List[BrokerInterface],
                primary_broker_index: int = 0,
                max_retries: int = 3,
                retry_delay: int = 2,
                auto_failover: bool = True,
                auto_refresh: bool = True,
                refresh_interval: int = 3600):
        """
        Initialize the multi-broker executor
        
        Args:
            brokers: List of broker interfaces
            primary_broker_index: Index of the primary broker in the list
            max_retries: Maximum number of retries per operation
            retry_delay: Delay between retries in seconds
            auto_failover: Whether to automatically failover to another broker
            auto_refresh: Whether to automatically refresh broker connections
            refresh_interval: Interval for refreshing connections in seconds
        """
        self.brokers = brokers
        self.primary_broker_index = primary_broker_index
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_failover = auto_failover
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        
        # Last refresh time
        self.last_refresh_time = datetime.now(timezone.utc)
        
        # Active broker index
        self.active_broker_index = primary_broker_index
        
        # Check if we have brokers
        if not brokers:
            raise ValueError("No brokers provided")
            
        # Initialize trade history
        self.trade_history = []
        self.active_trades = []
        
        # Verify broker connections
        self._verify_connections()
        
        logger.info(f"Multi-broker executor initialized with {len(brokers)} brokers")
    
    def _verify_connections(self):
        """Verify connections to all brokers"""
        for i, broker in enumerate(self.brokers):
            status = broker.status
            logger.info(f"Broker {i} ({broker.name}): {status}")
            
            if status != "connected" and self.auto_refresh:
                success = broker.refresh_connection()
                logger.info(f"Broker {i} ({broker.name}) refresh: {'Success' if success else 'Failed'}")
    
    @property
    def active_broker(self) -> BrokerInterface:
        """Get the currently active broker"""
        return self.brokers[self.active_broker_index]
    
    @property
    def primary_broker(self) -> BrokerInterface:
        """Get the primary broker"""
        return self.brokers[self.primary_broker_index]
    
    def _execute_with_retry(self, operation, *args, **kwargs) -> Any:
        """
        Execute an operation with retry and failover logic
        
        Args:
            operation: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation
        """
        # Check if it's time to refresh connections
        now = datetime.now(timezone.utc)
        if self.auto_refresh and (now - self.last_refresh_time).total_seconds() > self.refresh_interval:
            self._verify_connections()
            self.last_refresh_time = now
        
        # Try the operation with the active broker
        for attempt in range(self.max_retries):
            try:
                return operation(self.active_broker, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Error with broker {self.active_broker_index} ({self.active_broker.name}): {str(e)}")
                
                # Last attempt with current broker
                if attempt == self.max_retries - 1:
                    if self.auto_failover and len(self.brokers) > 1:
                        # Try to failover to another broker
                        old_index = self.active_broker_index
                        
                        # Temporarily exclude the current broker
                        available_indices = [i for i in range(len(self.brokers)) if i != old_index]
                        
                        if available_indices:
                            # Select a new broker
                            self.active_broker_index = random.choice(available_indices)
                            logger.info(f"Failing over from broker {old_index} to {self.active_broker_index}")
                            
                            # Try one more time with the new broker
                            try:
                                result = operation(self.active_broker, *args, **kwargs)
                                logger.info(f"Operation succeeded with failover broker {self.active_broker_index}")
                                return result
                            except Exception as failover_error:
                                logger.error(f"Failover also failed: {str(failover_error)}")
                                # Revert to primary broker for next operation
                                self.active_broker_index = self.primary_broker_index
                                raise
                    else:
                        # No failover, just raise the exception
                        raise
                
                # Wait before retry
                time.sleep(self.retry_delay)
    
    def refresh_account_data(self):
        """Refresh account data from all brokers"""
        for i, broker in enumerate(self.brokers):
            try:
                broker.get_account_balances()
                logger.debug(f"Refreshed account data for broker {i} ({broker.name})")
            except Exception as e:
                logger.warning(f"Error refreshing account data for broker {i} ({broker.name}): {str(e)}")
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        # Try with active broker first
        try:
            return self._execute_with_retry(lambda b, *a, **k: b.is_market_open())
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            
            # Fallback to MarketSession utility
            now = datetime.now(timezone.utc)
            return MarketSession.is_regular_hours(now)
    
    def wait_for_market_open(self, check_interval: int = 60) -> bool:
        """
        Wait for the market to open
        
        Args:
            check_interval: Interval between checks in seconds
            
        Returns:
            bool: True if market opened, False if interrupted
        """
        logger.info("Waiting for market to open...")
        
        while not self.is_market_open():
            # Get time until next market open
            try:
                next_open = self._execute_with_retry(lambda b, *a, **k: b.get_next_market_open())
                now = datetime.now(timezone.utc)
                
                wait_seconds = int((next_open - now).total_seconds())
                if wait_seconds > 0:
                    logger.info(f"Market opens in {wait_seconds//3600} hours, {(wait_seconds%3600)//60} minutes")
                    
                    # If more than 10 minutes away, wait longer between checks
                    sleep_time = min(wait_seconds, check_interval * 10) if wait_seconds > 600 else check_interval
                    time.sleep(sleep_time)
                else:
                    # We might be very close to market open, check more frequently
                    time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error getting next market open time: {str(e)}")
                time.sleep(check_interval)
                
        logger.info("Market is now open")
        return True
    
    def execute_trade(self, 
                     symbol: str, 
                     action: str, 
                     quantity: Optional[int] = None, 
                     dollar_amount: Optional[float] = None, 
                     order_type: str = "market", 
                     price: Optional[float] = None, 
                     stop_price: Optional[float] = None, 
                     take_profit_price: Optional[float] = None, 
                     stop_loss_pct: Optional[float] = None, 
                     take_profit_pct: Optional[float] = None,
                     time_in_force: str = "day",
                     wait_for_fill: bool = True,
                     check_market_hours: bool = True) -> Dict[str, Any]:
        """
        Execute a trade with the active broker
        
        Args:
            symbol: Symbol to trade
            action: 'buy' or 'sell'
            quantity: Number of shares (if None, dollar_amount must be provided)
            dollar_amount: Dollar amount to trade (if quantity is None)
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            take_profit_price: Take profit price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            time_in_force: Time in force ('day', 'gtc', 'opg')
            wait_for_fill: Whether to wait for the order to fill
            check_market_hours: Whether to check if market is open before trading
            
        Returns:
            Dict: Trade result
        """
        # Check market hours if required
        if check_market_hours and not self.is_market_open():
            return {
                'success': False,
                'error': 'Market is closed',
                'ticker': symbol,
                'action': action
            }
        
        # If not provided quantity but dollar amount, calculate quantity
        if quantity is None and dollar_amount is not None:
            # Get latest quote
            quote = self._execute_with_retry(lambda b, s: b.get_quote(s), symbol)
            
            # Calculate quantity
            if action.lower() == 'buy':
                price_to_use = quote.get('ask', 0) or quote.get('last', 0)
            else:
                price_to_use = quote.get('bid', 0) or quote.get('last', 0)
                
            if price_to_use <= 0:
                return {
                    'success': False,
                    'error': 'Unable to get price for quantity calculation',
                    'ticker': symbol,
                    'action': action
                }
                
            # Calculate shares
            quantity = int(dollar_amount / price_to_use)
            
            # Check minimum
            if quantity <= 0:
                return {
                    'success': False,
                    'error': f'Calculated quantity ({quantity}) is too small',
                    'ticker': symbol,
                    'action': action
                }
        
        # Execute the trade with retry and failover
        try:
            # Map our parameters to broker-specific parameters
            side = action.lower()
            
            # Make the actual broker call
            def execute_order(broker, *args, **kwargs):
                # Different brokers have different parameter names
                if hasattr(broker, 'place_equity_order'):
                    # Standard broker interface
                    return broker.place_equity_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type=order_type,
                        duration=time_in_force,
                        price=price,
                        stop_price=stop_price
                    )
                else:
                    # Legacy interface (for TradierClient)
                    return broker.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        type=order_type,
                        duration=time_in_force,
                        price=price,
                        stop=stop_price
                    )
            
            # Place the order
            order_result = self._execute_with_retry(execute_order)
            
            if not order_result or not order_result.get('id'):
                return {
                    'success': False,
                    'error': 'Failed to place order',
                    'ticker': symbol,
                    'action': action,
                    'raw_result': order_result
                }
                
            # Store the order ID
            order_id = order_result.get('id')
            
            # Wait for fill if requested
            if wait_for_fill and order_type.lower() == 'market':
                max_wait = 60  # Maximum 60 seconds to wait for fill
                wait_interval = 2  # Check every 2 seconds
                
                for _ in range(max_wait // wait_interval):
                    # Check order status
                    status = self._execute_with_retry(lambda b, o: b.get_order_status(o), order_id)
                    
                    # If filled, break
                    if status.get('status', '').lower() in ['filled', 'executed', 'complete', 'closed']:
                        order_result = status
                        break
                        
                    # If rejected, return error
                    if status.get('status', '').lower() in ['rejected', 'expired', 'canceled', 'cancelled']:
                        return {
                            'success': False,
                            'error': f"Order was {status.get('status', 'rejected')}",
                            'ticker': symbol,
                            'action': action,
                            'order_id': order_id
                        }
                        
                    # Wait and check again
                    time.sleep(wait_interval)
            
            # Create trade record
            trade_record = {
                'trade_id': order_id,
                'ticker': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'time_in_force': time_in_force,
                'status': order_result.get('status', 'pending'),
                'timestamp': datetime.now(timezone.utc),
                'broker': self.active_broker.name,
                'broker_index': self.active_broker_index,
                'order_id': order_id,
                'filled_price': order_result.get('price', price),
                'filled_quantity': order_result.get('size', quantity),
                'commission': order_result.get('commission', 0),
                'success': True
            }
            
            # Add to trade history
            self.trade_history.append(trade_record)
            
            # If it's a buy, add to active trades
            if action.lower() == 'buy':
                self.active_trades.append(trade_record)
            
            # Process stop loss and take profit if requested
            if action.lower() == 'buy' and (stop_loss_pct or take_profit_pct):
                filled_price = order_result.get('price', price) or price
                
                # Set stop loss order
                if stop_loss_pct and filled_price:
                    stop_price = filled_price * (1 - stop_loss_pct / 100)
                    try:
                        self._place_stop_loss(symbol, quantity, stop_price, order_id, time_in_force)
                    except Exception as e:
                        logger.error(f"Error setting stop loss: {str(e)}")
                
                # Set take profit order
                if take_profit_pct and filled_price:
                    target_price = filled_price * (1 + take_profit_pct / 100)
                    try:
                        self._place_take_profit(symbol, quantity, target_price, order_id, time_in_force)
                    except Exception as e:
                        logger.error(f"Error setting take profit: {str(e)}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ticker': symbol,
                'action': action
            }
    
    def _place_stop_loss(self, symbol, quantity, stop_price, parent_order_id, time_in_force):
        """Place a stop loss order"""
        # Execute with retry and failover
        return self._execute_with_retry(
            lambda b, *args, **kwargs: b.place_equity_order(
                symbol=symbol,
                side='sell',
                quantity=quantity,
                order_type='stop',
                price=None,
                stop_price=stop_price,
                duration=time_in_force
            )
        )
    
    def _place_take_profit(self, symbol, quantity, target_price, parent_order_id, time_in_force):
        """Place a take profit order"""
        # Execute with retry and failover
        return self._execute_with_retry(
            lambda b, *args, **kwargs: b.place_equity_order(
                symbol=symbol,
                side='sell',
                quantity=quantity,
                order_type='limit',
                price=target_price,
                stop_price=None,
                duration=time_in_force
            )
        )
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from active broker
        
        Returns:
            List[Dict]: Current positions
        """
        try:
            return self._execute_with_retry(lambda b, *args, **kwargs: b.get_positions())
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current orders from active broker
        
        Returns:
            List[Dict]: Current orders
        """
        try:
            return self._execute_with_retry(lambda b, *args, **kwargs: b.get_orders())
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get quote for a symbol
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Dict: Quote data
        """
        try:
            return self._execute_with_retry(lambda b, s: b.get_quote(s), symbol)
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {}
    
    def get_broker_account_info(self) -> Dict[str, Any]:
        """
        Get account information from the active broker
        
        Returns:
            Dict: Account information
        """
        try:
            return self._execute_with_retry(lambda b, *args, **kwargs: b.get_account_balances())
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get active trades
        
        Returns:
            List[Dict]: Active trades
        """
        # Update active trades based on positions
        self._update_active_trades()
        return self.active_trades
    
    def _update_active_trades(self):
        """Update active trades based on current positions"""
        if not self.active_trades:
            return
            
        # Get current positions
        positions = self.get_positions()
        
        # Create position lookup
        position_by_symbol = {p.get('symbol', ''): p for p in positions if 'symbol' in p}
        
        # Update active trades
        updated_active_trades = []
        
        for trade in self.active_trades:
            symbol = trade.get('ticker', '')
            
            # Check if we still have this position
            if symbol in position_by_symbol:
                # Update trade with position info
                position = position_by_symbol[symbol]
                
                trade['current_price'] = position.get('current_price', 0)
                trade['current_value'] = position.get('size', 0) * position.get('current_price', 0)
                trade['unrealized_pnl'] = position.get('profit_loss', 0)
                
                # Keep in active trades
                updated_active_trades.append(trade)
            else:
                # Position is closed
                trade['status'] = 'closed'
                
                # Try to get filled price from broker
                try:
                    order_id = trade.get('order_id', '')
                    if order_id:
                        order_status = self._execute_with_retry(lambda b, o: b.get_order_status(o), order_id)
                        if order_status:
                            trade['filled_price'] = order_status.get('price', trade.get('filled_price', 0))
                            trade['filled_quantity'] = order_status.get('size', trade.get('filled_quantity', 0))
                except Exception as e:
                    logger.warning(f"Error updating trade {trade.get('trade_id', '')}: {str(e)}")
        
        # Update active trades
        self.active_trades = updated_active_trades
    
    def can_trade_now(self, check_market_hours: bool = True) -> bool:
        """
        Check if trading is possible right now
        
        Args:
            check_market_hours: Whether to check market hours
            
        Returns:
            bool: True if trading is possible
        """
        # Check broker connection
        if self.active_broker.status != "connected":
            try:
                self.active_broker.refresh_connection()
            except Exception:
                return False
        
        # Check market hours if required
        if check_market_hours:
            return self.is_market_open()
            
        return True


# Factory function to create a multi-broker executor with available brokers
def create_multi_broker_executor(config=None) -> MultiBrokerExecutor:
    """
    Create a multi-broker executor with available brokers
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MultiBrokerExecutor: Initialized multi-broker executor
    """
    brokers = []
    config = config or {}
    
    # Tradier client support
    if config.get('tradier', {}).get('api_key'):
        try:
            tradier_client = TradierClient(
                api_key=config['tradier']['api_key'],
                account_id=config['tradier'].get('account_id'),
                paper=config['tradier'].get('paper', True)
            )
            brokers.append(tradier_client)
            logger.info("Added Tradier broker")
        except Exception as e:
            logger.error(f"Error initializing Tradier client: {str(e)}")
    
    # IG Markets support
    if IG_AVAILABLE and config.get('ig', {}).get('api_key'):
        try:
            ig_adapter = IGAdapter(
                api_key=config['ig']['api_key'],
                username=config['ig']['username'],
                password=config['ig']['password'],
                demo=config['ig'].get('demo', True)
            )
            brokers.append(ig_adapter)
            logger.info("Added IG Markets broker")
        except Exception as e:
            logger.error(f"Error initializing IG adapter: {str(e)}")
    
    # If no brokers available, raise error
    if not brokers:
        raise ValueError("No brokers available, check configuration")
    
    # Create multi-broker executor
    executor = MultiBrokerExecutor(
        brokers=brokers,
        primary_broker_index=0,
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay', 2),
        auto_failover=config.get('auto_failover', True),
        auto_refresh=config.get('auto_refresh', True),
        refresh_interval=config.get('refresh_interval', 3600)
    )
    
    return executor
