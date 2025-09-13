from typing import Any, Dict, List, Optional
from datetime import datetime

class BrokerageClientAdapter(OrderExecutor):
    """
    Adapter class that bridges between the new BrokerageClient interface
    and the existing OrderExecutor interface.
    
    This allows using the new broker API integration with existing code
    that expects an OrderExecutor.
    """
    
    def __init__(self, broker_client, asset_class: str = "equity", **kwargs):
        """
        Initialize the adapter with a broker client.
        
        Args:
            broker_client: The BrokerageClient instance to adapt
            asset_class: The asset class this adapter is handling
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.broker_client = broker_client
        self.asset_class = asset_class
        
        logger.info(f"Initialized BrokerageClientAdapter for {asset_class}")
    
    def place_order(self, symbol: str, quantity: float, side: str, order_type: str, 
                   price: Optional[float] = None, time_in_force: str = 'day',
                   stop_price: Optional[float] = None, 
                   take_profit_price: Optional[float] = None,
                   strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order using the broker client.
        
        Args:
            symbol: Security symbol
            quantity: Order quantity
            side: Order side ('buy', 'sell', 'sell_short', 'buy_to_cover')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            price: Limit price (required for limit and stop_limit orders)
            time_in_force: Time in force for the order ('day', 'gtc', 'ioc', 'fok')
            stop_price: Stop price (required for stop and stop_limit orders)
            take_profit_price: Take profit price for OCO orders
            strategy: Optional strategy name for tracking
            
        Returns:
            Order details dictionary
        """
        try:
            # Map parameters to the broker client interface
            from trading_bot.brokers.brokerage_client import OrderType, OrderSide, TimeInForce
            
            # Map order type
            if order_type.lower() == 'market':
                broker_order_type = OrderType.MARKET
            elif order_type.lower() == 'limit':
                broker_order_type = OrderType.LIMIT
            elif order_type.lower() == 'stop':
                broker_order_type = OrderType.STOP
            elif order_type.lower() == 'stop_limit':
                broker_order_type = OrderType.STOP_LIMIT
            else:
                logger.warning(f"Unknown order type {order_type}, using MARKET")
                broker_order_type = OrderType.MARKET
            
            # Map order side
            if side.lower() == 'buy':
                broker_side = OrderSide.BUY
            elif side.lower() == 'sell':
                broker_side = OrderSide.SELL
            elif side.lower() == 'sell_short':
                broker_side = OrderSide.SELL_SHORT
            elif side.lower() == 'buy_to_cover':
                broker_side = OrderSide.BUY_TO_COVER
            else:
                logger.warning(f"Unknown order side {side}, using BUY")
                broker_side = OrderSide.BUY
            
            # Map time in force
            if time_in_force.lower() == 'day':
                broker_tif = TimeInForce.DAY
            elif time_in_force.lower() == 'gtc':
                broker_tif = TimeInForce.GTC
            elif time_in_force.lower() == 'ioc':
                broker_tif = TimeInForce.IOC
            elif time_in_force.lower() == 'fok':
                broker_tif = TimeInForce.FOK
            else:
                logger.warning(f"Unknown time in force {time_in_force}, using DAY")
                broker_tif = TimeInForce.DAY
            
            # Place the order using the broker client
            order_result = self.broker_client.place_order(
                symbol=symbol,
                side=broker_side,
                quantity=quantity,
                order_type=broker_order_type,
                time_in_force=broker_tif,
                limit_price=price,
                stop_price=stop_price,
                client_order_id=f"strategy_{strategy}" if strategy else None
            )
            
            # Add strategy information to the result
            if isinstance(order_result, dict):
                order_result['strategy'] = strategy
            
            # Track the order
            self.orders.append(order_result)
            
            logger.info(f"Order placed: {symbol} {side} {quantity} at {price if price else 'market price'}")
            return order_result
            
        except Exception as e:
            error_msg = f"Error placing order for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type,
                'price': price,
                'error': str(e)
            }
            
            self.errors.append(error_details)
            return {'error': error_msg, 'details': error_details}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order using the broker client.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Status dictionary
        """
        try:
            return self.broker_client.cancel_order(order_id)
        except Exception as e:
            error_msg = f"Error cancelling order {order_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status using the broker client.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            return self.broker_client.get_order(order_id)
        except Exception as e:
            error_msg = f"Error retrieving order status for {order_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg,
                'order_id': order_id
            }
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get open orders using the broker client.
        
        Returns:
            List of open order dictionaries
        """
        try:
            return self.broker_client.get_orders(status='open')
        except Exception as e:
            error_msg = f"Error retrieving open orders: {str(e)}"
            logger.error(error_msg)
            
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get positions using the broker client.
        
        Returns:
            List of position dictionaries
        """
        try:
            return self.broker_client.get_positions()
        except Exception as e:
            error_msg = f"Error retrieving positions: {str(e)}"
            logger.error(error_msg)
            
            return []
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol using the broker client.
        
        Args:
            symbol: Security symbol
            
        Returns:
            Position dictionary
        """
        try:
            positions = self.broker_client.get_positions()
            
            for position in positions:
                if position.get('symbol') == symbol:
                    return position
            
            return {
                'symbol': symbol,
                'quantity': 0,
                'side': 'none',
                'avg_entry_price': 0,
                'current_price': 0,
                'market_value': 0,
                'cost_basis': 0,
                'unrealized_pl': 0,
                'unrealized_plpc': 0
            }
        except Exception as e:
            error_msg = f"Error retrieving position for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg,
                'symbol': symbol
            }
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance using the broker client.
        
        Returns:
            Account balance dictionary
        """
        try:
            return self.broker_client.get_account_info()
        except Exception as e:
            error_msg = f"Error retrieving account balance: {str(e)}"
            logger.error(error_msg)
            
            return {
                'error': error_msg
            } 