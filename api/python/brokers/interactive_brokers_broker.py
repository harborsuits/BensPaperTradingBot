#!/usr/bin/env python3
"""
Interactive Brokers Adapter for BensBot

Implements the BrokerInterface for Interactive Brokers using the IB API.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set

# Import the Interactive Brokers API (ibapi)
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.execution import Execution
    from ibapi.commission_report import CommissionReport
    from ibapi.ticktype import TickTypeEnum
    HAS_IBAPI = True
except ImportError:
    HAS_IBAPI = False

# Import BrokerInterface
from trading_bot.brokers.broker_interface import BrokerInterface

# Configure logging
logger = logging.getLogger(__name__)

class IBConnectionError(Exception):
    """Exception raised for Interactive Brokers connection errors"""
    pass

class IBError(Exception):
    """Exception raised for Interactive Brokers API errors"""
    pass

class IBClientWrapper(EWrapper, EClient):
    """
    Wrapper class for the IB API client
    Handles IB API callbacks and request/response management
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Store request/response data
        self.next_req_id = 1
        self.req_lock = threading.Lock()
        self.request_map = {}
        self.response_map = {}
        self.account_positions = {}
        self.account_values = {}
        self.order_status = {}
        self.executions = {}
        self.market_data = {}
        
        # Connection status
        self.is_connected = False
        self.connect_event = threading.Event()
        
        # Error handling
        self.errors = []
        self.last_error_time = 0
    
    def connectAck(self):
        """Called when connection is established"""
        super().connectAck()
        self.is_connected = True
        self.connect_event.set()
        logger.info("Connected to Interactive Brokers")
    
    def connectionClosed(self):
        """Called when connection is closed"""
        super().connectionClosed()
        self.is_connected = False
        self.connect_event.clear()
        logger.info("Disconnected from Interactive Brokers")
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Called when IB returns an error"""
        self.errors.append((reqId, errorCode, errorString))
        self.last_error_time = time.time()
        
        # Log the error
        if reqId > 0:
            logger.error(f"IB API Error {errorCode} for request {reqId}: {errorString}")
        else:
            logger.error(f"IB API Error {errorCode}: {errorString}")
        
        # Store error in response map
        if reqId in self.request_map:
            req_type = self.request_map[reqId]
            if reqId not in self.response_map:
                self.response_map[reqId] = {"type": req_type, "data": [], "error": (errorCode, errorString)}
            else:
                self.response_map[reqId]["error"] = (errorCode, errorString)
    
    def nextValidId(self, orderId: int):
        """Called when IB sends the next valid order ID"""
        super().nextValidId(orderId)
        self.next_req_id = orderId
    
    def getNextReqId(self) -> int:
        """Get the next request ID (thread-safe)"""
        with self.req_lock:
            req_id = self.next_req_id
            self.next_req_id += 1
            return req_id
    
    def makeRequest(self, req_type: str, req_func, *args, **kwargs) -> Tuple[int, Any]:
        """
        Make a request to IB API with timeout and error handling
        
        Args:
            req_type: Type of request
            req_func: Function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Tuple of (request ID, response data)
            
        Raises:
            IBError: If request fails or times out
        """
        # Get the next request ID
        req_id = self.getNextReqId()
        
        # Store request type
        self.request_map[req_id] = req_type
        self.response_map[req_id] = {"type": req_type, "data": [], "error": None, "completed": False}
        
        # Make the request
        try:
            # Call the request function with the request ID and other arguments
            req_func(req_id, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error making IB request: {str(e)}")
            raise IBError(f"Error making IB request: {str(e)}")
        
        # Wait for response with timeout
        timeout = 30  # seconds
        start_time = time.time()
        
        while not self.response_map[req_id]["completed"] and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        # Check for timeout
        if not self.response_map[req_id]["completed"]:
            logger.error(f"Request {req_id} ({req_type}) timed out")
            raise IBError(f"Request {req_id} ({req_type}) timed out")
        
        # Check for error
        if self.response_map[req_id]["error"]:
            error_code, error_msg = self.response_map[req_id]["error"]
            logger.error(f"Request {req_id} ({req_type}) failed: {error_code} - {error_msg}")
            raise IBError(f"Request {req_id} ({req_type}) failed: {error_code} - {error_msg}")
        
        # Return response data
        return req_id, self.response_map[req_id]["data"]
    
    # Account and Position callbacks
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Called when position information is received"""
        super().position(account, contract, position, avgCost)
        
        # Store position
        symbol = contract.symbol
        if account not in self.account_positions:
            self.account_positions[account] = {}
        
        self.account_positions[account][symbol] = {
            "symbol": symbol,
            "position": position,
            "avg_cost": avgCost,
            "contract": contract
        }
    
    def positionEnd(self):
        """Called when all positions have been received"""
        super().positionEnd()
        
        # Mark position requests as completed
        for req_id, req_info in self.request_map.items():
            if req_info == "positions":
                self.response_map[req_id]["completed"] = True
                self.response_map[req_id]["data"] = self.account_positions
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Called when account summary information is received"""
        super().accountSummary(reqId, account, tag, value, currency)
        
        # Store account value
        if account not in self.account_values:
            self.account_values[account] = {}
        
        self.account_values[account][tag] = {
            "value": value,
            "currency": currency
        }
    
    def accountSummaryEnd(self, reqId: int):
        """Called when all account summary data has been received"""
        super().accountSummaryEnd(reqId)
        
        # Mark account request as completed
        if reqId in self.request_map and self.request_map[reqId] == "account_summary":
            self.response_map[reqId]["completed"] = True
            self.response_map[reqId]["data"] = self.account_values
    
    # Order callbacks
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float, avgFillPrice: float,
                   permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """Called when order status information is received"""
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        
        # Store order status
        self.order_status[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avgFillPrice,
            "last_fill_price": lastFillPrice,
            "why_held": whyHeld
        }
        
        # Mark order status request as completed if order is done
        if status in ["Filled", "Cancelled", "Inactive"]:
            for req_id, req_info in self.request_map.items():
                if req_info == f"order_status_{orderId}":
                    self.response_map[req_id]["completed"] = True
                    self.response_map[req_id]["data"] = self.order_status[orderId]
    
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState):
        """Called when open order information is received"""
        super().openOrder(orderId, contract, order, orderState)
        
        # Process open order information
        if "open_orders" not in self.response_map:
            self.response_map["open_orders"] = {"type": "open_orders", "data": {}, "error": None, "completed": False}
        
        self.response_map["open_orders"]["data"][orderId] = {
            "contract": contract,
            "order": order,
            "order_state": orderState
        }
    
    def openOrderEnd(self):
        """Called when all open orders have been received"""
        super().openOrderEnd()
        
        # Mark open orders request as completed
        for req_id, req_info in self.request_map.items():
            if req_info == "open_orders":
                self.response_map[req_id]["completed"] = True
                self.response_map[req_id]["data"] = self.response_map["open_orders"]["data"]

class InteractiveBrokersBroker(BrokerInterface):
    """
    Interactive Brokers implementation of the BrokerInterface
    
    Connects to Interactive Brokers Trader Workstation (TWS) or IB Gateway
    using the IB API to provide trading functionality.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # Default port for TWS Paper Trading
        client_id: int = 1,
        account_id: str = "",
        timeout: int = 30,
        auto_reconnect: bool = True
    ):
        """
        Initialize the Interactive Brokers broker adapter
        
        Args:
            host: IB API hostname or IP
            port: IB API port (7496 for TWS, 7497 for TWS Paper)
            client_id: IB API client ID
            account_id: IB account ID
            timeout: Request timeout in seconds
            auto_reconnect: Whether to automatically reconnect
        """
        if not HAS_IBAPI:
            raise ImportError("Interactive Brokers API (ibapi) not installed. Please install it with pip install ibapi.")
        
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_id = account_id
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        
        # Create the API client
        self.client = IBClientWrapper()
        self.connected = False
        
        # Threading
        self._api_thread = None
        self._lock = threading.Lock()
        
        # Connection management
        self._last_activity = 0
        self._watchdog_thread = None
        self._stop_watchdog = threading.Event()
        
        logger.info(f"Initialized Interactive Brokers adapter (host={host}, port={port}, client_id={client_id})")
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers API
        
        Returns:
            True if connection successful
        """
        if self.connected:
            logger.info("Already connected to Interactive Brokers")
            return True
        
        try:
            logger.info(f"Connecting to Interactive Brokers at {self.host}:{self.port}")
            
            # Connect to the API
            self.client.connect(self.host, self.port, self.client_id)
            
            # Wait for connection acknowledgement
            if not self.client.connect_event.wait(timeout=self.timeout):
                logger.error("Timed out waiting for IB connection")
                return False
            
            # Start the API thread
            if not self._api_thread or not self._api_thread.is_alive():
                self._api_thread = threading.Thread(target=self._run_client, daemon=True)
                self._api_thread.start()
            
            # Start the watchdog
            if self.auto_reconnect and (not self._watchdog_thread or not self._watchdog_thread.is_alive()):
                self._stop_watchdog.clear()
                self._watchdog_thread = threading.Thread(target=self._connection_watchdog, daemon=True)
                self._watchdog_thread.start()
            
            # Get account info if needed
            if not self.account_id:
                # Placeholder for getting the account ID dynamically
                # In a real implementation, we would query accounts and use the first one
                logger.warning("No account ID provided, functionality may be limited")
            
            self.connected = True
            self._last_activity = time.time()
            
            logger.info("Successfully connected to Interactive Brokers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Interactive Brokers: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Interactive Brokers API
        
        Returns:
            True if disconnection successful
        """
        if not self.connected:
            return True
        
        try:
            # Stop the watchdog
            if self._watchdog_thread and self._watchdog_thread.is_alive():
                self._stop_watchdog.set()
                self._watchdog_thread.join(timeout=5)
            
            # Disconnect API
            self.client.disconnect()
            
            # Wait for API thread to exit
            if self._api_thread and self._api_thread.is_alive():
                self._api_thread.join(timeout=5)
            
            self.connected = False
            logger.info("Disconnected from Interactive Brokers")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Interactive Brokers: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to Interactive Brokers API
        
        Returns:
            True if connected
        """
        return self.connected and self.client.is_connected
    
    def _run_client(self):
        """Run the IB API client (background thread)"""
        try:
            self.client.run()
        except Exception as e:
            logger.error(f"IB API client thread error: {str(e)}")
            self.connected = False
    
    def _connection_watchdog(self):
        """Monitor connection and reconnect if needed (background thread)"""
        while not self._stop_watchdog.is_set():
            time.sleep(30)  # Check every 30 seconds
            
            # If connected but no activity for a while, check connection
            if self.connected and time.time() - self._last_activity > 300:  # 5 minutes
                if not self.client.is_connected:
                    logger.warning("IB connection lost, attempting to reconnect")
                    self.connect()
    
    def _ensure_connected(self):
        """Ensure connection is established"""
        if not self.is_connected():
            if not self.connect():
                raise IBConnectionError("Not connected to Interactive Brokers")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dictionary of account information
        """
        self._ensure_connected()
        
        try:
            # Request account summary
            account_tags = "TotalCashValue,NetLiquidation,EquityWithLoanValue,BuyingPower"
            req_id, response = self.client.makeRequest(
                "account_summary",
                self.client.reqAccountSummary,
                account_tags
            )
            
            # Format response
            account_info = {}
            if self.account_id in response:
                for tag, data in response[self.account_id].items():
                    account_info[tag] = float(data["value"]) if data["value"].replace(".", "", 1).isdigit() else data["value"]
            
            self._last_activity = time.time()
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise IBError(f"Failed to get account info: {str(e)}")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
            List of position dictionaries
        """
        self._ensure_connected()
        
        try:
            # Request positions
            req_id, response = self.client.makeRequest(
                "positions",
                self.client.reqPositions
            )
            
            # Format response
            positions = []
            for account, account_positions in response.items():
                for symbol, position_data in account_positions.items():
                    if position_data["position"] != 0:  # Only include non-zero positions
                        positions.append({
                            "symbol": symbol,
                            "quantity": position_data["position"],
                            "avg_price": position_data["avg_cost"],
                            "asset_type": "equity"  # Default to equity
                        })
            
            self._last_activity = time.time()
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise IBError(f"Failed to get positions: {str(e)}")
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get open orders
        
        Returns:
            List of order dictionaries
        """
        self._ensure_connected()
        
        try:
            # Request open orders
            req_id, response = self.client.makeRequest(
                "open_orders",
                self.client.reqOpenOrders
            )
            
            # Format response
            orders = []
            for order_id, order_data in response.items():
                contract = order_data["contract"]
                order = order_data["order"]
                
                orders.append({
                    "order_id": str(order_id),
                    "symbol": contract.symbol,
                    "quantity": order.totalQuantity,
                    "order_type": order.orderType,
                    "side": "buy" if order.action == "BUY" else "sell",
                    "limit_price": order.lmtPrice if order.orderType == "LMT" else None,
                    "stop_price": order.auxPrice if order.orderType == "STP" else None,
                    "time_in_force": order.tif,
                    "status": self.client.order_status.get(order_id, {}).get("status", "Unknown")
                })
            
            self._last_activity = time.time()
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise IBError(f"Failed to get orders: {str(e)}")
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status dictionary
        """
        self._ensure_connected()
        
        try:
            # Check if order status is already available
            if int(order_id) in self.client.order_status:
                status = self.client.order_status[int(order_id)]
                
                return {
                    "order_id": order_id,
                    "status": status["status"],
                    "filled_quantity": status["filled"],
                    "remaining_quantity": status["remaining"],
                    "avg_fill_price": status["avg_fill_price"]
                }
            
            # Request order status
            req_id, response = self.client.makeRequest(
                f"order_status_{order_id}",
                self.client.reqOrderStatus,
                int(order_id)
            )
            
            # Format response
            if int(order_id) in self.client.order_status:
                status = self.client.order_status[int(order_id)]
                
                return {
                    "order_id": order_id,
                    "status": status["status"],
                    "filled_quantity": status["filled"],
                    "remaining_quantity": status["remaining"],
                    "avg_fill_price": status["avg_fill_price"]
                }
            
            self._last_activity = time.time()
            return {"order_id": order_id, "status": "Unknown"}
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            raise IBError(f"Failed to get order status: {str(e)}")
    
    def place_equity_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str,
        limit_price: float = None,
        stop_price: float = None,
        time_in_force: str = "DAY"
    ) -> Dict[str, Any]:
        """
        Place an equity order
        
        Args:
            symbol: Ticker symbol
            quantity: Order quantity
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop/stop_limit)
            limit_price: Limit price (for limit and stop-limit orders)
            stop_price: Stop price (for stop and stop-limit orders)
            time_in_force: Time in force (DAY/GTC/IOC/FOK)
            
        Returns:
            Order confirmation dictionary
        """
        self._ensure_connected()
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create order
            order = Order()
            order.action = "BUY" if side.upper() == "BUY" else "SELL"
            order.totalQuantity = quantity
            order.tif = time_in_force
            
            # Set order type and prices
            if order_type.upper() == "MARKET":
                order.orderType = "MKT"
            elif order_type.upper() == "LIMIT":
                order.orderType = "LMT"
                order.lmtPrice = limit_price
            elif order_type.upper() == "STOP":
                order.orderType = "STP"
                order.auxPrice = stop_price
            elif order_type.upper() == "STOP_LIMIT":
                order.orderType = "STP LMT"
                order.lmtPrice = limit_price
                order.auxPrice = stop_price
            else:
                raise IBError(f"Unsupported order type: {order_type}")
            
            # Place the order
            order_id = self.client.getNextReqId()
            self.client.placeOrder(order_id, contract, order)
            
            # Wait briefly for order acknowledgement
            time.sleep(1)
            
            self._last_activity = time.time()
            
            return {
                "order_id": str(order_id),
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
                "order_type": order_type,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "time_in_force": time_in_force,
                "status": "Submitted"
            }
            
        except Exception as e:
            logger.error(f"Error placing equity order: {str(e)}")
            raise IBError(f"Failed to place equity order: {str(e)}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancellation successful
        """
        self._ensure_connected()
        
        try:
            # Cancel the order
            self.client.cancelOrder(int(order_id))
            
            # Wait briefly for cancellation acknowledgement
            time.sleep(1)
            
            self._last_activity = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            raise IBError(f"Failed to cancel order: {str(e)}")
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get a quote for a symbol
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Quote dictionary
        """
        self._ensure_connected()
        
        # Not fully implemented - would require market data subscription handling
        # This is a simplified version
        raise NotImplementedError("Quote functionality not fully implemented for Interactive Brokers")
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical candles for a symbol
        
        Args:
            symbol: Ticker symbol
            timeframe: Candle timeframe (e.g., 1m, 5m, 1h, 1d)
            count: Number of candles to retrieve
            
        Returns:
            List of candle dictionaries
        """
        self._ensure_connected()
        
        # Not fully implemented - would require historical data handling
        # This is a simplified version
        raise NotImplementedError("Historical data functionality not fully implemented for Interactive Brokers")


# Usage example
def test_connection(host="127.0.0.1", port=7497, client_id=1):
    """Test connection to Interactive Brokers"""
    broker = InteractiveBrokersBroker(host=host, port=port, client_id=client_id)
    
    try:
        # Connect to IB
        print("Connecting to Interactive Brokers...")
        if broker.connect():
            print("Connected successfully!")
            
            # Get account info
            print("\nGetting account info...")
            account_info = broker.get_account_info()
            print(f"Account info: {account_info}")
            
            # Get positions
            print("\nGetting positions...")
            positions = broker.get_positions()
            print(f"Positions: {positions}")
            
            # Disconnect
            print("\nDisconnecting...")
            broker.disconnect()
            print("Disconnected!")
            
            return True
        else:
            print("Connection failed")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Run connection test
    logging.basicConfig(level=logging.INFO)
    test_connection()
