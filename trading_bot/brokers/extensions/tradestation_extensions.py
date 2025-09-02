"""
TradeStation Broker Extensions

Implements broker-specific extensions for TradeStation, providing access to
TradeStation's unique features like futures trading and advanced market data.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import re

# Import the extension base classes
from trading_bot.brokers.broker_extensions import (
    FuturesExtension,
    TechnicalIndicatorExtension,
    StreamingDataExtension
)

# Configure logging
logger = logging.getLogger(__name__)


class TradeStationFuturesExtension(FuturesExtension):
    """
    Implements futures trading features for TradeStation.
    
    TradeStation provides comprehensive futures trading capabilities,
    margin requirements, contract specifications, and roll date information.
    """
    
    def __init__(self, tradestation_client):
        """
        Initialize the extension with a TradeStation client instance
        
        Args:
            tradestation_client: Instance of TradeStationClient
        """
        self.client = tradestation_client
        
        # Cache for contract specifications to avoid repeated API calls
        self._contract_specs_cache = {}
        self._cache_expiry = {}
        self._cache_ttl = timedelta(hours=24)  # Cache for 24 hours
    
    def get_futures_contracts(self, 
                            root_symbol: str, 
                            include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        Get available futures contracts for a root symbol
        
        Args:
            root_symbol: Futures root symbol (e.g., "ES", "CL", "GC")
            include_expired: Whether to include expired contracts
            
        Returns:
            List[Dict]: Available futures contracts
        """
        try:
            # Make API request to get futures contracts
            endpoint = f"/marketdata/symbols/futures/{root_symbol}"
            response = self.client._make_request("GET", endpoint)
            
            if not response or 'Contracts' not in response:
                return []
            
            # Parse contracts
            contracts = []
            now = datetime.now()
            
            for contract in response['Contracts']:
                # Extract contract details
                symbol = contract.get('Symbol', '')
                name = contract.get('Name', '')
                
                # Extract expiration date from name (e.g., "Dec 2023")
                expiration_match = re.search(r'([A-Za-z]+)\s+(\d{4})', name)
                if expiration_match:
                    month_name = expiration_match.group(1)
                    year = int(expiration_match.group(2))
                    
                    # Convert month name to month number
                    month_map = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    month = month_map.get(month_name[:3], 1)
                    
                    # Estimate expiration date (typically third Friday of month)
                    day = 15  # Middle of month as estimate
                    expiration_date = datetime(year, month, day)
                    
                    # Skip expired contracts if not requested
                    if not include_expired and expiration_date < now:
                        continue
                    
                    # Get contract specs
                    contract_specs = self.get_futures_contract_specs(symbol)
                    
                    contracts.append({
                        "symbol": symbol,
                        "root_symbol": root_symbol,
                        "name": name,
                        "expiration_date": expiration_date.strftime("%Y-%m-%d"),
                        "is_front_month": contract.get('IsFrontMonth', False),
                        "tick_size": contract_specs.get('tick_size', 0.01),
                        "tick_value": contract_specs.get('tick_value', 1.0),
                        "contract_size": contract_specs.get('contract_size', 1)
                    })
            
            # Sort by expiration date (ascending)
            contracts = sorted(contracts, key=lambda x: x['expiration_date'])
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting futures contracts for {root_symbol}: {str(e)}")
            return []
    
    def get_futures_contract_specs(self, symbol: str) -> Dict[str, Any]:
        """
        Get specifications for a futures contract
        
        Args:
            symbol: Futures contract symbol
            
        Returns:
            Dict: Contract specifications
        """
        try:
            # Check cache first
            now = datetime.now()
            if symbol in self._contract_specs_cache:
                if now < self._cache_expiry.get(symbol, now):
                    return self._contract_specs_cache[symbol]
            
            # Make API request to get contract specifications
            endpoint = f"/marketdata/symbols/{symbol}/info"
            response = self.client._make_request("GET", endpoint)
            
            if not response:
                return {}
            
            # Extract relevant contract specifications
            tick_size = 0.01  # Default
            tick_value = 1.0  # Default
            contract_size = 1  # Default
            
            # Extract from response
            if 'TickSize' in response:
                tick_size = float(response['TickSize'])
            
            if 'PointValue' in response:
                tick_value = float(response['PointValue'])
            
            if 'ContractSize' in response:
                try:
                    contract_size = int(response['ContractSize'])
                except ValueError:
                    # Handle case where contract size is not a simple number
                    # It might be descriptive like "5,000 bushels"
                    contract_size_str = response['ContractSize']
                    contract_size_match = re.search(r'(\d+),?(\d*)', contract_size_str)
                    if contract_size_match:
                        contract_size = int(contract_size_match.group(1) + contract_size_match.group(2))
            
            # Extract other specifications
            specs = {
                "symbol": symbol,
                "exchange": response.get('Exchange', ''),
                "tick_size": tick_size,
                "tick_value": tick_value,
                "contract_size": contract_size,
                "initial_margin": response.get('InitialMargin', 0),
                "maintenance_margin": response.get('MaintenanceMargin', 0),
                "currency": response.get('Currency', 'USD'),
                "trading_hours": response.get('TradingHours', ''),
                "product_type": response.get('ProductType', '')
            }
            
            # Cache the result
            self._contract_specs_cache[symbol] = specs
            self._cache_expiry[symbol] = now + self._cache_ttl
            
            return specs
            
        except Exception as e:
            logger.error(f"Error getting futures contract specs for {symbol}: {str(e)}")
            return {}
    
    def get_futures_margin_requirements(self, symbol: str) -> Dict[str, float]:
        """
        Get margin requirements for a futures contract
        
        Args:
            symbol: Futures contract symbol
            
        Returns:
            Dict: Initial and maintenance margin requirements
        """
        try:
            # Get contract specifications which include margin requirements
            specs = self.get_futures_contract_specs(symbol)
            
            # Extract margin requirements
            margins = {
                "initial_margin": specs.get('initial_margin', 0),
                "maintenance_margin": specs.get('maintenance_margin', 0),
                "day_trade_margin": specs.get('day_trade_margin', 0)
            }
            
            # If margin requirements are not available via the API,
            # use predefined values for common futures
            if margins["initial_margin"] == 0:
                # Extract root symbol from the contract symbol
                root_match = re.match(r'^([A-Z]+)', symbol)
                if root_match:
                    root = root_match.group(1)
                    
                    # Common margins for popular futures contracts (as of 2023)
                    common_margins = {
                        "ES": {"initial": 12100, "maintenance": 11000, "day_trade": 500},  # E-mini S&P 500
                        "NQ": {"initial": 15000, "maintenance": 13500, "day_trade": 600},  # E-mini Nasdaq 100
                        "CL": {"initial": 5800, "maintenance": 5250, "day_trade": 400},    # Crude Oil
                        "GC": {"initial": 7750, "maintenance": 7050, "day_trade": 550},    # Gold
                        "ZB": {"initial": 4050, "maintenance": 3000, "day_trade": 300},    # 30-Year Treasury Bond
                        "ZN": {"initial": 2300, "maintenance": 1700, "day_trade": 200},    # 10-Year Treasury Note
                        "6E": {"initial": 2750, "maintenance": 2500, "day_trade": 250}     # Euro FX
                    }
                    
                    if root in common_margins:
                        margins = {
                            "initial_margin": common_margins[root]["initial"],
                            "maintenance_margin": common_margins[root]["maintenance"],
                            "day_trade_margin": common_margins[root]["day_trade"]
                        }
            
            return margins
            
        except Exception as e:
            logger.error(f"Error getting futures margin requirements for {symbol}: {str(e)}")
            return {"initial_margin": 0, "maintenance_margin": 0, "day_trade_margin": 0}
    
    def get_futures_roll_dates(self, root_symbol: str) -> Dict[str, datetime]:
        """
        Get roll dates for a futures contract
        
        Args:
            root_symbol: Futures root symbol
            
        Returns:
            Dict: Upcoming roll dates
        """
        try:
            # Get available contracts for root
            contracts = self.get_futures_contracts(root_symbol)
            
            if not contracts:
                return {}
            
            # Find front month contract
            front_month = None
            for contract in contracts:
                if contract.get('is_front_month', False):
                    front_month = contract
                    break
            
            if not front_month:
                front_month = contracts[0]  # Assume first contract is front month
            
            # Calculate roll dates based on contract specs and standard practices
            expiration_date = datetime.strptime(front_month['expiration_date'], "%Y-%m-%d")
            
            # Common roll date rules by product type
            
            # Financial futures (ES, NQ, etc.) typically roll 8 days before expiration
            financial_roll_date = expiration_date - timedelta(days=8)
            
            # Energy futures (CL, NG, etc.) typically roll 3 days before expiration
            energy_roll_date = expiration_date - timedelta(days=3)
            
            # Agricultural futures vary but often roll 2 weeks before expiration
            agricultural_roll_date = expiration_date - timedelta(days=14)
            
            # Metals typically roll a week before expiration
            metals_roll_date = expiration_date - timedelta(days=7)
            
            # Return all possible roll dates, caller can decide which is appropriate
            roll_dates = {
                "expiration_date": expiration_date,
                "financial_roll_date": financial_roll_date,
                "energy_roll_date": energy_roll_date,
                "agricultural_roll_date": agricultural_roll_date,
                "metals_roll_date": metals_roll_date
            }
            
            # Identify the probable roll date based on root symbol
            financial_futures = ["ES", "NQ", "YM", "RTY", "ZB", "ZN", "ZF", "ZT", "6E", "6J", "6B", "6C", "6A"]
            energy_futures = ["CL", "NG", "RB", "HO", "QG"]
            agricultural_futures = ["ZC", "ZS", "ZW", "ZM", "ZL", "KE", "LE", "GF", "HE"]
            metals_futures = ["GC", "SI", "HG", "PL", "PA"]
            
            if root_symbol in financial_futures:
                roll_dates["recommended_roll_date"] = financial_roll_date
            elif root_symbol in energy_futures:
                roll_dates["recommended_roll_date"] = energy_roll_date
            elif root_symbol in agricultural_futures:
                roll_dates["recommended_roll_date"] = agricultural_roll_date
            elif root_symbol in metals_futures:
                roll_dates["recommended_roll_date"] = metals_roll_date
            else:
                roll_dates["recommended_roll_date"] = financial_roll_date  # Default to financial
            
            return roll_dates
            
        except Exception as e:
            logger.error(f"Error getting futures roll dates for {root_symbol}: {str(e)}")
            return {}
    
    def get_extension_name(self) -> str:
        return "TradeStationFuturesExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"futures_contracts", "futures_specs", "futures_margins", "futures_roll_dates"}


class TradeStationStreamingExtension(StreamingDataExtension):
    """
    Implements real-time streaming data for TradeStation.
    
    TradeStation provides WebSocket-based streaming for quotes, bars, and other data.
    """
    
    def __init__(self, tradestation_client):
        """
        Initialize the extension with a TradeStation client instance
        
        Args:
            tradestation_client: Instance of TradeStationClient
        """
        self.client = tradestation_client
        self.ws = None
        self.callbacks = {}
        self.subscriptions = set()
        self.running = False
        self.ws_thread = None
    
    def _connect_websocket(self):
        """
        Connect to TradeStation's WebSocket API
        """
        try:
            # Import websocket here to avoid dependency if not used
            import websocket
            import threading
            import json
            import time
            
            # Initialize connection
            base_url = "wss://api.tradestation.com/v3/MarketData/Stream"
            
            # Get access token
            access_token = self.client.get_access_token()
            if not access_token:
                logger.error("Failed to get access token for WebSocket connection")
                return False
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                base_url,
                header={"Authorization": f"Bearer {access_token}"},
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a background thread
            self.running = True
            self.ws_thread = threading.Thread(target=self._run_websocket)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection to be established
            timeout = 10
            start_time = time.time()
            while not self.ws.sock or not self.ws.sock.connected:
                if time.time() - start_time > timeout:
                    raise ConnectionError("Timed out connecting to WebSocket")
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to TradeStation WebSocket: {str(e)}")
            return False
    
    def _run_websocket(self):
        """Run the WebSocket connection"""
        try:
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket thread error: {str(e)}")
        finally:
            self.running = False
    
    def _on_open(self, ws):
        """Called when WebSocket connection is opened"""
        logger.info("TradeStation WebSocket connection opened")
        
        # Resubscribe to any existing subscriptions
        if self.subscriptions:
            for subscription in self.subscriptions:
                self._send_subscription(subscription)
    
    def _on_message(self, ws, message):
        """Called when a message is received from WebSocket"""
        try:
            data = json.loads(message)
            
            # Check if it's a heartbeat message
            if 'HeartBeat' in data:
                return
            
            # Handle quotes
            if 'Quote' in data:
                quote_data = data['Quote']
                symbol = quote_data.get('Symbol')
                if symbol and 'quote' in self.callbacks:
                    callback = self.callbacks.get('quote')
                    if callback:
                        quote = {
                            'symbol': symbol,
                            'bid_price': float(quote_data.get('BidPrice', 0)),
                            'ask_price': float(quote_data.get('AskPrice', 0)),
                            'bid_size': int(quote_data.get('BidSize', 0)),
                            'ask_size': int(quote_data.get('AskSize', 0)),
                            'last_price': float(quote_data.get('LastPrice', 0)),
                            'timestamp': pd.Timestamp(quote_data.get('TimeStamp')).to_pydatetime()
                        }
                        callback(quote)
            
            # Handle bars
            elif 'Bar' in data:
                bar_data = data['Bar']
                symbol = bar_data.get('Symbol')
                if symbol and 'bar' in self.callbacks:
                    callback = self.callbacks.get('bar')
                    if callback:
                        bar = {
                            'symbol': symbol,
                            'open': float(bar_data.get('Open', 0)),
                            'high': float(bar_data.get('High', 0)),
                            'low': float(bar_data.get('Low', 0)),
                            'close': float(bar_data.get('Close', 0)),
                            'volume': int(bar_data.get('Volume', 0)),
                            'timestamp': pd.Timestamp(bar_data.get('TimeStamp')).to_pydatetime()
                        }
                        callback(bar)
                        
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Called when a WebSocket error occurs"""
        logger.error(f"TradeStation WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket connection is closed"""
        logger.info(f"TradeStation WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.running = False
        
        # Attempt to reconnect
        if not self.running:
            logger.info("Attempting to reconnect to TradeStation WebSocket...")
            time.sleep(5)  # Wait before reconnecting
            self._connect_websocket()
    
    def _send_subscription(self, subscription):
        """Send subscription request to WebSocket"""
        if not self.ws:
            return False
        
        self.ws.send(json.dumps(subscription))
        return True
    
    def subscribe_to_quotes(self, symbols: List[str], callback: callable) -> bool:
        """
        Subscribe to real-time quote updates for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call with each update
            
        Returns:
            bool: Success status
        """
        if not symbols:
            return False
            
        # Connect WebSocket if not already connected
        if not self.ws:
            if not self._connect_websocket():
                return False
        
        # Store callback
        self.callbacks['quote'] = callback
        
        # Create subscription request
        subscription = {
            "Request": "Subscribe",
            "Symbol": ",".join(symbols),
            "Service": "Quote"
        }
        
        # Add to subscriptions
        self.subscriptions.add(json.dumps(subscription))
        
        # Send subscription request
        return self._send_subscription(subscription)
    
    def subscribe_to_bars(self, symbols: List[str], timeframe: str, callback: callable) -> bool:
        """
        Subscribe to real-time bar updates for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            timeframe: Bar timeframe (e.g., "1min", "5min", "1h")
            callback: Function to call with each update
            
        Returns:
            bool: Success status
        """
        if not symbols:
            return False
        
        # Map user-friendly timeframe to TradeStation format
        timeframe_map = {
            "1min": "1", "5min": "5", "15min": "15", "30min": "30",
            "1h": "60", "4h": "240", "1d": "D"
        }
        
        ts_timeframe = timeframe_map.get(timeframe.lower(), "1")
        
        # Connect WebSocket if not already connected
        if not self.ws:
            if not self._connect_websocket():
                return False
        
        # Store callback
        self.callbacks['bar'] = callback
        
        # Create subscription request
        subscription = {
            "Request": "Subscribe",
            "Symbol": ",".join(symbols),
            "Service": "Bar",
            "Interval": ts_timeframe
        }
        
        # Add to subscriptions
        self.subscriptions.add(json.dumps(subscription))
        
        # Send subscription request
        return self._send_subscription(subscription)
    
    def unsubscribe_all(self) -> bool:
        """
        Unsubscribe from all streaming data
        
        Returns:
            bool: Success status
        """
        if not self.ws:
            return True  # Nothing to unsubscribe from
        
        try:
            for subscription in self.subscriptions:
                # Parse the subscription
                sub_data = json.loads(subscription)
                
                # Create unsubscribe request
                unsub_data = {
                    "Request": "Unsubscribe",
                    "Symbol": sub_data.get("Symbol", ""),
                    "Service": sub_data.get("Service", "")
                }
                
                if "Interval" in sub_data:
                    unsub_data["Interval"] = sub_data["Interval"]
                
                # Send unsubscribe request
                self.ws.send(json.dumps(unsub_data))
            
            self.subscriptions.clear()
            self.callbacks.clear()
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from TradeStation streams: {str(e)}")
            return False
    
    def get_extension_name(self) -> str:
        return "TradeStationStreamingExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"streaming_quotes", "streaming_bars"}
