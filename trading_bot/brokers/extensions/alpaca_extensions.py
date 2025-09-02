"""
Alpaca Broker Extensions

Implements broker-specific extensions for Alpaca, providing access to
Alpaca's unique features like streaming data and crypto trading.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
import pandas as pd
import threading
import time
import json

# Import the extension base classes
from trading_bot.brokers.broker_extensions import (
    StreamingDataExtension,
    CryptoExtension,
    PortfolioAnalysisExtension
)

# Configure logging
logger = logging.getLogger(__name__)


class AlpacaStreamingExtension(StreamingDataExtension):
    """
    Implements real-time streaming data for Alpaca.
    
    Alpaca provides WebSocket-based streaming for quotes, trades, and bars.
    This extension wraps those capabilities in a standardized interface.
    """
    
    def __init__(self, alpaca_client):
        """
        Initialize the extension with an Alpaca client instance
        
        Args:
            alpaca_client: Instance of AlpacaClient
        """
        self.client = alpaca_client
        self.ws = None
        self.callbacks = {}
        self.subscriptions = set()
        self.running = False
        self.ws_thread = None
    
    def _connect_websocket(self):
        """
        Connect to Alpaca's WebSocket API
        """
        try:
            # Initialize connection based on environment
            base_url = "wss://stream.data.alpaca.markets/v2"
            if self.client.paper_trading:
                base_url = "wss://stream.data.paper-api.alpaca.markets/v2"
            
            # Import websocket here to avoid dependency if not used
            import websocket
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                base_url,
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
            
            # Authenticate
            self._authenticate()
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Alpaca WebSocket: {str(e)}")
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
        logger.info("Alpaca WebSocket connection opened")
    
    def _on_message(self, ws, message):
        """Called when a message is received from WebSocket"""
        try:
            data = json.loads(message)
            
            # Handle authorization response
            if 'type' in data and data['type'] == 'authorization':
                if data['status'] == 'authorized':
                    logger.info("Alpaca WebSocket authorized successfully")
                    # Resubscribe to any existing subscriptions
                    if self.subscriptions:
                        self._subscribe()
                else:
                    logger.error(f"Alpaca WebSocket authorization failed: {data.get('message', 'Unknown error')}")
                return
            
            # Handle quotes
            if 'T' in data and data['T'] == 'q':  # Quote data
                symbol = data.get('S')
                if symbol and 'quote' in self.callbacks:
                    callback = self.callbacks.get('quote')
                    if callback:
                        quote = {
                            'symbol': symbol,
                            'bid_price': float(data.get('bp', 0)),
                            'ask_price': float(data.get('ap', 0)),
                            'bid_size': int(data.get('bs', 0)),
                            'ask_size': int(data.get('as', 0)),
                            'timestamp': pd.Timestamp(data.get('t')).to_pydatetime()
                        }
                        callback(quote)
            
            # Handle trades
            elif 'T' in data and data['T'] == 't':  # Trade data
                symbol = data.get('S')
                if symbol and 'trade' in self.callbacks:
                    callback = self.callbacks.get('trade')
                    if callback:
                        trade = {
                            'symbol': symbol,
                            'price': float(data.get('p', 0)),
                            'size': int(data.get('s', 0)),
                            'timestamp': pd.Timestamp(data.get('t')).to_pydatetime()
                        }
                        callback(trade)
            
            # Handle bars
            elif 'T' in data and data['T'] == 'b':  # Bar data
                symbol = data.get('S')
                if symbol and 'bar' in self.callbacks:
                    callback = self.callbacks.get('bar')
                    if callback:
                        bar = {
                            'symbol': symbol,
                            'open': float(data.get('o', 0)),
                            'high': float(data.get('h', 0)),
                            'low': float(data.get('l', 0)),
                            'close': float(data.get('c', 0)),
                            'volume': int(data.get('v', 0)),
                            'timestamp': pd.Timestamp(data.get('t')).to_pydatetime()
                        }
                        callback(bar)
                        
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Called when a WebSocket error occurs"""
        logger.error(f"Alpaca WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket connection is closed"""
        logger.info(f"Alpaca WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.running = False
        
        # Attempt to reconnect
        if not self.running:
            logger.info("Attempting to reconnect to Alpaca WebSocket...")
            time.sleep(5)  # Wait before reconnecting
            self._connect_websocket()
    
    def _authenticate(self):
        """Authenticate with Alpaca WebSocket API"""
        if not self.ws:
            return False
        
        auth_msg = {
            "action": "auth",
            "key": self.client.api_key,
            "secret": self.client.api_secret
        }
        
        self.ws.send(json.dumps(auth_msg))
        return True
    
    def _subscribe(self):
        """Send subscription request for current subscriptions"""
        if not self.ws or not self.subscriptions:
            return False
        
        sub_msg = {
            "action": "subscribe",
            "trades": [s for s in self.subscriptions if "trade:" in s],
            "quotes": [s.replace("quote:", "") for s in self.subscriptions if "quote:" in s],
            "bars": [s.replace("bar:", "") for s in self.subscriptions if "bar:" in s]
        }
        
        # Clean up empty lists
        for key in list(sub_msg.keys()):
            if not sub_msg[key]:
                del sub_msg[key]
        
        if len(sub_msg) <= 1:  # Just the action key
            return False
            
        self.ws.send(json.dumps(sub_msg))
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
        
        # Add to subscriptions
        for symbol in symbols:
            self.subscriptions.add(f"quote:{symbol}")
        
        # Send subscription request
        return self._subscribe()
    
    def subscribe_to_bars(self, symbols: List[str], timeframe: str, callback: callable) -> bool:
        """
        Subscribe to real-time bar updates for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            timeframe: Bar timeframe (currently only "1Min" supported by Alpaca)
            callback: Function to call with each update
            
        Returns:
            bool: Success status
        """
        if not symbols:
            return False
        
        # Verify timeframe (Alpaca only supports 1-minute bars)
        if timeframe not in ["1Min", "1M", "1m"]:
            logger.warning("Alpaca only supports 1-minute bars for streaming. Using 1-minute bars.")
        
        # Connect WebSocket if not already connected
        if not self.ws:
            if not self._connect_websocket():
                return False
        
        # Store callback
        self.callbacks['bar'] = callback
        
        # Add to subscriptions
        for symbol in symbols:
            self.subscriptions.add(f"bar:{symbol}")
        
        # Send subscription request
        return self._subscribe()
    
    def unsubscribe_all(self) -> bool:
        """
        Unsubscribe from all streaming data
        
        Returns:
            bool: Success status
        """
        if not self.ws:
            return True  # Nothing to unsubscribe from
        
        try:
            unsub_msg = {
                "action": "unsubscribe",
                "trades": [s for s in self.subscriptions if "trade:" in s],
                "quotes": [s.replace("quote:", "") for s in self.subscriptions if "quote:" in s],
                "bars": [s.replace("bar:", "") for s in self.subscriptions if "bar:" in s]
            }
            
            # Clean up empty lists
            for key in list(unsub_msg.keys()):
                if not unsub_msg[key]:
                    del unsub_msg[key]
            
            if len(unsub_msg) <= 1:  # Just the action key
                return True
                
            self.ws.send(json.dumps(unsub_msg))
            self.subscriptions.clear()
            self.callbacks.clear()
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from Alpaca streams: {str(e)}")
            return False
    
    def get_extension_name(self) -> str:
        return "AlpacaStreamingExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"streaming_quotes", "streaming_bars", "streaming_trades"}


class AlpacaCryptoExtension(CryptoExtension):
    """
    Implements crypto-specific features for Alpaca.
    
    Alpaca provides access to cryptocurrency trading and data.
    """
    
    def __init__(self, alpaca_client):
        """
        Initialize the extension with an Alpaca client instance
        
        Args:
            alpaca_client: Instance of AlpacaClient
        """
        self.client = alpaca_client
    
    def get_crypto_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get order book data for a cryptocurrency
        
        Args:
            symbol: Crypto symbol (e.g., "BTC/USD")
            depth: Depth of the order book to return
            
        Returns:
            Dict: Order book with bids and asks
        """
        try:
            # Format symbol correctly for Alpaca API
            formatted_symbol = symbol.replace("/", "")
            
            # Make API request to get orderbook
            endpoint = f"/v2/crypto/{formatted_symbol}/orderbook"
            params = {"limit": depth}
            
            response = self.client._make_request("GET", endpoint, params)
            
            # Format response to standard structure
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "bids": [{"price": float(bid[0]), "size": float(bid[1])} for bid in response.get("bids", [])],
                "asks": [{"price": float(ask[0]), "size": float(ask[1])} for ask in response.get("asks", [])]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting crypto orderbook: {str(e)}")
            return {"symbol": symbol, "timestamp": datetime.now(), "bids": [], "asks": [], "error": str(e)}
    
    def get_crypto_trading_pairs(self) -> List[str]:
        """
        Get available cryptocurrency trading pairs
        
        Returns:
            List[str]: Available trading pairs
        """
        try:
            # Make API request to get available crypto assets
            response = self.client._make_request("GET", "/v2/crypto/assets")
            
            # Extract and format trading pairs
            pairs = []
            for asset in response:
                symbol = asset.get("symbol", "")
                if symbol:
                    # Convert format to standard (e.g., "BTCUSD" to "BTC/USD")
                    base = symbol[:-3]  # Extract base currency
                    quote = symbol[-3:]  # Extract quote currency
                    pairs.append(f"{base}/{quote}")
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error getting crypto trading pairs: {str(e)}")
            return []
    
    def get_crypto_account_history(self, 
                                 symbol: Optional[str] = None,
                                 start: Optional[datetime] = None,
                                 end: Optional[datetime] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get crypto account history (deposits, withdrawals, transfers)
        
        Args:
            symbol: Optional symbol to filter by
            start: Start date
            end: End date
            limit: Maximum records to return
            
        Returns:
            List[Dict]: Account history records
        """
        try:
            # Format parameters
            params = {"limit": limit}
            
            if symbol:
                params["symbol"] = symbol.replace("/", "")
                
            if start:
                params["start"] = start.isoformat()
                
            if end:
                params["end"] = end.isoformat()
            
            # Make API request
            response = self.client._make_request("GET", "/v2/account/activities", params)
            
            # Filter to crypto-related activities
            crypto_activities = []
            for activity in response:
                activity_type = activity.get("activity_type", "")
                if "CRYPTO" in activity_type:
                    crypto_activities.append({
                        "id": activity.get("id", ""),
                        "type": activity_type,
                        "symbol": activity.get("symbol", ""),
                        "timestamp": pd.Timestamp(activity.get("date")).to_pydatetime(),
                        "quantity": float(activity.get("qty", 0)),
                        "price": float(activity.get("price", 0)),
                        "net_amount": float(activity.get("net_amount", 0))
                    })
            
            return crypto_activities
            
        except Exception as e:
            logger.error(f"Error getting crypto account history: {str(e)}")
            return []
    
    def get_extension_name(self) -> str:
        return "AlpacaCryptoExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"crypto_orderbook", "crypto_pairs", "crypto_history"}


class AlpacaPortfolioExtension(PortfolioAnalysisExtension):
    """
    Implements portfolio analysis features for Alpaca.
    
    Alpaca provides access to portfolio metrics and performance data.
    """
    
    def __init__(self, alpaca_client):
        """
        Initialize the extension with an Alpaca client instance
        
        Args:
            alpaca_client: Instance of AlpacaClient
        """
        self.client = alpaca_client
    
    def get_portfolio_risk_metrics(self) -> Dict[str, float]:
        """
        Get risk metrics for the current portfolio
        
        Returns:
            Dict: Risk metrics (beta, VaR, etc.)
        """
        try:
            # Make API request to get portfolio metrics
            response = self.client._make_request("GET", "/v2/account/portfolio/metrics")
            
            # Extract relevant metrics
            metrics = {
                "equity": float(response.get("equity", 0)),
                "profit_loss": float(response.get("profit_loss", 0)),
                "profit_loss_pct": float(response.get("profit_loss_pct", 0)) * 100,  # Convert to percentage
                "beta": float(response.get("beta", 0)),
                "alpha": float(response.get("alpha", 0)),
                "sharpe": float(response.get("sharpe", 0)),
                "volatility": float(response.get("volatility", 0)) * 100,  # Convert to percentage
                "max_drawdown": float(response.get("max_drawdown", 0)) * 100,  # Convert to percentage
                "sortino": float(response.get("sortino", 0)),
                "treynor": float(response.get("treynor", 0))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting portfolio risk metrics: {str(e)}")
            return {}
    
    def get_position_performance(self, 
                               symbol: Optional[str] = None, 
                               timeframe: str = "1D",
                               start: Optional[datetime] = None,
                               end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get performance metrics for positions
        
        Args:
            symbol: Optional symbol to filter by
            timeframe: Analysis timeframe
            start: Start date
            end: End date
            
        Returns:
            DataFrame: Performance metrics
        """
        try:
            # Get current positions
            positions = self.client.get_positions()
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
            
            if not positions:
                return pd.DataFrame()
            
            # Get historical bars for each position for performance calculation
            results = []
            for position in positions:
                # Set default dates if not provided
                if not end:
                    end = datetime.now()
                if not start:
                    start = end - timedelta(days=30)
                
                # Get historical data
                bars = self.client.get_bars(
                    position.symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end
                )
                
                if not bars:
                    continue
                
                # Calculate performance metrics
                entry_price = position.avg_entry_price
                current_price = position.current_price
                price_change = current_price - entry_price
                price_change_pct = (price_change / entry_price) * 100 if entry_price else 0
                
                # Calculate additional metrics from historical data
                prices = [bar.close_price for bar in bars]
                pct_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                
                # Add to results
                results.append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "market_value": position.market_value,
                    "unrealized_pl": position.unrealized_pl,
                    "price_change": price_change,
                    "price_change_pct": price_change_pct,
                    "volatility": float(pd.Series(pct_changes).std() * 100) if pct_changes else 0,  # Convert to percentage
                    "max_price": float(pd.Series(prices).max()) if prices else 0,
                    "min_price": float(pd.Series(prices).min()) if prices else 0
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error getting position performance: {str(e)}")
            return pd.DataFrame()
    
    def get_portfolio_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for current portfolio holdings
        
        Returns:
            DataFrame: Correlation matrix
        """
        try:
            # Get current positions
            positions = self.client.get_positions()
            if not positions:
                return pd.DataFrame()
            
            # Get historical data for correlation calculation
            symbols = [position.symbol for position in positions]
            end = datetime.now()
            start = end - timedelta(days=30)  # Use 30 days of data
            
            price_data = {}
            for symbol in symbols:
                bars = self.client.get_bars(
                    symbol,
                    timeframe="1D",  # Daily data for correlation
                    start=start,
                    end=end
                )
                
                if bars:
                    # Create time series of prices
                    times = [bar.timestamp for bar in bars]
                    prices = [bar.close_price for bar in bars]
                    price_data[symbol] = pd.Series(prices, index=times)
            
            if not price_data:
                return pd.DataFrame()
            
            # Create DataFrame from price data
            price_df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            correlation = price_df.pct_change().corr()
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error getting portfolio correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def get_extension_name(self) -> str:
        return "AlpacaPortfolioExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"portfolio_risk", "position_performance", "portfolio_correlation"}
