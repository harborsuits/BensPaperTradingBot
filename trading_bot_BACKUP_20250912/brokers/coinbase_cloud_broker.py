#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Cloud API Broker Implementation

This module implements a broker that connects to Coinbase using the Cloud API format
with JWT authentication. It uses native Python libraries to minimize dependencies.
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.request
import urllib.error
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import sys

# Add parent directory to path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_bot.brokers.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)

class CoinbaseCloudBroker(BrokerInterface):
    """
    A broker implementation for Coinbase using their Cloud API with JWT authentication.
    This is a more modern API compared to the standard REST API.
    """
    
    def __init__(self, api_key_name: str, private_key: str, sandbox: bool = False):
        """
        Initialize the Coinbase Cloud broker with API credentials.
        
        Args:
            api_key_name: The API key name/path from Coinbase Cloud
            private_key: The EC private key in PEM format
            sandbox: Whether to use the sandbox environment
        """
        self.api_key_name = api_key_name
        self.private_key = private_key
        self.sandbox = sandbox
        
        # Set base URLs based on environment - we know api.exchange.coinbase.com works with our IP whitelist
        # This is different from the original implementation which used api.coinbase.com
        self.base_url = "https://api.exchange.coinbase.com"
        
        # For historical reference, keeping the sandbox distinction even though we use the same URL
        self.is_sandbox = sandbox
            
        # API version paths
        self.cloud_api_path = "/api/v3/"
        self.rest_api_path = "/v2/"
        
        # Check if we can use optimized crypto libraries
        try:
            # Try to import cryptography for EC private key operations
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec, utils
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            import jwt  # PyJWT package
            
            self.crypto_libraries_available = True
            self.crypto = {
                "hashes": hashes,
                "ec": ec,
                "utils": utils, 
                "load_pem_private_key": load_pem_private_key,
                "jwt": jwt
            }
            logger.info("Using optimized crypto libraries for Coinbase Cloud API")
        except ImportError:
            self.crypto_libraries_available = False
            logger.warning(
                "Cryptography libraries not available. Coinbase Cloud API authentication "
                "will use a minimal implementation with limited functionality."
            )
        
        # Test connection at startup
        self.check_connection()
    
    def create_jwt_token(self) -> Optional[str]:
        """
        Create a JWT token signed with the EC private key.
        
        Returns:
            JWT token or None if creation failed
        """
        if self.crypto_libraries_available:
            try:
                # Use cryptography and PyJWT if available
                private_key = self.crypto["load_pem_private_key"](
                    self.private_key.encode(), 
                    password=None
                )
                
                # Create JWT payload
                now = int(time.time())
                payload = {
                    'sub': self.api_key_name,
                    'iss': 'coinbase-cloud',
                    'nbf': now,
                    'exp': now + 60,  # Token expires in 60 seconds
                    'aud': ['brokerage']  # This is the working audience value we confirmed
                }
                
                # Create and sign token
                token = self.crypto["jwt"].encode(
                    payload,
                    private_key,
                    algorithm='ES256'
                )
                
                return token
            except Exception as e:
                logger.error(f"Error creating JWT token: {e}")
                return None
        else:
            # Minimal implementation
            logger.error(
                "Cannot create JWT token without cryptography libraries. "
                "Please install 'cryptography' and 'pyjwt' packages."
            )
            return None
    
    def make_cloud_api_request(self, method: str, endpoint: str, body: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Make an authenticated request to the Coinbase Cloud API.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint (without base URL)
            body: Request body for POST/PUT requests
            
        Returns:
            Tuple of (success, response_data)
        """
        # Generate JWT token
        token = self.create_jwt_token()
        if not token:
            return False, {"error": "Failed to generate JWT token"}
            
        # Create headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Build URL - IMPORTANT: We know https://api.exchange.coinbase.com works with our whitelist
        # Use the endpoint directly without api_path since the Exchange API has a different structure
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            # Make request
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=body)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return False, {"error": f"Unsupported HTTP method: {method}"}
            
            # Handle response
            if response.status_code in [200, 201]:
                return True, response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return False, {"error": f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return False, {"error": f"Request error: {str(e)}"}
    
    def make_public_request(self, endpoint: str) -> Tuple[bool, Dict]:
        """
        Make a request to a public Coinbase API endpoint.
        
        Args:
            endpoint: API endpoint (without base URL)
            
        Returns:
            Tuple of (success, response_data)
        """
        url = self.base_url + self.rest_api_path + endpoint
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                return True, response_data
        except urllib.error.HTTPError as e:
            error_message = e.read().decode('utf-8')
            logger.error(f"HTTP error: {e.code} - {error_message}")
            return False, {"status_code": e.code, "error": error_message}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return False, {"error": str(e)}
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check connection to Coinbase API.
        
        Returns:
            Dict with connection status information
        """
        # Try public endpoint first
        success, response = self.make_public_request("currencies")
        
        if not success:
            logger.error("Failed to connect to Coinbase public API")
            return {
                "connected": False,
                "message": "Failed to connect to Coinbase public API",
                "error": response.get("error", "Unknown error")
            }
        
        # If crypto libraries aren't available, we can't do authenticated checks
        if not self.crypto_libraries_available:
            logger.warning("Crypto libraries not available - skipping authenticated connection check")
            return {
                "connected": True,
                "authenticated": False,
                "message": "Connected to public API only. Authenticated requests require cryptography libraries."
            }
        
        # Test authenticated endpoint
        auth_success, auth_response = self.make_cloud_api_request("GET", "brokerage/products")
        
        if not auth_success:
            logger.error("Connected to public API but authentication failed")
            return {
                "connected": True,
                "authenticated": False,
                "message": "Connected to public API but authentication failed",
                "error": auth_response.get("error", "Unknown error")
            }
        
        logger.info("Successfully connected to Coinbase Cloud API")
        return {
            "connected": True,
            "authenticated": True,
            "message": "Successfully connected to Coinbase Cloud API"
        }
    
    def get_account_balances(self) -> Dict[str, Any]:
        """
        Get account balances.
        
        Returns:
            Dict with account balance information
        """
        success, response = self.make_cloud_api_request("GET", "brokerage/accounts")
        
        if not success:
            logger.error("Failed to get account balances")
            return {}
        
        # Process and format account balances
        balances = {}
        for account in response.get("accounts", []):
            currency = account.get("currency", "")
            if currency:
                balances[currency] = {
                    "available": float(account.get("available_balance", {}).get("value", 0)),
                    "hold": float(account.get("hold", {}).get("value", 0)),
                    "total": float(account.get("balance", {}).get("value", 0))
                }
        
        return balances
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict with account information
        """
        # Get account balances
        balances = self.get_account_balances()
        
        # Calculate total value (in USD if available)
        total_value = 0
        for currency, balance in balances.items():
            if currency == "USD":
                total_value += balance["total"]
            else:
                # Try to get current price for this asset
                price = self.get_latest_price(f"{currency}-USD")
                if price > 0:
                    total_value += balance["total"] * price
        
        return {
            "total_value": total_value,
            "balances": balances,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            
        Returns:
            Latest price as a float
        """
        success, response = self.make_cloud_api_request(
            "GET", 
            f"brokerage/products/{symbol}/ticker"
        )
        
        if not success:
            logger.error(f"Failed to get price for {symbol}")
            return 0
        
        return float(response.get("price", 0))
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get a quote for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            
        Returns:
            Dict with quote information
        """
        success, response = self.make_cloud_api_request(
            "GET", 
            f"brokerage/products/{symbol}/ticker"
        )
        
        if not success:
            logger.error(f"Failed to get quote for {symbol}")
            return {}
        
        # Format in a standardized way
        return {
            "symbol": symbol,
            "bid": float(response.get("bid", 0)),
            "ask": float(response.get("ask", 0)),
            "last": float(response.get("price", 0)),
            "timestamp": response.get("time", datetime.now().isoformat())
        }
    
    def get_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> List[Dict]:
        """
        Get historical bars for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            timeframe: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start: Start datetime
            end: End datetime
            
        Returns:
            List of bar data
        """
        # Convert timeframe to granularity parameter
        granularity_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        granularity = granularity_map.get(timeframe, 3600)  # Default to 1h
        
        # Convert datetimes to ISO strings
        start_str = start.isoformat()
        end_str = end.isoformat()
        
        # Make request
        success, response = self.make_cloud_api_request(
            "GET",
            f"brokerage/products/{symbol}/candles",
            {
                "start": start_str,
                "end": end_str,
                "granularity": granularity
            }
        )
        
        if not success:
            logger.error(f"Failed to get bars for {symbol}")
            return []
        
        # Format candles in a standardized way
        candles = []
        for candle in response.get("candles", []):
            candles.append({
                "timestamp": candle.get("start", ""),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0))
            })
        
        return candles
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position data
        """
        # Get account balances
        balances = self.get_account_balances()
        
        # Convert balances to positions
        positions = []
        for currency, balance in balances.items():
            if currency != "USD" and balance["total"] > 0:
                # Get current price
                price = self.get_latest_price(f"{currency}-USD")
                
                positions.append({
                    "symbol": f"{currency}-USD",
                    "quantity": balance["total"],
                    "entry_price": 0,  # Not available without order history
                    "current_price": price,
                    "value_usd": balance["total"] * price,
                    "unrealized_pl": 0,  # Not available without entry price
                    "unrealized_pl_pct": 0,  # Not available without entry price
                    "last_updated": datetime.now().isoformat()
                })
        
        return positions
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            quantity: Quantity to buy/sell
            order_type: Order type ('market' or 'limit')
            
        Returns:
            Dict with order information
        """
        # Implement order placement logic using Cloud API
        # This is a placeholder implementation
        logger.warning("Order placement not fully implemented - running in read-only mode")
        
        return {
            "status": "simulated",
            "message": "Order placement simulated - running in read-only mode",
            "order_id": f"sim-{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type
        }
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get orders with optional status filter.
        
        Args:
            status: Order status filter (e.g., 'open', 'filled')
            
        Returns:
            List of order data
        """
        # Implement order retrieval logic using Cloud API
        # This is a placeholder implementation
        logger.warning("Order retrieval not fully implemented - running in read-only mode")
        
        return []
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict with cancellation status
        """
        # Implement order cancellation logic using Cloud API
        # This is a placeholder implementation
        logger.warning("Order cancellation not fully implemented - running in read-only mode")
        
        return {
            "status": "simulated",
            "message": "Order cancellation simulated - running in read-only mode",
            "order_id": order_id
        }
