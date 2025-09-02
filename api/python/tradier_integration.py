#!/usr/bin/env python3
"""
Tradier Integration Module

This module serves as a bridge between the trading bot and the Tradier API client,
providing methods for trade execution, position management, and market data access.
"""

import os
import logging
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from trading_bot.tradier_client import TradierClient, TradierAPIError
from trading_bot.config_manager import ConfigManager
from trading_bot.security_utils import get_secure_env_var

logger = logging.getLogger(__name__)

class TradierIntegration:
    """
    Integration class for Tradier broker API that provides high-level
    trading functions for the trading bot.
    
    This class wraps the TradierClient and provides additional functionality
    for order management, position tracking, and data analysis.
    """
    
    def __init__(
        self,
        api_key: str,
        account_id: str,
        use_sandbox: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Tradier integration.
        
        Args:
            api_key: Tradier API key
            account_id: Tradier account ID
            use_sandbox: Whether to use the sandbox environment
            config: Additional configuration options
        """
        self.account_id = account_id
        self.use_sandbox = use_sandbox
        
        # Default configuration
        self.config = {
            "max_retries": 3,
            "retry_delay": 1,
            "timeout": 30,
            "order_tags": {"source": "trading_bot"},
            "max_positions": 10,
            "max_position_size_percent": 5.0,
            "default_stop_loss_percent": 2.0,
            "default_take_profit_percent": 4.0,
            "default_order_type": "limit",
            "default_time_in_force": "day"
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Initialize the Tradier client
        self.client = TradierClient(
            api_key=api_key,
            account_id=account_id,
            use_sandbox=use_sandbox,
            timeout=self.config["timeout"],
            max_retries=self.config["max_retries"],
            retry_delay=self.config["retry_delay"]
        )
        
        # Cache for data
        self._cache = {
            "account_summary": {"data": None, "timestamp": None},
            "positions": {"data": None, "timestamp": None},
            "orders": {"data": None, "timestamp": None},
            "market_status": {"data": None, "timestamp": None}
        }
        
        # Cache expiry (in seconds)
        self._cache_expiry = 60
        
        logger.info(f"Initialized Tradier integration with {'sandbox' if use_sandbox else 'live'} environment")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid."""
        if not self._cache[cache_key]["timestamp"]:
            return False
        
        elapsed = (datetime.now() - self._cache[cache_key]["timestamp"]).total_seconds()
        return elapsed < self._cache_expiry
    
    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Update cache with new data."""
        self._cache[cache_key]["data"] = data
        self._cache[cache_key]["timestamp"] = datetime.now()
    
    def get_account_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get account summary including balances, positions, and open orders.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Account summary information
        """
        # Check cache first
        if use_cache and self._is_cache_valid("account_summary"):
            logger.debug("Using cached account summary")
            return self._cache["account_summary"]["data"]
        
        try:
            # Get account balances
            balances = self.client.get_account_balance()
            
            # Get positions
            try:
                positions_response = self.client.get_account_positions()
                positions = positions_response.get("positions", {}).get("position", [])
                # Ensure positions is a list
                if positions and not isinstance(positions, list):
                    positions = [positions]
            except TradierAPIError as e:
                logger.warning(f"Failed to get positions: {str(e)}")
                positions = []
            
            # Get open orders
            try:
                orders_response = self.client.get_account_orders()
                orders = orders_response.get("orders", {}).get("order", [])
                # Ensure orders is a list
                if orders and not isinstance(orders, list):
                    orders = [orders]
                # Filter for open orders
                open_orders = [order for order in orders if order.get("status") == "open"]
            except TradierAPIError as e:
                logger.warning(f"Failed to get orders: {str(e)}")
                open_orders = []
            
            # Calculate today's P&L
            today_pnl = 0
            try:
                # Get today's history
                history_response = self.client.get_account_history()
                history_items = history_response.get("history", {}).get("item", [])
                
                # Ensure history_items is a list
                if history_items and not isinstance(history_items, list):
                    history_items = [history_items]
                
                # Look for trades today
                today = datetime.now().strftime("%Y-%m-%d")
                for item in history_items:
                    if item.get("date", "").startswith(today) and item.get("type") == "trade":
                        today_pnl += float(item.get("amount", 0))
            except TradierAPIError as e:
                logger.warning(f"Failed to get account history: {str(e)}")
            
            # Build the summary
            account_summary = {
                "account_number": self.account_id,
                "environment": "sandbox" if self.use_sandbox else "live",
                "balances": {
                    "total_equity": float(balances["balances"]["total_equity"]),
                    "option_buying_power": float(balances["balances"]["option_buying_power"]),
                    "stock_buying_power": float(balances["balances"]["stock_buying_power"]),
                    "cash": float(balances["balances"]["cash"]),
                    "market_value": float(balances["balances"]["market_value"]),
                },
                "positions_count": len(positions),
                "open_orders_count": len(open_orders),
                "today_pnl": today_pnl,
                "today_pnl_percent": (today_pnl / float(balances["balances"]["total_equity"])) * 100 if float(balances["balances"]["total_equity"]) > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cache
            self._update_cache("account_summary", account_summary)
            
            return account_summary
            
        except TradierAPIError as e:
            logger.error(f"Failed to get account summary: {str(e)}")
            raise
    
    def get_positions_dataframe(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get current positions as a pandas DataFrame.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame of positions with calculated metrics
        """
        # Check cache first
        if use_cache and self._is_cache_valid("positions"):
            if self._cache["positions"]["data"] is not None:
                logger.debug("Using cached positions")
                return self._cache["positions"]["data"]
        
        try:
            # Get positions
            positions_response = self.client.get_account_positions()
            positions = positions_response.get("positions", {}).get("position", [])
            
            # Ensure positions is a list
            if positions and not isinstance(positions, list):
                positions = [positions]
            
            if not positions:
                empty_df = pd.DataFrame(columns=[
                    "symbol", "quantity", "cost_basis", "date_acquired", 
                    "current_price", "current_value", "unrealized_pnl", 
                    "unrealized_pnl_percent", "today_pnl", "today_pnl_percent"
                ])
                self._update_cache("positions", empty_df)
                return empty_df
            
            # Get current quotes
            symbols = [position["symbol"] for position in positions]
            quotes_response = self.client.get_quotes(symbols)
            quotes = quotes_response.get("quotes", {}).get("quote", [])
            
            # Ensure quotes is a list
            if quotes and not isinstance(quotes, list):
                quotes = [quotes]
            
            # Build a lookup for quotes
            quotes_lookup = {quote["symbol"]: quote for quote in quotes}
            
            # Build position data
            position_data = []
            for position in positions:
                symbol = position["symbol"]
                quantity = float(position["quantity"])
                cost_basis = float(position["cost_basis"])
                date_acquired = position.get("date_acquired", "")
                
                # Get current price
                current_price = float(quotes_lookup.get(symbol, {}).get("last", 0))
                
                # Calculate metrics
                current_value = quantity * current_price
                unrealized_pnl = current_value - cost_basis
                unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                
                # Calculate today's P&L
                today_pnl = quantity * float(quotes_lookup.get(symbol, {}).get("change", 0))
                today_pnl_percent = (today_pnl / cost_basis * 100) if cost_basis > 0 else 0
                
                position_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "cost_basis": cost_basis,
                    "date_acquired": date_acquired,
                    "current_price": current_price,
                    "current_value": current_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": unrealized_pnl_percent,
                    "today_pnl": today_pnl,
                    "today_pnl_percent": today_pnl_percent
                })
            
            # Convert to DataFrame
            positions_df = pd.DataFrame(position_data)
            
            # Update cache
            self._update_cache("positions", positions_df)
            
            return positions_df
            
        except TradierAPIError as e:
            logger.error(f"Failed to get positions: {str(e)}")
            # Return empty DataFrame
            empty_df = pd.DataFrame(columns=[
                "symbol", "quantity", "cost_basis", "date_acquired", 
                "current_price", "current_value", "unrealized_pnl", 
                "unrealized_pnl_percent", "today_pnl", "today_pnl_percent"
            ])
            return empty_df
    
    def get_order_history(self, days: int = 7) -> pd.DataFrame:
        """
        Get order history for the past specified days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame of order history
        """
        try:
            # Get orders
            orders_response = self.client.get_account_orders()
            orders = orders_response.get("orders", {}).get("order", [])
            
            # Ensure orders is a list
            if orders and not isinstance(orders, list):
                orders = [orders]
            
            if not orders:
                return pd.DataFrame(columns=[
                    "id", "type", "symbol", "side", "quantity", "status", 
                    "price", "avg_fill_price", "exec_quantity", "transaction_date", 
                    "create_date", "class", "duration"
                ])
            
            # Filter orders for the past specified days
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            filtered_orders = [
                order for order in orders 
                if order.get("create_date", "").split("T")[0] >= cutoff_date
            ]
            
            # Convert to DataFrame
            orders_df = pd.DataFrame(filtered_orders)
            
            # Convert date columns to datetime
            if "create_date" in orders_df.columns:
                orders_df["create_date"] = pd.to_datetime(orders_df["create_date"])
            if "transaction_date" in orders_df.columns:
                orders_df["transaction_date"] = pd.to_datetime(orders_df["transaction_date"])
            
            # Convert numeric columns
            numeric_columns = ["quantity", "price", "avg_fill_price", "exec_quantity"]
            for col in numeric_columns:
                if col in orders_df.columns:
                    orders_df[col] = pd.to_numeric(orders_df[col], errors="coerce")
            
            return orders_df
            
        except TradierAPIError as e:
            logger.error(f"Failed to get order history: {str(e)}")
            # Return empty DataFrame
            return pd.DataFrame(columns=[
                "id", "type", "symbol", "side", "quantity", "status", 
                "price", "avg_fill_price", "exec_quantity", "transaction_date", 
                "create_date", "class", "duration"
            ])
    
    def get_historical_data(
        self, 
        symbol: str, 
        interval: str = "daily", 
        start_date: str = None, 
        end_date: str = None, 
        days: int = None
    ) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Data interval ('daily', 'weekly', 'monthly', or 'minute')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            days: Number of days to look back (alternative to start_date)
            
        Returns:
            DataFrame of historical data
        """
        try:
            # Set date range
            if not start_date and days:
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Get historical data
            history_response = self.client.get_historical_quotes(
                symbol=symbol,
                interval=interval,
                start=start_date,
                end=end_date
            )
            
            # Extract history data
            if interval == "minute":
                history_data = history_response.get("series", {}).get("data", [])
            else:
                history_data = history_response.get("history", {}).get("day", [])
            
            # Ensure history_data is a list
            if history_data and not isinstance(history_data, list):
                history_data = [history_data]
            
            if not history_data:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
            
            # Convert to DataFrame
            history_df = pd.DataFrame(history_data)
            
            # Convert date column to datetime
            if "date" in history_df.columns:
                history_df["date"] = pd.to_datetime(history_df["date"])
            
            # Convert numeric columns
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in history_df.columns:
                    history_df[col] = pd.to_numeric(history_df[col], errors="coerce")
            
            return history_df
            
        except TradierAPIError as e:
            logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            # Return empty DataFrame
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: int, 
        tag: str = None
    ) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            symbol: Symbol to trade
            side: Order side ('buy', 'sell', 'buy_to_cover', 'sell_short')
            quantity: Number of shares
            tag: Custom tag for the order
            
        Returns:
            Order confirmation
        """
        try:
            # Generate tag if not provided
            if not tag:
                tag_prefix = self.config["order_tags"].get("source", "trading_bot")
                tag = f"{tag_prefix}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Place order
            order_response = self.client.place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",
                duration=self.config["default_time_in_force"],
                tag=tag
            )
            
            # Clear positions cache
            self._cache["positions"]["timestamp"] = None
            
            return order_response
            
        except TradierAPIError as e:
            logger.error(f"Failed to place market order for {symbol}: {str(e)}")
            raise
    
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: int, 
        price: float, 
        tag: str = None
    ) -> Dict[str, Any]:
        """
        Place a limit order.
        
        Args:
            symbol: Symbol to trade
            side: Order side ('buy', 'sell', 'buy_to_cover', 'sell_short')
            quantity: Number of shares
            price: Limit price
            tag: Custom tag for the order
            
        Returns:
            Order confirmation
        """
        try:
            # Generate tag if not provided
            if not tag:
                tag_prefix = self.config["order_tags"].get("source", "trading_bot")
                tag = f"{tag_prefix}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Place order
            order_response = self.client.place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="limit",
                price=price,
                duration=self.config["default_time_in_force"],
                tag=tag
            )
            
            # Clear positions cache
            self._cache["positions"]["timestamp"] = None
            
            return order_response
            
        except TradierAPIError as e:
            logger.error(f"Failed to place limit order for {symbol}: {str(e)}")
            raise
    
    def place_stop_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: int, 
        stop_price: float, 
        tag: str = None
    ) -> Dict[str, Any]:
        """
        Place a stop order.
        
        Args:
            symbol: Symbol to trade
            side: Order side ('buy', 'sell', 'buy_to_cover', 'sell_short')
            quantity: Number of shares
            stop_price: Stop price
            tag: Custom tag for the order
            
        Returns:
            Order confirmation
        """
        try:
            # Generate tag if not provided
            if not tag:
                tag_prefix = self.config["order_tags"].get("source", "trading_bot")
                tag = f"{tag_prefix}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Place order
            order_response = self.client.place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="stop",
                stop=stop_price,
                duration=self.config["default_time_in_force"],
                tag=tag
            )
            
            # Clear positions cache
            self._cache["positions"]["timestamp"] = None
            
            return order_response
            
        except TradierAPIError as e:
            logger.error(f"Failed to place stop order for {symbol}: {str(e)}")
            raise
    
    def get_option_chain_dataframe(
        self, 
        symbol: str, 
        expiration: str = None,
        strike_count: int = None
    ) -> pd.DataFrame:
        """
        Get option chain data as a pandas DataFrame.
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date (YYYY-MM-DD)
            strike_count: Number of strikes to include (centered around ATM)
            
        Returns:
            DataFrame with option chain data
        """
        try:
            # Get current stock price
            quote_response = self.client.get_quotes(symbol)
            quote = quote_response.get("quotes", {}).get("quote", {})
            current_price = float(quote.get("last", 0))
            
            # Get expirations if not provided
            if not expiration:
                expiration_response = self.client.get_option_expirations(symbol)
                expirations = expiration_response.get("expirations", {}).get("expiration", [])
                
                # Ensure expirations is a list
                if expirations and not isinstance(expirations, list):
                    expirations = [expirations]
                
                if not expirations:
                    logger.warning(f"No option expirations found for {symbol}")
                    return pd.DataFrame()
                
                # Get closest expiration (at least 7 days out)
                min_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                valid_expirations = [
                    exp["date"] for exp in expirations 
                    if exp["date"] >= min_date
                ]
                
                if not valid_expirations:
                    logger.warning(f"No valid option expirations found for {symbol}")
                    return pd.DataFrame()
                
                expiration = min(valid_expirations)
            
            # Get option chain
            chain_response = self.client.get_option_chain(
                symbol=symbol,
                expiration=expiration,
                greeks=True
            )
            
            # Extract options
            options = chain_response.get("options", {}).get("option", [])
            
            # Ensure options is a list
            if options and not isinstance(options, list):
                options = [options]
            
            if not options:
                logger.warning(f"No options found for {symbol} with expiration {expiration}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            options_df = pd.DataFrame(options)
            
            # Filter by strike count if specified
            if strike_count and current_price > 0:
                # Find ATM strike
                options_df["strike_diff"] = abs(pd.to_numeric(options_df["strike"]) - current_price)
                atm_strike = options_df.loc[options_df["strike_diff"].idxmin(), "strike"]
                
                # Get unique strikes
                all_strikes = pd.to_numeric(options_df["strike"].unique())
                all_strikes.sort()
                
                # Find index of ATM strike
                atm_idx = np.where(all_strikes == float(atm_strike))[0][0]
                
                # Calculate range
                half_count = strike_count // 2
                min_idx = max(0, atm_idx - half_count)
                max_idx = min(len(all_strikes) - 1, atm_idx + half_count)
                
                # Get strikes to include
                selected_strikes = all_strikes[min_idx:max_idx+1]
                
                # Filter options
                options_df = options_df[pd.to_numeric(options_df["strike"]).isin(selected_strikes)]
                
                # Drop temporary column
                options_df = options_df.drop("strike_diff", axis=1)
            
            # Convert numeric columns
            numeric_columns = [
                "strike", "bid", "ask", "last", "volume", "open_interest", 
                "delta", "gamma", "theta", "vega", "rho", "implied_volatility"
            ]
            
            for col in numeric_columns:
                if col in options_df.columns:
                    options_df[col] = pd.to_numeric(options_df[col], errors="coerce")
            
            # Add underlying price
            options_df["underlying_price"] = current_price
            
            # Calculate mid price
            if "bid" in options_df.columns and "ask" in options_df.columns:
                options_df["mid"] = (options_df["bid"] + options_df["ask"]) / 2
            
            return options_df
            
        except TradierAPIError as e:
            logger.error(f"Failed to get option chain for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def close_position(
        self, 
        symbol: str, 
        quantity: int = None, 
        order_type: str = "market", 
        price: float = None
    ) -> Dict[str, Any]:
        """
        Close a position (partially or fully).
        
        Args:
            symbol: Symbol to close
            quantity: Number of shares to close (None for all)
            order_type: Order type ('market' or 'limit')
            price: Limit price (required for limit orders)
            
        Returns:
            Order confirmation
        """
        try:
            # Get positions
            positions_df = self.get_positions_dataframe(use_cache=False)
            
            # Check if position exists
            position = positions_df[positions_df["symbol"] == symbol]
            if position.empty:
                logger.warning(f"No position found for {symbol}")
                raise ValueError(f"No position found for {symbol}")
            
            # Get position details
            position_quantity = position["quantity"].values[0]
            
            # If quantity not specified, close entire position
            if quantity is None:
                quantity = abs(position_quantity)
            
            # Check if quantity is valid
            if quantity > abs(position_quantity):
                logger.warning(f"Requested quantity {quantity} exceeds position size {abs(position_quantity)}")
                quantity = abs(position_quantity)
            
            # Determine order side
            if position_quantity > 0:
                side = "sell"
            else:
                side = "buy_to_cover"
            
            # Place order based on type
            if order_type == "market":
                return self.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity
                )
            elif order_type == "limit":
                if price is None:
                    raise ValueError("Price is required for limit orders")
                
                return self.place_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
        except (TradierAPIError, ValueError) as e:
            logger.error(f"Failed to close position for {symbol}: {str(e)}")
            raise
    
    def cancel_pending_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Cancel pending orders for a specific symbol or all symbols.
        
        Args:
            symbol: Symbol to cancel orders for (None for all)
            
        Returns:
            List of cancellation confirmations
        """
        try:
            # Get open orders
            orders_response = self.client.get_account_orders()
            orders = orders_response.get("orders", {}).get("order", [])
            
            # Ensure orders is a list
            if orders and not isinstance(orders, list):
                orders = [orders]
            
            # Filter for open orders
            open_orders = [order for order in orders if order.get("status") == "open"]
            
            # Filter by symbol if specified
            if symbol:
                open_orders = [order for order in open_orders if order.get("symbol") == symbol]
            
            if not open_orders:
                logger.info(f"No open orders found{' for ' + symbol if symbol else ''}")
                return []
            
            # Cancel orders
            cancellations = []
            for order in open_orders:
                order_id = order["id"]
                try:
                    cancel_response = self.client.cancel_order(order_id)
                    cancellations.append({
                        "order_id": order_id,
                        "symbol": order.get("symbol"),
                        "status": "cancelled",
                        "response": cancel_response
                    })
                    logger.info(f"Cancelled order {order_id} for {order.get('symbol')}")
                except TradierAPIError as e:
                    logger.error(f"Failed to cancel order {order_id}: {str(e)}")
                    cancellations.append({
                        "order_id": order_id,
                        "symbol": order.get("symbol"),
                        "status": "error",
                        "error": str(e)
                    })
            
            return cancellations
            
        except TradierAPIError as e:
            logger.error(f"Failed to cancel pending orders: {str(e)}")
            raise
    
    def get_market_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get market status information.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Market status information
        """
        # Check cache first
        if use_cache and self._is_cache_valid("market_status"):
            logger.debug("Using cached market status")
            return self._cache["market_status"]["data"]
        
        try:
            # Get market clock
            clock_response = self.client.get_clock()
            clock = clock_response.get("clock", {})
            
            # Format the data
            market_status = {
                "is_open": clock.get("state") == "open",
                "timestamp": clock.get("timestamp"),
                "next_open": clock.get("next_open"),
                "next_close": clock.get("next_close"),
                "description": clock.get("description")
            }
            
            # Update cache
            self._update_cache("market_status", market_status)
            
            return market_status
            
        except TradierAPIError as e:
            logger.error(f"Failed to get market status: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example: Initialize client
    try:
        tradier = TradierIntegration(
            api_key=os.environ.get("TRADIER_API_KEY", "demo_key"),
            account_id=os.environ.get("TRADIER_ACCOUNT_ID", "demo_account"),
            use_sandbox=True
        )
        
        # Example: Check market status
        market_status = tradier.get_market_status()
        print(f"Market is {'open' if market_status['is_open'] else 'closed'}")
        
        # Example: Get account summary
        account_summary = tradier.get_account_summary()
        print(f"Account value: ${account_summary['balances']['total_equity']}")
        print(f"Positions: {account_summary['positions_count']}")
        print(f"Today's P&L: ${account_summary['today_pnl']} ({account_summary['today_pnl_percent']:.2f}%)")
        
        # Example: Get positions
        positions = tradier.get_positions_dataframe()
        if not positions.empty:
            print("\nCurrent Positions:")
            print(positions[["symbol", "quantity", "cost_basis", "current_value", "unrealized_pnl", "unrealized_pnl_percent"]])
        
        # Example: Get historical data for SPY
        spy_data = tradier.get_historical_data("SPY", days=30)
        print(f"\nSPY Historical Data ({len(spy_data)} days):")
        print(spy_data.tail())
        
        # Example: Get option chain for AAPL
        aapl_options = tradier.get_option_chain_dataframe("AAPL", strike_count=5)
        if not aapl_options.empty:
            print("\nAAPL Option Chain:")
            print(aapl_options[["option_symbol", "strike", "bid", "ask", "delta", "implied_volatility"]])
        
    except Exception as e:
        print(f"Error: {str(e)}") 