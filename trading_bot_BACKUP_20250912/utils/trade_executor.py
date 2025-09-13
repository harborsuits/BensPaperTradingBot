"""
Trade Executor

This module provides connectivity to trading platforms to execute trades based on 
strategy allocations from the rotation system.
"""

import os
import json
import logging
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests

# Configure logging
logger = logging.getLogger("trade_executor")

class TradeExecutor:
    """Base class for trade execution integrations."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_secret: Optional[str] = None,
                config_path: Optional[str] = None):
        """
        Initialize the trade executor.
        
        Args:
            api_key: API key for the trading platform
            api_secret: API secret for the trading platform
            config_path: Path to configuration file
        """
        self.api_key = api_key or os.getenv("TRADING_API_KEY")
        self.api_secret = api_secret or os.getenv("TRADING_API_SECRET")
        
        # Load configuration if path provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Track execution history
        self.execution_history = []
    
    def execute_allocation_change(self, 
                                 strategy_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute allocation changes by placing necessary trades.
        
        Args:
            strategy_allocations: Dictionary mapping strategies to allocation percentages
            
        Returns:
            Dictionary with execution results
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def get_account_status(self) -> Dict[str, Any]:
        """
        Get current account status.
        
        Returns:
            Dictionary with account information
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def _log_execution(self, 
                     strategy: str, 
                     action: str, 
                     amount: float, 
                     details: Dict[str, Any]) -> None:
        """
        Log execution details.
        
        Args:
            strategy: Strategy name
            action: Action taken (buy, sell)
            amount: Amount executed
            details: Additional execution details
        """
        execution = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "action": action,
            "amount": amount,
            "details": details
        }
        
        self.execution_history.append(execution)
        
        logger.info(f"Executed {action} for {strategy}: ${amount:.2f}")


class MockTradeExecutor(TradeExecutor):
    """Mock implementation for testing trade execution without real orders."""
    
    def __init__(self, 
                config_path: Optional[str] = None,
                initial_cash: float = 100000.0):
        """
        Initialize the mock trade executor.
        
        Args:
            config_path: Path to configuration file
            initial_cash: Initial cash balance
        """
        super().__init__(api_key="mock_key", api_secret="mock_secret", config_path=config_path)
        
        # Initialize mock account
        self.portfolio = {"cash": initial_cash, "positions": {}}
        
        # Strategy to instrument mapping
        self.strategy_instruments = self.config.get("strategy_instruments", {})
        
        # If no mapping in config, use default mapping
        if not self.strategy_instruments:
            self.strategy_instruments = {
                "momentum": ["AAPL", "MSFT", "GOOGL"],
                "mean_reversion": ["SPY", "QQQ"],
                "trend_following": ["TSLA", "NVDA", "AMD"],
                "breakout_swing": ["AMZN", "NFLX", "FB"],
                "volatility_breakout": ["VXX", "UVXY"],
                "option_spreads": ["SPX_CALL_SPREAD", "QQQ_PUT_SPREAD"]
            }
        
        # Mock prices for instruments
        self.instrument_prices = {}
        for instruments in self.strategy_instruments.values():
            for instrument in instruments:
                # Generate a random price between 10 and 1000
                price = hash(instrument) % 990 + 10
                self.instrument_prices[instrument] = price
        
        logger.info(f"Mock trade executor initialized with ${initial_cash:.2f}")
    
    def execute_allocation_change(self, 
                                 strategy_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute allocation changes in the mock environment.
        
        Args:
            strategy_allocations: Dictionary mapping strategies to allocation percentages
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing allocation change: {strategy_allocations}")
        
        # Calculate target value for each strategy
        total_portfolio_value = self._get_total_portfolio_value()
        target_values = {
            strategy: (allocation / 100) * total_portfolio_value
            for strategy, allocation in strategy_allocations.items()
        }
        
        # Calculate current value of each strategy
        current_values = self._get_strategy_values()
        
        # Determine rebalancing actions
        actions = []
        for strategy, target_value in target_values.items():
            current_value = current_values.get(strategy, 0)
            difference = target_value - current_value
            
            if abs(difference) < 1:  # Less than $1 difference, ignore
                continue
            
            if difference > 0:
                # Need to buy more of this strategy
                actions.append({
                    "strategy": strategy,
                    "action": "buy",
                    "amount": difference
                })
            else:
                # Need to sell some of this strategy
                actions.append({
                    "strategy": strategy,
                    "action": "sell",
                    "amount": abs(difference)
                })
        
        # Execute actions
        execution_results = []
        for action in actions:
            strategy = action["strategy"]
            action_type = action["action"]
            amount = action["amount"]
            
            # Get instruments for this strategy
            instruments = self.strategy_instruments.get(strategy, [])
            if not instruments:
                logger.warning(f"No instruments defined for strategy {strategy}")
                continue
            
            # Divide amount among instruments
            amount_per_instrument = amount / len(instruments)
            
            for instrument in instruments:
                price = self.instrument_prices.get(instrument, 10)
                quantity = amount_per_instrument / price
                
                if action_type == "buy":
                    self._execute_buy(strategy, instrument, quantity, price)
                else:
                    self._execute_sell(strategy, instrument, quantity, price)
                
                execution_results.append({
                    "strategy": strategy,
                    "instrument": instrument,
                    "action": action_type,
                    "quantity": quantity,
                    "price": price,
                    "value": quantity * price
                })
        
        # Return execution summary
        return {
            "timestamp": datetime.now().isoformat(),
            "total_value": total_portfolio_value,
            "target_allocations": strategy_allocations,
            "actions_taken": len(execution_results),
            "execution_details": execution_results
        }
    
    def get_account_status(self) -> Dict[str, Any]:
        """
        Get current account status from the mock environment.
        
        Returns:
            Dictionary with account information
        """
        total_value = self._get_total_portfolio_value()
        strategy_values = self._get_strategy_values()
        
        # Calculate current allocations
        strategy_allocations = {}
        for strategy, value in strategy_values.items():
            allocation = (value / total_value) * 100 if total_value > 0 else 0
            strategy_allocations[strategy] = allocation
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cash_balance": self.portfolio["cash"],
            "positions_value": total_value - self.portfolio["cash"],
            "total_value": total_value,
            "positions": self.portfolio["positions"],
            "strategy_allocations": strategy_allocations
        }
    
    def _execute_buy(self, 
                   strategy: str, 
                   instrument: str, 
                   quantity: float, 
                   price: float) -> None:
        """
        Execute a buy order in the mock environment.
        
        Args:
            strategy: Strategy name
            instrument: Instrument to buy
            quantity: Quantity to buy
            price: Price per unit
        """
        value = quantity * price
        
        # Check if we have enough cash
        if value > self.portfolio["cash"]:
            # If not enough cash, adjust quantity
            quantity = self.portfolio["cash"] / price
            value = quantity * price
        
        # Update cash balance
        self.portfolio["cash"] -= value
        
        # Update position
        if instrument not in self.portfolio["positions"]:
            self.portfolio["positions"][instrument] = {
                "quantity": 0,
                "strategy": strategy
            }
        
        self.portfolio["positions"][instrument]["quantity"] += quantity
        
        # Log execution
        self._log_execution(
            strategy=strategy,
            action="buy",
            amount=value,
            details={
                "instrument": instrument,
                "quantity": quantity,
                "price": price
            }
        )
    
    def _execute_sell(self, 
                    strategy: str, 
                    instrument: str, 
                    quantity: float, 
                    price: float) -> None:
        """
        Execute a sell order in the mock environment.
        
        Args:
            strategy: Strategy name
            instrument: Instrument to sell
            quantity: Quantity to sell
            price: Price per unit
        """
        current_quantity = 0
        if instrument in self.portfolio["positions"]:
            current_quantity = self.portfolio["positions"][instrument]["quantity"]
        
        # Adjust quantity if needed
        quantity = min(quantity, current_quantity)
        value = quantity * price
        
        # Update cash balance
        self.portfolio["cash"] += value
        
        # Update position
        if instrument in self.portfolio["positions"]:
            self.portfolio["positions"][instrument]["quantity"] -= quantity
            
            # Remove position if quantity is zero
            if self.portfolio["positions"][instrument]["quantity"] <= 0:
                del self.portfolio["positions"][instrument]
        
        # Log execution
        self._log_execution(
            strategy=strategy,
            action="sell",
            amount=value,
            details={
                "instrument": instrument,
                "quantity": quantity,
                "price": price
            }
        )
    
    def _get_total_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value
        """
        positions_value = 0
        for instrument, position in self.portfolio["positions"].items():
            price = self.instrument_prices.get(instrument, 0)
            positions_value += position["quantity"] * price
        
        return self.portfolio["cash"] + positions_value
    
    def _get_strategy_values(self) -> Dict[str, float]:
        """
        Calculate current value of each strategy.
        
        Returns:
            Dictionary mapping strategies to their current values
        """
        strategy_values = {}
        
        # Get value of positions by strategy
        for instrument, position in self.portfolio["positions"].items():
            strategy = position["strategy"]
            price = self.instrument_prices.get(instrument, 0)
            value = position["quantity"] * price
            
            if strategy not in strategy_values:
                strategy_values[strategy] = 0
            
            strategy_values[strategy] += value
        
        return strategy_values


class AlpacaTradeExecutor(TradeExecutor):
    """
    Integration with Alpaca trading API for live trading.
    This is a simplified implementation - a real one would handle more edge cases.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_secret: Optional[str] = None,
                config_path: Optional[str] = None,
                paper_trading: bool = True):
        """
        Initialize the Alpaca trade executor.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            config_path: Path to configuration file
            paper_trading: Whether to use paper trading
        """
        super().__init__(api_key, api_secret, config_path)
        
        # API base URL
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        # Strategy to instrument mapping
        self.strategy_instruments = self.config.get("strategy_instruments", {})
        
        # Check if API credentials are available
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API credentials not provided")
            raise ValueError("API key and secret are required for Alpaca integration")
        
        logger.info(f"Alpaca trade executor initialized (paper_trading={paper_trading})")
    
    def execute_allocation_change(self, 
                                 strategy_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute allocation changes using Alpaca API.
        
        Args:
            strategy_allocations: Dictionary mapping strategies to allocation percentages
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing allocation change via Alpaca: {strategy_allocations}")
        
        # Get account information
        account = self.get_account_status()
        total_equity = float(account.get("total_equity", 0))
        
        if total_equity <= 0:
            logger.error("Account equity is zero or negative")
            return {
                "success": False,
                "error": "Insufficient account equity"
            }
        
        # Get current positions
        positions = self._get_positions()
        
        # Calculate target allocation for each instrument
        target_allocations = {}
        for strategy, allocation in strategy_allocations.items():
            instruments = self.strategy_instruments.get(strategy, [])
            
            # Skip if no instruments for this strategy
            if not instruments:
                logger.warning(f"No instruments defined for strategy {strategy}")
                continue
            
            # Divide allocation evenly among instruments
            allocation_per_instrument = allocation / len(instruments)
            
            for instrument in instruments:
                target_allocations[instrument] = {
                    "strategy": strategy,
                    "allocation": allocation_per_instrument,
                    "target_value": (allocation_per_instrument / 100) * total_equity
                }
        
        # Calculate rebalancing actions
        actions = []
        for symbol, target in target_allocations.items():
            current_position = positions.get(symbol, {"market_value": 0})
            current_value = float(current_position.get("market_value", 0))
            target_value = target["target_value"]
            
            # Calculate difference
            difference = target_value - current_value
            
            # Skip small differences
            if abs(difference) < 10:  # Less than $10 difference
                continue
            
            # Determine action
            if difference > 0:
                actions.append({
                    "symbol": symbol,
                    "strategy": target["strategy"],
                    "action": "buy",
                    "amount": difference
                })
            else:
                actions.append({
                    "symbol": symbol,
                    "strategy": target["strategy"],
                    "action": "sell",
                    "amount": abs(difference)
                })
        
        # Execute orders
        execution_results = []
        for action in actions:
            symbol = action["symbol"]
            strategy = action["strategy"]
            action_type = action["action"]
            amount = action["amount"]
            
            try:
                # Get current price
                price_data = self._get_price(symbol)
                current_price = float(price_data.get("last", 0))
                
                if current_price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                
                # Calculate quantity
                quantity = amount / current_price
                
                # Place order
                if action_type == "buy":
                    order = self._place_order(symbol, quantity, "buy")
                else:
                    order = self._place_order(symbol, quantity, "sell")
                
                # Record result
                execution_results.append({
                    "strategy": strategy,
                    "symbol": symbol,
                    "action": action_type,
                    "quantity": quantity,
                    "price": current_price,
                    "value": quantity * current_price,
                    "order_id": order.get("id")
                })
                
                # Log execution
                self._log_execution(
                    strategy=strategy,
                    action=action_type,
                    amount=quantity * current_price,
                    details={
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": current_price,
                        "order_id": order.get("id")
                    }
                )
                
            except Exception as e:
                logger.error(f"Error executing {action_type} for {symbol}: {str(e)}")
                execution_results.append({
                    "strategy": strategy,
                    "symbol": symbol,
                    "action": action_type,
                    "error": str(e)
                })
        
        # Return execution summary
        return {
            "timestamp": datetime.now().isoformat(),
            "total_equity": total_equity,
            "target_allocations": strategy_allocations,
            "actions_taken": len(execution_results),
            "execution_details": execution_results
        }
    
    def get_account_status(self) -> Dict[str, Any]:
        """
        Get current account status from Alpaca.
        
        Returns:
            Dictionary with account information
        """
        endpoint = f"{self.base_url}/v2/account"
        headers = self._get_headers()
        
        try:
            response = requests.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting account status: {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            logger.error(f"Error getting account status: {str(e)}")
            return {"error": str(e)}
    
    def _get_positions(self) -> Dict[str, Any]:
        """
        Get current positions from Alpaca.
        
        Returns:
            Dictionary mapping symbols to position information
        """
        endpoint = f"{self.base_url}/v2/positions"
        headers = self._get_headers()
        
        try:
            response = requests.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                positions = response.json()
                return {position["symbol"]: position for position in positions}
            else:
                logger.error(f"Error getting positions: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    def _get_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Dictionary with price information
        """
        endpoint = f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"
        headers = self._get_headers()
        
        try:
            response = requests.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting price for {symbol}: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return {}
    
    def _place_order(self, 
                   symbol: str, 
                   quantity: float, 
                   side: str) -> Dict[str, Any]:
        """
        Place an order via Alpaca API.
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade
            side: Order side (buy or sell)
            
        Returns:
            Dictionary with order information
        """
        endpoint = f"{self.base_url}/v2/orders"
        headers = self._get_headers()
        
        # Round quantity to nearest whole number
        quantity = int(quantity)
        
        # Ensure minimum quantity
        quantity = max(1, quantity)
        
        # Create order payload
        payload = {
            "symbol": symbol,
            "qty": quantity,
            "side": side,
            "type": "market",
            "time_in_force": "day"
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error placing order: {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {"error": str(e)}
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers for Alpaca API.
        
        Returns:
            Dictionary with request headers
        """
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }


class AllocationExecutor:
    """
    High-level executor that uses strategy allocations from the rotator to
    execute trades using a configured trade executor.
    """
    
    def __init__(self, 
                 strategy_rotator,
                 trade_executor,
                 monitor=None):
        """
        Initialize the allocation executor.
        
        Args:
            strategy_rotator: IntegratedStrategyRotator instance
            trade_executor: TradeExecutor instance
            monitor: Optional StrategyMonitor instance
        """
        self.rotator = strategy_rotator
        self.executor = trade_executor
        self.monitor = monitor
        
        logger.info("Allocation executor initialized")
    
    def execute_current_allocations(self) -> Dict[str, Any]:
        """
        Execute the current allocations from the rotator.
        
        Returns:
            Dictionary with execution results
        """
        # Get current allocations
        allocations = self.rotator.get_current_allocations()
        
        # Execute allocation change
        results = self.executor.execute_allocation_change(allocations)
        
        # Update monitor if available
        if self.monitor:
            self.monitor.update_metrics_from_rotator()
        
        return results
    
    def rotate_and_execute(self, 
                         market_context: Optional[Dict[str, Any]] = None,
                         force_rotation: bool = False) -> Dict[str, Any]:
        """
        Perform strategy rotation and execute the new allocations.
        
        Args:
            market_context: Optional market context data
            force_rotation: Whether to force rotation regardless of schedule
            
        Returns:
            Dictionary with rotation and execution results
        """
        # Perform rotation
        rotation_result = self.rotator.rotate_strategies(
            market_context=market_context,
            force_rotation=force_rotation
        )
        
        # Update monitor if available
        if self.monitor:
            self.monitor.monitor_rotation(rotation_result)
        
        # Check if rotation was performed
        if not rotation_result.get("rotated", False):
            logger.info(f"No rotation performed: {rotation_result.get('message', 'No reason provided')}")
            return rotation_result
        
        # Get the new allocations
        new_allocations = rotation_result.get("new_allocations", {})
        
        # Execute allocation change
        execution_results = self.executor.execute_allocation_change(new_allocations)
        
        # Combine results
        results = {
            "timestamp": datetime.now().isoformat(),
            "rotation": rotation_result,
            "execution": execution_results
        }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of allocations and account.
        
        Returns:
            Dictionary with status information
        """
        # Get account status
        account_status = self.executor.get_account_status()
        
        # Get rotator allocations
        rotator_allocations = self.rotator.get_current_allocations()
        
        # Get actual allocations
        actual_allocations = account_status.get("strategy_allocations", {})
        
        # Calculate differences
        allocation_diff = {}
        for strategy, target in rotator_allocations.items():
            actual = actual_allocations.get(strategy, 0)
            allocation_diff[strategy] = {
                "target": target,
                "actual": actual,
                "difference": actual - target
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "account_status": account_status,
            "rotator_allocations": rotator_allocations,
            "allocation_differences": allocation_diff
        } 