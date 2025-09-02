"""
Simulation environment for EvoTrader.

This module provides a controlled environment for simulating trading bots
with realistic market conditions, including slippage, fees, and latency.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Type
import random
from datetime import datetime, timedelta

from ..core.trading_bot import TradingBot, Order, OrderSide, OrderType
from ..data.market_data_provider import MarketDataProvider, RandomWalkDataProvider


class SimulationEnvironment:
    """
    Simulates a trading environment with realistic market conditions.
    
    This class handles the mechanics of running trading bots in a simulated
    market environment with configurable parameters for realism.
    """
    
    def __init__(self, 
                 data_provider: MarketDataProvider,
                 initial_balance: float = 1000.0,
                 trading_fee: float = 0.001,  # 0.1% fee
                 slippage_model: str = "basic",
                 latency_ms: int = 10,
                 max_days: int = 30,
                 record_interval: int = 1):
        """
        Initialize simulation environment.
        
        Args:
            data_provider: Provider of market data
            initial_balance: Starting balance for each bot
            trading_fee: Fee as a proportion of trade value (0.001 = 0.1%)
            slippage_model: Model for price slippage ('none', 'basic', 'volume')
            latency_ms: Simulated latency in milliseconds
            max_days: Maximum number of days to simulate
            record_interval: Interval (in days) for recording detailed metrics
        """
        self.logger = logging.getLogger(__name__)
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.slippage_model = slippage_model
        self.latency_ms = latency_ms
        self.max_days = max_days
        self.record_interval = record_interval
        
        # Initialize simulation state
        self.current_day = 0
        self.available_symbols = self.data_provider.get_symbols()
        self.current_market_data = None
        
        # Metrics tracking
        self.metrics = {
            "total_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "simulation_start_time": None,
            "simulation_end_time": None,
            "slippage_cost": 0.0,
        }
        
        self.logger.info(f"Simulation environment initialized with {len(self.available_symbols)} symbols")
    
    def register_bot(self, bot: TradingBot) -> None:
        """
        Register a bot with the simulation.
        
        Args:
            bot: Trading bot to register
        """
        bot.initialize(self.initial_balance)
        self.logger.info(f"Registered bot {bot.bot_id} with initial balance {self.initial_balance}")
    
    def run_simulation(self, bots: List[TradingBot]) -> Dict[str, Any]:
        """
        Run a full simulation with the given bots.
        
        Args:
            bots: List of trading bots to simulate
            
        Returns:
            Dict of simulation results and metrics
        """
        self.metrics["simulation_start_time"] = time.time()
        
        # Initialize all bots
        for bot in bots:
            self.register_bot(bot)
            
        self.logger.info(f"Starting simulation with {len(bots)} bots for {self.max_days} days")
        
        # Run simulation for each day
        for day in range(self.max_days):
            self.current_day = day
            self.step_simulation(bots)
            
            # Record metrics at intervals
            if day % self.record_interval == 0:
                self._record_metrics(bots, day)
                
        self.metrics["simulation_end_time"] = time.time()
        
        # Finalize all bots
        results = {}
        for bot in bots:
            bot_results = bot.finalize()
            results[bot.bot_id] = bot_results
            
        # Compile overall statistics
        overall_stats = self._compile_simulation_stats(bots, results)
        results["overall"] = overall_stats
        
        self.logger.info(f"Simulation completed in {self.metrics['simulation_end_time'] - self.metrics['simulation_start_time']:.2f} seconds")
        return results
    
    def step_simulation(self, bots: List[TradingBot]) -> None:
        """
        Advance simulation by one day.
        
        Args:
            bots: List of trading bots to update
        """
        # Get market data for current day
        self.current_market_data = self.data_provider.get_data(self.current_day)
        
        # Process each bot
        for bot in bots:
            # Process pending orders first (with slippage and fees)
            self._process_orders(bot)
            
            # Update bot with current market data
            bot.on_data(self.current_market_data)
            
            # Update equity calculations
            bot.update_equity(self.current_market_data)
    
    def _process_orders(self, bot: TradingBot) -> None:
        """
        Process pending orders for a bot.
        
        Args:
            bot: Trading bot whose orders to process
        """
        # Make a copy of open orders to avoid modification during iteration
        open_orders = list(bot.open_orders.values())
        
        for order in open_orders:
            # Check if symbol exists in current data
            if order.symbol not in self.current_market_data:
                self.logger.warning(f"Symbol {order.symbol} not found in market data, cancelling order")
                bot.cancel_order(order.id)
                continue
                
            # Get current price for the symbol
            market_data = self.current_market_data[order.symbol]
            current_price = market_data["price"]
            
            # Process different order types
            if order.order_type == OrderType.MARKET:
                # Apply slippage to market orders
                fill_price = self._apply_slippage(current_price, order.side, order.quantity)
                
                # Calculate fees
                trade_value = order.quantity * fill_price
                fee_amount = trade_value * self.trading_fee
                
                # Execute the order
                bot.handle_order_fill(order, fill_price, time.time(), fee_amount)
                
                # Update metrics
                self.metrics["total_trades"] += 1
                self.metrics["total_volume"] += trade_value
                self.metrics["total_fees"] += fee_amount
                
            elif order.order_type == OrderType.LIMIT:
                # For limit orders, check if price conditions are met
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    
                    # Calculate fees
                    trade_value = order.quantity * order.price
                    fee_amount = trade_value * self.trading_fee
                    
                    # Execute the order at limit price (no slippage)
                    bot.handle_order_fill(order, order.price, time.time(), fee_amount)
                    
                    # Update metrics
                    self.metrics["total_trades"] += 1
                    self.metrics["total_volume"] += trade_value
                    self.metrics["total_fees"] += fee_amount
                    
            elif order.order_type == OrderType.STOP:
                # For stop orders, check if stop price is triggered
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
                   (order.side == OrderSide.SELL and current_price <= order.stop_price):
                    
                    # Convert to market order once triggered
                    fill_price = self._apply_slippage(current_price, order.side, order.quantity)
                    
                    # Calculate fees
                    trade_value = order.quantity * fill_price
                    fee_amount = trade_value * self.trading_fee
                    
                    # Execute the order
                    bot.handle_order_fill(order, fill_price, time.time(), fee_amount)
                    
                    # Update metrics
                    self.metrics["total_trades"] += 1
                    self.metrics["total_volume"] += trade_value
                    self.metrics["total_fees"] += fee_amount
    
    def _apply_slippage(self, price: float, side: OrderSide, quantity: float) -> float:
        """
        Apply slippage model to price.
        
        Args:
            price: Current market price
            side: Order side (buy/sell)
            quantity: Order quantity
            
        Returns:
            Adjusted price with slippage
        """
        if self.slippage_model == "none":
            return price
            
        elif self.slippage_model == "basic":
            # Basic slippage model: 0.1% - 0.3% price impact based on side
            slippage_factor = 0.001 + (random.random() * 0.002)  # 0.1% - 0.3%
            
            if side == OrderSide.BUY:
                # Buys executed at higher prices
                adjusted_price = price * (1 + slippage_factor)
                slippage_cost = (adjusted_price - price) * quantity
            else:
                # Sells executed at lower prices
                adjusted_price = price * (1 - slippage_factor)
                slippage_cost = (price - adjusted_price) * quantity
                
            # Track slippage cost
            self.metrics["slippage_cost"] += slippage_cost
            return adjusted_price
            
        elif self.slippage_model == "volume":
            # Volume-based slippage: more slippage for larger orders
            # Assuming average daily volume is roughly 1000 * price
            estimated_volume = 1000 * price
            volume_ratio = min(1.0, quantity / estimated_volume)
            
            # Slippage increases non-linearly with volume ratio
            slippage_factor = 0.001 * (1 + (volume_ratio * 10)**2)
            
            if side == OrderSide.BUY:
                adjusted_price = price * (1 + slippage_factor)
                slippage_cost = (adjusted_price - price) * quantity
            else:
                adjusted_price = price * (1 - slippage_factor)
                slippage_cost = (price - adjusted_price) * quantity
                
            # Track slippage cost
            self.metrics["slippage_cost"] += slippage_cost
            return adjusted_price
        
        # Default
        return price
    
    def _record_metrics(self, bots: List[TradingBot], day: int) -> None:
        """
        Record metrics for the current day.
        
        Args:
            bots: List of trading bots
            day: Current simulation day
        """
        # Log basic statistics
        alive_bots = [bot for bot in bots if bot.balance > 0]
        self.logger.info(f"Day {day}: {len(alive_bots)}/{len(bots)} bots active")
        
        total_equity = sum(bot.equity for bot in bots)
        avg_equity = total_equity / len(bots) if bots else 0
        
        self.logger.info(f"Day {day}: Avg equity ${avg_equity:.2f}, Total equity ${total_equity:.2f}")
        
        # Could save detailed metrics to a file for later analysis
    
    def _compile_simulation_stats(self, 
                               bots: List[TradingBot], 
                               bot_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compile overall statistics from bot results.
        
        Args:
            bots: List of trading bots
            bot_results: Results from each bot's finalize method
            
        Returns:
            Overall simulation statistics
        """
        # Calculate overall statistics
        final_equities = [bot.equity for bot in bots]
        
        return {
            "initial_balance": self.initial_balance,
            "total_starting_capital": self.initial_balance * len(bots),
            "final_total_equity": sum(final_equities),
            "mean_final_equity": sum(final_equities) / len(bots) if bots else 0,
            "median_final_equity": sorted(final_equities)[len(final_equities)//2] if final_equities else 0,
            "max_final_equity": max(final_equities) if final_equities else 0,
            "min_final_equity": min(final_equities) if final_equities else 0,
            "bot_count": len(bots),
            "simulation_days": self.current_day + 1,
            "total_trades": self.metrics["total_trades"],
            "total_volume": self.metrics["total_volume"],
            "total_fees": self.metrics["total_fees"],
            "total_slippage_cost": self.metrics["slippage_cost"],
            "simulation_duration": self.metrics["simulation_end_time"] - self.metrics["simulation_start_time"]
            if self.metrics["simulation_end_time"] else 0,
        }
