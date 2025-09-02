#!/usr/bin/env python3
"""
Trading Simulator for backtesting and paper trading strategies.

This module provides a flexible simulation environment for testing trading strategies
under various market conditions. It supports multiple asset classes, custom market
scenarios, and comprehensive performance reporting.
"""

import logging
import datetime
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from enum import Enum, auto
import uuid
from dataclasses import dataclass

from trading_bot.risk_manager import RiskManager
from trading_bot.multi_asset_adapter import MultiAssetAdapter
from trading_bot.data_providers.base_provider import DataProvider
from trading_bot.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown

# Configure logging
logger = logging.getLogger(__name__)

class SimulationMode(Enum):
    """Simulation execution modes"""
    BACKTEST = auto()
    PAPER_TRADING = auto()
    HYBRID = auto()  # Uses historical data with real-time overlay

class MarketScenario(Enum):
    """Pre-defined market scenarios for simulation"""
    NORMAL = auto()
    HIGH_VOLATILITY = auto()
    LOW_LIQUIDITY = auto()
    FLASH_CRASH = auto()
    SIDEWAYS = auto()
    CUSTOM = auto()  # For user-defined scenarios

@dataclass
class SimulationConfig:
    """Configuration for simulation runs"""
    mode: SimulationMode
    start_date: datetime.datetime
    end_date: Optional[datetime.datetime] = None
    initial_capital: float = 100000.0
    symbols: List[str] = None
    market_scenario: MarketScenario = MarketScenario.NORMAL
    data_frequency: str = "1min"  # "1min", "5min", "1h", "1d"
    slippage_model: str = "fixed"  # "fixed", "percentage", "market_impact"
    slippage_value: float = 0.0
    commission_model: str = "fixed"  # "fixed", "percentage"
    commission_value: float = 0.0
    enable_fractional_shares: bool = True
    random_seed: Optional[int] = None

class TradingSimulator:
    """
    Trading simulator for backtesting and paper trading.
    
    This class provides a comprehensive environment for simulating trading strategies
    with configurable market conditions, risk parameters, and performance tracking.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        data_provider: DataProvider,
        risk_manager: Optional[RiskManager] = None,
        strategy_factory: Optional[Callable] = None,
        output_dir: str = "simulation_results"
    ):
        """
        Initialize the trading simulator.
        
        Args:
            config: Configuration parameters for the simulation
            data_provider: Data provider for market data
            risk_manager: Optional risk manager for position sizing and risk control
            strategy_factory: Optional factory function to create strategy instances
            output_dir: Directory to save simulation results
        """
        self.config = config
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        self.strategy_factory = strategy_factory
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            
        # Initialize simulation state
        self.current_time = config.start_date
        self.portfolio = self._initialize_portfolio()
        self.trades = []
        self.portfolio_history = []
        self.market_data_cache = {}
        self.performance_metrics = {}
        self.simulation_id = str(uuid.uuid4())
        
        # Initialize multi-asset adapter if not provided in strategy factory
        self.multi_asset_adapter = MultiAssetAdapter(data_provider=data_provider)
        
        # Initialize strategy instances
        self.strategies = self._initialize_strategies()
        
        logger.info(f"Initialized TradingSimulator with ID: {self.simulation_id}")
        logger.info(f"Mode: {config.mode.name}, Symbols: {config.symbols}, Period: {config.start_date} to {config.end_date}")
    
    def _initialize_portfolio(self) -> Dict:
        """Initialize portfolio with starting capital"""
        return {
            "cash": self.config.initial_capital,
            "positions": {},
            "total_value": self.config.initial_capital,
            "margin_used": 0.0,
            "margin_available": self.config.initial_capital
        }
    
    def _initialize_strategies(self) -> Dict:
        """Initialize trading strategies using the provided factory"""
        strategies = {}
        
        if self.strategy_factory is not None:
            for symbol in self.config.symbols:
                strategies[symbol] = self.strategy_factory(
                    symbol=symbol,
                    data_provider=self.data_provider,
                    risk_manager=self.risk_manager
                )
                logger.info(f"Initialized strategy for {symbol}")
        
        return strategies
    
    def run_simulation(self) -> Dict:
        """
        Run the trading simulation according to the configured parameters.
        
        Returns:
            Dict: Simulation results including performance metrics
        """
        logger.info(f"Starting simulation run {self.simulation_id}")
        
        if self.config.mode == SimulationMode.BACKTEST:
            results = self._run_backtest()
        elif self.config.mode == SimulationMode.PAPER_TRADING:
            results = self._run_paper_trading()
        elif self.config.mode == SimulationMode.HYBRID:
            results = self._run_hybrid_simulation()
        else:
            raise ValueError(f"Unsupported simulation mode: {self.config.mode}")
        
        # Save results
        self._save_simulation_results(results)
        
        logger.info(f"Completed simulation run {self.simulation_id}")
        return results
    
    def _run_backtest(self) -> Dict:
        """Run historical backtest simulation"""
        logger.info("Running backtest simulation")
        
        # Load all required historical data
        self._preload_market_data()
        
        # Iterate through each time step
        time_index = self._generate_time_index()
        for timestamp in time_index:
            self.current_time = timestamp
            
            # Update portfolio with latest market data
            self._update_portfolio_values()
            
            # Execute strategy logic for each symbol
            for symbol in self.config.symbols:
                if symbol in self.strategies:
                    signals = self.strategies[symbol].generate_signals(timestamp)
                    self._process_signals(symbol, signals)
            
            # Record portfolio state
            self._record_portfolio_snapshot()
            
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        return self._prepare_simulation_results()
    
    def _run_paper_trading(self) -> Dict:
        """Run paper trading simulation with real-time data"""
        logger.info("Running paper trading simulation")
        # Implementation would vary based on real-time data handling
        # This is a placeholder for the basic structure
        
        end_time = self.config.end_date or datetime.datetime.now()
        
        while self.current_time < end_time:
            # Get latest market data
            market_data = self._fetch_realtime_market_data()
            
            # Update portfolio with latest market data
            self._update_portfolio_values()
            
            # Execute strategy logic for each symbol
            for symbol in self.config.symbols:
                if symbol in self.strategies:
                    signals = self.strategies[symbol].generate_signals(self.current_time)
                    self._process_signals(symbol, signals)
            
            # Record portfolio state
            self._record_portfolio_snapshot()
            
            # Advance simulation time
            self.current_time += self._get_time_increment()
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        return self._prepare_simulation_results()
    
    def _run_hybrid_simulation(self) -> Dict:
        """Run hybrid simulation with historical data and real-time overlays"""
        logger.info("Running hybrid simulation")
        # Implementation would combine historical and synthetic real-time data
        # This is a placeholder for the basic structure
        
        # Preload historical data
        self._preload_market_data()
        
        # Generate time index
        time_index = self._generate_time_index()
        
        for timestamp in time_index:
            self.current_time = timestamp
            
            # Apply market scenario overlay to historical data
            market_data = self._apply_market_scenario(timestamp)
            
            # Update portfolio with latest market data
            self._update_portfolio_values(market_data)
            
            # Execute strategy logic for each symbol
            for symbol in self.config.symbols:
                if symbol in self.strategies:
                    signals = self.strategies[symbol].generate_signals(timestamp)
                    self._process_signals(symbol, signals)
            
            # Record portfolio state
            self._record_portfolio_snapshot()
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        return self._prepare_simulation_results()
    
    def _preload_market_data(self):
        """Preload historical market data for all symbols"""
        logger.info("Preloading market data for simulation")
        
        for symbol in self.config.symbols:
            try:
                data = self.data_provider.get_historical_data(
                    symbol=symbol,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    interval=self.config.data_frequency
                )
                self.market_data_cache[symbol] = data
                logger.info(f"Loaded {len(data)} data points for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                raise
    
    def _generate_time_index(self) -> List[datetime.datetime]:
        """Generate time index for simulation based on data frequency"""
        if not self.market_data_cache:
            raise ValueError("Market data must be preloaded before generating time index")
        
        # Use the first symbol's data to generate time index
        symbol = list(self.market_data_cache.keys())[0]
        return self.market_data_cache[symbol].index.tolist()
    
    def _get_time_increment(self) -> datetime.timedelta:
        """Get time increment based on data frequency"""
        if self.config.data_frequency == "1min":
            return datetime.timedelta(minutes=1)
        elif self.config.data_frequency == "5min":
            return datetime.timedelta(minutes=5)
        elif self.config.data_frequency == "1h":
            return datetime.timedelta(hours=1)
        elif self.config.data_frequency == "1d":
            return datetime.timedelta(days=1)
        else:
            raise ValueError(f"Unsupported data frequency: {self.config.data_frequency}")
    
    def _apply_market_scenario(self, timestamp) -> Dict:
        """Apply market scenario overlay to historical data at given timestamp"""
        modified_data = {}
        
        for symbol in self.config.symbols:
            if symbol not in self.market_data_cache or timestamp not in self.market_data_cache[symbol].index:
                continue
                
            base_data = self.market_data_cache[symbol].loc[timestamp].to_dict()
            
            if self.config.market_scenario == MarketScenario.HIGH_VOLATILITY:
                # Increase price volatility
                volatility_factor = 1.5 + 0.5 * np.random.randn()
                base_data['high'] = base_data['open'] + volatility_factor * (base_data['high'] - base_data['open'])
                base_data['low'] = base_data['open'] - volatility_factor * (base_data['open'] - base_data['low'])
                price_range = base_data['high'] - base_data['low']
                base_data['close'] = base_data['low'] + np.random.random() * price_range
                
            elif self.config.market_scenario == MarketScenario.LOW_LIQUIDITY:
                # Widen bid-ask spread
                if 'bid' in base_data and 'ask' in base_data:
                    mid_price = (base_data['bid'] + base_data['ask']) / 2
                    spread = base_data['ask'] - base_data['bid']
                    base_data['bid'] = mid_price - 2.0 * spread / 2
                    base_data['ask'] = mid_price + 2.0 * spread / 2
                
            elif self.config.market_scenario == MarketScenario.FLASH_CRASH:
                # Sudden price drop followed by recovery
                day_of_simulation = (timestamp - self.config.start_date).days
                if day_of_simulation == len(self._generate_time_index()) // 2:  # Middle of simulation
                    base_data['close'] = base_data['open'] * 0.9  # 10% drop
                    base_data['low'] = base_data['open'] * 0.85   # 15% intraday drop
                
            elif self.config.market_scenario == MarketScenario.SIDEWAYS:
                # Reduce price movements
                base_data['close'] = base_data['open'] + 0.1 * (base_data['close'] - base_data['open'])
                range_factor = 0.3
                base_data['high'] = base_data['open'] + range_factor * (base_data['high'] - base_data['open'])
                base_data['low'] = base_data['open'] - range_factor * (base_data['open'] - base_data['low'])
            
            modified_data[symbol] = base_data
            
        return modified_data
    
    def _update_portfolio_values(self, market_data=None):
        """Update portfolio values based on current market prices"""
        if market_data is None:
            # Get market data for current timestamp
            market_data = {}
            for symbol in self.config.symbols:
                if symbol in self.market_data_cache and self.current_time in self.market_data_cache[symbol].index:
                    market_data[symbol] = self.market_data_cache[symbol].loc[self.current_time].to_dict()
        
        # Update position values
        total_position_value = 0.0
        
        for symbol, position in self.portfolio["positions"].items():
            if symbol in market_data:
                current_price = market_data[symbol].get('close')
                position["current_price"] = current_price
                position["current_value"] = position["quantity"] * current_price
                total_position_value += position["current_value"]
        
        # Update portfolio total value
        self.portfolio["total_value"] = self.portfolio["cash"] + total_position_value
        
        # Update margin available
        self.portfolio["margin_available"] = self.portfolio["total_value"] - self.portfolio["margin_used"]
    
    def _process_signals(self, symbol, signals):
        """Process trading signals and execute orders"""
        if not signals:
            return
            
        for signal in signals:
            if signal["action"] == "BUY":
                self._execute_buy(symbol, signal)
            elif signal["action"] == "SELL":
                self._execute_sell(symbol, signal)
            elif signal["action"] == "CLOSE":
                self._execute_close_position(symbol)
    
    def _execute_buy(self, symbol, signal):
        """Execute buy order based on signal"""
        quantity = signal.get("quantity", 0)
        price = signal.get("price")
        
        if not price:
            # Get current market price
            price = self._get_current_price(symbol)
            
        # Apply slippage
        execution_price = self._apply_slippage(price, "BUY")
        
        # Calculate order cost
        order_cost = execution_price * quantity
        commission = self._calculate_commission(order_cost)
        total_cost = order_cost + commission
        
        # Verify sufficient funds
        if total_cost > self.portfolio["cash"]:
            logger.warning(f"Insufficient funds for {symbol} buy order. Required: {total_cost}, Available: {self.portfolio['cash']}")
            # Adjust quantity if fractional shares are enabled
            if self.config.enable_fractional_shares:
                quantity = (self.portfolio["cash"] - commission) / execution_price
                order_cost = execution_price * quantity
                total_cost = order_cost + commission
                logger.info(f"Adjusted {symbol} buy quantity to {quantity} based on available funds")
            else:
                return
        
        # Update portfolio
        self.portfolio["cash"] -= total_cost
        
        # Update or create position
        if symbol in self.portfolio["positions"]:
            position = self.portfolio["positions"][symbol]
            # Calculate new average entry price
            total_shares = position["quantity"] + quantity
            position["entry_price"] = ((position["entry_price"] * position["quantity"]) + 
                                    (execution_price * quantity)) / total_shares
            position["quantity"] = total_shares
            position["current_price"] = execution_price
            position["current_value"] = position["quantity"] * execution_price
        else:
            self.portfolio["positions"][symbol] = {
                "quantity": quantity,
                "entry_price": execution_price,
                "current_price": execution_price,
                "current_value": quantity * execution_price,
                "entry_time": self.current_time
            }
        
        # Record trade
        trade = {
            "time": self.current_time,
            "symbol": symbol,
            "action": "BUY",
            "quantity": quantity,
            "price": execution_price,
            "commission": commission,
            "total_cost": total_cost
        }
        self.trades.append(trade)
        
        logger.info(f"Executed BUY: {quantity} shares of {symbol} @ {execution_price}")
    
    def _execute_sell(self, symbol, signal):
        """Execute sell order based on signal"""
        quantity = signal.get("quantity", 0)
        price = signal.get("price")
        
        if not price:
            # Get current market price
            price = self._get_current_price(symbol)
            
        # Apply slippage
        execution_price = self._apply_slippage(price, "SELL")
        
        # Check if we have the position
        if symbol not in self.portfolio["positions"]:
            logger.warning(f"Cannot sell {symbol} - no position exists")
            return
            
        position = self.portfolio["positions"][symbol]
        
        # Adjust quantity if needed
        if quantity > position["quantity"]:
            logger.warning(f"Sell quantity ({quantity}) exceeds position size ({position['quantity']}). Adjusting to available quantity.")
            quantity = position["quantity"]
            
        # Calculate order proceeds
        order_proceeds = execution_price * quantity
        commission = self._calculate_commission(order_proceeds)
        net_proceeds = order_proceeds - commission
        
        # Update portfolio
        self.portfolio["cash"] += net_proceeds
        
        # Update position
        position["quantity"] -= quantity
        position["current_price"] = execution_price
        position["current_value"] = position["quantity"] * execution_price
        
        # Remove position if fully closed
        if position["quantity"] <= 0:
            del self.portfolio["positions"][symbol]
        
        # Record trade
        trade = {
            "time": self.current_time,
            "symbol": symbol,
            "action": "SELL",
            "quantity": quantity,
            "price": execution_price,
            "commission": commission,
            "total_proceeds": net_proceeds
        }
        self.trades.append(trade)
        
        logger.info(f"Executed SELL: {quantity} shares of {symbol} @ {execution_price}")
    
    def _execute_close_position(self, symbol):
        """Close entire position for a symbol"""
        if symbol not in self.portfolio["positions"]:
            logger.warning(f"Cannot close position for {symbol} - no position exists")
            return
            
        position = self.portfolio["positions"][symbol]
        
        # Create a sell signal for the entire position
        signal = {
            "action": "SELL",
            "quantity": position["quantity"]
        }
        
        # Execute sell order
        self._execute_sell(symbol, signal)
        
        logger.info(f"Closed entire position for {symbol}")
    
    def _get_current_price(self, symbol):
        """Get current market price for a symbol"""
        if symbol in self.market_data_cache and self.current_time in self.market_data_cache[symbol].index:
            return self.market_data_cache[symbol].loc[self.current_time]["close"]
        else:
            # Try to get price from data provider
            try:
                data = self.data_provider.get_current_price(symbol)
                return data["price"]
            except Exception as e:
                logger.error(f"Error getting current price for {symbol}: {str(e)}")
                # Fall back to last known price if possible
                if symbol in self.portfolio["positions"]:
                    return self.portfolio["positions"][symbol]["current_price"]
                raise ValueError(f"Cannot determine current price for {symbol}")
    
    def _apply_slippage(self, price, action):
        """Apply slippage model to price"""
        if self.config.slippage_model == "fixed":
            # Apply fixed amount
            if action == "BUY":
                return price + self.config.slippage_value
            else:  # SELL
                return price - self.config.slippage_value
                
        elif self.config.slippage_model == "percentage":
            # Apply percentage of price
            if action == "BUY":
                return price * (1 + self.config.slippage_value / 100)
            else:  # SELL
                return price * (1 - self.config.slippage_value / 100)
                
        elif self.config.slippage_model == "market_impact":
            # More complex model based on order size and market conditions
            # This is a simplified implementation
            # In reality, would consider volume, liquidity, etc.
            impact_factor = 1 + (0.5 * np.random.random() * self.config.slippage_value / 100)
            if action == "BUY":
                return price * impact_factor
            else:  # SELL
                return price / impact_factor
                
        return price  # Default: no slippage
    
    def _calculate_commission(self, order_value):
        """Calculate commission based on configured model"""
        if self.config.commission_model == "fixed":
            return self.config.commission_value
        elif self.config.commission_model == "percentage":
            return order_value * (self.config.commission_value / 100)
        return 0.0  # Default: no commission
    
    def _record_portfolio_snapshot(self):
        """Record portfolio state at current timestamp"""
        snapshot = {
            "timestamp": self.current_time,
            "cash": self.portfolio["cash"],
            "total_value": self.portfolio["total_value"],
            "positions": {symbol: pos["current_value"] for symbol, pos in self.portfolio["positions"].items()}
        }
        self.portfolio_history.append(snapshot)
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics based on portfolio history"""
        if not self.portfolio_history:
            logger.warning("No portfolio history to calculate performance metrics")
            return
            
        # Convert portfolio history to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df.set_index("timestamp", inplace=True)
        
        # Calculate daily returns
        df["daily_return"] = df["total_value"].pct_change().fillna(0)
        
        # Calculate cumulative returns
        initial_value = df["total_value"].iloc[0]
        df["cumulative_return"] = (df["total_value"] / initial_value) - 1
        
        # Calculate various metrics
        total_days = (df.index[-1] - df.index[0]).days or 1  # Avoid division by zero
        
        metrics = {
            "initial_capital": self.config.initial_capital,
            "final_portfolio_value": df["total_value"].iloc[-1],
            "total_return": (df["total_value"].iloc[-1] / initial_value) - 1,
            "annualized_return": ((1 + (df["total_value"].iloc[-1] / initial_value - 1)) ** (365 / total_days)) - 1,
            "sharpe_ratio": calculate_sharpe_ratio(df["daily_return"]),
            "max_drawdown": calculate_max_drawdown(df["total_value"]),
            "win_rate": self._calculate_win_rate(),
            "total_trades": len(self.trades)
        }
        
        self.performance_metrics = metrics
        logger.info(f"Calculated performance metrics: {metrics}")
    
    def _calculate_win_rate(self):
        """Calculate win rate based on completed trades"""
        if not self.trades:
            return 0.0
            
        # Group trades by symbol to match buys with sells
        trades_by_symbol = {}
        for trade in self.trades:
            symbol = trade["symbol"]
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Calculate profit/loss for completed round trips
        winning_trades = 0
        completed_trades = 0
        
        for symbol, symbol_trades in trades_by_symbol.items():
            # Sort trades by time
            sorted_trades = sorted(symbol_trades, key=lambda x: x["time"])
            
            # Match buys with sells (simplified FIFO method)
            remaining_shares = 0
            entry_value = 0
            
            for trade in sorted_trades:
                if trade["action"] == "BUY":
                    remaining_shares += trade["quantity"]
                    entry_value += trade["quantity"] * trade["price"]
                elif trade["action"] == "SELL":
                    # Only count if we had existing shares
                    if remaining_shares > 0:
                        # Calculate average entry price
                        avg_entry = entry_value / remaining_shares
                        
                        # Determine shares to match with this sell
                        matched_shares = min(trade["quantity"], remaining_shares)
                        
                        # Calculate profit/loss
                        pnl = matched_shares * (trade["price"] - avg_entry)
                        
                        # Count winning trade
                        if pnl > 0:
                            winning_trades += 1
                        
                        completed_trades += 1
                        
                        # Update remaining shares and entry value
                        exit_ratio = matched_shares / remaining_shares
                        entry_value *= (1 - exit_ratio)
                        remaining_shares -= matched_shares
        
        if completed_trades == 0:
            return 0.0
            
        return winning_trades / completed_trades
    
    def _prepare_simulation_results(self) -> Dict:
        """Prepare final simulation results"""
        # Convert portfolio history to DataFrame for analysis
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index("timestamp", inplace=True)
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        results = {
            "simulation_id": self.simulation_id,
            "configuration": {
                "mode": self.config.mode.name,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat() if self.config.end_date else None,
                "initial_capital": self.config.initial_capital,
                "symbols": self.config.symbols,
                "market_scenario": self.config.market_scenario.name,
                "data_frequency": self.config.data_frequency
            },
            "performance_metrics": self.performance_metrics,
            "final_portfolio": {
                "cash": self.portfolio["cash"],
                "total_value": self.portfolio["total_value"],
                "positions": {symbol: pos.copy() for symbol, pos in self.portfolio["positions"].items()}
            },
            "portfolio_history": portfolio_df.to_dict(orient="records"),
            "trades": trades_df.to_dict(orient="records") if not trades_df.empty else []
        }
        
        return results
    
    def _save_simulation_results(self, results):
        """Save simulation results to file"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define filenames
        results_file = os.path.join(self.output_dir, f"simulation_{self.simulation_id}.json")
        portfolio_file = os.path.join(self.output_dir, f"portfolio_{self.simulation_id}.csv")
        trades_file = os.path.join(self.output_dir, f"trades_{self.simulation_id}.csv")
        
        # Convert results to JSON-compatible format
        json_results = results.copy()
        
        # Convert DataFrames to lists
        if "portfolio_history" in results:
            df = pd.DataFrame(results["portfolio_history"])
            df.to_csv(portfolio_file)
            json_results["portfolio_history"] = "saved to separate file"
        
        if "trades" in results:
            df = pd.DataFrame(results["trades"])
            if not df.empty:
                df.to_csv(trades_file)
            json_results["trades"] = "saved to separate file"
        
        # Save results JSON
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Saved simulation results to {results_file}")
        
    def plot_portfolio_performance(self, save_path=None):
        """
        Plot portfolio performance over time.
        
        Args:
            save_path: Optional path to save the plot image
        """
        if not self.portfolio_history:
            logger.warning("No portfolio history to plot")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df.set_index("timestamp", inplace=True)
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 2, 2]})
        
        # Plot portfolio value
        axs[0].plot(df.index, df["total_value"], label="Portfolio Value", color="blue")
        axs[0].set_title("Portfolio Value Over Time")
        axs[0].set_ylabel("Value ($)")
        axs[0].grid(True)
        
        # Highlight drawdowns
        rolling_max = df["total_value"].cummax()
        drawdowns = (df["total_value"] - rolling_max) / rolling_max
        
        # Color the background during drawdown periods
        for i in range(1, len(drawdowns)):
            if drawdowns.iloc[i] < -0.05:  # Only highlight significant drawdowns (>5%)
                axs[0].axvspan(df.index[i-1], df.index[i], color='red', alpha=0.2)
        
        # Plot drawdowns
        axs[1].fill_between(df.index, 0, drawdowns.values, color="red", alpha=0.3)
        axs[1].plot(df.index, drawdowns.values, color="red", linestyle="-")
        axs[1].set_title("Drawdowns")
        axs[1].set_ylabel("Drawdown (%)")
        axs[1].grid(True)
        
        # Plot daily returns
        daily_returns = df["total_value"].pct_change().fillna(0)
        axs[2].bar(df.index, daily_returns.values, color=["green" if r > 0 else "red" for r in daily_returns])
        axs[2].set_title("Daily Returns")
        axs[2].set_ylabel("Return (%)")
        axs[2].grid(True)
        
        # Add performance metrics as text
        metrics_text = (
            f"Total Return: {self.performance_metrics.get('total_return', 0):.2%}\n"
            f"Annualized Return: {self.performance_metrics.get('annualized_return', 0):.2%}\n"
            f"Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {self.performance_metrics.get('max_drawdown', 0):.2%}\n"
            f"Win Rate: {self.performance_metrics.get('win_rate', 0):.2%}\n"
            f"Total Trades: {self.performance_metrics.get('total_trades', 0)}"
        )
        
        # Add text box for metrics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[0].text(0.02, 0.05, metrics_text, transform=axs[0].transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Format x-axis
        for ax in axs:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved performance plot to {save_path}")
        else:
            plt.show()
            
        plt.close(fig) 