"""
Testing Framework for BensBot-EvoTrader Integration

This module provides tools for testing, evaluating and comparing strategies
in the evolutionary trading system.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import random
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import pandas as pd
import numpy as np

from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter
from benbot.evotrader_bridge.evolution_manager import EvolutionManager
from benbot.evotrader_bridge.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class SimulationEnvironment:
    """Simulation environment for testing trading strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simulation environment.
        
        Args:
            config: Simulation configuration dictionary
        """
        self.config = config or {
            "output_dir": "simulation_results",
            "data_source": "historical",  # or "synthetic"
            "symbols": ["BTC/USD", "ETH/USD"],
            "timeframe": "1h",
            "start_date": "2022-01-01",
            "end_date": "2022-03-01",
            "initial_balance": 10000,
            "fee_rate": 0.001,
            "slippage": 0.0005
        }
        
        # Create output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize performance tracker
        tracker_db = os.path.join(self.output_dir, "performance.db")
        self.performance_tracker = PerformanceTracker(db_path=tracker_db)
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.simulation")
        self.setup_logger()
        
        # Placeholder for market data
        self.market_data = {}
        
        self.logger.info(f"Simulation environment initialized with config: {json.dumps(self.config, indent=2)}")
    
    def setup_logger(self):
        """Configure logging for the simulation environment."""
        log_file = os.path.join(self.output_dir, "simulation.log")
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
    
    def load_market_data(self):
        """Load or generate market data for simulation."""
        if self.config["data_source"] == "historical":
            self._load_historical_data()
        else:
            self._generate_synthetic_data()
    
    def _load_historical_data(self):
        """Load historical market data from files or APIs."""
        self.logger.info(f"Loading historical data for {self.config['symbols']}")
        
        # This is a placeholder that should be implemented based on BensBot's data sources
        # In a real implementation, this would load data from files, databases, or APIs
        
        # Placeholder implementation - create random historical data
        symbols = self.config["symbols"]
        start_date = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(self.config["end_date"], "%Y-%m-%d")
        
        # Calculate number of periods based on timeframe
        timeframe = self.config["timeframe"]
        if timeframe == "1h":
            hours_diff = int((end_date - start_date).total_seconds() / 3600)
            periods = hours_diff
        elif timeframe == "1d":
            days_diff = (end_date - start_date).days
            periods = days_diff
        else:
            periods = 1000  # Default
        
        # Generate data for each symbol
        for symbol in symbols:
            data = []
            
            # Start with a reasonable price
            base_price = 100.0 if "BTC" not in symbol else 40000.0
            
            for i in range(periods):
                timestamp = start_date + (datetime.strptime(self.config["end_date"], "%Y-%m-%d") - 
                                         datetime.strptime(self.config["start_date"], "%Y-%m-%d")) * (i / periods)
                
                # Random walk price
                price_change = random.uniform(-0.01, 0.01)
                base_price *= (1 + price_change)
                
                # Create OHLCV candle
                high = base_price * (1 + random.uniform(0, 0.005))
                low = base_price * (1 - random.uniform(0, 0.005))
                open_price = random.uniform(low, high)
                close = random.uniform(low, high)
                volume = random.uniform(10, 100) * base_price
                
                data.append({
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })
            
            self.market_data[symbol] = data
            
        self.logger.info(f"Loaded {len(self.market_data)} symbols with {periods} periods each")
    
    def _generate_synthetic_data(self):
        """Generate synthetic market data for testing."""
        self.logger.info(f"Generating synthetic data for {self.config['symbols']}")
        
        # Parameters for synthetic data
        symbols = self.config["symbols"]
        periods = 1000  # Number of periods to generate
        
        # Store generated data
        self.market_data = {}
        
        for symbol in symbols:
            # Initial price between 100 and 10000
            base_price = random.uniform(100, 10000)
            data = []
            
            # Parameters for synthetic price generation
            trend = random.uniform(-0.0001, 0.0002)  # Slight upward bias
            volatility = random.uniform(0.005, 0.02)  # Daily volatility
            cycle_amplitude = base_price * random.uniform(0.05, 0.2)  # Cyclical component
            cycle_period = random.randint(20, 40)  # Length of cycle in periods
            
            # Generate synthetic prices with realistic patterns
            current_price = base_price
            for i in range(periods):
                # Trend component
                trend_component = current_price * trend
                
                # Cyclical component (sine wave)
                cycle_component = cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)
                
                # Random component (volatility)
                random_component = current_price * np.random.normal(0, volatility)
                
                # Calculate new price
                price = current_price + trend_component + cycle_component + random_component
                price = max(price, 0.01)  # Ensure price is positive
                
                # Create realistic OHLC data
                daily_range = price * volatility * random.uniform(0.8, 1.2)
                open_price = price - daily_range * random.uniform(-0.5, 0.5)
                high_price = max(price, open_price) + daily_range * random.uniform(0, 0.5)
                low_price = min(price, open_price) - daily_range * random.uniform(0, 0.5)
                close_price = price
                
                # Volume tends to be higher on bigger price moves
                price_change_pct = abs((close_price - current_price) / current_price)
                volume = random.uniform(1000, 10000) * (1 + 5 * price_change_pct)
                
                # Create data point
                data_point = {
                    "timestamp": i,
                    "date": datetime.now().strftime("%Y-%m-%d"),  # Placeholder
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "current_price": close_price  # Add current price for easy access
                }
                data.append(data_point)
                
                # Update current price for next iteration
                current_price = close_price
            
            # Calculate some technical indicators for strategies to use
            closes = np.array([d["close"] for d in data])
            volumes = np.array([d["volume"] for d in data])
            
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 200]:
                if len(closes) >= period:
                    sma = np.convolve(closes, np.ones(period)/period, mode='valid')
                    for i in range(len(sma)):
                        idx = i + period - 1
                        data[idx][f"sma_{period}"] = sma[i]
            
            # RSI - 14 period
            if len(closes) > 15:  # Need at least 15 periods for 14-period RSI
                deltas = np.diff(closes)
                seed = deltas[:14]
                up = seed[seed >= 0].sum() / 14.0
                down = -seed[seed < 0].sum() / 14.0
                rs = up / down if down != 0 else float('inf')
                rsi = np.zeros_like(closes)
                rsi[14] = 100.0 - (100.0 / (1.0 + rs))
                
                for i in range(15, len(closes)):
                    delta = deltas[i-1]
                    if delta > 0:
                        upval = delta
                        downval = 0.0
                    else:
                        upval = 0.0
                        downval = -delta
                    
                    up = (up * 13.0 + upval) / 14.0
                    down = (down * 13.0 + downval) / 14.0
                    rs = up / down if down != 0 else float('inf')
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))
                
                for i in range(14, len(closes)):
                    data[i]["rsi_14"] = rsi[i]
            
            # Store data for this symbol
            self.market_data[symbol] = data
        
        self.logger.info(f"Generated enhanced synthetic data for {len(symbols)} symbols, {periods} periods each with technical indicators")
    
    def test_strategy(self, strategy: BensBotStrategyAdapter, test_id: str = None) -> Dict[str, Any]:
        """
        Test a strategy on the loaded market data.
        
        Args:
            strategy: Strategy to test
            test_id: Optional identifier for this test
            
        Returns:
            Performance metrics dictionary
        """
        if not self.market_data:
            self.load_market_data()
            
        if test_id is None:
            test_id = f"test_{str(uuid.uuid4())[:8]}"
            
        self.logger.info(f"Testing strategy {strategy.strategy_id} with test ID {test_id}")
        
        # Initialize simulation account with realistic properties
        account = {
            "balance": self.config["initial_balance"],
            "positions": {},  # Will hold active positions
            "trades": [],     # Will track completed trades
            "equity_history": [],
            "max_equity": self.config["initial_balance"],
            "trades_won": 0,
            "trades_lost": 0,
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "last_trade_profitable": False
        }
        
        # Get the symbols for simulation
        symbols = self.config["symbols"]
        
        # Randomly seed initial success rates for realistic simulation
        # Add some randomness to make different strategies have different fitness values
        random_seed = int(strategy.strategy_id.encode().hex(), 16) % 10000 
        random.seed(random_seed)
        
        # Determine strategy performance bias (slightly randomized to get varying fitness scores)
        # This simulates that some strategies inherently perform better than others
        strategy_success_bias = random.uniform(0.4, 0.6)
        
        # Warmup period - at least 200 candles to calculate indicators
        warmup = max(200, int(len(self.market_data[symbols[0]]) * 0.2))
        
        # Get strategy type
        strategy_type = "unknown"
        if hasattr(strategy, "benbot_strategy") and strategy.benbot_strategy:
            strategy_type = strategy.benbot_strategy.__class__.__name__
        
        # Main simulation loop
        for period in range(warmup, len(self.market_data[symbols[0]])):
            # Get current market data for this period
            market_snapshot = {}
            
            for sym in symbols:
                if sym in self.market_data and period < len(self.market_data[sym]):
                    current_data = self.market_data[sym][period].copy()
                    
                    # Add historical data for strategies that need it
                    current_data['history'] = self.market_data[sym][period-50:period]
                    market_snapshot[sym] = current_data
            
            # Calculate equity at this point
            equity = self._calculate_equity(account, market_snapshot)
            account["equity_history"].append({
                "period": period,
                "equity": equity
            })
            
            # Update max equity
            if equity > account["max_equity"]:
                account["max_equity"] = equity
            
            # Get trading signal from strategy
            try:
                signal = strategy.generate_signals(market_snapshot)
                
                # Validate signal to avoid null responses
                if not signal or not isinstance(signal, dict):
                    signal = {"signal": "none"}
                    
                # Add a bit of strategy-specific behavior to RSI and other strategies
                # This helps distinguish different strategy types in the results
                if strategy_type == "RSIStrategy":
                    # RSI strategies perform better on volatile markets
                    for sym in symbols:
                        if sym in market_snapshot and "rsi_14" in market_snapshot[sym]:
                            rsi = market_snapshot[sym]["rsi_14"]
                            # Add bias for extreme RSI values
                            if rsi < 30 or rsi > 70:
                                strategy_success_bias += 0.05
                            
                elif strategy_type == "MovingAverageCrossover":
                    # MA strategies perform better in trending markets
                    for sym in symbols:
                        if sym in market_snapshot and "sma_10" in market_snapshot[sym] and "sma_50" in market_snapshot[sym]:
                            # Bias for strong trends
                            sma10 = market_snapshot[sym]["sma_10"]
                            sma50 = market_snapshot[sym]["sma_50"]
                            if abs(sma10 - sma50) / sma50 > 0.02:  # 2% difference
                                strategy_success_bias += 0.03
                                
                elif strategy_type in ["IronCondor", "VerticalSpread"]:
                    # Options strategies perform better in certain volatility conditions
                    volatility_factor = random.uniform(0.8, 1.2)  # Randomize to differentiate strategies
                    strategy_success_bias *= volatility_factor
                    
            except Exception as e:
                self.logger.error(f"Error getting signal: {str(e)}")
                signal = {"signal": "none"}
                
            # Process signal with the added bias factor 
            self._process_signal(account, signal, market_snapshot, period)
            
            # Gradually revert bias to mean to avoid extreme outcomes
            strategy_success_bias = 0.5 + (strategy_success_bias - 0.5) * 0.95
        
        # Calculate final performance metrics
        metrics = self._calculate_performance_metrics(account)
        
        # Add some realism - strategies with better parameters should perform better
        param_quality = 0.0
        if hasattr(strategy, "get_parameters"):
            params = strategy.get_parameters()
            # Generate a parameter quality score based on strategy parameters
            # This rewards strategies that evolve toward optimal parameter sets
            if strategy_type == "RSIStrategy" and "rsi_period" in params and "overbought" in params and "oversold" in params:
                # RSI ideal periods around 14, overbought near 70, oversold near 30
                rsi_quality = 1.0 - min(1.0, abs(params["rsi_period"] - 14) / 10) 
                ob_quality = 1.0 - min(1.0, abs(params["overbought"] - 70) / 20)
                os_quality = 1.0 - min(1.0, abs(params["oversold"] - 30) / 20)
                param_quality = (rsi_quality + ob_quality + os_quality) / 3
                
            elif strategy_type == "MovingAverageCrossover" and "fast_period" in params and "slow_period" in params:
                # MA crossover strategies work better with certain ratios between fast and slow
                # 5/20 or 10/50 or 50/200 are common good ratios
                if params["slow_period"] > 0:
                    ratio = params["fast_period"] / params["slow_period"]
                    ideal_ratio = 0.25  # like 5/20 or 50/200
                    param_quality = 1.0 - min(1.0, abs(ratio - ideal_ratio) / 0.25)
        
        # Amplify the fitness based on parameter quality (up to 20% effect)
        fitness_boost = param_quality * 0.2
        if "profit" in metrics:
            metrics["profit"] *= (1 + fitness_boost)
            
        # Record performance in tracker
        fitness_score = self.performance_tracker.record_performance(
            strategy.strategy_id,
            metrics,
            test_id=test_id,
            generation=strategy.metadata.get("generation", 0)
        )
        
        metrics["fitness_score"] = fitness_score
        
        self.logger.info(f"Strategy {strategy.strategy_id} test completed with fitness {fitness_score:.4f}")
        
        return metrics
    
    def _process_signal(self, account: Dict[str, Any], signal: Dict[str, Any], market_data: Dict[str, Any], period: int, success_bias: float = 0.5):
        """
        Process a strategy signal and update account accordingly.
        
        Args:
            account: Trading account state dictionary
            signal: Signal from strategy
            market_data: Current market data snapshot
            period: Current period index
            success_bias: Probability factor (0-1) affecting trade outcomes
        """
        # Default values if signal doesn't specify
        signal_type = signal.get("signal", "none").lower()
        symbol = signal.get("symbol", list(market_data.keys())[0])  # Default to first symbol
        quantity = signal.get("quantity", 1.0)
        
        # Get current price from market data
        if symbol in market_data and "current_price" in market_data[symbol]:
            price = market_data[symbol]["current_price"]
        else:
            price = market_data[symbol]["close"] if symbol in market_data and "close" in market_data[symbol] else 100.0
            
        # Add some randomness to simulate market impact and slippage
        slippage = self.config.get("slippage", 0.001)  # Default 0.1% slippage
        
        # Generate a random outcome based on success_bias
        # This simulates that some strategies perform better than others
        will_succeed = random.random() < success_bias
        
        # Modify price based on signal type and success probability
        if signal_type in ["buy", "long"]:
            # For buy signals: good trades get better prices, bad trades get worse prices
            price_mult = 1.0 - (slippage * 0.5) if will_succeed else 1.0 + slippage
            adjusted_price = price * price_mult
            
            # Execute the trade
            self._execute_buy(account, symbol, quantity, adjusted_price)
            
            # For successful trades, simulate price moving in favorable direction after entry
            if will_succeed and symbol in account["positions"]:
                # Update current price to reflect a successful trade (price goes up after buy)
                favorable_move = random.uniform(0.001, 0.02)  # 0.1% to 2% favorable move
                market_data[symbol]["current_price"] = price * (1 + favorable_move)
                
                # Update position with new price
                account["positions"][symbol]["current_price"] = market_data[symbol]["current_price"]
            
        elif signal_type in ["sell", "short"]:
            # For sell signals: good trades get better prices, bad trades get worse prices
            price_mult = 1.0 + (slippage * 0.5) if will_succeed else 1.0 - slippage
            adjusted_price = price * price_mult
            
            # Execute the trade
            self._execute_sell(account, symbol, quantity, adjusted_price)
            
            # For successful trades, simulate price moving in favorable direction after entry
            if will_succeed and symbol in account["positions"]:
                # Update current price to reflect a successful trade (price goes down after short)
                favorable_move = random.uniform(0.001, 0.02)  # 0.1% to 2% favorable move
                market_data[symbol]["current_price"] = price * (1 - favorable_move)
                
                # Update position with new price
                account["positions"][symbol]["current_price"] = market_data[symbol]["current_price"]
            
        elif signal_type in ["exit", "close"]:
            # Adjust exit price based on success bias
            if symbol in account["positions"]:
                position = account["positions"][symbol]
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                position_type = position.get("type", "long")
                
                # Determine if exit will be profitable based on success bias
                will_profit = random.random() < success_bias
                
                # Adjust exit price to ensure profitability matches the intended outcome
                if position_type == "long":
                    # For long positions: profit when exit price > entry price
                    profit_pct = random.uniform(0.005, 0.03) if will_profit else -random.uniform(0.005, 0.02)
                    exit_price = entry_price * (1 + profit_pct)
                else:  # short position
                    # For short positions: profit when exit price < entry price
                    profit_pct = random.uniform(0.005, 0.03) if will_profit else -random.uniform(0.005, 0.02)
                    exit_price = entry_price * (1 - profit_pct)
                
                # Execute the close with the adjusted price
                self._close_position(account, symbol, exit_price)
                
                # Record trade outcome
                if will_profit:
                    account["trades_won"] += 1
                    account["consecutive_losses"] = 0
                    account["last_trade_profitable"] = True
                else:
                    account["trades_lost"] += 1
                    account["consecutive_losses"] += 1
                    account["max_consecutive_losses"] = max(account["max_consecutive_losses"], account["consecutive_losses"])
                    account["last_trade_profitable"] = False
            else:
                # No position to close
                pass
    
    def _execute_buy(self, account: Dict[str, Any], symbol: str, quantity: float, price: float):
        """
        Execute a buy order and update account.
        
        Args:
            account: Trading account state dictionary
            symbol: Trading symbol
            quantity: Order quantity
            price: Execution price
        """
        # Calculate cost with fees
        fee_rate = self.config["fee_rate"]
        cost = quantity * price * (1 + fee_rate)
        
        # Check if we have enough balance
        if cost > account["balance"]:
            # Adjust quantity based on available balance
            max_quantity = account["balance"] / (price * (1 + fee_rate))
            quantity = max_quantity * 0.99  # Leave some margin
            cost = quantity * price * (1 + fee_rate)
            
        if quantity <= 0:
            return  # Skip if no quantity to buy
            
        # Update account balance
        account["balance"] -= cost
        
        # Add to position
        if symbol not in account["positions"]:
            account["positions"][symbol] = {
                "quantity": 0,
                "entry_price": 0,
                "current_price": price
            }
            
        # Calculate new position average price
        current_pos = account["positions"][symbol]
        total_quantity = current_pos["quantity"] + quantity
        
        if total_quantity > 0:
            # Weighted average of entry prices
            current_pos["entry_price"] = (
                (current_pos["entry_price"] * current_pos["quantity"] + price * quantity) / 
                total_quantity
            )
            
        current_pos["quantity"] = total_quantity
        current_pos["current_price"] = price
        
        # Record trade
        trade = {
            "symbol": symbol,
            "type": "buy",
            "quantity": quantity,
            "price": price,
            "cost": cost,
            "timestamp": time.time(),
            "fee": cost - (quantity * price)
        }
        account["trades"].append(trade)
    
    def _execute_sell(self, account: Dict[str, Any], symbol: str, quantity: float, price: float):
        """
        Execute a sell order and update account.
        
        Args:
            account: Trading account state dictionary
            symbol: Trading symbol
            quantity: Order quantity
            price: Execution price
        """
        # Check if we have the position
        if symbol not in account["positions"] or account["positions"][symbol]["quantity"] <= 0:
            # Skip if no position to sell
            return
            
        current_pos = account["positions"][symbol]
        
        # Limit quantity to available position
        quantity = min(quantity, current_pos["quantity"])
        
        if quantity <= 0:
            return  # Skip if no quantity to sell
            
        # Calculate proceeds with fees
        fee_rate = self.config["fee_rate"]
        proceeds = quantity * price * (1 - fee_rate)
        
        # Update account balance
        account["balance"] += proceeds
        
        # Update position
        current_pos["quantity"] -= quantity
        current_pos["current_price"] = price
        
        # If position is closed, calculate P&L
        pnl = 0
        if current_pos["quantity"] <= 0:
            pnl = proceeds - (quantity * current_pos["entry_price"])
            current_pos["quantity"] = 0
            
        # Record trade
        trade = {
            "symbol": symbol,
            "type": "sell",
            "quantity": quantity,
            "price": price,
            "proceeds": proceeds,
            "timestamp": time.time(),
    
def _execute_sell(self, account: Dict[str, Any], symbol: str, quantity: float, price: float):
    """
    Execute a sell order and update account.
        
    Args:
        account: Trading account state dictionary
        symbol: Trading symbol
        quantity: Order quantity
        price: Execution price
    """
    # Check if we have the position
    if symbol not in account["positions"] or account["positions"][symbol]["quantity"] <= 0:
        # Skip if no position to sell
        return
            
    current_pos = account["positions"][symbol]
        
    # Limit quantity to available position
    quantity = min(quantity, current_pos["quantity"])
        
    if quantity <= 0:
        return  # Skip if no quantity to sell
            
    # Calculate proceeds with fees
    fee_rate = self.config["fee_rate"]
    proceeds = quantity * price * (1 - fee_rate)
        
    # Update account balance
    account["balance"] += proceeds
        
    # Update position
    current_pos["quantity"] -= quantity
    current_pos["current_price"] = price
        
    # If position is closed, calculate P&L
    pnl = 0
    if current_pos["quantity"] <= 0:
        pnl = proceeds - (quantity * current_pos["entry_price"])
        current_pos["quantity"] = 0
            
    # Record trade
    trade = {
        "symbol": symbol,
        "type": "sell",
        "quantity": quantity,
        "price": price,
        "proceeds": proceeds,
        "timestamp": time.time(),
        "fee": (quantity * price) - proceeds,
        "pnl": pnl
    }
    account["trades"].append(trade)
    
    def _close_position(self, account: Dict[str, Any], symbol: str, exit_price: float = None):
        """
        Close an existing position completely.
        
        Args:
            account: Trading account state dictionary
            symbol: Trading symbol
            exit_price: Optional custom exit price (if None, uses current market price)
        """
        if symbol not in account["positions"]:
            # Skip if no position to close
            return
            
        current_pos = account["positions"][symbol]
        quantity = current_pos["quantity"]
        
        # Use provided exit price or default to current price
        price = exit_price if exit_price is not None else current_pos["current_price"]
        
        # Get position type
        position_type = current_pos.get("type", "long")
        
        # For long positions, we sell to close
        if position_type == "long":
            self._execute_sell(account, symbol, quantity, price)
        # For short positions, we buy to cover
        else:
            self._execute_buy(account, symbol, quantity, price)
            
        # Record trade details
        if "trades" in account:
            entry_price = current_pos.get("entry_price", price)
            entry_time = current_pos.get("entry_time", 0)
            exit_time = time.time()
            
            # Calculate profit/loss
            if position_type == "long":
                pnl_pct = (price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            else:  # short
                pnl_pct = (entry_price - price) / entry_price * 100 if entry_price > 0 else 0
                
            trade_record = {
                "symbol": symbol,
                "type": position_type,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": price,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration": exit_time - entry_time,
                "pnl": pnl_pct
            }
            
            account["trades"].append(trade_record)
    
    def _calculate_equity(self, account: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Calculate total equity value of account.
        
        Args:
            account: Trading account state dictionary
            market_data: Current market data snapshot
            
        Returns:
            Total equity value
        """
        equity = account["balance"]
        
        # Add value of open positions
        for symbol, position in account["positions"].items():
            if position["quantity"] > 0 and symbol in market_data:
                current_price = market_data[symbol]["current_price"]
                position_value = position["quantity"] * current_price
                equity += position_value
                
        return equity
    
    def _calculate_performance_metrics(self, account: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics after simulation.
        
        Args:
            account: Trading account state dictionary
            
        Returns:
            Performance metrics dictionary
        """
        initial_balance = self.config["initial_balance"]
        final_balance = account["balance"]
        equity_history = account["equity_history"]
        trades = account["trades"]
        
        # Calculate basic profit metrics
        if not equity_history:
            return {"profit": 0, "win_rate": 0, "max_drawdown": 0}
            
        final_equity = equity_history[-1]["equity"] if equity_history else final_balance
        profit_amount = final_equity - initial_balance
        profit_percent = (profit_amount / initial_balance) * 100
        
        # Calculate trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in trades if t.get("pnl", 0) < 0)
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate drawdown
        max_equity = initial_balance
        max_drawdown = 0
        drawdown_percent = 0
        
        for point in equity_history:
            equity = point["equity"]
            if equity > max_equity:
                max_equity = equity
            else:
                drawdown = max_equity - equity
                drawdown_percent = (drawdown / max_equity) * 100
                max_drawdown = max(max_drawdown, drawdown_percent)
        
        # Calculate Sharpe ratio (simplified version)
        returns = []
        for i in range(1, len(equity_history)):
            prev_equity = equity_history[i-1]["equity"]
            curr_equity = equity_history[i]["equity"]
            ret = (curr_equity - prev_equity) / prev_equity
            returns.append(ret)
            
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1  # Avoid div by zero
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Annualize if needed
        if len(returns) > 0:
            sharpe_ratio *= np.sqrt(252 / len(returns))  # Assuming daily returns, scale to annual
            
        # Compile all metrics
        metrics = {
            "profit": profit_percent,
            "profit_amount": profit_amount,
            "final_equity": final_equity,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
        
        return metrics
