#!/usr/bin/env python3
"""
Paper Trading Simulation with Anomaly-Based Risk Management and Parameter Optimization

This script simulates paper trading with:
- Anomaly detection for risk management
- Parameter optimization for trading strategy and risk parameters
- Multiple asset classes support
- Advanced error handling and logging
- Enhanced visualization options
- Real-time monitoring capabilities
- Historical data overlay capability
- Volatility regime simulation

Usage:
    python paper_trading_simulation.py --assets BTC ETH --periods 1000 --anomaly-prob 0.05 --display-interval 50
"""

import argparse
import logging
import os
import sys
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import traceback
from itertools import product

# Add parent directory to path to import from trading_bot
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from trading_bot.ml.market_anomaly_detector import MarketAnomalyDetector
    from trading_bot.risk_manager import RiskManager, RiskLevel
    from trading_bot.utils.market_simulator import MarketDataGenerator
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Could not import dependencies. Using mock implementations.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trading_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PaperTradingSimulation")

# Create rich console for pretty output
console = Console()

class SimulationException(Exception):
    """Custom exception class for simulation errors with detailed context"""
    def __init__(self, message, component=None, context=None, recommendation=None):
        self.message = message
        self.component = component
        self.context = context or {}
        self.recommendation = recommendation
        super().__init__(self.formatted_message())
        
    def formatted_message(self):
        msg = f"[{self.component}] {self.message}" if self.component else self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" (Context: {context_str})"
        if self.recommendation:
            msg += f" Recommendation: {self.recommendation}"
        return msg

class ParamOptimizer:
    """Parameter optimizer for trading strategies and risk management"""
    
    def __init__(self, 
                 simulator,
                 param_grid: Dict[str, List[Any]], 
                 metric: str = 'sharpe_ratio',
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        Initialize the parameter optimizer.
        
        Args:
            simulator: Reference to the simulator object
            param_grid: Dictionary of parameters to optimize with lists of possible values
            metric: Performance metric to optimize ('sharpe_ratio', 'total_return', etc.)
            n_jobs: Number of parallel jobs for optimization
            verbose: Whether to print optimization progress
        """
        self.simulator = simulator
        self.param_grid = param_grid
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results = []
        self.best_params = None
        self.best_score = -float('inf') if metric != 'max_drawdown' else float('inf')
        
    def _evaluate_params(self, params):
        """Evaluate a single parameter combination"""
        try:
            # Configure simulator with these parameters
            self.simulator.configure(params)
            
            # Run simulation
            self.simulator.run_simulation(show_progress=False)
            
            # Get performance metrics
            metrics = self.simulator.calculate_performance_metrics()
            score = metrics.get(self.metric, 0)
            
            # For metrics where lower is better (like max_drawdown)
            if self.metric == 'max_drawdown':
                is_better = score < self.best_score
            else:
                is_better = score > self.best_score
                
            result = {
                'params': params,
                'score': score,
                'all_metrics': metrics
            }
            
            # Log result
            if self.verbose:
                logger.info(f"Evaluated parameters {params}: {self.metric}={score:.4f}")
                
            return result, is_better, score
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {str(e)}")
            logger.error(traceback.format_exc())
            return {'params': params, 'score': float('-inf'), 'error': str(e)}, False, float('-inf')
    
    def optimize(self):
        """Run the optimization process"""
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        
        if self.verbose:
            logger.info(f"Starting parameter optimization with {total_combinations} combinations")
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Optimizing {self.metric}", total=total_combinations)
            
            if self.n_jobs > 1:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = []
                    for params in param_combinations:
                        futures.append(executor.submit(self._evaluate_params, params))
                    
                    for future in futures:
                        result, is_better, score = future.result()
                        self.results.append(result)
                        
                        if is_better:
                            self.best_score = score
                            self.best_params = result['params']
                            
                        progress.update(task, advance=1)
            else:
                # Sequential execution
                for params in param_combinations:
                    result, is_better, score = self._evaluate_params(params)
                    self.results.append(result)
                    
                    if is_better:
                        self.best_score = score
                        self.best_params = result['params']
                        
                    progress.update(task, advance=1)
        
        # Sort results
        self.results.sort(key=lambda x: x['score'], reverse=(self.metric != 'max_drawdown'))
        
        if self.verbose:
            logger.info(f"Parameter optimization complete. Best {self.metric}: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
        return self.best_params, self.best_score
    
    def _generate_param_combinations(self):
        """Generate all combinations of parameters from param_grid"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def plot_param_importance(self, top_n=10):
        """Plot parameter importance based on optimization results"""
        if not self.results:
            logger.error("No optimization results available")
            return
        
        param_scores = {}
        
        # Group scores by parameter values
        for param in self.param_grid.keys():
            param_scores[param] = {}
            for result in self.results:
                param_value = result['params'].get(param)
                if param_value not in param_scores[param]:
                    param_scores[param][param_value] = []
                param_scores[param][param_value].append(result['score'])
        
        # Calculate average score for each parameter value
        param_impact = {}
        for param, value_scores in param_scores.items():
            values = []
            avg_scores = []
            for value, scores in value_scores.items():
                values.append(str(value))
                avg_scores.append(np.mean(scores))
            
            # Calculate variance in scores as impact measure
            if len(avg_scores) > 1:
                param_impact[param] = np.std(avg_scores)
            else:
                param_impact[param] = 0
        
        # Plot top N parameters by impact
        sorted_params = sorted(param_impact.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig = plt.figure(figsize=(10, 6))
        plt.bar([x[0] for x in sorted_params], [x[1] for x in sorted_params])
        plt.title('Parameter Importance')
        plt.xlabel('Parameter')
        plt.ylabel('Impact (Standard Deviation of Scores)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save and show
        plt.savefig('param_importance.png')
        plt.close()
        
        return fig
    
    def get_top_configurations(self, n=5):
        """Return the top N parameter configurations"""
        if not self.results:
            return []
            
        sorted_results = sorted(self.results, 
                                key=lambda x: x['score'], 
                                reverse=(self.metric != 'max_drawdown'))
        return sorted_results[:n]

class MockMultiAssetAdapter(MultiAssetAdapter):
    """Mock adapter for paper trading simulation."""
    
    def __init__(self, initial_balance: float = 100000.0):
        """Initialize the mock adapter."""
        super().__init__({})  # Empty config
        self.account_balance = initial_balance
        self.positions = {}
        self.orders = []
        self.market_data = {}
        self.transaction_history = []
        
        logger.info(f"Initialized mock adapter with ${initial_balance:.2f}")
    
    def get_account_balance(self) -> float:
        """Get the current account balance."""
        return self.account_balance
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return {
            "balance": self.account_balance,
            "equity": self.calculate_equity(),
            "margin_available": self.account_balance,
            "positions": len(self.positions)
        }
    
    def calculate_equity(self) -> float:
        """Calculate total equity including positions."""
        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol].iloc[-1]["close"]
                position_value += position["quantity"] * current_price
        
        return self.account_balance + position_value
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """Update market data for a symbol."""
        self.market_data[symbol] = data
    
    def place_order(self, symbol: str, order_type: str, quantity: float, 
                   side: str, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a mock order."""
        if symbol not in self.market_data:
            return {"status": "error", "message": f"No market data for {symbol}"}
        
        # Get current price from market data
        current_price = self.market_data[symbol].iloc[-1]["close"]
        execution_price = price or current_price
        
        # Calculate order cost
        order_cost = quantity * execution_price
        commission = max(1.0, order_cost * 0.001)  # $1 min or 0.1%
        
        # Check if we have enough balance for a buy
        if side == "buy" and order_cost + commission > self.account_balance:
            return {"status": "error", "message": "Insufficient funds"}
        
        # Process the order
        order_id = f"ord_{len(self.orders) + 1}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "quantity": quantity,
            "price": execution_price,
            "commission": commission,
            "timestamp": datetime.now(),
            "status": "filled"
        }
        
        # Update positions
        if side == "buy":
            # Deduct from balance
            self.account_balance -= (order_cost + commission)
            
            # Add to position
            if symbol in self.positions:
                # Update existing position
                current_position = self.positions[symbol]
                avg_price = ((current_position["quantity"] * current_position["avg_price"]) + 
                             (quantity * execution_price)) / (current_position["quantity"] + quantity)
                current_position["quantity"] += quantity
                current_position["avg_price"] = avg_price
            else:
                # Create new position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": execution_price,
                    "unrealized_pnl": 0.0
                }
        else:  # sell
            # Check if we have the position
            if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
                return {"status": "error", "message": "Insufficient position"}
            
            # Update position
            position = self.positions[symbol]
            position["quantity"] -= quantity
            
            # Calculate P&L
            pnl = (execution_price - position["avg_price"]) * quantity - commission
            self.account_balance += order_cost + pnl
            
            # Record realized P&L
            order["realized_pnl"] = pnl
            
            # Remove position if quantity is zero
            if position["quantity"] <= 0:
                del self.positions[symbol]
        
        # Record the order
        self.orders.append(order)
        self.transaction_history.append({
            "timestamp": order["timestamp"],
            "type": "order",
            "details": order
        })
        
        logger.info(f"Order executed: {side.upper()} {quantity} {symbol} @ ${execution_price:.2f}")
        return {"status": "success", "order": order}
    
    def update_positions(self):
        """Update positions with current market data."""
        for symbol, position in list(self.positions.items()):
            if symbol in self.market_data:
                current_price = self.market_data[symbol].iloc[-1]["close"]
                position["current_price"] = current_price
                position["unrealized_pnl"] = (current_price - position["avg_price"]) * position["quantity"]
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of current positions."""
        self.update_positions()
        
        total_value = 0.0
        total_unrealized_pnl = 0.0
        positions_data = []
        
        for symbol, position in self.positions.items():
            position_value = position["quantity"] * position.get("current_price", position["avg_price"])
            total_value += position_value
            total_unrealized_pnl += position.get("unrealized_pnl", 0.0)
            
            positions_data.append({
                "symbol": symbol,
                "quantity": position["quantity"],
                "avg_price": position["avg_price"],
                "current_price": position.get("current_price", position["avg_price"]),
                "value": position_value,
                "unrealized_pnl": position.get("unrealized_pnl", 0.0)
            })
        
        return {
            "positions": positions_data,
            "total_value": total_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "count": len(positions_data)
        }

class PaperTradingSimulator:
    """Paper trading simulator with anomaly-based risk management."""
    
    def __init__(self, 
                 assets: List[str] = None, 
                 initial_balance: float = 100000.0,
                 lookback_window: int = 20,
                 anomaly_config_path: str = None):
        """
        Initialize the paper trading simulator.
        
        Args:
            assets: List of asset symbols to simulate
            initial_balance: Initial account balance
            lookback_window: Window size for anomaly detection
            anomaly_config_path: Path to anomaly risk configuration
        """
        # Default assets if none provided
        self.assets = assets or ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
        
        # Create directories for output
        os.makedirs("output", exist_ok=True)
        os.makedirs("models/anomaly_detection", exist_ok=True)
        os.makedirs("journal", exist_ok=True)
        
        # Initialize components
        self.adapter = MockMultiAssetAdapter(initial_balance)
        
        # If config path is not specified, use default
        if anomaly_config_path is None:
            anomaly_config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "anomaly_risk_rules.yaml"
            )
        
        # Risk manager with anomaly integration
        self.risk_manager = RiskManager(
            multi_asset_adapter=self.adapter,
            journal_dir="journal",
            anomaly_config_path=anomaly_config_path
        )
        
        # Anomaly detectors for each asset
        self.anomaly_detectors = {}
        for symbol in self.assets:
            self.anomaly_detectors[symbol] = MarketAnomalyDetector(
                symbol=symbol,
                lookback_window=lookback_window,
                alert_threshold=0.75,
                model_dir="models/anomaly_detection",
                use_autoencoder=True,
                contamination=0.01
            )
        
        # Market data buffers (deques for efficient window operations)
        self.data_buffers = {symbol: deque(maxlen=200) for symbol in self.assets}
        self.lookback_window = lookback_window
        
        # Simulation state
        self.current_timestamp = datetime.now()
        self.trading_enabled = True
        self.simulation_stats = {
            "trades_executed": 0,
            "anomalies_detected": 0,
            "risk_events": 0,
            "days_simulated": 0
        }
        
        # Tracking metrics
        self.equity_curve = []
        self.anomaly_scores = {symbol: [] for symbol in self.assets}
        self.risk_level_history = []
        
        logger.info(f"Paper trading simulator initialized with {len(self.assets)} assets")
        for symbol in self.assets:
            logger.info(f"  - {symbol}")
    
    def generate_market_data(self, 
                           periods: int = 100, 
                           anomaly_prob: float = 0.05) -> Generator[Dict[str, Any], None, None]:
        """
        Generate synthetic market data for simulation.
        
        Args:
            periods: Number of periods to generate
            anomaly_prob: Probability of injecting an anomaly
            
        Yields:
            Dictionary with market data for each asset
        """
        # Base prices for each asset
        base_prices = {
            "AAPL": 170.0,
            "MSFT": 290.0,
            "AMZN": 130.0,
            "GOOGL": 140.0,
            "TSLA": 180.0,
            "BTC/USD": 35000.0,
            "ETH/USD": 2000.0,
            "EUR/USD": 1.1,
            "SPY": 420.0,
            "QQQ": 350.0
        }
        
        # Use defaults for assets not in the dictionary
        for symbol in self.assets:
            if symbol not in base_prices:
                base_prices[symbol] = 100.0
        
        # Volatility for each asset
        volatilities = {symbol: price * 0.01 for symbol, price in base_prices.items()}
        
        # Correlation matrix for price movements (simplified)
        correlations = np.array([
            [1.0, 0.7, 0.5, 0.6, 0.4],
            [0.7, 1.0, 0.6, 0.7, 0.3],
            [0.5, 0.6, 1.0, 0.5, 0.2],
            [0.6, 0.7, 0.5, 1.0, 0.3],
            [0.4, 0.3, 0.2, 0.3, 1.0]
        ])
        
        # Pad or truncate correlation matrix to match assets
        num_assets = len(self.assets)
        if correlations.shape[0] > num_assets:
            correlations = correlations[:num_assets, :num_assets]
        elif correlations.shape[0] < num_assets:
            new_corr = np.eye(num_assets)
            new_corr[:correlations.shape[0], :correlations.shape[1]] = correlations
            correlations = new_corr
        
        # Cholesky decomposition for correlated price movements
        cholesky = np.linalg.cholesky(correlations)
        
        # Current prices
        current_prices = {symbol: price for symbol, price in base_prices.items()}
        
        # Time increment
        time_increment = timedelta(minutes=5)
        current_time = self.current_timestamp
        
        # Types of anomalies to inject
        anomaly_types = ["price_spike", "volume_spike", "spread_widening", "mini_flash_crash"]
        
        # Generate data for each period
        for _ in range(periods):
            # Update timestamp
            current_time += time_increment
            
            # Generate correlated random returns
            uncorrelated_returns = np.random.normal(0, 1, num_assets)
            correlated_returns = np.dot(cholesky, uncorrelated_returns)
            
            # Update prices based on correlated returns
            for i, symbol in enumerate(self.assets):
                volatility = volatilities.get(symbol, 0.01)
                return_pct = correlated_returns[i] * volatility
                current_prices[symbol] *= (1 + return_pct)
            
            # Market data for this period
            market_data = {}
            
            # Check for market-wide anomaly
            market_wide_anomaly = random.random() < anomaly_prob / 3
            market_wide_anomaly_type = None
            
            if market_wide_anomaly:
                market_wide_anomaly_type = random.choice(anomaly_types)
                logger.info(f"Injecting market-wide {market_wide_anomaly_type} at {current_time}")
                self.simulation_stats["anomalies_detected"] += 1
            
            # Generate data for each asset
            for symbol in self.assets:
                price = current_prices[symbol]
                
                # Determine if we inject an anomaly for this asset
                inject_anomaly = market_wide_anomaly or (random.random() < anomaly_prob)
                anomaly_type = market_wide_anomaly_type if market_wide_anomaly else (
                    random.choice(anomaly_types) if inject_anomaly else None
                )
                
                if inject_anomaly and not market_wide_anomaly:
                    logger.info(f"Injecting {anomaly_type} for {symbol} at {current_time}")
                    self.simulation_stats["anomalies_detected"] += 1
                
                # Generate OHLCV data with potential anomaly
                if anomaly_type == "price_spike":
                    # Sudden price jump
                    direction = random.choice([-1, 1])
                    factor = 1 + direction * random.uniform(0.02, 0.05)
                    price *= factor
                    
                    # Create OHLCV
                    open_price = price / factor  # Previous price
                    close_price = price
                    high_price = max(open_price, close_price) * 1.002
                    low_price = min(open_price, close_price) * 0.998
                    volume = random.lognormal(10, 1)
                    
                elif anomaly_type == "volume_spike":
                    # Unusual volume
                    volume_factor = random.uniform(5, 10)
                    
                    # Create OHLCV with normal price but high volume
                    open_price = price * (1 - 0.001 * random.random())
                    close_price = price
                    high_price = price * 1.002
                    low_price = price * 0.998
                    volume = random.lognormal(10, 1) * volume_factor
                    
                elif anomaly_type == "spread_widening":
                    # Bid-ask spread widens
                    # Create OHLCV with wider spread
                    open_price = price * (1 - 0.001 * random.random())
                    close_price = price
                    high_price = price * 1.002
                    low_price = price * 0.998
                    volume = random.lognormal(10, 1)
                    bid = price * 0.98  # 2% lower bid
                    ask = price * 1.02  # 2% higher ask
                    
                elif anomaly_type == "mini_flash_crash":
                    # Brief price drop
                    crash_factor = random.uniform(0.9, 0.95)
                    price *= crash_factor
                    
                    # Create OHLCV data for flash crash
                    open_price = price / crash_factor  # Previous price
                    close_price = price
                    high_price = open_price * 1.001
                    low_price = price * 0.99
                    volume = random.lognormal(10, 1) * 2  # Higher volume
                    
                else:
                    # Normal market data
                    open_price = price * (1 - 0.001 * random.random())
                    close_price = price
                    high_price = price * (1 + 0.001 + 0.001 * random.random())
                    low_price = price * (1 - 0.001 - 0.001 * random.random())
                    volume = random.lognormal(10, 1)
                
                # Create bid/ask if not already set
                if 'bid' not in locals():
                    bid = price * 0.999
                    ask = price * 1.001
                
                # Create bid/ask size
                bid_size = random.lognormal(8, 0.5)
                ask_size = random.lognormal(8, 0.5)
                
                # Create dataframe for this asset
                df = pd.DataFrame({
                    'open': [open_price],
                    'high': [high_price],
                    'low': [low_price],
                    'close': [close_price],
                    'volume': [volume],
                    'bid': [bid],
                    'ask': [ask],
                    'bid_size': [bid_size],
                    'ask_size': [ask_size]
                }, index=[current_time])
                
                # Store the data
                market_data[symbol] = df
                
                # Update current price after potential anomaly
                current_prices[symbol] = close_price
            
            # Update timestamp
            self.current_timestamp = current_time
            
            # Yield the market data for all assets
            yield market_data
    
    def update_data_buffers(self, market_data: Dict[str, pd.DataFrame]):
        """
        Update data buffers with new market data.
        
        Args:
            market_data: Dictionary of market data for each asset
        """
        for symbol, df in market_data.items():
            # Add to buffer
            self.data_buffers[symbol].append(df)
            
            # Convert buffer to dataframe
            buffer_df = pd.concat(list(self.data_buffers[symbol]))
            
            # Update adapter with latest data
            self.adapter.update_market_data(symbol, buffer_df)
    
    def run_anomaly_detection(self) -> Dict[str, Dict[str, Any]]:
        """
        Run anomaly detection for all assets.
        
        Returns:
            Dictionary of anomaly results for each asset
        """
        anomaly_results = {}
        
        for symbol, detector in self.anomaly_detectors.items():
            # Get data from buffer
            if len(self.data_buffers[symbol]) < self.lookback_window:
                # Not enough data for detection
                continue
            
            # Create dataframe from buffer
            buffer_df = pd.concat(list(self.data_buffers[symbol]))
            
            # Run detection
            try:
                result = detector.detect_anomalies(buffer_df)
                anomaly_results[symbol] = result
                
                # Store anomaly score
                self.anomaly_scores[symbol].append(result.get("latest_score", 0))
                
                # Log high anomaly scores
                if result.get("latest_score", 0) > detector.alert_threshold:
                    logger.warning(f"High anomaly score for {symbol}: {result.get('latest_score', 0):.4f}")
            except Exception as e:
                logger.error(f"Error in anomaly detection for {symbol}: {e}")
        
        return anomaly_results
    
    def update_risk_management(self, anomaly_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update risk management based on anomaly detection results.
        
        Args:
            anomaly_results: Dictionary of anomaly results for each asset
            
        Returns:
            Dictionary with risk management actions
        """
        # Use the highest anomaly score for risk management
        highest_anomaly = {"score": 0, "symbol": None, "result": None}
        
        for symbol, result in anomaly_results.items():
            score = result.get("latest_score", 0)
            if score > highest_anomaly["score"]:
                highest_anomaly = {"score": score, "symbol": symbol, "result": result}
        
        # If we found anomalies, update risk management
        if highest_anomaly["score"] > 0:
            try:
                risk_actions = self.risk_manager.update_risk_from_anomalies(highest_anomaly["result"])
                
                # Store risk level
                risk_status = self.risk_manager.get_anomaly_risk_status()
                self.risk_level_history.append({
                    "timestamp": self.current_timestamp,
                    "risk_level": risk_status.get("risk_level", "minimal"),
                    "anomaly_score": highest_anomaly["score"],
                    "symbol": highest_anomaly["symbol"]
                })
                
                # Check if trading should be disabled
                is_restricted = risk_status.get("trading_restricted", False)
                if is_restricted:
                    self.trading_enabled = False
                    logger.warning(f"Trading disabled due to high risk - level: {risk_status.get('risk_level', 'unknown')}")
                    self.simulation_stats["risk_events"] += 1
                
                return {
                    "actions": risk_actions,
                    "status": risk_status,
                    "highest_anomaly": highest_anomaly
                }
            except Exception as e:
                logger.error(f"Error updating risk management: {e}")
        
        return {"actions": {}, "status": {}, "highest_anomaly": highest_anomaly}
    
    def make_trading_decisions(self) -> List[Dict[str, Any]]:
        """
        Make trading decisions based on current data and risk status.
        
        Returns:
            List of trading decisions
        """
        if not self.trading_enabled:
            logger.info("Trading disabled due to risk management")
            return []
        
        decisions = []
        
        # Simple trading strategy for demonstration
        for symbol in self.assets:
            # Skip if not enough data
            if len(self.data_buffers[symbol]) < self.lookback_window:
                continue
            
            # Get latest data
            buffer_df = pd.concat(list(self.data_buffers[symbol]))
            latest_price = buffer_df['close'].iloc[-1]
            
            # Simple moving average strategy
            short_ma = buffer_df['close'].iloc[-5:].mean()
            long_ma = buffer_df['close'].iloc[-15:].mean() if len(buffer_df) >= 15 else short_ma
            
            # Check for existing position
            has_position = symbol in self.adapter.positions
            position_qty = self.adapter.positions.get(symbol, {}).get("quantity", 0)
            
            # Decision logic
            decision = {
                "symbol": symbol,
                "timestamp": self.current_timestamp,
                "current_price": latest_price,
                "short_ma": short_ma,
                "long_ma": long_ma,
                "has_position": has_position,
                "position_qty": position_qty,
                "action": "hold"
            }
            
            # Buy signal
            if short_ma > long_ma and not has_position:
                decision["action"] = "buy"
                decision["quantity"] = self._calculate_position_size(symbol, latest_price)
                decision["stop_loss"] = latest_price * 0.95
                decision["target"] = latest_price * 1.05
            
            # Sell signal
            elif short_ma < long_ma and has_position:
                decision["action"] = "sell"
                decision["quantity"] = position_qty
            
            decisions.append(decision)
        
        return decisions
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Asset symbol
            price: Current price
            
        Returns:
            Position size
        """
        # Get account equity
        equity = self.adapter.calculate_equity()
        
        # Risk per trade (1% of equity)
        risk_amount = equity * 0.01
        
        # Position size based on 5% stop loss
        stop_loss_pct = 0.05
        quantity = risk_amount / (price * stop_loss_pct)
        
        # Get position size modifier from risk manager
        if hasattr(self.risk_manager, 'active_risk_modifiers'):
            position_modifier = self.risk_manager.active_risk_modifiers.get("position_size_modifier", 1.0)
            quantity *= position_modifier
        
        # Round to 2 decimal places
        return round(quantity, 2)
    
    def execute_trading_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute trading decisions.
        
        Args:
            decisions: List of trading decisions
            
        Returns:
            List of execution results
        """
        results = []
        
        for decision in decisions:
            if decision["action"] == "buy":
                result = self.adapter.place_order(
                    symbol=decision["symbol"],
                    order_type="market",
                    quantity=decision["quantity"],
                    side="buy"
                )
                
                if result["status"] == "success":
                    self.simulation_stats["trades_executed"] += 1
                
                results.append({
                    "decision": decision,
                    "result": result
                })
                
            elif decision["action"] == "sell":
                result = self.adapter.place_order(
                    symbol=decision["symbol"],
                    order_type="market",
                    quantity=decision["quantity"],
                    side="sell"
                )
                
                if result["status"] == "success":
                    self.simulation_stats["trades_executed"] += 1
                
                results.append({
                    "decision": decision,
                    "result": result
                })
        
        return results
    
    def update_simulation_metrics(self):
        """Update simulation metrics."""
        # Update positions with latest prices
        self.adapter.update_positions()
        
        # Calculate current equity
        equity = self.adapter.calculate_equity()
        
        # Store in equity curve
        self.equity_curve.append({
            "timestamp": self.current_timestamp,
            "equity": equity
        })
    
    def display_simulation_status(self):
        """Display current simulation status."""
        console.print(Panel(f"Simulation Status - {self.current_timestamp}", style="cyan"))
        
        # Display account information
        account_info = self.adapter.get_account_info()
        
        account_table = Table(title="Account Information")
        account_table.add_column("Metric", style="cyan")
        account_table.add_column("Value", style="yellow")
        
        account_table.add_row("Balance", f"${account_info['balance']:.2f}")
        account_table.add_row("Equity", f"${account_info['equity']:.2f}")
        account_table.add_row("Open Positions", str(account_info['positions']))
        
        console.print(account_table)
        
        # Display positions if any
        position_summary = self.adapter.get_position_summary()
        if position_summary["count"] > 0:
            position_table = Table(title="Open Positions")
            position_table.add_column("Symbol", style="cyan")
            position_table.add_column("Quantity", style="yellow")
            position_table.add_column("Avg Price", style="yellow")
            position_table.add_column("Current Price", style="yellow")
            position_table.add_column("Value", style="yellow")
            position_table.add_column("Unrealized P&L", style="green")
            
            for position in position_summary["positions"]:
                pnl = position["unrealized_pnl"]
                pnl_style = "green" if pnl >= 0 else "red"
                
                position_table.add_row(
                    position["symbol"],
                    f"{position['quantity']:.2f}",
                    f"${position['avg_price']:.2f}",
                    f"${position['current_price']:.2f}",
                    f"${position['value']:.2f}",
                    f"[{pnl_style}]${pnl:.2f}[/{pnl_style}]"
                )
            
            console.print(position_table)
        
        # Display risk status
        risk_status = self.risk_manager.get_anomaly_risk_status()
        
        risk_table = Table(title="Risk Status")
        risk_table.add_column("Parameter", style="cyan")
        risk_table.add_column("Value", style="yellow")
        
        # Add risk status details
        for key, value in risk_status.items():
            if key == "anomaly_score":
                risk_table.add_row("Anomaly Score", f"{value:.4f}")
            elif key == "risk_level":
                style = "green"
                if value == "moderate":
                    style = "yellow"
                elif value == "high":
                    style = "orange3"
                elif value == "critical":
                    style = "red"
                risk_table.add_row("Risk Level", f"[{style}]{value}[/{style}]")
            elif key == "in_cooldown":
                risk_table.add_row("In Cooldown", "Yes" if value else "No")
            elif key == "cooldown_end" and value:
                remaining = value - datetime.now()
                minutes = remaining.total_seconds() / 60
                risk_table.add_row("Cooldown Remaining", f"{minutes:.1f} minutes")
            elif key == "position_size_modifier":
                risk_table.add_row("Position Size Modifier", f"{value:.2f} ({(1-value)*100:.0f}% reduction)")
            elif key == "trading_restricted":
                status = "[red]Restricted[/red]" if value else "[green]Allowed[/green]"
                risk_table.add_row("Trading Status", status)
        
        console.print(risk_table)
        
        # Display summary stats
        stats_table = Table(title="Simulation Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        for key, value in self.simulation_stats.items():
            stats_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(stats_table)
    
    def plot_simulation_results(self):
        """Plot simulation results."""
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        if self.equity_curve:
            dates = [entry["timestamp"] for entry in self.equity_curve]
            equity = [entry["equity"] for entry in self.equity_curve]
            
            ax1.plot(dates, equity, label="Equity", color="blue")
            ax1.set_title("Paper Trading Simulation Results")
            ax1.set_ylabel("Equity ($)")
            ax1.legend()
            ax1.grid(True)
        
        # Plot anomaly scores
        if any(self.anomaly_scores.values()):
            for symbol, scores in self.anomaly_scores.items():
                if scores:
                    # Use the same dates as equity curve but truncate to match scores length
                    score_dates = dates[:len(scores)]
                    ax2.plot(score_dates, scores, label=f"{symbol} Anomaly Score", alpha=0.7)
            
            # Plot threshold line
            if score_dates:
                ax2.axhline(0.75, color="red", linestyle="--", label="Alert Threshold")
            
            ax2.set_ylabel("Anomaly Score")
            ax2.legend()
            ax2.grid(True)
        
        # Plot risk levels
        if self.risk_level_history:
            risk_dates = [entry["timestamp"] for entry in self.risk_level_history]
            
            # Convert risk levels to numeric
            risk_level_map = {"minimal": 0, "moderate": 1, "high": 2, "critical": 3}
            risk_levels = [risk_level_map.get(entry["risk_level"], 0) for entry in self.risk_level_history]
            
            # Plot risk levels
            ax3.step(risk_dates, risk_levels, label="Risk Level", color="purple", where="post")
            
            # Add risk level labels
            ax3.set_yticks(list(risk_level_map.values()))
            ax3.set_yticklabels(list(risk_level_map.keys()))
            
            ax3.set_ylabel("Risk Level")
            ax3.set_xlabel("Date")
            ax3.legend()
            ax3.grid(True)
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save figure
        filename = f"output/paper_trading_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        
        console.print(f"[green]Simulation results saved to {filename}[/green]")
        
        # Show figure
        plt.show()
    
    def run_simulation(self, 
                     periods: int = 100, 
                     anomaly_prob: float = 0.05, 
                     display_interval: int = 10):
        """
        Run the paper trading simulation.
        
        Args:
            periods: Number of periods to simulate
            anomaly_prob: Probability of anomaly injection
            display_interval: How often to display status
        """
        console.print(Panel("Starting Paper Trading Simulation", style="bold blue"))
        
        # Train anomaly detectors
        console.print(Panel("Training anomaly detectors...", style="cyan"))
        
        # Generate initial data for training
        training_periods = 50
        training_data = {}
        
        for market_data in self.generate_market_data(training_periods, anomaly_prob=0):
            for symbol, df in market_data.items():
                if symbol not in training_data:
                    training_data[symbol] = []
                training_data[symbol].append(df)
        
        # Train each detector
        for symbol, detector in self.anomaly_detectors.items():
            if symbol in training_data:
                training_df = pd.concat(training_data[symbol])
                try:
                    detector.train(training_df)
                    logger.info(f"Trained anomaly detector for {symbol}")
                except Exception as e:
                    logger.error(f"Error training detector for {symbol}: {e}")
        
        # Run simulation
        console.print(Panel("Running paper trading simulation", style="cyan"))
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Simulating...", total=periods)
            
            for i, market_data in enumerate(self.generate_market_data(periods, anomaly_prob)):
                # Update data buffers
                self.update_data_buffers(market_data)
                
                # Run anomaly detection
                anomaly_results = self.run_anomaly_detection()
                
                # Update risk management
                risk_update = self.update_risk_management(anomaly_results)
                
                # Make trading decisions
                decisions = self.make_trading_decisions()
                
                # Execute trading decisions
                execution_results = self.execute_trading_decisions(decisions)
                
                # Update simulation metrics
                self.update_simulation_metrics()
                
                # Display status at intervals
                if i % display_interval == 0:
                    self.display_simulation_status()
                
                # Update progress
                progress.update(task, advance=1)
        
        # Final status display
        self.display_simulation_status()
        
        # Plot results
        self.plot_simulation_results()
        
        console.print(Panel("Paper Trading Simulation Complete", style="bold green"))

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Paper Trading Simulation with Anomaly Risk Management")
    parser.add_argument("--assets", type=str, nargs="+", default=["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"],
                      help="Asset symbols to simulate")
    parser.add_argument("--periods", type=int, default=100, 
                      help="Number of periods to simulate")
    parser.add_argument("--anomaly-prob", type=float, default=0.05,
                      help="Probability of anomaly injection")
    parser.add_argument("--display-interval", type=int, default=10,
                      help="How often to display status")
    args = parser.parse_args()
    
    try:
        # Initialize simulator
        simulator = PaperTradingSimulator(assets=args.assets)
        
        # Run simulation
        simulator.run_simulation(
            periods=args.periods,
            anomaly_prob=args.anomaly_prob,
            display_interval=args.display_interval
        )
    except KeyboardInterrupt:
        console.print("[yellow]Simulation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error in simulation: {e}[/bold red]")
        import traceback
        traceback.print_exc()
    
    console.print("\n[bold green]Program completed[/bold green]")

if __name__ == "__main__":
    main() 