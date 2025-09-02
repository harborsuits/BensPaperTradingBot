#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Generation Engine

This module provides a comprehensive strategy generation engine that can:
1. Create different types of trading strategies
2. Optimize strategy parameters
3. Detect stock characteristics suitable for each strategy
4. Manage collections of strategies
5. Create ensemble strategies
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Type, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import uuid

from trading_bot.strategies.strategy_template import (
    StrategyTemplate,
    StrategyOptimizable,
    StrategyEnsemble,
    StrategyMetadata,
    Signal,
    SignalType,
    TimeFrame,
    MarketRegime
)

# Import strategy implementations
from trading_bot.strategies.mean_reversion_template import MeanReversionTemplate
from trading_bot.strategies.trend_following_template import TrendFollowingTemplate
from trading_bot.strategies.breakout_template import BreakoutTemplate

# Import optimization tools
from trading_bot.optimization import (
    ParameterOptimizer, 
    OptimizationMethod, 
    ParameterType
)

# Setup logging
logger = logging.getLogger(__name__)

class StrategyGenerator:
    """
    Strategy Generation Engine responsible for creating, optimizing, and managing trading strategies.
    
    Features:
    - Strategy creation from templates
    - Parameter optimization using grid search and other methods
    - Stock characteristic detection for strategy matching
    - Ensemble strategy creation
    - Strategy serialization and persistence
    - Strategy performance evaluation
    """
    
    def __init__(
        self,
        base_dir: str = "strategies",
        optimization_dir: str = "optimization_results",
        default_optimization_method: str = "grid_search",
        max_optimization_iterations: int = 100
    ):
        """
        Initialize the strategy generator.
        
        Args:
            base_dir: Base directory for strategy storage
            optimization_dir: Directory for optimization results
            default_optimization_method: Default optimization method
            max_optimization_iterations: Maximum optimization iterations
        """
        self.base_dir = base_dir
        self.optimization_dir = optimization_dir
        self.default_optimization_method = default_optimization_method
        self.max_optimization_iterations = max_optimization_iterations
        
        # Create directories if they don't exist
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(optimization_dir, exist_ok=True)
        
        # Initialize registry of available strategy types
        self.strategy_registry = {
            "mean_reversion": MeanReversionTemplate,
            "trend_following": TrendFollowingTemplate,
            "breakout": BreakoutTemplate,
            "ensemble": StrategyEnsemble
        }
        
        # Initialize dict to track created strategies
        self.strategies: Dict[str, StrategyTemplate] = {}
        
        logger.info(f"Initialized StrategyGenerator with {len(self.strategy_registry)} available strategy types")
    
    def create_strategy(
        self,
        strategy_type: str,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StrategyTemplate:
        """
        Create a new strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            name: Strategy name (if None, auto-generated)
            parameters: Strategy parameters
            metadata: Strategy metadata
            
        Returns:
            Strategy instance
        """
        # Check if strategy type is supported
        if strategy_type not in self.strategy_registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}. Available types: {list(self.strategy_registry.keys())}")
        
        # Generate name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = str(uuid.uuid4())[:6]
            name = f"{strategy_type}_{timestamp}_{random_suffix}"
        
        # Create metadata object if provided as dict
        strategy_metadata = None
        if metadata:
            # Convert timeframes if provided as strings
            if "timeframes" in metadata:
                timeframes = []
                for tf in metadata["timeframes"]:
                    if isinstance(tf, str):
                        timeframes.append(TimeFrame(tf))
                    else:
                        timeframes.append(tf)
                metadata["timeframes"] = timeframes
            
            # Convert dates if provided as strings
            for date_field in ["created_at", "updated_at"]:
                if date_field in metadata and isinstance(metadata[date_field], str):
                    metadata[date_field] = datetime.fromisoformat(metadata[date_field])
            
            strategy_metadata = StrategyMetadata(**metadata)
        
        # Create strategy instance
        strategy_class = self.strategy_registry[strategy_type]
        strategy = strategy_class(
            name=name,
            parameters=parameters,
            metadata=strategy_metadata
        )
        
        # Store in strategies dict
        self.strategies[name] = strategy
        
        logger.info(f"Created {strategy_type} strategy: {name}")
        return strategy
    
    def optimize_strategy(
        self,
        strategy: Union[str, StrategyOptimizable],
        market_data: Dict[str, pd.DataFrame],
        evaluation_func: Optional[Callable] = None,
        parameter_space: Optional[Dict[str, List[Any]]] = None,
        method: Optional[str] = None,
        max_iterations: Optional[int] = None,
        cv_folds: int = 1,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy: Strategy instance or name
            market_data: Historical market data for optimization
            evaluation_func: Function to evaluate parameter sets
            parameter_space: Parameter space to optimize (None = use strategy's space)
            method: Optimization method
            max_iterations: Maximum optimization iterations
            cv_folds: Cross-validation folds
            save_results: Whether to save optimization results
            
        Returns:
            Optimization results
        """
        # Get strategy instance if name was provided
        if isinstance(strategy, str):
            if strategy not in self.strategies:
                raise ValueError(f"Strategy not found: {strategy}")
            strategy_instance = self.strategies[strategy]
        else:
            strategy_instance = strategy
        
        # Check if strategy is optimizable
        if not isinstance(strategy_instance, StrategyOptimizable):
            raise TypeError(f"Strategy {strategy_instance.name} is not optimizable")
        
        # Set defaults if not provided
        method = method or self.default_optimization_method
        max_iterations = max_iterations or self.max_optimization_iterations
        
        # Run optimization
        optimization_result = strategy_instance.optimize(
            data=market_data,
            evaluation_func=evaluation_func,
            parameter_space=parameter_space,
            method=method,
            max_iterations=max_iterations,
            cv_folds=cv_folds,
            verbose=True
        )
        
        # Save optimization results if requested
        if save_results:
            self._save_optimization_results(strategy_instance.name, optimization_result)
        
        return optimization_result
    
    def _save_optimization_results(self, strategy_name: str, results: Dict[str, Any]) -> str:
        """
        Save optimization results to file.
        
        Args:
            strategy_name: Name of the strategy
            results: Optimization results
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_optimization_{timestamp}.json"
        filepath = os.path.join(self.optimization_dir, filename)
        
        # Convert any non-serializable values to strings
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = str(value) if not isinstance(value, (int, float, str, bool, list, dict, type(None))) else value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved optimization results to {filepath}")
        return filepath
    
    def detect_characteristics(
        self,
        market_data: Dict[str, pd.DataFrame],
        strategy_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Detect stock characteristics suitable for different strategy types.
        
        Args:
            market_data: Historical market data
            strategy_types: List of strategy types to check (None = all)
            
        Returns:
            Dictionary of symbol -> strategy type -> characteristics
        """
        if strategy_types is None:
            strategy_types = ["mean_reversion", "trend_following", "breakout"]
        
        results = {}
        
        # Create temporary instances of each strategy type
        strategy_instances = {}
        for strategy_type in strategy_types:
            if strategy_type in self.strategy_registry:
                strategy_class = self.strategy_registry[strategy_type]
                strategy_instances[strategy_type] = strategy_class(name=f"temp_{strategy_type}")
        
        # Detect characteristics for each symbol
        for symbol in market_data.keys():
            symbol_data = {symbol: market_data[symbol]}
            results[symbol] = {}
            
            for strategy_type, strategy in strategy_instances.items():
                # Only if strategy has characteristic detection
                if hasattr(strategy, "detect_stock_characteristics"):
                    try:
                        characteristics = strategy.detect_stock_characteristics(symbol_data)
                        if symbol in characteristics:
                            results[symbol][strategy_type] = characteristics[symbol]
                    except Exception as e:
                        logger.error(f"Error detecting {strategy_type} characteristics for {symbol}: {e}")
        
        return results
    
    def analyze_strategy_suitability(
        self,
        characteristics: Dict[str, Dict[str, Dict[str, Any]]],
        threshold: float = 0.6
    ) -> Dict[str, List[str]]:
        """
        Analyze which strategy types are best suited for each symbol.
        
        Args:
            characteristics: Output from detect_characteristics
            threshold: Minimum suitability score threshold
            
        Returns:
            Dictionary of symbol -> list of suitable strategy types
        """
        recommendations = {}
        
        for symbol, strategies in characteristics.items():
            suitable_strategies = []
            
            for strategy_type, props in strategies.items():
                # Check if strategy is suitable
                is_suitable = props.get("suitable_for_strategy", False)
                if is_suitable:
                    suitable_strategies.append(strategy_type)
            
            recommendations[symbol] = suitable_strategies
        
        return recommendations
    
    def create_ensemble(
        self,
        name: str,
        strategies: List[Union[str, StrategyTemplate]],
        weights: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StrategyEnsemble:
        """
        Create an ensemble of multiple strategies.
        
        Args:
            name: Ensemble name
            strategies: List of strategy instances or names
            weights: Dictionary of strategy name -> weight
            parameters: Ensemble parameters
            metadata: Ensemble metadata
            
        Returns:
            StrategyEnsemble instance
        """
        # Resolve strategy names to instances
        strategy_instances = []
        for strategy in strategies:
            if isinstance(strategy, str):
                if strategy not in self.strategies:
                    raise ValueError(f"Strategy not found: {strategy}")
                strategy_instances.append(self.strategies[strategy])
            else:
                strategy_instances.append(strategy)
        
        # Create strategy metadata if provided as dict
        strategy_metadata = None
        if metadata:
            strategy_metadata = StrategyMetadata(**metadata)
        
        # Create ensemble
        ensemble = StrategyEnsemble(
            name=name,
            strategies=strategy_instances,
            weights=weights,
            parameters=parameters,
            metadata=strategy_metadata
        )
        
        # Store in strategies dict
        self.strategies[name] = ensemble
        
        logger.info(f"Created ensemble strategy {name} with {len(strategy_instances)} strategies")
        return ensemble
    
    def save_strategy(self, strategy_name: str) -> str:
        """
        Save a strategy to file.
        
        Args:
            strategy_name: Name of strategy to save
            
        Returns:
            Path to saved strategy file
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.save(directory=self.base_dir)
    
    def load_strategy(self, strategy_path: str) -> StrategyTemplate:
        """
        Load a strategy from file.
        
        Args:
            strategy_path: Path to strategy file
            
        Returns:
            Loaded strategy instance
        """
        # Load strategy configuration
        with open(strategy_path, 'r') as f:
            config = json.load(f)
        
        # Determine strategy type
        strategy_type = config.get("type")
        if not strategy_type or strategy_type not in self.strategy_registry:
            raise ValueError(f"Unknown or missing strategy type in file: {strategy_path}")
        
        # Load using appropriate strategy class
        strategy_class = self.strategy_registry[strategy_type]
        strategy = strategy_class.load(strategy_path)
        
        # Store in strategies dict
        self.strategies[strategy.name] = strategy
        
        logger.info(f"Loaded {strategy_type} strategy: {strategy.name}")
        return strategy
    
    def delete_strategy(self, strategy_name: str) -> bool:
        """
        Delete a strategy from the generator.
        
        Args:
            strategy_name: Name of strategy to delete
            
        Returns:
            True if deleted, False if not found
        """
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Deleted strategy: {strategy_name}")
            return True
        
        logger.warning(f"Strategy not found: {strategy_name}")
        return False
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        List all strategies managed by this generator.
        
        Returns:
            List of strategy information dictionaries
        """
        strategy_info = []
        
        for name, strategy in self.strategies.items():
            # Get strategy type
            strategy_type = strategy.__class__.__name__
            
            # Basic info
            info = {
                "name": name,
                "type": strategy_type,
                "parameters_count": len(strategy.parameters) if hasattr(strategy, "parameters") else 0,
                "optimizable": isinstance(strategy, StrategyOptimizable),
                "is_ensemble": isinstance(strategy, StrategyEnsemble)
            }
            
            # Add metadata if available
            if hasattr(strategy, "metadata"):
                info["metadata"] = {
                    "version": strategy.metadata.version,
                    "description": strategy.metadata.description,
                    "author": strategy.metadata.author,
                    "created_at": strategy.metadata.created_at.isoformat(),
                    "timeframes": [tf.value for tf in strategy.metadata.timeframes],
                    "asset_classes": strategy.metadata.asset_classes
                }
            
            strategy_info.append(info)
        
        return strategy_info

    def generate_backtest_report(
        self,
        strategy: Union[str, StrategyTemplate],
        market_data: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        position_sizing: str = "equal",
        risk_per_trade: float = 0.01
    ) -> Dict[str, Any]:
        """
        Generate a backtest report for a strategy.
        
        Args:
            strategy: Strategy instance or name
            market_data: Historical market data
            initial_capital: Initial capital for backtest
            position_sizing: Position sizing method ('equal', 'risk_parity', 'kelly')
            risk_per_trade: Risk per trade (as fraction of capital)
            
        Returns:
            Backtest report
        """
        # Get strategy instance if name was provided
        if isinstance(strategy, str):
            if strategy not in self.strategies:
                raise ValueError(f"Strategy not found: {strategy}")
            strategy_instance = self.strategies[strategy]
        else:
            strategy_instance = strategy
        
        # Simple backtest simulation
        signals = {}
        trades = []
        capital = initial_capital
        equity_curve = [capital]
        positions = {}
        
        # Get signals for all time periods
        time_periods = sorted(set(idx for df in market_data.values() for idx in df.index))
        
        for time in time_periods:
            # Prepare data up to this time point
            current_data = {}
            for symbol, df in market_data.items():
                # Skip if this symbol doesn't have data at this time
                if time not in df.index:
                    continue
                
                # Get data up to this point
                current_data[symbol] = df.loc[:time]
            
            # Skip if no data
            if not current_data:
                continue
            
            # Generate signals
            time_signals = strategy_instance.generate_signals(current_data)
            signals[time] = time_signals
            
            # Process signals
            for symbol, signal in time_signals.items():
                # Skip if symbol not in data
                if symbol not in current_data:
                    continue
                
                # Process buy signals
                if signal.signal_type == SignalType.BUY and symbol not in positions:
                    # Calculate position size
                    price = signal.price
                    size = capital * risk_per_trade / price
                    
                    # Open position
                    positions[symbol] = {
                        "entry_price": price,
                        "entry_time": time,
                        "size": size,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit
                    }
                    
                    # Record trade
                    trades.append({
                        "symbol": symbol,
                        "entry_time": time,
                        "entry_price": price,
                        "size": size,
                        "direction": "long",
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit
                    })
                
                # Process sell signals
                elif signal.signal_type == SignalType.SELL and symbol in positions:
                    # Close position
                    position = positions[symbol]
                    exit_price = signal.price
                    profit = (exit_price - position["entry_price"]) * position["size"]
                    
                    # Update capital
                    capital += profit
                    
                    # Update trade record
                    for trade in reversed(trades):
                        if trade["symbol"] == symbol and "exit_time" not in trade:
                            trade["exit_time"] = time
                            trade["exit_price"] = exit_price
                            trade["profit"] = profit
                            trade["pnl_pct"] = (exit_price / position["entry_price"] - 1) * 100
                            break
                    
                    # Remove from positions
                    del positions[symbol]
            
            # Check stop loss and take profit
            for symbol, position in list(positions.items()):
                if symbol not in current_data:
                    continue
                
                current_price = current_data[symbol].iloc[-1]["close"]
                
                # Check stop loss
                if position["stop_loss"] and (
                    (position["entry_price"] > position["stop_loss"] and current_price <= position["stop_loss"]) or
                    (position["entry_price"] < position["stop_loss"] and current_price >= position["stop_loss"])
                ):
                    # Stop loss hit
                    exit_price = position["stop_loss"]
                    profit = (exit_price - position["entry_price"]) * position["size"]
                    
                    # Update capital
                    capital += profit
                    
                    # Update trade record
                    for trade in reversed(trades):
                        if trade["symbol"] == symbol and "exit_time" not in trade:
                            trade["exit_time"] = time
                            trade["exit_price"] = exit_price
                            trade["profit"] = profit
                            trade["pnl_pct"] = (exit_price / position["entry_price"] - 1) * 100
                            trade["exit_reason"] = "stop_loss"
                            break
                    
                    # Remove from positions
                    del positions[symbol]
                    continue
                
                # Check take profit
                if position["take_profit"] and (
                    (position["entry_price"] < position["take_profit"] and current_price >= position["take_profit"]) or
                    (position["entry_price"] > position["take_profit"] and current_price <= position["take_profit"])
                ):
                    # Take profit hit
                    exit_price = position["take_profit"]
                    profit = (exit_price - position["entry_price"]) * position["size"]
                    
                    # Update capital
                    capital += profit
                    
                    # Update trade record
                    for trade in reversed(trades):
                        if trade["symbol"] == symbol and "exit_time" not in trade:
                            trade["exit_time"] = time
                            trade["exit_price"] = exit_price
                            trade["profit"] = profit
                            trade["pnl_pct"] = (exit_price / position["entry_price"] - 1) * 100
                            trade["exit_reason"] = "take_profit"
                            break
                    
                    # Remove from positions
                    del positions[symbol]
            
            # Update equity curve
            equity_curve.append(capital)
        
        # Close any remaining positions using last price
        for symbol, position in list(positions.items()):
            if symbol in market_data:
                exit_price = market_data[symbol].iloc[-1]["close"]
                profit = (exit_price - position["entry_price"]) * position["size"]
                
                # Update capital
                capital += profit
                
                # Update trade record
                for trade in reversed(trades):
                    if trade["symbol"] == symbol and "exit_time" not in trade:
                        trade["exit_time"] = time_periods[-1]
                        trade["exit_price"] = exit_price
                        trade["profit"] = profit
                        trade["pnl_pct"] = (exit_price / position["entry_price"] - 1) * 100
                        trade["exit_reason"] = "end_of_period"
                        break
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("profit", 0) > 0)
        losing_trades = sum(1 for t in trades if t.get("profit", 0) < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = sum(t.get("profit", 0) for t in trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(abs(t.get("profit", 0)) for t in trades if t.get("profit", 0) < 0) / losing_trades if losing_trades > 0 else 0
        
        profit_factor = avg_win * winning_trades / (avg_loss * losing_trades) if losing_trades > 0 and avg_loss > 0 else float('inf')
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        total_return = (equity_array[-1] / equity_array[0]) - 1
        annualized_return = ((1 + total_return) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
        
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = 1 - equity_array / peak
        max_drawdown = drawdown.max()
        
        # Create report
        report = {
            "strategy_name": strategy_instance.name,
            "initial_capital": initial_capital,
            "final_capital": capital,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "avg_profit": avg_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "equity_curve": equity_curve,
            "trades": trades
        }
        
        return report 