#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Adapter Module

This module provides adapters that standardize the interface between various 
strategy implementations and the autonomous engine. It ensures all strategies
implement the required methods for generation, sizing, and management.
"""

import logging
from typing import Dict, List, Any, Union, Optional, Type, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategy templates and base classes
from trading_bot.strategies.strategy_template import StrategyTemplate, StrategyOptimizable
from trading_bot.strategies.base_strategy import Strategy, SignalType

# Import specific strategy classes for type checking
try:
    from trading_bot.strategies.iron_condor_strategy import IronCondorStrategy
    from trading_bot.strategies.strangle_strategy import StrangleStrategy
    # Add other strategy imports as needed
except ImportError:
    pass

logger = logging.getLogger(__name__)

class StrategyAdapter:
    """
    Adapter that provides a standardized interface to various strategy implementations.
    
    This adapter ensures all strategies expose the required methods:
    - generate_signals
    - size_position
    - manage_open_trades
    
    It handles translation between different strategy paradigms and the expected
    interface for the autonomous engine.
    """
    
    def __init__(self, strategy_instance, strategy_type: str = None):
        """
        Initialize the adapter with a strategy instance.
        
        Args:
            strategy_instance: Instance of a strategy class
            strategy_type: Type of strategy (optional, inferred if not provided)
        """
        self.strategy = strategy_instance
        self.strategy_type = strategy_type or self._infer_strategy_type()
        self.metadata = {}
        
        logger.info(f"Created strategy adapter for {self.strategy_type}")
    
    def _infer_strategy_type(self) -> str:
        """Infer the strategy type from the instance class name."""
        class_name = self.strategy.__class__.__name__
        # Remove 'Strategy' suffix if present
        if class_name.endswith('Strategy'):
            return class_name[:-8]
        return class_name
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Dictionary containing market data
                - OHLCV data frames
                - Option chains if applicable
                - Other market data
                
        Returns:
            Dictionary with generated signals
        """
        # Route to appropriate implementation based on strategy type
        if hasattr(self.strategy, 'generate_signals'):
            # Direct call if method exists
            return self.strategy.generate_signals(data)
        
        # For strategies using the Strategy base class
        if isinstance(self.strategy, Strategy):
            current_time = data.get('current_time', pd.Timestamp.now())
            ohlcv_data = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
            return self.strategy.generate_signals(ohlcv_data, current_time)
            
        # For IronCondorStrategy and similar options strategies
        if isinstance(self.strategy, (IronCondorStrategy, StrangleStrategy)):
            # Options strategies may need different data formats
            market_data = data.get('market_data')
            option_chains = data.get('option_chains')
            
            # Check if strategy meets selection criteria
            symbols = self.strategy.define_universe(market_data)
            signals = {}
            
            for symbol in symbols:
                if self.strategy.check_selection_criteria(symbol, market_data, option_chains):
                    # Select option contracts
                    contracts = self.strategy.select_option_contract(symbol, market_data, option_chains)
                    if contracts:
                        signals[symbol] = contracts
            
            return signals
        
        # If no matching implementation, log warning and return empty
        logger.warning(f"No signal generation implementation found for {self.strategy_type}")
        return {}
    
    def size_position(self, symbol: str, signal: Dict[str, Any], 
                     account_size: float, risk_params: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size based on signal and risk parameters.
        
        Args:
            symbol: Trading symbol
            signal: Signal information
            account_size: Current account size
            risk_params: Risk management parameters
            
        Returns:
            Position size (units, contracts, or capital amount)
        """
        # Direct call if method exists
        if hasattr(self.strategy, 'size_position'):
            return self.strategy.size_position(symbol, signal, account_size, risk_params)
            
        # For Strategy base class
        if isinstance(self.strategy, Strategy):
            # Convert signal to SignalType if needed
            signal_type = signal.get('signal_type', SignalType.FLAT)
            price = signal.get('price', 0.0)
            volatility = signal.get('volatility', risk_params.get('volatility', 0.01))
            return self.strategy.calculate_position_size(symbol, signal_type, price, volatility, account_size)
            
        # For options strategies
        if isinstance(self.strategy, (IronCondorStrategy, StrangleStrategy)):
            position_sizer = risk_params.get('position_sizer')
            if position_sizer and hasattr(self.strategy, 'calculate_position_size'):
                return self.strategy.calculate_position_size(signal, position_sizer)
            
        # Default implementation based on fixed percentage
        risk_pct = risk_params.get('risk_percent', 0.02)  # Default 2% risk
        return account_size * risk_pct
    
    def manage_open_trades(self, positions: Dict[str, Any], 
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage existing open positions.
        
        Args:
            positions: Current open positions
            market_data: Current market data
            
        Returns:
            Dictionary with management decisions (exits, adjustments, etc.)
        """
        # Direct call if method exists
        if hasattr(self.strategy, 'manage_open_trades'):
            return self.strategy.manage_open_trades(positions, market_data)
            
        # For Strategy base class
        if isinstance(self.strategy, Strategy):
            current_time = market_data.get('current_time', pd.Timestamp.now())
            ohlcv_data = {k: v for k, v in market_data.items() if isinstance(v, pd.DataFrame)}
            account_size = market_data.get('account_size', 10000.0)
            return self.strategy.update(ohlcv_data, current_time, account_size)
            
        # For options strategies
        if isinstance(self.strategy, (IronCondorStrategy, StrangleStrategy)):
            # Check exit conditions for each position
            management_decisions = {}
            
            for position_id, position in positions.items():
                symbol = position.get('symbol')
                
                # Check if we should exit
                if hasattr(self.strategy, 'check_exit_conditions'):
                    should_exit = self.strategy.check_exit_conditions(position, market_data)
                    if should_exit:
                        exit_orders = []
                        if hasattr(self.strategy, 'prepare_exit_orders'):
                            exit_orders = self.strategy.prepare_exit_orders(position)
                        management_decisions[position_id] = {
                            'action': 'exit',
                            'orders': exit_orders
                        }
                    else:
                        # Check if we should adjust
                        if hasattr(self.strategy, 'prepare_adjustment_orders'):
                            option_chains = market_data.get('option_chains')
                            adjustment_orders = self.strategy.prepare_adjustment_orders(
                                position, market_data, option_chains
                            )
                            if adjustment_orders:
                                management_decisions[position_id] = {
                                    'action': 'adjust',
                                    'orders': adjustment_orders
                                }
            
            return management_decisions
        
        # Default implementation - simple stop-loss and take-profit
        management_decisions = {}
        for position_id, position in positions.items():
            symbol = position.get('symbol')
            entry_price = position.get('entry_price', 0.0)
            direction = position.get('direction', 'long')
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')
            
            # Get current price
            current_price = 0.0
            if symbol in market_data:
                if isinstance(market_data[symbol], pd.DataFrame):
                    current_price = market_data[symbol].iloc[-1]['close']
                else:
                    current_price = market_data[symbol].get('price', 0.0)
            
            # Check stop loss
            if stop_loss and ((direction == 'long' and current_price <= stop_loss) or
                             (direction == 'short' and current_price >= stop_loss)):
                management_decisions[position_id] = {
                    'action': 'exit',
                    'reason': 'stop_loss',
                    'price': current_price
                }
            
            # Check take profit
            if take_profit and ((direction == 'long' and current_price >= take_profit) or
                               (direction == 'short' and current_price <= take_profit)):
                management_decisions[position_id] = {
                    'action': 'exit',
                    'reason': 'take_profit',
                    'price': current_price
                }
        
        return management_decisions
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters for optimization."""
        if hasattr(self.strategy, 'get_parameters'):
            return self.strategy.get_parameters()
            
        if hasattr(self.strategy, 'parameters'):
            return self.strategy.parameters
            
        if isinstance(self.strategy, StrategyOptimizable) and hasattr(self.strategy, 'get_parameter_space'):
            return {param: values[0] if values else None 
                    for param, values in self.strategy.get_parameter_space().items()}
        
        if hasattr(self.strategy, 'DEFAULT_PARAMS'):
            return self.strategy.DEFAULT_PARAMS.copy()
            
        # Empty parameters if nothing found
        logger.warning(f"No parameters found for {self.strategy_type}")
        return {}
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters for optimization."""
        if hasattr(self.strategy, 'set_parameters'):
            self.strategy.set_parameters(parameters)
        elif hasattr(self.strategy, 'parameters'):
            # Directly update parameters if accessible
            if isinstance(self.strategy.parameters, dict):
                self.strategy.parameters.update(parameters)
        else:
            # Set parameters individually if accessible
            for param, value in parameters.items():
                if hasattr(self.strategy, param):
                    setattr(self.strategy, param, value)
                    
    def get_optimization_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters that can be optimized with their ranges."""
        if isinstance(self.strategy, StrategyOptimizable) and hasattr(self.strategy, 'get_optimization_params'):
            return self.strategy.get_optimization_params()
            
        if hasattr(self.strategy, 'get_parameter_space'):
            param_space = self.strategy.get_parameter_space()
            # Convert to format expected by optimizer
            return {param: {'type': 'choice', 'values': values} for param, values in param_space.items()}
        
        # Default optimization parameters
        return {
            'risk_percent': {'type': 'float', 'min': 0.01, 'max': 0.05, 'step': 0.01},
            'stop_loss_atr': {'type': 'float', 'min': 1.0, 'max': 3.0, 'step': 0.5},
            'take_profit_atr': {'type': 'float', 'min': 1.0, 'max': 5.0, 'step': 0.5},
        }
        
    def backtest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest with current parameters."""
        if hasattr(self.strategy, 'backtest'):
            return self.strategy.backtest(data)
            
        # Simple backtest implementation
        logger.info(f"Running simplified backtest for {self.strategy_type}")
        
        results = {
            'trades': [],
            'equity_curve': [],
            'performance': {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0,
                'num_trades': 0
            }
        }
        
        # Get performance metrics if available
        if hasattr(self.strategy, 'get_performance_metrics'):
            metrics = self.strategy.get_performance_metrics()
            if metrics:
                results['performance'].update(metrics)
        
        return results

# Factory function to create adapter for any strategy type
def create_strategy_adapter(strategy_instance_or_class, *args, **kwargs) -> StrategyAdapter:
    """
    Create a strategy adapter for the given strategy instance or class.
    
    Args:
        strategy_instance_or_class: Strategy instance or class
        *args, **kwargs: Arguments to pass to strategy constructor if class provided
        
    Returns:
        StrategyAdapter instance
    """
    # If a class is provided, instantiate it
    if isinstance(strategy_instance_or_class, type):
        strategy_instance = strategy_instance_or_class(*args, **kwargs)
    else:
        strategy_instance = strategy_instance_or_class
        
    # Create and return adapter
    return StrategyAdapter(strategy_instance)
