#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Template for BensBot Trading System

This template defines the standard structure that all strategies should follow
to ensure proper integration with the backtester, strategy finder, and live trading.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from trading_bot.core.events import Event, EventType
from trading_bot.market.market_data import MarketData
from trading_bot.market.universe import Universe

logger = logging.getLogger(__name__)

class StrategyTemplate(ABC):
    """
    Base template for all trading strategies that ensures consistent
    interface for the backtester, strategy finder, and live trading systems.
    
    All strategies should inherit from this template and implement the 
    required methods with the same signatures.
    """
    
    # Default parameters shared by all strategies
    DEFAULT_PARAMS = {
        'strategy_name': 'template_strategy',
        'strategy_version': '1.0.0',
        'asset_class': 'multi_asset',    # Options: forex, stocks, options, crypto, multi_asset
        'strategy_type': 'all_weather',  # See StrategyType enum in registry
        'timeframe': 'daily',            # Options: scalping, intraday, swing, position, multi_timeframe
        'market_regime': 'all_weather',  # Options: trending, ranging, volatile, low_volatility, all_weather
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the strategy with parameters.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        # Merge default parameters with strategy-specific defaults and provided parameters
        all_params = self.DEFAULT_PARAMS.copy()
        strategy_defaults = getattr(self, 'DEFAULT_PARAMS', {})
        all_params.update(strategy_defaults)
        
        if parameters:
            all_params.update(parameters)
        
        self.strategy_id = strategy_id or f"{all_params['strategy_name']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.name = name or all_params['strategy_name']
        self.parameters = all_params
        self.logger = logging.getLogger(f"{__name__}.{self.strategy_id}")
        
        # Strategy state tracking
        self.active_positions = {}
        self.pending_orders = {}
        self.historical_trades = []
        self.performance_metrics = {}
        
        self.logger.info(f"Initialized strategy: {self.name} ({self.strategy_id})")
    
    @abstractmethod
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable assets for this strategy.
        
        Args:
            market_data: Market data to use for filtering
            
        Returns:
            Universe object containing filtered symbols
        """
        pass
    
    @abstractmethod
    def generate_signals(self, market_data: Any) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data: Market data to analyze (type depends on asset class)
            
        Returns:
            List of signal dictionaries with standard format
        """
        pass
    
    @abstractmethod
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for a given signal.
        
        Args:
            signal: Trading signal dictionary
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Position size in units appropriate for the asset class
        """
        pass
    
    def calculate_risk_metrics(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics for a potential trade.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary of risk metrics
        """
        # Default implementation, can be overridden
        return {
            'max_loss_pct': 0.01,  # 1% of account
            'max_loss_amount': 1000.0,
            'risk_reward_ratio': 2.0,
        }
    
    def on_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a generated signal and make a final trading decision.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Modified signal with final decision
        """
        # Default implementation passes through the signal
        return signal
    
    def on_trade_executed(self, trade: Dict[str, Any]) -> None:
        """
        Handle a trade execution notification.
        
        Args:
            trade: Trade execution details
        """
        # Update internal state
        self.historical_trades.append(trade)
        
        # Update active positions
        symbol = trade.get('symbol')
        if trade.get('action') == 'BUY':
            if symbol in self.active_positions:
                self.active_positions[symbol]['quantity'] += trade.get('quantity', 0)
            else:
                self.active_positions[symbol] = trade
        elif trade.get('action') == 'SELL':
            if symbol in self.active_positions:
                current_qty = self.active_positions[symbol]['quantity'] 
                new_qty = current_qty - trade.get('quantity', 0)
                if new_qty <= 0:
                    del self.active_positions[symbol]
                else:
                    self.active_positions[symbol]['quantity'] = new_qty
    
    def on_market_data(self, data: Dict[str, Any]) -> None:
        """
        Process new market data updates.
        
        Args:
            data: Market data update
        """
        # Default implementation does nothing
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics for this strategy.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for this strategy.
        
        Args:
            metrics: Dictionary of updated metrics
        """
        self.performance_metrics.update(metrics)
    
    # Standard signal format for all strategies
    def create_signal(self, symbol: str, action: str, reason: str, strength: float = 1.0,
                     entry_price: Optional[float] = None, stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a standardized signal dictionary.
        
        Args:
            symbol: Trading symbol
            action: Signal action ('BUY', 'SELL', 'EXIT')
            reason: Reason for the signal
            strength: Signal strength from 0.0 to 1.0
            entry_price: Suggested entry price
            stop_loss: Suggested stop loss price
            take_profit: Suggested take profit price
            
        Returns:
            Standardized signal dictionary
        """
        return {
            'symbol': symbol,
            'strategy_id': self.strategy_id,
            'strategy_name': self.name,
            'action': action,
            'reason': reason,
            'strength': strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now(),
            'parameters': self.parameters.copy(),
        }

# Function to register all strategy classes with the registry
def register_strategy_with_registry(strategy_class):
    """
    Register a strategy class with the registry.
    
    Args:
        strategy_class: The strategy class to register
    """
    try:
        from trading_bot.core.strategy_registry import StrategyRegistry
        
        # Extract metadata from the strategy class
        metadata = {
            'asset_class': strategy_class.DEFAULT_PARAMS.get('asset_class', 'multi_asset'),
            'strategy_type': strategy_class.DEFAULT_PARAMS.get('strategy_type', 'all_weather'),
            'market_regime': strategy_class.DEFAULT_PARAMS.get('market_regime', 'all_weather'),
            'timeframe': strategy_class.DEFAULT_PARAMS.get('timeframe', 'daily'),
        }
        
        # Register with registry
        StrategyRegistry.register(strategy_class, metadata)
        logger.info(f"Registered strategy {strategy_class.__name__} with registry")
        
    except Exception as e:
        logger.error(f"Failed to register strategy {strategy_class.__name__}: {e}")
