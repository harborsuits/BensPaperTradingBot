#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks Base Strategy

Base class for all stocks trading strategies, providing common functionality.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from trading_bot.strategies.strategy_template import Strategy, Signal, SignalType
from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class StocksBaseStrategy(Strategy):
    """
    Base class for stocks trading strategies.
    
    This class provides common functionality for stocks strategies,
    including specialized risk management, position sizing, and
    market-specific logic.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize StocksBaseStrategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        super().__init__(name, parameters, metadata)
        
        # Default parameters specific to stocks
        self.default_params = {
            'max_positions': 5,
            'max_risk_per_trade_percent': 0.01,  # 1% risk per trade
        }
        
        # Override defaults with provided parameters
        if parameters:
            self.parameters.update(parameters)
        else:
            self.parameters = self.default_params.copy()
        
        # Asset class specific state
        self.positions = {}
        self.pending_orders = {}
        
        logger.info(f"StocksBaseStrategy initialized")
    
    def register_events(self, event_bus: EventBus) -> None:
        """
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        """
        self.event_bus = event_bus
        
        # Register for market data events
        event_bus.register(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        event_bus.register(EventType.TIMEFRAME_COMPLETED, self._on_timeframe_completed)
        
        # Register for additional stocks-specific events if needed
        
        logger.info(f"Strategy registered for events")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """Handle market data updated events."""
        pass  # Implement in child classes
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """Handle timeframe completed events."""
        pass  # Implement in child classes
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        """
        raise NotImplementedError("Subclasses must implement calculate_indicators")
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size for the signal based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units
        """
        # Extract parameters
        max_risk_percent = self.parameters['max_risk_per_trade_percent']
        risk_amount = account_balance * max_risk_percent
        
        # Calculate position size
        if signal.stop_loss is None:
            # Use a default risk amount if no stop loss is specified
            position_size = risk_amount / account_balance * 100
        else:
            # Calculate based on stop loss
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            
            # Calculate risk per share/unit
            if signal.signal_type == SignalType.LONG:
                risk_per_unit = entry_price - stop_loss
            else:
                risk_per_unit = stop_loss - entry_price
            
            # Calculate position size
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                # Fallback if stop loss is not properly defined
                position_size = risk_amount / entry_price
        
        return position_size
