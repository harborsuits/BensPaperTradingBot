#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Base Strategy

This module provides the base class for all forex trading strategies.
It handles common functionality such as event registration, data processing,
position management, and risk management that is specific to forex markets.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.core.session import Session
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus

# Configure logging
logger = logging.getLogger(__name__)

class ForexSession(Session):
    """
    Forex-specific session class that extends the base Session.
    
    This handles forex-specific metadata, session timing, and market hours.
    """
    
    def __init__(self, symbol: str, timeframe: TimeFrame, **kwargs):
        super().__init__(symbol=symbol, timeframe=timeframe, market_type=MarketType.FOREX, **kwargs)
        
        # Forex-specific session attributes
        self.base_currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        self.quote_currency = symbol.split('/')[1] if '/' in symbol else symbol[3:6]
        self.pip_value = kwargs.get('pip_value', self._calculate_pip_value())
        self.spread = kwargs.get('spread', 0.0)
        self.is_major_pair = self._is_major_pair()
        
    def _calculate_pip_value(self) -> float:
        """Calculate the pip value based on the currency pair."""
        # Default pip size is 0.0001 for most pairs
        pip_size = 0.0001
        
        # For JPY pairs, pip size is 0.01
        if self.quote_currency == 'JPY':
            pip_size = 0.01
            
        # Simplified calculation, would use real exchange rates in production
        return pip_size
    
    def _is_major_pair(self) -> bool:
        """Determine if this is a major currency pair."""
        major_pairs = [
            'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 
            'AUD/USD', 'USD/CAD', 'NZD/USD'
        ]
        return self.symbol in major_pairs


class ForexBaseStrategy(ABC):
    """
    Base class for all forex trading strategies.
    
    This class handles the common functionality of forex strategies:
    - Event registration and handling
    - Data processing through the DataPipeline
    - Basic position management
    - Risk management
    - Performance tracking
    
    Child classes must implement:
    - calculate_indicators: Technical/fundamental indicators specific to the strategy
    - generate_signals: Trading signals based on the indicators
    - calculate_position_size: Position sizing logic
    - regime_compatibility: How well the strategy fits the current market regime
    """
    
    def __init__(self, session: ForexSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the forex base strategy.
        
        Args:
            session: Forex trading session with symbol, timeframe, etc.
            data_pipeline: Data processing pipeline for cleaning and quality assurance
            parameters: Strategy-specific parameters (will be provided by factory)
        """
        self.name = self.__class__.__name__
        self.session = session
        self.data_pipeline = data_pipeline
        self.parameters = parameters or {}
        self.event_bus = None
        
        # Strategy state
        self.is_active = False
        self.positions = []
        self.market_data = pd.DataFrame()
        self.indicators = {}
        self.signals = {}
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0
        }
        
        logger.info(f"Initialized {self.name} for {session.symbol} on {session.timeframe}")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register the strategy for relevant events.
        
        Args:
            event_bus: Event bus to register with
        """
        self.event_bus = event_bus
        
        # Register for market data events
        event_bus.register(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        event_bus.register(EventType.TIMEFRAME_COMPLETED, self._on_timeframe_completed)
        
        # Register for position-related events
        event_bus.register(EventType.POSITION_OPENED, self._on_position_opened)
        event_bus.register(EventType.POSITION_CLOSED, self._on_position_closed)
        event_bus.register(EventType.POSITION_MODIFIED, self._on_position_modified)
        
        # Register for forex-specific events
        event_bus.register(EventType.ECONOMIC_CALENDAR_UPDATE, self._on_economic_calendar_update)
        event_bus.register(EventType.CENTRAL_BANK_ANNOUNCEMENT, self._on_central_bank_announcement)
        
        logger.info(f"{self.name} registered for events")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        Args:
            event: Market data updated event
        """
        if event.data.get('symbol') != self.session.symbol:
            return
            
        # Process and store the market data
        data = event.data.get('data')
        if data is not None:
            # Process data through the pipeline
            processed_data = self.data_pipeline.process(data, 
                                                       symbol=self.session.symbol, 
                                                       source=event.data.get('source', 'unknown'))
            
            self.market_data = processed_data
            
            # Calculate strategy-specific indicators
            self.indicators = self.calculate_indicators(processed_data)
            
            # Generate trading signals
            self.signals = self.generate_signals(processed_data, self.indicators)
            
            # Check for trade opportunities
            self._check_for_trade_opportunities()
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        Args:
            event: Timeframe completed event
        """
        if event.data.get('symbol') != self.session.symbol or event.data.get('timeframe') != self.session.timeframe:
            return
            
        # Update performance metrics
        self._update_performance_metrics()
        
        # Publish strategy status update
        if self.event_bus:
            self.event_bus.publish(EventType.STRATEGY_STATUS_UPDATED, {
                'strategy_name': self.name,
                'symbol': self.session.symbol,
                'timeframe': self.session.timeframe,
                'is_active': self.is_active,
                'performance_metrics': self.performance_metrics,
                'positions_count': len(self.positions),
                'current_signals': self.signals
            })
    
    def _on_position_opened(self, event: Event) -> None:
        """Handle position opened events."""
        if event.data.get('strategy_name') != self.name:
            return
            
        position = event.data.get('position')
        if position:
            self.positions.append(position)
            logger.info(f"{self.name}: Position opened - {position}")
    
    def _on_position_closed(self, event: Event) -> None:
        """Handle position closed events."""
        if event.data.get('strategy_name') != self.name:
            return
            
        position_id = event.data.get('position_id')
        self.positions = [p for p in self.positions if p.id != position_id]
        logger.info(f"{self.name}: Position closed - ID {position_id}")
    
    def _on_position_modified(self, event: Event) -> None:
        """Handle position modified events."""
        if event.data.get('strategy_name') != self.name:
            return
            
        modified_position = event.data.get('position')
        if modified_position:
            for i, position in enumerate(self.positions):
                if position.id == modified_position.id:
                    self.positions[i] = modified_position
                    break
    
    def _on_economic_calendar_update(self, event: Event) -> None:
        """Handle economic calendar update events."""
        # Forex-specific logic for economic events
        pass
    
    def _on_central_bank_announcement(self, event: Event) -> None:
        """Handle central bank announcement events."""
        # Forex-specific logic for central bank events
        pass
    
    def _check_for_trade_opportunities(self) -> None:
        """
        Check for trade opportunities based on current signals.
        
        This method evaluates the current signals and market conditions,
        then opens, closes, or modifies positions as appropriate.
        """
        if not self.is_active or not self.signals:
            return
            
        # Check for entry signals
        entry_signals = self.signals.get('entry', {})
        for direction, strength in entry_signals.items():
            if strength > self.parameters.get('entry_threshold', 0.7):
                # Calculate position size based on current risk parameters
                position_size = self.calculate_position_size(
                    direction, 
                    self.market_data, 
                    self.indicators
                )
                
                # Open position if size is valid
                if position_size > 0:
                    self._open_position(direction, position_size)
        
        # Check for exit signals
        exit_signals = self.signals.get('exit', {})
        for position in self.positions:
            if position.status == PositionStatus.OPEN:
                # Check if we should exit this position
                if (position.direction == 'long' and exit_signals.get('long', 0) > 
                        self.parameters.get('exit_threshold', 0.5)) or \
                   (position.direction == 'short' and exit_signals.get('short', 0) > 
                        self.parameters.get('exit_threshold', 0.5)):
                    self._close_position(position.id)
    
    def _open_position(self, direction: str, size: float) -> None:
        """
        Open a new position.
        
        Args:
            direction: Direction of the position ('long' or 'short')
            size: Size of the position
        """
        # Create position object
        position = Position(
            symbol=self.session.symbol,
            direction=direction,
            size=size,
            entry_price=self.market_data['close'].iloc[-1],
            stop_loss=self._calculate_stop_loss(direction),
            take_profit=self._calculate_take_profit(direction),
            strategy_name=self.name
        )
        
        # Publish position opening event
        if self.event_bus:
            self.event_bus.publish(EventType.POSITION_OPEN_REQUESTED, {
                'strategy_name': self.name,
                'position': position
            })
            
        logger.info(f"{self.name}: Requested to open {direction} position of size {size}")
    
    def _close_position(self, position_id: str) -> None:
        """
        Close an existing position.
        
        Args:
            position_id: ID of the position to close
        """
        # Find the position
        position = next((p for p in self.positions if p.id == position_id), None)
        if not position:
            logger.warning(f"{self.name}: Cannot close position {position_id} - not found")
            return
            
        # Publish position closing event
        if self.event_bus:
            self.event_bus.publish(EventType.POSITION_CLOSE_REQUESTED, {
                'strategy_name': self.name,
                'position_id': position_id,
                'exit_price': self.market_data['close'].iloc[-1]
            })
            
        logger.info(f"{self.name}: Requested to close position {position_id}")
    
    def _calculate_stop_loss(self, direction: str) -> float:
        """
        Calculate the stop loss price for a position.
        
        Args:
            direction: Direction of the position ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        current_price = self.market_data['close'].iloc[-1]
        atr = self.indicators.get('atr', self.market_data['high'].iloc[-1] - self.market_data['low'].iloc[-1])
        
        # Default multiplier
        multiplier = self.parameters.get('stop_loss_atr_multiplier', 2.0)
        
        if direction == 'long':
            return current_price - (atr * multiplier)
        else:  # short
            return current_price + (atr * multiplier)
    
    def _calculate_take_profit(self, direction: str) -> float:
        """
        Calculate the take profit price for a position.
        
        Args:
            direction: Direction of the position ('long' or 'short')
            
        Returns:
            Take profit price
        """
        current_price = self.market_data['close'].iloc[-1]
        atr = self.indicators.get('atr', self.market_data['high'].iloc[-1] - self.market_data['low'].iloc[-1])
        
        # Default risk:reward
        risk_reward = self.parameters.get('risk_reward_ratio', 2.0)
        stop_multiplier = self.parameters.get('stop_loss_atr_multiplier', 2.0)
        take_profit_distance = atr * stop_multiplier * risk_reward
        
        if direction == 'long':
            return current_price + take_profit_distance
        else:  # short
            return current_price - take_profit_distance
    
    def _update_performance_metrics(self) -> None:
        """Update the strategy's performance metrics based on closed positions."""
        # This would be implemented with actual position history in production
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical or fundamental indicators for the strategy.
        
        Args:
            data: Market data as a pandas DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on indicators and market data.
        
        Args:
            data: Market data as a pandas DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data as a pandas DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in base currency units
        """
        pass
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile', etc.)
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Default compatibilities - to be overridden by child classes
        compatibility_map = {
            'trending': 0.5,
            'ranging': 0.5,
            'volatile': 0.5,
            'calm': 0.5
        }
        
        return compatibility_map.get(market_regime, 0.3)
