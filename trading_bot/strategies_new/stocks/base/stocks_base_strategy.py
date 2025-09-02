#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks Base Strategy

This module provides the base class for all stock trading strategies.
It handles common functionality such as event registration, data processing,
position management, and risk management that is specific to stock markets.
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

class StocksSession(Session):
    """
    Stocks-specific session class that extends the base Session.
    
    This handles stocks-specific metadata, session timing, and market hours.
    """
    
    def __init__(self, symbol: str, timeframe: TimeFrame, **kwargs):
        super().__init__(symbol=symbol, timeframe=timeframe, market_type=MarketType.STOCKS, **kwargs)
        
        # Stocks-specific session attributes
        self.exchange = kwargs.get('exchange', 'NYSE')
        self.sector = kwargs.get('sector', 'Unknown')
        self.industry = kwargs.get('industry', 'Unknown')
        self.market_cap_category = kwargs.get('market_cap_category', 'Unknown')
        self.session_type = kwargs.get('session_type', 'regular')  # regular, pre-market, after-hours
        self.lot_size = kwargs.get('lot_size', 100)
        self.is_market_open = self._check_market_open()
        
    def _check_market_open(self) -> bool:
        """Check if the market is currently open for trading."""
        # In a real implementation this would check current time against market hours
        # and consider holidays, weekends, etc.
        return True  # Simplified placeholder


class StocksBaseStrategy(ABC):
    """
    Base class for all stock trading strategies.
    
    This class handles the common functionality of stock strategies:
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
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the stocks base strategy.
        
        Args:
            session: Stock trading session with symbol, timeframe, etc.
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
        
        # Register for stocks-specific events
        event_bus.register(EventType.EARNINGS_ANNOUNCEMENT, self._on_earnings_announcement)
        event_bus.register(EventType.MARKET_HOURS_CHANGE, self._on_market_hours_change)
        event_bus.register(EventType.ECONOMIC_INDICATOR_RELEASE, self._on_economic_indicator_release)
        
        logger.info(f"Strategy registered for events")
    
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
    
    def _on_earnings_announcement(self, event: Event) -> None:
        """Handle earnings announcement events."""
        # Stocks-specific logic for earnings events
        if event.data.get('symbol') != self.session.symbol:
            return
            
        # Get earnings data
        earnings_data = event.data.get('earnings_data', {})
        
        # Decide if we need to adjust positions before/after earnings
        expected_volatility = earnings_data.get('expected_volatility', 'medium')
        
        # Adjust risk based on earnings volatility
        if expected_volatility == 'high':
            # Consider reducing position sizes or hedging
            self._reduce_positions_for_earnings()
            
        logger.info(f"{self.name}: Processed earnings announcement for {self.session.symbol}")
    
    def _on_market_hours_change(self, event: Event) -> None:
        """Handle market hours change events."""
        # Update session market hours status
        is_open = event.data.get('is_open', False)
        self.session.is_market_open = is_open
        
        # Handle market close
        if not is_open and len(self.positions) > 0:
            logger.info(f"{self.name}: Market closed with {len(self.positions)} open positions")
    
    def _on_economic_indicator_release(self, event: Event) -> None:
        """Handle economic indicator release events."""
        # Check if this indicator affects our stock
        indicator = event.data.get('indicator', '')
        impact = event.data.get('impact', 'low')
        
        if impact in ['high', 'medium'] and self.session.sector in event.data.get('affected_sectors', []):
            logger.info(f"{self.name}: Economic indicator {indicator} may impact {self.session.symbol}")
    
    def _reduce_positions_for_earnings(self) -> None:
        """Reduce position sizes before high-volatility events like earnings."""
        for position in self.positions:
            # Request position size reduction
            if self.event_bus and position.status == PositionStatus.OPEN:
                self.event_bus.publish(EventType.POSITION_MODIFY_REQUESTED, {
                    'strategy_name': self.name,
                    'position_id': position.id,
                    'new_size': position.size * 0.5,  # Reduce by 50%
                    'reason': 'earnings_risk_reduction'
                })
    
    def _check_for_trade_opportunities(self) -> None:
        """
        Check for trade opportunities based on current signals.
        
        This method evaluates the current signals and market conditions,
        then opens, closes, or modifies positions as appropriate.
        """
        if not self.is_active or not self.signals:
            return
            
        # Only trade during market hours
        if not self.session.is_market_open and not self.parameters.get('trade_after_hours', False):
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
        # Round to lot size for stocks
        lot_size = self.session.lot_size
        size = round(size / lot_size) * lot_size
        
        if size < lot_size:
            logger.warning(f"{self.name}: Position size {size} too small, minimum is {lot_size}")
            return
            
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
            Position size in number of shares
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
