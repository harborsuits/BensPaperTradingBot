#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Base Strategy

This module provides the base class for all cryptocurrency trading strategies.
It handles common functionality such as event registration, data processing,
position management, and risk management that is specific to crypto markets.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.core.session import Session
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus

# Configure logging
logger = logging.getLogger(__name__)

class CryptoSession(Session):
    """
    Crypto-specific session class that extends the base Session.
    
    This handles cryptocurrency-specific metadata, exchange settings, and 
    market characteristics.
    """
    
    def __init__(self, symbol: str, timeframe: TimeFrame, **kwargs):
        super().__init__(symbol=symbol, timeframe=timeframe, market_type=MarketType.CRYPTO, **kwargs)
        
        # Crypto-specific session attributes
        self.exchange = kwargs.get('exchange', 'Binance')
        self.quote_currency = kwargs.get('quote_currency', 'USDT')
        self.min_trade_size = kwargs.get('min_trade_size', 0.001)
        self.trading_fees = kwargs.get('trading_fees', 0.001)  # 0.1% default
        self.is_perpetual = kwargs.get('is_perpetual', False)
        self.funding_rate = kwargs.get('funding_rate', 0.0)
        self.market_type = 'spot' if not self.is_perpetual else 'futures'
        
        # Crypto markets trade 24/7
        self.is_market_open = True
        
    def update_funding_rate(self, funding_rate: float) -> None:
        """Update the funding rate for perpetual futures."""
        if self.is_perpetual:
            self.funding_rate = funding_rate
            logger.debug(f"Updated funding rate for {self.symbol}: {funding_rate}")
        

class CryptoBaseStrategy(ABC):
    """
    Base class for all cryptocurrency trading strategies.
    
    This class handles the common functionality of crypto strategies:
    - Event registration and handling
    - Data processing through the DataPipeline
    - Basic position management
    - Risk management with crypto-specific considerations
    - Performance tracking
    
    Child classes must implement:
    - calculate_indicators: Technical indicators specific to the strategy
    - generate_signals: Trading signals based on the indicators
    - calculate_position_size: Position sizing logic
    - regime_compatibility: How well the strategy fits the current market regime
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the crypto base strategy.
        
        Args:
            session: Crypto trading session with symbol, timeframe, etc.
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
        
        # Crypto-specific state
        self.market_depth = {}
        self.orderbook = {}
        self.recent_trades = []
        self.latest_funding_rate = 0.0
        
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
        
        # Register for crypto-specific events
        event_bus.register(EventType.ORDERBOOK_UPDATED, self._on_orderbook_updated)
        event_bus.register(EventType.MARKET_DEPTH_UPDATED, self._on_market_depth_updated)
        event_bus.register(EventType.FUNDING_RATE_UPDATED, self._on_funding_rate_updated)
        event_bus.register(EventType.LIQUIDATION_EVENT, self._on_liquidation_event)
        
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
    
    def _on_orderbook_updated(self, event: Event) -> None:
        """
        Handle orderbook updated events.
        
        This is crypto-specific as orderbook data is more commonly used
        in crypto trading strategies.
        """
        if event.data.get('symbol') != self.session.symbol:
            return
            
        self.orderbook = event.data.get('orderbook', {})
        
        # Some strategies might want to adjust positions based on orderbook imbalances
        # or other orderbook features
        if self.is_active and len(self.positions) > 0 and self.parameters.get('use_orderbook_for_exits', False):
            self._check_orderbook_for_exit_signals()
    
    def _on_market_depth_updated(self, event: Event) -> None:
        """Handle market depth updated events."""
        if event.data.get('symbol') != self.session.symbol:
            return
            
        self.market_depth = event.data.get('market_depth', {})
    
    def _on_funding_rate_updated(self, event: Event) -> None:
        """
        Handle funding rate updated events for perpetual futures.
        """
        if event.data.get('symbol') != self.session.symbol or not self.session.is_perpetual:
            return
            
        funding_rate = event.data.get('funding_rate', 0.0)
        funding_time = event.data.get('funding_time', datetime.now(timezone.utc))
        
        self.latest_funding_rate = funding_rate
        self.session.update_funding_rate(funding_rate)
        
        # Check if we should adjust positions before funding
        if self.parameters.get('funding_rate_sensitive', False):
            self._adjust_for_funding_rate(funding_rate, funding_time)
    
    def _on_liquidation_event(self, event: Event) -> None:
        """
        Handle large liquidation events which can indicate market volatility.
        """
        if event.data.get('symbol') != self.session.symbol:
            return
            
        liquidation_amount = event.data.get('liquidation_amount', 0.0)
        liquidation_direction = event.data.get('direction', 'unknown')
        
        # If there's a large liquidation event, consider adjusting risk
        threshold = self.parameters.get('liquidation_threshold', 100.0)
        if liquidation_amount > threshold:
            logger.info(f"Large liquidation detected: {liquidation_amount} {liquidation_direction}")
            
            # Temporarily increase risk management
            if self.is_active and len(self.positions) > 0:
                self._tighten_risk_parameters()
    
    def _adjust_for_funding_rate(self, funding_rate: float, funding_time: datetime) -> None:
        """
        Adjust positions based on funding rate for futures.
        
        Positive funding rate: shorts pay longs
        Negative funding rate: longs pay shorts
        """
        if not self.positions:
            return
            
        # Check how long until next funding
        now = datetime.now(timezone.utc)
        seconds_to_funding = (funding_time - now).total_seconds()
        
        # Only act if we're close to funding time
        if abs(seconds_to_funding) > 3600:  # More than an hour away
            return
            
        # Adjust positions based on funding rate
        for position in self.positions:
            if (position.direction == 'long' and funding_rate < -0.001) or \
               (position.direction == 'short' and funding_rate > 0.001):
                # Consider closing the position to avoid paying high funding
                logger.info(f"Considering closing {position.direction} position due to unfavorable funding of {funding_rate}")
                
                # Check if expected funding cost exceeds our threshold
                position_value = position.size * self.market_data['close'].iloc[-1]
                funding_cost = position_value * abs(funding_rate)
                funding_threshold = self.parameters.get('max_funding_cost', 0.0005) * position_value
                
                if funding_cost > funding_threshold:
                    self._close_position(position.id)
    
    def _tighten_risk_parameters(self) -> None:
        """Temporarily tighten risk parameters during high volatility."""
        for position in self.positions:
            # Move stop loss closer
            current_price = self.market_data['close'].iloc[-1]
            
            if position.direction == 'long':
                new_stop = max(position.stop_loss, current_price - (current_price - position.stop_loss) * 0.5)
                
                if new_stop > position.stop_loss:
                    self.event_bus.publish(EventType.POSITION_MODIFY_REQUESTED, {
                        'strategy_name': self.name,
                        'position_id': position.id,
                        'stop_loss': new_stop,
                        'reason': 'high_volatility_risk_management'
                    })
            else:  # short
                new_stop = min(position.stop_loss, current_price + (position.stop_loss - current_price) * 0.5)
                
                if new_stop < position.stop_loss:
                    self.event_bus.publish(EventType.POSITION_MODIFY_REQUESTED, {
                        'strategy_name': self.name,
                        'position_id': position.id,
                        'stop_loss': new_stop,
                        'reason': 'high_volatility_risk_management'
                    })
    
    def _check_orderbook_for_exit_signals(self) -> None:
        """Check orderbook for potential exit signals."""
        if not self.orderbook or not self.positions:
            return
            
        # Calculate orderbook imbalance (simplified)
        bid_volume = sum(qty for price, qty in self.orderbook.get('bids', [])[:5])
        ask_volume = sum(qty for price, qty in self.orderbook.get('asks', [])[:5])
        
        if bid_volume == 0 or ask_volume == 0:
            return
            
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Check positions against imbalance
        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue
                
            # If there's a strong imbalance against our position, consider exiting
            if (position.direction == 'long' and imbalance < -0.7) or \
               (position.direction == 'short' and imbalance > 0.7):
                logger.info(f"Orderbook imbalance {imbalance:.2f} suggests closing {position.direction} position")
                self._close_position(position.id)
    
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
                
                # Ensure position size meets minimums
                if position_size >= self.session.min_trade_size:
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
        
        # For crypto, we might want wider stops due to higher volatility
        if self.parameters.get('use_wider_stops_for_crypto', True):
            multiplier = multiplier * 1.2
        
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
        
        # For crypto, we might want wider stops but also higher profit targets
        if self.parameters.get('use_wider_stops_for_crypto', True):
            stop_multiplier = stop_multiplier * 1.2
        
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
            Position size in crypto units
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
