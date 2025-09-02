#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap Trading Strategy (Gap and Go/Fade)

An advanced gap trading strategy for stocks that implements both:
- Gap and Go: Trading in the direction of the gap (continuation)
- Gap Fade: Trading against the gap direction (reversal)

This implementation adapts the legacy gap trading strategy to the new 
modular, event-driven architecture.
"""

import logging
import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time

from trading_bot.core.signals import Signal, SignalType

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.stocks.gap.gap_trade_analyzer import GapTradeAnalyzer
from trading_bot.strategies_new.factory.registry import register_strategy

@register_strategy(
    name="GapTradingStrategy",
    market_type="stocks",
    description="A strategy that trades price gaps between previous close and current open, including both Gap and Go (continuation) and Gap Fade (reversal) approaches",
    timeframes=["1m", "5m", "15m", "1h"],
    parameters={
        "strategy_mode": {"description": "Trading mode (continuation, fade, or both)", "type": "string"},
        "min_gap_percent": {"description": "Minimum gap size as percentage", "type": "float"},
        "volume_threshold": {"description": "Minimum volume vs average for confirmation", "type": "float"},
        "max_entry_time_minutes": {"description": "Maximum time to enter after market open", "type": "integer"}
    }
)
class GapTradingStrategy(StocksBaseStrategy):
    """
    Gap Trading Strategy
    
    This strategy identifies and trades price gaps between the previous close and current open:
    - Gap and Go: Trades in the direction of the gap (continuation pattern)
    - Gap Fade: Trades against the direction of the gap (reversal pattern)
    
    Features:
    - Sophisticated gap detection with filters for significance and quality
    - Volume confirmation requirements
    - Multiple technical indicators for signal confirmation
    - Adaptive time-based signal decay
    - Comprehensive risk management
    - Position sizing based on ATR/volatility
    - Support for both long and short trades
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Gap Trading Strategy.
        
        Args:
            session: StocksSession for the specific symbol and timeframe
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base strategy
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Gap detection parameters
            'min_gap_percent': 1.5,       # Minimum gap size as percentage
            'max_gap_percent': 10.0,      # Maximum gap size to filter outliers
            'gap_directions': ['up', 'down'], # Which gap directions to trade
            
            # Trading approach parameters
            'strategy_mode': 'both',      # 'continuation', 'fade', or 'both'
            'continuation_preference': 0.5, # 0.0-1.0 preference for continuation vs fade
            
            # Entry parameters
            'volume_threshold': 1.5,      # Minimum volume vs average for confirmation
            'max_entry_time_minutes': 30, # Maximum time to enter after market open
            'require_first_bar_confirmation': True, # Require 1st bar to close in expected direction
            
            # Technical analysis parameters
            'vwap_periods': [20],         # VWAP periods for confirmation
            'rsi_period': 14,             # RSI period
            'rsi_overbought': 70,         # RSI overbought threshold
            'rsi_oversold': 30,           # RSI oversold threshold
            'atr_period': 14,             # ATR period for volatility measurement
            
            # Risk management
            'atr_multiplier': 2.0,        # ATR multiplier for stop loss
            'max_risk_per_trade_percent': 1.0, # Max risk per trade as % of account
            'profit_factor': 1.5,         # Profit target as multiple of risk
            'max_gap_trades_per_day': 3,  # Maximum gap trades per day
            
            # Time-based management
            'signal_decay_minutes': 45,   # How long a gap signal remains valid
            'close_by_end_of_day': True,  # Close all positions by end of day
            
            # Additional filters
            'min_avg_volume': 500000,     # Minimum average volume
            'min_price': 5.0,             # Minimum price for tradable stocks
            'exclude_earnings_days': True, # Avoid trading gaps on earnings days
            'earnings_calendar': {},      # Dict of symbols -> upcoming earnings dates
        }
        
        # Update parameters with defaults for any missing keys
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Strategy state
        self.gaps_detected = {}  # Dict to store detected gaps by symbol
        self.gap_trades_today = 0  # Counter for gap trades made today
        self.last_market_date = None  # Track the last market date to reset counters
        
        # Register for market events if event bus is available
        if self.event_bus:
            self.register_for_events(self.event_bus)
        
        logger.info(f"Initialized Gap Trading Strategy for {session.symbol} on {session.timeframe}")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register for relevant market events.
        
        Args:
            event_bus: EventBus to register with
        """
        # First register for common events via base class
        super().register_for_events(event_bus)
        
        # Register for gap-specific events
        event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        event_bus.subscribe(EventType.MARKET_CLOSE, self._on_market_close)
        event_bus.subscribe(EventType.PRE_MARKET_DATA, self._on_pre_market_data)
        event_bus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self._on_earnings_announcement)
        
        logger.debug(f"Gap Trading Strategy registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open event.
        
        This is a critical event for gap trading as we need to detect
        gaps between previous close and today's open.
        
        Args:
            event: Market open event
        """
        # Reset daily counters if it's a new trading day
        current_date = datetime.now().date()
        if self.last_market_date != current_date:
            self.gap_trades_today = 0
            self.last_market_date = current_date
            logger.info(f"New trading day {current_date}, reset gap trade counter")
        
        # Check if we have enough data to detect gaps
        if self.market_data is None or len(self.market_data) < 2:
            logger.warning(f"Insufficient data for gap detection on market open")
            return
        
        # Detect gaps at market open
        symbol = self.session.symbol
        self._detect_gap(self.market_data, symbol)
        
        # Log detected gaps
        if symbol in self.gaps_detected:
            gap_info = self.gaps_detected[symbol]
            logger.info(f"Gap detected for {symbol}: {gap_info['gap_percent']:.2f}% {gap_info['gap_direction']} gap")
    
    def _on_market_close(self, event: Event) -> None:
        """
        Handle market close event.
        
        Close any open positions if configured to do so.
        
        Args:
            event: Market close event
        """
        # Check if we should close positions at end of day
        if self.parameters['close_by_end_of_day']:
            # Close all open positions
            for position in self.positions:
                if position.status == PositionStatus.OPEN:
                    self._close_position(position.id)
                    logger.info(f"Closed position {position.id} at market close")
        
        # Clear any pending gap signals
        for symbol in list(self.gaps_detected.keys()):
            if self.gaps_detected[symbol].get('active', False):
                self.gaps_detected[symbol]['active'] = False
                logger.info(f"Deactivated gap signal for {symbol} at market close")
    
    def _on_pre_market_data(self, event: Event) -> None:
        """
        Handle pre-market data events.
        
        This allows us to prepare for potential gaps before market open.
        
        Args:
            event: Pre-market data event
        """
        # Check if the event data contains our symbol
        symbol = self.session.symbol
        if event.data.get('symbol') != symbol:
            return
        
        # Extract pre-market data
        pre_market_data = event.data.get('data')
        if pre_market_data is None or len(pre_market_data) < 1:
            return
        
        # Get previous close
        if self.market_data is not None and len(self.market_data) > 0:
            prev_close = self.market_data['close'].iloc[-1]
            
            # Get pre-market price
            pre_market_price = pre_market_data['close'].iloc[-1]
            
            # Calculate potential gap
            gap_percent = ((pre_market_price / prev_close) - 1) * 100
            
            # Log potential gap
            if abs(gap_percent) >= self.parameters['min_gap_percent']:
                gap_direction = 'up' if gap_percent > 0 else 'down'
                logger.info(f"Potential {gap_direction} gap of {gap_percent:.2f}% detected in pre-market for {symbol}")
    
    def _on_earnings_announcement(self, event: Event) -> None:
        """
        Handle earnings announcement events.
        
        Skip gap trading around earnings if configured to do so.
        
        Args:
            event: Earnings announcement event
        """
        # Check if we should avoid trading on earnings days
        if not self.parameters['exclude_earnings_days']:
            return
        
        # Check if the event data contains our symbol
        symbols = event.data.get('symbols', [])
        symbol = self.session.symbol
        
        if symbol in symbols:
            # Get days to announcement
            days_to_announcement = event.data.get('days_to_announcement', 0)
            
            # If earnings are today or tomorrow, deactivate any gap signals
            if days_to_announcement <= 1:
                if symbol in self.gaps_detected:
                    self.gaps_detected[symbol]['active'] = False
                    logger.info(f"Deactivated gap signal for {symbol} due to earnings announcement")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated event.
        
        Update our internal data and check entry/exit conditions.
        
        Args:
            event: Market data updated event
        """
        # Let the base class handle common functionality first
        super()._on_market_data_updated(event)
        
        # Check if the event contains data for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
            
        # Process gap trades if market is open and we have detected gaps
        if self.session.is_market_open and self.gaps_detected:
            symbol = self.session.symbol
            
            # Check if we have a detected gap for this symbol
            if symbol in self.gaps_detected and self.gaps_detected[symbol].get('active', False):
                # Check entry conditions
                self._check_entry_conditions(self.market_data, symbol)
                
                # Update active signals and remove expired ones
                self._update_signals(self.market_data, symbol)
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed event.
        
        Calculate indicators and update our strategy state.
        
        Args:
            event: Timeframe completed event
        """
        # Let the base class handle common functionality first
        super()._on_timeframe_completed(event)
        
        # Check if the event contains data for our symbol and timeframe
        if (event.data.get('symbol') != self.session.symbol or 
            event.data.get('timeframe') != self.session.timeframe):
            return
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
    
    def _is_in_trading_window(self) -> bool:
        """
        Check if the current time is within the gap trading window.
        
        Returns:
            True if within trading window, False otherwise
        """
        # Get current time
        now = datetime.now().time()
        
        # Get market open time (default to 9:30 AM EST for US markets)
        market_open = time(9, 30)
        
        # Calculate max entry time
        max_entry_time_minutes = self.parameters['max_entry_time_minutes']
        max_entry_time = (datetime.combine(datetime.today(), market_open) + 
                         timedelta(minutes=max_entry_time_minutes)).time()
        
        # Check if current time is within trading window
        return market_open <= now <= max_entry_time
    
    def _detect_gap(self, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Detect if a gap exists between previous close and current open.
        
        Args:
            data: Market data DataFrame
            symbol: Symbol to check for gaps
            
        Returns:
            Gap details dictionary if gap detected, None otherwise
        """
        # Check if we have enough data
        if len(data) < 2:
            logger.warning(f"Insufficient data for gap detection for {symbol}")
            return None
        
        # Get previous close and current open
        prev_close = data['close'].iloc[-2]
        curr_open = data['open'].iloc[-1]
        
        # Calculate gap percentage
        gap_percent = ((curr_open / prev_close) - 1) * 100
        
        # Determine if this is a valid gap
        min_gap = self.parameters['min_gap_percent']
        max_gap = self.parameters['max_gap_percent']
        
        if abs(gap_percent) < min_gap or abs(gap_percent) > max_gap:
            return None
        
        # Determine gap direction
        gap_direction = 'up' if gap_percent > 0 else 'down'
        
        # Check if we're trading this type of gap
        if gap_direction not in self.parameters['gap_directions']:
            return None
        
        # Create gap info dictionary
        gap_info = {
            'symbol': symbol,
            'detection_time': datetime.now(),
            'prev_close': prev_close,
            'curr_open': curr_open,
            'gap_percent': gap_percent,
            'gap_direction': gap_direction,
            'active': True,
            'volume_confirmed': False,
            'first_bar_confirmed': False,
            'entry_price': None,
            'atr': self.indicators.get('atr', data['high'].iloc[-1] - data['low'].iloc[-1]),
            'trade_direction': None  # Will be set during entry condition check
        }
        
        # Store gap info
        self.gaps_detected[symbol] = gap_info
        
        # Log gap detection
        logger.info(f"Detected {gap_direction} gap of {gap_percent:.2f}% for {symbol}")
        
        return gap_info
    
    def _check_entry_conditions(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Check if entry conditions are met for a detected gap.
        
        Args:
            data: Market data DataFrame
            symbol: Symbol to check
            
        Returns:
            True if entry conditions are met, False otherwise
        """
        # Check if we have a detected gap for this symbol
        if symbol not in self.gaps_detected or not self.gaps_detected[symbol].get('active', False):
            return False
        
        # Check if we're within the trading window
        if not self._is_in_trading_window():
            logger.debug(f"Outside trading window for gap trade on {symbol}")
            return False
        
        # Check if we've reached max trades for today
        if self.gap_trades_today >= self.parameters['max_gap_trades_per_day']:
            logger.info(f"Reached maximum gap trades for today ({self.parameters['max_gap_trades_per_day']})")
            return False
        
        # Get gap info
        gap_info = self.gaps_detected[symbol]
        
        # Check if this gap has already triggered a trade
        if gap_info.get('trade_direction') is not None:
            return False
        
        # Check volume confirmation if not already confirmed
        if not gap_info.get('volume_confirmed', False):
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            curr_volume = data['volume'].iloc[-1]
            
            volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 0
            volume_threshold = self.parameters['volume_threshold']
            
            if volume_ratio >= volume_threshold:
                gap_info['volume_confirmed'] = True
                logger.debug(f"Volume confirmed for gap on {symbol} (ratio: {volume_ratio:.2f})")
            else:
                logger.debug(f"Waiting for volume confirmation for gap on {symbol}")
                return False
        
        # Check first bar confirmation if required
        if self.parameters['require_first_bar_confirmation'] and not gap_info.get('first_bar_confirmed', False):
            # Get first completed bar after open
            if len(data) < 2:
                return False
                
            # Check if bar closed in expected direction
            first_bar_open = data['open'].iloc[-2]
            first_bar_close = data['close'].iloc[-2]
            
            # For gap up, we want first bar to close above its open (continuation)
            # For gap down, we want first bar to close below its open (continuation)
            is_continuation_confirmed = False
            if gap_info['gap_direction'] == 'up' and first_bar_close > first_bar_open:
                is_continuation_confirmed = True
            elif gap_info['gap_direction'] == 'down' and first_bar_close < first_bar_open:
                is_continuation_confirmed = True
            
            # Store confirmation result
            gap_info['first_bar_confirmed'] = True
            gap_info['is_continuation_confirmed'] = is_continuation_confirmed
            
            logger.debug(f"First bar confirmation: {'success' if is_continuation_confirmed else 'failed'} for gap on {symbol}")
        
        # Determine trade direction based on gap direction and strategy mode
        strategy_mode = self.parameters['strategy_mode']
        gap_direction = gap_info['gap_direction']
        
        # Decide on trade direction
        if strategy_mode == 'continuation':
            # Gap and Go: Trade in direction of gap
            gap_info['trade_direction'] = 'long' if gap_direction == 'up' else 'short'
        elif strategy_mode == 'fade':
            # Gap Fade: Trade against direction of gap
            gap_info['trade_direction'] = 'short' if gap_direction == 'up' else 'long'
        else:  # 'both' mode
            # Either continuation or fade based on first bar and confirmation indicators
            continuation_preference = self.parameters['continuation_preference']
            
            # Get first bar confirmation result
            is_continuation_confirmed = gap_info.get('is_continuation_confirmed', False)
            
            # Use RSI to help determine direction
            rsi = self.indicators.get('rsi', 50)
            
            # For gap up: High RSI favors fade, low RSI favors continuation
            # For gap down: Low RSI favors fade, high RSI favors continuation
            rsi_favors_continuation = False
            if gap_direction == 'up' and rsi < self.parameters['rsi_overbought']:
                rsi_favors_continuation = True
            elif gap_direction == 'down' and rsi > self.parameters['rsi_oversold']:
                rsi_favors_continuation = True
            
            # Weight the factors
            continuation_score = continuation_preference
            if is_continuation_confirmed:
                continuation_score += 0.25
            if rsi_favors_continuation:
                continuation_score += 0.25
                
            # Decide final direction
            if continuation_score >= 0.5:
                # Continuation trade (Gap and Go)
                gap_info['trade_direction'] = 'long' if gap_direction == 'up' else 'short'
            else:
                # Fade trade (Gap Fade)
                gap_info['trade_direction'] = 'short' if gap_direction == 'up' else 'long'
        
        # Set entry price
        gap_info['entry_price'] = data['close'].iloc[-1]
        
        # Log entry conditions met
        trade_direction = gap_info['trade_direction']
        trade_type = 'continuation' if (
            (gap_direction == 'up' and trade_direction == 'long') or
            (gap_direction == 'down' and trade_direction == 'short')
        ) else 'fade'
        
        logger.info(f"Entry conditions met for {trade_type} gap trade on {symbol}: {gap_direction} gap, {trade_direction} position")
        
        # Increment gap trades counter
        self.gap_trades_today += 1
        
        return True
    
    def _update_signals(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Update active signals and remove expired ones.
        
        Args:
            data: Market data DataFrame
            symbol: Symbol to update signals for
        """
        # Check if we have a detected gap for this symbol
        if symbol not in self.gaps_detected:
            return
        
        gap_info = self.gaps_detected[symbol]
        
        # Check if the signal is still active
        if not gap_info.get('active', False):
            return
        
        # Check if the signal has expired
        detection_time = gap_info.get('detection_time')
        if detection_time is not None:
            # Calculate signal age in minutes
            signal_age = (datetime.now() - detection_time).total_seconds() / 60
            
            # Check if signal has expired
            if signal_age > self.parameters['signal_decay_minutes']:
                gap_info['active'] = False
                logger.info(f"Gap signal for {symbol} expired after {signal_age:.1f} minutes")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for gap trading.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate VWAP
        for period in self.parameters['vwap_periods']:
            # Calculate Typical Price
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            
            # Calculate VWAP
            cumulative_tp_vol = (typical_price * data['volume']).rolling(window=period).sum()
            cumulative_vol = data['volume'].rolling(window=period).sum()
            
            vwap = cumulative_tp_vol / cumulative_vol
            indicators[f'vwap_{period}'] = vwap.iloc[-1] if not vwap.empty else None
        
        # Calculate RSI
        period = self.parameters['rsi_period']
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate ATR for volatility measurement
        atr_period = self.parameters['atr_period']
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()
        indicators['atr'] = atr.iloc[-1] if not atr.empty else (data['high'].iloc[-1] - data['low'].iloc[-1])
        
        # Additional indicators for confirmation
        
        # Calculate moving averages for trend confirmation
        indicators['sma_20'] = data['close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else None
        indicators['sma_50'] = data['close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else None
        
        # Calculate Bollinger Bands for volatility
        sma_20 = data['close'].rolling(window=20).mean()
        std_20 = data['close'].rolling(window=20).std()
        
        indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1] if len(data) >= 20 else None
        indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1] if len(data) >= 20 else None
        indicators['bb_width'] = ((indicators['bb_upper'] - indicators['bb_lower']) / sma_20.iloc[-1]) if len(data) >= 20 else None
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on detected gaps and indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {}
        symbol = self.session.symbol
        
        # Check if we have a detected gap for this symbol
        if symbol not in self.gaps_detected or not self.gaps_detected[symbol].get('active', False):
            return signals
        
        # Get gap info
        gap_info = self.gaps_detected[symbol]
        
        # Check if entry conditions are met
        if self._check_entry_conditions(data, symbol):
            # Create entry signal
            trade_direction = gap_info['trade_direction']
            entry_price = gap_info['entry_price']
            
            # Create a unique signal ID
            signal_id = str(uuid.uuid4())
            
            # Determine signal type
            signal_type = SignalType.LONG if trade_direction == 'long' else SignalType.SHORT
            
            # Create signal object
            signal = Signal(
                id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                timeframe=self.session.timeframe,
                price=entry_price,
                confidence=0.8,  # High confidence for gap trades
                metadata={
                    'strategy': self.name,
                    'gap_percent': gap_info['gap_percent'],
                    'gap_direction': gap_info['gap_direction'],
                    'trade_type': 'continuation' if (
                        (gap_info['gap_direction'] == 'up' and trade_direction == 'long') or
                        (gap_info['gap_direction'] == 'down' and trade_direction == 'short')
                    ) else 'fade',
                    'atr': gap_info['atr']
                }
            )
            
            signals[signal_id] = signal
            
            # Log signal generation
            logger.info(f"Generated {trade_direction} signal for {symbol} gap trade: {gap_info['gap_percent']:.2f}% {gap_info['gap_direction']} gap")
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size based on gap characteristics and risk parameters.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in number of shares
        """
        symbol = self.session.symbol
        
        # Get account balance (placeholder - would come from the actual broker in production)
        account_balance = 100000.0  # Example value
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Get ATR for volatility-based position sizing
        atr = indicators.get('atr', data['high'].iloc[-1] - data['low'].iloc[-1])
        
        # Calculate stop loss distance based on ATR
        atr_multiplier = self.parameters['atr_multiplier']
        stop_distance = atr * atr_multiplier
        
        # Calculate risk amount based on max risk per trade
        max_risk_percent = self.parameters['max_risk_per_trade_percent'] / 100.0
        risk_amount = account_balance * max_risk_percent
        
        # Calculate position size based on stop distance
        shares = risk_amount / stop_distance
        
        # Convert to nearest lot size (typically 100 shares for stocks)
        lot_size = self.session.lot_size
        shares = round(shares / lot_size) * lot_size
        
        # Ensure we have at least one lot
        if shares < lot_size:
            shares = lot_size
        
        logger.info(f"Calculated position size for {symbol}: {shares} shares ({direction})")
        return shares
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile', etc.)
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Gap trading works well in choppy and volatile markets,
        # especially around earnings seasons.
        compatibility_map = {
            'trending': 0.6,     # Gap and Go works in trending markets
            'ranging': 0.7,      # Gap Fade works well in ranging markets
            'volatile': 0.8,     # Both approaches work well in volatile markets
            'calm': 0.4          # Fewer gaps in calm markets
        }
        
        return compatibility_map.get(market_regime, 0.5)
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"GapTradingStrategy(symbol={self.session.symbol}, mode={self.parameters['strategy_mode']})"
    
    def __repr__(self) -> str:
        """Detailed representation of the strategy."""
        return f"GapTradingStrategy(symbol={self.session.symbol}, timeframe={self.session.timeframe}, " \
               f"mode={self.parameters['strategy_mode']}, min_gap={self.parameters['min_gap_percent']}%, " \
               f"max_gap={self.parameters['max_gap_percent']}%)"
