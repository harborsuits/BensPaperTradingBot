#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Breakout Strategy Module

This module implements a breakout trading strategy for forex markets. It identifies 
significant price levels and generates entry signals when price breaks through these levels,
indicating potential trend continuation or reversal.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Import base strategy
from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="ForexBreakoutStrategy",
    market_type="forex",
    description="A strategy that identifies key support/resistance levels and trades breakouts when price moves beyond these levels with sufficient momentum",
    timeframes=["15m", "1h", "4h", "1d"],
    parameters={
        # Breakout detection parameters
        "breakout_type": {
            "type": "str",
            "default": "range",
            "enum": ["range", "channel", "pattern", "horizontal_level", "pivot"],
            "description": "Type of breakout to detect"
        },
        "lookback_periods": {
            "type": "int",
            "default": 20,
            "description": "Number of periods to look back for establishing range/channel"
        },
        "min_range_pips": {
            "type": "float",
            "default": 30.0,
            "description": "Minimum range size in pips for valid breakout setup"
        },
        "consolidation_periods": {
            "type": "int",
            "default": 8,
            "description": "Minimum periods of consolidation required"
        },
        
        # Confirmation parameters
        "breakout_confirmation": {
            "type": "str",
            "default": "momentum",
            "enum": ["momentum", "close", "time", "volume", "all"],
            "description": "Method to confirm breakout validity"
        },
        "momentum_threshold": {
            "type": "float",
            "default": 1.5,
            "description": "Multiple of ATR required for momentum confirmation"
        },
        "close_confirmation_periods": {
            "type": "int",
            "default": 2,
            "description": "Number of closes above/below level required for confirmation"
        },
        "volume_increase_factor": {
            "type": "float",
            "default": 1.5,
            "description": "Required volume increase for volume confirmation"
        },
        
        # Trade management parameters
        "entry_type": {
            "type": "str",
            "default": "market",
            "enum": ["market", "limit", "stop"],
            "description": "Order type for breakout entries"
        },
        "stop_placement": {
            "type": "str",
            "default": "atr",
            "enum": ["atr", "swing", "percentage", "support_resistance"],
            "description": "Method for stop loss placement"
        },
        "stop_distance_atr": {
            "type": "float",
            "default": 1.5,
            "description": "Stop distance as ATR multiple"
        },
        "profit_target_method": {
            "type": "str",
            "default": "risk_reward",
            "enum": ["risk_reward", "fibonacci", "swing", "atr", "next_level"],
            "description": "Method for determining profit targets"
        },
        "risk_reward_ratio": {
            "type": "float",
            "default": 2.0,
            "description": "Target risk-reward ratio for profit targets"
        },
        "partial_exit_levels": {
            "type": "list",
            "default": [0.5, 0.75, 1.0],
            "description": "Levels for partial profit taking as multiples of risk"
        },
        "trailing_stop_enable": {
            "type": "bool",
            "default": True,
            "description": "Whether to use trailing stops"
        },
        "trailing_stop_activation": {
            "type": "float",
            "default": 1.0,
            "description": "Risk multiple at which to activate trailing stop"
        },
        
        # Timeframes and session parameters
        "enable_session_filter": {
            "type": "bool",
            "default": True,
            "description": "Whether to filter by trading session"
        },
        "target_sessions": {
            "type": "list",
            "default": ["london", "new_york", "asian", "overlap"],
            "description": "Trading sessions to generate signals in"
        },
        "multi_timeframe_confirmation": {
            "type": "bool",
            "default": True,
            "description": "Whether to use multiple timeframes for confirmation"
        },
        "higher_timeframe_factor": {
            "type": "int",
            "default": 4,
            "description": "Factor to determine higher timeframe (e.g., 4 for 1h -> a 4h)"
        },
        
        # Risk management
        "max_risk_per_trade": {
            "type": "float",
            "default": 0.01,
            "description": "Maximum risk per trade as fraction of account"
        },
        "max_daily_loss": {
            "type": "float",
            "default": 0.03,
            "description": "Maximum daily loss as fraction of account"
        },
        "max_open_trades": {
            "type": "int",
            "default": 3,
            "description": "Maximum number of concurrent open trades"
        }
    }
)
class ForexBreakoutStrategy(ForexBaseStrategy):
    """
    A forex breakout strategy that identifies key support/resistance levels and
    trades breakouts when price moves beyond these levels with sufficient momentum.
    
    This strategy:
    1. Identifies potential breakout levels based on price ranges, channels, or pivots
    2. Monitors for breakouts of these levels with confirmation criteria
    3. Enters trades in the direction of the breakout with appropriate risk management
    4. Uses multiple timeframes and session awareness for improved accuracy
    5. Manages trades with trailing stops and partial profit taking
    """
    
    def __init__(self, session: ForexSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Forex Breakout strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize strategy-specific state variables
        self.breakout_levels = {}  # Store detected breakout levels
        self.pending_breakouts = {}  # Store breakouts awaiting confirmation
        self.confirmed_breakouts = {}  # Store confirmed breakouts
        self.active_trades = {}  # Track active trades
        self.trade_stats = {}  # Track trade performance stats
        
        # Session windows (in UTC)
        self.trading_sessions = {
            "asian": {"start": 0, "end": 9},  # 00:00-09:00 UTC
            "london": {"start": 8, "end": 16},  # 08:00-16:00 UTC
            "new_york": {"start": 13, "end": 21},  # 13:00-21:00 UTC
            "overlap": {"start": 13, "end": 16}  # 13:00-16:00 UTC (London/NY overlap)
        }
        
        logger.info(f"Forex Breakout Strategy initialized with {self.parameters['breakout_type']} breakout type")
    
    def _identify_current_session(self, timestamp: datetime) -> List[str]:
        """
        Identify which trading sessions are currently active.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of active sessions
        """
        hour_utc = timestamp.hour
        active_sessions = []
        
        for session_name, session_hours in self.trading_sessions.items():
            if session_hours["start"] <= hour_utc < session_hours["end"]:
                active_sessions.append(session_name)
        
        return active_sessions
    
    def _is_session_tradable(self, timestamp: datetime) -> bool:
        """
        Check if current session is in the tradable sessions list.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Boolean indicating if current session is tradable
        """
        if not self.parameters["enable_session_filter"]:
            return True
            
        active_sessions = self._identify_current_session(timestamp)
        target_sessions = self.parameters["target_sessions"]
        
        return any(session in target_sessions for session in active_sessions)
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        Args:
            data: Price data
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period:
            return 0.0
            
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _detect_range_breakout(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect range breakouts in the data.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with breakout information
        """
        if len(data) < self.parameters["lookback_periods"]:
            return {}
            
        # Get relevant window for range detection
        window = data.iloc[-self.parameters["lookback_periods"]:]
        
        # Calculate range
        range_high = window["high"].max()
        range_low = window["low"].min()
        range_size = range_high - range_low
        
        # Convert to pips (approximate for example, would be currency-pair specific)
        pair = self.session.symbol
        point_value = 0.0001 if "JPY" not in pair else 0.01
        range_pips = range_size / point_value
        
        # Check if range is significant enough
        if range_pips < self.parameters["min_range_pips"]:
            return {}
        
        # Check for consolidation
        recent_window = data.iloc[-self.parameters["consolidation_periods"]:]
        recent_high = recent_window["high"].max()
        recent_low = recent_window["low"].min()
        recent_range = recent_high - recent_low
        
        # If recent range is much smaller than overall range, it's consolidating
        is_consolidating = recent_range < range_size * 0.7
        
        # Get current price and check for breakout
        current_price = data["close"].iloc[-1]
        previous_close = data["close"].iloc[-2]
        
        breakout_upward = current_price > range_high and previous_close <= range_high
        breakout_downward = current_price < range_low and previous_close >= range_low
        
        if not (breakout_upward or breakout_downward):
            return {}
            
        # Calculate momentum
        atr = self.calculate_atr(data)
        if breakout_upward:
            momentum = (current_price - range_high) / atr if atr > 0 else 0
            breakout_level = range_high
            direction = "long"
        else:  # breakout_downward
            momentum = (range_low - current_price) / atr if atr > 0 else 0
            breakout_level = range_low
            direction = "short"
        
        # Create breakout data
        breakout_data = {
            "type": "range",
            "level": breakout_level,
            "direction": direction,
            "momentum": momentum,
            "range_high": range_high,
            "range_low": range_low,
            "range_pips": range_pips,
            "atr": atr,
            "timestamp": data.index[-1],
            "confirmed": momentum >= self.parameters["momentum_threshold"]
        }
        
        return breakout_data
    
    def _detect_channel_breakout(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect channel breakouts (upper and lower trendlines).
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with breakout information
        """
        # This would implement trendline detection and breakout identification
        # For simplicity, using a placeholder implementation
        return {}
    
    def _detect_pivot_breakout(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect breakouts of significant pivot points.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with breakout information
        """
        # This would implement pivot point detection and breakout identification
        # For simplicity, using a placeholder implementation
        return {}
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for breakout detection and confirmation.
        
        Args:
            data: Market data DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty or len(data) < self.parameters["lookback_periods"]:
            return {}
        
        indicators = {}
        
        # Calculate ATR for volatility assessment
        indicators["atr"] = self.calculate_atr(data)
        
        # Calculate volume indicators if needed for confirmation
        if "volume" in self.parameters["breakout_confirmation"]:
            indicators["volume_sma"] = data["volume"].rolling(window=20).mean()
            indicators["relative_volume"] = data["volume"] / indicators["volume_sma"]
        
        # Calculate momentum indicators
        indicators["rsi"] = self._calculate_rsi(data["close"])
        
        # Calculate moving averages for trend context
        indicators["sma_50"] = data["close"].rolling(window=50).mean()
        indicators["sma_200"] = data["close"].rolling(window=200).mean()
        
        # Determine current market structure
        if len(data) >= 50:
            indicators["market_structure"] = "uptrend" if indicators["sma_50"].iloc[-1] > indicators["sma_200"].iloc[-1] else "downtrend"
        else:
            indicators["market_structure"] = "unknown"
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_breakout(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential breakouts based on specified breakout type.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Breakout data if detected, empty dict otherwise
        """
        breakout_type = self.parameters["breakout_type"]
        
        if breakout_type == "range":
            return self._detect_range_breakout(data)
        elif breakout_type == "channel":
            return self._detect_channel_breakout(data)
        elif breakout_type == "pivot":
            return self._detect_pivot_breakout(data)
        else:
            # Other breakout types would be implemented similarly
            return {}
    
    def confirm_breakout(self, breakout_data: Dict[str, Any], data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Apply confirmation criteria to a detected breakout.
        
        Args:
            breakout_data: Detected breakout information
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Whether the breakout is confirmed
        """
        if not breakout_data:
            return False
            
        confirmation_type = self.parameters["breakout_confirmation"]
        
        # Momentum confirmation
        if confirmation_type in ["momentum", "all"]:
            if breakout_data["momentum"] < self.parameters["momentum_threshold"]:
                return False
        
        # Close confirmation (multiple closes beyond level)
        if confirmation_type in ["close", "all"]:
            level = breakout_data["level"]
            direction = breakout_data["direction"]
            closes = data["close"].iloc[-self.parameters["close_confirmation_periods"]:]
            
            if direction == "long":
                if not all(close > level for close in closes):
                    return False
            else:  # short
                if not all(close < level for close in closes):
                    return False
        
        # Volume confirmation
        if confirmation_type in ["volume", "all"] and "relative_volume" in indicators:
            if indicators["relative_volume"].iloc[-1] < self.parameters["volume_increase_factor"]:
                return False
        
        # Time confirmation (not implemented here)
        
        return True
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on breakout detection and confirmation.
        
        Args:
            data: Market data DataFrame with OHLCV data
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        if data.empty or not indicators:
            return {}
        
        signals = {
            "long_entry": False,
            "long_exit": False,
            "short_entry": False,
            "short_exit": False,
            "signal_strength": 0.0,
            "entry_type": None,
            "exit_type": None
        }
        
        symbol = self.session.symbol
        current_price = data["close"].iloc[-1]
        current_time = data.index[-1]
        
        # Check if current session is tradable
        if not self._is_session_tradable(current_time):
            return signals
        
        # Detect potential breakout
        breakout_data = self.detect_breakout(data, indicators)
        
        # If breakout detected, check for confirmation
        if breakout_data:
            # Store/update breakout data
            if symbol not in self.pending_breakouts:
                self.pending_breakouts[symbol] = breakout_data
            
            # Check if breakout is confirmed
            is_confirmed = self.confirm_breakout(breakout_data, data, indicators)
            
            if is_confirmed:
                direction = breakout_data["direction"]
                
                if direction == "long":
                    signals["long_entry"] = True
                    signals["entry_type"] = f"{breakout_data['type']}_breakout_long"
                else:  # short
                    signals["short_entry"] = True
                    signals["entry_type"] = f"{breakout_data['type']}_breakout_short"
                
                # Calculate signal strength based on momentum and confirmation
                signals["signal_strength"] = min(1.0, breakout_data["momentum"] / self.parameters["momentum_threshold"])
                
                # Store confirmed breakout
                self.confirmed_breakouts[symbol] = breakout_data
                
                logger.info(f"Confirmed {direction} breakout for {symbol} at {current_price}")
        
        # Check for exit signals on existing positions
        position_exists = self._position_exists(symbol)
        if position_exists:
            position = self._get_position(symbol)
            position_direction = position.get("direction", "")
            
            # Implement exit logic (taking profits, trailing stops, etc.)
            # For simplicity, this is placeholder logic
            
            # Example: Exit on opposite breakout
            if breakout_data and is_confirmed and breakout_data["direction"] != position_direction:
                if position_direction == "long":
                    signals["long_exit"] = True
                else:
                    signals["short_exit"] = True
                
                signals["exit_type"] = "opposite_breakout"
                signals["signal_strength"] = 0.8
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size as a decimal (0.0-1.0) representing account percentage
        """
        if data.empty or not indicators:
            return 0.0
        
        symbol = self.session.symbol
        current_price = data["close"].iloc[-1]
        
        # Get breakout data to determine stop placement
        breakout_data = self.confirmed_breakouts.get(symbol, {})
        if not breakout_data:
            return 0.0
        
        # Calculate stop distance based on selected method
        stop_distance_pips = 0.0
        point_value = 0.0001 if "JPY" not in symbol else 0.01
        
        if self.parameters["stop_placement"] == "atr":
            atr = indicators.get("atr", 0.0)
            stop_distance_pips = (atr * self.parameters["stop_distance_atr"]) / point_value
        
        elif self.parameters["stop_placement"] == "swing":
            # For simplicity, using a fixed percentage of the range
            range_pips = breakout_data.get("range_pips", 0.0)
            stop_distance_pips = range_pips * 0.3
        
        elif self.parameters["stop_placement"] == "percentage":
            # Fixed percentage of current price
            stop_distance_pips = (current_price * 0.01) / point_value  # 1% stop
        
        elif self.parameters["stop_placement"] == "support_resistance":
            # Use the breakout level as the stop
            level = breakout_data.get("level", current_price)
            if direction == "long":
                stop_distance_pips = (current_price - level) / point_value
            else:
                stop_distance_pips = (level - current_price) / point_value
        
        # Ensure minimum stop distance
        stop_distance_pips = max(stop_distance_pips, 10.0)  # Minimum 10 pips
        
        # Calculate position size based on risk per trade
        # This is a simplified calculation - would need actual account balance in production
        account_balance = 10000.0  # Placeholder
        risk_amount = account_balance * self.parameters["max_risk_per_trade"]
        pip_value = 1.0  # Placeholder - would be calculated based on lot size and pair
        
        # Calculate position size in lots based on risk
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Apply checks against max open trades
        current_open_trades = len(self._get_active_positions())
        if current_open_trades >= self.parameters["max_open_trades"]:
            position_size = 0.0
        
        # Apply daily loss checks
        daily_loss = self._calculate_daily_loss()
        if daily_loss >= self.parameters["max_daily_loss"]:
            position_size = 0.0
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.2f} lots with {stop_distance_pips:.1f} pips stop")
        
        return position_size
    
    def _position_exists(self, symbol: str) -> bool:
        """Check if a position exists for the symbol."""
        # In a real implementation, this would check with the position manager
        return symbol in self.active_trades
    
    def _get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position data for a symbol."""
        return self.active_trades.get(symbol, {})
    
    def _get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        return self.active_trades
    
    def _calculate_daily_loss(self) -> float:
        """Calculate the loss for the current day as a fraction of account."""
        # This would track actual trades and calculate daily P&L
        return 0.0  # Placeholder
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the forex breakout strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "ranging": 0.30,               # Poor in pure ranging markets
            "trending": 0.85,              # Very good in trending markets
            "volatile_trending": 0.75,     # Good in volatile trending markets
            "volatile_ranging": 0.40,      # Below average in volatile ranging markets
            "low_volatility": 0.25,        # Poor in low volatility (no breakouts)
            "high_volatility": 0.80,       # Very good in high volatility
            "news_driven": 0.90,           # Excellent in news-driven markets 
            "asian_session": 0.30,         # Poor in typically quiet Asian sessions
            "london_session": 0.85,        # Very good in typically active London session
            "new_york_session": 0.80,      # Very good in typically active NY session
            "session_overlap": 0.95,       # Excellent during session overlaps
            "pre_news": 0.40,              # Below average before major news (false breakouts)
            "post_news": 0.90,             # Excellent after major news (real breakouts)
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
