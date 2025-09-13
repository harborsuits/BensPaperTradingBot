#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session-Aware Forex Scalping Strategy Module

This module implements a scalping strategy for forex markets that is specifically 
designed to adapt to different trading sessions (Asian, London, New York) and their 
unique volatility and liquidity characteristics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta, time
import pytz

# Import base strategy
from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="SessionAwareForexScalpingStrategy",
    market_type="forex",
    description="A session-aware forex scalping strategy that adapts parameters and execution based on the current trading session (Asian, London, New York) and their unique characteristics",
    timeframes=["1m", "5m", "15m"],
    parameters={
        # Session configuration
        "enabled_sessions": {
            "type": "list",
            "default": ["asian", "london", "new_york", "london_ny_overlap"],
            "description": "Trading sessions to enable for this strategy"
        },
        "session_timezone": {
            "type": "str",
            "default": "UTC",
            "description": "Timezone for session calculations"
        },
        "asian_session_start": {
            "type": "str",
            "default": "00:00",
            "description": "Asian session start time (24-hour format in session_timezone)"
        },
        "asian_session_end": {
            "type": "str",
            "default": "09:00",
            "description": "Asian session end time (24-hour format in session_timezone)"
        },
        "london_session_start": {
            "type": "str",
            "default": "08:00",
            "description": "London session start time (24-hour format in session_timezone)"
        },
        "london_session_end": {
            "type": "str",
            "default": "16:00",
            "description": "London session end time (24-hour format in session_timezone)"
        },
        "newyork_session_start": {
            "type": "str",
            "default": "13:00",
            "description": "New York session start time (24-hour format in session_timezone)"
        },
        "newyork_session_end": {
            "type": "str",
            "default": "21:00",
            "description": "New York session end time (24-hour format in session_timezone)"
        },
        
        # Entry parameters (will have session-specific overrides)
        "volatility_filter": {
            "type": "bool",
            "default": True,
            "description": "Whether to use volatility filtering for entries"
        },
        "min_volatility_atr": {
            "type": "float",
            "default": 0.0001,
            "description": "Minimum ATR (in price terms) for entry"
        },
        "max_volatility_atr": {
            "type": "float",
            "default": 0.0010,
            "description": "Maximum ATR (in price terms) for entry"
        },
        "min_spread_pips": {
            "type": "float", 
            "default": 0.5,
            "description": "Minimum acceptable spread in pips"
        },
        "max_spread_pips": {
            "type": "float",
            "default": 3.0,
            "description": "Maximum acceptable spread in pips"
        },
        
        # Scalping trade parameters
        "atr_period": {
            "type": "int",
            "default": 14,
            "description": "Period for ATR calculation"
        },
        "profit_target_pips": {
            "type": "float",
            "default": 10.0,
            "description": "Profit target in pips"
        },
        "stop_loss_pips": {
            "type": "float",
            "default": 5.0,
            "description": "Stop loss in pips"
        },
        "profit_target_atr_multiple": {
            "type": "float",
            "default": 1.5,
            "description": "Profit target as multiple of ATR"
        },
        "stop_loss_atr_multiple": {
            "type": "float",
            "default": 0.75,
            "description": "Stop loss as multiple of ATR"
        },
        "use_dynamic_targets": {
            "type": "bool",
            "default": True,
            "description": "Whether to use ATR-based targets/stops or fixed pips"
        },
        
        # Risk management
        "max_risk_per_trade": {
            "type": "float",
            "default": 0.005,
            "description": "Maximum risk per trade as fraction of account"
        },
        "max_daily_loss": {
            "type": "float",
            "default": 0.02,
            "description": "Maximum daily loss as fraction of account"
        },
        "max_session_loss": {
            "type": "float",
            "default": 0.01,
            "description": "Maximum loss per session as fraction of account"
        },
        "max_open_trades": {
            "type": "int",
            "default": 3,
            "description": "Maximum number of concurrent open trades"
        },
        
        # Session-specific pair preferences (1-10 scale, 10 being best)
        "asian_pairs_ranking": {
            "type": "dict",
            "default": {
                "USDJPY": 9, "AUDJPY": 8, "EURJPY": 7, "GBPJPY": 6, 
                "AUDUSD": 7, "NZDUSD": 6, "EURUSD": 5
            },
            "description": "Ranking of pairs for Asian session"
        },
        "london_pairs_ranking": {
            "type": "dict",
            "default": {
                "GBPUSD": 9, "EURGBP": 8, "GBPJPY": 7, "EURUSD": 8,
                "EURJPY": 7, "USDCHF": 6, "GBPCHF": 6
            },
            "description": "Ranking of pairs for London session"
        },
        "newyork_pairs_ranking": {
            "type": "dict",
            "default": {
                "EURUSD": 9, "USDJPY": 8, "USDCAD": 8, "GBPUSD": 7,
                "AUDUSD": 6, "USDCHF": 6
            },
            "description": "Ranking of pairs for New York session"
        }
    }
)
class SessionAwareForexScalpingStrategy(ForexBaseStrategy):
    """
    A session-aware scalping strategy for forex markets that adapts to the unique
    characteristics of different trading sessions.
    
    This strategy:
    1. Identifies the current active trading session(s)
    2. Applies session-specific parameters for entry, exit, and risk management
    3. Filters currency pairs based on session-specific liquidity and volatility profiles
    4. Uses tight stops and quick profit targets optimized for scalping
    5. Monitors market microstructure and spread conditions to avoid high-cost environments
    """
    
    def __init__(self, session: ForexSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Session-Aware Forex Scalping strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize strategy-specific state variables
        self.active_trades = {}
        self.daily_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0
        }
        self.session_stats = {
            "asian": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "london": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "new_york": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "overlap": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        }
        
        # Session definition (parsed from time strings)
        self.sessions = self._initialize_sessions()
        
        # Track current session
        self.current_session = None
        self.last_session_update = None
        
        # Session-specific parameters (initialized from base parameters)
        self.session_parameters = self._initialize_session_parameters()
        
        logger.info(f"Session-Aware Forex Scalping Strategy initialized for {self.session.symbol}")
    
    def _initialize_sessions(self) -> Dict[str, Dict[str, time]]:
        """
        Initialize session time windows based on parameters.
        
        Returns:
            Dictionary of session definitions with start and end times
        """
        sessions = {}
        
        # Parse time strings to datetime.time objects
        try:
            # Asian session
            asian_start = datetime.strptime(self.parameters["asian_session_start"], "%H:%M").time()
            asian_end = datetime.strptime(self.parameters["asian_session_end"], "%H:%M").time()
            sessions["asian"] = {"start": asian_start, "end": asian_end}
            
            # London session
            london_start = datetime.strptime(self.parameters["london_session_start"], "%H:%M").time()
            london_end = datetime.strptime(self.parameters["london_session_end"], "%H:%M").time()
            sessions["london"] = {"start": london_start, "end": london_end}
            
            # New York session
            ny_start = datetime.strptime(self.parameters["newyork_session_start"], "%H:%M").time()
            ny_end = datetime.strptime(self.parameters["newyork_session_end"], "%H:%M").time()
            sessions["new_york"] = {"start": ny_start, "end": ny_end}
            
            # London-NY overlap (derived from individual sessions)
            overlap_start = max(london_start, ny_start)
            overlap_end = min(london_end, ny_end)
            sessions["london_ny_overlap"] = {"start": overlap_start, "end": overlap_end}
            
        except ValueError as e:
            logger.error(f"Error parsing session times: {str(e)}")
            # Fall back to defaults
            sessions = {
                "asian": {"start": time(0, 0), "end": time(9, 0)},
                "london": {"start": time(8, 0), "end": time(16, 0)},
                "new_york": {"start": time(13, 0), "end": time(21, 0)},
                "london_ny_overlap": {"start": time(13, 0), "end": time(16, 0)}
            }
        
        return sessions
    
    def _initialize_session_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize session-specific parameter sets.
        
        Returns:
            Dictionary of session-specific parameters
        """
        # Base parameters
        base_params = {k: v for k, v in self.parameters.items()}
        
        # Create session-specific parameter sets
        session_parameters = {
            # Asian session - typically lower volatility, tighter ranges
            "asian": {
                "profit_target_pips": 5.0,
                "stop_loss_pips": 4.0,
                "profit_target_atr_multiple": 1.2,
                "stop_loss_atr_multiple": 0.6,
                "max_spread_pips": 1.5,
                "max_risk_per_trade": 0.004
            },
            
            # London session - higher volatility, trending moves
            "london": {
                "profit_target_pips": 8.0,
                "stop_loss_pips": 5.0,
                "profit_target_atr_multiple": 1.5,
                "stop_loss_atr_multiple": 0.75,
                "max_spread_pips": 2.0,
                "max_risk_per_trade": 0.005
            },
            
            # New York session - can be volatile, good for momentum
            "new_york": {
                "profit_target_pips": 7.0,
                "stop_loss_pips": 5.0,
                "profit_target_atr_multiple": 1.4,
                "stop_loss_atr_multiple": 0.7,
                "max_spread_pips": 2.0,
                "max_risk_per_trade": 0.005
            },
            
            # London-NY overlap - highest volatility, best for scalping
            "london_ny_overlap": {
                "profit_target_pips": 10.0,
                "stop_loss_pips": 6.0,
                "profit_target_atr_multiple": 1.7,
                "stop_loss_atr_multiple": 0.8,
                "max_spread_pips": 2.5,
                "max_risk_per_trade": 0.006
            }
        }
        
        # For each session, merge with base parameters
        for session_name, session_params in session_parameters.items():
            session_parameters[session_name] = {**base_params, **session_params}
        
        return session_parameters
    
    def _detect_active_sessions(self, timestamp: datetime) -> List[str]:
        """
        Detect which forex trading sessions are currently active.
        
        Args:
            timestamp: Current datetime
            
        Returns:
            List of active session names
        """
        # Convert to session timezone if not already
        session_tz = pytz.timezone(self.parameters["session_timezone"])
        if timestamp.tzinfo is None:
            timestamp = pytz.utc.localize(timestamp)
        session_time = timestamp.astimezone(session_tz).time()
        
        active_sessions = []
        
        # Check each session to see if current time falls within its range
        for session_name, session_times in self.sessions.items():
            start_time = session_times["start"]
            end_time = session_times["end"]
            
            # Handle sessions that cross midnight
            if start_time <= end_time:
                if start_time <= session_time <= end_time:
                    active_sessions.append(session_name)
            else:  # Session crosses midnight
                if session_time >= start_time or session_time <= end_time:
                    active_sessions.append(session_name)
        
        return active_sessions
    
    def _update_current_session(self, timestamp: datetime) -> None:
        """
        Update the current active session based on timestamp.
        
        Args:
            timestamp: Current datetime
        """
        # Only update periodically (every 5 minutes) to avoid excessive processing
        if (self.last_session_update is not None and 
                timestamp - self.last_session_update < timedelta(minutes=5)):
            return
        
        # Detect active sessions
        active_sessions = self._detect_active_sessions(timestamp)
        
        # Determine primary session - prioritize overlap, then individual sessions
        if "london_ny_overlap" in active_sessions:
            self.current_session = "london_ny_overlap"
        elif "london" in active_sessions:
            self.current_session = "london"
        elif "new_york" in active_sessions:
            self.current_session = "new_york"
        elif "asian" in active_sessions:
            self.current_session = "asian"
        else:
            self.current_session = None
        
        self.last_session_update = timestamp
        
        if self.current_session:
            logger.info(f"Current trading session: {self.current_session}")
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for the current session.
        
        Returns:
            Dictionary of session-specific parameters
        """
        if not self.current_session or self.current_session not in self.session_parameters:
            # Fall back to default parameters if no active session
            return self.parameters
        
        return self.session_parameters[self.current_session]
    
    def _is_pair_suitable_for_session(self, pair: str, session: str) -> Tuple[bool, float]:
        """
        Determine if a currency pair is suitable for the current session.
        
        Args:
            pair: Currency pair symbol
            session: Session name
            
        Returns:
            Tuple of (is_suitable, ranking_score)
        """
        # Get the ranking for this pair in the current session
        if session == "asian":
            rankings = self.parameters["asian_pairs_ranking"]
        elif session == "london":
            rankings = self.parameters["london_pairs_ranking"]
        elif session == "new_york":
            rankings = self.parameters["newyork_pairs_ranking"]
        elif session == "london_ny_overlap":
            # For overlap, use the best ranking from either London or NY
            london_rankings = self.parameters["london_pairs_ranking"]
            ny_rankings = self.parameters["newyork_pairs_ranking"]
            london_score = london_rankings.get(pair, 0)
            ny_score = ny_rankings.get(pair, 0)
            return (london_score >= 6 or ny_score >= 6), max(london_score, ny_score) / 10.0
        else:
            # Default to empty rankings if session not recognized
            rankings = {}
        
        # Get ranking score, defaulting to 0 if not found
        score = rankings.get(pair, 0)
        
        # Consider pairs with ranking 6 or higher as suitable
        return score >= 6, score / 10.0
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy, with parameters adapted to the current session.
        
        Args:
            data: Market data DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty or len(data) < 20:  # Need at least 20 bars for reliable indicators
            return {}
        
        # Update current session based on the timestamp of the latest bar
        latest_time = data.index[-1]
        self._update_current_session(latest_time)
        
        # Get session-specific parameters
        params = self._get_current_parameters()
        
        # Initialize indicators dictionary
        indicators = {
            "session": self.current_session,
            "session_score": 0.0,
        }
        
        # Calculate pair suitability for current session
        pair = self.session.symbol
        is_suitable, session_score = self._is_pair_suitable_for_session(
            pair, self.current_session or "asian"  # Default to Asian if no active session
        )
        indicators["session_suitable"] = is_suitable
        indicators["session_score"] = session_score
        
        # Calculate ATR for volatility assessment
        indicators["atr"] = self._calculate_atr(data, params["atr_period"])
        
        # Convert ATR to pips for easier interpretation
        point_value = 0.0001 if "JPY" not in pair else 0.01
        indicators["atr_pips"] = indicators["atr"] / point_value
        
        # Determine if volatility is within acceptable range for the session
        min_volatility = params["min_volatility_atr"]
        max_volatility = params["max_volatility_atr"]
        indicators["volatility_suitable"] = min_volatility <= indicators["atr"] <= max_volatility
        
        # Calculate short-term momentum indicators (useful for scalping)
        indicators["rsi"] = self._calculate_rsi(data["close"], 9)  # Shorter period for faster signals
        indicators["macd"], indicators["macd_signal"], indicators["macd_hist"] = \
            self._calculate_macd(data["close"], 12, 26, 9)
            
        # Calculate Bollinger Bands (useful for scalping range breakouts/mean reversions)
        indicators["bb_middle"], indicators["bb_upper"], indicators["bb_lower"] = \
            self._calculate_bollinger_bands(data["close"], 20, 2.0)
        
        # Calculate additional short-term indicators specifically useful for scalping
        indicators["ema_fast"] = data["close"].ewm(span=8, adjust=False).mean()
        indicators["ema_slow"] = data["close"].ewm(span=21, adjust=False).mean()
        indicators["stochastic_k"], indicators["stochastic_d"] = self._calculate_stochastic(
            data["high"], data["low"], data["close"], 14, 3
        )
        
        # Market microstructure indicators (specific to the current timeframe)
        # These help identify high-probability scalping setups
        indicators["spread_estimate"] = self._estimate_spread(data)
        indicators["spread_suitable"] = indicators["spread_estimate"] <= params["max_spread_pips"]
        
        # Detect price action patterns relevant for scalping
        indicators["potential_fakeout"] = self._detect_fakeout_pattern(data)
        indicators["potential_reversal"] = self._detect_reversal_pattern(data)
        indicators["potential_breakout"] = self._detect_breakout_pattern(data, indicators)
        
        # Calculate session-specific indicators
        if self.current_session == "asian":
            # Asian session specifics - range detection, lower volatility moves
            indicators["range_high"], indicators["range_low"] = self._calculate_session_range(data)
            indicators["range_middle"] = (indicators["range_high"] + indicators["range_low"]) / 2
            indicators["range_width"] = (indicators["range_high"] - indicators["range_low"]) / point_value  # in pips
            
        elif self.current_session in ["london", "new_york", "london_ny_overlap"]:
            # European/US session specifics - momentum, trend moves
            indicators["momentum"] = self._calculate_momentum(data["close"], 5)
            indicators["adx"] = self._calculate_adx(data, 14)  # Trend strength
        
        return indicators
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility assessment.
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period:
            return 0.0
            
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
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
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, 
                       signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Number of standard deviations for bands
            
        Returns:
            Tuple of (Middle Band, Upper Band, Lower Band)
        """
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return middle_band, upper_band, lower_band
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: K period
            d_period: D period
            
        Returns:
            Tuple of (K, D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def _estimate_spread(self, data: pd.DataFrame) -> float:
        """
        Estimate current spread based on recent data.
        This is a placeholder - in production, would use actual bid/ask data.
        
        Args:
            data: OHLCV data
            
        Returns:
            Estimated spread in pips
        """
        # In a real implementation, this would use actual bid/ask data
        # For now, use a simplified estimation based on volatility
        pair = self.session.symbol
        point_value = 0.0001 if "JPY" not in pair else 0.01
        
        # Simplistic spread estimate - higher during Asian, lower during London/NY
        if self.current_session == "asian":
            base_spread = 1.5
        elif self.current_session == "london_ny_overlap":
            base_spread = 0.7
        else:
            base_spread = 1.0
            
        # Adjust for volatility - higher volatility often means wider spreads
        if "atr" in data:
            volatility_factor = min(3.0, max(0.8, data["atr"].iloc[-1] * 100))
            base_spread *= volatility_factor
            
        return base_spread  # in pips
    
    def _detect_fakeout_pattern(self, data: pd.DataFrame) -> bool:
        """
        Detect potential fakeout patterns - price briefly breaks a level then reverses.
        These are common in forex and provide good scalping opportunities.
        
        Args:
            data: OHLCV data
            
        Returns:
            Boolean indicating if a potential fakeout pattern is detected
        """
        if len(data) < 5:
            return False
            
        # Get recent price action
        recent_data = data.iloc[-5:]
        highs = recent_data["high"]
        lows = recent_data["low"]
        closes = recent_data["close"]
        
        # Define local swing high/low over last 5 bars
        local_high = highs.iloc[:-1].max()
        local_low = lows.iloc[:-1].min()
        current_close = closes.iloc[-1]
        previous_close = closes.iloc[-2]
        
        # Check for high fakeout (price briefly broke above local high then closed back below)
        high_fakeout = (highs.iloc[-1] > local_high) and (current_close < local_high) \
                      and (current_close < previous_close)
                      
        # Check for low fakeout (price briefly broke below local low then closed back above)
        low_fakeout = (lows.iloc[-1] < local_low) and (current_close > local_low) \
                    and (current_close > previous_close)
        
        return high_fakeout or low_fakeout
    
    def _detect_reversal_pattern(self, data: pd.DataFrame) -> bool:
        """
        Detect potential reversal patterns suitable for scalping.
        
        Args:
            data: OHLCV data
            
        Returns:
            Boolean indicating if a potential reversal pattern is detected
        """
        if len(data) < 5:
            return False
            
        # Get recent price action
        recent_data = data.iloc[-4:]
        opens = recent_data["open"]
        closes = recent_data["close"]
        highs = recent_data["high"]
        lows = recent_data["low"]
        
        # Check for bullish reversal pattern
        # Look for: downtrend into support, bullish engulfing or hammer
        if closes.iloc[0] < opens.iloc[0] and closes.iloc[1] < opens.iloc[1] and \
           closes.iloc[-1] > opens.iloc[-1]:
            # Check for bullish engulfing
            bullish_engulfing = (opens.iloc[-1] <= closes.iloc[-2]) and \
                              (closes.iloc[-1] > opens.iloc[-2])
                              
            # Check for hammer (long lower wick, small body, little/no upper wick)
            lower_wick = opens.iloc[-1] - lows.iloc[-1] if opens.iloc[-1] < closes.iloc[-1] \
                        else closes.iloc[-1] - lows.iloc[-1]
            upper_wick = highs.iloc[-1] - closes.iloc[-1] if opens.iloc[-1] < closes.iloc[-1] \
                        else highs.iloc[-1] - opens.iloc[-1]
            body_size = abs(closes.iloc[-1] - opens.iloc[-1])
            
            hammer = (lower_wick > body_size * 2) and (upper_wick < body_size * 0.5)
            
            return bullish_engulfing or hammer
            
        # Check for bearish reversal pattern
        # Look for: uptrend into resistance, bearish engulfing or shooting star
        elif closes.iloc[0] > opens.iloc[0] and closes.iloc[1] > opens.iloc[1] and \
              closes.iloc[-1] < opens.iloc[-1]:
            # Check for bearish engulfing
            bearish_engulfing = (opens.iloc[-1] >= closes.iloc[-2]) and \
                              (closes.iloc[-1] < opens.iloc[-2])
                              
            # Check for shooting star (long upper wick, small body, little/no lower wick)
            upper_wick = highs.iloc[-1] - opens.iloc[-1] if opens.iloc[-1] > closes.iloc[-1] \
                        else highs.iloc[-1] - closes.iloc[-1]
            lower_wick = closes.iloc[-1] - lows.iloc[-1] if opens.iloc[-1] > closes.iloc[-1] \
                        else opens.iloc[-1] - lows.iloc[-1]
            body_size = abs(closes.iloc[-1] - opens.iloc[-1])
            
            shooting_star = (upper_wick > body_size * 2) and (lower_wick < body_size * 0.5)
            
            return bearish_engulfing or shooting_star
            
        return False
    
    def _detect_breakout_pattern(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Detect potential breakout patterns suitable for scalping.
        Adjusts based on session characteristics.
        
        Args:
            data: OHLCV data
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating if a potential breakout pattern is detected
        """
        if len(data) < 20 or not indicators:
            return False
            
        # Get key price levels
        current_close = data["close"].iloc[-1]
        current_volume = data["volume"].iloc[-1] if "volume" in data else None
        avg_volume = data["volume"].rolling(window=20).mean().iloc[-1] if "volume" in data else None
        
        # Check if price breaks Bollinger Bands with momentum
        if "bb_upper" in indicators and "bb_lower" in indicators:
            bb_upper = indicators["bb_upper"].iloc[-1] if isinstance(indicators["bb_upper"], pd.Series) else indicators["bb_upper"]
            bb_lower = indicators["bb_lower"].iloc[-1] if isinstance(indicators["bb_lower"], pd.Series) else indicators["bb_lower"]
            
            # Adjust breakout detection based on session
            if self.current_session == "asian":
                # Asian session has more false breakouts, so be more conservative
                upper_breakout = current_close > bb_upper * 1.001  # Require a bit more confirmation
                lower_breakout = current_close < bb_lower * 0.999
            else:
                # London/NY sessions have stronger momentum, so standard breakouts work better
                upper_breakout = current_close > bb_upper
                lower_breakout = current_close < bb_lower
            
            # Volume confirmation if available
            if current_volume is not None and avg_volume is not None:
                # Higher volume standards during London/NY for trend confirmation
                if self.current_session in ["london", "new_york", "london_ny_overlap"]:
                    volume_confirm = current_volume > avg_volume * 1.2  # 20% above average
                else:
                    volume_confirm = current_volume > avg_volume  # Just above average for Asian
                
                return (upper_breakout or lower_breakout) and volume_confirm
            else:
                return upper_breakout or lower_breakout
        
        return False
    
    def _calculate_session_range(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate the high/low range for the current session.
        Useful for Asian session range trading strategies.
        
        Args:
            data: OHLCV data
            
        Returns:
            Tuple of (range_high, range_low)
        """
        # For simplicity, use last 4 hours of data as the "session range"
        # In production, would detect actual session boundaries
        recent_data = data.iloc[-16:] if len(data) >= 16 else data
        
        range_high = recent_data["high"].max()
        range_low = recent_data["low"].min()
        
        return range_high, range_low
    
    def _calculate_momentum(self, prices: pd.Series, period: int = 5) -> float:
        """
        Calculate short-term price momentum (latest close vs N periods ago).
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Momentum value
        """
        if len(prices) <= period:
            return 0.0
            
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-period-1]
        
        return (current_price / past_price - 1) * 100  # Percentage change
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index for trend strength.
        
        Args:
            data: OHLCV data
            period: ADX period
            
        Returns:
            ADX values
        """
        # This is a simplified ADX calculation
        if len(data) < period + 1:
            return pd.Series([0] * len(data))
            
        # Calculate directional movement
        plus_dm = data["high"].diff()
        minus_dm = data["low"].diff().multiply(-1)
        
        # Clean up positive and negative values
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate true range
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate directional indicators
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # Calculate directional index
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 0.001)
        adx = dx.rolling(window=period).mean()
        
        return adx
        
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for forex pairs, adapting to current trading session.
        
        Args:
            data_dict: Dictionary of market data for different pairs
            
        Returns:
            Dictionary of trading signals
        """
        generated_signals = {}
        current_time = datetime.now(self.parameters["timezone"])
        self._update_current_session(current_time)
        
        # Early return if no active session
        if not self.current_session:
            self.logger.info("No active trading session - not generating signals")
            return {}
        
        # Only process symbols that match our criteria
        for symbol, data in data_dict.items():
            if data.empty or len(data) < 20:
                continue
                
            # Set current symbol for context in helper methods
            self.session.symbol = symbol
            
            # Get session-specific parameters
            params = self._get_current_parameters()
            
            # Calculate indicators for this pair
            indicators = self.calculate_indicators(data)
            
            # Skip if indicators couldn't be calculated
            if not indicators:
                continue
                
            # Skip if pair not suitable for current session or volatility/spread not suitable
            if not indicators.get("session_suitable", False) or \
               not indicators.get("volatility_suitable", False) or \
               not indicators.get("spread_suitable", False):
                continue
                
            # Check entry conditions based on current session
            entry_signal, signal_strength = self._check_entry_conditions(
                data, indicators, params
            )
            
            # If valid entry signal detected
            if entry_signal:
                # Get current market data
                current_price = data["close"].iloc[-1]
                
                # Calculate risk parameters (adaptive to session and volatility)
                stop_loss_pips, take_profit_pips = self._calculate_risk_parameters(
                    data, indicators, entry_signal, params
                )
                
                # Convert pips to price
                point_value = 0.0001 if "JPY" not in symbol else 0.01
                stop_loss_price = current_price - (stop_loss_pips * point_value) if entry_signal == "buy" \
                                 else current_price + (stop_loss_pips * point_value)
                take_profit_price = current_price + (take_profit_pips * point_value) if entry_signal == "buy" \
                                  else current_price - (take_profit_pips * point_value)
                
                # Calculate position size
                account_balance = self.session.account_balance
                risk_amount = account_balance * params["risk_per_trade"]
                position_size = self._calculate_position_size(
                    symbol, current_price, stop_loss_price, risk_amount
                )
                
                # Construct signal
                signal = Signal(
                    symbol=symbol,
                    signal_type="buy" if entry_signal == "buy" else "sell",
                    entry_price=current_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    size=position_size,
                    timestamp=current_time,
                    timeframe=data.index.freq or "1H",  # Use actual timeframe or default
                    strategy=self.name,
                    session=self.current_session,
                    strength=signal_strength,
                    metadata={
                        "session": self.current_session,
                        "session_score": indicators.get("session_score", 0),
                        "spread_pips": indicators.get("spread_estimate", 0),
                        "atr_pips": indicators.get("atr_pips", 0),
                        "setup_type": self._determine_setup_type(indicators),
                        "expiry_bars": params["max_holding_bars"]  # Signal expires after this many bars
                    }
                )
                
                # Add to signals dictionary
                generated_signals[symbol] = signal
                self.logger.info(f"Generated {entry_signal} signal for {symbol} in {self.current_session} session")
        
        return generated_signals
    
    def _check_entry_conditions(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                              params: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Check if entry conditions are met, with logic adapted to current session.
        
        Args:
            data: Price data
            indicators: Pre-calculated indicators
            params: Session-specific parameters
            
        Returns:
            Tuple of (signal_type, signal_strength) where signal_type is 'buy', 'sell', or None
        """
        # Default - no signal
        signal = None
        strength = 0.0
        
        # Get current session and important indicators
        session = self.current_session
        rsi = indicators.get("rsi", pd.Series()).iloc[-1] if isinstance(indicators.get("rsi"), pd.Series) else indicators.get("rsi", 50)
        macd = indicators.get("macd", pd.Series()).iloc[-1] if isinstance(indicators.get("macd"), pd.Series) else indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", pd.Series()).iloc[-1] if isinstance(indicators.get("macd_signal"), pd.Series) else indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_hist", pd.Series()).iloc[-1] if isinstance(indicators.get("macd_hist"), pd.Series) else indicators.get("macd_hist", 0)
        
        # Variables to track signal validity across different conditions
        buy_conditions_met = 0
        sell_conditions_met = 0
        total_conditions = 0
        
        # Asian session - focus on range trading and key level bounces
        if session == "asian":
            total_conditions = 3
            
            # Get range data
            current_price = data["close"].iloc[-1]
            range_high = indicators.get("range_high", current_price * 1.01)
            range_low = indicators.get("range_low", current_price * 0.99)
            range_middle = indicators.get("range_middle", current_price)
            bb_upper = indicators.get("bb_upper", pd.Series()).iloc[-1] if isinstance(indicators.get("bb_upper"), pd.Series) else indicators.get("bb_upper", range_high)
            bb_lower = indicators.get("bb_lower", pd.Series()).iloc[-1] if isinstance(indicators.get("bb_lower"), pd.Series) else indicators.get("bb_lower", range_low)
            
            # Buy near range lows with oversold RSI
            if current_price < range_low * 1.005 and rsi < params["rsi_oversold"]:
                buy_conditions_met += 1
            # Buy on bullish reversal patterns near support
            if indicators.get("potential_reversal", False) and current_price < range_middle:
                buy_conditions_met += 1
            # Buy on BB squeeze with bullish momentum
            if current_price > bb_lower and macd_hist > 0 and macd > macd_signal:
                buy_conditions_met += 1
                
            # Sell near range highs with overbought RSI
            if current_price > range_high * 0.995 and rsi > params["rsi_overbought"]:
                sell_conditions_met += 1
            # Sell on bearish reversal patterns near resistance
            if indicators.get("potential_reversal", False) and current_price > range_middle:
                sell_conditions_met += 1
            # Sell on BB squeeze with bearish momentum
            if current_price < bb_upper and macd_hist < 0 and macd < macd_signal:
                sell_conditions_met += 1
                
        # London/NY/Overlap sessions - focus on momentum and breakouts
        else:
            total_conditions = 4
            
            # Get momentum/trend indicators
            ema_fast = indicators.get("ema_fast", pd.Series()).iloc[-1] if isinstance(indicators.get("ema_fast"), pd.Series) else None
            ema_slow = indicators.get("ema_slow", pd.Series()).iloc[-1] if isinstance(indicators.get("ema_slow"), pd.Series) else None
            adx = indicators.get("adx", pd.Series()).iloc[-1] if isinstance(indicators.get("adx"), pd.Series) else indicators.get("adx", 0)
            momentum = indicators.get("momentum", 0)
            
            # Buy conditions for momentum-based trading
            if ema_fast is not None and ema_slow is not None and ema_fast > ema_slow:
                buy_conditions_met += 1
            if macd > 0 and macd > macd_signal and macd_hist > 0:
                buy_conditions_met += 1
            if rsi > 50 and rsi < 70:  # Some upside momentum but not overbought
                buy_conditions_met += 1
            if adx > params["min_adx"] and momentum > 0:  # Strong trend with positive momentum
                buy_conditions_met += 1
                
            # Sell conditions for momentum-based trading
            if ema_fast is not None and ema_slow is not None and ema_fast < ema_slow:
                sell_conditions_met += 1
            if macd < 0 and macd < macd_signal and macd_hist < 0:
                sell_conditions_met += 1
            if rsi < 50 and rsi > 30:  # Some downside momentum but not oversold
                sell_conditions_met += 1
            if adx > params["min_adx"] and momentum < 0:  # Strong trend with negative momentum
                sell_conditions_met += 1
        
        # Determine signal based on number of conditions met
        required_conditions = params["min_entry_conditions"]
        
        if buy_conditions_met >= required_conditions and buy_conditions_met > sell_conditions_met:
            signal = "buy"
            strength = buy_conditions_met / total_conditions
        elif sell_conditions_met >= required_conditions and sell_conditions_met > buy_conditions_met:
            signal = "sell"
            strength = sell_conditions_met / total_conditions
        
        return signal, strength
    
    def _calculate_risk_parameters(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                 signal_type: str, params: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate adaptive stop loss and take profit levels based on session, volatility, and signal type.
        
        Args:
            data: Price data
            indicators: Technical indicators
            signal_type: 'buy' or 'sell'
            params: Session-specific parameters
            
        Returns:
            Tuple of (stop_loss_pips, take_profit_pips)
        """
        # Get ATR for dynamic risk sizing
        atr_pips = indicators.get("atr_pips", 10)  # Default to 10 pips if not available
        spread_pips = indicators.get("spread_estimate", 1)  # Default to 1 pip if not available
        
        # Base stop loss on ATR and session
        if self.current_session == "asian":
            # Tighter stops for Asian range trading
            stop_loss_multiplier = params["asian_stop_loss_atr_multiplier"]
        else:
            # Wider stops for London/NY trend trading
            stop_loss_multiplier = params["main_session_stop_loss_atr_multiplier"]
        
        # Calculate basic stop loss
        stop_loss_pips = max(atr_pips * stop_loss_multiplier, params["min_stop_loss_pips"])
        
        # Calculate take profit with consideration for the spread
        risk_reward_ratio = params["risk_reward_ratio"]
        take_profit_pips = stop_loss_pips * risk_reward_ratio
        
        # Ensure minimum take profit covers the spread plus minimum profit
        min_take_profit = spread_pips * params["min_profit_multiple"]
        take_profit_pips = max(take_profit_pips, min_take_profit)
        
        # Round to 0.1 pip precision
        stop_loss_pips = round(stop_loss_pips * 10) / 10
        take_profit_pips = round(take_profit_pips * 10) / 10
        
        return stop_loss_pips, take_profit_pips
    
    def _calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float, risk_amount: float) -> float:
        """
        Calculate position size based on risk amount and stop loss distance.
        
        Args:
            symbol: Currency pair
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_amount: Amount to risk in account currency
            
        Returns:
            Position size in lots
        """
        # Calculate pip value
        base_currency = symbol[:3]
        quote_currency = symbol[3:6] if len(symbol) >= 6 else "USD"
        account_currency = "USD"  # Default, would be configurable in production
        
        # Price difference in absolute terms
        price_diff = abs(entry_price - stop_loss_price)
        
        # Calculate pip value for standard lot (100,000 units)        
        standard_lot = 100000
        pip_size = 0.0001 if "JPY" not in symbol else 0.01
        pip_value_in_quote = standard_lot * pip_size
        
        # Convert to account currency if necessary (simplified)
        pip_value_in_account = pip_value_in_quote
        if quote_currency != account_currency:
            # In production, would lookup conversion rate
            conversion_rate = 1.0  # Placeholder
            pip_value_in_account = pip_value_in_quote * conversion_rate
        
        # Calculate number of pips risked
        pips_risked = price_diff / pip_size
        
        # Calculate position size in standard lots
        if pips_risked > 0 and pip_value_in_account > 0:
            position_size_in_standard_lots = risk_amount / (pips_risked * pip_value_in_account)
        else:
            position_size_in_standard_lots = 0
        
        # Apply position size limits
        min_lot = self.parameters["min_position_size"]
        max_lot = self.parameters["max_position_size"]
        position_size_in_standard_lots = max(min(position_size_in_standard_lots, max_lot), min_lot)
        
        return position_size_in_standard_lots
    
    def _determine_setup_type(self, indicators: Dict[str, Any]) -> str:
        """
        Determine the specific setup type for the generated signal.
        Useful for post-trade analysis and strategy refinement.
        
        Args:
            indicators: Calculated indicators
            
        Returns:
            Setup type label
        """
        # Determine setup type based on indicators and patterns
        if indicators.get("potential_breakout", False):
            return "breakout"
        elif indicators.get("potential_fakeout", False):
            return "fakeout"
        elif indicators.get("potential_reversal", False):
            return "reversal"
        elif self.current_session == "asian":
            return "range"
        else:
            return "momentum"
    
    def check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Check if the position should be exited based on current market conditions.
        This supplements fixed stop loss/take profit orders with dynamic exit conditions.
        
        Args:
            position: Current position information
            data: Latest market data
            
        Returns:
            True if position should be exited, False otherwise
        """
        if data.empty or len(data) < 5:
            return False
            
        # Extract position details
        symbol = position["symbol"]
        entry_time = position["entry_time"]
        position_type = position["type"]  # 'buy' or 'sell'
        self.session.symbol = symbol
        
        # Calculate current profit/loss (in pips)
        current_price = data["close"].iloc[-1]
        entry_price = position["entry_price"]
        point_value = 0.0001 if "JPY" not in symbol else 0.01
        pips_diff = (current_price - entry_price) / point_value if position_type == "buy" \
                   else (entry_price - current_price) / point_value
        
        # Update current session
        current_time = data.index[-1]
        self._update_current_session(current_time)
        params = self._get_current_parameters()
        
        # Check time-based exit (if position has been open too long)
        # Calculate bars passed since entry
        entry_idx = None
        for i, time in enumerate(data.index):
            if time >= entry_time:
                entry_idx = i
                break
                
        if entry_idx is not None:
            bars_held = len(data) - entry_idx
            if bars_held >= params["max_holding_bars"]:
                self.logger.info(f"Exiting {symbol} due to max holding time reached ({bars_held} bars)")
                return True
        
        # Calculate indicators for exit decision
        indicators = self.calculate_indicators(data)
        
        # Exit on session change if configured to do so
        original_session = position.get("metadata", {}).get("session")
        if original_session and original_session != self.current_session and params["exit_on_session_change"]:
            self.logger.info(f"Exiting {symbol} due to session change from {original_session} to {self.current_session}")
            return True
        
        # Exit on trend reversal for trend-following trades
        if position.get("metadata", {}).get("setup_type") in ["momentum", "breakout"]:
            # Check for trend reversal
            if position_type == "buy" and indicators.get("macd_hist", 0) < 0 and \
               indicators.get("rsi", 50) < 45:
                self.logger.info(f"Exiting {symbol} buy position due to bearish reversal")
                return True
            elif position_type == "sell" and indicators.get("macd_hist", 0) > 0 and \
                 indicators.get("rsi", 50) > 55:
                self.logger.info(f"Exiting {symbol} sell position due to bullish reversal")
                return True
        
        # Exit on range expansion/contraction for range trades
        if position.get("metadata", {}).get("setup_type") == "range" and self.current_session != "asian":
            # Exit range trades when transitioning to directional sessions
            self.logger.info(f"Exiting {symbol} range trade as session changed to directional {self.current_session}")
            return True
        
        # Exit if market volatility increases dramatically (risk management)
        original_atr = position.get("metadata", {}).get("atr_pips", 10)
        current_atr = indicators.get("atr_pips", original_atr)
        if current_atr > original_atr * params["volatility_exit_multiplier"]:
            self.logger.info(f"Exiting {symbol} due to volatility spike: original ATR={original_atr}, current ATR={current_atr}")
            return True
        
        # If in profit, consider trailing stops
        if pips_diff > original_atr and params["use_trailing_stop"]:
            trailing_stop_activated = position.get("metadata", {}).get("trailing_stop_activated", False)
            
            # If trailing stop already activated, check if price pulled back too far
            if trailing_stop_activated:
                max_profit_pips = position.get("metadata", {}).get("max_profit_pips", pips_diff)
                trail_buffer = params["trailing_stop_buffer"]
                
                if pips_diff < max_profit_pips - trail_buffer:
                    self.logger.info(f"Exiting {symbol} due to trailing stop: max profit={max_profit_pips}, current={pips_diff}, buffer={trail_buffer}")
                    return True
            else:
                # Activate trailing stop and update position metadata
                # In a real implementation, this would update the position object
                self.logger.info(f"Activating trailing stop for {symbol} at {pips_diff} pips profit")
                # position["metadata"]["trailing_stop_activated"] = True
                # position["metadata"]["max_profit_pips"] = pips_diff
        
        return False
