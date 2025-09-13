#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Day Trading Strategy

This strategy focuses on short-term price movements within a single day.
It aims to enter and exit positions during the same day, capturing smaller
price movements with higher frequency trading.

Key characteristics:
- Short holding periods (minutes to hours)
- Multiple trades per day
- Uses intraday momentum and reversal signals
- Employs strict risk management
- Closes all positions by end of trading session
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoDayTradingStrategy",
    market_type="crypto",
    description="Intraday trading strategy for crypto markets focusing on momentum and reversal setups",
    timeframes=["M5", "M15", "H1"],
    parameters={
        # Technical indicators
        "vwap_enabled": {"type": "bool", "default": True},
        "ema_short": {"type": "int", "default": 9, "min": 5, "max": 20},
        "ema_medium": {"type": "int", "default": 21, "min": 15, "max": 30},
        "rsi_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "rsi_overbought": {"type": "float", "default": 70.0, "min": 65.0, "max": 80.0},
        "rsi_oversold": {"type": "float", "default": 30.0, "min": 20.0, "max": 35.0},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        
        # Session parameters
        "session_start_hour": {"type": "int", "default": 0, "min": 0, "max": 23},
        "session_end_hour": {"type": "int", "default": 23, "min": 0, "max": 23},
        "force_session_close": {"type": "bool", "default": True},
        
        # Trade execution
        "profit_target_atr_multi": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        "stop_loss_atr_multi": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
        "max_trades_per_day": {"type": "int", "default": 5, "min": 1, "max": 10},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.005, "min": 0.001, "max": 0.02},
        "max_daily_risk": {"type": "float", "default": 0.02, "min": 0.005, "max": 0.05},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 5}
    }
)
class CryptoDayTradingStrategy(CryptoBaseStrategy):
    """
    An intraday trading strategy for cryptocurrency markets.
    
    This strategy:
    1. Uses intraday momentum and reversal patterns
    2. Focuses on high-probability setups with defined support/resistance
    3. Manages trades within a single day's session
    4. Enforces strict daily risk limits
    5. Utilizes volume and price action for entries/exits
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto day trading strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Day trading state
        self.daily_trades = 0
        self.daily_risk_used = 0.0
        self.current_session_date = None
        self.in_active_session = False
        self.pending_trade_setups = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.best_performing_setups = {}
        
        logger.info(f"Initialized crypto day trading strategy for {self.session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for day trading strategy."""
        if data.empty or len(data) < 30:
            return {}
        
        indicators = {}
        
        # Moving averages
        indicators["ema_short"] = data["close"].ewm(span=self.parameters["ema_short"], adjust=False).mean()
        indicators["ema_medium"] = data["close"].ewm(span=self.parameters["ema_medium"], adjust=False).mean()
        
        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.parameters["rsi_period"]).mean()
        avg_loss = loss.rolling(window=self.parameters["rsi_period"]).mean()
        rs = avg_gain / avg_loss
        indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR for volatility
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = true_range.rolling(window=self.parameters["atr_period"]).mean()
        
        # VWAP (Volume Weighted Average Price)
        if self.parameters["vwap_enabled"]:
            # Get the current day's data only
            if not data.empty and isinstance(data.index[0], pd.Timestamp):
                today = data.index[-1].date()
                day_data = data[data.index.date == today]
                
                if not day_data.empty:
                    # Calculate VWAP
                    v = day_data["volume"]
                    tp = (day_data["high"] + day_data["low"] + day_data["close"]) / 3
                    vwap = (tp * v).cumsum() / v.cumsum()
                    
                    # Extend VWAP to match original data length
                    full_vwap = pd.Series(index=data.index)
                    full_vwap.loc[day_data.index] = vwap
                    indicators["vwap"] = full_vwap
        
        # Support and resistance levels
        pivots = self._find_pivot_points(data)
        indicators["support_levels"] = pivots["supports"]
        indicators["resistance_levels"] = pivots["resistances"]
        
        # Near-term momentum
        indicators["momentum"] = data["close"].diff(3)
        
        return indicators
    
    def _find_pivot_points(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate pivot points based on recent price action."""
        result = {"supports": [], "resistances": []}
        
        # Simple approach: use recent local maxima and minima
        if len(data) < 20:
            return result
            
        # Recent price range
        recent_high = data["high"].iloc[-20:].max()
        recent_low = data["low"].iloc[-20:].min()
        current_price = data["close"].iloc[-1]
        
        # Basic pivot formula
        pivot = (recent_high + recent_low + current_price) / 3
        
        # Support levels
        s1 = (2 * pivot) - recent_high
        s2 = pivot - (recent_high - recent_low)
        result["supports"] = [s1, s2]
        
        # Resistance levels
        r1 = (2 * pivot) - recent_low
        r2 = pivot + (recent_high - recent_low)
        result["resistances"] = [r1, r2]
        
        return result
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on calculated indicators."""
        signals = {
            "long_entry": False,
            "short_entry": False,
            "long_exit": False,
            "short_exit": False,
            "signal_strength": 0.0,
            "setup_type": None
        }
        
        if not indicators or data.empty or len(data) < 20:
            return signals
        
        # Check if we're in an active session
        current_time = data.index[-1]
        session_start_hour = self.parameters["session_start_hour"]
        session_end_hour = self.parameters["session_end_hour"]
        
        # Update session status
        self._update_session_status(current_time)
        
        # Only generate new signals during active session
        if not self.in_active_session:
            # Force close positions at session end if configured
            if self.parameters["force_session_close"] and self.positions:
                signals["long_exit"] = True
                signals["short_exit"] = True
                signals["setup_type"] = "session_close"
                logger.info(f"Forcing position close at session end for {self.session.symbol}")
            return signals
        
        # Check if we've reached daily trade limit
        if self.daily_trades >= self.parameters["max_trades_per_day"]:
            logger.info(f"Daily trade limit reached for {self.session.symbol}, no new signals")
            return signals
        
        # Check if we've reached daily risk limit
        if self.daily_risk_used >= self.parameters["max_daily_risk"]:
            logger.info(f"Daily risk limit reached for {self.session.symbol}, no new signals")
            return signals
        
        # Current indicator values
        current_close = data["close"].iloc[-1]
        current_rsi = indicators["rsi"].iloc[-1]
        prev_rsi = indicators["rsi"].iloc[-2] if len(indicators["rsi"]) > 1 else 50
        ema_short = indicators["ema_short"].iloc[-1]
        ema_medium = indicators["ema_medium"].iloc[-1]
        
        # Price relative to VWAP
        above_vwap = False
        if "vwap" in indicators and not pd.isna(indicators["vwap"].iloc[-1]):
            above_vwap = current_close > indicators["vwap"].iloc[-1]
        
        # Support and resistance proximity
        near_support = False
        near_resistance = False
        
        if "support_levels" in indicators and indicators["support_levels"]:
            nearest_support = min(indicators["support_levels"], key=lambda x: abs(x - current_close))
            near_support = (current_close - nearest_support) / current_close < 0.01  # Within 1%
            
        if "resistance_levels" in indicators and indicators["resistance_levels"]:
            nearest_resistance = min(indicators["resistance_levels"], key=lambda x: abs(x - current_close))
            near_resistance = (nearest_resistance - current_close) / current_close < 0.01  # Within 1%
        
        # Momentum and trend
        bullish_trend = ema_short > ema_medium
        bearish_trend = ema_short < ema_medium
        bullish_momentum = indicators["momentum"].iloc[-1] > 0 if "momentum" in indicators else False
        bearish_momentum = indicators["momentum"].iloc[-1] < 0 if "momentum" in indicators else False
        
        # Long entry setups
        momentum_long = bullish_trend and bullish_momentum and above_vwap
        reversal_long = near_support and current_rsi < self.parameters["rsi_oversold"] and prev_rsi < current_rsi
        
        # Short entry setups
        momentum_short = bearish_trend and bearish_momentum and not above_vwap
        reversal_short = near_resistance and current_rsi > self.parameters["rsi_overbought"] and prev_rsi > current_rsi
        
        # Generate signals based on setups
        if momentum_long or reversal_long:
            signals["long_entry"] = True
            signals["signal_strength"] = 0.7 if momentum_long else 0.6
            signals["setup_type"] = "momentum_long" if momentum_long else "reversal_long"
            logger.info(f"Day trading LONG signal for {self.session.symbol} ({signals['setup_type']})")
            
        elif momentum_short or reversal_short:
            signals["short_entry"] = True
            signals["signal_strength"] = 0.7 if momentum_short else 0.6
            signals["setup_type"] = "momentum_short" if momentum_short else "reversal_short"
            logger.info(f"Day trading SHORT signal for {self.session.symbol} ({signals['setup_type']})")
            
        # Exit signals for existing positions
        for position in self.positions:
            if position.direction == "long":
                # Exit long on bearish signals or near resistance
                if bearish_momentum or near_resistance or (current_rsi > self.parameters["rsi_overbought"]):
                    signals["long_exit"] = True
                    
            elif position.direction == "short":
                # Exit short on bullish signals or near support
                if bullish_momentum or near_support or (current_rsi < self.parameters["rsi_oversold"]):
                    signals["short_exit"] = True
        
        return signals
    
    def _update_session_status(self, current_time: datetime) -> None:
        """Update the session status based on current time."""
        if not isinstance(current_time, datetime):
            logger.warning(f"Invalid time format: {current_time}")
            return
            
        # Check if we're in a new day
        current_date = current_time.date()
        if self.current_session_date != current_date:
            # Reset daily counters
            self.current_session_date = current_date
            self.daily_trades = 0
            self.daily_risk_used = 0.0
            self.daily_pnl = 0.0
            logger.info(f"New trading day started for {self.session.symbol}")
        
        # Check if we're in active trading hours
        current_hour = current_time.hour
        session_start_hour = self.parameters["session_start_hour"]
        session_end_hour = self.parameters["session_end_hour"]
        
        # Handle different session configurations
        if session_start_hour <= session_end_hour:
            # Normal session (e.g., 9 to 17)
            self.in_active_session = session_start_hour <= current_hour < session_end_hour
        else:
            # Overnight session (e.g., 22 to 4)
            self.in_active_session = current_hour >= session_start_hour or current_hour < session_end_hour
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """Calculate position size based on risk parameters and ATR."""
        if data.empty or not indicators or "atr" not in indicators:
            return 0.0
        
        # Account balance
        account_balance = 10000.0  # Mock value, would come from exchange API
        
        # Risk per trade
        risk_per_trade = self.parameters["risk_per_trade"]
        
        # Remaining risk budget for the day
        remaining_risk = self.parameters["max_daily_risk"] - self.daily_risk_used
        
        # Use the smaller of the two risk limits
        actual_risk = min(risk_per_trade, remaining_risk)
        
        # If we're out of risk budget, don't trade
        if actual_risk <= 0:
            return 0.0
            
        risk_amount = account_balance * actual_risk
        
        # Calculate stop distance using ATR
        atr = indicators["atr"].iloc[-1]
        stop_loss_multiple = self.parameters["stop_loss_atr_multi"]
        stop_distance = atr * stop_loss_multiple
        
        # Calculate position size
        current_price = data["close"].iloc[-1]
        if stop_distance > 0 and current_price > 0:
            position_size_base = risk_amount / (stop_distance / current_price)
            position_size_crypto = position_size_base / current_price
            
            # Apply signal strength adjustment if available
            signal_strength = indicators.get("signal_strength", 0.6)
            position_size_crypto *= signal_strength
            
            # Ensure minimum trade size
            min_trade_size = self.session.min_trade_size
            position_size_crypto = max(position_size_crypto, min_trade_size)
            
            # Round to appropriate precision
            decimals = 8 if self.session.symbol.startswith("BTC") else 6
            position_size_crypto = round(position_size_crypto, decimals)
            
            # Update daily risk tracking
            self.daily_risk_used += actual_risk
            
            return position_size_crypto
        
        return 0.0
    
    def _on_position_opened(self, event: Event) -> None:
        """Handle position opened events for day trading."""
        super()._on_position_opened(event)
        
        # Track daily trades
        if event.data.get('position_id') in [p.id for p in self.positions]:
            self.daily_trades += 1
            logger.info(f"Day trade opened - daily count: {self.daily_trades}/{self.parameters['max_trades_per_day']}")
    
    def _on_position_closed(self, event: Event) -> None:
        """Handle position closed events for day trading."""
        super()._on_position_closed(event)
        
        # Update daily P&L
        position_pnl = event.data.get('pnl', 0.0)
        self.daily_pnl += position_pnl
        
        # Track setup performance
        setup_type = event.data.get('metadata', {}).get('setup_type')
        if setup_type:
            if setup_type not in self.best_performing_setups:
                self.best_performing_setups[setup_type] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0
                }
            
            self.best_performing_setups[setup_type]['count'] += 1
            self.best_performing_setups[setup_type]['total_pnl'] += position_pnl
            
            if position_pnl > 0:
                self.best_performing_setups[setup_type]['wins'] += 1
            else:
                self.best_performing_setups[setup_type]['losses'] += 1
    
    def regime_compatibility(self, market_regime: str) -> float:
        """Calculate compatibility with the current market regime."""
        compatibility_map = {
            "trending": 0.75,        # Good in trending markets
            "ranging": 0.80,         # Very good in ranging markets
            "volatile": 0.65,        # Moderate in volatile markets
            "calm": 0.60,            # Lower in calm markets (fewer opportunities)
            "breakout": 0.75,        # Good during breakouts
            "high_volume": 0.90,     # Excellent during high volume
            "low_volume": 0.40,      # Poor during low volume
            "high_liquidity": 0.85,  # Very good in high liquidity markets
            "low_liquidity": 0.50,   # Moderate in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.65)  # Default compatibility
