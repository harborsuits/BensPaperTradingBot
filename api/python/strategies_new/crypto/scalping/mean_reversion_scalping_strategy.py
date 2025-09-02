#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Mean Reversion Scalping Strategy

This strategy focuses on rapid mean reversion trades in cryptocurrency markets,
identifying when prices have deviated significantly from their short-term moving
averages or statistical norms, and executing quick scalp trades to capture the
reversion to the mean.

Key characteristics:
- Ultra-short holding periods (minutes to hours)
- Uses statistical deviation metrics (z-scores, Bollinger Bands)
- Focuses on overbought/oversold conditions via multiple indicators
- Quick profit taking at mean levels
- Tight stop losses to manage downside risk
- Designed for choppy, ranging market conditions
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies_new.crypto.scalping.crypto_scalping_strategy import CryptoScalpingStrategy
from trading_bot.strategies_new.crypto.base import CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoMeanReversionScalping",
    market_type="crypto",
    description="Ultra-short term mean reversion strategy for crypto markets focusing on statistical deviations",
    timeframes=["M1", "M5", "M15"],
    parameters={
        # Mean reversion specific parameters
        "z_score_window": {"type": "int", "default": 20, "min": 10, "max": 50},
        "z_score_entry_threshold": {"type": "float", "default": 2.0, "min": 1.5, "max": 3.0},
        "z_score_exit_threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        "bb_entry_pct": {"type": "float", "default": 0.95, "min": 0.8, "max": 0.99},
        "use_rsi_filter": {"type": "bool", "default": True},
        "rsi_upper_threshold": {"type": "float", "default": 70.0, "min": 65.0, "max": 85.0},
        "rsi_lower_threshold": {"type": "float", "default": 30.0, "min": 15.0, "max": 35.0},
        
        # Moving average parameters
        "fast_ma_period": {"type": "int", "default": 5, "min": 3, "max": 10},
        "slow_ma_period": {"type": "int", "default": 20, "min": 15, "max": 50},
        "use_ma_filter": {"type": "bool", "default": True},
        
        # Volume filters
        "volume_spike_threshold": {"type": "float", "default": 1.5, "min": 1.2, "max": 3.0},
        "volume_ma_period": {"type": "int", "default": 20, "min": 10, "max": 30},
        
        # Trade execution parameters
        "profit_target_pct": {"type": "float", "default": 0.005, "min": 0.002, "max": 0.02},
        "stop_loss_atr_multiplier": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
        "max_holding_periods": {"type": "int", "default": 12, "min": 3, "max": 24},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.003, "min": 0.001, "max": 0.008},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 4},
        "cooldown_periods": {"type": "int", "default": 3, "min": 1, "max": 10},
    }
)
class CryptoMeanReversionScalping(CryptoScalpingStrategy):
    """
    A specialized scalping strategy focused on mean reversion in cryptocurrency markets.
    
    This strategy:
    1. Identifies statistically significant price deviations from short-term averages
    2. Uses multiple confirmation indicators (RSI, BB, Volume patterns)
    3. Executes rapid entries when extreme deviations are detected
    4. Takes quick profits when price reverts toward the mean
    5. Uses tight risk controls with pre-defined stop losses and holding periods
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the mean reversion scalping strategy.
        
        Args:
            session: Crypto trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific state
        self.z_score_history = []
        self.reversion_signals = {
            "oversold": False,
            "overbought": False,
            "mean_crossover": False,
            "last_signal_time": None,
            "cooldown_until": None,
        }
        
        # Trade management
        self.active_mean_reversion_trades = {}
        self.trade_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_holding_time": 0,
        }
        
        # Register for real-time updates
        self.register_event_handlers()
        logger.info(f"Initialized CryptoMeanReversionScalping strategy for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate mean reversion specific indicators.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty or len(data) < 50:
            return {}
            
        # First get the base scalping indicators
        indicators = super().calculate_indicators(data)
        
        # Mean reversion specific indicators
        try:
            # Z-score calculation (price deviation from moving average)
            window = self.parameters["z_score_window"]
            rolling_mean = data["close"].rolling(window=window).mean()
            rolling_std = data["close"].rolling(window=window).std()
            z_score = (data["close"] - rolling_mean) / rolling_std
            indicators["z_score"] = z_score
            
            # Moving averages
            fast_ma = data["close"].rolling(window=self.parameters["fast_ma_period"]).mean()
            slow_ma = data["close"].rolling(window=self.parameters["slow_ma_period"]).mean()
            indicators["fast_ma"] = fast_ma
            indicators["slow_ma"] = slow_ma
            indicators["ma_spread"] = 100 * (fast_ma - slow_ma) / slow_ma
            
            # Volume analysis
            volume_ma = data["volume"].rolling(window=self.parameters["volume_ma_period"]).mean()
            volume_ratio = data["volume"] / volume_ma
            indicators["volume_ratio"] = volume_ratio
            indicators["volume_spike"] = volume_ratio > self.parameters["volume_spike_threshold"]
            
            # Mean reversion strength indicator (0-100)
            # Combines z-score, RSI and Bollinger extremes
            mr_strength = 50  # Neutral starting point
            
            # Z-score contribution (high z-score = high mean reversion potential)
            z_score_factor = min(abs(z_score.iloc[-1]) / 3.0 * 50, 50)
            z_score_direction = -1 if z_score.iloc[-1] > 0 else 1  # Negative z-score = bullish reversion
            mr_strength += z_score_factor * z_score_direction
            
            # RSI contribution
            if "rsi" in indicators:
                rsi = indicators["rsi"].iloc[-1]
                if rsi < 30:  # Oversold
                    mr_strength += (30 - rsi) * 1.5  # More oversold = stronger bullish reversion
                elif rsi > 70:  # Overbought
                    mr_strength -= (rsi - 70) * 1.5  # More overbought = stronger bearish reversion
            
            # Bollinger contribution
            if "bb_upper" in indicators and "bb_lower" in indicators:
                upper = indicators["bb_upper"].iloc[-1]
                lower = indicators["bb_lower"].iloc[-1]
                close = data["close"].iloc[-1]
                
                if close > upper:  # Above upper band
                    band_pct = (close - upper) / (upper - lower) * 100
                    mr_strength -= min(band_pct * 2, 20)  # Strong bearish reversion signal
                elif close < lower:  # Below lower band
                    band_pct = (lower - close) / (upper - lower) * 100
                    mr_strength += min(band_pct * 2, 20)  # Strong bullish reversion signal
            
            # Ensure within 0-100 range
            indicators["mean_reversion_strength"] = max(0, min(100, mr_strength))
            
            # Mean reversion signal - direction and strength
            indicators["reversion_direction"] = "bearish" if z_score.iloc[-1] > 0 else "bullish"
            indicators["reversion_signal"] = abs(z_score.iloc[-1]) > self.parameters["z_score_entry_threshold"]
            
            # Store for reference and analysis
            self.z_score_history.append(z_score.iloc[-1])
            if len(self.z_score_history) > 100:
                self.z_score_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error calculating mean reversion indicators: {e}")
            
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mean reversion trading signals.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "long": False,
            "short": False,
            "exit_long": False,
            "exit_short": False,
            "signal_strength": 0.0,
        }
        
        if data.empty or not indicators or "z_score" not in indicators:
            return signals
            
        # Current market data
        current_price = data["close"].iloc[-1]
        current_time = data.index[-1] if not data.index.empty else datetime.now()
        
        # Check if we're in cooldown period after a trade
        if self.reversion_signals["cooldown_until"] and current_time < self.reversion_signals["cooldown_until"]:
            logger.info(f"In cooldown period until {self.reversion_signals['cooldown_until']}")
            return signals
        
        # Extract key indicators
        z_score = indicators["z_score"].iloc[-1]
        z_score_threshold = self.parameters["z_score_entry_threshold"]
        
        # RSI filter
        rsi_filter_passed = True
        if self.parameters["use_rsi_filter"] and "rsi" in indicators:
            rsi = indicators["rsi"].iloc[-1]
            rsi_upper = self.parameters["rsi_upper_threshold"]
            rsi_lower = self.parameters["rsi_lower_threshold"]
            
            # For long signals, need RSI < lower threshold (oversold)
            # For short signals, need RSI > upper threshold (overbought)
            if z_score > 0:  # Potential short (overbought)
                rsi_filter_passed = rsi > rsi_upper
            else:  # Potential long (oversold)
                rsi_filter_passed = rsi < rsi_lower
        
        # Moving average filter
        ma_filter_passed = True
        if self.parameters["use_ma_filter"] and "fast_ma" in indicators and "slow_ma" in indicators:
            fast_ma = indicators["fast_ma"].iloc[-1]
            slow_ma = indicators["slow_ma"].iloc[-1]
            
            # For short signals, fast MA should be below slow MA (downtrend)
            # For long signals, fast MA should be above slow MA (uptrend)
            # This is counterintuitive for mean reversion, but helps filter out false signals
            if z_score > 0:  # Potential short
                ma_filter_passed = fast_ma < slow_ma  # Confirm downtrend
            else:  # Potential long
                ma_filter_passed = fast_ma > slow_ma  # Confirm uptrend
        
        # Volume confirmation
        volume_confirmed = not self.parameters["use_volume_filter"]
        if self.parameters["use_volume_filter"] and "volume_ratio" in indicators:
            volume_ratio = indicators["volume_ratio"].iloc[-1]
            volume_threshold = self.parameters["volume_spike_threshold"]
            volume_confirmed = volume_ratio > volume_threshold
        
        # Generate entry signals
        signal_strength = min(abs(z_score) / z_score_threshold, 1.0)
        
        # Mean reversion entry logic
        if z_score > z_score_threshold and rsi_filter_passed and volume_confirmed:
            # Overbought condition - short signal
            signals["short"] = True
            signals["signal_strength"] = signal_strength
            self.reversion_signals["overbought"] = True
            logger.info(f"Mean reversion SHORT signal: z-score={z_score:.2f}, RSI={indicators.get('rsi', pd.Series()).iloc[-1]:.1f}")
            
        elif z_score < -z_score_threshold and rsi_filter_passed and volume_confirmed:
            # Oversold condition - long signal
            signals["long"] = True
            signals["signal_strength"] = signal_strength
            self.reversion_signals["oversold"] = True
            logger.info(f"Mean reversion LONG signal: z-score={z_score:.2f}, RSI={indicators.get('rsi', pd.Series()).iloc[-1]:.1f}")
        
        # Exit logic - when price reverts toward the mean
        exit_threshold = self.parameters["z_score_exit_threshold"]
        
        # Check for exits on open positions
        for position in self.positions:
            position_z_score = None
            if 'entry_z_score' in position.metadata:
                position_z_score = position.metadata['entry_z_score']
                
            if position.direction == "long":
                # Exit long when z-score reverts to mean (becomes less negative or positive)
                if position_z_score and position_z_score < 0 and z_score > -exit_threshold:
                    signals["exit_long"] = True
                    logger.info(f"Mean reversion exit LONG: z-score reverted from {position_z_score:.2f} to {z_score:.2f}")
                    
                # Also exit if trade duration exceeds max holding periods
                elif position.entry_time and (current_time - position.entry_time) > timedelta(minutes=self.parameters["max_holding_periods"]):
                    signals["exit_long"] = True
                    logger.info(f"Mean reversion exit LONG: max holding period exceeded")
                    
            elif position.direction == "short":
                # Exit short when z-score reverts to mean (becomes less positive or negative)
                if position_z_score and position_z_score > 0 and z_score < exit_threshold:
                    signals["exit_short"] = True
                    logger.info(f"Mean reversion exit SHORT: z-score reverted from {position_z_score:.2f} to {z_score:.2f}")
                    
                # Also exit if trade duration exceeds max holding periods
                elif position.entry_time and (current_time - position.entry_time) > timedelta(minutes=self.parameters["max_holding_periods"]):
                    signals["exit_short"] = True
                    logger.info(f"Mean reversion exit SHORT: max holding period exceeded")
        
        # Record signal time and state
        if signals["long"] or signals["short"]:
            self.reversion_signals["last_signal_time"] = current_time
            
            # Store signal metadata for position management
            self.signals = signals
            self.signals["current_z_score"] = z_score
            self.signals["entry_price"] = current_price
            self.signals["signal_time"] = current_time
            
        return signals
        
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on mean reversion signal strength and risk parameters.
        
        For mean reversion scalping, we adjust position size based on:
        1. Signal strength (z-score deviation)
        2. Market volatility (ATR)
        3. Risk per trade setting
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        # Get the base position size from parent
        base_position_size = super().calculate_position_size(direction, data, indicators)
        
        # Adjust based on mean reversion signal strength
        if not self.signals or "signal_strength" not in self.signals:
            return base_position_size
            
        signal_strength = self.signals["signal_strength"]
        
        # For very strong signals (high z-score), increase position size
        if signal_strength > 0.8:
            position_size = base_position_size * 1.2
        # For moderate signals, use normal position size
        elif signal_strength > 0.5:
            position_size = base_position_size
        # For weaker signals, reduce position size
        else:
            position_size = base_position_size * 0.8
            
        # Additional adjustments based on recent performance
        if self.trade_stats["total_trades"] > 10:
            win_rate = self.trade_stats["winning_trades"] / self.trade_stats["total_trades"]
            
            # If recent performance is strong, cautiously increase size
            if win_rate > 0.6:
                position_size *= 1.1
            # If recent performance is poor, reduce size
            elif win_rate < 0.4:
                position_size *= 0.8
                
        # Apply precision appropriate for the asset
        decimals = 8 if self.session.symbol.startswith("BTC") else 6
        position_size = round(position_size, decimals)
        
        # Ensure minimum trade size
        min_trade_size = self.session.min_trade_size
        position_size = max(position_size, min_trade_size)
        
        logger.info(f"Mean reversion position size: {position_size} {self.session.symbol.split('-')[0]} "
                   f"(signal strength: {signal_strength:.2f})")
                  
        return position_size
    
    def _on_position_opened(self, event: Event) -> None:
        """
        Handle position opened events.
        
        For mean reversion trades, we capture initial conditions for exit logic.
        """
        super()._on_position_opened(event)
        
        position_data = event.data
        if position_data["strategy_id"] != self.id:
            return
            
        # Store mean reversion metadata with the position
        position_id = position_data["position_id"]
        position = self.get_position_by_id(position_id)
        
        if position and self.signals:
            # Record entry z-score for exit logic
            if "current_z_score" in self.signals:
                position.metadata["entry_z_score"] = self.signals["current_z_score"]
                
            # Start cooldown timer for this trade
            self.active_mean_reversion_trades[position_id] = {
                "entry_time": datetime.now(),
                "entry_price": position.entry_price,
                "entry_z_score": self.signals.get("current_z_score"),
                "direction": position.direction,
            }
            
            logger.info(f"Mean reversion {position.direction} position opened at {position.entry_price} "
                      f"with z-score: {position.metadata.get('entry_z_score', 'N/A')}")
    
    def _on_position_closed(self, event: Event) -> None:
        """
        Handle position closed events.
        
        Updates strategy statistics and implements post-trade cooldown period.
        """
        super()._on_position_closed(event)
        
        position_data = event.data
        if position_data["strategy_id"] != self.id:
            return
            
        position_id = position_data["position_id"]
        
        # Update trade statistics
        if position_id in self.active_mean_reversion_trades:
            trade_data = self.active_mean_reversion_trades[position_id]
            entry_time = trade_data["entry_time"]
            exit_time = datetime.now()
            holding_time = (exit_time - entry_time).total_seconds() / 60  # minutes
            
            # Update stats
            self.trade_stats["total_trades"] += 1
            
            profit_loss = position_data.get("profit_loss", 0)
            if profit_loss > 0:
                self.trade_stats["winning_trades"] += 1
                trade_result = "winning"
            else:
                self.trade_stats["losing_trades"] += 1
                trade_result = "losing"
                
            # Update average holding time
            n = self.trade_stats["total_trades"]
            old_avg = self.trade_stats["avg_holding_time"]
            self.trade_stats["avg_holding_time"] = ((n-1) * old_avg + holding_time) / n
            
            logger.info(f"Mean reversion {trade_data['direction']} trade closed: {trade_result}, "
                      f"holding time: {holding_time:.1f} minutes, "
                      f"entry z-score: {trade_data.get('entry_z_score', 'N/A')}")
            
            # Set cooldown period after trade
            cooldown_minutes = self.parameters["cooldown_periods"]
            self.reversion_signals["cooldown_until"] = datetime.now() + timedelta(minutes=cooldown_minutes)
            
            # Clean up
            del self.active_mean_reversion_trades[position_id]
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Mean reversion scalping works best in ranging, choppy markets and performs
        poorly in trending or breakout conditions.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "ranging": 0.95,        # Excellent in ranging markets
            "volatile": 0.75,       # Good in volatile but non-trending markets
            "trending": 0.30,       # Poor in trending markets
            "calm": 0.70,           # Good in calm markets
            "breakout": 0.25,       # Very poor during breakouts
            "high_volume": 0.80,    # Good in high volume (more rapid mean reversion)
            "low_volume": 0.45,     # Poor in low volume (slower mean reversion)
            "high_liquidity": 0.90, # Very good in high liquidity (easier to exit)
            "low_liquidity": 0.35,  # Poor in low liquidity (slippage risk)
        }
        
        return compatibility_map.get(market_regime, 0.60)
