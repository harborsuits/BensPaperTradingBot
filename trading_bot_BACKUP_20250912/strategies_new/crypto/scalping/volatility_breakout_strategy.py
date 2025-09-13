#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Volatility Breakout Scalping Strategy

This strategy identifies and trades short-term breakouts during periods of high volatility
in cryptocurrency markets. It monitors volatility bands and enters trades when price 
breaks out of consolidation patterns with volume confirmation.

Key characteristics:
- Very short holding periods (minutes to hours)
- Volatility-based entry triggers
- ATR-based position sizing and stop placement
- Multi-timeframe confirmation
- Volume surge detection
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.crypto.scalping.crypto_scalping_strategy import CryptoScalpingStrategy
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="VolatilityBreakoutScalpingStrategy",
    market_type="crypto",
    description="High-frequency volatility breakout strategy for crypto markets targeting short-term price explosions",
    timeframes=["M1", "M5", "M15"],
    parameters={
        # Volatility indicators
        "atr_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        "atr_multiplier": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        "bollinger_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        "bollinger_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
        "keltner_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        "keltner_multiplier": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        
        # Breakout parameters
        "consolidation_bars": {"type": "int", "default": 5, "min": 3, "max": 15},
        "breakout_threshold": {"type": "float", "default": 0.6, "min": 0.2, "max": 1.0},
        "min_volatility_percentile": {"type": "float", "default": 75.0, "min": 50.0, "max": 95.0},
        "volume_surge_threshold": {"type": "float", "default": 2.0, "min": 1.5, "max": 5.0},
        
        # Entry/exit management
        "stop_loss_atr_multiplier": {"type": "float", "default": 1.2, "min": 0.5, "max": 2.5},
        "take_profit_atr_multiplier": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
        "trailing_stop_activation": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
        "trailing_stop_distance": {"type": "float", "default": 0.5, "min": 0.2, "max": 1.5},
        "max_trade_duration_minutes": {"type": "int", "default": 120, "min": 15, "max": 480},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 5},
        "max_trades_per_hour": {"type": "int", "default": 4, "min": 1, "max": 12},
        "min_risk_reward": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
    }
)
class VolatilityBreakoutScalpingStrategy(CryptoScalpingStrategy):
    """
    Volatility Breakout Scalping Strategy for cryptocurrencies.
    
    This strategy builds on the base CryptoScalpingStrategy with specific focus on:
    1. Detecting volatility compression and expansion cycles
    2. Identifying breakout signals when price exceeds volatility bands
    3. Using volume surge confirmation for entry validation
    4. Implementing precise exit rules for quick profit-taking in volatile conditions
    5. Advanced risk management specifically calibrated for high-volatility environments
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the volatility breakout scalping strategy.
        
        Args:
            session: Crypto trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base scalping strategy
        super().__init__(session, data_pipeline, parameters)
        
        # Volatility-specific state tracking
        self.volatility_history = []  # Track recent volatility metrics
        self.consolidation_periods = {}  # Track detected consolidation periods
        self.breakout_signals = {}  # Track detected breakout signals
        self.volatility_percentile = 0.0  # Current volatility percentile
        self.is_high_volatility = False  # Flag for high volatility conditions
        
        # Performance tracking specific to volatility breakouts
        self.successful_breakouts = 0
        self.failed_breakouts = 0
        self.breakout_win_rate = 0.0
        
        logger.info(f"Volatility Breakout Scalping Strategy initialized for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators focusing on volatility measurements.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Get base indicators from parent class
        indicators = super().calculate_indicators(data)
        
        if len(data) < 30:  # Need sufficient data
            return indicators
        
        # Calculate volatility indicators
        atr_period = self.parameters["atr_period"]
        atr = self._calculate_atr(data, atr_period)
        
        # Calculate Bollinger Bands (volatility-based bands)
        bb_period = self.parameters["bollinger_period"]
        bb_std = self.parameters["bollinger_std"]
        middle_band = data['close'].rolling(window=bb_period).mean()
        std_dev = data['close'].rolling(window=bb_period).std()
        upper_band = middle_band + (std_dev * bb_std)
        lower_band = middle_band - (std_dev * bb_std)
        
        # Calculate Keltner Channels (ATR-based bands)
        kc_period = self.parameters["keltner_period"]
        kc_multiplier = self.parameters["keltner_multiplier"]
        kc_middle = data['close'].rolling(window=kc_period).mean()
        kc_atr = self._calculate_atr(data, kc_period)
        kc_upper = kc_middle + (kc_atr[-1] * kc_multiplier)
        kc_lower = kc_middle - (kc_atr[-1] * kc_multiplier)
        
        # Calculate volatility squeeze (when BBs are inside KCs)
        is_squeeze = (lower_band.iloc[-1] > kc_lower.iloc[-1]) and (upper_band.iloc[-1] < kc_upper.iloc[-1])
        
        # Detect consolidation pattern
        close = data['close'].values
        is_consolidating = self._is_consolidating(close, self.parameters["consolidation_bars"])
        
        # Volume analysis
        volume_sma = data['volume'].rolling(window=20).mean()
        volume_ratio = data['volume'].iloc[-1] / volume_sma.iloc[-1] if not np.isnan(volume_sma.iloc[-1]) else 1.0
        
        # Calculate historical volatility percentile
        if len(atr) > 20:
            recent_atr = atr[-20:]
            self.volatility_percentile = np.percentile(recent_atr, 75)  # 75th percentile
            self.is_high_volatility = atr[-1] > self.volatility_percentile
        
        # Update volatility history
        self.volatility_history.append(atr[-1])
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
        
        # Store all calculated indicators
        indicators.update({
            'atr': atr[-1],
            'bollinger_middle': middle_band.iloc[-1],
            'bollinger_upper': upper_band.iloc[-1],
            'bollinger_lower': lower_band.iloc[-1],
            'keltner_middle': kc_middle.iloc[-1],
            'keltner_upper': kc_upper.iloc[-1],
            'keltner_lower': kc_lower.iloc[-1],
            'is_squeeze': is_squeeze,
            'is_consolidating': is_consolidating,
            'volume_ratio': volume_ratio,
            'volatility_percentile': self.volatility_percentile,
            'is_high_volatility': self.is_high_volatility
        })
        
        # Detect breakout conditions
        if is_consolidating:
            # Track the consolidation period
            self.consolidation_periods[data.index[-1]] = {
                'high': np.max(close[-self.parameters["consolidation_bars"]:]),
                'low': np.min(close[-self.parameters["consolidation_bars"]:]),
                'atr': atr[-1],
                'duration': self.parameters["consolidation_bars"]
            }
        
        return indicators
    
    def _is_consolidating(self, prices: np.ndarray, lookback: int) -> bool:
        """
        Determine if the market is in a consolidation phase.
        
        Args:
            prices: Array of price data
            lookback: Number of bars to analyze
            
        Returns:
            True if market is consolidating
        """
        if len(prices) < lookback:
            return False
            
        # Get the range of the lookback period
        recent_prices = prices[-lookback:]
        price_range = np.max(recent_prices) - np.min(recent_prices)
        avg_price = np.mean(recent_prices)
        
        # Calculate range as a percentage of average price
        range_pct = price_range / avg_price
        
        # Compare to the average range of previous periods
        if len(prices) >= (lookback * 3):
            prev_periods = []
            for i in range(1, 3):  # Check 2 previous periods
                prev_slice = prices[-(i+1)*lookback:-i*lookback]
                prev_range = np.max(prev_slice) - np.min(prev_slice)
                prev_avg = np.mean(prev_slice)
                prev_periods.append(prev_range / prev_avg)
            
            avg_prev_range = np.mean(prev_periods)
            
            # Consolidation means current range is smaller than previous ranges
            return range_pct < (avg_prev_range * 0.7)  # 30% less volatile
        
        # Default if not enough history
        return range_pct < 0.03  # Less than 3% range is considered consolidation
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on volatility breakout patterns.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'entry': None,
            'direction': None,
            'confidence': 0.0,
            'exit': []
        }
        
        # Only proceed if we have enough data
        if len(data) < 30:
            return signals
            
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        
        # Check if we're in high volatility environment
        if not indicators.get('is_high_volatility', False):
            logger.debug(f"Current volatility below threshold, no breakout signals for {self.session.symbol}")
            return signals
            
        # Check for squeeze and breakout
        is_squeeze = indicators.get('is_squeeze', False)
        was_consolidating = indicators.get('is_consolidating', False)
        volume_surge = indicators.get('volume_ratio', 1.0) > self.parameters["volume_surge_threshold"]
        
        # Define breakout thresholds
        atr = indicators.get('atr', current_price * 0.01)  # Default to 1% if ATR not available
        breakout_threshold = atr * self.parameters["breakout_threshold"]
        
        # Check for recent consolidation periods
        recent_consolidation = None
        for timestamp, period in sorted(self.consolidation_periods.items(), reverse=True):
            if timestamp + pd.Timedelta(minutes=30) >= data.index[-1]:  # Within last 30 minutes
                recent_consolidation = period
                break
                
        if recent_consolidation and not indicators.get('is_consolidating', True):
            # We have a recent consolidation and now we're not consolidating - potential breakout
            high = recent_consolidation['high']
            low = recent_consolidation['low']
            
            # Check breakout direction
            breakout_up = current_price > (high + breakout_threshold) and current_price > prev_price
            breakout_down = current_price < (low - breakout_threshold) and current_price < prev_price
            
            if (breakout_up or breakout_down) and volume_surge:
                direction = "long" if breakout_up else "short"
                
                # Calculate confidence based on multiple factors
                # 1. Volatility percentile
                vol_factor = min(1.0, indicators.get('volatility_percentile', 75) / 100)
                
                # 2. Volume surge strength
                volume_factor = min(1.0, indicators.get('volume_ratio', 1.0) / self.parameters["volume_surge_threshold"])
                
                # 3. Breakout strength
                if breakout_up:
                    breakout_strength = (current_price - high) / breakout_threshold
                else:
                    breakout_strength = (low - current_price) / breakout_threshold
                breakout_factor = min(1.0, breakout_strength)
                
                # Combined confidence score
                confidence = 0.4 * vol_factor + 0.3 * volume_factor + 0.3 * breakout_factor
                confidence = min(0.95, confidence)  # Cap at 0.95
                
                signals['entry'] = True
                signals['direction'] = direction
                signals['confidence'] = confidence
                signals['entry_price'] = current_price
                
                # Record the breakout signal
                self.breakout_signals[data.index[-1]] = {
                    'direction': direction,
                    'price': current_price,
                    'confidence': confidence,
                    'consolidation': recent_consolidation
                }
                
                logger.info(f"Volatility breakout signal: {direction.upper()} {self.session.symbol} @ {current_price:.8f} (conf: {confidence:.2f})")
        
        # Check for exit signals on existing positions
        for position in self.positions:
            # Simple take-profit and stop-loss logic handled by base strategy
            # Add specific exit signals for volatility reversal
            
            # 1. Exit long on bearish reversal
            if position.direction == "long" and indicators.get('is_high_volatility', False):
                if current_price < indicators.get('keltner_middle', current_price):
                    signals['exit'].append({
                        'position_id': position.id,
                        'reason': 'volatility_reversal',
                        'price': current_price
                    })
            
            # 2. Exit short on bullish reversal
            elif position.direction == "short" and indicators.get('is_high_volatility', False):
                if current_price > indicators.get('keltner_middle', current_price):
                    signals['exit'].append({
                        'position_id': position.id,
                        'reason': 'volatility_reversal',
                        'price': current_price
                    })
            
            # 3. Time-based exit (volatility often reverts quickly)
            entry_time = position.entry_time
            current_time = datetime.utcnow()
            if (current_time - entry_time).total_seconds() > self.parameters["max_trade_duration_minutes"] * 60:
                signals['exit'].append({
                    'position_id': position.id,
                    'reason': 'max_duration',
                    'price': current_price
                })
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size adjusted for volatility conditions.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        # Use base calculation from parent class
        base_position_size = super().calculate_position_size(direction, data, indicators)
        
        # Adjust for current volatility level
        atr = indicators.get('atr', data['close'].iloc[-1] * 0.01)
        
        # Calculate volatility adjustment factor
        # In high volatility, we might want to reduce position size
        if self.is_high_volatility:
            # Calculate volatility percentile relative to history
            if len(self.volatility_history) > 0:
                vol_percentile = sum(1 for x in self.volatility_history if x < atr) / len(self.volatility_history)
                
                # Adjust position size down as volatility increases
                # 90th percentile = 70% position size, 99th percentile = 50% position size
                if vol_percentile > 0.9:
                    volatility_factor = max(0.5, 1.0 - (vol_percentile - 0.9) * 5)
                    adjusted_size = base_position_size * volatility_factor
                    
                    logger.info(f"High volatility adjustment: {volatility_factor:.2f} - Position size: {base_position_size:.8f} → {adjusted_size:.8f}")
                    return adjusted_size
        
        return base_position_size
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events with focus on volatility analysis.
        
        Args:
            event: Timeframe completed event
        """
        # Call the parent method first
        super()._on_timeframe_completed(event)
        
        # Additional processing for breakout analysis
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe')
        
        # Only process events for our symbol and timeframe
        if symbol != self.session.symbol or timeframe != self.session.timeframe:
            return
            
        # Clean up old consolidation periods and breakout signals
        current_time = datetime.utcnow()
        cutoff = pd.Timestamp(current_time - timedelta(hours=24))
        
        # Remove old consolidation periods
        self.consolidation_periods = {
            ts: data for ts, data in self.consolidation_periods.items()
            if ts >= cutoff
        }
        
        # Remove old breakout signals
        self.breakout_signals = {
            ts: data for ts, data in self.breakout_signals.items()
            if ts >= cutoff
        }
        
        # Calculate breakout performance metrics if we have enough data
        self._update_breakout_performance()
    
    def _update_breakout_performance(self) -> None:
        """Update performance metrics for breakout signals."""
        if len(self.positions) == 0 and len(self.breakout_signals) > 10:
            # Calculate win rate if we have closed positions
            if self.successful_breakouts + self.failed_breakouts > 0:
                self.breakout_win_rate = self.successful_breakouts / (self.successful_breakouts + self.failed_breakouts)
                logger.info(f"Breakout win rate: {self.breakout_win_rate:.2f} ({self.successful_breakouts}/{self.successful_breakouts + self.failed_breakouts})")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate strategy compatibility with current market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Volatility breakout strategy works best in volatile and breakout regimes
        compatibility_map = {
            "volatile": 0.95,         # Excellent in volatile markets
            "ranging": 0.30,          # Poor in ranging markets
            "trending": 0.60,         # Moderate in trending markets
            "breakout": 0.90,         # Excellent during breakouts
            "high_volume": 0.85,      # Great in high volume
            "low_volume": 0.30,       # Poor in low volume
            "high_volatility": 0.95,  # Excellent in high volatility
            "low_volatility": 0.20,   # Very poor in low volatility
        }
        
        return compatibility_map.get(market_regime, 0.50)
