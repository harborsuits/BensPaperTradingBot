#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Trend Following Strategy

This strategy identifies and follows strong trends in forex markets using:
- Moving averages (SMA, EMA, WMA)
- ADX for trend strength detection
- MACD for trend confirmation
- ATR for volatility measurement and position sizing
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from trading_bot.strategies_new.forex.base import ForexBaseStrategy, ForexSession
from trading_bot.core.events import EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="ForexTrendFollowingStrategy",
    market_type="forex",
    description="Follows strong trends in forex markets using multiple indicators",
    timeframes=["H1", "H4", "D1"],
    parameters={
        "fast_ma_period": {"type": "int", "default": 9, "min": 5, "max": 20},
        "slow_ma_period": {"type": "int", "default": 21, "min": 15, "max": 50},
        "adx_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "adx_threshold": {"type": "float", "default": 25.0, "min": 15.0, "max": 35.0},
        "macd_fast": {"type": "int", "default": 12, "min": 8, "max": 20},
        "macd_slow": {"type": "int", "default": 26, "min": 20, "max": 40},
        "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 15},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.03},
        "risk_reward_ratio": {"type": "float", "default": 2.0, "min": 1.5, "max": 3.5},
        "stop_loss_atr_multiplier": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
        "entry_threshold": {"type": "float", "default": 0.7, "min": 0.5, "max": 0.9},
        "exit_threshold": {"type": "float", "default": 0.6, "min": 0.4, "max": 0.8},
    }
)
class ForexTrendFollowingStrategy(ForexBaseStrategy):
    """
    A trend following strategy for forex markets.
    
    This strategy identifies and follows strong trends using:
    - Multiple moving averages (SMA, EMA, WMA) for trend direction
    - ADX (Average Directional Index) to measure trend strength
    - MACD (Moving Average Convergence Divergence) for trend confirmation
    - ATR (Average True Range) for volatility measurement and position sizing
    
    The strategy only enters trades when:
    1. Multiple indicators confirm a trend
    2. Trend strength is above threshold
    3. Risk parameters are satisfied
    """
    
    def __init__(self, session: ForexSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the forex trend following strategy.
        
        Args:
            session: Forex trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Merge default parameters with any provided parameters
        default_params = {
            "fast_ma_period": 9,
            "slow_ma_period": 21,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
            "risk_per_trade": 0.01,
            "risk_reward_ratio": 2.0,
            "stop_loss_atr_multiplier": 2.0,
            "entry_threshold": 0.7,
            "exit_threshold": 0.6,
        }
        
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy
        super().__init__(session, data_pipeline, default_params)
        
        # Strategy-specific initialization
        self.last_signal = None
        
        logger.info(f"Initialized {self.name} with parameters: {self.parameters}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        if len(data) < self.parameters["slow_ma_period"] + 10:
            return {}
            
        indicators = {}
        
        # Moving Averages
        indicators["sma_fast"] = self._calculate_sma(data, self.parameters["fast_ma_period"])
        indicators["sma_slow"] = self._calculate_sma(data, self.parameters["slow_ma_period"])
        indicators["ema_fast"] = self._calculate_ema(data, self.parameters["fast_ma_period"])
        indicators["ema_slow"] = self._calculate_ema(data, self.parameters["slow_ma_period"])
        indicators["wma_fast"] = self._calculate_wma(data, self.parameters["fast_ma_period"])
        indicators["wma_slow"] = self._calculate_wma(data, self.parameters["slow_ma_period"])
        
        # Moving Average Cross Status
        indicators["ma_cross_status"] = indicators["sma_fast"][-1] > indicators["sma_slow"][-1]
        indicators["ema_cross_status"] = indicators["ema_fast"][-1] > indicators["ema_slow"][-1]
        indicators["wma_cross_status"] = indicators["wma_fast"][-1] > indicators["wma_slow"][-1]
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            data, 
            self.parameters["macd_fast"], 
            self.parameters["macd_slow"], 
            self.parameters["macd_signal"]
        )
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = signal_line
        indicators["macd_histogram"] = histogram
        indicators["macd_above_signal"] = macd_line[-1] > signal_line[-1]
        
        # ADX for trend strength
        adx = self._calculate_adx(data, self.parameters["adx_period"])
        indicators["adx"] = adx
        indicators["trend_strength"] = "strong" if adx[-1] > self.parameters["adx_threshold"] else "weak"
        
        # ATR for volatility
        indicators["atr"] = self._calculate_atr(data, self.parameters["atr_period"])
        
        # Trend direction based on price vs moving averages
        indicators["price_above_sma_slow"] = data["close"].iloc[-1] > indicators["sma_slow"][-1]
        indicators["price_above_ema_slow"] = data["close"].iloc[-1] > indicators["ema_slow"][-1]
        
        # Overall trend direction (consensus of indicators)
        ma_bullish_signals = sum([
            indicators["price_above_sma_slow"],
            indicators["price_above_ema_slow"],
            indicators["ma_cross_status"],
            indicators["ema_cross_status"],
            indicators["wma_cross_status"]
        ])
        
        if ma_bullish_signals >= 3 and indicators["macd_above_signal"]:
            indicators["trend_direction"] = "bullish"
        elif ma_bullish_signals <= 2 and not indicators["macd_above_signal"]:
            indicators["trend_direction"] = "bearish"
        else:
            indicators["trend_direction"] = "neutral"
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on calculated indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        if not indicators or len(data) < 2:
            return {"entry": {}, "exit": {}, "position_adjust": {}}
            
        signals = {
            "entry": {"long": 0.0, "short": 0.0},
            "exit": {"long": 0.0, "short": 0.0},
            "position_adjust": {"long": 0.0, "short": 0.0}
        }
        
        # Check for strong trend
        if indicators.get("trend_strength") != "strong":
            return signals
            
        # Entry Signals
        if indicators.get("trend_direction") == "bullish":
            # Calculate strength of bullish signal
            ma_agreement = sum([
                indicators.get("ma_cross_status", False),
                indicators.get("ema_cross_status", False),
                indicators.get("wma_cross_status", False)
            ]) / 3.0
            
            adx_strength = min(1.0, indicators.get("adx", [0])[-1] / 50.0)
            macd_strength = 1.0 if indicators.get("macd_above_signal", False) else 0.0
            
            # Weighted average of signals
            long_signal = (0.4 * ma_agreement) + (0.4 * adx_strength) + (0.2 * macd_strength)
            signals["entry"]["long"] = long_signal
            
            # If we have an open short position, signal to exit
            signals["exit"]["short"] = max(0.7, long_signal)
            
        elif indicators.get("trend_direction") == "bearish":
            # Calculate strength of bearish signal
            ma_agreement = sum([
                not indicators.get("ma_cross_status", True),
                not indicators.get("ema_cross_status", True),
                not indicators.get("wma_cross_status", True)
            ]) / 3.0
            
            adx_strength = min(1.0, indicators.get("adx", [0])[-1] / 50.0)
            macd_strength = 1.0 if not indicators.get("macd_above_signal", True) else 0.0
            
            # Weighted average of signals
            short_signal = (0.4 * ma_agreement) + (0.4 * adx_strength) + (0.2 * macd_strength)
            signals["entry"]["short"] = short_signal
            
            # If we have an open long position, signal to exit
            signals["exit"]["long"] = max(0.7, short_signal)
        
        # Save the signal for reference
        self.last_signal = signals
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters and ATR.
        
        This method uses the ATR to determine a volatility-adjusted position size
        that risks a fixed percentage of the account on each trade.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in base currency units
        """
        if not indicators or "atr" not in indicators or len(data) < 1:
            return 0.0
            
        # Get the current ATR value
        atr = indicators["atr"][-1]
        
        # Account equity would come from the broker/portfolio in a real system
        # Using a placeholder value here
        account_equity = 10000.0  # Example value
        
        # Calculate the risk amount in account currency
        risk_amount = account_equity * self.parameters["risk_per_trade"]
        
        # Stop loss distance in price
        stop_loss_distance = atr * self.parameters["stop_loss_atr_multiplier"]
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Calculate position size
        if stop_loss_distance > 0:
            # Position size in base units
            position_size = risk_amount / stop_loss_distance
            
            # Convert to lot size (standard lot = 100,000 units)
            lot_size = position_size / 100000.0
            
            # Round to mini lots (0.1 lot increments)
            lot_size = round(lot_size * 10) / 10
            
            # Convert back to base units
            return lot_size * 100000.0
        
        return 0.0
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Trend following strategies perform best in trending markets, poorly in
        ranging markets, and moderately in other regimes.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.95,       # Excellent in trending markets
            "ranging": 0.30,        # Poor in ranging markets
            "volatile": 0.60,       # Moderate in volatile markets
            "calm": 0.70,           # Good in calm markets
            "breakout": 0.80,       # Good in breakout markets
        }
        
        return compatibility_map.get(market_regime, 0.50)
    
    def _calculate_sma(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        return data["close"].rolling(window=period).mean().values
    
    def _calculate_ema(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        return data["close"].ewm(span=period, adjust=False).mean().values
    
    def _calculate_wma(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return data["close"].rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        ).values
    
    def _calculate_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal line, and Histogram."""
        ema_fast = data["close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data["close"].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, histogram.values
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.values
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average Directional Index (ADX)."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        # Handle first row NaN
        plus_dm = plus_dm.fillna(0)
        minus_dm = minus_dm.fillna(0)
        
        # Calculate true +DM and -DM
        plus_dm = np.where(
            (plus_dm > 0) & (plus_dm > minus_dm.abs()),
            plus_dm,
            0
        )
        minus_dm = np.where(
            (minus_dm < 0) & (minus_dm.abs() > plus_dm),
            minus_dm.abs(),
            0
        )
        
        # Calculate TR (True Range)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        
        # Calculate smoothed +DM, -DM, and TR
        smoothed_plus_dm = pd.Series(plus_dm).rolling(window=period).sum()
        smoothed_minus_dm = pd.Series(minus_dm).rolling(window=period).sum()
        smoothed_tr = tr.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Calculate DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx.values
