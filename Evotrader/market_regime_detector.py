#!/usr/bin/env python3
"""
Market Regime Detector

Detects market regimes based on price data and technical indicators.
"""

import numpy as np
import pandas as pd
import logging
import talib
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_regime_detector')

class MarketRegimeDetector:
    """
    Detects current market regime based on price action and indicators.
    
    Regimes:
    - bullish: Strong uptrend with low volatility
    - bearish: Strong downtrend with low volatility
    - volatile_bullish: Uptrend with high volatility
    - volatile_bearish: Downtrend with high volatility
    - ranging: Sideways movement with low volatility
    - choppy: Sideways movement with high volatility
    """
    
    # Regime definitions
    REGIME_BULLISH = "bullish"
    REGIME_BEARISH = "bearish"
    REGIME_VOLATILE_BULLISH = "volatile_bullish"
    REGIME_VOLATILE_BEARISH = "volatile_bearish"
    REGIME_RANGING = "ranging"
    REGIME_CHOPPY = "choppy"
    
    # Volatility thresholds
    LOW_VOLATILITY = 0.8  # Below this multiple of avg volatility is considered low
    HIGH_VOLATILITY = 1.5  # Above this multiple of avg volatility is considered high
    
    # Trend thresholds
    STRONG_TREND = 25  # ADX value for strong trend
    
    def __init__(self):
        """Initialize the market regime detector."""
        pass
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            price_data: OHLCV price data as DataFrame with 'open', 'high', 'low', 'close' columns
            
        Returns:
            Dictionary with regime classification and metrics
        """
        if len(price_data) < 30:
            logger.warning("Not enough data to detect regime (minimum 30 periods required)")
            return {
                'regime': self.REGIME_RANGING,
                'confidence': 0.5,
                'volatility': 'medium',
                'metrics': {}
            }
        
        try:
            # Calculate indicators
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            
            # 1. Trend detection
            # - Use 20-period moving average direction
            ma20 = self._safe_sma(close, 20)
            ma50 = self._safe_sma(close, 50)
            
            # Calculate trend direction using moving averages
            short_ma_slope = (ma20[-1] - ma20[-6]) / ma20[-6] if ma20[-6] > 0 else 0
            long_ma_slope = (ma50[-1] - ma50[-10]) / ma50[-10] if ma50[-10] > 0 else 0
            
            ma_cross_trend = 1 if ma20[-1] > ma50[-1] else -1
            
            # 2. Volatility detection
            # - Use ATR relative to its moving average
            atr = self._calculate_atr(high, low, close, 14)
            atr_ma = self._safe_sma(atr, 20)
            
            relative_volatility = atr[-1] / atr_ma[-1] if atr_ma[-1] > 0 else 1.0
            
            # Classify volatility
            if relative_volatility < self.LOW_VOLATILITY:
                volatility_class = "low"
            elif relative_volatility > self.HIGH_VOLATILITY:
                volatility_class = "high"
            else:
                volatility_class = "medium"
            
            # 3. Range vs Trend detection
            # - Use ADX to determine if market is trending or ranging
            adx = self._calculate_adx(high, low, close, 14)
            is_trending = adx[-1] > self.STRONG_TREND
            
            # Determine regime
            regime = self.REGIME_RANGING  # Default
            
            if is_trending:
                if short_ma_slope > 0 and ma_cross_trend > 0:
                    if volatility_class == "high":
                        regime = self.REGIME_VOLATILE_BULLISH
                    else:
                        regime = self.REGIME_BULLISH
                elif short_ma_slope < 0 and ma_cross_trend < 0:
                    if volatility_class == "high":
                        regime = self.REGIME_VOLATILE_BEARISH
                    else:
                        regime = self.REGIME_BEARISH
            else:
                if volatility_class == "high":
                    regime = self.REGIME_CHOPPY
                else:
                    regime = self.REGIME_RANGING
            
            # Calculate confidence
            confidence = min(1.0, (adx[-1] / 30)) if is_trending else min(1.0, (30 - adx[-1]) / 15)
            
            return {
                'regime': regime,
                'confidence': float(confidence),
                'volatility': volatility_class,
                'metrics': {
                    'adx': float(adx[-1]) if len(adx) > 0 else 0.0,
                    'short_ma_slope': float(short_ma_slope),
                    'long_ma_slope': float(long_ma_slope),
                    'relative_volatility': float(relative_volatility)
                }
            }
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': self.REGIME_RANGING,
                'confidence': 0.5,
                'volatility': 'medium',
                'metrics': {}
            }
    
    def regime_history(self, price_data: pd.DataFrame, lookback_periods: int = 20) -> List[Dict[str, Any]]:
        """
        Get a history of regime changes.
        
        Args:
            price_data: OHLCV price data
            lookback_periods: Number of periods to look back
            
        Returns:
            List of regime classifications for each period
        """
        if len(price_data) < lookback_periods + 30:
            logger.warning(f"Not enough data for regime history (need {lookback_periods + 30} periods)")
            return []
        
        regimes = []
        
        for i in range(lookback_periods):
            # Create a slice of data up to the point in time
            end_idx = len(price_data) - lookback_periods + i
            historical_data = price_data.iloc[:end_idx].copy()
            
            # Detect regime for this period
            regime = self.detect_regime(historical_data)
            regime['date'] = price_data.index[end_idx - 1] if hasattr(price_data, 'index') else end_idx - 1
            regimes.append(regime)
        
        return regimes
    
    def _safe_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA safely handling edge cases."""
        if len(data) < period:
            return np.array([0.0])
        
        try:
            if hasattr(talib, 'SMA'):
                return talib.SMA(data, timeperiod=period)
            else:
                # Fallback if talib is not available
                result = np.zeros_like(data)
                for i in range(len(data)):
                    if i < period - 1:
                        result[i] = np.nan
                    else:
                        result[i] = np.mean(data[i-period+1:i+1])
                return np.nan_to_num(result, nan=0.0)
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return np.zeros_like(data)
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            if hasattr(talib, 'ATR'):
                return talib.ATR(high, low, close, timeperiod=period)
            else:
                # Fallback if talib is not available
                tr = np.zeros(len(high))
                
                # Calculate True Range
                for i in range(1, len(high)):
                    tr[i] = max(
                        high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1])
                    )
                
                # Calculate ATR
                atr = np.zeros(len(high))
                for i in range(period, len(high)):
                    atr[i] = np.mean(tr[i-period+1:i+1])
                
                return np.nan_to_num(atr, nan=0.0)
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return np.zeros_like(high)
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average Directional Index."""
        try:
            if hasattr(talib, 'ADX'):
                return talib.ADX(high, low, close, timeperiod=period)
            else:
                # Simplified fallback for ADX if talib is not available
                # This is a basic approximation; ADX calculation is complex
                # In production, use talib or implement the full calculation
                smooth_dm_plus = np.zeros(len(high))
                smooth_dm_minus = np.zeros(len(high))
                tr = np.zeros(len(high))
                
                for i in range(1, len(high)):
                    # Calculate directional movement
                    up_move = high[i] - high[i-1]
                    down_move = low[i-1] - low[i]
                    
                    dm_plus = max(0, up_move) if up_move > down_move else 0
                    dm_minus = max(0, down_move) if down_move > up_move else 0
                    
                    # True Range
                    tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    
                    # Apply smoothing after sufficient periods
                    if i >= period:
                        smooth_dm_plus[i] = smooth_dm_plus[i-1] - (smooth_dm_plus[i-1] / period) + dm_plus
                        smooth_dm_minus[i] = smooth_dm_minus[i-1] - (smooth_dm_minus[i-1] / period) + dm_minus
                    else:
                        # Initial smoothing
                        smooth_dm_plus[i] = dm_plus
                        smooth_dm_minus[i] = dm_minus
                
                # Calculate ADX
                atr = np.zeros(len(high))
                di_plus = np.zeros(len(high))
                di_minus = np.zeros(len(high))
                dx = np.zeros(len(high))
                adx = np.zeros(len(high))
                
                for i in range(period, len(high)):
                    atr[i] = np.mean(tr[i-period+1:i+1])
                    
                    if atr[i] > 0:
                        di_plus[i] = 100 * smooth_dm_plus[i] / atr[i]
                        di_minus[i] = 100 * smooth_dm_minus[i] / atr[i]
                        
                        if di_plus[i] + di_minus[i] > 0:
                            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
                    
                    # Calculate final ADX
                    if i >= 2 * period - 1:
                        adx[i] = np.mean(dx[i-period+1:i+1])
                
                return np.nan_to_num(adx, nan=0.0)
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return np.zeros_like(high)


if __name__ == "__main__":
    # Test with sample data
    detector = MarketRegimeDetector()
    
    # Create sample price data
    dates = pd.date_range('2025-01-01', periods=100)
    
    # Bullish trend
    bullish_data = pd.DataFrame({
        'open': np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
        'high': np.linspace(102, 153, 100) + np.random.normal(0, 2, 100),
        'low': np.linspace(98, 147, 100) + np.random.normal(0, 2, 100),
        'close': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)
    
    # Detect regime
    regime = detector.detect_regime(bullish_data)
    print(f"Detected regime: {regime['regime']} (confidence: {regime['confidence']:.2f})")
    
    # Get regime history
    history = detector.regime_history(bullish_data, 10)
    print(f"Regime history (last 10 periods):")
    for r in history:
        print(f"  {r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else r['date']}: {r['regime']} (volatility: {r['volatility']})")
