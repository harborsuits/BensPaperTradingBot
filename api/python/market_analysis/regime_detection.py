"""
Market Regime Detection Module

This module provides models and algorithms for detecting market regimes
based on technical indicators, price patterns, and volatility metrics.
The regime classification helps guide trading strategy selection.
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from trading_bot.data_sources.market_data_adapter import (
    OHLCV, MarketIndicator, TimeFrame
)

logger = logging.getLogger("market_analysis.regime")

class MarketRegimeType(str, Enum):
    """Types of market regimes"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"

class RegimeMethod(str, Enum):
    """Methods for market regime detection"""
    TREND_BASED = "trend_based"
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM_BASED = "momentum_based"
    ML_BASED = "ml_based"
    MULTI_FACTOR = "multi_factor"

class RegimeFeature(BaseModel):
    """Feature used in regime detection"""
    name: str
    value: float
    weight: float
    contribution: float  # The weighted contribution to the regime score
    
    class Config:
        allow_population_by_field_name = True

class MarketRegimeResult(BaseModel):
    """Result of market regime detection"""
    primary_regime: MarketRegimeType
    confidence: float
    secondary_regime: Optional[MarketRegimeType] = None
    secondary_confidence: Optional[float] = None
    since: Optional[str] = None  # ISO date when regime began
    features: Dict[str, float] = {}  # Feature values that determined regime
    feature_details: Optional[List[RegimeFeature]] = None
    method: RegimeMethod = RegimeMethod.MULTI_FACTOR
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        allow_population_by_field_name = True

class RegimeDetector:
    """Base class for market regime detection algorithms"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("regime_detector")
    
    def detect(
        self,
        ohlcv_data: List[OHLCV],
        indicators: Dict[str, List[MarketIndicator]] = None,
        additional_data: Dict[str, Any] = None
    ) -> MarketRegimeResult:
        """
        Detect the current market regime from price and indicator data
        
        Args:
            ohlcv_data: List of price candles (most recent first)
            indicators: Dictionary of technical indicators
            additional_data: Any additional data needed for detection
            
        Returns:
            Market regime detection result
        """
        raise NotImplementedError("Subclasses must implement detect")
    
    def _calculate_basic_trend_score(self, ohlcv_data: List[OHLCV]) -> float:
        """Calculate a basic trend score from -1 (bearish) to 1 (bullish)"""
        if not ohlcv_data or len(ohlcv_data) < 5:
            return 0.0
        
        # Get close prices, most recent first
        closes = [candle.close for candle in ohlcv_data]
        
        # Calculate percentage changes
        changes = []
        for i in range(len(closes) - 1):
            pct_change = (closes[i] - closes[i+1]) / closes[i+1]
            changes.append(pct_change)
        
        # Calculate short-term and medium-term trends
        short_term = sum(changes[:5]) / 5 if len(changes) >= 5 else 0
        medium_term = sum(changes[:20]) / 20 if len(changes) >= 20 else short_term
        
        # Calculate trend persistence (consistency of direction)
        trend_direction = np.sign(changes)
        persistence = np.abs(np.mean(trend_direction[:10])) if len(changes) >= 10 else 0.5
        
        # Calculate normalized trend score between -1 and 1
        # Weight recent price action more heavily
        trend_score = 0.6 * np.sign(short_term) * min(1.0, abs(short_term) * 100) + \
                     0.4 * np.sign(medium_term) * min(1.0, abs(medium_term) * 100)
        
        # Adjust by persistence factor
        trend_score *= (0.5 + 0.5 * persistence)
        
        return max(-1.0, min(1.0, trend_score))
    
    def _calculate_volatility_score(self, ohlcv_data: List[OHLCV]) -> float:
        """Calculate volatility score from 0 (low) to 1 (high)"""
        if not ohlcv_data or len(ohlcv_data) < 5:
            return 0.5
        
        # Get close prices, most recent first
        closes = [candle.close for candle in ohlcv_data]
        
        # Calculate daily returns
        returns = []
        for i in range(len(closes) - 1):
            ret = (closes[i] - closes[i+1]) / closes[i+1]
            returns.append(ret)
        
        # Calculate volatility as the standard deviation of returns
        # Use different windows to capture short and long-term volatility
        short_vol = np.std(returns[:10]) if len(returns) >= 10 else np.std(returns)
        long_vol = np.std(returns[:30]) if len(returns) >= 30 else short_vol
        
        # Calculate ranges relative to price
        ranges = []
        for candle in ohlcv_data[:20]:
            ranges.append((candle.high - candle.low) / candle.close)
        
        # Average true range component
        avg_range = np.mean(ranges)
        
        # Combine volatility metrics with weights
        vol_score = 0.4 * (short_vol * 100) + 0.3 * (long_vol * 100) + 0.3 * (avg_range * 100)
        
        # Normalize to 0-1 scale
        # Typical daily volatility ranges from 0.5% to 4%
        norm_vol = min(1.0, vol_score / 4.0)
        
        return norm_vol


class MultifactorRegimeDetector(RegimeDetector):
    """
    Detects market regimes using multiple factors including trend, momentum, and volatility
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.lookback_days = config.get("lookback_days", 60) if config else 60
        self.trend_weight = config.get("trend_weight", 0.4) if config else 0.4
        self.momentum_weight = config.get("momentum_weight", 0.3) if config else 0.3
        self.volatility_weight = config.get("volatility_weight", 0.3) if config else 0.3
        self.previous_regimes = []  # Tracks history of regime changes
    
    def detect(
        self,
        ohlcv_data: List[OHLCV],
        indicators: Dict[str, List[MarketIndicator]] = None,
        additional_data: Dict[str, Any] = None
    ) -> MarketRegimeResult:
        """
        Detect market regime using multiple factors
        """
        if not ohlcv_data or len(ohlcv_data) < 10:
            self.logger.warning("Insufficient data for regime detection")
            return MarketRegimeResult(
                primary_regime=MarketRegimeType.SIDEWAYS,
                confidence=0.5,
                method=RegimeMethod.MULTI_FACTOR,
                features={}
            )
        
        # 1. Calculate trend score (-1 to 1)
        trend_score = self._calculate_basic_trend_score(ohlcv_data)
        
        # 2. Calculate momentum score (using indicators if available)
        momentum_score = self._calculate_momentum_score(ohlcv_data, indicators)
        
        # 3. Calculate volatility score (0 to 1)
        volatility_score = self._calculate_volatility_score(ohlcv_data)
        
        # 4. Calculate correlation score (market breadth/health)
        correlation_score = self._calculate_correlation_score(additional_data)
        
        # 5. Calculate volume pattern score
        volume_score = self._calculate_volume_pattern_score(ohlcv_data)
        
        # 6. Create feature details for transparency
        features = {
            "trend_strength": abs(trend_score),
            "trend_direction": trend_score,
            "momentum": momentum_score,
            "volatility": volatility_score,
            "correlation": correlation_score,
            "volume_pattern": volume_score
        }
        
        feature_details = [
            RegimeFeature(
                name="trend_strength",
                value=abs(trend_score),
                weight=self.trend_weight,
                contribution=abs(trend_score) * self.trend_weight
            ),
            RegimeFeature(
                name="trend_direction",
                value=trend_score,
                weight=0.0,  # Informational only, not used in final score
                contribution=0.0
            ),
            RegimeFeature(
                name="momentum",
                value=momentum_score,
                weight=self.momentum_weight,
                contribution=momentum_score * self.momentum_weight
            ),
            RegimeFeature(
                name="volatility",
                value=volatility_score,
                weight=self.volatility_weight,
                contribution=volatility_score * self.volatility_weight
            ),
            RegimeFeature(
                name="correlation",
                value=correlation_score,
                weight=0.0,  # Used for secondary regime classification
                contribution=0.0
            ),
            RegimeFeature(
                name="volume_pattern",
                value=volume_score,
                weight=0.0,  # Used for secondary regime classification
                contribution=0.0
            )
        ]
        
        # 7. Determine primary regime based on combined factors
        # High volatility suggests volatile regime
        if volatility_score > 0.7:
            primary_regime = MarketRegimeType.VOLATILE
            confidence = volatility_score
        # Strong trend suggests trend-following regime
        elif abs(trend_score) > 0.6 and momentum_score > 0.5:
            if trend_score > 0:
                primary_regime = MarketRegimeType.BULLISH
            else:
                primary_regime = MarketRegimeType.BEARISH
            confidence = (abs(trend_score) * self.trend_weight + 
                         momentum_score * self.momentum_weight) / (self.trend_weight + self.momentum_weight)
        # Lack of trend with moderate volatility suggests mean-reverting
        elif abs(trend_score) < 0.3 and 0.3 < volatility_score < 0.7:
            primary_regime = MarketRegimeType.MEAN_REVERTING
            confidence = (1 - abs(trend_score)) * 0.7 + volatility_score * 0.3
        # Sideways market
        elif abs(trend_score) < 0.2 and volatility_score < 0.3:
            primary_regime = MarketRegimeType.SIDEWAYS
            confidence = (1 - abs(trend_score)) * 0.7 + (1 - volatility_score) * 0.3
        # Trending market
        elif abs(trend_score) > 0.4:
            primary_regime = MarketRegimeType.TRENDING
            confidence = abs(trend_score)
        # Default to trending for moderate conditions
        else:
            primary_regime = MarketRegimeType.TRENDING if trend_score > 0 else MarketRegimeType.MEAN_REVERTING
            confidence = 0.5 + abs(trend_score) * 0.2 + momentum_score * 0.2
        
        # 8. Determine secondary regime (often risk-on/risk-off)
        if correlation_score > 0.7:
            secondary_regime = MarketRegimeType.RISK_ON if trend_score > 0 else MarketRegimeType.RISK_OFF
            secondary_confidence = correlation_score
        else:
            secondary_regime = None
            secondary_confidence = None
        
        # 9. Calculate regime start date (when did the current regime begin)
        since = self._determine_regime_start(primary_regime, ohlcv_data)
        
        # Create the result
        result = MarketRegimeResult(
            primary_regime=primary_regime,
            confidence=min(0.95, confidence),  # Cap confidence at 95%
            secondary_regime=secondary_regime,
            secondary_confidence=secondary_confidence,
            since=since,
            features=features,
            feature_details=feature_details,
            method=RegimeMethod.MULTI_FACTOR
        )
        
        # Store this result in history
        self._update_regime_history(result)
        
        return result
    
    def _calculate_momentum_score(
        self, 
        ohlcv_data: List[OHLCV],
        indicators: Dict[str, List[MarketIndicator]] = None
    ) -> float:
        """Calculate momentum score from 0 (weak) to 1 (strong)"""
        if not ohlcv_data or len(ohlcv_data) < 10:
            return 0.5
        
        # Use indicators if available
        rsi_value = None
        macd_value = None
        macd_hist_value = None
        
        if indicators:
            # Get RSI value if available
            if 'rsi' in indicators and indicators['rsi']:
                rsi_value = indicators['rsi'][0].value if indicators['rsi'] else None
            
            # Get MACD values if available
            if 'macd_line' in indicators and indicators['macd_line']:
                macd_value = indicators['macd_line'][0].value if indicators['macd_line'] else None
                
            if 'macd_hist' in indicators and indicators['macd_hist']:
                macd_hist_value = indicators['macd_hist'][0].value if indicators['macd_hist'] else None
        
        # Calculate momentum from price data
        closes = [candle.close for candle in ohlcv_data]
        
        # Rate of change
        roc_5 = (closes[0] - closes[5]) / closes[5] if len(closes) > 5 else 0
        roc_20 = (closes[0] - closes[20]) / closes[20] if len(closes) > 20 else roc_5
        
        # Moving average alignment
        ma_10 = np.mean(closes[:10]) if len(closes) >= 10 else np.mean(closes)
        ma_20 = np.mean(closes[:20]) if len(closes) >= 20 else ma_10
        ma_50 = np.mean(closes[:50]) if len(closes) >= 50 else ma_20
        
        ma_alignment = 0
        if closes[0] > ma_10 and ma_10 > ma_20:
            ma_alignment += 0.5
        if ma_20 > ma_50:
            ma_alignment += 0.5
        
        # Combine factors, prioritizing indicators if available
        momentum_score = 0
        count = 0
        
        # RSI: rescale from 0-100 to 0-1 and adjust to prioritize extremes
        if rsi_value is not None:
            rsi_contribution = 0
            if rsi_value > 70:
                rsi_contribution = 0.7 + (rsi_value - 70) / 100
            elif rsi_value > 50:
                rsi_contribution = 0.5 + (rsi_value - 50) / 100
            elif rsi_value < 30:
                rsi_contribution = 0.3 - (30 - rsi_value) / 100
            else:
                rsi_contribution = rsi_value / 100
            momentum_score += rsi_contribution
            count += 1
        
        # MACD: normalize and use both line and histogram
        if macd_value is not None and macd_hist_value is not None:
            # Normalize MACD relative to price
            norm_factor = closes[0] * 0.01  # 1% of price
            macd_norm = min(1.0, max(0.0, (macd_value / norm_factor + 1) / 2))
            hist_norm = min(1.0, max(0.0, (macd_hist_value / norm_factor + 1) / 2))
            
            momentum_score += (macd_norm * 0.6 + hist_norm * 0.4)
            count += 1
        
        # Price-based momentum
        price_momentum = (
            0.4 * min(1.0, max(0.0, (roc_5 * 100 + 3) / 6)) +
            0.3 * min(1.0, max(0.0, (roc_20 * 100 + 10) / 20)) +
            0.3 * ma_alignment
        )
        momentum_score += price_momentum
        count += 1
        
        # Average all components
        momentum_score = momentum_score / count if count > 0 else 0.5
        
        return momentum_score
    
    def _calculate_correlation_score(self, additional_data: Dict[str, Any] = None) -> float:
        """
        Calculate market correlation/breadth score
        Higher values indicate stronger correlation (risk-on/risk-off behavior)
        """
        # Default to moderate correlation if no additional data
        if not additional_data:
            return 0.5
        
        # Extract correlation data if provided
        if 'sector_correlations' in additional_data:
            correlations = additional_data['sector_correlations']
            if isinstance(correlations, list) and len(correlations) > 0:
                # Average of absolute correlations
                avg_corr = np.mean([abs(c) for c in correlations])
                return avg_corr
        
        # Extract breadth data if provided
        if 'market_breadth' in additional_data:
            breadth = additional_data['market_breadth']
            if 'advancers' in breadth and 'decliners' in breadth:
                total = breadth['advancers'] + breadth['decliners']
                if total > 0:
                    # Calculate how one-sided the market is
                    breadth_score = max(breadth['advancers'], breadth['decliners']) / total
                    return breadth_score
        
        return 0.5
    
    def _calculate_volume_pattern_score(self, ohlcv_data: List[OHLCV]) -> float:
        """
        Calculate a score based on volume patterns
        Higher values indicate higher volume supporting price direction
        """
        if not ohlcv_data or len(ohlcv_data) < 5:
            return 0.5
        
        # Extract price and volume data
        closes = [candle.close for candle in ohlcv_data]
        volumes = [candle.volume for candle in ohlcv_data]
        
        # Calculate price changes
        changes = []
        for i in range(len(closes) - 1):
            changes.append(closes[i] - closes[i+1])
        
        # Calculate volume-weighted price change
        vol_price_score = 0
        total_vol = sum(volumes[:5])
        
        if total_vol > 0:
            for i in range(min(5, len(changes))):
                # Weight by volume
                vol_weight = volumes[i] / total_vol
                # Contribute to score if volume confirms direction
                vol_price_score += vol_weight * np.sign(changes[i])
        
        # Normalize to 0-1
        vol_price_score = (vol_price_score + 1) / 2
        
        # Check for volume trend
        vol_trend = 0
        if len(volumes) >= 10:
            avg_recent = np.mean(volumes[:5])
            avg_older = np.mean(volumes[5:10])
            vol_trend = avg_recent / avg_older if avg_older > 0 else 1.0
            # Normalize to 0-1 (1.0 = 100% increase, 0.5 = no change, 0.0 = complete drop)
            vol_trend = min(1.0, vol_trend / 2)
        else:
            vol_trend = 0.5
        
        # Combine scores
        final_score = 0.7 * vol_price_score + 0.3 * vol_trend
        
        return final_score
    
    def _determine_regime_start(
        self, 
        current_regime: MarketRegimeType, 
        ohlcv_data: List[OHLCV]
    ) -> str:
        """
        Determine when the current regime began
        This is a simplified approach - in production would use more sophisticated change point detection
        """
        # If we have regime history, use it
        if self.previous_regimes:
            for i, prev_regime in enumerate(self.previous_regimes):
                if prev_regime.primary_regime != current_regime:
                    if i > 0:
                        return self.previous_regimes[i-1].timestamp
                    break
        
        # Fallback: estimate based on price data
        if ohlcv_data and len(ohlcv_data) > 5:
            # Very simple: just use 30 days ago as an approximation
            # In a real system, this would be more sophisticated
            return (datetime.now() - timedelta(days=30)).isoformat()
        
        # Last resort
        return datetime.now().isoformat()
    
    def _update_regime_history(self, result: MarketRegimeResult):
        """Update the history of regime transitions"""
        self.previous_regimes.insert(0, result)
        # Keep only last 10 regime snapshots to avoid memory growth
        if len(self.previous_regimes) > 10:
            self.previous_regimes = self.previous_regimes[:10]


# Factory function for creating regime detectors
def create_regime_detector(method: str, config: Dict[str, Any] = None) -> RegimeDetector:
    """
    Create an appropriate regime detector based on the method
    
    Args:
        method: The regime detection method to use
        config: Configuration parameters for the detector
        
    Returns:
        An instance of the appropriate RegimeDetector subclass
    """
    if method == RegimeMethod.MULTI_FACTOR:
        return MultifactorRegimeDetector(config)
    elif method == RegimeMethod.TREND_BASED:
        # Could implement specific trend-based detector
        return MultifactorRegimeDetector({
            **(config or {}),
            "trend_weight": 0.7,
            "momentum_weight": 0.2,
            "volatility_weight": 0.1
        })
    elif method == RegimeMethod.VOLATILITY_BASED:
        # Could implement specific volatility-based detector
        return MultifactorRegimeDetector({
            **(config or {}),
            "trend_weight": 0.2,
            "momentum_weight": 0.1,
            "volatility_weight": 0.7
        })
    elif method == RegimeMethod.MOMENTUM_BASED:
        # Could implement specific momentum-based detector
        return MultifactorRegimeDetector({
            **(config or {}),
            "trend_weight": 0.2,
            "momentum_weight": 0.7,
            "volatility_weight": 0.1
        })
    elif method == RegimeMethod.ML_BASED:
        # Would implement ML-based detector in production
        logger.warning("ML-based regime detection requested but not implemented, using multi-factor instead")
        return MultifactorRegimeDetector(config)
    else:
        logger.warning(f"Unknown regime detection method: {method}, using multi-factor")
        return MultifactorRegimeDetector(config)
