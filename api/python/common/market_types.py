#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common market types used across the trading bot system.
This centralizes definitions to avoid duplication and ensure consistency.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Enum representing different market regimes."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class MarketData:
    """Class representing market data for regime detection."""
    timestamp: datetime
    price: float
    volume: float
    volatility: float
    trend_strength: float
    momentum: float
    additional_metrics: Dict[str, float]


@dataclass
class MarketRegimeEvent:
    """Class representing a market regime change event."""
    timestamp: datetime
    old_regime: MarketRegime
    new_regime: MarketRegime
    confidence: float
    trigger_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'old_regime': self.old_regime.value,
            'new_regime': self.new_regime.value,
            'confidence': self.confidence,
            'trigger_metrics': self.trigger_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketRegimeEvent':
        """Create event from dictionary format."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            old_regime=MarketRegime(data['old_regime']),
            new_regime=MarketRegime(data['new_regime']),
            confidence=data['confidence'],
            trigger_metrics=data['trigger_metrics']
        )


class MarketRegimeDetector:
    """Class for detecting market regimes from market data."""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[MarketRegimeEvent] = []

    def detect_regime(self, market_data: List[MarketData]) -> MarketRegimeEvent:
        """
        Detect the current market regime based on market data.
        
        Args:
            market_data: List of market data points
            
        Returns:
            MarketRegimeEvent containing the regime change information
        """
        if len(market_data) < self.lookback_period:
            return MarketRegimeEvent(
                timestamp=market_data[-1].timestamp,
                old_regime=self.current_regime,
                new_regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trigger_metrics={}
            )

        # Calculate key metrics
        prices = np.array([d.price for d in market_data[-self.lookback_period:]])
        volumes = np.array([d.volume for d in market_data[-self.lookback_period:]])
        
        # Calculate trend metrics
        price_change = (prices[-1] - prices[0]) / prices[0]
        volatility = np.std(prices) / np.mean(prices)
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[:5])
        
        # Determine regime
        new_regime = self._determine_regime(price_change, volatility, volume_trend)
        confidence = self._calculate_confidence(price_change, volatility, volume_trend)
        
        # Create event
        event = MarketRegimeEvent(
            timestamp=market_data[-1].timestamp,
            old_regime=self.current_regime,
            new_regime=new_regime,
            confidence=confidence,
            trigger_metrics={
                'price_change': price_change,
                'volatility': volatility,
                'volume_trend': volume_trend
            }
        )
        
        # Update state
        self.current_regime = new_regime
        self.regime_history.append(event)
        
        return event

    def _determine_regime(self, price_change: float, volatility: float, volume_trend: float) -> MarketRegime:
        """Determine market regime based on metrics."""
        if volatility > 0.02:  # High volatility threshold
            return MarketRegime.VOLATILE
        elif price_change > 0.05:  # Bullish threshold
            return MarketRegime.BULLISH
        elif price_change < -0.05:  # Bearish threshold
            return MarketRegime.BEARISH
        else:
            return MarketRegime.SIDEWAYS

    def _calculate_confidence(self, price_change: float, volatility: float, volume_trend: float) -> float:
        """Calculate confidence in regime detection."""
        # Simple confidence calculation based on metric strength
        confidence = min(1.0, max(0.0, abs(price_change) * 10))
        return confidence


@dataclass
class MarketData:
    """Basic market data point."""
    symbol: str
    timestamp: datetime
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "close": self.close,
        }
        
        # Add optional fields if available
        if self.open is not None:
            result["open"] = self.open
        if self.high is not None:
            result["high"] = self.high
        if self.low is not None:
            result["low"] = self.low
        if self.volume is not None:
            result["volume"] = self.volume
        if self.source:
            result["source"] = self.source
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        return cls(
            symbol=data["symbol"],
            timestamp=timestamp,
            close=data["close"],
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            volume=data.get("volume"),
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )


class MarketRegimeDetector:
    """
    Base class for market regime detectors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration for the detector
        """
        self.config = config or {}
        self.current_regime = MarketRegime.UNKNOWN
        self.confidence = 0.0
        
    def detect_regime(self, data: Dict[str, Any]) -> MarketRegimeEvent:
        """
        Detect the current market regime based on the provided data.
        
        Args:
            data: Market data for regime detection
            
        Returns:
            MarketRegimeEvent with the detected regime
        """
        old_regime = self.current_regime
        metrics = self._calculate_metrics(data)
        regime, confidence = self._classify_regime(metrics)
        
        # Update current regime
        self.current_regime = regime
        self.confidence = confidence
        
        # Create event
        event = MarketRegimeEvent(
            timestamp=datetime.now(),
            old_regime=old_regime,
            new_regime=regime,
            confidence=confidence,
            trigger_metrics=metrics
        )
        
        return event
    
    def _calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for regime detection.
        
        Args:
            data: Market data for metric calculation
            
        Returns:
            Dictionary of calculated metrics
        """
        # Base implementation returns empty metrics
        return {}
    
    def _classify_regime(self, metrics: Dict[str, Any]) -> tuple[MarketRegime, float]:
        """
        Classify the market regime based on metrics.
        
        Args:
            metrics: Calculated market metrics
            
        Returns:
            Tuple of (MarketRegime, confidence)
        """
        # Base implementation returns UNKNOWN with zero confidence
        return MarketRegime.UNKNOWN, 0.0 