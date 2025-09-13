"""
Market Regime Detector

This module provides the core functionality for detecting market regimes 
(trending, range-bound, volatile, etc.) using various statistical and 
machine learning techniques.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum

# Import feature calculation modules
from trading_bot.analytics.market_regime.features import RegimeFeaturesCalculator
from trading_bot.analytics.market_regime.classifier import RegimeClassifier
from trading_bot.analytics.market_regime.adaptation import ParameterOptimizer
from trading_bot.analytics.market_regime.performance import RegimePerformanceTracker

# For tracking strategy performance across regimes
from trading_bot.analytics.market_regime.utils import exponential_smoothing

logger = logging.getLogger(__name__)

class MarketRegimeType(str, Enum):
    """Market regime classification types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    REVERSAL = "reversal"
    CHOPPY = "choppy"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"

class MarketRegimeInfo:
    """Container for market regime information."""
    
    def __init__(
        self, 
        regime_type: MarketRegimeType,
        confidence: float,
        features: Dict[str, float],
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.regime_type = regime_type
        self.confidence = confidence
        self.features = features
        self.timestamp = timestamp
        self.metadata = metadata or {}
        
    def __str__(self) -> str:
        return f"MarketRegime({self.regime_type}, confidence={self.confidence:.2f}, time={self.timestamp})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "regime_type": self.regime_type,
            "confidence": self.confidence,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketRegimeInfo':
        """Create from dictionary."""
        return cls(
            regime_type=data["regime_type"],
            confidence=data["confidence"],
            features=data["features"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )

class MarketRegimeDetector:
    """
    Detects market regimes using multiple methodologies and provides
    real-time classification of market conditions.
    
    The detector uses a combination of:
    - Technical indicators (trend, volatility, momentum)
    - Statistical measures (mean reversion, autocorrelation)
    - Machine learning classification
    
    It maintains a history of regime changes and can be used to
    optimize trading parameters for different market conditions.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_provider = None,
        event_bus = None
    ):
        """
        Initialize market regime detector.
        
        Args:
            config: Configuration dictionary
            data_provider: Data provider for market data
            event_bus: Event bus for system events
        """
        self.config = config or {}
        self.data_provider = data_provider
        self.event_bus = event_bus
        
        # Default timeframes to analyze (can be overridden in config)
        self.timeframes = self.config.get("timeframes", ["1h", "4h", "1d"])
        
        # Default symbols to track (can be overridden in config)
        self.symbols = self.config.get("symbols", [])
        
        # Lookback window size for regime detection (in bars)
        self.lookback_window = self.config.get("lookback_window", 100)
        
        # Minimum confidence threshold for regime classification
        self.confidence_threshold = self.config.get("confidence_threshold", 0.65)
        
        # Smoothing factor for regime transitions (0-1)
        self.smoothing_factor = self.config.get("smoothing_factor", 0.3)
        
        # Create components
        self.features_calculator = RegimeFeaturesCalculator(config=self.config.get("features", {}))
        self.classifier = RegimeClassifier(config=self.config.get("classifier", {}))
        self.parameter_optimizer = ParameterOptimizer(config=self.config.get("optimizer", {}))
        self.performance_tracker = RegimePerformanceTracker(config=self.config.get("tracker", {}))
        
        # Current regime state by symbol and timeframe
        self.current_regimes: Dict[str, Dict[str, MarketRegimeInfo]] = {}
        
        # Historical regimes (limited to last 1000 per symbol/timeframe)
        self.regime_history: Dict[str, Dict[str, List[MarketRegimeInfo]]] = {}
        
        # Last update time by symbol and timeframe
        self.last_updates: Dict[str, Dict[str, datetime]] = {}
        
        # Feature cache to avoid redundant calculations
        self.feature_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Strategy adaptation parameters
        self.strategy_parameters: Dict[str, Dict[MarketRegimeType, Dict[str, Any]]] = {}
        
        # Threading control
        self.monitoring_active = False
        self.monitoring_thread = None
        self._lock = threading.RLock()
        
        # Initialize
        self._initialize_state()
        
        logger.info("Market Regime Detector initialized")
    
    def _initialize_state(self) -> None:
        """Initialize internal state structures."""
        for symbol in self.symbols:
            self.current_regimes[symbol] = {}
            self.regime_history[symbol] = {}
            self.last_updates[symbol] = {}
            self.feature_cache[symbol] = {}
            
            for timeframe in self.timeframes:
                self.current_regimes[symbol][timeframe] = None
                self.regime_history[symbol][timeframe] = []
                self.last_updates[symbol][timeframe] = datetime.now() - timedelta(days=1)
                self.feature_cache[symbol][timeframe] = {}
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a new symbol for regime detection.
        
        Args:
            symbol: Symbol to add
        """
        if symbol in self.symbols:
            return
            
        with self._lock:
            self.symbols.append(symbol)
            self.current_regimes[symbol] = {}
            self.regime_history[symbol] = {}
            self.last_updates[symbol] = {}
            self.feature_cache[symbol] = {}
            
            for timeframe in self.timeframes:
                self.current_regimes[symbol][timeframe] = None
                self.regime_history[symbol][timeframe] = []
                self.last_updates[symbol][timeframe] = datetime.now() - timedelta(days=1)
                self.feature_cache[symbol][timeframe] = {}
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from regime detection.
        
        Args:
            symbol: Symbol to remove
        """
        if symbol not in self.symbols:
            return
            
        with self._lock:
            self.symbols.remove(symbol)
            if symbol in self.current_regimes:
                del self.current_regimes[symbol]
            if symbol in self.regime_history:
                del self.regime_history[symbol]
            if symbol in self.last_updates:
                del self.last_updates[symbol]
            if symbol in self.feature_cache:
                del self.feature_cache[symbol]
    
    def start_monitoring(self) -> bool:
        """
        Start background monitoring of market regimes.
        
        Returns:
            bool: Success status
        """
        if self.monitoring_active:
            logger.warning("Market regime monitoring already active")
            return True
        
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="MarketRegimeMonitor"
            )
            self.monitoring_thread.start()
            logger.info("Started market regime monitoring")
            return True
        except Exception as e:
            self.monitoring_active = False
            logger.error(f"Error starting regime monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop background monitoring of market regimes.
        
        Returns:
            bool: Success status
        """
        if not self.monitoring_active:
            return True
            
        try:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Stopped market regime monitoring")
            return True
        except Exception as e:
            logger.error(f"Error stopping regime monitoring: {str(e)}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for regime detection."""
        logger.info("Market regime monitoring loop started")
        
        update_interval = self.config.get("update_interval_seconds", 60)
        
        while self.monitoring_active:
            try:
                # Update all symbols and timeframes
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        try:
                            # Check if update is needed
                            last_update = self.last_updates[symbol][timeframe]
                            now = datetime.now()
                            
                            # Determine update frequency based on timeframe
                            update_frequency = self._get_update_frequency(timeframe)
                            
                            if (now - last_update).total_seconds() >= update_frequency:
                                self.detect_regime(symbol, timeframe)
                        except Exception as e:
                            logger.error(f"Error updating regime for {symbol} {timeframe}: {str(e)}")
                
                # Sleep until next update
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in regime monitoring loop: {str(e)}")
                time.sleep(max(update_interval, 60))  # Sleep longer on error
        
        logger.info("Market regime monitoring loop stopped")
    
    def _get_update_frequency(self, timeframe: str) -> float:
        """
        Determine update frequency in seconds based on timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            
        Returns:
            float: Update frequency in seconds
        """
        # Default frequencies based on timeframe
        if timeframe == '1m':
            return 30  # 30 seconds
        elif timeframe == '5m':
            return 60  # 1 minute
        elif timeframe == '15m':
            return 300  # 5 minutes
        elif timeframe == '30m':
            return 600  # 10 minutes
        elif timeframe == '1h':
            return 1800  # 30 minutes
        elif timeframe == '4h':
            return 3600  # 1 hour
        elif timeframe == '1d':
            return 14400  # 4 hours
        elif timeframe == '1w':
            return 86400  # 1 day
        else:
            return 3600  # 1 hour default
    
    def detect_regime(self, symbol: str, timeframe: str) -> Optional[MarketRegimeInfo]:
        """
        Detect the current market regime for a symbol and timeframe.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            
        Returns:
            MarketRegimeInfo or None if detection fails
        """
        try:
            # Update last update time
            self.last_updates[symbol][timeframe] = datetime.now()
            
            # Get price data
            price_data = self._get_price_data(symbol, timeframe)
            if price_data is None or price_data.empty:
                logger.error(f"No price data available for {symbol} {timeframe}")
                return None
            
            # Calculate features
            features = self.features_calculator.calculate(symbol, timeframe, price_data)
            
            # Cache features
            self.feature_cache[symbol][timeframe] = features
            
            # Classify regime
            regime_type, confidence = self.classifier.classify(features, symbol, timeframe)
            
            # Create regime info
            regime_info = MarketRegimeInfo(
                regime_type=regime_type,
                confidence=confidence,
                features=features,
                timestamp=datetime.now()
            )
            
            # Update current regime with smoothing
            self._update_current_regime(symbol, timeframe, regime_info)
            
            # Track regime history
            self._update_regime_history(symbol, timeframe, regime_info)
            
            # Emit event if confidence is high enough
            if confidence >= self.confidence_threshold:
                self._emit_regime_event(symbol, timeframe, regime_info)
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol} {timeframe}: {str(e)}")
            return None
    
    def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get price data for regime detection.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            
        Returns:
            DataFrame with price data or None if unavailable
        """
        try:
            # Check if we have a data provider
            if self.data_provider is None:
                logger.error("No data provider configured")
                return None
            
            # Get data using provider
            if hasattr(self.data_provider, 'get_historical_data'):
                data = self.data_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=self.lookback_window + 50  # Add buffer for calculations
                )
                return data
            
            # Fallback to other methods if available
            if hasattr(self.data_provider, 'get_ohlcv'):
                data = self.data_provider.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=self.lookback_window + 50
                )
                return data
            
            logger.error("Data provider missing required methods")
            return None
            
        except Exception as e:
            logger.error(f"Error getting price data: {str(e)}")
            return None
    
    def _update_current_regime(self, symbol: str, timeframe: str, regime_info: MarketRegimeInfo) -> None:
        """
        Update current regime with smoothing for transitions.
        
        Args:
            symbol: Symbol being updated
            timeframe: Timeframe being updated
            regime_info: New regime information
        """
        with self._lock:
            current = self.current_regimes[symbol].get(timeframe)
            
            # If no current regime, just set it
            if current is None:
                self.current_regimes[symbol][timeframe] = regime_info
                return
            
            # If same regime type, update with exponential smoothing
            if current.regime_type == regime_info.regime_type:
                # Smooth confidence
                smoothed_confidence = exponential_smoothing(
                    current.confidence, 
                    regime_info.confidence, 
                    self.smoothing_factor
                )
                
                # Smooth features
                smoothed_features = {}
                for feature, value in regime_info.features.items():
                    if feature in current.features:
                        smoothed_features[feature] = exponential_smoothing(
                            current.features[feature],
                            value,
                            self.smoothing_factor
                        )
                    else:
                        smoothed_features[feature] = value
                
                # Create updated regime info
                updated_info = MarketRegimeInfo(
                    regime_type=regime_info.regime_type,
                    confidence=smoothed_confidence,
                    features=smoothed_features,
                    timestamp=regime_info.timestamp,
                    metadata=regime_info.metadata
                )
                
                self.current_regimes[symbol][timeframe] = updated_info
            else:
                # Different regime type - check confidence
                # Only switch if new regime has higher confidence
                if regime_info.confidence > current.confidence:
                    self.current_regimes[symbol][timeframe] = regime_info
    
    def _update_regime_history(self, symbol: str, timeframe: str, regime_info: MarketRegimeInfo) -> None:
        """
        Update regime history.
        
        Args:
            symbol: Symbol being updated
            timeframe: Timeframe being updated
            regime_info: New regime information
        """
        with self._lock:
            # Check if this is a regime change
            current = self.current_regimes[symbol].get(timeframe)
            is_regime_change = (current is None or 
                              current.regime_type != regime_info.regime_type or
                              abs(current.confidence - regime_info.confidence) > 0.2)
            
            # Add to history if it's a regime change or periodically
            history = self.regime_history[symbol][timeframe]
            if is_regime_change or len(history) == 0 or \
               (regime_info.timestamp - history[-1].timestamp).total_seconds() > 3600:
                
                # Add to history
                history.append(regime_info)
                
                # Limit history size
                max_history = self.config.get("max_regime_history", 1000)
                if len(history) > max_history:
                    self.regime_history[symbol][timeframe] = history[-max_history:]
    
    def _emit_regime_event(self, symbol: str, timeframe: str, regime_info: MarketRegimeInfo) -> None:
        """
        Emit market regime event if conditions are met.
        
        Args:
            symbol: Symbol for the event
            timeframe: Timeframe for the event
            regime_info: Regime information
        """
        # Check if we have an event bus
        if self.event_bus is None or not hasattr(self.event_bus, 'emit'):
            return
            
        # Check for regime change
        current = self.current_regimes[symbol].get(timeframe)
        
        # Emit event
        event_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'regime': regime_info.regime_type,
            'confidence': regime_info.confidence,
            'timestamp': regime_info.timestamp.isoformat(),
            'features': regime_info.features,
            'is_change': current is None or current.regime_type != regime_info.regime_type
        }
        
        try:
            self.event_bus.emit('market_regime_update', event_data)
            
            # Emit regime change event if applicable
            if event_data['is_change'] and current is not None:
                change_data = event_data.copy()
                change_data['prev_regime'] = current.regime_type
                change_data['prev_confidence'] = current.confidence
                self.event_bus.emit('market_regime_change', change_data)
                
                # Log regime change
                logger.info(f"Market regime change: {symbol} {timeframe} "
                          f"{current.regime_type} -> {regime_info.regime_type} "
                          f"(confidence: {regime_info.confidence:.2f})")
        except Exception as e:
            logger.error(f"Error emitting regime event: {str(e)}")
    
    def get_current_regime(self, symbol: str, timeframe: str) -> Optional[MarketRegimeInfo]:
        """
        Get current market regime for a symbol and timeframe.
        
        Args:
            symbol: Symbol to get regime for
            timeframe: Timeframe to get regime for
            
        Returns:
            MarketRegimeInfo or None if not available
        """
        if symbol not in self.current_regimes or timeframe not in self.current_regimes[symbol]:
            return None
            
        return self.current_regimes[symbol][timeframe]
    
    def get_regime_history(self, symbol: str, timeframe: str, 
                         limit: Optional[int] = None) -> List[MarketRegimeInfo]:
        """
        Get historical regimes for a symbol and timeframe.
        
        Args:
            symbol: Symbol to get history for
            timeframe: Timeframe to get history for
            limit: Optional limit for number of entries
            
        Returns:
            List of MarketRegimeInfo objects
        """
        if symbol not in self.regime_history or timeframe not in self.regime_history[symbol]:
            return []
            
        history = self.regime_history[symbol][timeframe]
        
        if limit is not None:
            return history[-limit:]
        return history
    
    def get_regime_features(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """
        Get current regime features for a symbol and timeframe.
        
        Args:
            symbol: Symbol to get features for
            timeframe: Timeframe to get features for
            
        Returns:
            Dict of feature values
        """
        if symbol not in self.feature_cache or timeframe not in self.feature_cache[symbol]:
            return {}
            
        return self.feature_cache[symbol][timeframe]
    
    def get_optimal_parameters(self, strategy_id: str, symbol: str, 
                             timeframe: str) -> Dict[str, Any]:
        """
        Get optimal strategy parameters for current market regime.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Symbol
            timeframe: Timeframe
            
        Returns:
            Dict of optimal parameters
        """
        # Get current regime
        regime_info = self.get_current_regime(symbol, timeframe)
        if regime_info is None:
            logger.warning(f"No regime info available for {symbol} {timeframe}")
            return {}
            
        # Get optimized parameters from parameter optimizer
        return self.parameter_optimizer.get_optimal_parameters(
            strategy_id=strategy_id,
            regime_type=regime_info.regime_type,
            symbol=symbol,
            timeframe=timeframe,
            confidence=regime_info.confidence
        )
    
    def update_strategy_performance(self, strategy_id: str, performance_metrics: Dict[str, Any],
                                  symbol: str, timeframe: str) -> None:
        """
        Update strategy performance for the current regime.
        
        Args:
            strategy_id: Strategy identifier
            performance_metrics: Performance metrics
            symbol: Symbol
            timeframe: Timeframe
        """
        # Get current regime
        regime_info = self.get_current_regime(symbol, timeframe)
        if regime_info is None:
            logger.warning(f"No regime info available for {symbol} {timeframe}")
            return
            
        # Update performance tracker
        self.performance_tracker.update_performance(
            strategy_id=strategy_id,
            regime_type=regime_info.regime_type,
            performance_metrics=performance_metrics,
            symbol=symbol,
            timeframe=timeframe
        )
    
    def get_strategy_performance_by_regime(self, strategy_id: str) -> Dict[MarketRegimeType, Dict[str, Any]]:
        """
        Get strategy performance metrics by regime type.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict mapping regime types to performance metrics
        """
        return self.performance_tracker.get_performance_by_regime(strategy_id)
    
    def adapt_strategy(self, strategy_id: str, strategy_instance: Any, 
                     symbol: str, timeframe: str) -> bool:
        """
        Adapt strategy parameters to current market regime.
        
        Args:
            strategy_id: Strategy identifier
            strategy_instance: Strategy instance to adapt
            symbol: Symbol
            timeframe: Timeframe
            
        Returns:
            bool: Success status
        """
        try:
            # Get optimal parameters
            params = self.get_optimal_parameters(strategy_id, symbol, timeframe)
            if not params:
                return False
                
            # Apply parameters to strategy
            if hasattr(strategy_instance, 'update_parameters'):
                strategy_instance.update_parameters(params)
                
                # Get current regime
                regime_info = self.get_current_regime(symbol, timeframe)
                if regime_info:
                    logger.info(f"Adapted strategy {strategy_id} for {regime_info.regime_type} "
                              f"regime (confidence: {regime_info.confidence:.2f})")
                return True
            else:
                logger.warning(f"Strategy {strategy_id} does not support parameter updates")
                return False
                
        except Exception as e:
            logger.error(f"Error adapting strategy {strategy_id}: {str(e)}")
            return False
    
    def calculate_position_size_adjustment(self, symbol: str, timeframe: str) -> float:
        """
        Calculate position size adjustment factor based on current regime.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            
        Returns:
            float: Adjustment factor (0.0-2.0, where 1.0 is neutral)
        """
        # Get current regime
        regime_info = self.get_current_regime(symbol, timeframe)
        if regime_info is None:
            return 1.0  # Neutral if no regime info
            
        # Default adjustments by regime type
        regime_adjustments = {
            MarketRegimeType.TRENDING_UP: 1.2,
            MarketRegimeType.TRENDING_DOWN: 1.2,
            MarketRegimeType.RANGE_BOUND: 0.8,
            MarketRegimeType.HIGH_VOLATILITY: 0.6,
            MarketRegimeType.LOW_VOLATILITY: 1.3,
            MarketRegimeType.BREAKOUT: 1.3,
            MarketRegimeType.BREAKDOWN: 1.3,
            MarketRegimeType.REVERSAL: 0.7,
            MarketRegimeType.CHOPPY: 0.5,
            MarketRegimeType.NORMAL: 1.0,
            MarketRegimeType.UNCERTAIN: 0.7
        }
        
        # Get base adjustment for regime type
        base_adjustment = regime_adjustments.get(regime_info.regime_type, 1.0)
        
        # Scale by confidence
        confidence_scaling = regime_info.confidence / self.confidence_threshold
        confidence_scaling = min(max(confidence_scaling, 0.5), 1.5)
        
        # Calculate final adjustment
        adjustment = base_adjustment * confidence_scaling
        
        # Ensure within reasonable bounds
        adjustment = min(max(adjustment, 0.2), 2.0)
        
        return adjustment
