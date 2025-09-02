"""
Market Regime Detector

This module provides advanced market regime detection capabilities using
technical indicators, volatility patterns, and machine learning techniques.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enum for different market regimes"""
    TRENDING_BULLISH = auto()
    TRENDING_BEARISH = auto()
    RANGE_BOUND = auto()
    VOLATILE_BULLISH = auto()
    VOLATILE_BEARISH = auto()
    HIGH_VOLATILITY = auto()
    CORRECTIVE = auto()
    EARLY_TREND = auto()
    LATE_TREND = auto()
    BREAKOUT = auto()
    REVERSAL = auto()
    UNDEFINED = auto()

class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple techniques including:
    - Technical indicators pattern recognition
    - Volatility-based analysis
    - Statistical pattern recognition
    - Optional ML-based classification
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        use_ml_models: bool = False,
        lookback_periods: Dict[str, int] = None
    ):
        """
        Initialize the MarketRegimeDetector with configuration parameters
        
        Args:
            config: Configuration dictionary
            use_ml_models: Whether to use machine learning models
            lookback_periods: Dictionary of lookback periods for different timeframes
        """
        self.config = config or {}
        self.use_ml_models = use_ml_models
        self.lookback_periods = lookback_periods or {
            "short_term": 10,
            "medium_term": 50,
            "long_term": 200
        }
        
        # Cache for regime calculations
        self.regime_cache = {}
        self.cache_expiry = self.config.get("regime_cache_minutes", 30)
        
        # Store results of last detection
        self.last_detection = None
        
        logger.info(f"MarketRegimeDetector initialized with ML models: {use_ml_models}")
        
    def detect_regime(self, market_data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Detect the current market regime using multiple indicators
        
        Args:
            market_data: DataFrame with OHLCV data
            symbol: Optional symbol name for logging
            
        Returns:
            Dictionary with regime details including:
            - primary_regime: Main regime classification
            - confidence: Confidence level (0-1)
            - sub_regimes: List of secondary regime characteristics
            - indicators: Dictionary of supporting indicator values
        """
        symbol_label = f" for {symbol}" if symbol else ""
        logger.info(f"Detecting market regime{symbol_label} using {len(market_data)} data points")
        
        if market_data.empty or len(market_data) < 20:
            logger.warning(f"Insufficient data to detect regime{symbol_label}")
            return self._create_undefined_regime()
            
        try:
            # Check cache first
            cache_key = f"{symbol}_regime" if symbol else "global_regime"
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(market_data)
            
            # Perform rule-based regime detection
            regime_results = self._rule_based_regime_detection(indicators, market_data)
            
            # Enhance with ML if enabled
            if self.use_ml_models:
                ml_results = self._ml_based_regime_detection(indicators, market_data)
                # Blend rule-based and ML results
                regime_results = self._blend_detection_results(regime_results, ml_results)
                
            # Add timestamp and cache
            regime_results['timestamp'] = datetime.now().isoformat()
            self._update_cache(cache_key, regime_results)
            
            # Store last detection result
            self.last_detection = regime_results
            
            return regime_results
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return self._create_undefined_regime()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators from market data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of indicator values
        """
        indicators = {}
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column {col}")
                df[col] = 0
        
        # Trend indicators
        
        # 1. Moving Averages
        for period in [10, 20, 50, 200]:
            if len(df) >= period:
                indicators[f'sma_{period}'] = df['close'].rolling(window=period).mean().iloc[-1]
                indicators[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
        
        # 2. MA Crossovers and relationships
        if 'sma_20' in indicators and 'sma_50' in indicators:
            indicators['sma_20_50_diff'] = indicators['sma_20'] / indicators['sma_50'] - 1
            
        if 'sma_50' in indicators and 'sma_200' in indicators:
            indicators['sma_50_200_diff'] = indicators['sma_50'] / indicators['sma_200'] - 1
        
        # 3. Relative position to MAs
        current_price = df['close'].iloc[-1]
        for ma in ['sma_20', 'sma_50', 'sma_200']:
            if ma in indicators:
                indicators[f'{ma}_rel_pos'] = current_price / indicators[ma] - 1
        
        # Volatility indicators
        
        # 1. ATR (Average True Range)
        if len(df) >= 14:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr_14'] = tr.rolling(window=14).mean().iloc[-1]
            indicators['atr_14_pct'] = indicators['atr_14'] / current_price
            
        # 2. Bollinger Bands
        if len(df) >= 20:
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_width'] = (2 * std_20 / sma_20).iloc[-1]  # Width as percentage of price
            indicators['bb_pos'] = (df['close'].iloc[-1] - sma_20.iloc[-1]) / (std_20.iloc[-1] * 2)  # -1 to 1 position within bands
        
        # Momentum indicators
        
        # 1. RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi_14'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 2. MACD
        if len(df) >= 26:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_hist'] = (macd - signal).iloc[-1]
        
        # 3. Rate of change
        for period in [5, 10, 20]:
            if len(df) >= period:
                indicators[f'roc_{period}'] = (df['close'].iloc[-1] / df['close'].iloc[-period-1] - 1) * 100
        
        # Volume indicators
        if 'volume' in df.columns and not (df['volume'] == 0).all():
            # 1. Volume moving average
            if len(df) >= 20:
                indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean().iloc[-1]
                indicators['rel_volume'] = df['volume'].iloc[-1] / indicators['volume_sma_20']
            
            # 2. Volume trend
            if len(df) >= 10:
                vol_change = df['volume'].pct_change(periods=10).iloc[-1] * 100
                indicators['volume_trend_10'] = vol_change
        
        # Pattern detection
        if len(df) >= 5:
            # Recent consecutive moves
            closes = df['close'].iloc[-5:]
            ups = sum(1 for i in range(1, len(closes)) if closes.iloc[i] > closes.iloc[i-1])
            indicators['consecutive_direction'] = ups / (len(closes) - 1) * 2 - 1  # -1 to 1 scale
        
        return indicators
    
    def _rule_based_regime_detection(self, indicators: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regime using rule-based approach
        
        Args:
            indicators: Dictionary of technical indicators
            data: Original OHLCV DataFrame
            
        Returns:
            Dictionary with regime classification
        """
        # Set up default response
        result = {
            'primary_regime': MarketRegime.UNDEFINED.name,
            'confidence': 0.5,
            'sub_regimes': [],
            'indicators': {}
        }
        
        # Add key indicators used for decision making
        result['indicators'] = {
            'ma_alignment': 0,
            'volatility': 0,
            'momentum': 0,
            'trend_strength': 0,
            'volume_confirmation': 0
        }
        
        # Exit if we don't have enough indicator data
        if not indicators or len(indicators) < 5:
            return result
            
        # Trend detection based on MA alignment
        ma_alignment = 0
        if 'sma_20_50_diff' in indicators and 'sma_50_200_diff' in indicators:
            short_aligned = indicators['sma_20_50_diff'] > 0
            long_aligned = indicators['sma_50_200_diff'] > 0
            
            if short_aligned and long_aligned:
                ma_alignment = 1  # Strong bullish alignment
            elif not short_aligned and not long_aligned:
                ma_alignment = -1  # Strong bearish alignment
            elif short_aligned and not long_aligned:
                ma_alignment = 0.5  # Potential early bullish trend
            else:
                ma_alignment = -0.5  # Potential early bearish trend
                
        result['indicators']['ma_alignment'] = ma_alignment
        
        # Volatility assessment
        volatility = 0
        if 'bb_width' in indicators:
            if indicators['bb_width'] > 0.06:  # High volatility threshold
                volatility = 1
            elif indicators['bb_width'] < 0.02:  # Low volatility threshold
                volatility = -1
            else:
                volatility = 0  # Normal volatility
                
        result['indicators']['volatility'] = volatility
        
        # Momentum assessment
        momentum = 0
        if 'rsi_14' in indicators:
            if indicators['rsi_14'] > 70:
                momentum = 1  # Overbought
            elif indicators['rsi_14'] < 30:
                momentum = -1  # Oversold
            else:
                momentum = (indicators['rsi_14'] - 50) / 20  # Normalized -1 to 1
                
        result['indicators']['momentum'] = momentum
        
        # Trend strength
        trend_strength = 0
        if 'macd_hist' in indicators:
            if abs(indicators['macd_hist']) > abs(indicators['macd_signal']) * 0.5:
                trend_strength = 1 if indicators['macd_hist'] > 0 else -1
        
        result['indicators']['trend_strength'] = trend_strength
        
        # Volume confirmation
        volume_confirmation = 0
        if 'rel_volume' in indicators:
            volume_confirmation = min(1, max(-1, indicators['rel_volume'] - 1)) 
        
        result['indicators']['volume_confirmation'] = volume_confirmation
        
        # Combine indicators to determine regime
        # Trending regime detection
        if ma_alignment >= 0.5 and trend_strength > 0:
            if volatility < 0.5:
                result['primary_regime'] = MarketRegime.TRENDING_BULLISH.name
                result['confidence'] = min(0.9, 0.6 + ma_alignment * 0.2 + trend_strength * 0.1)
            else:
                result['primary_regime'] = MarketRegime.VOLATILE_BULLISH.name
                result['confidence'] = min(0.85, 0.6 + volatility * 0.15 + trend_strength * 0.1)
                
        elif ma_alignment <= -0.5 and trend_strength < 0:
            if volatility < 0.5:
                result['primary_regime'] = MarketRegime.TRENDING_BEARISH.name
                result['confidence'] = min(0.9, 0.6 + abs(ma_alignment) * 0.2 + abs(trend_strength) * 0.1)
            else:
                result['primary_regime'] = MarketRegime.VOLATILE_BEARISH.name
                result['confidence'] = min(0.85, 0.6 + volatility * 0.15 + abs(trend_strength) * 0.1)
                
        # Range-bound regime detection
        elif abs(ma_alignment) < 0.3 and abs(trend_strength) < 0.3:
            result['primary_regime'] = MarketRegime.RANGE_BOUND.name
            result['confidence'] = min(0.8, 0.5 + (0.3 - abs(ma_alignment)) * 0.5 + (0.3 - abs(trend_strength)) * 0.5)
            
        # Corrective regime detection
        elif (ma_alignment > 0 and trend_strength < -0.3) or (ma_alignment < 0 and trend_strength > 0.3):
            result['primary_regime'] = MarketRegime.CORRECTIVE.name
            result['confidence'] = min(0.75, 0.5 + abs(trend_strength) * 0.25)
            
        # High volatility regime
        elif volatility > 0.8:
            result['primary_regime'] = MarketRegime.HIGH_VOLATILITY.name
            result['confidence'] = min(0.9, 0.6 + volatility * 0.3)
        
        # Breakout detection
        if 'bb_pos' in indicators and abs(indicators['bb_pos']) > 0.9:
            result['sub_regimes'].append(MarketRegime.BREAKOUT.name)
                
        # Early trend detection
        if (ma_alignment >= 0.5 and abs(indicators.get('sma_50_200_diff', 0)) < 0.01) or \
           (ma_alignment <= -0.5 and abs(indicators.get('sma_50_200_diff', 0)) < 0.01):
            result['sub_regimes'].append(MarketRegime.EARLY_TREND.name)
            
        # Late trend detection
        if 'rsi_14' in indicators and ((ma_alignment > 0.8 and indicators['rsi_14'] > 75) or \
                                       (ma_alignment < -0.8 and indicators['rsi_14'] < 25)):
            result['sub_regimes'].append(MarketRegime.LATE_TREND.name)
            
        # Reversal signals
        if 'rsi_14' in indicators and 'macd_hist' in indicators:
            if (ma_alignment > 0.5 and indicators['rsi_14'] > 70 and indicators['macd_hist'] < 0) or \
               (ma_alignment < -0.5 and indicators['rsi_14'] < 30 and indicators['macd_hist'] > 0):
                result['sub_regimes'].append(MarketRegime.REVERSAL.name)
        
        return result
    
    def _ml_based_regime_detection(self, indicators: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Use ML models to detect market regime (stub for future implementation)
        
        Args:
            indicators: Dictionary of technical indicators
            data: Original OHLCV DataFrame
            
        Returns:
            Dictionary with regime classification
        """
        # This is a stub for future ML-based regime detection
        # In a production implementation, this would load trained models and make predictions
        
        # Use rule-based as fallback
        return self._rule_based_regime_detection(indicators, data)
    
    def _blend_detection_results(self, rule_results: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Blend results from different detection methods
        
        Args:
            rule_results: Results from rule-based detection
            ml_results: Results from ML-based detection
            
        Returns:
            Blended results
        """
        # This is a simple implementation that gives more weight to ML results when confidence is high
        ml_weight = 0.7 if ml_results['confidence'] > 0.8 else 0.3
        rule_weight = 1 - ml_weight
        
        # If regimes match, keep that regime with combined confidence
        if ml_results['primary_regime'] == rule_results['primary_regime']:
            return {
                'primary_regime': rule_results['primary_regime'],
                'confidence': (ml_results['confidence'] * ml_weight + rule_results['confidence'] * rule_weight),
                'sub_regimes': list(set(rule_results['sub_regimes'] + ml_results['sub_regimes'])),
                'indicators': rule_results['indicators']
            }
        
        # Otherwise use the higher confidence result
        if ml_results['confidence'] > rule_results['confidence']:
            return ml_results
        return rule_results
    
    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Check if we have a valid cached result
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        if key in self.regime_cache:
            timestamp_str, data = self.regime_cache[key]
            timestamp = datetime.fromisoformat(timestamp_str)
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            if age_minutes < self.cache_expiry:
                return data
        return None
    
    def _update_cache(self, key: str, data: Dict[str, Any]) -> None:
        """
        Update the cache with new data
        
        Args:
            key: Cache key
            data: Data to cache
        """
        self.regime_cache[key] = (datetime.now().isoformat(), data)
    
    def _create_undefined_regime(self) -> Dict[str, Any]:
        """
        Create a default undefined regime response
        
        Returns:
            Dictionary with undefined regime
        """
        return {
            'primary_regime': MarketRegime.UNDEFINED.name,
            'confidence': 0.0,
            'sub_regimes': [],
            'indicators': {},
            'timestamp': datetime.now().isoformat()
        }
