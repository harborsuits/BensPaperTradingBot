"""
Market Regime Detector

This module provides tools to detect and classify market regimes (trending, ranging, volatile)
to help adaptive trading strategies adjust their parameters for optimal performance.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
from enum import Enum
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enum representing different market regimes"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

@dataclass
class RegimeData:
    """Data class for market regime information"""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    strength: float  # 0.0 to 1.0
    duration: int  # Days in current regime
    metrics: Dict[str, float]  # Additional metrics

class MarketRegimeDetector:
    """
    Detects and classifies market regimes to help strategies adapt.
    
    The MarketRegimeDetector analyzes price data to determine the current
    market condition (trend, range, volatility) and provides confidence 
    scores to help strategies adjust their parameters for the current regime.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market regime detector.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Analysis parameters
        self.atr_period = self.config.get('atr_period', 14)
        self.trend_period = self.config.get('trend_period', 20)
        self.ma_fast_period = self.config.get('ma_fast_period', 20)
        self.ma_slow_period = self.config.get('ma_slow_period', 50)
        self.volatility_threshold = self.config.get('volatility_threshold', 1.5)
        self.trend_strength_threshold = self.config.get('trend_strength_threshold', 0.6)
        self.range_threshold = self.config.get('range_threshold', 0.3)
        self.lookback_periods = self.config.get('lookback_periods', [10, 20, 50, 100, 200])  # Days
        
        # Regime data storage
        self.regime_history = {}  # symbol -> list of regime data points
        self.current_regimes = {}  # symbol -> current regime data
        
        # Cache for market data
        self.price_data = {}  # symbol -> DataFrame
        self.data_updated = {}  # symbol -> last update time
        
        # Data saving/loading
        self.data_dir = self.config.get('data_dir', './data/regimes')
        self.auto_save = self.config.get('auto_save', True)
        self.save_interval = self.config.get('save_interval', 24*60*60)  # 24 hours
        self.last_save_time = datetime.now()
        
        # Ensure data directory exists
        if self.auto_save and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        
        # Try to load existing data
        self._load_data()
        
        logger.info(f"Initialized MarketRegimeDetector with {len(self.current_regimes)} symbols")
    
    def update_prices(self, 
                      symbol: str, 
                      prices_df: pd.DataFrame) -> None:
        """
        Update price data for a symbol.
        
        Args:
            symbol: Trading symbol
            prices_df: DataFrame with OHLCV data
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Convert column names to lowercase
        df = prices_df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        # Check if all required columns are present
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' for {symbol}")
                return
        
        # Ensure DataFrame is sorted by date
        if 'date' in df.columns or 'timestamp' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            df = df.sort_values(by=date_col)
        
        # Store data
        self.price_data[symbol] = df
        self.data_updated[symbol] = datetime.now()
        
        # Analyze regime
        self._analyze_regime(symbol)
        
        # Auto-save if enabled
        if self.auto_save:
            time_since_save = (datetime.now() - self.last_save_time).total_seconds()
            if time_since_save > self.save_interval:
                self._save_data()
    
    def get_current_regime(self, symbol: str) -> Optional[RegimeData]:
        """
        Get the current market regime for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            RegimeData object or None if unavailable
        """
        return self.current_regimes.get(symbol)
    
    def get_regime_history(self, 
                          symbol: str, 
                          days: int = 30) -> List[Dict[str, Any]]:
        """
        Get regime history for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of history to return
            
        Returns:
            List of regime data dictionaries
        """
        if symbol not in self.regime_history:
            return []
        
        # Get regime history entries within specified days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        history = []
        for entry in self.regime_history[symbol]:
            try:
                entry_date = datetime.fromisoformat(entry.get('date', ''))
                if entry_date >= cutoff_date:
                    history.append(entry)
            except (ValueError, TypeError):
                continue
        
        return history
    
    def get_all_regimes(self) -> Dict[str, RegimeData]:
        """
        Get current regimes for all tracked symbols.
        
        Returns:
            Dict mapping symbols to RegimeData objects
        """
        return self.current_regimes.copy()
    
    def is_suitable_for_strategy(self, 
                                symbol: str, 
                                strategy_type: str, 
                                min_confidence: float = 0.6) -> bool:
        """
        Check if current market regime is suitable for a strategy type.
        
        Args:
            symbol: Trading symbol
            strategy_type: Strategy type (e.g., 'trend_following', 'mean_reversion', 'breakout')
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if regime is suitable for strategy
        """
        regime_data = self.get_current_regime(symbol)
        if not regime_data:
            return False
        
        # Check confidence
        if regime_data.confidence < min_confidence:
            return False
        
        # Map strategy types to suitable regimes
        strategy_regime_map = {
            'trend_following': [
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN
            ],
            'mean_reversion': [
                MarketRegime.RANGING,
                MarketRegime.REVERSAL
            ],
            'breakout': [
                MarketRegime.BREAKOUT,
                MarketRegime.VOLATILE
            ],
            'volatility': [
                MarketRegime.VOLATILE
            ]
        }
        
        # Check if current regime is suitable for strategy
        suitable_regimes = strategy_regime_map.get(strategy_type.lower(), [])
        return regime_data.regime in suitable_regimes
    
    def get_strategy_suitability(self, symbol: str) -> Dict[str, float]:
        """
        Get suitability scores for different strategy types.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict mapping strategy types to suitability scores (0.0 to 1.0)
        """
        regime_data = self.get_current_regime(symbol)
        if not regime_data:
            return {}
        
        # Base scores
        suitability = {
            'trend_following': 0.0,
            'mean_reversion': 0.0,
            'breakout': 0.0,
            'volatility': 0.0
        }
        
        # Adjust scores based on regime
        if regime_data.regime == MarketRegime.TRENDING_UP or regime_data.regime == MarketRegime.TRENDING_DOWN:
            suitability['trend_following'] = regime_data.confidence * regime_data.strength
            suitability['mean_reversion'] = 0.3 * (1 - regime_data.strength)
            
        elif regime_data.regime == MarketRegime.RANGING:
            suitability['mean_reversion'] = regime_data.confidence * regime_data.strength
            suitability['trend_following'] = 0.2 * (1 - regime_data.strength)
            
        elif regime_data.regime == MarketRegime.VOLATILE:
            suitability['volatility'] = regime_data.confidence * regime_data.strength
            suitability['breakout'] = 0.5 * regime_data.confidence
            
        elif regime_data.regime == MarketRegime.BREAKOUT:
            suitability['breakout'] = regime_data.confidence * regime_data.strength
            suitability['trend_following'] = 0.7 * regime_data.confidence
            
        elif regime_data.regime == MarketRegime.REVERSAL:
            suitability['mean_reversion'] = regime_data.confidence * regime_data.strength
            suitability['trend_following'] = 0.4 * (1 - regime_data.strength)
        
        return suitability
    
    def get_optimal_parameters(self, 
                              symbol: str, 
                              strategy_type: str) -> Dict[str, Any]:
        """
        Get optimal parameters for a strategy type in current regime.
        
        Args:
            symbol: Trading symbol
            strategy_type: Strategy type (e.g., 'trend_following', 'mean_reversion')
            
        Returns:
            Dict with recommended parameters
        """
        regime_data = self.get_current_regime(symbol)
        if not regime_data or not regime_data.regime:
            return {}
        
        # Base parameter adjustments
        params = {}
        
        # Trend following parameters
        if strategy_type.lower() == 'trend_following':
            if regime_data.regime == MarketRegime.TRENDING_UP or regime_data.regime == MarketRegime.TRENDING_DOWN:
                # Strong trend, use faster settings
                params['trailing_stop_pct'] = 2.0 * (1 + regime_data.strength)
                params['profit_target_pct'] = 3.0 * (1 + regime_data.strength)
                params['entry_filter_strength'] = 0.5 * (1 - regime_data.strength)  # More permissive in strong trends
                
            elif regime_data.regime == MarketRegime.RANGING:
                # Ranging market, more conservative
                params['trailing_stop_pct'] = 1.5
                params['profit_target_pct'] = 2.0
                params['entry_filter_strength'] = 0.8  # More selective
                
            elif regime_data.regime == MarketRegime.VOLATILE:
                # Volatile market, very tight parameters
                params['trailing_stop_pct'] = 3.0
                params['profit_target_pct'] = 4.0
                params['entry_filter_strength'] = 0.9  # Very selective
        
        # Mean reversion parameters
        elif strategy_type.lower() == 'mean_reversion':
            if regime_data.regime == MarketRegime.RANGING:
                # Ideal for mean reversion
                params['entry_threshold'] = 2.0 * regime_data.strength
                params['profit_target_pct'] = 1.5 * regime_data.strength
                params['stop_loss_pct'] = 1.0 * (1 + regime_data.strength)
                
            elif regime_data.regime == MarketRegime.TRENDING_UP or regime_data.regime == MarketRegime.TRENDING_DOWN:
                # Trend - harder for mean reversion
                params['entry_threshold'] = 3.0  # Higher threshold to enter
                params['profit_target_pct'] = 1.0  # Lower target
                params['stop_loss_pct'] = 1.5  # Tighter stop
                
            elif regime_data.regime == MarketRegime.VOLATILE:
                # Volatile market, very selective
                params['entry_threshold'] = 4.0
                params['profit_target_pct'] = 2.0
                params['stop_loss_pct'] = 2.0
        
        # Breakout parameters  
        elif strategy_type.lower() == 'breakout':
            if regime_data.regime == MarketRegime.BREAKOUT:
                # Ideal for breakout
                params['breakout_threshold'] = 1.5 * (1 + regime_data.strength)
                params['confirmation_period'] = 2 if regime_data.strength > 0.7 else 3
                params['trailing_stop_pct'] = 2.0 * regime_data.strength
                
            elif regime_data.regime == MarketRegime.VOLATILE:
                # Good for breakout but need confirmation
                params['breakout_threshold'] = 2.0
                params['confirmation_period'] = 3
                params['trailing_stop_pct'] = 2.5
                
            elif regime_data.regime == MarketRegime.RANGING:
                # Range bound - wait for stronger breakouts
                params['breakout_threshold'] = 2.5
                params['confirmation_period'] = 4
                params['trailing_stop_pct'] = 1.5
        
        # Volatility parameters
        elif strategy_type.lower() == 'volatility':
            if regime_data.regime == MarketRegime.VOLATILE:
                # Ideal for volatility strategies
                params['vix_threshold'] = 20 * (1 - regime_data.strength)  # Lower threshold when already volatile
                params['position_size_scale'] = 0.7 * (1 - regime_data.strength)  # Smaller when more volatile
                params['profit_target_mult'] = 1.5 * regime_data.strength
                
            elif regime_data.regime == MarketRegime.RANGING:
                # Less volatile, higher thresholds
                params['vix_threshold'] = 25
                params['position_size_scale'] = 0.8
                params['profit_target_mult'] = 1.2
                
            elif regime_data.regime == MarketRegime.TRENDING_UP or regime_data.regime == MarketRegime.TRENDING_DOWN:
                # Use volatility expansion in trends
                params['vix_threshold'] = 15
                params['position_size_scale'] = 0.9
                params['profit_target_mult'] = 1.3
        
        return params
    
    def _analyze_regime(self, symbol: str) -> None:
        """
        Analyze price data to determine current market regime.
        
        Args:
            symbol: Trading symbol
        """
        if symbol not in self.price_data:
            return
        
        df = self.price_data[symbol]
        
        if len(df) < self.ma_slow_period:
            logger.warning(f"Insufficient data for {symbol} regime analysis. Need at least {self.ma_slow_period} bars.")
            return
        
        # Calculate indicators
        # Add moving averages
        df['ma_fast'] = df['close'].rolling(window=self.ma_fast_period).mean()
        df['ma_slow'] = df['close'].rolling(window=self.ma_slow_period).mean()
        
        # Calculate Average True Range (ATR)
        df['tr'] = np.maximum(
            np.maximum(
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1))
            ),
            abs(df['low'] - df['close'].shift(1))
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Percent volatility (ATR as percentage of price)
        df['volatility_pct'] = df['atr'] / df['close'] * 100
        
        # Directional Movement Index (DMI)
        df['plus_dm'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        df['minus_dm'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        df['plus_di'] = 100 * df['plus_dm'].rolling(window=self.trend_period).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(window=self.trend_period).mean() / df['atr']
        
        # Calculate ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=self.trend_period).mean()
        
        # Slope of moving averages
        df['ma_fast_slope'] = df['ma_fast'] - df['ma_fast'].shift(5)
        df['ma_slow_slope'] = df['ma_slow'] - df['ma_slow'].shift(5)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Relative volatility
        df['relative_volatility'] = df['volatility_pct'] / df['volatility_pct'].rolling(window=50).mean()
        
        # Price relative to moving averages
        df['price_vs_ma_fast'] = df['close'] / df['ma_fast'] - 1
        df['price_vs_ma_slow'] = df['close'] / df['ma_slow'] - 1
        df['ma_diff'] = df['ma_fast'] / df['ma_slow'] - 1
        
        # Analyze the most recent data
        recent = df.iloc[-1].copy()
        
        # Combine multiple indicators to determine regime
        regime = MarketRegime.UNKNOWN
        confidence = 0.5
        strength = 0.5
        metrics = {}
        
        # Record key metrics
        metrics['adx'] = recent['adx']
        metrics['volatility_pct'] = recent['volatility_pct']
        metrics['relative_volatility'] = recent['relative_volatility']
        metrics['bb_width'] = recent['bb_width']
        metrics['ma_diff'] = recent['ma_diff']
        metrics['price_vs_ma_fast'] = recent['price_vs_ma_fast']
        metrics['price_vs_ma_slow'] = recent['price_vs_ma_slow']
        
        # Check for trending market
        if recent['adx'] > 25:  # Strong trend when ADX > 25
            # Trending up or down?
            if recent['plus_di'] > recent['minus_di'] and recent['ma_fast_slope'] > 0:
                regime = MarketRegime.TRENDING_UP
                strength = min((recent['adx'] - 25) / 25, 1.0)  # Normalized 0-1
            elif recent['minus_di'] > recent['plus_di'] and recent['ma_fast_slope'] < 0:
                regime = MarketRegime.TRENDING_DOWN
                strength = min((recent['adx'] - 25) / 25, 1.0)  # Normalized 0-1
            
            confidence = min(0.5 + (recent['adx'] - 25) / 50, 0.95)  # Higher ADX = higher confidence
        
        # Check for ranging market
        elif recent['adx'] < 20 and recent['bb_width'] < df['bb_width'].rolling(window=50).mean().iloc[-1]:
            regime = MarketRegime.RANGING
            strength = max(1 - recent['adx'] / 20, 0.3)  # Lower ADX = stronger range
            confidence = min(0.5 + (20 - recent['adx']) / 20, 0.9)  # Lower ADX = higher confidence
        
        # Check for volatile market
        elif recent['relative_volatility'] > self.volatility_threshold:
            regime = MarketRegime.VOLATILE
            strength = min(recent['relative_volatility'] / 3, 1.0)  # Normalized 0-1
            confidence = min(0.5 + (recent['relative_volatility'] - 1.5) / 3, 0.9)
        
        # Check for breakout
        elif (abs(recent['price_vs_ma_fast']) > 0.02 and 
              abs(recent['ma_fast_slope']) > df['ma_fast_slope'].abs().rolling(window=20).mean().iloc[-1] * 2):
            regime = MarketRegime.BREAKOUT
            strength = min(abs(recent['price_vs_ma_fast']) / 0.04, 1.0)
            confidence = min(0.5 + abs(recent['price_vs_ma_fast']) / 0.04, 0.9)
        
        # Check for potential reversal
        elif ((recent['ma_fast_slope'] * df['ma_fast_slope'].shift(5).iloc[-1] < 0) or  # Slope changed direction
              (abs(recent['price_vs_ma_fast']) > 0.03 and recent['price_vs_ma_fast'] * recent['ma_fast_slope'] < 0)):  # Price and MA slope diverge
            regime = MarketRegime.REVERSAL
            strength = min(abs(recent['price_vs_ma_fast']) / 0.03, 1.0)
            confidence = 0.6  # Reversals are harder to predict
        
        # Create regime data
        regime_data = RegimeData(
            regime=regime,
            confidence=confidence,
            strength=strength,
            duration=self._get_regime_duration(symbol, regime),
            metrics=metrics
        )
        
        # Update current regime
        self.current_regimes[symbol] = regime_data
        
        # Add to history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        # Convert to serializable dict
        regime_dict = {
            'date': datetime.now().isoformat(),
            'regime': regime.value,
            'confidence': confidence,
            'strength': strength,
            'duration': regime_data.duration,
            'metrics': metrics
        }
        
        self.regime_history[symbol].append(regime_dict)
        
        # Keep history reasonably sized (last 365 days)
        if len(self.regime_history[symbol]) > 365:
            self.regime_history[symbol] = self.regime_history[symbol][-365:]
        
        logger.info(f"Analyzed regime for {symbol}: {regime.value} (conf: {confidence:.2f}, strength: {strength:.2f})")
    
    def _get_regime_duration(self, symbol: str, current_regime: MarketRegime) -> int:
        """
        Calculate how many consecutive days the symbol has been in the current regime.
        
        Args:
            symbol: Trading symbol
            current_regime: Current regime
            
        Returns:
            Number of days in current regime
        """
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return 1  # First day
        
        # Check previous regimes
        duration = 1
        for entry in reversed(self.regime_history[symbol]):
            if entry.get('regime') == current_regime.value:
                duration += 1
            else:
                break
        
        return duration
    
    def _save_data(self) -> bool:
        """
        Save regime data to disk.
        
        Returns:
            bool: True if successful
        """
        try:
            # Save history
            history_file = os.path.join(self.data_dir, 'regime_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.regime_history, f)
            
            # Save current regimes as serializable dict
            current_regimes_dict = {}
            for symbol, regime_data in self.current_regimes.items():
                current_regimes_dict[symbol] = {
                    'regime': regime_data.regime.value,
                    'confidence': regime_data.confidence,
                    'strength': regime_data.strength,
                    'duration': regime_data.duration,
                    'metrics': regime_data.metrics
                }
            
            current_file = os.path.join(self.data_dir, 'current_regimes.json')
            with open(current_file, 'w') as f:
                json.dump(current_regimes_dict, f)
            
            self.last_save_time = datetime.now()
            logger.info(f"Saved regime data for {len(self.current_regimes)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error saving regime data: {str(e)}")
            return False
    
    def _load_data(self) -> bool:
        """
        Load regime data from disk.
        
        Returns:
            bool: True if successful
        """
        try:
            # Load history
            history_file = os.path.join(self.data_dir, 'regime_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.regime_history = json.load(f)
            
            # Load current regimes
            current_file = os.path.join(self.data_dir, 'current_regimes.json')
            if os.path.exists(current_file):
                with open(current_file, 'r') as f:
                    current_regimes_dict = json.load(f)
                
                # Convert back to RegimeData objects
                for symbol, data in current_regimes_dict.items():
                    self.current_regimes[symbol] = RegimeData(
                        regime=MarketRegime(data['regime']),
                        confidence=data['confidence'],
                        strength=data['strength'],
                        duration=data['duration'],
                        metrics=data['metrics']
                    )
            
            logger.info(f"Loaded regime data for {len(self.current_regimes)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error loading regime data: {str(e)}")
            return False
