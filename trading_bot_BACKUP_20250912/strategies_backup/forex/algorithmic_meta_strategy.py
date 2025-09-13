#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Algorithmic Trading Meta-Strategy

This strategy functions as an ensemble approach, dynamically combining
multiple sub-strategies based on market conditions, performance metrics,
and machine learning-based optimization. It represents the most sophisticated
strategy in the institutional forex suite, capable of auto-optimizing and
adapting to changing market regimes.

Key Features:
1. Ensemble-based strategy combination
2. ML-driven weight optimization
3. Dynamic parameter tuning
4. Adaptive regime detection
5. Performance-based strategy rotation
6. Continuous backtesting and validation

Author: Ben Dickinson
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set, Callable
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
from collections import defaultdict
import warnings

# For ML components (we'll handle import errors gracefully)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML features disabled for Algorithmic Meta-Strategy.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib not available. Model persistence disabled for Algorithmic Meta-Strategy.")

from trading_bot.strategies.base.forex_base import ForexBaseStrategy
from trading_bot.utils.event_bus import EventBus
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.strategies.strategy_template import (
    Strategy, SignalType, PositionSizing, OrderStatus, 
    TradeDirection, MarketRegime, TimeFrame
)

logger = logging.getLogger(__name__)

class WeightingMethod(Enum):
    """Methods for weighting sub-strategies."""
    EQUAL = "equal"               # Equal weights
    PERFORMANCE = "performance"   # Based on historical performance
    REGIME = "regime"             # Based on regime compatibility
    ML_OPTIMIZED = "ml_optimized" # ML-optimized weights
    ADAPTIVE = "adaptive"         # Dynamically adaptive weighting

class PerformanceMetric(Enum):
    """Key performance metrics for strategy evaluation."""
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTANCY = "expectancy"
    RECOVERY_FACTOR = "recovery_factor"

class AlgorithmicMetaStrategy(ForexBaseStrategy):
    """
    Forex Algorithmic Trading Meta-Strategy
    
    This strategy functions as an ensemble framework that dynamically combines
    multiple sub-strategies based on market conditions, performance metrics,
    and machine learning-based optimization.
    """
    
    def __init__(self,
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Algorithmic Meta-Strategy.
        
        Args:
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # Strategy selection and weighting
            'sub_strategies': [
                'forex_trend_following',
                'forex_breakout',
                'forex_momentum',
                'forex_range',
                'forex_price_action',
                'forex_retracement'
            ],
            'weighting_method': WeightingMethod.ADAPTIVE.value,
            'min_strategies': 2,     # Minimum number of sub-strategies to use
            'max_strategies': 5,     # Maximum number of sub-strategies to use
            'min_strategy_weight': 0.1,  # Minimum weight for any strategy
            
            # Performance evaluation
            'performance_window': 30,  # Days to consider for performance history
            'performance_metrics': [
                PerformanceMetric.WIN_RATE.value,
                PerformanceMetric.PROFIT_FACTOR.value,
                PerformanceMetric.SHARPE_RATIO.value
            ],
            'metric_weights': {
                'win_rate': 0.3,
                'profit_factor': 0.4,
                'sharpe_ratio': 0.3
            },
            
            # ML optimization
            'use_ml_optimization': True,  # Enable ML-based optimization
            'ml_model_type': 'random_forest',  # Model to use: random_forest, gradient_boost, elastic_net
            'training_window': 90,  # Days of data to use for training
            'retraining_interval': 7,  # Days between model retraining
            'prediction_features': [
                'volatility', 'trend_strength', 'range_bound', 
                'momentum', 'day_of_week', 'hour_of_day', 'prev_performance'
            ],
            
            # Signal generation
            'signal_threshold': 0.6,  # Minimum combined signal strength
            'agreement_threshold': 0.65,  # Required agreement among strategies (0-1)
            'conflict_resolution': 'strongest',  # strongest, majority, or consensus
            
            # Risk management
            'position_sizing_method': 'risk_adjusted',  # equal, performance_based, risk_adjusted
            'max_correlated_exposure': 0.2,  # Maximum exposure to correlated strategies
            'use_dynamic_stops': True,  # Dynamically adjust stop losses
            'use_trailing_stops': True,  # Use trailing stops
            'trailing_stop_activation': 0.5,  # ATR multiples before trailing activates
            
            # Regime detection
            'regime_detection_method': 'ml',  # statistical, indicator_based, or ml
            'fast_regime_period': 10,  # Days for fast-changing regime detection
            'slow_regime_period': 30,  # Days for slow-changing regime detection
            
            # Persistence and monitoring
            'save_ml_models': True,  # Save ML models periodically
            'model_directory': 'models/algorithmic',  # Directory for model storage
            'monitoring_metrics': [
                'win_rate', 'profit_factor', 'sharpe_ratio', 
                'sortino_ratio', 'max_drawdown', 'recovery_factor'
            ],
            
            # Advanced
            'use_reinforcement_learning': False,  # RL for weight optimization
            'feature_selection_method': 'importance',  # none, correlation, importance
            'ensemble_method': 'weighted_vote',  # weighted_vote, stacked, boosted
            'use_hyperparameter_tuning': True,  # Auto-tune ML parameters
        }
        
        # Initialize base class
        default_metadata = {
            'name': 'Forex Algorithmic Meta-Strategy',
            'description': 'Advanced ensemble meta-strategy that combines multiple forex strategies adaptively',
            'version': '1.0.0',
            'author': 'Ben Dickinson',
            'type': 'forex_algorithmic_meta',
            'tags': ['forex', 'algorithmic', 'meta_strategy', 'ensemble', 'machine_learning', 'adaptive'],
            'preferences': {
                'timeframes': ['15M', '1H', '4H', 'D'],
                'default_timeframe': '1H'
            }
        }
        
        # Update metadata if provided
        if metadata:
            default_metadata.update(metadata)
            
        super().__init__('forex_algorithmic_meta', parameters, default_metadata)
        
        # Merge provided parameters with defaults
        if parameters:
            for key, value in parameters.items():
                if key in default_params:
                    self.parameters[key] = value
        else:
            self.parameters = default_params
            
        # Initialize sub-strategies
        self.strategy_factory = StrategyFactory()
        self.sub_strategies = {}  # strategy_type -> strategy instance
        self._initialize_sub_strategies()
        
        # Performance tracking
        self.strategy_performance = {}  # strategy_type -> performance metrics
        self.strategy_signals = {}      # strategy_type -> recent signals
        self.strategy_weights = {}      # strategy_type -> current weight
        
        # ML models
        self.ml_models = {}           # symbol -> ML model
        self.feature_scalers = {}     # symbol -> feature scaler
        self.last_training_time = {}  # symbol -> last training timestamp
        
        # State tracking
        self.current_regime = {}      # symbol -> detected regime
        self.regime_history = {}      # symbol -> list of historical regimes
        self.signal_history = {}      # symbol -> list of historical signals
        self.weight_history = {}      # strategy_type -> list of historical weights
        
        # Register with event bus
        EventBus.get_instance().register(self)
        
    def _detect_market_regime(self, data: pd.DataFrame, symbol: str) -> MarketRegime:
        """
        Detect the current market regime using advanced techniques.
        
        Args:
            data: OHLCV DataFrame
            symbol: Currency pair symbol
            
        Returns:
            Detected market regime
        """
        if len(data) < 50:
            return MarketRegime.UNKNOWN
            
        # Get detection method from parameters
        method = self.parameters['regime_detection_method']
        
        # Calculate key metrics
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
        
        # Calculate average true range for volatility
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / close.iloc[-1]
        
        # Calculate trend metrics
        sma20 = data['close'].rolling(20).mean()
        sma50 = data['close'].rolling(50).mean()
        trend_direction = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1
        price_to_sma_ratio = data['close'].iloc[-1] / sma50.iloc[-1]
        
        # Calculate momentum
        momentum = data['close'].pct_change(20).iloc[-1]
        
        # Calculate range-bound indicators
        highest_high = data['high'].rolling(20).max()
        lowest_low = data['low'].rolling(20).min()
        price_range_pct = (highest_high.iloc[-1] - lowest_low.iloc[-1]) / lowest_low.iloc[-1]
        
        # Statistical method
        if method == 'statistical':
            # High volatility check
            if volatility > 0.25 or atr_pct > 0.015:  # Very high volatility
                return MarketRegime.HIGH_VOLATILITY
                
            # Low volatility check
            if volatility < 0.05 and atr_pct < 0.005:  # Very low volatility
                return MarketRegime.LOW_VOLATILITY
                
            # Trending check
            if abs(momentum) > 0.03 and abs(price_to_sma_ratio - 1) > 0.02:
                if momentum > 0 and trend_direction > 0:
                    return MarketRegime.BULLISH
                elif momentum < 0 and trend_direction < 0:
                    return MarketRegime.BEARISH
                return MarketRegime.TRENDING
                
            # Breakout check
            if price_range_pct < 0.02 and abs(momentum) > 0.02:
                return MarketRegime.BREAKOUT
                
            # Ranging check
            if price_range_pct < 0.03 and abs(momentum) < 0.01:
                return MarketRegime.RANGING
                
            # Consolidation check
            if price_range_pct < 0.02 and abs(momentum) < 0.005:
                return MarketRegime.CONSOLIDATION
                
            # Default to volatile if nothing else fits
            return MarketRegime.VOLATILE
            
        # Indicator-based method 
        elif method == 'indicator_based':
            # Calculate ADX for trend strength
            # This is a simplified version - in practice, use proper ADX calculation
            adx = data['close'].diff().abs().rolling(14).mean() / data['close'] * 100
            adx_value = adx.iloc[-1]
            
            # Calculate Bollinger Band width for volatility/ranging
            sma20 = data['close'].rolling(20).mean()
            std20 = data['close'].rolling(20).std()
            bb_width = (2 * std20 / sma20).iloc[-1]
            
            # Determine regime
            if adx_value > 30 and bb_width > 0.05:  # Strong trend with volatility
                if trend_direction > 0:
                    return MarketRegime.BULLISH
                else:
                    return MarketRegime.BEARISH
            elif adx_value > 25:  # Strong trend
                return MarketRegime.TRENDING
            elif bb_width < 0.02:  # Tight Bollinger Bands
                if adx_value < 15:  # Low ADX
                    return MarketRegime.CONSOLIDATION
                else:
                    return MarketRegime.BREAKOUT  # Potential breakout forming
            elif bb_width < 0.04 and adx_value < 20:  # Medium-width bands with low ADX
                return MarketRegime.RANGING
            elif bb_width > 0.06:  # Wide bands
                return MarketRegime.VOLATILE
            elif bb_width > 0.08:  # Very wide bands
                return MarketRegime.HIGH_VOLATILITY
            elif bb_width < 0.02 and adx_value < 10:  # Very tight bands, very low ADX
                return MarketRegime.LOW_VOLATILITY
                
            # Default
            return MarketRegime.UNKNOWN
            
        # ML-based regime detection
        elif method == 'ml' and ML_AVAILABLE:
            # Use ML model if available, otherwise fall back to statistical
            regime_model_path = os.path.join(
                self.parameters['model_directory'], 
                f"regime_detector_{symbol}.joblib"
            )
            
            if os.path.exists(regime_model_path) and JOBLIB_AVAILABLE:
                try:
                    # Load the model
                    model = joblib.load(regime_model_path)
                    
                    # Prepare feature vector
                    features = [
                        volatility,
                        atr_pct,
                        price_to_sma_ratio - 1.0,  # Deviation from SMA
                        momentum,
                        price_range_pct,
                        adx.iloc[-1] if 'adx' in locals() else 0.0,
                        bb_width if 'bb_width' in locals() else 0.0
                    ]
                    
                    # Reshape for prediction
                    X = np.array(features).reshape(1, -1)
                    
                    # Predict regime
                    regime_idx = model.predict(X)[0]
                    
                    # Convert to MarketRegime enum
                    regimes = list(MarketRegime)
                    if 0 <= regime_idx < len(regimes):
                        return regimes[int(regime_idx)]
                        
                except Exception as e:
                    logger.error(f"Error using ML regime detector: {str(e)}")
                    # Fall back to statistical method
                    return self._detect_market_regime_statistical(data)
            else:
                # Fall back to statistical method
                return self._detect_market_regime_statistical(data)
        else:
            # Default to statistical method
            return self._detect_market_regime_statistical(data)
        
        # Default fallback
        return MarketRegime.UNKNOWN
        
    def _detect_market_regime_statistical(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect market regime using statistical approach (fallback method).
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Detected market regime
        """
        if len(data) < 50:
            return MarketRegime.UNKNOWN
            
        # Calculate key metrics
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        momentum = data['close'].pct_change(20).iloc[-1]
        
        # Calculate trend metrics
        sma20 = data['close'].rolling(20).mean()
        sma50 = data['close'].rolling(50).mean()
        trend_direction = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1
        
        # Calculate range metrics
        highest_high = data['high'].rolling(20).max()
        lowest_low = data['low'].rolling(20).min()
        price_range_pct = (highest_high.iloc[-1] - lowest_low.iloc[-1]) / lowest_low.iloc[-1]
        
        # Determine regime
        if volatility > 0.2:  # High volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.05:  # Low volatility
            return MarketRegime.LOW_VOLATILITY
        elif abs(momentum) > 0.03 and trend_direction > 0:  # Strong up momentum
            return MarketRegime.BULLISH
        elif abs(momentum) > 0.03 and trend_direction < 0:  # Strong down momentum
            return MarketRegime.BEARISH
        elif price_range_pct < 0.02:  # Tight range
            return MarketRegime.CONSOLIDATION
        elif price_range_pct < 0.04:  # Medium range
            return MarketRegime.RANGING
        else:  # Default
            return MarketRegime.TRENDING
            
    def _calculate_strategy_weights(self, data: Dict[str, pd.DataFrame], current_regimes: Dict[str, MarketRegime]) -> Dict[str, float]:
        """
        Calculate optimal weights for each sub-strategy based on the weighting method.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_regimes: Dictionary of symbol -> detected market regime
            
        Returns:
            Dictionary of strategy_type -> weight
        """
        if not self.sub_strategies:
            return {}
            
        weighting_method = self.parameters['weighting_method']
        
        # Equal weighting (baseline)
        if weighting_method == WeightingMethod.EQUAL.value:
            equal_weight = 1.0 / len(self.sub_strategies)
            return {s_type: equal_weight for s_type in self.sub_strategies}
            
        # Performance-based weighting
        elif weighting_method == WeightingMethod.PERFORMANCE.value:
            return self._calculate_performance_weights()
            
        # Regime-based weighting
        elif weighting_method == WeightingMethod.REGIME.value:
            return self._calculate_regime_weights(current_regimes)
            
        # ML-optimized weighting
        elif weighting_method == WeightingMethod.ML_OPTIMIZED.value and ML_AVAILABLE:
            return self._calculate_ml_weights(data, current_regimes)
            
        # Adaptive weighting (combines multiple methods)
        elif weighting_method == WeightingMethod.ADAPTIVE.value:
            return self._calculate_adaptive_weights(data, current_regimes)
            
        # Default to equal weighting
        equal_weight = 1.0 / len(self.sub_strategies)
        return {s_type: equal_weight for s_type in self.sub_strategies}
        
    def _calculate_performance_weights(self) -> Dict[str, float]:
        """
        Calculate weights based on historical performance metrics.
        
        Returns:
            Dictionary of strategy_type -> weight
        """
        if not self.strategy_performance:
            # Fall back to equal weights if no performance data
            equal_weight = 1.0 / len(self.sub_strategies)
            return {s_type: equal_weight for s_type in self.sub_strategies}
            
        weights = {}
        total_score = 0.0
        min_weight = self.parameters['min_strategy_weight']
        
        # Get performance metrics to use
        perf_metrics = self.parameters['performance_metrics']
        metric_weights = self.parameters['metric_weights']
        
        # Calculate score for each strategy
        for strategy_type in self.sub_strategies:
            if strategy_type not in self.strategy_performance:
                weights[strategy_type] = min_weight
                continue
                
            perf = self.strategy_performance[strategy_type]
            score = 0.0
            
            # Calculate weighted score from metrics
            for metric in perf_metrics:
                if metric in perf and metric in metric_weights:
                    # Normalize based on metric type (higher is better except for drawdown)
                    value = perf[metric]
                    if metric == PerformanceMetric.MAX_DRAWDOWN.value:
                        # Convert drawdown to positive score (lower is better)
                        if value != 0:
                            value = 1.0 / value
                    
                    score += value * metric_weights.get(metric, 1.0)
            
            weights[strategy_type] = max(min_weight, score)
            total_score += weights[strategy_type]
        
        # Normalize weights
        if total_score > 0:
            for strategy_type in weights:
                weights[strategy_type] /= total_score
        else:
            # Fall back to equal weights
            equal_weight = 1.0 / len(self.sub_strategies)
            return {s_type: equal_weight for s_type in self.sub_strategies}
        
        return weights
    
    def _calculate_regime_weights(self, current_regimes: Dict[str, MarketRegime]) -> Dict[str, float]:
        """
        Calculate weights based on strategy compatibility with current market regimes.
        
        Args:
            current_regimes: Dictionary of symbol -> detected market regime
            
        Returns:
            Dictionary of strategy_type -> weight
        """
        if not current_regimes:
            # Fall back to equal weights if no regime data
            equal_weight = 1.0 / len(self.sub_strategies)
            return {s_type: equal_weight for s_type in self.sub_strategies}
        
        # Get the most common regime across symbols
        all_regimes = list(current_regimes.values())
        if not all_regimes:
            regime = MarketRegime.UNKNOWN
        else:
            # Count occurrences of each regime
            regime_counts = {}
            for r in all_regimes:
                regime_counts[r] = regime_counts.get(r, 0) + 1
            
            # Get the most common regime
            regime = max(regime_counts.items(), key=lambda x: x[1])[0]
        
        weights = {}
        total_score = 0.0
        min_weight = self.parameters['min_strategy_weight']
        
        # Calculate compatibility score for each strategy
        for strategy_type, strategy in self.sub_strategies.items():
            # Get regime compatibility score
            compatibility = strategy.get_regime_compatibility_score(regime)
            
            # Apply minimum weight
            weights[strategy_type] = max(min_weight, compatibility)
            total_score += weights[strategy_type]
        
        # Normalize weights
        if total_score > 0:
            for strategy_type in weights:
                weights[strategy_type] /= total_score
        else:
            # Fall back to equal weights
            equal_weight = 1.0 / len(self.sub_strategies)
            return {s_type: equal_weight for s_type in self.sub_strategies}
        
        return weights
        
    def _calculate_ml_weights(self, data: Dict[str, pd.DataFrame], current_regimes: Dict[str, MarketRegime]) -> Dict[str, float]:
        """
        Calculate weights using machine learning optimization.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_regimes: Dictionary of symbol -> detected market regime
            
        Returns:
            Dictionary of strategy_type -> weight
        """
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available. Falling back to regime-based weights.")
            return self._calculate_regime_weights(current_regimes)
        
        # Check if models need training/updating
        self._update_ml_models(data, current_regimes)
        
        # Get features for prediction
        features = self._extract_ml_features(data, current_regimes)
        
        # If no features or models, fall back to regime-based weights
        if not features or not self.ml_models:
            return self._calculate_regime_weights(current_regimes)
        
        # Get primary symbol
        primary_symbol = list(data.keys())[0] if data else None
        if not primary_symbol or primary_symbol not in self.ml_models:
            return self._calculate_regime_weights(current_regimes)
        
        try:
            # Get model and scaler
            model = self.ml_models[primary_symbol]
            scaler = self.feature_scalers.get(primary_symbol)
            
            # Preprocess features
            X = np.array(features).reshape(1, -1)
            if scaler:
                X = scaler.transform(X)
            
            # Predict weights
            raw_weights = model.predict(X)[0]
            
            # Convert to dictionary and apply minimum weight
            min_weight = self.parameters['min_strategy_weight']
            weights = {}
            
            for i, strategy_type in enumerate(self.sub_strategies):
                if i < len(raw_weights):
                    weights[strategy_type] = max(min_weight, raw_weights[i])
                else:
                    weights[strategy_type] = min_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for strategy_type in weights:
                    weights[strategy_type] /= total_weight
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating ML weights: {str(e)}")
            return self._calculate_regime_weights(current_regimes)
        
    def _calculate_adaptive_weights(self, data: Dict[str, pd.DataFrame], current_regimes: Dict[str, MarketRegime]) -> Dict[str, float]:
        """
        Calculate weights using an adaptive approach that combines multiple methods.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_regimes: Dictionary of symbol -> detected market regime
            
        Returns:
            Dictionary of strategy_type -> weight
        """
        # Calculate weights using different methods
        performance_weights = self._calculate_performance_weights()
        regime_weights = self._calculate_regime_weights(current_regimes)
        
        # Try to use ML weights if available
        if ML_AVAILABLE and self.parameters['use_ml_optimization']:
            ml_weights = self._calculate_ml_weights(data, current_regimes)
        else:
            ml_weights = None
        
        # Combine weights with appropriate balance
        # If we have a lot of performance data, weight it more heavily
        perf_data_quality = min(1.0, len(self.strategy_performance) / len(self.sub_strategies))
        
        # Define adaptive weighting factors
        if ml_weights:
            perf_factor = 0.3 * perf_data_quality
            regime_factor = 0.3
            ml_factor = 0.4
        else:
            perf_factor = 0.4 * perf_data_quality
            regime_factor = 0.6
            ml_factor = 0.0
        
        # Combine weights
        combined_weights = {}
        min_weight = self.parameters['min_strategy_weight']
        
        for strategy_type in self.sub_strategies:
            combined_weights[strategy_type] = (
                performance_weights.get(strategy_type, min_weight) * perf_factor +
                regime_weights.get(strategy_type, min_weight) * regime_factor
            )
            
            if ml_weights:
                combined_weights[strategy_type] += ml_weights.get(strategy_type, min_weight) * ml_factor
        
        # Normalize weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            for strategy_type in combined_weights:
                combined_weights[strategy_type] /= total_weight
        
        return combined_weights
    
    def _update_ml_models(self, data: Dict[str, pd.DataFrame], current_regimes: Dict[str, MarketRegime]) -> None:
        """
        Update ML models for strategy weight optimization.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_regimes: Dictionary of symbol -> detected market regime
        """
        if not ML_AVAILABLE or not self.parameters['use_ml_optimization']:
            return
            
        # Ensure directory exists
        model_dir = self.parameters['model_directory']
        os.makedirs(model_dir, exist_ok=True)
        
        # Update models for each symbol
        for symbol, ohlcv_data in data.items():
            # Check if model needs updating
            current_time = pd.Timestamp.now()
            last_update = self.last_training_time.get(symbol, pd.Timestamp.min)
            
            # Get retraining interval in days
            retraining_interval = self.parameters['retraining_interval']
            
            # Check if retraining is needed
            needs_retraining = (
                symbol not in self.ml_models or
                (current_time - last_update).days >= retraining_interval
            )
            
            # Skip if no retraining needed
            if not needs_retraining:
                continue
                
            # Check if we have enough performance data for training
            if not self.strategy_performance or len(self.strategy_performance) < 2:
                logger.info(f"Insufficient performance data for training model for {symbol}")
                continue
                
            # Check if we have enough signal history
            if symbol not in self.signal_history or len(self.signal_history[symbol]) < 30:
                logger.info(f"Insufficient signal history for training model for {symbol}")
                continue
                
            try:
                # Train model
                logger.info(f"Training ML model for {symbol}")
                self._train_ml_model(symbol, ohlcv_data, current_regimes.get(symbol, MarketRegime.UNKNOWN))
                
                # Update last training time
                self.last_training_time[symbol] = current_time
                
                logger.info(f"Successfully trained ML model for {symbol}")
            except Exception as e:
                logger.error(f"Error training ML model for {symbol}: {str(e)}")
    
    def _train_ml_model(self, symbol: str, data: pd.DataFrame, regime: MarketRegime) -> None:
        """
        Train an ML model for strategy weight optimization.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV DataFrame
            regime: Current market regime
        """
        if not ML_AVAILABLE:
            return
            
        # Prepare feature data
        X, y = self._prepare_training_data(symbol, data, regime)
        
        if X is None or y is None or len(X) < 10:
            logger.warning(f"Insufficient data for ML training for {symbol}")
            return
            
        # Split data for training and testing
        try:
            # Use time series split for financial data
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                break  # Just use the last split
        except Exception:
            # Fall back to regular train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save scaler
        self.feature_scalers[symbol] = scaler
        
        # Select model type
        model_type = self.parameters['ml_model_type']
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=42, n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42
            )
        elif model_type == 'elastic_net':
            model = ElasticNet(
                alpha=0.1, l1_ratio=0.5,
                max_iter=1000, random_state=42
            )
        else:  # Default to random forest
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=42, n_jobs=-1
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        logger.info(f"Model for {symbol} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}, MSE: {mse:.4f}")
        
        # Save model
        self.ml_models[symbol] = model
        
        # Persist model if configured
        if self.parameters['save_ml_models'] and JOBLIB_AVAILABLE:
            model_path = os.path.join(
                self.parameters['model_directory'],
                f"{symbol}_weight_model.joblib"
            )
            try:
                joblib.dump(model, model_path)
                logger.info(f"Saved model to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
    
    def _prepare_training_data(self, symbol: str, data: pd.DataFrame, regime: MarketRegime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare training data for ML model.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV DataFrame
            regime: Current market regime
            
        Returns:
            Tuple of (features, targets) arrays or (None, None) if insufficient data
        """
        if symbol not in self.signal_history or not self.strategy_performance:
            return None, None
            
        signals = self.signal_history[symbol]
        
        if len(signals) < 10:
            return None, None
            
        # Prepare feature matrix and target vector
        X_list = []
        y_list = []
        
        # For each historical signal, create a sample
        for i, signal_group in enumerate(signals):
            if i < 5:  # Skip the first few signals as we need history
                continue
                
            # Extract features for this signal
            try:
                # Get the timestamp for this signal group
                timestamp = signal_group.get('timestamp', None)
                if timestamp is None:
                    continue
                    
                # Convert to pd.Timestamp if string
                if isinstance(timestamp, str):
                    timestamp = pd.Timestamp(timestamp)
                    
                # Find the corresponding data
                idx = data.index.get_indexer([timestamp], method='nearest')[0]
                if idx < 50:  # Need enough history
                    continue
                    
                # Extract signal data
                signals_dict = signal_group.get('signals', {})
                strategies = list(signals_dict.keys())
                
                # Extract targets (next period returns)
                returns = {}
                for strategy_type, signal in signals_dict.items():
                    if 'realized_return' in signal:
                        returns[strategy_type] = float(signal['realized_return'])
                    elif 'pnl_pips' in signal:
                        returns[strategy_type] = float(signal['pnl_pips']) / 100.0  # Normalize pips
                    
                # Skip if no return data
                if not returns:
                    continue
                    
                # Create features
                features = self._extract_ml_features(data.iloc[:idx+1], {symbol: regime})
                
                # Skip if insufficient features
                if not features:
                    continue
                    
                # Create target array (optimal weights)
                all_returns = np.array([returns.get(s, 0.0) for s in self.sub_strategies])
                
                # Skip if all returns are zero
                if np.sum(np.abs(all_returns)) < 0.0001:
                    continue
                    
                # Normalize returns to sum to 1 (for weighting)
                # Handle both positive and negative returns
                pos_returns = np.maximum(all_returns, 0)
                if np.sum(pos_returns) > 0:
                    target = pos_returns / np.sum(pos_returns)
                else:
                    # If all negative, invert and normalize
                    neg_returns = np.minimum(all_returns, 0)
                    target = 1.0 + neg_returns / np.sum(np.abs(neg_returns))
                    target = target / np.sum(target)  # Renormalize
                
                X_list.append(features)
                y_list.append(target)
                
            except Exception as e:
                logger.error(f"Error preparing training sample: {str(e)}")
                continue
                
        # Convert to numpy arrays
        if not X_list or not y_list:
            return None, None
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _extract_ml_features(self, data: Dict[str, pd.DataFrame], current_regimes: Dict[str, MarketRegime]) -> List[float]:
        """
        Extract features for ML model.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_regimes: Dictionary of symbol -> detected market regime
            
        Returns:
            List of features
        """
        features = []
        
        # Process each symbol
        for symbol, ohlcv_data in data.items():
            if len(ohlcv_data) < 50:  # Need enough history
                continue
                
            # Extract features from the allowed feature list
            allowed_features = self.parameters['prediction_features']
            
            # Volatility features
            if 'volatility' in allowed_features:
                # Calculate volatility (standard deviation of returns)
                returns = ohlcv_data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
                features.append(volatility)
                
                # ATR-based volatility
                high = ohlcv_data['high']
                low = ohlcv_data['low']
                close = ohlcv_data['close']
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                atr_pct = atr / close.iloc[-1]
                features.append(atr_pct)
            
            # Trend strength features
            if 'trend_strength' in allowed_features:
                # Moving average relationships
                sma20 = ohlcv_data['close'].rolling(20).mean().iloc[-1]
                sma50 = ohlcv_data['close'].rolling(50).mean().iloc[-1]
                sma_ratio = sma20 / sma50 - 1.0  # Deviation from longer MA
                features.append(sma_ratio)
                
                # Directional movement
                close = ohlcv_data['close']
                price_change_10 = (close.iloc[-1] / close.iloc[-10] - 1.0) * 100
                price_change_20 = (close.iloc[-1] / close.iloc[-20] - 1.0) * 100
                features.append(price_change_10)
                features.append(price_change_20)
            
            # Range-bound features
            if 'range_bound' in allowed_features:
                # Price range relative to recent history
                high_20 = ohlcv_data['high'].rolling(20).max().iloc[-1]
                low_20 = ohlcv_data['low'].rolling(20).min().iloc[-1]
                price_range = (high_20 - low_20) / low_20
                features.append(price_range)
                
                # Distance from center of range
                current = ohlcv_data['close'].iloc[-1]
                range_center = (high_20 + low_20) / 2
                range_pos = (current - range_center) / (high_20 - low_20) if (high_20 - low_20) > 0 else 0
                features.append(range_pos)
            
            # Momentum features
            if 'momentum' in allowed_features:
                # RSI (simplified)
                returns = ohlcv_data['close'].diff()
                up = returns.clip(lower=0)
                down = -returns.clip(upper=0)
                ma_up = up.rolling(14).mean()
                ma_down = down.rolling(14).mean()
                rsi = 100 - (100 / (1 + ma_up / ma_down))
                features.append(rsi.iloc[-1] / 100.0)  # Normalize to 0-1
                
                # ROC (Rate of Change)
                roc = ohlcv_data['close'].pct_change(10).iloc[-1]
                features.append(roc)
            
            # Time-based features
            current_time = ohlcv_data.index[-1]
            
            if 'day_of_week' in allowed_features:
                # Day of week (0-6, Monday to Sunday)
                day_of_week = current_time.dayofweek / 6.0  # Normalize to 0-1
                features.append(day_of_week)
                
            if 'hour_of_day' in allowed_features:
                # Hour of day (0-23)
                hour_of_day = current_time.hour / 23.0  # Normalize to 0-1
                features.append(hour_of_day)
            
            # Previous performance features
            if 'prev_performance' in allowed_features and self.strategy_performance:
                # Get aggregate performance metrics across strategies
                win_rates = []
                profit_factors = []
                sharpe_ratios = []
                
                for strategy_type, perf in self.strategy_performance.items():
                    if 'win_rate' in perf:
                        win_rates.append(perf['win_rate'])
                    if 'profit_factor' in perf:
                        profit_factors.append(min(10.0, perf['profit_factor']))  # Cap at 10.0
                    if 'sharpe_ratio' in perf:
                        sharpe_ratios.append(min(5.0, perf['sharpe_ratio']))  # Cap at 5.0
                
                # Add aggregate performance metrics
                if win_rates:
                    features.append(np.mean(win_rates))
                else:
                    features.append(0.5)  # Default
                    
                if profit_factors:
                    features.append(np.mean(profit_factors) / 10.0)  # Normalize to 0-1
                else:
                    features.append(0.1)  # Default
                    
                if sharpe_ratios:
                    features.append(np.mean(sharpe_ratios) / 5.0)  # Normalize to 0-1
                else:
                    features.append(0.1)  # Default
            
            # Market regime features
            if symbol in current_regimes:
                regime = current_regimes[symbol]
                # Convert regime to numeric features (one-hot)
                regimes = list(MarketRegime)
                regime_idx = regimes.index(regime) if regime in regimes else -1
                
                # One-hot encode the regime
                for i in range(len(regimes)):
                    features.append(1.0 if i == regime_idx else 0.0)
            
            # Only process the first symbol if we have multiple
            break
        
        return features
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals using the ensemble of sub-strategies.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current time
            
        Returns:
            Dictionary of signal information
        """
        signals = {}
        
        # Check for active trading session
        if not self.is_active_trading_session(current_time):
            return signals
            
        # Detect market regimes for each symbol
        current_regimes = {}
        for symbol, ohlcv in data.items():
            if len(ohlcv) < 50:  # Need enough history
                continue
                
            regime = self._detect_market_regime(ohlcv, symbol)
            current_regimes[symbol] = regime
            
            # Update regime history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
                
            self.regime_history[symbol].append({
                'regime': regime,
                'time': current_time
            })
            
            # Trim history if too long
            max_history = 100
            if len(self.regime_history[symbol]) > max_history:
                self.regime_history[symbol] = self.regime_history[symbol][-max_history:]
        
        # Calculate strategy weights
        strategy_weights = self._calculate_strategy_weights(data, current_regimes)
        
        # Update weight history
        for strategy_type, weight in strategy_weights.items():
            if strategy_type not in self.weight_history:
                self.weight_history[strategy_type] = []
                
            self.weight_history[strategy_type].append({
                'weight': weight,
                'time': current_time
            })
            
            # Trim history if too long
            max_history = 100
            if len(self.weight_history[strategy_type]) > max_history:
                self.weight_history[strategy_type] = self.weight_history[strategy_type][-max_history:]
        
        # Collect signals from all sub-strategies
        sub_signals = {}
        
        for strategy_type, strategy in self.sub_strategies.items():
            try:
                # Weight to apply to this strategy
                weight = strategy_weights.get(strategy_type, 0.0)
                
                # Skip if weight is too low
                if weight < self.parameters['min_strategy_weight']:
                    continue
                    
                # Generate signals from this strategy
                strategy_signals = strategy.generate_signals(data, current_time)
                
                # Store signals with their weights
                if strategy_signals:
                    for symbol, signal in strategy_signals.items():
                        if symbol not in sub_signals:
                            sub_signals[symbol] = []
                            
                        # Add strategy type and weight information
                        signal['strategy_type'] = strategy_type
                        signal['weight'] = weight
                        
                        # Add to signals list
                        sub_signals[symbol].append(signal)
                        
            except Exception as e:
                logger.error(f"Error generating signals from {strategy_type}: {str(e)}")
        
        # Process collected signals
        for symbol, symbol_signals in sub_signals.items():
            if not symbol_signals:
                continue
                
            # Separate signals by direction
            long_signals = [s for s in symbol_signals if s['direction'] == TradeDirection.LONG]
            short_signals = [s for s in symbol_signals if s['direction'] == TradeDirection.SHORT]
            
            # Check for direction conflicts
            if long_signals and short_signals:
                # Resolve conflicts based on configured method
                conflict_method = self.parameters['conflict_resolution']
                
                if conflict_method == 'strongest':
                    # Use direction with the strongest signal
                    strongest_long = max(long_signals, key=lambda x: x['strength'] * x['weight']) if long_signals else None
                    strongest_short = max(short_signals, key=lambda x: x['strength'] * x['weight']) if short_signals else None
                    
                    if strongest_long and strongest_short:
                        long_score = strongest_long['strength'] * strongest_long['weight']
                        short_score = strongest_short['strength'] * strongest_short['weight']
                        
                        if long_score > short_score:
                            active_signals = long_signals
                            conflict_resolution = "strongest signal (long)"
                        else:
                            active_signals = short_signals
                            conflict_resolution = "strongest signal (short)"
                    elif strongest_long:
                        active_signals = long_signals
                        conflict_resolution = "only long signals"
                    else:
                        active_signals = short_signals
                        conflict_resolution = "only short signals"
                    
                elif conflict_method == 'majority':
                    # Use direction with the most signals
                    if len(long_signals) > len(short_signals):
                        active_signals = long_signals
                        conflict_resolution = "majority direction (long)"
                    elif len(short_signals) > len(long_signals):
                        active_signals = short_signals
                        conflict_resolution = "majority direction (short)"
                    else:
                        # Tie, use strongest signal
                        strongest_long = max(long_signals, key=lambda x: x['strength'] * x['weight']) if long_signals else None
                        strongest_short = max(short_signals, key=lambda x: x['strength'] * x['weight']) if short_signals else None
                        
                        if strongest_long and strongest_short:
                            long_score = strongest_long['strength'] * strongest_long['weight']
                            short_score = strongest_short['strength'] * strongest_short['weight']
                            
                            if long_score > short_score:
                                active_signals = long_signals
                                conflict_resolution = "tie, strongest signal (long)"
                            else:
                                active_signals = short_signals
                                conflict_resolution = "tie, strongest signal (short)"
                        elif strongest_long:
                            active_signals = long_signals
                            conflict_resolution = "tie, only long signals"
                        else:
                            active_signals = short_signals
                            conflict_resolution = "tie, only short signals"
                
                elif conflict_method == 'consensus':
                    # Cancel out if no consensus
                    active_signals = []
                    conflict_resolution = "no consensus, signals canceled"
                    
                else:  # Default to strongest
                    strongest_long = max(long_signals, key=lambda x: x['strength'] * x['weight']) if long_signals else None
                    strongest_short = max(short_signals, key=lambda x: x['strength'] * x['weight']) if short_signals else None
                    
                    if strongest_long and strongest_short:
                        long_score = strongest_long['strength'] * strongest_long['weight']
                        short_score = strongest_short['strength'] * strongest_short['weight']
                        
                        if long_score > short_score:
                            active_signals = long_signals
                            conflict_resolution = "default, strongest signal (long)"
                        else:
                            active_signals = short_signals
                            conflict_resolution = "default, strongest signal (short)"
                    elif strongest_long:
                        active_signals = long_signals
                        conflict_resolution = "default, only long signals"
                    else:
                        active_signals = short_signals
                        conflict_resolution = "default, only short signals"
            
            elif long_signals:
                active_signals = long_signals
                conflict_resolution = "only long signals present"
            elif short_signals:
                active_signals = short_signals
                conflict_resolution = "only short signals present"
            else:
                active_signals = []
                conflict_resolution = "no signals"
            
            # Skip if no active signals
            if not active_signals:
                continue
                
            # Combine signals based on weights
            direction = active_signals[0]['direction']
            total_weight = sum(s['weight'] for s in active_signals)
            weighted_strength = sum(s['strength'] * s['weight'] for s in active_signals) / total_weight if total_weight > 0 else 0
            
            # Check if combined strength exceeds threshold
            if weighted_strength < self.parameters['signal_threshold']:
                continue
                
            # Calculate weighted entry price
            weighted_entry = sum(s['entry_price'] * s['weight'] for s in active_signals) / total_weight if total_weight > 0 else 0
            
            # Calculate weighted stop loss and take profit levels
            # For simplicity, use the levels from the strongest signal
            strongest_signal = max(active_signals, key=lambda x: x['strength'] * x['weight'])
            stop_loss = strongest_signal['stop_loss']
            take_profit = strongest_signal['take_profit']
            
            # Generate unique signal ID
            signal_id = f"algo_{symbol}_{current_time.strftime('%Y%m%d%H%M%S')}")
            
            # Create the combined signal
            combined_signal = {
                'id': signal_id,
                'symbol': symbol,
                'direction': direction,
                'strength': weighted_strength,
                'entry_price': weighted_entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sub_strategies': [s['strategy_type'] for s in active_signals],
                'weights': {s['strategy_type']: s['weight'] for s in active_signals},
                'regime': current_regimes.get(symbol, MarketRegime.UNKNOWN).name,
                'conflict_resolution': conflict_resolution,
                'time': current_time
            }
            
            # Add the combined signal
            signals[symbol] = combined_signal
            
            # Store signal history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
                
            # Store sub-signals and combined signal
            self.signal_history[symbol].append({
                'timestamp': current_time,
                'combined': combined_signal,
                'signals': {s['strategy_type']: s for s in active_signals},
                'regime': current_regimes.get(symbol, MarketRegime.UNKNOWN).name
            })
            
            # Trim history if too long
            max_history = 1000
            if len(self.signal_history[symbol]) > max_history:
                self.signal_history[symbol] = self.signal_history[symbol][-max_history:]
            
            # Publish signal event
            EventBus.get_instance().publish('algorithmic_meta_signal', {
                'id': signal_id,
                'symbol': symbol,
                'direction': direction.name,
                'strength': weighted_strength,
                'sub_strategies': [s['strategy_type'] for s in active_signals],
                'regime': current_regimes.get(symbol, MarketRegime.UNKNOWN).name,
                'time': current_time.isoformat()
            })
        
        return signals
        
    def _initialize_sub_strategies(self) -> None:
        """Initialize all sub-strategies from the factory."""
        for strategy_type in self.parameters['sub_strategies']:
            try:
                strategy = self.strategy_factory.create_strategy(strategy_type)
                if strategy:
                    self.sub_strategies[strategy_type] = strategy
                    logger.info(f"Initialized sub-strategy: {strategy_type}")
                else:
                    logger.warning(f"Failed to create sub-strategy: {strategy_type}")
            except Exception as e:
                logger.error(f"Error initializing sub-strategy {strategy_type}: {str(e)}")
                
        # Initialize equal weights if no weights exist
        if not self.strategy_weights and self.sub_strategies:
            equal_weight = 1.0 / len(self.sub_strategies)
            self.strategy_weights = {strategy_type: equal_weight for strategy_type in self.sub_strategies}
            
        logger.info(f"Initialized {len(self.sub_strategies)} sub-strategies")
