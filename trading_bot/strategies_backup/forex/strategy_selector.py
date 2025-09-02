#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Strategy Selector

This module implements intelligent selection of forex trading strategies based on:
1. Current market conditions (regime detection)
2. Time-awareness (trading sessions, economic calendars)
3. Risk tolerance parameters
4. Historical performance in similar conditions

It serves as a decision layer that implements the "Principles & Selection" logic
for optimal forex strategy deployment.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, time, timedelta
import pytz
import joblib  # For loading ML models
import requests  # For economic calendar API access
import json
from collections import defaultdict

from trading_bot.strategies.strategy_template import MarketRegime, TimeFrame
from trading_bot.strategies.base.forex_base import ForexSession

# Import all forex strategies
from trading_bot.strategies.forex.trend_following_strategy import ForexTrendFollowingStrategy
from trading_bot.strategies.forex.range_trading_strategy import ForexRangeTradingStrategy
from trading_bot.strategies.forex.breakout_strategy import ForexBreakoutStrategy
from trading_bot.strategies.forex.momentum_strategy import ForexMomentumStrategy
from trading_bot.strategies.forex.scalping_strategy import ForexScalpingStrategy
from trading_bot.strategies.forex.swing_trading_strategy import ForexSwingTradingStrategy

logger = logging.getLogger(__name__)

# Risk tolerance levels
class RiskTolerance:
    LOW = "low"           # Conservative - smaller positions, tighter stops
    MEDIUM = "medium"     # Balanced - moderate position sizing
    HIGH = "high"         # Aggressive - larger positions, wider stops

class ForexStrategySelector:
    """
    Intelligence layer for selecting optimal forex strategies based on
    market conditions, time, and risk parameters.
    """
    
    def __init__(self, 
                 risk_tolerance: str = RiskTolerance.MEDIUM,
                 time_zone: str = "UTC",
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy selector with given parameters.
        
        Args:
            risk_tolerance: Risk tolerance level (low, medium, high)
            time_zone: Time zone for session-based decisions
            parameters: Additional configuration parameters
        """
        self.risk_tolerance = risk_tolerance
        self.time_zone = pytz.timezone(time_zone)
        self.parameters = parameters or {}
        
        # Initialize strategy compatibility knowledge base
        self.strategy_compatibility = {
            # Strategy compatibility scores by market regime
            # Scores are 0.0-1.0 where 1.0 is perfect match
            "forex_trend_following": {
                MarketRegime.BULL_TREND: 0.95,  # Increased based on testing
                MarketRegime.BEAR_TREND: 0.80,
                MarketRegime.CONSOLIDATION: 0.30, 
                MarketRegime.HIGH_VOLATILITY: 0.65,
                MarketRegime.LOW_VOLATILITY: 0.96,  # Increased based on testing
                MarketRegime.UNKNOWN: 0.50
            },
            "forex_range_trading": {
                MarketRegime.BULL_TREND: 0.30,
                MarketRegime.BEAR_TREND: 0.35,
                MarketRegime.CONSOLIDATION: 0.96,  # Increased based on testing
                MarketRegime.HIGH_VOLATILITY: 0.95,  # Increased based on testing
                MarketRegime.LOW_VOLATILITY: 0.95,  # Increased based on testing
                MarketRegime.UNKNOWN: 0.50
            },
            "forex_breakout": {
                MarketRegime.BULL_TREND: 0.65,
                MarketRegime.BEAR_TREND: 0.97,  # Increased based on testing
                MarketRegime.CONSOLIDATION: 0.60,
                MarketRegime.HIGH_VOLATILITY: 0.90,
                MarketRegime.LOW_VOLATILITY: 0.40, 
                MarketRegime.UNKNOWN: 0.55
            },
            "forex_momentum": {
                MarketRegime.BULL_TREND: 0.80,
                MarketRegime.BEAR_TREND: 0.95,  # Increased based on testing
                MarketRegime.CONSOLIDATION: 0.40,
                MarketRegime.HIGH_VOLATILITY: 0.85,
                MarketRegime.LOW_VOLATILITY: 0.30,
                MarketRegime.UNKNOWN: 0.60
            },
            "forex_scalping": {
                MarketRegime.BULL_TREND: 0.60,
                MarketRegime.BEAR_TREND: 0.60,
                MarketRegime.CONSOLIDATION: 0.50, 
                MarketRegime.HIGH_VOLATILITY: 0.90,  # Increased based on testing
                MarketRegime.LOW_VOLATILITY: 0.50,
                MarketRegime.UNKNOWN: 0.60
            },
            "forex_swing_trading": {
                MarketRegime.BULL_TREND: 0.90,  # Increased based on testing
                MarketRegime.BEAR_TREND: 0.90,  # Increased based on testing
                MarketRegime.CONSOLIDATION: 0.60, 
                MarketRegime.HIGH_VOLATILITY: 0.70,
                MarketRegime.LOW_VOLATILITY: 0.50,
                MarketRegime.UNKNOWN: 0.65
            }
        }
        
        # Session preferences for strategies
        self.session_preferences = {
            # Each session has a list of preferred strategies
            ForexSession.SYDNEY: ["forex_range_trading", "forex_scalping"],
            ForexSession.TOKYO: ["forex_range_trading", "forex_momentum"],
            ForexSession.LONDON: ["forex_trend_following", "forex_breakout", "forex_momentum"],
            ForexSession.NEWYORK: ["forex_breakout", "forex_momentum", "forex_swing_trading"],
            ForexSession.LONDON_NEWYORK_OVERLAP: ["forex_breakout", "forex_scalping", "forex_momentum"]
        }
        
        # Risk adjustments
        self.risk_adjustments = {
            RiskTolerance.LOW: {
                "position_size_multiplier": 0.7,
                "stop_loss_multiplier": 0.8,   # Tighter stops for lower risk
                "take_profit_multiplier": 1.2,  # Closer take profits
                "max_trades_per_session": 3,
                "max_risk_per_trade_pct": 1.0,
                "max_risk_per_day_pct": 3.0,
                "preferred_strategies": ["forex_range_trading", "forex_trend_following"]
            },
            RiskTolerance.MEDIUM: {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "max_trades_per_session": 5,
                "max_risk_per_trade_pct": 2.0,
                "max_risk_per_day_pct": 6.0,
                "preferred_strategies": ["forex_momentum", "forex_swing_trading"]
            },
            RiskTolerance.HIGH: {
                "position_size_multiplier": 1.3,
                "stop_loss_multiplier": 1.2,  # Wider stops for higher risk
                "take_profit_multiplier": 0.8, # Further take profits
                "max_trades_per_session": 8,
                "max_risk_per_trade_pct": 3.0,
                "max_risk_per_day_pct": 10.0,
                "preferred_strategies": ["forex_breakout", "forex_scalping"]
            }
        }
        
        # Initialize ML model components
        self.ml_model = None
        self.ml_scaler = None
        self.ml_feature_columns = None
        self.ml_model_loaded = False
        self.ml_model_package = None
        
        # Try to load ML model
        try:
            # Get model path from parameters or use default path
            ml_model_path = self.parameters.get('ml_model_path', os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'ml/models/forex_regime_classifier.joblib'
            ))
            
            if os.path.exists(ml_model_path):
                # Load the model package (contains model, scaler, and metadata)
                self.ml_model_package = joblib.load(ml_model_path)
                
                # Extract components from the package
                if isinstance(self.ml_model_package, dict):
                    # New format with full package
                    self.ml_model = self.ml_model_package.get('model')
                    self.ml_scaler = self.ml_model_package.get('scaler')
                    self.ml_feature_columns = self.ml_model_package.get('feature_columns')
                    training_date = self.ml_model_package.get('training_date', 'unknown')
                    model_accuracy = self.ml_model_package.get('accuracy', 0.0)
                    
                    logger.info(f"Loaded ML model package for regime detection (trained: {training_date}, "
                               f"accuracy: {model_accuracy:.2f})")
                else:
                    # Legacy format with just the model
                    self.ml_model = self.ml_model_package
                    logger.info(f"Loaded legacy ML model for regime detection from {ml_model_path}")
                
                self.ml_model_loaded = self.ml_model is not None
            else:
                logger.warning(f"ML model not found at {ml_model_path}. Using traditional regime detection.")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}. Using traditional regime detection.")
            
        # Economic calendar integration
        # Initialize with empty calendar
        self.economic_calendar = {}
        self.last_calendar_update = None
        self.calendar_update_frequency = timedelta(hours=6)  # Update every 6 hours
        
        # API configuration for economic calendar
        self.economic_api_key = self.parameters.get('economic_calendar_api_key', '')
        self.economic_api_url = self.parameters.get('economic_calendar_api_url', 
                              'https://api.example.com/economic-calendar')  # Replace with actual API
        
        # Initialize historical performance tracking
        # Structure: {strategy_name: {market_regime: [performance_scores]}}
        self.strategy_performance = {
            "forex_trend_following": {regime: [] for regime in MarketRegime},
            "forex_range_trading": {regime: [] for regime in MarketRegime},
            "forex_breakout": {regime: [] for regime in MarketRegime},
            "forex_momentum": {regime: [] for regime in MarketRegime},
            "forex_scalping": {regime: [] for regime in MarketRegime},
            "forex_swing_trading": {regime: [] for regime in MarketRegime}
        }
        
        # Time-based strategy preferences
        # Mapping forex sessions to preferred strategies
        self.session_strategy_preferences = {
            ForexSession.SYDNEY: ["forex_range_trading", "forex_breakout"],  # Lower volatility, look for ranges and breakouts
            ForexSession.TOKYO: ["forex_range_trading", "forex_momentum"],   # Asian session often range-bound
            ForexSession.LONDON: ["forex_trend_following", "forex_momentum"], # London often creates trends
            ForexSession.NEWYORK: ["forex_breakout", "forex_momentum"],      # NY can be volatile with breakouts
            ForexSession.LONDON_NEWYORK_OVERLAP: ["forex_breakout", "forex_trend_following"]  # Highest volatility session
        }
        
        # Risk tolerance adjustments
        # How risk tolerance affects strategy selection and parameter tuning
        self.risk_adjustments = {
            RiskTolerance.LOW: {
                "position_size_multiplier": 0.7,  # Reduce position size
                "stop_loss_multiplier": 0.8,      # Tighter stops
                "take_profit_multiplier": 1.2,    # More conservative targets
                "preferred_strategies": ["forex_range_trading", "forex_trend_following"],  # Less volatile strategies
                "avoid_strategies": []  # No strategies to explicitly avoid
            },
            RiskTolerance.MEDIUM: {
                "position_size_multiplier": 1.0,  # Standard position size
                "stop_loss_multiplier": 1.0,      # Standard stops
                "take_profit_multiplier": 1.0,    # Standard targets
                "preferred_strategies": [],       # No specific preferences
                "avoid_strategies": []            # No strategies to avoid
            },
            RiskTolerance.HIGH: {
                "position_size_multiplier": 1.3,  # Larger positions
                "stop_loss_multiplier": 1.2,      # Wider stops for more room
                "take_profit_multiplier": 0.8,    # More aggressive targets
                "preferred_strategies": ["forex_breakout", "forex_momentum"],  # More volatile strategies
                "avoid_strategies": []            # No strategies to explicitly avoid
            }
        }
        
        logger.info(f"Initialized ForexStrategySelector with risk tolerance: {risk_tolerance}")
    
    def select_optimal_strategy(self, 
                              market_data: Dict[str, pd.DataFrame],
                              current_time: datetime,
                              detected_regime: Optional[MarketRegime] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Select the optimal forex strategy based on current conditions.
        
        Args:
            market_data: Dictionary of symbol to OHLCV dataframes
            current_time: Current datetime (timezone-aware)
            detected_regime: Detected market regime, or None to auto-detect
            
        Returns:
            Tuple containing (strategy_name, optimized_parameters)
        """
        # Standardize timezone
        if current_time.tzinfo is None:
            current_time = self.time_zone.localize(current_time)
        else:
            current_time = current_time.astimezone(self.time_zone)
        
        # 1. Detect market regime if not provided
        if detected_regime is None:
            detected_regime = self._detect_market_regime(market_data)
        
        logger.info(f"Detected market regime: {detected_regime.name}")
        
        # Identify active trading sessions
        active_sessions = self._identify_active_sessions(current_time)
        logger.info(f"Active forex sessions: {[s.name for s in active_sessions]}")
        
        # Check for significant economic news
        economic_data = self._check_economic_calendar(current_time)
        has_high_impact_news = economic_data['has_high_impact']
        
        # Get base scores from regime compatibility
        scores = self._get_regime_based_scores(detected_regime)
        
        # Adjust scores based on trading session
        scores = self._apply_session_preferences(scores, active_sessions)
        
        # Adjust scores based on risk tolerance
        scores = self._apply_risk_adjustments(scores)
        
        # Adjust based on historical performance in similar conditions
        scores = self._apply_historical_performance_adjustments(scores, detected_regime)
        
        # Adjust for economic news
        if has_high_impact_news:
            scores = self._apply_news_adjustments(scores)
            logger.info("Adjusted strategy selection for high-impact economic news")
        
        # Select the highest scoring strategy
        strategy_name = max(scores.items(), key=lambda x: x[1])[0]
        optimal_params = self._get_optimized_parameters(strategy_name, detected_regime, active_sessions, has_high_impact_news)
        
        logger.info(f"Selected optimal strategy: {strategy_name} with score {scores[strategy_name]:.2f}")
        
        return strategy_name, optimal_params
    
    def _detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """
        Detect the current market regime based on price action and indicators.
        Uses machine learning model if available, otherwise falls back to conventional detection.
        
        Args:
            market_data: Dictionary of symbol to OHLCV dataframes
            
        Returns:
            Detected MarketRegime
        """
        # Default to unknown if we can't determine
        if not market_data or len(market_data) == 0:
            logger.warning("No market data provided for regime detection")
            return MarketRegime.UNKNOWN
        
        # Take the first symbol's data
        symbol = list(market_data.keys())[0]
        data = market_data[symbol]
        
        # Need enough data to compute indicators
        if len(data) < 30:  # Need at least 30 bars
            logger.warning("Not enough data to detect market regime")
            return MarketRegime.UNKNOWN
        
        # Check if we can use ML model
        if self.ml_model_loaded and self.ml_model is not None:
            try:
                logger.debug(f"Using ML model for regime detection on {symbol}")
                
                # Extract features for ML prediction
                features = self._extract_ml_features(data)
                
                # If we have a scaler and feature columns defined, use them
                if hasattr(self, 'ml_scaler') and self.ml_scaler is not None and hasattr(self, 'ml_feature_columns'):
                    # Create DataFrame with proper column names for consistent order
                    features_df = pd.DataFrame([features], columns=self.ml_feature_columns)
                    # Standardize features
                    features_standardized = self.ml_scaler.transform(features_df)
                    # Make prediction
                    regime_prediction = self.ml_model.predict(features_standardized)[0]
                else:
                    # Fall back to direct prediction
                    regime_prediction = self.ml_model.predict([features])[0]
                
                # Convert prediction to MarketRegime enum
                # Depending on how the model was trained, we might need to map integers to enum values
                if isinstance(regime_prediction, (int, np.integer)):
                    regime_mapping = {
                        1: MarketRegime.BULL_TREND,
                        2: MarketRegime.BEAR_TREND,
                        3: MarketRegime.CONSOLIDATION,
                        4: MarketRegime.HIGH_VOLATILITY,
                        5: MarketRegime.LOW_VOLATILITY,
                        6: MarketRegime.UNKNOWN
                    }
                    predicted_regime = regime_mapping.get(regime_prediction, MarketRegime.UNKNOWN)
                else:
                    # Directly use the enum
                    predicted_regime = MarketRegime(regime_prediction)
                
                logger.info(f"ML model predicted market regime: {predicted_regime.name}")
                return predicted_regime
            
            except Exception as e:
                logger.warning(f"Error using ML model for regime detection: {e}. Falling back to traditional method.")
                # Continue with traditional method as fallback
        
        # Traditional method using technical indicators
        logger.debug(f"Using traditional regime detection on {symbol}")
        
        # Compute trend indicators
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Calculate moving averages
        data['sma20'] = close.rolling(window=20).mean()
        data['sma50'] = close.rolling(window=50).mean()
        data['sma200'] = close.rolling(window=200).mean()
        
        # Calculate ATR for volatility
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        data['atr_pct'] = (data['atr'] / close) * 100
        
        # Calculate ADX for trend strength
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
        
        # Smooth the directional movement
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / data['atr'])
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / data['atr'])
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        data['adx'] = dx.rolling(window=14).mean()
        
        # RSI for momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['volatility'] = close.rolling(window=20).std()
        
        # Calculate price direction
        data['price_direction'] = (close > data['sma20']).astype(int)
        
        # Price movement
        data['price_change'] = close.pct_change(periods=5) * 100
        
        # Drop NaN values that may have been introduced by calculations
        data = data.dropna()
        
        # If we don't have enough data after calculations, return UNKNOWN
        if len(data) < 5:
            logger.warning("Insufficient data after calculations")
            return MarketRegime.UNKNOWN
        
        # Get most recent values
        current_close = close.iloc[-1]
        current_sma20 = data['sma20'].iloc[-1]
        current_sma50 = data['sma50'].iloc[-1]
        current_adx = data['adx'].iloc[-1]
        current_rsi = data['rsi'].iloc[-1]
        current_volatility = data['volatility'].iloc[-1]
        current_atr_pct = data['atr_pct'].iloc[-1]
        
        # Calculate historical context
        avg_volatility = data['volatility'].rolling(window=30).mean().iloc[-1]
        avg_atr_pct = data['atr_pct'].rolling(window=30).mean().iloc[-1]
        sma20_slope = (data['sma20'].iloc[-1] / data['sma20'].iloc[-6] - 1) * 100
        
        # Evidence-based approach to regime detection
        evidence = {
            MarketRegime.BULL_TREND: 0,
            MarketRegime.BEAR_TREND: 0,
            MarketRegime.CONSOLIDATION: 0,
            MarketRegime.HIGH_VOLATILITY: 0,
            MarketRegime.LOW_VOLATILITY: 0
        }
        
        # Bull Trend evidence
        if current_close > current_sma20 > current_sma50:
            evidence[MarketRegime.BULL_TREND] += 2
        if sma20_slope > 0.3:
            evidence[MarketRegime.BULL_TREND] += 1
        if current_adx > 25 and current_rsi > 50:
            evidence[MarketRegime.BULL_TREND] += 1
        
        # Bear Trend evidence
        if current_close < current_sma20 < current_sma50:
            evidence[MarketRegime.BEAR_TREND] += 2
        if sma20_slope < -0.3:
            evidence[MarketRegime.BEAR_TREND] += 1
        if current_adx > 25 and current_rsi < 50:
            evidence[MarketRegime.BEAR_TREND] += 1
        
        # Consolidation evidence
        if abs(current_close/current_sma50 - 1) < 0.005:
            evidence[MarketRegime.CONSOLIDATION] += 1
        if abs(sma20_slope) < 0.1:
            evidence[MarketRegime.CONSOLIDATION] += 1
        if 40 < current_rsi < 60:
            evidence[MarketRegime.CONSOLIDATION] += 1
        if current_adx < 20:
            evidence[MarketRegime.CONSOLIDATION] += 1
        
        # High Volatility evidence
        if current_atr_pct > avg_atr_pct * 1.5:
            evidence[MarketRegime.HIGH_VOLATILITY] += 2
        if current_volatility > avg_volatility * 1.5:
            evidence[MarketRegime.HIGH_VOLATILITY] += 1
        if abs(data['price_change'].iloc[-1]) > 3.0:
            evidence[MarketRegime.HIGH_VOLATILITY] += 1
        
        # Low Volatility evidence
        if current_atr_pct < avg_atr_pct * 0.6:
            evidence[MarketRegime.LOW_VOLATILITY] += 2
        if current_volatility < avg_volatility * 0.6:
            evidence[MarketRegime.LOW_VOLATILITY] += 1
        if current_adx < 15:
            evidence[MarketRegime.LOW_VOLATILITY] += 1
        
        # Find regime with most evidence
        max_evidence = max(evidence.values())
        
        # If we have clear evidence for a regime, return it
        if max_evidence >= 2:
            # Find all regimes that have this max evidence
            top_regimes = [regime for regime, score in evidence.items() if score == max_evidence]
            
            if len(top_regimes) == 1:
                logger.info(f"Traditional regime detection: {top_regimes[0].name} (evidence: {max_evidence})")
                return top_regimes[0]
            else:
                # In case of a tie, use additional criteria
                if current_adx > 30 and MarketRegime.BULL_TREND in top_regimes:
                    logger.info(f"Traditional regime detection (tiebreak): {MarketRegime.BULL_TREND.name}")
                    return MarketRegime.BULL_TREND
                
                if current_atr_pct > avg_atr_pct * 2 and MarketRegime.HIGH_VOLATILITY in top_regimes:
                    logger.info(f"Traditional regime detection (tiebreak): {MarketRegime.HIGH_VOLATILITY.name}")
                    return MarketRegime.HIGH_VOLATILITY
                
                # Default to first regime in the tie
                logger.info(f"Traditional regime detection (tie): {top_regimes[0].name}")
                return top_regimes[0]
        
        # If no strong evidence, default to UNKNOWN
        logger.info("No clear regime detected, defaulting to UNKNOWN")
        return MarketRegime.UNKNOWN
    
    def _extract_ml_features(self, data: pd.DataFrame) -> List[float]:
        """
        Extract features from price data for ML-based regime detection.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of features for ML model input
        """
        # Calculate technical indicators as features
        # Price-based features
        close = data['close']
        returns = close.pct_change().dropna()
        
        # Moving averages
        sma20 = close.rolling(window=20).mean().iloc[-1]
        sma50 = close.rolling(window=50).mean().iloc[-1]
        sma_ratio = sma20 / sma50 if sma50 != 0 else 1.0
        
        # Volatility measures
        volatility_20 = returns.rolling(window=20).std().iloc[-1] * 100  # Convert to percentage
        volatility_50 = returns.rolling(window=50).std().iloc[-1] * 100
        volatility_ratio = volatility_20 / volatility_50 if volatility_50 != 0 else 1.0
        
        # Momentum indicators
        momentum_10 = (close.iloc[-1] / close.iloc[-10] - 1) * 100
        momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        
        # Trend strength (ADX)
        high = data['high']
        low = data['low']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Calculate ADX
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean().iloc[-1]
        
        # Combine all features
        features = [
            sma_ratio,
            volatility_20,
            volatility_ratio,
            momentum_10,
            momentum_20,
            adx,
            plus_di.iloc[-1],
            minus_di.iloc[-1],
            close.iloc[-1] > sma20,  # Price above/below 20-day SMA
            (high.rolling(window=20).max().iloc[-1] - close.iloc[-1]) / close.iloc[-1] * 100,  # Distance from 20-day high
            (close.iloc[-1] - low.rolling(window=20).min().iloc[-1]) / close.iloc[-1] * 100  # Distance from 20-day low
        ]
        
        return features
    
    def _identify_active_sessions(self, current_time: datetime) -> List[ForexSession]:
        """
        Identify which forex trading sessions are currently active.
        
        Args:
            current_time: Current timezone-aware datetime
            
        Returns:
            List of active ForexSession enums
        """
        # Convert to UTC for consistent session checking
        utc_time = current_time.astimezone(pytz.UTC)
        current_hour = utc_time.hour
        current_minute = utc_time.minute
        current_weekday = utc_time.weekday()  # Monday=0, Sunday=6
        
        # Skip if weekend (forex market closed)
        if current_weekday >= 5:  # Saturday or Sunday
            return []
        
        active_sessions = []
        
        # Sydney: 22:00-07:00 UTC
        if (current_hour >= 22) or (current_hour < 7):
            active_sessions.append(ForexSession.SYDNEY)
            
        # Tokyo: 00:00-09:00 UTC
        if (current_hour >= 0) and (current_hour < 9):
            active_sessions.append(ForexSession.TOKYO)
            
        # London: 08:00-17:00 UTC
        if (current_hour >= 8) and (current_hour < 17):
            active_sessions.append(ForexSession.LONDON)
            
        # New York: 13:00-22:00 UTC
        if (current_hour >= 13) and (current_hour < 22):
            active_sessions.append(ForexSession.NEWYORK)
            
        # London-NY Overlap: 13:00-17:00 UTC (high volatility window)
        if (current_hour >= 13) and (current_hour < 17):
            active_sessions.append(ForexSession.LONDON_NEWYORK_OVERLAP)
            
        return active_sessions
    
    def _check_economic_calendar(self, current_time: datetime) -> Dict[str, Any]:
        """
        Check if there are important economic news releases around the current time.
        Fetches data from economic calendar API if available and caches results.
        
        Args:
            current_time: Current datetime
            
        Returns:
            Dictionary with 'has_high_impact' boolean flag and 'events' list
        """
        # Default return value
        result = {'has_high_impact': False, 'events': []}
        
        # Check if we need to update the calendar data
        calendar_expired = (self.last_calendar_update is None or 
                         current_time - self.last_calendar_update > self.calendar_update_frequency)
        
        # Update calendar if needed
        if calendar_expired:
            try:
                # Only attempt to fetch if we have an API key
                if self.economic_api_key:
                    self._update_economic_calendar(current_time)
                    logger.info("Successfully updated economic calendar data")
                else:
                    logger.debug("No economic calendar API key provided. Skipping update.")
            except Exception as e:
                logger.warning(f"Failed to update economic calendar: {e}")
        
        # Initialize currencies of interest (common forex currencies)
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
        
        # Look for high impact events in the next 24 hours
        look_ahead_window = timedelta(hours=24)
        look_back_window = timedelta(minutes=30)  # Also check recent events that might still affect the market
        
        events_found = []
        has_high_impact = False
        
        # Check cached calendar data for relevant events
        for currency in currencies:
            if currency in self.economic_calendar:
                for event in self.economic_calendar[currency]:
                    event_time = event.get('datetime')
                    impact = event.get('impact', 'low')  # Default to low if not specified
                    
                    # Check if event is within our time window
                    if event_time and (current_time - look_back_window <= event_time <= current_time + look_ahead_window):
                        # Add to our list of events
                        events_found.append(event)
                        
                        # Check if it's high impact
                        if impact.lower() in ['high', 'h', '3']:
                            has_high_impact = True
        
        # Return the result
        result['has_high_impact'] = has_high_impact
        result['events'] = events_found
        
        if has_high_impact:
            logger.info(f"Detected {len(events_found)} economic events, including high-impact events")
        elif events_found:
            logger.debug(f"Detected {len(events_found)} economic events, but none are high-impact")
        
        return result
    
    def _update_economic_calendar(self, current_time: datetime) -> None:
        """
        Fetch the latest economic calendar data from the API.
        
        Args:
            current_time: Current datetime for reference
        """
        # Calculate date range for API request (next 7 days)
        start_date = current_time.strftime('%Y-%m-%d')
        end_date = (current_time + timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Prepare API request parameters
        params = {
            'api_key': self.economic_api_key,
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        try:
            # Make API request
            response = requests.get(self.economic_api_url, params=params, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse response
            data = response.json()
            
            # Process the data into our format
            calendar_data = defaultdict(list)
            
            for event in data.get('events', []):
                # Extract relevant info
                currency = event.get('currency')
                if not currency:
                    continue
                    
                # Parse event datetime
                try:
                    event_date = event.get('date', '')
                    event_time = event.get('time', '00:00:00')
                    event_datetime = datetime.strptime(f"{event_date} {event_time}", '%Y-%m-%d %H:%M:%S')
                    event_datetime = event_datetime.replace(tzinfo=pytz.UTC)
                except Exception as e:
                    logger.warning(f"Failed to parse event datetime: {e}")
                    continue
                
                # Create event record
                event_record = {
                    'datetime': event_datetime,
                    'currency': currency,
                    'event': event.get('event', 'Unknown Event'),
                    'impact': event.get('impact', 'low'),
                    'forecast': event.get('forecast'),
                    'previous': event.get('previous')
                }
                
                # Add to our calendar data
                calendar_data[currency].append(event_record)
            
            # Update the calendar data
            self.economic_calendar = dict(calendar_data)
            self.last_calendar_update = current_time
            
            logger.info(f"Economic calendar updated with {sum(len(events) for events in calendar_data.values())} events")
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar data: {e}")
            # Keep using the old calendar data if available
    
    def _get_regime_based_scores(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get strategy scores based on market regime compatibility.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary mapping strategy names to scores
        """
        scores = {}
        for strategy_name, compatibility in self.strategy_compatibility.items():
            scores[strategy_name] = compatibility.get(regime, 0.5)
        return scores
    
    def _apply_session_preferences(self, 
                                 scores: Dict[str, float], 
                                 active_sessions: List[ForexSession]) -> Dict[str, float]:
        """
        Adjust strategy scores based on active forex sessions.
        
        Args:
            scores: Initial strategy scores
            active_sessions: List of currently active forex sessions
            
        Returns:
            Updated strategy scores
        """
        if not active_sessions:
            return scores  # No active sessions, no adjustments
            
        # Copy to avoid modifying the original
        updated_scores = scores.copy()
        
        # Apply session-based adjustments
        for session in active_sessions:
            preferred_strategies = self.session_strategy_preferences.get(session, [])
            for strategy in preferred_strategies:
                if strategy in updated_scores:
                    # Boost score for preferred strategies in this session
                    updated_scores[strategy] += 0.1
                    
            # If it's the overlap session, give extra emphasis to the adjustments
            if session == ForexSession.LONDON_NEWYORK_OVERLAP:
                for strategy in preferred_strategies:
                    if strategy in updated_scores:
                        updated_scores[strategy] += 0.1  # Additional boost
        
        # Normalize scores to keep them in the 0-1 range
        max_score = max(updated_scores.values())
        if max_score > 1.0:
            for strategy in updated_scores:
                updated_scores[strategy] /= max_score
                
        return updated_scores
    
    def _apply_risk_adjustments(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust strategy scores based on risk tolerance settings.
        
        Args:
            scores: Current strategy scores
            
        Returns:
            Risk-adjusted strategy scores
        """
        # Copy to avoid modifying the original
        adjusted_scores = scores.copy()
        
        # Get risk adjustment parameters
        risk_params = self.risk_adjustments.get(self.risk_tolerance, 
                                              self.risk_adjustments[RiskTolerance.MEDIUM])
        
        # Apply adjustments based on risk tolerance
        preferred_strategies = risk_params.get("preferred_strategies", [])
        avoid_strategies = risk_params.get("avoid_strategies", [])
        
        # Boost scores for preferred strategies
        for strategy in preferred_strategies:
            if strategy in adjusted_scores:
                adjusted_scores[strategy] += 0.15
                
        # Reduce scores for strategies to avoid
        for strategy in avoid_strategies:
            if strategy in adjusted_scores:
                adjusted_scores[strategy] -= 0.25
                
        # Normalize scores to keep them in the 0-1 range
        max_score = max(adjusted_scores.values())
        if max_score > 1.0:
            for strategy in adjusted_scores:
                adjusted_scores[strategy] /= max_score
                
        return adjusted_scores
    
    def _apply_news_adjustments(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust strategy scores when significant economic news is expected.
        
        Args:
            scores: Current strategy scores
            
        Returns:
            News-adjusted strategy scores
        """
        # During high-impact news:
        # - Reduce scores for range trading (ranges often break)
        # - Boost scores for breakout strategies
        # - Reduce scores for trend following (trends can reverse)
        
        adjusted_scores = scores.copy()
        
        # Adjust scores for high-impact news
        if "forex_range_trading" in adjusted_scores:
            adjusted_scores["forex_range_trading"] *= 0.6  # Significantly reduce
            
        if "forex_breakout" in adjusted_scores:
            adjusted_scores["forex_breakout"] *= 1.5  # Significantly boost
            
        if "forex_trend_following" in adjusted_scores:
            adjusted_scores["forex_trend_following"] *= 0.8  # Moderately reduce
            
        # Normalize scores to keep them in the 0-1 range
        max_score = max(adjusted_scores.values())
        if max_score > 1.0:
            for strategy in adjusted_scores:
                adjusted_scores[strategy] /= max_score
                
        return adjusted_scores
    
    def _apply_historical_performance_adjustments(self, scores: Dict[str, float], regime: MarketRegime) -> Dict[str, float]:
        """
        Adjust strategy scores based on historical performance in similar market conditions.
        
        Args:
            scores: Current strategy scores
            regime: Current market regime
            
        Returns:
            Performance-adjusted strategy scores
        """
        # Create a copy of scores to modify
        adjusted_scores = scores.copy()
        
        # Apply historical performance adjustments
        for strategy_name, base_score in scores.items():
            # Check if we have historical performance data for this strategy and regime
            if strategy_name in self.strategy_performance and regime in self.strategy_performance[strategy_name]:
                performance_data = self.strategy_performance[strategy_name][regime]
                
                # Only adjust if we have enough data points
                if len(performance_data) >= 5:  # Require at least 5 performance records
                    # Calculate average performance
                    avg_performance = sum(performance_data) / len(performance_data)
                    
                    # Normalize to 0.0-1.0 range where 0.5 is neutral
                    # Values above 0.5 boost the score, below 0.5 reduce it
                    normalized_performance = min(max(avg_performance, 0.0), 1.0)
                    
                    # Apply performance-based adjustment (up to +/- 20%)
                    adjustment = (normalized_performance - 0.5) * 0.4  # Scale to +/- 20%
                    adjusted_scores[strategy_name] = base_score * (1 + adjustment)
                    
                    logger.debug(f"Applied historical performance adjustment to {strategy_name}: {adjustment:+.2f}")
        
        return adjusted_scores
    
    def record_strategy_performance(self, strategy_name: str, regime: MarketRegime, performance_score: float) -> None:
        """
        Record the performance of a strategy in a specific market regime.
        
        Args:
            strategy_name: Name of the strategy
            regime: Market regime during the strategy execution
            performance_score: Performance score (0.0-1.0 where 1.0 is perfect)
        """
        # Ensure score is in valid range
        performance_score = min(max(performance_score, 0.0), 1.0)
        
        # Record the performance
        if strategy_name in self.strategy_performance and regime in self.strategy_performance[strategy_name]:
            # Add to existing records, keeping the most recent 50 data points
            self.strategy_performance[strategy_name][regime].append(performance_score)
            if len(self.strategy_performance[strategy_name][regime]) > 50:
                self.strategy_performance[strategy_name][regime].pop(0)  # Remove oldest record
        else:
            # Create new record
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {}
            self.strategy_performance[strategy_name][regime] = [performance_score]
        
        logger.info(f"Recorded performance for {strategy_name} in {regime.name} regime: {performance_score:.2f}")
        
    def calculate_strategy_performance(self, 
                                     strategy_name: str, 
                                     initial_equity: float, 
                                     final_equity: float,
                                     max_drawdown: float,
                                     win_rate: float,
                                     profit_factor: float) -> float:
        """
        Calculate a normalized performance score for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            initial_equity: Starting account value
            final_equity: Ending account value
            max_drawdown: Maximum drawdown percentage (0-100)
            win_rate: Percentage of winning trades (0-100)
            profit_factor: Ratio of gross profits to gross losses
            
        Returns:
            Performance score (0.0-1.0)
        """
        # Calculate return percentage
        return_pct = (final_equity / initial_equity - 1) * 100 if initial_equity > 0 else 0
        
        # Calculate adjusted return based on drawdown (penalize high drawdowns)
        if max_drawdown > 0:
            risk_adjusted_return = return_pct / max_drawdown
        else:
            risk_adjusted_return = return_pct * 2  # Bonus for no drawdown
        
        # Consider the win rate (normalized to 0-1)
        win_rate_normalized = win_rate / 100.0 if win_rate <= 100 else 1.0
        
        # Consider profit factor (cap at 5.0 for normalization)
        profit_factor_normalized = min(profit_factor / 5.0, 1.0) if profit_factor > 0 else 0
        
        # Calculate overall performance score with weighted components
        # Weights: Risk-adjusted Return 50%, Win Rate 25%, Profit Factor 25%
        if return_pct > 0:
            # Normalize risk-adjusted return (cap at 5.0 for normalization)
            return_score = min(risk_adjusted_return / 5.0, 1.0)
        else:
            # Negative returns get lower scores
            return_score = max(0, 1 + (return_pct / 100))  # E.g., -5% return => 0.95 score
        
        # Combine all factors with weights
        performance_score = (return_score * 0.5) + (win_rate_normalized * 0.25) + (profit_factor_normalized * 0.25)
        
        # Ensure final score is in 0-1 range
        return min(max(performance_score, 0.0), 1.0)
    
    def _get_optimized_parameters(self, 
                               strategy_name: str, 
                               regime: MarketRegime,
                               active_sessions: List[ForexSession],
                               has_high_impact_news: bool) -> Dict[str, Any]:
        """
        Get optimized strategy parameters based on current conditions.
        
        Args:
            strategy_name: Selected strategy name
            regime: Current market regime
            active_sessions: Active forex sessions
            has_high_impact_news: Whether high-impact news is expected
            
        Returns:
            Dictionary of optimized parameters
        """
        # This would be more comprehensive in a production system.
        # For now, we'll return a basic set of optimized parameters.
        
        # Base parameters (these would normally come from strategy default params)
        base_params = {
            "pip_value": 0.0001,  # Standard for most forex pairs
            "atr_period": 14,
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 3.0,
            "trading_sessions": [session.name for session in active_sessions],
            "avoid_news_releases": True
        }
        
        # Apply risk tolerance adjustments
        risk_params = self.risk_adjustments.get(self.risk_tolerance, 
                                            self.risk_adjustments[RiskTolerance.MEDIUM])
        
        base_params["stop_loss_atr_mult"] *= risk_params["stop_loss_multiplier"]
        base_params["take_profit_atr_mult"] *= risk_params["take_profit_multiplier"]
        
        # Apply strategy-specific optimizations
        if strategy_name == "forex_trend_following":
            base_params.update({
                "fast_ma_period": 20 if regime == MarketRegime.HIGH_VOLATILITY else 10,
                "slow_ma_period": 50 if regime == MarketRegime.HIGH_VOLATILITY else 30,
                "adx_period": 14,
                "adx_threshold": 25,
                "trend_strength_threshold": 30 if self.risk_tolerance == RiskTolerance.LOW else 20
            })
            
        elif strategy_name == "forex_range_trading":
            base_params.update({
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "range_detection_lookback": 20,
                "min_range_periods": 10
            })
            
        elif strategy_name == "forex_breakout":
            # For breakout, adjust confirmation thresholds based on risk tolerance
            base_params.update({
                "donchian_period": 20,
                "atr_multiplier": 0.5 if self.risk_tolerance == RiskTolerance.LOW else 1.0,
                "volume_confirmation": self.risk_tolerance != RiskTolerance.HIGH,  # Skip volume confirmation for high risk
                "confirmation_candles": 3 if self.risk_tolerance == RiskTolerance.LOW else 1
            })
            
        elif strategy_name == "forex_momentum":
            base_params.update({
                "roc_period": 14,
                "rsi_period": 14,
                "momentum_period": 10,
                "adx_period": 14,
                "adx_threshold": 20 if self.risk_tolerance == RiskTolerance.HIGH else 25
            })
            
        # Add news handling parameters if there's high impact news
        if has_high_impact_news:
            base_params["news_volatility_adjustment"] = True
            base_params["wider_stops_for_news"] = True
            
        return base_params
        
    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Get risk management parameters based on current risk tolerance.
        
        Returns:
            Dictionary of risk parameters
        """
        risk_params = self.risk_adjustments.get(self.risk_tolerance, 
                                              self.risk_adjustments[RiskTolerance.MEDIUM])
        
        return {
            "risk_tolerance": self.risk_tolerance,
            "position_size_multiplier": risk_params["position_size_multiplier"],
            "stop_loss_multiplier": risk_params["stop_loss_multiplier"],
            "take_profit_multiplier": risk_params["take_profit_multiplier"],
            "max_trades_per_session": 3 if self.risk_tolerance == RiskTolerance.LOW else 
                                     5 if self.risk_tolerance == RiskTolerance.MEDIUM else 8,
            "max_risk_per_trade_pct": 1.0 if self.risk_tolerance == RiskTolerance.LOW else
                                     2.0 if self.risk_tolerance == RiskTolerance.MEDIUM else 3.0,
            "max_risk_per_day_pct": 3.0 if self.risk_tolerance == RiskTolerance.LOW else
                                   6.0 if self.risk_tolerance == RiskTolerance.MEDIUM else 10.0,
        }
