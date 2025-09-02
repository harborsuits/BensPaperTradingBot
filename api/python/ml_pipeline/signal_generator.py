"""
ML-Powered Signal Generator

This module uses trained ML models to generate trading signals 
with confidence scores and risk assessment.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generates trading signals using ML models and market data"""
    
    def __init__(self, models=None, confidence_threshold=0.65, config=None):
        """
        Initialize the signal generator
        
        Args:
            models: Dictionary of trained models (or paths to models)
            confidence_threshold: Minimum confidence score to generate a signal
            config: Configuration dictionary with parameters
        """
        self.models = models or {}
        self.confidence_threshold = confidence_threshold
        self.config = config or {}
        self.signal_history = []
        
        # Set default signal parameters
        self.default_params = {
            'signal_lookback': 3,         # Days to look back for signal confirmation
            'min_signal_streak': 2,       # Minimum consecutive signals in same direction
            'max_positions': 5,           # Maximum number of concurrent positions
            'position_sizing_method': 'confidence',  # 'equal', 'confidence', 'kelly'
            'risk_per_trade': 0.02,       # 2% risk per trade
            'trail_stop_atr_multiple': 2.0 # ATR multiple for trailing stops
        }
        
        # Override defaults with config values if provided
        for param, value in self.config.get('signal_params', {}).items():
            if param in self.default_params:
                self.default_params[param] = value
        
        logger.info(f"Signal Generator initialized with confidence threshold: {confidence_threshold}")
    
    def add_model(self, name: str, model, metadata: Dict = None):
        """
        Add a trained model to the signal generator
        
        Args:
            name: Name of the model
            model: Trained model object
            metadata: Optional metadata about the model
        """
        self.models[name] = {
            'model': model,
            'metadata': metadata or {}
        }
        logger.info(f"Added model {name} to signal generator")
    
    def generate_signals(self, market_data: pd.DataFrame, 
                       ticker: str,
                       timeframe: str = '1d') -> Dict[str, Any]:
        """
        Generate trading signals from market data using ML models
        
        Args:
            market_data: DataFrame with market data and features
            ticker: Ticker symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with signal information
        """
        if not self.models:
            logger.warning("No models available for signal generation")
            return {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'signal': 'neutral',
                'confidence': 0.0,
                'position_size': 0.0,
                'reason': 'No models available'
            }
        
        # Ensure market data is not empty
        if market_data.empty:
            logger.warning(f"Empty market data for {ticker}")
            return {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'signal': 'neutral',
                'confidence': 0.0,
                'position_size': 0.0,
                'reason': 'Empty market data'
            }
        
        # Ensure all required features are present
        feature_missing = False
        for model_info in self.models.values():
            required_features = model_info.get('metadata', {}).get('features', [])
            for feature in required_features:
                if feature not in market_data.columns:
                    logger.warning(f"Missing required feature: {feature}")
                    feature_missing = True
        
        if feature_missing:
            return {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'signal': 'neutral',
                'confidence': 0.0,
                'position_size': 0.0,
                'reason': 'Missing required features'
            }
        
        # Get the latest data point
        latest_data = market_data.iloc[-1:].copy()
        
        # Generate predictions from each model
        predictions = []
        model_weights = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            metadata = model_info['metadata']
            
            # Get required features for this model
            features = metadata.get('features', [col for col in market_data.columns 
                                               if col not in ['open', 'high', 'low', 'close', 'volume', 'date']])
            
            # Handle missing features
            missing_features = [f for f in features if f not in market_data.columns]
            if missing_features:
                logger.warning(f"Model {model_name} missing features: {missing_features}")
                continue
            
            # Get model weight (default to 1.0 if not specified)
            weight = metadata.get('weight', 1.0)
            model_weights[model_name] = weight
            
            # Generate prediction
            try:
                X = latest_data[features]
                
                # Handle probability-based models vs. direct score models
                if hasattr(model, 'predict_proba'):
                    # For classifiers that output probabilities
                    proba = model.predict_proba(X)[0]
                    if len(proba) >= 2:
                        # Assuming binary classification with [down, up] probabilities
                        prediction = {
                            'model': model_name,
                            'signal': 'buy' if proba[1] > 0.5 else 'sell',
                            'raw_confidence': proba[1] if proba[1] > 0.5 else proba[0],
                            'weight': weight
                        }
                    else:
                        # Handle unusual case
                        prediction = {
                            'model': model_name,
                            'signal': 'neutral',
                            'raw_confidence': 0.5,
                            'weight': weight
                        }
                else:
                    # For models that output a direction score directly
                    score = model.predict(X)[0]
                    prediction = {
                        'model': model_name,
                        'signal': 'buy' if score > 0 else 'sell' if score < 0 else 'neutral',
                        'raw_confidence': abs(score),
                        'weight': weight
                    }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error generating prediction with model {model_name}: {e}")
        
        # If no predictions could be generated, return neutral signal
        if not predictions:
            return {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'signal': 'neutral',
                'confidence': 0.0,
                'position_size': 0.0,
                'reason': 'Failed to generate predictions'
            }
        
        # Calculate weighted consensus and confidence
        total_weight = sum(model_weights.values())
        buy_confidence = sum(p['raw_confidence'] * p['weight'] 
                            for p in predictions if p['signal'] == 'buy') / total_weight
        sell_confidence = sum(p['raw_confidence'] * p['weight'] 
                             for p in predictions if p['signal'] == 'sell') / total_weight
        
        # Determine final signal and confidence
        if buy_confidence > sell_confidence:
            signal = 'buy'
            confidence = buy_confidence
        elif sell_confidence > buy_confidence:
            signal = 'sell'
            confidence = sell_confidence
        else:
            signal = 'neutral'
            confidence = 0.5
        
        # Only generate actionable signals if confidence exceeds threshold
        if confidence < self.confidence_threshold:
            signal = 'neutral'
            reason = f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}"
        else:
            reason = f"ML model consensus with {confidence:.2f} confidence"
        
        # Calculate market conditions for additional context
        market_conditions = self._analyze_market_conditions(market_data)
        
        # Calculate recommended position size
        position_size = self._calculate_position_size(confidence, market_conditions)
        
        # Calculate risk parameters
        risk_params = self._calculate_risk_parameters(market_data, signal, confidence)
        
        # Create signal
        signal_data = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'timeframe': timeframe,
            'signal': signal,
            'confidence': confidence,
            'position_size': position_size,
            'reason': reason,
            'market_conditions': market_conditions,
            'risk_params': risk_params,
            'model_predictions': predictions
        }
        
        # Store in signal history
        self.signal_history.append(signal_data)
        if len(self.signal_history) > 100:  # Keep history manageable
            self.signal_history = self.signal_history[-100:]
        
        # Log signal generation
        logger.info(f"Generated {signal} signal for {ticker} with {confidence:.2f} confidence")
        
        return signal_data
    
    def generate_signals_batch(self, market_data_dict: Dict[str, pd.DataFrame], 
                             timeframe: str = '1d') -> Dict[str, Dict[str, Any]]:
        """
        Generate signals for multiple tickers
        
        Args:
            market_data_dict: Dictionary of DataFrames with market data, keyed by ticker
            timeframe: Data timeframe
            
        Returns:
            Dictionary of signal information, keyed by ticker
        """
        signals = {}
        
        for ticker, data in market_data_dict.items():
            signals[ticker] = self.generate_signals(data, ticker, timeframe)
        
        return signals
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market conditions for context
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Dictionary with market condition analysis
        """
        try:
            # Use last 20 days for analysis if available
            lookback = min(20, len(market_data))
            recent_data = market_data.iloc[-lookback:].copy()
            
            # Calculate basic metrics
            latest_close = recent_data['close'].iloc[-1]
            avg_volume = recent_data['volume'].mean()
            latest_volume = recent_data['volume'].iloc[-1]
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine trend
            sma20 = recent_data['close'].mean()
            sma50 = market_data['close'].rolling(window=50).mean().iloc[-1] if len(market_data) >= 50 else sma20
            trend = 'bullish' if latest_close > sma20 and sma20 > sma50 else \
                   'bearish' if latest_close < sma20 and sma20 < sma50 else 'neutral'
            
            # Calculate volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Determine volatility regime
            vol_regime = 'high' if volatility > 0.3 else \
                        'medium' if volatility > 0.15 else 'low'
            
            # Recent price movement
            five_day_return = latest_close / market_data['close'].iloc[-5] - 1 if len(market_data) >= 5 else 0
            
            return {
                'trend': trend,
                'volatility': volatility,
                'volatility_regime': vol_regime,
                'volume_ratio': volume_ratio,
                'volume_trend': 'increasing' if volume_ratio > 1.2 else \
                               'decreasing' if volume_ratio < 0.8 else 'normal',
                'five_day_return': five_day_return,
                'above_sma20': latest_close > sma20,
                'above_sma50': latest_close > sma50
            }
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                'trend': 'unknown',
                'volatility': 0.0,
                'volatility_regime': 'unknown',
                'volume_ratio': 1.0,
                'volume_trend': 'normal',
                'five_day_return': 0.0,
                'above_sma20': False,
                'above_sma50': False
            }
    
    def _calculate_position_size(self, confidence: float, 
                               market_conditions: Dict[str, Any]) -> float:
        """
        Calculate recommended position size based on confidence and market conditions
        
        Args:
            confidence: Signal confidence score
            market_conditions: Dictionary with market condition analysis
            
        Returns:
            Recommended position size as a percentage
        """
        method = self.default_params['position_sizing_method']
        base_risk = self.default_params['risk_per_trade']
        
        # Volatility adjustment factor
        vol_regime = market_conditions.get('volatility_regime', 'medium')
        vol_factor = 0.7 if vol_regime == 'high' else \
                    1.2 if vol_regime == 'low' else 1.0
        
        # Trend alignment factor
        trend = market_conditions.get('trend', 'neutral')
        trend_factor = 1.2 if trend in ['bullish', 'bearish'] else 1.0
        
        if method == 'equal':
            # Equal position sizing (just use base risk)
            position_size = base_risk * vol_factor
            
        elif method == 'confidence':
            # Scale by confidence level
            position_size = base_risk * confidence * vol_factor * trend_factor
            
        elif method == 'kelly':
            # Simplified Kelly criterion
            win_rate = confidence  # Approximate win rate with confidence
            # Assume reward:risk ratio of 2:1 for simplicity
            reward_risk_ratio = 2.0
            
            # Kelly formula: f* = (p*b - q) / b
            # where p = win rate, q = 1-p, b = reward/risk ratio
            kelly_pct = (win_rate * reward_risk_ratio - (1 - win_rate)) / reward_risk_ratio
            
            # Use half-Kelly for safety
            position_size = max(0, kelly_pct * 0.5) * vol_factor * trend_factor
            
        else:
            # Default to base risk if unknown method
            position_size = base_risk
        
        # Ensure position size doesn't exceed maximum
        max_position = base_risk * 2.0  # Maximum twice the base risk
        return min(position_size, max_position)
    
    def _calculate_risk_parameters(self, market_data: pd.DataFrame, 
                                 signal: str, 
                                 confidence: float) -> Dict[str, Any]:
        """
        Calculate risk management parameters for the trade
        
        Args:
            market_data: DataFrame with market data
            signal: Signal direction ('buy', 'sell', 'neutral')
            confidence: Signal confidence score
            
        Returns:
            Dictionary with risk management parameters
        """
        # Default parameters
        risk_params = {
            'stop_loss_pct': 0.0,
            'take_profit_pct': 0.0,
            'trail_stop_pct': 0.0,
            'max_holding_days': 10,
            'entry_window_hours': 24,
            'risk_reward_ratio': 0.0
        }
        
        if signal == 'neutral':
            return risk_params
            
        try:
            # Calculate ATR for volatility-based stops
            atr_periods = 14
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high.iloc[-atr_periods:] - low.iloc[-atr_periods:]
            tr2 = abs(high.iloc[-atr_periods:] - close.iloc[-atr_periods-1:-1].values)
            tr3 = abs(low.iloc[-atr_periods:] - close.iloc[-atr_periods-1:-1].values)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.mean()
            
            # Latest close price
            latest_close = close.iloc[-1]
            
            # ATR as percentage of price
            atr_pct = atr / latest_close * 100
            
            # Set stops based on ATR, confidence and signal
            trail_stop_multiple = self.default_params['trail_stop_atr_multiple']
            
            # Dynamic stop loss based on ATR and confidence
            # Lower confidence = wider stop to reduce false exits
            confidence_factor = 0.8 + (confidence * 0.4)  # Range from 0.8 to 1.2
            stop_loss_pct = atr_pct * trail_stop_multiple * confidence_factor
            
            # Take profit is a multiple of stop loss
            # Higher confidence = larger profit target
            take_profit_multiple = 1.5 + confidence  # Range from 1.5 to 2.5
            take_profit_pct = stop_loss_pct * take_profit_multiple
            
            # Trailing stop percentage
            trail_stop_pct = atr_pct * trail_stop_multiple
            
            # Adjust max holding days based on confidence
            # Higher confidence = can hold longer
            base_holding_days = 10
            max_holding_days = int(base_holding_days * (0.7 + confidence * 0.6))  # Range from 7 to 13
            
            # Calculate risk-reward ratio
            risk_reward_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
            
            risk_params = {
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'trail_stop_pct': round(trail_stop_pct, 2),
                'max_holding_days': max_holding_days,
                'entry_window_hours': 24,
                'risk_reward_ratio': round(risk_reward_ratio, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk parameters: {e}")
        
        return risk_params
    
    def get_signal_history(self, ticker: Optional[str] = None, 
                         days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical signals
        
        Args:
            ticker: Optional ticker to filter by
            days_back: Optional number of days to look back
            
        Returns:
            List of historical signals
        """
        if not self.signal_history:
            return []
            
        filtered_history = self.signal_history
        
        # Filter by ticker if specified
        if ticker:
            filtered_history = [s for s in filtered_history if s['ticker'] == ticker]
            
        # Filter by date if specified
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_history = [s for s in filtered_history 
                              if s['timestamp'] > cutoff_date]
            
        return filtered_history
    
    def export_signals(self, filepath: str):
        """
        Export signal history to JSON file
        
        Args:
            filepath: Path to export to
        """
        # Convert datetime objects to strings
        export_data = []
        for signal in self.signal_history:
            export_signal = signal.copy()
            export_signal['timestamp'] = export_signal['timestamp'].isoformat()
            export_data.append(export_signal)
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(export_data)} signals to {filepath}")
    
    def import_signals(self, filepath: str):
        """
        Import signal history from JSON file
        
        Args:
            filepath: Path to import from
        """
        with open(filepath, 'r') as f:
            import_data = json.load(f)
            
        # Convert string timestamps back to datetime
        for signal in import_data:
            signal['timestamp'] = datetime.fromisoformat(signal['timestamp'])
            
        self.signal_history = import_data
        logger.info(f"Imported {len(import_data)} signals from {filepath}")
        
    def get_signal_performance(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze performance of historical signals
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with signal performance metrics
        """
        if not self.signal_history:
            return {
                'total_signals': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'correlation': 0.0
            }
            
        # Filter to signals within lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_signals = [s for s in self.signal_history 
                         if s['timestamp'] > cutoff_date]
        
        if not recent_signals:
            return {
                'total_signals': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'correlation': 0.0
            }
            
        # Calculate basic metrics
        total_signals = len(recent_signals)
        buy_signals = sum(1 for s in recent_signals if s['signal'] == 'buy')
        sell_signals = sum(1 for s in recent_signals if s['signal'] == 'sell')
        neutral_signals = total_signals - buy_signals - sell_signals
        
        # Calculate average confidence
        avg_confidence = sum(s['confidence'] for s in recent_signals) / total_signals
        
        # Calculate correlation between confidence and outcome
        # This would require outcome data which we don't have here
        # For now, just return placeholder
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'avg_confidence': avg_confidence,
            'signal_types': {
                'buy_pct': buy_signals / total_signals if total_signals > 0 else 0,
                'sell_pct': sell_signals / total_signals if total_signals > 0 else 0,
                'neutral_pct': neutral_signals / total_signals if total_signals > 0 else 0
            },
            'lookback_days': lookback_days,
            'latest_signal_date': max(s['timestamp'] for s in recent_signals).isoformat()
        }
