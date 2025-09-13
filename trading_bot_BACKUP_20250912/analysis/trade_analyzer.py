#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Analyzer Module for Trading Strategies

This module provides tools for tracking, analyzing, and explaining model predictions
and trading outcomes, creating a feedback loop for continuous improvement.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class TradeAnalyzer:
    """
    Trade analyzer class for tracking and analyzing trading decisions.
    
    This class provides:
    1. Logging of model predictions with feature importance
    2. Comparison of predictions vs actual outcomes
    3. Performance tracking by market regime
    4. Feature stability analysis
    5. Trade explanation generation
    6. Performance visualization
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the trade analyzer module.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.prediction_log = []
        self.performance_by_regime = defaultdict(list)
        self.feature_stability = {}
        self.trades_history = []
        self.feature_usage_history = defaultdict(list)
        
        # Create log directory
        self.log_dir = self.params.get('log_dir', './logs/trades')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Load previous logs if available
        self._load_previous_logs()
    
    def _load_previous_logs(self):
        """Load previous trade logs if available."""
        log_file = os.path.join(self.log_dir, 'prediction_log.pkl')
        if os.path.exists(log_file):
            try:
                with open(log_file, 'rb') as f:
                    self.prediction_log = pickle.load(f)
                    
                # Also load feature usage history
                for prediction in self.prediction_log:
                    if 'top_features' in prediction:
                        for feature, value in prediction['top_features'].items():
                            self.feature_usage_history[feature].append({
                                'timestamp': prediction['timestamp'],
                                'importance': value,
                                'regime': prediction.get('regime', 'unknown')
                            })
            except Exception as e:
                print(f"Error loading previous logs: {str(e)}")
    
    def log_prediction(self, timestamp: datetime, features: pd.DataFrame, 
                      prediction: Any, confidence: float, 
                      top_features: Dict[str, float], regime: str = 'unknown',
                      model_name: str = 'default', metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Log a model prediction with feature importance.
        
        Args:
            timestamp: Time of prediction
            features: Feature values used for prediction
            prediction: Model prediction (class or value)
            confidence: Confidence score (for classification)
            top_features: Dictionary of top features and their importance
            regime: Market regime at time of prediction
            model_name: Name of the model used
            metadata: Additional metadata about the prediction
            
        Returns:
            Dictionary representing the logged prediction
        """
        # Create prediction entry
        prediction_entry = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'top_features': top_features,
            'regime': regime,
            'model_name': model_name,
            'feature_values': features.iloc[0].to_dict() if len(features) == 1 else {},
            'metadata': metadata or {}
        }
        
        # Store in log
        self.prediction_log.append(prediction_entry)
        
        # Update feature usage history
        for feature, value in top_features.items():
            self.feature_usage_history[feature].append({
                'timestamp': timestamp,
                'importance': value,
                'regime': regime
            })
        
        # Save to disk periodically
        if len(self.prediction_log) % self.params.get('log_save_frequency', 100) == 0:
            self._save_logs()
        
        return prediction_entry
    
    def log_trade_outcome(self, prediction_id: Union[int, datetime], 
                         actual_outcome: Any, pnl: float, 
                         trade_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Log the actual outcome of a trade.
        
        Args:
            prediction_id: ID or timestamp of the prediction
            actual_outcome: Actual outcome (class or value)
            pnl: Profit/loss realized
            trade_metadata: Additional trade details
            
        Returns:
            Updated prediction entry
        """
        # Find the corresponding prediction
        prediction_entry = None
        if isinstance(prediction_id, datetime):
            # Find by timestamp
            for entry in self.prediction_log:
                if entry['timestamp'] == prediction_id:
                    prediction_entry = entry
                    break
        else:
            # Find by index
            if 0 <= prediction_id < len(self.prediction_log):
                prediction_entry = self.prediction_log[prediction_id]
        
        if prediction_entry is None:
            raise ValueError(f"Prediction with ID {prediction_id} not found")
        
        # Update prediction with actual outcome
        prediction_entry['actual_outcome'] = actual_outcome
        prediction_entry['pnl'] = pnl
        prediction_entry['trade_metadata'] = trade_metadata or {}
        
        # Calculate if prediction was correct
        prediction_entry['correct'] = (
            prediction_entry['prediction'] == actual_outcome 
            if isinstance(actual_outcome, (int, str, bool)) 
            else np.sign(prediction_entry['prediction']) == np.sign(actual_outcome)
        )
        
        # Add to trades history
        self.trades_history.append(prediction_entry)
        
        # Update performance by regime
        regime = prediction_entry['regime']
        self.performance_by_regime[regime].append({
            'timestamp': prediction_entry['timestamp'],
            'correct': prediction_entry['correct'],
            'pnl': pnl
        })
        
        # Save to disk periodically
        if len(self.trades_history) % self.params.get('log_save_frequency', 100) == 0:
            self._save_logs()
        
        return prediction_entry
    
    def analyze_model_performance(self, regime: str = None, 
                                 timeframe: Tuple[datetime, datetime] = None,
                                 model_name: str = None) -> Dict[str, Any]:
        """
        Analyze model prediction performance.
        
        Args:
            regime: Optional filter by market regime
            timeframe: Optional timeframe filter (start, end)
            model_name: Optional filter by model
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter trades
        filtered_trades = self._filter_trades(regime, timeframe, model_name)
        
        # Calculate metrics
        if not filtered_trades:
            return {
                'total_trades': 0,
                'accuracy': None,
                'avg_pnl': None,
                'total_pnl': None,
                'win_rate': None,
                'profit_factor': None
            }
        
        # Basic metrics
        correct_trades = [t for t in filtered_trades if t.get('correct', False)]
        profitable_trades = [t for t in filtered_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in filtered_trades if t.get('pnl', 0) < 0]
        
        total_trades = len(filtered_trades)
        total_correct = len(correct_trades)
        accuracy = total_correct / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        pnls = [t.get('pnl', 0) for t in filtered_trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        total_pnl = np.sum(pnls) if pnls else 0
        
        # Trading metrics
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(t.get('pnl', 0) for t in profitable_trades) if profitable_trades else 0
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Create result
        return {
            'total_trades': total_trades,
            'accuracy': accuracy,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'correct_predictions': total_correct,
            'incorrect_predictions': total_trades - total_correct
        }
    
    def analyze_feature_stability(self, top_n: int = 10, 
                                 timeframe: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """
        Analyze how feature importance changes over time.
        
        Args:
            top_n: Number of top features to include
            timeframe: Optional timeframe filter (start, end)
            
        Returns:
            Dictionary with feature stability metrics
        """
        if not self.feature_usage_history:
            return {'error': 'No feature usage data available'}
        
        # Get feature usage in timeframe
        feature_usage = {}
        for feature, history in self.feature_usage_history.items():
            if timeframe:
                filtered_history = [h for h in history 
                                   if timeframe[0] <= h['timestamp'] <= timeframe[1]]
            else:
                filtered_history = history
                
            if filtered_history:
                # Calculate average importance and frequency
                avg_importance = np.mean([h['importance'] for h in filtered_history])
                frequency = len(filtered_history)
                
                # Calculate variance of importance
                if len(filtered_history) > 1:
                    importance_variance = np.var([h['importance'] for h in filtered_history])
                else:
                    importance_variance = 0
                
                # Store metrics
                feature_usage[feature] = {
                    'avg_importance': avg_importance,
                    'frequency': frequency,
                    'importance_variance': importance_variance,
                    'regime_breakdown': self._feature_regime_breakdown(feature, filtered_history)
                }
        
        # Sort by average importance
        sorted_features = sorted(feature_usage.items(), 
                               key=lambda x: x[1]['avg_importance'], 
                               reverse=True)
        
        # Get top N
        top_features = dict(sorted_features[:top_n])
        
        # Calculate stability metrics
        stability_scores = {}
        for feature, metrics in top_features.items():
            # Stability score: higher is more stable
            # Based on inverse of variance and frequency of appearance
            if metrics['importance_variance'] > 0:
                stability = metrics['frequency'] / (metrics['importance_variance'] * 10 + 1)
            else:
                stability = metrics['frequency']
                
            stability_scores[feature] = stability
        
        return {
            'top_features': top_features,
            'stability_scores': stability_scores
        }
    
    def _feature_regime_breakdown(self, feature: str, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate feature importance breakdown by regime."""
        regime_importance = defaultdict(list)
        
        for entry in history:
            regime = entry.get('regime', 'unknown')
            regime_importance[regime].append(entry['importance'])
        
        # Calculate average importance per regime
        return {regime: np.mean(importances) for regime, importances in regime_importance.items()}
    
    def get_trade_explanation(self, trade_id: Union[int, datetime], 
                            include_feature_values: bool = True) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a specific trade.
        
        Args:
            trade_id: ID or timestamp of the trade
            include_feature_values: Whether to include actual feature values
            
        Returns:
            Dictionary with detailed trade explanation
        """
        # Find the trade
        trade = None
        if isinstance(trade_id, datetime):
            # Find by timestamp
            for entry in self.trades_history:
                if entry['timestamp'] == trade_id:
                    trade = entry
                    break
        else:
            # Find by index
            if 0 <= trade_id < len(self.trades_history):
                trade = self.trades_history[trade_id]
        
        if trade is None:
            raise ValueError(f"Trade with ID {trade_id} not found")
        
        # Build explanation
        explanation = {
            'timestamp': trade['timestamp'],
            'model': trade['model_name'],
            'regime': trade['regime'],
            'prediction': trade['prediction'],
            'confidence': trade['confidence'],
            'actual_outcome': trade.get('actual_outcome'),
            'pnl': trade.get('pnl'),
            'correct': trade.get('correct'),
            'top_features': trade['top_features']
        }
        
        # Add feature values if requested
        if include_feature_values and 'feature_values' in trade:
            # Only include values for top features
            top_feature_values = {f: trade['feature_values'].get(f) for f in trade['top_features']}
            explanation['feature_values'] = top_feature_values
        
        # Add market context
        if 'metadata' in trade and trade['metadata']:
            explanation['market_context'] = {
                'regime': trade['regime'],
                'market_condition': trade['metadata'].get('market_condition'),
                'volatility': trade['metadata'].get('volatility')
            }
        
        # Add explanation text
        explanation['explanation_text'] = self._generate_explanation_text(trade)
        
        return explanation
    
    def _generate_explanation_text(self, trade: Dict[str, Any]) -> str:
        """Generate human-readable explanation for a trade."""
        # Basic template
        text = f"Trade on {trade['timestamp']} "
        
        # Decision
        if isinstance(trade['prediction'], (int, bool)):
            # Classification
            decision = "BUY" if trade['prediction'] > 0 else "SELL" if trade['prediction'] < 0 else "HOLD"
            text += f"decision: {decision} "
        else:
            # Regression
            direction = "upward" if trade['prediction'] > 0 else "downward" if trade['prediction'] < 0 else "flat"
            text += f"predicted {direction} movement of {trade['prediction']:.2f}% "
        
        # Confidence
        if trade['confidence'] is not None:
            text += f"with {trade['confidence']:.1%} confidence. "
        else:
            text += ". "
        
        # Market regime
        text += f"Market regime was {trade['regime']}. "
        
        # Feature explanation
        text += "This prediction was driven by: "
        feature_explanations = []
        for feature, importance in list(trade['top_features'].items())[:3]:  # Top 3 features
            # Get feature value
            feature_value = trade.get('feature_values', {}).get(feature, 'N/A')
            
            # Direction of influence
            direction = "positive" if importance > 0 else "negative"
            
            feature_explanations.append(
                f"{feature} ({feature_value}) with {direction} influence of {abs(importance):.4f}"
            )
        
        text += ", ".join(feature_explanations)
        
        # Outcome if available
        if 'actual_outcome' in trade:
            text += f". Actual outcome: {trade['actual_outcome']}, resulting in {trade.get('pnl', 0):.2f} P&L."
            if trade.get('correct'):
                text += " Prediction was correct."
            else:
                text += " Prediction was incorrect."
        
        return text
    
    def compare_regimes(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare model performance across different market regimes.
        
        Returns:
            Dictionary mapping regimes to performance metrics
        """
        results = {}
        
        # Analyze each regime
        for regime in self.performance_by_regime:
            trades = [t for entry in self.performance_by_regime[regime] 
                    for t in self.trades_history 
                    if t['timestamp'] == entry['timestamp']]
            
            if trades:
                results[regime] = self.analyze_model_performance(regime=regime)
            
        # Add overall performance
        results['overall'] = self.analyze_model_performance()
        
        return results
    
    def _filter_trades(self, regime: str = None, timeframe: Tuple[datetime, datetime] = None, 
                     model_name: str = None) -> List[Dict[str, Any]]:
        """Filter trades based on criteria."""
        filtered = self.trades_history
        
        # Filter by regime
        if regime:
            filtered = [t for t in filtered if t['regime'] == regime]
        
        # Filter by timeframe
        if timeframe:
            start, end = timeframe
            filtered = [t for t in filtered if start <= t['timestamp'] <= end]
        
        # Filter by model
        if model_name:
            filtered = [t for t in filtered if t['model_name'] == model_name]
        
        return filtered
    
    def get_performance_summary(self, timeframes: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get performance summary across different timeframes.
        
        Args:
            timeframes: List of timeframes to analyze ('week', 'month', 'quarter', 'year')
            
        Returns:
            Dictionary with performance metrics for each timeframe
        """
        if not self.trades_history:
            return {'error': 'No trade history available'}
        
        # Default timeframes
        if timeframes is None:
            timeframes = ['week', 'month', 'quarter', 'year', 'all']
            
        result = {}
        now = datetime.now()
        
        for period in timeframes:
            if period == 'week':
                start = now - timedelta(days=7)
            elif period == 'month':
                start = now - timedelta(days=30)
            elif period == 'quarter':
                start = now - timedelta(days=90)
            elif period == 'year':
                start = now - timedelta(days=365)
            elif period == 'all':
                start = datetime.min
            else:
                continue
                
            timeframe = (start, now)
            result[period] = self.analyze_model_performance(timeframe=timeframe)
            
        return result
    
    def get_feature_trends(self, top_n: int = 5, 
                         timeframe: Tuple[datetime, datetime] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get trends in feature importance over time.
        
        Args:
            top_n: Number of top features to include
            timeframe: Optional timeframe filter
            
        Returns:
            Dictionary mapping features to importance history
        """
        # Get top features overall
        stability_data = self.analyze_feature_stability(top_n=top_n, timeframe=timeframe)
        
        if 'error' in stability_data:
            return stability_data
            
        top_features = list(stability_data['top_features'].keys())
        
        # Get history for these features
        result = {}
        for feature in top_features:
            history = self.feature_usage_history.get(feature, [])
            
            if timeframe:
                history = [h for h in history if timeframe[0] <= h['timestamp'] <= timeframe[1]]
                
            # Sort by timestamp
            history = sorted(history, key=lambda x: x['timestamp'])
            
            # Store for result
            result[feature] = history
            
        return result
    
    def plot_performance_by_regime(self, save_path: str = None) -> None:
        """
        Plot performance metrics by market regime.
        
        Args:
            save_path: Optional path to save the plot
        """
        regime_data = self.compare_regimes()
        
        # Extract metrics
        regimes = list(regime_data.keys())
        accuracy = [regime_data[r]['accuracy'] * 100 for r in regimes]
        win_rate = [regime_data[r]['win_rate'] * 100 for r in regimes]
        avg_pnl = [regime_data[r]['avg_pnl'] for r in regimes]
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Bar plots for accuracy and win rate
        x = np.arange(len(regimes))
        width = 0.35
        
        ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='skyblue')
        ax1.bar(x + width/2, win_rate, width, label='Win Rate (%)', color='lightgreen')
        
        # Line plot for average PnL
        ax2 = ax1.twinx()
        ax2.plot(x, avg_pnl, 'ro-', linewidth=2, markersize=8, label='Avg PnL')
        
        # Set labels and title
        ax1.set_xlabel('Market Regime')
        ax1.set_ylabel('Percentage (%)')
        ax2.set_ylabel('Average P&L')
        plt.title('Trading Performance by Market Regime')
        
        # Set x-ticks
        ax1.set_xticks(x)
        ax1.set_xticklabels(regimes, rotation=45)
        
        # Add legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
    
    def plot_feature_importance_stability(self, save_path: str = None) -> None:
        """
        Plot feature importance stability over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        stability_data = self.analyze_feature_stability(top_n=10)
        
        if 'error' in stability_data:
            print(stability_data['error'])
            return
            
        # Extract data
        features = list(stability_data['top_features'].keys())
        avg_importance = [stability_data['top_features'][f]['avg_importance'] for f in features]
        stability = [stability_data['stability_scores'][f] for f in features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot with size based on stability
        scatter = ax.scatter(avg_importance, range(len(features)), 
                           s=[s*100 for s in stability], 
                           alpha=0.6, 
                           c=stability, 
                           cmap='viridis')
        
        # Set labels and title
        ax.set_xlabel('Average Importance')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        plt.title('Feature Importance and Stability')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Stability Score')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
    
    def _save_logs(self):
        """Save logs to disk."""
        # Save prediction log
        log_file = os.path.join(self.log_dir, 'prediction_log.pkl')
        with open(log_file, 'wb') as f:
            pickle.dump(self.prediction_log, f)
        
        # Save trades history
        trades_file = os.path.join(self.log_dir, 'trades_history.pkl')
        with open(trades_file, 'wb') as f:
            pickle.dump(self.trades_history, f)
        
        # Save performance by regime
        regime_file = os.path.join(self.log_dir, 'regime_performance.json')
        with open(regime_file, 'w') as f:
            json.dump(
                {regime: [dict(p) for p in perfs] for regime, perfs in self.performance_by_regime.items()}, 
                f, 
                default=self._json_serializer
            )
    
    def _json_serializer(self, obj):
        """Helper method to serialize datetime objects to JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable") 