"""
Hybrid Strategy Optimizer Module

Integrates the advanced optimizer framework with the hybrid strategy system
to optimize strategy weights across different market regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import os
import json

from trading_bot.strategies.hybrid_strategy_system import HybridStrategySystem
from trading_bot.strategies.hybrid_strategy_adapter import HybridStrategyAdapter
from trading_bot.ml_pipeline.optimizer_integration import OptimizerIntegration
from trading_bot.ml_pipeline.market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

class HybridStrategyOptimizer:
    """
    Specialized optimizer for the Hybrid Strategy System
    
    Optimizes the weights between technical, ML, and custom components
    while considering different market regimes for optimal performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the hybrid strategy optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.optimizer_integration = OptimizerIntegration(config=self.config)
        self.regime_detector = MarketRegimeDetector(config=self.config.get('market_regime', {}))
        
        # Results directory
        self.results_dir = self.config.get('results_dir', 'hybrid_optimization_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Hybrid Strategy Optimizer initialized")
    
    def optimize_strategy_weights(self,
                                symbols: List[str],
                                timeframes: List[str] = ['1h', '4h', 'D'],
                                optimization_method: str = 'genetic',
                                metric: str = 'sharpe_ratio',
                                use_multi_timeframe: bool = True,
                                include_regime_detection: bool = True,
                                use_walk_forward: bool = False,
                                days: int = 180) -> Dict[str, Any]:
        """
        Optimize the weights for the hybrid strategy system
        
        Args:
            symbols: List of symbols to optimize for
            timeframes: List of timeframes to test
            optimization_method: Optimization method ('genetic', 'bayesian', etc.)
            metric: Metric to optimize ('sharpe_ratio', 'total_profit', etc.)
            use_multi_timeframe: Whether to test across multiple timeframes
            include_regime_detection: Whether to use market regime detection
            use_walk_forward: Whether to use walk-forward testing
            days: Number of days of historical data to use
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hybrid strategy weight optimization for {len(symbols)} symbols")
        
        # Define parameter space for hybrid strategy weights
        param_space = {
            'tech_weight': np.linspace(0.1, 0.6, 6).tolist(),  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            'ml_weight': np.linspace(0.1, 0.6, 6).tolist(),    # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            'custom_weight': np.linspace(0.1, 0.6, 6).tolist(),  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            'min_confidence': np.linspace(0.5, 0.9, 5).tolist(),  # [0.5, 0.6, 0.7, 0.8, 0.9]
            'use_regime_adaptation': [True, False]
        }
        
        # Configure optimizer
        optimizer_config = {
            # General configuration
            'results_dir': self.results_dir,
            
            # Genetic algorithm configuration (if used)
            'population_size': 30,
            'generations': 10,
            'crossover_rate': 0.7,
            'mutation_rate': 0.2,
            
            # Bayesian optimization configuration (if used)
            'n_calls': 50,
            'n_initial_points': 10,
            
            # Walk-forward configuration (if used)
            'in_sample_days': 90,
            'out_sample_days': 30,
            'step_days': 30,
            'n_windows': 4
        }
        
        # Run optimization
        results = self.optimizer_integration.run_optimization(
            strategy_type='hybrid',
            param_space=param_space,
            symbols=symbols,
            timeframes=timeframes,
            optimization_method=optimization_method,
            metric=metric,
            use_multi_timeframe=use_multi_timeframe,
            include_regime_detection=include_regime_detection,
            use_walk_forward=use_walk_forward,
            days=days,
            optimizer_config=optimizer_config
        )
        
        # Extract optimization results
        if 'best_params' in results:
            best_params = results['best_params']
            self._normalize_weights(best_params)
            
            logger.info(f"Optimal hybrid strategy weights: {best_params}")
            
            # Apply the optimized parameters
            self._apply_optimized_weights(best_params)
        
        # Save regime-specific weights if available
        if include_regime_detection and 'regime_results' in results:
            self._save_regime_specific_weights(results)
        
        return results
    
    def _normalize_weights(self, weights: Dict[str, Any]):
        """
        Normalize strategy weights to sum to 1.0
        
        Args:
            weights: Dictionary with weights to normalize
        """
        weight_keys = ['tech_weight', 'ml_weight', 'custom_weight']
        
        # Extract weights
        weight_values = [weights.get(key, 0.33) for key in weight_keys]
        
        # Calculate sum
        weight_sum = sum(weight_values)
        
        # Normalize if sum is not 0
        if weight_sum > 0:
            for i, key in enumerate(weight_keys):
                weights[key] = weight_values[i] / weight_sum
    
    def _apply_optimized_weights(self, weights: Dict[str, Any]):
        """
        Apply optimized weights to the hybrid strategy system
        
        Args:
            weights: Dictionary with optimized weights
        """
        # In a real implementation, would update system config
        # For now, just log
        logger.info(f"Applying optimized weights to hybrid strategy system: {weights}")
    
    def _save_regime_specific_weights(self, results: Dict[str, Any]):
        """
        Save regime-specific strategy weights
        
        Args:
            results: Optimization results with regime-specific data
        """
        # Extract regime results
        regime_results = results.get('regime_results', {})
        
        # Create regime-specific weights if they exist
        regime_weights = {}
        
        for regime, symbols_results in regime_results.items():
            # Aggregate metrics across symbols
            regime_metrics = {}
            
            for symbol, metrics in symbols_results.items():
                for metric_name, value in metrics.items():
                    if metric_name not in regime_metrics:
                        regime_metrics[metric_name] = []
                    regime_metrics[metric_name].append(value)
            
            # Calculate average metrics
            avg_metrics = {
                metric: np.mean(values) for metric, values in regime_metrics.items()
                if len(values) > 0
            }
            
            # Create recommended weights for this regime
            # For simplicity, we'll modify the optimal weights slightly based on regime
            base_weights = results.get('best_params', {}).copy()
            
            if regime == 'trending_up':
                # In trending up markets, increase ML weight
                if 'ml_weight' in base_weights:
                    base_weights['ml_weight'] = min(base_weights['ml_weight'] * 1.2, 0.6)
                    self._normalize_weights(base_weights)
            elif regime == 'trending_down':
                # In trending down markets, increase custom_weight
                if 'custom_weight' in base_weights:
                    base_weights['custom_weight'] = min(base_weights['custom_weight'] * 1.2, 0.6)
                    self._normalize_weights(base_weights)
            elif regime == 'volatile':
                # In volatile markets, balance weights more evenly
                base_weights['tech_weight'] = 0.33
                base_weights['ml_weight'] = 0.33
                base_weights['custom_weight'] = 0.34
            elif regime == 'ranging':
                # In ranging markets, increase technical weight
                if 'tech_weight' in base_weights:
                    base_weights['tech_weight'] = min(base_weights['tech_weight'] * 1.2, 0.6)
                    self._normalize_weights(base_weights)
            
            # Store regime-specific weights
            regime_weights[regime] = {
                'weights': base_weights,
                'metrics': avg_metrics
            }
        
        # Save regime-specific weights
        filename = f"regime_specific_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(regime_weights, f, indent=2, default=str)
        
        logger.info(f"Saved regime-specific weights to {filepath}")
    
    def get_regime_specific_weights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get regime-specific weights for current market conditions
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with regime-specific weights
        """
        # Detect current market regime
        current_regime = self.regime_detector.detect_regime(data)
        
        # Find latest regime-specific weights file
        weights_files = [f for f in os.listdir(self.results_dir) if f.startswith('regime_specific_weights_')]
        
        if not weights_files:
            logger.warning("No regime-specific weights found")
            return {}
        
        # Sort by timestamp (newest first)
        weights_files.sort(reverse=True)
        latest_file = os.path.join(self.results_dir, weights_files[0])
        
        try:
            with open(latest_file, 'r') as f:
                regime_weights = json.load(f)
            
            # Get weights for current regime
            if current_regime['regime'] in regime_weights:
                regime_data = regime_weights[current_regime['regime']]
                
                logger.info(f"Using regime-specific weights for {current_regime['regime']} " +
                          f"regime (confidence: {current_regime['confidence']:.2f})")
                
                return {
                    'regime': current_regime['regime'],
                    'confidence': current_regime['confidence'],
                    'weights': regime_data['weights'],
                    'description': current_regime['description']
                }
            else:
                logger.warning(f"No weights found for regime: {current_regime['regime']}")
                return current_regime
        except Exception as e:
            logger.error(f"Error loading regime-specific weights: {e}")
            return current_regime
