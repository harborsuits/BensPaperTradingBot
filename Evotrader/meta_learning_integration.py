#!/usr/bin/env python3
"""
Meta-Learning Integration Module

Combines insights from meta-learning database, regime detection, and pattern analysis
to optimize evolutionary parameters.
"""

import os
import json
import datetime
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector
from strategy_pattern_analyzer import StrategyPatternAnalyzer
from prop_strategy_registry import PropStrategyRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meta_learning_integration')

class MetaLearningIntegrator:
    """
    Integrates meta-learning insights into the evolution process.
    
    This class:
    1. Detects current market regime
    2. Extracts relevant pattern insights from meta-learning database
    3. Generates optimized evolution parameters
    4. Creates bias configurations for strategy evolution
    """
    
    def __init__(self, 
                config_path: str = None,
                meta_db_path: str = "./meta_learning/meta_db.sqlite",
                output_dir: str = "./evolution/configs"):
        """
        Initialize the meta-learning integrator.
        
        Args:
            config_path: Path to evolution configuration file
            meta_db_path: Path to meta-learning database
            output_dir: Directory to save bias configurations
        """
        self.meta_db = MetaLearningDB(db_path=meta_db_path)
        self.regime_detector = MarketRegimeDetector()
        self.pattern_analyzer = StrategyPatternAnalyzer()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load base evolution configuration
        self.base_config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.base_config = yaml.safe_load(f)
                logger.info(f"Loaded base evolution configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self.base_config = {}
    
    def generate_evolution_config(self, 
                                  price_data: pd.DataFrame,
                                  registry: PropStrategyRegistry = None,
                                  strategy_types: List[str] = None,
                                  min_confidence: float = 0.6,
                                  apply_biasing: bool = True) -> Dict[str, Any]:
        """
        Generate an optimized evolution configuration based on meta-learning insights.
        
        Args:
            price_data: Recent price data for regime detection
            registry: Strategy registry instance (optional)
            strategy_types: List of strategy types to consider (optional)
            min_confidence: Minimum confidence score for biasing
            apply_biasing: Whether to apply biasing based on meta-learning
            
        Returns:
            Evolution configuration dictionary
        """
        # Start with base configuration or defaults
        config = self.base_config.copy() if self.base_config else self._get_default_config()
        
        if not apply_biasing:
            logger.info("Biasing disabled - returning base configuration")
            return config
        
        try:
            # 1. Detect current market regime
            regime_info = self.regime_detector.detect_regime(price_data)
            current_regime = regime_info['regime']
            regime_confidence = regime_info['confidence']
            
            logger.info(f"Detected market regime: {current_regime} (confidence: {regime_confidence:.2f})")
            
            # 2. Get strategy pool from registry (if available)
            strategy_pool = []
            if registry:
                # Get promoted strategies as successful examples
                promoted = registry.get_promotion_candidates(min_confidence=min_confidence)
                strategy_pool.extend(promoted)
                
                # Get some demoted strategies as negative examples
                demoted = registry.get_demotion_candidates(max_confidence=0.4)
                strategy_pool.extend(demoted)
                
                # Get active strategies too
                active = registry.get_active_strategies()
                strategy_pool.extend(active)
                
                logger.info(f"Collected {len(strategy_pool)} strategies from registry for analysis")
            
            # 3. Query meta-learning database for regime-specific insights
            regime_insights = self.meta_db.get_regime_insights(current_regime)
            
            # 4. Analyze strategy patterns for additional insights
            pattern_insights = {}
            if strategy_pool:
                pattern_insights = self.pattern_analyzer.analyze_strategy_pool(
                    strategy_pool, 
                    performance_threshold=min_confidence
                )
            
            # 5. Generate bias specifications
            bias_config = self._generate_bias_config(regime_insights, pattern_insights, current_regime)
            
            # 6. Apply biases to configuration
            config = self._apply_biases_to_config(config, bias_config, strategy_types)
            
            # 7. Save the configuration
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = os.path.join(self.output_dir, f"evolution_config_{current_regime}_{timestamp}.yaml")
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Generated optimized evolution configuration: {config_path}")
            
            return config
        
        except Exception as e:
            logger.error(f"Error generating evolution configuration: {e}")
            return config
    
    def _generate_bias_config(self, 
                             regime_insights: Dict[str, Any],
                             pattern_insights: Dict[str, Any],
                             current_regime: str) -> Dict[str, Any]:
        """
        Generate evolution bias configuration.
        
        Args:
            regime_insights: Insights from meta-learning database
            pattern_insights: Insights from pattern analysis
            current_regime: Current market regime
            
        Returns:
            Bias configuration dictionary
        """
        bias_config = {
            'strategy_type_weights': {},
            'indicator_weights': {},
            'parameter_biases': {},
            'regime': current_regime
        }
        
        # 1. Strategy type biasing based on regime performance
        strategy_performance = regime_insights.get('strategy_type_performance', {})
        
        # Set initial weights based on historical performance
        for strategy_type, performance in strategy_performance.items():
            # Higher weight means more likely to be selected
            weight = 1.0  # Default
            
            # Adjust weight based on win rate and profit factor
            win_rate = performance.get('win_rate', 0.5)
            profit_factor = performance.get('profit_factor', 1.0)
            
            if win_rate > 0.6 and profit_factor > 1.5:
                weight = 2.0
            elif win_rate > 0.5 and profit_factor > 1.2:
                weight = 1.5
            elif win_rate < 0.4 or profit_factor < 0.8:
                weight = 0.5
            
            bias_config['strategy_type_weights'][strategy_type] = weight
        
        # 2. Indicator biasing based on effectiveness
        indicator_effectiveness = {}
        
        # From regime insights
        for indicator, stats in regime_insights.get('indicator_performance', {}).items():
            indicator_effectiveness[indicator] = stats.get('effectiveness', 1.0)
        
        # From pattern analysis
        if 'indicator_patterns' in pattern_insights:
            pattern_effectiveness = pattern_insights['indicator_patterns'].get('relative_effectiveness', {})
            
            for indicator, stats in pattern_effectiveness.items():
                if indicator in indicator_effectiveness:
                    # Average with existing data
                    effectiveness = (indicator_effectiveness[indicator] + stats.get('effectiveness', 1.0)) / 2
                else:
                    effectiveness = stats.get('effectiveness', 1.0)
                
                indicator_effectiveness[indicator] = effectiveness
        
        # Set weights based on effectiveness
        for indicator, effectiveness in indicator_effectiveness.items():
            weight = max(0.5, min(2.0, effectiveness))  # Clamp between 0.5 and 2.0
            bias_config['indicator_weights'][indicator] = weight
        
        # 3. Parameter biases based on cluster analysis
        parameter_biases = {}
        
        # From regime insights
        param_clusters = regime_insights.get('parameter_clusters', {})
        
        for param, clusters in param_clusters.items():
            if not clusters:
                continue
                
            # Find highest performing cluster
            best_cluster = max(clusters, key=lambda c: c.get('performance', 0))
            
            parameter_biases[param] = {
                'center': best_cluster.get('center', 0),
                'range': best_cluster.get('range', [0, 0]),
                'performance': best_cluster.get('performance', 0)
            }
        
        # From pattern analysis
        if 'parameter_patterns' in pattern_insights:
            for strategy_type, params in pattern_insights['parameter_patterns'].items():
                for param, stats in params.items():
                    clustered_values = stats.get('clustered_values', [])
                    
                    if not clustered_values:
                        continue
                    
                    # Find largest cluster
                    best_cluster = max(clustered_values, key=lambda c: c.get('count', 0))
                    param_key = f"{strategy_type}.{param}"
                    
                    parameter_biases[param_key] = {
                        'center': best_cluster.get('center', stats.get('mean', 0)),
                        'range': best_cluster.get('range', [stats.get('min', 0), stats.get('max', 0)]),
                        'strategy_type': strategy_type
                    }
        
        bias_config['parameter_biases'] = parameter_biases
        
        return bias_config
    
    def _apply_biases_to_config(self, 
                               config: Dict[str, Any],
                               bias_config: Dict[str, Any],
                               strategy_types: List[str] = None) -> Dict[str, Any]:
        """
        Apply biases to evolution configuration.
        
        Args:
            config: Base evolution configuration
            bias_config: Bias configuration
            strategy_types: List of strategy types to consider
            
        Returns:
            Updated evolution configuration
        """
        if not bias_config:
            return config
        
        # Track what we modified
        modifications = []
        
        # 1. Apply strategy type weights to population generation
        if 'strategy_type_weights' in bias_config and 'population' in config:
            weights = bias_config['strategy_type_weights']
            
            # Filter to specified strategy types if provided
            if strategy_types:
                weights = {k: v for k, v in weights.items() if k in strategy_types}
            
            if weights:
                config['population']['strategy_type_weights'] = weights
                modifications.append("Applied strategy type weights to population generation")
        
        # 2. Apply indicator weights
        if 'indicator_weights' in bias_config and 'indicators' in config:
            weights = bias_config['indicator_weights']
            
            for indicator, weight in weights.items():
                if indicator in config['indicators']:
                    config['indicators'][indicator]['weight'] = weight
            
            modifications.append("Applied indicator weights based on effectiveness")
        
        # 3. Apply parameter biases to parameter ranges
        if 'parameter_biases' in bias_config and 'parameters' in config:
            biases = bias_config['parameter_biases']
            
            for param, bias in biases.items():
                # Check if parameter has strategy type prefix
                if '.' in param:
                    strategy_type, param_name = param.split('.', 1)
                    
                    # Skip if strategy type filtering is enabled and this type isn't included
                    if strategy_types and strategy_type not in strategy_types:
                        continue
                    
                    # Add to strategy-specific parameter bias
                    if 'strategy_parameters' not in config:
                        config['strategy_parameters'] = {}
                    
                    if strategy_type not in config['strategy_parameters']:
                        config['strategy_parameters'][strategy_type] = {}
                    
                    if param_name not in config['strategy_parameters'][strategy_type]:
                        config['strategy_parameters'][strategy_type][param_name] = {}
                    
                    param_config = config['strategy_parameters'][strategy_type][param_name]
                    
                    # Apply bias
                    self._apply_param_bias(param_config, bias)
                else:
                    # Apply to general parameter if it exists
                    if param in config['parameters']:
                        param_config = config['parameters'][param]
                        self._apply_param_bias(param_config, bias)
            
            modifications.append("Applied parameter biases based on successful clusters")
        
        # 4. Apply market regime to configuration
        current_regime = bias_config.get('regime', 'unknown')
        if current_regime != 'unknown':
            config['market_regime'] = current_regime
            modifications.append(f"Set current market regime to {current_regime}")
        
        # Log modifications
        if modifications:
            logger.info("Applied the following biases to evolution configuration:")
            for mod in modifications:
                logger.info(f"- {mod}")
        else:
            logger.info("No biases were applied to evolution configuration")
        
        return config
    
    def _apply_param_bias(self, param_config: Dict[str, Any], bias: Dict[str, Any]) -> None:
        """
        Apply parameter bias to parameter configuration.
        
        Args:
            param_config: Parameter configuration
            bias: Parameter bias
        """
        # Only apply bias to numeric parameters
        if not isinstance(param_config.get('min', None), (int, float)) or not isinstance(param_config.get('max', None), (int, float)):
            return
        
        # Get current parameter range
        current_min = param_config.get('min', 0)
        current_max = param_config.get('max', 0)
        
        # Get bias information
        bias_center = bias.get('center', (current_min + current_max) / 2)
        bias_range = bias.get('range', [current_min, current_max])
        
        # Calculate new range bounds based on bias and current range
        # We want to shift the range toward the bias center while
        # not completely overriding the original range
        
        # Weight for blending (higher means more bias influence)
        weight = 0.7
        
        # Calculate adjusted min/max
        new_min = current_min * (1 - weight) + bias_range[0] * weight
        new_max = current_max * (1 - weight) + bias_range[1] * weight
        
        # Ensure new range is valid
        if new_min >= new_max:
            new_min = bias_center * 0.9
            new_max = bias_center * 1.1
        
        # Round to original precision
        min_precision = self._get_precision(current_min)
        max_precision = self._get_precision(current_max)
        
        new_min = round(new_min, min_precision)
        new_max = round(new_max, max_precision)
        
        # Update parameter configuration
        param_config['min'] = new_min
        param_config['max'] = new_max
        
        # Adjust default value if present
        if 'default' in param_config:
            current_default = param_config['default']
            
            # Only adjust numeric defaults
            if isinstance(current_default, (int, float)):
                # Bias default toward the center of successful values
                new_default = current_default * (1 - weight) + bias_center * weight
                
                # Round to original precision
                default_precision = self._get_precision(current_default)
                param_config['default'] = round(new_default, default_precision)
    
    def _get_precision(self, value: float) -> int:
        """Get decimal precision of a number."""
        value_str = str(value)
        if '.' in value_str:
            return len(value_str) - value_str.index('.') - 1
        return 0
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default evolution configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'population': {
                'size': 100,
                'initial_diversity': 0.8,
                'strategy_type_weights': {
                    'trend_following': 1.0,
                    'mean_reversion': 1.0,
                    'breakout': 1.0,
                    'pattern_recognition': 1.0,
                    'multi_timeframe': 1.0
                }
            },
            'evolution': {
                'generations': 20,
                'selection_percent': 0.3,
                'mutation_rate': 0.2,
                'crossover_rate': 0.7,
                'elitism_count': 5
            },
            'indicators': {
                'macd': {'weight': 1.0},
                'rsi': {'weight': 1.0},
                'bollinger_bands': {'weight': 1.0},
                'ema': {'weight': 1.0},
                'sma': {'weight': 1.0},
                'atr': {'weight': 1.0},
                'adx': {'weight': 1.0},
                'stochastic': {'weight': 1.0},
                'ichimoku': {'weight': 1.0},
                'volume': {'weight': 1.0}
            },
            'parameters': {
                'rsi_period': {'min': 7, 'max': 21, 'default': 14},
                'macd_fast': {'min': 8, 'max': 16, 'default': 12},
                'macd_slow': {'min': 18, 'max': 30, 'default': 26},
                'macd_signal': {'min': 5, 'max': 12, 'default': 9},
                'bollinger_period': {'min': 10, 'max': 30, 'default': 20},
                'bollinger_std': {'min': 1.5, 'max': 3.0, 'default': 2.0},
                'ema_period': {'min': 5, 'max': 50, 'default': 21},
                'sma_period': {'min': 5, 'max': 50, 'default': 21},
                'atr_period': {'min': 7, 'max': 21, 'default': 14},
                'adx_period': {'min': 7, 'max': 21, 'default': 14},
                'stoch_k': {'min': 5, 'max': 21, 'default': 14},
                'stoch_d': {'min': 3, 'max': 9, 'default': 3},
                'stop_loss': {'min': 1.0, 'max': 5.0, 'default': 2.0},
                'take_profit': {'min': 1.5, 'max': 8.0, 'default': 3.0}
            }
        }


if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Example usage
    integrator = MetaLearningIntegrator()
    
    # Get some sample price data
    today = datetime.now()
    start_date = today - timedelta(days=100)
    end_date = today
    
    try:
        # Try to get EURUSD data (if yfinance is available)
        df = yf.download('EURUSD=X', start=start_date, end=end_date, interval='1d')
        
        if len(df) > 0:
            # Generate optimized evolution configuration
            config = integrator.generate_evolution_config(df)
            print(f"Generated evolution configuration with biases for {config.get('market_regime', 'unknown')} regime")
        else:
            print("No price data available")
    except Exception as e:
        print(f"Error downloading price data: {e}")
        
        # Fall back to dummy data for demonstration
        dates = pd.date_range(start=start_date, end=end_date)
        data = {
            'Open': np.random.normal(100, 2, len(dates)),
            'High': np.random.normal(101, 2, len(dates)),
            'Low': np.random.normal(99, 2, len(dates)),
            'Close': np.random.normal(100, 2, len(dates)),
            'Volume': np.random.normal(1000, 200, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        
        # Convert column names to lowercase to match expected format
        df.columns = [c.lower() for c in df.columns]
        
        # Generate optimized evolution configuration
        config = integrator.generate_evolution_config(df)
        print(f"Generated evolution configuration with biases for {config.get('market_regime', 'unknown')} regime using dummy data")
