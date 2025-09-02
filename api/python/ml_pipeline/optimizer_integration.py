"""
Strategy Optimizer Integration Module

Connects the Strategy Optimizer to the main application and provides
convenience functions for optimization.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime

from trading_bot.ml_pipeline.optimizer import (
    BaseOptimizer, 
    GeneticOptimizer, 
    BayesianOptimizer,
    MultiTimeframeOptimizer,
    WalkForwardOptimizer
)
from trading_bot.ml_pipeline.market_regime_detector import MarketRegimeDetector
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.data_handlers.data_loader import DataLoader

logger = logging.getLogger(__name__)

class OptimizerIntegration:
    """
    Integration class for the Strategy Optimizer
    
    Provides high-level functions to run optimizations from
    the main application and dashboard.
    """
    
    def __init__(self, config=None):
        """
        Initialize the optimizer integration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_loader = DataLoader(config=self.config.get('data_loader', {}))
        self.regime_detector = MarketRegimeDetector(config=self.config.get('market_regime', {}))
        
        # Create results directory
        self.results_dir = self.config.get('results_dir', 'optimization_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Strategy Optimizer Integration initialized")
    
    def create_optimizer(self, optimization_method: str, optimizer_config: Dict[str, Any] = None) -> BaseOptimizer:
        """
        Create an optimizer instance based on method
        
        Args:
            optimization_method: Optimization method ('grid', 'random', 'genetic', 'bayesian')
            optimizer_config: Configuration for the optimizer
            
        Returns:
            BaseOptimizer instance
        """
        config = optimizer_config or {}
        
        if optimization_method == 'genetic':
            return GeneticOptimizer(config=config)
        elif optimization_method == 'bayesian':
            return BayesianOptimizer(config=config)
        else:
            return BaseOptimizer(config=config)
    
    def load_data_for_optimization(self, 
                                  symbols: List[str], 
                                  timeframes: List[str],
                                  days: int = 180) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load historical data for optimization
        
        Args:
            symbols: List of symbols to load data for
            timeframes: List of timeframes to load data for
            days: Number of days of history to load
            
        Returns:
            Dict of timeframe -> symbol -> DataFrame with historical data
        """
        logger.info(f"Loading data for optimization: {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        historical_data_sets = {}
        
        for timeframe in timeframes:
            historical_data_sets[timeframe] = {}
            
            for symbol in symbols:
                try:
                    # Load historical data
                    df = self.data_loader.load_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        days=days
                    )
                    
                    if df is not None and not df.empty:
                        historical_data_sets[timeframe][symbol] = df
                    else:
                        logger.warning(f"No data loaded for {symbol} on {timeframe}")
                except Exception as e:
                    logger.error(f"Error loading data for {symbol} on {timeframe}: {e}")
        
        logger.info(f"Loaded data for optimization: {sum(len(data) for data in historical_data_sets.values())} symbol-timeframe combinations")
        
        return historical_data_sets
    
    def run_optimization(self, 
                        strategy_type: str,
                        param_space: Dict[str, Union[List, Tuple]],
                        symbols: List[str],
                        timeframes: List[str],
                        optimization_method: str = 'genetic',
                        metric: str = 'sharpe_ratio',
                        use_multi_timeframe: bool = True,
                        include_regime_detection: bool = True,
                        use_walk_forward: bool = False,
                        days: int = 180,
                        optimizer_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete optimization for a strategy
        
        Args:
            strategy_type: Type of strategy to optimize
            param_space: Dictionary of parameter names and possible values
            symbols: List of symbols to optimize for
            timeframes: List of timeframes to optimize for
            optimization_method: Optimization method ('grid', 'random', 'genetic')
            metric: Metric to optimize for
            use_multi_timeframe: Whether to use multi-timeframe testing
            include_regime_detection: Whether to include market regime detection
            days: Number of days of history to use
            optimizer_config: Additional configuration for optimizer
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting optimization for {strategy_type} strategy")
        
        start_time = datetime.now()
        
        # Load historical data
        historical_data_sets = self.load_data_for_optimization(symbols, timeframes, days)
        
        # Get strategy class
        strategy_class = self._get_strategy_class(strategy_type)
        
        # Create optimizer
        config = optimizer_config or {}
        optimizer = self.create_optimizer(optimization_method, config)
        
        # Run optimization
        if use_walk_forward:
            # Create walk-forward optimizer
            walk_forward_optimizer = WalkForwardOptimizer(optimizer, config)
            
            # Use first timeframe for walk-forward optimization
            first_timeframe = timeframes[0] if timeframes else '1h'
            
            if first_timeframe in historical_data_sets:
                # Run walk-forward optimization
                results = walk_forward_optimizer.optimize(
                    strategy_class,
                    param_space,
                    historical_data_sets[first_timeframe],
                    metric
                )
            else:
                logger.error(f"No data available for timeframe: {first_timeframe}")
                return {"error": f"No data available for timeframe: {first_timeframe}"}
        elif use_multi_timeframe and len(timeframes) > 1:
            # Create multi-timeframe optimizer
            multi_optimizer = MultiTimeframeOptimizer(optimizer, config)
            
            # Run multi-timeframe optimization
            results = multi_optimizer.run_multi_timeframe_test(
                strategy_class,
                param_space,
                historical_data_sets,
                metric,
                None,  # No custom metric function
                self.regime_detector if include_regime_detection else None
            )
        else:
            # Use first timeframe for single-timeframe optimization
            first_timeframe = timeframes[0] if timeframes else '1h'
            
            # Run single-timeframe optimization
            if first_timeframe in historical_data_sets:
                # If regime detection is enabled, test parameters in different regimes
                if include_regime_detection and not use_multi_timeframe:
                    # First get optimal parameters
                    base_results = optimizer.optimize(
                        strategy_class,
                        param_space,
                        historical_data_sets[first_timeframe],
                        metric
                    )
                    
                    # Then test across regimes
                    regime_results = self._test_parameters_by_regime(
                        strategy_class,
                        base_results['best_params'],
                        historical_data_sets[first_timeframe],
                        metric
                    )
                    
                    # Combine results
                    results = base_results
                    results['regime_results'] = regime_results
                else:
                    # Standard optimization without regime testing
                    results = optimizer.optimize(
                        strategy_class,
                        param_space,
                        historical_data_sets[first_timeframe],
                        metric
                    )
            else:
                logger.error(f"No data available for timeframe: {first_timeframe}")
                return {"error": f"No data available for timeframe: {first_timeframe}"}
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Add additional information
        results.update({
            'strategy_type': strategy_type,
            'symbols': symbols,
            'timeframes': timeframes,
            'use_multi_timeframe': use_multi_timeframe,
            'include_regime_detection': include_regime_detection,
            'elapsed_time': elapsed_time,
            'timestamp': end_time.isoformat()
        })
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Optimization completed in {elapsed_time:.1f} seconds")
        return results
    
    def _get_strategy_class(self, strategy_type: str):
        """
        Get strategy class by type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Strategy class
        """
        # This is a simplified implementation - in practice would be more complex
        # to handle different strategy types
        
        # Create a strategy instance of the specified type
        strategy = StrategyFactory.create_strategy(strategy_type)
        
        # Return the class of the strategy
        return strategy.__class__
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save optimization results to disk
        
        Args:
            results: Optimization results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = results.get('strategy_type', 'unknown')
        method = results.get('optimization_method', 'unknown')
        
        filename = f"{strategy_name}_{method}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # In a real implementation, would save to disk here
        logger.info(f"Would save optimization results to {filepath}")
    
    def _test_parameters_by_regime(self,
                              strategy_class,
                              parameters: Dict[str, Any],
                              historical_data: Dict[str, pd.DataFrame],
                              metric: str) -> Dict[str, Dict[str, float]]:
        """
        Test parameters under different market regimes
        
        Args:
            strategy_class: Strategy class to test
            parameters: Parameters to test
            historical_data: Dictionary of symbol -> DataFrame with historical data
            metric: Metric to evaluate
            
        Returns:
            Dict with performance under different regimes
        """
        regime_results = {}
        
        # For each symbol
        for symbol, df in historical_data.items():
            # Skip if data is too small
            if len(df) < 100:
                continue
                
            # Detect regimes
            try:
                regimes = self.regime_detector.detect_regimes(df)
                
                # Group data by regime
                regime_data = {}
                for i, regime in enumerate(regimes):
                    if regime not in regime_data:
                        regime_data[regime] = []
                    regime_data[regime].append(i)
                
                # Test strategy on each regime
                for regime, indices in regime_data.items():
                    if len(indices) < 50:  # Skip if too few data points
                        continue
                        
                    # Create subset of data for this regime
                    regime_df = df.iloc[indices].copy()
                    
                    # Initialize strategy
                    strategy = strategy_class(parameters=parameters)
                    
                    # Generate signals
                    signals = []
                    for i in range(len(regime_df) - 1):
                        df_subset = regime_df.iloc[:i+1].copy()
                        try:
                            signal = strategy.generate_signals(df_subset)
                            if signal:
                                signal['timestamp'] = regime_df.index[i]
                                signal['symbol'] = symbol
                                signals.append(signal)
                        except Exception as e:
                            logger.debug(f"Error generating signal for regime {regime}: {e}")
                    
                    # Calculate metrics
                    if signals:
                        signals_df = pd.DataFrame(signals)
                        # Create a dummy optimizer to access the metrics calculation methods
                        dummy_optimizer = BaseOptimizer()
                        metrics = dummy_optimizer._calculate_metrics(signals_df, regime_df, symbol)
                        
                        # Store results
                        if regime not in regime_results:
                            regime_results[regime] = {}
                        
                        regime_results[regime][symbol] = metrics
            
            except Exception as e:
                logger.error(f"Error detecting regimes for {symbol}: {e}")
        
        return regime_results
    
    def apply_optimized_parameters(self, strategy_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Apply optimized parameters to a strategy
        
        Args:
            strategy_type: Type of strategy
            parameters: Parameters to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real implementation, would update strategy parameters
            # For now, just log
            logger.info(f"Would apply parameters {parameters} to {strategy_type} strategy")
            return True
        except Exception as e:
            logger.error(f"Error applying parameters: {e}")
            return False
