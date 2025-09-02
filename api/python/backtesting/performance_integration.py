#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Performance Integration

This module connects backtest results to the strategy selector's
performance tracking system, creating a feedback loop where strategies 
are selected based on their historical performance in similar market conditions.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from trading_bot.strategies.strategy_template import MarketRegime
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector
from trading_bot.backtesting.backtest_results import BacktestResultsManager

logger = logging.getLogger(__name__)

class PerformanceIntegration:
    """Integrates backtest results with strategy selector performance tracking."""
    
    def __init__(self, backtest_manager: Optional[BacktestResultsManager] = None):
        """
        Initialize the performance integration module.
        
        Args:
            backtest_manager: BacktestResultsManager instance or None to create a new one
        """
        # Initialize backtest results manager
        self.backtest_manager = backtest_manager or BacktestResultsManager()
        
        # Initialize strategy selector for performance tracking
        self.strategy_selector = ForexStrategySelector()
        
        # Track when we last updated performance records
        self.last_update = None
    
    def update_performance_records(self) -> Dict[str, int]:
        """
        Update strategy selector performance records based on backtest results.
        
        Returns:
            Dictionary with counts of updates by strategy
        """
        logger.info("Updating strategy performance records from backtest results")
        
        # Track counts for reporting
        update_counts = {}
        
        # Process each market regime
        for regime in MarketRegime:
            # Skip UNKNOWN regime
            if regime == MarketRegime.UNKNOWN:
                continue
                
            # Get results for this regime
            regime_results = self.backtest_manager.get_regime_results(regime.name)
            
            if not regime_results:
                logger.debug(f"No backtest results found for regime {regime.name}")
                continue
            
            # Process results for each strategy
            for strategy_name in self.strategy_selector.strategy_compatibility.keys():
                # Format to match backtest storage format (remove forex_ prefix if needed)
                backtest_strategy_name = strategy_name
                
                # Get results for this strategy and regime
                strategy_regime_results = [r for r in regime_results 
                                         if r['strategy_name'] == backtest_strategy_name]
                
                if not strategy_regime_results:
                    logger.debug(f"No backtest results for {strategy_name} in {regime.name} regime")
                    continue
                
                # Calculate performance scores for each result
                performance_scores = []
                
                for result in strategy_regime_results:
                    score = self.backtest_manager.calculate_performance_score(result)
                    performance_scores.append(score)
                
                # Record performance in strategy selector
                for score in performance_scores:
                    self.strategy_selector.record_strategy_performance(
                        strategy_name=strategy_name,
                        regime=regime,
                        performance_score=score
                    )
                
                # Update count
                update_counts[strategy_name] = update_counts.get(strategy_name, 0) + len(performance_scores)
        
        # Update last update timestamp
        self.last_update = datetime.now()
        
        logger.info(f"Updated performance records for {sum(update_counts.values())} backtest results")
        
        return update_counts
    
    def optimize_strategy_parameters(self, 
                                   strategy_name: str, 
                                   regime: MarketRegime) -> Optional[Dict[str, Any]]:
        """
        Get optimized parameters for a strategy based on backtest results.
        
        Args:
            strategy_name: Strategy name
            regime: Market regime
            
        Returns:
            Optimized parameters or None if no results found
        """
        # Format strategy name to match backtest storage
        backtest_strategy_name = strategy_name
        
        # Get best parameters from backtest results
        return self.backtest_manager.get_best_parameters(
            strategy_name=backtest_strategy_name,
            market_regime=regime.name
        )
    
    def get_strategy_regime_performance(self, 
                                      strategy_name: str, 
                                      regime: MarketRegime) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy in a specific regime.
        
        Args:
            strategy_name: Strategy name
            regime: Market regime
            
        Returns:
            Dictionary with performance metrics
        """
        # Format strategy name to match backtest storage
        backtest_strategy_name = strategy_name
        
        # Get results
        results = self.backtest_manager.get_strategy_regime_results(
            strategy_name=backtest_strategy_name,
            market_regime=regime.name
        )
        
        if not results:
            return {
                'strategy': strategy_name,
                'regime': regime.name,
                'backtest_count': 0,
                'avg_performance': 0.0,
                'best_performance': 0.0,
                'worst_performance': 0.0,
                'avg_return': 0.0,
                'avg_drawdown': 0.0
            }
        
        # Calculate metrics
        performances = [self.backtest_manager.calculate_performance_score(r) for r in results]
        returns = [r.get('metrics', {}).get('total_return', 0.0) for r in results]
        drawdowns = [r.get('metrics', {}).get('max_drawdown', 0.0) for r in results]
        
        return {
            'strategy': strategy_name,
            'regime': regime.name,
            'backtest_count': len(results),
            'avg_performance': np.mean(performances),
            'best_performance': max(performances),
            'worst_performance': min(performances),
            'avg_return': np.mean(returns),
            'avg_drawdown': np.mean(drawdowns)
        }
    
    def get_performance_matrix(self) -> pd.DataFrame:
        """
        Generate a performance matrix showing how each strategy performs in each regime.
        
        Returns:
            DataFrame with strategy/regime performance matrix
        """
        # Get all strategies
        strategies = list(self.strategy_selector.strategy_compatibility.keys())
        
        # Get all regimes except UNKNOWN
        regimes = [r for r in MarketRegime if r != MarketRegime.UNKNOWN]
        
        # Create matrix
        matrix = []
        
        for strategy in strategies:
            row = {'strategy': strategy}
            
            for regime in regimes:
                # Get performance for this strategy/regime combination
                perf = self.get_strategy_regime_performance(strategy, regime)
                row[regime.name] = perf['avg_performance']
            
            matrix.append(row)
        
        # Convert to DataFrame
        return pd.DataFrame(matrix)
    
    def export_performance_matrix(self, filepath: str = None) -> str:
        """
        Export the performance matrix to a CSV file.
        
        Args:
            filepath: Path to save the file or None for default
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            filepath = os.path.join(
                project_root, 
                'backtest_results', 
                f'performance_matrix_{datetime.now().strftime("%Y%m%d")}.csv'
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get matrix and save
        matrix = self.get_performance_matrix()
        matrix.to_csv(filepath, index=False)
        
        logger.info(f"Exported performance matrix to {filepath}")
        
        return filepath
