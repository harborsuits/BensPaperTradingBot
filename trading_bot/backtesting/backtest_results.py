#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Results Storage & Processing

This module handles the storage and retrieval of backtesting results
for strategy performance tracking and machine learning integration.
"""

import os
import json
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

class BacktestResultsManager:
    """Manages storage and retrieval of backtest results for strategy performance analysis."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize the backtest results manager.
        
        Args:
            results_dir: Directory where backtest results are stored
        """
        if results_dir is None:
            # Default to project_root/backtest_results
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            results_dir = os.path.join(project_root, 'backtest_results')
            
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Default filename for backtest summary
        self.summary_file = os.path.join(results_dir, 'backtest_summary.json')
        
        # Initialize or load summary data
        self.summary_data = self._load_summary()
    
    def _load_summary(self) -> Dict[str, Any]:
        """Load the backtest summary data from disk."""
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading backtest summary: {e}")
                return {'backtest_runs': []}
        else:
            return {'backtest_runs': []}
    
    def save_backtest_result(self, 
                           strategy_name: str,
                           market_regime: str,
                           start_date: str,
                           end_date: str,
                           symbols: List[str],
                           metrics: Dict[str, float],
                           parameters: Dict[str, Any]) -> str:
        """
        Save a backtest result to disk.
        
        Args:
            strategy_name: Name of the strategy
            market_regime: Market regime during backtest
            start_date: Start date of backtest (str format)
            end_date: End date of backtest (str format)
            symbols: List of symbols tested
            metrics: Performance metrics (e.g., return, drawdown, etc.)
            parameters: Strategy parameters used
            
        Returns:
            ID of the saved backtest
        """
        # Generate unique ID for this backtest
        backtest_id = f"bt_{strategy_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create backtest result record
        result = {
            'id': backtest_id,
            'strategy_name': strategy_name,
            'market_regime': market_regime,
            'start_date': start_date,
            'end_date': end_date,
            'symbols': symbols,
            'metrics': metrics,
            'parameters': parameters,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to summary data
        self.summary_data['backtest_runs'].append(result)
        
        # Save to disk
        with open(self.summary_file, 'w') as f:
            json.dump(self.summary_data, f, indent=2)
            
        return backtest_id
    
    def get_strategy_results(self, strategy_name: str) -> List[Dict[str, Any]]:
        """
        Get all backtest results for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            List of backtest results for the strategy
        """
        return [r for r in self.summary_data['backtest_runs'] 
                if r['strategy_name'] == strategy_name]
    
    def get_regime_results(self, market_regime: str) -> List[Dict[str, Any]]:
        """
        Get all backtest results for a specific market regime.
        
        Args:
            market_regime: Market regime
            
        Returns:
            List of backtest results for the market regime
        """
        return [r for r in self.summary_data['backtest_runs'] 
                if r['market_regime'] == market_regime]
    
    def get_strategy_regime_results(self, 
                                  strategy_name: str, 
                                  market_regime: str) -> List[Dict[str, Any]]:
        """
        Get all backtest results for a specific strategy and market regime.
        
        Args:
            strategy_name: Name of the strategy
            market_regime: Market regime
            
        Returns:
            List of backtest results for the strategy and market regime
        """
        return [r for r in self.summary_data['backtest_runs'] 
                if r['strategy_name'] == strategy_name and r['market_regime'] == market_regime]
    
    def calculate_performance_score(self, backtest_result: Dict[str, Any]) -> float:
        """
        Calculate a normalized performance score (0.0-1.0) from backtest metrics.
        
        Args:
            backtest_result: Backtest result dict
            
        Returns:
            Performance score (0.0-1.0)
        """
        metrics = backtest_result.get('metrics', {})
        
        # Extract key performance metrics
        total_return = metrics.get('total_return', 0.0)
        max_drawdown = max(0.1, metrics.get('max_drawdown', 100.0))  # Avoid division by zero
        win_rate = metrics.get('win_rate', 0.0)
        profit_factor = metrics.get('profit_factor', 0.0)
        
        # Normalize return to a 0-1 scale
        norm_return = min(max(0, (total_return / 50) + 0.5), 1.0)  # 0% maps to 0.5, -50% to 0, +50% to 1.0
        
        # Normalize drawdown (lower is better)
        norm_drawdown = min(max(0, 1.0 - (max_drawdown / 50)), 1.0)  # 0% is best (1.0), 50%+ is worst (0.0)
        
        # Normalize win rate
        norm_win_rate = min(max(0, win_rate / 100), 1.0)
        
        # Normalize profit factor (capped at 3.0 for normalization)
        norm_profit_factor = min(max(0, profit_factor / 3.0), 1.0)
        
        # Combine metrics with weights
        # Return and drawdown are most important
        score = (
            norm_return * 0.35 +
            norm_drawdown * 0.35 +
            norm_win_rate * 0.15 +
            norm_profit_factor * 0.15
        )
        
        return max(0.0, min(1.0, score))  # Ensure score is in 0-1 range

    def get_best_parameters(self, 
                          strategy_name: str, 
                          market_regime: str) -> Optional[Dict[str, Any]]:
        """
        Get the parameters from the best-performing backtest for a strategy/regime.
        
        Args:
            strategy_name: Name of the strategy
            market_regime: Market regime
            
        Returns:
            Parameters dict or None if no results found
        """
        results = self.get_strategy_regime_results(strategy_name, market_regime)
        
        if not results:
            return None
            
        # Calculate performance scores for each result
        scored_results = [(r, self.calculate_performance_score(r)) for r in results]
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return parameters from the best result
        if scored_results:
            return scored_results[0][0].get('parameters', {})
        
        return None
