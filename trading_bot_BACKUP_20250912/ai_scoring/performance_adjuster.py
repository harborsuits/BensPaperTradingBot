"""
Performance Adjuster Module

This module provides functionality to adjust strategy allocations based on
recent performance metrics, enabling dynamic capital allocation for strategies.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerformanceAdjuster")


class PerformanceAdjuster:
    """
    Adjusts strategy allocations based on recent performance metrics.
    """
    
    def __init__(
        self,
        performance_weight: float = 0.3,
        performance_metrics: List[str] = None
    ):
        """
        Initialize the PerformanceAdjuster.
        
        Args:
            performance_weight: Weight to give performance metrics vs market regime (0-1)
            performance_metrics: List of metrics to use for performance evaluation
        """
        self.performance_weight = performance_weight
        self.performance_metrics = performance_metrics or ['sharpe_ratio', 'total_return', 'win_rate', 'max_drawdown']
        
    def adjust_allocations(
        self, 
        allocations: Dict[str, float], 
        performance_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Adjust strategy allocations based on recent performance metrics.
        
        Args:
            allocations: Original allocations based on market regime
            performance_data: Dictionary of performance metrics by strategy
            
        Returns:
            Adjusted allocations with performance weighting
        """
        logger.info("Adjusting allocations based on performance metrics")
        
        # Create a copy of original allocations
        adjusted_allocations = allocations.copy()
        
        # Calculate performance scores for each strategy
        performance_scores = {}
        metrics_available = False
        
        for strategy, metrics in performance_data.items():
            if strategy not in allocations:
                continue
                
            # Skip if no metrics available
            if not metrics:
                performance_scores[strategy] = 1.0  # Neutral score
                continue
                
            # Calculate weighted score from available metrics
            score = 0
            weight_sum = 0
            
            # Sharpe ratio (higher is better)
            if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] is not None:
                score += max(0.1, min(3.0, metrics['sharpe_ratio'])) * 0.4
                weight_sum += 0.4
                metrics_available = True
                
            # Total return (higher is better)
            if 'total_return' in metrics and metrics['total_return'] is not None:
                # Convert from percentage to decimal if needed
                return_value = metrics['total_return']
                if abs(return_value) > 10:  # Likely a percentage
                    return_value /= 100
                score += max(-1.0, min(3.0, return_value * 10)) * 0.3
                weight_sum += 0.3
                metrics_available = True
                
            # Win rate (higher is better)
            if 'win_rate' in metrics and metrics['win_rate'] is not None:
                win_rate = metrics['win_rate']
                # Convert from percentage to decimal if needed
                if win_rate > 1:
                    win_rate /= 100
                score += max(0.1, min(1.5, win_rate * 2)) * 0.2
                weight_sum += 0.2
                metrics_available = True
                
            # Max drawdown (lower is better, so we invert)
            if 'max_drawdown' in metrics and metrics['max_drawdown'] is not None:
                drawdown = abs(metrics['max_drawdown'])
                # Convert from percentage to decimal if needed
                if drawdown > 1:
                    drawdown /= 100
                # Invert so lower drawdown gives higher score
                score += max(0.1, min(1.0, (1 - drawdown) * 1.5)) * 0.1
                weight_sum += 0.1
                metrics_available = True
            
            # Normalize score if we have metrics
            if weight_sum > 0:
                performance_scores[strategy] = score / weight_sum
            else:
                performance_scores[strategy] = 1.0  # Neutral score
        
        # If no metrics available, return original allocations
        if not metrics_available:
            logger.warning("No valid performance metrics found, using original allocations")
            return allocations
            
        # Scale scores relative to average (to avoid all positive/negative adjustments)
        score_values = list(performance_scores.values())
        if score_values:
            avg_score = sum(score_values) / len(score_values)
            relative_scores = {s: score / avg_score for s, score in performance_scores.items()}
        else:
            relative_scores = {s: 1.0 for s in performance_scores}
        
        # Apply performance adjustment with configurable weight
        for strategy in allocations:
            if strategy in relative_scores:
                # Original allocation with blend of performance adjustment
                original = allocations[strategy]
                perf_factor = relative_scores[strategy]
                
                # Apply adjustment with configured weight
                adjusted = original * (1 - self.performance_weight) + \
                        (original * perf_factor) * self.performance_weight
                
                # Ensure we don't reduce too drastically
                min_allocation = original * 0.25
                adjusted = max(min_allocation, adjusted)
                
                adjusted_allocations[strategy] = adjusted
        
        # Normalize to sum to 100%
        total = sum(adjusted_allocations.values())
        if total > 0:
            normalized = {s: (alloc / total) * 100 for s, alloc in adjusted_allocations.items()}
        else:
            normalized = allocations  # Fallback to original
            
        # Log adjustments for debugging
        logger.debug("Performance scores: %s", relative_scores)
        logger.debug("Original allocations: %s", {s: f"{v:.1f}%" for s, v in allocations.items()})
        logger.debug("Performance-adjusted: %s", {s: f"{v:.1f}%" for s, v in normalized.items()})
        
        return normalized


# Command-line testing
if __name__ == "__main__":
    # Set up some test data
    allocations = {
        'trend_following': 30.0,
        'momentum': 20.0, 
        'mean_reversion': 15.0,
        'breakout_swing': 15.0,
        'volatility_breakout': 10.0,
        'option_spreads': 10.0
    }
    
    # Sample performance metrics
    performance_data = {
        'trend_following': {'sharpe_ratio': 1.2, 'total_return': 8.5, 'win_rate': 0.55, 'max_drawdown': -12.0},
        'momentum': {'sharpe_ratio': 1.8, 'total_return': 14.2, 'win_rate': 0.62, 'max_drawdown': -18.0},
        'mean_reversion': {'sharpe_ratio': 0.9, 'total_return': 4.2, 'win_rate': 0.48, 'max_drawdown': -8.0},
        'breakout_swing': {'sharpe_ratio': 1.5, 'total_return': 11.0, 'win_rate': 0.58, 'max_drawdown': -15.0},
        'volatility_breakout': {'sharpe_ratio': 1.3, 'total_return': 12.5, 'win_rate': 0.51, 'max_drawdown': -22.0},
        'option_spreads': {'sharpe_ratio': 1.0, 'total_return': 7.5, 'win_rate': 0.65, 'max_drawdown': -10.0}
    }
    
    # Initialize adjuster
    adjuster = PerformanceAdjuster(performance_weight=0.4)
    
    # Apply adjustment
    adjusted = adjuster.adjust_allocations(allocations, performance_data)
    
    # Print results
    print("\nOriginal Allocations:")
    for strategy, alloc in allocations.items():
        print(f"  {strategy}: {alloc:.1f}%")
        
    print("\nAdjusted Allocations:")
    for strategy, alloc in adjusted.items():
        print(f"  {strategy}: {alloc:.1f}%")
