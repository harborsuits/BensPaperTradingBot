#!/usr/bin/env python
"""
Strategy Analysis Module for Nightly Recap System

This module contains functions for analyzing trading strategy performance,
detecting deterioration in key metrics, and generating alerts.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def analyze_strategy_performance(
    strategy_metrics: Dict[str, Dict[str, Any]], 
    thresholds: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Analyze strategy performance and flag strategies with deteriorating metrics
    
    Args:
        strategy_metrics: Dictionary of strategy metrics
        thresholds: Dictionary of threshold values for alerts
    
    Returns:
        List of strategy alerts
    """
    if not strategy_metrics:
        logger.warning("No strategy metrics available for analysis")
        return []
    
    # Get thresholds
    sharpe_threshold = thresholds.get('sharpe_ratio', 0.5)
    win_rate_threshold = thresholds.get('win_rate', 45.0)
    drawdown_threshold = thresholds.get('max_drawdown', -10.0)
    
    # Get rolling windows
    windows = thresholds.get('rolling_windows', [5, 10, 20, 60])
    
    # Initialize alerts list
    alerts = []
    
    # Analyze each strategy
    for strategy, metrics in strategy_metrics.items():
        # Initialize alert for this strategy
        strategy_alerts = []
        
        # Check overall metrics against thresholds
        if metrics['sharpe_ratio'] < sharpe_threshold:
            strategy_alerts.append({
                'metric': 'sharpe_ratio',
                'value': metrics['sharpe_ratio'],
                'threshold': sharpe_threshold,
                'severity': 'warning',
                'message': f"Sharpe ratio ({metrics['sharpe_ratio']:.2f}) below threshold ({sharpe_threshold:.2f})"
            })
        
        if metrics['win_rate'] < win_rate_threshold:
            strategy_alerts.append({
                'metric': 'win_rate',
                'value': metrics['win_rate'],
                'threshold': win_rate_threshold,
                'severity': 'warning',
                'message': f"Win rate ({metrics['win_rate']:.1f}%) below threshold ({win_rate_threshold:.1f}%)"
            })
        
        if metrics['max_drawdown'] < drawdown_threshold:
            strategy_alerts.append({
                'metric': 'max_drawdown',
                'value': metrics['max_drawdown'],
                'threshold': drawdown_threshold,
                'severity': 'warning',
                'message': f"Max drawdown ({metrics['max_drawdown']:.1f}%) exceeded threshold ({drawdown_threshold:.1f}%)"
            })
        
        # Check deterioration in rolling windows
        for window in windows:
            # Skip if we don't have data for this window
            if f'sharpe_ratio_{window}d' not in metrics:
                continue
            
            # Check Sharpe ratio deterioration
            if metrics['sharpe_ratio'] > 0 and metrics[f'sharpe_ratio_{window}d'] > 0:
                sharpe_deterioration = (metrics[f'sharpe_ratio_{window}d'] / metrics['sharpe_ratio'] - 1) * 100
                # Flag if recent Sharpe is much worse than overall
                if sharpe_deterioration < -20:  # 20% deterioration
                    strategy_alerts.append({
                        'metric': f'sharpe_ratio_{window}d',
                        'value': metrics[f'sharpe_ratio_{window}d'],
                        'compared_to': metrics['sharpe_ratio'],
                        'deterioration': sharpe_deterioration,
                        'severity': 'high' if sharpe_deterioration < -50 else 'medium',
                        'message': f"{window}-day Sharpe ratio ({metrics[f'sharpe_ratio_{window}d']:.2f}) has deteriorated by {abs(sharpe_deterioration):.1f}% compared to overall ({metrics['sharpe_ratio']:.2f})"
                    })
            
            # Check win rate deterioration
            if metrics['win_rate'] > 0 and metrics[f'win_rate_{window}d'] > 0:
                win_rate_deterioration = metrics[f'win_rate_{window}d'] - metrics['win_rate']
                # Flag if recent win rate is much worse than overall
                if win_rate_deterioration < -10:  # 10 percentage points
                    strategy_alerts.append({
                        'metric': f'win_rate_{window}d',
                        'value': metrics[f'win_rate_{window}d'],
                        'compared_to': metrics['win_rate'],
                        'deterioration': win_rate_deterioration,
                        'severity': 'high' if win_rate_deterioration < -20 else 'medium',
                        'message': f"{window}-day win rate ({metrics[f'win_rate_{window}d']:.1f}%) has deteriorated by {abs(win_rate_deterioration):.1f} percentage points compared to overall ({metrics['win_rate']:.1f}%)"
                    })
            
            # Check drawdown deterioration
            if f'max_drawdown_{window}d' in metrics and metrics['max_drawdown'] < 0 and metrics[f'max_drawdown_{window}d'] < 0:
                drawdown_deterioration = metrics[f'max_drawdown_{window}d'] - metrics['max_drawdown']
                # Flag if recent drawdown is much worse than overall
                if drawdown_deterioration < -5:  # 5 percentage points
                    strategy_alerts.append({
                        'metric': f'max_drawdown_{window}d',
                        'value': metrics[f'max_drawdown_{window}d'],
                        'compared_to': metrics['max_drawdown'],
                        'deterioration': drawdown_deterioration,
                        'severity': 'high' if drawdown_deterioration < -10 else 'medium',
                        'message': f"{window}-day max drawdown ({metrics[f'max_drawdown_{window}d']:.1f}%) has deteriorated by {abs(drawdown_deterioration):.1f} percentage points compared to overall ({metrics['max_drawdown']:.1f}%)"
                    })
        
        # If we have alerts, add an entry to the main alerts list
        if strategy_alerts:
            # Determine overall severity
            severity_level = max([alert['severity'] for alert in strategy_alerts], 
                                key=lambda x: {'warning': 1, 'medium': 2, 'high': 3}.get(x, 0))
            
            # Determine if action is required based on severity and weight
            current_weight = metrics['current_weight']
            action_required = (severity_level == 'high' or 
                              (severity_level == 'medium' and current_weight > 0.1) or
                              (severity_level == 'warning' and current_weight > 0.25))
            
            alerts.append({
                'strategy': strategy,
                'current_weight': current_weight,
                'alerts': strategy_alerts,
                'severity': severity_level,
                'action_required': action_required,
                'suggestion': None  # Will be filled in by generate_insights
            })
            
            logger.info(f"Strategy {strategy} has {len(strategy_alerts)} alerts with severity {severity_level}")
        else:
            logger.info(f"Strategy {strategy} has no alerts")
    
    return alerts


def generate_insights(
    alerts: List[Dict[str, Any]], 
    optimization_threshold: float = -20.0
) -> List[Dict[str, Any]]:
    """
    Generate insights and suggestions based on performance analysis
    
    Args:
        alerts: List of strategy alerts
        optimization_threshold: Threshold for triggering optimization (percentage)
    
    Returns:
        List of suggestions
    """
    if not alerts:
        logger.info("No alerts to generate insights from")
        return []
    
    # Initialize suggestions list
    suggestions = []
    
    # Process each alert and generate suggestions
    for alert in alerts:
        strategy = alert['strategy']
        current_weight = alert['current_weight']
        severity = alert['severity']
        
        # Skip if no action required
        if not alert['action_required']:
            continue
        
        # Generate suggestion based on severity and metrics
        if severity == 'high':
            # For high severity, suggest significant weight reduction
            suggested_reduction = min(current_weight, 0.2)  # At most 20% reduction
            new_weight = max(0, current_weight - suggested_reduction)
            
            alert['suggestion'] = {
                'action': 'reduce_allocation',
                'current_weight': current_weight,
                'suggested_weight': new_weight,
                'reduction_percentage': (suggested_reduction / current_weight * 100) if current_weight > 0 else 0,
                'reason': f"Critical deterioration in performance metrics",
                'details': [a['message'] for a in alert['alerts'] if a['severity'] == 'high']
            }
            
        elif severity == 'medium':
            # For medium severity, suggest moderate weight reduction
            suggested_reduction = min(current_weight, 0.1)  # At most 10% reduction
            new_weight = max(0, current_weight - suggested_reduction)
            
            alert['suggestion'] = {
                'action': 'reduce_allocation',
                'current_weight': current_weight,
                'suggested_weight': new_weight,
                'reduction_percentage': (suggested_reduction / current_weight * 100) if current_weight > 0 else 0,
                'reason': f"Significant deterioration in performance metrics",
                'details': [a['message'] for a in alert['alerts'] if a['severity'] in ['high', 'medium']]
            }
            
        elif severity == 'warning' and current_weight > 0.25:
            # For warning severity on high-weighted strategies, suggest slight reduction
            suggested_reduction = min(current_weight, 0.05)  # At most 5% reduction
            new_weight = max(0, current_weight - suggested_reduction)
            
            alert['suggestion'] = {
                'action': 'reduce_allocation',
                'current_weight': current_weight,
                'suggested_weight': new_weight,
                'reduction_percentage': (suggested_reduction / current_weight * 100) if current_weight > 0 else 0,
                'reason': f"Minor deterioration in performance metrics on heavily weighted strategy",
                'details': [a['message'] for a in alert['alerts']]
            }
        
        # Add to suggestions list
        if 'suggestion' in alert and alert['suggestion']:
            suggestions.append({
                'strategy': strategy,
                'suggestion': alert['suggestion']
            })
    
    logger.info(f"Generated {len(suggestions)} suggestions from {len(alerts)} alerts")
    return suggestions
