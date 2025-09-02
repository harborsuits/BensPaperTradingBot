#!/usr/bin/env python3
"""
Production Approval Workflow Integration

This module builds directly on our successful enhanced_approval_integration.py,
adding five major advanced features:

1. Performance Benchmark Integration - Compare against static strategies
2. Adaptive Rejection Threshold - Dynamic approval criteria
3. Historical Performance Database - Repository for meta-learning
4. Advanced Learning Algorithms - Bayesian optimization techniques
5. Production Testing Framework - Low-risk shadow trading infrastructure

All enhancements maintain backward compatibility with our existing system.
"""

import uuid
import time
import os
import json
import random
import datetime
import math
import statistics
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict, deque

# Import our successful components as a foundation
from enhanced_approval_integration import (
    EnhancedApprovalWorkflowIntegration, 
    EnhancedApprovalWorkflowManager
)
from multi_objective_simplified import (
    MultiObjectiveFeedbackLoop, 
    SimpleStrategy, 
    REAL_WORLD_PARAMS
)
from feedback_loop_enhancements import patch_feedback_loop
from feedback_loop_text_visualized import TextVisualization

# Try to import the approval workflow components
try:
    from trading_bot.autonomous.approval_workflow import get_approval_workflow_manager
    from trading_bot.event_system import EventBus, Event, EventType
    WORKFLOW_AVAILABLE = True
except ImportError:
    print("Approval workflow components not available. Running in simulation mode.")
    WORKFLOW_AVAILABLE = False
    
    # Import simulation classes from our enhanced implementation
    from enhanced_approval_integration import (
        get_enhanced_approval_workflow_manager,
        EventBus, Event, EventType
    )

#-----------------------------------------------------------
# 1. PERFORMANCE BENCHMARK INTEGRATION
#-----------------------------------------------------------

class BenchmarkComparison:
    """Compares optimized strategies to static benchmark strategies."""
    
    def __init__(self, feedback_loop):
        """
        Initialize the benchmark comparison system.
        
        Args:
            feedback_loop: The feedback loop containing market generation capability
        """
        # Static benchmark strategies for comparison
        self.benchmarks = {
            "buy_and_hold": {"trend_period": 20, "entry_threshold": 0, "exit_threshold": 0, 
                           "stop_loss": 100, "position_size": 1.0},
            "conservative": {"trend_period": 50, "entry_threshold": 2.0, "exit_threshold": -1.0, 
                           "stop_loss": 5.0, "position_size": 0.2},
            "aggressive": {"trend_period": 10, "entry_threshold": 0.5, "exit_threshold": -0.5, 
                         "stop_loss": 20.0, "position_size": 0.8}
        }
        self.feedback_loop = feedback_loop
        
    def evaluate_all_regimes(self, strategy_params):
        """
        Evaluate a strategy against benchmarks across all regimes.
        
        Args:
            strategy_params: Parameters of the strategy to evaluate
            
        Returns:
            Dict with benchmark comparison results by regime
        """
        results = {}
        
        # Test the strategy and benchmarks on each regime
        for regime in ['bullish', 'bearish', 'sideways', 'volatile']:
            # Generate test data for this regime
            market_params = self.feedback_loop.market_models.get(regime, {})
            if not market_params:
                continue
                
            # Generate market data for this regime
            trend = market_params.get('trend', 0.001)
            volatility = market_params.get('volatility', 0.02)
            mean_reversion = market_params.get('mean_reversion', 0.3)
            
            # Use the generate_market_data function from feedback loop
            from multi_objective_simplified import generate_market_data
            market_data = generate_market_data(
                days=252, 
                trend=trend, 
                volatility=volatility, 
                mean_reversion=mean_reversion
            )
            
            # Evaluate the optimized strategy
            strategy = SimpleStrategy(strategy_params)
            opt_performance = strategy.backtest(market_data)
            
            # Evaluate benchmark strategies
            benchmark_results = {}
            for name, params in self.benchmarks.items():
                bench_strategy = SimpleStrategy(params)
                benchmark_results[name] = bench_strategy.backtest(market_data)
            
            # Calculate outperformance metrics
            outperformance = {}
            for name, perf in benchmark_results.items():
                return_diff = opt_performance['return'] - perf['return']
                risk_ratio = perf['max_drawdown'] / max(0.1, opt_performance['max_drawdown'])
                sharpe_diff = opt_performance['sharpe'] - perf['sharpe']
                
                outperformance[name] = {
                    'return_diff': return_diff,
                    'risk_ratio': risk_ratio,
                    'sharpe_diff': sharpe_diff,
                    'overall_improvement': (return_diff * 0.4) + (risk_ratio * 0.3) + (sharpe_diff * 0.3)
                }
                
            results[regime] = {
                'strategy_performance': opt_performance,
                'benchmark_performance': benchmark_results,
                'outperformance': outperformance
            }
            
        return results
    
    def summarize_benchmark_results(self, results):
        """
        Create a readable text summary of benchmark comparison results.
        
        Args:
            results: Results from evaluate_all_regimes method
            
        Returns:
            Text summary of benchmark results
        """
        summary = []
        summary.append("=" * 50)
        summary.append("BENCHMARK PERFORMANCE COMPARISON")
        summary.append("=" * 50)
        
        # Overall summary across all regimes
        overall_improvement = 0
        regime_count = 0
        
        for regime, data in results.items():
            regime_summary = []
            regime_summary.append(f"\n{regime.upper()} REGIME")
            regime_summary.append("-" * 20)
            
            strategy_perf = data['strategy_performance']
            regime_summary.append(f"Strategy: Return={strategy_perf['return']:.2f}%, " +
                               f"Drawdown={strategy_perf['max_drawdown']:.2f}%, " +
                               f"Sharpe={strategy_perf['sharpe']:.2f}")
            
            # Benchmark comparisons
            for bench_name, bench_perf in data['benchmark_performance'].items():
                outperf = data['outperformance'][bench_name]
                
                # Add a visual indicator of outperformance
                indicator = "✓" if outperf['overall_improvement'] > 0 else "✗"
                
                regime_summary.append(f"{indicator} vs {bench_name}: " +
                                   f"Return {outperf['return_diff']:+.2f}%, " +
                                   f"Risk {outperf['risk_ratio']:.2f}x, " +
                                   f"Sharpe {outperf['sharpe_diff']:+.2f}")
                                   
            # Average improvement for this regime
            avg_improvement = sum(o['overall_improvement'] 
                                for o in data['outperformance'].values()) / len(data['outperformance'])
            regime_summary.append(f"Avg improvement: {avg_improvement:.2f}")
            
            # Add to overall stats
            overall_improvement += avg_improvement
            regime_count += 1
            
            # Add regime summary to main summary
            summary.extend(regime_summary)
        
        # Add overall summary if we have data
        if regime_count > 0:
            avg_overall = overall_improvement / regime_count
            summary.append("\n" + "=" * 50)
            summary.append(f"OVERALL IMPROVEMENT: {avg_overall:.2f}")
            
            # Classification
            if avg_overall > 1.0:
                summary.append("ASSESSMENT: Exceptional outperformance")
            elif avg_overall > 0.5:
                summary.append("ASSESSMENT: Strong outperformance")
            elif avg_overall > 0:
                summary.append("ASSESSMENT: Modest outperformance")
            else:
                summary.append("ASSESSMENT: Underperformance relative to benchmarks")
            
            summary.append("=" * 50)
        
        return "\n".join(summary)

#-----------------------------------------------------------
# 2. ADAPTIVE REJECTION THRESHOLD
#-----------------------------------------------------------

class AdaptiveApprovalManager(EnhancedApprovalWorkflowManager):
    """Approval manager with adaptive rejection thresholds based on strategy quality."""
    
    def __init__(self):
        """Initialize the adaptive approval manager."""
        super().__init__()
        self.quality_history = []
        self.rejection_threshold = 0.2  # Initial rejection threshold
        self.adaptation_rate = 0.05     # How quickly to adapt
        self.min_threshold = 0.05       # Minimum rejection threshold
        self.max_threshold = 0.4        # Maximum rejection threshold
        
    def create_request(self, **kwargs):
        """
        Create a request with enhanced tracking and adaptive thresholds.
        
        Args:
            **kwargs: Strategy data including performance expectations
            
        Returns:
            Request object or dict
        """
        strategy_quality = self._assess_strategy_quality(kwargs)
        
        # Store quality before creating request
        self.quality_history.append(strategy_quality)
        
        # Adapt rejection threshold based on recent quality trends
        self._adapt_rejection_threshold()
        
        # Create the request using parent implementation
        request = super().create_request(**kwargs)
        
        # Extract request_id based on return type
        request_id = request['request_id'] if isinstance(request, dict) else request.request_id
        
        # Store quality score in the pending request
        self.approval_delays['pending_requests'][request_id]['strategy_quality'] = strategy_quality
        
        print(f"Strategy quality: {strategy_quality:.2f}, Rejection threshold: {self.rejection_threshold:.2f}")
        
        return request
        
    def _assess_strategy_quality(self, strategy_data):
        """
        More sophisticated strategy quality assessment.
        
        Args:
            strategy_data: Strategy data including expected performance
            
        Returns:
            Quality score (0-1 scale)
        """
        # Base quality is random between 0.3 and 0.7
        quality = 0.3 + (random.random() * 0.4)
        
        # Extract expected performance if available
        if 'expected_performance' in strategy_data:
            perf = strategy_data['expected_performance']
            
            # Prefer strategies with positive return in bullish markets
            if 'bullish' in perf and perf['bullish'].get('return', 0) > 10:
                quality += 0.1
                
            # Prefer strategies that limit losses in bearish markets
            if 'bearish' in perf and perf['bearish'].get('return', -100) > -20:
                quality += 0.1
                
            # Penalize very high drawdowns
            max_dd = 0
            for regime_data in perf.values():
                if isinstance(regime_data, dict) and 'max_drawdown' in regime_data:
                    max_dd = max(max_dd, regime_data['max_drawdown'])
                    
            if max_dd > 30:
                quality -= 0.2
                
            # Bonus for positive performance in multiple regimes
            positive_regimes = sum(1 for regime_data in perf.values() 
                                if isinstance(regime_data, dict) 
                                and regime_data.get('return', 0) > 0)
            if positive_regimes >= 2:
                quality += 0.15
        
        # If parameters are available, check for excessive risk
        if 'params' in strategy_data:
            params = strategy_data['params']
            
            # Penalize very high position sizes
            if params.get('position_size', 0) > 0.8:
                quality -= 0.1
                
            # Penalize very tight stop losses
            if params.get('stop_loss', 100) < 2.0:
                quality -= 0.1
        
        # Normalize quality score to 0-1 range
        return max(0, min(1, quality))
        
    def _adapt_rejection_threshold(self):
        """Dynamically adjust rejection threshold based on recent strategy quality."""
        if len(self.quality_history) < 5:
            return  # Not enough history
            
        # Calculate average recent quality
        recent_quality = sum(self.quality_history[-5:]) / 5
        
        # If recent strategies have been high quality, lower rejection threshold
        if recent_quality > 0.7:
            self.rejection_threshold = max(
                self.min_threshold, 
                self.rejection_threshold - self.adaptation_rate
            )
            
        # If recent strategies have been low quality, raise rejection threshold
        elif recent_quality < 0.4:
            self.rejection_threshold = min(
                self.max_threshold,
                self.rejection_threshold + self.adaptation_rate
            )
            
        # Update the simulation parameters
        self.approval_delays['rejection_probability'] = self.rejection_threshold


# Function to get the adaptive approval workflow manager
def get_adaptive_approval_workflow_manager():
    """Get the adaptive approval workflow manager singleton."""
    return AdaptiveApprovalManager()


#-----------------------------------------------------------
# 3. HISTORICAL PERFORMANCE DATABASE
#-----------------------------------------------------------

class StrategyDatabase:
    """Database for storing and retrieving historical strategy performance."""
    
    def __init__(self, db_path="strategy_database.json"):
        """Initialize the strategy database."""
        self.db_path = db_path
        self.strategies = self._load_database()
        
    def _load_database(self):
        """Load the strategy database from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Error loading database from {self.db_path}, creating new one")
                return {"strategies": [], "metadata": {"last_updated": None}}
        else:
            return {"strategies": [], "metadata": {"last_updated": None}}
            
    def save_database(self):
        """Save the strategy database to disk."""
        self.strategies["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.db_path, 'w') as f:
            json.dump(self.strategies, f, indent=2)
            
    def add_strategy(self, strategy_data):
        """Add a strategy to the database."""
        # Make a copy to avoid modifying the original
        strategy_record = strategy_data.copy() if isinstance(strategy_data, dict) else {}
        
        # Add timestamp
        strategy_record["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add unique ID if not present
        if "id" not in strategy_record:
            strategy_record["id"] = str(uuid.uuid4())
            
        # Add to database
        self.strategies["strategies"].append(strategy_record)
        self.save_database()
        return strategy_record["id"]
        
    def get_strategy(self, strategy_id):
        """Retrieve a strategy by ID."""
        for strategy in self.strategies["strategies"]:
            if strategy.get("id") == strategy_id:
                return strategy
        return None
        
    def get_strategies_by_performance(self, min_accuracy=None, regime=None, limit=10):
        """Query strategies with filters."""
        results = self.strategies["strategies"]
        
        # Filter by minimum accuracy
        if min_accuracy is not None:
            results = [s for s in results if s.get("accuracy", 0) >= min_accuracy]
            
        # Filter by regime performance
        if regime is not None:
            results = [s for s in results if regime in s.get("expected_performance", {})]
            
        # Sort by accuracy (descending)
        results = sorted(results, key=lambda s: s.get("accuracy", 0), reverse=True)
        
        # Apply limit
        return results[:limit]
        
    def analyze_patterns(self):
        """Analyze patterns in successful strategies."""
        if len(self.strategies["strategies"]) < 10:
            return {"message": "Not enough data for pattern analysis (need at least 10 strategies)"}
            
        # Group strategies by accuracy quartiles
        all_strategies = sorted(self.strategies["strategies"], 
                                key=lambda s: s.get("accuracy", 0))
        quartile_size = max(1, len(all_strategies) // 4)
        
        low_performers = all_strategies[:quartile_size]
        high_performers = all_strategies[-quartile_size:]
        
        # Analyze parameter patterns
        param_patterns = {}
        
        # Compare average parameter values between high and low performers
        for param in ["trend_period", "entry_threshold", "exit_threshold", "stop_loss", "position_size"]:
            # Extract values, handling missing data
            high_values = []
            low_values = []
            
            for s in high_performers:
                if "params" in s and param in s["params"]:
                    high_values.append(s["params"][param])
                    
            for s in low_performers:
                if "params" in s and param in s["params"]:
                    low_values.append(s["params"][param])
            
            # Calculate statistics if we have data
            if high_values and low_values:
                high_avg = sum(high_values) / len(high_values)
                low_avg = sum(low_values) / len(low_values)
                
                param_patterns[param] = {
                    "high_performer_avg": high_avg,
                    "low_performer_avg": low_avg,
                    "difference": high_avg - low_avg,
                    "difference_pct": (high_avg - low_avg) / max(0.0001, low_avg) * 100
                }
            
        # Analyze regime performance patterns
        regime_patterns = {}
        for regime in ["bullish", "bearish", "sideways", "volatile"]:
            high_regime_success = 0
            low_regime_success = 0
            
            # Calculate success rate in each regime
            for s in high_performers:
                if "expected_performance" in s and regime in s["expected_performance"]:
                    perf = s["expected_performance"][regime]
                    if isinstance(perf, dict) and perf.get("return", 0) > 0:
                        high_regime_success += 1
                        
            for s in low_performers:
                if "expected_performance" in s and regime in s["expected_performance"]:
                    perf = s["expected_performance"][regime]
                    if isinstance(perf, dict) and perf.get("return", 0) > 0:
                        low_regime_success += 1
            
            # Calculate regime success percentages
            if high_performers and low_performers:
                high_pct = high_regime_success / len(high_performers) * 100
                low_pct = low_regime_success / len(low_performers) * 100
                
                regime_patterns[regime] = {
                    "high_performer_success": high_pct,
                    "low_performer_success": low_pct,
                    "difference": high_pct - low_pct
                }
            
        return {
            "param_patterns": param_patterns,
            "regime_patterns": regime_patterns,
            "high_performer_count": len(high_performers),
            "strategy_count": len(all_strategies)
        }
    
    def generate_insights_report(self):
        """Generate a human-readable insights report from the database."""
        analysis = self.analyze_patterns()
        
        if "message" in analysis:
            return analysis["message"]
            
        report = []
        report.append("=" * 70)
        report.append("STRATEGY DATABASE INSIGHTS REPORT")
        report.append("=" * 70)
        report.append(f"\nBased on analysis of {analysis['strategy_count']} strategies")
        
        # Parameter insights
        report.append("\nPARAMETER PATTERNS IN SUCCESSFUL STRATEGIES\n" + "-" * 45)
        
        for param, data in analysis["param_patterns"].items():
            direction = "HIGHER" if data["difference"] > 0 else "LOWER"
            significance = ""  # Assess significance
            if abs(data["difference_pct"]) > 50:
                significance = "(STRONG SIGNAL)"
            elif abs(data["difference_pct"]) > 25:
                significance = "(MODERATE SIGNAL)"
                
            report.append(f"{param}: {direction} {significance}")
            report.append(f"  Top performers: {data['high_performer_avg']:.2f}")
            report.append(f"  Bottom performers: {data['low_performer_avg']:.2f}")
            report.append(f"  Difference: {data['difference_pct']:+.1f}%\n")
            
        # Regime insights
        report.append("REGIME PERFORMANCE PATTERNS\n" + "-" * 25)
        
        for regime, data in analysis["regime_patterns"].items():
            report.append(f"{regime.capitalize()} regime:")
            report.append(f"  Top performers success rate: {data['high_performer_success']:.1f}%")
            report.append(f"  Bottom performers success rate: {data['low_performer_success']:.1f}%")
            report.append(f"  Difference: {data['difference']:+.1f}%\n")
            
        # Summary of recommendations
        report.append("STRATEGY RECOMMENDATIONS\n" + "-" * 25)
        
        # Extract key recommendations
        recommendations = []
        
        for param, data in analysis["param_patterns"].items():
            if abs(data["difference_pct"]) > 25:  # Only significant differences
                direction = "higher" if data["difference"] > 0 else "lower"
                recommendations.append(f"Use {direction} {param} values")
                
        # Add regime-specific recommendations
        strong_regimes = []
        weak_regimes = []
        
        for regime, data in analysis["regime_patterns"].items():
            if data["difference"] > 10:  # Strong advantage
                strong_regimes.append(regime)
            elif data["difference"] < -10:  # Strong disadvantage
                weak_regimes.append(regime)
                
        if strong_regimes:
            recommendations.append(f"Optimize for {', '.join(strong_regimes)} regimes")
            
        if weak_regimes:
            recommendations.append(f"Improve handling of {', '.join(weak_regimes)} regimes")
            
        # Add the recommendations to the report
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
            
        # Overall conclusion
        report.append("\n" + "=" * 70)
        report.append(f"Analysis completed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 70)
        
        return "\n".join(report)


#-----------------------------------------------------------
# 4. ADVANCED LEARNING ALGORITHMS
#-----------------------------------------------------------

class BayesianParameterLearner:
    """Applies Bayesian optimization techniques to parameter learning."""
    
    def __init__(self, real_params=None, learning_rate=0.2):
        """Initialize the Bayesian parameter learner."""
        self.real_params = real_params
        self.param_history = []
        self.performance_history = []
        self.learning_rate = learning_rate
        
        # Prior distributions for each parameter
        self.priors = {
            'trend': {'mean': 0.001, 'std': 0.005, 'confidence': 0.5},
            'volatility': {'mean': 0.02, 'std': 0.01, 'confidence': 0.5},
            'mean_reversion': {'mean': 0.3, 'std': 0.3, 'confidence': 0.5}
        }
        
        # Parameter constraints
        self.constraints = {
            'trend': {'min': -0.01, 'max': 0.01},
            'volatility': {'min': 0.001, 'max': 0.1},
            'mean_reversion': {'min': 0.01, 'max': 0.99}
        }
        
        # Adaptive learning rates by parameter
        self.adaptive_rates = {
            'trend': 0.2,
            'volatility': 0.2,
            'mean_reversion': 0.2
        }
        
    def update(self, params, actual_performance, expected_performance):
        """Update parameter beliefs based on new performance data using Bayesian methods."""
        # Store history
        self.param_history.append(params)
        self.performance_history.append({
            'actual': actual_performance,
            'expected': expected_performance
        })
        
        # Calculate prediction errors
        errors = self._calculate_error(actual_performance, expected_performance)
        
        # Update parameter beliefs using a Bayesian approach
        updated_params = {}
        
        for regime, regime_params in params.items():
            updated_params[regime] = {}
            regime_error = errors.get(regime, {'overall': 1.0})
            
            # Update each parameter based on performance
            for param_name, param_value in regime_params.items():
                if param_name not in self.priors:
                    updated_params[regime][param_name] = param_value
                    continue
                
                # Get prior distribution for this parameter
                prior = self.priors[param_name]
                constraints = self.constraints.get(param_name, {'min': -float('inf'), 'max': float('inf')})
                
                # Calculate confidence based on prediction accuracy
                # Lower error means we trust our current value more
                prediction_confidence = max(0.1, 1.0 - regime_error.get('overall', 0.5))
                
                # Prior confidence increases with each update (Bayesian learning)
                prior['confidence'] = min(0.9, prior['confidence'] + 0.05)
                
                # Calculate adaptive learning rate
                # If error is high and consistent across multiple updates, increase learning rate
                consistency_factor = 1.0
                if len(self.performance_history) > 2:
                    # Check if errors are consistently high for this parameter
                    prev_errors = [self._calculate_error(
                                    self.performance_history[i]['actual'],
                                    self.performance_history[i]['expected']
                                  ).get(regime, {'overall': 0.5})['overall']
                                  for i in range(-3, -1)]
                    
                    if all(e > 0.3 for e in prev_errors) and regime_error.get('overall', 0) > 0.3:
                        # Errors consistently high - increase learning rate
                        consistency_factor = 1.5
                    elif all(e < 0.2 for e in prev_errors) and regime_error.get('overall', 0) < 0.2:
                        # Errors consistently low - decrease learning rate
                        consistency_factor = 0.8
                
                # Adaptive rate for this parameter
                adaptive_rate = self.adaptive_rates[param_name] * consistency_factor
                
                # Calculate parameter update using Bayesian fusion
                # Balance prior beliefs with new evidence
                prior_weight = prior['confidence']
                evidence_weight = prediction_confidence
                
                # Normalize weights
                total_weight = prior_weight + evidence_weight
                prior_weight /= total_weight
                evidence_weight /= total_weight
                
                # Direction of update depends on specific error metrics
                error_metrics = regime_error
                update_direction = self._determine_update_direction(param_name, regime, error_metrics)
                
                # Calculate the update magnitude
                update_magnitude = adaptive_rate * evidence_weight * update_direction
                
                # Apply the update
                updated_value = param_value * (1 + update_magnitude)
                
                # Apply constraints
                updated_value = max(constraints['min'], min(constraints['max'], updated_value))
                
                # Update the prior for next iteration
                self.priors[param_name]['mean'] = (prior['mean'] * prior_weight + 
                                                  updated_value * evidence_weight)
                
                # Store the result
                updated_params[regime][param_name] = updated_value
                
                # Update adaptive learning rate based on success of this update
                if len(self.param_history) > 1:
                    previous_value = self.param_history[-1][regime][param_name]
                    change_pct = abs((updated_value - previous_value) / previous_value) if previous_value else 0
                    
                    # If change is very small, increase learning rate slightly
                    if change_pct < 0.01:
                        self.adaptive_rates[param_name] = min(0.5, self.adaptive_rates[param_name] * 1.05)
                    # If change is very large, decrease learning rate
                    elif change_pct > 0.2:
                        self.adaptive_rates[param_name] = max(0.05, self.adaptive_rates[param_name] * 0.95)
        
        return updated_params
    
    def _determine_update_direction(self, param_name, regime, error_metrics):
        """Determine direction of parameter update based on error metrics."""
        # Default direction (no strong signal)
        direction = 0
        
        # Different parameters need different update logic
        if param_name == 'trend':
            # If return error is high, adjust trend
            return_error = error_metrics.get('return', 0)
            expected_return = error_metrics.get('expected_return', 0)
            actual_return = error_metrics.get('actual_return', 0)
            
            if return_error > 0.2:  # Significant error
                # If actual < expected, trend might be too high
                if actual_return < expected_return:
                    direction = -1
                else:  # actual > expected, trend might be too low
                    direction = 1
                    
                # Special cases for regimes
                if regime == 'bearish' and direction > 0:
                    # For bearish regimes, negative trend is expected
                    direction = -2  # Stronger update for bearish trend
        
        elif param_name == 'volatility':
            # If drawdown error is high, adjust volatility
            dd_error = error_metrics.get('drawdown', 0)
            expected_dd = error_metrics.get('expected_drawdown', 0)
            actual_dd = error_metrics.get('actual_drawdown', 0)
            
            if dd_error > 0.2:  # Significant error
                # If actual drawdown > expected, volatility might be too low
                if actual_dd > expected_dd:
                    direction = 1
                else:  # actual < expected, volatility might be too high
                    direction = -1
                    
                # Special case for volatile regime
                if regime == 'volatile':
                    direction *= 1.5  # Stronger updates for volatile regime
        
        elif param_name == 'mean_reversion':
            # Mean reversion affects overall behavior
            overall_error = error_metrics.get('overall', 0)
            
            if overall_error > 0.3:  # Significant overall error
                if regime == 'sideways':
                    # Sideways markets usually have higher mean reversion
                    direction = 1
                elif regime in ['bullish', 'bearish']:
                    # Trending markets usually have lower mean reversion
                    direction = -1
        
        return direction
    
    def _calculate_error(self, actual, expected):
        """Calculate detailed error metrics between actual and expected performance."""
        errors = {}
        
        for regime in actual:
            if regime not in expected:
                errors[regime] = {
                    'overall': 1.0,
                    'return': 1.0,
                    'drawdown': 1.0,
                    'sharpe': 1.0,
                    'expected_return': 0,
                    'actual_return': 0,
                    'expected_drawdown': 0,
                    'actual_drawdown': 0
                }
                continue
                
            # Extract metrics
            actual_return = actual[regime].get('return', 0)
            expected_return = expected[regime].get('return', 0)
            actual_dd = actual[regime].get('max_drawdown', 0)
            expected_dd = expected[regime].get('max_drawdown', 0)
            actual_sharpe = actual[regime].get('sharpe', 0)
            expected_sharpe = expected[regime].get('sharpe', 0)
            
            # Calculate normalized errors
            return_error = abs(actual_return - expected_return) / max(1.0, abs(expected_return))
            dd_error = abs(actual_dd - expected_dd) / max(1.0, abs(expected_dd))
            sharpe_error = abs(actual_sharpe - expected_sharpe) / max(1.0, abs(expected_sharpe))
            
            # Cap errors at 1.0
            return_error = min(1.0, return_error)
            dd_error = min(1.0, dd_error)
            sharpe_error = min(1.0, sharpe_error)
            
            # Combined error with weights
            overall_error = return_error * 0.5 + dd_error * 0.3 + sharpe_error * 0.2
            
            # Store detailed error metrics
            errors[regime] = {
                'overall': overall_error,
                'return': return_error,
                'drawdown': dd_error,
                'sharpe': sharpe_error,
                'expected_return': expected_return,
                'actual_return': actual_return,
                'expected_drawdown': expected_dd,
                'actual_drawdown': actual_dd
            }
            
        return errors
    
    def get_parameter_beliefs(self):
        """Get current parameter beliefs for visualization and analysis."""
        return self.priors
    
    def get_adaptive_rates(self):
        """Get current adaptive learning rates."""
        return self.adaptive_rates
    
    def generate_update_report(self, params, updated_params):
        """Generate a detailed report about parameter updates."""
        report = []
        report.append("=" * 60)
        report.append("BAYESIAN PARAMETER UPDATE REPORT")
        report.append("=" * 60)
        
        for regime in params:
            if regime not in updated_params:
                continue
                
            report.append(f"\n{regime.upper()} REGIME")
            report.append("-" * 20)
            
            for param_name in params[regime]:
                if param_name not in updated_params[regime]:
                    continue
                    
                old_value = params[regime][param_name]
                new_value = updated_params[regime][param_name]
                change = new_value - old_value
                change_pct = change / old_value * 100 if old_value else float('inf')
                
                # Show adaptive learning rate
                adaptive_rate = self.adaptive_rates.get(param_name, "N/A")
                
                report.append(f"{param_name}:")
                report.append(f"  Old value: {old_value:.6f}")
                report.append(f"  New value: {new_value:.6f}")
                report.append(f"  Change: {change:.6f} ({change_pct:+.2f}%)")
                report.append(f"  Learning rate: {adaptive_rate}")
                
                # Show prior beliefs
                if param_name in self.priors:
                    prior = self.priors[param_name]
                    report.append(f"  Prior belief: mean={prior['mean']:.6f}, confidence={prior['confidence']:.2f}")
                
                # If we have real parameters for comparison, show distance
                if self.real_params and regime in self.real_params and param_name in self.real_params[regime]:
                    real_value = self.real_params[regime][param_name]
                    old_distance = abs(old_value - real_value)
                    new_distance = abs(new_value - real_value)
                    improvement = old_distance - new_distance
                    
                    # Describe if we're getting closer or further from real value
                    if improvement > 0:
                        direction = "CLOSER TO REAL VALUE"
                    elif improvement < 0:
                        direction = "FURTHER FROM REAL VALUE"
                    else:
                        direction = "NO CHANGE"
                        
                    report.append(f"  Real value: {real_value:.6f}")
                    report.append(f"  Distance change: {improvement:.6f} ({direction})")
                
                report.append("")
                
        return "\n".join(report)


#-----------------------------------------------------------
# 5. PRODUCTION TESTING FRAMEWORK
#-----------------------------------------------------------

# Import datetime if not already imported
import datetime

class ShadowTradingFramework:
    """Framework for shadow trading approved strategies without real money."""
    
    def __init__(self, risk_limit=0.01, tracking_window=30):
        """Initialize the shadow trading framework."""
        self.shadow_strategies = {}  # Track all shadow strategies
        self.risk_limit = risk_limit  # Maximum portfolio % for shadow trading
        self.tracking_window = tracking_window  # Days to track before evaluation
        self.trading_log = []  # Overall trading log
        self.valuation_history = []  # Track portfolio value over time
        self.last_valuation = {"timestamp": None, "value": 1000000}  # Initial portfolio value
        
    def deploy_strategy(self, strategy_id, strategy_params, portfolio_fraction=0.001, strategy_name=None):
        """Deploy a strategy for shadow trading with risk limits."""
        # Enforce risk limits
        actual_fraction = min(portfolio_fraction, self.risk_limit)
        
        # Calculate initial allocation
        initial_allocation = self.last_valuation["value"] * actual_fraction
        
        # Create deployment record
        deployment = {
            'strategy_id': strategy_id,
            'name': strategy_name or f"Strategy {strategy_id}",
            'params': strategy_params,
            'deployment_date': datetime.datetime.now().isoformat(),
            'portfolio_fraction': actual_fraction,
            'initial_allocation': initial_allocation,
            'current_allocation': initial_allocation,
            'trades': [],
            'positions': {},  # Current open positions
            'status': 'active',
            'metrics': {
                'win_rate': None,
                'profit_factor': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'returns': None
            }
        }
        
        self.shadow_strategies[strategy_id] = deployment
        
        print(f"Shadow deployed: {strategy_id} with {actual_fraction:.4%} allocation (${initial_allocation:.2f})")
        
        return {
            'strategy_id': strategy_id,
            'portfolio_fraction': actual_fraction,
            'initial_allocation': initial_allocation,
            'status': 'active',
            'message': f"Strategy {strategy_id} deployed for shadow trading"
        }
        
    def record_trade(self, strategy_id, trade_data):
        """Record a trade executed by the shadow strategy."""
        if strategy_id not in self.shadow_strategies:
            return {'error': f"Strategy {strategy_id} not found"}
            
        # Ensure we have required trade data
        required_fields = ['symbol', 'direction', 'quantity', 'price']
        missing_fields = [field for field in required_fields if field not in trade_data]
        
        if missing_fields:
            return {'error': f"Missing required trade fields: {', '.join(missing_fields)}"}
            
        # Add timestamp if not provided
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.datetime.now().isoformat()
            
        # Calculate trade value
        price = trade_data['price']
        quantity = trade_data['quantity']
        trade_value = price * quantity
        
        # Add trade value to record
        trade_data['value'] = trade_value
        
        # Update strategy's positions
        symbol = trade_data['symbol']
        direction = trade_data['direction']  # 'buy' or 'sell'
        
        # Get current positions
        positions = self.shadow_strategies[strategy_id]['positions']
        
        # Update position based on trade direction
        if direction == 'buy':
            if symbol not in positions:
                positions[symbol] = {'quantity': 0, 'value': 0, 'avg_price': 0}
                
            # Calculate new average price
            current_qty = positions[symbol]['quantity']
            current_value = positions[symbol]['value']
            
            new_qty = current_qty + quantity
            new_value = current_value + trade_value
            
            if new_qty > 0:
                new_avg_price = new_value / new_qty
            else:
                new_avg_price = 0
                
            # Update position
            positions[symbol] = {
                'quantity': new_qty,
                'value': new_value,
                'avg_price': new_avg_price
            }
        elif direction == 'sell':
            if symbol not in positions:
                # Selling something we don't have - short position
                positions[symbol] = {'quantity': -quantity, 'value': -trade_value, 'avg_price': price}
            else:
                # Calculate profit/loss on this trade
                current_qty = positions[symbol]['quantity']
                current_avg_price = positions[symbol]['avg_price']
                
                # Calculate P&L if closing a position
                if current_qty > 0 and quantity > 0:
                    # Closing long position (full or partial)
                    close_qty = min(current_qty, quantity)
                    pnl = (price - current_avg_price) * close_qty
                    trade_data['pnl'] = pnl
                    
                # Update position
                new_qty = current_qty - quantity
                if new_qty == 0:
                    # Position closed
                    del positions[symbol]
                else:
                    # Position reduced or flipped
                    positions[symbol]['quantity'] = new_qty
                    # Recalculate value based on direction
                    if new_qty > 0:
                        # Still long, value doesn't change for remaining shares
                        positions[symbol]['value'] = new_qty * current_avg_price
                    else:
                        # Flipped to short
                        positions[symbol]['value'] = new_qty * price
                        positions[symbol]['avg_price'] = price
        
        # Add to strategy's trade log
        self.shadow_strategies[strategy_id]['trades'].append(trade_data)
        
        # Add to global trading log
        self.trading_log.append({
            'strategy_id': strategy_id,
            'trade': trade_data
        })
        
        # Update strategy metrics
        self._update_metrics(strategy_id)
        
        return {
            'strategy_id': strategy_id,
            'trade_id': len(self.shadow_strategies[strategy_id]['trades']),
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'timestamp': trade_data['timestamp'],
            'message': f"Trade recorded for strategy {strategy_id}"
        }
        
    def update_portfolio_value(self, date=None, value=None):
        """Update the total portfolio value (typically called daily)."""
        # Use current date/time if not provided
        if date is None:
            date = datetime.datetime.now().isoformat()
            
        # Use previous value if not provided
        if value is None:
            value = self.last_valuation["value"]
            
        # Record the valuation
        valuation = {"timestamp": date, "value": value}
        self.valuation_history.append(valuation)
        self.last_valuation = valuation
        
        # Update each strategy's allocation based on portfolio fraction
        for strategy_id, strategy in self.shadow_strategies.items():
            if strategy['status'] == 'active':
                new_allocation = value * strategy['portfolio_fraction']
                strategy['current_allocation'] = new_allocation
        
        return {
            'timestamp': date,
            'portfolio_value': value,
            'active_strategies': len([s for s in self.shadow_strategies.values() if s['status'] == 'active']),
            'message': f"Portfolio value updated to ${value:,.2f}"
        }
        
    def _update_metrics(self, strategy_id):
        """Update performance metrics for a strategy."""
        strategy = self.shadow_strategies[strategy_id]
        trades = strategy['trades']
        
        if len(trades) < 3:
            return  # Not enough trades for meaningful metrics
            
        # Extract completed trades (those with PnL)
        completed_trades = [t for t in trades if 'pnl' in t]
        
        if not completed_trades:
            return  # No completed trades yet
            
        # Calculate metrics
        wins = [t for t in completed_trades if t['pnl'] > 0]
        losses = [t for t in completed_trades if t['pnl'] <= 0]
        
        # Win rate
        win_rate = len(wins) / len(completed_trades) if completed_trades else 0
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = sum(abs(t['pnl']) for t in losses) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Returns
        initial_allocation = strategy['initial_allocation']
        current_value = self._calculate_strategy_value(strategy_id)
        returns = (current_value - initial_allocation) / initial_allocation * 100
        
        # Calculate drawdown series and max drawdown
        equity_curve = self._calculate_equity_curve(strategy_id)
        max_dd = self._calculate_max_drawdown(equity_curve)
        
        # Update strategy metrics
        strategy['metrics'] = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'returns': returns,
            'max_drawdown': max_dd,
            'trade_count': len(completed_trades)
        }
        
    def _calculate_strategy_value(self, strategy_id):
        """Calculate current value of a strategy including open positions."""
        strategy = self.shadow_strategies[strategy_id]
        
        # Start with current allocated cash
        value = strategy['current_allocation']
        
        # Add value of open positions
        for symbol, position in strategy['positions'].items():
            value += position['value']
            
        # Add in realized P&L from completed trades
        for trade in strategy['trades']:
            if 'pnl' in trade:
                value += trade['pnl']
                
        return value
        
    def _calculate_equity_curve(self, strategy_id):
        """Calculate equity curve for a strategy from trades."""
        strategy = self.shadow_strategies[strategy_id]
        initial_value = strategy['initial_allocation']
        
        # Create time series of equity
        equity_curve = [{'timestamp': strategy['deployment_date'], 'equity': initial_value}]
        running_equity = initial_value
        
        # Sort trades by timestamp
        sorted_trades = sorted(strategy['trades'], key=lambda t: t['timestamp'])
        
        # Add equity points for each trade with PnL
        for trade in sorted_trades:
            if 'pnl' in trade:
                running_equity += trade['pnl']
                equity_curve.append({
                    'timestamp': trade['timestamp'],
                    'equity': running_equity
                })
                
        return equity_curve
        
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
            
        max_dd = 0.0
        peak = equity_curve[0]['equity']
        
        for point in equity_curve[1:]:
            if point['equity'] > peak:
                peak = point['equity']
            else:
                dd = (peak - point['equity']) / peak
                max_dd = max(max_dd, dd)
                
        return max_dd * 100  # Return as percentage
        
    def evaluate_readiness(self, strategy_id):
        """Evaluate if a shadow strategy is ready for live trading."""
        if strategy_id not in self.shadow_strategies:
            return {'error': f"Strategy {strategy_id} not found"}
            
        strategy = self.shadow_strategies[strategy_id]
        metrics = strategy['metrics']
        
        # Check if we have metrics yet
        if not metrics['trade_count']:
            return {
                'strategy_id': strategy_id,
                'name': strategy['name'],
                'ready': False,
                'reason': "No completed trades yet",
                'metrics': metrics
            }
            
        # Check minimum trade count
        if metrics['trade_count'] < 20:
            return {
                'strategy_id': strategy_id,
                'name': strategy['name'],
                'ready': False,
                'reason': f"Insufficient trades ({metrics['trade_count']}/20 required)",
                'metrics': metrics
            }
            
        # Check performance criteria
        if metrics['win_rate'] < 0.5:
            return {
                'strategy_id': strategy_id,
                'name': strategy['name'],
                'ready': False,
                'reason': f"Win rate too low ({metrics['win_rate']:.2%} < 50%)",
                'metrics': metrics
            }
            
        if metrics['profit_factor'] < 1.5:
            return {
                'strategy_id': strategy_id,
                'name': strategy['name'],
                'ready': False,
                'reason': f"Profit factor too low ({metrics['profit_factor']:.2f} < 1.5)",
                'metrics': metrics
            }
            
        if metrics['max_drawdown'] > 15:
            return {
                'strategy_id': strategy_id,
                'name': strategy['name'],
                'ready': False,
                'reason': f"Max drawdown too high ({metrics['max_drawdown']:.2f}% > 15%)",
                'metrics': metrics
            }
            
        # Strategy meets criteria
        return {
            'strategy_id': strategy_id,
            'name': strategy['name'],
            'ready': True,
            'message': "Strategy meets criteria for live trading consideration",
            'metrics': metrics
        }
        
    def generate_shadow_report(self, strategy_id=None):
        """Generate a comprehensive report on shadow trading performance."""
        if strategy_id and strategy_id not in self.shadow_strategies:
            return f"Strategy {strategy_id} not found"
            
        report = []
        report.append("=" * 70)
        report.append("SHADOW TRADING PERFORMANCE REPORT")
        report.append("=" * 70)
        
        # Overall stats
        active_count = len([s for s in self.shadow_strategies.values() if s['status'] == 'active'])
        total_count = len(self.shadow_strategies)
        trade_count = len(self.trading_log)
        
        report.append(f"\nTotal Strategies: {total_count}")
        report.append(f"Active Strategies: {active_count}")
        report.append(f"Total Trades: {trade_count}")
        
        # Portfolio value trend
        if len(self.valuation_history) > 1:
            initial = self.valuation_history[0]['value']
            current = self.valuation_history[-1]['value']
            change = (current - initial) / initial * 100
            
            report.append(f"\nPortfolio Value: ${current:,.2f} ({change:+.2f}%)")
            
        # If specific strategy requested
        if strategy_id:
            strategy = self.shadow_strategies[strategy_id]
            report.append(f"\n{'-' * 30}")
            report.append(f"STRATEGY: {strategy['name']} ({strategy_id})")
            report.append(f"{'-' * 30}")
            
            # Strategy details
            report.append(f"Deployed: {strategy['deployment_date']}")
            report.append(f"Status: {strategy['status']}")
            report.append(f"Allocation: ${strategy['current_allocation']:,.2f} ({strategy['portfolio_fraction']:.2%})")
            
            # Performance metrics
            metrics = strategy['metrics']
            if metrics['trade_count']:
                report.append(f"\nPERFORMANCE METRICS:")
                report.append(f"  Win Rate: {metrics['win_rate']:.2%}")
                report.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
                report.append(f"  Returns: {metrics['returns']:+.2f}%")
                report.append(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
                report.append(f"  Completed Trades: {metrics['trade_count']}")
                
                # Readiness assessment
                readiness = self.evaluate_readiness(strategy_id)
                if readiness['ready']:
                    report.append("\nREADINESS: READY FOR LIVE TRADING ✓")
                else:
                    report.append(f"\nREADINESS: NOT READY - {readiness['reason']}")
            else:
                report.append("\nInsufficient data for performance metrics")
                
            # Open positions
            positions = strategy['positions']
            if positions:
                report.append("\nOPEN POSITIONS:")
                for symbol, pos in positions.items():
                    direction = "LONG" if pos['quantity'] > 0 else "SHORT"
                    report.append(f"  {symbol}: {direction} {abs(pos['quantity'])} @ {pos['avg_price']:.2f}")
            else:
                report.append("\nNo open positions")
                
            # Recent trades
            recent_trades = strategy['trades'][-5:] if len(strategy['trades']) > 5 else strategy['trades']
            if recent_trades:
                report.append("\nRECENT TRADES:")
                for trade in recent_trades:
                    report.append(f"  {trade['timestamp']}: {trade['direction'].upper()} {trade['quantity']} {trade['symbol']} @ {trade['price']:.2f}")
                    if 'pnl' in trade:
                        report.append(f"    P&L: ${trade['pnl']:+,.2f}")
        else:
            # Summary of all strategies
            report.append("\nSTRATEGY SUMMARY:")
            
            for sid, strategy in self.shadow_strategies.items():
                metrics = strategy['metrics']
                if metrics['trade_count']:
                    status = "✓" if self.evaluate_readiness(sid)['ready'] else "✗"
                    report.append(f"  {strategy['name']}: {metrics['returns']:+.2f}% ({metrics['trade_count']} trades) {status}")
                else:
                    report.append(f"  {strategy['name']}: No completed trades yet")
            
        report.append("\n" + "=" * 70)
        return "\n".join(report)


#-----------------------------------------------------------
# 6. INTEGRATION: PRODUCTION-READY WORKFLOW
#-----------------------------------------------------------

class ProductionReadyWorkflow:
    """Comprehensive integration of all production-ready trading workflow components."""
    
    def __init__(self, synthetic_params=None, trading_strategy_class=None):
        """Initialize the production-ready workflow with all components."""
        # Core components
        self.benchmarks = BenchmarkComparison(None)  # Pass None since we don't have a feedback loop here
        self.adaptive_approval = AdaptiveApprovalManager()
        self.history_db = StrategyDatabase()
        self.bayesian_optimizer = BayesianParameterLearner()
        self.shadow_trading = ShadowTradingFramework()
        
        # Initialize optional trading strategy class and synthetic parameters
        self._strategy_class = trading_strategy_class
        self._synthetic_params = synthetic_params or {}
        
        # Workflow state
        self.active_strategies = {}
        self.pending_strategies = {}
        self.rejected_strategies = {}
        self.shadow_strategies = {}
        
        # Event tracking
        self.events = []
        
        # Logging and reporting
        self.logs = []
        self.approval_log = []
        self.optimization_history = {}
        
        print("Production-ready trading workflow initialized successfully")
    
    def log(self, message, level="INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.datetime.now().isoformat()
        entry = {"timestamp": timestamp, "level": level, "message": message}
        self.logs.append(entry)
        if level == "ERROR":
            print(f"[ERROR] {message}")
        elif level == "WARNING":
            print(f"[WARNING] {message}")
        else:
            print(f"[{level}] {message}")
        return entry
    
    def start_optimization_cycle(self, strategy_id, market_data, initial_params=None, regimes=None, max_iterations=50):
        """Start a new optimization cycle for a strategy."""
        self.log(f"Starting optimization cycle for strategy {strategy_id} (max {max_iterations} iterations)")
        
        # Initialize data structures for this optimization
        if strategy_id not in self.optimization_history:
            self.optimization_history[strategy_id] = []
        
        # Default regimes if not provided
        if not regimes:
            regimes = ["bullish", "bearish", "sideways", "volatile"]
            
        # Initialize Bayesian parameter learning for this strategy
        # No explicit setup needed - will be initialized with first update
        
        # Create the optimization run object
        optimization_run = {
            "strategy_id": strategy_id,
            "started_at": datetime.datetime.now().isoformat(),
            "regimes": regimes,
            "max_iterations": max_iterations,
            "iterations": [],
            "status": "running",
            "best_params": None,
            "converged": False,
            "accuracy": 0.0
        }
        
        # Run the optimization iterations
        current_params = initial_params or {}
        best_accuracy = 0.0
        convergence_count = 0
        accuracy_history = []
        
        for iteration in range(1, max_iterations + 1):
            # Use Bayesian optimization to suggest next parameters
            if iteration > 1 and self.bayesian_optimizer.is_ready(strategy_id):
                current_params = self.bayesian_optimizer.suggest_parameters(strategy_id, current_params)
            
            # Evaluate parameters across all regimes
            results = self._evaluate_strategy(strategy_id, current_params, market_data, regimes)
            
            # Record iteration results
            iteration_data = {
                "iteration": iteration,
                "params": current_params.copy(),
                "results": results,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Update optimization run record
            optimization_run["iterations"].append(iteration_data)
            
            # Extract accuracy from results
            accuracy = results.get("overall_accuracy", 0.0)
            accuracy_history.append(accuracy)
            
            # Check if this is the best result so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimization_run["best_params"] = current_params.copy()
                convergence_count = 0  # Reset convergence counter when we find better params
                self.log(f"New best parameters found: accuracy={accuracy:.2%}")
            else:
                convergence_count += 1
            
            # Update the Bayesian optimizer with results
            self.bayesian_optimizer.update(strategy_id, current_params, results)
            
            # Check for convergence
            if convergence_count >= 5 and accuracy > 0.75:  # Converged if no improvement for 5 iterations and accuracy > 75%
                self.log(f"Optimization converged after {iteration} iterations with accuracy {best_accuracy:.2%}")
                optimization_run["converged"] = True
                break
                
            # Progress update every 5 iterations
            if iteration % 5 == 0:
                self.log(f"Optimization progress: {iteration}/{max_iterations} iterations, best accuracy: {best_accuracy:.2%}")
        
        # Complete the optimization run
        optimization_run["completed_at"] = datetime.datetime.now().isoformat()
        optimization_run["status"] = "completed"
        optimization_run["accuracy"] = best_accuracy
        
        # Store in history
        self.optimization_history[strategy_id].append(optimization_run)
        
        # If we have a strong result, submit for approval
        if best_accuracy >= 0.7:  # At least 70% accuracy to submit
            self.submit_for_approval(strategy_id, optimization_run["best_params"], best_accuracy)
        else:
            self.log(f"Optimization result below approval threshold. Accuracy: {best_accuracy:.2%}", "WARNING")
        
        return optimization_run
    
    def _evaluate_strategy(self, strategy_id, params, market_data, regimes):
        """Evaluate strategy with given parameters across multiple regimes."""
        # Check if we have a strategy class
        if not self._strategy_class:
            # Return simulated evaluation with random performance
            import random
            
            # Generate more realistic simulation results based on params consistency
            # More consistent params = better performance
            consistency = sum(1 for p in params.values() if isinstance(p, (int, float)) and 0.2 <= p <= 0.8) / max(1, len(params))
            base_accuracy = 0.5 + (consistency * 0.3)  # 50-80% base accuracy based on param consistency
            
            regime_results = {}
            overall_metrics = {}
            
            for regime in regimes:
                # Simulate performance for this regime
                regime_accuracy = max(0.0, min(1.0, base_accuracy + random.uniform(-0.1, 0.1)))
                
                # Regime-specific performance metrics
                returns = (regime_accuracy - 0.5) * 20  # -10% to +10% returns
                sharpe = (regime_accuracy - 0.5) * 4  # -2 to +2 Sharpe ratio
                drawdown = (1.0 - regime_accuracy) * 15  # 0% to 15% drawdown
                
                regime_results[regime] = {
                    "accuracy": regime_accuracy,
                    "return": returns,
                    "sharpe": sharpe,
                    "max_drawdown": drawdown,
                    "win_rate": 0.4 + (regime_accuracy * 0.3)  # 40-70% win rate
                }
                
            # Overall accuracy is average of regime accuracies
            overall_accuracy = sum(r["accuracy"] for r in regime_results.values()) / len(regime_results)
            
            # Get benchmark comparison
            benchmark_comparison = self.benchmarks.compare_strategy_performance(regime_results)
            
            return {
                "strategy_id": strategy_id,
                "params": params.copy(),
                "overall_accuracy": overall_accuracy,
                "regime_results": regime_results,
                "benchmark_comparison": benchmark_comparison
            }
        else:
            # Real strategy evaluation code would go here
            # For demonstration, we'll just return simulated results
            return self._evaluate_strategy(strategy_id, params, market_data, regimes)
    
    def submit_for_approval(self, strategy_id, params, expected_accuracy, test_id=None):
        """Submit optimized strategy for approval."""
        test_id = test_id or f"test_{int(datetime.datetime.now().timestamp())}"
        
        # Create a pending strategy record
        pending_strategy = {
            "strategy_id": strategy_id,
            "test_id": test_id,
            "params": params.copy(),
            "expected_accuracy": expected_accuracy,
            "submitted_at": datetime.datetime.now().isoformat(),
            "status": "pending",
            "approval_data": None,
            "verification_data": None
        }
        
        # Store in pending strategies
        self.pending_strategies[strategy_id] = pending_strategy
        
        # Log the approval request
        approval_request = {
            "strategy_id": strategy_id,
            "test_id": test_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "submitted",
            "params": params.copy(),
            "expected_accuracy": expected_accuracy
        }
        self.approval_log.append(approval_request)
        
        # Check if adaptive approval threshold allows auto-approval
        approval_threshold = self.adaptive_approval.get_current_threshold(strategy_id)
        
        if expected_accuracy >= approval_threshold["auto_approve_threshold"]:
            # Auto-approve high-accuracy strategies
            self.log(f"Strategy {strategy_id} auto-approved (accuracy: {expected_accuracy:.2%} >= threshold: {approval_threshold['auto_approve_threshold']:.2%})")
            self.approve_strategy(strategy_id, "auto_approval_system", "Auto-approved based on high expected accuracy")
        else:
            self.log(f"Strategy {strategy_id} submitted for approval (accuracy: {expected_accuracy:.2%})")
        
        return approval_request
    
    def approve_strategy(self, strategy_id, approver="system", comments=None):
        """Approve a pending strategy."""
        if strategy_id not in self.pending_strategies:
            return {"error": f"Strategy {strategy_id} not in pending state"}
        
        # Get the pending strategy data
        strategy = self.pending_strategies[strategy_id]
        
        # Update approval data
        strategy["approval_data"] = {
            "approved_at": datetime.datetime.now().isoformat(),
            "approver": approver,
            "comments": comments
        }
        
        # Update status
        strategy["status"] = "approved"
        
        # Move to active strategies
        self.active_strategies[strategy_id] = strategy
        del self.pending_strategies[strategy_id]
        
        # Log the approval
        approval_event = {
            "strategy_id": strategy_id,
            "test_id": strategy["test_id"],
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "approved",
            "approver": approver,
            "comments": comments
        }
        self.approval_log.append(approval_event)
        
        # Update the adaptive approval threshold based on this approval
        self.adaptive_approval.record_approval(strategy_id, strategy["expected_accuracy"])
        
        # Store in historical database
        self.history_db.store_strategy(strategy_id, {
            "params": strategy["params"],
            "expected_accuracy": strategy["expected_accuracy"],
            "approved_at": strategy["approval_data"]["approved_at"],
            "phase": "approved"
        })
        
        # Deploy to shadow trading
        self.deploy_to_shadow_trading(strategy_id, strategy["params"])
        
        self.log(f"Strategy {strategy_id} approved by {approver}")
        
        return approval_event
    
    def reject_strategy(self, strategy_id, reviewer="system", reason=None):
        """Reject a pending strategy."""
        if strategy_id not in self.pending_strategies:
            return {"error": f"Strategy {strategy_id} not in pending state"}
        
        # Get the pending strategy data
        strategy = self.pending_strategies[strategy_id]
        
        # Update rejection data
        strategy["rejection_data"] = {
            "rejected_at": datetime.datetime.now().isoformat(),
            "reviewer": reviewer,
            "reason": reason
        }
        
        # Update status
        strategy["status"] = "rejected"
        
        # Move to rejected strategies
        self.rejected_strategies[strategy_id] = strategy
        del self.pending_strategies[strategy_id]
        
        # Log the rejection
        rejection_event = {
            "strategy_id": strategy_id,
            "test_id": strategy["test_id"],
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "rejected",
            "reviewer": reviewer,
            "reason": reason
        }
        self.approval_log.append(rejection_event)
        
        # Update the adaptive approval threshold based on this rejection
        self.adaptive_approval.record_rejection(strategy_id, strategy["expected_accuracy"])
        
        # Store in historical database
        self.history_db.store_strategy(strategy_id, {
            "params": strategy["params"],
            "expected_accuracy": strategy["expected_accuracy"],
            "rejected_at": strategy["rejection_data"]["rejected_at"],
            "phase": "rejected",
            "reason": reason
        })
        
        self.log(f"Strategy {strategy_id} rejected by {reviewer}: {reason}")
        
        return rejection_event
    
    def deploy_to_shadow_trading(self, strategy_id, params, portfolio_fraction=0.001):
        """Deploy an approved strategy to shadow trading."""
        # Check if strategy is approved
        if strategy_id not in self.active_strategies:
            return {"error": f"Strategy {strategy_id} not in active approved state"}
            
        # Deploy to shadow trading framework
        result = self.shadow_trading.deploy_strategy(
            strategy_id=strategy_id,
            strategy_params=params,
            portfolio_fraction=portfolio_fraction,
            strategy_name=f"Strategy {strategy_id}"
        )
        
        # Track in shadow strategies
        self.shadow_strategies[strategy_id] = {
            "deployed_at": datetime.datetime.now().isoformat(),
            "params": params.copy(),
            "portfolio_fraction": portfolio_fraction,
            "status": "active"
        }
        
        # Store in historical database
        self.history_db.store_strategy(strategy_id, {
            "params": params,
            "deployed_at": datetime.datetime.now().isoformat(),
            "phase": "shadow_deployed",
            "portfolio_fraction": portfolio_fraction
        })
        
        self.log(f"Strategy {strategy_id} deployed to shadow trading with {portfolio_fraction:.4%} allocation")
        
        return result
    
    def record_shadow_trade(self, strategy_id, trade_data):
        """Record a trade for a shadow-deployed strategy."""
        if strategy_id not in self.shadow_strategies:
            return {"error": f"Strategy {strategy_id} not in shadow trading"}
            
        # Record the trade in shadow trading framework
        result = self.shadow_trading.record_trade(strategy_id, trade_data)
        
        # If successful, update the historical database
        if "error" not in result:
            self.history_db.store_trade(strategy_id, trade_data)
            
        return result
    
    def evaluate_shadow_readiness(self, strategy_id):
        """Evaluate if a shadow strategy is ready for live trading."""
        if strategy_id not in self.shadow_strategies:
            return {"error": f"Strategy {strategy_id} not in shadow trading"}
            
        # Get readiness assessment from shadow trading framework
        readiness = self.shadow_trading.evaluate_readiness(strategy_id)
        
        # If strategy is ready, record in historical database
        if readiness.get("ready", False):
            self.history_db.store_strategy(strategy_id, {
                "readiness_achieved_at": datetime.datetime.now().isoformat(),
                "phase": "ready_for_live",
                "metrics": readiness.get("metrics", {})
            })
            self.log(f"Strategy {strategy_id} evaluated as READY for live trading")
        else:
            self.log(f"Strategy {strategy_id} not yet ready: {readiness.get('reason', 'Unknown')}")
            
        return readiness
    
    def update_portfolio_value(self, value=None):
        """Update the portfolio value for shadow trading."""
        return self.shadow_trading.update_portfolio_value(value=value)
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of the entire workflow."""
        report = []
        report.append("=" * 70)
        report.append("PRODUCTION TRADING WORKFLOW REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall statistics
        total_strategies = len(self.active_strategies) + len(self.pending_strategies) + len(self.rejected_strategies)
        total_shadow = len(self.shadow_strategies)
        
        report.append(f"Total strategies: {total_strategies}")
        report.append(f"  - Active approved: {len(self.active_strategies)}")
        report.append(f"  - Pending approval: {len(self.pending_strategies)}")
        report.append(f"  - Rejected: {len(self.rejected_strategies)}")
        report.append(f"Shadow deployed: {total_shadow}")
        report.append("")
        
        # Optimization statistics
        total_optimizations = sum(len(runs) for runs in self.optimization_history.values())
        converged_count = sum(1 for runs in self.optimization_history.values() 
                            for run in runs if run.get("converged", False))
        
        report.append(f"Optimization statistics:")
        report.append(f"  - Total optimization runs: {total_optimizations}")
        report.append(f"  - Converged runs: {converged_count}")
        if total_optimizations > 0:
            convergence_rate = converged_count / total_optimizations * 100
            report.append(f"  - Convergence rate: {convergence_rate:.1f}%")
        report.append("")
        
        # Adaptive approval thresholds
        report.append("Adaptive approval thresholds:")
        for strategy_id, threshold in self.adaptive_approval.thresholds.items():
            report.append(f"  - Strategy {strategy_id}: {threshold['approval_threshold']:.2%} (auto: {threshold['auto_approve_threshold']:.2%})")
        report.append("")
        
        # Shadow trading summary (from shadow trading framework)
        shadow_report = self.shadow_trading.generate_shadow_report()
        report.append(shadow_report)
        
        # Historical database insights
        db_insights = self.history_db.generate_insights_report()
        report.append("\n" + db_insights)
        
        # Bayesian learning status
        bayesian_report = self.bayesian_optimizer.generate_status_report()
        report.append("\n" + bayesian_report)
        
        # Benchmark comparisons
        benchmark_report = self.benchmarks.generate_benchmark_report()
        report.append("\n" + benchmark_report)
        
        # Recent events (last 10)
        recent_logs = self.logs[-10:] if len(self.logs) > 10 else self.logs
        report.append("\nRecent activity:")
        for log in recent_logs:
            report.append(f"  {log['timestamp']}: {log['level']} - {log['message']}")
        
        return "\n".join(report)


#-----------------------------------------------------------
# 7. MAIN DEMONSTRATION
#-----------------------------------------------------------

def run_production_workflow_demo(days=60, strategies=3):
    """Run a comprehensive demonstration of the production-ready trading workflow."""
    print("\n" + "=" * 70)
    print("PRODUCTION-READY TRADING WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    # Initialize the workflow
    workflow = ProductionReadyWorkflow()
    
    # Initial strategy parameters for optimization
    initial_params = {
        "ma_short": 10,
        "ma_long": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "volatility_lookback": 20,
        "profit_target": 0.05,
        "stop_loss": 0.03,
        "position_size": 0.1,
    }
    
    # Define market regimes for testing
    regimes = ["bullish", "bearish", "sideways", "volatile"]
    
    print(f"\nRunning {days} day simulation with {strategies} strategies...")
    
    # Step 1: Optimization Phase
    print("\n[1] OPTIMIZATION PHASE")
    print("-" * 30)
    
    for i in range(1, strategies + 1):
        strategy_id = f"strat_{i:02d}"
        print(f"Optimizing strategy {strategy_id}...")
        
        # Slightly modify initial params for each strategy to simulate different strategies
        strategy_params = initial_params.copy()
        strategy_params["ma_short"] += i * 2
        strategy_params["ma_long"] += i * 5
        
        # Run the optimization cycle
        result = workflow.start_optimization_cycle(
            strategy_id=strategy_id,
            market_data=None,  # Simulated data
            initial_params=strategy_params,
            regimes=regimes,
            max_iterations=20
        )
        
        print(f"  Strategy {strategy_id} optimization complete. Accuracy: {result['accuracy']:.2%}")
        print(f"  Convergence: {'Yes' if result['converged'] else 'No'}")
        print()
    
    # Step 2: Approval Phase
    print("\n[2] APPROVAL WORKFLOW PHASE")
    print("-" * 30)
    
    # Get all pending strategies
    pending_strategies = list(workflow.pending_strategies.keys())
    
    # Manual approvals and rejections for demonstration
    for i, strategy_id in enumerate(pending_strategies):
        if i % 3 == 0:  # Reject every 3rd strategy
            workflow.reject_strategy(
                strategy_id=strategy_id,
                reviewer="demo_user",
                reason="Demo rejection - parameter sensitivity too high"
            )
            print(f"Rejected strategy {strategy_id}")
        else:
            workflow.approve_strategy(
                strategy_id=strategy_id,
                approver="demo_user",
                comments="Demo approval - parameters look good"
            )
            print(f"Approved strategy {strategy_id}")
    
    # Step 3: Shadow Trading Phase
    print("\n[3] SHADOW TRADING PHASE")
    print("-" * 30)
    
    # Get all active strategies (those that were approved)
    active_strategies = list(workflow.active_strategies.keys())
    
    if not active_strategies:
        print("No active strategies to shadow trade!")
    else:
        # Simulate shadow trading over the time period
        for day in range(1, days + 1):
            # Update portfolio value with a small random drift
            import random
            current_value = workflow.shadow_trading.last_valuation["value"]
            new_value = current_value * (1 + random.uniform(-0.01, 0.02))  # -1% to +2% daily change
            workflow.update_portfolio_value(value=new_value)
            
            if day % 10 == 0 or day == 1:
                print(f"Day {day}: Portfolio value ${new_value:,.2f}")
            
            # For each active strategy, simulate 1-3 trades per week
            if day % 2 == 0:  # Every other day
                for strategy_id in active_strategies:
                    # Decide if we make a trade today
                    if random.random() < 0.4:  # 40% chance of a trade
                        # Generate a random trade
                        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
                        direction = random.choice(["buy", "sell"])
                        quantity = random.randint(10, 100)
                        price = random.uniform(50, 500)
                        
                        trade_data = {
                            "symbol": random.choice(symbols),
                            "direction": direction,
                            "quantity": quantity,
                            "price": price
                        }
                        
                        # Record the trade
                        workflow.record_shadow_trade(strategy_id, trade_data)
                        
                        if day % 10 == 0:  # Only print some trades to avoid spam
                            print(f"  Strategy {strategy_id}: {direction.upper()} {quantity} {trade_data['symbol']} @ ${price:.2f}")
    
    # Step 4: Evaluation Phase
    print("\n[4] STRATEGY EVALUATION PHASE")
    print("-" * 30)
    
    for strategy_id in active_strategies:
        readiness = workflow.evaluate_shadow_readiness(strategy_id)
        if readiness.get("ready", False):
            print(f"Strategy {strategy_id} is READY for live trading")
        else:
            reason = readiness.get("reason", "Unknown")
            print(f"Strategy {strategy_id} is NOT READY: {reason}")
    
    # Generate comprehensive report
    print("\n[5] COMPREHENSIVE WORKFLOW REPORT")
    print("-" * 30)
    
    report = workflow.generate_comprehensive_report()
    print(report)
    
    print("\nWorkflow demonstration complete!")
    return workflow


# Main entry point
if __name__ == "__main__":
    # Run the complete production workflow demonstration
    run_production_workflow_demo(days=60, strategies=5)
