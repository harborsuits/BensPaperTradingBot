#!/usr/bin/env python3
"""
A/B Testing Analysis

This module provides statistical analysis capabilities for the A/B Testing Framework,
including significance testing, confidence interval calculation, and regime-specific
performance comparison. It builds directly upon our established event-driven architecture
and integrates with our existing components.

Classes:
    ABTestAnalyzer: Statistical analysis for A/B tests
    SignificanceTest: Statistical significance testing
    ConfidenceInterval: Confidence interval calculation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy import stats

# Import A/B testing core components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalTestType(str, Enum):
    """Types of statistical significance tests."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    BOOTSTRAP = "bootstrap"
    MANN_WHITNEY = "mann_whitney"


class SignificanceTest:
    """
    Performs statistical significance testing on A/B test results.
    
    This class implements various statistical tests to determine if
    the difference between two variants is statistically significant.
    
    Methods:
        t_test: Performs Student's t-test
        wilcoxon: Performs Wilcoxon signed-rank test
        mann_whitney: Performs Mann-Whitney U test
        bootstrap: Performs bootstrap resampling test
    """
    
    @staticmethod
    def t_test(
        data_a: List[float],
        data_b: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform Student's t-test for independent samples.
        
        Args:
            data_a: Sample data for variant A
            data_b: Sample data for variant B
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with test results
        """
        if len(data_a) < 2 or len(data_b) < 2:
            return {
                "is_significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": 1 - alpha,
                "error": "Insufficient data for t-test"
            }
        
        try:
            t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
            
            return {
                "is_significant": p_value < alpha,
                "p_value": p_value,
                "test_statistic": t_stat,
                "confidence_level": 1 - alpha,
                "sample_size_a": len(data_a),
                "sample_size_b": len(data_b),
                "mean_a": np.mean(data_a),
                "mean_b": np.mean(data_b),
                "std_a": np.std(data_a, ddof=1),
                "std_b": np.std(data_b, ddof=1)
            }
        except Exception as e:
            logger.error(f"Error in t-test: {str(e)}")
            return {
                "is_significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": 1 - alpha,
                "error": str(e)
            }
    
    @staticmethod
    def wilcoxon(
        data_a: List[float],
        data_b: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        
        Args:
            data_a: Sample data for variant A
            data_b: Sample data for variant B
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with test results
        """
        if len(data_a) < 10 or len(data_b) < 10:
            return {
                "is_significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": 1 - alpha,
                "error": "Insufficient data for Wilcoxon test (need at least 10 samples)"
            }
            
        # Ensure equal length for paired test
        min_len = min(len(data_a), len(data_b))
        data_a = data_a[:min_len]
        data_b = data_b[:min_len]
        
        try:
            w_stat, p_value = stats.wilcoxon(data_a, data_b)
            
            return {
                "is_significant": p_value < alpha,
                "p_value": p_value,
                "test_statistic": w_stat,
                "confidence_level": 1 - alpha,
                "sample_size": min_len,
                "mean_diff": np.mean(np.array(data_b) - np.array(data_a))
            }
        except Exception as e:
            logger.error(f"Error in Wilcoxon test: {str(e)}")
            return {
                "is_significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": 1 - alpha,
                "error": str(e)
            }
    
    @staticmethod
    def mann_whitney(
        data_a: List[float],
        data_b: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test for independent samples.
        
        This is a non-parametric test that doesn't require normal distribution.
        
        Args:
            data_a: Sample data for variant A
            data_b: Sample data for variant B
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with test results
        """
        if len(data_a) < 5 or len(data_b) < 5:
            return {
                "is_significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": 1 - alpha,
                "error": "Insufficient data for Mann-Whitney test"
            }
        
        try:
            u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
            
            return {
                "is_significant": p_value < alpha,
                "p_value": p_value,
                "test_statistic": u_stat,
                "confidence_level": 1 - alpha,
                "sample_size_a": len(data_a),
                "sample_size_b": len(data_b),
                "median_a": np.median(data_a),
                "median_b": np.median(data_b)
            }
        except Exception as e:
            logger.error(f"Error in Mann-Whitney test: {str(e)}")
            return {
                "is_significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": 1 - alpha,
                "error": str(e)
            }
    
    @staticmethod
    def bootstrap(
        data_a: List[float],
        data_b: List[float],
        statistic: str = 'mean',
        n_iterations: int = 1000,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform bootstrap resampling test.
        
        Args:
            data_a: Sample data for variant A
            data_b: Sample data for variant B
            statistic: Statistic to bootstrap ('mean', 'median', etc.)
            n_iterations: Number of bootstrap iterations
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with test results
        """
        if len(data_a) < 5 or len(data_b) < 5:
            return {
                "is_significant": False,
                "p_value": None,
                "confidence_level": 1 - alpha,
                "error": "Insufficient data for bootstrap test"
            }
        
        try:
            # Define function to compute statistic
            if statistic == 'mean':
                stat_func = np.mean
            elif statistic == 'median':
                stat_func = np.median
            else:
                stat_func = np.mean  # Default to mean
            
            # Original difference
            original_diff = stat_func(data_b) - stat_func(data_a)
            
            # Bootstrap resampling
            bootstrap_diffs = []
            
            for _ in range(n_iterations):
                # Resample with replacement
                resample_a = np.random.choice(data_a, size=len(data_a), replace=True)
                resample_b = np.random.choice(data_b, size=len(data_b), replace=True)
                
                # Compute difference in statistic
                diff = stat_func(resample_b) - stat_func(resample_a)
                bootstrap_diffs.append(diff)
            
            # Calculate p-value
            # Two-tailed test: check how many differences are more extreme than original
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(original_diff))
            
            # Calculate confidence interval
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
            
            return {
                "is_significant": p_value < alpha,
                "p_value": p_value,
                "confidence_level": 1 - alpha,
                "sample_size_a": len(data_a),
                "sample_size_b": len(data_b),
                "original_diff": original_diff,
                "confidence_interval": (ci_lower, ci_upper),
                "statistic": statistic,
                "n_iterations": n_iterations
            }
        except Exception as e:
            logger.error(f"Error in bootstrap test: {str(e)}")
            return {
                "is_significant": False,
                "p_value": None,
                "confidence_level": 1 - alpha,
                "error": str(e)
            }


class ConfidenceInterval:
    """
    Calculates confidence intervals for various metrics.
    
    This class provides methods to calculate confidence intervals
    for different types of performance metrics.
    """
    
    @staticmethod
    def mean_confidence_interval(
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for mean.
        
        Args:
            data: Sample data
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if not data:
            return 0.0, 0.0, 0.0
            
        a = np.array(data)
        n = len(a)
        
        if n < 2:
            return float(a[0]) if n == 1 else 0.0, 0.0, 0.0
            
        m = np.mean(a)
        se = stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return m, m - h, m + h
    
    @staticmethod
    def sharpe_ratio_confidence_interval(
        returns: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for Sharpe ratio.
        
        Args:
            returns: List of returns
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (sharpe_ratio, lower_bound, upper_bound)
        """
        if not returns or len(returns) < 5:
            return 0.0, 0.0, 0.0
            
        a = np.array(returns)
        n = len(a)
        
        # Calculate Sharpe ratio
        mu = np.mean(a)
        sigma = np.std(a, ddof=1)
        
        if sigma == 0:
            return 0.0, 0.0, 0.0
            
        sharpe = mu / sigma
        
        # Calculate standard error of Sharpe ratio
        # Using Lo (2002) approximation
        se = np.sqrt((1 + sharpe**2 / 2) / n)
        
        # Calculate confidence interval
        z = stats.norm.ppf((1 + confidence) / 2)
        
        return sharpe, sharpe - z * se, sharpe + z * se
    
    @staticmethod
    def win_rate_confidence_interval(
        wins: int,
        total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for win rate.
        
        Args:
            wins: Number of winning trades
            total: Total number of trades
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (win_rate, lower_bound, upper_bound)
        """
        if total == 0:
            return 0.0, 0.0, 0.0
            
        # Calculate win rate
        p = wins / total
        
        # Calculate standard error
        # Using normal approximation to binomial
        se = np.sqrt(p * (1 - p) / total)
        
        # Calculate confidence interval
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate bounds
        lower = max(0.0, p - z * se)
        upper = min(1.0, p + z * se)
        
        return p, lower, upper
    
    @staticmethod
    def drawdown_confidence_interval(
        returns: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for maximum drawdown.
        
        Args:
            returns: List of returns
            confidence: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Tuple of (max_drawdown, lower_bound, upper_bound)
        """
        if not returns or len(returns) < 10:
            return 0.0, 0.0, 0.0
            
        a = np.array(returns)
        
        # Calculate actual max drawdown
        cum_returns = np.cumprod(1 + a) - 1
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / (1 + running_max)
        max_dd = np.min(drawdowns)
        
        # Bootstrap for confidence interval
        bootstrap_dd = []
        for _ in range(n_bootstrap):
            # Resample returns
            sample = np.random.choice(a, size=len(a), replace=True)
            
            # Calculate drawdown
            cum_returns = np.cumprod(1 + sample) - 1
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / (1 + running_max)
            bootstrap_dd.append(np.min(drawdowns))
        
        # Calculate bounds
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 - (1 - confidence) / 2) * 100
        
        lower = np.percentile(bootstrap_dd, lower_percentile)
        upper = np.percentile(bootstrap_dd, upper_percentile)
        
        return max_dd, lower, upper


class ABTestAnalyzer:
    """
    Performs statistical analysis on A/B test results.
    
    This class provides methods to analyze A/B test results, including
    significance testing, confidence intervals, and regime-specific analysis.
    It builds upon our established event-driven architecture and integrates
    with our existing components.
    """
    
    def __init__(self):
        """Initialize the A/B test analyzer."""
        pass
    
    def analyze_test(self, test: ABTest) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on an A/B test.
        
        Args:
            test: A/B test to analyze
            
        Returns:
            Analysis results dictionary
        """
        if test.status not in (TestStatus.COMPLETED, TestStatus.INCONCLUSIVE):
            return {
                "error": f"Cannot analyze test in {test.status} state"
            }
            
        # Initialize results
        results = {
            "test_id": test.test_id,
            "name": test.name,
            "variant_a": {
                "id": test.variant_a.variant_id,
                "name": test.variant_a.name,
                "metrics": test.variant_a.metrics
            },
            "variant_b": {
                "id": test.variant_b.variant_id,
                "name": test.variant_b.name,
                "metrics": test.variant_b.metrics
            },
            "significance_tests": {},
            "confidence_intervals": {},
            "regime_analysis": {}
        }
        
        # Get trade data (if available)
        trades_a = test.variant_a.trade_history
        trades_b = test.variant_b.trade_history
        
        # Extract returns from trades (if available)
        returns_a = [t.get('return', 0) for t in trades_a]
        returns_b = [t.get('return', 0) for t in trades_b]
        
        # Run significance tests if we have enough data
        if len(returns_a) >= 10 and len(returns_b) >= 10:
            # T-test for returns
            results["significance_tests"]["returns_t_test"] = SignificanceTest.t_test(
                returns_a, returns_b
            )
            
            # Mann-Whitney test (non-parametric)
            results["significance_tests"]["returns_mann_whitney"] = SignificanceTest.mann_whitney(
                returns_a, returns_b
            )
            
            # Bootstrap test
            results["significance_tests"]["returns_bootstrap"] = SignificanceTest.bootstrap(
                returns_a, returns_b, statistic='mean'
            )
        
        # Calculate confidence intervals for key metrics
        if returns_a and returns_b:
            # Sharpe ratio confidence intervals
            results["confidence_intervals"]["sharpe_ratio_a"] = ConfidenceInterval.sharpe_ratio_confidence_interval(
                returns_a
            )
            results["confidence_intervals"]["sharpe_ratio_b"] = ConfidenceInterval.sharpe_ratio_confidence_interval(
                returns_b
            )
            
            # Win rate confidence intervals
            wins_a = sum(1 for r in returns_a if r > 0)
            wins_b = sum(1 for r in returns_b if r > 0)
            
            results["confidence_intervals"]["win_rate_a"] = ConfidenceInterval.win_rate_confidence_interval(
                wins_a, len(returns_a)
            )
            results["confidence_intervals"]["win_rate_b"] = ConfidenceInterval.win_rate_confidence_interval(
                wins_b, len(returns_b)
            )
            
            # Drawdown confidence intervals
            results["confidence_intervals"]["drawdown_a"] = ConfidenceInterval.drawdown_confidence_interval(
                returns_a
            )
            results["confidence_intervals"]["drawdown_b"] = ConfidenceInterval.drawdown_confidence_interval(
                returns_b
            )
        
        # Analyze performance by regime
        regime_performance_a = test.variant_a.regime_performance
        regime_performance_b = test.variant_b.regime_performance
        
        all_regimes = set(regime_performance_a.keys()) | set(regime_performance_b.keys())
        
        for regime in all_regimes:
            regime_metrics_a = regime_performance_a.get(regime, {})
            regime_metrics_b = regime_performance_b.get(regime, {})
            
            results["regime_analysis"][regime] = self._compare_regime_metrics(
                regime, regime_metrics_a, regime_metrics_b
            )
        
        # Determine overall recommendation
        results["recommendation"] = self._generate_recommendation(
            test, results
        )
        
        return results
    
    def _compare_regime_metrics(
        self,
        regime: str,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare metrics between variants for a specific regime.
        
        Args:
            regime: Market regime identifier
            metrics_a: Metrics for variant A in this regime
            metrics_b: Metrics for variant B in this regime
            
        Returns:
            Comparison results
        """
        comparison = {
            "regime": regime,
            "metrics_comparison": {},
            "winner": None,
            "conclusion": ""
        }
        
        if not metrics_a or not metrics_b:
            comparison["conclusion"] = "Insufficient data for comparison"
            return comparison
            
        # Compare key metrics
        key_metrics = ['sharpe_ratio', 'win_rate', 'max_drawdown']
        winner_score = 0
        
        for metric in key_metrics:
            value_a = metrics_a.get(metric)
            value_b = metrics_b.get(metric)
            
            if value_a is None or value_b is None:
                continue
                
            # Calculate difference
            diff = value_b - value_a
            
            # For metrics where lower is better, invert the comparison
            is_lower_better = metric in ('max_drawdown', 'volatility')
            if is_lower_better:
                diff = -diff
                
            # Determine which variant is better
            better_variant = 'B' if diff > 0 else 'A'
            
            # Update winner score
            if better_variant == 'B':
                winner_score += 1
            else:
                winner_score -= 1
                
            comparison["metrics_comparison"][metric] = {
                "value_a": value_a,
                "value_b": value_b,
                "difference": diff,
                "better_variant": better_variant
            }
        
        # Determine overall winner for this regime
        if winner_score > 0:
            comparison["winner"] = 'B'
            comparison["conclusion"] = f"Variant B outperforms in {regime} regime"
        elif winner_score < 0:
            comparison["winner"] = 'A'
            comparison["conclusion"] = f"Variant A outperforms in {regime} regime"
        else:
            comparison["winner"] = 'tie'
            comparison["conclusion"] = f"No clear winner in {regime} regime"
        
        return comparison
    
    def _generate_recommendation(
        self,
        test: ABTest,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an overall recommendation based on analysis results.
        
        Args:
            test: A/B test
            analysis_results: Analysis results dictionary
            
        Returns:
            Recommendation dictionary
        """
        # Initialize recommendation
        recommendation = {
            "promote_variant_b": False,
            "confidence": "low",
            "explanation": "",
            "regime_specific": {}
        }
        
        # Check if any significance tests show B is better
        significant_tests = [
            test for test_name, test in analysis_results.get("significance_tests", {}).items()
            if test.get("is_significant", False)
        ]
        
        significant_improvements = []
        
        # Check key metrics
        metrics_a = test.variant_a.metrics
        metrics_b = test.variant_b.metrics
        
        for metric in ['sharpe_ratio', 'sortino_ratio', 'win_rate']:
            value_a = metrics_a.get(metric)
            value_b = metrics_b.get(metric)
            
            if value_a is None or value_b is None:
                continue
                
            improvement = (value_b - value_a) / abs(value_a) if value_a != 0 else float('inf')
            
            if improvement > 0.05:  # 5% improvement
                significant_improvements.append({
                    "metric": metric,
                    "improvement": improvement
                })
        
        # Check drawdown improvement
        dd_a = metrics_a.get('max_drawdown')
        dd_b = metrics_b.get('max_drawdown')
        
        if dd_a is not None and dd_b is not None:
            # Calculate absolute improvement (less negative is better)
            improvement = (abs(dd_a) - abs(dd_b)) / abs(dd_a) if dd_a != 0 else float('inf')
            
            if improvement > 0.05:  # 5% improvement
                significant_improvements.append({
                    "metric": "max_drawdown",
                    "improvement": improvement
                })
        
        # Determine if B should be promoted
        if len(significant_tests) >= 2 and len(significant_improvements) >= 2:
            recommendation["promote_variant_b"] = True
            recommendation["confidence"] = "high"
            recommendation["explanation"] = (
                f"Variant B ({test.variant_b.name}) shows statistically significant "
                f"improvements across multiple metrics and tests. Recommend promotion."
            )
        elif len(significant_tests) >= 1 and len(significant_improvements) >= 1:
            recommendation["promote_variant_b"] = True
            recommendation["confidence"] = "medium"
            recommendation["explanation"] = (
                f"Variant B ({test.variant_b.name}) shows some statistically significant "
                f"improvements. Consider promotion with monitoring."
            )
        else:
            recommendation["promote_variant_b"] = False
            recommendation["confidence"] = "low"
            recommendation["explanation"] = (
                f"Variant B ({test.variant_b.name}) does not show consistent "
                f"statistically significant improvements. Not recommended for promotion."
            )
        
        # Check if we have regime-specific recommendations
        regime_analysis = analysis_results.get("regime_analysis", {})
        
        for regime, analysis in regime_analysis.items():
            if analysis.get("winner") == 'B':
                recommendation["regime_specific"][regime] = {
                    "use_variant_b": True,
                    "explanation": f"Use variant B in {regime} market conditions"
                }
            elif analysis.get("winner") == 'A':
                recommendation["regime_specific"][regime] = {
                    "use_variant_b": False,
                    "explanation": f"Use variant A in {regime} market conditions"
                }
        
        # Check if we have regime-specific switching recommendation
        if len(recommendation["regime_specific"]) > 1:
            b_regimes = [
                regime for regime, rec in recommendation["regime_specific"].items()
                if rec.get("use_variant_b")
            ]
            
            a_regimes = [
                regime for regime, rec in recommendation["regime_specific"].items()
                if not rec.get("use_variant_b")
            ]
            
            if b_regimes and a_regimes:
                recommendation["regime_switching"] = {
                    "recommended": True,
                    "explanation": (
                        f"Consider regime-switching approach: use variant B in "
                        f"{', '.join(b_regimes)} regimes and variant A in "
                        f"{', '.join(a_regimes)} regimes."
                    )
                }
        
        return recommendation


# Singleton instance
_ab_test_analyzer = None


def get_ab_test_analyzer() -> ABTestAnalyzer:
    """
    Get singleton instance of ABTestAnalyzer.
    
    Returns:
        ABTestAnalyzer instance
    """
    global _ab_test_analyzer
    
    if _ab_test_analyzer is None:
        _ab_test_analyzer = ABTestAnalyzer()
    
    return _ab_test_analyzer
