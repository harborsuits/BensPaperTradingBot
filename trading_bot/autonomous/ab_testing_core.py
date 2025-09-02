#!/usr/bin/env python3
"""
A/B Testing Core Framework

This module provides the core classes and data structures for the A/B Testing Framework,
which enables systematic comparison between strategy variants to validate improvements.

Classes:
    TestStatus: Enum for test status tracking
    TestVariant: Represents a strategy variant in an A/B test
    TestMetrics: Statistical metrics for comparing variants
    ABTest: Core class for defining and managing A/B tests
"""

import os
import json
import logging
import uuid
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import threading
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Enum for A/B test status tracking."""
    CREATED = "created"         # Test created but not started
    RUNNING = "running"         # Test is currently running
    COMPLETED = "completed"     # Test completed successfully
    STOPPED = "stopped"         # Test stopped manually
    FAILED = "failed"           # Test failed due to an error
    INCONCLUSIVE = "inconclusive"  # Test completed but results inconclusive


class TestVariant:
    """
    Represents a strategy variant in an A/B test.
    
    This class contains information about a specific strategy configuration being
    tested, including its performance metrics and trade history. It links to a
    strategy version in the lifecycle manager.
    
    Attributes:
        variant_id: Unique identifier for this variant
        strategy_id: ID of the strategy
        version_id: Version ID in the lifecycle manager
        name: Human-readable name (e.g., "Variant A", "Baseline")
        parameters: Strategy parameters for this variant
        metrics: Performance metrics for this variant
        trade_history: History of trades during the test
        metadata: Additional metadata about this variant
    """
    
    def __init__(
        self,
        strategy_id: str,
        version_id: str,
        name: str,
        parameters: Dict[str, Any],
        variant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new test variant.
        
        Args:
            strategy_id: ID of the strategy
            version_id: Version ID in the lifecycle manager
            name: Human-readable name for this variant
            parameters: Strategy parameters for this variant
            variant_id: Optional variant ID, generated if not provided
            metadata: Additional metadata about this variant
        """
        self.variant_id = variant_id or str(uuid.uuid4())
        self.strategy_id = strategy_id
        self.version_id = version_id
        self.name = name
        self.parameters = parameters or {}
        self.metadata = metadata or {}
        
        # Performance tracking
        self.metrics: Dict[str, Any] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.regime_performance: Dict[str, Dict[str, Any]] = {}
        
        # Test timestamps
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert variant to dictionary for serialization."""
        return {
            "variant_id": self.variant_id,
            "strategy_id": self.strategy_id,
            "version_id": self.version_id,
            "name": self.name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "trade_history": self.trade_history,
            "regime_performance": self.regime_performance,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestVariant':
        """Create variant from dictionary representation."""
        variant = cls(
            strategy_id=data["strategy_id"],
            version_id=data["version_id"],
            name=data["name"],
            parameters=data["parameters"],
            variant_id=data["variant_id"],
            metadata=data["metadata"]
        )
        
        variant.metrics = data.get("metrics", {})
        variant.trade_history = data.get("trade_history", [])
        variant.regime_performance = data.get("regime_performance", {})
        variant.created_at = datetime.fromisoformat(data["created_at"])
        variant.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return variant
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics for this variant.
        
        Args:
            metrics: Performance metrics to update
        """
        self.metrics.update(metrics)
        self.updated_at = datetime.now()
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the history.
        
        Args:
            trade: Trade details to add
        """
        self.trade_history.append(trade)
        self.updated_at = datetime.now()
    
    def update_regime_performance(self, regime: str, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics for a specific market regime.
        
        Args:
            regime: Market regime identifier
            metrics: Performance metrics for this regime
        """
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {}
            
        self.regime_performance[regime].update(metrics)
        self.updated_at = datetime.now()
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """
        Get a specific performance metric.
        
        Args:
            name: Name of the metric to retrieve
            default: Default value if metric not found
            
        Returns:
            Metric value or default
        """
        return self.metrics.get(name, default)
    
    def get_regime_metric(self, regime: str, name: str, default: Any = None) -> Any:
        """
        Get a specific performance metric for a market regime.
        
        Args:
            regime: Market regime identifier
            name: Name of the metric to retrieve
            default: Default value if metric not found
            
        Returns:
            Metric value or default
        """
        if regime not in self.regime_performance:
            return default
            
        return self.regime_performance[regime].get(name, default)
    
    def __str__(self) -> str:
        """String representation of the variant."""
        metrics_str = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in self.metrics.items()
            if k in ('sharpe_ratio', 'win_rate', 'max_drawdown')
        )
        return f"{self.name} ({self.strategy_id}.{self.version_id}): {metrics_str}"


class TestMetrics:
    """
    Statistical metrics for comparing test variants.
    
    This class computes statistical metrics for comparing the performance of
    variants in an A/B test, including significance tests and confidence intervals.
    
    Attributes:
        comparison_metrics: Metrics for comparing variants
        significance_tests: Results of statistical significance tests
        confidence_intervals: Confidence intervals for metric differences
        metrics_by_regime: Performance metrics segmented by market regime
    """
    
    def __init__(self, variant_a: TestVariant, variant_b: TestVariant):
        """
        Initialize metrics for comparing variants.
        
        Args:
            variant_a: First variant (typically baseline)
            variant_b: Second variant (typically the improved version)
        """
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.comparison_metrics: Dict[str, Dict[str, Any]] = {}
        self.significance_tests: Dict[str, Dict[str, Any]] = {}
        self.confidence_intervals: Dict[str, Dict[str, Any]] = {}
        self.metrics_by_regime: Dict[str, Dict[str, Any]] = {}
        
        # Default metrics to compare
        self.default_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'win_rate', 'max_drawdown',
            'profit_factor', 'annualized_return', 'volatility'
        ]
    
    def calculate_metrics(self, metrics_to_compare: Optional[List[str]] = None) -> None:
        """
        Calculate comparison metrics between variants.
        
        Args:
            metrics_to_compare: List of metric names to compare, uses defaults if None
        """
        metrics = metrics_to_compare or self.default_metrics
        
        # Calculate basic comparison metrics
        for metric in metrics:
            value_a = self.variant_a.get_metric(metric)
            value_b = self.variant_b.get_metric(metric)
            
            if value_a is None or value_b is None:
                continue
                
            # Calculate absolute and relative differences
            abs_diff = value_b - value_a
            rel_diff = abs_diff / abs(value_a) if value_a != 0 else float('inf')
            
            # For metrics where lower is better, invert the sign
            if metric in ('max_drawdown', 'volatility'):
                abs_diff = -abs_diff
                rel_diff = -rel_diff
            
            self.comparison_metrics[metric] = {
                'value_a': value_a,
                'value_b': value_b,
                'absolute_difference': abs_diff,
                'relative_difference': rel_diff,
                'improvement': abs_diff > 0  # True if B is better than A
            }
        
        # Calculate per-regime metrics
        all_regimes = set(self.variant_a.regime_performance.keys()) | set(self.variant_b.regime_performance.keys())
        
        for regime in all_regimes:
            regime_metrics = {}
            
            for metric in metrics:
                value_a = self.variant_a.get_regime_metric(regime, metric)
                value_b = self.variant_b.get_regime_metric(regime, metric)
                
                if value_a is None or value_b is None:
                    continue
                    
                # Calculate differences
                abs_diff = value_b - value_a
                rel_diff = abs_diff / abs(value_a) if value_a != 0 else float('inf')
                
                # For metrics where lower is better, invert the sign
                if metric in ('max_drawdown', 'volatility'):
                    abs_diff = -abs_diff
                    rel_diff = -rel_diff
                
                regime_metrics[metric] = {
                    'value_a': value_a,
                    'value_b': value_b,
                    'absolute_difference': abs_diff,
                    'relative_difference': rel_diff,
                    'improvement': abs_diff > 0
                }
            
            if regime_metrics:
                self.metrics_by_regime[regime] = regime_metrics
    
    def calculate_significance(self, confidence_level: float = 0.95) -> None:
        """
        Calculate statistical significance of performance differences.
        
        This is a simplified implementation. In a real system, this would use
        proper statistical tests (t-tests, bootstrap methods, etc.)
        
        Args:
            confidence_level: Confidence level for statistical tests (0-1)
        """
        # This is a placeholder for real statistical testing
        # In a production system, we would implement proper t-tests, 
        # bootstrap methods, etc. based on trade-level data
        
        # Simplified example for now
        trade_count_a = len(self.variant_a.trade_history)
        trade_count_b = len(self.variant_b.trade_history)
        
        if trade_count_a < 30 or trade_count_b < 30:
            # Not enough trades for statistical significance
            for metric in self.comparison_metrics:
                self.significance_tests[metric] = {
                    'is_significant': False,
                    'p_value': None,
                    'confidence_level': confidence_level,
                    'reason': 'Insufficient trades for statistical testing'
                }
            return
        
        # Extract returns from trades for analysis
        # In a real implementation, this would calculate proper returns series
        returns_a = [t.get('return', 0) for t in self.variant_a.trade_history]
        returns_b = [t.get('return', 0) for t in self.variant_b.trade_history]
        
        for metric, comparison in self.comparison_metrics.items():
            # Calculate simplified p-value based on metric
            # This is just a placeholder for actual statistical tests
            is_significant = False
            p_value = 0.5  # Placeholder
            
            # For sharpe ratio, we could use Jobson-Korkie test
            if metric == 'sharpe_ratio' and returns_a and returns_b:
                # Simulate a p-value based on relative difference
                rel_diff = abs(comparison['relative_difference'])
                if rel_diff > 0.2:
                    p_value = 0.01  # Significant difference
                    is_significant = True
                elif rel_diff > 0.1:
                    p_value = 0.08  # Borderline significant
                    is_significant = p_value < (1 - confidence_level)
                else:
                    p_value = 0.3  # Not significant
                    is_significant = False
            
            # Store results
            self.significance_tests[metric] = {
                'is_significant': is_significant,
                'p_value': p_value,
                'confidence_level': confidence_level,
                'sample_size_a': trade_count_a,
                'sample_size_b': trade_count_b
            }
            
            # Calculate confidence intervals
            if is_significant:
                diff = comparison['absolute_difference']
                # Simplified confidence interval calculation
                # This would be proper bootstrap or parametric CI in production
                ci_width = diff * 0.4  # Placeholder
                
                self.confidence_intervals[metric] = {
                    'lower_bound': diff - ci_width,
                    'upper_bound': diff + ci_width,
                    'confidence_level': confidence_level
                }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of comparison metrics and significance tests.
        
        Returns:
            Summary dictionary with comparison results
        """
        # Calculate overall assessment
        is_significant_improvement = any(
            test['is_significant'] and self.comparison_metrics[metric]['improvement']
            for metric, test in self.significance_tests.items()
            if metric in ('sharpe_ratio', 'sortino_ratio', 'profit_factor')
        )
        
        significant_metrics = [
            metric for metric, test in self.significance_tests.items()
            if test['is_significant'] and self.comparison_metrics[metric]['improvement']
        ]
        
        # Determine best variant for each regime
        regime_winners = {}
        for regime, metrics in self.metrics_by_regime.items():
            # Count metrics where each variant is better
            a_better_count = sum(1 for m in metrics.values() if not m['improvement'])
            b_better_count = sum(1 for m in metrics.values() if m['improvement'])
            
            if b_better_count > a_better_count:
                regime_winners[regime] = 'B'
            elif a_better_count > b_better_count:
                regime_winners[regime] = 'A'
            else:
                regime_winners[regime] = 'tie'
        
        return {
            'comparison_metrics': self.comparison_metrics,
            'significance_tests': self.significance_tests,
            'confidence_intervals': self.confidence_intervals,
            'metrics_by_regime': self.metrics_by_regime,
            'regime_winners': regime_winners,
            'is_significant_improvement': is_significant_improvement,
            'significant_improved_metrics': significant_metrics,
            'variant_a': {
                'id': self.variant_a.variant_id,
                'name': self.variant_a.name
            },
            'variant_b': {
                'id': self.variant_b.variant_id,
                'name': self.variant_b.name
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'comparison_metrics': self.comparison_metrics,
            'significance_tests': self.significance_tests,
            'confidence_intervals': self.confidence_intervals,
            'metrics_by_regime': self.metrics_by_regime
        }
    
    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any],
        variant_a: TestVariant,
        variant_b: TestVariant
    ) -> 'TestMetrics':
        """Create metrics from dictionary representation."""
        metrics = cls(variant_a, variant_b)
        
        metrics.comparison_metrics = data.get('comparison_metrics', {})
        metrics.significance_tests = data.get('significance_tests', {})
        metrics.confidence_intervals = data.get('confidence_intervals', {})
        metrics.metrics_by_regime = data.get('metrics_by_regime', {})
        
        return metrics


class ABTest:
    """
    Core class for defining and managing A/B tests.
    
    This class manages the comparison between two strategy variants (A and B),
    including test configuration, execution, and analysis of results.
    
    Attributes:
        test_id: Unique identifier for this test
        name: Human-readable name for the test
        description: Detailed description of the test
        variant_a: Baseline variant (A)
        variant_b: Comparison variant (B)
        status: Current test status
        config: Test configuration parameters
        metrics: Comparison metrics between variants
        created_at: When the test was created
        started_at: When the test was started
        completed_at: When the test was completed
        metadata: Additional test metadata
    """
    
    def __init__(
        self,
        name: str,
        variant_a: TestVariant,
        variant_b: TestVariant,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        test_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new A/B test.
        
        Args:
            name: Human-readable name for the test
            variant_a: Baseline variant (A)
            variant_b: Comparison variant (B)
            config: Test configuration parameters
            description: Detailed description of the test
            test_id: Optional test ID, generated if not provided
            metadata: Additional test metadata
        """
        self.test_id = test_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.status = TestStatus.CREATED
        self.config = config or {
            'duration_days': 30,  # Test duration in days
            'confidence_level': 0.95,  # Statistical confidence level
            'metrics_to_compare': [  # Metrics to compare
                'sharpe_ratio', 'sortino_ratio', 'win_rate', 'max_drawdown',
                'profit_factor', 'annualized_return', 'volatility'
            ],
            'auto_promote_threshold': 0.1,  # Min improvement for auto-promotion
            'min_trade_count': 30  # Minimum trades for statistical validity
        }
        self.metrics = None  # Will be populated after test runs
        self.metadata = metadata or {}
        
        # Timestamps
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
        # Results
        self.winner = None  # 'A', 'B', or None if inconclusive
        self.conclusion = ""  # Text conclusion of test
        self.result_summary = {}  # Detailed result summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "variant_a": self.variant_a.to_dict(),
            "variant_b": self.variant_b.to_dict(),
            "status": self.status.value,
            "config": self.config,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "winner": self.winner,
            "conclusion": self.conclusion,
            "result_summary": self.result_summary,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTest':
        """Create test from dictionary representation."""
        variant_a = TestVariant.from_dict(data["variant_a"])
        variant_b = TestVariant.from_dict(data["variant_b"])
        
        test = cls(
            name=data["name"],
            variant_a=variant_a,
            variant_b=variant_b,
            config=data["config"],
            description=data["description"],
            test_id=data["test_id"],
            metadata=data["metadata"]
        )
        
        test.status = TestStatus(data["status"])
        
        if data["metrics"] and variant_a and variant_b:
            test.metrics = TestMetrics.from_dict(data["metrics"], variant_a, variant_b)
            
        if data["started_at"]:
            test.started_at = datetime.fromisoformat(data["started_at"])
            
        if data["completed_at"]:
            test.completed_at = datetime.fromisoformat(data["completed_at"])
            
        test.created_at = datetime.fromisoformat(data["created_at"])
        test.winner = data["winner"]
        test.conclusion = data["conclusion"]
        test.result_summary = data["result_summary"]
        
        return test
    
    def start_test(self) -> None:
        """Start the A/B test."""
        if self.status != TestStatus.CREATED:
            raise ValueError(f"Cannot start test in {self.status} state")
            
        self.status = TestStatus.RUNNING
        self.started_at = datetime.now()
    
    def complete_test(self) -> None:
        """Mark the test as completed and analyze results."""
        if self.status != TestStatus.RUNNING:
            raise ValueError(f"Cannot complete test in {self.status} state")
            
        self.status = TestStatus.COMPLETED
        self.completed_at = datetime.now()
        
        # Calculate metrics and determine winner
        self.analyze_results()
    
    def stop_test(self, reason: str = "") -> None:
        """
        Stop the test before completion.
        
        Args:
            reason: Reason for stopping the test
        """
        if self.status not in (TestStatus.RUNNING, TestStatus.CREATED):
            raise ValueError(f"Cannot stop test in {self.status} state")
            
        self.status = TestStatus.STOPPED
        self.completed_at = datetime.now()
        self.conclusion = f"Test stopped: {reason}"
    
    def fail_test(self, error: str) -> None:
        """
        Mark the test as failed.
        
        Args:
            error: Error that caused the test to fail
        """
        self.status = TestStatus.FAILED
        self.completed_at = datetime.now()
        self.conclusion = f"Test failed: {error}"
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze test results and determine the winner.
        
        Returns:
            Result summary dictionary
        """
        # Initialize metrics if needed
        if not self.metrics:
            self.metrics = TestMetrics(self.variant_a, self.variant_b)
            
        # Calculate comparison metrics
        metrics_to_compare = self.config.get('metrics_to_compare')
        self.metrics.calculate_metrics(metrics_to_compare)
        
        # Calculate statistical significance
        confidence_level = self.config.get('confidence_level', 0.95)
        self.metrics.calculate_significance(confidence_level)
        
        # Get result summary
        self.result_summary = self.metrics.get_summary()
        
        # Determine overall winner
        is_significant = self.result_summary.get('is_significant_improvement', False)
        significant_metrics = self.result_summary.get('significant_improved_metrics', [])
        
        if is_significant:
            # B is significantly better than A
            self.winner = 'B'
            self.conclusion = (
                f"Variant B ({self.variant_b.name}) shows statistically significant "
                f"improvement over Variant A ({self.variant_a.name}) in: "
                f"{', '.join(significant_metrics)}."
            )
        elif significant_metrics:
            # Some metrics improved but not the critical ones
            self.winner = None
            self.conclusion = (
                f"Test is inconclusive. Variant B shows some improvements in "
                f"{', '.join(significant_metrics)}, but not in critical metrics."
            )
            self.status = TestStatus.INCONCLUSIVE
        else:
            # No significant improvements
            self.winner = 'A'
            self.conclusion = (
                f"Variant B ({self.variant_b.name}) does not show statistically "
                f"significant improvement over Variant A ({self.variant_a.name})."
            )
        
        # Add regime-specific insights
        regime_winners = self.result_summary.get('regime_winners', {})
        if regime_winners:
            regime_insights = []
            for regime, winner in regime_winners.items():
                if winner == 'B':
                    regime_insights.append(
                        f"Variant B performs better in {regime} markets."
                    )
                elif winner == 'A':
                    regime_insights.append(
                        f"Variant A performs better in {regime} markets."
                    )
            
            if regime_insights:
                self.conclusion += " " + " ".join(regime_insights)
        
        return self.result_summary
    
    def should_promote_variant_b(self) -> bool:
        """
        Determine if variant B should be promoted based on test results.
        
        Returns:
            True if variant B should be promoted, False otherwise
        """
        if self.status != TestStatus.COMPLETED or not self.result_summary:
            return False
            
        if self.winner != 'B':
            return False
            
        # Check if improvement exceeds auto-promote threshold
        auto_threshold = self.config.get('auto_promote_threshold', 0.1)
        
        # Check critical metrics
        for metric in ('sharpe_ratio', 'sortino_ratio'):
            if metric in self.metrics.comparison_metrics:
                rel_diff = self.metrics.comparison_metrics[metric].get('relative_difference', 0)
                
                if rel_diff >= auto_threshold:
                    return True
                    
        return False
    
    def __str__(self) -> str:
        """String representation of the test."""
        status_str = self.status.value.capitalize()
        if self.completed_at:
            duration = (self.completed_at - self.started_at).days if self.started_at else 0
            return f"{self.name} ({status_str}, {duration} days): {self.conclusion}"
        elif self.started_at:
            duration = (datetime.now() - self.started_at).days
            return f"{self.name} ({status_str}, running for {duration} days)"
        else:
            return f"{self.name} ({status_str})"
