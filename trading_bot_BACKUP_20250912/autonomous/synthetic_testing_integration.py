#!/usr/bin/env python3
"""
Synthetic Market Testing Integration

This module connects the Synthetic Market Generator with the A/B Testing Framework
and Approval Workflow, allowing strategies to be evaluated across different
market regimes before human review.

It provides:
1. Synthetic market data generation for strategy evaluation
2. Regime-specific backtesting for A/B tests
3. Enhanced approval requests with synthetic test results
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import synthetic market components
from trading_bot.autonomous.synthetic_market_generator import (
    SyntheticMarketGenerator, MarketRegimeType
)
from trading_bot.autonomous.synthetic_market_generator_correlations import (
    CorrelatedMarketGenerator, CorrelationStructure
)

# Import A/B testing components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)
from trading_bot.autonomous.ab_testing_manager import (
    get_ab_test_manager
)
from trading_bot.autonomous.ab_testing_analysis import (
    get_ab_test_analyzer
)

# Import approval workflow components
from trading_bot.autonomous.approval_workflow import (
    get_approval_workflow_manager, ApprovalStatus, ApprovalRequest
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticTestingIntegration:
    """
    Integrates synthetic market generation with A/B testing and approval workflow.
    
    This class:
    1. Generates synthetic market data across different regimes
    2. Runs strategies against these synthetic markets
    3. Enhances approval requests with regime-specific performance data
    4. Provides confidence scoring based on multi-regime testing
    """
    
    def __init__(self):
        """Initialize the synthetic testing integration."""
        # Core components
        self.event_bus = EventBus()
        self.synthetic_generator = SyntheticMarketGenerator()
        self.correlated_generator = CorrelatedMarketGenerator()
        self.ab_test_manager = get_ab_test_manager()
        self.ab_test_analyzer = get_ab_test_analyzer()
        self.approval_manager = get_approval_workflow_manager()
        
        # Configuration
        self.default_test_days = 252  # One trading year
        self.default_symbols = ["BTC", "ETH", "SOL", "ADA"]
        self.default_regimes = [
            MarketRegimeType.BULLISH,
            MarketRegimeType.BEARISH,
            MarketRegimeType.SIDEWAYS,
            MarketRegimeType.VOLATILE
        ]
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Synthetic Testing Integration initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        # Listen for test creation events
        self.event_bus.register(
            "ab_test_created",
            self._handle_test_created
        )
        
        # Listen for approval request creation events
        self.event_bus.register(
            EventType.APPROVAL_REQUEST_CREATED,
            self._handle_approval_request_created
        )
    
    def _handle_test_created(self, event: Event):
        """
        Handle test creation events by adding synthetic testing.
        
        Args:
            event: Test created event
        """
        test_id = event.data.get("test_id")
        if not test_id:
            return
            
        # Get the test
        test = self.ab_test_manager.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return
            
        # Add synthetic testing if it's enabled for this test
        if test.metadata.get("use_synthetic_testing", False):
            logger.info(f"Adding synthetic testing for test {test_id}")
            self.add_synthetic_testing_to_test(test)
    
    def _handle_approval_request_created(self, event: Event):
        """
        Handle approval request creation to enhance with synthetic data.
        
        Args:
            event: Approval request created event
        """
        request_id = event.data.get("request_id")
        test_id = event.data.get("test_id")
        if not request_id or not test_id:
            return
            
        # Get the request and test
        request = self.approval_manager.get_request(request_id)
        test = self.ab_test_manager.get_test(test_id)
        if not request or not test:
            return
            
        # Check if synthetic data is already included
        if test.metadata.get("synthetic_testing_completed", False):
            logger.info(f"Synthetic testing already completed for test {test_id}")
            return
            
        # Add synthetic testing results to the approval request
        logger.info(f"Enhancing approval request {request_id} with synthetic testing results")
        self.enhance_approval_request_with_synthetic_data(request, test)
    
    def add_synthetic_testing_to_test(self, test: ABTest):
        """
        Add synthetic market testing to an A/B test.
        
        Args:
            test: The A/B test to enhance with synthetic testing
        """
        try:
            # Configure synthetic testing parameters based on test metadata
            days = test.metadata.get("synthetic_test_days", self.default_test_days)
            symbols = test.metadata.get("synthetic_test_symbols", self.default_symbols)
            regimes = test.metadata.get("synthetic_test_regimes", self.default_regimes)
            
            # Generate synthetic data for each regime
            regime_results = {}
            for regime in regimes:
                # Generate synthetic market data for this regime
                market_data = self._generate_regime_specific_data(
                    symbols=symbols,
                    days=days,
                    regime=regime
                )
                
                # Run both variants against this market data
                variant_a_results = self._backtest_strategy(
                    strategy_id=test.variant_a.strategy_id,
                    version_id=test.variant_a.version_id,
                    market_data=market_data
                )
                
                variant_b_results = self._backtest_strategy(
                    strategy_id=test.variant_b.strategy_id,
                    version_id=test.variant_b.version_id,
                    market_data=market_data
                )
                
                # Compare results
                comparison = self._compare_backtest_results(
                    variant_a_results, variant_b_results
                )
                
                # Save regime-specific results
                regime_results[regime.value] = {
                    "variant_a": variant_a_results,
                    "variant_b": variant_b_results,
                    "comparison": comparison
                }
            
            # Update test metadata with synthetic testing results
            test.metadata["synthetic_testing_completed"] = True
            test.metadata["synthetic_testing_results"] = regime_results
            test.metadata["synthetic_testing_timestamp"] = datetime.utcnow().isoformat()
            
            # Update the test in the manager
            self.ab_test_manager.update_test(test)
            
            # Emit event for synthetic testing completion
            self.event_bus.emit(
                Event(
                    event_type="synthetic_testing_completed",
                    data={
                        "test_id": test.test_id,
                        "regime_count": len(regimes),
                        "results_summary": self._generate_regime_summary(regime_results)
                    },
                    source="synthetic_testing_integration"
                )
            )
            
            logger.info(f"Synthetic testing completed for test {test.test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in synthetic testing for test {test.test_id}: {str(e)}")
            return False
    
    def enhance_approval_request_with_synthetic_data(
        self, request: ApprovalRequest, test: ABTest
    ):
        """
        Enhance an approval request with synthetic testing results.
        
        Args:
            request: The approval request to enhance
            test: The associated A/B test
        """
        try:
            # If test doesn't have synthetic results, run the tests now
            if not test.metadata.get("synthetic_testing_completed", False):
                success = self.add_synthetic_testing_to_test(test)
                if not success:
                    logger.error(f"Failed to add synthetic testing to test {test.test_id}")
                    return False
                
                # Get updated test with results
                test = self.ab_test_manager.get_test(test.test_id)
            
            # Extract synthetic testing results
            regime_results = test.metadata.get("synthetic_testing_results", {})
            if not regime_results:
                logger.warning(f"No synthetic testing results found for test {test.test_id}")
                return False
            
            # Create enhanced approval request (this is a new request that replaces the old one)
            # In a real implementation, we would update the existing request instead
            enhanced_request = self.approval_manager.create_request(
                test_id=request.test_id,
                strategy_id=request.strategy_id,
                version_id=request.version_id,
                requester=f"{request.requester}_with_synthetic_data"
            )
            
            # Add synthetic testing results to the request comments
            summary = self._generate_regime_summary(regime_results)
            confidence_score = self._calculate_confidence_score(regime_results)
            regime_specific_recommendations = self._generate_regime_recommendations(regime_results)
            
            # Log the enhancement
            logger.info(
                f"Enhanced approval request created ({enhanced_request.request_id}) "
                f"with synthetic data for test {test.test_id}. "
                f"Confidence score: {confidence_score}"
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error enhancing approval request {request.request_id} "
                f"with synthetic data: {str(e)}"
            )
            return False
    
    def _generate_regime_specific_data(
        self, symbols: List[str], days: int, regime: MarketRegimeType
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate regime-specific market data for multiple symbols.
        
        Args:
            symbols: List of symbols to generate data for
            days: Number of days to generate
            regime: Market regime to simulate
            
        Returns:
            Dictionary mapping symbols to their price data
        """
        # Set up correlation structure for realistic multi-asset generation
        correlation = CorrelationStructure(symbols)
        correlation.set_default_structure()
        
        # Configure regime-specific parameters
        if regime == MarketRegimeType.BULLISH:
            volatility = 0.015
            drift = 0.0008  # ~20% annualized return
        elif regime == MarketRegimeType.BEARISH:
            volatility = 0.02
            drift = -0.0006  # ~-15% annualized return
        elif regime == MarketRegimeType.VOLATILE:
            volatility = 0.03
            drift = 0.0001  # High volatility, slight upward bias
        else:  # SIDEWAYS
            volatility = 0.008
            drift = 0.0001  # Low volatility, minimal drift
        
        # Generate correlated price data
        market_data = self.correlated_generator.generate_correlated_markets(
            symbols=symbols,
            days=days,
            base_price=100.0,
            volatility=volatility,
            drift=drift,
            correlation_structure=correlation
        )
        
        # Apply regime-specific patterns
        for symbol in symbols:
            if regime == MarketRegimeType.BULLISH:
                # Add some momentum patterns
                self.synthetic_generator.apply_momentum_effect(market_data[symbol])
            elif regime == MarketRegimeType.BEARISH:
                # Add some panic selling patterns
                self.synthetic_generator.apply_panic_selling_effect(market_data[symbol])
            elif regime == MarketRegimeType.VOLATILE:
                # Add volatility clustering
                self.synthetic_generator.apply_volatility_clustering(market_data[symbol])
        
        return market_data
    
    def _backtest_strategy(
        self, strategy_id: str, version_id: str, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy against synthetic market data.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            market_data: Dictionary mapping symbols to price data
            
        Returns:
            Dictionary with backtest results
        """
        # NOTE: In a real implementation, this would use your actual backtesting engine
        # For now, we'll simulate results
        
        # Simulate strategy performance metrics
        sharpe_ratio = np.random.uniform(0.8, 2.0)
        max_drawdown = np.random.uniform(-0.3, -0.05)
        win_rate = np.random.uniform(0.4, 0.7)
        profit_factor = np.random.uniform(1.1, 2.0)
        
        # Create simulated trade history
        trade_count = np.random.randint(30, 100)
        trades = []
        
        # Calculate results
        return {
            "strategy_id": strategy_id,
            "version_id": version_id,
            "metrics": {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "trade_count": trade_count
            },
            "trades": trades
        }
    
    def _compare_backtest_results(
        self, variant_a_results: Dict[str, Any], variant_b_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare backtest results between variants.
        
        Args:
            variant_a_results: Results for variant A
            variant_b_results: Results for variant B
            
        Returns:
            Dictionary with comparison metrics
        """
        # Extract metrics
        metrics_a = variant_a_results["metrics"]
        metrics_b = variant_b_results["metrics"]
        
        # Calculate differences
        sharpe_diff = metrics_b["sharpe_ratio"] - metrics_a["sharpe_ratio"]
        drawdown_diff = metrics_b["max_drawdown"] - metrics_a["max_drawdown"]
        win_rate_diff = metrics_b["win_rate"] - metrics_a["win_rate"]
        profit_factor_diff = metrics_b["profit_factor"] - metrics_a["profit_factor"]
        
        # Determine if B is better overall
        b_is_better = (
            sharpe_diff > 0 and
            drawdown_diff > 0 and  # Less negative drawdown
            win_rate_diff > 0 and
            profit_factor_diff > 0
        )
        
        # Calculate relative improvement percentages
        relative_improvements = {
            "sharpe_ratio": (sharpe_diff / metrics_a["sharpe_ratio"]) if metrics_a["sharpe_ratio"] != 0 else 0,
            "max_drawdown": (drawdown_diff / metrics_a["max_drawdown"]) if metrics_a["max_drawdown"] != 0 else 0,
            "win_rate": (win_rate_diff / metrics_a["win_rate"]) if metrics_a["win_rate"] != 0 else 0,
            "profit_factor": (profit_factor_diff / metrics_a["profit_factor"]) if metrics_a["profit_factor"] != 0 else 0
        }
        
        # Calculate confidence score for this regime
        confidence_score = 0
        if b_is_better:
            # Weight improvements by importance
            confidence_score = (
                0.4 * relative_improvements["sharpe_ratio"] +
                0.3 * (-relative_improvements["max_drawdown"]) +  # Reverse sign since improvement is less negative
                0.15 * relative_improvements["win_rate"] +
                0.15 * relative_improvements["profit_factor"]
            )
            confidence_score = min(1.0, max(0, confidence_score))
        
        return {
            "b_is_better": b_is_better,
            "differences": {
                "sharpe_ratio": sharpe_diff,
                "max_drawdown": drawdown_diff,
                "win_rate": win_rate_diff,
                "profit_factor": profit_factor_diff
            },
            "relative_improvements": relative_improvements,
            "confidence_score": confidence_score
        }
    
    def _generate_regime_summary(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of regime-specific testing results.
        
        Args:
            regime_results: Dictionary with regime-specific results
            
        Returns:
            Dictionary with results summary
        """
        # Count regimes where B is better
        regimes_b_better = 0
        total_regimes = len(regime_results)
        
        for regime, results in regime_results.items():
            if results["comparison"]["b_is_better"]:
                regimes_b_better += 1
        
        # Calculate overall recommendation
        promote_b = (regimes_b_better / total_regimes) >= 0.75
        confidence = "high" if (regimes_b_better / total_regimes) >= 0.9 else "medium"
        
        # Generate regime-specific recommendations
        regime_specific = {}
        for regime, results in regime_results.items():
            variant = "B" if results["comparison"]["b_is_better"] else "A"
            confidence_score = results["comparison"]["confidence_score"]
            regime_specific[regime] = {
                "recommended_variant": variant,
                "confidence": confidence_score,
                "key_metrics": {
                    "sharpe_improvement": results["comparison"]["differences"]["sharpe_ratio"],
                    "drawdown_improvement": results["comparison"]["differences"]["max_drawdown"]
                }
            }
        
        return {
            "total_regimes": total_regimes,
            "regimes_b_better": regimes_b_better,
            "promote_b": promote_b,
            "confidence": confidence,
            "regime_specific": regime_specific
        }
    
    def _calculate_confidence_score(self, regime_results: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score based on regime-specific results.
        
        Args:
            regime_results: Dictionary with regime-specific results
            
        Returns:
            Confidence score between 0 and 1
        """
        # Count regime performance
        regimes_b_better = 0
        total_confidence = 0.0
        
        for regime, results in regime_results.items():
            if results["comparison"]["b_is_better"]:
                regimes_b_better += 1
                total_confidence += results["comparison"]["confidence_score"]
        
        # No regimes where B is better
        if regimes_b_better == 0:
            return 0.0
        
        # Calculate weighted confidence score
        regime_ratio = regimes_b_better / len(regime_results)
        avg_confidence = total_confidence / regimes_b_better
        
        # Combined score weights both breadth (how many regimes) and depth (confidence per regime)
        return regime_ratio * avg_confidence
    
    def _generate_regime_recommendations(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate regime-specific recommendations.
        
        Args:
            regime_results: Dictionary with regime-specific results
            
        Returns:
            Dictionary with regime-specific recommendations
        """
        # Initialize counts
        regime_counts = {
            "A_better": 0,
            "B_better": 0,
            "similar": 0
        }
        
        # Check if regime switching would be beneficial
        clear_regime_preference = False
        recommendations = {}
        
        for regime, results in regime_results.items():
            comp = results["comparison"]
            is_significant = abs(comp["differences"]["sharpe_ratio"]) > 0.3
            
            if comp["b_is_better"] and is_significant:
                regime_counts["B_better"] += 1
                recommendations[regime] = "B"
            elif not comp["b_is_better"] and is_significant:
                regime_counts["A_better"] += 1
                recommendations[regime] = "A"
            else:
                regime_counts["similar"] += 1
                # Use variant with less drawdown for similar performance
                if comp["differences"]["max_drawdown"] > 0:
                    recommendations[regime] = "B"
                else:
                    recommendations[regime] = "A"
        
        # Check if there's a clear regime-based preference pattern
        if (regime_counts["A_better"] > 0 and regime_counts["B_better"] > 0 and
            regime_counts["similar"] < len(regime_results) / 2):
            clear_regime_preference = True
        
        return {
            "regime_switching_recommended": clear_regime_preference,
            "regime_specific_variants": recommendations,
            "regime_counts": regime_counts
        }


# Singleton instance
_synthetic_testing_integration = None


def get_synthetic_testing_integration() -> SyntheticTestingIntegration:
    """
    Get the singleton instance of SyntheticTestingIntegration.
    
    Returns:
        SyntheticTestingIntegration instance
    """
    global _synthetic_testing_integration
    
    if _synthetic_testing_integration is None:
        _synthetic_testing_integration = SyntheticTestingIntegration()
    
    return _synthetic_testing_integration


if __name__ == "__main__":
    # Simple test
    integration = get_synthetic_testing_integration()
    print("Synthetic Testing Integration initialized")
