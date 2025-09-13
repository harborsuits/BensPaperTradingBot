#!/usr/bin/env python3
"""
A/B Testing Integration

This module integrates the A/B Testing Framework with the event system, strategy lifecycle
manager, and autonomous engine. It provides a seamless connection between all components
of our trading system, enabling automated test creation, execution, and result application.

Classes:
    ABTestingIntegration: Connects A/B testing with other system components
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

# Import A/B testing components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)
from trading_bot.autonomous.ab_testing_manager import (
    get_ab_test_manager, ABTestEventType
)
from trading_bot.autonomous.ab_testing_analysis import (
    get_ab_test_analyzer
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Import strategy lifecycle components
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_lifecycle_manager, StrategyStatus, VersionStatus
)

# Import autonomous engine components
from trading_bot.autonomous.autonomous_engine import (
    get_autonomous_engine
)

# Import approval workflow components
from trading_bot.autonomous.approval_workflow import (
    get_approval_workflow_manager, ApprovalStatus
)

# Import optimization components
try:
    from trading_bot.autonomous.optimization_integration import (
        get_optimization_integration
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestingIntegration:
    """
    Integrates A/B Testing with other system components.
    
    This class provides the connection layer between the A/B Testing Framework
    and other system components, including:
    
    - Strategy Lifecycle Manager
    - Autonomous Engine
    - Event System
    - Optimization System (if available)
    
    It enables:
    - Automatic A/B test creation from optimization results
    - Automatic promotion of successful variants
    - Integration with the event system for observability
    - Regime-aware test analysis and application
    """
    
    def __init__(self):
        """Initialize the A/B testing integration."""
        # Get component instances
        self.ab_test_manager = get_ab_test_manager()
        self.ab_test_analyzer = get_ab_test_analyzer()
        self.event_bus = EventBus()
        self.lifecycle_manager = get_lifecycle_manager()
        self.autonomous_engine = get_autonomous_engine()
        self.approval_manager = get_approval_workflow_manager()
        
        # Initialize optimization integration if available
        self.optimization_integration = None
        if OPTIMIZATION_AVAILABLE:
            self.optimization_integration = get_optimization_integration()
        
        # Configuration
        self.auto_create_approval_requests = True
        self.auto_analyze_tests = True
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start A/B test manager
        self.ab_test_manager.start()
        
        logger.info("A/B Testing Integration initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        # Register handlers for A/B testing events
        self.event_bus.register(
            ABTestEventType.TEST_COMPLETED,
            self._handle_test_completed
        )
        
        self.event_bus.register(
            ABTestEventType.VARIANT_PROMOTED,
            self._handle_variant_promoted
        )
        
        # Register handlers for lifecycle events
        self.event_bus.register(
            EventType.STRATEGY_VERSION_CREATED,
            self._handle_version_created
        )
        
        self.event_bus.register(
            EventType.STRATEGY_DEPLOYED,
            self._handle_strategy_deployed
        )
        
        # Register handlers for approval workflow events
        self.event_bus.register(
            EventType.APPROVAL_REQUEST_APPROVED,
            self._handle_approval_approved
        )
        
        self.event_bus.register(
            EventType.APPROVAL_REQUEST_REJECTED,
            self._handle_approval_rejected
        )
        
        # Register handlers for optimization events
        if OPTIMIZATION_AVAILABLE:
            self.event_bus.register(
                "optimization_completed",
                self._handle_optimization_completed
            )
    
    def _handle_test_completed(self, event: Event):
        """
        Handle test completed events.
        
        Args:
            event: Test completed event
        """
        test_id = event.data.get("test_id")
        if not test_id:
            logger.error("Received test_completed event without test_id")
            return
            
        # Retrieve the completed test
        test = self.ab_test_manager.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return
            
        # Analyze the test
        try:
            analysis = self.ab_test_analyzer.analyze_test(test)
            
            # Extract recommendation
            recommendation = analysis.get("recommendation", {})
            promote_b = recommendation.get("promote_variant_b", False)
            confidence = recommendation.get("confidence", "low")
            explanation = recommendation.get("explanation", "No detailed explanation provided")
            
            # Log analysis results
            logger.info(
                f"Test {test_id} completed. Recommendation: "
                f"promote variant B: {promote_b} ({confidence} confidence)"
            )
            
            # Emit analysis event
            self.event_bus.emit(
                Event(
                    event_type="test_analysis_completed",
                    data={
                        "test_id": test_id,
                        "recommendation": recommendation,
                        "analysis": analysis
                    },
                    source="ab_testing_integration"
                )
            )
            
            # Create approval request if recommendation is to promote
            if promote_b and self.auto_create_approval_requests:
                logger.info(f"Creating approval request for test {test_id}")
                self._create_approval_request(test, explanation, confidence)
            # Auto-promote only for very low impact changes with very high confidence
            elif promote_b and confidence == "very_high" and self._is_low_impact_change(test):
                logger.info(f"Auto-promoting low-impact variant B for test {test_id}")
                self._promote_variant(test, test.variant_b)
                
            # Apply regime-specific recommendations if available
            regime_switching = recommendation.get("regime_switching", {})
            if regime_switching.get("recommended", False):
                # Create regime-switching approval request
                if self.auto_create_approval_requests:
                    logger.info(f"Creating regime-switching approval request for test {test_id}")
                    self._create_regime_approval_request(test, regime_switching, recommendation)
                
        except Exception as e:
            logger.error(f"Error analyzing test {test_id}: {str(e)}")
    
    def _handle_variant_promoted(self, event: Event):
        """
        Handle variant promoted events.
        
        Args:
            event: Variant promoted event
        """
        data = event.data
        if not data:
            return
            
        test_id = data.get("test_id")
        variant_id = data.get("variant_id")
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        
        if not test_id or not variant_id or not strategy_id or not version_id:
            return
            
        # Notify lifecycle manager to promote this version
        logger.info(
            f"Promoting version {version_id} of strategy {strategy_id} "
            f"from test {test_id}"
        )
        
        # In a production system, we might want to have an approval workflow here
        # For now, we'll directly promote the version
        try:
            self.lifecycle_manager.promote_version(
                strategy_id, version_id, 
                reason=f"Promoted from A/B test {test_id}"
            )
            
            logger.info(
                f"Successfully promoted version {version_id} of strategy {strategy_id}"
            )
            
        except Exception as e:
            logger.error(
                f"Error promoting version {version_id} of strategy {strategy_id}: {str(e)}"
            )
    
    def _handle_version_created(self, event: Event):
        """
        Handle strategy version created events.
        
        This can automatically create A/B tests for new versions.
        
        Args:
            event: Version created event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        source = data.get("source")
        
        if not strategy_id or not version_id:
            return
            
        # If this is not from an optimization or other known source, we might
        # want to create an A/B test against the current production version
        if source != "optimization" and source != "abtest":
            # Get current production version
            production_version = self.lifecycle_manager.get_production_version(strategy_id)
            
            if production_version:
                # Create an A/B test
                self._create_test_for_versions(
                    strategy_id,
                    production_version.version_id,
                    version_id,
                    description=f"Testing new version {version_id} against production version"
                )
    
    def _handle_strategy_deployed(self, event: Event):
        """
        Handle strategy deployed events.
        
        Args:
            event: Strategy deployed event
        """
        # When a strategy is deployed, we might want to create A/B tests
        # against similar strategies or baseline strategies
        pass
    
    def _handle_optimization_completed(self, event: Event):
        """
        Handle optimization completed events.
        
        Args:
            event: Optimization completed event
        """
        data = event.data
        if not data:
            return
            
        # The AB test manager already handles this event directly,
        # so we don't need to duplicate that logic here
        pass
    
    def _handle_approval_approved(self, event: Event):
        """
        Handle approval request approved events.
        
        Args:
            event: Approval approved event
        """
        request_id = event.data.get("request_id")
        test_id = event.data.get("test_id")
        strategy_id = event.data.get("strategy_id")
        version_id = event.data.get("version_id")
        reviewer = event.data.get("reviewer")
        
        if not all([request_id, test_id, strategy_id, version_id]):
            logger.error("Received incomplete approval event data")
            return
        
        logger.info(f"Processing approved request {request_id} by {reviewer}")
        
        # Get the test
        test = self.ab_test_manager.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found for approval request {request_id}")
            return
        
        # Determine which variant to promote based on version_id
        if test.variant_b and test.variant_b.version_id == version_id:
            logger.info(f"Promoting variant B (version {version_id}) for test {test_id}")
            self._promote_variant(test, test.variant_b)
        elif test.variant_a and test.variant_a.version_id == version_id:
            logger.info(f"Promoting variant A (version {version_id}) for test {test_id}")
            self._promote_variant(test, test.variant_a)
        else:
            logger.error(f"Version {version_id} not found in test {test_id}")
    
    def _handle_approval_rejected(self, event: Event):
        """
        Handle approval request rejected events.
        
        Args:
            event: Approval rejected event
        """
        request_id = event.data.get("request_id")
        test_id = event.data.get("test_id")
        reviewer = event.data.get("reviewer")
        comments = event.data.get("comments")
        
        logger.info(
            f"Approval request {request_id} for test {test_id} was rejected by {reviewer}"
            + (f": {comments}" if comments else "")
        )
        
        # Mark the test as reviewed but not promoted
        test = self.ab_test_manager.get_test(test_id)
        if test:
            self.event_bus.emit(
                Event(
                    event_type="test_promotion_rejected",
                    data={
                        "test_id": test_id,
                        "reviewer": reviewer,
                        "comments": comments
                    },
                    source="ab_testing_integration"
                )
            )
    
    def _create_approval_request(self, test: ABTest, explanation: str, confidence: str):
        """
        Create an approval request for promoting a test variant.
        
        Args:
            test: The A/B test
            explanation: Analysis explanation
            confidence: Confidence level of the recommendation
        
        Returns:
            Created approval request or None if error
        """
        try:
            # Format a detailed description for the reviewer
            description = f"Test {test.name} ({test.test_id}) recommends promoting variant B ({test.variant_b.name}).\n"
            description += f"Confidence: {confidence}\n"
            description += f"Explanation: {explanation}\n"
            description += f"Metrics: {json.dumps(test.variant_b.metrics, indent=2)}\n"
            
            # Create the approval request
            request = self.approval_manager.create_request(
                test_id=test.test_id,
                strategy_id=test.variant_b.strategy_id,
                version_id=test.variant_b.version_id,
                requester="ab_testing_system"
            )
            
            logger.info(f"Created approval request {request.request_id} for test {test.test_id}")
            return request
            
        except Exception as e:
            logger.error(f"Error creating approval request for test {test.test_id}: {str(e)}")
            return None
    
    def _create_regime_approval_request(self, test: ABTest, regime_switching: dict, recommendation: dict):
        """
        Create an approval request for regime-switching strategy.
        
        Args:
            test: The A/B test
            regime_switching: Regime switching recommendation
            recommendation: Overall recommendation
        
        Returns:
            Created approval request or None if error
        """
        try:
            # Format a detailed description
            description = f"Test {test.name} ({test.test_id}) recommends a regime-switching approach.\n"
            description += f"Explanation: {regime_switching.get('explanation', 'No explanation provided')}\n"
            
            # Add details about each regime
            regimes = recommendation.get("regime_specific", {})
            for regime, details in regimes.items():
                description += f"\nRegime: {regime}\n"
                description += f"Recommended variant: {details.get('recommended_variant', 'Unknown')}\n"
                description += f"Confidence: {details.get('confidence', 'Unknown')}\n"
            
            # Create the approval request - we'll use variant_a's strategy_id but a special version_id
            request = self.approval_manager.create_request(
                test_id=test.test_id,
                strategy_id=test.variant_a.strategy_id,
                version_id="regime_switching",  # Special marker
                requester="ab_testing_system"
            )
            
            logger.info(f"Created regime-switching approval request {request.request_id} for test {test.test_id}")
            return request
            
        except Exception as e:
            logger.error(f"Error creating regime approval request for test {test.test_id}: {str(e)}")
            return None
    
    def _is_low_impact_change(self, test: ABTest) -> bool:
        """
        Determine if a test represents a low-impact change that could potentially
        be auto-promoted without human review.
        
        Args:
            test: The A/B test to evaluate
        
        Returns:
            True if the change is considered low impact, False otherwise
        """
        # Check if this test has the low_impact tag in metadata
        if test.metadata.get("low_impact", False):
            return True
        
        # By default, require human review for safety
        return False
    
    def _promote_variant(self, test: ABTest, variant: TestVariant):
        """
        Promote a variant to production.
        
        Args:
            test: A/B test
            variant: Variant to promote
        """
        try:
            # Get version details
            strategy_id = variant.strategy_id
            version_id = variant.version_id
            
            # Check if version exists
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
            if not version:
                logger.error(
                    f"Version {version_id} of strategy {strategy_id} not found, "
                    f"cannot promote"
                )
                return
                
            # Check version status
            if version.status == VersionStatus.PRODUCTION:
                logger.info(
                    f"Version {version_id} of strategy {strategy_id} is already "
                    f"in production"
                )
                return
                
            # Promote the version
            self.lifecycle_manager.promote_version(
                strategy_id, version_id, 
                reason=f"Promoted from A/B test {test.test_id}"
            )
            
            logger.info(
                f"Promoted version {version_id} of strategy {strategy_id} "
                f"from test {test.test_id}"
            )
            
        except Exception as e:
            logger.error(
                f"Error promoting variant {variant.variant_id} "
                f"from test {test.test_id}: {str(e)}"
            )
    
    def _apply_regime_switching(
        self,
        test: ABTest,
        regime_recommendations: Dict[str, Dict[str, Any]]
    ):
        """
        Apply regime-specific strategy selection.
        
        Args:
            test: A/B test
            regime_recommendations: Regime-specific recommendations
        """
        try:
            strategy_id = test.variant_a.strategy_id
            
            # Convert recommendations to regime map
            regime_map = {}
            
            for regime, rec in regime_recommendations.items():
                use_variant_b = rec.get("use_variant_b", False)
                version_id = (
                    test.variant_b.version_id if use_variant_b
                    else test.variant_a.version_id
                )
                
                regime_map[regime] = version_id
            
            # Register regime-specific version selection with the engine or lifecycle manager
            logger.info(
                f"Registering regime-specific versions for strategy {strategy_id}: "
                f"{regime_map}"
            )
            
            # This assumes the autonomous engine has regime-aware strategy selection
            # In a production system, we would implement this integration
            # For now, we'll just log it
            
        except Exception as e:
            logger.error(f"Error applying regime switching: {str(e)}")
    
    def _create_test_for_versions(
        self,
        strategy_id: str,
        version_a_id: str,
        version_b_id: str,
        description: str = ""
    ) -> Optional[ABTest]:
        """
        Create an A/B test for two versions of a strategy.
        
        Args:
            strategy_id: Strategy ID
            version_a_id: Version A ID
            version_b_id: Version B ID
            description: Test description
            
        Returns:
            Created ABTest or None if error
        """
        try:
            # Get version details
            version_a = self.lifecycle_manager.get_version(strategy_id, version_a_id)
            version_b = self.lifecycle_manager.get_version(strategy_id, version_b_id)
            
            if not version_a or not version_b:
                logger.error(
                    f"Versions not found for strategy {strategy_id}: "
                    f"A={version_a_id}, B={version_b_id}"
                )
                return None
                
            # Create variants
            variant_a = TestVariant(
                strategy_id=strategy_id,
                version_id=version_a_id,
                name=f"{version_a.name or 'Version A'}",
                parameters=version_a.parameters or {}
            )
            
            variant_b = TestVariant(
                strategy_id=strategy_id,
                version_id=version_b_id,
                name=f"{version_b.name or 'Version B'}",
                parameters=version_b.parameters or {}
            )
            
            # Test configuration
            config = {
                'duration_days': 30,
                'confidence_level': 0.95,
                'metrics_to_compare': [
                    'sharpe_ratio', 'sortino_ratio', 'win_rate', 'max_drawdown',
                    'profit_factor', 'annualized_return', 'volatility'
                ],
                'auto_promote_threshold': 0.1,
                'min_trade_count': 30
            }
            
            # Create test
            test = self.ab_test_manager.create_test(
                name=f"Version Comparison: {strategy_id}",
                variant_a=variant_a,
                variant_b=variant_b,
                config=config,
                description=description,
                metadata={
                    "source": "version_comparison",
                    "strategy_id": strategy_id
                }
            )
            
            # Start test
            self.ab_test_manager.start_test(test.test_id)
            
            logger.info(
                f"Created and started A/B test {test.test_id} comparing "
                f"versions {version_a_id} and {version_b_id} of strategy {strategy_id}"
            )
            
            return test
            
        except Exception as e:
            logger.error(
                f"Error creating test for versions {version_a_id} and {version_b_id} "
                f"of strategy {strategy_id}: {str(e)}"
            )
            return None
    
    def create_test_from_optimization(
        self,
        strategy_id: str,
        original_version_id: str,
        new_parameters: Dict[str, Any],
        job_id: str = "",
        auto_start: bool = True
    ) -> Optional[ABTest]:
        """
        Create A/B test from optimization results.
        
        Args:
            strategy_id: Strategy ID
            original_version_id: Original version ID
            new_parameters: New optimized parameters
            job_id: Optimization job ID
            auto_start: Whether to auto-start the test
            
        Returns:
            Created ABTest or None if error
        """
        try:
            # Get original version
            original_version = self.lifecycle_manager.get_version(
                strategy_id, original_version_id
            )
            
            if not original_version:
                logger.error(
                    f"Original version {original_version_id} of strategy {strategy_id} "
                    f"not found"
                )
                return None
                
            # Create a new version with optimized parameters
            new_version_id = f"{original_version_id}.opt"
            if job_id:
                new_version_id = f"{original_version_id}.opt-{job_id[:8]}"
                
            # In a production system, we would actually create the version
            # For now, we'll simulate it
            
            # Create variants
            variant_a = TestVariant(
                strategy_id=strategy_id,
                version_id=original_version_id,
                name="Original",
                parameters=original_version.parameters or {}
            )
            
            variant_b = TestVariant(
                strategy_id=strategy_id,
                version_id=new_version_id,
                name="Optimized",
                parameters=new_parameters
            )
            
            # Test configuration
            config = {
                'duration_days': 30,
                'confidence_level': 0.95,
                'metrics_to_compare': [
                    'sharpe_ratio', 'sortino_ratio', 'win_rate', 'max_drawdown',
                    'profit_factor', 'annualized_return', 'volatility'
                ],
                'auto_promote_threshold': 0.1,
                'min_trade_count': 30
            }
            
            # Create test
            test = self.ab_test_manager.create_test(
                name=f"Optimization Test: {strategy_id}",
                variant_a=variant_a,
                variant_b=variant_b,
                config=config,
                description=f"Testing optimization results for {strategy_id}",
                metadata={
                    "source": "optimization",
                    "job_id": job_id,
                    "strategy_id": strategy_id
                }
            )
            
            # Start test if auto_start is True
            if auto_start:
                self.ab_test_manager.start_test(test.test_id)
                
                logger.info(
                    f"Created and started A/B test {test.test_id} for optimized "
                    f"parameters of strategy {strategy_id}"
                )
            else:
                logger.info(
                    f"Created A/B test {test.test_id} for optimized "
                    f"parameters of strategy {strategy_id} (not started)"
                )
            
            return test
            
        except Exception as e:
            logger.error(
                f"Error creating test from optimization for strategy {strategy_id}: {str(e)}"
            )
            return None
            
    def analyze_and_apply_test_results(self, test_id: str) -> bool:
        """
        Analyze and apply the results of an A/B test.
        
        Args:
            test_id: ID of test to analyze and apply
            
        Returns:
            True if successful, False otherwise
        """
        test = self.ab_test_manager.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return False
            
        try:
            # Complete the test if it's not already completed
            if test.status == TestStatus.RUNNING:
                self.ab_test_manager.complete_test(test_id)
                
            # If the test is already completed or is now completed, analyze it
            if test.status == TestStatus.COMPLETED:
                # Analyze the test
                analysis = self.ab_test_analyzer.analyze_test(test)
                
                # Check if we should promote variant B
                recommendation = analysis.get("recommendation", {})
                promote_b = recommendation.get("promote_variant_b", False)
                
                if promote_b:
                    # Promote variant B
                    self._promote_variant(test, test.variant_b)
                    return True
                    
                # Check for regime-specific recommendations
                regime_switching = recommendation.get("regime_switching", {})
                if regime_switching.get("recommended", False):
                    # Apply regime-specific strategy selection
                    self._apply_regime_switching(
                        test, recommendation.get("regime_specific", {})
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error analyzing and applying test {test_id}: {str(e)}")
            return False


# Singleton instance
_ab_testing_integration = None


def get_ab_testing_integration() -> ABTestingIntegration:
    """
    Get singleton instance of ABTestingIntegration.
    
    Returns:
        ABTestingIntegration instance
    """
    global _ab_testing_integration
    
    if _ab_testing_integration is None:
        _ab_testing_integration = ABTestingIntegration()
    
    return _ab_testing_integration


if __name__ == "__main__":
    # Example usage
    integration = get_ab_testing_integration()
    print("A/B Testing Integration initialized")
