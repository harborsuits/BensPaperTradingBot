#!/usr/bin/env python3
"""
A/B Testing Manager

This module provides the management layer for A/B tests, handling test creation,
scheduling, persistence, and event integration. It follows our established
patterns of singleton access, event-driven architecture, and persistence.

Classes:
    ABTestManager: Singleton class managing all A/B tests
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import time

# Import A/B testing core
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Import lifecycle management
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_lifecycle_manager, StrategyStatus
)

# Import correlation components (optional, for regime detection)
try:
    from trading_bot.autonomous.correlation_regime_detector import (
        get_correlation_regime_detector, RegimeType
    )
    CORRELATION_AVAILABLE = True
except ImportError:
    CORRELATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define A/B testing event types
class ABTestEventType(str):
    """Event types for A/B testing."""
    TEST_CREATED = "abtest_created"
    TEST_STARTED = "abtest_started"
    TEST_COMPLETED = "abtest_completed"
    TEST_STOPPED = "abtest_stopped"
    TEST_FAILED = "abtest_failed"
    VARIANT_UPDATED = "abtest_variant_updated"
    VARIANT_PROMOTED = "abtest_variant_promoted"


class ABTestManager:
    """
    Manages A/B tests throughout their lifecycle.
    
    This class is responsible for:
    - Creating and configuring A/B tests
    - Persisting test configurations and results
    - Scheduling and executing tests
    - Processing test results and promotion decisions
    - Integrating with the event system for monitoring
    
    It follows the singleton pattern for global access.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the A/B test manager.
        
        Args:
            storage_path: Path to store A/B test data
        """
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".trading_bot", "abtests"
        )
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Current tests
        self.tests: Dict[str, ABTest] = {}
        self.lock = threading.RLock()
        
        # Event system integration
        self.event_bus = EventBus()
        
        # Lifecycle manager integration
        self.lifecycle_manager = get_lifecycle_manager()
        
        # Regime detector integration (optional)
        self.regime_detector = None
        if CORRELATION_AVAILABLE:
            self.regime_detector = get_correlation_regime_detector()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Load existing tests
        self._load_tests()
        
        # Thread control
        self.running = False
        self.scheduler_thread = None
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        # Strategy performance events
        self.event_bus.register(
            EventType.STRATEGY_PERFORMANCE_UPDATED,
            self._handle_performance_updated
        )
        
        # Strategy deployment events
        self.event_bus.register(
            EventType.STRATEGY_DEPLOYED,
            self._handle_strategy_deployed
        )
        
        # Optimization events
        self.event_bus.register(
            "optimization_completed",
            self._handle_optimization_completed
        )
        
        # Market regime events
        if CORRELATION_AVAILABLE:
            self.event_bus.register(
                "regime_changed",
                self._handle_regime_changed
            )
    
    def _load_tests(self):
        """Load tests from disk."""
        test_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
        
        with self.lock:
            for filename in test_files:
                try:
                    filepath = os.path.join(self.storage_path, filename)
                    with open(filepath, 'r') as f:
                        test_data = json.load(f)
                    
                    test = ABTest.from_dict(test_data)
                    self.tests[test.test_id] = test
                    
                except Exception as e:
                    logger.error(f"Error loading test from {filename}: {str(e)}")
        
        logger.info(f"Loaded {len(self.tests)} A/B tests from {self.storage_path}")
    
    def _save_test(self, test: ABTest):
        """
        Save test to disk.
        
        Args:
            test: Test to save
        """
        filepath = os.path.join(self.storage_path, f"{test.test_id}.json")
        
        try:
            # Write to temporary file first
            temp_filepath = f"{filepath}.tmp"
            with open(temp_filepath, 'w') as f:
                json.dump(test.to_dict(), f, indent=2)
            
            # Atomic replace
            os.replace(temp_filepath, filepath)
            
        except Exception as e:
            logger.error(f"Error saving test {test.test_id}: {str(e)}")
    
    def _handle_performance_updated(self, event: Event):
        """
        Handle strategy performance update events.
        
        Args:
            event: Performance update event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        performance = data.get("performance", {})
        
        if not strategy_id or not version_id or not performance:
            return
            
        # Find tests that include this strategy/version
        with self.lock:
            for test in self.tests.values():
                # Skip tests that aren't running
                if test.status != TestStatus.RUNNING:
                    continue
                    
                # Check variant A
                if (test.variant_a.strategy_id == strategy_id and
                    test.variant_a.version_id == version_id):
                    test.variant_a.update_metrics(performance)
                    self._save_test(test)
                    self._emit_variant_updated_event(test, test.variant_a)
                
                # Check variant B
                if (test.variant_b.strategy_id == strategy_id and
                    test.variant_b.version_id == version_id):
                    test.variant_b.update_metrics(performance)
                    self._save_test(test)
                    self._emit_variant_updated_event(test, test.variant_b)
    
    def _handle_strategy_deployed(self, event: Event):
        """
        Handle strategy deployment events.
        
        Args:
            event: Strategy deployment event
        """
        # Implementation for automatically creating A/B tests
        # when new strategies are deployed
        pass
    
    def _handle_optimization_completed(self, event: Event):
        """
        Handle optimization completion events.
        
        Args:
            event: Optimization completion event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        original_version_id = data.get("version_id")
        results = data.get("results", {})
        
        if not strategy_id or not original_version_id or not results:
            return
            
        # Check if this optimization produced a new version
        new_parameters = results.get("parameters")
        if not new_parameters:
            return
            
        # Get information about original version
        original_version = self.lifecycle_manager.get_version(
            strategy_id, original_version_id
        )
        if not original_version:
            return
            
        # See if a new version was created
        # In a real system, this would look up the newly created version
        # For now, we'll simulate by creating a test manually
        
        # Create test name
        test_name = f"Optimization Test: {strategy_id}"
        
        # Create variants
        variant_a = TestVariant(
            strategy_id=strategy_id,
            version_id=original_version_id,
            name="Original",
            parameters=original_version.parameters or {}
        )
        
        # For variant B, we'd use the actual new version
        # For now, we'll create a simulated variant
        new_version_id = f"{original_version_id}.opt"
        variant_b = TestVariant(
            strategy_id=strategy_id,
            version_id=new_version_id,
            name="Optimized",
            parameters=new_parameters
        )
        
        # Create test configuration
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
        test = self.create_test(
            name=test_name,
            variant_a=variant_a,
            variant_b=variant_b,
            config=config,
            description=f"Testing optimization results for {strategy_id}",
            metadata={"source": "optimization", "job_id": data.get("job_id")}
        )
        
        # Start test
        self.start_test(test.test_id)
    
    def _handle_regime_changed(self, event: Event):
        """
        Handle market regime change events.
        
        Args:
            event: Regime change event
        """
        data = event.data
        if not data:
            return
            
        # Get new regime
        new_regime = data.get("new_regime")
        if not new_regime:
            return
            
        # Currently running tests
        with self.lock:
            running_tests = [
                test for test in self.tests.values()
                if test.status == TestStatus.RUNNING
            ]
        
        for test in running_tests:
            # Update regime information for the test
            self._update_test_regime_data(test, new_regime)
    
    def _update_test_regime_data(self, test: ABTest, regime: str):
        """
        Update regime-specific performance data for a test.
        
        Args:
            test: Test to update
            regime: Current market regime
        """
        # In a real implementation, this would retrieve regime-specific
        # performance metrics for both variants
        
        # For now, we'll just log it
        logger.info(f"Updating regime data for test {test.test_id}, regime: {regime}")
        
        # This is where we would update variant_a.regime_performance and
        # variant_b.regime_performance with regime-specific metrics
    
    def _emit_test_created_event(self, test: ABTest):
        """
        Emit test created event.
        
        Args:
            test: Created test
        """
        self.event_bus.emit(
            ABTestEventType.TEST_CREATED,
            {
                "test_id": test.test_id,
                "name": test.name,
                "variant_a": {
                    "strategy_id": test.variant_a.strategy_id,
                    "version_id": test.variant_a.version_id,
                    "name": test.variant_a.name
                },
                "variant_b": {
                    "strategy_id": test.variant_b.strategy_id,
                    "version_id": test.variant_b.version_id,
                    "name": test.variant_b.name
                },
                "config": test.config,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _emit_test_started_event(self, test: ABTest):
        """
        Emit test started event.
        
        Args:
            test: Started test
        """
        self.event_bus.emit(
            ABTestEventType.TEST_STARTED,
            {
                "test_id": test.test_id,
                "name": test.name,
                "variant_a": {
                    "strategy_id": test.variant_a.strategy_id,
                    "version_id": test.variant_a.version_id
                },
                "variant_b": {
                    "strategy_id": test.variant_b.strategy_id,
                    "version_id": test.variant_b.version_id
                },
                "timestamp": datetime.now().isoformat(),
                "duration_days": test.config.get("duration_days", 30)
            }
        )
    
    def _emit_test_completed_event(self, test: ABTest):
        """
        Emit test completed event.
        
        Args:
            test: Completed test
        """
        self.event_bus.emit(
            ABTestEventType.TEST_COMPLETED,
            {
                "test_id": test.test_id,
                "name": test.name,
                "result": {
                    "winner": test.winner,
                    "conclusion": test.conclusion,
                    "should_promote": test.should_promote_variant_b()
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _emit_variant_updated_event(self, test: ABTest, variant: TestVariant):
        """
        Emit variant updated event.
        
        Args:
            test: Test containing the variant
            variant: Updated variant
        """
        # Select key metrics to include
        key_metrics = {}
        for metric in ('sharpe_ratio', 'win_rate', 'max_drawdown'):
            if metric in variant.metrics:
                key_metrics[metric] = variant.metrics[metric]
        
        self.event_bus.emit(
            ABTestEventType.VARIANT_UPDATED,
            {
                "test_id": test.test_id,
                "variant_id": variant.variant_id,
                "strategy_id": variant.strategy_id,
                "version_id": variant.version_id,
                "name": variant.name,
                "metrics": key_metrics,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _emit_variant_promoted_event(self, test: ABTest, variant: TestVariant):
        """
        Emit variant promoted event.
        
        Args:
            test: Test containing the variant
            variant: Promoted variant
        """
        self.event_bus.emit(
            ABTestEventType.VARIANT_PROMOTED,
            {
                "test_id": test.test_id,
                "variant_id": variant.variant_id,
                "strategy_id": variant.strategy_id,
                "version_id": variant.version_id,
                "name": variant.name,
                "conclusion": test.conclusion,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def create_test(
        self,
        name: str,
        variant_a: TestVariant,
        variant_b: TestVariant,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ABTest:
        """
        Create a new A/B test.
        
        Args:
            name: Human-readable name for the test
            variant_a: Baseline variant (A)
            variant_b: Comparison variant (B)
            config: Test configuration parameters
            description: Detailed description of the test
            metadata: Additional test metadata
            
        Returns:
            Created ABTest instance
        """
        # Create the test
        test = ABTest(
            name=name,
            variant_a=variant_a,
            variant_b=variant_b,
            config=config,
            description=description,
            metadata=metadata
        )
        
        # Store the test
        with self.lock:
            self.tests[test.test_id] = test
        
        # Save to disk
        self._save_test(test)
        
        # Emit event
        self._emit_test_created_event(test)
        
        logger.info(f"Created A/B test '{name}' with ID {test.test_id}")
        return test
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """
        Get a test by ID.
        
        Args:
            test_id: Test ID to retrieve
            
        Returns:
            ABTest instance or None if not found
        """
        with self.lock:
            return self.tests.get(test_id)
    
    def get_all_tests(self) -> List[ABTest]:
        """
        Get all tests.
        
        Returns:
            List of all ABTest instances
        """
        with self.lock:
            return list(self.tests.values())
    
    def start_test(self, test_id: str) -> bool:
        """
        Start an A/B test.
        
        Args:
            test_id: ID of test to start
            
        Returns:
            True if test was started, False otherwise
        """
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return False
            
        try:
            test.start_test()
            self._save_test(test)
            self._emit_test_started_event(test)
            
            logger.info(f"Started A/B test {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting test {test_id}: {str(e)}")
            return False
    
    def stop_test(self, test_id: str, reason: str = "") -> bool:
        """
        Stop an A/B test.
        
        Args:
            test_id: ID of test to stop
            reason: Reason for stopping
            
        Returns:
            True if test was stopped, False otherwise
        """
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return False
            
        try:
            test.stop_test(reason)
            self._save_test(test)
            
            self.event_bus.emit(
                ABTestEventType.TEST_STOPPED,
                {
                    "test_id": test_id,
                    "name": test.name,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Stopped A/B test {test_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping test {test_id}: {str(e)}")
            return False
    
    def complete_test(self, test_id: str) -> bool:
        """
        Complete an A/B test and analyze results.
        
        Args:
            test_id: ID of test to complete
            
        Returns:
            True if test was completed, False otherwise
        """
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return False
            
        try:
            test.complete_test()
            
            # Save test with results
            self._save_test(test)
            
            # Emit completion event
            self._emit_test_completed_event(test)
            
            # Check if promotion is recommended
            if test.should_promote_variant_b():
                self._emit_variant_promoted_event(test, test.variant_b)
                
                # In a real implementation, we would integrate with
                # the lifecycle manager to promote the variant
                logger.info(
                    f"Variant B ({test.variant_b.name}) recommended for promotion "
                    f"from test {test_id}"
                )
            
            logger.info(f"Completed A/B test {test_id} with winner: {test.winner}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing test {test_id}: {str(e)}")
            test.fail_test(str(e))
            self._save_test(test)
            
            self.event_bus.emit(
                ABTestEventType.TEST_FAILED,
                {
                    "test_id": test_id,
                    "name": test.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return False
    
    def delete_test(self, test_id: str) -> bool:
        """
        Delete an A/B test.
        
        Args:
            test_id: ID of test to delete
            
        Returns:
            True if test was deleted, False otherwise
        """
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found")
            return False
            
        # Remove from memory
        with self.lock:
            if test_id in self.tests:
                del self.tests[test_id]
                
        # Remove from disk
        filepath = os.path.join(self.storage_path, f"{test_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error deleting test file {filepath}: {str(e)}")
                return False
        
        logger.info(f"Deleted A/B test {test_id}")
        return True
    
    def get_test_summary(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a test.
        
        Args:
            test_id: ID of test to summarize
            
        Returns:
            Test summary or None if not found
        """
        test = self.get_test(test_id)
        if not test:
            return None
            
        return {
            "test_id": test.test_id,
            "name": test.name,
            "description": test.description,
            "status": test.status.value,
            "created_at": test.created_at.isoformat(),
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "completed_at": test.completed_at.isoformat() if test.completed_at else None,
            "variant_a": {
                "name": test.variant_a.name,
                "strategy_id": test.variant_a.strategy_id,
                "version_id": test.variant_a.version_id,
                "key_metrics": {
                    k: v for k, v in test.variant_a.metrics.items()
                    if k in ('sharpe_ratio', 'win_rate', 'max_drawdown')
                }
            },
            "variant_b": {
                "name": test.variant_b.name,
                "strategy_id": test.variant_b.strategy_id,
                "version_id": test.variant_b.version_id,
                "key_metrics": {
                    k: v for k, v in test.variant_b.metrics.items()
                    if k in ('sharpe_ratio', 'win_rate', 'max_drawdown')
                }
            },
            "winner": test.winner,
            "conclusion": test.conclusion,
            "should_promote_b": test.should_promote_variant_b() if test.status == TestStatus.COMPLETED else False
        }
    
    def start(self):
        """Start the test manager scheduler."""
        with self.lock:
            if self.running:
                logger.warning("A/B Test Manager is already running")
                return
                
            self.running = True
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="ABTestScheduler"
            )
            self.scheduler_thread.start()
            
        logger.info("A/B Test Manager started")
    
    def stop(self):
        """Stop the test manager scheduler."""
        with self.lock:
            self.running = False
            
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
            
        logger.info("A/B Test Manager stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop that checks test status."""
        while self.running:
            try:
                # Check for tests that are due for completion
                self._check_test_completion()
                
            except Exception as e:
                logger.error(f"Error in A/B test scheduler: {str(e)}")
                
            # Sleep before next check
            time.sleep(60)  # Check every minute
    
    def _check_test_completion(self):
        """Check for tests that are due for completion."""
        now = datetime.now()
        
        # Get running tests
        with self.lock:
            running_tests = [
                test for test in self.tests.values()
                if test.status == TestStatus.RUNNING
            ]
        
        for test in running_tests:
            # Skip if no start date
            if not test.started_at:
                continue
                
            # Check if test duration has elapsed
            duration_days = test.config.get('duration_days', 30)
            end_date = test.started_at + timedelta(days=duration_days)
            
            if now >= end_date:
                # Test is due for completion
                logger.info(f"Test {test.test_id} has reached its duration and will be completed")
                self.complete_test(test.test_id)


# Singleton instance
_ab_test_manager = None


def get_ab_test_manager() -> ABTestManager:
    """
    Get singleton instance of ABTestManager.
    
    Returns:
        ABTestManager instance
    """
    global _ab_test_manager
    
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    
    return _ab_test_manager


if __name__ == "__main__":
    # Example usage
    manager = get_ab_test_manager()
    
    # Start the manager
    manager.start()
    
    try:
        # Wait for manager to run
        while True:
            time.sleep(10)
            tests = manager.get_all_tests()
            print(f"Active tests: {len(tests)}")
    except KeyboardInterrupt:
        # Stop on Ctrl+C
        manager.stop()
