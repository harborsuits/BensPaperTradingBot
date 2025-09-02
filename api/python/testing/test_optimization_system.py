#!/usr/bin/env python3
"""
Test Optimization System

This module contains tests for the optimization system components:
- OptimizationJob and JobStore
- OptimizationScheduler
- OptimizationHistoryTracker
- OptimizationIntegration

These tests verify the core functionality of the optimization system without
requiring external dependencies, using synthetic data and mock components.
"""

import unittest
import os
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import time
import threading

# Import optimization components
from trading_bot.autonomous.optimization_jobs import (
    OptimizationJob, OptimizationStatus, OptimizationMethod,
    OptimizationPriority, PriorityCalculator, OptimizationJobStore,
    get_job_store, create_optimization_job
)
from trading_bot.autonomous.optimization_history import (
    OptimizationHistoryTracker, OptimizationEffectiveness,
    OptimizationHistoryEntry, get_optimization_history_tracker
)
from trading_bot.autonomous.optimization_scheduler import (
    OptimizationScheduler, ResourceMonitor, OptimizationEventType,
    get_optimization_scheduler
)
from trading_bot.autonomous.optimization_integration import (
    OptimizationIntegration, get_optimization_integration
)

# Mock EventBus for testing
class MockEventBus:
    def __init__(self):
        self.handlers = {}
        self.emitted_events = []
    
    def register(self, event_type, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def emit(self, event_type, data=None):
        self.emitted_events.append((event_type, data))
        
        # Create Event
        event = MagicMock()
        event.type = event_type
        event.data = data
        
        # Call handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)
    
    def get_emitted_events(self, event_type=None):
        if event_type:
            return [e for e in self.emitted_events if e[0] == event_type]
        return self.emitted_events


# Mock LifecycleManager for testing
class MockLifecycleManager:
    def __init__(self):
        self.versions = {}
    
    def get_version(self, strategy_id, version_id):
        key = f"{strategy_id}:{version_id}"
        return self.versions.get(key)
    
    def create_version(self, strategy_id, version_id, parameters, performance, status, metadata=None):
        key = f"{strategy_id}:{version_id}"
        version = MagicMock()
        version.strategy_id = strategy_id
        version.version_id = version_id
        version.parameters = parameters
        version.performance = performance
        version.status = status
        version.metadata = metadata
        version.created_at = datetime.now()
        
        self.versions[key] = version
        return version
    
    def update_status(self, strategy_id, version_id, status):
        key = f"{strategy_id}:{version_id}"
        if key in self.versions:
            self.versions[key].status = status
            return True
        return False
    
    def get_versions(self, strategy_id):
        return [v for k, v in self.versions.items() if k.startswith(f"{strategy_id}:")]


class TestOptimizationJob(unittest.TestCase):
    """Tests for OptimizationJob and related functionality."""
    
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock datetime.now to return a fixed value
        self.now_patcher = patch('datetime.datetime')
        self.mock_datetime = self.now_patcher.start()
        self.mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0)
        self.mock_datetime.fromisoformat = datetime.fromisoformat
    
    def tearDown(self):
        # Remove the test directory
        shutil.rmtree(self.test_dir)
        
        # Stop patching
        self.now_patcher.stop()
    
    def test_job_creation(self):
        """Test creating an optimization job."""
        job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now() + timedelta(hours=1),
            priority=75,
            reason="test"
        )
        
        self.assertEqual(job.strategy_id, "test_strategy")
        self.assertEqual(job.version_id, "v1.0")
        self.assertEqual(job.priority, 75)
        self.assertEqual(job.status, OptimizationStatus.PENDING)
        self.assertEqual(job.reason, "test")
    
    def test_job_serialization(self):
        """Test job serialization and deserialization."""
        original_job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now() + timedelta(hours=1),
            priority=75,
            reason="test"
        )
        
        # Serialize
        job_dict = original_job.to_dict()
        
        # Deserialize
        loaded_job = OptimizationJob.from_dict(job_dict)
        
        # Compare
        self.assertEqual(loaded_job.job_id, original_job.job_id)
        self.assertEqual(loaded_job.strategy_id, original_job.strategy_id)
        self.assertEqual(loaded_job.version_id, original_job.version_id)
        self.assertEqual(loaded_job.priority, original_job.priority)
        self.assertEqual(loaded_job.status, original_job.status)
    
    def test_job_status_updates(self):
        """Test job status updates."""
        job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now() + timedelta(hours=1)
        )
        
        # Update status
        job.update_status(OptimizationStatus.SCHEDULED)
        self.assertEqual(job.status, OptimizationStatus.SCHEDULED)
        
        # Set results
        results = {"test": "results"}
        job.set_results(results)
        self.assertEqual(job.status, OptimizationStatus.COMPLETED)
        self.assertEqual(job.results, results)
        
        # Set error
        job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now() + timedelta(hours=1)
        )
        job.set_error("test error")
        self.assertEqual(job.status, OptimizationStatus.FAILED)
        self.assertEqual(job.error, "test error")
    
    def test_job_priority_comparison(self):
        """Test job priority comparison for priority queue."""
        high_priority = OptimizationJob(
            strategy_id="high",
            version_id="v1",
            scheduled_time=datetime.now() + timedelta(hours=2),
            priority=80
        )
        
        medium_priority = OptimizationJob(
            strategy_id="medium",
            version_id="v1",
            scheduled_time=datetime.now() + timedelta(hours=1),
            priority=50
        )
        
        low_priority = OptimizationJob(
            strategy_id="low",
            version_id="v1",
            scheduled_time=datetime.now() + timedelta(hours=1),
            priority=20
        )
        
        # Higher priority should come first
        self.assertTrue(high_priority < medium_priority)
        self.assertTrue(medium_priority < low_priority)
        
        # Same priority, earlier time should come first
        early_medium = OptimizationJob(
            strategy_id="early",
            version_id="v1",
            scheduled_time=datetime.now() + timedelta(minutes=30),
            priority=50
        )
        
        self.assertTrue(early_medium < medium_priority)
    
    def test_job_store(self):
        """Test job store operations."""
        # Create store with test directory
        store = OptimizationJobStore(self.test_dir)
        
        # Create jobs
        job1 = OptimizationJob(
            strategy_id="strategy1",
            version_id="v1",
            scheduled_time=datetime.now() + timedelta(hours=1),
            priority=50
        )
        
        job2 = OptimizationJob(
            strategy_id="strategy2",
            version_id="v1",
            scheduled_time=datetime.now() + timedelta(hours=2),
            priority=75
        )
        
        # Add jobs
        store.add_job(job1)
        store.add_job(job2)
        
        # Retrieve jobs
        self.assertEqual(len(store.get_all_jobs()), 2)
        self.assertEqual(store.get_job(job1.job_id).strategy_id, "strategy1")
        
        # Update job
        job1.update_status(OptimizationStatus.SCHEDULED)
        store.update_job(job1)
        
        updated_job = store.get_job(job1.job_id)
        self.assertEqual(updated_job.status, OptimizationStatus.SCHEDULED)
        
        # Get by status
        scheduled_jobs = store.get_jobs_by_status(OptimizationStatus.SCHEDULED)
        self.assertEqual(len(scheduled_jobs), 1)
        self.assertEqual(scheduled_jobs[0].job_id, job1.job_id)
        
        # Get by strategy
        strategy_jobs = store.get_jobs_by_strategy("strategy1")
        self.assertEqual(len(strategy_jobs), 1)
        self.assertEqual(strategy_jobs[0].job_id, job1.job_id)
        
        # Remove job
        store.remove_job(job1.job_id)
        self.assertIsNone(store.get_job(job1.job_id))
        self.assertEqual(len(store.get_all_jobs()), 1)
    
    def test_priority_calculator(self):
        """Test priority calculation based on strategy performance."""
        # Calculate priority for a poor-performing strategy
        priority1 = PriorityCalculator.calculate_priority(
            strategy_id="poor_strategy",
            strategy_type="iron_condor",
            universe="options",
            performance_metrics={
                "sharpe_ratio": 0.4,
                "max_drawdown": -0.25,
                "win_rate": 0.52
            },
            optimization_history=[],
            last_optimization=None
        )
        
        # Calculate priority for a good-performing strategy
        priority2 = PriorityCalculator.calculate_priority(
            strategy_id="good_strategy",
            strategy_type="iron_condor",
            universe="options",
            performance_metrics={
                "sharpe_ratio": 2.3,
                "max_drawdown": -0.08,
                "win_rate": 0.75
            },
            optimization_history=[],
            last_optimization=None
        )
        
        # Poor strategy should have higher priority
        self.assertGreater(priority1, priority2)
        
        # Test time-based priority
        recent_opt = PriorityCalculator.calculate_priority(
            strategy_id="recent_opt",
            strategy_type="iron_condor",
            universe="options",
            performance_metrics={
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.12
            },
            optimization_history=[],
            last_optimization=datetime.now() - timedelta(days=5)
        )
        
        old_opt = PriorityCalculator.calculate_priority(
            strategy_id="old_opt",
            strategy_type="iron_condor",
            universe="options",
            performance_metrics={
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.12
            },
            optimization_history=[],
            last_optimization=datetime.now() - timedelta(days=60)
        )
        
        # Older optimization should have higher priority
        self.assertGreater(old_opt, recent_opt)


class TestOptimizationHistory(unittest.TestCase):
    """Tests for OptimizationHistoryTracker."""
    
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create tracker with test directory
        self.tracker = OptimizationHistoryTracker(self.test_dir)
    
    def tearDown(self):
        # Remove the test directory
        shutil.rmtree(self.test_dir)
    
    def test_add_entry(self):
        """Test adding a history entry."""
        entry = self.tracker.add_entry(
            strategy_id="test_strategy",
            old_version_id="v1.0",
            new_version_id="v1.1",
            optimization_parameters={
                "method": "bayesian",
                "iterations": 100
            },
            old_metrics={
                "sharpe_ratio": 0.8,
                "max_drawdown": -0.15,
                "win_rate": 0.6
            },
            new_metrics={
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.11,
                "win_rate": 0.7
            },
            job_id="job123"
        )
        
        # Check effectiveness calculation
        self.assertEqual(entry.effectiveness, OptimizationEffectiveness.SIGNIFICANT_IMPROVEMENT)
        
        # Check improvement calculation
        self.assertAlmostEqual(entry.improvement["sharpe_ratio"], 0.5, places=2)  # 50% improvement
        self.assertAlmostEqual(entry.improvement["max_drawdown"], 0.2667, places=2)  # 26.7% improvement
        self.assertAlmostEqual(entry.improvement["win_rate"], 0.1667, places=2)  # 16.7% improvement
        
        # Check retrieval
        history = self.tracker.get_strategy_history("test_strategy")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["strategy_id"], "test_strategy")
        self.assertEqual(history[0]["effectiveness"], OptimizationEffectiveness.SIGNIFICANT_IMPROVEMENT.value)
    
    def test_effectiveness_evaluation(self):
        """Test effectiveness evaluation with various improvements."""
        # Add entries with different levels of improvement
        self.tracker.add_entry(
            strategy_id="significant",
            old_version_id="v1.0",
            new_version_id="v1.1",
            optimization_parameters={},
            old_metrics={"sharpe_ratio": 1.0},
            new_metrics={"sharpe_ratio": 1.2},
            job_id="job1"
        )
        
        self.tracker.add_entry(
            strategy_id="moderate",
            old_version_id="v1.0",
            new_version_id="v1.1",
            optimization_parameters={},
            old_metrics={"sharpe_ratio": 1.0},
            new_metrics={"sharpe_ratio": 1.07},
            job_id="job2"
        )
        
        self.tracker.add_entry(
            strategy_id="minimal",
            old_version_id="v1.0",
            new_version_id="v1.1",
            optimization_parameters={},
            old_metrics={"sharpe_ratio": 1.0},
            new_metrics={"sharpe_ratio": 1.02},
            job_id="job3"
        )
        
        self.tracker.add_entry(
            strategy_id="no_change",
            old_version_id="v1.0",
            new_version_id="v1.1",
            optimization_parameters={},
            old_metrics={"sharpe_ratio": 1.0},
            new_metrics={"sharpe_ratio": 1.0},
            job_id="job4"
        )
        
        self.tracker.add_entry(
            strategy_id="regression",
            old_version_id="v1.0",
            new_version_id="v1.1",
            optimization_parameters={},
            old_metrics={"sharpe_ratio": 1.0},
            new_metrics={"sharpe_ratio": 0.95},
            job_id="job5"
        )
        
        # Check statistics
        stats = self.tracker.get_optimization_effectiveness_stats()
        
        self.assertEqual(stats["total_optimizations"], 5)
        self.assertEqual(stats["effectiveness_counts"][OptimizationEffectiveness.SIGNIFICANT_IMPROVEMENT.value], 1)
        self.assertEqual(stats["effectiveness_counts"][OptimizationEffectiveness.MODERATE_IMPROVEMENT.value], 1)
        self.assertEqual(stats["effectiveness_counts"][OptimizationEffectiveness.MINIMAL_IMPROVEMENT.value], 1)
        self.assertEqual(stats["effectiveness_counts"][OptimizationEffectiveness.NO_CHANGE.value], 1)
        self.assertEqual(stats["effectiveness_counts"][OptimizationEffectiveness.REGRESSION.value], 1)


class TestOptimizationScheduler(unittest.TestCase):
    """Tests for OptimizationScheduler."""
    
    def setUp(self):
        # Patch the event bus
        self.event_bus_patcher = patch('trading_bot.event_system.EventBus')
        self.mock_event_bus_class = self.event_bus_patcher.start()
        self.mock_event_bus = MockEventBus()
        self.mock_event_bus_class.return_value = self.mock_event_bus
        
        # Patch job store
        self.test_dir = tempfile.mkdtemp()
        self.job_store = OptimizationJobStore(self.test_dir)
        self.job_store_patcher = patch('trading_bot.autonomous.optimization_scheduler.get_job_store')
        self.mock_get_job_store = self.job_store_patcher.start()
        self.mock_get_job_store.return_value = self.job_store
        
        # Create scheduler with test config
        self.scheduler = OptimizationScheduler({
            "max_concurrent_jobs": 2,
            "check_interval_seconds": 1,
            "retry_limit": 1,
            "job_timeout_hours": 1,
            "auto_schedule": True
        })
    
    def tearDown(self):
        # Stop patchers
        self.event_bus_patcher.stop()
        self.job_store_patcher.stop()
        
        # Remove the test directory
        shutil.rmtree(self.test_dir)
    
    def test_schedule_optimization(self):
        """Test scheduling an optimization job."""
        job = self.scheduler.schedule_optimization(
            strategy_id="test_strategy",
            strategy_type="iron_condor",
            universe="options",
            version_id="v1.0",
            performance_metrics={
                "sharpe_ratio": 0.8,
                "max_drawdown": -0.15
            },
            reason="test"
        )
        
        # Check job in store
        stored_job = self.job_store.get_job(job.job_id)
        self.assertIsNotNone(stored_job)
        self.assertEqual(stored_job.status, OptimizationStatus.SCHEDULED)
        
        # Check event emitted
        events = self.mock_event_bus.get_emitted_events(OptimizationEventType.OPTIMIZATION_SCHEDULED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1]["strategy_id"], "test_strategy")
    
    def test_resource_monitor(self):
        """Test resource monitor capacity management."""
        monitor = ResourceMonitor(max_concurrent_jobs=2)
        
        # Should have capacity for 2 jobs
        self.assertTrue(monitor.can_start_job())
        self.assertTrue(monitor.register_job("job1"))
        self.assertTrue(monitor.can_start_job())
        self.assertTrue(monitor.register_job("job2"))
        
        # Should be at capacity
        self.assertFalse(monitor.can_start_job())
        self.assertFalse(monitor.register_job("job3"))
        
        # Free up a job
        monitor.unregister_job("job1")
        self.assertTrue(monitor.can_start_job())
        self.assertTrue(monitor.register_job("job3"))
    
    def test_scheduler_loop(self):
        """Test scheduler loop processing jobs."""
        # Create a due job
        job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now() - timedelta(minutes=5),
            priority=50
        )
        job.update_status(OptimizationStatus.SCHEDULED)
        self.job_store.add_job(job)
        
        # Register a mock execution callback
        execution_called = threading.Event()
        
        def mock_execution(job):
            job.set_results({"result": "success"})
            execution_called.set()
        
        self.scheduler.register_execution_callback(mock_execution)
        
        # Start scheduler
        self.scheduler.start()
        
        # Wait for job to be processed
        self.assertTrue(execution_called.wait(timeout=5))
        
        # Check job status
        updated_job = self.job_store.get_job(job.job_id)
        self.assertEqual(updated_job.status, OptimizationStatus.COMPLETED)
        
        # Stop scheduler
        self.scheduler.stop()


class TestOptimizationIntegration(unittest.TestCase):
    """Tests for OptimizationIntegration with mock components."""
    
    def setUp(self):
        # Create temp directory
        self.test_dir = tempfile.mkdtemp()
        
        # Patch event bus
        self.event_bus_patcher = patch('trading_bot.event_system.EventBus')
        self.mock_event_bus_class = self.event_bus_patcher.start()
        self.mock_event_bus = MockEventBus()
        self.mock_event_bus_class.return_value = self.mock_event_bus
        
        # Patch job store
        self.job_store = OptimizationJobStore(self.test_dir)
        self.job_store_patcher = patch('trading_bot.autonomous.optimization_integration.get_job_store')
        self.mock_get_job_store = self.job_store_patcher.start()
        self.mock_get_job_store.return_value = self.job_store
        
        # Patch scheduler
        self.scheduler = OptimizationScheduler({"auto_schedule": True})
        self.scheduler_patcher = patch('trading_bot.autonomous.optimization_integration.get_optimization_scheduler')
        self.mock_get_scheduler = self.scheduler_patcher.start()
        self.mock_get_scheduler.return_value = self.scheduler
        
        # Patch history tracker
        self.history_tracker = OptimizationHistoryTracker(self.test_dir)
        self.history_tracker_patcher = patch('trading_bot.autonomous.optimization_integration.get_optimization_history_tracker')
        self.mock_get_history_tracker = self.history_tracker_patcher.start()
        self.mock_get_history_tracker.return_value = self.history_tracker
        
        # Patch lifecycle manager
        self.lifecycle_manager = MockLifecycleManager()
        self.lifecycle_manager_patcher = patch('trading_bot.autonomous.optimization_integration.get_lifecycle_manager')
        self.mock_get_lifecycle_manager = self.lifecycle_manager_patcher.start()
        self.mock_get_lifecycle_manager.return_value = self.lifecycle_manager
        
        # Create integration
        self.integration = OptimizationIntegration()
    
    def tearDown(self):
        # Stop patchers
        self.event_bus_patcher.stop()
        self.job_store_patcher.stop()
        self.scheduler_patcher.stop()
        self.history_tracker_patcher.stop()
        self.lifecycle_manager_patcher.stop()
        
        # Remove test directory
        shutil.rmtree(self.test_dir)
    
    def test_version_created_handler(self):
        """Test handling of strategy version creation events."""
        # Create a strategy version
        version = self.lifecycle_manager.create_version(
            strategy_id="test_strategy",
            version_id="v1.0",
            parameters={"param1": 10},
            performance={"sharpe_ratio": 1.2},
            status="active",
            metadata={"strategy_type": "iron_condor", "universe": "options"}
        )
        
        # Emit event
        self.mock_event_bus.emit(
            "strategy_version_created",
            {
                "strategy_id": "test_strategy",
                "version_id": "v1.0",
                "performance_metrics": {"sharpe_ratio": 1.2},
                "strategy_type": "iron_condor",
                "universe": "options"
            }
        )
        
        # Check for scheduled optimization
        scheduled_jobs = self.job_store.get_jobs_by_strategy("test_strategy")
        self.assertEqual(len(scheduled_jobs), 1)
        self.assertEqual(scheduled_jobs[0].strategy_id, "test_strategy")
        self.assertEqual(scheduled_jobs[0].version_id, "v1.0")
        self.assertEqual(scheduled_jobs[0].reason, "new_version_baseline")
    
    def test_optimization_completed_handler(self):
        """Test handling of optimization completion events."""
        # Create a strategy version
        old_version = self.lifecycle_manager.create_version(
            strategy_id="test_strategy",
            version_id="v1.0",
            parameters={"param1": 10},
            performance={"sharpe_ratio": 1.0},
            status="active",
            metadata={"strategy_type": "iron_condor", "universe": "options"}
        )
        
        # Create an optimization job
        job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now(),
            priority=50
        )
        job.update_status(OptimizationStatus.RUNNING)
        self.job_store.add_job(job)
        
        # Emit optimization completed event
        self.mock_event_bus.emit(
            OptimizationEventType.OPTIMIZATION_COMPLETED,
            {
                "job_id": job.job_id,
                "strategy_id": "test_strategy",
                "version_id": "v1.0",
                "results": {
                    "parameters": {"param1": 15},
                    "old_performance": {"sharpe_ratio": 1.0},
                    "new_performance": {"sharpe_ratio": 1.5},
                    "performance_improvement": {"sharpe_ratio": 0.5}
                }
            }
        )
        
        # Check for new version creation
        versions = self.lifecycle_manager.get_versions("test_strategy")
        self.assertEqual(len(versions), 2)  # Original + optimized
        
        # Find new version
        new_version = None
        for v in versions:
            if v.version_id != "v1.0":
                new_version = v
                break
        
        self.assertIsNotNone(new_version)
        self.assertEqual(new_version.parameters["param1"], 15)
        self.assertEqual(new_version.performance["sharpe_ratio"], 1.5)
        self.assertTrue(new_version.metadata["from_optimization"])
        
        # Check for history entry
        history = self.history_tracker.get_strategy_history("test_strategy")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["old_version_id"], "v1.0")
        self.assertEqual(history[0]["effectiveness"], OptimizationEffectiveness.SIGNIFICANT_IMPROVEMENT.value)
    
    def test_optimization_execution(self):
        """Test execution of optimization jobs."""
        # Create a strategy version
        self.lifecycle_manager.create_version(
            strategy_id="test_strategy",
            version_id="v1.0",
            parameters={"param1": 10},
            performance={"sharpe_ratio": 1.0},
            status="active",
            metadata={"strategy_type": "iron_condor", "universe": "options"}
        )
        
        # Create a job
        job = OptimizationJob(
            strategy_id="test_strategy",
            version_id="v1.0",
            scheduled_time=datetime.now(),
            priority=50
        )
        job.update_status(OptimizationStatus.RUNNING)
        self.job_store.add_job(job)
        
        # Execute job
        self.integration._execute_optimization(job)
        
        # Check job status
        self.assertEqual(job.status, OptimizationStatus.COMPLETED)
        self.assertIsNotNone(job.results)
        
        # Check event emission
        events = self.mock_event_bus.get_emitted_events(OptimizationEventType.OPTIMIZATION_COMPLETED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1]["job_id"], job.job_id)
        self.assertEqual(events[0][1]["strategy_id"], "test_strategy")


if __name__ == '__main__':
    unittest.main()
