#!/usr/bin/env python3
"""
Approval Workflow Test Suite

This module provides comprehensive tests for the Approval Workflow component,
including persistence, event emission, and integration with the A/B Testing Framework.

Tests cover:
- Core data structures (ApprovalStatus, ApprovalRequest)
- ApprovalWorkflowManager with persistence
- Event emission and handling
- Integration with A/B Testing components
"""

import os
import json
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Import approval workflow components
from trading_bot.autonomous.approval_workflow import (
    ApprovalStatus, ApprovalRequest, ApprovalWorkflowManager,
    get_approval_workflow_manager
)

# Import event system components
from trading_bot.event_system import EventBus, Event, EventType

# Import A/B testing components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestStatus
)
from trading_bot.autonomous.ab_testing_integration import (
    ABTestingIntegration, get_ab_testing_integration
)


class TestApprovalWorkflowCore(unittest.TestCase):
    """Test core approval workflow data structures."""

    def test_approval_status_values(self):
        """Test ApprovalStatus enum."""
        # Ensure enum values are as expected
        self.assertEqual(ApprovalStatus.PENDING.value, "pending")
        self.assertEqual(ApprovalStatus.APPROVED.value, "approved")
        self.assertEqual(ApprovalStatus.REJECTED.value, "rejected")

    def test_approval_request_creation(self):
        """Test ApprovalRequest creation and basic properties."""
        # Create a test request
        req = ApprovalRequest(
            test_id="test123",
            strategy_id="strategy_a",
            version_id="v1.0",
            requester="test_system"
        )

        # Check basic properties
        self.assertEqual(req.test_id, "test123")
        self.assertEqual(req.strategy_id, "strategy_a")
        self.assertEqual(req.version_id, "v1.0")
        self.assertEqual(req.requester, "test_system")
        self.assertEqual(req.status, ApprovalStatus.PENDING)
        self.assertIsNone(req.reviewer)
        self.assertIsNone(req.decision_time)
        self.assertIsNone(req.comments)
        self.assertIsInstance(req.request_id, str)
        self.assertIsInstance(req.request_time, datetime)

    def test_approval_request_approve(self):
        """Test approving a request."""
        req = ApprovalRequest(
            test_id="test123",
            strategy_id="strategy_a",
            version_id="v1.0"
        )

        # Approve the request
        before_time = datetime.utcnow()
        req.approve(reviewer="reviewer1", comments="Looks good")
        after_time = datetime.utcnow()

        # Check approval status
        self.assertEqual(req.status, ApprovalStatus.APPROVED)
        self.assertEqual(req.reviewer, "reviewer1")
        self.assertEqual(req.comments, "Looks good")
        self.assertIsNotNone(req.decision_time)
        
        # Check decision time is set correctly
        self.assertGreaterEqual(req.decision_time, before_time)
        self.assertLessEqual(req.decision_time, after_time)

    def test_approval_request_reject(self):
        """Test rejecting a request."""
        req = ApprovalRequest(
            test_id="test123",
            strategy_id="strategy_a",
            version_id="v1.0"
        )

        # Reject the request
        before_time = datetime.utcnow()
        req.reject(reviewer="reviewer2", comments="Not enough data")
        after_time = datetime.utcnow()

        # Check rejection status
        self.assertEqual(req.status, ApprovalStatus.REJECTED)
        self.assertEqual(req.reviewer, "reviewer2")
        self.assertEqual(req.comments, "Not enough data")
        self.assertIsNotNone(req.decision_time)
        
        # Check decision time is set correctly
        self.assertGreaterEqual(req.decision_time, before_time)
        self.assertLessEqual(req.decision_time, after_time)

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        # Create a request with all fields populated
        original = ApprovalRequest(
            test_id="test123",
            strategy_id="strategy_a",
            version_id="v1.0",
            requester="test_system"
        )
        original.approve(reviewer="reviewer1", comments="Looks good")

        # Convert to dict and back
        data = original.to_dict()
        restored = ApprovalRequest.from_dict(data)

        # Check all fields are preserved
        self.assertEqual(restored.test_id, original.test_id)
        self.assertEqual(restored.strategy_id, original.strategy_id)
        self.assertEqual(restored.version_id, original.version_id)
        self.assertEqual(restored.requester, original.requester)
        self.assertEqual(restored.request_id, original.request_id)
        self.assertEqual(restored.status, original.status)
        self.assertEqual(restored.reviewer, original.reviewer)
        self.assertEqual(restored.comments, original.comments)
        
        # Check timestamps (allowing for microsecond differences in serialization)
        self.assertAlmostEqual(
            restored.request_time.timestamp(),
            original.request_time.timestamp(),
            delta=0.001
        )
        self.assertAlmostEqual(
            restored.decision_time.timestamp(),
            original.decision_time.timestamp(),
            delta=0.001
        )


class TestApprovalWorkflowManager(unittest.TestCase):
    """Test the ApprovalWorkflowManager."""

    def setUp(self):
        """Set up test environment with temporary storage."""
        # Create a temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_approvals.json"
        
        # Create manager with mock event bus
        self.event_bus_mock = MagicMock()
        with patch('trading_bot.event_system.EventBus', return_value=self.event_bus_mock):
            self.manager = ApprovalWorkflowManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_create_request(self):
        """Test creating an approval request."""
        # Create a request
        req = self.manager.create_request(
            test_id="test123",
            strategy_id="strategy_a",
            version_id="v1.0",
            requester="test_system"
        )

        # Check request was created correctly
        self.assertIsInstance(req, ApprovalRequest)
        self.assertEqual(req.test_id, "test123")
        self.assertEqual(req.strategy_id, "strategy_a")
        self.assertEqual(req.version_id, "v1.0")
        self.assertEqual(req.requester, "test_system")
        
        # Check manager has the request
        stored_req = self.manager.get_request(req.request_id)
        self.assertEqual(stored_req, req)
        
        # Check event was emitted
        self.event_bus_mock.publish.assert_called_once()
        event = self.event_bus_mock.publish.call_args[0][0]
        self.assertEqual(event.event_type, EventType.APPROVAL_REQUEST_CREATED)
        self.assertEqual(event.data["request_id"], req.request_id)
        self.assertEqual(event.data["test_id"], "test123")

    def test_list_requests(self):
        """Test listing requests by status."""
        # Create requests with different statuses
        req1 = self.manager.create_request(
            test_id="test1", strategy_id="strat1", version_id="v1"
        )
        req2 = self.manager.create_request(
            test_id="test2", strategy_id="strat2", version_id="v2"
        )
        req3 = self.manager.create_request(
            test_id="test3", strategy_id="strat3", version_id="v3"
        )
        
        # Approve and reject some requests
        self.manager.approve_request(req1.request_id, "reviewer1")
        self.manager.reject_request(req2.request_id, "reviewer2")
        
        # Check listing by status
        pending = self.manager.list_requests(status=ApprovalStatus.PENDING)
        approved = self.manager.list_requests(status=ApprovalStatus.APPROVED)
        rejected = self.manager.list_requests(status=ApprovalStatus.REJECTED)
        all_reqs = self.manager.list_requests(status=None)
        
        # Verify counts
        self.assertEqual(len(pending), 1)
        self.assertEqual(len(approved), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(all_reqs), 3)
        
        # Verify specific requests are in the correct lists
        self.assertEqual(pending[0].request_id, req3.request_id)
        self.assertEqual(approved[0].request_id, req1.request_id)
        self.assertEqual(rejected[0].request_id, req2.request_id)

    def test_approve_request(self):
        """Test approving a request."""
        # Create a request
        req = self.manager.create_request(
            test_id="test123", strategy_id="strategy_a", version_id="v1.0"
        )
        
        # Reset mock to clear creation event
        self.event_bus_mock.reset_mock()
        
        # Approve the request
        success = self.manager.approve_request(
            req.request_id, reviewer="reviewer1", comments="Good"
        )
        
        # Check approval succeeded
        self.assertTrue(success)
        updated_req = self.manager.get_request(req.request_id)
        self.assertEqual(updated_req.status, ApprovalStatus.APPROVED)
        self.assertEqual(updated_req.reviewer, "reviewer1")
        self.assertEqual(updated_req.comments, "Good")
        
        # Check approval event was emitted
        self.event_bus_mock.publish.assert_called_once()
        event = self.event_bus_mock.publish.call_args[0][0]
        self.assertEqual(event.event_type, EventType.APPROVAL_REQUEST_APPROVED)
        self.assertEqual(event.data["request_id"], req.request_id)
        self.assertEqual(event.data["reviewer"], "reviewer1")

    def test_reject_request(self):
        """Test rejecting a request."""
        # Create a request
        req = self.manager.create_request(
            test_id="test123", strategy_id="strategy_a", version_id="v1.0"
        )
        
        # Reset mock to clear creation event
        self.event_bus_mock.reset_mock()
        
        # Reject the request
        success = self.manager.reject_request(
            req.request_id, reviewer="reviewer2", comments="Not enough data"
        )
        
        # Check rejection succeeded
        self.assertTrue(success)
        updated_req = self.manager.get_request(req.request_id)
        self.assertEqual(updated_req.status, ApprovalStatus.REJECTED)
        self.assertEqual(updated_req.reviewer, "reviewer2")
        self.assertEqual(updated_req.comments, "Not enough data")
        
        # Check rejection event was emitted
        self.event_bus_mock.publish.assert_called_once()
        event = self.event_bus_mock.publish.call_args[0][0]
        self.assertEqual(event.event_type, EventType.APPROVAL_REQUEST_REJECTED)
        self.assertEqual(event.data["request_id"], req.request_id)
        self.assertEqual(event.data["reviewer"], "reviewer2")

    def test_approve_nonexistent_request(self):
        """Test approving a nonexistent request."""
        # Try to approve a nonexistent request
        success = self.manager.approve_request(
            "nonexistent", reviewer="reviewer1"
        )
        
        # Check approval failed
        self.assertFalse(success)
        
        # Check no event was emitted
        self.event_bus_mock.publish.assert_not_called()

    def test_approve_already_approved_request(self):
        """Test approving an already approved request."""
        # Create and approve a request
        req = self.manager.create_request(
            test_id="test123", strategy_id="strategy_a", version_id="v1.0"
        )
        self.manager.approve_request(req.request_id, reviewer="reviewer1")
        
        # Reset mock to clear previous events
        self.event_bus_mock.reset_mock()
        
        # Try to approve again
        success = self.manager.approve_request(
            req.request_id, reviewer="reviewer2"
        )
        
        # Check second approval failed
        self.assertFalse(success)
        
        # Check no event was emitted
        self.event_bus_mock.publish.assert_not_called()

    def test_persistence(self):
        """Test that requests are saved to and loaded from disk."""
        # Create some requests
        req1 = self.manager.create_request(
            test_id="test1", strategy_id="strat1", version_id="v1"
        )
        req2 = self.manager.create_request(
            test_id="test2", strategy_id="strat2", version_id="v2"
        )
        
        # Change status of one request
        self.manager.approve_request(req1.request_id, reviewer="reviewer1")
        
        # Create a new manager instance that should load from the same file
        with patch('trading_bot.event_system.EventBus'):
            new_manager = ApprovalWorkflowManager(storage_path=self.storage_path)
        
        # Check all requests were loaded
        loaded_req1 = new_manager.get_request(req1.request_id)
        loaded_req2 = new_manager.get_request(req2.request_id)
        
        self.assertIsNotNone(loaded_req1)
        self.assertIsNotNone(loaded_req2)
        
        # Check status was preserved
        self.assertEqual(loaded_req1.status, ApprovalStatus.APPROVED)
        self.assertEqual(loaded_req2.status, ApprovalStatus.PENDING)


class TestApprovalWorkflowIntegration(unittest.TestCase):
    """Test integration with A/B Testing framework."""

    def setUp(self):
        """Set up test environment with mocks."""
        # Create temp storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_approvals.json"
        
        # Mock dependencies
        self.event_bus_mock = MagicMock()
        self.ab_test_manager_mock = MagicMock()
        self.ab_test_analyzer_mock = MagicMock()
        self.lifecycle_manager_mock = MagicMock()
        
        # Create a real approval manager
        with patch('trading_bot.event_system.EventBus', return_value=self.event_bus_mock):
            self.approval_manager = ApprovalWorkflowManager(storage_path=self.storage_path)
        
        # Patch the approval workflow singleton
        self.get_approval_workflow_patcher = patch(
            'trading_bot.autonomous.approval_workflow.get_approval_workflow_manager',
            return_value=self.approval_manager
        )
        self.get_approval_workflow_mock = self.get_approval_workflow_patcher.start()
        
        # Patch other singletons
        self.get_event_bus_patcher = patch(
            'trading_bot.event_system.EventBus',
            return_value=self.event_bus_mock
        )
        self.get_ab_test_manager_patcher = patch(
            'trading_bot.autonomous.ab_testing_manager.get_ab_test_manager',
            return_value=self.ab_test_manager_mock
        )
        self.get_ab_test_analyzer_patcher = patch(
            'trading_bot.autonomous.ab_testing_analysis.get_ab_test_analyzer',
            return_value=self.ab_test_analyzer_mock
        )
        self.get_lifecycle_manager_patcher = patch(
            'trading_bot.autonomous.strategy_lifecycle_manager.get_lifecycle_manager',
            return_value=self.lifecycle_manager_mock
        )
        
        # Start all patches
        self.get_event_bus_mock = self.get_event_bus_patcher.start()
        self.get_ab_test_manager_mock = self.get_ab_test_manager_patcher.start()
        self.get_ab_test_analyzer_mock = self.get_ab_test_analyzer_patcher.start()
        self.get_lifecycle_manager_mock = self.get_lifecycle_manager_patcher.start()

    def tearDown(self):
        """Clean up test environment."""
        # Stop all patches
        self.get_approval_workflow_patcher.stop()
        self.get_event_bus_patcher.stop()
        self.get_ab_test_manager_patcher.stop()
        self.get_ab_test_analyzer_patcher.stop()
        self.get_lifecycle_manager_patcher.stop()
        
        # Remove temp directory
        shutil.rmtree(self.temp_dir)

    def test_test_completion_creates_approval_request(self):
        """Test that test completion creates an approval request."""
        # Create a mock test
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.name = "Test 123"
        mock_test.variant_b = MagicMock()
        mock_test.variant_b.strategy_id = "strategy_a"
        mock_test.variant_b.version_id = "v2.0"
        mock_test.variant_b.name = "Variant B"
        mock_test.variant_b.metrics = {"sharpe_ratio": 1.5, "win_rate": 0.6}
        
        # Configure manager to return our mock test
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Configure analyzer to return recommendation
        self.ab_test_analyzer_mock.analyze_test.return_value = {
            "recommendation": {
                "promote_variant_b": True,
                "confidence": "high",
                "explanation": "B is better",
                "regime_switching": {
                    "recommended": False
                }
            }
        }
        
        # Create A/B testing integration
        integration = ABTestingIntegration()
        
        # Create event with test completed data
        event = MagicMock()
        event.data = {
            "test_id": "test123",
            "name": "Test 123",
            "result": {
                "winner": "B",
                "conclusion": "B is better"
            }
        }
        
        # Call handler
        integration._handle_test_completed(event)
        
        # Check that test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that test was analyzed
        self.ab_test_analyzer_mock.analyze_test.assert_called_once_with(mock_test)
        
        # Check that approval request was created
        pending_requests = self.approval_manager.list_requests(status=ApprovalStatus.PENDING)
        self.assertEqual(len(pending_requests), 1)
        
        req = pending_requests[0]
        self.assertEqual(req.test_id, "test123")
        self.assertEqual(req.strategy_id, "strategy_a")
        self.assertEqual(req.version_id, "v2.0")
        self.assertEqual(req.requester, "ab_testing_system")

    def test_approval_triggers_promotion(self):
        """Test that approval triggers strategy promotion."""
        # Create a mock test
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.variant_b = MagicMock()
        mock_test.variant_b.strategy_id = "strategy_a"
        mock_test.variant_b.version_id = "v2.0"
        
        # Configure manager to return our mock test
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Configure lifecycle manager
        mock_version = MagicMock()
        self.lifecycle_manager_mock.get_version.return_value = mock_version
        
        # Create A/B testing integration
        integration = ABTestingIntegration()
        
        # Create approval request
        req = self.approval_manager.create_request(
            test_id="test123",
            strategy_id="strategy_a",
            version_id="v2.0"
        )
        
        # Approve the request
        self.approval_manager.approve_request(
            req.request_id, 
            reviewer="reviewer1", 
            comments="Looks good"
        )
        
        # Create event with approval data
        event = MagicMock()
        event.data = {
            "request_id": req.request_id,
            "test_id": "test123",
            "strategy_id": "strategy_a",
            "version_id": "v2.0",
            "reviewer": "reviewer1",
            "comments": "Looks good"
        }
        
        # Call handler
        integration._handle_approval_approved(event)
        
        # Check that test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that version was retrieved
        self.lifecycle_manager_mock.get_version.assert_called_once_with("strategy_a", "v2.0")
        
        # Check that promotion was attempted
        self.lifecycle_manager_mock.promote_version.assert_called_once()

    def test_full_workflow_simulation(self):
        """Simulate the full workflow from test completion to promotion."""
        # Create a mock test
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.name = "Test 123"
        mock_test.variant_b = MagicMock()
        mock_test.variant_b.strategy_id = "strategy_a"
        mock_test.variant_b.version_id = "v2.0"
        mock_test.variant_b.name = "Variant B"
        mock_test.variant_b.metrics = {"sharpe_ratio": 1.5, "win_rate": 0.6}
        
        # Configure AB test manager
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Configure analyzer
        self.ab_test_analyzer_mock.analyze_test.return_value = {
            "recommendation": {
                "promote_variant_b": True,
                "confidence": "high",
                "explanation": "B is better"
            }
        }
        
        # Configure lifecycle manager
        mock_version = MagicMock()
        self.lifecycle_manager_mock.get_version.return_value = mock_version
        
        # Create A/B testing integration
        integration = ABTestingIntegration()
        
        # STEP 1: Test completion
        test_completion_event = MagicMock()
        test_completion_event.data = {
            "test_id": "test123",
            "name": "Test 123"
        }
        
        integration._handle_test_completed(test_completion_event)
        
        # Check approval request was created
        pending_requests = self.approval_manager.list_requests(status=ApprovalStatus.PENDING)
        self.assertEqual(len(pending_requests), 1)
        req = pending_requests[0]
        
        # STEP 2: Human approval
        self.approval_manager.approve_request(
            req.request_id,
            reviewer="reviewer1",
            comments="Looks good"
        )
        
        # STEP 3: Approval event handling
        approval_event = MagicMock()
        approval_event.data = {
            "request_id": req.request_id,
            "test_id": "test123",
            "strategy_id": "strategy_a",
            "version_id": "v2.0",
            "reviewer": "reviewer1",
            "comments": "Looks good"
        }
        
        # Reset mocks to clear previous calls
        self.ab_test_manager_mock.reset_mock()
        self.lifecycle_manager_mock.reset_mock()
        
        integration._handle_approval_approved(approval_event)
        
        # Check that test was retrieved again
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that version was retrieved
        self.lifecycle_manager_mock.get_version.assert_called_once_with("strategy_a", "v2.0")
        
        # Check that promotion was attempted
        self.lifecycle_manager_mock.promote_version.assert_called_once()


if __name__ == "__main__":
    unittest.main()
