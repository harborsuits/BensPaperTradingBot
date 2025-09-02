#!/usr/bin/env python3
"""
Feedback Loop Approval Workflow Integration

This script integrates the multi-objective feedback loop with the approval workflow system.
It demonstrates how the optimization, approval, and verification processes work together
to create a complete self-improving system.

It builds upon our successful components:
1. MultiObjectiveFeedbackLoop - for optimizing strategies
2. ApprovalWorkflowManager - for strategy approval
3. PerformanceVerifier - for tracking real performance
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import our multi-objective feedback loop
from feedback_loop_multi_objective_demo import MultiObjectiveFeedbackLoop

# Import from approval workflow and verification systems
try:
    from trading_bot.autonomous.approval_workflow import (
        get_approval_workflow_manager,
        ApprovalRequest,
        ApprovalStatus
    )
    from trading_bot.autonomous.performance_verification import (
        PerformanceVerifier,
        StrategyPerformanceRecord
    )
    from trading_bot.event_system import (
        EventBus,
        Event,
        EventType
    )
    FULL_INTEGRATION_AVAILABLE = True
except ImportError:
    print("Full integration components not available. Running in simulation mode.")
    FULL_INTEGRATION_AVAILABLE = False
    
    # Define minimal simulation classes
    class ApprovalRequest:
        def __init__(self, test_id, strategy_id, version_id, requester="system"):
            self.request_id = str(uuid.uuid4())
            self.test_id = test_id
            self.strategy_id = strategy_id
            self.version_id = version_id
            self.requester = requester
            self.status = "PENDING"
            self.reviewer = None
            self.decision_time = None
            self.comments = None
            
        def approve(self, reviewer, comments=None):
            self.status = "APPROVED"
            self.reviewer = reviewer
            self.decision_time = datetime.utcnow()
            self.comments = comments
    
    class ApprovalWorkflowManager:
        def __init__(self):
            self.requests = {}
            
        def create_request(self, **kwargs):
            req = ApprovalRequest(**kwargs)
            self.requests[req.request_id] = req
            return req
            
        def approve_request(self, request_id, reviewer, comments=None):
            if request_id in self.requests:
                self.requests[request_id].approve(reviewer, comments)
                return self.requests[request_id]
            return None
    
    class EventBus:
        def __init__(self):
            self.handlers = {}
            
        def subscribe(self, event_type, handler):
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
        def emit(self, event):
            if event.event_type in self.handlers:
                for handler in self.handlers[event.event_type]:
                    handler(event)
    
    class Event:
        def __init__(self, event_type, data=None):
            self.event_type = event_type
            self.data = data or {}
            self.timestamp = datetime.utcnow()
    
    class EventType:
        APPROVAL_REQUEST_CREATED = "approval_request_created"
        APPROVAL_REQUEST_APPROVED = "approval_request_approved"
        APPROVAL_REQUEST_REJECTED = "approval_request_rejected"
        STRATEGY_PERFORMANCE_UPDATE = "strategy_performance_update"
    
    class PerformanceVerifier:
        def __init__(self):
            self.records = {}
            
        def register_strategy(self, strategy_id, params, expected_performance):
            self.records[strategy_id] = {
                "params": params,
                "expected": expected_performance,
                "actual": None
            }
            
        def update_performance(self, strategy_id, actual_performance):
            if strategy_id in self.records:
                self.records[strategy_id]["actual"] = actual_performance
                return True
            return False
    
    def get_approval_workflow_manager():
        return ApprovalWorkflowManager()


class FeedbackLoopApproval:
    """
    Integrates the feedback loop with the approval workflow system.
    
    This class demonstrates how the optimization, approval, and verification
    processes work together in a complete self-improving system.
    """
    
    def __init__(self):
        """Initialize the integration components."""
        # Core components
        self.approval_manager = get_approval_workflow_manager()
        self.event_bus = EventBus()
        self.optimizer = MultiObjectiveFeedbackLoop()
        self.performance_verifier = PerformanceVerifier()
        
        # Tracking for strategies
        self.strategy_registry = {}
        
        # Register event handlers
        if FULL_INTEGRATION_AVAILABLE:
            self.event_bus.subscribe(
                EventType.APPROVAL_REQUEST_APPROVED, 
                self.handle_approval
            )
            self.event_bus.subscribe(
                EventType.STRATEGY_PERFORMANCE_UPDATE, 
                self.handle_performance
            )
        
        print("Initialized FeedbackLoopApproval integration")
    
    def handle_approval(self, event):
        """
        Handle strategy approval event.
        
        This starts tracking the approved strategy for performance verification.
        
        Args:
            event: Approval event containing strategy details
        """
        if not hasattr(event, 'data') or 'request_id' not in event.data:
            print("Invalid approval event format")
            return
        
        request_id = event.data['request_id']
        
        # Find the strategy details in our registry
        if request_id not in self.strategy_registry:
            print(f"Strategy with request_id {request_id} not found in registry")
            return
        
        strategy_info = self.strategy_registry[request_id]
        
        print(f"\nStrategy approved: {strategy_info['strategy_id']}")
        
        # Register with performance verifier for tracking
        self.performance_verifier.register_strategy(
            strategy_info['strategy_id'],
            strategy_info['params'],
            strategy_info['expected_performance']
        )
        
        print(f"Started performance tracking for strategy {strategy_info['strategy_id']}")
    
    def handle_performance(self, event):
        """
        Process performance update event.
        
        This feeds real performance data back to the optimization system.
        
        Args:
            event: Performance event containing actual results
        """
        if not hasattr(event, 'data'):
            print("Invalid performance event format")
            return
        
        data = event.data
        
        if 'strategy_id' not in data or 'performance' not in data:
            print("Missing strategy_id or performance data")
            return
        
        strategy_id = data['strategy_id']
        performance = data['performance']
        
        print(f"\nReceived performance update for strategy {strategy_id}")
        
        # Update performance verifier
        success = self.performance_verifier.update_performance(
            strategy_id, 
            performance
        )
        
        if not success:
            print(f"Strategy {strategy_id} not found in performance verifier")
            return
        
        # Get the strategy record for verification
        record = self.performance_verifier.records[strategy_id]
        
        # Verify prediction accuracy
        expected = record['expected']
        actual = record['actual']
        
        # Feed back to the optimizer for learning
        accuracy = self.optimizer.verify_accuracy(expected, actual)
        
        # Update the market models
        self.optimizer.update_market_models()
        
        print(f"Updated market models based on performance of strategy {strategy_id}")
        print(f"New prediction accuracy: {accuracy:.2%}")
    
    def submit_strategy(self, params, expected_perf, name=None):
        """
        Submit a strategy for approval.
        
        Args:
            params: Strategy parameters
            expected_perf: Expected performance metrics
            name: Strategy name (optional)
            
        Returns:
            request_id: ID of the created approval request
        """
        # Generate IDs
        strategy_id = f"strategy_{uuid.uuid4().hex[:8]}"
        version_id = "v1"
        test_id = f"test_{uuid.uuid4().hex[:8]}"
        
        # Use provided name or generate one
        if name is None:
            name = f"Optimized Strategy {strategy_id}"
        
        # Create approval request
        request = self.approval_manager.create_request(
            test_id=test_id,
            strategy_id=strategy_id,
            version_id=version_id,
            requester="feedback_loop_system"
        )
        
        # Store in registry
        self.strategy_registry[request.request_id] = {
            'strategy_id': strategy_id,
            'params': params,
            'expected_performance': expected_perf,
            'name': name
        }
        
        print(f"\nSubmitted strategy {name} for approval")
        print(f"Request ID: {request.request_id}")
        
        # In a full system, we would emit an event here
        if FULL_INTEGRATION_AVAILABLE:
            self.event_bus.emit(Event(
                EventType.APPROVAL_REQUEST_CREATED,
                {'request_id': request.request_id}
            ))
        
        return request.request_id


def run_integration_demo():
    """
    Run a demonstration of the integrated feedback loop system.
    
    This function demonstrates how the complete system works together:
    1. Optimize a strategy
    2. Submit for approval
    3. Track real performance
    4. Feed results back to improve future optimization
    """
    print("=" * 70)
    print("FEEDBACK LOOP APPROVAL INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how the complete system works:")
    print("1. Generate optimized strategies")
    print("2. Submit strategies for approval")
    print("3. Track post-approval performance")
    print("4. Feed real performance back to improve the optimizer")
    print("\n" + "=" * 70)
    
    # Initialize the integration
    integration = FeedbackLoopApproval()
    
    # Run through multiple iterations
    for i in range(3):
        print(f"\n=== Iteration {i+1}/3 ===")
        
        # 1. Optimize strategy
        params = integration.optimizer.optimize()
        
        # Get expected performance
        expected_perf = integration.optimizer.expected_perf[-1]
        
        # 2. Submit for approval
        request_id = integration.submit_strategy(
            params, 
            expected_perf,
            name=f"Optimized Strategy {i+1}"
        )
        
        # 3. Simulate approval
        integration.approval_manager.approve_request(
            request_id, 
            "test_reviewer", 
            "Approved for demonstration"
        )
        
        # Emit approval event
        integration.event_bus.emit(Event(
            EventType.APPROVAL_REQUEST_APPROVED,
            {'request_id': request_id}
        ))
        
        # 4. Simulate real-world performance
        strategy_info = integration.strategy_registry[request_id]
        
        # Get actual performance from the optimizer's test method
        actual_perf = integration.optimizer.test_strategy(params)
        
        # 5. Create performance event
        integration.event_bus.emit(Event(
            EventType.STRATEGY_PERFORMANCE_UPDATE,
            {
                'strategy_id': strategy_info['strategy_id'],
                'performance': actual_perf
            }
        ))
        
        # 6. Show current state
        print("\nCurrent state after iteration:")
        print("Market models:")
        for regime, params in integration.optimizer.market_models.items():
            print(f"  {regime}: {params}")
        
        print(f"Overall prediction accuracy: {integration.optimizer.accuracy['overall'][-1]:.2%}")
        
        # Small delay for readability
        time.sleep(1)
    
    # Final summary
    print("\n" + "=" * 70)
    print("INTEGRATION DEMONSTRATION RESULTS")
    print("=" * 70)
    
    # Show accuracy improvement
    print("\nPrediction Accuracy Improvement:")
    for i, accuracy in enumerate(integration.optimizer.accuracy['overall']):
        print(f"Iteration {i+1}: {accuracy:.2%}")
    
    if len(integration.optimizer.accuracy['overall']) > 1:
        initial = integration.optimizer.accuracy['overall'][0]
        final = integration.optimizer.accuracy['overall'][-1]
        improvement = (final - initial) / initial if initial > 0 else 0
        
        print(f"\nInitial accuracy: {initial:.2%}")
        print(f"Final accuracy: {final:.2%}")
        print(f"Improvement: {improvement:.2%}")
    
    # Show model convergence
    print("\nMarket Model Convergence:")
    for regime in integration.optimizer.market_models:
        if regime in integration.optimizer.real_world_params:
            print(f"\n{regime.capitalize()} regime:")
            print("Parameter | Initial    | Final      | Real       | Convergence")
            print("-" * 64)
            
            initial_params = integration.optimizer.param_history[0][regime]
            final_params = integration.optimizer.market_models[regime]
            real_params = integration.optimizer.real_world_params[regime]
            
            for param in ['trend', 'volatility', 'reversion']:
                initial = initial_params[param]
                final = final_params[param]
                real = real_params[param]
                
                # Calculate convergence percentage
                initial_diff = abs(initial - real)
                final_diff = abs(final - real)
                
                if initial_diff == 0:
                    convergence = 100.0
                else:
                    convergence = (1 - final_diff / initial_diff) * 100.0
                    convergence = max(0, convergence)  # Don't show negative convergence
                
                print(f"{param:<9} | {initial:<10.6f} | {final:<10.6f} | {real:<10.6f} | {convergence:>6.2f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if len(integration.optimizer.accuracy['overall']) > 1 and integration.optimizer.accuracy['overall'][-1] > integration.optimizer.accuracy['overall'][0]:
        print("\nThe integrated feedback loop successfully improved prediction accuracy!")
        print(f"The system learned from real performance data and adjusted its models,")
        print(f"resulting in a {improvement:.2%} improvement in prediction accuracy.")
    else:
        print("\nThe integration demonstration did not show significant improvement.")
    
    print("\nThis demonstrates how the complete self-improving system works,")
    print("connecting optimization, approval, verification, and learning.")
    print("=" * 70)


if __name__ == "__main__":
    run_integration_demo()
