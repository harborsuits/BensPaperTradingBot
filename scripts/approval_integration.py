#!/usr/bin/env python3
"""
Approval Workflow Integration

This module integrates our enhanced feedback loop with the approval workflow system,
creating a complete end-to-end system that:

1. Optimizes strategies using the enhanced feedback loop
2. Submits them through the approval workflow
3. Processes performance data to improve future optimizations
4. Implements convergence detection and early stopping
5. Applies specialized learning rules for challenging market regimes

This builds directly on our successful components:
- multi_objective_simplified.py (core feedback loop)
- feedback_loop_enhancements.py (adaptive learning)
- feedback_loop_text_visualized.py (text visualization)
"""

import uuid
import time
import os
import json
from typing import Dict, List, Any, Tuple, Optional

# Import our successful components
from multi_objective_simplified import MultiObjectiveFeedbackLoop, REAL_WORLD_PARAMS
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
    
    # Define minimal simulation classes
    class ApprovalWorkflowManager:
        def __init__(self):
            self.requests = {}
            
        def create_request(self, **kwargs):
            request_id = str(uuid.uuid4())
            self.requests[request_id] = {
                'request_id': request_id,
                'status': 'PENDING',
                **kwargs
            }
            return self.requests[request_id]
            
        def approve_request(self, request_id, reviewer, comments=None):
            if request_id in self.requests:
                self.requests[request_id]['status'] = 'APPROVED'
                self.requests[request_id]['reviewer'] = reviewer
                self.requests[request_id]['comments'] = comments
                return self.requests[request_id]
            return None
    
    def get_approval_workflow_manager():
        return ApprovalWorkflowManager()
    
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
    
    class EventType:
        APPROVAL_REQUEST_CREATED = "approval_request_created"
        APPROVAL_REQUEST_APPROVED = "approval_request_approved"
        APPROVAL_REQUEST_REJECTED = "approval_request_rejected"
        STRATEGY_PERFORMANCE_UPDATE = "strategy_performance_update"


class ApprovalWorkflowIntegration:
    """
    Integrates the enhanced feedback loop with the approval workflow system.
    """
    
    def __init__(self):
        """Initialize the integrated system components."""
        # Initialize feedback loop with enhanced learning
        self.feedback_loop = MultiObjectiveFeedbackLoop()
        patch_feedback_loop(self.feedback_loop)
        
        # Store reference to real-world params for visualization and metrics
        self.feedback_loop.real_world_params = REAL_WORLD_PARAMS
        
        # Initialize approval workflow components
        self.approval_manager = get_approval_workflow_manager()
        self.event_bus = EventBus()
        
        # Strategy registry for tracking submitted strategies
        self.strategy_registry = {}
        
        # Results tracking
        self.results_dir = "workflow_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Register event handlers
        self.event_bus.subscribe(
            EventType.APPROVAL_REQUEST_APPROVED, 
            self.handle_approval
        )
        self.event_bus.subscribe(
            EventType.STRATEGY_PERFORMANCE_UPDATE, 
            self.handle_performance
        )
        
        print("Initialized ApprovalWorkflowIntegration")
    
    def submit_strategy(self, params, expected_perf, name=None):
        """
        Submit an optimized strategy for approval.
        
        Args:
            params: Strategy parameters
            expected_perf: Expected performance metrics
            name: Optional strategy name
            
        Returns:
            request_id: ID of the created approval request
        """
        # Generate IDs
        strategy_id = f"strategy_{uuid.uuid4().hex[:8]}"
        version_id = "v1"
        test_id = f"test_{uuid.uuid4().hex[:8]}"
        
        # Use provided name or generate one
        name = name or f"Strategy {strategy_id}"
        
        # Create approval request
        request = self.approval_manager.create_request(
            test_id=test_id,
            strategy_id=strategy_id,
            version_id=version_id,
            requester="feedback_loop_system"
        )
        
        # Extract request ID based on the return type
        if isinstance(request, dict):
            request_id = request['request_id']
        else:
            request_id = request.request_id
        
        # Store in registry
        self.strategy_registry[request_id] = {
            'strategy_id': strategy_id,
            'params': params,
            'expected_performance': expected_perf,
            'name': name
        }
        
        print(f"Submitted strategy {name} for approval (Request ID: {request_id})")
        
        # Emit event
        if WORKFLOW_AVAILABLE:
            self.event_bus.emit(Event(
                EventType.APPROVAL_REQUEST_CREATED,
                {'request_id': request_id}
            ))
        
        return request_id
    
    def handle_approval(self, event):
        """
        Handle an approved strategy by starting performance tracking.
        
        Args:
            event: Approval event containing strategy details
        """
        request_id = event.data['request_id']
        
        # Find strategy in registry
        if request_id not in self.strategy_registry:
            print(f"Strategy with request_id {request_id} not found")
            return
        
        strategy_info = self.strategy_registry[request_id]
        
        # In a real system, this would register with the performance verification system
        print(f"Started performance tracking for strategy: {strategy_info['name']}")
    
    def handle_performance(self, event):
        """
        Process real performance data to update the feedback loop.
        
        Args:
            event: Performance event containing actual results
        """
        strategy_id = event.data['strategy_id']
        performance = event.data['performance']
        
        # Find the associated strategy
        request_id = None
        for req_id, info in self.strategy_registry.items():
            if info['strategy_id'] == strategy_id:
                request_id = req_id
                break
        
        if request_id is None:
            print(f"Strategy {strategy_id} not found in registry")
            return
        
        # Get expected performance and update the feedback loop
        expected_perf = self.strategy_registry[request_id]['expected_performance']
        
        # Store the expected and actual performance in the feedback loop
        self.feedback_loop.expected_perf.append(expected_perf)
        self.feedback_loop.actual_perf.append(performance)
        
        # Verify accuracy and update market models
        self.feedback_loop.verify_accuracy()
        
        # Apply regime-specific refinements
        self.apply_regime_specific_refinements()
        
        # Update market models
        self.feedback_loop.update_market_models()
        
        print(f"Updated market models based on performance of {strategy_id}")
    
    def run_optimization_cycle(self):
        """
        Run a complete optimization, approval, and feedback cycle.
        
        Returns:
            Dict with cycle results
        """
        # Step 1: Optimize strategy using current market models
        print("\nOptimizing strategy...")
        params = self.feedback_loop.optimize()
        expected_perf = self.feedback_loop.expected_perf[-1]
        
        # Step 2: Submit for approval
        print("\nSubmitting strategy for approval...")
        request_id = self.submit_strategy(params, expected_perf)
        
        # Step 3: Simulate approval (in a real system, this would be done by a human)
        print("\nAwaiting approval...")
        self.approval_manager.approve_request(
            request_id, 
            "auto_approver", 
            "Auto-approved for demonstration"
        )
        
        # Emit approval event
        self.event_bus.emit(Event(
            EventType.APPROVAL_REQUEST_APPROVED,
            {'request_id': request_id}
        ))
        
        # Step 4: Simulate performance data (in a real system, this would come from actual trading)
        print("\nCollecting performance data...")
        strategy_info = self.strategy_registry[request_id]
        
        # Get actual performance by testing in "real world" conditions
        actual_perf = self.feedback_loop.test_strategy(params)
        
        # Step 5: Feed performance data back
        print("\nProcessing performance data...")
        self.event_bus.emit(Event(
            EventType.STRATEGY_PERFORMANCE_UPDATE,
            {
                'strategy_id': strategy_info['strategy_id'],
                'performance': actual_perf
            }
        ))
        
        # Record results
        cycle_result = {
            'strategy_id': strategy_info['strategy_id'],
            'request_id': request_id,
            'accuracy': self.feedback_loop.accuracy_history['overall'][-1] if self.feedback_loop.accuracy_history['overall'] else 0,
            'expected_performance': expected_perf,
            'actual_performance': actual_perf
        }
        
        return cycle_result
    
    def apply_regime_specific_refinements(self):
        """
        Apply specialized learning rules for challenging regimes.
        
        Returns:
            Boolean indicating if refinements were applied
        """
        # Special handling for bearish regime which showed poor convergence
        bearish_accuracy = self.feedback_loop.accuracy_history.get('bearish', [])
        
        if bearish_accuracy and len(bearish_accuracy) >= 2:
            recent_accuracy = bearish_accuracy[-1]
            
            # If accuracy is poor, apply more aggressive learning for bearish regime
            if recent_accuracy < 0.4:  # Threshold for poor accuracy
                # Get latest expected vs actual performance
                if not self.feedback_loop.expected_perf or not self.feedback_loop.actual_perf:
                    return False
                    
                expected = self.feedback_loop.expected_perf[-1].get('bearish', {})
                actual = self.feedback_loop.actual_perf[-1].get('bearish', {})
                
                if expected and actual:
                    # Calculate adjustment directions
                    return_diff = actual.get('return', 0) - expected.get('return', 0)
                    vol_diff = actual.get('max_drawdown', 0) - expected.get('max_drawdown', 0)
                    
                    # Get current parameters
                    bearish_params = self.feedback_loop.market_models.get('bearish', {})
                    
                    if bearish_params and 'trend' in bearish_params and 'volatility' in bearish_params:
                        # More aggressive adjustment for trend parameter based on return difference
                        if return_diff > 0:
                            # If actual returns are better than expected, decrease negative trend
                            bearish_params['trend'] *= 0.7  # More aggressive reduction
                        else:
                            # If actual returns are worse than expected, increase negative trend
                            bearish_params['trend'] *= 1.3  # More aggressive increase
                        
                        # Ensure trend remains negative for bearish market
                        bearish_params['trend'] = min(bearish_params['trend'], -0.001)
                        
                        # Volatility adjustment based on drawdown difference
                        if vol_diff > 0:
                            # If actual drawdowns are larger, increase volatility more aggressively
                            bearish_params['volatility'] *= 1.2
                        
                        print("Applied specialized bearish regime adjustments")
                        return True
        
        return False
    
    def check_regime_convergence(self):
        """
        Check if specific regimes have converged to high accuracy.
        
        Returns:
            List of converged regimes or None
        """
        converged_regimes = []
        
        for regime in self.feedback_loop.accuracy_history:
            if regime == 'overall':
                continue
                
            # Need at least 3 data points
            if len(self.feedback_loop.accuracy_history[regime]) < 3:
                continue
                
            recent_accuracy = self.feedback_loop.accuracy_history[regime][-1]
            
            # Consider a regime converged if accuracy > 80%
            if recent_accuracy > 0.8:
                converged_regimes.append(regime)
        
        return converged_regimes if converged_regimes else None
    
    def run_until_convergence(self, max_iterations=20, convergence_threshold=0.85, 
                             stability_window=3, min_iterations=5):
        """
        Run optimization cycles until parameter convergence or max iterations.
        
        Args:
            max_iterations: Maximum number of iterations to run
            convergence_threshold: Accuracy threshold to consider converged
            stability_window: Number of iterations to check for stable accuracy
            min_iterations: Minimum iterations before checking convergence
            
        Returns:
            List of cycle results
        """
        results = []
        
        for i in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"OPTIMIZATION CYCLE {i+1}/{max_iterations}")
            print(f"{'='*50}")
            
            # Run a complete cycle
            cycle_result = self.run_optimization_cycle()
            results.append(cycle_result)
            
            # Show current convergence status
            self.show_convergence_status()
            
            # Check for convergence after minimum iterations
            if i >= min_iterations:
                # Check if accuracy exceeds threshold
                current_accuracy = self.feedback_loop.accuracy_history['overall'][-1]
                
                if current_accuracy >= convergence_threshold:
                    print(f"\nConvergence detected! Accuracy threshold {convergence_threshold:.2%} reached.")
                    break
                    
                # Check for stability (little improvement over window)
                if i >= min_iterations + stability_window:
                    window_accuracies = self.feedback_loop.accuracy_history['overall'][-(stability_window+1):]
                    improvement = window_accuracies[-1] - window_accuracies[0]
                    
                    if abs(improvement) < 0.05:  # Less than 5% change over window
                        print(f"\nStable accuracy detected over {stability_window} iterations. Early stopping.")
                        break
            
            # Check for regime-specific convergence
            regime_convergence = self.check_regime_convergence()
            if regime_convergence:
                print(f"\nRegime-specific convergence detected: {regime_convergence}")
                
                # If most regimes have converged, we can stop
                if len(regime_convergence) >= 3:  # 3 out of 4 regimes
                    print("Majority of regimes have converged. Early stopping.")
                    break
        
        # Generate final report
        self.generate_final_report(results)
        
        return results
    
    def show_convergence_status(self):
        """Display current convergence status."""
        if not self.feedback_loop.param_history:
            print("No parameter history available yet.")
            return
            
        print("\nCurrent Convergence Status:")
        print("-" * 28)
        
        # Show accuracy trend
        if self.feedback_loop.accuracy_history['overall']:
            accuracies = self.feedback_loop.accuracy_history['overall']
            current = accuracies[-1]
            
            print(f"Overall Accuracy: {current:.2%}")
            
            if len(accuracies) > 1:
                improvement = current - accuracies[0]
                print(f"Improvement: {improvement:.2%} from initial {accuracies[0]:.2%}")
        
        # Show key parameter convergence
        for regime, real_params in self.feedback_loop.real_world_params.items():
            if regime not in self.feedback_loop.param_history[0]:
                continue
                
            print(f"\n{regime.capitalize()} Regime Convergence:")
            
            for param in ['trend', 'volatility', 'mean_reversion']:
                if param not in real_params:
                    continue
                    
                initial = self.feedback_loop.param_history[0][regime][param]
                current = self.feedback_loop.market_models[regime][param]
                real = real_params[param]
                
                # Calculate convergence
                initial_diff = abs(initial - real)
                current_diff = abs(current - real)
                
                if initial_diff > 0:
                    convergence = (1 - current_diff / initial_diff) * 100
                    convergence = max(0, convergence)
                else:
                    convergence = 100.0
                
                # Simple text-based visualization
                status = "●" if convergence > 80 else "◐" if convergence > 40 else "○"
                print(f"  {param}: {status} {convergence:.1f}% converged")
    
    def generate_final_report(self, results):
        """
        Generate a comprehensive final report.
        
        Args:
            results: List of cycle results
            
        Returns:
            Path to the report file
        """
        timestamp = int(time.time())
        
        # Create a report directory
        report_dir = os.path.join(self.results_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Export results to JSON
        json_file = os.path.join(report_dir, "workflow_results.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate text report
        report_file = os.path.join(report_dir, "workflow_report.txt")
        
        # Create the report content
        report = []
        report.append("=" * 70)
        report.append("APPROVAL WORKFLOW INTEGRATION REPORT")
        report.append("=" * 70)
        
        # Summary statistics
        report.append("\nSUMMARY STATISTICS")
        report.append("-" * 18)
        report.append(f"Total Optimization Cycles: {len(results)}")
        
        if self.feedback_loop.accuracy_history['overall']:
            final_accuracy = self.feedback_loop.accuracy_history['overall'][-1]
            initial_accuracy = self.feedback_loop.accuracy_history['overall'][0]
            improvement = final_accuracy - initial_accuracy
            
            report.append(f"Initial Prediction Accuracy: {initial_accuracy:.2%}")
            report.append(f"Final Prediction Accuracy: {final_accuracy:.2%}")
            report.append(f"Accuracy Improvement: {improvement:.2%}")
        
        # Parameter convergence
        report.append("\nPARAMETER CONVERGENCE")
        report.append("-" * 20)
        
        param_table = TextVisualization.create_parameter_table(
            self.feedback_loop.param_history,
            self.feedback_loop.real_world_params
        )
        report.append(param_table)
        
        # Accuracy history
        report.append("\nACCURACY HISTORY")
        report.append("-" * 15)
        
        accuracy_table = TextVisualization.create_accuracy_table(
            self.feedback_loop.accuracy_history
        )
        report.append(accuracy_table)
        
        # Performance gap analysis
        report.append("\nPERFORMANCE GAP ANALYSIS")
        report.append("-" * 23)
        
        gap_table = TextVisualization.create_performance_gap_table(
            self.feedback_loop.expected_perf,
            self.feedback_loop.actual_perf
        )
        report.append(gap_table)
        
        # Write the report
        with open(report_file, 'w') as f:
            f.write("\n".join(report))
        
        print(f"\nFinal report generated at: {report_file}")
        return report_file


def main():
    """Run the integrated approval workflow demonstration."""
    print("=" * 70)
    print("INTEGRATED APPROVAL WORKFLOW DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstrates the complete end-to-end system:")
    print("1. Enhanced feedback loop with adaptive learning")
    print("2. Integration with approval workflow")
    print("3. Performance verification and learning")
    print("4. Convergence detection and early stopping")
    print("5. Regime-specific refinements for bearish markets")
    print("\n" + "=" * 70)
    
    # Create the integrated system
    integrated_system = ApprovalWorkflowIntegration()
    
    # Run until convergence
    results = integrated_system.run_until_convergence(
        max_iterations=10,
        convergence_threshold=0.85,
        stability_window=3,
        min_iterations=5
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW INTEGRATION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal optimization cycles: {len(results)}")
    print(f"Final prediction accuracy: {integrated_system.feedback_loop.accuracy_history['overall'][-1]:.2%}")
    
    # Print parameter convergence
    param_table = TextVisualization.create_parameter_table(
        integrated_system.feedback_loop.param_history,
        integrated_system.feedback_loop.real_world_params
    )
    print("\n" + param_table)
    
    print("\n" + "=" * 70)
    print("\nThis completes the demonstration of the integrated approval workflow.")
    print("The system is now ready for extended optimization or real-world testing.")
    
    return integrated_system


if __name__ == "__main__":
    main()
