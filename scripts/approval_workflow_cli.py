#!/usr/bin/env python3
"""
Approval Workflow CLI

A simple command-line interface for reviewers to interact with approval requests.
This tool allows reviewers to:
- List pending approval requests
- Display details about specific requests
- Approve or reject requests with comments

Usage:
    python approval_workflow_cli.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add the project root to the sys path to enable imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import approval workflow and A/B testing components
from trading_bot.autonomous.approval_workflow import (
    get_approval_workflow_manager, ApprovalStatus, ApprovalRequest
)
from trading_bot.autonomous.ab_testing_core import (
    ABTest
)
from trading_bot.autonomous.ab_testing_manager import (
    get_ab_test_manager
)
from trading_bot.autonomous.synthetic_market_generator import (
    MarketRegimeType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ApprovalCLI")


class ApprovalWorkflowCLI:
    """Command-line interface for approval workflow interaction."""

    def __init__(self):
        """Initialize the CLI with necessary components."""
        self.approval_manager = get_approval_workflow_manager()
        self.ab_test_manager = get_ab_test_manager()
        self.running = False
        self.commands = {
            'list': self.list_requests,
            'detail': self.show_request_detail,
            'approve': self.approve_request,
            'reject': self.reject_request,
            'help': self.show_help,
            'exit': self.exit_cli,
            'quit': self.exit_cli,
        }

    def start(self):
        """Start the CLI loop."""
        self.running = True
        self.show_welcome()
        
        while self.running:
            try:
                command = input("\nEnter command (or 'help' for commands): ").strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in self.commands:
                    self.commands[cmd](*args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                self.running = False
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def show_welcome(self):
        """Display welcome message and summary of pending requests."""
        print("\n==================================================")
        print("Welcome to the Approval Workflow CLI")
        print("==================================================")
        
        # Show summary of pending requests
        pending_requests = self.approval_manager.list_requests(status=ApprovalStatus.PENDING)
        print(f"\nThere are {len(pending_requests)} pending approval requests.")
        
        if pending_requests:
            print("\nPending requests:")
            for i, req in enumerate(pending_requests[:5], 1):
                print(f"  {i}. ID: {req.request_id[:8]}... - Strategy: {req.strategy_id} - Version: {req.version_id}")
            
            if len(pending_requests) > 5:
                print(f"  ... and {len(pending_requests) - 5} more.")
            
            print("\nUse 'list' to see all pending requests.")
    
    def show_help(self, *args):
        """Show available commands."""
        print("\nAvailable commands:")
        print("  list [status]       - List requests (status: pending, approved, rejected, all)")
        print("  detail <request_id> - Show details of a specific request")
        print("  approve <request_id> [comments] - Approve a request with optional comments")
        print("  reject <request_id> [comments]  - Reject a request with optional comments")
        print("  help                - Show this help message")
        print("  exit                - Exit the CLI")
        print("\nSynthetic market testing results will be shown automatically when available.")
    
    def list_requests(self, status_arg="pending"):
        """
        List approval requests, filtered by status.
        
        Args:
            status_arg: Status filter ("pending", "approved", "rejected", "all")
        """
        # Map status argument to ApprovalStatus
        status_map = {
            "pending": ApprovalStatus.PENDING,
            "approved": ApprovalStatus.APPROVED,
            "rejected": ApprovalStatus.REJECTED,
            "all": None
        }
        
        if status_arg not in status_map:
            print(f"Unknown status: {status_arg}. Valid options: pending, approved, rejected, all")
            return
        
        status = status_map[status_arg]
        requests = self.approval_manager.list_requests(status=status)
        
        if not requests:
            print(f"No {status_arg} requests found.")
            return
        
        print(f"\n{len(requests)} {status_arg} request(s):")
        print("{:<12} {:<20} {:<15} {:<15} {:<20}".format(
            "ID", "Test", "Strategy", "Version", "Request Time"
        ))
        print("-" * 85)
        
        for req in requests:
            # Format date for display
            req_time = req.request_time.strftime("%Y-%m-%d %H:%M")
            print("{:<12} {:<20} {:<15} {:<15} {:<20}".format(
                req.request_id[:8] + "...", 
                req.test_id[:16] + "..." if len(req.test_id) > 16 else req.test_id,
                req.strategy_id,
                req.version_id,
                req_time
            ))
    
    def show_request_detail(self, *args):
        """
        Show detailed information about a specific request.
        
        Args:
            args[0]: Request ID
        """
        if not args:
            print("Error: Missing request ID. Usage: detail <request_id>")
            return
        
        request_id = args[0]
        request = self.approval_manager.get_request(request_id)
        
        if not request:
            print(f"Request with ID {request_id} not found.")
            return
        
        # Get test details
        test = self.ab_test_manager.get_test(request.test_id)
        
        print("\n==================================================")
        print(f"Approval Request: {request.request_id}")
        print("==================================================")
        print(f"Status: {request.status.value}")
        print(f"Test ID: {request.test_id}")
        print(f"Strategy ID: {request.strategy_id}")
        print(f"Version ID: {request.version_id}")
        print(f"Requester: {request.requester}")
        print(f"Request Time: {request.request_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if request.status != ApprovalStatus.PENDING:
            print(f"Reviewer: {request.reviewer}")
            print(f"Decision Time: {request.decision_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Comments: {request.comments or 'None'}")
        
        # Show test details if available
        if test:
            print("\nTest Details:")
            print(f"Name: {test.name}")
            print(f"Status: {test.status.value}")
            print(f"Start Time: {test.start_time.strftime('%Y-%m-%d %H:%M:%S') if test.start_time else 'Not started'}")
            print(f"End Time: {test.end_time.strftime('%Y-%m-%d %H:%M:%S') if test.end_time else 'Not ended'}")
            
            # Show variant details
            if test.variant_a and test.variant_b:
                self._print_variant_comparison(test, show_synthetic=True)
            
        else:
            print("\nTest details not available.")
    
    def _print_variant_comparison(self, test: ABTest, show_synthetic=False):
        """Print comparison table between variants."""
        # Extract metrics for comparison
        metrics_a = None
        metrics_b = None
        
        if test.metrics and hasattr(test.metrics, "variant_a_metrics"):
            metrics_a = test.metrics.variant_a_metrics
        
        if test.metrics and hasattr(test.metrics, "variant_b_metrics"):
            metrics_b = test.metrics.variant_b_metrics
        
        # Create a formatted comparison table
        if metrics_a or metrics_b:
            print("{:<20} {:<15} {:<15} {:<15} {}".format(
                "Metric", "Variant A", "Variant B", "Difference", "Recommendation" if show_synthetic else ""
            ))
            print("-" * (70 + (15 if show_synthetic else 0)))
            
            # Combine all metrics keys
            all_metrics = set()
            if metrics_a and hasattr(metrics_a, "__dict__"):
                all_metrics.update(metrics_a.__dict__.keys())
            if metrics_b and hasattr(metrics_b, "__dict__"):
                all_metrics.update(metrics_b.__dict__.keys())
            
            # Filter out special/private attributes
            all_metrics = [m for m in all_metrics if not m.startswith("_")]
            
            # Get synthetic recommendations if available
            regime_recommendations = {}
            if show_synthetic and "synthetic_testing_results" in test.metadata:
                results = test.metadata["synthetic_testing_results"]
                for regime, data in results.items():
                    if "comparison" in data and "b_is_better" in data["comparison"]:
                        regime_recommendations[regime] = data["comparison"]["b_is_better"]
            
            # Print each metric with comparison
            for metric in sorted(all_metrics):
                # Get values or None if not available
                value_a = getattr(metrics_a, metric, None) if metrics_a else None
                value_b = getattr(metrics_b, metric, None) if metrics_b else None
                
                # Calculate difference if possible
                diff = ""
                if value_a is not None and value_b is not None:
                    try:
                        diff_val = float(value_b) - float(value_a)
                        diff = f"{diff_val:+.4f}"
                    except (ValueError, TypeError):
                        diff = "N/A"
                
                # Format values for display
                disp_a = f"{value_a:.4f}" if isinstance(value_a, float) else str(value_a)
                disp_b = f"{value_b:.4f}" if isinstance(value_b, float) else str(value_b)
                
                # Add recommendation indicator if applicable
                recommendation = ""
                if show_synthetic and metric in ["sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]:
                    if diff and diff != "N/A" and diff.startswith("+"):
                        recommendation = "B outperforms →"
                    elif diff and diff != "N/A" and diff.startswith("-"):
                        recommendation = "← A outperforms"
                
                print("{:<20} {:<15} {:<15} {:<15} {}".format(
                    metric, disp_a, disp_b, diff, recommendation
                ))
        else:
            print("\nNo metrics available for comparison.")
    
    def approve_request(self, *args):
        """
        Approve a request.
        
        Args:
            args[0]: Request ID
            args[1:]: Comments (joined with spaces)
        """
        if not args:
            print("Error: Missing request ID. Usage: approve <request_id> [comments]")
            return
        
        request_id = args[0]
        comments = " ".join(args[1:]) if len(args) > 1 else None
        
        # Get reviewer name - in a real system, this would be the authenticated user
        reviewer = os.environ.get("USER", "cli_user")
        
        # Approve the request
        success = self.approval_manager.approve_request(
            request_id=request_id,
            reviewer=reviewer,
            comments=comments
        )
        
        if success:
            print(f"Request {request_id} approved successfully.")
        else:
            print(f"Failed to approve request {request_id}. It may not exist or is not in PENDING state.")
    
    def reject_request(self, *args):
        """
        Reject a request.
        
        Args:
            args[0]: Request ID
            args[1:]: Comments (joined with spaces)
        """
        if not args:
            print("Error: Missing request ID. Usage: reject <request_id> [comments]")
            return
        
        request_id = args[0]
        comments = " ".join(args[1:]) if len(args) > 1 else None
        
        # Get reviewer name - in a real system, this would be the authenticated user
        reviewer = os.environ.get("USER", "cli_user")
        
        # Reject the request
        success = self.approval_manager.reject_request(
            request_id=request_id,
            reviewer=reviewer,
            comments=comments
        )
        
        if success:
            print(f"Request {request_id} rejected successfully.")
        else:
            print(f"Failed to reject request {request_id}. It may not exist or is not in PENDING state.")
    
    def _print_synthetic_testing_summary(self, test: ABTest):
        """Print summary of synthetic testing results."""
        if not test.metadata.get("synthetic_testing_results"):
            print("No synthetic testing results available.")
            return
        
        results = test.metadata["synthetic_testing_results"]
        regimes_tested = len(results)
        
        # Get regime summary if available
        regime_summary = None
        if "synthetic_testing_results" in test.metadata:
            for regime_data in results.values():
                if "comparison" in regime_data:
                    regime_summary = self._generate_regime_summary(results)
                    break
        
        # Display overall recommendation if available
        if regime_summary:
            regimes_b_better = regime_summary.get("regimes_b_better", 0)
            total_regimes = regime_summary.get("total_regimes", 0)
            promote_b = regime_summary.get("promote_b", False)
            confidence = regime_summary.get("confidence", "low")
            
            print(f"Tested across {total_regimes} market regimes")
            print(f"Variant B performed better in {regimes_b_better}/{total_regimes} regimes")
            print(f"Overall recommendation: {'Promote Variant B' if promote_b else 'Keep Variant A'}")
            print(f"Confidence: {confidence.upper()}\n")
        
        # Display regime-specific results
        print("Regime-Specific Performance:")
        print("{:<15} {:<15} {:<15}".format("Regime", "Better Variant", "Confidence"))
        print("-" * 45)
        
        for regime, data in results.items():
            if "comparison" in data:
                comparison = data["comparison"]
                b_is_better = comparison.get("b_is_better", False)
                better_variant = "B" if b_is_better else "A"
                confidence = comparison.get("confidence_score", 0.0)
                confidence_str = f"{confidence:.2f}"
                
                print("{:<15} {:<15} {:<15}".format(
                    regime, better_variant, confidence_str
                ))
        
        # Display regime-switching recommendation if available
        regime_recommendations = test.metadata.get("regime_switching_recommendation", {})
        if regime_recommendations and regime_recommendations.get("regime_switching_recommended", False):
            print("\n⚠️ REGIME-SPECIFIC STRATEGY RECOMMENDED")
            print("This strategy shows significantly different performance across market regimes.")
            print("Consider implementing a regime-switching approach.")
        
        print("\nKey observations:")
        # Show best and worst regimes for each variant
        best_regime_b = max(results.items(), key=lambda x: 
                          x[1]["comparison"]["differences"]["sharpe_ratio"] 
                          if "comparison" in x[1] and "differences" in x[1]["comparison"] 
                          and "sharpe_ratio" in x[1]["comparison"]["differences"] else -float('inf'))
        
        worst_regime_b = min(results.items(), key=lambda x: 
                           x[1]["comparison"]["differences"]["sharpe_ratio"] 
                           if "comparison" in x[1] and "differences" in x[1]["comparison"] 
                           and "sharpe_ratio" in x[1]["comparison"]["differences"] else float('inf'))
        
        if "comparison" in best_regime_b[1] and "differences" in best_regime_b[1]["comparison"]:
            print(f"- Variant B performs best in {best_regime_b[0]} markets")
        
        if "comparison" in worst_regime_b[1] and "differences" in worst_regime_b[1]["comparison"]:
            print(f"- Variant B performs worst in {worst_regime_b[0]} markets")
    
    def _generate_regime_summary(self, regime_results):
        """Generate a summary from regime-specific testing results."""
        # Count regimes where B is better
        regimes_b_better = 0
        total_regimes = len(regime_results)
        
        for regime, results in regime_results.items():
            if "comparison" in results and results["comparison"].get("b_is_better", False):
                regimes_b_better += 1
        
        # Calculate overall recommendation
        promote_b = (regimes_b_better / total_regimes) >= 0.75 if total_regimes > 0 else False
        confidence = "high" if (regimes_b_better / total_regimes) >= 0.9 else "medium"
        
        return {
            "total_regimes": total_regimes,
            "regimes_b_better": regimes_b_better,
            "promote_b": promote_b,
            "confidence": confidence
        }
    
    def exit_cli(self, *args):
        """Exit the CLI."""
        print("Exiting the Approval Workflow CLI. Goodbye!")
        self.running = False


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Approval Workflow CLI")
    
    # Optional batch mode arguments
    parser.add_argument("--list", action="store_true", help="List pending requests and exit")
    parser.add_argument("--approve", metavar="REQUEST_ID", help="Approve specified request and exit")
    parser.add_argument("--reject", metavar="REQUEST_ID", help="Reject specified request and exit")
    parser.add_argument("--comments", metavar="TEXT", help="Comments for approve/reject actions")
    
    args = parser.parse_args()
    cli = ApprovalWorkflowCLI()
    
    # Handle batch mode if any arguments were provided
    if args.list:
        cli.list_requests("pending")
        return
    
    if args.approve:
        comments = args.comments or ""
        cli.approve_request(args.approve, comments)
        return
    
    if args.reject:
        comments = args.comments or ""
        cli.reject_request(args.reject, comments)
        return
    
    # Start interactive mode if no batch arguments were provided
    cli.start()


if __name__ == "__main__":
    main()
