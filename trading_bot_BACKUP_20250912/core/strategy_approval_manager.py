#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Approval Manager

Handles the go-live process for strategies, including:
1. Approving paper strategies for live trading
2. Managing the transition of positions (close paper positions or transfer to live)
3. Initializing the strategy in live mode with proper allocations
4. Tracking the approval process with auditing
"""

import logging
from enum import Enum
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, OrderSide
from trading_bot.core.strategy_broker_router import StrategyBrokerRouter
from trading_bot.core.strategy_trial_workflow import StrategyTrialWorkflow, StrategyStatus, TrialPhase
from trading_bot.core.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class PositionTransitionMode(Enum):
    """Options for handling positions during paper to live transition."""
    CLOSE_PAPER_START_FLAT = "close_paper_start_flat"  # Close paper positions, start live with no positions
    MIRROR_TO_LIVE = "mirror_to_live"                  # Mirror paper positions to live account at market
    WAIT_FOR_FLAT = "wait_for_flat"                    # Only allow promotion when paper positions are flat
    STAGED_TRANSITION = "staged_transition"            # Close paper positions gradually and build live gradually


class ApprovalResult:
    """Result of a strategy approval operation."""
    
    def __init__(
        self,
        success: bool,
        strategy_id: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize approval result.
        
        Args:
            success: Whether the approval was successful
            strategy_id: Strategy ID
            message: Result message
            details: Additional details
        """
        self.success = success
        self.strategy_id = strategy_id
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "strategy_id": self.strategy_id,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class StrategyApprovalManager:
    """
    Manages the approval and go-live process for strategies.
    
    This class handles the transition from paper trading to live trading,
    including position management, broker switching, and tracking.
    """
    
    def __init__(
        self,
        workflow: StrategyTrialWorkflow,
        router: StrategyBrokerRouter,
        performance_tracker: PerformanceTracker,
        default_transition_mode: PositionTransitionMode = PositionTransitionMode.CLOSE_PAPER_START_FLAT,
        approval_required: bool = True,
        default_position_limit_pct: Optional[float] = 0.02,  # Start with 2% position limit by default
        pre_approval_checks: Optional[List[Callable[[str], Tuple[bool, str]]]] = None
    ):
        """
        Initialize strategy approval manager.
        
        Args:
            workflow: Strategy trial workflow
            router: Strategy broker router
            performance_tracker: Performance tracker
            default_transition_mode: Default mode for position transition
            approval_required: Whether manual approval is required
            default_position_limit_pct: Default position limit percentage for newly approved strategies
            pre_approval_checks: List of functions that perform pre-approval validation
        """
        self._workflow = workflow
        self._router = router
        self._performance_tracker = performance_tracker
        self._default_transition_mode = default_transition_mode
        self._approval_required = approval_required
        self._default_position_limit_pct = default_position_limit_pct
        self._pre_approval_checks = pre_approval_checks or []
        
        # Get event bus
        self._event_bus = ServiceRegistry.get_instance().get_service(EventBus)
        
        # Approval request tracking
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
        
        # Approval history
        self._approval_history: List[Dict[str, Any]] = []
        
        logger.info("Strategy approval manager initialized")
    
    def request_approval(
        self,
        strategy_id: str,
        requested_by: str,
        notes: Optional[str] = None
    ) -> ApprovalResult:
        """
        Request approval for a strategy to go live.
        
        Args:
            strategy_id: Strategy ID
            requested_by: User who requested approval
            notes: Optional notes about the request
            
        Returns:
            ApprovalResult: Result of the request
        """
        # Check if strategy exists
        metadata = self._workflow.get_strategy_metadata(strategy_id)
        if not metadata:
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Strategy {strategy_id} not found"
            )
        
        # Check current status
        current_status = metadata.get("status")
        if current_status not in [StrategyStatus.PAPER_TESTING.value, StrategyStatus.READY_FOR_REVIEW.value]:
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Strategy {strategy_id} is not eligible for approval (status: {current_status})"
            )
        
        # Check if already approved or pending
        if current_status == StrategyStatus.APPROVED.value:
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Strategy {strategy_id} is already approved"
            )
        
        if strategy_id in self._pending_approvals:
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Strategy {strategy_id} already has a pending approval request"
            )
        
        # Create approval request
        request = {
            "strategy_id": strategy_id,
            "requested_by": requested_by,
            "requested_at": datetime.now().isoformat(),
            "notes": notes,
            "status": "pending"
        }
        
        self._pending_approvals[strategy_id] = request
        
        # If no approval required, auto-approve
        if not self._approval_required:
            return self.approve_strategy(
                strategy_id=strategy_id,
                approved_by="system",
                transition_mode=self._default_transition_mode,
                position_limit_pct=self._default_position_limit_pct,
                notes="Auto-approved (no approval required)"
            )
        
        # Publish event
        self._publish_approval_event(
            strategy_id=strategy_id,
            event_type="approval_requested",
            details={
                "requested_by": requested_by,
                "requested_at": request["requested_at"],
                "notes": notes
            }
        )
        
        logger.info(f"Approval requested for strategy {strategy_id} by {requested_by}")
        
        return ApprovalResult(
            success=True,
            strategy_id=strategy_id,
            message=f"Approval requested for strategy {strategy_id}",
            details={"request": request}
        )
    
    def approve_strategy(
        self,
        strategy_id: str,
        approved_by: str,
        transition_mode: Optional[PositionTransitionMode] = None,
        position_limit_pct: Optional[float] = None,
        notes: Optional[str] = None,
        live_broker_id: Optional[str] = None
    ) -> ApprovalResult:
        """
        Approve a strategy for live trading.
        
        Args:
            strategy_id: Strategy ID
            approved_by: User who approved
            transition_mode: How to handle existing positions
            position_limit_pct: Position size limit as percentage
            notes: Optional notes about the approval
            live_broker_id: Specific broker to use for live trading
            
        Returns:
            ApprovalResult: Result of the approval
        """
        # Check if strategy exists
        metadata = self._workflow.get_strategy_metadata(strategy_id)
        if not metadata:
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Strategy {strategy_id} not found"
            )
        
        # Run pre-approval checks
        for check_func in self._pre_approval_checks:
            passed, message = check_func(strategy_id)
            if not passed:
                return ApprovalResult(
                    success=False,
                    strategy_id=strategy_id,
                    message=f"Pre-approval check failed: {message}"
                )
        
        # If transition mode requires flat positions, check if positions are flat
        mode = transition_mode or self._default_transition_mode
        if mode == PositionTransitionMode.WAIT_FOR_FLAT:
            positions = self._router.get_positions_for_strategy(strategy_id)
            if positions and any(p.get("quantity", 0) != 0 for p in positions):
                return ApprovalResult(
                    success=False,
                    strategy_id=strategy_id,
                    message=f"Strategy {strategy_id} has open positions and transition mode is WAIT_FOR_FLAT"
                )
        
        # Update status to APPROVED
        self._workflow.update_strategy_status(
            strategy_id=strategy_id,
            status=StrategyStatus.APPROVED,
            remarks=f"Approved by {approved_by}: {notes or 'No notes provided'}"
        )
        
        # Handle position transition
        transition_result = self._handle_position_transition(
            strategy_id=strategy_id,
            mode=mode,
            live_broker_id=live_broker_id
        )
        
        if not transition_result.get("success", False):
            logger.error(f"Position transition failed for {strategy_id}: {transition_result.get('message')}")
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Position transition failed: {transition_result.get('message')}",
                details={"transition_result": transition_result}
            )
        
        # Promote to live
        promotion_success = self._workflow.promote_to_live(
            strategy_id=strategy_id,
            live_broker_id=live_broker_id,
            position_limit_pct=position_limit_pct or self._default_position_limit_pct,
            approved_by=approved_by,
            remarks=notes
        )
        
        if not promotion_success:
            logger.error(f"Promotion failed for {strategy_id}")
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"Promotion failed for strategy {strategy_id}",
                details={"transition_result": transition_result}
            )
        
        # Update pending approval status
        if strategy_id in self._pending_approvals:
            self._pending_approvals[strategy_id]["status"] = "approved"
            self._pending_approvals[strategy_id]["approved_by"] = approved_by
            self._pending_approvals[strategy_id]["approved_at"] = datetime.now().isoformat()
            self._pending_approvals[strategy_id]["transition_mode"] = mode.value
            
            # Add to approval history
            self._approval_history.append(self._pending_approvals[strategy_id])
            
            # Remove from pending
            del self._pending_approvals[strategy_id]
        
        # Publish event
        self._publish_approval_event(
            strategy_id=strategy_id,
            event_type="strategy_approved",
            details={
                "approved_by": approved_by,
                "transition_mode": mode.value,
                "position_limit_pct": position_limit_pct or self._default_position_limit_pct,
                "notes": notes,
                "transition_result": transition_result
            }
        )
        
        logger.info(f"Strategy {strategy_id} approved by {approved_by} and promoted to live")
        
        return ApprovalResult(
            success=True,
            strategy_id=strategy_id,
            message=f"Strategy {strategy_id} approved and promoted to live",
            details={
                "transition_result": transition_result,
                "position_limit_pct": position_limit_pct or self._default_position_limit_pct
            }
        )
    
    def reject_approval(
        self,
        strategy_id: str,
        rejected_by: str,
        reason: str
    ) -> ApprovalResult:
        """
        Reject a pending approval request.
        
        Args:
            strategy_id: Strategy ID
            rejected_by: User who rejected
            reason: Reason for rejection
            
        Returns:
            ApprovalResult: Result of the rejection
        """
        if strategy_id not in self._pending_approvals:
            return ApprovalResult(
                success=False,
                strategy_id=strategy_id,
                message=f"No pending approval request for strategy {strategy_id}"
            )
        
        # Update pending approval status
        self._pending_approvals[strategy_id]["status"] = "rejected"
        self._pending_approvals[strategy_id]["rejected_by"] = rejected_by
        self._pending_approvals[strategy_id]["rejected_at"] = datetime.now().isoformat()
        self._pending_approvals[strategy_id]["rejection_reason"] = reason
        
        # Add to approval history
        self._approval_history.append(self._pending_approvals[strategy_id])
        
        # Remove from pending
        del self._pending_approvals[strategy_id]
        
        # Publish event
        self._publish_approval_event(
            strategy_id=strategy_id,
            event_type="approval_rejected",
            details={
                "rejected_by": rejected_by,
                "reason": reason
            }
        )
        
        logger.info(f"Approval rejected for strategy {strategy_id} by {rejected_by}: {reason}")
        
        return ApprovalResult(
            success=True,
            strategy_id=strategy_id,
            message=f"Approval rejected for strategy {strategy_id}",
            details={"reason": reason}
        )
    
    def get_pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all pending approval requests.
        
        Returns:
            Dict[str, Dict[str, Any]]: Pending approvals by strategy ID
        """
        return self._pending_approvals.copy()
    
    def get_approval_history(self) -> List[Dict[str, Any]]:
        """
        Get approval history.
        
        Returns:
            List[Dict[str, Any]]: Approval history
        """
        return self._approval_history.copy()
    
    def get_strategies_eligible_for_approval(self) -> List[Dict[str, Any]]:
        """
        Get all strategies eligible for approval.
        
        Returns:
            List[Dict[str, Any]]: Eligible strategies
        """
        eligible = []
        
        # Get all strategies
        all_metadata = self._workflow.get_all_strategy_metadata()
        
        for strategy_id, metadata in all_metadata.items():
            # Skip if already pending approval
            if strategy_id in self._pending_approvals:
                continue
            
            # Check if eligible based on status
            status = metadata.get("status")
            if status in [StrategyStatus.PAPER_TESTING.value, StrategyStatus.READY_FOR_REVIEW.value]:
                eligible.append(metadata)
        
        return eligible
    
    def _handle_position_transition(
        self,
        strategy_id: str,
        mode: PositionTransitionMode,
        live_broker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle transition of positions from paper to live.
        
        Args:
            strategy_id: Strategy ID
            mode: Position transition mode
            live_broker_id: Live broker ID
            
        Returns:
            Dict: Result of position transition
        """
        # Get current paper positions
        paper_positions = self._router.get_positions_for_strategy(strategy_id)
        
        if not paper_positions:
            return {
                "success": True,
                "message": "No paper positions to transition",
                "positions_closed": 0
            }
        
        # Handle based on mode
        if mode == PositionTransitionMode.CLOSE_PAPER_START_FLAT:
            return self._close_paper_positions(strategy_id, paper_positions)
            
        elif mode == PositionTransitionMode.MIRROR_TO_LIVE:
            return self._mirror_positions_to_live(strategy_id, paper_positions, live_broker_id)
            
        elif mode == PositionTransitionMode.WAIT_FOR_FLAT:
            # Already checked earlier, just verify again
            if any(p.get("quantity", 0) != 0 for p in paper_positions):
                return {
                    "success": False,
                    "message": "Strategy has open positions and transition mode is WAIT_FOR_FLAT"
                }
            return {
                "success": True,
                "message": "No open positions to transition"
            }
            
        elif mode == PositionTransitionMode.STAGED_TRANSITION:
            return {
                "success": True,
                "message": "Staged transition initiated - paper positions will be closed gradually",
                "positions": paper_positions
            }
        
        return {
            "success": False,
            "message": f"Unknown transition mode: {mode}"
        }
    
    def _close_paper_positions(
        self,
        strategy_id: str,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Close all paper positions for a strategy.
        
        Args:
            strategy_id: Strategy ID
            positions: Current positions
            
        Returns:
            Dict: Result of position closing
        """
        closed_positions = 0
        
        for position in positions:
            symbol = position.get("symbol")
            quantity = position.get("quantity", 0)
            
            if quantity == 0:
                continue
                
            # Determine side for closing
            side = "sell" if quantity > 0 else "buy"
            abs_quantity = abs(quantity)
            
            # Place closing order
            try:
                result = self._router.place_order_for_strategy(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    side=side,
                    quantity=abs_quantity,
                    order_type="market",
                    tags=["PAPER", "POSITION_CLOSE", "TRANSITION"]
                )
                
                if result:
                    closed_positions += 1
                
            except Exception as e:
                logger.error(f"Error closing position for {strategy_id} {symbol}: {str(e)}")
        
        return {
            "success": True,
            "message": f"Closed {closed_positions} paper positions",
            "positions_closed": closed_positions
        }
    
    def _mirror_positions_to_live(
        self,
        strategy_id: str,
        paper_positions: List[Dict[str, Any]],
        live_broker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mirror paper positions to live account.
        
        Args:
            strategy_id: Strategy ID
            paper_positions: Current paper positions
            live_broker_id: Live broker ID
            
        Returns:
            Dict: Result of position mirroring
        """
        # This would require setting up the strategy with the live broker first,
        # then placing equivalent orders - for this implementation, we'll
        # just simulate the concept
        
        mirrored_positions = 0
        
        for position in paper_positions:
            symbol = position.get("symbol")
            quantity = position.get("quantity", 0)
            
            if quantity == 0:
                continue
                
            # Determine side
            side = "buy" if quantity > 0 else "sell"
            abs_quantity = abs(quantity)
            
            # Record that we would create this position
            logger.info(f"Would mirror position for {strategy_id} {symbol}: {side} {abs_quantity}")
            mirrored_positions += 1
        
        return {
            "success": True,
            "message": f"Mirrored {mirrored_positions} positions to live account (simulated)",
            "positions_mirrored": mirrored_positions
        }
    
    def _publish_approval_event(self, strategy_id: str, event_type: str, details: Dict[str, Any]) -> None:
        """
        Publish an approval event.
        
        Args:
            strategy_id: Strategy ID
            event_type: Event type
            details: Event details
        """
        if not self._event_bus:
            return
        
        event_data = {
            "strategy_id": strategy_id,
            "approval_event": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        try:
            self._event_bus.publish(Event(
                event_type=EventType.APPROVAL_EVENT,
                data=event_data
            ))
        except Exception as e:
            logger.error(f"Error publishing approval event: {str(e)}")
