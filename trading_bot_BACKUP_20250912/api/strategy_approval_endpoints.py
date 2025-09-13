#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Approval API Endpoints

Provides REST API endpoints for strategy approval workflow:
1. Listing strategies eligible for approval
2. Requesting approval
3. Approving/rejecting strategies
4. Handling position transitions
5. Getting approval history
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from pydantic import BaseModel, Field

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.strategy_approval_manager import (
    StrategyApprovalManager, PositionTransitionMode, ApprovalResult
)
from trading_bot.core.strategy_trial_workflow import StrategyStatus

logger = logging.getLogger(__name__)

# Define API models
class ApprovalRequestModel(BaseModel):
    """Model for approval request."""
    strategy_id: str = Field(..., description="Strategy ID")
    requested_by: str = Field(..., description="User requesting approval")
    notes: Optional[str] = Field(None, description="Notes about the request")

class ApprovalActionModel(BaseModel):
    """Model for approval action."""
    strategy_id: str = Field(..., description="Strategy ID")
    approved_by: str = Field(..., description="User approving/rejecting")
    notes: Optional[str] = Field(None, description="Notes about the action")
    position_transition_mode: str = Field(
        "CLOSE_PAPER_START_FLAT", 
        description="How to handle open positions"
    )
    position_limit_pct: Optional[float] = Field(
        None, 
        description="Position size limit as percentage"
    )
    live_broker_id: Optional[str] = Field(
        None, 
        description="Specific broker for live trading"
    )

class RejectionModel(BaseModel):
    """Model for rejection."""
    strategy_id: str = Field(..., description="Strategy ID")
    rejected_by: str = Field(..., description="User rejecting")
    reason: str = Field(..., description="Reason for rejection")

class ApiResponseModel(BaseModel):
    """Generic API response model."""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


# Create router
router = APIRouter(
    prefix="/api/strategies/approval",
    tags=["strategy_approval"],
    responses={404: {"description": "Not found"}},
)


# Dependency to get approval manager
def get_approval_manager() -> StrategyApprovalManager:
    """Get approval manager from service registry."""
    service_registry = ServiceRegistry.get_instance()
    approval_manager = service_registry.get_service(StrategyApprovalManager)
    if not approval_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Strategy approval service not available"
        )
    return approval_manager


@router.get(
    "/eligible",
    response_model=ApiResponseModel,
    summary="Get strategies eligible for approval",
    description="Returns a list of strategies that are eligible for approval"
)
async def get_eligible_strategies(
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Get strategies eligible for approval."""
    try:
        eligible = approval_manager.get_strategies_eligible_for_approval()
        return {
            "success": True,
            "message": f"Found {len(eligible)} eligible strategies",
            "data": {"strategies": eligible}
        }
    except Exception as e:
        logger.error(f"Error getting eligible strategies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting eligible strategies: {str(e)}"
        )


@router.get(
    "/pending",
    response_model=ApiResponseModel,
    summary="Get pending approval requests",
    description="Returns a list of strategies with pending approval requests"
)
async def get_pending_approvals(
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Get pending approval requests."""
    try:
        pending = approval_manager.get_pending_approvals()
        return {
            "success": True,
            "message": f"Found {len(pending)} pending approval requests",
            "data": {"pending_approvals": pending}
        }
    except Exception as e:
        logger.error(f"Error getting pending approvals: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting pending approvals: {str(e)}"
        )


@router.get(
    "/history",
    response_model=ApiResponseModel,
    summary="Get approval history",
    description="Returns the history of approval requests and actions"
)
async def get_approval_history(
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Get approval history."""
    try:
        history = approval_manager.get_approval_history()
        return {
            "success": True,
            "message": f"Found {len(history)} approval history entries",
            "data": {"history": history}
        }
    except Exception as e:
        logger.error(f"Error getting approval history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting approval history: {str(e)}"
        )


@router.post(
    "/request",
    response_model=ApiResponseModel,
    summary="Request approval for a strategy",
    description="Request approval for a strategy to transition from paper to live"
)
async def request_approval(
    request: ApprovalRequestModel,
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Request approval for a strategy."""
    try:
        result = approval_manager.request_approval(
            strategy_id=request.strategy_id,
            requested_by=request.requested_by,
            notes=request.notes
        )
        
        if not result.success:
            return {
                "success": False,
                "message": result.message,
                "data": result.details
            }
        
        return {
            "success": True,
            "message": result.message,
            "data": result.details
        }
    except Exception as e:
        logger.error(f"Error requesting approval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error requesting approval: {str(e)}"
        )


@router.post(
    "/approve",
    response_model=ApiResponseModel,
    summary="Approve a strategy for live trading",
    description="Approve a strategy to transition from paper to live trading"
)
async def approve_strategy(
    approval: ApprovalActionModel,
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Approve a strategy for live trading."""
    try:
        # Convert position transition mode string to enum
        try:
            transition_mode = PositionTransitionMode[approval.position_transition_mode]
        except (KeyError, ValueError):
            return {
                "success": False,
                "message": f"Invalid position transition mode: {approval.position_transition_mode}",
                "data": {
                    "valid_modes": [mode.name for mode in PositionTransitionMode]
                }
            }
        
        result = approval_manager.approve_strategy(
            strategy_id=approval.strategy_id,
            approved_by=approval.approved_by,
            transition_mode=transition_mode,
            position_limit_pct=approval.position_limit_pct,
            notes=approval.notes,
            live_broker_id=approval.live_broker_id
        )
        
        if not result.success:
            return {
                "success": False,
                "message": result.message,
                "data": result.details
            }
        
        return {
            "success": True,
            "message": result.message,
            "data": result.details
        }
    except Exception as e:
        logger.error(f"Error approving strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error approving strategy: {str(e)}"
        )


@router.post(
    "/reject",
    response_model=ApiResponseModel,
    summary="Reject an approval request",
    description="Reject a pending approval request"
)
async def reject_approval(
    rejection: RejectionModel,
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Reject an approval request."""
    try:
        result = approval_manager.reject_approval(
            strategy_id=rejection.strategy_id,
            rejected_by=rejection.rejected_by,
            reason=rejection.reason
        )
        
        if not result.success:
            return {
                "success": False,
                "message": result.message,
                "data": result.details
            }
        
        return {
            "success": True,
            "message": result.message,
            "data": result.details
        }
    except Exception as e:
        logger.error(f"Error rejecting approval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rejecting approval: {str(e)}"
        )


@router.get(
    "/{strategy_id}/status",
    response_model=ApiResponseModel,
    summary="Get strategy approval status",
    description="Get the current approval status of a strategy"
)
async def get_strategy_approval_status(
    strategy_id: str = Path(..., description="Strategy ID"),
    approval_manager: StrategyApprovalManager = Depends(get_approval_manager)
):
    """Get strategy approval status."""
    try:
        # Check if in pending approvals
        pending = approval_manager.get_pending_approvals()
        if strategy_id in pending:
            return {
                "success": True,
                "message": f"Strategy {strategy_id} has pending approval",
                "data": {"status": "pending", "details": pending[strategy_id]}
            }
        
        # Check strategy status in workflow
        workflow = ServiceRegistry.get_instance().get_service("StrategyTrialWorkflow")
        if workflow:
            metadata = workflow.get_strategy_metadata(strategy_id)
            if metadata:
                status = metadata.get("status")
                return {
                    "success": True,
                    "message": f"Strategy {strategy_id} status: {status}",
                    "data": {
                        "status": status,
                        "metadata": metadata
                    }
                }
        
        return {
            "success": False,
            "message": f"Strategy {strategy_id} not found or no approval data available",
            "data": None
        }
    except Exception as e:
        logger.error(f"Error getting strategy approval status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting strategy approval status: {str(e)}"
        )
