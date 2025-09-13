#!/usr/bin/env python3
"""
Approval Workflow Component (Minimal Skeleton)

This module introduces a minimal Approval Workflow manager that will allow
human operators to review and approve or reject strategy promotions suggested
by the A/B Testing Framework.

Increment 1 (small chunk):
    • Core data structures (`ApprovalStatus`, `ApprovalRequest`).
    • In-memory manager (`ApprovalWorkflowManager`) with CRUD operations.
    • Singleton accessor (`get_approval_workflow_manager`).

Next increments will add:
    • Persistence to disk.
    • Event-bus integration for notifications.
    • Role-based access & simple CLI / REST hooks.
"""

from __future__ import annotations

import uuid
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
import json
from pathlib import Path

import logging

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_FILE = Path.home() / ".trading_bot" / "approval_requests.json"

class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    """Represents a single approval request for strategy promotion."""

    test_id: str
    strategy_id: str
    version_id: str
    requester: str = "ab_testing_system"
    request_time: datetime = field(default_factory=datetime.utcnow)

    # Auto-generated fields
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: Optional[str] = None
    decision_time: Optional[datetime] = None
    comments: Optional[str] = None

    # Convenience helpers -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "strategy_id": self.strategy_id,
            "version_id": self.version_id,
            "requester": self.requester,
            "request_time": self.request_time.isoformat(),
            "request_id": self.request_id,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "decision_time": self.decision_time.isoformat() if self.decision_time else None,
            "comments": self.comments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalRequest":
        obj = cls(
            test_id=data["test_id"],
            strategy_id=data["strategy_id"],
            version_id=data["version_id"],
            requester=data.get("requester", "ab_testing_system"),
        )
        # Restore mutable / generated fields
        obj.request_time = datetime.fromisoformat(data["request_time"])
        obj.request_id = data["request_id"]
        obj.status = ApprovalStatus(data["status"])
        obj.reviewer = data.get("reviewer")
        obj.decision_time = (
            datetime.fromisoformat(data["decision_time"]) if data.get("decision_time") else None
        )
        obj.comments = data.get("comments")
        return obj

    def approve(self, reviewer: str, comments: str | None = None):
        self.status = ApprovalStatus.APPROVED
        self.reviewer = reviewer
        self.decision_time = datetime.utcnow()
        self.comments = comments
        logger.info(
            "ApprovalRequest %s approved by %s", self.request_id, reviewer
        )

    def reject(self, reviewer: str, comments: str | None = None):
        self.status = ApprovalStatus.REJECTED
        self.reviewer = reviewer
        self.decision_time = datetime.utcnow()
        self.comments = comments
        logger.info(
            "ApprovalRequest %s rejected by %s", self.request_id, reviewer
        )


class ApprovalWorkflowManager:
    """In-memory manager for approval workflow requests (increment 1)."""

    def __init__(self, storage_path: str | Path | None = None):
        self._storage_path = Path(storage_path) if storage_path else DEFAULT_STORAGE_FILE
        self._requests: Dict[str, ApprovalRequest] = {}
        self._event_bus = EventBus()
        self._load_from_disk()

    # CRUD operations -----------------------------------------------------

    def create_request(
        self, *, test_id: str, strategy_id: str, version_id: str, requester: str = "ab_testing_system"
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            test_id=test_id,
            strategy_id=strategy_id,
            version_id=version_id,
            requester=requester,
        )
        self._requests[req.request_id] = req
        logger.info("Created ApprovalRequest %s for test %s", req.request_id, test_id)
        self._save_to_disk()
        
        # Emit event for request creation
        self._emit_event(
            EventType.APPROVAL_REQUEST_CREATED,
            {
                "request_id": req.request_id,
                "test_id": req.test_id,
                "strategy_id": req.strategy_id,
                "version_id": req.version_id,
                "requester": req.requester
            }
        )
        
        return req

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        return self._requests.get(request_id)

    def list_requests(self, status: ApprovalStatus | None = None) -> List[ApprovalRequest]:
        if status is None:
            return list(self._requests.values())
        return [r for r in self._requests.values() if r.status == status]

    def approve_request(self, request_id: str, reviewer: str, comments: str | None = None) -> bool:
        req = self.get_request(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False
        req.approve(reviewer, comments)
        self._save_to_disk()
        
        # Emit event for request approval
        self._emit_event(
            EventType.APPROVAL_REQUEST_APPROVED,
            {
                "request_id": req.request_id,
                "test_id": req.test_id,
                "strategy_id": req.strategy_id,
                "version_id": req.version_id,
                "reviewer": reviewer,
                "comments": comments
            }
        )
        
        return True

    def reject_request(self, request_id: str, reviewer: str, comments: str | None = None) -> bool:
        req = self.get_request(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False
        req.reject(reviewer, comments)
        self._save_to_disk()
        
        # Emit event for request rejection
        self._emit_event(
            EventType.APPROVAL_REQUEST_REJECTED,
            {
                "request_id": req.request_id,
                "test_id": req.test_id,
                "strategy_id": req.strategy_id,
                "version_id": req.version_id,
                "reviewer": reviewer,
                "comments": comments
            }
        )
        
        return True

    # Internal persistence helpers ---------------------------------------

    def _load_from_disk(self) -> None:
        """Load requests from disk into memory (best-effort)."""
        if not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                req = ApprovalRequest.from_dict(item)
                self._requests[req.request_id] = req
            logger.info(
                "Loaded %d approval requests from %s",
                len(self._requests),
                self._storage_path,
            )
        except Exception as e:
            logger.exception("Failed to load approval requests: %s", e)

    def _save_to_disk(self) -> None:
        """Persist current requests to disk (best-effort)."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._storage_path.open("w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._requests.values()], f, indent=2)
        except Exception as e:
            logger.exception("Failed to save approval requests: %s", e)
            
    def _emit_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Emit an event to the event bus.
        
        Args:
            event_type: Type of event to emit
            data: Event data payload
        """
        event = Event(
            event_type=event_type,
            data=data,
            source="approval_workflow"
        )
        self._event_bus.publish(event)


# Singleton helper ---------------------------------------------------------

_approval_workflow_manager: Optional[ApprovalWorkflowManager] = None


def get_approval_workflow_manager(storage_path: str | Path | None = None) -> ApprovalWorkflowManager:
    global _approval_workflow_manager
    if _approval_workflow_manager is None:
        _approval_workflow_manager = ApprovalWorkflowManager(storage_path=storage_path)
    return _approval_workflow_manager


# Simple manual test when run directly ------------------------------------
if __name__ == "__main__":
    mgr = get_approval_workflow_manager()

    r = mgr.create_request(test_id="test123", strategy_id="stratA", version_id="v2")
    print("Created:", r)

    mgr.approve_request(r.request_id, reviewer="alice", comments="Looks good")
    print("After approval:", mgr.get_request(r.request_id))
