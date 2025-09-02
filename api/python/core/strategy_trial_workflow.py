#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Trial Workflow

This module implements a workflow system for strategy trials:
1. New strategies automatically start in paper trading mode
2. Performance metrics are tracked identically for paper and live strategies
3. Strategies can be promoted from paper to live based on performance
4. Clear status indicators are maintained throughout the system
"""

import logging
import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_base import Strategy
from trading_bot.core.strategy_broker_router import StrategyBrokerRouter
from trading_bot.core.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy trial status within the workflow."""
    NEW = "new"                       # Just created, no trades yet
    PAPER_TESTING = "paper_testing"   # Running in paper mode
    READY_FOR_REVIEW = "ready_for_review"  # Paper testing complete, pending review
    APPROVED = "approved"             # Approved for live trading
    LIVE = "live"                     # Currently live trading
    PAUSED = "paused"                 # Temporarily paused
    REJECTED = "rejected"             # Failed evaluation
    DEPRECATED = "deprecated"         # No longer in use
    ARCHIVED = "archived"             # Saved for reference only


class TrialPhase(Enum):
    """Phases of the strategy trial process."""
    DEVELOPMENT = "development"       # Initial development/coding
    BACKTEST = "backtest"             # Historical backtesting
    PAPER_TRADE = "paper_trade"       # Paper trading (forward testing)
    INCUBATION = "incubation"         # Proven in paper, waiting for promotion
    LIVE_LIMITED = "live_limited"     # Live trading with position limits
    LIVE_FULL = "live_full"           # Full live trading


class PromotionCriteria:
    """Criteria for promoting a strategy from paper to live."""
    
    def __init__(
        self,
        min_trading_days: int = 30,
        min_trades: int = 20,
        min_profit_factor: float = 1.5,
        min_win_rate: float = 0.50,
        min_sharpe_ratio: float = 1.0,
        max_drawdown_pct: float = -15.0,
        custom_criteria: Optional[List[Callable[[Dict[str, Any]], bool]]] = None
    ):
        """
        Initialize promotion criteria.
        
        Args:
            min_trading_days: Minimum days in paper trading
            min_trades: Minimum number of trades executed
            min_profit_factor: Minimum profit factor (gross_profit / gross_loss)
            min_win_rate: Minimum win rate (wins / total_trades)
            min_sharpe_ratio: Minimum Sharpe ratio
            max_drawdown_pct: Maximum drawdown percentage allowed (negative value)
            custom_criteria: List of custom functions that evaluate performance
        """
        self.min_trading_days = min_trading_days
        self.min_trades = min_trades
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown_pct = max_drawdown_pct
        self.custom_criteria = custom_criteria or []
    
    def check(self, performance_metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if performance meets promotion criteria.
        
        Args:
            performance_metrics: Strategy performance metrics
            
        Returns:
            Tuple of (passes_criteria, details)
        """
        results = {}
        
        # Trading days check
        days_active = performance_metrics.get("days_active", 0)
        results["days_active"] = {
            "value": days_active,
            "required": self.min_trading_days,
            "passed": days_active >= self.min_trading_days
        }
        
        # Trade count check
        trade_count = performance_metrics.get("total_trades", 0)
        results["trade_count"] = {
            "value": trade_count,
            "required": self.min_trades,
            "passed": trade_count >= self.min_trades
        }
        
        # Profit factor check
        profit_factor = performance_metrics.get("profit_factor", 0)
        results["profit_factor"] = {
            "value": profit_factor,
            "required": self.min_profit_factor,
            "passed": profit_factor >= self.min_profit_factor
        }
        
        # Win rate check
        win_rate = performance_metrics.get("win_rate", 0)
        results["win_rate"] = {
            "value": win_rate,
            "required": self.min_win_rate,
            "passed": win_rate >= self.min_win_rate
        }
        
        # Sharpe ratio check
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
        results["sharpe_ratio"] = {
            "value": sharpe_ratio,
            "required": self.min_sharpe_ratio,
            "passed": sharpe_ratio >= self.min_sharpe_ratio
        }
        
        # Max drawdown check
        max_drawdown = performance_metrics.get("max_drawdown_pct", 0)
        results["max_drawdown"] = {
            "value": max_drawdown,
            "required": self.max_drawdown_pct,
            "passed": max_drawdown >= self.max_drawdown_pct
        }
        
        # Custom criteria checks
        for i, criterion in enumerate(self.custom_criteria):
            try:
                passed = criterion(performance_metrics)
                results[f"custom_criterion_{i+1}"] = {
                    "passed": passed
                }
            except Exception as e:
                logger.error(f"Error evaluating custom criterion {i+1}: {str(e)}")
                results[f"custom_criterion_{i+1}"] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Overall result
        all_passed = all(item.get("passed", False) for item in results.values())
        
        return all_passed, results


class StrategyTrialWorkflow:
    """
    Manages the workflow for strategy trials from paper to live.
    
    This class enforces a promotion workflow:
    1. New strategies start in paper trading mode
    2. Performance is tracked and evaluated against criteria
    3. Strategies can be promoted to live once criteria are met
    4. Status tracking and notifications at each step
    """
    
    def __init__(
        self,
        broker_router: StrategyBrokerRouter,
        performance_tracker: PerformanceTracker,
        promotion_criteria: Optional[PromotionCriteria] = None,
        workflow_config_path: Optional[str] = None
    ):
        """
        Initialize the strategy trial workflow.
        
        Args:
            broker_router: Strategy broker router
            performance_tracker: Performance tracking system
            promotion_criteria: Criteria for promotion to live
            workflow_config_path: Path to workflow configuration file
        """
        self._broker_router = broker_router
        self._performance_tracker = performance_tracker
        self._promotion_criteria = promotion_criteria or PromotionCriteria()
        
        # Strategy metadata tracking
        self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Event bus for notifications
        self._event_bus = ServiceRegistry.get_instance().get_service(EventBus)
        
        # Load workflow configuration if provided
        if workflow_config_path and os.path.exists(workflow_config_path):
            self._load_workflow_config(workflow_config_path)
        
        logger.info("Strategy trial workflow initialized")
    
    def _load_workflow_config(self, config_path: str) -> None:
        """
        Load workflow configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load promotion criteria if present
            if "promotion_criteria" in config:
                criteria = config["promotion_criteria"]
                self._promotion_criteria = PromotionCriteria(
                    min_trading_days=criteria.get("min_trading_days", 30),
                    min_trades=criteria.get("min_trades", 20),
                    min_profit_factor=criteria.get("min_profit_factor", 1.5),
                    min_win_rate=criteria.get("min_win_rate", 0.50),
                    min_sharpe_ratio=criteria.get("min_sharpe_ratio", 1.0),
                    max_drawdown_pct=criteria.get("max_drawdown_pct", -15.0)
                )
            
            # Load existing strategy metadata if present
            if "strategies" in config:
                self._strategy_metadata = config["strategies"]
            
            logger.info(f"Loaded workflow configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading workflow configuration: {str(e)}")
    
    def register_new_strategy(
        self,
        strategy: Strategy,
        initial_config: Dict[str, Any],
        paper_broker_id: Optional[str] = None
    ) -> None:
        """
        Register a new strategy in the workflow.
        
        Args:
            strategy: Strategy instance
            initial_config: Strategy configuration
            paper_broker_id: Specific paper broker to use (optional)
        """
        strategy_id = strategy.get_id()
        
        # Enforce paper trading mode for new strategies
        if "mode" in initial_config and initial_config["mode"] != "paper":
            logger.warning(f"New strategy {strategy_id} must start in paper mode. "
                          f"Overriding mode '{initial_config['mode']}' to 'paper'.")
        
        # Override mode to ensure paper trading
        config = initial_config.copy()
        config["mode"] = "paper"
        
        # Register with broker router
        self._broker_router.register_strategy_from_config(strategy, config)
        
        # Create metadata entry
        now = datetime.now()
        metadata = {
            "id": strategy_id,
            "name": config.get("name", strategy_id),
            "status": StrategyStatus.NEW.value,
            "phase": TrialPhase.PAPER_TRADE.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "broker_id": paper_broker_id or self._broker_router.get_strategy_broker_id(strategy_id),
            "evaluation_start": now.isoformat(),
            "evaluation_end": None,
            "review_date": None,
            "approved_by": None,
            "remarks": [],
            "config": config
        }
        
        self._strategy_metadata[strategy_id] = metadata
        
        # Initialize performance tracking
        self._performance_tracker.register_strategy(strategy_id)
        
        # Publish event
        self._publish_workflow_event(
            strategy_id=strategy_id,
            event_type="strategy_registered",
            details={
                "status": StrategyStatus.NEW.value,
                "phase": TrialPhase.PAPER_TRADE.value
            }
        )
        
        logger.info(f"Registered new strategy '{strategy_id}' in paper testing mode")
    
    def update_strategy_status(
        self,
        strategy_id: str,
        status: StrategyStatus,
        remarks: Optional[str] = None
    ) -> None:
        """
        Update a strategy's status in the workflow.
        
        Args:
            strategy_id: Strategy ID
            status: New status
            remarks: Optional remarks about the status change
        """
        if strategy_id not in self._strategy_metadata:
            logger.error(f"Strategy {strategy_id} not found in workflow")
            return
        
        metadata = self._strategy_metadata[strategy_id]
        old_status = metadata["status"]
        
        # Update metadata
        metadata["status"] = status.value
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Add remarks if provided
        if remarks:
            metadata["remarks"].append({
                "timestamp": datetime.now().isoformat(),
                "status_change": f"{old_status} -> {status.value}",
                "note": remarks
            })
        
        # Special handling for specific status changes
        if status == StrategyStatus.READY_FOR_REVIEW:
            metadata["review_date"] = (datetime.now() + timedelta(days=3)).isoformat()
        
        elif status == StrategyStatus.APPROVED:
            # Record approval but don't change mode yet
            metadata["approved_by"] = "system"  # In a real app, this would be the user
        
        elif status == StrategyStatus.LIVE:
            # Only allow transition to LIVE if previously APPROVED
            if old_status != StrategyStatus.APPROVED.value:
                logger.warning(f"Strategy {strategy_id} must be APPROVED before going LIVE")
                return
        
        # Publish event
        self._publish_workflow_event(
            strategy_id=strategy_id,
            event_type="status_updated",
            details={
                "old_status": old_status,
                "new_status": status.value,
                "remarks": remarks
            }
        )
        
        logger.info(f"Updated strategy '{strategy_id}' status: {old_status} -> {status.value}")
    
    def update_strategy_phase(
        self,
        strategy_id: str,
        phase: TrialPhase,
        remarks: Optional[str] = None
    ) -> None:
        """
        Update a strategy's trial phase.
        
        Args:
            strategy_id: Strategy ID
            phase: New phase
            remarks: Optional remarks about the phase change
        """
        if strategy_id not in self._strategy_metadata:
            logger.error(f"Strategy {strategy_id} not found in workflow")
            return
        
        metadata = self._strategy_metadata[strategy_id]
        old_phase = metadata["phase"]
        
        # Update metadata
        metadata["phase"] = phase.value
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Add remarks if provided
        if remarks:
            metadata["remarks"].append({
                "timestamp": datetime.now().isoformat(),
                "phase_change": f"{old_phase} -> {phase.value}",
                "note": remarks
            })
        
        # Publish event
        self._publish_workflow_event(
            strategy_id=strategy_id,
            event_type="phase_updated",
            details={
                "old_phase": old_phase,
                "new_phase": phase.value,
                "remarks": remarks
            }
        )
        
        logger.info(f"Updated strategy '{strategy_id}' phase: {old_phase} -> {phase.value}")
    
    def promote_to_live(
        self,
        strategy_id: str,
        live_broker_id: Optional[str] = None,
        position_limit_pct: Optional[float] = None,
        approved_by: Optional[str] = None,
        remarks: Optional[str] = None
    ) -> bool:
        """
        Promote a strategy from paper to live trading.
        
        Args:
            strategy_id: Strategy ID
            live_broker_id: Broker ID for live trading
            position_limit_pct: Optional position size limit as percentage
            approved_by: Name of approver
            remarks: Optional remarks about the promotion
            
        Returns:
            bool: Success status
        """
        if strategy_id not in self._strategy_metadata:
            logger.error(f"Strategy {strategy_id} not found in workflow")
            return False
        
        metadata = self._strategy_metadata[strategy_id]
        
        # Check current status
        if metadata["status"] != StrategyStatus.APPROVED.value:
            logger.error(f"Strategy {strategy_id} must be APPROVED before promotion to live")
            return False
        
        # Get original config
        config = metadata["config"].copy()
        
        # Update config for live trading
        config["mode"] = "live"
        
        if live_broker_id:
            config["broker"] = live_broker_id
        elif "broker" in config and "_paper" in config["broker"]:
            # Try to derive live broker from paper broker
            live_broker_id = config["broker"].replace("_paper", "")
            config["broker"] = live_broker_id
        
        # Apply position limit if specified
        if position_limit_pct is not None:
            if "risk" not in config:
                config["risk"] = {}
            config["risk"]["position_limit_pct"] = position_limit_pct
        
        # Update metadata
        metadata["status"] = StrategyStatus.LIVE.value
        metadata["phase"] = TrialPhase.LIVE_LIMITED.value if position_limit_pct else TrialPhase.LIVE_FULL.value
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["approved_by"] = approved_by or metadata.get("approved_by", "system")
        metadata["config"] = config
        
        # Add remarks
        promotion_note = remarks or "Promoted to live trading"
        metadata["remarks"].append({
            "timestamp": datetime.now().isoformat(),
            "status_change": f"{StrategyStatus.APPROVED.value} -> {StrategyStatus.LIVE.value}",
            "note": promotion_note
        })
        
        # Update broker assignment
        broker_id = config.get("broker", live_broker_id)
        if not broker_id:
            logger.error(f"No live broker specified for strategy {strategy_id}")
            return False
        
        # Re-register with broker router in live mode
        try:
            # Get the strategy instance via the performance tracker
            # (In a real system, you'd have a cleaner way to get the strategy instance)
            strategy = self._performance_tracker.get_strategy(strategy_id)
            if not strategy:
                logger.error(f"Could not find strategy instance for {strategy_id}")
                return False
            
            # Register with live broker
            self._broker_router.register_strategy(
                strategy=strategy,
                mode="live",
                broker_id=broker_id
            )
            
            # Publish event
            self._publish_workflow_event(
                strategy_id=strategy_id,
                event_type="promoted_to_live",
                details={
                    "broker_id": broker_id,
                    "position_limit_pct": position_limit_pct,
                    "approved_by": metadata["approved_by"],
                    "remarks": promotion_note
                }
            )
            
            logger.info(f"Promoted strategy '{strategy_id}' to live trading with broker '{broker_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting strategy {strategy_id} to live: {str(e)}")
            return False
    
    def evaluate_performance(
        self,
        strategy_id: str,
        min_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy's performance against promotion criteria.
        
        Args:
            strategy_id: Strategy ID
            min_days: Minimum days to evaluate (overrides default)
            
        Returns:
            Dict: Evaluation results with metrics and criteria checks
        """
        if strategy_id not in self._strategy_metadata:
            logger.error(f"Strategy {strategy_id} not found in workflow")
            return {"error": "Strategy not found"}
        
        metadata = self._strategy_metadata[strategy_id]
        
        # Use specified min_days or default from promotion criteria
        min_days_to_check = min_days or self._promotion_criteria.min_trading_days
        
        # Get performance metrics
        metrics = self._performance_tracker.get_strategy_metrics(
            strategy_id=strategy_id,
            days=min_days_to_check
        )
        
        if not metrics:
            return {
                "strategy_id": strategy_id,
                "status": metadata["status"],
                "phase": metadata["phase"],
                "error": "No performance data available",
                "meets_criteria": False
            }
        
        # Check promotion criteria
        meets_criteria, criteria_details = self._promotion_criteria.check(metrics)
        
        # Build evaluation result
        result = {
            "strategy_id": strategy_id,
            "name": metadata.get("name", strategy_id),
            "status": metadata["status"],
            "phase": metadata["phase"],
            "evaluation_date": datetime.now().isoformat(),
            "days_evaluated": metrics.get("days_active", 0),
            "meets_all_criteria": meets_criteria,
            "criteria_checks": criteria_details,
            "metrics": metrics,
            "recommendation": "PROMOTE" if meets_criteria else "CONTINUE_PAPER"
        }
        
        # If meets criteria and in paper testing, suggest ready for review
        if meets_criteria and metadata["status"] == StrategyStatus.PAPER_TESTING.value:
            # Auto-update status if configured to do so
            self.update_strategy_status(
                strategy_id=strategy_id,
                status=StrategyStatus.READY_FOR_REVIEW,
                remarks="Automatically marked ready for review based on performance criteria"
            )
        
        # Publish evaluation event
        self._publish_workflow_event(
            strategy_id=strategy_id,
            event_type="performance_evaluated",
            details={
                "meets_criteria": meets_criteria,
                "recommendation": result["recommendation"]
            }
        )
        
        return result
    
    def get_strategies_by_status(self, status: StrategyStatus) -> List[Dict[str, Any]]:
        """
        Get all strategies with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List[Dict]: List of strategy metadata
        """
        return [
            metadata for metadata in self._strategy_metadata.values()
            if metadata["status"] == status.value
        ]
    
    def get_strategies_by_phase(self, phase: TrialPhase) -> List[Dict[str, Any]]:
        """
        Get all strategies in a specific phase.
        
        Args:
            phase: Phase to filter by
            
        Returns:
            List[Dict]: List of strategy metadata
        """
        return [
            metadata for metadata in self._strategy_metadata.values()
            if metadata["phase"] == phase.value
        ]
    
    def get_strategy_metadata(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dict: Strategy metadata or None if not found
        """
        return self._strategy_metadata.get(strategy_id)
    
    def get_all_strategy_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all strategies.
        
        Returns:
            Dict: All strategy metadata
        """
        return self._strategy_metadata.copy()
    
    def save_workflow_state(self, file_path: str) -> bool:
        """
        Save current workflow state to file.
        
        Args:
            file_path: Path to save file
            
        Returns:
            bool: Success status
        """
        try:
            # Create output structure
            output = {
                "timestamp": datetime.now().isoformat(),
                "strategies": self._strategy_metadata,
                "promotion_criteria": {
                    "min_trading_days": self._promotion_criteria.min_trading_days,
                    "min_trades": self._promotion_criteria.min_trades,
                    "min_profit_factor": self._promotion_criteria.min_profit_factor,
                    "min_win_rate": self._promotion_criteria.min_win_rate,
                    "min_sharpe_ratio": self._promotion_criteria.min_sharpe_ratio,
                    "max_drawdown_pct": self._promotion_criteria.max_drawdown_pct
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Saved workflow state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving workflow state: {str(e)}")
            return False
    
    def _publish_workflow_event(self, strategy_id: str, event_type: str, details: Dict[str, Any]) -> None:
        """
        Publish a workflow event.
        
        Args:
            strategy_id: Strategy ID
            event_type: Workflow event type
            details: Event details
        """
        if not self._event_bus:
            return
        
        event_data = {
            "strategy_id": strategy_id,
            "workflow_event": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        try:
            self._event_bus.publish(Event(
                event_type=EventType.WORKFLOW_EVENT,
                data=event_data
            ))
        except Exception as e:
            logger.error(f"Error publishing workflow event: {str(e)}")
