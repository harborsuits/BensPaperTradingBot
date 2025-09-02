#!/usr/bin/env python3
"""
Strategy Lifecycle Extensions

This module extends the Strategy Lifecycle Manager with:
- Promotion criteria for identifying deployment candidates
- Succession logic for replacing underperforming strategies
- Version management and rollback procedures
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading

# Import core components
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_strategy_lifecycle_manager, StrategyVersion, VersionStatus, VersionSource
)
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline, DeploymentStatus
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class PromotionCriteria:
    """
    Criteria for promoting a strategy version to candidate or deployment.
    """
    
    def __init__(self, 
                 min_sharpe: float = 0.8,
                 min_profit_factor: float = 1.5, 
                 min_win_rate: float = 0.55,
                 max_drawdown: float = 15.0,
                 min_trades: int = 20,
                 min_days: int = 30,
                 required_metrics: Optional[List[str]] = None):
        """
        Initialize promotion criteria.
        
        Args:
            min_sharpe: Minimum Sharpe ratio
            min_profit_factor: Minimum profit factor
            min_win_rate: Minimum win rate (0-1)
            max_drawdown: Maximum drawdown percentage
            min_trades: Minimum number of trades
            min_days: Minimum backtest duration in days
            required_metrics: List of metrics that must be present
        """
        self.min_sharpe = min_sharpe
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades
        self.min_days = min_days
        self.required_metrics = required_metrics or [
            "sharpe_ratio", "profit_factor", "win_rate", 
            "max_drawdown_pct", "total_trades", "backtest_days"
        ]
    
    def meets_criteria(self, version: StrategyVersion) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a version meets promotion criteria.
        
        Args:
            version: Strategy version to check
            
        Returns:
            Tuple of (meets criteria, details)
        """
        backtest = version.metrics.get("backtest", {})
        
        # Check if required metrics are present
        for metric in self.required_metrics:
            if metric not in backtest:
                return False, {"missing_metric": metric}
        
        # Check each criterion
        checks = {
            "sharpe_ratio": backtest.get("sharpe_ratio", 0) >= self.min_sharpe,
            "profit_factor": backtest.get("profit_factor", 0) >= self.min_profit_factor,
            "win_rate": backtest.get("win_rate", 0) >= self.min_win_rate,
            "max_drawdown": backtest.get("max_drawdown_pct", 100) <= self.max_drawdown,
            "trades": backtest.get("total_trades", 0) >= self.min_trades,
            "duration": backtest.get("backtest_days", 0) >= self.min_days
        }
        
        # All checks must pass
        meets_all = all(checks.values())
        
        return meets_all, checks


class SuccessionCriteria:
    """
    Criteria for determining when a strategy should be replaced.
    """
    
    def __init__(self, 
                 trailing_window_days: int = 30,
                 min_sharpe_diff: float = 0.3,
                 min_return_diff_pct: float = 5.0,
                 max_drawdown_increase: float = 5.0,
                 min_trades_for_evaluation: int = 10,
                 max_consecutive_losses: int = 5):
        """
        Initialize succession criteria.
        
        Args:
            trailing_window_days: Days to look back for performance comparison
            min_sharpe_diff: Minimum improvement in Sharpe ratio to justify replacement
            min_return_diff_pct: Minimum improvement in return percentage
            max_drawdown_increase: Maximum acceptable increase in drawdown
            min_trades_for_evaluation: Minimum trades before evaluating
            max_consecutive_losses: Maximum consecutive losses before considering replacement
        """
        self.trailing_window_days = trailing_window_days
        self.min_sharpe_diff = min_sharpe_diff
        self.min_return_diff_pct = min_return_diff_pct
        self.max_drawdown_increase = max_drawdown_increase
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.max_consecutive_losses = max_consecutive_losses
    
    def should_replace(self, 
                      current: StrategyVersion, 
                      candidate: StrategyVersion) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if current version should be replaced by candidate.
        
        Args:
            current: Currently deployed version
            candidate: Candidate replacement version
            
        Returns:
            Tuple of (should replace, details)
        """
        # Get metrics
        current_live = current.metrics.get("live", {})
        current_backtest = current.metrics.get("backtest", {})
        candidate_backtest = candidate.metrics.get("backtest", {})
        
        # Check if we have enough trades to evaluate
        if current_live.get("total_trades", 0) < self.min_trades_for_evaluation:
            return False, {"reason": "insufficient_trades"}
        
        # Check consecutive losses
        if current_live.get("consecutive_losses", 0) >= self.max_consecutive_losses:
            return True, {"reason": "consecutive_losses"}
        
        # Compare performance metrics
        comparisons = {
            "sharpe_ratio": candidate_backtest.get("sharpe_ratio", 0) - current_live.get("sharpe_ratio", 0),
            "return_pct": candidate_backtest.get("total_return_pct", 0) - current_live.get("total_return_pct", 0),
            "drawdown_diff": current_live.get("max_drawdown_pct", 0) - candidate_backtest.get("max_drawdown_pct", 0),
            "win_rate_diff": candidate_backtest.get("win_rate", 0) - current_live.get("win_rate", 0)
        }
        
        # Check if candidate is better
        better_sharpe = comparisons["sharpe_ratio"] >= self.min_sharpe_diff
        better_return = comparisons["return_pct"] >= self.min_return_diff_pct
        acceptable_drawdown = comparisons["drawdown_diff"] >= -self.max_drawdown_increase
        
        # Decision rules
        should_replace = (better_sharpe and better_return and acceptable_drawdown)
        
        return should_replace, comparisons


class StrategyLifecycleExtension:
    """
    Extends the Strategy Lifecycle Manager with promotion and succession logic.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the extension.
        
        Args:
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus or EventBus()
        
        # Get core components
        self.lifecycle_manager = get_strategy_lifecycle_manager(event_bus)
        self.deployment_pipeline = get_deployment_pipeline()
        self.risk_manager = get_autonomous_risk_manager()
        
        # Criteria
        self.promotion_criteria = PromotionCriteria()
        self.succession_criteria = SuccessionCriteria()
        
        # Configuration
        self.config = {
            "auto_promotion_enabled": True,
            "auto_succession_enabled": True,
            "promotion_check_interval_hours": 24,
            "succession_check_interval_hours": 24,
            "max_candidates_per_strategy": 3,
            "rollback_performance_threshold": -10.0,  # % return before auto-rollback
            "version_probation_days": 5  # Days to give a new version before considering rollback
        }
        
        # State
        self.last_promotion_check = {}
        self.last_succession_check = {}
        self.monitoring_thread = None
        self.is_running = False
        self._lock = threading.RLock()
        
        # Register for events
        self._register_for_events()
        
        logger.info("Strategy Lifecycle Extension initialized")
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        # Strategy optimization events
        self.event_bus.register(EventType.STRATEGY_OPTIMISED, self._handle_strategy_optimised)
        
        # Performance events
        self.event_bus.register(EventType.PERFORMANCE_REPORT, self._handle_performance_report)
        
        logger.info("Registered for events")
    
    def start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started lifecycle monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring thread."""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
        
        logger.info("Stopped lifecycle monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Check for promotion candidates
                self._check_for_promotion_candidates()
                
                # Check for succession opportunities
                self._check_for_succession_opportunities()
                
                # Check for rollback conditions
                self._check_for_rollback_conditions()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for an hour
            for _ in range(60):  # Check every minute if we should stop
                if not self.is_running:
                    break
                import time
                time.sleep(60)
    
    def _check_for_promotion_candidates(self) -> None:
        """Check for strategies that should be promoted to candidates."""
        with self._lock:
            # Skip if auto-promotion is disabled
            if not self.config["auto_promotion_enabled"]:
                return
            
            # Get all strategy IDs
            all_strategy_ids = list(self.lifecycle_manager.versions.keys())
            
            for strategy_id in all_strategy_ids:
                # Check if we need to evaluate this strategy
                last_check = self.last_promotion_check.get(strategy_id, datetime.min)
                hours_since_check = (datetime.now() - last_check).total_seconds() / 3600
                
                if hours_since_check < self.config["promotion_check_interval_hours"]:
                    continue
                
                # Update last check time
                self.last_promotion_check[strategy_id] = datetime.now()
                
                # Get development versions
                dev_versions = self.lifecycle_manager.get_versions_by_status(
                    strategy_id, VersionStatus.DEVELOPMENT
                )
                
                if not dev_versions:
                    continue
                
                # Check each version against promotion criteria
                promoted_count = 0
                for version in dev_versions:
                    meets_criteria, details = self.promotion_criteria.meets_criteria(version)
                    
                    if meets_criteria:
                        # Promote to candidate
                        self.lifecycle_manager.set_version_status(
                            strategy_id,
                            version.version_id,
                            VersionStatus.CANDIDATE,
                            reason="Automatically promoted: meets criteria"
                        )
                        
                        logger.info(f"Promoted {strategy_id} version {version.version_id} to candidate status")
                        promoted_count += 1
                        
                        # Emit event
                        self._emit_event(
                            event_type="STRATEGY_VERSION_PROMOTED",
                            data={
                                "strategy_id": strategy_id,
                                "version_id": version.version_id,
                                "promotion_details": details,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                if promoted_count > 0:
                    logger.info(f"Promoted {promoted_count} versions for strategy {strategy_id}")
    
    def _check_for_succession_opportunities(self) -> None:
        """Check if any strategies should be replaced with better versions."""
        with self._lock:
            # Skip if auto-succession is disabled
            if not self.config["auto_succession_enabled"]:
                return
            
            # Get all strategy IDs with active versions
            active_strategy_ids = list(self.lifecycle_manager.active_versions.keys())
            
            for strategy_id in active_strategy_ids:
                # Check if we need to evaluate this strategy
                last_check = self.last_succession_check.get(strategy_id, datetime.min)
                hours_since_check = (datetime.now() - last_check).total_seconds() / 3600
                
                if hours_since_check < self.config["succession_check_interval_hours"]:
                    continue
                
                # Update last check time
                self.last_succession_check[strategy_id] = datetime.now()
                
                # Get the active version
                active_version = self.lifecycle_manager.get_active_version(strategy_id)
                
                if not active_version:
                    continue
                
                # Get candidate versions
                candidate_versions = self.lifecycle_manager.get_versions_by_status(
                    strategy_id, VersionStatus.CANDIDATE
                )
                
                if not candidate_versions:
                    continue
                
                # Check each candidate against succession criteria
                for candidate in candidate_versions:
                    should_replace, details = self.succession_criteria.should_replace(
                        active_version, candidate
                    )
                    
                    if should_replace:
                        # Replace with candidate
                        self._replace_strategy_version(
                            strategy_id,
                            active_version.version_id,
                            candidate.version_id,
                            reason=f"Auto-succession: {details.get('reason', 'performance improvement')}"
                        )
                        
                        # Only replace with one candidate at a time
                        break
    
    def _check_for_rollback_conditions(self) -> None:
        """Check if any strategies need to be rolled back due to poor performance."""
        with self._lock:
            # Get all active strategies
            active_strategy_ids = list(self.lifecycle_manager.active_versions.keys())
            
            for strategy_id in active_strategy_ids:
                # Get the active version
                active_version = self.lifecycle_manager.get_active_version(strategy_id)
                
                if not active_version:
                    continue
                
                # Skip if this version is too new
                if not active_version.last_deployed:
                    continue
                    
                days_deployed = (datetime.now() - active_version.last_deployed).days
                if days_deployed < self.config["version_probation_days"]:
                    continue
                
                # Check live performance
                live_metrics = active_version.metrics.get("live", {})
                total_return = live_metrics.get("total_return_pct", 0)
                
                # Check if performance is below threshold
                if total_return < self.config["rollback_performance_threshold"]:
                    # Get previous versions
                    all_versions = list(self.lifecycle_manager.versions.get(strategy_id, {}).values())
                    previous_versions = [
                        v for v in all_versions
                        if v.created_at < active_version.created_at and 
                        v.status != VersionStatus.FAILED
                    ]
                    
                    if not previous_versions:
                        continue
                    
                    # Sort by created_at descending
                    previous_versions.sort(key=lambda v: v.created_at, reverse=True)
                    previous_version = previous_versions[0]
                    
                    # Roll back to previous version
                    self._rollback_version(
                        strategy_id,
                        active_version.version_id,
                        previous_version.version_id,
                        reason=f"Auto-rollback: poor performance ({total_return:.1f}%)"
                    )
    
    def promote_strategy_to_candidate(self, 
                                     strategy_id: str, 
                                     version_id: str,
                                     reason: str = "Manual promotion") -> bool:
        """
        Promote a strategy version to candidate status.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            reason: Reason for promotion
            
        Returns:
            True if successful
        """
        with self._lock:
            # Get the version
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
            
            if not version:
                logger.warning(f"Cannot promote unknown version {version_id} of strategy {strategy_id}")
                return False
            
            # Check if it's already a candidate or deployed
            if version.status in [VersionStatus.CANDIDATE, VersionStatus.DEPLOYED]:
                logger.info(f"Version {version_id} of strategy {strategy_id} is already a {version.status.value}")
                return True
            
            # Update the status
            success = self.lifecycle_manager.set_version_status(
                strategy_id,
                version_id,
                VersionStatus.CANDIDATE,
                reason=reason
            )
            
            if success:
                # Emit event
                self._emit_event(
                    event_type="STRATEGY_VERSION_PROMOTED",
                    data={
                        "strategy_id": strategy_id,
                        "version_id": version_id,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return success
    
    def replace_underperforming_strategy(self, 
                                       strategy_id: str, 
                                       new_version_id: str,
                                       reason: str = "Manual replacement") -> bool:
        """
        Replace an active strategy with a new version.
        
        Args:
            strategy_id: Strategy identifier
            new_version_id: New version identifier
            reason: Reason for replacement
            
        Returns:
            True if successful
        """
        with self._lock:
            # Get the active version
            active_version = self.lifecycle_manager.get_active_version(strategy_id)
            
            if not active_version:
                logger.warning(f"Cannot replace non-active strategy {strategy_id}")
                return False
            
            # Get the new version
            new_version = self.lifecycle_manager.get_version(strategy_id, new_version_id)
            
            if not new_version:
                logger.warning(f"Cannot replace with unknown version {new_version_id} of strategy {strategy_id}")
                return False
            
            # Replace the version
            return self._replace_strategy_version(
                strategy_id,
                active_version.version_id,
                new_version_id,
                reason=reason
            )
    
    def _replace_strategy_version(self, 
                                 strategy_id: str, 
                                 old_version_id: str,
                                 new_version_id: str,
                                 reason: str = "") -> bool:
        """
        Replace a strategy version with a new one.
        
        Args:
            strategy_id: Strategy identifier
            old_version_id: Current version identifier
            new_version_id: New version identifier
            reason: Reason for replacement
            
        Returns:
            True if successful
        """
        with self._lock:
            # Get the versions
            old_version = self.lifecycle_manager.get_version(strategy_id, old_version_id)
            new_version = self.lifecycle_manager.get_version(strategy_id, new_version_id)
            
            if not old_version or not new_version:
                logger.warning(f"Cannot replace: missing version for strategy {strategy_id}")
                return False
            
            # Find the deployed strategy
            deployment_id = None
            for deployment in self.deployment_pipeline.get_deployments(status=DeploymentStatus.ACTIVE):
                if deployment.get("strategy_id") == strategy_id:
                    deployment_id = deployment.get("deployment_id")
                    break
            
            if not deployment_id:
                logger.warning(f"Cannot find deployment for strategy {strategy_id}")
                return False
            
            # Stop the current deployment
            self.deployment_pipeline.stop_deployment(deployment_id, f"Replacing with version {new_version_id}: {reason}")
            
            # Record the end of deployment for old version
            old_version.record_deployment_end(f"Replaced by version {new_version_id}: {reason}")
            
            # Update old version status
            old_version.update_status(VersionStatus.RETIRED, f"Replaced by version {new_version_id}: {reason}")
            
            # Update new version status
            new_version.update_status(VersionStatus.DEPLOYED, f"Replacing version {old_version_id}: {reason}")
            
            # Deploy the new version
            parameters = new_version.parameters
            risk_parameters = {
                # Copy risk parameters from the old deployment if available
                # This would typically be retrieved from the deployment pipeline
                "allocation_percentage": 5.0,
                "stop_loss_pct": 10.0,
                "risk_level": "MEDIUM"
            }
            
            # Deploy with risk manager
            deployment_result = self.deployment_pipeline.deploy_strategy(
                strategy_id=strategy_id,
                parameters=parameters,
                risk_params=risk_parameters,
                metadata={
                    "version_id": new_version_id,
                    "replaced_version": old_version_id,
                    "reason": reason
                }
            )
            
            # Update active version
            self.lifecycle_manager.active_versions[strategy_id] = new_version_id
            
            # Emit event
            self._emit_event(
                event_type="STRATEGY_VERSION_REPLACED",
                data={
                    "strategy_id": strategy_id,
                    "old_version_id": old_version_id,
                    "new_version_id": new_version_id,
                    "reason": reason,
                    "deployment_id": deployment_result.get("deployment_id"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Replaced strategy {strategy_id} version {old_version_id} with {new_version_id}: {reason}")
            
            return True
    
    def rollback_to_version(self, 
                           strategy_id: str, 
                           target_version_id: str,
                           reason: str = "Manual rollback") -> bool:
        """
        Roll back a strategy to a previous version.
        
        Args:
            strategy_id: Strategy identifier
            target_version_id: Target version to roll back to
            reason: Reason for rollback
            
        Returns:
            True if successful
        """
        with self._lock:
            # Get the active version
            active_version = self.lifecycle_manager.get_active_version(strategy_id)
            
            if not active_version:
                logger.warning(f"Cannot roll back non-active strategy {strategy_id}")
                return False
            
            # Get the target version
            target_version = self.lifecycle_manager.get_version(strategy_id, target_version_id)
            
            if not target_version:
                logger.warning(f"Cannot roll back to unknown version {target_version_id} of strategy {strategy_id}")
                return False
            
            # Roll back
            return self._rollback_version(
                strategy_id,
                active_version.version_id,
                target_version_id,
                reason=reason
            )
    
    def _rollback_version(self, 
                         strategy_id: str, 
                         current_version_id: str,
                         target_version_id: str,
                         reason: str = "") -> bool:
        """
        Roll back a strategy version.
        
        Args:
            strategy_id: Strategy identifier
            current_version_id: Current version identifier
            target_version_id: Target version identifier
            reason: Reason for rollback
            
        Returns:
            True if successful
        """
        with self._lock:
            # Create a specific rollback reason
            rollback_reason = f"Rolling back from {current_version_id} to {target_version_id}: {reason}"
            
            # Use the same replacement logic
            success = self._replace_strategy_version(
                strategy_id,
                current_version_id,
                target_version_id,
                reason=rollback_reason
            )
            
            if success:
                # Update version source to indicate it's a rollback
                target_version = self.lifecycle_manager.get_version(strategy_id, target_version_id)
                if target_version:
                    target_version.source = VersionSource.ROLLBACK
                
                # Mark the current version as failed
                current_version = self.lifecycle_manager.get_version(strategy_id, current_version_id)
                if current_version:
                    current_version.update_status(VersionStatus.FAILED, reason=rollback_reason)
                
                # Emit specific rollback event
                self._emit_event(
                    event_type="STRATEGY_VERSION_ROLLBACK",
                    data={
                        "strategy_id": strategy_id,
                        "from_version_id": current_version_id,
                        "to_version_id": target_version_id,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return success
    
    def get_best_version(self, 
                        strategy_id: str, 
                        metric_name: str = "sharpe_ratio") -> Optional[StrategyVersion]:
        """
        Get the best version of a strategy based on a metric.
        
        Args:
            strategy_id: Strategy identifier
            metric_name: Metric to compare
            
        Returns:
            Best StrategyVersion or None
        """
        return self.lifecycle_manager.get_best_performing_version(
            strategy_id, metric_name
        )
    
    def _handle_strategy_optimised(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle strategy optimised event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        strategy_id = data.get("strategy_id")
        
        if not strategy_id:
            return
        
        # Extract parameters and metrics
        parameters = data.get("parameters", {})
        metrics = data.get("metrics", {})
        parent_version = data.get("parent_version")
        
        # Track as a new version
        version_id = self.lifecycle_manager.track_strategy_version(
            strategy_id=strategy_id,
            parameters=parameters,
            metrics=metrics,
            source=VersionSource.OPTIMIZATION,
            parent_version=parent_version,
            metadata={
                "optimization_id": data.get("optimization_id"),
                "optimization_method": data.get("method")
            }
        )
        
        logger.info(f"Tracked optimized version {version_id} for strategy {strategy_id}")
        
        # Check against promotion criteria
        version = self.lifecycle_manager.get_version(strategy_id, version_id)
        
        if version:
            meets_criteria, details = self.promotion_criteria.meets_criteria(version)
            
            if meets_criteria:
                # Promote to candidate
                self.promote_strategy_to_candidate(
                    strategy_id,
                    version_id,
                    reason="Optimization met promotion criteria"
                )
    
    def _handle_performance_report(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle performance report event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        strategy_id = data.get("strategy_id")
        
        if not strategy_id:
            return
        
        # Get metrics
        metrics = data.get("metrics", {})
        
        # Check if we have version information
        version_id = data.get("version_id")
        
        if not version_id:
            # Try to get active version
            active_version = self.lifecycle_manager.get_active_version(strategy_id)
            
            if not active_version:
                return
                
            version_id = active_version.version_id
        
        # Update metrics
        self.lifecycle_manager.update_version_metrics(
            strategy_id=strategy_id,
            version_id=version_id,
            metrics=metrics,
            is_live=True
        )
        
        logger.debug(f"Updated live metrics for strategy {strategy_id} version {version_id}")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit event to event bus.
        
        Args:
            event_type: Event type
            data: Event data
        """
        if not self.event_bus:
            return
            
        try:
            event = Event(
                event_type=event_type,
                source="StrategyLifecycleExtension",
                data=data,
                timestamp=datetime.now()
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error emitting event: {e}")


# Singleton instance for global access
_strategy_lifecycle_extension = None

def get_strategy_lifecycle_extension(event_bus: Optional[EventBus] = None) -> StrategyLifecycleExtension:
    """
    Get singleton instance of strategy lifecycle extension.
    
    Args:
        event_bus: Event bus for communication
        
    Returns:
        StrategyLifecycleExtension instance
    """
    global _strategy_lifecycle_extension
    
    if _strategy_lifecycle_extension is None:
        _strategy_lifecycle_extension = StrategyLifecycleExtension(event_bus)
        
    return _strategy_lifecycle_extension
