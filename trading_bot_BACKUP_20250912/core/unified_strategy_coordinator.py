#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Strategy Coordinator

Coordinates signals from multiple strategy types, handles conflicts,
and ensures consistent processing regardless of signal source.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import uuid
from collections import defaultdict

from trading_bot.core.strategy_base import (
    Strategy, StrategyPriority, ConflictResolutionMode, SignalTag, StrategyType
)
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.strategy_manager import StrategyPerformanceManager

logger = logging.getLogger(__name__)


class SignalConflict:
    """Represents a conflict between signals from different strategies."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        strategies: List[str],
        signals: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize a signal conflict.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe of the signals
            strategies: List of strategy IDs involved
            signals: List of conflicting signals
            timestamp: When the conflict was detected
        """
        self.conflict_id = str(uuid.uuid4())
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategies = strategies
        self.signals = signals
        self.timestamp = timestamp or datetime.now()
        self.resolution = None
        self.resolved = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conflict_id": self.conflict_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategies": self.strategies,
            "signals": self.signals,
            "timestamp": self.timestamp.isoformat(),
            "resolution": self.resolution,
            "resolved": self.resolved
        }


class UnifiedStrategyCoordinator:
    """
    Coordinates signals from multiple strategy types.
    
    Key responsibilities:
    1. Signal normalization across strategy types
    2. Conflict detection and resolution
    3. Unified performance tracking
    4. Dynamic priority adjustment
    """
    
    def __init__(
        self,
        default_conflict_mode: ConflictResolutionMode = ConflictResolutionMode.PRIORITY_BASED,
        normalization_config: Optional[Dict[str, Any]] = None,
        performance_manager: Optional[StrategyPerformanceManager] = None
    ):
        """
        Initialize the coordinator.
        
        Args:
            default_conflict_mode: Default conflict resolution mode
            normalization_config: Signal normalization configuration
            performance_manager: Strategy performance manager
        """
        self.default_conflict_mode = default_conflict_mode
        self.normalization_config = normalization_config or {}
        
        # Get or create performance manager
        self.performance_manager = performance_manager or ServiceRegistry.get("strategy_performance_manager")
        if not self.performance_manager:
            self.performance_manager = StrategyPerformanceManager()
            ServiceRegistry.register("strategy_performance_manager", self.performance_manager)
        
        # Strategy registry
        self.strategies: Dict[str, Strategy] = {}
        
        # Signal tracking
        self.received_signals: List[Dict[str, Any]] = []
        self.normalized_signals: List[Dict[str, Any]] = []
        self.pending_signals: Dict[str, Dict[str, Any]] = {}
        self.processed_signals: Dict[str, Dict[str, Any]] = {}
        
        # Conflict tracking
        self.active_conflicts: Dict[str, SignalConflict] = {}
        self.resolved_conflicts: List[SignalConflict] = []
        
        # Signal history by symbol and timeframe
        self.signal_history = defaultdict(lambda: defaultdict(list))
        
        # Priority overrides
        self.priority_overrides: Dict[str, StrategyPriority] = {}
        
        # Subscribe to events
        self.event_bus = EventBus()
        self._subscribe_to_events()
        
        logger.info("UnifiedStrategyCoordinator initialized")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Strategy signals
        self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self.handle_strategy_signal)
        
        # External signals
        self.event_bus.subscribe(EventType.EXTERNAL_SIGNAL, self.handle_external_signal)
        
        # Pattern signals
        self.event_bus.subscribe(EventType.PATTERN_DETECTED, self.handle_pattern_signal)
        
        # Performance updates
        self.event_bus.subscribe(EventType.STRATEGY_PERFORMANCE_UPDATE, self.handle_performance_update)
    
    def register_strategy(self, strategy: Strategy) -> None:
        """
        Register a strategy with the coordinator.
        
        Args:
            strategy: The strategy to register
        """
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Registered strategy: {strategy.name} ({strategy.strategy_id})")
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """
        Unregister a strategy from the coordinator.
        
        Args:
            strategy_id: ID of the strategy to unregister
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            logger.info(f"Unregistered strategy: {strategy_id}")
    
    def set_priority_override(self, strategy_id: str, priority: StrategyPriority) -> None:
        """
        Set a manual priority override for a strategy.
        
        Args:
            strategy_id: Strategy ID
            priority: New priority level
        """
        self.priority_overrides[strategy_id] = priority
        logger.info(f"Set priority override for {strategy_id}: {priority.name}")
    
    def get_effective_priority(self, strategy_id: str) -> StrategyPriority:
        """
        Get the effective priority for a strategy, considering overrides.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Effective priority level
        """
        # Check overrides first
        if strategy_id in self.priority_overrides:
            return self.priority_overrides[strategy_id]
        
        # Use strategy's own priority
        if strategy_id in self.strategies:
            return self.strategies[strategy_id].priority
        
        # Default
        return StrategyPriority.NORMAL
    
    def handle_strategy_signal(self, event: Event) -> None:
        """
        Handle a signal from a strategy.
        
        Args:
            event: Strategy signal event
        """
        signal_data = event.data
        if not signal_data:
            return
        
        strategy_id = signal_data.get("strategy_id")
        if not strategy_id:
            logger.warning("Received strategy signal without strategy_id")
            return
        
        # Normalize the signal
        normalized = self._normalize_strategy_signal(signal_data)
        
        # Process the normalized signal
        self._process_signal(normalized)
    
    def handle_external_signal(self, event: Event) -> None:
        """
        Handle an external signal event.
        
        Args:
            event: External signal event
        """
        signal_data = event.data
        if not signal_data:
            return
        
        # Normalize the external signal
        normalized = self._normalize_external_signal(signal_data)
        
        # Process the normalized signal
        self._process_signal(normalized)
    
    def handle_pattern_signal(self, event: Event) -> None:
        """
        Handle a pattern detection event.
        
        Args:
            event: Pattern detection event
        """
        pattern_data = event.data
        if not pattern_data:
            return
        
        # Normalize the pattern signal
        normalized = self._normalize_pattern_signal(pattern_data)
        
        # Process the normalized signal
        self._process_signal(normalized)
    
    def handle_performance_update(self, event: Event) -> None:
        """
        Handle a strategy performance update.
        
        Args:
            event: Performance update event
        """
        performance_data = event.data
        if not performance_data:
            return
        
        strategy_id = performance_data.get("strategy_id")
        if not strategy_id:
            return
        
        # Update our internal tracking
        self._update_strategy_performance(strategy_id, performance_data)
    
    def _normalize_strategy_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a strategy signal to the standard format.
        
        Args:
            signal: Strategy signal data
            
        Returns:
            Normalized signal
        """
        # Generate a unique signal ID if not present
        signal_id = signal.get("id") or str(uuid.uuid4())
        
        # Ensure timestamp exists
        timestamp = signal.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not timestamp:
            timestamp = datetime.now()
        
        # Create normalized signal
        normalized = {
            "id": signal_id,
            "timestamp": timestamp,
            "source": "strategy",
            "strategy_id": signal.get("strategy_id"),
            "strategy_type": signal.get("strategy_type"),
            "priority": signal.get("priority", StrategyPriority.NORMAL.value),
            "symbol": signal.get("symbol"),
            "timeframe": signal.get("timeframe"),
            "direction": signal.get("direction"),
            "signal_type": signal.get("signal_type"),
            "price": signal.get("price"),
            "tags": signal.get("tags", []),
            "confidence": signal.get("confidence", 0.5),
            "metadata": signal.get("metadata", {}),
            "raw_signal": signal
        }
        
        # Record the normalized signal
        self.normalized_signals.append(normalized)
        
        return normalized
    
    def _normalize_external_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize an external signal to the standard format.
        
        Args:
            signal: External signal data
            
        Returns:
            Normalized signal
        """
        signal_data = signal.get("signal", {})
        if not signal_data:
            signal_data = signal
        
        # Generate a unique signal ID
        signal_id = str(uuid.uuid4())
        
        # Extract the source
        source = signal_data.get("source", "unknown")
        
        # Map external source to strategy type
        strategy_type_map = {
            "tradingview": StrategyType.EXTERNAL_SIGNAL.value,
            "custom_script": StrategyType.CUSTOM.value,
            "api": StrategyType.EXTERNAL_SIGNAL.value,
            "alpaca": StrategyType.EXTERNAL_SIGNAL.value,
            "finnhub": StrategyType.EXTERNAL_SIGNAL.value
        }
        
        strategy_type = strategy_type_map.get(source, StrategyType.EXTERNAL_SIGNAL.value)
        
        # Ensure timestamp exists
        timestamp = signal_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not timestamp:
            timestamp = datetime.now()
        
        # Create normalized signal
        normalized = {
            "id": signal_id,
            "timestamp": timestamp,
            "source": "external",
            "strategy_id": f"external_{source}",
            "strategy_type": strategy_type,
            "priority": StrategyPriority.NORMAL.value,
            "symbol": signal_data.get("symbol"),
            "timeframe": signal_data.get("timeframe", "1h"),
            "direction": signal_data.get("direction"),
            "signal_type": signal_data.get("signal_type", "entry"),
            "price": signal_data.get("price"),
            "tags": ["external", source],
            "confidence": signal_data.get("confidence", 0.5),
            "metadata": signal_data.get("metadata", {}),
            "raw_signal": signal
        }
        
        # Record the normalized signal
        self.normalized_signals.append(normalized)
        
        return normalized
    
    def _normalize_pattern_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a pattern signal to the standard format.
        
        Args:
            signal: Pattern signal data
            
        Returns:
            Normalized signal
        """
        # Generate a unique signal ID
        signal_id = str(uuid.uuid4())
        
        # Ensure timestamp exists
        timestamp = signal.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not timestamp:
            timestamp = datetime.now()
        
        # Map pattern direction to standard direction
        direction_map = {
            "bullish": "long",
            "bearish": "short",
            "neutral": "flat"
        }
        
        direction = signal.get("direction", "unknown")
        if direction in direction_map:
            direction = direction_map[direction]
        
        # Create normalized signal
        normalized = {
            "id": signal_id,
            "timestamp": timestamp,
            "source": "pattern",
            "strategy_id": f"pattern_{signal.get('pattern_type', 'unknown')}",
            "strategy_type": StrategyType.PATTERN_RECOGNITION.value,
            "priority": StrategyPriority.NORMAL.value,
            "symbol": signal.get("symbol"),
            "timeframe": signal.get("timeframe", "1h"),
            "direction": direction,
            "signal_type": "entry",
            "price": signal.get("price"),
            "tags": ["pattern", signal.get("pattern_type", "unknown")],
            "confidence": signal.get("confidence", 0.5),
            "metadata": {
                "pattern_type": signal.get("pattern_type"),
                "pattern_strength": signal.get("strength")
            },
            "raw_signal": signal
        }
        
        # Record the normalized signal
        self.normalized_signals.append(normalized)
        
        return normalized
    
    def _process_signal(self, signal: Dict[str, Any]) -> None:
        """
        Process a normalized signal.
        
        Args:
            signal: Normalized signal
        """
        # Add to pending signals
        signal_id = signal["id"]
        self.pending_signals[signal_id] = signal
        
        # Add to signal history
        symbol = signal["symbol"]
        timeframe = signal["timeframe"]
        self.signal_history[symbol][timeframe].append(signal)
        
        # Check for conflicts
        conflict = self._check_for_conflicts(signal)
        
        if conflict:
            # Handle conflict
            self._handle_conflict(conflict)
        else:
            # No conflict, proceed with the signal
            self._execute_signal(signal)
    
    def _check_for_conflicts(self, signal: Dict[str, Any]) -> Optional[SignalConflict]:
        """
        Check if a signal conflicts with other pending signals.
        
        Args:
            signal: The signal to check
            
        Returns:
            A SignalConflict if conflicts exist, None otherwise
        """
        symbol = signal["symbol"]
        timeframe = signal["timeframe"]
        
        # Find other recent signals for the same symbol and timeframe
        recent_signals = []
        for s in self.pending_signals.values():
            if (s["symbol"] == symbol and 
                s["timeframe"] == timeframe and 
                s["id"] != signal["id"] and
                (datetime.now() - s["timestamp"]).total_seconds() < 3600):  # Within last hour
                recent_signals.append(s)
        
        if not recent_signals:
            return None
        
        # Check for direction conflicts
        conflicts = []
        for s in recent_signals:
            if s["direction"] != signal["direction"] and s["direction"] != "flat" and signal["direction"] != "flat":
                conflicts.append(s)
        
        if not conflicts:
            return None
        
        # Create conflict
        strategies = [signal["strategy_id"]] + [s["strategy_id"] for s in conflicts]
        conflict_signals = [signal] + conflicts
        
        return SignalConflict(
            symbol=symbol,
            timeframe=timeframe,
            strategies=strategies,
            signals=conflict_signals
        )
    
    def _handle_conflict(self, conflict: SignalConflict) -> None:
        """
        Handle a signal conflict.
        
        Args:
            conflict: The conflict to handle
        """
        logger.info(f"Handling conflict for {conflict.symbol} on {conflict.timeframe} between {len(conflict.strategies)} strategies")
        
        # Record the conflict
        self.active_conflicts[conflict.conflict_id] = conflict
        
        # Get conflict resolution mode
        resolution_mode = self._get_conflict_resolution_mode(conflict)
        
        # Resolve based on mode
        if resolution_mode == ConflictResolutionMode.PRIORITY_BASED:
            self._resolve_by_priority(conflict)
        elif resolution_mode == ConflictResolutionMode.PERFORMANCE_BASED:
            self._resolve_by_performance(conflict)
        elif resolution_mode == ConflictResolutionMode.NEWER_SIGNAL:
            self._resolve_by_timestamp(conflict)
        elif resolution_mode == ConflictResolutionMode.CONSERVATIVE:
            self._resolve_conservative(conflict)
        elif resolution_mode == ConflictResolutionMode.SPLIT_ALLOCATION:
            self._resolve_by_splitting(conflict)
        elif resolution_mode == ConflictResolutionMode.MANUAL:
            self._queue_for_manual_resolution(conflict)
        else:
            # Default to priority-based
            self._resolve_by_priority(conflict)
    
    def _get_conflict_resolution_mode(self, conflict: SignalConflict) -> ConflictResolutionMode:
        """
        Determine the appropriate conflict resolution mode.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Conflict resolution mode
        """
        # Check if all strategies use the same mode
        modes = []
        for strategy_id in conflict.strategies:
            if strategy_id in self.strategies:
                modes.append(self.strategies[strategy_id].conflict_resolution)
        
        if len(set(modes)) == 1:
            # All strategies have the same preference
            return modes[0]
        
        # Default to the coordinator's default mode
        return self.default_conflict_mode
    
    def _resolve_by_priority(self, conflict: SignalConflict) -> None:
        """
        Resolve conflict by priority - highest priority strategy wins.
        
        Args:
            conflict: The conflict to resolve
        """
        # Get priorities for each signal
        priorities = []
        for signal in conflict.signals:
            strategy_id = signal["strategy_id"]
            priority = self.get_effective_priority(strategy_id)
            priorities.append((signal, priority))
        
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: x[1].value, reverse=True)
        
        # Execute the signal with the highest priority
        winning_signal = priorities[0][0]
        
        # Record resolution
        conflict.resolution = {
            "mode": "priority_based",
            "winning_strategy": winning_signal["strategy_id"],
            "winning_priority": priorities[0][1].value
        }
        conflict.resolved = True
        
        # Move to resolved conflicts
        del self.active_conflicts[conflict.conflict_id]
        self.resolved_conflicts.append(conflict)
        
        # Execute the winning signal
        self._execute_signal(winning_signal)
        
        # Mark other signals as rejected
        for signal, _ in priorities[1:]:
            self._reject_signal(signal, "lower_priority")
    
    def _resolve_by_performance(self, conflict: SignalConflict) -> None:
        """
        Resolve conflict by performance - best performing strategy wins.
        
        Args:
            conflict: The conflict to resolve
        """
        # Get performance for each strategy
        performances = []
        for signal in conflict.signals:
            strategy_id = signal["strategy_id"]
            performance = self._get_strategy_performance(strategy_id)
            performances.append((signal, performance))
        
        # Sort by performance score (highest first)
        performances.sort(key=lambda x: x[1], reverse=True)
        
        # Execute the signal with the best performance
        winning_signal = performances[0][0]
        
        # Record resolution
        conflict.resolution = {
            "mode": "performance_based",
            "winning_strategy": winning_signal["strategy_id"],
            "winning_performance": performances[0][1]
        }
        conflict.resolved = True
        
        # Move to resolved conflicts
        del self.active_conflicts[conflict.conflict_id]
        self.resolved_conflicts.append(conflict)
        
        # Execute the winning signal
        self._execute_signal(winning_signal)
        
        # Mark other signals as rejected
        for signal, _ in performances[1:]:
            self._reject_signal(signal, "lower_performance")
    
    def _resolve_by_timestamp(self, conflict: SignalConflict) -> None:
        """
        Resolve conflict by timestamp - newest signal wins.
        
        Args:
            conflict: The conflict to resolve
        """
        # Sort by timestamp (newest first)
        signals = list(conflict.signals)
        signals.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Execute the newest signal
        winning_signal = signals[0]
        
        # Record resolution
        conflict.resolution = {
            "mode": "newer_signal",
            "winning_strategy": winning_signal["strategy_id"],
            "winning_timestamp": winning_signal["timestamp"].isoformat()
        }
        conflict.resolved = True
        
        # Move to resolved conflicts
        del self.active_conflicts[conflict.conflict_id]
        self.resolved_conflicts.append(conflict)
        
        # Execute the winning signal
        self._execute_signal(winning_signal)
        
        # Mark other signals as rejected
        for signal in signals[1:]:
            self._reject_signal(signal, "older_signal")
    
    def _resolve_conservative(self, conflict: SignalConflict) -> None:
        """
        Resolve conflict conservatively - reject all conflicting signals.
        
        Args:
            conflict: The conflict to resolve
        """
        # Record resolution
        conflict.resolution = {
            "mode": "conservative",
            "rejection_reason": "conflicting_signals"
        }
        conflict.resolved = True
        
        # Move to resolved conflicts
        del self.active_conflicts[conflict.conflict_id]
        self.resolved_conflicts.append(conflict)
        
        # Reject all signals
        for signal in conflict.signals:
            self._reject_signal(signal, "conflict_conservative")
    
    def _resolve_by_splitting(self, conflict: SignalConflict) -> None:
        """
        Resolve conflict by splitting allocation between strategies.
        
        Args:
            conflict: The conflict to resolve
        """
        # Group signals by direction
        direction_groups = defaultdict(list)
        for signal in conflict.signals:
            direction_groups[signal["direction"]].append(signal)
        
        # Record resolution
        conflict.resolution = {
            "mode": "split_allocation",
            "directions": list(direction_groups.keys()),
            "signals_per_direction": {d: len(s) for d, s in direction_groups.items()}
        }
        conflict.resolved = True
        
        # Move to resolved conflicts
        del self.active_conflicts[conflict.conflict_id]
        self.resolved_conflicts.append(conflict)
        
        # Execute all signals with reduced allocation
        for direction, signals in direction_groups.items():
            allocation_factor = 1.0 / len(signals)
            for signal in signals:
                # Modify signal to reduce allocation
                signal_copy = signal.copy()
                signal_copy["allocation_factor"] = allocation_factor
                signal_copy["metadata"]["split_allocation"] = True
                signal_copy["metadata"]["original_allocation"] = 1.0
                signal_copy["metadata"]["adjusted_allocation"] = allocation_factor
                
                # Execute with reduced allocation
                self._execute_signal(signal_copy)
    
    def _queue_for_manual_resolution(self, conflict: SignalConflict) -> None:
        """
        Queue conflict for manual resolution.
        
        Args:
            conflict: The conflict to queue
        """
        # Publish event for UI to handle
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_CONFLICT,
            data=conflict.to_dict()
        ))
        
        logger.info(f"Conflict {conflict.conflict_id} queued for manual resolution")
    
    def _execute_signal(self, signal: Dict[str, Any]) -> None:
        """
        Execute a signal.
        
        Args:
            signal: Signal to execute
        """
        # Remove from pending
        signal_id = signal["id"]
        if signal_id in self.pending_signals:
            del self.pending_signals[signal_id]
        
        # Add to processed
        self.processed_signals[signal_id] = signal
        
        # Create trade signal event
        self.event_bus.publish(Event(
            event_type=EventType.TRADE_SIGNAL,
            data=signal
        ))
        
        logger.info(f"Executed signal {signal_id} for {signal['symbol']} {signal['direction']}")
    
    def _reject_signal(self, signal: Dict[str, Any], reason: str) -> None:
        """
        Reject a signal.
        
        Args:
            signal: Signal to reject
            reason: Reason for rejection
        """
        # Remove from pending
        signal_id = signal["id"]
        if signal_id in self.pending_signals:
            del self.pending_signals[signal_id]
        
        # Mark as rejected
        signal["rejected"] = True
        signal["rejection_reason"] = reason
        signal["rejection_timestamp"] = datetime.now()
        
        # Add to processed
        self.processed_signals[signal_id] = signal
        
        # Publish rejection event
        self.event_bus.publish(Event(
            event_type=EventType.SIGNAL_REJECTED,
            data=signal
        ))
        
        logger.info(f"Rejected signal {signal_id} for {signal['symbol']} {signal['direction']}: {reason}")
    
    def _get_strategy_performance(self, strategy_id: str) -> float:
        """
        Get the performance score for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Performance score (higher is better)
        """
        # Check if we have the performance manager
        if not self.performance_manager:
            return 0.5  # Default mid-point
        
        # Get status from performance manager
        status = self.performance_manager.get_strategy_status(strategy_id)
        
        # Map status to score
        status_scores = {
            "PROMOTED": 0.9,
            "ACTIVE": 0.7,
            "PROBATION": 0.3,
            "RETIRED": 0.1,
            "NEW": 0.5,
            "INACTIVE": 0.0
        }
        
        return status_scores.get(status, 0.5)
    
    def _update_strategy_performance(self, strategy_id: str, performance_data: Dict[str, Any]) -> None:
        """
        Update strategy performance tracking.
        
        Args:
            strategy_id: Strategy ID
            performance_data: Performance metrics
        """
        # If this strategy has a priority override, check if we should remove it
        if strategy_id in self.priority_overrides:
            # Auto-adjust priority based on performance
            metrics = performance_data.get("metrics", {})
            sharpe_ratio = metrics.get("sharpe_ratio", 0)
            win_rate = metrics.get("win_rate", 0)
            
            # If performance is poor, remove the override
            if sharpe_ratio < 0.5 and win_rate < 0.4:
                del self.priority_overrides[strategy_id]
                logger.info(f"Removed priority override for {strategy_id} due to poor performance")


# Create singleton instance
coordinator = UnifiedStrategyCoordinator()
ServiceRegistry.register("unified_strategy_coordinator", coordinator)
