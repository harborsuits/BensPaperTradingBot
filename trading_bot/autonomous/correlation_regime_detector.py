#!/usr/bin/env python3
"""
Correlation Regime Detector

This module detects market regime changes based on correlation structure analysis
and provides risk parameter adjustments based on the current regime.
"""

import os
import json
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import time

# For PCA and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import correlation components
from trading_bot.autonomous.correlation_monitor import CorrelationMonitor, get_correlation_monitor

# Import risk and deployment components
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.risk.risk_manager import RiskLevel

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """Types of market regimes."""
    STABLE = "stable"          # Low correlation, normal volatility
    VOLATILE = "volatile"      # High correlation, high volatility  
    TRANSITIONAL = "transitional"  # Changing correlation structure
    UNKNOWN = "unknown"        # Not enough data or unclear regime


class CorrelationRegimeDetector:
    """
    Detects market regime changes based on correlation structure analysis
    and provides risk parameter adjustments based on the current regime.
    """
    
    def __init__(self, 
                 correlation_monitor: Optional[CorrelationMonitor] = None,
                 event_bus: Optional[EventBus] = None,
                 persistence_dir: Optional[str] = None):
        """
        Initialize the regime detector.
        
        Args:
            correlation_monitor: CorrelationMonitor instance
            event_bus: Event bus for communication
            persistence_dir: Directory for persisting state
        """
        self.correlation_monitor = correlation_monitor or get_correlation_monitor()
        self.event_bus = event_bus or EventBus()
        
        # Directory for persisting state
        self.persistence_dir = persistence_dir or os.path.join(
            os.path.expanduser("~"), 
            ".trading_bot", 
            "correlation_regime"
        )
        
        # Ensure directory exists
        os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Risk manager
        self.risk_manager = get_autonomous_risk_manager()
        
        # Regime detection parameters
        self.config = {
            "lookback_period": 90,  # 90 days for regime detection
            "regime_change_threshold": 0.3,  # Significant change threshold
            "detection_interval_days": 1,  # Check daily
            "min_strategies": 3,  # Minimum strategies needed for detection
            "pca_components": 2,  # Principal components for dimension reduction
            "cluster_count": 3,  # Number of regime clusters
            "detection_enabled": True,  # Enable/disable regime detection
            "auto_adjust_risk": True,  # Auto-adjust risk parameters on regime change
            "detection_interval_seconds": 86400,  # Check daily (in seconds)
            "eigenvalue_history_size": 30,  # Store 30 days of eigenvalues
            "regime_stability_window": 5,  # 5 days to confirm regime change
            "max_regime_history": 100,  # Maximum regime history events to store
        }
        
        # Current regime state
        self.current_regime = RegimeType.UNKNOWN
        self.current_regime_start = datetime.now()
        self.regime_confidence = 0.0
        
        # Regime history: (timestamp, regime, confidence, description)
        self.regime_history: List[Tuple[datetime, RegimeType, float, str]] = []
        
        # Eigenvalue history for PCA tracking
        self.eigenvalue_history: List[Tuple[datetime, List[float], float]] = []
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.last_detection_time = None
        self._lock = threading.RLock()
        
        # Load state if available
        self._load_state()
        
        # Register for events
        self._register_for_events()
        
        logger.info("Correlation Regime Detector initialized")
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        # Register for correlation report events
        self.event_bus.register("CORRELATION_REPORT_GENERATED", self._handle_correlation_report)
        
        logger.info("Registered for events")
    
    def start_monitoring(self) -> None:
        """Start regime detection monitoring."""
        if self.is_running:
            logger.warning("Regime detection already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started regime detection monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop regime detection monitoring."""
        if not self.is_running:
            logger.warning("Regime detection not running")
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped regime detection monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Check if it's time to detect regime
                now = datetime.now()
                if (not self.last_detection_time or 
                    (now - self.last_detection_time).total_seconds() >= self.config["detection_interval_seconds"]):
                    
                    # Detect regime change
                    if self.config["detection_enabled"]:
                        self.detect_regime_change()
                        self.last_detection_time = now
                
            except Exception as e:
                logger.error(f"Error in regime detection loop: {e}")
            
            # Sleep for a while
            time.sleep(3600)  # Check hourly if it's time to detect regime
    
    def detect_regime_change(self) -> bool:
        """
        Detect market regime change using correlation structure analysis.
        
        Returns:
            True if regime changed
        """
        with self._lock:
            # Get correlation matrix
            correlation_matrix = self.correlation_monitor.get_correlation_matrix()
            
            # If not enough strategies, we can't detect regime
            strategies = list(correlation_matrix.keys())
            if len(strategies) < self.config["min_strategies"]:
                logger.info(f"Not enough strategies ({len(strategies)}) for regime detection")
                return False
            
            # Convert to numpy matrix for analysis
            matrix_values = []
            for s1 in strategies:
                row = []
                for s2 in strategies:
                    row.append(correlation_matrix[s1][s2])
                matrix_values.append(row)
            
            correlation_np = np.array(matrix_values)
            
            # Calculate eigenvalues
            eigenvalues, eigenvectors = self.calculate_correlation_eigenvalues(correlation_np)
            
            # Store eigenvalue history
            self._add_to_eigenvalue_history(eigenvalues)
            
            # Classify regime
            new_regime, confidence, description = self.classify_regime(correlation_np, eigenvalues)
            
            # Check if regime changed
            regime_changed = (new_regime != self.current_regime)
            
            if regime_changed:
                # Check if we need to confirm the change
                if self.config["regime_stability_window"] > 1:
                    # Count recent regime classifications
                    recent_regimes = [r for t, r, c, d in self.regime_history[-self.config["regime_stability_window"]:]]
                    
                    # Only change if most recent regimes agree
                    if len(recent_regimes) < self.config["regime_stability_window"]:
                        # Not enough history yet
                        regime_changed = False
                    else:
                        # Count occurrences of new regime
                        new_regime_count = recent_regimes.count(new_regime)
                        
                        # Require majority for confirmation
                        if new_regime_count < len(recent_regimes) // 2 + 1:
                            regime_changed = False
                            logger.info(f"Potential regime change to {new_regime.value} not confirmed yet")
            
            # Add to history regardless of change
            self.regime_history.append((datetime.now(), new_regime, confidence, description))
            
            # Trim history if needed
            if len(self.regime_history) > self.config["max_regime_history"]:
                self.regime_history = self.regime_history[-self.config["max_regime_history"]:]
            
            # If regime changed, update state and take actions
            if regime_changed:
                old_regime = self.current_regime
                self.current_regime = new_regime
                self.current_regime_start = datetime.now()
                self.regime_confidence = confidence
                
                logger.warning(
                    f"Market regime changed from {old_regime.value} to {new_regime.value} "
                    f"(confidence: {confidence:.2f})"
                )
                
                # Issue warning
                self.issue_regime_warning(old_regime, new_regime, confidence, description)
                
                # Adjust risk parameters if enabled
                if self.config["auto_adjust_risk"]:
                    self.adapt_risk_parameters(new_regime)
                
                # Emit event
                self._emit_event(
                    event_type="REGIME_CHANGE_DETECTED",
                    data={
                        "old_regime": old_regime.value,
                        "new_regime": new_regime.value,
                        "confidence": confidence,
                        "description": description,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Save state
                self._save_state()
            
            return regime_changed
    
    def calculate_correlation_eigenvalues(self, correlation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate eigenvalues of correlation matrix for PCA analysis.
        
        Args:
            correlation_matrix: Numpy array of correlation matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Ensure matrix is symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _add_to_eigenvalue_history(self, eigenvalues: np.ndarray) -> None:
        """
        Add eigenvalues to history.
        
        Args:
            eigenvalues: Array of eigenvalues
        """
        # Calculate dispersion ratio (largest/sum)
        dispersion = float(eigenvalues[0] / np.sum(eigenvalues))
        
        # Store in history
        self.eigenvalue_history.append((
            datetime.now(),
            eigenvalues.tolist()[:3],  # Store top 3 eigenvalues
            dispersion
        ))
        
        # Trim history if needed
        if len(self.eigenvalue_history) > self.config["eigenvalue_history_size"]:
            self.eigenvalue_history = self.eigenvalue_history[-self.config["eigenvalue_history_size"]:]
    
    def classify_regime(self, 
                      correlation_matrix: np.ndarray, 
                      eigenvalues: np.ndarray) -> Tuple[RegimeType, float, str]:
        """
        Classify market regime based on correlation structure.
        
        Uses eigenvalue analysis and heuristic rules.
        
        Args:
            correlation_matrix: Numpy array of correlation matrix
            eigenvalues: Eigenvalues of correlation matrix
            
        Returns:
            Tuple of (regime_type, confidence, description)
        """
        # Calculate key metrics
        dispersion = float(eigenvalues[0] / np.sum(eigenvalues))
        avg_correlation = float(np.mean(correlation_matrix) - np.trace(correlation_matrix) / len(correlation_matrix))
        
        # Check for eigenvalue dispersion change
        dispersion_change = 0.0
        if len(self.eigenvalue_history) > 5:
            # Calculate average dispersion over last 5 periods
            prev_dispersion = np.mean([d for _, _, d in self.eigenvalue_history[-6:-1]])
            dispersion_change = dispersion - prev_dispersion
        
        # Decision rules for regime classification
        # These are heuristic rules based on financial literature
        
        # High dispersion + high correlation = Volatile regime
        if dispersion > 0.6 and avg_correlation > 0.4:
            regime = RegimeType.VOLATILE
            confidence = min(1.0, dispersion * avg_correlation * 2)
            description = "High correlation across strategies, market in crisis/volatile regime"
        
        # Low dispersion + low correlation = Stable regime
        elif dispersion < 0.4 and avg_correlation < 0.3:
            regime = RegimeType.STABLE
            confidence = min(1.0, (0.4 - dispersion) * (0.3 - avg_correlation) * 10)
            description = "Low correlation across strategies, market in stable regime"
        
        # Significant dispersion change = Transitional regime
        elif abs(dispersion_change) > 0.1:
            regime = RegimeType.TRANSITIONAL
            confidence = min(1.0, abs(dispersion_change) * 5)
            description = f"Correlation structure changing, market in transitional regime (dispersion change: {dispersion_change:.3f})"
        
        # Fallback to current regime if confidence is low
        else:
            # If confidence would be low, keep current regime
            calculated_confidence = 0.5  # Medium confidence
            
            if self.current_regime != RegimeType.UNKNOWN:
                regime = self.current_regime
                confidence = max(0.3, self.regime_confidence * 0.9)  # Decay confidence slightly
                description = f"No significant change detected, maintaining {regime.value} regime"
            else:
                # Make a best guess
                if avg_correlation > 0.4:
                    regime = RegimeType.VOLATILE
                    confidence = 0.3
                    description = "Moderately high correlation, possible volatile regime"
                else:
                    regime = RegimeType.STABLE
                    confidence = 0.3
                    description = "Moderate correlation, possible stable regime"
        
        return regime, confidence, description
    
    def adapt_risk_parameters(self, regime: RegimeType) -> None:
        """
        Adapt risk parameters based on detected regime.
        
        Args:
            regime: Current market regime
        """
        logger.info(f"Adapting risk parameters for {regime.value} regime")
        
        # Get active strategies from correlation monitor
        correlation_matrix = self.correlation_monitor.get_correlation_matrix()
        strategies = list(correlation_matrix.keys())
        
        # Apply regime-specific adjustments
        if regime == RegimeType.VOLATILE:
            # Reduce risk in volatile regime
            for strategy_id in strategies:
                # Lower risk level
                self.risk_manager.set_strategy_risk_level(strategy_id, RiskLevel.LOW)
                
                # Reduce allocation
                current_allocation = self.risk_manager.get_strategy_allocation(strategy_id)
                if current_allocation:
                    new_allocation = current_allocation * 0.7  # 30% reduction
                    self.risk_manager.adjust_allocation(strategy_id, new_allocation)
                    
                    logger.info(f"Reduced allocation for {strategy_id} to {new_allocation:.1f}% in volatile regime")
            
            # Emit event
            self._emit_event(
                event_type="RISK_PARAMETERS_ADJUSTED",
                data={
                    "regime": regime.value,
                    "adjustment_type": "risk_reduction",
                    "affected_strategies": len(strategies),
                    "timestamp": datetime.now().isoformat()
                }
            )
                
        elif regime == RegimeType.STABLE:
            # Optimize for return in stable regime
            for strategy_id in strategies:
                # Standard risk level
                self.risk_manager.set_strategy_risk_level(strategy_id, RiskLevel.MEDIUM)
                
                # Normal allocation (no adjustment needed)
                pass
            
            # Emit event
            self._emit_event(
                event_type="RISK_PARAMETERS_ADJUSTED",
                data={
                    "regime": regime.value,
                    "adjustment_type": "standard_risk",
                    "affected_strategies": len(strategies),
                    "timestamp": datetime.now().isoformat()
                }
            )
                
        elif regime == RegimeType.TRANSITIONAL:
            # Cautious approach in transitional regime
            for strategy_id in strategies:
                # Medium-low risk level
                self.risk_manager.set_strategy_risk_level(strategy_id, RiskLevel.MEDIUM_LOW)
                
                # Moderate allocation reduction
                current_allocation = self.risk_manager.get_strategy_allocation(strategy_id)
                if current_allocation:
                    new_allocation = current_allocation * 0.85  # 15% reduction
                    self.risk_manager.adjust_allocation(strategy_id, new_allocation)
                    
                    logger.info(f"Adjusted allocation for {strategy_id} to {new_allocation:.1f}% in transitional regime")
            
            # Emit event
            self._emit_event(
                event_type="RISK_PARAMETERS_ADJUSTED",
                data={
                    "regime": regime.value,
                    "adjustment_type": "cautious_adjustment",
                    "affected_strategies": len(strategies),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def issue_regime_warning(self, 
                           old_regime: RegimeType, 
                           new_regime: RegimeType,
                           confidence: float,
                           description: str) -> None:
        """
        Issue warning for regime change.
        
        Args:
            old_regime: Previous regime
            new_regime: New regime
            confidence: Confidence in regime classification
            description: Detailed description
        """
        # Determine warning level
        if new_regime == RegimeType.VOLATILE:
            warning_level = "critical"
        elif new_regime == RegimeType.TRANSITIONAL:
            warning_level = "warning"
        else:
            warning_level = "info"
        
        # Log warning
        if warning_level == "critical":
            logger.critical(f"REGIME CHANGE ALERT: {old_regime.value} -> {new_regime.value}: {description}")
        elif warning_level == "warning":
            logger.warning(f"Regime change: {old_regime.value} -> {new_regime.value}: {description}")
        else:
            logger.info(f"Regime change: {old_regime.value} -> {new_regime.value}: {description}")
        
        # Emit warning event
        self._emit_event(
            event_type="REGIME_WARNING_ISSUED",
            data={
                "old_regime": old_regime.value,
                "new_regime": new_regime.value,
                "warning_level": warning_level,
                "confidence": confidence,
                "description": description,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_regime_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get regime history.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of regime history items
        """
        history = []
        for timestamp, regime, confidence, description in self.regime_history[-limit:]:
            history.append({
                "timestamp": timestamp.isoformat(),
                "regime": regime.value,
                "confidence": confidence,
                "description": description
            })
        
        return history
    
    def get_current_regime_info(self) -> Dict[str, Any]:
        """
        Get information about current regime.
        
        Returns:
            Dict with regime information
        """
        return {
            "regime": self.current_regime.value,
            "confidence": self.regime_confidence,
            "since": self.current_regime_start.isoformat(),
            "duration_days": (datetime.now() - self.current_regime_start).days,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_eigenvalue_trends(self) -> Dict[str, Any]:
        """
        Get eigenvalue trends for visualization.
        
        Returns:
            Dict with eigenvalue trends
        """
        if not self.eigenvalue_history:
            return {
                "timestamps": [],
                "eigenvalues": [],
                "dispersion": []
            }
        
        # Unpack history
        timestamps = [t.isoformat() for t, _, _ in self.eigenvalue_history]
        eigenvalues = []
        dispersion = [d for _, _, d in self.eigenvalue_history]
        
        # Extract top eigenvalues
        for _, evals, _ in self.eigenvalue_history:
            eigenvalues.append(evals)
        
        return {
            "timestamps": timestamps,
            "eigenvalues": eigenvalues,
            "dispersion": dispersion
        }
    
    def _handle_correlation_report(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle correlation report event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # When correlation report is generated, we can detect regime change
        if self.config["detection_enabled"]:
            try:
                self.detect_regime_change()
            except Exception as e:
                logger.error(f"Error detecting regime change: {e}")
    
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
            # Create event
            event = Event(
                event_type=event_type,
                source="CorrelationRegimeDetector",
                data=data,
                timestamp=datetime.now()
            )
            
            # Publish event
            self.event_bus.publish(event)
                
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    def _get_state_file_path(self) -> str:
        """Get path to state file."""
        return os.path.join(self.persistence_dir, "regime_state.json")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        with self._lock:
            try:
                # Convert regime history to serializable form
                regime_history = [
                    {
                        "timestamp": t.isoformat(),
                        "regime": r.value,
                        "confidence": c,
                        "description": d
                    }
                    for t, r, c, d in self.regime_history
                ]
                
                # Convert eigenvalue history to serializable form
                eigenvalue_history = [
                    {
                        "timestamp": t.isoformat(),
                        "eigenvalues": e,
                        "dispersion": d
                    }
                    for t, e, d in self.eigenvalue_history
                ]
                
                # Create state dictionary
                state = {
                    "config": self.config,
                    "current_regime": self.current_regime.value,
                    "current_regime_start": self.current_regime_start.isoformat(),
                    "regime_confidence": self.regime_confidence,
                    "regime_history": regime_history,
                    "eigenvalue_history": eigenvalue_history,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Write to file
                with open(self._get_state_file_path(), 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.info("Saved regime detector state to disk")
                
            except Exception as e:
                logger.error(f"Error saving state: {e}")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self._get_state_file_path()
        
        if not os.path.exists(state_file):
            logger.info("No state file found, starting with default state")
            return
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore configuration
            if "config" in state:
                saved_config = state["config"]
                # Only update keys that exist in both
                for key in self.config:
                    if key in saved_config:
                        self.config[key] = saved_config[key]
            
            # Restore current regime
            if "current_regime" in state:
                self.current_regime = RegimeType(state["current_regime"])
            
            if "current_regime_start" in state:
                self.current_regime_start = datetime.fromisoformat(state["current_regime_start"])
            
            if "regime_confidence" in state:
                self.regime_confidence = state["regime_confidence"]
            
            # Restore regime history
            if "regime_history" in state:
                self.regime_history = [
                    (
                        datetime.fromisoformat(item["timestamp"]),
                        RegimeType(item["regime"]),
                        item["confidence"],
                        item["description"]
                    )
                    for item in state["regime_history"]
                ]
            
            # Restore eigenvalue history
            if "eigenvalue_history" in state:
                self.eigenvalue_history = [
                    (
                        datetime.fromisoformat(item["timestamp"]),
                        item["eigenvalues"],
                        item["dispersion"]
                    )
                    for item in state["eigenvalue_history"]
                ]
            
            logger.info(
                f"Loaded regime detector state from disk, current regime: {self.current_regime.value}"
            )
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Continue with default state


# Singleton instance for global access
_correlation_regime_detector = None

def get_correlation_regime_detector(correlation_monitor: Optional[CorrelationMonitor] = None,
                                   event_bus: Optional[EventBus] = None,
                                   persistence_dir: Optional[str] = None) -> CorrelationRegimeDetector:
    """
    Get singleton instance of correlation regime detector.
    
    Args:
        correlation_monitor: CorrelationMonitor instance
        event_bus: Event bus for communication
        persistence_dir: Directory for persisting state
        
    Returns:
        CorrelationRegimeDetector instance
    """
    global _correlation_regime_detector
    
    if _correlation_regime_detector is None:
        _correlation_regime_detector = CorrelationRegimeDetector(
            correlation_monitor=correlation_monitor,
            event_bus=event_bus,
            persistence_dir=persistence_dir
        )
        
    return _correlation_regime_detector


# Define custom event types for regime detection
def register_regime_event_types(event_bus: EventBus) -> None:
    """
    Register regime-related event types with event bus.
    
    Args:
        event_bus: Event bus to register with
    """
    # Define custom event types
    regime_event_types = [
        "REGIME_CHANGE_DETECTED",
        "REGIME_WARNING_ISSUED",
        "RISK_PARAMETERS_ADJUSTED"
    ]
    
    # Register each type
    for event_type in regime_event_types:
        if not hasattr(EventType, event_type):
            # Only add if not already defined
            setattr(EventType, event_type, event_type)
    
    logger.info("Registered regime event types")
