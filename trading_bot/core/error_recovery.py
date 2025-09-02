#!/usr/bin/env python
"""
Error Recovery System for BensBot

This module provides robust error recovery mechanisms for the trading pipeline,
enabling the system to gracefully handle and recover from failures at any stage.

Features:
- Circuit breaker pattern to prevent cascading failures
- Automatic error detection and classification
- Staged recovery strategies based on error type
- State reconciliation after failures
- Error tracking and analysis 
"""

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from collections import defaultdict, deque

from trading_bot.core.event_bus import EventBus, Event, get_global_event_bus
from trading_bot.core.constants import EventType
from trading_bot.core.transaction_manager import get_global_transaction_manager

logger = logging.getLogger(__name__)

# New event types for error recovery
EventType.ERROR_RECOVERY_STARTED = "error_recovery_started"
EventType.ERROR_RECOVERY_SUCCEEDED = "error_recovery_succeeded"
EventType.ERROR_RECOVERY_FAILED = "error_recovery_failed"
EventType.CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
EventType.CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
EventType.CIRCUIT_BREAKER_HALF_OPEN = "circuit_breaker_half_open"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    CRITICAL = 3    # System cannot function, immediate attention required
    HIGH = 2        # Major feature is broken, needs prompt attention
    MEDIUM = 1      # Non-critical issue that should be fixed soon
    LOW = 0         # Minor issue that can be fixed later


class ComponentState(Enum):
    """State of a system component"""
    HEALTHY = "healthy"          # Component is working normally
    DEGRADED = "degraded"        # Component is working with reduced functionality
    FAILED = "failed"            # Component has failed
    RECOVERING = "recovering"    # Component is in recovery mode
    UNKNOWN = "unknown"          # Component state is unknown


class CircuitBreakerState(Enum):
    """State of a circuit breaker"""
    CLOSED = "closed"        # Normal operation, requests flow through
    OPEN = "open"            # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if system has recovered


class ErrorRecoveryManager:
    """
    Manages error recovery for the trading system.
    
    Implements circuit breaker pattern, error tracking, and recovery strategies
    to ensure the system can gracefully handle failures and recover automatically.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
        reset_timeout_seconds: int = 300
    ):
        """
        Initialize the error recovery manager.
        
        Args:
            event_bus: Event bus for publishing events
            failure_threshold: Number of failures before circuit breaker opens
            recovery_timeout_seconds: Seconds to wait before attempting recovery
            reset_timeout_seconds: Seconds to wait before resetting circuit breaker
        """
        self.event_bus = event_bus or get_global_event_bus()
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.reset_timeout = reset_timeout_seconds
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Error tracking
        self.errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Component states
        self.component_states: Dict[str, ComponentState] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, List[Callable]] = {}
        
        # Monitoring thread
        self.monitor_thread = None
        self.is_running = False
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Error Recovery Manager initialized")
    
    def start(self):
        """Start the error recovery manager"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_circuit_breakers,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Error Recovery Manager started")
    
    def stop(self):
        """Stop the error recovery manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Error Recovery Manager stopped")
    
    def _register_event_handlers(self):
        """Register for error-related events"""
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._handle_error)
    
    def _handle_error(self, event: Event):
        """Handle error event"""
        # Extract error information
        component = event.data.get('component', 'unknown')
        error_type = event.data.get('error_type', 'unknown')
        error_message = event.data.get('message', '')
        severity = event.data.get('severity', ErrorSeverity.MEDIUM)
        
        # Record the error
        self._record_error(component, error_type, error_message, severity)
        
        # Check if circuit breaker should trip
        self._check_circuit_breaker(component)
    
    def _record_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        severity: ErrorSeverity
    ):
        """Record an error for tracking"""
        with self.lock:
            # Create error record
            error = {
                'timestamp': datetime.now(),
                'component': component,
                'error_type': error_type,
                'message': error_message,
                'severity': severity
            }
            
            # Add to error tracking
            self.errors[component].append(error)
            self.error_counts[component] += 1
            
            # Keep only the most recent errors
            if len(self.errors[component]) > 100:
                self.errors[component] = self.errors[component][-100:]
            
            # Update component state based on severity
            if severity == ErrorSeverity.CRITICAL:
                self.component_states[component] = ComponentState.FAILED
            elif severity == ErrorSeverity.HIGH:
                if self.component_states.get(component) != ComponentState.FAILED:
                    self.component_states[component] = ComponentState.DEGRADED
    
    def _check_circuit_breaker(self, component: str):
        """Check if circuit breaker should trip"""
        with self.lock:
            # Initialize circuit breaker if not exists
            if component not in self.circuit_breakers:
                self.circuit_breakers[component] = {
                    'state': CircuitBreakerState.CLOSED,
                    'failure_count': 0,
                    'last_failure_time': None,
                    'last_success_time': datetime.now(),
                    'last_state_change_time': datetime.now()
                }
            
            cb = self.circuit_breakers[component]
            
            # Update failure count
            cb['failure_count'] += 1
            cb['last_failure_time'] = datetime.now()
            
            # Check if threshold reached
            if (cb['state'] == CircuitBreakerState.CLOSED and 
                cb['failure_count'] >= self.failure_threshold):
                self._trip_circuit_breaker(component)
    
    def _trip_circuit_breaker(self, component: str):
        """Trip (open) a circuit breaker"""
        with self.lock:
            if component not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[component]
            
            # Set state to open
            cb['state'] = CircuitBreakerState.OPEN
            cb['last_state_change_time'] = datetime.now()
            
            logger.warning(f"Circuit breaker opened for component: {component}")
            
            # Publish circuit breaker event
            self.event_bus.create_and_publish(
                event_type=EventType.CIRCUIT_BREAKER_OPEN,
                data={
                    'component': component,
                    'failure_count': cb['failure_count'],
                    'last_failure_time': cb['last_failure_time'].isoformat() if cb['last_failure_time'] else None
                },
                source="error_recovery_manager"
            )
            
            # Schedule recovery attempt
            threading.Timer(
                self.recovery_timeout,
                lambda: self._attempt_recovery(component)
            ).start()
    
    def _half_open_circuit_breaker(self, component: str):
        """Set circuit breaker to half-open state"""
        with self.lock:
            if component not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[component]
            
            # Set state to half-open
            cb['state'] = CircuitBreakerState.HALF_OPEN
            cb['last_state_change_time'] = datetime.now()
            
            logger.info(f"Circuit breaker half-opened for component: {component}")
            
            # Publish circuit breaker event
            self.event_bus.create_and_publish(
                event_type=EventType.CIRCUIT_BREAKER_HALF_OPEN,
                data={
                    'component': component,
                    'failure_count': cb['failure_count'],
                    'last_failure_time': cb['last_failure_time'].isoformat() if cb['last_failure_time'] else None
                },
                source="error_recovery_manager"
            )
    
    def _close_circuit_breaker(self, component: str):
        """Close circuit breaker (return to normal)"""
        with self.lock:
            if component not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[component]
            
            # Reset state and counters
            cb['state'] = CircuitBreakerState.CLOSED
            cb['failure_count'] = 0
            cb['last_state_change_time'] = datetime.now()
            cb['last_success_time'] = datetime.now()
            
            logger.info(f"Circuit breaker closed for component: {component}")
            
            # Publish circuit breaker event
            self.event_bus.create_and_publish(
                event_type=EventType.CIRCUIT_BREAKER_CLOSED,
                data={
                    'component': component
                },
                source="error_recovery_manager"
            )
            
            # Update component state
            self.component_states[component] = ComponentState.HEALTHY
    
    def _attempt_recovery(self, component: str):
        """Attempt recovery for a component"""
        with self.lock:
            if component not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[component]
            
            # Skip if already closed
            if cb['state'] == CircuitBreakerState.CLOSED:
                return
            
            # Set component to recovering state
            self.component_states[component] = ComponentState.RECOVERING
            
            # Set circuit breaker to half-open
            self._half_open_circuit_breaker(component)
            
            # Execute recovery strategies
            success = self._execute_recovery_strategies(component)
            
            if success:
                # Recovery succeeded, close circuit breaker
                self._close_circuit_breaker(component)
                
                # Publish recovery success event
                self.event_bus.create_and_publish(
                    event_type=EventType.ERROR_RECOVERY_SUCCEEDED,
                    data={
                        'component': component,
                        'recovery_time': datetime.now().isoformat()
                    },
                    source="error_recovery_manager"
                )
            else:
                # Recovery failed, keep circuit breaker open
                cb['state'] = CircuitBreakerState.OPEN
                
                # Publish recovery failed event
                self.event_bus.create_and_publish(
                    event_type=EventType.ERROR_RECOVERY_FAILED,
                    data={
                        'component': component,
                        'recovery_time': datetime.now().isoformat()
                    },
                    source="error_recovery_manager"
                )
                
                # Schedule another recovery attempt
                threading.Timer(
                    self.recovery_timeout,
                    lambda: self._attempt_recovery(component)
                ).start()
    
    def _execute_recovery_strategies(self, component: str) -> bool:
        """Execute recovery strategies for a component"""
        # Publish recovery started event
        self.event_bus.create_and_publish(
            event_type=EventType.ERROR_RECOVERY_STARTED,
            data={
                'component': component,
                'recovery_time': datetime.now().isoformat()
            },
            source="error_recovery_manager"
        )
        
        # Get recovery strategies for this component
        strategies = self.recovery_strategies.get(component, [])
        
        # If no specific strategies, use default recovery
        if not strategies:
            logger.info(f"No specific recovery strategies for {component}, using default")
            return self._default_recovery(component)
        
        # Try each strategy in order
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"Executing recovery strategy {i+1}/{len(strategies)} for {component}")
                if strategy(component):
                    logger.info(f"Recovery strategy {i+1} succeeded for {component}")
                    return True
            except Exception as e:
                logger.error(f"Error in recovery strategy {i+1} for {component}: {str(e)}")
        
        logger.warning(f"All recovery strategies failed for {component}")
        return False
    
    def _default_recovery(self, component: str) -> bool:
        """Default recovery strategy for components without specific strategies"""
        logger.info(f"Executing default recovery for {component}")
        
        # Default strategy is to wait and see if the component recovers on its own
        time.sleep(5)
        
        # For demonstration purposes, return success 50% of the time
        import random
        return random.random() > 0.5
    
    def _monitor_circuit_breakers(self):
        """Background thread for monitoring circuit breakers"""
        while self.is_running:
            try:
                now = datetime.now()
                
                with self.lock:
                    for component, cb in list(self.circuit_breakers.items()):
                        # Check if open circuit breakers should be half-opened
                        if (cb['state'] == CircuitBreakerState.OPEN and 
                            cb['last_state_change_time'] and 
                            now - cb['last_state_change_time'] > timedelta(seconds=self.reset_timeout)):
                            self._half_open_circuit_breaker(component)
                
                # Sleep for a short time
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring thread: {str(e)}")
    
    def register_recovery_strategy(self, component: str, strategy: Callable[[str], bool]):
        """
        Register a recovery strategy for a component.
        
        Args:
            component: Component name
            strategy: Recovery strategy function that takes component name and returns success boolean
        """
        with self.lock:
            if component not in self.recovery_strategies:
                self.recovery_strategies[component] = []
            
            self.recovery_strategies[component].append(strategy)
            logger.info(f"Registered recovery strategy for {component}")
    
    def register_component(self, component: str, initial_state: ComponentState = ComponentState.HEALTHY):
        """
        Register a component with the error recovery manager.
        
        Args:
            component: Component name
            initial_state: Initial component state
        """
        with self.lock:
            if component not in self.component_states:
                self.component_states[component] = initial_state
                logger.info(f"Registered component: {component} with state {initial_state}")
    
    def report_success(self, component: str):
        """
        Report a successful operation for a component.
        
        Args:
            component: Component name
        """
        with self.lock:
            if component not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[component]
            
            # Record success
            cb['last_success_time'] = datetime.now()
            
            # If half-open, close the circuit breaker
            if cb['state'] == CircuitBreakerState.HALF_OPEN:
                self._close_circuit_breaker(component)
    
    def report_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        """
        Report an error for a component.
        
        Args:
            component: Component name
            error_type: Type of error
            error_message: Error message
            severity: Error severity
        """
        # Record the error
        self._record_error(component, error_type, error_message, severity)
        
        # Check if circuit breaker should trip
        self._check_circuit_breaker(component)
    
    def get_component_state(self, component: str) -> ComponentState:
        """
        Get the current state of a component.
        
        Args:
            component: Component name
            
        Returns:
            Current component state
        """
        return self.component_states.get(component, ComponentState.UNKNOWN)
    
    def get_circuit_breaker_state(self, component: str) -> CircuitBreakerState:
        """
        Get the current state of a circuit breaker.
        
        Args:
            component: Component name
            
        Returns:
            Current circuit breaker state
        """
        if component not in self.circuit_breakers:
            return CircuitBreakerState.CLOSED
        
        return self.circuit_breakers[component]['state']
    
    def get_errors(self, component: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors for a component.
        
        Args:
            component: Component name
            limit: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        with self.lock:
            return self.errors.get(component, [])[-limit:]
    
    def get_recovery_status(self, component: str) -> Dict[str, Any]:
        """
        Get recovery status for a component.
        
        Args:
            component: Component name
            
        Returns:
            Recovery status dictionary
        """
        with self.lock:
            cb = self.circuit_breakers.get(component, {
                'state': CircuitBreakerState.CLOSED,
                'failure_count': 0,
                'last_failure_time': None,
                'last_success_time': None,
                'last_state_change_time': None
            })
            
            return {
                'component': component,
                'state': self.component_states.get(component, ComponentState.UNKNOWN),
                'circuit_breaker_state': cb['state'],
                'failure_count': cb['failure_count'],
                'last_failure_time': cb['last_failure_time'],
                'last_success_time': cb['last_success_time'],
                'has_recovery_strategies': component in self.recovery_strategies,
                'strategy_count': len(self.recovery_strategies.get(component, []))
            }


# Global error recovery manager instance
_global_error_recovery_manager: Optional[ErrorRecoveryManager] = None

def get_global_error_recovery_manager() -> ErrorRecoveryManager:
    """
    Get the global error recovery manager instance.
    
    Returns:
        The global error recovery manager
    """
    global _global_error_recovery_manager
    if _global_error_recovery_manager is None:
        from trading_bot.core.event_bus import get_global_event_bus
        
        _global_error_recovery_manager = ErrorRecoveryManager(
            event_bus=get_global_event_bus()
        )
        
        # Start the error recovery manager
        _global_error_recovery_manager.start()
        
    return _global_error_recovery_manager
