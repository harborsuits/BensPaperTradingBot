"""
Exit Strategy Manager Safeguards

This module provides robust safeguards for the Exit Strategy Manager component,
ensuring reliable exit execution, error recovery, and protection against
execution failures that could lead to capital loss.

Features:
- Execution verification with retry logic
- State monitoring and validation
- Exit condition redundancy checks
- Circuit breaker protections
- Fail-safe emergency exits
"""

import logging
import threading
import time
import traceback
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import copy

# Import needed components
from trading_bot.strategy.exit_manager import ExitStrategyManager
from trading_bot.robustness.system_safeguards import ComponentType, SafeguardState

logger = logging.getLogger(__name__)

class ExitManagerSafeguards:
    """
    Enhances the ExitStrategyManager with robust safeguards to ensure reliable
    exit execution and protection mechanisms.
    """
    
    def __init__(self, exit_manager: ExitStrategyManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize exit manager safeguards.
        
        Args:
            exit_manager: The exit strategy manager instance to enhance
            config: Configuration parameters
        """
        self.exit_manager = exit_manager
        self.config = config or {}
        
        # Exit execution tracking
        self.exit_execution_log: List[Dict[str, Any]] = []
        self.failed_exits: Dict[str, List[Dict[str, Any]]] = {}
        self.retry_queue: List[Dict[str, Any]] = []
        
        # Monitoring state
        self.monitoring_health: Dict[str, Any] = {
            'last_check': None,
            'is_running': False,
            'last_exception': None,
            'error_count': 0
        }
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {
            'exit_execution': {
                'tripped': False,
                'trip_time': None,
                'error_count': 0,
                'reset_time': None
            },
            'monitoring_thread': {
                'tripped': False,
                'trip_time': None,
                'error_count': 0,
                'reset_time': None
            }
        }
        
        # Emergency exit trigger state
        self.emergency_exit = {
            'triggered': False,
            'trigger_time': None,
            'trigger_reason': None
        }
        
        # Exit condition verification
        self.verification_log: List[Dict[str, Any]] = []
        self.last_verification: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._setup_safeguards()
        
        logger.info("Exit Manager safeguards initialized")
    
    def _setup_safeguards(self) -> None:
        """Setup safety mechanisms and hooks."""
        # Add method wrapping for execution safety
        if hasattr(self.exit_manager, 'execute_exit'):
            original_execute_exit = self.exit_manager.execute_exit
            
            def safe_execute_exit(*args, **kwargs):
                return self._safe_execute_exit(original_execute_exit, *args, **kwargs)
            
            self.exit_manager.execute_exit = safe_execute_exit
        
        # Enhance monitoring thread with safety checks
        if hasattr(self.exit_manager, '_start_monitoring'):
            original_start_monitoring = self.exit_manager._start_monitoring
            
            def safe_start_monitoring(*args, **kwargs):
                return self._safe_start_monitoring(original_start_monitoring, *args, **kwargs)
            
            self.exit_manager._start_monitoring = safe_start_monitoring
        
        # Enhance stop monitoring with safeguards
        if hasattr(self.exit_manager, '_stop_monitoring'):
            original_stop_monitoring = self.exit_manager._stop_monitoring
            
            def safe_stop_monitoring(*args, **kwargs):
                return self._safe_stop_monitoring(original_stop_monitoring, *args, **kwargs)
            
            self.exit_manager._stop_monitoring = safe_stop_monitoring
        
        # Add validation to check exits method
        if hasattr(self.exit_manager, '_check_exits'):
            original_check_exits = self.exit_manager._check_exits
            
            def safe_check_exits(*args, **kwargs):
                return self._safe_check_exits(original_check_exits, *args, **kwargs)
            
            self.exit_manager._check_exits = safe_check_exits
    
    def _safe_execute_exit(self, original_func, *args, **kwargs) -> Any:
        """
        Execute exit with safety mechanisms and retries.
        
        Args:
            original_func: Original exit execution function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or error handling result
        """
        # Extract position_id from args or kwargs
        position_id = None
        if len(args) > 0:
            position_id = args[0]
        elif 'position_id' in kwargs:
            position_id = kwargs['position_id']
        
        # Log exit attempt
        exit_attempt = {
            'position_id': position_id,
            'timestamp': datetime.now().isoformat(),
            'args': str(args),
            'kwargs': str(kwargs),
            'status': 'attempt'
        }
        self.exit_execution_log.append(exit_attempt)
        
        # Check if circuit breaker is tripped
        if self.circuit_breakers['exit_execution']['tripped']:
            logger.warning(f"Exit execution circuit breaker tripped, using emergency exit for position {position_id}")
            return self._emergency_exit_position(*args, **kwargs)
        
        try:
            # Execute original exit function
            result = original_func(*args, **kwargs)
            
            # Log successful exit
            exit_attempt['status'] = 'success'
            exit_attempt['result'] = str(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing exit for position {position_id}: {str(e)}")
            
            # Log failed exit
            exit_attempt['status'] = 'failed'
            exit_attempt['error'] = str(e)
            exit_attempt['traceback'] = traceback.format_exc()
            
            # Track failed exits by position
            if position_id not in self.failed_exits:
                self.failed_exits[position_id] = []
            self.failed_exits[position_id].append(exit_attempt)
            
            # Increment circuit breaker error count
            with self._lock:
                self.circuit_breakers['exit_execution']['error_count'] += 1
                
                # Trip circuit breaker if too many errors
                error_threshold = self.config.get('exit_error_threshold', 5)
                if self.circuit_breakers['exit_execution']['error_count'] >= error_threshold:
                    self._trip_circuit_breaker('exit_execution', f"Too many exit failures: {error_threshold}")
            
            # Check if retry is appropriate
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                # Add to retry queue for temporary connection issues
                retry_info = {
                    'position_id': position_id,
                    'args': args,
                    'kwargs': kwargs,
                    'attempt': 1,
                    'next_retry': datetime.now() + timedelta(seconds=5)
                }
                self.retry_queue.append(retry_info)
                logger.warning(f"Added position {position_id} to exit retry queue")
                
                # Don't raise exception, return info about retry
                return {'status': 'retry_scheduled', 'position_id': position_id}
            
            # For other errors, attempt emergency exit
            logger.warning(f"Attempting emergency exit for position {position_id}")
            return self._emergency_exit_position(*args, **kwargs)
    
    def _emergency_exit_position(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Attempt emergency exit for a position when normal exit fails.
        This is a last resort to close positions and protect capital.
        
        Args:
            *args, **kwargs: Original exit arguments
            
        Returns:
            Dict with emergency exit results
        """
        # Extract position_id from args or kwargs
        position_id = None
        if len(args) > 0:
            position_id = args[0]
        elif 'position_id' in kwargs:
            position_id = kwargs['position_id']
        
        result = {
            'status': 'emergency_exit_failed',
            'position_id': position_id,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': []
        }
        
        try:
            logger.critical(f"EMERGENCY EXIT triggered for position {position_id}")
            
            # Record emergency exit trigger
            self.emergency_exit['triggered'] = True
            self.emergency_exit['trigger_time'] = datetime.now().isoformat()
            self.emergency_exit['trigger_reason'] = f"Exit execution failure for position {position_id}"
            
            # Try direct broker exit if available
            if hasattr(self.exit_manager, 'broker_manager'):
                broker_manager = self.exit_manager.broker_manager
                
                try:
                    # Get position details
                    position = None
                    if hasattr(self.exit_manager, 'position_manager'):
                        position_manager = self.exit_manager.position_manager
                        if hasattr(position_manager, 'get_position'):
                            position = position_manager.get_position(position_id)
                    
                    if position:
                        # Create market order to close position
                        symbol = position.get('symbol')
                        quantity = position.get('quantity', 0)
                        direction = position.get('direction', 'long')
                        
                        # Determine opposing side for closing
                        side = 'sell' if direction == 'long' else 'buy'
                        
                        result['actions_taken'].append("Retrieved position details")
                        
                        # Try to close via broker manager
                        if hasattr(broker_manager, 'place_market_order'):
                            order_result = broker_manager.place_market_order(
                                symbol=symbol,
                                quantity=quantity,
                                side=side,
                                time_in_force='day',
                                order_class='simple'
                            )
                            
                            result['actions_taken'].append("Placed emergency market order")
                            result['order_result'] = str(order_result)
                            result['status'] = 'emergency_exit_success'
                            
                            logger.warning(f"Emergency exit market order placed for position {position_id}")
                            return result
                except Exception as broker_ex:
                    error_msg = f"Emergency exit broker action failed: {str(broker_ex)}"
                    logger.error(error_msg)
                    result['actions_taken'].append(f"Broker exit failed: {str(broker_ex)}")
            
            # If broker exit fails, try a more direct approach
            result['actions_taken'].append("Attempting direct position closure")
            
            # This is very implementation dependent, but we'll make a best effort
            if hasattr(self.exit_manager, 'position_manager') and hasattr(self.exit_manager.position_manager, 'close_position'):
                try:
                    close_result = self.exit_manager.position_manager.close_position(
                        position_id, 
                        exit_price=None,  # Market price
                        exit_time=datetime.now(),
                        exit_reason="EMERGENCY_EXIT_FAIL_SAFE"
                    )
                    
                    result['actions_taken'].append("Used position_manager.close_position")
                    result['close_result'] = str(close_result)
                    result['status'] = 'emergency_exit_success'
                    
                    logger.warning(f"Emergency position closure executed for {position_id}")
                    return result
                except Exception as pos_ex:
                    error_msg = f"Emergency position closure failed: {str(pos_ex)}"
                    logger.error(error_msg)
                    result['actions_taken'].append(f"Direct closure failed: {str(pos_ex)}")
            
            # If all else fails, try to mark the position as requiring manual intervention
            result['actions_taken'].append("All automatic emergency exits failed")
            result['status'] = 'emergency_exit_failed_manual_intervention_required'
            
            # Emit emergency event if possible
            if hasattr(self.exit_manager, 'event_bus') and hasattr(self.exit_manager.event_bus, 'emit'):
                self.exit_manager.event_bus.emit(
                    'emergency_exit_failure',
                    {
                        'position_id': position_id,
                        'timestamp': datetime.now().isoformat(),
                        'reason': "All automatic exit mechanisms failed",
                        'manual_intervention_required': True
                    }
                )
                result['actions_taken'].append("Emitted emergency_exit_failure event")
            
            logger.critical(f"EMERGENCY EXIT FAILED for position {position_id} - MANUAL INTERVENTION REQUIRED")
            return result
            
        except Exception as e:
            error_msg = f"Critical failure in emergency exit procedure: {str(e)}"
            logger.critical(error_msg)
            result['actions_taken'].append(f"Emergency procedure failed: {str(e)}")
            result['status'] = 'critical_failure'
            return result
    
    def _safe_start_monitoring(self, original_func, *args, **kwargs) -> Any:
        """
        Start monitoring thread with safety checks.
        
        Args:
            original_func: Original monitoring start function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result
        """
        try:
            # Reset monitoring health stats
            self.monitoring_health['is_running'] = True
            self.monitoring_health['last_check'] = datetime.now().isoformat()
            self.monitoring_health['last_exception'] = None
            
            # Run original start monitoring function
            result = original_func(*args, **kwargs)
            
            logger.info("Exit monitoring thread started with safety checks")
            return result
            
        except Exception as e:
            logger.error(f"Error starting exit monitoring thread: {str(e)}")
            self.monitoring_health['is_running'] = False
            self.monitoring_health['last_exception'] = str(e)
            
            # Re-raise to preserve original behavior
            raise
    
    def _safe_stop_monitoring(self, original_func, *args, **kwargs) -> Any:
        """
        Stop monitoring thread with safety checks.
        
        Args:
            original_func: Original monitoring stop function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result
        """
        try:
            # Update monitoring health stats
            self.monitoring_health['is_running'] = False
            
            # Run original stop monitoring function
            result = original_func(*args, **kwargs)
            
            logger.info("Exit monitoring thread stopped cleanly")
            return result
            
        except Exception as e:
            logger.error(f"Error stopping exit monitoring thread: {str(e)}")
            self.monitoring_health['last_exception'] = str(e)
            
            # Force monitoring thread to False regardless of exception
            self.monitoring_health['is_running'] = False
            
            # Re-raise to preserve original behavior
            raise
    
    def _safe_check_exits(self, original_func, *args, **kwargs) -> Any:
        """
        Add validation to the check exits function.
        
        Args:
            original_func: Original check exits function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or error handling result
        """
        # Update verification log
        verification_entry = {
            'timestamp': datetime.now().isoformat(),
            'status': 'started'
        }
        self.verification_log.append(verification_entry)
        self.last_verification = datetime.now()
        
        try:
            # Call original check exits function
            result = original_func(*args, **kwargs)
            
            # Update verification log
            verification_entry['status'] = 'completed'
            
            # Reset monitoring thread error count on successful check
            if self.monitoring_health['error_count'] > 0:
                self.monitoring_health['error_count'] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error in exit condition check: {str(e)}")
            
            # Update verification log
            verification_entry['status'] = 'failed'
            verification_entry['error'] = str(e)
            
            # Update monitoring health
            self.monitoring_health['last_exception'] = str(e)
            self.monitoring_health['error_count'] += 1
            
            # Check if we need to trip circuit breaker
            error_threshold = self.config.get('monitoring_error_threshold', 5)
            if self.monitoring_health['error_count'] >= error_threshold:
                self._trip_circuit_breaker('monitoring_thread', f"Too many monitoring thread errors: {error_threshold}")
            
            # Re-raise exception to preserve original behavior
            # This will likely be caught by the monitoring thread's exception handler
            raise
    
    def _trip_circuit_breaker(self, breaker_name: str, reason: str) -> None:
        """
        Trip a circuit breaker to prevent further operations.
        
        Args:
            breaker_name: Name of circuit breaker to trip
            reason: Reason for tripping
        """
        with self._lock:
            logger.critical(f"CIRCUIT BREAKER TRIPPED: {breaker_name} - {reason}")
            
            if breaker_name in self.circuit_breakers:
                self.circuit_breakers[breaker_name]['tripped'] = True
                self.circuit_breakers[breaker_name]['trip_time'] = datetime.now().isoformat()
                self.circuit_breakers[breaker_name]['trip_reason'] = reason
                
                # If monitoring thread breaker is tripped, try to safely shut down monitoring
                if breaker_name == 'monitoring_thread' and hasattr(self.exit_manager, '_stop_monitoring'):
                    try:
                        logger.warning("Attempting to safely shut down exit monitoring due to circuit breaker")
                        self.exit_manager._stop_monitoring()
                    except Exception as e:
                        logger.error(f"Error shutting down monitoring after circuit breaker: {str(e)}")
    
    def reset_circuit_breaker(self, breaker_name: str) -> bool:
        """
        Reset a tripped circuit breaker.
        
        Args:
            breaker_name: Name of circuit breaker to reset
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if breaker_name not in self.circuit_breakers:
                logger.warning(f"Unknown circuit breaker: {breaker_name}")
                return False
                
            if not self.circuit_breakers[breaker_name]['tripped']:
                logger.info(f"Circuit breaker {breaker_name} not tripped")
                return True
                
            # Reset circuit breaker
            self.circuit_breakers[breaker_name]['tripped'] = False
            self.circuit_breakers[breaker_name]['reset_time'] = datetime.now().isoformat()
            self.circuit_breakers[breaker_name]['error_count'] = 0
            
            logger.warning(f"Reset circuit breaker: {breaker_name}")
            
            # If monitoring thread was reset, restart monitoring
            if breaker_name == 'monitoring_thread' and hasattr(self.exit_manager, '_start_monitoring'):
                try:
                    logger.info("Restarting exit monitoring after circuit breaker reset")
                    self.exit_manager._start_monitoring()
                except Exception as e:
                    logger.error(f"Error restarting monitoring after circuit breaker reset: {str(e)}")
                    return False
            
            return True
    
    def process_retry_queue(self) -> Dict[str, Any]:
        """
        Process the exit retry queue.
        
        Returns:
            Dict with retry results
        """
        if not self.retry_queue:
            return {'status': 'no_retries_pending'}
        
        results = {
            'retries_processed': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'remaining_retries': 0
        }
        
        # Current time for comparison
        now = datetime.now()
        
        # Process each retry
        retry_items = list(self.retry_queue)  # Create copy to avoid modification during iteration
        for retry_item in retry_items:
            # Check if it's time to retry
            if retry_item['next_retry'] <= now:
                results['retries_processed'] += 1
                
                try:
                    # Remove from queue
                    self.retry_queue.remove(retry_item)
                    
                    # Execute exit
                    if hasattr(self.exit_manager, 'execute_exit'):
                        logger.info(f"Retrying exit for position {retry_item['position_id']}, attempt {retry_item['attempt']}")
                        result = self.exit_manager.execute_exit(*retry_item['args'], **retry_item['kwargs'])
                        
                        # Check result
                        if isinstance(result, dict) and result.get('status') == 'retry_scheduled':
                            # Exit failed again and was rescheduled
                            results['failed_retries'] += 1
                        else:
                            # Assume success
                            results['successful_retries'] += 1
                except Exception as e:
                    logger.error(f"Error processing retry for position {retry_item['position_id']}: {str(e)}")
                    results['failed_retries'] += 1
                    
                    # Check if we should retry again
                    max_retries = self.config.get('max_exit_retries', 3)
                    if retry_item['attempt'] < max_retries:
                        # Schedule next retry with exponential backoff
                        retry_item['attempt'] += 1
                        backoff_seconds = 5 * (2 ** (retry_item['attempt'] - 1))  # 5, 10, 20, 40...
                        retry_item['next_retry'] = now + timedelta(seconds=backoff_seconds)
                        self.retry_queue.append(retry_item)
                    else:
                        logger.warning(f"Max retries ({max_retries}) reached for position {retry_item['position_id']}")
        
        # Count remaining retries
        results['remaining_retries'] = len(self.retry_queue)
        
        return results
    
    def verify_exit_conditions(self) -> Tuple[bool, List[str]]:
        """
        Verify that exit conditions are being checked properly.
        
        Returns:
            Tuple of (is_valid, messages)
        """
        messages = []
        
        # Check if monitoring is running
        if not self.monitoring_health['is_running']:
            messages.append("Exit monitoring thread is not running")
        
        # Check last verification time
        if self.last_verification:
            seconds_since_check = (datetime.now() - self.last_verification).total_seconds()
            max_seconds = self.config.get('max_seconds_between_checks', 300)  # 5 minutes default
            
            if seconds_since_check > max_seconds:
                messages.append(f"Exit condition check overdue: {seconds_since_check:.1f} seconds since last check")
        else:
            messages.append("No exit condition checks recorded")
        
        # Check for failed verifications
        recent_verifications = self.verification_log[-10:] if self.verification_log else []
        failed_verifications = [v for v in recent_verifications if v.get('status') == 'failed']
        
        if failed_verifications:
            messages.append(f"Found {len(failed_verifications)} failed exit condition checks in recent history")
        
        # Check circuit breakers
        if self.circuit_breakers['monitoring_thread']['tripped']:
            messages.append("Monitoring thread circuit breaker is tripped")
        
        if self.circuit_breakers['exit_execution']['tripped']:
            messages.append("Exit execution circuit breaker is tripped")
        
        # Check emergency exit state
        if self.emergency_exit['triggered']:
            messages.append(f"Emergency exit triggered at {self.emergency_exit['trigger_time']}: {self.emergency_exit['trigger_reason']}")
        
        # Check failed exits
        if self.failed_exits:
            messages.append(f"Found {len(self.failed_exits)} positions with failed exits")
        
        # Check retry queue
        if self.retry_queue:
            messages.append(f"{len(self.retry_queue)} exits pending retry")
        
        return len(messages) == 0, messages
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """
        Check if the exit manager is healthy.
        
        Returns:
            Tuple of (is_healthy, messages)
        """
        # First check exit conditions
        is_valid, messages = self.verify_exit_conditions()
        
        # If we have circuit breakers tripped, always unhealthy
        for breaker_name, breaker_info in self.circuit_breakers.items():
            if breaker_info['tripped']:
                return False, messages
        
        # If emergency exit triggered, always unhealthy
        if self.emergency_exit['triggered']:
            return False, messages
        
        # Return overall health status
        return is_valid, messages

# Recovery function for system safeguards
def recover_exit_manager(exit_manager_safeguards: ExitManagerSafeguards) -> bool:
    """
    Attempt to recover exit manager from error state.
    
    Args:
        exit_manager_safeguards: Exit manager safeguards instance
        
    Returns:
        bool: Success status
    """
    try:
        logger.warning("Attempting to recover exit manager")
        success = True
        
        # Process any pending retries
        try:
            retry_results = exit_manager_safeguards.process_retry_queue()
            logger.info(f"Processed exit retries: {retry_results}")
        except Exception as e:
            logger.error(f"Error processing retry queue: {str(e)}")
            success = False
        
        # Reset circuit breakers
        for breaker_name in exit_manager_safeguards.circuit_breakers:
            if exit_manager_safeguards.circuit_breakers[breaker_name]['tripped']:
                try:
                    exit_manager_safeguards.reset_circuit_breaker(breaker_name)
                except Exception as e:
                    logger.error(f"Error resetting circuit breaker {breaker_name}: {str(e)}")
                    success = False
        
        # Restart monitoring if needed
        if not exit_manager_safeguards.monitoring_health['is_running']:
            try:
                if hasattr(exit_manager_safeguards.exit_manager, '_start_monitoring'):
                    exit_manager_safeguards.exit_manager._start_monitoring()
                    logger.info("Restarted exit monitoring thread")
            except Exception as e:
                logger.error(f"Error restarting monitoring thread: {str(e)}")
                success = False
        
        return success
        
    except Exception as e:
        logger.error(f"Exit manager recovery failed: {str(e)}")
        return False

# Function to create validation function for system safeguards
def create_exit_manager_validator(exit_manager_safeguards: ExitManagerSafeguards):
    """Create a validation function for the exit manager component."""
    
    def validate_exit_manager(component: Any) -> Tuple[bool, List[str]]:
        """
        Validate exit manager state.
        
        Args:
            component: Exit manager component
            
        Returns:
            Tuple of (is_valid, messages)
        """
        return exit_manager_safeguards.is_healthy()
    
    return validate_exit_manager
