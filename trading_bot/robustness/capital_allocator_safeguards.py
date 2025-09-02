"""
Capital Allocator Safeguards

This module provides enhanced robustness, validation, and protection mechanisms
for the CapitalAllocator component, ensuring reliable capital allocation
and protecting against allocation errors that could lead to excessive risk.

Features:
- Allocation boundary enforcement
- Transition smoothing for stability
- State validation and sanity checks
- Allocation history and version tracking
- Emergency risk reduction mechanisms
"""

import logging
import threading
import time
import traceback
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
import copy

# Import capital allocator
from trading_bot.portfolio.capital_allocator import CapitalAllocator
from trading_bot.robustness.system_safeguards import ComponentType, SafeguardState

logger = logging.getLogger(__name__)

class CapitalAllocatorSafeguards:
    """
    Enhances the CapitalAllocator with robust safeguards to ensure reliable and safe
    allocation of capital across strategies.
    """
    
    def __init__(self, capital_allocator: CapitalAllocator, config: Optional[Dict[str, Any]] = None):
        """
        Initialize capital allocator safeguards.
        
        Args:
            capital_allocator: The capital allocator instance to enhance
            config: Configuration parameters
        """
        self.capital_allocator = capital_allocator
        self.config = config or {}
        
        # Allocation state tracking
        self.allocation_history: List[Dict[str, Any]] = []
        self.allocation_versions: Dict[str, Dict[str, Any]] = {}
        self.current_version: str = "initial"
        
        # Protection boundaries
        self.max_allocation_change: float = self.config.get('max_allocation_change', 0.20)  # 20% max change
        self.min_allocation: float = self.config.get('min_allocation', 0.01)  # 1% min allocation
        self.max_allocation: float = self.config.get('max_allocation', 0.50)  # 50% max allocation
        
        # Risk management
        self.risk_levels: Dict[str, Dict[str, Any]] = {
            'normal': {
                'active': True,
                'max_portfolio_drawdown': self.config.get('normal_max_portfolio_drawdown', 0.10),
                'reserved_capital': self.config.get('normal_reserved_capital', 0.10)
            },
            'elevated': {
                'active': False,
                'max_portfolio_drawdown': self.config.get('elevated_max_portfolio_drawdown', 0.07),
                'reserved_capital': self.config.get('elevated_reserved_capital', 0.20)
            },
            'high': {
                'active': False,
                'max_portfolio_drawdown': self.config.get('high_max_portfolio_drawdown', 0.05),
                'reserved_capital': self.config.get('high_reserved_capital', 0.30)
            },
            'extreme': {
                'active': False,
                'max_portfolio_drawdown': self.config.get('extreme_max_portfolio_drawdown', 0.02),
                'reserved_capital': self.config.get('extreme_reserved_capital', 0.50)
            }
        }
        
        # Smoothing parameters
        self.smoothing_enabled: bool = self.config.get('smoothing_enabled', True)
        self.smoothing_factor: float = self.config.get('smoothing_factor', 0.25)  # 25% of difference per rebalance
        
        # Fault detection
        self.last_successful_allocation: Optional[datetime] = None
        self.allocation_errors: List[Dict[str, Any]] = []
        
        # Emergency state
        self.emergency_allocation: Dict[str, float] = {}
        self.emergency_allocation_active: bool = False
        self.emergency_trigger_reason: Optional[str] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._setup_safeguards()
        
        logger.info("Capital Allocator safeguards initialized")
    
    def _setup_safeguards(self) -> None:
        """Setup safety mechanisms and hooks."""
        # Wrap critical methods with safeguards
        if hasattr(self.capital_allocator, 'calculate_allocations'):
            original_calculate = self.capital_allocator.calculate_allocations
            
            def safe_calculate_allocations(*args, **kwargs):
                return self._safe_calculate_allocations(original_calculate, *args, **kwargs)
            
            self.capital_allocator.calculate_allocations = safe_calculate_allocations
        
        if hasattr(self.capital_allocator, 'allocate_capital'):
            original_allocate = self.capital_allocator.allocate_capital
            
            def safe_allocate_capital(*args, **kwargs):
                return self._safe_allocate_capital(original_allocate, *args, **kwargs)
            
            self.capital_allocator.allocate_capital = safe_allocate_capital
        
        if hasattr(self.capital_allocator, 'calculate_position_size'):
            original_position_size = self.capital_allocator.calculate_position_size
            
            def safe_calculate_position_size(*args, **kwargs):
                return self._safe_calculate_position_size(original_position_size, *args, **kwargs)
            
            self.capital_allocator.calculate_position_size = safe_calculate_position_size
        
        # Store initial allocation
        self._capture_current_allocation('initial')
    
    def _capture_current_allocation(self, version_name: str) -> None:
        """
        Capture current capital allocation for history and versioning.
        
        Args:
            version_name: Name for this allocation version
        """
        try:
            current_allocations = {}
            
            # Get current allocations from CapitalAllocator
            if hasattr(self.capital_allocator, 'current_allocations'):
                current_allocations = copy.deepcopy(self.capital_allocator.current_allocations)
            
            # Capture allocation state
            allocation_record = {
                'version': version_name,
                'timestamp': datetime.now().isoformat(),
                'allocations': current_allocations,
                'risk_level': self._get_active_risk_level(),
                'reserved_capital': self.capital_allocator.reserved_capital_percentage 
                                   if hasattr(self.capital_allocator, 'reserved_capital_percentage') else None
            }
            
            # Add to history
            self.allocation_history.append(allocation_record)
            
            # Keep history to a reasonable size
            max_history = self.config.get('max_allocation_history', 100)
            if len(self.allocation_history) > max_history:
                self.allocation_history = self.allocation_history[-max_history:]
            
            # Store version
            self.allocation_versions[version_name] = allocation_record
            self.current_version = version_name
            
        except Exception as e:
            logger.error(f"Error capturing allocation state: {str(e)}")
    
    def _get_active_risk_level(self) -> str:
        """
        Get currently active risk level.
        
        Returns:
            str: Active risk level name
        """
        for level_name, level_info in self.risk_levels.items():
            if level_info['active']:
                return level_name
        
        # Default to normal if none active
        return 'normal'
    
    def _safe_calculate_allocations(self, original_func, *args, **kwargs) -> Any:
        """
        Safely calculate allocations with validation and protection.
        
        Args:
            original_func: Original allocation calculation function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or modified result
        """
        try:
            # Call original function
            allocations = original_func(*args, **kwargs)
            
            # Validate allocations
            validated_allocations = self._validate_and_protect_allocations(allocations)
            
            # Capture this allocation
            version_name = f"allocation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self._capture_current_allocation(version_name)
            
            # Update last successful allocation time
            self.last_successful_allocation = datetime.now()
            
            return validated_allocations
            
        except Exception as e:
            logger.error(f"Error calculating allocations: {str(e)}")
            
            # Log error
            self.allocation_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Use previous allocation if available
            previous_allocations = self._get_last_successful_allocation()
            if previous_allocations:
                logger.warning("Using previous successful allocation due to calculation error")
                return previous_allocations
            else:
                # If no previous allocation, use emergency equal allocation
                logger.warning("Using emergency equal allocation due to calculation error")
                return self._create_emergency_allocation('calculation_error')
    
    def _validate_and_protect_allocations(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and protect allocations from excessive risk or errors.
        
        Args:
            allocations: Raw allocations from calculation
            
        Returns:
            Dict[str, float]: Validated and protected allocations
        """
        if not allocations:
            return {}
            
        protected_allocations = {}
        previous_allocations = self._get_last_successful_allocation()
        
        # Apply risk level constraints
        risk_level = self._get_active_risk_level()
        risk_config = self.risk_levels[risk_level]
        
        # Apply smoothing if enabled and we have previous allocations
        if self.smoothing_enabled and previous_allocations:
            for strategy_id, allocation in allocations.items():
                # Get previous allocation or default to 0
                prev_allocation = previous_allocations.get(strategy_id, 0.0)
                
                # Calculate smoothed allocation
                if abs(allocation - prev_allocation) > self.max_allocation_change:
                    # Apply max change limit
                    if allocation > prev_allocation:
                        protected_allocations[strategy_id] = prev_allocation + self.max_allocation_change
                    else:
                        protected_allocations[strategy_id] = prev_allocation - self.max_allocation_change
                else:
                    # Apply general smoothing
                    change = allocation - prev_allocation
                    smoothed_allocation = prev_allocation + (change * self.smoothing_factor)
                    protected_allocations[strategy_id] = smoothed_allocation
        else:
            # No smoothing
            protected_allocations = allocations.copy()
        
        # Apply min/max allocation limits
        for strategy_id, allocation in list(protected_allocations.items()):
            # Enforce minimum allocation
            if allocation < self.min_allocation:
                if allocation > 0:  # Only apply min if strategy is allocated at all
                    protected_allocations[strategy_id] = self.min_allocation
            
            # Enforce maximum allocation
            if allocation > self.max_allocation:
                protected_allocations[strategy_id] = self.max_allocation
        
        # Normalize if sum exceeds 1.0 or reserved capital constraint
        max_allocatable = 1.0 - risk_config['reserved_capital']
        sum_allocations = sum(protected_allocations.values())
        
        if sum_allocations > max_allocatable:
            # Scale down proportionally
            scaling_factor = max_allocatable / sum_allocations
            for strategy_id in protected_allocations:
                protected_allocations[strategy_id] *= scaling_factor
        
        return protected_allocations
    
    def _safe_allocate_capital(self, original_func, *args, **kwargs) -> Any:
        """
        Safely allocate capital with validation and protection.
        
        Args:
            original_func: Original capital allocation function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or error handling result
        """
        try:
            # Check if emergency allocation is active
            if self.emergency_allocation_active:
                logger.warning("Using emergency allocation for capital allocation")
                return self.emergency_allocation
            
            # Call original function
            result = original_func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error allocating capital: {str(e)}")
            
            # Log error
            self.allocation_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'function': 'allocate_capital'
            })
            
            # Use previous allocation if available
            previous_allocations = self._get_last_successful_allocation()
            if previous_allocations:
                logger.warning("Using previous successful allocation due to allocation error")
                return previous_allocations
            else:
                # If no previous allocation, use emergency equal allocation
                logger.warning("Using emergency equal allocation due to allocation error")
                return self._create_emergency_allocation('allocation_error')
    
    def _safe_calculate_position_size(self, original_func, *args, **kwargs) -> Any:
        """
        Safely calculate position size with protection against excessive sizes.
        
        Args:
            original_func: Original position size calculation function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or modified result
        """
        try:
            # Extract strategy_id if available in args or kwargs
            strategy_id = None
            if len(args) > 0:
                strategy_id = args[0]
            elif 'strategy_id' in kwargs:
                strategy_id = kwargs['strategy_id']
            
            # Call original function
            position_size = original_func(*args, **kwargs)
            
            # Apply protection based on active risk level
            risk_level = self._get_active_risk_level()
            
            if risk_level != 'normal':
                # Reduce position size based on risk level
                if risk_level == 'elevated':
                    position_size *= 0.8  # 20% reduction
                elif risk_level == 'high':
                    position_size *= 0.6  # 40% reduction
                elif risk_level == 'extreme':
                    position_size *= 0.3  # 70% reduction
                
                logger.info(f"Applied {risk_level} risk reduction to position size for {strategy_id}")
            
            # Apply maximum position size limit if specified
            max_position_size = kwargs.get('max_position_size')
            if max_position_size is not None and position_size > max_position_size:
                logger.warning(f"Position size {position_size} exceeds maximum {max_position_size}, reducing")
                position_size = max_position_size
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            
            # Log error
            self.allocation_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'function': 'calculate_position_size'
            })
            
            # Return conservative position size on error
            default_position_size = self.config.get('default_position_size', 1)
            logger.warning(f"Using conservative default position size: {default_position_size}")
            return default_position_size
    
    def _get_last_successful_allocation(self) -> Dict[str, float]:
        """
        Get the last successful allocation.
        
        Returns:
            Dict[str, float]: Last successful allocation or empty dict
        """
        if not self.allocation_history:
            return {}
            
        # Find the latest allocation
        latest_allocation = self.allocation_history[-1]
        return latest_allocation.get('allocations', {})
    
    def _create_emergency_allocation(self, reason: str) -> Dict[str, float]:
        """
        Create emergency allocation for use during failures.
        
        Args:
            reason: Reason for emergency allocation
            
        Returns:
            Dict[str, float]: Emergency allocation
        """
        # Check if we already have emergency allocation
        if self.emergency_allocation and self.emergency_allocation_active:
            return self.emergency_allocation
            
        # Create equal allocation for all strategies
        all_strategies = set()
        
        # Get strategies from allocation history
        for alloc_record in self.allocation_history:
            all_strategies.update(alloc_record.get('allocations', {}).keys())
        
        # Get current strategies from capital allocator
        if hasattr(self.capital_allocator, 'current_allocations'):
            all_strategies.update(self.capital_allocator.current_allocations.keys())
        
        # If we still have no strategies, try to find them from config
        if not all_strategies and hasattr(self.capital_allocator, 'strategies'):
            all_strategies = set(self.capital_allocator.strategies)
        
        # Create equal allocation
        if not all_strategies:
            logger.error("No strategies found for emergency allocation")
            return {}
            
        # Get risk level reserved capital
        risk_level = 'extreme'  # Use extreme risk level for emergency
        reserved_capital = self.risk_levels[risk_level]['reserved_capital']
        
        # Calculate equal allocation with high reserved capital
        available_capital = 1.0 - reserved_capital
        equal_allocation = available_capital / len(all_strategies)
        
        # Create allocation dict
        emergency_allocation = {strategy_id: equal_allocation for strategy_id in all_strategies}
        
        # Save emergency allocation
        self.emergency_allocation = emergency_allocation
        self.emergency_allocation_active = True
        self.emergency_trigger_reason = reason
        
        logger.warning(f"Created emergency allocation: {emergency_allocation}")
        
        return emergency_allocation
    
    def set_risk_level(self, level: str) -> bool:
        """
        Set active risk level.
        
        Args:
            level: Risk level to set active
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if level not in self.risk_levels:
                logger.error(f"Unknown risk level: {level}")
                return False
                
            # Deactivate all levels
            for level_name in self.risk_levels:
                self.risk_levels[level_name]['active'] = False
                
            # Activate specified level
            self.risk_levels[level]['active'] = True
            
            logger.warning(f"Set risk level to {level}")
            
            # Update capital allocator reserved capital if it has the attribute
            if hasattr(self.capital_allocator, 'reserved_capital_percentage'):
                self.capital_allocator.reserved_capital_percentage = self.risk_levels[level]['reserved_capital']
                
            # Capture new allocation state
            self._capture_current_allocation(f"risk_level_{level}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            return True
    
    def disable_emergency_allocation(self) -> bool:
        """
        Disable emergency allocation mode.
        
        Returns:
            bool: Success status
        """
        if not self.emergency_allocation_active:
            return True
            
        self.emergency_allocation_active = False
        logger.warning("Disabled emergency allocation mode")
        
        # Recalculate allocations
        if hasattr(self.capital_allocator, 'calculate_allocations'):
            try:
                self.capital_allocator.calculate_allocations()
                logger.info("Recalculated allocations after disabling emergency mode")
                return True
            except Exception as e:
                logger.error(f"Error recalculating allocations: {str(e)}")
                return False
        
        return True
    
    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback allocations to a previous version.
        
        Args:
            version: Version name to rollback to
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if version not in self.allocation_versions:
                logger.error(f"Unknown allocation version: {version}")
                return False
                
            try:
                # Get version allocations
                version_record = self.allocation_versions[version]
                allocations = version_record['allocations']
                
                # Set allocations in capital allocator
                if hasattr(self.capital_allocator, 'current_allocations'):
                    self.capital_allocator.current_allocations = copy.deepcopy(allocations)
                    
                # Set risk level if available
                if 'risk_level' in version_record:
                    self.set_risk_level(version_record['risk_level'])
                    
                # Set reserved capital if available
                if 'reserved_capital' in version_record and version_record['reserved_capital'] is not None:
                    if hasattr(self.capital_allocator, 'reserved_capital_percentage'):
                        self.capital_allocator.reserved_capital_percentage = version_record['reserved_capital']
                
                logger.warning(f"Rolled back allocations to version: {version}")
                
                # Capture rollback as new version
                self._capture_current_allocation(f"rollback_to_{version}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error rolling back to version {version}: {str(e)}")
                return False
    
    def verify_allocations(self) -> Tuple[bool, List[str]]:
        """
        Verify current allocations for validity.
        
        Returns:
            Tuple of (is_valid, messages)
        """
        messages = []
        
        try:
            # Check if capital allocator has current_allocations
            if not hasattr(self.capital_allocator, 'current_allocations'):
                messages.append("Capital allocator missing current_allocations attribute")
                return False, messages
                
            current_allocations = self.capital_allocator.current_allocations
            
            # Check if allocations exist
            if not current_allocations:
                messages.append("No current allocations found")
                return False, messages
                
            # Check allocation sum
            allocation_sum = sum(current_allocations.values())
            
            # Account for reserved capital
            reserved_capital = 0.0
            if hasattr(self.capital_allocator, 'reserved_capital_percentage'):
                reserved_capital = self.capital_allocator.reserved_capital_percentage
                
            expected_sum = 1.0 - reserved_capital
            if not (0.99 <= allocation_sum <= 1.01) and not (expected_sum - 0.01 <= allocation_sum <= expected_sum + 0.01):
                messages.append(f"Allocation sum {allocation_sum:.4f} is invalid (expected: {expected_sum:.4f})")
            
            # Check negative or excessive allocations
            for strategy_id, allocation in current_allocations.items():
                if allocation < 0:
                    messages.append(f"Negative allocation for {strategy_id}: {allocation}")
                    
                if allocation > self.max_allocation:
                    messages.append(f"Excessive allocation for {strategy_id}: {allocation} > {self.max_allocation}")
            
            return len(messages) == 0, messages
            
        except Exception as e:
            messages.append(f"Error verifying allocations: {str(e)}")
            return False, messages
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """
        Check if the capital allocator is healthy.
        
        Returns:
            Tuple of (is_healthy, messages)
        """
        messages = []
        
        # Check allocation validity
        is_valid, validation_messages = self.verify_allocations()
        if not is_valid:
            messages.extend(validation_messages)
        
        # Check if emergency allocation is active
        if self.emergency_allocation_active:
            messages.append(f"Emergency allocation active: {self.emergency_trigger_reason}")
        
        # Check last successful allocation
        if self.last_successful_allocation:
            hours_since_allocation = (datetime.now() - self.last_successful_allocation).total_seconds() / 3600
            max_hours = self.config.get('max_hours_since_allocation', 24)
            
            if hours_since_allocation > max_hours:
                messages.append(f"Allocation overdue: {hours_since_allocation:.1f} hours since last successful allocation")
        else:
            messages.append("No successful allocation recorded")
        
        # Check for allocation errors
        recent_errors = [
            e for e in self.allocation_errors
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        if len(recent_errors) > 3:
            messages.append(f"High error rate: {len(recent_errors)} allocation errors in last 24 hours")
        
        return len(messages) == 0, messages

# Recovery function for system safeguards
def recover_capital_allocator(capital_allocator_safeguards: CapitalAllocatorSafeguards) -> bool:
    """
    Attempt to recover capital allocator from error state.
    
    Args:
        capital_allocator_safeguards: Capital allocator safeguards instance
        
    Returns:
        bool: Success status
    """
    try:
        logger.warning("Attempting to recover capital allocator")
        
        # First check allocation validity
        is_valid, validation_messages = capital_allocator_safeguards.verify_allocations()
        
        if not is_valid:
            logger.warning(f"Invalid allocations detected: {validation_messages}")
            
            # Try to rollback to a known good version
            if capital_allocator_safeguards.allocation_versions:
                # Find a recent known good version
                for version_name in reversed(list(capital_allocator_safeguards.allocation_versions.keys())):
                    if version_name != capital_allocator_safeguards.current_version:
                        logger.info(f"Attempting rollback to version: {version_name}")
                        rollback_success = capital_allocator_safeguards.rollback_to_version(version_name)
                        if rollback_success:
                            logger.info(f"Successfully rolled back to version: {version_name}")
                            return True
                        
            # If rollback fails, use emergency allocation
            logger.warning("Enabling emergency allocation as last resort")
            capital_allocator_safeguards._create_emergency_allocation('recovery_rollback_failed')
            return True
        
        # If emergency allocation is active but not needed, disable it
        if capital_allocator_safeguards.emergency_allocation_active and is_valid:
            logger.info("Disabling emergency allocation, allocations are now valid")
            capital_allocator_safeguards.disable_emergency_allocation()
        
        # Check if risk level needs adjustment
        if capital_allocator_safeguards.last_successful_allocation:
            hours_since_allocation = (datetime.now() - capital_allocator_safeguards.last_successful_allocation).total_seconds() / 3600
            if hours_since_allocation > 24:
                # Set conservative risk level if allocations are stale
                logger.warning("Setting elevated risk level due to stale allocations")
                capital_allocator_safeguards.set_risk_level('elevated')
                
                # Try to recalculate allocations
                if hasattr(capital_allocator_safeguards.capital_allocator, 'calculate_allocations'):
                    try:
                        capital_allocator_safeguards.capital_allocator.calculate_allocations()
                        logger.info("Successfully recalculated allocations")
                    except Exception as e:
                        logger.error(f"Error recalculating allocations: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Capital allocator recovery failed: {str(e)}")
        return False

# Function to create validation function for system safeguards
def create_capital_allocator_validator(capital_allocator_safeguards: CapitalAllocatorSafeguards):
    """Create a validation function for the capital allocator component."""
    
    def validate_capital_allocator(component: Any) -> Tuple[bool, List[str]]:
        """
        Validate capital allocator state.
        
        Args:
            component: Capital allocator component
            
        Returns:
            Tuple of (is_valid, messages)
        """
        return capital_allocator_safeguards.is_healthy()
    
    return validate_capital_allocator
