"""
Position Manager Safeguards

This module provides enhanced robustness, error detection, and recovery mechanisms 
for the PositionManager component, ensuring reliable position tracking and reconciliation
even in challenging conditions.

Features:
- Transaction atomicity with rollback capability
- Reconciliation error handling and resolution
- Position data integrity checks
- Automatic retry logic for broker connectivity issues
- State checkpointing and recovery
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

# Import position manager
from trading_bot.position.position_manager import PositionManager
from trading_bot.robustness.system_safeguards import ComponentType, SafeguardState

logger = logging.getLogger(__name__)

class PositionManagerSafeguards:
    """
    Enhances PositionManager with robust safeguards to ensure reliable operation
    and automatic recovery from error conditions.
    """
    
    def __init__(self, position_manager: PositionManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize position manager safeguards.
        
        Args:
            position_manager: The position manager instance to enhance
            config: Configuration parameters
        """
        self.position_manager = position_manager
        self.config = config or {}
        
        # State backup and transaction handling
        self.state_backups: Dict[str, Dict[str, Any]] = {}
        self.transaction_log: List[Dict[str, Any]] = []
        self.transaction_counter = 0
        self.pending_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Reconciliation error handling
        self.reconciliation_errors: List[Dict[str, Any]] = []
        self.last_successful_reconciliation: Optional[datetime] = None
        self.reconciliation_retries: int = 0
        
        # State checkpointing
        self.checkpoint_interval_minutes = self.config.get('checkpoint_interval_minutes', 30)
        self.last_checkpoint_time: Optional[datetime] = None
        self.checkpoints: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._setup_safeguards()
        
        logger.info("Position Manager safeguards initialized")
    
    def _setup_safeguards(self) -> None:
        """Setup safety mechanisms and hooks."""
        # Create initial state backup
        self._create_state_backup()
        
        # Add error handling hooks
        if hasattr(self.position_manager, 'reconcile_positions'):
            original_reconcile = self.position_manager.reconcile_positions
            
            def safe_reconcile(*args, **kwargs):
                return self._safe_reconcile(original_reconcile, *args, **kwargs)
            
            self.position_manager.reconcile_positions = safe_reconcile
        
        # Schedule periodic checkpoints
        self._schedule_next_checkpoint()
    
    def _create_state_backup(self) -> str:
        """
        Create a backup of the current position manager state.
        
        Returns:
            str: Backup ID
        """
        with self._lock:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.transaction_counter}"
            
            try:
                # Create deep copy of positions
                positions_backup = {}
                if hasattr(self.position_manager, 'internal_positions'):
                    positions_backup = copy.deepcopy(self.position_manager.internal_positions)
                
                # Create deep copy of reconciliation state
                reconciliation_backup = {}
                if hasattr(self.position_manager, 'reconciliation_status'):
                    reconciliation_backup = copy.deepcopy(self.position_manager.reconciliation_status)
                
                # Store backup
                self.state_backups[backup_id] = {
                    'positions': positions_backup,
                    'reconciliation': reconciliation_backup,
                    'timestamp': datetime.now().isoformat(),
                    'transaction_counter': self.transaction_counter
                }
                
                # Prune old backups (keep last 10)
                if len(self.state_backups) > 10:
                    oldest_backup = min(self.state_backups.keys(), 
                                       key=lambda k: self.state_backups[k]['timestamp'])
                    del self.state_backups[oldest_backup]
                
                self.transaction_counter += 1
                return backup_id
                
            except Exception as e:
                logger.error(f"Error creating state backup: {str(e)}")
                return ""
    
    def _restore_from_backup(self, backup_id: str) -> bool:
        """
        Restore position manager state from a backup.
        
        Args:
            backup_id: Backup ID to restore from
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if backup_id not in self.state_backups:
                logger.error(f"Backup {backup_id} not found")
                return False
                
            try:
                # Get backup data
                backup = self.state_backups[backup_id]
                
                # Restore positions
                if hasattr(self.position_manager, 'internal_positions'):
                    self.position_manager.internal_positions = copy.deepcopy(backup['positions'])
                
                # Restore reconciliation state
                if hasattr(self.position_manager, 'reconciliation_status'):
                    self.position_manager.reconciliation_status = copy.deepcopy(backup['reconciliation'])
                
                logger.warning(f"Restored state from backup {backup_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring from backup: {str(e)}")
                return False
    
    def _safe_reconcile(self, original_func, *args, **kwargs) -> Any:
        """
        Execute position reconciliation with error handling and retries.
        
        Args:
            original_func: Original reconciliation function
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or error handling result
        """
        # Create backup before reconciliation
        backup_id = self._create_state_backup()
        
        try:
            # Call original reconciliation function
            result = original_func(*args, **kwargs)
            
            # Update successful reconciliation time
            self.last_successful_reconciliation = datetime.now()
            self.reconciliation_retries = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error during position reconciliation: {str(e)}")
            self.reconciliation_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Increment retry counter
            self.reconciliation_retries += 1
            
            # Determine action based on error type and retry count
            if "connection" in str(e).lower() and self.reconciliation_retries < 3:
                # Connection error, retry after delay
                logger.warning(f"Connection error during reconciliation, retry {self.reconciliation_retries}/3")
                time.sleep(5)  # Wait 5 seconds before retry
                return self._safe_reconcile(original_func, *args, **kwargs)
                
            elif "timeout" in str(e).lower() and self.reconciliation_retries < 3:
                # Timeout error, retry with longer timeout
                logger.warning(f"Timeout during reconciliation, retry {self.reconciliation_retries}/3")
                # Add longer timeout to kwargs if applicable
                if 'timeout' in kwargs:
                    kwargs['timeout'] = kwargs.get('timeout', 30) * 2
                time.sleep(3)
                return self._safe_reconcile(original_func, *args, **kwargs)
                
            elif "data" in str(e).lower() or "format" in str(e).lower():
                # Data or format error, try to continue with last known good state
                logger.warning("Data error during reconciliation, restoring from backup")
                if backup_id and self._restore_from_backup(backup_id):
                    return {'error': str(e), 'status': 'restored_from_backup'}
                    
            else:
                # Other error or too many retries, restore from backup
                logger.error(f"Unrecoverable error during reconciliation, restoring from backup: {str(e)}")
                if backup_id and self._restore_from_backup(backup_id):
                    return {'error': str(e), 'status': 'restored_from_backup'}
            
            # If we get here, couldn't recover
            return {'error': str(e), 'status': 'reconciliation_failed'}
    
    def _schedule_next_checkpoint(self) -> None:
        """Schedule the next state checkpoint."""
        # This would set up a timer in a real implementation
        self.last_checkpoint_time = datetime.now()
    
    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a complete state checkpoint for disaster recovery.
        
        Returns:
            Dict with checkpoint data
        """
        with self._lock:
            try:
                # Create checkpoint data
                checkpoint = {
                    'timestamp': datetime.now().isoformat(),
                    'positions': copy.deepcopy(self.position_manager.internal_positions),
                    'reconciliation_status': (
                        copy.deepcopy(self.position_manager.reconciliation_status) 
                        if hasattr(self.position_manager, 'reconciliation_status') else {}
                    ),
                    'broker_positions': (
                        copy.deepcopy(self.position_manager.broker_positions)
                        if hasattr(self.position_manager, 'broker_positions') else {}
                    ),
                    'position_history': (
                        copy.deepcopy(self.position_manager.position_history)
                        if hasattr(self.position_manager, 'position_history') else {}
                    )
                }
                
                # Save checkpoint
                self.checkpoints.append(checkpoint)
                
                # Keep only last 5 checkpoints
                if len(self.checkpoints) > 5:
                    self.checkpoints = self.checkpoints[-5:]
                
                # Update last checkpoint time
                self.last_checkpoint_time = datetime.now()
                
                logger.info("Created position manager checkpoint")
                
                # Optionally save to disk for permanent storage
                checkpoint_dir = self.config.get('checkpoint_dir')
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 
                        f"position_checkpoint_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                    )
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f)
                
                return checkpoint
                
            except Exception as e:
                logger.error(f"Error creating checkpoint: {str(e)}")
                return {'error': str(e)}
    
    def restore_from_checkpoint(self, checkpoint_idx: int = -1) -> bool:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_idx: Index of checkpoint (-1 for latest)
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                # Verify checkpoint exists
                if not self.checkpoints:
                    logger.error("No checkpoints available")
                    return False
                
                if checkpoint_idx < -len(self.checkpoints) or checkpoint_idx >= len(self.checkpoints):
                    logger.error(f"Invalid checkpoint index: {checkpoint_idx}")
                    return False
                
                # Get checkpoint
                checkpoint = self.checkpoints[checkpoint_idx]
                
                # Restore state
                if hasattr(self.position_manager, 'internal_positions'):
                    self.position_manager.internal_positions = copy.deepcopy(checkpoint['positions'])
                
                if hasattr(self.position_manager, 'reconciliation_status') and 'reconciliation_status' in checkpoint:
                    self.position_manager.reconciliation_status = copy.deepcopy(checkpoint['reconciliation_status'])
                
                if hasattr(self.position_manager, 'broker_positions') and 'broker_positions' in checkpoint:
                    self.position_manager.broker_positions = copy.deepcopy(checkpoint['broker_positions'])
                
                if hasattr(self.position_manager, 'position_history') and 'position_history' in checkpoint:
                    self.position_manager.position_history = copy.deepcopy(checkpoint['position_history'])
                
                logger.warning(f"Restored state from checkpoint {checkpoint['timestamp']}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring from checkpoint: {str(e)}")
                return False
    
    def verify_position_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of position data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check for basic structural integrity
            if not hasattr(self.position_manager, 'internal_positions'):
                errors.append("Internal positions attribute missing")
                return False, errors
            
            # Check each position for required fields
            for pos_id, position in self.position_manager.internal_positions.items():
                # Check for required fields
                required_fields = ['position_id', 'symbol', 'quantity', 'direction', 'entry_price']
                for field in required_fields:
                    if field not in position:
                        errors.append(f"Position {pos_id} missing required field: {field}")
                
                # Check data types
                if 'quantity' in position and not isinstance(position['quantity'], (int, float)):
                    errors.append(f"Position {pos_id} has invalid quantity type: {type(position['quantity'])}")
                
                if 'entry_price' in position and not isinstance(position['entry_price'], (int, float)):
                    errors.append(f"Position {pos_id} has invalid entry_price type: {type(position['entry_price'])}")
            
            # Check for duplicate symbols with same direction
            symbols_and_directions = [
                (p.get('symbol'), p.get('direction')) 
                for p in self.position_manager.internal_positions.values()
            ]
            duplicates = set([
                item for item in symbols_and_directions 
                if symbols_and_directions.count(item) > 1
            ])
            if duplicates:
                for dup in duplicates:
                    errors.append(f"Duplicate position found: {dup[0]} direction {dup[1]}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error during integrity check: {str(e)}")
            return False, errors
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """
        Check if the position manager is healthy.
        
        Returns:
            Tuple of (is_healthy, messages)
        """
        messages = []
        
        # Check integrity
        integrity_valid, integrity_errors = self.verify_position_integrity()
        if not integrity_valid:
            messages.extend(integrity_errors)
        
        # Check reconciliation health
        if self.last_successful_reconciliation:
            # Check if reconciliation is overdue
            hours_since_reconciliation = (datetime.now() - self.last_successful_reconciliation).total_seconds() / 3600
            max_hours = self.config.get('max_reconciliation_hours', 24)
            
            if hours_since_reconciliation > max_hours:
                messages.append(f"Position reconciliation overdue: {hours_since_reconciliation:.1f} hours since last success")
        else:
            messages.append("No successful reconciliation recorded")
        
        # Check error count
        recent_errors = [
            e for e in self.reconciliation_errors
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        if len(recent_errors) > 5:
            messages.append(f"High reconciliation error rate: {len(recent_errors)} errors in last 24 hours")
        
        # Overall health status
        is_healthy = len(messages) == 0
        
        return is_healthy, messages

# Recovery function for system safeguards
def recover_position_manager(position_manager_safeguards: PositionManagerSafeguards) -> bool:
    """
    Attempt to recover position manager from error state.
    
    Args:
        position_manager_safeguards: Position manager safeguards instance
        
    Returns:
        bool: Success status
    """
    try:
        # Verify integrity first
        integrity_valid, integrity_errors = position_manager_safeguards.verify_position_integrity()
        
        if not integrity_valid:
            # Try to restore from most recent checkpoint
            logger.warning(f"Position integrity check failed, attempting checkpoint restore: {integrity_errors}")
            return position_manager_safeguards.restore_from_checkpoint()
        
        # Try to force reconciliation
        logger.info("Attempting forced position reconciliation")
        
        # Access the wrapped position manager
        position_manager = position_manager_safeguards.position_manager
        
        # Perform reconciliation if the method exists
        if hasattr(position_manager, 'reconcile_positions'):
            try:
                # Use the wrapped, safe reconciliation method
                result = position_manager.reconcile_positions(force=True)
                
                # Check result
                if isinstance(result, dict) and result.get('status') == 'restored_from_backup':
                    logger.warning("Reconciliation failed but restored from backup")
                    return True
                
                return True
            except Exception as e:
                logger.error(f"Error during recovery reconciliation: {str(e)}")
                
                # Last resort: restore from checkpoint
                return position_manager_safeguards.restore_from_checkpoint()
        
        return False
        
    except Exception as e:
        logger.error(f"Position manager recovery failed: {str(e)}")
        return False

# Function to create validation function for system safeguards
def create_position_manager_validator(position_manager_safeguards: PositionManagerSafeguards):
    """Create a validation function for the position manager component."""
    
    def validate_position_manager(component: Any) -> Tuple[bool, List[str]]:
        """
        Validate position manager state.
        
        Args:
            component: Position manager component
            
        Returns:
            Tuple of (is_valid, messages)
        """
        return position_manager_safeguards.is_healthy()
    
    return validate_position_manager
