#!/usr/bin/env python3
"""
State Manager for Trading Bot

This module implements state persistence and recovery mechanisms to ensure
the trading bot can recover from crashes or disconnects without human intervention.
It handles saving and restoring state for various components of the system.
"""

import os
import json
import time
import logging
import threading
import pickle
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class StatePersistenceFormat(Enum):
    """Format options for state persistence."""
    JSON = "json"
    PICKLE = "pickle"
    

class StateManager:
    """
    Manages state persistence and recovery for the trading bot.
    
    Features:
    - Periodic state snapshotting to disk
    - Safe state serialization/deserialization
    - Atomic writes to prevent corruption
    - Component-specific state handling
    - Transaction logging to ensure idempotence
    """
    
    def __init__(
        self,
        state_dir: str,
        snapshot_interval_seconds: int = 60,
        format: StatePersistenceFormat = StatePersistenceFormat.JSON,
        max_snapshots: int = 5,
        compress: bool = False
    ):
        """
        Initialize the state manager.
        
        Args:
            state_dir: Directory to store state files
            snapshot_interval_seconds: How often to take snapshots (0 = manual only)
            format: Format to use for state persistence
            max_snapshots: Maximum number of snapshot files to keep
            compress: Whether to compress state files (only for pickle format)
        """
        self.state_dir = state_dir
        self.snapshot_interval = snapshot_interval_seconds
        self.format = format
        self.max_snapshots = max_snapshots
        self.compress = compress
        
        # Create state directory if it doesn't exist
        if not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)
        
        # Component registrations
        self.components = {}
        
        # Internal state
        self._snapshot_thread = None
        self._running = False
        self._lock = threading.RLock()
        self._last_snapshot_time = 0
        
        # Transaction log
        self.transaction_log_path = os.path.join(self.state_dir, "transaction_log.json")
        self.transaction_log: Dict[str, Dict] = self._load_transaction_log()
        
    def register_component(self, name: str, component: Any, get_state_method: str = "get_state", 
                          restore_state_method: str = "restore_state") -> None:
        """
        Register a component for state persistence.
        
        Args:
            name: Unique name for the component
            component: The component object
            get_state_method: Name of the method to call to get component state
            restore_state_method: Name of the method to call to restore component state
        """
        if name in self.components:
            logger.warning(f"Component {name} already registered, overwriting")
            
        self.components[name] = {
            "instance": component,
            "get_state_method": get_state_method,
            "restore_state_method": restore_state_method
        }
        
        logger.info(f"Registered component {name} for state persistence")
        
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component.
        
        Args:
            name: Name of the component to unregister
            
        Returns:
            True if the component was unregistered, False if it wasn't registered
        """
        if name in self.components:
            del self.components[name]
            logger.info(f"Unregistered component {name}")
            return True
        
        return False
        
    def create_snapshot(self) -> bool:
        """
        Create a snapshot of the current state.
        
        Returns:
            True if the snapshot was created successfully, False otherwise
        """
        with self._lock:
            try:
                # Collect state from all components
                state = {}
                for name, component_info in self.components.items():
                    try:
                        component = component_info["instance"]
                        get_state_method = getattr(component, component_info["get_state_method"])
                        component_state = get_state_method()
                        state[name] = component_state
                    except Exception as e:
                        logger.error(f"Error getting state for component {name}: {str(e)}")
                        # Continue with other components despite this error
                
                # Add metadata
                state["_metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    "components": list(self.components.keys())
                }
                
                # Create snapshot file
                timestamp = int(time.time())
                snapshot_file = os.path.join(self.state_dir, f"state_snapshot_{timestamp}.{self.format.value}")
                
                # Write to temp file first for atomic operation
                temp_file = snapshot_file + ".tmp"
                
                if self.format == StatePersistenceFormat.JSON:
                    with open(temp_file, 'w') as f:
                        json.dump(state, f, indent=2, default=self._json_serializer)
                else:  # PICKLE
                    with open(temp_file, 'wb') as f:
                        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Atomic rename
                os.rename(temp_file, snapshot_file)
                
                # Update last snapshot time
                self._last_snapshot_time = time.time()
                
                # Rotate old snapshots
                self._rotate_snapshots()
                
                logger.info(f"Created state snapshot: {snapshot_file}")
                return True
                
            except Exception as e:
                logger.error(f"Error creating state snapshot: {str(e)}")
                return False
    
    def restore_latest_snapshot(self) -> bool:
        """
        Restore the latest state snapshot.
        
        Returns:
            True if the state was restored successfully, False otherwise
        """
        with self._lock:
            try:
                # Find the latest snapshot
                snapshots = self._get_snapshot_files()
                if not snapshots:
                    logger.warning("No state snapshots found to restore")
                    return False
                
                latest_snapshot = snapshots[-1]
                
                # Load the snapshot
                state = self._load_state_file(latest_snapshot)
                if not state:
                    logger.error(f"Failed to load state from {latest_snapshot}")
                    return False
                
                # Restore state to each component
                for name, component_info in self.components.items():
                    if name in state:
                        try:
                            component = component_info["instance"]
                            restore_method = getattr(component, component_info["restore_state_method"])
                            restore_method(state[name])
                            logger.info(f"Restored state for component {name}")
                        except Exception as e:
                            logger.error(f"Error restoring state for component {name}: {str(e)}")
                    else:
                        logger.warning(f"No state found for component {name} in snapshot")
                
                logger.info(f"Restored state from snapshot: {latest_snapshot}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring state: {str(e)}")
                return False
    
    def start_auto_snapshot(self) -> bool:
        """
        Start automatic state snapshotting.
        
        Returns:
            True if auto snapshotting was started, False otherwise
        """
        if self.snapshot_interval <= 0:
            logger.warning("Cannot start auto snapshot with interval <= 0")
            return False
            
        if self._running:
            logger.warning("Auto snapshot already running")
            return False
            
        self._running = True
        self._snapshot_thread = threading.Thread(
            target=self._auto_snapshot_loop,
            daemon=True,
            name="StateManagerSnapshotThread"
        )
        self._snapshot_thread.start()
        
        logger.info(f"Started automatic state snapshots every {self.snapshot_interval} seconds")
        return True
    
    def stop_auto_snapshot(self) -> None:
        """Stop automatic state snapshotting."""
        self._running = False
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            # Let the thread terminate naturally at next interval
            # No need to join - it's a daemon thread
            logger.info("Stopping automatic state snapshots")
    
    def log_transaction(self, transaction_type: str, transaction_id: str, 
                        data: Dict[str, Any], expiry_seconds: Optional[int] = None) -> None:
        """
        Log a transaction to prevent duplication on restart.
        
        Args:
            transaction_type: Type of transaction (e.g., "order", "position")
            transaction_id: Unique ID for the transaction
            data: Transaction data
            expiry_seconds: Optional expiry time in seconds (None = never expire)
        """
        with self._lock:
            if transaction_type not in self.transaction_log:
                self.transaction_log[transaction_type] = {}
                
            # Add timestamp and expiry
            timestamp = time.time()
            transaction_data = {
                "data": data,
                "timestamp": timestamp,
                "expiry": timestamp + expiry_seconds if expiry_seconds else None
            }
            
            # Add to transaction log
            self.transaction_log[transaction_type][transaction_id] = transaction_data
            
            # Save transaction log
            self._save_transaction_log()
            
    def check_transaction(self, transaction_type: str, transaction_id: str) -> Optional[Dict]:
        """
        Check if a transaction has been processed already.
        
        Args:
            transaction_type: Type of transaction
            transaction_id: Transaction ID to check
            
        Returns:
            Transaction data if it exists, None otherwise
        """
        with self._lock:
            # Clean expired transactions first
            self._clean_expired_transactions()
            
            # Check if transaction exists
            if transaction_type in self.transaction_log:
                return self.transaction_log[transaction_type].get(transaction_id)
            
            return None
            
    def remove_transaction(self, transaction_type: str, transaction_id: str) -> bool:
        """
        Remove a transaction from the log.
        
        Args:
            transaction_type: Type of transaction
            transaction_id: Transaction ID to remove
            
        Returns:
            True if the transaction was removed, False otherwise
        """
        with self._lock:
            if transaction_type in self.transaction_log:
                if transaction_id in self.transaction_log[transaction_type]:
                    del self.transaction_log[transaction_type][transaction_id]
                    self._save_transaction_log()
                    return True
            
            return False
    
    def _auto_snapshot_loop(self) -> None:
        """Background thread for automatic state snapshots."""
        logger.info("Auto snapshot thread started")
        
        while self._running:
            try:
                # Sleep until next snapshot time
                time.sleep(1.0)  # Check every second
                
                # Check if it's time for a snapshot
                if (time.time() - self._last_snapshot_time) >= self.snapshot_interval:
                    self.create_snapshot()
                    
            except Exception as e:
                logger.error(f"Error in auto snapshot loop: {str(e)}")
                time.sleep(5.0)  # Sleep longer on error
        
        logger.info("Auto snapshot thread stopped")
    
    def _load_state_file(self, filepath: str) -> Optional[Dict]:
        """Load state from a file."""
        try:
            if filepath.endswith(StatePersistenceFormat.JSON.value):
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:  # PICKLE
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading state file {filepath}: {str(e)}")
            return None
    
    def _rotate_snapshots(self) -> None:
        """Remove old snapshots to maintain max_snapshots."""
        snapshots = self._get_snapshot_files()
        
        # If we have more than max_snapshots, delete the oldest ones
        if len(snapshots) > self.max_snapshots:
            for snapshot in snapshots[:-self.max_snapshots]:
                try:
                    os.remove(snapshot)
                    logger.debug(f"Deleted old snapshot: {snapshot}")
                except Exception as e:
                    logger.error(f"Error deleting old snapshot {snapshot}: {str(e)}")
    
    def _get_snapshot_files(self) -> List[str]:
        """Get a list of snapshot files, sorted by timestamp (oldest first)."""
        snapshots = []
        for filename in os.listdir(self.state_dir):
            if filename.startswith("state_snapshot_") and (
                filename.endswith(f".{StatePersistenceFormat.JSON.value}") or
                filename.endswith(f".{StatePersistenceFormat.PICKLE.value}")
            ):
                filepath = os.path.join(self.state_dir, filename)
                snapshots.append(filepath)
        
        # Sort by timestamp in filename
        return sorted(snapshots)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for handling non-serializable objects."""
        if isinstance(obj, (datetime, )):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.name
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            # Try to convert to string
            return str(obj)
    
    def _load_transaction_log(self) -> Dict:
        """Load the transaction log from disk."""
        if os.path.exists(self.transaction_log_path):
            try:
                with open(self.transaction_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading transaction log: {str(e)}")
                # Backup the corrupted file
                backup_path = f"{self.transaction_log_path}.corrupted.{int(time.time())}"
                try:
                    os.rename(self.transaction_log_path, backup_path)
                    logger.warning(f"Backed up corrupted transaction log to {backup_path}")
                except Exception:
                    pass
        
        return {}
    
    def _save_transaction_log(self) -> None:
        """Save the transaction log to disk."""
        try:
            # Write to temp file first for atomic operation
            temp_file = f"{self.transaction_log_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.transaction_log, f)
            
            # Atomic rename
            os.rename(temp_file, self.transaction_log_path)
        except Exception as e:
            logger.error(f"Error saving transaction log: {str(e)}")
    
    def _clean_expired_transactions(self) -> None:
        """Remove expired transactions from the log."""
        now = time.time()
        modified = False
        
        for tx_type in list(self.transaction_log.keys()):
            for tx_id in list(self.transaction_log[tx_type].keys()):
                tx_data = self.transaction_log[tx_type][tx_id]
                
                # Check if expired
                if tx_data.get("expiry") and tx_data["expiry"] < now:
                    del self.transaction_log[tx_type][tx_id]
                    modified = True
            
            # Remove empty transaction types
            if not self.transaction_log[tx_type]:
                del self.transaction_log[tx_type]
                modified = True
        
        # Save if modified
        if modified:
            self._save_transaction_log()
    
    def generate_transaction_id(self, transaction_data: Dict) -> str:
        """
        Generate a deterministic transaction ID from transaction data.
        This helps with idempotence - the same transaction will get the same ID.
        
        Args:
            transaction_data: Data that uniquely identifies the transaction
            
        Returns:
            A hash string that can be used as a transaction ID
        """
        # Convert to JSON and hash
        json_str = json.dumps(transaction_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
