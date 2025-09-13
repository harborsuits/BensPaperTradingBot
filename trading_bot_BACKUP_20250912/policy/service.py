"""
Policy management service.

This module provides a service for loading, validating, and managing
trading policies, with support for versioning and hot reloading.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, Optional, Any

from trading_bot.policy.types import Policy
from trading_bot.policy.default_policy import default_policy


logger = logging.getLogger(__name__)


class PolicyService:
    """Service for managing trading policies."""
    
    def __init__(self, policy_dir: str = "config/policies"):
        """
        Initialize the policy service.
        
        Args:
            policy_dir: Directory containing policy JSON files
        """
        self.policy_dir = policy_dir
        self.current_policy = default_policy
        self.policy_lock = threading.RLock()
        self.last_reload_time = 0
        self.policy_file_path = os.path.join(policy_dir, "active_policy.json")
        
        # Create policy directory if it doesn't exist
        os.makedirs(policy_dir, exist_ok=True)
        
        # Try to load policy from file
        self._load_policy()
    
    def get_policy(self) -> Policy:
        """
        Get the current active policy.
        
        Returns:
            The current policy
        """
        with self.policy_lock:
            return self.current_policy
    
    def update_policy(self, new_policy: Policy) -> bool:
        """
        Update the current policy.
        
        Args:
            new_policy: The new policy to apply
            
        Returns:
            True if the policy was updated, False otherwise
        """
        if not self._validate_policy(new_policy):
            logger.error("Invalid policy, not updating")
            return False
        
        with self.policy_lock:
            self.current_policy = new_policy
            self._save_policy()
            logger.info(f"Policy updated to version {new_policy['version']}")
            return True
    
    def reload_if_changed(self) -> bool:
        """
        Reload the policy from file if it has changed.
        
        Returns:
            True if the policy was reloaded, False otherwise
        """
        try:
            if not os.path.exists(self.policy_file_path):
                return False
            
            mtime = os.path.getmtime(self.policy_file_path)
            if mtime <= self.last_reload_time:
                return False
            
            return self._load_policy()
        except Exception as e:
            logger.error(f"Error checking for policy changes: {e}")
            return False
    
    def _load_policy(self) -> bool:
        """
        Load the policy from file.
        
        Returns:
            True if the policy was loaded, False otherwise
        """
        try:
            if not os.path.exists(self.policy_file_path):
                # Save default policy if no file exists
                self._save_policy()
                logger.info("Created default policy file")
                return True
            
            with open(self.policy_file_path, 'r') as f:
                policy_data = json.load(f)
            
            if not self._validate_policy(policy_data):
                logger.error("Invalid policy file, using default")
                return False
            
            with self.policy_lock:
                self.current_policy = policy_data
                self.last_reload_time = os.path.getmtime(self.policy_file_path)
            
            logger.info(f"Loaded policy from file, version {policy_data['version']}")
            return True
        except Exception as e:
            logger.error(f"Error loading policy: {e}")
            return False
    
    def _save_policy(self) -> bool:
        """
        Save the current policy to file.
        
        Returns:
            True if the policy was saved, False otherwise
        """
        try:
            with open(self.policy_file_path, 'w') as f:
                json.dump(self.current_policy, f, indent=2)
            
            self.last_reload_time = os.path.getmtime(self.policy_file_path)
            return True
        except Exception as e:
            logger.error(f"Error saving policy: {e}")
            return False
    
    def _validate_policy(self, policy: Any) -> bool:
        """
        Validate a policy object.
        
        Args:
            policy: The policy to validate
            
        Returns:
            True if the policy is valid, False otherwise
        """
        # Basic structure validation
        required_keys = ["version", "risk", "compliance", "weights", "toggles", "stale_after_ms"]
        for key in required_keys:
            if key not in policy:
                logger.error(f"Policy missing required key: {key}")
                return False
        
        # Risk limits validation
        risk = policy.get("risk", {})
        if not isinstance(risk.get("max_daily_loss_pct"), (int, float)):
            logger.error("Invalid max_daily_loss_pct")
            return False
        
        if not isinstance(risk.get("max_gross_exposure_pct"), (int, float)):
            logger.error("Invalid max_gross_exposure_pct")
            return False
        
        # More validation could be added here...
        
        return True
