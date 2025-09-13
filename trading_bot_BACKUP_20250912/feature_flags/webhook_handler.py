"""
Feature Flag Webhook Handler

Provides API endpoints for remote management of feature flags.
Allows external systems to view, enable, disable, and create feature flags.
"""

import logging
import json
import hmac
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple

from .service import (
    get_feature_flag_service, 
    FlagCategory,
    FeatureFlag
)

logger = logging.getLogger(__name__)

class WebhookSecurityError(Exception):
    """Exception raised for webhook security issues."""
    pass

class FeatureFlagWebhookHandler:
    """Handler for feature flag webhook operations."""
    
    def __init__(
        self, 
        secret_key: str,
        max_timestamp_diff: int = 300,
        require_signature: bool = True,
        admin_keys: Optional[List[str]] = None
    ):
        """Initialize the webhook handler.
        
        Args:
            secret_key: Secret key for validating webhook signatures
            max_timestamp_diff: Maximum allowed timestamp difference in seconds
            require_signature: Whether to require valid signatures
            admin_keys: List of API keys allowed to perform admin operations
        """
        self.secret_key = secret_key
        self.max_timestamp_diff = max_timestamp_diff
        self.require_signature = require_signature
        self.admin_keys = set(admin_keys or [])
        self.service = get_feature_flag_service()
    
    def _verify_signature(self, payload: Dict[str, Any], signature: str) -> bool:
        """Verify the signature of a webhook payload.
        
        Args:
            payload: Webhook payload
            signature: Provided signature
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        if not self.require_signature:
            return True
            
        # Extract timestamp from payload
        timestamp = payload.get("timestamp")
        if not timestamp:
            logger.warning("Missing timestamp in webhook payload")
            return False
        
        # Check if timestamp is recent
        current_time = int(time.time())
        if abs(current_time - timestamp) > self.max_timestamp_diff:
            logger.warning(f"Timestamp too old: {timestamp} vs {current_time}")
            return False
        
        # Compute expected signature
        payload_json = json.dumps(payload, sort_keys=True)
        expected_signature = hmac.new(
            self.secret_key.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(expected_signature, signature)
    
    def _verify_admin_key(self, api_key: str) -> bool:
        """Verify if the API key has admin privileges.
        
        Args:
            api_key: API key to verify
            
        Returns:
            bool: True if key has admin privileges, False otherwise
        """
        return api_key in self.admin_keys
    
    def handle_webhook(
        self, 
        endpoint: str, 
        payload: Dict[str, Any], 
        signature: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[Dict[str, Any], int]:
        """Handle a webhook request.
        
        Args:
            endpoint: Webhook endpoint (e.g., "list", "enable", "disable")
            payload: Webhook payload
            signature: Request signature for verification
            api_key: API key for authentication
            
        Returns:
            Tuple[Dict[str, Any], int]: Response payload and HTTP status code
        """
        try:
            # Verify signature if required
            if self.require_signature and not self._verify_signature(payload, signature or ""):
                return {"error": "Invalid signature"}, 401
            
            # Handle endpoint
            if endpoint == "list":
                return self._handle_list_flags(payload), 200
            elif endpoint == "get":
                return self._handle_get_flag(payload), 200
            elif endpoint == "enable":
                # Admin check for modifying flags
                if api_key and not self._verify_admin_key(api_key):
                    return {"error": "Insufficient permissions"}, 403
                return self._handle_enable_flag(payload), 200
            elif endpoint == "disable":
                # Admin check for modifying flags
                if api_key and not self._verify_admin_key(api_key):
                    return {"error": "Insufficient permissions"}, 403
                return self._handle_disable_flag(payload), 200
            elif endpoint == "create":
                # Admin check for creating flags
                if api_key and not self._verify_admin_key(api_key):
                    return {"error": "Insufficient permissions"}, 403
                return self._handle_create_flag(payload), 200
            elif endpoint == "history":
                return self._handle_flag_history(payload), 200
            else:
                return {"error": f"Unknown endpoint: {endpoint}"}, 400
                
        except WebhookSecurityError as e:
            logger.warning(f"Security error in webhook: {str(e)}")
            return {"error": str(e)}, 403
        except Exception as e:
            logger.error(f"Error handling webhook: {str(e)}", exc_info=True)
            return {"error": f"Internal error: {str(e)}"}, 500
    
    def _handle_list_flags(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to list flags.
        
        Args:
            payload: Request payload, may contain filters
            
        Returns:
            Dict[str, Any]: Response with flag list
        """
        category_str = payload.get("category")
        category = None
        
        if category_str:
            try:
                category = FlagCategory[category_str.upper()]
            except KeyError:
                return {
                    "error": f"Invalid category: {category_str}",
                    "valid_categories": [c.name for c in FlagCategory]
                }
        
        flags = self.service.list_flags(category)
        
        # Convert flags to dict representation
        return {
            "flags": [
                {
                    "id": flag.id,
                    "name": flag.name,
                    "category": flag.category.name,
                    "enabled": flag.enabled,
                    "description": flag.description,
                    "modified_at": flag.modified_at.isoformat()
                }
                for flag in flags
            ],
            "count": len(flags)
        }
    
    def _handle_get_flag(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to get a specific flag.
        
        Args:
            payload: Request payload with flag ID
            
        Returns:
            Dict[str, Any]: Response with flag details
        """
        flag_id = payload.get("flag_id")
        if not flag_id:
            return {"error": "Missing flag_id parameter"}
        
        flag = self.service.get_flag(flag_id)
        if not flag:
            return {"error": f"Flag not found: {flag_id}"}
        
        # Get full flag details including history
        return {
            "flag": {
                "id": flag.id,
                "name": flag.name,
                "description": flag.description,
                "category": flag.category.name,
                "enabled": flag.enabled,
                "default": flag.default,
                "requires_confirmation": flag.requires_confirmation,
                "rollback_after_seconds": flag.rollback_after_seconds,
                "dependent_flags": list(flag.dependent_flags),
                "created_at": flag.created_at.isoformat(),
                "modified_at": flag.modified_at.isoformat()
            }
        }
    
    def _handle_enable_flag(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to enable a flag.
        
        Args:
            payload: Request payload with flag ID
            
        Returns:
            Dict[str, Any]: Response with result
        """
        flag_id = payload.get("flag_id")
        if not flag_id:
            return {"error": "Missing flag_id parameter"}
        
        reason = payload.get("reason", "Enabled via webhook")
        changed_by = payload.get("user", "webhook")
        
        success, message = self.service.set_flag(
            flag_id=flag_id,
            enabled=True,
            changed_by=changed_by,
            reason=reason
        )
        
        if success:
            logger.info(f"Flag {flag_id} enabled via webhook by {changed_by}")
            return {"success": True, "message": message}
        else:
            return {"success": False, "error": message}
    
    def _handle_disable_flag(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to disable a flag.
        
        Args:
            payload: Request payload with flag ID
            
        Returns:
            Dict[str, Any]: Response with result
        """
        flag_id = payload.get("flag_id")
        if not flag_id:
            return {"error": "Missing flag_id parameter"}
        
        reason = payload.get("reason", "Disabled via webhook")
        changed_by = payload.get("user", "webhook")
        
        success, message = self.service.set_flag(
            flag_id=flag_id,
            enabled=False,
            changed_by=changed_by,
            reason=reason
        )
        
        if success:
            logger.info(f"Flag {flag_id} disabled via webhook by {changed_by}")
            return {"success": True, "message": message}
        else:
            return {"success": False, "error": message}
    
    def _handle_create_flag(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to create a new flag.
        
        Args:
            payload: Request payload with flag details
            
        Returns:
            Dict[str, Any]: Response with result
        """
        required_fields = ["id", "name", "description", "category"]
        for field in required_fields:
            if field not in payload:
                return {"error": f"Missing required field: {field}"}
        
        # Parse category
        category_str = payload["category"].upper()
        try:
            category = FlagCategory[category_str]
        except KeyError:
            return {
                "error": f"Invalid category: {category_str}",
                "valid_categories": [c.name for c in FlagCategory]
            }
        
        # Optional parameters
        default = payload.get("default", False)
        requires_confirmation = payload.get("requires_confirmation", False)
        rollback_after_seconds = payload.get("rollback_after_seconds")
        dependent_flags = set(payload.get("dependent_flags", []))
        
        # Create the flag
        success, message = self.service.create_flag(
            id=payload["id"],
            name=payload["name"],
            description=payload["description"],
            category=category,
            default=default,
            requires_confirmation=requires_confirmation,
            rollback_after_seconds=rollback_after_seconds,
            dependent_flags=dependent_flags
        )
        
        if success:
            logger.info(f"Flag {payload['id']} created via webhook")
            return {"success": True, "message": message}
        else:
            return {"success": False, "error": message}
    
    def _handle_flag_history(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request for flag change history.
        
        Args:
            payload: Request payload with flag ID
            
        Returns:
            Dict[str, Any]: Response with flag history
        """
        flag_id = payload.get("flag_id")
        if not flag_id:
            return {"error": "Missing flag_id parameter"}
        
        limit = int(payload.get("limit", 10))
        
        flag = self.service.get_flag(flag_id)
        if not flag:
            return {"error": f"Flag not found: {flag_id}"}
        
        # Get history events sorted by timestamp (newest first)
        history = sorted(
            flag.history,
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit]
        
        return {
            "flag_id": flag_id,
            "flag_name": flag.name,
            "history": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "enabled": event.enabled,
                    "changed_by": event.changed_by,
                    "reason": event.reason
                }
                for event in history
            ],
            "count": len(history)
        } 