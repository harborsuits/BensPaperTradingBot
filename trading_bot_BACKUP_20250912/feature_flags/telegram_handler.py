#!/usr/bin/env python3
"""
Feature Flag Telegram Handler

This module provides Telegram commands for managing feature flags remotely.
It allows operators to enable/disable features during market volatility
without requiring a full deployment.
"""

import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any

from ..telegram.command_handler import CommandHandler
from .service import (
    get_feature_flag_service, 
    FlagCategory,
    FeatureFlag,
    AssetClass
)

logger = logging.getLogger(__name__)

class FeatureFlagCommandHandler(CommandHandler):
    """Handles feature flag commands from Telegram."""
    
    def __init__(
        self, 
        command_prefix: str = "/",
        admin_users: Optional[Set[int]] = None,
        confirmation_expiry: int = 60,  # 60 seconds
        cooldown_period: int = 3        # 3 seconds between commands
    ):
        """Initialize the feature flag command handler.
        
        Args:
            command_prefix: Prefix for commands
            admin_users: Set of user IDs allowed to make changes
            confirmation_expiry: Seconds until confirmation expires
            cooldown_period: Seconds between commands
        """
        super().__init__(command_prefix, cooldown_period)
        self.service = get_feature_flag_service()
        self.admin_users = admin_users or set()
        self.confirmation_expiry = confirmation_expiry
        self.pending_toggles: Dict[str, Dict[str, Any]] = {}
        
        # Register commands
        self.register_command("flags", self.handle_list_flags, "List all feature flags")
        self.register_command("flag", self.handle_flag_details, "Show details for a flag")
        self.register_command("enable", self.handle_enable_flag, "Enable a flag")
        self.register_command("disable", self.handle_disable_flag, "Disable a flag")
        self.register_command("toggle", self.handle_toggle_flag, "Toggle a flag")
        self.register_command("confirm", self.handle_confirm_toggle, "Confirm a pending toggle")
        self.register_command("cancel", self.handle_cancel_toggle, "Cancel a pending toggle")
        self.register_command("create_flag", self.handle_create_flag, "Create a new flag")
        self.register_command("flag_history", self.handle_flag_history, "Show history for a flag")
        # New commands for gradual rollout
        self.register_command("rollout", self.handle_update_rollout, "Update rollout percentage")
        self.register_command("asset_classes", self.handle_update_asset_classes, "Update applicable asset classes")
        self.register_command("add_rule", self.handle_add_rule, "Add a context rule")
        self.register_command("remove_rule", self.handle_remove_rule, "Remove a context rule")
        self.register_command("gradual_flags", self.handle_list_gradual_flags, "List flags with gradual rollout")
        
        logger.info("Feature flag commands registered")
    
    def _check_admin(self, user_id: int) -> bool:
        """Check if a user has admin privileges.
        
        Args:
            user_id: User ID to check
            
        Returns:
            bool: True if admin, False otherwise
        """
        if not self.admin_users:
            return True  # No admins defined means everyone is admin
        
        return user_id in self.admin_users
    
    def _format_flag_details(self, flag: FeatureFlag) -> str:
        """Format the details of a flag for display.
        
        Args:
            flag: The flag to format
            
        Returns:
            str: Formatted flag details
        """
        status = "✅ ENABLED" if flag.enabled else "❌ DISABLED"
        
        details = [
            f"*{flag.name}* (`{flag.id}`) - {status}",
            f"Description: {flag.description}",
            f"Category: {flag.category.name}",
            f"Modified: {flag.modified_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if flag.rollout_percentage < 100:
            details.append(f"Rollout: {flag.rollout_percentage}%")
        
        if "ALL" not in flag.applicable_asset_classes:
            details.append(f"Asset Classes: {', '.join(flag.applicable_asset_classes)}")
        
        if flag.context_rules:
            details.append(f"Rules: {len(flag.context_rules)} active")
        
        return "\n".join(details)
    
    def _generate_confirmation_id(self) -> str:
        """Generate a unique confirmation ID.
        
        Returns:
            str: Confirmation ID
        """
        return uuid.uuid4().hex[:8]
    
    def _cleanup_expired_toggles(self):
        """Clean up expired pending toggles."""
        now = datetime.now()
        expired = []
        
        for confirm_id, data in self.pending_toggles.items():
            if now > data["expires_at"]:
                expired.append(confirm_id)
        
        for confirm_id in expired:
            del self.pending_toggles[confirm_id]
    
    def handle_list_flags(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /flags command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        # Check for category parameter
        parts = message_text.strip().split()
        category = None
        
        if len(parts) > 1:
            category_name = parts[1].upper()
            try:
                category = FlagCategory[category_name]
            except KeyError:
                categories = ", ".join(c.name for c in FlagCategory)
                return f"Invalid category: {category_name}\nAvailable categories: {categories}"
        
        # Get flags
        flags = self.service.list_flags(category)
        
        if not flags:
            if category:
                return f"No flags found for category: {category.name}"
            else:
                return "No flags found"
        
        # Group by category
        by_category = {}
        for flag in flags:
            cat_name = flag.category.name
            if cat_name not in by_category:
                by_category[cat_name] = []
            by_category[cat_name].append(flag)
        
        # Format response
        response = []
        for cat_name, cat_flags in sorted(by_category.items()):
            response.append(f"*{cat_name}*")
            for flag in sorted(cat_flags, key=lambda f: f.id):
                status = "✅" if flag.enabled else "❌"
                response.append(f"{status} `{flag.id}` - {flag.name}")
            response.append("")
        
        return "\n".join(response).strip()
    
    def handle_flag_details(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /flag command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        parts = message_text.strip().split()
        if len(parts) < 2:
            return "Usage: /flag <flag_id>"
        
        flag_id = parts[1]
        flag = self.service.get_flag(flag_id)
        
        if not flag:
            return f"Flag not found: {flag_id}"
        
        return self._format_flag_details(flag)
    
    def handle_enable_flag(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /enable command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to enable flags"
        
        parts = message_text.strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /enable <flag_id> [reason]"
        
        flag_id_part = parts[1].strip()
        
        # Check if reason is provided
        if " " in flag_id_part:
            flag_id, reason = flag_id_part.split(maxsplit=1)
        else:
            flag_id = flag_id_part
            reason = "Enabled via Telegram"
        
        # Get flag
        flag = self.service.get_flag(flag_id)
        if not flag:
            return f"Flag not found: {flag_id}"
        
        # Check if already enabled
        if flag.enabled:
            return f"Flag `{flag_id}` is already enabled"
        
        # Check if confirmation is required
        if flag.requires_confirmation:
            confirm_id = self._generate_confirmation_id()
            expires_at = datetime.now() + timedelta(seconds=self.confirmation_expiry)
            
            self.pending_toggles[confirm_id] = {
                "flag_id": flag_id,
                "enabled": True,
                "reason": reason,
                "user_id": user_id,
                "expires_at": expires_at
            }
            
            return (
                f"Flag `{flag_id}` requires confirmation to enable. "
                f"Type `/confirm {confirm_id}` to confirm or `/cancel {confirm_id}` to cancel. "
                f"This request will expire in {self.confirmation_expiry} seconds."
            )
        
        # Enable the flag
        success, message = self.service.set_flag(
            flag_id=flag_id,
            enabled=True,
            changed_by=f"telegram:{user_id}",
            reason=reason
        )
        
        if success:
            return f"Flag `{flag_id}` enabled successfully"
        else:
            return f"Failed to enable flag: {message}"
    
    def handle_disable_flag(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /disable command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to disable flags"
        
        parts = message_text.strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /disable <flag_id> [reason]"
        
        flag_id_part = parts[1].strip()
        
        # Check if reason is provided
        if " " in flag_id_part:
            flag_id, reason = flag_id_part.split(maxsplit=1)
        else:
            flag_id = flag_id_part
            reason = "Disabled via Telegram"
        
        # Get flag
        flag = self.service.get_flag(flag_id)
        if not flag:
            return f"Flag not found: {flag_id}"
        
        # Check if already disabled
        if not flag.enabled:
            return f"Flag `{flag_id}` is already disabled"
        
        # Check if confirmation is required
        if flag.requires_confirmation:
            confirm_id = self._generate_confirmation_id()
            expires_at = datetime.now() + timedelta(seconds=self.confirmation_expiry)
            
            self.pending_toggles[confirm_id] = {
                "flag_id": flag_id,
                "enabled": False,
                "reason": reason,
                "user_id": user_id,
                "expires_at": expires_at
            }
            
            return (
                f"Flag `{flag_id}` requires confirmation to disable. "
                f"Type `/confirm {confirm_id}` to confirm or `/cancel {confirm_id}` to cancel. "
                f"This request will expire in {self.confirmation_expiry} seconds."
            )
        
        # Disable the flag
        success, message = self.service.set_flag(
            flag_id=flag_id,
            enabled=False,
            changed_by=f"telegram:{user_id}",
            reason=reason
        )
        
        if success:
            return f"Flag `{flag_id}` disabled successfully"
        else:
            return f"Failed to disable flag: {message}"
    
    def handle_toggle_flag(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /toggle command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to toggle flags"
        
        parts = message_text.strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /toggle <flag_id> [reason]"
        
        flag_id_part = parts[1].strip()
        
        # Check if reason is provided
        if " " in flag_id_part:
            flag_id, reason = flag_id_part.split(maxsplit=1)
        else:
            flag_id = flag_id_part
            reason = "Toggled via Telegram"
        
        # Get flag
        flag = self.service.get_flag(flag_id)
        if not flag:
            return f"Flag not found: {flag_id}"
        
        # Determine new state
        new_state = not flag.enabled
        action = "enable" if new_state else "disable"
        
        # Check if confirmation is required
        if flag.requires_confirmation:
            confirm_id = self._generate_confirmation_id()
            expires_at = datetime.now() + timedelta(seconds=self.confirmation_expiry)
            
            self.pending_toggles[confirm_id] = {
                "flag_id": flag_id,
                "enabled": new_state,
                "reason": reason,
                "user_id": user_id,
                "expires_at": expires_at
            }
            
            return (
                f"Flag `{flag_id}` requires confirmation to {action}. "
                f"Type `/confirm {confirm_id}` to confirm or `/cancel {confirm_id}` to cancel. "
                f"This request will expire in {self.confirmation_expiry} seconds."
            )
        
        # Toggle the flag
        success, message = self.service.set_flag(
            flag_id=flag_id,
            enabled=new_state,
            changed_by=f"telegram:{user_id}",
            reason=reason
        )
        
        if success:
            return f"Flag `{flag_id}` {action}d successfully"
        else:
            return f"Failed to {action} flag: {message}"
    
    def handle_confirm_toggle(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /confirm command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to confirm flag changes"
        
        parts = message_text.strip().split()
        if len(parts) < 2:
            return "Usage: /confirm <confirmation_id>"
        
        confirm_id = parts[1]
        
        # Clean up expired toggles
        self._cleanup_expired_toggles()
        
        # Check if confirmation exists
        if confirm_id not in self.pending_toggles:
            return f"Confirmation ID not found or expired: {confirm_id}"
        
        # Get toggle data
        toggle_data = self.pending_toggles.pop(confirm_id)
        flag_id = toggle_data["flag_id"]
        enabled = toggle_data["enabled"]
        reason = toggle_data["reason"]
        action = "enable" if enabled else "disable"
        
        # Apply the change
        success, message = self.service.set_flag(
            flag_id=flag_id,
            enabled=enabled,
            changed_by=f"telegram:{user_id}",
            reason=reason
        )
        
        if success:
            return f"Flag `{flag_id}` {action}d successfully"
        else:
            return f"Failed to {action} flag: {message}"
    
    def handle_cancel_toggle(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /cancel command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        parts = message_text.strip().split()
        if len(parts) < 2:
            return "Usage: /cancel <confirmation_id>"
        
        confirm_id = parts[1]
        
        # Clean up expired toggles
        self._cleanup_expired_toggles()
        
        # Check if confirmation exists
        if confirm_id not in self.pending_toggles:
            return f"Confirmation ID not found or expired: {confirm_id}"
        
        # Get toggle data and remove it
        toggle_data = self.pending_toggles.pop(confirm_id)
        flag_id = toggle_data["flag_id"]
        enabled = toggle_data["enabled"]
        action = "enable" if enabled else "disable"
        
        return f"Cancelled request to {action} flag `{flag_id}`"
    
    def handle_create_flag(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /create_flag command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to create flags"
        
        # Parse arguments
        # Format: /create_flag <id> <name> <category> <description>
        parts = message_text.strip().split(maxsplit=4)
        if len(parts) < 5:
            categories = ", ".join(c.name for c in FlagCategory)
            return f"Usage: /create_flag <id> <name> <category> <description>\nCategories: {categories}"
        
        flag_id = parts[1]
        name = parts[2]
        category_name = parts[3].upper()
        description = parts[4]
        
        # Validate category
        try:
            category = FlagCategory[category_name]
        except KeyError:
            categories = ", ".join(c.name for c in FlagCategory)
            return f"Invalid category: {category_name}\nAvailable categories: {categories}"
        
        # Create the flag
        success, message = self.service.create_flag(
            id=flag_id,
            name=name,
            description=description,
            category=category,
            default=False
        )
        
        if success:
            return f"Flag `{flag_id}` created successfully"
        else:
            return f"Failed to create flag: {message}"
    
    def handle_flag_history(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /flag_history command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        parts = message_text.strip().split()
        if len(parts) < 2:
            return "Usage: /flag_history <flag_id> [limit]"
        
        flag_id = parts[1]
        
        # Check if limit is provided
        limit = 5
        if len(parts) > 2:
            try:
                limit = int(parts[2])
                limit = max(1, min(limit, 20))  # Limit between 1 and 20
            except ValueError:
                pass
        
        # Get flag
        flag = self.service.get_flag(flag_id)
        if not flag:
            return f"Flag not found: {flag_id}"
        
        # Get history
        history = self.service.get_flag_history(flag_id)
        
        if not history:
            return f"No history found for flag: {flag_id}"
        
        # Sort by timestamp (newest first) and limit
        history = sorted(history, key=lambda e: e.timestamp, reverse=True)[:limit]
        
        # Format response
        response = [f"*History for {flag.name} (`{flag_id}`)*"]
        for event in history:
            state = "Enabled" if event.enabled else "Disabled"
            timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            reason = f" - {event.reason}" if event.reason else ""
            response.append(f"{timestamp}: {state} by {event.changed_by}{reason}")
        
        return "\n".join(response)
    
    def handle_update_rollout(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /rollout command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to update rollout settings"
        
        # Parse arguments
        # Format: /rollout <flag_id> <percentage>
        parts = message_text.strip().split()
        if len(parts) < 3:
            return "Usage: /rollout <flag_id> <percentage>"
        
        flag_id = parts[1]
        
        # Parse percentage
        try:
            percentage = int(parts[2])
            if not 0 <= percentage <= 100:
                return "Percentage must be between 0 and 100"
        except ValueError:
            return "Invalid percentage. Must be a number between 0 and 100."
        
        # Update rollout percentage
        success, message = self.service.update_flag_rollout(flag_id, percentage)
        
        if success:
            return f"Updated rollout percentage for `{flag_id}` to {percentage}%"
        else:
            return f"Failed to update rollout percentage: {message}"
    
    def handle_update_asset_classes(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /asset_classes command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to update asset classes"
        
        # Parse arguments
        # Format: /asset_classes <flag_id> <class1,class2,...>
        parts = message_text.strip().split(maxsplit=2)
        if len(parts) < 3:
            asset_classes = ", ".join(c.name for c in AssetClass)
            return f"Usage: /asset_classes <flag_id> <class1,class2,...>\nAsset Classes: {asset_classes}"
        
        flag_id = parts[1]
        asset_class_str = parts[2]
        
        # Parse asset classes
        asset_classes = set(ac.strip() for ac in asset_class_str.split(","))
        
        # Update asset classes
        success, message = self.service.update_flag_asset_classes(flag_id, asset_classes)
        
        if success:
            return f"Updated asset classes for `{flag_id}` to {', '.join(asset_classes)}"
        else:
            return f"Failed to update asset classes: {message}"
    
    def handle_add_rule(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /add_rule command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to add rules"
        
        # Format: /add_rule <flag_id> <rule_type> <parameters>
        parts = message_text.strip().split(maxsplit=3)
        if len(parts) < 4:
            return (
                "Usage: /add_rule <flag_id> <rule_type> <parameters>\n"
                "Rule Types: asset_class, time_window, account_value, market_condition\n"
                "Examples:\n"
                "/add_rule my_flag asset_class FOREX,CRYPTO\n"
                "/add_rule my_flag time_window 09:30-16:00\n"
                "/add_rule my_flag account_value min=10000,max=500000\n"
                "/add_rule my_flag market_condition NORMAL,LOW_VOLATILITY"
            )
        
        flag_id = parts[1]
        rule_type = parts[2]
        params_str = parts[3]
        
        # Parse parameters based on rule type
        parameters = {}
        
        if rule_type == "asset_class":
            # Format: FOREX,CRYPTO,...
            asset_classes = [ac.strip() for ac in params_str.split(",")]
            parameters["asset_classes"] = asset_classes
        
        elif rule_type == "time_window":
            # Format: 09:30-16:00
            match = re.match(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", params_str)
            if not match:
                return "Invalid time window format. Use format: 09:30-16:00"
            
            start_time, end_time = match.groups()
            parameters["start_time"] = start_time
            parameters["end_time"] = end_time
        
        elif rule_type == "account_value":
            # Format: min=10000,max=500000
            for param in params_str.split(","):
                if "=" in param:
                    key, value = param.split("=", 1)
                    try:
                        parameters[key.strip()] = float(value.strip())
                    except ValueError:
                        return f"Invalid value for {key}: {value}. Must be a number."
        
        elif rule_type == "market_condition":
            # Format: NORMAL,LOW_VOLATILITY,...
            conditions = [c.strip() for c in params_str.split(",")]
            parameters["conditions"] = conditions
        
        else:
            return f"Invalid rule type: {rule_type}"
        
        # Add the rule
        success, message = self.service.add_context_rule(flag_id, rule_type, parameters)
        
        if success:
            return f"Added {rule_type} rule to `{flag_id}`"
        else:
            return f"Failed to add rule: {message}"
    
    def handle_remove_rule(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /remove_rule command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        if not self._check_admin(user_id):
            return "You don't have permission to remove rules"
        
        # Format: /remove_rule <flag_id> <rule_index>
        parts = message_text.strip().split()
        if len(parts) < 3:
            return "Usage: /remove_rule <flag_id> <rule_index>"
        
        flag_id = parts[1]
        
        # Parse rule index
        try:
            rule_index = int(parts[2])
            if rule_index < 0:
                return "Rule index must be a positive number"
        except ValueError:
            return "Invalid rule index. Must be a number."
        
        # Remove the rule
        success, message = self.service.remove_context_rule(flag_id, rule_index)
        
        if success:
            return f"Removed rule from `{flag_id}`"
        else:
            return f"Failed to remove rule: {message}"
    
    def handle_list_gradual_flags(self, message_text: str, user_id: int, chat_id: int) -> str:
        """Handle the /gradual_flags command.
        
        Args:
            message_text: Full message text
            user_id: User ID
            chat_id: Chat ID
            
        Returns:
            str: Response message
        """
        # Get all flags
        all_flags = self.service.list_flags()
        
        # Filter to flags with gradual rollout settings
        gradual_flags = [
            flag for flag in all_flags 
            if flag.rollout_percentage < 100 or 
               "ALL" not in flag.applicable_asset_classes or 
               flag.context_rules
        ]
        
        if not gradual_flags:
            return "No flags with gradual rollout settings found"
        
        # Format response
        response = ["*Flags with Gradual Rollout Settings*"]
        
        for flag in sorted(gradual_flags, key=lambda f: f.id):
            status = "✅" if flag.enabled else "❌"
            rollout = f"{flag.rollout_percentage}%"
            
            # Format asset classes
            asset_classes = list(flag.applicable_asset_classes)
            if "ALL" in asset_classes:
                asset_text = "All"
            else:
                asset_text = ", ".join(asset_classes)
            
            # Count of rules
            rules_count = len(flag.context_rules)
            rules_text = f"{rules_count} rules" if rules_count > 0 else "No rules"
            
            response.append(
                f"{status} `{flag.id}` - {flag.name}\n"
                f"   Rollout: {rollout} | Assets: {asset_text} | {rules_text}"
            )
        
        return "\n".join(response)


def register_feature_flag_commands(telegram_bot, admin_users: Optional[List[str]] = None):
    """
    Register feature flag commands with the Telegram bot
    
    Args:
        telegram_bot: Telegram bot instance
        admin_users: List of usernames that can manage flags
    """
    handler = FeatureFlagCommandHandler(admin_users)
    telegram_bot.register_command_handler(handler)
    logger.info("Registered feature flag commands with Telegram") 