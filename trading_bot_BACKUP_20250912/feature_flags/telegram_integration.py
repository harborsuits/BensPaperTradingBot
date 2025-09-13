#!/usr/bin/env python3
"""
Telegram Integration for Feature Flag Management

This module provides integration with Telegram for remote management of feature flags.
It allows users to enable/disable features, list all flags, and get flag details
through Telegram commands.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum

from ..feature_flags.feature_flag_service import (
    FeatureFlag, 
    FeatureFlagCategory, 
    FeatureFlagService,
    get_feature_flag_service
)

logger = logging.getLogger(__name__)

class CommandPermission(Enum):
    """Permission levels for feature flag commands"""
    READ = "read"  # Can view flags
    WRITE = "write"  # Can modify flags
    ADMIN = "admin"  # Can add/delete flags and modify permissions

class TelegramFeatureFlagHandler:
    """
    Handler for feature flag management via Telegram.
    
    This class registers commands for managing feature flags and
    processes incoming Telegram messages to execute those commands.
    """
    
    def __init__(
        self,
        feature_flag_service: Optional[FeatureFlagService] = None,
        authorized_users: Optional[Dict[str, CommandPermission]] = None
    ):
        """
        Initialize the Telegram feature flag handler
        
        Args:
            feature_flag_service: Feature flag service instance (optional)
            authorized_users: Dict mapping user IDs to permission levels
        """
        self.feature_flag_service = feature_flag_service or get_feature_flag_service()
        self.authorized_users = authorized_users or {}
        self.commands = self._register_commands()
        
        logger.info("Telegram feature flag handler initialized")
    
    def _register_commands(self) -> Dict[str, Tuple[Callable, CommandPermission, str]]:
        """Register all available commands with their handlers and permission levels"""
        return {
            "flags": (self._handle_list_flags, CommandPermission.READ, "List all feature flags"),
            "flag": (self._handle_flag_details, CommandPermission.READ, "Get details for a specific flag: /flag <name>"),
            "enable": (self._handle_enable_flag, CommandPermission.WRITE, "Enable a flag: /enable <name>"),
            "disable": (self._handle_disable_flag, CommandPermission.WRITE, "Disable a flag: /disable <name>"),
            "toggle": (self._handle_toggle_flag, CommandPermission.WRITE, "Toggle a flag: /toggle <name>"),
            "addflag": (self._handle_add_flag, CommandPermission.ADMIN, "Add a new flag: /addflag <name> [enabled=True|False] [category=SYSTEM] [description=...]"),
            "deleteflag": (self._handle_delete_flag, CommandPermission.ADMIN, "Delete a flag: /deleteflag <name>"),
            "category": (self._handle_list_category, CommandPermission.READ, "List flags in category: /category <name>"),
            "categories": (self._handle_list_categories, CommandPermission.READ, "List all categories"),
            "flaghelp": (self._handle_help, CommandPermission.READ, "Show this help message"),
        }
    
    def get_command_list(self) -> List[Dict[str, str]]:
        """
        Get list of commands for BotFather registration
        
        Returns:
            List of dictionaries with command and description
        """
        return [
            {"command": cmd, "description": help_text}
            for cmd, (_, _, help_text) in self.commands.items()
        ]
    
    def handle_message(self, message: str, user_id: str, username: str) -> Optional[str]:
        """
        Handle an incoming message from Telegram
        
        Args:
            message: The message text
            user_id: The Telegram user ID
            username: The Telegram username
            
        Returns:
            Optional response message
        """
        if not message.startswith('/'):
            return None
        
        # Extract command and arguments
        parts = message[1:].split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command not in self.commands:
            return None
        
        handler, required_permission, _ = self.commands[command]
        
        # Check permissions
        if not self._check_permission(user_id, required_permission):
            return f"⛔ You don't have permission to use the /{command} command"
        
        try:
            return handler(args, user_id, username)
        except Exception as e:
            logger.error(f"Error handling command '{command}': {e}")
            return f"❌ Error: {str(e)}"
    
    def _check_permission(self, user_id: str, required_permission: CommandPermission) -> bool:
        """Check if user has required permission"""
        if user_id not in self.authorized_users:
            return False
        
        user_permission = self.authorized_users[user_id]
        
        # Admin can do everything
        if user_permission == CommandPermission.ADMIN:
            return True
        
        # Write can do read
        if user_permission == CommandPermission.WRITE and required_permission == CommandPermission.READ:
            return True
        
        # Exact match
        return user_permission == required_permission
    
    def _handle_list_flags(self, args: str, user_id: str, username: str) -> str:
        """Handle command to list all flags"""
        flags = self.feature_flag_service.get_all_flags()
        
        if not flags:
            return "No feature flags found"
        
        # Sort by enabled status (enabled first) then by name
        flags.sort(key=lambda f: (not f.enabled, f.name))
        
        result = "🚩 *Feature Flags:*\n\n"
        for flag in flags:
            status = "✅" if flag.enabled else "❌"
            result += f"{status} `{flag.name}` ({flag.category.value})\n"
        
        return result
    
    def _handle_flag_details(self, args: str, user_id: str, username: str) -> str:
        """Handle command to get details of a specific flag"""
        if not args:
            return "⚠️ Usage: /flag <name>"
        
        flag_name = args.strip()
        flag = self.feature_flag_service.get_flag(flag_name)
        
        if not flag:
            return f"❌ Flag '{flag_name}' not found"
        
        status = "✅ Enabled" if flag.enabled else "❌ Disabled"
        
        result = f"🚩 *Flag: {flag.name}*\n\n"
        result += f"Status: {status}\n"
        result += f"Category: {flag.category.value}\n"
        result += f"Description: {flag.description}\n"
        
        if flag.metadata:
            result += "\nMetadata:\n"
            for k, v in flag.metadata.items():
                result += f"- {k}: {v}\n"
        
        result += f"\nLast updated: {flag.last_updated}\n"
        result += f"Modified by: {flag.last_modified_by}"
        
        return result
    
    def _handle_enable_flag(self, args: str, user_id: str, username: str) -> str:
        """Handle command to enable a flag"""
        if not args:
            return "⚠️ Usage: /enable <name>"
        
        flag_name = args.strip()
        success = self.feature_flag_service.update_flag(
            flag_name, True, f"telegram:{username}"
        )
        
        if success:
            return f"✅ Flag '{flag_name}' has been enabled"
        else:
            return f"❌ Flag '{flag_name}' not found"
    
    def _handle_disable_flag(self, args: str, user_id: str, username: str) -> str:
        """Handle command to disable a flag"""
        if not args:
            return "⚠️ Usage: /disable <name>"
        
        flag_name = args.strip()
        success = self.feature_flag_service.update_flag(
            flag_name, False, f"telegram:{username}"
        )
        
        if success:
            return f"✅ Flag '{flag_name}' has been disabled"
        else:
            return f"❌ Flag '{flag_name}' not found"
    
    def _handle_toggle_flag(self, args: str, user_id: str, username: str) -> str:
        """Handle command to toggle a flag"""
        if not args:
            return "⚠️ Usage: /toggle <name>"
        
        flag_name = args.strip()
        flag = self.feature_flag_service.get_flag(flag_name)
        
        if not flag:
            return f"❌ Flag '{flag_name}' not found"
        
        new_state = not flag.enabled
        success = self.feature_flag_service.update_flag(
            flag_name, new_state, f"telegram:{username}"
        )
        
        if success:
            status = "enabled" if new_state else "disabled"
            return f"✅ Flag '{flag_name}' has been {status}"
        else:
            return f"❌ Failed to toggle flag '{flag_name}'"
    
    def _handle_add_flag(self, args: str, user_id: str, username: str) -> str:
        """Handle command to add a new flag"""
        if not args:
            return "⚠️ Usage: /addflag <name> [enabled=True|False] [category=SYSTEM] [description=...]"
        
        # Extract arguments
        match = re.match(r'(\w+)(?:\s+(.*))?', args)
        if not match:
            return "⚠️ Invalid format for flag name"
        
        name = match.group(1)
        params_str = match.group(2) or ""
        
        # Parse parameters
        enabled = True  # Default
        category = FeatureFlagCategory.OTHER
        description = ""
        
        # Extract key=value parameters
        params_dict = {}
        for param in re.finditer(r'(\w+)=([^\s]+|"[^"]*"|\'[^\']*\')', params_str):
            key = param.group(1).lower()
            value = param.group(2)
            
            # Strip quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            params_dict[key] = value
        
        if 'enabled' in params_dict:
            enabled_str = params_dict['enabled'].lower()
            enabled = enabled_str in ('true', 'yes', '1', 'on')
        
        if 'category' in params_dict:
            category_str = params_dict['category'].upper()
            try:
                category = FeatureFlagCategory(category_str.lower())
            except ValueError:
                return f"⚠️ Invalid category '{category_str}'"
        
        if 'description' in params_dict:
            description = params_dict['description']
        
        # Create flag
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            category=category,
            description=description
        )
        
        success = self.feature_flag_service.add_flag(flag, f"telegram:{username}")
        
        if success:
            status = "enabled" if enabled else "disabled"
            return f"✅ Flag '{name}' added and {status}"
        else:
            return f"❌ Flag '{name}' already exists"
    
    def _handle_delete_flag(self, args: str, user_id: str, username: str) -> str:
        """Handle command to delete a flag"""
        if not args:
            return "⚠️ Usage: /deleteflag <name>"
        
        flag_name = args.strip()
        success = self.feature_flag_service.delete_flag(flag_name, f"telegram:{username}")
        
        if success:
            return f"✅ Flag '{flag_name}' has been deleted"
        else:
            return f"❌ Flag '{flag_name}' not found"
    
    def _handle_list_category(self, args: str, user_id: str, username: str) -> str:
        """Handle command to list flags in a category"""
        if not args:
            return "⚠️ Usage: /category <name>"
        
        category_name = args.strip().lower()
        
        try:
            category = FeatureFlagCategory(category_name)
        except ValueError:
            return f"❌ Invalid category '{category_name}'"
        
        flags = self.feature_flag_service.get_flags_by_category(category)
        
        if not flags:
            return f"No flags found in category '{category_name}'"
        
        # Sort by enabled status then by name
        flags.sort(key=lambda f: (not f.enabled, f.name))
        
        result = f"🚩 *Flags in category '{category_name}':*\n\n"
        for flag in flags:
            status = "✅" if flag.enabled else "❌"
            result += f"{status} `{flag.name}`"
            if flag.description:
                result += f" - {flag.description}"
            result += "\n"
        
        return result
    
    def _handle_list_categories(self, args: str, user_id: str, username: str) -> str:
        """Handle command to list all categories"""
        flags = self.feature_flag_service.get_all_flags()
        
        if not flags:
            return "No feature flags found"
        
        # Count flags per category
        categories = {}
        for flag in flags:
            cat = flag.category.value
            if cat not in categories:
                categories[cat] = {"total": 0, "enabled": 0}
            
            categories[cat]["total"] += 1
            if flag.enabled:
                categories[cat]["enabled"] += 1
        
        result = "🏷️ *Flag Categories:*\n\n"
        for cat_name, counts in sorted(categories.items()):
            result += f"• `{cat_name}`: {counts['enabled']}/{counts['total']} enabled\n"
        
        return result
    
    def _handle_help(self, args: str, user_id: str, username: str) -> str:
        """Handle help command"""
        result = "🚩 *Feature Flag Commands:*\n\n"
        
        for cmd, (_, perm, help_text) in sorted(self.commands.items()):
            # Only show commands the user has permission to use
            if self._check_permission(user_id, perm):
                result += f"/{cmd} - {help_text}\n"
        
        return result


def register_with_telegram_bot(bot, authorized_users=None):
    """
    Register feature flag commands with a Telegram bot
    
    Args:
        bot: The Telegram bot instance
        authorized_users: Dict mapping user IDs to permission levels
    
    Returns:
        The handler instance
    """
    handler = TelegramFeatureFlagHandler(authorized_users=authorized_users)
    
    # This assumes a typical bot structure with a message handler
    # Adjust as needed based on your Telegram bot implementation
    @bot.message_handler(func=lambda message: message.text and message.text.startswith('/'))
    def handle_commands(message):
        user_id = str(message.from_user.id)
        username = message.from_user.username or str(user_id)
        
        response = handler.handle_message(message.text, user_id, username)
        if response:
            bot.reply_to(message, response, parse_mode='Markdown')
    
    return handler 