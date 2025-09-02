#!/usr/bin/env python3
"""
Telegram commands for feature flag management.

This module provides Telegram bot commands for managing feature flags.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CallbackContext, 
    CommandHandler, 
    CallbackQueryHandler, 
    ConversationHandler,
    MessageHandler,
    Filters
)

from trading_bot.feature_flags.feature_flags import (
    get_feature_flag_service,
    FeatureFlag
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
SELECTING_ACTION, SELECTING_CATEGORY, SELECTING_FLAG, UPDATING_FLAG, ADDING_FLAG = range(5)

# Callback data identifiers
PREFIX_CATEGORY = "category"
PREFIX_FLAG = "flag"
PREFIX_ACTION = "action"
PREFIX_ENABLE = "enable"
PREFIX_DISABLE = "disable"
PREFIX_BACK = "back"
PREFIX_CANCEL = "cancel"
PREFIX_CONFIRM = "confirm"
PREFIX_EXPIRY = "expiry"
PREFIX_ADD_FLAG = "add_flag"
PREFIX_ADD_NAME = "add_name"
PREFIX_ADD_DESC = "add_desc"
PREFIX_ADD_CATG = "add_catg"

# Admin user IDs that are allowed to manage feature flags
ADMIN_USER_IDS = []  # Add Telegram user IDs here, or load from config


def is_admin(user_id: int) -> bool:
    """Check if the user is an admin."""
    return len(ADMIN_USER_IDS) == 0 or user_id in ADMIN_USER_IDS


def format_flag(flag: FeatureFlag) -> str:
    """Format a feature flag for display."""
    status = "✅ ENABLED" if flag.enabled else "❌ DISABLED"
    expiry_str = f"Expires: {flag.expiry.isoformat()}" if flag.expiry else "No expiry"
    
    return (
        f"*{flag.name}* - {status}\n"
        f"_{flag.description}_\n"
        f"Category: {flag.category}\n"
        f"{expiry_str}\n"
        f"Last updated by: {flag.updated_by} at {flag.updated_at.isoformat()}"
    )


def build_category_keyboard() -> InlineKeyboardMarkup:
    """Build a keyboard with feature flag categories."""
    service = get_feature_flag_service()
    flags = service.get_all_flags()
    
    # Get unique categories
    categories = set(flag.category for flag in flags)
    
    # Build keyboard with categories
    keyboard = []
    for category in sorted(categories):
        keyboard.append([
            InlineKeyboardButton(
                category.capitalize(), 
                callback_data=f"{PREFIX_CATEGORY}:{category}"
            )
        ])
    
    # Add option to see all flags
    keyboard.append([
        InlineKeyboardButton("All Flags", callback_data=f"{PREFIX_CATEGORY}:all")
    ])
    
    # Add option to add a new flag
    keyboard.append([
        InlineKeyboardButton("➕ Add New Flag", callback_data=f"{PREFIX_ADD_FLAG}")
    ])
    
    # Add cancel button
    keyboard.append([InlineKeyboardButton("Cancel", callback_data=PREFIX_CANCEL)])
    
    return InlineKeyboardMarkup(keyboard)


def build_flag_keyboard(category: str) -> InlineKeyboardMarkup:
    """Build a keyboard with feature flags for a category."""
    service = get_feature_flag_service()
    
    if category == "all":
        flags = service.get_all_flags()
    else:
        flags = service.get_all_flags(category=category)
    
    # Build keyboard with flags
    keyboard = []
    for flag in sorted(flags, key=lambda f: f.name):
        status = "✅" if flag.enabled else "❌"
        keyboard.append([
            InlineKeyboardButton(
                f"{status} {flag.name}", 
                callback_data=f"{PREFIX_FLAG}:{flag.name}"
            )
        ])
    
    # Add back button
    keyboard.append([
        InlineKeyboardButton("◀️ Back", callback_data=PREFIX_BACK)
    ])
    
    return InlineKeyboardMarkup(keyboard)


def build_flag_action_keyboard(flag: FeatureFlag) -> InlineKeyboardMarkup:
    """Build a keyboard with actions for a flag."""
    keyboard = []
    
    # Toggle button
    if flag.enabled:
        keyboard.append([
            InlineKeyboardButton(
                "❌ Disable", 
                callback_data=f"{PREFIX_DISABLE}:{flag.name}"
            )
        ])
    else:
        keyboard.append([
            InlineKeyboardButton(
                "✅ Enable", 
                callback_data=f"{PREFIX_ENABLE}:{flag.name}"
            )
        ])
    
    # Temporary enable/disable buttons with different expirations
    if not flag.enabled:
        keyboard.append([
            InlineKeyboardButton(
                "Enable for 1 hour", 
                callback_data=f"{PREFIX_EXPIRY}:{flag.name}:enable:3600"
            ),
            InlineKeyboardButton(
                "Enable for 1 day", 
                callback_data=f"{PREFIX_EXPIRY}:{flag.name}:enable:86400"
            )
        ])
    else:
        keyboard.append([
            InlineKeyboardButton(
                "Disable for 1 hour", 
                callback_data=f"{PREFIX_EXPIRY}:{flag.name}:disable:3600"
            ),
            InlineKeyboardButton(
                "Disable for 1 day", 
                callback_data=f"{PREFIX_EXPIRY}:{flag.name}:disable:86400"
            )
        ])
    
    # Back button
    keyboard.append([
        InlineKeyboardButton(
            "◀️ Back", 
            callback_data=f"{PREFIX_CATEGORY}:{flag.category}"
        )
    ])
    
    return InlineKeyboardMarkup(keyboard)


def flags_command(update: Update, context: CallbackContext) -> int:
    """Handle the /flags command."""
    user_id = update.effective_user.id
    
    # Check if user is admin
    if not is_admin(user_id):
        update.message.reply_text("You are not authorized to manage feature flags.")
        return ConversationHandler.END
    
    # Show categories
    update.message.reply_text(
        "Select a category of feature flags to manage:",
        reply_markup=build_category_keyboard()
    )
    
    return SELECTING_CATEGORY


def status_command(update: Update, context: CallbackContext) -> None:
    """Handle the /flagstatus command to show current status of all flags."""
    user_id = update.effective_user.id
    
    # Check if user is admin
    if not is_admin(user_id):
        update.message.reply_text("You are not authorized to view feature flags.")
        return
    
    service = get_feature_flag_service()
    flags = service.get_all_flags()
    
    # Group flags by category
    flags_by_category = {}
    for flag in flags:
        if flag.category not in flags_by_category:
            flags_by_category[flag.category] = []
        flags_by_category[flag.category].append(flag)
    
    # Build message
    message = "🚩 *Feature Flag Status* 🚩\n\n"
    
    for category, category_flags in sorted(flags_by_category.items()):
        message += f"*{category.upper()}*\n"
        
        for flag in sorted(category_flags, key=lambda f: f.name):
            status = "✅" if flag.enabled else "❌"
            message += f"{status} {flag.name}: _{flag.description}_\n"
        
        message += "\n"
    
    update.message.reply_text(message, parse_mode="Markdown")


def category_callback(update: Update, context: CallbackContext) -> int:
    """Handle category selection."""
    query = update.callback_query
    query.answer()
    
    data = query.data.split(":")
    if len(data) < 2:
        return SELECTING_CATEGORY
    
    category = data[1]
    context.user_data["selected_category"] = category
    
    # Show flags for the selected category
    query.edit_message_text(
        f"Feature flags in category: *{category.capitalize() if category != 'all' else 'All Categories'}*\n"
        "Select a flag to manage:",
        reply_markup=build_flag_keyboard(category),
        parse_mode="Markdown"
    )
    
    return SELECTING_FLAG


def flag_callback(update: Update, context: CallbackContext) -> int:
    """Handle flag selection."""
    query = update.callback_query
    query.answer()
    
    data = query.data.split(":")
    if len(data) < 2:
        return SELECTING_FLAG
    
    flag_name = data[1]
    service = get_feature_flag_service()
    flag = service.get_flag(flag_name)
    
    if not flag:
        query.edit_message_text(
            f"Flag {flag_name} not found. Please try again.",
            reply_markup=build_category_keyboard()
        )
        return SELECTING_CATEGORY
    
    # Store the selected flag
    context.user_data["selected_flag"] = flag_name
    
    # Show flag details and actions
    query.edit_message_text(
        f"Flag: *{flag_name}*\n\n{format_flag(flag)}\n\nSelect an action:",
        reply_markup=build_flag_action_keyboard(flag),
        parse_mode="Markdown"
    )
    
    return UPDATING_FLAG


def back_callback(update: Update, context: CallbackContext) -> int:
    """Handle back button press."""
    query = update.callback_query
    query.answer()
    
    # Go back to category selection
    query.edit_message_text(
        "Select a category of feature flags to manage:",
        reply_markup=build_category_keyboard()
    )
    
    return SELECTING_CATEGORY


def cancel_callback(update: Update, context: CallbackContext) -> int:
    """Handle cancel button press."""
    query = update.callback_query
    query.answer()
    
    # End the conversation
    query.edit_message_text("Feature flag management cancelled.")
    
    return ConversationHandler.END


def enable_flag_callback(update: Update, context: CallbackContext) -> int:
    """Handle enable flag button press."""
    query = update.callback_query
    query.answer()
    
    data = query.data.split(":")
    if len(data) < 2:
        return UPDATING_FLAG
    
    flag_name = data[1]
    service = get_feature_flag_service()
    
    # Enable the flag
    user_name = update.effective_user.username or str(update.effective_user.id)
    result = service.update_flag(
        name=flag_name,
        enabled=True,
        updated_by=f"telegram:{user_name}"
    )
    
    if result:
        # Get the updated flag
        flag = service.get_flag(flag_name)
        
        # Show updated flag details
        query.edit_message_text(
            f"Flag *{flag_name}* has been enabled.\n\n{format_flag(flag)}\n\nSelect an action:",
            reply_markup=build_flag_action_keyboard(flag),
            parse_mode="Markdown"
        )
    else:
        query.edit_message_text(
            f"Failed to enable flag {flag_name}. Please try again.",
            reply_markup=build_category_keyboard()
        )
        return SELECTING_CATEGORY
    
    return UPDATING_FLAG


def disable_flag_callback(update: Update, context: CallbackContext) -> int:
    """Handle disable flag button press."""
    query = update.callback_query
    query.answer()
    
    data = query.data.split(":")
    if len(data) < 2:
        return UPDATING_FLAG
    
    flag_name = data[1]
    service = get_feature_flag_service()
    
    # Disable the flag
    user_name = update.effective_user.username or str(update.effective_user.id)
    result = service.update_flag(
        name=flag_name,
        enabled=False,
        updated_by=f"telegram:{user_name}"
    )
    
    if result:
        # Get the updated flag
        flag = service.get_flag(flag_name)
        
        # Show updated flag details
        query.edit_message_text(
            f"Flag *{flag_name}* has been disabled.\n\n{format_flag(flag)}\n\nSelect an action:",
            reply_markup=build_flag_action_keyboard(flag),
            parse_mode="Markdown"
        )
    else:
        query.edit_message_text(
            f"Failed to disable flag {flag_name}. Please try again.",
            reply_markup=build_category_keyboard()
        )
        return SELECTING_CATEGORY
    
    return UPDATING_FLAG


def expiry_flag_callback(update: Update, context: CallbackContext) -> int:
    """Handle temporary enable/disable with expiry."""
    query = update.callback_query
    query.answer()
    
    data = query.data.split(":")
    if len(data) < 4:
        return UPDATING_FLAG
    
    flag_name = data[1]
    action = data[2]  # enable or disable
    seconds = int(data[3])
    
    service = get_feature_flag_service()
    
    # Calculate expiry
    expiry = datetime.now() + timedelta(seconds=seconds)
    
    # Update the flag
    user_name = update.effective_user.username or str(update.effective_user.id)
    result = service.update_flag(
        name=flag_name,
        enabled=(action == "enable"),
        expiry=expiry,
        updated_by=f"telegram:{user_name}"
    )
    
    if result:
        # Get the updated flag
        flag = service.get_flag(flag_name)
        
        # Show updated flag details
        duration = "1 hour" if seconds == 3600 else "1 day"
        query.edit_message_text(
            f"Flag *{flag_name}* has been {action}d for {duration}.\n\n"
            f"{format_flag(flag)}\n\nSelect an action:",
            reply_markup=build_flag_action_keyboard(flag),
            parse_mode="Markdown"
        )
    else:
        query.edit_message_text(
            f"Failed to update flag {flag_name}. Please try again.",
            reply_markup=build_category_keyboard()
        )
        return SELECTING_CATEGORY
    
    return UPDATING_FLAG


def add_flag_callback(update: Update, context: CallbackContext) -> int:
    """Handle add new flag button press."""
    query = update.callback_query
    query.answer()
    
    # Ask for flag name
    query.edit_message_text(
        "Enter the name for the new feature flag (e.g., enable_new_strategy):"
    )
    
    return ADDING_FLAG


def add_flag_name(update: Update, context: CallbackContext) -> int:
    """Handle flag name input."""
    # Get flag name from user message
    flag_name = update.message.text.strip()
    
    # Validate flag name
    if not flag_name or " " in flag_name:
        update.message.reply_text(
            "Invalid flag name. Flag names should not contain spaces. Please try again:"
        )
        return ADDING_FLAG
    
    # Check if flag already exists
    service = get_feature_flag_service()
    if service.get_flag(flag_name):
        update.message.reply_text(
            f"A flag with the name '{flag_name}' already exists. Please enter a different name:"
        )
        return ADDING_FLAG
    
    # Store flag name
    context.user_data["new_flag_name"] = flag_name
    
    # Ask for description
    update.message.reply_text(
        f"Flag name: *{flag_name}*\n\n"
        "Enter a description for this feature flag:",
        parse_mode="Markdown"
    )
    
    return ADD_FLAG_DESC


def add_flag_desc(update: Update, context: CallbackContext) -> int:
    """Handle flag description input."""
    # Get description from user message
    description = update.message.text.strip()
    
    # Store description
    context.user_data["new_flag_desc"] = description
    
    # Ask for category
    update.message.reply_text(
        f"Flag name: *{context.user_data['new_flag_name']}*\n"
        f"Description: _{description}_\n\n"
        "Enter a category for this flag (e.g., trading, risk, system):",
        parse_mode="Markdown"
    )
    
    return ADD_FLAG_CATG


def add_flag_category(update: Update, context: CallbackContext) -> int:
    """Handle flag category input and create the flag."""
    # Get category from user message
    category = update.message.text.strip().lower()
    
    # Store category
    context.user_data["new_flag_catg"] = category
    
    # Build confirmation keyboard
    keyboard = [
        [
            InlineKeyboardButton("Create Enabled", callback_data=f"{PREFIX_CONFIRM}:enabled"),
            InlineKeyboardButton("Create Disabled", callback_data=f"{PREFIX_CONFIRM}:disabled")
        ],
        [InlineKeyboardButton("Cancel", callback_data=PREFIX_CANCEL)]
    ]
    
    # Ask for confirmation
    update.message.reply_text(
        f"Ready to create new feature flag:\n\n"
        f"Name: *{context.user_data['new_flag_name']}*\n"
        f"Description: _{context.user_data['new_flag_desc']}_\n"
        f"Category: {category}\n\n"
        f"Do you want to create this flag as enabled or disabled?",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
    
    return ADD_FLAG_CONFIRM


def add_flag_confirm(update: Update, context: CallbackContext) -> int:
    """Handle flag creation confirmation."""
    query = update.callback_query
    query.answer()
    
    data = query.data.split(":")
    if len(data) < 2:
        return ADD_FLAG_CONFIRM
    
    action = data[1]  # enabled or disabled
    
    # Get flag data from context
    flag_name = context.user_data.get("new_flag_name")
    description = context.user_data.get("new_flag_desc")
    category = context.user_data.get("new_flag_catg")
    
    if not all([flag_name, description, category]):
        query.edit_message_text(
            "Error: Missing flag information. Please try again.",
            reply_markup=build_category_keyboard()
        )
        return SELECTING_CATEGORY
    
    # Create the flag
    service = get_feature_flag_service()
    user_name = update.effective_user.username or str(update.effective_user.id)
    
    result = service.create_flag(
        name=flag_name,
        enabled=(action == "enabled"),
        description=description,
        category=category,
        created_by=f"telegram:{user_name}"
    )
    
    if result:
        # Get the created flag
        flag = service.get_flag(flag_name)
        
        # Show created flag details
        query.edit_message_text(
            f"Feature flag *{flag_name}* has been created.\n\n"
            f"{format_flag(flag)}\n\n"
            f"Select a category to continue:",
            reply_markup=build_category_keyboard(),
            parse_mode="Markdown"
        )
    else:
        query.edit_message_text(
            f"Failed to create flag {flag_name}. Please try again.",
            reply_markup=build_category_keyboard()
        )
    
    # Clear user data
    if "new_flag_name" in context.user_data:
        del context.user_data["new_flag_name"]
    if "new_flag_desc" in context.user_data:
        del context.user_data["new_flag_desc"]
    if "new_flag_catg" in context.user_data:
        del context.user_data["new_flag_catg"]
    
    return SELECTING_CATEGORY


def add_handlers(dispatcher):
    """Add feature flag command handlers to the dispatcher."""
    # Add conversation handler for feature flag management
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("flags", flags_command)],
        states={
            SELECTING_CATEGORY: [
                CallbackQueryHandler(category_callback, pattern=f"^{PREFIX_CATEGORY}:"),
                CallbackQueryHandler(add_flag_callback, pattern=f"^{PREFIX_ADD_FLAG}$"),
                CallbackQueryHandler(cancel_callback, pattern=f"^{PREFIX_CANCEL}$")
            ],
            SELECTING_FLAG: [
                CallbackQueryHandler(flag_callback, pattern=f"^{PREFIX_FLAG}:"),
                CallbackQueryHandler(back_callback, pattern=f"^{PREFIX_BACK}$")
            ],
            UPDATING_FLAG: [
                CallbackQueryHandler(enable_flag_callback, pattern=f"^{PREFIX_ENABLE}:"),
                CallbackQueryHandler(disable_flag_callback, pattern=f"^{PREFIX_DISABLE}:"),
                CallbackQueryHandler(expiry_flag_callback, pattern=f"^{PREFIX_EXPIRY}:"),
                CallbackQueryHandler(category_callback, pattern=f"^{PREFIX_CATEGORY}:")
            ],
            ADDING_FLAG: [MessageHandler(Filters.text & ~Filters.command, add_flag_name)],
            ADD_FLAG_DESC: [MessageHandler(Filters.text & ~Filters.command, add_flag_desc)],
            ADD_FLAG_CATG: [MessageHandler(Filters.text & ~Filters.command, add_flag_category)],
            ADD_FLAG_CONFIRM: [
                CallbackQueryHandler(add_flag_confirm, pattern=f"^{PREFIX_CONFIRM}:"),
                CallbackQueryHandler(cancel_callback, pattern=f"^{PREFIX_CANCEL}$")
            ]
        },
        fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)]
    )
    
    dispatcher.add_handler(conv_handler)
    
    # Add status command
    dispatcher.add_handler(CommandHandler("flagstatus", status_command)) 