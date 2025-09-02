#!/usr/bin/env python3
"""
Telegram Bot Handlers for Feature Flag Management

This module provides handlers for managing feature flags via Telegram commands.
"""

import logging
from typing import Dict, List, Optional, Tuple
import textwrap

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CommandHandler, CallbackQueryHandler, Filters, MessageHandler

from trading_bot.feature_flags.feature_flag_service import (
    get_feature_flag_service, 
    FlagCategory,
    FeatureFlag
)

logger = logging.getLogger(__name__)

# Command prefixes
FLAG_PREFIX = "flag_"
CATEGORY_PREFIX = "category_"
ENABLE_PREFIX = "enable_"
DISABLE_PREFIX = "disable_"
TOGGLE_PREFIX = "toggle_"

def add_feature_flag_handlers(dispatcher):
    """
    Add feature flag command handlers to the dispatcher
    
    Args:
        dispatcher: The telegram dispatcher to add handlers to
    """
    dispatcher.add_handler(CommandHandler("flags", cmd_flags))
    dispatcher.add_handler(CommandHandler("flag", cmd_flag_details))
    dispatcher.add_handler(CommandHandler("addflag", cmd_add_flag, pass_args=True))
    dispatcher.add_handler(CommandHandler("enableflag", cmd_enable_flag, pass_args=True))
    dispatcher.add_handler(CommandHandler("disableflag", cmd_disable_flag, pass_args=True))
    dispatcher.add_handler(CommandHandler("toggleflag", cmd_toggle_flag, pass_args=True))
    dispatcher.add_handler(CallbackQueryHandler(flag_button_callback))

def cmd_flags(update: Update, context: CallbackContext):
    """
    Command handler for /flags - lists all feature flags by category
    """
    service = get_feature_flag_service()
    categories = service.get_categories()
    
    # Create inline keyboard with categories
    keyboard = []
    for i in range(0, len(categories), 2):
        row = []
        for j in range(2):
            idx = i + j
            if idx < len(categories):
                category = categories[idx]
                row.append(InlineKeyboardButton(
                    f"{category.value.capitalize()}", 
                    callback_data=f"{CATEGORY_PREFIX}{category.value}"
                ))
        keyboard.append(row)
    
    # Add an "All Flags" button
    keyboard.append([InlineKeyboardButton("All Flags", callback_data="all_flags")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        "🚩 *Feature Flag Management*\n\n"
        "Select a category to view flags:",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

def cmd_flag_details(update: Update, context: CallbackContext):
    """
    Command handler for /flag <name> - shows details for a specific flag
    """
    if not context.args or len(context.args) < 1:
        update.message.reply_text(
            "Please provide a flag name. Example: `/flag enable_automatic_trading`",
            parse_mode="Markdown"
        )
        return
    
    flag_name = context.args[0]
    service = get_feature_flag_service()
    flag = service.get_flag(flag_name)
    
    if not flag:
        update.message.reply_text(f"❌ Flag '{flag_name}' not found")
        return
    
    keyboard = [
        [
            InlineKeyboardButton("Enable", callback_data=f"{ENABLE_PREFIX}{flag_name}"),
            InlineKeyboardButton("Disable", callback_data=f"{DISABLE_PREFIX}{flag_name}"),
            InlineKeyboardButton("Toggle", callback_data=f"{TOGGLE_PREFIX}{flag_name}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        format_flag_details(flag),
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

def cmd_add_flag(update: Update, context: CallbackContext):
    """
    Command handler for /addflag <name> <enabled> <category> <description>
    Adds a new feature flag
    """
    if not context.args or len(context.args) < 3:
        update.message.reply_text(
            "Please provide flag name, enabled state, and category.\n"
            "Example: `/addflag new_strategy_alpha true strategy This is a new alpha strategy`",
            parse_mode="Markdown"
        )
        return
    
    flag_name = context.args[0]
    
    # Parse enabled state
    enabled_str = context.args[1].lower()
    if enabled_str in ("true", "yes", "1", "on"):
        enabled = True
    elif enabled_str in ("false", "no", "0", "off"):
        enabled = False
    else:
        update.message.reply_text(
            f"Invalid enabled state '{enabled_str}'. Please use true/false."
        )
        return
    
    # Parse category
    category_str = context.args[2].lower()
    try:
        category = FlagCategory(category_str)
    except ValueError:
        valid_categories = [cat.value for cat in FlagCategory]
        update.message.reply_text(
            f"Invalid category '{category_str}'. Valid categories: {', '.join(valid_categories)}"
        )
        return
    
    # Get description (all remaining args)
    description = " ".join(context.args[3:]) if len(context.args) > 3 else ""
    
    # Add the flag
    service = get_feature_flag_service()
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.username or update.effective_user.first_name
    modified_by = f"telegram:{user_id}:{user_name}"
    
    success = service.add_flag(
        name=flag_name,
        enabled=enabled,
        description=description,
        category=category,
        modified_by=modified_by
    )
    
    if success:
        flag = service.get_flag(flag_name)
        update.message.reply_text(
            f"✅ Flag added successfully:\n\n{format_flag_details(flag)}",
            parse_mode="Markdown"
        )
    else:
        update.message.reply_text(f"❌ Flag '{flag_name}' already exists")

def cmd_enable_flag(update: Update, context: CallbackContext):
    """
    Command handler for /enableflag <name> - enables a feature flag
    """
    if not context.args or len(context.args) < 1:
        update.message.reply_text(
            "Please provide a flag name. Example: `/enableflag enable_automatic_trading`",
            parse_mode="Markdown"
        )
        return
    
    flag_name = context.args[0]
    service = get_feature_flag_service()
    
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.username or update.effective_user.first_name
    modified_by = f"telegram:{user_id}:{user_name}"
    
    success = service.enable_flag(flag_name, modified_by)
    
    if success:
        flag = service.get_flag(flag_name)
        update.message.reply_text(
            f"✅ Flag enabled:\n\n{format_flag_details(flag)}",
            parse_mode="Markdown"
        )
    else:
        update.message.reply_text(f"❌ Flag '{flag_name}' not found")

def cmd_disable_flag(update: Update, context: CallbackContext):
    """
    Command handler for /disableflag <name> - disables a feature flag
    """
    if not context.args or len(context.args) < 1:
        update.message.reply_text(
            "Please provide a flag name. Example: `/disableflag enable_automatic_trading`",
            parse_mode="Markdown"
        )
        return
    
    flag_name = context.args[0]
    service = get_feature_flag_service()
    
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.username or update.effective_user.first_name
    modified_by = f"telegram:{user_id}:{user_name}"
    
    success = service.disable_flag(flag_name, modified_by)
    
    if success:
        flag = service.get_flag(flag_name)
        update.message.reply_text(
            f"✅ Flag disabled:\n\n{format_flag_details(flag)}",
            parse_mode="Markdown"
        )
    else:
        update.message.reply_text(f"❌ Flag '{flag_name}' not found")

def cmd_toggle_flag(update: Update, context: CallbackContext):
    """
    Command handler for /toggleflag <name> - toggles a feature flag
    """
    if not context.args or len(context.args) < 1:
        update.message.reply_text(
            "Please provide a flag name. Example: `/toggleflag enable_automatic_trading`",
            parse_mode="Markdown"
        )
        return
    
    flag_name = context.args[0]
    service = get_feature_flag_service()
    
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.username or update.effective_user.first_name
    modified_by = f"telegram:{user_id}:{user_name}"
    
    new_state = service.toggle_flag(flag_name, modified_by)
    
    if new_state is not None:
        flag = service.get_flag(flag_name)
        update.message.reply_text(
            f"✅ Flag toggled to {'enabled' if new_state else 'disabled'}:\n\n{format_flag_details(flag)}",
            parse_mode="Markdown"
        )
    else:
        update.message.reply_text(f"❌ Flag '{flag_name}' not found")

def flag_button_callback(update: Update, context: CallbackContext):
    """
    Callback handler for inline keyboard buttons related to feature flags
    """
    query = update.callback_query
    query.answer()
    
    data = query.data
    service = get_feature_flag_service()
    
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.username or update.effective_user.first_name
    modified_by = f"telegram:{user_id}:{user_name}"
    
    # Handle category selection
    if data.startswith(CATEGORY_PREFIX):
        category_name = data[len(CATEGORY_PREFIX):]
        try:
            category = FlagCategory(category_name)
            flags = service.get_flags_by_category(category)
            show_flags_list(query, flags, f"Category: {category.value.capitalize()}")
        except ValueError:
            query.edit_message_text(f"Invalid category: {category_name}")
    
    # Handle "All Flags" button
    elif data == "all_flags":
        flags = service.get_all_flags()
        show_flags_list(query, flags, "All Feature Flags")
    
    # Handle flag selection
    elif data.startswith(FLAG_PREFIX):
        flag_name = data[len(FLAG_PREFIX):]
        flag = service.get_flag(flag_name)
        if flag:
            keyboard = [
                [
                    InlineKeyboardButton("Enable", callback_data=f"{ENABLE_PREFIX}{flag_name}"),
                    InlineKeyboardButton("Disable", callback_data=f"{DISABLE_PREFIX}{flag_name}"),
                    InlineKeyboardButton("Toggle", callback_data=f"{TOGGLE_PREFIX}{flag_name}")
                ],
                [InlineKeyboardButton("« Back", callback_data=f"{CATEGORY_PREFIX}{flag.category.value}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            query.edit_message_text(
                format_flag_details(flag),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            query.edit_message_text(f"Flag '{flag_name}' not found")
    
    # Handle enable/disable/toggle actions
    elif data.startswith(ENABLE_PREFIX) or data.startswith(DISABLE_PREFIX) or data.startswith(TOGGLE_PREFIX):
        if data.startswith(ENABLE_PREFIX):
            flag_name = data[len(ENABLE_PREFIX):]
            success = service.enable_flag(flag_name, modified_by)
            action = "enabled"
        elif data.startswith(DISABLE_PREFIX):
            flag_name = data[len(DISABLE_PREFIX):]
            success = service.disable_flag(flag_name, modified_by)
            action = "disabled"
        else:  # TOGGLE_PREFIX
            flag_name = data[len(TOGGLE_PREFIX):]
            new_state = service.toggle_flag(flag_name, modified_by)
            success = new_state is not None
            action = f"toggled to {'enabled' if new_state else 'disabled'}" if success else "toggled"
        
        if success:
            flag = service.get_flag(flag_name)
            keyboard = [
                [
                    InlineKeyboardButton("Enable", callback_data=f"{ENABLE_PREFIX}{flag_name}"),
                    InlineKeyboardButton("Disable", callback_data=f"{DISABLE_PREFIX}{flag_name}"),
                    InlineKeyboardButton("Toggle", callback_data=f"{TOGGLE_PREFIX}{flag_name}")
                ],
                [InlineKeyboardButton("« Back", callback_data=f"{CATEGORY_PREFIX}{flag.category.value}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            query.edit_message_text(
                f"✅ Flag {action}:\n\n{format_flag_details(flag)}",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            query.edit_message_text(f"❌ Flag '{flag_name}' not found")

def show_flags_list(query, flags: Dict[str, FeatureFlag], title: str):
    """
    Show a list of flags with buttons
    
    Args:
        query: The callback query
        flags: Dictionary of flags to display
        title: Title for the message
    """
    if not flags:
        query.edit_message_text(f"{title}\n\nNo flags found in this category.")
        return
    
    # Create flag buttons
    keyboard = []
    
    # Sort flags by name
    sorted_flags = sorted(flags.items(), key=lambda x: x[0])
    
    for name, flag in sorted_flags:
        status = "✅" if flag.enabled else "❌"
        keyboard.append([
            InlineKeyboardButton(
                f"{status} {name}", 
                callback_data=f"{FLAG_PREFIX}{name}"
            )
        ])
    
    # Add a back button
    keyboard.append([InlineKeyboardButton("« Back to Categories", callback_data="back_to_categories")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(
        f"🚩 *{title}*\n\n"
        f"Total: {len(flags)} flags\n"
        f"Enabled: {sum(1 for f in flags.values() if f.enabled)}\n"
        f"Disabled: {sum(1 for f in flags.values() if not f.enabled)}\n\n"
        f"Select a flag to view details:",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

def format_flag_details(flag: FeatureFlag) -> str:
    """
    Format a feature flag as a Markdown string
    
    Args:
        flag: The feature flag to format
        
    Returns:
        str: Formatted flag details
    """
    status = "✅ Enabled" if flag.enabled else "❌ Disabled"
    
    # Format metadata as a string if it exists
    metadata_str = ""
    if flag.metadata:
        metadata_items = [f"  - {k}: {v}" for k, v in flag.metadata.items()]
        metadata_str = "\n*Metadata*:\n" + "\n".join(metadata_items)
    
    # Format last modified timestamp
    import datetime
    last_modified = datetime.datetime.fromtimestamp(flag.last_modified).strftime('%Y-%m-%d %H:%M:%S')
    
    return (
        f"*Flag*: `{flag.name}`\n"
        f"*Status*: {status}\n"
        f"*Category*: {flag.category.value.capitalize()}\n"
        f"*Description*: {flag.description}\n"
        f"*Last Modified*: {last_modified}\n"
        f"*Modified By*: {flag.modified_by}"
        f"{metadata_str}"
    ) 