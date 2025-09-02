from trading_bot.telegram_bot.market_commands import add_handlers as add_market_handlers
from trading_bot.telegram_bot.trade_commands import add_handlers as add_trade_handlers
from trading_bot.telegram_bot.system_commands import add_handlers as add_system_handlers
from trading_bot.telegram_bot.feature_flag_commands import add_handlers as add_feature_flag_handlers

def initialize_telegram_bot(token, allowed_users=None):
    """Initialize the Telegram bot with the given token."""
    # Create the bot and set allowed users
    global bot, updater, ALLOWED_USERS
    
    if allowed_users:
        ALLOWED_USERS = set(allowed_users)
    
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Create the Updater and dispatcher
    updater = Updater(token=token)
    dispatcher = updater.dispatcher
    
    # Add command handlers
    add_basic_handlers(dispatcher)
    add_market_handlers(dispatcher)
    add_trade_handlers(dispatcher)
    add_system_handlers(dispatcher)
    add_feature_flag_handlers(dispatcher)  # Add feature flag commands
    
    # Start the Bot
    updater.start_polling()
    
    # Store reference to the bot
    bot = updater.bot
    
    return bot 