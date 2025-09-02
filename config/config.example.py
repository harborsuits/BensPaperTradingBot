#!/usr/bin/env python3
"""
Configuration settings for the trading bot system.
IMPORTANT: Rename this file to config.py and add your actual API keys.
DO NOT commit config.py to version control with real API keys!
"""

# System settings
DEBUG = True
LOG_LEVEL = "INFO"
ENABLE_PAPER_TRADING = True  # Set to False for live trading (be careful!)

# Base URLs for API services
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
TRADIER_BASE_URL = "https://sandbox.tradier.com/v1"  # Use api.tradier.com for live
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"

# API keys - REPLACE THESE WITH YOUR ACTUAL KEYS
# AI Model API Keys
HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_API_KEY_SECONDARY = "your_backup_openai_api_key_here"
CLAUDE_API_KEY = "your_anthropic_claude_api_key_here"
MISTRAL_API_KEY = "your_mistral_api_key_here" 
COHERE_API_KEY = "your_cohere_api_key_here"
GEMINI_API_KEY = "your_google_gemini_api_key_here"

# Market Data API Keys  
ALPHAVANTAGE_API_KEY = "your_alphavantage_api_key_here"
FINNHUB_API_KEY = "your_finnhub_api_key_here"
TRADIER_API_KEY = "your_tradier_api_key_here"
ALPACA_API_KEY = "your_alpaca_api_key_here"
ALPACA_SECRET_KEY = "your_alpaca_secret_key_here"

# News API Keys
MARKETAUX_API_KEY = "your_marketaux_api_key_here"
NEWSDATA_API_KEY = "your_newsdata_api_key_here"
GNEWS_API_KEY = "your_gnews_api_key_here"
MEDIASTACK_API_KEY = "your_mediastack_api_key_here"
CURRENTS_API_KEY = "your_currents_api_key_here"
NYTIMES_API_KEY = "your_nytimes_api_key_here"

# Tradier account credentials
TRADIER_ACCOUNT_ID = "your_tradier_account_id_here"

# Telegram Bot
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token_here"
TELEGRAM_CHAT_ID = "your_telegram_chat_id_here"

# Database settings
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "trading_bot"
DB_USER = "username"
DB_PASSWORD = "password"

# Strategy settings
DEFAULT_STRATEGIES = ["MeanReversion", "TrendFollowing", "VolatilityBreakout"]
RISK_PERCENTAGE = 2.0  # Maximum percentage of account to risk per trade
MAX_POSITIONS = 5  # Maximum number of open positions
