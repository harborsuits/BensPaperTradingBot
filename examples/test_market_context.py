#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if available
load_dotenv()

from trading_bot.market_context.context_analyzer import MarketContextAnalyzer

def main():
    """Test the market context analyzer"""
    # Get API keys from environment
    marketaux_api_key = os.environ.get("MARKETAUX_API_KEY", "7PgROm6BE4m6ejBW8unmZnnYS6kIygu5lwzpfd9K")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    logger.info("Initializing MarketContextAnalyzer...")
    analyzer = MarketContextAnalyzer({
        "MARKETAUX_API_KEY": marketaux_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "CACHE_EXPIRY_MINUTES": 30
    })
    
    # Test 1: Global market context
    logger.info("Test 1: Getting global market context...")
    try:
        context = analyzer.get_market_context()
        logger.info(f"Market bias: {context.get('bias')}")
        logger.info(f"Confidence: {context.get('confidence')}")
        logger.info(f"Top triggers:")
        for trigger in context.get('triggers', [])[:3]:
            logger.info(f"  - {trigger}")
        logger.info(f"Reasoning: {context.get('reasoning')}")
        logger.info(f"Suggested strategies: {context.get('suggested_strategies')}")
    except Exception as e:
        logger.error(f"Error getting global market context: {str(e)}")
    
    # Test 2: Symbol-specific context
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        logger.info(f"\nTest 2: Getting context for {symbol}...")
        try:
            context = analyzer.get_market_context(focus_symbols=[symbol])
            logger.info(f"{symbol} bias: {context.get('bias')}")
            logger.info(f"Confidence: {context.get('confidence')}")
            logger.info(f"Top triggers:")
            for trigger in context.get('triggers', [])[:3]:
                logger.info(f"  - {trigger}")
            logger.info(f"Reasoning: {context.get('reasoning')}")
            logger.info(f"Suggested strategies: {context.get('suggested_strategies')}")
        except Exception as e:
            logger.error(f"Error getting context for {symbol}: {str(e)}")
    
    # Test 3: Raw data fetching
    logger.info("\nTest 3: Testing raw headline fetching...")
    try:
        marketaux_data = analyzer.get_marketaux_headlines(limit=5)
        logger.info(f"Fetched {len(marketaux_data.get('headlines', []))} headlines from Marketaux")
        if marketaux_data.get('headlines'):
            logger.info("Sample headlines:")
            for headline in marketaux_data.get('headlines', [])[:3]:
                logger.info(f"  - {headline}")
        
        finviz_data = analyzer.scrape_finviz_headlines(limit=5)
        logger.info(f"Fetched {len(finviz_data.get('headlines', []))} headlines from Finviz")
        if finviz_data.get('headlines'):
            logger.info("Sample headlines:")
            for headline in finviz_data.get('headlines', [])[:3]:
                logger.info(f"  - {headline}")
    except Exception as e:
        logger.error(f"Error fetching raw headlines: {str(e)}")
    
    logger.info("\nTests completed.")

if __name__ == "__main__":
    main() 