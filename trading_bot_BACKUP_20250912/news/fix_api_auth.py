#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from trading_bot.news.api_manager import NewsApiManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def update_api_keys():
    """Update API keys in the news API configuration"""
    
    # Initialize the news API manager
    news_api = NewsApiManager()
    
    # Load env vars or prompt for API keys
    api_keys = {
        'nytimes': os.environ.get('NYTIMES_API_KEY', ''),
        'finnhub': os.environ.get('FINNHUB_API_KEY', ''),
        'newsapi': os.environ.get('NEWSAPI_API_KEY', ''),
        'marketaux': os.environ.get('MARKETAUX_API_KEY', ''), 
        'newsdata': os.environ.get('NEWSDATA_API_KEY', ''),
        'gnews': os.environ.get('GNEWS_API_KEY', ''),
        'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY', '')
    }
    
    # If any key is missing from environment, prompt user
    missing_keys = False
    for api_name, key in api_keys.items():
        if not key:
            new_key = input(f"Enter API key for {api_name} (leave blank to skip): ")
            api_keys[api_name] = new_key
            if new_key:
                missing_keys = True
                # Update environment variable for this session
                os.environ[f"{api_name.upper()}_API_KEY"] = new_key
                
    if not missing_keys:
        logger.info("All API keys already set in environment variables")
    
    # Update API keys in configuration
    updated = False
    for api_name, key in api_keys.items():
        if api_name in news_api.api_data['apis'] and key:
            if news_api.api_data['apis'][api_name]['api_key'] != key:
                news_api.api_data['apis'][api_name]['api_key'] = key
                news_api.api_data['apis'][api_name]['enabled'] = True
                news_api.api_data['apis'][api_name]['error_count'] = 0
                updated = True
                logger.info(f"Updated API key for {api_name}")
    
    if updated:
        # Save updated configuration
        news_api._save_config()
        logger.info("Saved updated API configuration")
    else:
        logger.info("No API keys needed to be updated")
    
    # Reset error counts and re-enable all APIs
    news_api.reset_api_errors()
    
    return news_api.api_data

def fix_api_rate_limits():
    """Adjust API rate limits to prevent 429 errors"""
    
    # Initialize the news API manager
    news_api = NewsApiManager()
    
    # Update rate limits and cooldown periods
    rate_limit_updates = {
        'nytimes': {'daily_limit': 450, 'cooldown_minutes': 10},
        'finnhub': {'daily_limit': 220, 'cooldown_minutes': 15},
        'newsapi': {'daily_limit': 90, 'cooldown_minutes': 20},
        'marketaux': {'daily_limit': 90, 'cooldown_minutes': 360},
        'newsdata': {'daily_limit': 180, 'cooldown_minutes': 3900},
        'gnews': {'daily_limit': 90, 'cooldown_minutes': 1000},
        'alpha_vantage': {'daily_limit': 450, 'cooldown_minutes': 80}
    }
    
    updated = False
    for api_name, limits in rate_limit_updates.items():
        if api_name in news_api.api_data['apis']:
            for key, value in limits.items():
                if news_api.api_data['apis'][api_name][key] != value:
                    news_api.api_data['apis'][api_name][key] = value
                    updated = True
    
    if updated:
        # Save updated configuration
        news_api._save_config()
        logger.info("Updated API rate limits and cooldown periods")
    else:
        logger.info("API rate limits already properly configured")
    
    return news_api.api_data

def test_apis():
    """Test each API to verify authentication"""
    
    # Initialize the news API manager
    news_api = NewsApiManager()
    
    # Test queries
    test_queries = ['AAPL', 'MSFT', 'market news']
    
    results = {}
    
    logger.info("Testing each API with sample queries...")
    
    for api_name in news_api.api_data['apis']:
        if not news_api.api_data['apis'][api_name]['api_key']:
            logger.warning(f"Skipping {api_name} - no API key configured")
            results[api_name] = {'status': 'skipped', 'reason': 'No API key'}
            continue
            
        if not news_api.api_data['apis'][api_name]['enabled']:
            logger.warning(f"Skipping {api_name} - API disabled in configuration")
            results[api_name] = {'status': 'skipped', 'reason': 'Disabled in config'}
            continue
        
        # Force this API to be used
        original_select = news_api.select_api
        news_api.select_api = lambda *args, **kwargs: api_name
        
        try:
            # Try one of the test queries
            query = test_queries[0]
            logger.info(f"Testing {api_name} API with query: {query}")
            
            news = news_api.fetch_news(query, max_results=1)
            
            if news:
                results[api_name] = {'status': 'success', 'articles': len(news)}
                logger.info(f"✅ {api_name} API working - returned {len(news)} articles")
            else:
                results[api_name] = {'status': 'warning', 'reason': 'No articles returned'}
                logger.warning(f"⚠️ {api_name} API returned no articles")
        except Exception as e:
            results[api_name] = {'status': 'error', 'reason': str(e)}
            logger.error(f"❌ {api_name} API error: {str(e)}")
        finally:
            # Restore original select method
            news_api.select_api = original_select
    
    return results

if __name__ == "__main__":
    logger.info("Starting news API authentication fix utility")
    
    # Update API keys
    update_api_keys()
    
    # Fix rate limits
    fix_api_rate_limits()
    
    # Test APIs
    results = test_apis()
    
    # Summarize results
    working_apis = [api for api, result in results.items() if result['status'] == 'success']
    warning_apis = [api for api, result in results.items() if result['status'] == 'warning']
    error_apis = [api for api, result in results.items() if result['status'] == 'error']
    skipped_apis = [api for api, result in results.items() if result['status'] == 'skipped']
    
    logger.info("\n--- API Status Summary ---")
    logger.info(f"Working APIs: {len(working_apis)} ({', '.join(working_apis) if working_apis else 'None'})")
    logger.info(f"Warning APIs: {len(warning_apis)} ({', '.join(warning_apis) if warning_apis else 'None'})")
    logger.info(f"Failed APIs: {len(error_apis)} ({', '.join(error_apis) if error_apis else 'None'})")
    logger.info(f"Skipped APIs: {len(skipped_apis)} ({', '.join(skipped_apis) if skipped_apis else 'None'})")
    
    if working_apis:
        logger.info("\nAPI authentication has been fixed for some providers. The system will now use these working APIs.")
    else:
        logger.warning("\nNo working APIs found. You may need to check your API keys or subscription status.")
        logger.info("Please add valid API keys to make the system functional.")
    
    logger.info("News API fix utility completed") 