"""
Enhanced API integrations for the BensBot Trading Dashboard
Supports direct connections to multiple financial data sources
"""
import os
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import time
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger("dashboard.data_sources")

# API Keys from environment variables or config
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Cache for API responses
_api_cache = {}
_cache_expiry = {}

def _get_cached(cache_key, expiry_seconds=300):
    """Get data from cache if available and not expired"""
    if cache_key in _api_cache and cache_key in _cache_expiry:
        if datetime.now() < _cache_expiry[cache_key]:
            return _api_cache[cache_key]
    return None

def _cache_result(cache_key, data, expiry_seconds=300):
    """Cache API response with expiry time"""
    _api_cache[cache_key] = data
    _cache_expiry[cache_key] = datetime.now() + timedelta(seconds=expiry_seconds)

class AlphaVantageAPI:
    """Alpha Vantage API integration"""
    BASE_URL = "https://www.alphavantage.co/query"
    
    @staticmethod
    def get_market_data(symbol, interval="daily", outputsize="compact"):
        """Get market data from Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY:
            logger.warning("Alpha Vantage API key not configured")
            return None
            
        cache_key = f"av_{symbol}_{interval}_{outputsize}"
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
            
        params = {
            "function": "TIME_SERIES_DAILY" if interval == "daily" else 
                       "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": outputsize
        }
        
        if interval != "daily":
            params["interval"] = interval
            
        try:
            response = requests.get(AlphaVantageAPI.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process the data into a more usable format
            if interval == "daily" and "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume"
                })
                # Convert string values to float
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                
                df = df.sort_index()
                result = df.to_dict(orient="records")
                _cache_result(cache_key, result)
                return result
            return data
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
            return None
    
    @staticmethod
    def get_economic_indicators():
        """Get economic indicators from Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY:
            logger.warning("Alpha Vantage API key not configured")
            return None
            
        cache_key = "av_economic"
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
            
        indicators = ["REAL_GDP", "INFLATION", "UNEMPLOYMENT", "RETAIL_SALES"]
        results = {}
        
        for indicator in indicators:
            params = {
                "function": indicator,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            try:
                response = requests.get(AlphaVantageAPI.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                results[indicator] = data
                # Avoid rate limiting
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Alpha Vantage API error for {indicator}: {e}")
                
        _cache_result(cache_key, results)
        return results

class FinnhubAPI:
    """Finnhub API integration"""
    BASE_URL = "https://finnhub.io/api/v1"
    
    @staticmethod
    def get_company_news(symbol, from_date=None, to_date=None):
        """Get company news from Finnhub"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return None
            
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_key = f"finnhub_news_{symbol}_{from_date}_{to_date}"
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
            
        url = f"{FinnhubAPI.BASE_URL}/company-news"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            _cache_result(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            return None
    
    @staticmethod
    def get_market_sentiment(symbol):
        """Get market sentiment from Finnhub"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return None
            
        cache_key = f"finnhub_sentiment_{symbol}"
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
            
        url = f"{FinnhubAPI.BASE_URL}/news-sentiment"
        params = {
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            _cache_result(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            return None

class NewsDataAPI:
    """NewsData.io API integration"""
    BASE_URL = "https://newsdata.io/api/1/news"
    
    @staticmethod
    def get_financial_news(keywords=None, countries=None, language="en"):
        """Get financial news from NewsData.io"""
        if not NEWSDATA_API_KEY:
            logger.warning("NewsData.io API key not configured")
            return None
            
        # Generate cache key based on parameters
        cache_key_parts = ["newsdata"]
        if keywords:
            cache_key_parts.append("_".join(keywords))
        if countries:
            cache_key_parts.append("_".join(countries))
        cache_key_parts.append(language)
        cache_key = "_".join(cache_key_parts)
        
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
            
        params = {
            "apikey": NEWSDATA_API_KEY,
            "category": "business",
            "language": language
        }
        
        if keywords:
            params["q"] = " AND ".join(keywords)
        if countries:
            params["country"] = ",".join(countries)
            
        try:
            response = requests.get(NewsDataAPI.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Add impact assessment to each article
            if "results" in data and data["results"]:
                for article in data["results"]:
                    # Simple keyword based impact assessment
                    title = article.get("title", "").lower()
                    description = article.get("description", "").lower()
                    content = article.get("content", "").lower()
                    
                    positive_keywords = ["surge", "gain", "rise", "growth", "profit", "bullish"]
                    negative_keywords = ["drop", "fall", "decline", "loss", "bearish", "crash"]
                    
                    positive_count = sum(1 for kw in positive_keywords if kw in title or kw in description)
                    negative_count = sum(1 for kw in negative_keywords if kw in title or kw in description)
                    
                    if positive_count > negative_count:
                        article["impact"] = "positive"
                    elif negative_count > positive_count:
                        article["impact"] = "negative"
                    else:
                        article["impact"] = "neutral"
            
            _cache_result(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"NewsData.io API error: {e}")
            return None

# More specialized API integrations can be added here for:
# - Marketaux
# - Tradier
# - Alpaca
# - etc.

# API Cycling functionality
def get_best_api_for_data_type(data_type):
    """
    Get the best API to use based on rate limits, last usage, etc.
    
    Args:
        data_type: The type of data needed (market_data, news, etc.)
        
    Returns:
        API class to use
    """
    # This would implement the API cycling logic
    # For now, return a default based on data type
    if data_type == "market_data":
        return AlphaVantageAPI
    elif data_type == "news":
        return NewsDataAPI
    elif data_type == "sentiment":
        return FinnhubAPI
    else:
        return AlphaVantageAPI
