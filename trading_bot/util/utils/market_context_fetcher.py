"""
Market Context Fetcher

This module fetches and processes real-time market data from various sources
to create a comprehensive context for strategy evaluation and rotation.
"""

import os
import json
import time
import logging
import requests
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("MarketContextFetcher")

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Cache settings
CACHE_EXPIRY = {
    "vix": 15 * 60,  # 15 minutes for VIX
    "market_indices": 15 * 60,  # 15 minutes for market indices
    "sector_performance": 60 * 60,  # 1 hour for sector performance
    "news": 30 * 60,  # 30 minutes for news
    "economic_data": 24 * 60 * 60,  # 1 day for economic data
}

# Cache storage
_cache = {
    "vix": {"data": None, "timestamp": 0},
    "market_indices": {"data": None, "timestamp": 0},
    "sector_performance": {"data": None, "timestamp": 0},
    "news": {"data": None, "timestamp": 0},
    "economic_data": {"data": None, "timestamp": 0},
}


def _is_cache_valid(cache_key: str) -> bool:
    """Check if cache is valid and not expired."""
    if not _cache.get(cache_key) or not _cache[cache_key].get("data"):
        return False
    
    current_time = time.time()
    cache_time = _cache[cache_key].get("timestamp", 0)
    
    return (current_time - cache_time) < CACHE_EXPIRY.get(cache_key, 0)


def _update_cache(cache_key: str, data: Any) -> None:
    """Update cache with new data."""
    _cache[cache_key] = {"data": data, "timestamp": time.time()}


def get_vix_data() -> Dict[str, Any]:
    """
    Get current VIX data and historical levels.
    
    Returns:
        Dictionary with current VIX value, historical data, and volatility classification
    """
    # Check cache first
    if _is_cache_valid("vix"):
        return _cache["vix"]["data"]
    
    try:
        # Fetch VIX data from Yahoo Finance
        vix = yf.Ticker("^VIX")
        current_vix = vix.history(period="1d")["Close"][-1]
        
        # Get historical VIX data for context
        vix_history = vix.history(period="6mo")
        
        # Calculate percentile of current VIX relative to 6-month range
        vix_min = vix_history["Low"].min()
        vix_max = vix_history["High"].max()
        vix_percentile = (current_vix - vix_min) / (vix_max - vix_min) if vix_max > vix_min else 0.5
        
        # Determine volatility regime
        if current_vix < 15:
            regime = "low_volatility"
        elif current_vix < 25:
            regime = "normal_volatility"
        elif current_vix < 35:
            regime = "high_volatility"
        else:
            regime = "extreme_volatility"
        
        # Calculate 10-day change in VIX
        vix_10d_ago = vix_history["Close"][-10] if len(vix_history) >= 10 else vix_history["Close"][0]
        vix_change = ((current_vix - vix_10d_ago) / vix_10d_ago) * 100
        
        vix_data = {
            "current": current_vix,
            "percentile": vix_percentile,
            "min_6m": vix_min,
            "max_6m": vix_max,
            "regime": regime,
            "change_10d_pct": vix_change,
            "trend": "rising" if vix_change > 5 else "falling" if vix_change < -5 else "stable"
        }
        
        # Update cache
        _update_cache("vix", vix_data)
        return vix_data
        
    except Exception as e:
        logger.error(f"Error fetching VIX data: {str(e)}")
        # Return a default/fallback response
        return {
            "current": 15.0,  # Default moderate VIX
            "percentile": 0.5,
            "min_6m": 12.0,
            "max_6m": 20.0,
            "regime": "normal_volatility",
            "change_10d_pct": 0.0,
            "trend": "stable",
            "error": str(e)
        }


def get_market_indices() -> Dict[str, Any]:
    """
    Get current market indices data including price, momentum, and trend indicators.
    
    Returns:
        Dictionary with major indices data and trend indicators
    """
    # Check cache first
    if _is_cache_valid("market_indices"):
        return _cache["market_indices"]["data"]
    
    # List of indices to track
    indices = [
        {"symbol": "^GSPC", "name": "S&P 500"},
        {"symbol": "^NDX", "name": "Nasdaq 100"},
        {"symbol": "^DJI", "name": "Dow Jones"},
        {"symbol": "^RUT", "name": "Russell 2000"},
        {"symbol": "IWM", "name": "Russell 2000 ETF"}
    ]
    
    result = {}
    
    try:
        for idx in indices:
            symbol = idx["symbol"]
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                continue
            
            # Get current price and changes
            current_price = hist["Close"][-1]
            prev_close = hist["Close"][-2] if len(hist) > 1 else current_price
            daily_change = ((current_price - prev_close) / prev_close) * 100
            
            # Calculate moving averages
            ma50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            ma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
            
            # Calculate 1-month change
            price_1m_ago = hist["Close"][-22] if len(hist) >= 22 else hist["Close"][0]
            monthly_change = ((current_price - price_1m_ago) / price_1m_ago) * 100
            
            # Calculate RSI (14-period)
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            result[idx["name"]] = {
                "symbol": symbol,
                "price": current_price,
                "daily_change_pct": daily_change,
                "monthly_change_pct": monthly_change,
                "above_50ma": current_price > ma50,
                "above_200ma": current_price > ma200,
                "ma50": ma50,
                "ma200": ma200,
                "rsi": current_rsi,
                "ma_trend": "bullish" if ma50 > ma200 else "bearish"
            }
        
        # Determine overall market regime
        sp500 = result.get("S&P 500", {})
        nasdaq = result.get("Nasdaq 100", {})
        
        if sp500 and nasdaq:
            # Determine market regime
            if (sp500.get("above_200ma", False) and 
                nasdaq.get("above_200ma", False) and 
                sp500.get("ma_trend") == "bullish"):
                regime = "bullish"
            elif (not sp500.get("above_200ma", True) and 
                  not nasdaq.get("above_200ma", True) and 
                  sp500.get("ma_trend") == "bearish"):
                regime = "bearish"
            elif (abs(sp500.get("daily_change_pct", 0)) < 0.3 and 
                  abs(nasdaq.get("daily_change_pct", 0)) < 0.3):
                regime = "sideways"
            elif (abs(sp500.get("daily_change_pct", 0)) > 1.5 or 
                  abs(nasdaq.get("daily_change_pct", 0)) > 1.5):
                regime = "volatile"
            else:
                regime = "mixed"
        else:
            regime = "unknown"
            
        result["market_regime"] = regime
        
        # Update cache
        _update_cache("market_indices", result)
        return result
    
    except Exception as e:
        logger.error(f"Error fetching market indices: {str(e)}")
        # Return fallback data
        return {
            "S&P 500": {
                "symbol": "^GSPC",
                "price": 4000.0,
                "daily_change_pct": 0.0,
                "monthly_change_pct": 0.0,
                "above_50ma": True,
                "above_200ma": True,
                "rsi": 50.0,
                "ma_trend": "bullish"
            },
            "market_regime": "mixed",
            "error": str(e)
        }


def get_sector_performance() -> Dict[str, Any]:
    """
    Get sector performance data to identify sector rotation and relative strength.
    
    Returns:
        Dictionary with sector performance metrics
    """
    # Check cache first
    if _is_cache_valid("sector_performance"):
        return _cache["sector_performance"]["data"]
    
    # Sector ETFs
    sectors = [
        {"symbol": "XLK", "name": "technology"},
        {"symbol": "XLF", "name": "financials"},
        {"symbol": "XLV", "name": "healthcare"},
        {"symbol": "XLE", "name": "energy"},
        {"symbol": "XLY", "name": "consumer_discretionary"},
        {"symbol": "XLP", "name": "consumer_staples"},
        {"symbol": "XLI", "name": "industrials"},
        {"symbol": "XLB", "name": "materials"},
        {"symbol": "XLU", "name": "utilities"},
        {"symbol": "XLRE", "name": "real_estate"}
    ]
    
    result = {"sectors": {}, "relative_strength": {}}
    
    try:
        # Fetch data for all sectors
        symbols = [s["symbol"] for s in sectors]
        data = yf.download(symbols, period="1mo", group_by="ticker")
        
        for sector in sectors:
            symbol = sector["symbol"]
            name = sector["name"]
            
            if symbol in data.columns.levels[0]:
                sector_data = data[symbol]
                
                # Calculate performance metrics
                current_price = sector_data["Close"][-1]
                start_price = sector_data["Close"][0]
                daily_change = ((current_price - sector_data["Close"][-2]) / sector_data["Close"][-2]) * 100
                monthly_change = ((current_price - start_price) / start_price) * 100
                
                # Calculate 5-day and 20-day moving averages
                ma5 = sector_data["Close"].rolling(window=5).mean().iloc[-1]
                ma20 = sector_data["Close"].rolling(window=20).mean().iloc[-1]
                
                result["sectors"][name] = {
                    "symbol": symbol,
                    "daily_change_pct": daily_change,
                    "monthly_change_pct": monthly_change,
                    "above_5ma": current_price > ma5,
                    "above_20ma": current_price > ma20,
                    "ma_trend": "bullish" if ma5 > ma20 else "bearish"
                }
                
                # Store monthly change for relative strength calculation
                result["relative_strength"][name] = monthly_change
        
        # Calculate relative strength ranking
        if result["relative_strength"]:
            # Sort sectors by monthly performance
            sorted_sectors = sorted(
                result["relative_strength"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Add sector rankings
            result["top_sectors"] = [s[0] for s in sorted_sectors[:3]]
            result["bottom_sectors"] = [s[0] for s in sorted_sectors[-3:]]
            
            # Determine sector rotation pattern
            if ("technology" in result["top_sectors"] and 
                "consumer_discretionary" in result["top_sectors"]):
                result["rotation_pattern"] = "risk_on"
            elif ("utilities" in result["top_sectors"] and 
                  "consumer_staples" in result["top_sectors"]):
                result["rotation_pattern"] = "risk_off"
            elif "energy" in result["top_sectors"]:
                result["rotation_pattern"] = "inflation_sensitive"
            else:
                result["rotation_pattern"] = "mixed"
        
        # Update cache
        _update_cache("sector_performance", result)
        return result
    
    except Exception as e:
        logger.error(f"Error fetching sector performance: {str(e)}")
        return {
            "sectors": {},
            "relative_strength": {},
            "top_sectors": ["technology", "healthcare", "financials"],
            "bottom_sectors": ["utilities", "real_estate", "energy"],
            "rotation_pattern": "mixed",
            "error": str(e)
        }


def get_market_news(max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Get relevant market news and sentiment from news APIs.
    
    Args:
        max_items: Maximum number of news items to return
        
    Returns:
        List of news items with headlines, sources, and sentiment
    """
    # Check cache first
    if _is_cache_valid("news"):
        cached_news = _cache["news"]["data"]
        return cached_news[:max_items] if cached_news else []
    
    news_items = []
    
    # Try Marketaux API first if key is available
    if MARKETAUX_API_KEY:
        try:
            url = f"https://api.marketaux.com/v1/news/all?api_token={MARKETAUX_API_KEY}&language=en&limit=10"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    news_items.append({
                        "headline": item.get("title"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "published_at": item.get("published_at"),
                        "sentiment": item.get("sentiment"),
                        "tickers": [entity.get("symbol") for entity in item.get("entities", []) 
                                   if entity.get("type") == "ticker"]
                    })
        except Exception as e:
            logger.error(f"Error fetching news from Marketaux: {str(e)}")
    
    # Fallback to News API if Marketaux failed or no key
    if not news_items and NEWS_API_KEY:
        try:
            # Get financial news
            url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get("articles", []):
                    # Basic sentiment analysis based on keywords
                    headline = item.get("title", "").lower()
                    sentiment = "neutral"
                    
                    positive_words = ["surge", "jump", "rise", "gain", "rally", "bull", "growth", "positive"]
                    negative_words = ["fall", "drop", "decline", "crash", "bear", "recession", "risk", "fear", "worry"]
                    
                    if any(word in headline for word in positive_words):
                        sentiment = "positive"
                    elif any(word in headline for word in negative_words):
                        sentiment = "negative"
                    
                    news_items.append({
                        "headline": item.get("title"),
                        "source": item.get("source", {}).get("name"),
                        "url": item.get("url"),
                        "published_at": item.get("publishedAt"),
                        "sentiment": sentiment,
                        "tickers": []  # News API doesn't provide ticker extraction
                    })
        except Exception as e:
            logger.error(f"Error fetching news from News API: {str(e)}")
    
    # If both APIs failed, provide mock data
    if not news_items:
        # Get current date
        today = datetime.now().strftime("%Y-%m-%d")
        
        news_items = [
            {
                "headline": "Markets react to Federal Reserve decision",
                "source": "Financial Times",
                "published_at": today,
                "sentiment": "neutral",
                "tickers": ["SPY"]
            },
            {
                "headline": "Tech stocks rally amid earnings beats",
                "source": "CNBC",
                "published_at": today,
                "sentiment": "positive",
                "tickers": ["QQQ", "AAPL", "MSFT"]
            },
            {
                "headline": "Oil prices drop on supply concerns",
                "source": "Reuters",
                "published_at": today,
                "sentiment": "negative",
                "tickers": ["USO", "XLE"]
            }
        ]
    
    # Update cache
    _update_cache("news", news_items)
    
    # Return limited number of items
    return news_items[:max_items]


def get_economic_data() -> Dict[str, Any]:
    """
    Get key economic indicators from Alpha Vantage or other sources.
    
    Returns:
        Dictionary with economic indicators
    """
    # Check cache first
    if _is_cache_valid("economic_data"):
        return _cache["economic_data"]["data"]
    
    result = {}
    
    # Only try to fetch if we have an API key
    if ALPHA_VANTAGE_API_KEY:
        try:
            # Get GDP growth data
            url = f"https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={ALPHA_VANTAGE_API_KEY}"
            gdp_resp = requests.get(url, timeout=10)
            
            if gdp_resp.status_code == 200:
                gdp_data = gdp_resp.json()
                gdp_values = gdp_data.get("data", [])
                if gdp_values:
                    recent_gdp = float(gdp_values[0].get("value", 0))
                    prev_gdp = float(gdp_values[1].get("value", 0)) if len(gdp_values) > 1 else recent_gdp
                    gdp_change = ((recent_gdp - prev_gdp) / prev_gdp) * 100
                    result["gdp_growth"] = gdp_change
            
            # Avoid hitting rate limits
            time.sleep(1)
            
            # Get CPI data (inflation)
            url = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={ALPHA_VANTAGE_API_KEY}"
            cpi_resp = requests.get(url, timeout=10)
            
            if cpi_resp.status_code == 200:
                cpi_data = cpi_resp.json()
                cpi_values = cpi_data.get("data", [])
                if cpi_values:
                    recent_cpi = float(cpi_values[0].get("value", 0))
                    prev_cpi = float(cpi_values[12].get("value", 0)) if len(cpi_values) > 12 else 0
                    if prev_cpi > 0:
                        inflation = ((recent_cpi - prev_cpi) / prev_cpi) * 100
                        result["inflation"] = inflation
            
            # Avoid hitting rate limits
            time.sleep(1)
            
            # Get unemployment rate
            url = f"https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={ALPHA_VANTAGE_API_KEY}"
            unemp_resp = requests.get(url, timeout=10)
            
            if unemp_resp.status_code == 200:
                unemp_data = unemp_resp.json()
                unemp_values = unemp_data.get("data", [])
                if unemp_values:
                    result["unemployment"] = float(unemp_values[0].get("value", 0))
        
        except Exception as e:
            logger.error(f"Error fetching economic data: {str(e)}")
    
    # If API failed or no key, use default values
    if not result:
        result = {
            "gdp_growth": 2.0,  # Default moderate growth
            "inflation": 3.0,    # Default moderate inflation
            "unemployment": 4.0  # Default moderate unemployment
        }
    
    # Add some classification
    if "inflation" in result:
        if result["inflation"] < 2.0:
            result["inflation_regime"] = "low"
        elif result["inflation"] < 5.0:
            result["inflation_regime"] = "moderate"
        else:
            result["inflation_regime"] = "high"
    
    # Add interest rate environment (could be fetched from FRED or other sources)
    result["interest_rate_environment"] = "tightening"  # Default for current environment
    
    # Update cache
    _update_cache("economic_data", result)
    return result


def calculate_trend_strength(data: pd.DataFrame, column: str = "Close") -> float:
    """
    Calculate the strength of a trend based on price action.
    
    Args:
        data: Price data DataFrame
        column: Column name to analyze
        
    Returns:
        Trend strength score (0-1 scale)
    """
    if len(data) < 20:
        return 0.5  # Default medium strength if not enough data
    
    try:
        # Calculate daily returns
        returns = data[column].pct_change().dropna()
        
        # Calculate directional movement
        pos_days = (returns > 0).sum()
        neg_days = (returns < 0).sum()
        total_days = pos_days + neg_days
        
        if total_days == 0:
            return 0.5
        
        # Directional bias (how many days are in the same direction)
        directional_bias = max(pos_days, neg_days) / total_days
        
        # Calculate linear regression to determine trend consistency
        x = np.arange(len(data))
        y = data[column].values
        slope, _, r_value, _, _ = linregress(x, y)
        
        # Combine metrics for final score
        # R-squared measures how well the data fits a linear trend
        r_squared = r_value ** 2
        
        # Normalize slope direction with the bias
        slope_norm = 1 if (slope > 0 and pos_days > neg_days) or (slope < 0 and pos_days < neg_days) else 0
        
        # Combine for final trend strength
        trend_strength = (directional_bias * 0.4) + (r_squared * 0.4) + (slope_norm * 0.2)
        
        return min(max(trend_strength, 0), 1)  # Ensure result is between 0 and 1
        
    except Exception as e:
        logger.warning(f"Error calculating trend strength: {str(e)}")
        return 0.5  # Default to medium strength on error


def get_current_market_context() -> Dict[str, Any]:
    """
    Get complete market context by combining all data sources.
    
    Returns:
        Dictionary with comprehensive market data
    """
    # Check if we need to import numpy and scipy for calculations
    try:
        import numpy as np
        from scipy.stats import linregress
    except ImportError:
        logger.warning("NumPy or SciPy not available, some calculations will be skipped")
    
    try:
        # Get data from all sources
        vix_data = get_vix_data()
        indices_data = get_market_indices()
        sector_data = get_sector_performance()
        news_data = get_market_news(max_items=5)
        economic_data = get_economic_data()
        
        # Get SPY data for trend strength calculation
        spy_data = None
        trend_strength = 0.5  # Default value
        
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="3mo")
            if 'numpy' in globals() and 'linregress' in globals():
                trend_strength = calculate_trend_strength(spy_data)
        except Exception as e:
            logger.warning(f"Error fetching SPY data: {str(e)}")
        
        # Combine all data into a single context object
        market_context = {
            "timestamp": datetime.now().isoformat(),
            "volatility_index": vix_data.get("current"),
            "volatility_regime": vix_data.get("regime"),
            "volatility_trend": vix_data.get("trend"),
            "market_regime": indices_data.get("market_regime"),
            "trend_strength": trend_strength,
            "market_indices": {
                k: v for k, v in indices_data.items() if k != "market_regime"
            },
            "sector_performance": sector_data.get("sectors", {}),
            "top_sectors": sector_data.get("top_sectors", []),
            "bottom_sectors": sector_data.get("bottom_sectors", []),
            "sector_rotation": sector_data.get("rotation_pattern"),
            "recent_news": news_data,
            "economic_indicators": economic_data
        }
        
        # Generate market summary
        summary_parts = []
        
        if "market_regime" in market_context:
            summary_parts.append(f"Market is in a {market_context['market_regime']} regime")
        
        if "volatility_regime" in market_context:
            summary_parts.append(f"with {market_context['volatility_regime'].replace('_', ' ')}")
        
        if "trend_strength" in market_context:
            strength_desc = "strong" if trend_strength > 0.7 else "moderate" if trend_strength > 0.4 else "weak"
            summary_parts.append(f"showing {strength_desc} trend strength")
        
        if "top_sectors" in market_context and market_context["top_sectors"]:
            top_sectors_str = ", ".join(market_context["top_sectors"])
            summary_parts.append(f"with {top_sectors_str} leading")
        
        market_context["market_summary"] = " ".join(summary_parts)
        
        return market_context
        
    except Exception as e:
        logger.error(f"Error compiling market context: {str(e)}")
        
        # Return a minimal fallback context
        return {
            "timestamp": datetime.now().isoformat(),
            "volatility_index": 20.0,
            "volatility_regime": "normal_volatility",
            "market_regime": "mixed",
            "trend_strength": 0.5,
            "error": str(e),
            "market_summary": "Market data unavailable, using default mixed regime assumptions"
        }


def get_mock_market_context(scenario: str = "default") -> Dict[str, Any]:
    """
    Generate a mock market context for testing when APIs are unavailable.
    
    Args:
        scenario: Market scenario to simulate (bullish, bearish, volatile, sideways)
        
    Returns:
        Simulated market context dictionary
    """
    scenarios = {
        "bullish": {
            "market_regime": "bullish",
            "volatility_index": 15.2,
            "volatility_regime": "low_volatility",
            "trend_strength": 0.75,
            "market_indices": {
                "S&P 500": {"daily_change_pct": 0.8, "above_200ma": True, "ma_trend": "bullish"},
                "Nasdaq 100": {"daily_change_pct": 1.2, "above_200ma": True, "ma_trend": "bullish"}
            },
            "sector_performance": {
                "technology": {"daily_change_pct": 1.5, "monthly_change_pct": 4.2},
                "financials": {"daily_change_pct": 0.7, "monthly_change_pct": 3.1},
                "healthcare": {"daily_change_pct": 0.3, "monthly_change_pct": 1.8}
            },
            "top_sectors": ["technology", "financials", "consumer_discretionary"],
            "sector_rotation": "risk_on",
            "recent_news": [
                {
                    "headline": "Fed signals continuation of accommodative policy",
                    "sentiment": "positive",
                    "published_at": datetime.now().strftime("%Y-%m-%d")
                }
            ],
            "economic_indicators": {
                "gdp_growth": 3.2,
                "inflation": 2.3,
                "inflation_regime": "moderate",
                "unemployment": 3.8
            }
        },
        "bearish": {
            "market_regime": "bearish",
            "volatility_index": 32.5,
            "volatility_regime": "high_volatility",
            "trend_strength": 0.65,
            "market_indices": {
                "S&P 500": {"daily_change_pct": -1.8, "above_200ma": False, "ma_trend": "bearish"},
                "Nasdaq 100": {"daily_change_pct": -2.5, "above_200ma": False, "ma_trend": "bearish"}
            },
            "sector_performance": {
                "technology": {"daily_change_pct": -3.2, "monthly_change_pct": -8.5},
                "financials": {"daily_change_pct": -1.5, "monthly_change_pct": -5.2},
                "utilities": {"daily_change_pct": 0.3, "monthly_change_pct": 1.2}
            },
            "top_sectors": ["utilities", "consumer_staples", "healthcare"],
            "sector_rotation": "risk_off",
            "recent_news": [
                {
                    "headline": "Recession fears grow as economic indicators weaken",
                    "sentiment": "negative",
                    "published_at": datetime.now().strftime("%Y-%m-%d")
                }
            ],
            "economic_indicators": {
                "gdp_growth": 0.5,
                "inflation": 3.8,
                "inflation_regime": "moderate",
                "unemployment": 5.2
            }
        },
        "volatile": {
            "market_regime": "volatile",
            "volatility_index": 28.5,
            "volatility_regime": "high_volatility",
            "trend_strength": 0.3,
            "market_indices": {
                "S&P 500": {"daily_change_pct": -1.5, "above_200ma": True, "ma_trend": "mixed"},
                "Nasdaq 100": {"daily_change_pct": -2.1, "above_200ma": False, "ma_trend": "mixed"}
            },
            "sector_performance": {
                "technology": {"daily_change_pct": -2.3, "monthly_change_pct": -4.5},
                "energy": {"daily_change_pct": 1.8, "monthly_change_pct": 6.2},
                "utilities": {"daily_change_pct": 0.5, "monthly_change_pct": 2.8}
            },
            "top_sectors": ["energy", "utilities", "materials"],
            "sector_rotation": "inflation_sensitive",
            "recent_news": [
                {
                    "headline": "Market volatility spikes amid mixed economic data",
                    "sentiment": "neutral",
                    "published_at": datetime.now().strftime("%Y-%m-%d")
                }
            ],
            "economic_indicators": {
                "gdp_growth": 1.8,
                "inflation": 4.5,
                "inflation_regime": "moderate",
                "unemployment": 4.2
            }
        },
        "sideways": {
            "market_regime": "sideways",
            "volatility_index": 18.5,
            "volatility_regime": "normal_volatility",
            "trend_strength": 0.2,
            "market_indices": {
                "S&P 500": {"daily_change_pct": 0.1, "above_200ma": True, "ma_trend": "mixed"},
                "Nasdaq 100": {"daily_change_pct": -0.2, "above_200ma": True, "ma_trend": "mixed"}
            },
            "sector_performance": {
                "technology": {"daily_change_pct": 0.3, "monthly_change_pct": 1.2},
                "financials": {"daily_change_pct": -0.2, "monthly_change_pct": -0.8},
                "healthcare": {"daily_change_pct": 0.5, "monthly_change_pct": 1.5}
            },
            "top_sectors": ["healthcare", "consumer_staples", "technology"],
            "sector_rotation": "mixed",
            "recent_news": [
                {
                    "headline": "Markets await direction as earnings season begins",
                    "sentiment": "neutral",
                    "published_at": datetime.now().strftime("%Y-%m-%d")
                }
            ],
            "economic_indicators": {
                "gdp_growth": 2.1,
                "inflation": 2.8,
                "inflation_regime": "moderate",
                "unemployment": 3.9
            }
        }
    }
    
    # Use requested scenario or default to mixed
    scenario_data = scenarios.get(scenario, scenarios.get("sideways"))
    
    # Add timestamp
    scenario_data["timestamp"] = datetime.now().isoformat()
    
    # Generate market summary
    summary_parts = []
    
    if "market_regime" in scenario_data:
        summary_parts.append(f"Market is in a {scenario_data['market_regime']} regime")
    
    if "volatility_regime" in scenario_data:
        summary_parts.append(f"with {scenario_data['volatility_regime'].replace('_', ' ')}")
    
    if "trend_strength" in scenario_data:
        strength_desc = "strong" if scenario_data["trend_strength"] > 0.7 else "moderate" if scenario_data["trend_strength"] > 0.4 else "weak"
        summary_parts.append(f"showing {strength_desc} trend strength")
    
    if "top_sectors" in scenario_data and scenario_data["top_sectors"]:
        top_sectors_str = ", ".join(scenario_data["top_sectors"])
        summary_parts.append(f"with {top_sectors_str} leading")
    
    scenario_data["market_summary"] = " ".join(summary_parts)
    
    # Mark this as mock data
    scenario_data["is_mock_data"] = True
    
    return scenario_data


# Testing function
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing Market Context Fetcher")
    
    try:
        # Test real data fetching
        print("\nFetching real market data...")
        market_context = get_current_market_context()
        
        print(f"Market Regime: {market_context.get('market_regime', 'Unknown')}")
        print(f"VIX: {market_context.get('volatility_index', 'Unknown')}")
        print(f"Top Sectors: {', '.join(market_context.get('top_sectors', ['Unknown']))}")
        print(f"Market Summary: {market_context.get('market_summary', 'Unknown')}")
        
        # Test mock data
        print("\nGenerating mock market data for testing...")
        for scenario in ["bullish", "bearish", "volatile", "sideways"]:
            mock_data = get_mock_market_context(scenario)
            print(f"\n{scenario.title()} Scenario:")
            print(f"Market Regime: {mock_data.get('market_regime')}")
            print(f"VIX: {mock_data.get('volatility_index')}")
            print(f"Market Summary: {mock_data.get('market_summary')}")
    
    except Exception as e:
        print(f"Error testing market context fetcher: {str(e)}")


class MarketContextFetcher:
    """
    MarketContextFetcher is responsible for collecting market data from various sources
    and creating a comprehensive market context for decision making.
    
    It can either fetch real data from APIs or generate mock data for testing.
    """
    
    def __init__(self, use_mock: bool = False, cache_duration: int = 60, mock_scenario: str = "default"):
        """
        Initialize the MarketContextFetcher.
        
        Args:
            use_mock: Whether to use mock data instead of real API calls
            cache_duration: How long to cache results in minutes
            mock_scenario: Market scenario to use when in mock mode (bullish, bearish, volatile, sideways)
        """
        self.use_mock = use_mock
        self.cache_duration = cache_duration
        self.mock_scenario = mock_scenario
        self.cached_data = {}
        self.cache_timestamp = {}
        
        logger.info(f"Initializing MarketContextFetcher (mock_mode: {use_mock}, mock_scenario: {mock_scenario})")
    
    def get_market_context(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get current market context including volatility indicators, market regime,
        sector performance, and other relevant data.
        
        Args:
            force_refresh: Whether to force a refresh of the data even if cache is valid
            
        Returns:
            Dictionary with market context information
        """
        cache_key = 'market_context'
        
        # Check if we have cached data and it's still valid
        if not force_refresh and cache_key in self.cached_data:
            cache_time = self.cache_timestamp.get(cache_key)
            if cache_time:
                cache_age = (datetime.now() - cache_time).total_seconds() / 60
                if cache_age < self.cache_duration:
                    logger.debug(f"Using cached market context ({cache_age:.1f} min old)")
                    return self.cached_data[cache_key]
        
        # If we're using mock data or real data fetch fails, use mock
        try:
            if self.use_mock:
                market_context = self._get_mock_market_context()
            else:
                # Try to get real market data
                market_context = self._fetch_real_market_context()
                
                # If real data fetch fails, fall back to mock
                if not market_context:
                    logger.warning("Failed to fetch real market data, falling back to mock data")
                    market_context = self._get_mock_market_context()
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            market_context = self._get_mock_market_context()
        
        # Cache the result
        self.cached_data[cache_key] = market_context
        self.cache_timestamp[cache_key] = datetime.now()
        
        return market_context
    
    def _fetch_real_market_context(self) -> Dict[str, Any]:
        """
        Fetch real market data from APIs.
        
        Returns:
            Dictionary with market context
        """
        # This would include API calls to various data sources
        # For now, just return empty dict to indicate it's not implemented
        logger.warning("Real data fetching not implemented, using mock data")
        return {}
    
    def _get_mock_market_context(self) -> Dict[str, Any]:
        """
        Generate mock market context for testing.
        
        Returns:
            Dictionary with mock market context
        """
        # If a specific scenario was requested, use that
        if self.mock_scenario != "default":
            return get_mock_market_context(self.mock_scenario)
            
        # Otherwise generate a random market context for testing
        vix = round(random.uniform(12, 40), 2)
        
        # Determine market regime based on VIX
        if vix < 15:
            regime = 'bullish'
        elif vix < 20:
            regime = 'sideways'
        elif vix < 30:
            regime = 'bearish'
        else:
            regime = 'volatile'
        
        # Generate mock sector performance
        sectors = [
            'technology', 'financials', 'energy', 'healthcare', 
            'consumer_discretionary', 'consumer_staples', 'utilities', 
            'materials', 'industrials', 'real_estate', 'communication_services'
        ]
        
        sector_performance = {}
        
        # Generate sector performance aligned with market regime
        if regime == 'bullish':
            # In bullish regimes, most sectors do well, especially growth sectors
            for sector in sectors:
                if sector in ['technology', 'consumer_discretionary', 'communication_services']:
                    sector_performance[sector] = round(random.uniform(1, 5), 2)
                else:
                    sector_performance[sector] = round(random.uniform(-1, 3), 2)
        
        elif regime == 'bearish':
            # In bearish regimes, defensive sectors do better
            for sector in sectors:
                if sector in ['utilities', 'consumer_staples', 'healthcare']:
                    sector_performance[sector] = round(random.uniform(-1, 2), 2)
                else:
                    sector_performance[sector] = round(random.uniform(-4, 0), 2)
        
        elif regime == 'volatile':
            # In volatile regimes, performance is mixed with wider swings
            for sector in sectors:
                sector_performance[sector] = round(random.uniform(-6, 6), 2)
        
        else:  # sideways
            # In sideways markets, performance is muted
            for sector in sectors:
                sector_performance[sector] = round(random.uniform(-2, 2), 2)
        
        # Add some economic indicators
        economic_indicators = {
            'gdp_growth': round(random.uniform(-1, 4), 1),
            'unemployment': round(random.uniform(3, 8), 1),
            'inflation': round(random.uniform(1.5, 8), 1),
            'fed_funds_rate': round(random.uniform(0, 5), 2)
        }
        
        # Create trend strength indicator
        trend_strength = {
            'short_term': round(random.uniform(-1, 1), 2),  # -1 to 1 scale
            'medium_term': round(random.uniform(-1, 1), 2),
            'long_term': round(random.uniform(-1, 1), 2)
        }
        
        # Create market summary
        market_summary = self._generate_market_summary(
            regime=regime, 
            vix=vix, 
            trend_strength=trend_strength, 
            sector_performance=sector_performance,
            economic_indicators=economic_indicators
        )
        
        # Build the complete market context
        market_context = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'vix': vix,
            'regime': regime,
            'sector_performance': sector_performance,
            'economic_indicators': economic_indicators,
            'trend_strength': trend_strength,
            'market_summary': market_summary,
            'is_mock': True
        }
        
        return market_context
    
    def _generate_market_summary(self, regime: str, vix: float, 
                                trend_strength: Dict[str, float],
                                sector_performance: Dict[str, float],
                                economic_indicators: Dict[str, float]) -> str:
        """
        Generate a textual summary of market conditions.
        
        Args:
            regime: Market regime (bullish, bearish, volatile, sideways)
            vix: VIX value
            trend_strength: Dictionary with trend strength indicators
            sector_performance: Dictionary with sector performance data
            economic_indicators: Dictionary with economic indicators
            
        Returns:
            String with market summary
        """
        # Create market regime description
        if regime == 'bullish':
            regime_desc = "Market is in a bullish regime with strong upward momentum."
        elif regime == 'bearish':
            regime_desc = "Market is in a bearish regime with downward pressure."
        elif regime == 'volatile':
            regime_desc = "Market is in a volatile regime with high uncertainty."
        else:  # sideways
            regime_desc = "Market is in a sideways regime with lack of clear direction."
        
        # Describe volatility
        if vix < 15:
            vol_desc = f"Volatility is very low (VIX: {vix})."
        elif vix < 20:
            vol_desc = f"Volatility is normal (VIX: {vix})."
        elif vix < 30:
            vol_desc = f"Volatility is elevated (VIX: {vix})."
        else:
            vol_desc = f"Volatility is high (VIX: {vix})."
        
        # Describe trend strength
        avg_trend = sum(trend_strength.values()) / len(trend_strength)
        if avg_trend > 0.5:
            trend_desc = "Trend strength is strong."
        elif avg_trend > 0:
            trend_desc = "Trend strength is moderate."
        elif avg_trend > -0.5:
            trend_desc = "Trend strength is weak."
        else:
            trend_desc = "Trend strength is very weak."
        
        # Find top and bottom sectors
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [f"{sector} ({perf:+.1f}%)" for sector, perf in sorted_sectors[:3]]
        bottom_sectors = [f"{sector} ({perf:+.1f}%)" for sector, perf in sorted_sectors[-3:]]
        
        sector_desc = f"Top performing sectors: {', '.join(top_sectors)}. "
        sector_desc += f"Worst performing sectors: {', '.join(bottom_sectors)}."
        
        # Describe economic conditions
        if economic_indicators['gdp_growth'] > 2.5:
            econ_desc = "Economic growth is strong."
        elif economic_indicators['gdp_growth'] > 0:
            econ_desc = "Economic growth is moderate."
        else:
            econ_desc = "Economic growth is weak or contracting."
        
        if economic_indicators['inflation'] > 5:
            econ_desc += f" Inflation is high at {economic_indicators['inflation']}%."
        elif economic_indicators['inflation'] > 2.5:
            econ_desc += f" Inflation is moderate at {economic_indicators['inflation']}%."
        else:
            econ_desc += f" Inflation is low at {economic_indicators['inflation']}%."
        
        # Compile the summary
        summary = f"{regime_desc} {vol_desc} {trend_desc} {sector_desc} {econ_desc}"
        
        # Add note about mock data
        summary += " (Note: This is mock data for testing purposes.)"
        
        return summary 

    def get_market_summary(self, market_context: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the current market context.
        
        Args:
            market_context: The market context dictionary
            
        Returns:
            String with a formatted market summary
        """
        # If the context already has a summary, return it
        if "market_summary" in market_context:
            return market_context["market_summary"]
            
        # Otherwise, generate a summary based on the context
        summary_parts = []
        
        # Add market regime
        if "regime" in market_context:
            summary_parts.append(f"Market is in a {market_context['regime']} regime")
        elif "market_regime" in market_context:
            summary_parts.append(f"Market is in a {market_context['market_regime']} regime")
        
        # Add VIX information
        if "vix" in market_context:
            summary_parts.append(f"with VIX at {market_context['vix']:.2f}")
        elif "volatility_index" in market_context:
            summary_parts.append(f"with VIX at {market_context['volatility_index']:.2f}")
            
        # Add volatility regime if available
        if "volatility_regime" in market_context:
            vol_regime = market_context["volatility_regime"].replace("_", " ")
            summary_parts.append(f"({vol_regime})")
            
        # Add trend strength if available
        if "trend_strength" in market_context:
            if isinstance(market_context["trend_strength"], dict):
                # If it's a dictionary with different timeframes
                medium = market_context["trend_strength"].get("medium_term", 0)
                strength_desc = "strong" if medium > 0.7 else "moderate" if medium > 0.4 else "weak"
                summary_parts.append(f"showing {strength_desc} trend strength")
            else:
                # If it's a single value
                strength = market_context["trend_strength"]
                strength_desc = "strong" if strength > 0.7 else "moderate" if strength > 0.4 else "weak"
                summary_parts.append(f"showing {strength_desc} trend strength")
                
        # Add top sectors if available
        if "top_sectors" in market_context and market_context["top_sectors"]:
            top_sectors_str = ", ".join(market_context["top_sectors"])
            summary_parts.append(f"with {top_sectors_str} leading")
            
        # Return the formatted summary
        return " ".join(summary_parts) 