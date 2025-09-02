"""
Market Data Utilities

This module provides functions for analyzing and processing market data,
including implied volatility, technical indicators, and event detection.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available. Some functions may not work properly.")
    # Create a minimal placeholder for np
    class MinimalNumpy:
        def random(self): return {'random': lambda: 0.5}
        def sqrt(self, x): return x ** 0.5
    np = MinimalNumpy()

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas not available. Some functions may not work properly.")
    # Create a minimal placeholder for pd
    class MinimalPandas:
        class Series:
            pass
        class DataFrame:
            pass
    pd = MinimalPandas()

def calculate_iv_rank(price_data: pd.DataFrame) -> float:
    """
    Calculate the Implied Volatility Rank (IV Rank) based on historical data.
    
    IV Rank measures where the current implied volatility stands in relation to
    its historical range, expressed as a percentage.
    
    Args:
        price_data: DataFrame containing historical price data with IV column
                    or sufficient OHLC data to estimate historical volatility
                    
    Returns:
        IV Rank as a float between 0 and 100
    """
    # If we have an 'iv' column, use it directly
    if 'iv' in price_data.columns:
        iv_data = price_data['iv'].dropna()
        if len(iv_data) < 30:
            return 50.0  # Default to middle value if insufficient data
            
        current_iv = iv_data.iloc[-1]
        min_iv = iv_data.min()
        max_iv = iv_data.max()
        
        # Avoid division by zero
        if max_iv == min_iv:
            return 50.0
            
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        
    else:
        # Calculate historical volatility from close prices
        # as a proxy for IV when actual IV is not available
        if 'close' not in price_data.columns or len(price_data) < 30:
            return 50.0
            
        # Calculate 30-day rolling volatility
        returns = price_data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized
        
        if len(rolling_vol) < 30:
            return 50.0
            
        current_vol = rolling_vol.iloc[-1]
        min_vol = rolling_vol.min()
        max_vol = rolling_vol.max()
        
        # Avoid division by zero
        if max_vol == min_vol:
            return 50.0
            
        iv_rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
    
    # Ensure the result is between 0 and 100
    return max(0, min(100, iv_rank))

def get_upcoming_events(symbol: str) -> List[Dict[str, Any]]:
    """
    Get a list of upcoming events for a given symbol.
    
    Events include earnings announcements, dividend payments, stock splits,
    and other corporate actions that might affect trading decisions.
    
    Args:
        symbol: The ticker symbol to fetch events for
        
    Returns:
        List of dictionaries containing event details:
        - date: Event date
        - event_type: Type of event (earnings, dividend, etc.)
        - description: Event details
        - importance: Numeric indicator of event importance (1-10)
    """
    # This function would normally fetch data from an API or database
    # For now, we'll return a placeholder empty list
    # In a production environment, this would connect to a data provider
    
    # Placeholder logic to create sample events for testing
    events = []
    
    # Get current date for reference
    today = datetime.now().date()
    
    # Example logic to generate placeholder events
    # In practice, this would come from an external API or database
    if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
        # Generate a sample earnings event 14-21 days in the future
        earnings_date = today + timedelta(days=14 + (hash(symbol) % 7))
        events.append({
            "date": earnings_date,
            "event_type": "earnings",
            "description": f"Q{((earnings_date.month-1)//3)+1} Earnings Announcement",
            "importance": 8
        })
        
        # Generate a sample dividend event if applicable
        if symbol in ["AAPL", "MSFT"]:
            dividend_date = today + timedelta(days=7 + (hash(symbol) % 14))
            events.append({
                "date": dividend_date,
                "event_type": "dividend",
                "description": "Quarterly Dividend Payment",
                "importance": 5
            })
    
    return events

def detect_price_gaps(price_data: pd.DataFrame, min_gap_percent: float = 2.0) -> List[Dict[str, Any]]:
    """
    Detect significant price gaps in historical data.
    
    Gaps occur when the opening price differs significantly from the prior day's
    closing price, indicating potential volatility or major news events.
    
    Args:
        price_data: DataFrame with OHLC data
        min_gap_percent: Minimum gap size as percentage to be considered significant
        
    Returns:
        List of dictionaries containing gap details:
        - date: Date when the gap occurred
        - gap_percent: Size of the gap as a percentage
        - direction: 'up' or 'down'
    """
    if len(price_data) < 2 or 'open' not in price_data.columns or 'close' not in price_data.columns:
        return []
        
    gaps = []
    
    # Calculate gap percentage (open price vs previous close)
    prev_close = price_data['close'].shift(1)
    gap_percent = (price_data['open'] - prev_close) / prev_close * 100
    
    # Find gaps exceeding the threshold
    for date, pct in zip(price_data.index[1:], gap_percent[1:]):
        if abs(pct) >= min_gap_percent:
            gaps.append({
                "date": date.date() if isinstance(date, pd.Timestamp) else date,
                "gap_percent": pct,
                "direction": "up" if pct > 0 else "down"
            })
    
    return gaps

def get_market_sentiment(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Get overall market sentiment based on multiple indicators.
    
    Analyzes technical indicators, breadth metrics, and volatility
    to determine the current market sentiment.
    
    Args:
        symbols: Optional list of symbols to focus the analysis on
        
    Returns:
        Dictionary containing sentiment metrics:
        - sentiment: Overall sentiment ('bullish', 'bearish', 'neutral')
        - strength: Strength of the sentiment (0-100)
        - indicators: Dictionary of individual indicator readings
    """
    # In a real implementation, this would analyze market data
    # For now, return a neutral placeholder
    
    return {
        "sentiment": "neutral",
        "strength": 50,
        "indicators": {
            "advance_decline": 0.0,
            "vix": 20.0,
            "put_call_ratio": 1.0,
            "net_new_highs": 0
        }
    }

def check_recent_gaps(price_data: pd.DataFrame, days: int = 10, threshold: float = 2.0) -> List[Dict[str, Any]]:
    """
    Check for recent price gaps within the specified lookback period.
    
    A convenience wrapper around detect_price_gaps that focuses on recent gaps.
    
    Args:
        price_data: DataFrame with OHLC data
        days: Number of recent days to check for gaps
        threshold: Minimum gap percentage to report
        
    Returns:
        List of recent gaps meeting the threshold criteria
    """
    if len(price_data) <= days:
        return detect_price_gaps(price_data, threshold)
    
    # Get only recent data
    recent_data = price_data.iloc[-days-1:]  # Include one extra day for gap calculation
    
    return detect_price_gaps(recent_data, threshold)

def calculate_historical_volatility(
    price_data: Any,
    window: int = 20,
    annualize: bool = True
) -> Any:
    """
    Calculate historical volatility from price data.
    
    Args:
        price_data: DataFrame with price data
        window: Rolling window for volatility calculation (default: 20 days)
        annualize: Whether to annualize the result (default: True)
        
    Returns:
        Series with historical volatility
    """
    try:
        # Check if pandas is available
        if not hasattr(pd, 'DataFrame') or not isinstance(pd.DataFrame, type):
            logger.warning("Pandas not available for historical volatility calculation")
            return pd.Series() if hasattr(pd, 'Series') else None
            
        # Check if we have close prices
        if not hasattr(price_data, 'columns'):
            logger.warning("Input is not a proper DataFrame")
            return pd.Series()
            
        if 'close' not in price_data.columns:
            if 'Close' in price_data.columns:
                price_data['close'] = price_data['Close']
            else:
                raise ValueError("No close price column found in data")
        
        # Calculate log returns
        log_returns = np.log(price_data['close'] / price_data['close'].shift(1))
        
        # Calculate rolling standard deviation
        if hasattr(log_returns, 'rolling'):
            hist_vol = log_returns.rolling(window=window).std()
            
            # Annualize if requested (multiply by sqrt of trading days in a year)
            if annualize and hasattr(np, 'sqrt'):
                hist_vol = hist_vol * np.sqrt(252)
                
            return hist_vol
        else:
            logger.warning("Rolling calculation not available")
            return pd.Series()
        
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {e}")
        return pd.Series() if hasattr(pd, 'Series') else None 