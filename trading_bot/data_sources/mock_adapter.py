"""
Mock Market Data Adapter

This module provides a mock implementation of the MarketDataAdapter interface
for testing purposes when real API keys aren't available.
"""

import logging
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from trading_bot.data_sources.market_data_adapter import (
    MarketDataAdapter, 
    MarketDataConfig,
    DataSourceType,
    TimeFrame,
    OHLCV,
    MarketIndicator,
    MarketNewsItem
)

logger = logging.getLogger("data_sources.mock")

class MockMarketDataAdapter(MarketDataAdapter):
    """Mock implementation of MarketDataAdapter for testing"""
    
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.logger = logging.getLogger("mock_adapter")
        self.logger.info("Initialized mock market data adapter")
        
        # Cache for price data
        self._price_cache = {}
        
        # Some realistic starting prices for major symbols
        self._base_prices = {
            "SPY": 450.0,
            "QQQ": 380.0,
            "IWM": 200.0,
            "AAPL": 175.0,
            "MSFT": 340.0,
            "GOOGL": 135.0,
            "AMZN": 180.0,
            "META": 300.0,
            "NVDA": 430.0,
            "TSLA": 240.0,
        }
        
        # Default price for unknown symbols
        self._default_price = 100.0
    
    async def get_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame = TimeFrame.DAY_1,
        limit: int = 100,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> List[OHLCV]:
        """Generate mock OHLCV data for testing"""
        # Check cache first
        cache_key = f"{symbol}:{timeframe}:{limit}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        # Get base price for symbol or use default
        base_price = self._base_prices.get(symbol.upper(), self._default_price)
        
        # Generate realistic looking price data
        candles = []
        now = datetime.now()
        
        # Determine time interval based on timeframe
        if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.MINUTE_30]:
            interval = timedelta(minutes=timeframe.value)
        elif timeframe == TimeFrame.HOUR_1:
            interval = timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_4:
            interval = timedelta(hours=4)
        elif timeframe == TimeFrame.DAY_1:
            interval = timedelta(days=1)
        elif timeframe == TimeFrame.WEEK_1:
            interval = timedelta(weeks=1)
        else:
            interval = timedelta(days=30)  # Month
        
        # Start with base price and work backwards
        current_price = base_price
        timestamp = int(now.timestamp())
        
        # Add small random walk with momentum
        trend = random.choice([-1, 1]) * random.uniform(0.0001, 0.0005)  # Small trend bias
        volatility = random.uniform(0.005, 0.015)  # Base volatility
        
        for i in range(limit):
            # Adjust price with random walk plus trend
            price_change = random.normalvariate(trend, volatility) * current_price
            current_price += price_change
            
            # Make sure price stays positive
            current_price = max(current_price, current_price * 0.1)
            
            # Generate OHLC prices around the current price
            daily_volatility = current_price * volatility * 2
            open_price = current_price - price_change/2 + random.uniform(-daily_volatility, daily_volatility)
            high_price = max(open_price, current_price) + random.uniform(0, daily_volatility)
            low_price = min(open_price, current_price) - random.uniform(0, daily_volatility)
            close_price = current_price
            
            # Generate volume with some randomness
            base_volume = 1000000 if current_price > 100 else 10000000  # Higher volume for lower priced stocks
            volume = int(base_volume * (1 + random.uniform(-0.5, 1.0)))
            
            # Create candle
            candle = OHLCV(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                symbol=symbol
            )
            candles.append(candle)
            
            # Move backward in time
            timestamp -= int(interval.total_seconds())
            
            # Occasionally change trend direction
            if random.random() < 0.1:  # 10% chance to change trend
                trend = random.choice([-1, 1]) * random.uniform(0.0001, 0.0005)
                
            # Occasionally spike volatility
            if random.random() < 0.05:  # 5% chance of volatility spike
                volatility = random.uniform(0.01, 0.03)
            else:
                volatility = random.uniform(0.005, 0.015)
        
        # Reverse so most recent is first
        candles.reverse()
        
        # Cache the result
        self._price_cache[cache_key] = candles
        
        return candles
    
    async def get_price_history(
        self, 
        symbol: str, 
        timeframe: TimeFrame = TimeFrame.DAY_1,
        limit: int = 100,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> List[OHLCV]:
        """Alias for get_price_data for compatibility"""
        return await self.get_price_data(symbol, timeframe, limit, start_time, end_time)
    
    async def get_indicator(
        self,
        symbol: str,
        indicator: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        params: Dict[str, Any] = None
    ) -> List[MarketIndicator]:
        """Generate mock indicator data"""
        # Get price data first
        prices = await self.get_price_data(symbol, timeframe, limit=100)
        
        # Create indicator values based on the indicator type
        indicators = []
        
        if indicator.lower() in ['rsi', 'relative_strength_index']:
            # Generate RSI values between 30 and 70 with some realistic behavior
            baseline = 50.0  # Neutral RSI
            values = []
            
            # Use price trend to influence RSI
            for i in range(min(30, len(prices))):
                if i < len(prices) - 1:
                    # Positive price change pushes RSI up, negative pushes down
                    price_change = (prices[i].close - prices[i+1].close) / prices[i+1].close
                    rsi_change = price_change * 100  # Scale to RSI range
                    rsi_value = baseline + rsi_change + random.uniform(-5, 5)
                    baseline = max(10, min(90, rsi_value))  # Keep within bounds but allow trends
                else:
                    rsi_value = baseline + random.uniform(-5, 5)
                
                # Ensure within 0-100 range
                rsi_value = max(0, min(100, rsi_value))
                values.append(rsi_value)
            
        elif indicator.lower() in ['macd', 'macd_line']:
            # Generate MACD line values
            baseline = 0.0
            values = []
            
            for i in range(min(30, len(prices))):
                if i < len(prices) - 1:
                    # MACD tends to follow price momentum
                    price_change = (prices[i].close - prices[i+1].close) / prices[i+1].close
                    macd_change = price_change * prices[i].close * 0.1  # Scale to reasonable MACD range
                    macd_value = baseline + macd_change + random.uniform(-0.2, 0.2)
                    baseline = macd_value  # Allow trending
                else:
                    macd_value = baseline + random.uniform(-0.2, 0.2)
                
                values.append(macd_value)
        
        elif indicator.lower() in ['macd_signal']:
            # Generate MACD signal line - smoother version of MACD
            # First get mock MACD values
            macd_indicators = await self.get_indicator(symbol, 'macd', timeframe)
            macd_values = [ind.value for ind in macd_indicators]
            
            # Calculate a simple moving average of MACD to simulate signal line
            window_size = 9  # Traditional signal line is 9-day EMA
            values = []
            
            for i in range(len(macd_values)):
                if i < window_size - 1:
                    # Not enough data for full window, use available data
                    signal = sum(macd_values[:i+1]) / (i+1)
                else:
                    # Full window
                    signal = sum(macd_values[i-window_size+1:i+1]) / window_size
                
                values.append(signal)
        
        elif indicator.lower() in ['macd_hist', 'macd_histogram']:
            # MACD histogram is the difference between MACD and signal line
            macd_indicators = await self.get_indicator(symbol, 'macd', timeframe)
            signal_indicators = await self.get_indicator(symbol, 'macd_signal', timeframe)
            
            macd_values = [ind.value for ind in macd_indicators]
            signal_values = [ind.value for ind in signal_indicators]
            
            # Calculate histogram
            values = []
            for i in range(min(len(macd_values), len(signal_values))):
                hist = macd_values[i] - signal_values[i]
                values.append(hist)
        
        elif indicator.lower() in ['atr', 'average_true_range']:
            # Calculate ATR based on price data
            values = []
            
            for i in range(min(30, len(prices))):
                if i == 0:
                    # First candle, just use high-low
                    atr = prices[i].high - prices[i].low
                else:
                    # Calculate true range
                    high_low = prices[i].high - prices[i].low
                    high_close = abs(prices[i].high - prices[i-1].close)
                    low_close = abs(prices[i].low - prices[i-1].close)
                    true_range = max(high_low, high_close, low_close)
                    
                    # Simple 14-period average
                    if i < 14:
                        atr = (atr * (i - 1) + true_range) / i
                    else:
                        atr = (atr * 13 + true_range) / 14
                
                values.append(atr)
        
        else:
            # Generic indicator: just return some random values around zero
            values = [random.uniform(-2, 2) for _ in range(min(30, len(prices)))]
        
        # Create the indicator objects
        for i, value in enumerate(values):
            timestamp = prices[i].timestamp if i < len(prices) else int(datetime.now().timestamp()) - i * 86400
            
            indicators.append(MarketIndicator(
                name=indicator,
                value=value,
                timestamp=timestamp,
                symbol=symbol,
                timeframe=timeframe
            ))
        
        return indicators
    
    async def get_latest_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 20,
        categories: Optional[List[str]] = None
    ) -> List[MarketNewsItem]:
        """Generate mock news items"""
        news_items = []
        now = datetime.now()
        
        # Sample headlines and sources for realistic mock data
        headlines = [
            "Market Rallies on Strong Economic Data",
            "Fed Signals Future Rate Path",
            "Earnings Beat Expectations for Q2",
            "Inflation Data Shows Mixed Signals",
            "Tech Sector Leads Market Higher",
            "Oil Prices Fall on Supply Concerns", 
            "Retail Sales Data Shows Consumer Strength",
            "Housing Market Shows Signs of Cooling",
            "Job Market Remains Resilient",
            "Manufacturing Data Indicates Expansion",
            "Market Volatility Increases Amid Uncertainty",
            "Global Markets React to Policy Announcements",
            "Semiconductor Stocks Surge on AI Demand",
            "Treasury Yields Rise on Economic Outlook",
            "M&A Activity Picks Up in Technology Sector"
        ]
        
        sources = [
            "Bloomberg", "CNBC", "Reuters", "Financial Times", 
            "Wall Street Journal", "MarketWatch", "Barron's",
            "Yahoo Finance", "Investor's Business Daily"
        ]
        
        # Generate news specific to requested symbols if provided
        if symbols and len(symbols) > 0:
            for symbol in symbols:
                # Company-specific headlines
                company_headlines = [
                    f"{symbol} Reports Strong Quarterly Results",
                    f"{symbol} Announces New Product Launch",
                    f"{symbol} Shares Rise on Analyst Upgrade",
                    f"{symbol} Expands into New Markets",
                    f"CEO of {symbol} Discusses Future Growth Strategy",
                    f"{symbol} Partners with Industry Leader",
                    f"Institutional Investors Increase Stake in {symbol}",
                    f"{symbol} Addresses Supply Chain Challenges",
                    f"Analyst Raises Price Target on {symbol}"
                ]
                
                # Add some company-specific news
                for i in range(min(5, limit // len(symbols))):
                    headline = random.choice(company_headlines)
                    summary = f"Latest news about {symbol}: {headline.lower()} according to industry experts. The company continues to focus on strategic initiatives and shareholder value."
                    
                    # Randomly assign sentiment
                    if "Rise" in headline or "Strong" in headline or "Upgrade" in headline:
                        sentiment = random.uniform(0.3, 0.8)
                    elif "Challenges" in headline or "Concerns" in headline:
                        sentiment = random.uniform(-0.7, -0.2)
                    else:
                        sentiment = random.uniform(-0.3, 0.5)
                    
                    news_items.append(MarketNewsItem(
                        id=f"mock-{symbol}-{i}",
                        title=headline,
                        summary=summary,
                        url=f"https://example.com/news/{symbol.lower()}-{i}",
                        source=random.choice(sources),
                        image_url=f"https://placehold.co/600x400?text={symbol}+News",
                        published_at=(now - timedelta(hours=random.randint(1, 48))).isoformat(),
                        related_symbols=[symbol],
                        sentiment_score=sentiment
                    ))
        else:
            # General market news
            for i in range(limit):
                headline = random.choice(headlines)
                summary = f"Market update: {headline.lower()} according to recent reports. Analysts are monitoring the situation closely for potential market impact."
                
                # Randomly assign sentiment
                if "Rallies" in headline or "Higher" in headline or "Strong" in headline or "Beat" in headline:
                    sentiment = random.uniform(0.3, 0.8)
                elif "Fall" in headline or "Concerns" in headline or "Cooling" in headline or "Uncertainty" in headline:
                    sentiment = random.uniform(-0.7, -0.2)
                else:
                    sentiment = random.uniform(-0.3, 0.5)
                
                # Randomly assign related symbols (major market ETFs and large caps)
                related = []
                if random.random() < 0.7:  # 70% chance to include related symbols
                    possible_symbols = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
                    related = random.sample(possible_symbols, random.randint(1, 3))
                
                news_items.append(MarketNewsItem(
                    id=f"mock-general-{i}",
                    title=headline,
                    summary=summary,
                    url=f"https://example.com/news/market-{i}",
                    source=random.choice(sources),
                    image_url=f"https://placehold.co/600x400?text=Market+News+{i}",
                    published_at=(now - timedelta(hours=random.randint(1, 72))).isoformat(),
                    related_symbols=related,
                    sentiment_score=sentiment
                ))
        
        # Sort by recency
        news_items.sort(key=lambda x: x.published_at, reverse=True)
        
        return news_items[:limit]
    
    async def get_market_breadth(self) -> Dict[str, Any]:
        """Generate mock market breadth data"""
        total_issues = 500  # S&P 500 proxy
        
        # Randomize with a slight bullish bias (since markets trend up over time)
        if random.random() < 0.6:  # 60% chance of positive breadth
            advancing = random.randint(int(total_issues * 0.55), int(total_issues * 0.8))
        else:
            advancing = random.randint(int(total_issues * 0.2), int(total_issues * 0.45))
        
        declining = total_issues - advancing
        
        # Add other breadth metrics
        new_highs = random.randint(5, 50)
        new_lows = random.randint(1, 30)
        
        # Create a breadth composite score
        breadth_score = (advancing/total_issues) * 2 - 1  # -1 to 1 scale
        
        return {
            "advancers": advancing,
            "decliners": declining, 
            "unchanged": 0,
            "total": total_issues,
            "new_highs": new_highs,
            "new_lows": new_lows,
            "advance_decline_ratio": advancing / declining if declining > 0 else advancing,
            "breadth_score": breadth_score
        }
    
    async def get_sector_performance(self) -> Dict[str, Any]:
        """Generate mock sector performance data"""
        sectors = [
            "Technology", "Financial", "Healthcare", "Consumer Discretionary",
            "Consumer Staples", "Energy", "Materials", "Industrials",
            "Utilities", "Communication Services", "Real Estate"
        ]
        
        sector_data = {}
        
        # Base performance anchored around a market return
        market_return = random.uniform(-0.5, 1.5)  # Slight bullish bias
        
        for sector in sectors:
            # Sector return relative to market with some randomness
            beta = {
                "Technology": 1.2,
                "Financial": 1.1,
                "Healthcare": 0.8,
                "Consumer Discretionary": 1.1,
                "Consumer Staples": 0.6,
                "Energy": 1.3,
                "Materials": 1.0,
                "Industrials": 1.1,
                "Utilities": 0.5,
                "Communication Services": 0.9,
                "Real Estate": 0.7
            }.get(sector, 1.0)
            
            # Calculate return with beta and random noise
            sector_return = market_return * beta + random.uniform(-1.0, 1.0)
            
            sector_data[sector] = {
                "return_daily": sector_return,
                "return_weekly": sector_return * 2 + random.uniform(-2.0, 2.0),
                "return_monthly": sector_return * 4 + random.uniform(-4.0, 4.0),
                "relative_strength": (sector_return - market_return) * 100,
                "volume_change": random.uniform(-10, 20)
            }
        
        return {
            "sectors": sector_data,
            "market_return": market_return,
            "timestamp": int(datetime.now().timestamp())
        }
