"""
Finnhub Market Data Adapter

This module implements a specific adapter for retrieving market data from Finnhub.
Finnhub provides real-time market data and financial news through their REST API.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import aiohttp

from trading_bot.data_sources.market_data_adapter import (
    MarketDataAdapter, 
    MarketDataConfig,
    DataSourceType,
    TimeFrame,
    OHLCV,
    MarketIndicator,
    MarketNewsItem
)

class FinnhubAdapter(MarketDataAdapter):
    """Implementation of MarketDataAdapter for Finnhub"""
    
    def __init__(self, config: MarketDataConfig):
        if not config.base_url:
            config.base_url = "https://finnhub.io/api/v1"
        super().__init__(config)
    
    async def get_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame = TimeFrame.DAY_1,
        limit: int = 100,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> List[OHLCV]:
        """Retrieve OHLCV data from Finnhub"""
        # Map our timeframes to Finnhub's resolution parameter
        resolution_map = {
            TimeFrame.MINUTE_1: "1",
            TimeFrame.MINUTE_5: "5",
            TimeFrame.MINUTE_15: "15",
            TimeFrame.MINUTE_30: "30",
            TimeFrame.HOUR_1: "60",
            TimeFrame.HOUR_4: "D",
            TimeFrame.DAY_1: "D",
            TimeFrame.WEEK_1: "W",
            TimeFrame.MONTH_1: "M",
        }
        
        resolution = resolution_map.get(timeframe, "D")
        
        # Convert time parameters to UNIX timestamps
        if not end_time:
            end_time = datetime.now()
        elif isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
        if not start_time:
            # Default to limit data points based on timeframe
            if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5]:
                start_time = end_time - timedelta(days=1)
            elif timeframe in [TimeFrame.MINUTE_15, TimeFrame.MINUTE_30, TimeFrame.HOUR_1]:
                start_time = end_time - timedelta(days=7)
            elif timeframe == TimeFrame.HOUR_4:
                start_time = end_time - timedelta(days=30)
            elif timeframe == TimeFrame.DAY_1:
                start_time = end_time - timedelta(days=90)
            else:
                start_time = end_time - timedelta(days=365)
        elif isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        # Convert to UNIX timestamps for Finnhub API
        from_timestamp = int(start_time.timestamp())
        to_timestamp = int(end_time.timestamp())
        
        # Make request to Finnhub candles endpoint
        endpoint = "stock/candle"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_timestamp,
            "to": to_timestamp
        }
        
        response = await self._make_request(endpoint, params, request_id=f"candles_{symbol}")
        
        if not response or response.get("s") != "ok":
            self.logger.error(f"Failed to retrieve price data for {symbol}: {response}")
            return []
        
        # Parse the response into our OHLCV model
        candles = []
        timestamps = response.get("t", [])
        opens = response.get("o", [])
        highs = response.get("h", [])
        lows = response.get("l", [])
        closes = response.get("c", [])
        volumes = response.get("v", [])
        
        for i in range(min(len(timestamps), limit)):
            candles.append(OHLCV(
                timestamp=timestamps[i],
                open=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=volumes[i],
                symbol=symbol
            ))
        
        return candles
    
    async def get_indicators(
        self,
        symbol: str,
        indicators: List[str],
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> Dict[str, List[MarketIndicator]]:
        """Retrieve technical indicators from Finnhub"""
        result = {}
        
        # Map indicator names to Finnhub parameters
        indicator_map = {
            "rsi": {"indicator": "rsi", "timeperiod": 14},
            "macd": {"indicator": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9},
            "bbands": {"indicator": "bbands", "timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
            "sma": {"indicator": "sma", "timeperiod": 20},
            "ema": {"indicator": "ema", "timeperiod": 20}
        }
        
        # Resolve timeframe
        resolution_map = {
            TimeFrame.MINUTE_1: "1",
            TimeFrame.MINUTE_5: "5",
            TimeFrame.MINUTE_15: "15",
            TimeFrame.MINUTE_30: "30",
            TimeFrame.HOUR_1: "60",
            TimeFrame.HOUR_4: "D",
            TimeFrame.DAY_1: "D",
            TimeFrame.WEEK_1: "W",
            TimeFrame.MONTH_1: "M",
        }
        resolution = resolution_map.get(timeframe, "D")
        
        # Get price data first (needed for indicators)
        price_data = await self.get_price_data(symbol, timeframe, limit=60)
        if not price_data:
            self.logger.error(f"Failed to retrieve price data for indicators: {symbol}")
            return result
        
        # Calculate end time from most recent candle
        end_time = price_data[0].datetime if price_data else datetime.now()
        
        # Retrieve each requested indicator
        for indicator_name in indicators:
            if indicator_name not in indicator_map:
                self.logger.warning(f"Unsupported indicator: {indicator_name}")
                continue
                
            indicator_params = indicator_map[indicator_name].copy()
            
            # Make request for this indicator
            endpoint = "indicator"
            params = {
                "symbol": symbol,
                "resolution": resolution,
                **indicator_params
            }
            
            response = await self._make_request(endpoint, params, request_id=f"indicator_{symbol}_{indicator_name}")
            
            if not response or response.get("s") != "ok":
                self.logger.error(f"Failed to retrieve indicator {indicator_name} for {symbol}: {response}")
                continue
            
            # Parse indicator data
            indicator_values = []
            
            # Handle different indicator formats
            if indicator_name == "macd":
                # MACD has three output series
                timestamps = response.get("t", [])
                macd_values = response.get("macd", [])
                signal_values = response.get("signal", [])
                hist_values = response.get("histogram", [])
                
                for i in range(len(timestamps)):
                    indicator_values.append(MarketIndicator(
                        name=f"{indicator_name}_line",
                        value=macd_values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
                    indicator_values.append(MarketIndicator(
                        name=f"{indicator_name}_signal",
                        value=signal_values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
                    indicator_values.append(MarketIndicator(
                        name=f"{indicator_name}_hist",
                        value=hist_values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
            elif indicator_name == "bbands":
                # Bollinger bands have three output series
                timestamps = response.get("t", [])
                upper_values = response.get("upperband", [])
                middle_values = response.get("middleband", [])
                lower_values = response.get("lowerband", [])
                
                for i in range(len(timestamps)):
                    indicator_values.append(MarketIndicator(
                        name=f"{indicator_name}_upper",
                        value=upper_values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
                    indicator_values.append(MarketIndicator(
                        name=f"{indicator_name}_middle",
                        value=middle_values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
                    indicator_values.append(MarketIndicator(
                        name=f"{indicator_name}_lower",
                        value=lower_values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
            else:
                # Most indicators have a single output series
                timestamps = response.get("t", [])
                values = response.get(indicator_name, [])
                
                for i in range(len(timestamps)):
                    indicator_values.append(MarketIndicator(
                        name=indicator_name,
                        value=values[i],
                        timestamp=timestamps[i],
                        symbol=symbol,
                        timeframe=timeframe
                    ))
            
            result[indicator_name] = indicator_values
        
        return result
    
    async def get_latest_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 20,
        categories: Optional[List[str]] = None
    ) -> List[MarketNewsItem]:
        """Retrieve latest market news from Finnhub"""
        news_items = []
        
        if symbols and len(symbols) > 0:
            # If specific symbols are requested, get company news
            for symbol in symbols[:5]:  # Limit to 5 symbols to avoid too many requests
                endpoint = f"company-news"
                # Get news from the last 7 days
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                to_date = datetime.now().strftime("%Y-%m-%d")
                
                params = {
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date
                }
                
                response = await self._make_request(endpoint, params, request_id=f"news_{symbol}")
                
                if not response:
                    continue
                
                # Process company news
                for item in response[:limit]:
                    try:
                        news_items.append(MarketNewsItem(
                            id=f"finnhub-{item.get('id', hash(item.get('headline', '') + str(item.get('datetime', 0))))}",
                            title=item.get("headline", ""),
                            summary=item.get("summary", ""),
                            url=item.get("url", ""),
                            source=item.get("source", "Finnhub"),
                            image_url=item.get("image", None),
                            published_at=datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                            related_symbols=[symbol],
                            sentiment_score=None  # Finnhub doesn't provide sentiment in the basic API
                        ))
                    except Exception as e:
                        self.logger.error(f"Error parsing news item: {str(e)}")
        else:
            # Get general market news
            endpoint = "news"
            params = {
                "category": "general"
            }
            
            if categories and len(categories) > 0:
                # Finnhub only supports one category at a time
                params["category"] = categories[0]
            
            response = await self._make_request(endpoint, params, request_id="market_news")
            
            if not response:
                return []
            
            # Process market news
            for item in response[:limit]:
                try:
                    # Extract related symbols if available
                    related = item.get("related", "").split(",")
                    related_symbols = [sym.strip() for sym in related if sym.strip()]
                    
                    news_items.append(MarketNewsItem(
                        id=f"finnhub-{item.get('id', hash(item.get('headline', '') + str(item.get('datetime', 0))))}",
                        title=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        url=item.get("url", ""),
                        source=item.get("source", "Finnhub"),
                        image_url=item.get("image", None),
                        published_at=datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                        related_symbols=related_symbols,
                        sentiment_score=None
                    ))
                except Exception as e:
                    self.logger.error(f"Error parsing news item: {str(e)}")
        
        return news_items
