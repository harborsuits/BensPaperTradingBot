#!/usr/bin/env python3
"""
Alpaca Market Data Adapter

This module implements a specific adapter for retrieving market data from Alpaca.
Alpaca provides excellent historical data through their REST API with no rate limits for paper accounts.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import aiohttp
import requests

from trading_bot.data_sources.market_data_adapter import (
    MarketDataAdapter,
    MarketDataConfig,
    DataSourceType,
    TimeFrame,
    OHLCV,
    MarketIndicator,
    MarketNewsItem
)

class AlpacaAdapter(MarketDataAdapter):
    """Implementation of MarketDataAdapter for Alpaca"""

    def __init__(self, config: MarketDataConfig):
        if not config.base_url:
            config.base_url = "https://data.alpaca.markets"  # Historical data endpoint
        super().__init__(config)

        # Alpaca credentials
        self.api_key = "PKYBHCCT1DIMGZX6P64A"
        self.api_secret = "ssidJ2cJU0EGBOhdHrXJd7HegoaPaAMQqs0AU2PO"

        # Headers for API requests
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }

    async def get_price_data(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        limit: int = 100,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> List[OHLCV]:
        """Retrieve OHLCV data from Alpaca"""

        # Map our timeframes to Alpaca's timeframe parameter
        timeframe_map = {
            TimeFrame.MINUTE_1: "1Min",
            TimeFrame.MINUTE_5: "5Min",
            TimeFrame.MINUTE_15: "15Min",
            TimeFrame.MINUTE_30: "30Min",
            TimeFrame.HOUR_1: "1Hour",
            TimeFrame.DAY_1: "1Day",
            TimeFrame.WEEK_1: "1Week",
            TimeFrame.MONTH_1: "1Month"
        }

        tf = timeframe_map.get(timeframe, "1Day")

        # Set default dates if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            if timeframe == TimeFrame.DAY_1:
                start_time = end_time - timedelta(days=limit)
            else:
                start_time = end_time - timedelta(days=30)

        # Format dates for Alpaca API
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(start_time, datetime) else start_time
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(end_time, datetime) else end_time

        # Build API URL
        url = f"{self.config.base_url}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": tf,
            "start": start_str,
            "end": end_str,
            "limit": min(limit, 10000),  # Alpaca max limit
            "adjustment": "raw",
            "feed": "iex"  # Use IEX for free data
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        bars = data.get("bars", [])

                        # Convert to OHLCV format
                        ohlcv_data = []
                        for bar in bars:
                            ohlcv = OHLCV(
                                timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                                open=float(bar["o"]),
                                high=float(bar["h"]),
                                low=float(bar["l"]),
                                close=float(bar["c"]),
                                volume=int(bar["v"]),
                                symbol=symbol
                            )
                            ohlcv_data.append(ohlcv)

                        return ohlcv_data
                    else:
                        logger.error(f"Alpaca API error: {response.status} - {await response.text()}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
            return []

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # Get latest 1-minute bar
            latest_data = await self.get_price_data(symbol, TimeFrame.MINUTE_1, limit=1)
            if latest_data:
                return latest_data[0].close
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = await self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

# Synchronous version for easier use in scripts
def get_alpaca_data_sync(symbol: str, start_date: str, end_date: str, timeframe: str = "1Day") -> List[Dict]:
    """Synchronous wrapper for getting Alpaca data"""

    api_key = "PKYBHCCT1DIMGZX6P64A"
    api_secret = "ssidJ2cJU0EGBOhdHrXJd7HegoaPaAMQqs0AU2PO"

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        bars = []
        for bar in data.get("bars", []):
            bars.append({
                "Date": bar["t"][:10],  # YYYY-MM-DD
                "Open": float(bar["o"]),
                "High": float(bar["h"]),
                "Low": float(bar["l"]),
                "Close": float(bar["c"]),
                "Volume": int(bar["v"])
            })

        return bars

    except Exception as e:
        print(f"Error fetching Alpaca data: {e}")
        return []
