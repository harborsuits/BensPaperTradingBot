#!/usr/bin/env python3
"""
Market Data Provider

This module provides access to market data (both historical and live) for
the strategy system, including prices, technical indicators, and market metrics.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)

class MarketDataProvider:
    """
    Provides access to market data for the trading system.
    
    This class serves as a unified interface for accessing different types of
    market data, including price data, technical indicators, and market metrics.
    It handles caching to minimize redundant API calls and provides fallbacks
    for missing data.
    """
    
    def __init__(self, cache_dir: str = 'data/market_data_cache'):
        """
        Initialize the market data provider.
        
        Args:
            cache_dir: Directory for caching market data
        """
        self.cache_dir = cache_dir
        self.data_cache = {}
        self.trading_days_cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_market_data(self, date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Get comprehensive market data for a specific date.
        
        Args:
            date: Date to get data for (None for current/latest date)
            
        Returns:
            Dictionary with market data
        """
        # Use current date if none specified
        target_date = date or datetime.now().date()
        
        # Check cache first
        cache_key = target_date.isoformat()
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
            
        # Check if cached file exists
        cache_file = os.path.join(self.cache_dir, f"market_data_{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                self.data_cache[cache_key] = cached_data
                return cached_data.copy()
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
        
        # Fetch new data
        logger.info(f"Fetching market data for {target_date}")
        
        try:
            # Get market data components
            price_data = self._get_price_data(target_date)
            indicator_data = self._calculate_indicators(price_data)
            market_metrics = self._get_market_metrics(target_date)
            
            # Combine data
            market_data = {
                **price_data,
                **indicator_data,
                **market_metrics,
                "date": cache_key
            }
            
            # Cache the data
            self.data_cache[cache_key] = market_data
            
            # Save to cache file
            try:
                with open(cache_file, 'w') as f:
                    json.dump(market_data, f)
            except Exception as e:
                logger.error(f"Error saving data to cache: {e}")
                
            return market_data.copy()
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {
                "date": cache_key,
                "error": str(e)
            }
    
    def _get_price_data(self, date: datetime.date) -> Dict[str, float]:
        """
        Get price data for major indices and ETFs.
        
        Args:
            date: Target date
            
        Returns:
            Dictionary with price data
        """
        # Define symbols to fetch
        symbols = ["SPY", "QQQ", "IWM", "VIX", "GLD", "TLT"]
        
        # For historical data, use date range ending on target date
        end_date = date
        start_date = end_date - timedelta(days=252)  # Include 1 year of data for calculation
        
        # For current date, use latest data
        if date >= datetime.now().date():
            end_date = datetime.now().date()
            
        try:
            # Fetch data from Yahoo Finance
            data = {}
            
            # Get data for each symbol
            for symbol in symbols:
                # Fetch historical data
                df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
                
                if df.empty:
                    logger.warning(f"No data for {symbol} on {date}")
                    continue
                
                # Get the row for the target date or the last available date
                if date in df.index:
                    row = df.loc[date]
                else:
                    # Get the most recent data on or before the target date
                    df = df[df.index <= pd.Timestamp(date)]
                    if df.empty:
                        continue
                    row = df.iloc[-1]
                
                # Extract relevant fields
                data[f"{symbol.lower()}_close"] = row["Close"]
                data[f"{symbol.lower()}_open"] = row["Open"]
                data[f"{symbol.lower()}_high"] = row["High"]
                data[f"{symbol.lower()}_low"] = row["Low"]
                data[f"{symbol.lower()}_volume"] = row["Volume"]
                
                # Calculate additional metrics
                if len(df) >= 20:
                    data[f"{symbol.lower()}_sma20"] = df["Close"].rolling(20).mean().iloc[-1]
                if len(df) >= 50:
                    data[f"{symbol.lower()}_sma50"] = df["Close"].rolling(50).mean().iloc[-1]
                if len(df) >= 200:
                    data[f"{symbol.lower()}_sma200"] = df["Close"].rolling(200).mean().iloc[-1]
            
            # Map some key fields to standard names
            if "spy_close" in data:
                data["spy_close"] = data["spy_close"]
            if "spy_sma20" in data:
                data["spy_sma20"] = data["spy_sma20"]
            if "spy_sma50" in data:
                data["spy_sma50"] = data["spy_sma50"]
            if "vix_close" in data:
                data["vix"] = data["vix_close"]
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return {}
    
    def _calculate_indicators(self, price_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate technical indicators from price data.
        
        Args:
            price_data: Dictionary with price data
            
        Returns:
            Dictionary with technical indicators
        """
        indicators = {}
        
        try:
            # ATR (Average True Range) - simplified calculation
            # Normally would use a more sophisticated implementation
            if all(k in price_data for k in ["spy_high", "spy_low", "spy_close"]):
                true_range = price_data["spy_high"] - price_data["spy_low"]
                indicators["atr"] = true_range
                
            # ADX (Average Directional Index) - would normally need more data
            # Using a placeholder for demo
            indicators["adx"] = 25.0
            
            # RSI - simplified placeholder
            indicators["rsi"] = 50.0
            
            # MACD - simplified placeholder
            indicators["macd"] = 0.0
            if "spy_close" in price_data and "spy_sma50" in price_data:
                indicators["macd"] = price_data["spy_close"] - price_data["spy_sma50"]
                
            # Calculate volume ratio (current vs 20-day average)
            # In a real implementation, would need historical volume data
            indicators["volume_ratio"] = 1.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _get_market_metrics(self, date: datetime.date) -> Dict[str, Any]:
        """
        Get additional market metrics like breadth, sentiment, etc.
        
        Args:
            date: Target date
            
        Returns:
            Dictionary with market metrics
        """
        # In a real implementation, these would be fetched from various sources
        # Using placeholders for demo
        
        metrics = {
            "advance_decline_ratio": 1.5,  # Placeholder
            "sector_deviation": 3.5,       # Placeholder
            "market_sentiment": "neutral"  # Placeholder
        }
        
        # Add sector performance data (placeholders)
        metrics["sector_performance"] = {
            "Technology": 0.5,
            "Healthcare": 0.2,
            "Financials": -0.3,
            "Energy": 1.0,
            "Materials": 0.1,
            "Utilities": -0.2,
            "Consumer Discretionary": 0.4,
            "Consumer Staples": 0.0,
            "Industrials": 0.3,
            "Communication Services": 0.2
        }
        
        # Add sector leaders/laggards
        sorted_sectors = sorted(metrics["sector_performance"].items(), key=lambda x: x[1], reverse=True)
        metrics["sector_leaders"] = [sector for sector, _ in sorted_sectors[:3]]
        metrics["sector_laggards"] = [sector for sector, _ in sorted_sectors[-3:]]
        
        # Add recent market performance
        metrics["recent_performance"] = "Up 2.3% over past week"  # Placeholder
        
        return metrics
    
    def get_trading_days(self, start_date: datetime.date, 
                       end_date: datetime.date) -> List[datetime.date]:
        """
        Get list of trading days in a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        # Check cache
        cache_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
        if cache_key in self.trading_days_cache:
            return self.trading_days_cache[cache_key].copy()
        
        try:
            # Fetch SPY data to get trading days
            df = yf.download("SPY", start=start_date, end=end_date + timedelta(days=1), progress=False)
            trading_days = [d.date() for d in df.index]
            
            # Cache the result
            self.trading_days_cache[cache_key] = trading_days
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Error fetching trading days: {e}")
            
            # Fallback: generate weekdays and filter out common holidays
            # This is a simplified approach - not accurate for all markets/holidays
            all_days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # 0-4 are Monday to Friday
                    all_days.append(current)
                current += timedelta(days=1)
                
            return all_days
    
    def get_historical_data(self, symbols: List[str], start_date: datetime.date, 
                          end_date: datetime.date) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        result = {}
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
                if not df.empty:
                    result[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                
        return result
    
    def get_current_market_context(self) -> Dict[str, Any]:
        """
        Get comprehensive current market context, including regime, trends, and key metrics.
        
        Returns:
            Dictionary with current market context
        """
        # Get latest market data
        market_data = self.get_market_data()
        
        # For demonstration purposes, adding additional context
        # In a real system, this would include more sophisticated analysis
        
        # Get 1-week and 1-month historical data for SPY
        end_date = datetime.now().date()
        week_start = end_date - timedelta(days=7)
        month_start = end_date - timedelta(days=30)
        
        # Calculate short-term and medium-term trends
        try:
            spy_week = yf.download("SPY", start=week_start, end=end_date + timedelta(days=1), progress=False)
            spy_month = yf.download("SPY", start=month_start, end=end_date + timedelta(days=1), progress=False)
            
            if not spy_week.empty and len(spy_week) > 1:
                week_change = (spy_week["Close"].iloc[-1] / spy_week["Close"].iloc[0] - 1) * 100
                market_data["spy_week_change"] = week_change
                
            if not spy_month.empty and len(spy_month) > 1:
                month_change = (spy_month["Close"].iloc[-1] / spy_month["Close"].iloc[0] - 1) * 100
                market_data["spy_month_change"] = month_change
                
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
        
        # Add comparison of different market segments (large cap vs small cap)
        try:
            market_data["large_vs_small"] = (
                market_data.get("spy_close", 0) / market_data.get("spy_sma20", 1) - 
                market_data.get("iwm_close", 0) / market_data.get("iwm_sma20", 1)
            ) * 100
        except Exception:
            market_data["large_vs_small"] = 0
            
        # Add bond market context
        try:
            market_data["bond_trend"] = "rising" if market_data.get("tlt_close", 0) > market_data.get("tlt_sma50", 0) else "falling"
        except Exception:
            market_data["bond_trend"] = "unknown"
            
        # Add gold trend as proxy for inflation/fear
        try:
            market_data["gold_trend"] = "rising" if market_data.get("gld_close", 0) > market_data.get("gld_sma50", 0) else "falling"
        except Exception:
            market_data["gold_trend"] = "unknown"
            
        return market_data
        
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.data_cache.clear()
        self.trading_days_cache.clear()
        logger.info("Cleared market data cache")
    
    def update_cache(self, days: int = 30) -> None:
        """
        Update cache for recent days.
        
        Args:
            days: Number of recent days to update
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        trading_days = self.get_trading_days(start_date, end_date)
        
        for date in trading_days:
            self.get_market_data(date)
            
        logger.info(f"Updated market data cache for {len(trading_days)} trading days")


class MockMarketDataProvider(MarketDataProvider):
    """
    Mock version of the market data provider for testing and development.
    
    Provides predefined market data for different scenarios without making
    external API calls.
    """
    
    def __init__(self, scenario: str = "neutral"):
        """
        Initialize the mock market data provider.
        
        Args:
            scenario: Market scenario to simulate ("bullish", "bearish", "volatile", "neutral")
        """
        super().__init__(cache_dir="data/mock_market_data")
        self.scenario = scenario
        
    def get_market_data(self, date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Get mock market data for the specified scenario.
        
        Args:
            date: Date (ignored in mock provider)
            
        Returns:
            Dictionary with mock market data
        """
        # Create base data
        data = {
            "date": date.isoformat() if date else datetime.now().date().isoformat(),
            "spy_close": 450.0,
            "spy_open": 448.0,
            "spy_high": 452.0,
            "spy_low": 447.0,
            "spy_volume": 100000000,
            "spy_sma20": 445.0,
            "spy_sma50": 440.0,
            "spy_sma200": 430.0,
            "vix": 18.0,
            "atr": 5.0,
            "adx": 25.0,
            "rsi": 55.0,
            "macd": 2.0,
            "volume_ratio": 1.1,
            "advance_decline_ratio": 1.5,
            "sector_deviation": 3.5,
            "market_sentiment": "neutral"
        }
        
        # Modify data based on scenario
        if self.scenario == "bullish":
            data.update({
                "spy_close": 460.0,
                "spy_sma20": 450.0,
                "spy_sma50": 440.0,
                "vix": 14.0,
                "adx": 35.0,
                "rsi": 68.0,
                "macd": 5.0,
                "advance_decline_ratio": 2.5,
                "market_sentiment": "bullish"
            })
            
        elif self.scenario == "bearish":
            data.update({
                "spy_close": 430.0,
                "spy_sma20": 440.0,
                "spy_sma50": 450.0,
                "vix": 28.0,
                "adx": 30.0,
                "rsi": 32.0,
                "macd": -5.0,
                "advance_decline_ratio": 0.6,
                "market_sentiment": "bearish"
            })
            
        elif self.scenario == "volatile":
            data.update({
                "spy_close": 445.0,
                "spy_sma20": 445.0,
                "spy_sma50": 445.0,
                "vix": 35.0,
                "atr": 15.0,
                "adx": 20.0,
                "rsi": 45.0,
                "macd": 0.5,
                "volume_ratio": 2.0,
                "advance_decline_ratio": 1.0,
                "sector_deviation": 8.0,
                "market_sentiment": "fear"
            })
            
        # Add sector performance based on scenario
        if self.scenario == "bullish":
            data["sector_performance"] = {
                "Technology": 2.5,
                "Healthcare": 1.5,
                "Financials": 2.0,
                "Energy": 1.0,
                "Materials": 1.2,
                "Utilities": 0.2,
                "Consumer Discretionary": 1.8,
                "Consumer Staples": 0.5,
                "Industrials": 1.7,
                "Communication Services": 2.2
            }
        elif self.scenario == "bearish":
            data["sector_performance"] = {
                "Technology": -2.0,
                "Healthcare": -0.5,
                "Financials": -2.5,
                "Energy": -1.5,
                "Materials": -1.8,
                "Utilities": -0.3,
                "Consumer Discretionary": -2.2,
                "Consumer Staples": -0.2,
                "Industrials": -1.7,
                "Communication Services": -2.0
            }
        elif self.scenario == "volatile":
            data["sector_performance"] = {
                "Technology": -3.0,
                "Healthcare": 1.5,
                "Financials": -2.0,
                "Energy": 2.5,
                "Materials": 1.8,
                "Utilities": 1.2,
                "Consumer Discretionary": -2.5,
                "Consumer Staples": 1.0,
                "Industrials": -1.5,
                "Communication Services": -2.2
            }
        else:  # neutral
            data["sector_performance"] = {
                "Technology": 0.5,
                "Healthcare": 0.2,
                "Financials": -0.3,
                "Energy": 0.4,
                "Materials": 0.1,
                "Utilities": -0.1,
                "Consumer Discretionary": 0.3,
                "Consumer Staples": 0.0,
                "Industrials": 0.2,
                "Communication Services": -0.2
            }
            
        # Add sector leaders/laggards
        sorted_sectors = sorted(data["sector_performance"].items(), key=lambda x: x[1], reverse=True)
        data["sector_leaders"] = [sector for sector, _ in sorted_sectors[:3]]
        data["sector_laggards"] = [sector for sector, _ in sorted_sectors[-3:]]
        
        # Add recent performance description
        if self.scenario == "bullish":
            data["recent_performance"] = "Up 3.5% over past week"
        elif self.scenario == "bearish":
            data["recent_performance"] = "Down 4.2% over past week"
        elif self.scenario == "volatile":
            data["recent_performance"] = "Flat with daily swings of 2%"
        else:
            data["recent_performance"] = "Up 0.5% over past week"
            
        return data
        
    def get_trading_days(self, start_date: datetime.date, 
                       end_date: datetime.date) -> List[datetime.date]:
        """
        Get mock list of trading days in a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days (weekdays in this mock implementation)
        """
        trading_days = []
        current = start_date
        
        while current <= end_date:
            if current.weekday() < 5:  # 0-4 are Monday to Friday
                trading_days.append(current)
            current += timedelta(days=1)
            
        return trading_days
    
    def set_scenario(self, scenario: str) -> None:
        """
        Set the market scenario to simulate.
        
        Args:
            scenario: Market scenario ("bullish", "bearish", "volatile", "neutral")
        """
        valid_scenarios = ["bullish", "bearish", "volatile", "neutral"]
        if scenario not in valid_scenarios:
            logger.warning(f"Invalid scenario: {scenario}. Using 'neutral' instead.")
            scenario = "neutral"
            
        self.scenario = scenario
        logger.info(f"Set market scenario to: {scenario}")
        
    def get_current_market_context(self) -> Dict[str, Any]:
        """Get comprehensive current market context for the current scenario."""
        data = self.get_market_data()
        
        # Add additional context based on scenario
        if self.scenario == "bullish":
            data.update({
                "spy_week_change": 3.5,
                "spy_month_change": 8.2,
                "large_vs_small": 1.5,  # Large caps outperforming
                "bond_trend": "falling",
                "gold_trend": "neutral"
            })
        elif self.scenario == "bearish":
            data.update({
                "spy_week_change": -4.2,
                "spy_month_change": -12.5,
                "large_vs_small": -0.8,  # Small caps outperforming (less bad)
                "bond_trend": "rising",
                "gold_trend": "rising"
            })
        elif self.scenario == "volatile":
            data.update({
                "spy_week_change": 0.5,
                "spy_month_change": -5.5,
                "large_vs_small": 0.2,
                "bond_trend": "volatile",
                "gold_trend": "rising"
            })
        else:  # neutral
            data.update({
                "spy_week_change": 0.5,
                "spy_month_change": 1.8,
                "large_vs_small": 0.1,
                "bond_trend": "neutral",
                "gold_trend": "neutral"
            })
            
        # The mock data already includes primary_regime and traits 
        # from the MarketRegimeClassifier in a real implementation
            
        return data 