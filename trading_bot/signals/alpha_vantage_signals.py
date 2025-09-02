#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage Technical Signals

This module implements technical analysis-based trading signals using 
the Alpha Vantage API data.
"""

import logging
import time
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Union, Optional, Any

from trading_bot.market.market_data import MarketData
from trading_bot.signals.base_signal import BaseSignal

logger = logging.getLogger(__name__)

class AlphaVantageTechnicalSignals(BaseSignal):
    """Technical analysis signals generated from Alpha Vantage API data."""
    
    # Default configuration for signals
    DEFAULT_CONFIG = {
        "sma_short": 20,
        "sma_medium": 50,
        "sma_long": 200,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bbands_period": 20,
        "bbands_std": 2.0,
        "adx_period": 14,
        "adx_threshold": 25,
        "atr_period": 14,
        "volume_ma_period": 20,
        "max_api_calls_per_minute": 5,  # Alpha Vantage free tier limit
        "api_call_pause_seconds": 12    # Pause between API calls
    }
    
    def __init__(
        self, 
        market_data: MarketData, 
        api_key: str, 
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Alpha Vantage signals generator.
        
        Args:
            market_data: Market data instance for retrieving historical data
            api_key: Alpha Vantage API key
            config: Optional configuration parameters
        """
        super().__init__(market_data)
        self.api_key = api_key
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.last_api_call = 0
        logger.info("Initialized Alpha Vantage Technical Signals")
    
    def _throttle_api_call(self):
        """Throttle API calls to respect Alpha Vantage rate limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.config["api_call_pause_seconds"]:
            sleep_time = self.config["api_call_pause_seconds"] - time_since_last_call
            logger.debug(f"Throttling API call, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def fetch_daily_data(self, symbol: str, full: bool = True) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            full: Whether to fetch full or compact data
            
        Returns:
            DataFrame with OHLCV data or None if the request fails
        """
        self._throttle_api_call()
        
        output_size = "full" if full else "compact"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={self.api_key}"
        
        try:
            logger.info(f"Fetching daily data for {symbol}")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logger.error(f"Failed to get daily data for {symbol}: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                return None
            
            time_series = data["Time Series (Daily)"]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = [col.split(". ")[1] for col in df.columns]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Rename to standard OHLCV format
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "adjusted close": "Adj Close"
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return None
    
    def fetch_indicator(self, symbol: str, indicator: str, **params) -> Optional[pd.DataFrame]:
        """Fetch technical indicator data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            indicator: Indicator function name as used by Alpha Vantage
            **params: Additional parameters for the indicator
            
        Returns:
            DataFrame with indicator data or None if the request fails
        """
        self._throttle_api_call()
        
        # Build URL with parameters
        param_str = ''.join(f"&{k}={v}" for k, v in params.items())
        url = f"https://www.alphavantage.co/query?function={indicator}&symbol={symbol}{param_str}&apikey={self.api_key}"
        
        try:
            logger.info(f"Fetching {indicator} data for {symbol}")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Check for errors
            if "Technical Analysis" not in data and "Error Message" in data:
                logger.error(f"Failed to get {indicator} data for {symbol}: {data['Error Message']}")
                return None
            
            # Different indicators have different response formats
            if "Technical Analysis" in data:
                # Most technical indicators
                key = next(iter(data["Technical Analysis"].keys()), None)
                metadata = data["Meta Data"]
                time_series = data["Technical Analysis"]
                
                # Create DataFrame
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                return df
            
            # Fallback for other formats
            logger.warning(f"Unexpected response format for {indicator} data for {symbol}")
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching {indicator} data for {symbol}: {str(e)}")
            return None
    
    def analyze_price_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend using multiple moving averages.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with trend analysis results
        """
        if df is None or df.empty:
            return {"trend": "unknown", "strength": 0}
        
        # Calculate moving averages
        sma_short = df["Close"].rolling(window=self.config["sma_short"]).mean()
        sma_medium = df["Close"].rolling(window=self.config["sma_medium"]).mean()
        sma_long = df["Close"].rolling(window=self.config["sma_long"]).mean()
        
        # Get latest values
        latest_close = df["Close"].iloc[-1]
        latest_sma_short = sma_short.iloc[-1] if not pd.isna(sma_short.iloc[-1]) else 0
        latest_sma_medium = sma_medium.iloc[-1] if not pd.isna(sma_medium.iloc[-1]) else 0
        latest_sma_long = sma_long.iloc[-1] if not pd.isna(sma_long.iloc[-1]) else 0
        
        # Determine trend
        trend = "neutral"
        strength = 0
        
        # Primary trend (using long-term MA)
        if latest_close > latest_sma_long:
            trend = "bullish"
            strength += 1
        elif latest_close < latest_sma_long:
            trend = "bearish"
            strength -= 1
        
        # Medium-term trend
        if latest_close > latest_sma_medium:
            if trend == "bullish":
                strength += 1
            else:
                trend = "neutral"
        elif latest_close < latest_sma_medium:
            if trend == "bearish":
                strength -= 1
            else:
                trend = "neutral"
        
        # Short-term trend (crossovers)
        if latest_sma_short > latest_sma_medium:
            if trend in ["bullish", "neutral"]:
                trend = "bullish"
                strength += 1
        elif latest_sma_short < latest_sma_medium:
            if trend in ["bearish", "neutral"]:
                trend = "bearish"
                strength -= 1
        
        # Normalize strength
        if trend == "bullish":
            strength = min(strength, 3)  # 1 to 3 for bullish
        elif trend == "bearish":
            strength = max(strength, -3)  # -1 to -3 for bearish
        else:
            strength = 0  # 0 for neutral
        
        return {
            "trend": trend,
            "strength": strength,
            "price": latest_close,
            "sma_short": latest_sma_short,
            "sma_medium": latest_sma_medium,
            "sma_long": latest_sma_long
        }
    
    def analyze_momentum(self, symbol: str) -> Dict[str, Any]:
        """Analyze momentum using RSI and MACD.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with momentum analysis results
        """
        # Fetch RSI
        rsi_df = self.fetch_indicator(
            symbol=symbol,
            indicator="RSI",
            time_period=self.config["rsi_period"],
            series_type="close"
        )
        
        # Fetch MACD
        macd_df = self.fetch_indicator(
            symbol=symbol,
            indicator="MACD",
            fastperiod=self.config["macd_fast"],
            slowperiod=self.config["macd_slow"],
            signalperiod=self.config["macd_signal"],
            series_type="close"
        )
        
        results = {
            "momentum": "neutral",
            "strength": 0
        }
        
        # Process RSI
        if rsi_df is not None and not rsi_df.empty:
            latest_rsi = rsi_df["RSI"].iloc[-1]
            results["rsi"] = latest_rsi
            
            if latest_rsi > self.config["rsi_overbought"]:
                results["momentum"] = "bearish"
                results["strength"] -= 1
            elif latest_rsi < self.config["rsi_oversold"]:
                results["momentum"] = "bullish"
                results["strength"] += 1
        
        # Process MACD
        if macd_df is not None and not macd_df.empty:
            latest_macd = macd_df["MACD"].iloc[-1]
            latest_signal = macd_df["MACD_Signal"].iloc[-1]
            latest_hist = macd_df["MACD_Hist"].iloc[-1]
            
            results["macd"] = latest_macd
            results["macd_signal"] = latest_signal
            results["macd_hist"] = latest_hist
            
            # MACD Crossover
            if latest_macd > latest_signal:
                if results["momentum"] in ["bullish", "neutral"]:
                    results["momentum"] = "bullish"
                    results["strength"] += 1
            elif latest_macd < latest_signal:
                if results["momentum"] in ["bearish", "neutral"]:
                    results["momentum"] = "bearish"
                    results["strength"] -= 1
            
            # MACD Histogram direction
            if macd_df["MACD_Hist"].iloc[-1] > macd_df["MACD_Hist"].iloc[-2]:
                if results["momentum"] == "bullish":
                    results["strength"] += 1
            elif macd_df["MACD_Hist"].iloc[-1] < macd_df["MACD_Hist"].iloc[-2]:
                if results["momentum"] == "bearish":
                    results["strength"] -= 1
        
        # Normalize strength
        if results["momentum"] == "bullish":
            results["strength"] = min(results["strength"], 3)
        elif results["momentum"] == "bearish":
            results["strength"] = max(results["strength"], -3)
        else:
            results["strength"] = 0
        
        return results
    
    def analyze_volatility(self, symbol: str) -> Dict[str, Any]:
        """Analyze volatility using Bollinger Bands and ATR.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with volatility analysis results
        """
        # Fetch Bollinger Bands
        bbands_df = self.fetch_indicator(
            symbol=symbol,
            indicator="BBANDS",
            time_period=self.config["bbands_period"],
            series_type="close",
            nbdevup=self.config["bbands_std"],
            nbdevdn=self.config["bbands_std"],
            matype=0
        )
        
        # Fetch ATR
        atr_df = self.fetch_indicator(
            symbol=symbol,
            indicator="ATR",
            time_period=self.config["atr_period"]
        )
        
        # Fetch daily data to calculate relative ATR
        daily_df = self.fetch_daily_data(symbol, full=False)
        
        results = {
            "volatility": "medium",
            "signal": "neutral"
        }
        
        # Process Bollinger Bands
        if bbands_df is not None and not bbands_df.empty and daily_df is not None and not daily_df.empty:
            latest_close = daily_df["Close"].iloc[-1]
            latest_upper = bbands_df["Real Upper Band"].iloc[-1]
            latest_middle = bbands_df["Real Middle Band"].iloc[-1]
            latest_lower = bbands_df["Real Lower Band"].iloc[-1]
            
            # Calculate bandwidth
            bandwidth = (latest_upper - latest_lower) / latest_middle
            
            results["bbands_upper"] = latest_upper
            results["bbands_middle"] = latest_middle
            results["bbands_lower"] = latest_lower
            results["bbands_bandwidth"] = bandwidth
            
            # Price position relative to bands
            if latest_close > latest_upper:
                results["signal"] = "bearish"  # Overbought
            elif latest_close < latest_lower:
                results["signal"] = "bullish"  # Oversold
            
            # Volatility assessment
            if bandwidth > 0.1:
                results["volatility"] = "high"
            elif bandwidth < 0.05:
                results["volatility"] = "low"
            else:
                results["volatility"] = "medium"
        
        # Process ATR
        if atr_df is not None and not atr_df.empty and daily_df is not None and not daily_df.empty:
            latest_atr = atr_df["ATR"].iloc[-1]
            latest_close = daily_df["Close"].iloc[-1]
            
            # Relative ATR (as percentage of price)
            relative_atr = (latest_atr / latest_close) * 100
            
            results["atr"] = latest_atr
            results["relative_atr"] = relative_atr
            
            # Update volatility assessment
            if relative_atr > 3.0:
                results["volatility"] = "high"
            elif relative_atr < 1.5:
                results["volatility"] = "low"
        
        return results
    
    def analyze_trend_strength(self, symbol: str) -> Dict[str, Any]:
        """Analyze trend strength using ADX.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with trend strength analysis results
        """
        # Fetch ADX
        adx_df = self.fetch_indicator(
            symbol=symbol,
            indicator="ADX",
            time_period=self.config["adx_period"]
        )
        
        results = {
            "trend_strength": "weak",
            "adx": 0
        }
        
        # Process ADX
        if adx_df is not None and not adx_df.empty:
            latest_adx = adx_df["ADX"].iloc[-1]
            results["adx"] = latest_adx
            
            # Interpret ADX
            if latest_adx > 50:
                results["trend_strength"] = "very_strong"
            elif latest_adx > 40:
                results["trend_strength"] = "strong"
            elif latest_adx > self.config["adx_threshold"]:
                results["trend_strength"] = "moderate"
            elif latest_adx > 15:
                results["trend_strength"] = "weak"
            else:
                results["trend_strength"] = "absent"
        
        return results
    
    def get_technical_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive technical analysis summary.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with technical analysis summary
        """
        logger.info(f"Generating technical analysis summary for {symbol}")
        
        # Get daily data
        daily_df = self.fetch_daily_data(symbol, full=False)
        
        if daily_df is None or daily_df.empty:
            logger.error(f"Failed to get data for {symbol}")
            return {"error": f"Failed to get data for {symbol}"}
        
        # Analyze different aspects
        price_trend = self.analyze_price_trend(daily_df)
        momentum = self.analyze_momentum(symbol)
        volatility = self.analyze_volatility(symbol)
        trend_strength = self.analyze_trend_strength(symbol)
        
        # Current price
        current_price = daily_df["Close"].iloc[-1]
        
        # Combine all indicators
        indicators = {
            "sma_short": price_trend.get("sma_short", None),
            "sma_medium": price_trend.get("sma_medium", None),
            "sma_long": price_trend.get("sma_long", None),
            "rsi": momentum.get("rsi", None),
            "macd": momentum.get("macd", None),
            "macd_signal": momentum.get("macd_signal", None),
            "macd_hist": momentum.get("macd_hist", None),
            "bbands_upper": volatility.get("bbands_upper", None),
            "bbands_middle": volatility.get("bbands_middle", None),
            "bbands_lower": volatility.get("bbands_lower", None),
            "atr": volatility.get("atr", None),
            "relative_atr": volatility.get("relative_atr", None),
            "adx": trend_strength.get("adx", None)
        }
        
        # Generate signals
        signals = {
            "price_trend": price_trend.get("trend", "neutral"),
            "momentum": momentum.get("momentum", "neutral"),
            "volatility": volatility.get("volatility", "medium"),
            "trend_strength": trend_strength.get("trend_strength", "weak"),
            "volatility_signal": volatility.get("signal", "neutral")
        }
        
        # Generate overall signal
        overall_signal = self._generate_overall_signal(price_trend, momentum, volatility, trend_strength)
        
        # Create summary
        summary = {
            "symbol": symbol,
            "current_price": current_price,
            "date": daily_df.index[-1].strftime("%Y-%m-%d"),
            "indicators": indicators,
            "signals": signals,
            "overall_signal": overall_signal
        }
        
        return summary
    
    def _generate_overall_signal(self, price_trend, momentum, volatility, trend_strength) -> str:
        """Generate overall trading signal from component signals.
        
        Args:
            price_trend: Price trend analysis results
            momentum: Momentum analysis results
            volatility: Volatility analysis results
            trend_strength: Trend strength analysis results
            
        Returns:
            Overall signal string: "strong_buy", "buy", "neutral", "sell", "strong_sell"
        """
        # Score system: -5 to +5 scale
        score = 0
        
        # Price trend contribution
        if price_trend.get("trend") == "bullish":
            score += price_trend.get("strength", 0)
        elif price_trend.get("trend") == "bearish":
            score += price_trend.get("strength", 0)  # Already negative
        
        # Momentum contribution
        if momentum.get("momentum") == "bullish":
            score += momentum.get("strength", 0)
        elif momentum.get("momentum") == "bearish":
            score += momentum.get("strength", 0)  # Already negative
        
        # Volatility signal
        if volatility.get("signal") == "bullish":
            score += 1
        elif volatility.get("signal") == "bearish":
            score -= 1
        
        # Trend strength modifier
        trend_strength_value = trend_strength.get("adx", 0)
        if trend_strength_value > 25:
            # Amplify existing signal
            score = int(score * 1.2)
        elif trend_strength_value < 15:
            # Reduce conviction
            score = int(score * 0.8)
        
        # Convert score to signal
        if score >= 4:
            return "strong_buy"
        elif score >= 2:
            return "buy"
        elif score <= -4:
            return "strong_sell"
        elif score <= -2:
            return "sell"
        else:
            return "neutral"
    
    def generate_signals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate technical signals for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with technical signals for each symbol
        """
        signals = {}
        
        for symbol in symbols:
            try:
                signals[symbol] = self.get_technical_summary(symbol)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                signals[symbol] = {"error": str(e)}
        
        return signals
    
    def save_signals_to_json(self, signals: Dict[str, Dict[str, Any]], filename: str) -> bool:
        """Save generated signals to a JSON file.
        
        Args:
            signals: Dictionary with technical signals
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2)
            
            logger.info(f"Saved signals to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving signals to {filename}: {str(e)}")
            return False 