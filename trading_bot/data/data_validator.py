"""
Market Data Validation Module

This module provides data quality checks for market data to ensure integrity
before it's used for trading decisions.

Features:
- Stale data detection
- Market hours validation
- Quote/trade reasonability checks
- Data gap identification
"""
import logging
import re
from datetime import datetime, time, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pytz

from trading_bot.core.event_bus import Event, get_global_event_bus
from trading_bot.core.constants import EventType
from trading_bot.security.secure_logger import SecureLogger

logger = SecureLogger(name=__name__)

class DataValidator:
    """
    Validates market data quality and publishes alerts when issues are detected.
    """
    
    # Default thresholds
    DEFAULT_STALE_THRESHOLDS = {
        "1m": 120,  # 2 minutes for 1-minute bars
        "5m": 360,  # 6 minutes for 5-minute bars
        "15m": 900,  # 15 minutes for 15-minute bars
        "1h": 3600,  # 1 hour for hourly bars
        "4h": 14400,  # 4 hours for 4-hour bars
        "1d": 86400,  # 24 hours for daily bars
        "tick": 30   # 30 seconds for tick data
    }
    
    # Reasonable price movement thresholds (as percentage)
    DEFAULT_PRICE_THRESHOLDS = {
        "stock": 0.10,     # 10% for stocks
        "forex": 0.03,     # 3% for forex
        "crypto": 0.20,    # 20% for crypto
        "futures": 0.05    # 5% for futures
    }
    
    # Volume spike thresholds (multiplier of avg volume)
    VOLUME_SPIKE_THRESHOLD = 10.0
    
    def __init__(self, 
                 stale_thresholds: Optional[Dict[str, int]] = None,
                 price_thresholds: Optional[Dict[str, float]] = None,
                 enable_market_hours_check: bool = True,
                 timezone: str = "US/Eastern"):
        """
        Initialize the data validator.
        
        Args:
            stale_thresholds: Dict mapping timeframes to stale thresholds in seconds
            price_thresholds: Dict mapping asset types to price movement thresholds
            enable_market_hours_check: Whether to check for market hours
            timezone: Timezone for market hours checking
        """
        self.stale_thresholds = stale_thresholds or self.DEFAULT_STALE_THRESHOLDS
        self.price_thresholds = price_thresholds or self.DEFAULT_PRICE_THRESHOLDS
        self.enable_market_hours_check = enable_market_hours_check
        self.timezone = pytz.timezone(timezone)
        self.event_bus = get_global_event_bus()
        
        # Last seen data timestamps by symbol and timeframe
        self.last_data_time: Dict[str, Dict[str, datetime]] = {}
        
        # Historical price ranges for reasonability checks
        self.historical_ranges: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Data validator initialized")
    
    def check_stale_data(self, 
                         symbol: str, 
                         timeframe: str, 
                         timestamp: datetime) -> bool:
        """
        Check if data for a symbol/timeframe is stale.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., "1m", "5m")
            timestamp: Timestamp of the current data point
            
        Returns:
            True if data is fresh, False if stale
        """
        # Initialize if this is the first data point for this symbol/timeframe
        if symbol not in self.last_data_time:
            self.last_data_time[symbol] = {}
        
        # If we have no prior data, just record and return fresh
        if timeframe not in self.last_data_time[symbol]:
            self.last_data_time[symbol][timeframe] = timestamp
            return True
        
        # Get threshold for this timeframe (default to 60 seconds if not specified)
        threshold = self.stale_thresholds.get(timeframe, 60)
        
        # Check time difference
        last_time = self.last_data_time[symbol][timeframe]
        time_diff = (timestamp - last_time).total_seconds()
        
        # Update the last time seen
        self.last_data_time[symbol][timeframe] = timestamp
        
        # For tick data, check if no ticks for too long
        if timeframe == "tick":
            if time_diff > threshold:
                self._publish_data_alert(
                    symbol=symbol,
                    alert_type="stale_data",
                    message=f"Stale tick data detected for {symbol}. No updates for {time_diff:.1f} seconds.",
                    severity="warning",
                    data={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "seconds_since_last_update": time_diff,
                        "threshold": threshold
                    }
                )
                return False
            return True
        
        # For bars, expected update frequency should align with timeframe
        # Convert timeframe to seconds
        timeframe_seconds = self._timeframe_to_seconds(timeframe)
        
        # If data is coming too late (threshold is a multiple of the timeframe)
        if time_diff > (timeframe_seconds + threshold):
            self._publish_data_alert(
                symbol=symbol,
                alert_type="stale_data",
                message=f"Stale {timeframe} data detected for {symbol}. Last update was {time_diff:.1f} seconds ago.",
                severity="warning",
                data={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "seconds_since_last_update": time_diff,
                    "threshold": threshold,
                    "expected_update_frequency": timeframe_seconds
                }
            )
            return False
        
        return True
    
    def check_market_hours(self, 
                          symbol: str, 
                          timestamp: datetime, 
                          asset_class: str = "stock") -> bool:
        """
        Check if the current time is within market hours for the given asset.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp to check
            asset_class: Asset class (stock, forex, crypto, etc.)
            
        Returns:
            True if within market hours, False otherwise
        """
        if not self.enable_market_hours_check:
            return True
        
        # Convert to market timezone
        market_time = timestamp.astimezone(self.timezone)
        
        # Different asset classes have different market hours
        if asset_class.lower() == "stock":
            # US Stock market: 9:30 AM - 4:00 PM ET, Monday-Friday
            is_market_open = (
                market_time.weekday() < 5 and  # Monday-Friday
                time(9, 30) <= market_time.time() <= time(16, 0)
            )
            
            if not is_market_open:
                self._publish_data_alert(
                    symbol=symbol,
                    alert_type="out_of_market_hours",
                    message=f"Data received for {symbol} outside of market hours: {market_time}",
                    severity="info",
                    data={
                        "symbol": symbol,
                        "timestamp": timestamp.isoformat(),
                        "market_time": market_time.isoformat(),
                        "asset_class": asset_class
                    }
                )
                return False
                
        elif asset_class.lower() == "forex":
            # Forex market: 24 hours except weekend
            is_market_open = market_time.weekday() < 5 or (
                market_time.weekday() == 5 and market_time.time() < time(17, 0)
            ) or (
                market_time.weekday() == 6 and market_time.time() >= time(17, 0)
            )
            
            if not is_market_open:
                self._publish_data_alert(
                    symbol=symbol,
                    alert_type="out_of_market_hours",
                    message=f"Data received for {symbol} during forex weekend: {market_time}",
                    severity="info",
                    data={
                        "symbol": symbol,
                        "timestamp": timestamp.isoformat(),
                        "market_time": market_time.isoformat(),
                        "asset_class": asset_class
                    }
                )
                return False
                
        elif asset_class.lower() == "crypto":
            # Crypto market: 24/7
            return True
        
        return True
    
    def check_price_reasonability(self, 
                                symbol: str, 
                                price: float,
                                prev_price: Optional[float] = None,
                                asset_class: str = "stock",
                                update_history: bool = True) -> bool:
        """
        Check if a price update is reasonable based on historical data.
        
        Args:
            symbol: Trading symbol
            price: Current price
            prev_price: Previous price (if None, will use historical data)
            asset_class: Asset class for threshold selection
            update_history: Whether to update the historical record
            
        Returns:
            True if price is reasonable, False otherwise
        """
        # Initialize history for this symbol if needed
        if symbol not in self.historical_ranges:
            self.historical_ranges[symbol] = {
                "min_price": price,
                "max_price": price,
                "last_price": price,
                "last_50_prices": [price],
                "avg_price": price
            }
            return True
            
        history = self.historical_ranges[symbol]
        
        # If no previous price specified, use the last recorded price
        if prev_price is None:
            prev_price = history["last_price"]
            
        # Skip if this is the first price
        if prev_price is None:
            if update_history:
                self._update_price_history(symbol, price)
            return True
            
        # Get the appropriate threshold for this asset class
        threshold = self.price_thresholds.get(asset_class.lower(), 0.10)
        
        # Calculate price change as percentage
        if prev_price > 0:
            pct_change = abs(price - prev_price) / prev_price
        else:
            pct_change = 1.0  # Avoid division by zero
            
        # Check if change exceeds threshold
        if pct_change > threshold:
            self._publish_data_alert(
                symbol=symbol,
                alert_type="price_spike",
                message=f"Unusual price change detected for {symbol}: {prev_price:.4f} → {price:.4f} ({pct_change*100:.2f}%)",
                severity="warning",
                data={
                    "symbol": symbol,
                    "current_price": price,
                    "previous_price": prev_price,
                    "percent_change": pct_change * 100,
                    "threshold_percent": threshold * 100,
                    "asset_class": asset_class
                }
            )
            
            # Still update history but flag as unreasonable
            if update_history:
                self._update_price_history(symbol, price)
                
            return False
            
        # Update history
        if update_history:
            self._update_price_history(symbol, price)
            
        return True
    
    def check_volume_reasonability(self,
                                  symbol: str,
                                  volume: float,
                                  update_history: bool = True) -> bool:
        """
        Check if volume is reasonable compared to historical data.
        
        Args:
            symbol: Trading symbol
            volume: Current volume
            update_history: Whether to update history
            
        Returns:
            True if volume is reasonable, False otherwise
        """
        # Initialize history for this symbol if needed
        if symbol not in self.historical_ranges:
            self.historical_ranges[symbol] = {
                "volumes": [volume],
                "avg_volume": volume,
                "max_volume": volume
            }
            return True
            
        history = self.historical_ranges[symbol]
        
        # Calculate average volume (last 20 periods)
        avg_volume = np.mean(history.get("volumes", [volume])[-20:])
        
        # Check if volume exceeds threshold
        if avg_volume > 0 and volume > avg_volume * self.VOLUME_SPIKE_THRESHOLD:
            self._publish_data_alert(
                symbol=symbol,
                alert_type="volume_spike",
                message=f"Unusual volume spike detected for {symbol}: {volume:,.0f} vs avg {avg_volume:,.0f}",
                severity="info",
                data={
                    "symbol": symbol,
                    "current_volume": volume,
                    "average_volume": avg_volume,
                    "ratio": volume / avg_volume if avg_volume > 0 else 0,
                    "threshold": self.VOLUME_SPIKE_THRESHOLD
                }
            )
            
            # Still update history
            if update_history and "volumes" in history:
                history["volumes"].append(volume)
                if len(history["volumes"]) > 100:
                    history["volumes"] = history["volumes"][-100:]
                history["avg_volume"] = np.mean(history["volumes"])
                history["max_volume"] = max(history["max_volume"], volume)
                
            return False
            
        # Update history
        if update_history:
            if "volumes" not in history:
                history["volumes"] = []
            history["volumes"].append(volume)
            if len(history["volumes"]) > 100:
                history["volumes"] = history["volumes"][-100:]
            history["avg_volume"] = np.mean(history["volumes"])
            history["max_volume"] = max(history.get("max_volume", 0), volume)
            
        return True
    
    def validate_bar_data(self,
                        symbol: str,
                        timeframe: str,
                        timestamp: datetime,
                        open_price: float,
                        high_price: float,
                        low_price: float,
                        close_price: float,
                        volume: float,
                        asset_class: str = "stock") -> bool:
        """
        Perform comprehensive validation on a bar of market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            timestamp: Bar timestamp
            open_price: Opening price
            high_price: High price
            low_price: Low price
            close_price: Closing price
            volume: Volume
            asset_class: Asset class
            
        Returns:
            True if data passes all checks, False otherwise
        """
        # Check for NaN or invalid values
        if (np.isnan(open_price) or np.isnan(high_price) or 
            np.isnan(low_price) or np.isnan(close_price) or
            np.isnan(volume)):
            self._publish_data_alert(
                symbol=symbol,
                alert_type="invalid_data",
                message=f"NaN values detected in {timeframe} bar for {symbol}",
                severity="error",
                data={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": timestamp.isoformat()
                }
            )
            return False
        
        # Check for negative prices
        if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
            self._publish_data_alert(
                symbol=symbol,
                alert_type="invalid_data",
                message=f"Negative or zero prices detected in {timeframe} bar for {symbol}",
                severity="error",
                data={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": timestamp.isoformat(),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price
                }
            )
            return False
        
        # Check high/low consistency
        if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
            self._publish_data_alert(
                symbol=symbol,
                alert_type="inconsistent_data",
                message=f"Inconsistent OHLC values in {timeframe} bar for {symbol}",
                severity="error",
                data={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": timestamp.isoformat(),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price
                }
            )
            return False
        
        # Run the individual checks
        stale_check = self.check_stale_data(symbol, timeframe, timestamp)
        hours_check = self.check_market_hours(symbol, timestamp, asset_class)
        price_check = self.check_price_reasonability(symbol, close_price, asset_class=asset_class)
        volume_check = self.check_volume_reasonability(symbol, volume)
        
        # Return True only if all checks pass
        return stale_check and hours_check and price_check and volume_check
    
    def validate_tick_data(self,
                          symbol: str,
                          timestamp: datetime,
                          price: float,
                          volume: Optional[float] = None,
                          asset_class: str = "stock") -> bool:
        """
        Validate tick data for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume (optional)
            asset_class: Asset class
            
        Returns:
            True if data passes all checks, False otherwise
        """
        # Check for NaN or invalid values
        if np.isnan(price) or (volume is not None and np.isnan(volume)):
            self._publish_data_alert(
                symbol=symbol,
                alert_type="invalid_data",
                message=f"NaN values detected in tick data for {symbol}",
                severity="error",
                data={
                    "symbol": symbol,
                    "timestamp": timestamp.isoformat()
                }
            )
            return False
        
        # Check for negative prices
        if price <= 0:
            self._publish_data_alert(
                symbol=symbol,
                alert_type="invalid_data",
                message=f"Negative or zero price detected in tick data for {symbol}",
                severity="error",
                data={
                    "symbol": symbol,
                    "timestamp": timestamp.isoformat(),
                    "price": price
                }
            )
            return False
        
        # Run the individual checks
        stale_check = self.check_stale_data(symbol, "tick", timestamp)
        hours_check = self.check_market_hours(symbol, timestamp, asset_class)
        price_check = self.check_price_reasonability(symbol, price, asset_class=asset_class)
        
        # Volume check is optional for tick data
        volume_check = True
        if volume is not None:
            volume_check = self.check_volume_reasonability(symbol, volume)
        
        # Return True only if all checks pass
        return stale_check and hours_check and price_check and volume_check
    
    def reset_history(self, symbol: Optional[str] = None):
        """
        Reset historical data for a symbol or all symbols.
        
        Args:
            symbol: Symbol to reset, or None for all symbols
        """
        if symbol is None:
            self.historical_ranges = {}
            self.last_data_time = {}
            logger.info("Reset all historical data")
        else:
            if symbol in self.historical_ranges:
                del self.historical_ranges[symbol]
            
            if symbol in self.last_data_time:
                del self.last_data_time[symbol]
                
            logger.info(f"Reset historical data for {symbol}")
            
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol."""
        if symbol not in self.historical_ranges:
            self.historical_ranges[symbol] = {
                "min_price": price,
                "max_price": price,
                "last_price": price,
                "last_50_prices": [price],
                "avg_price": price
            }
            return
            
        history = self.historical_ranges[symbol]
        
        # Update min/max
        history["min_price"] = min(history["min_price"], price)
        history["max_price"] = max(history["max_price"], price)
        history["last_price"] = price
        
        # Update price history
        if "last_50_prices" not in history:
            history["last_50_prices"] = []
        history["last_50_prices"].append(price)
        if len(history["last_50_prices"]) > 50:
            history["last_50_prices"] = history["last_50_prices"][-50:]
        
        # Update average
        history["avg_price"] = np.mean(history["last_50_prices"])
    
    def _publish_data_alert(self, 
                           symbol: str, 
                           alert_type: str,
                           message: str,
                           severity: str = "warning",
                           data: Optional[Dict[str, Any]] = None):
        """Publish a data quality alert to the event bus."""
        alert_data = {
            "symbol": symbol,
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            **data if data else {}
        }
        
        # Log the alert
        if severity == "error":
            logger.error(message)
        elif severity == "warning":
            logger.warning(message)
        else:
            logger.info(message)
        
        # Publish to event bus
        self.event_bus.create_and_publish(
            event_type=EventType.DATA_QUALITY_ALERT,
            data=alert_data,
            source="data_validator"
        )
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert a timeframe string to seconds."""
        if timeframe == "tick":
            return 1
            
        # Parse the timeframe (e.g., "1m", "5m", "1h", "1d")
        match = re.match(r"(\d+)([mhdw])", timeframe.lower())
        if not match:
            return 60  # Default to 1 minute
            
        value, unit = match.groups()
        value = int(value)
        
        if unit == "m":
            return value * 60  # minutes to seconds
        elif unit == "h":
            return value * 3600  # hours to seconds
        elif unit == "d":
            return value * 86400  # days to seconds
        elif unit == "w":
            return value * 604800  # weeks to seconds
        else:
            return 60  # Default to 1 minute
