#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asset Correlation Manager

This module provides functionality for tracking and analyzing correlations
between different assets and asset classes, supporting multi-asset trading strategies.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from enum import Enum

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.strategies.external_signal_strategy import (
    ExternalSignal, SignalSource, SignalType, Direction
)

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Standard asset classes for correlation analysis."""
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"
    INDEX = "index"
    BOND = "bond"
    ETF = "etf"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_symbol(cls, symbol: str) -> 'AssetClass':
        """
        Determine asset class from symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            AssetClass enum value
        """
        symbol = symbol.upper()
        
        # Common forex pairs
        forex_pairs = [
            "EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"
        ]
        
        # Check for forex pairs
        for base in forex_pairs:
            for quote in forex_pairs:
                if base != quote and f"{base}{quote}" == symbol:
                    return cls.FOREX
        
        # Common cryptocurrencies
        if symbol.endswith("USD") or symbol.endswith("USDT") or symbol.endswith("BTC"):
            crypto_bases = [
                "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "SOL", "DOGE"
            ]
            for base in crypto_bases:
                if symbol.startswith(base):
                    return cls.CRYPTO
        
        # US equities typically have 1-5 character symbols
        if len(symbol) <= 5 and symbol.isalpha():
            return cls.STOCK
        
        # Common commodities
        commodities = [
            "XAUUSD", "GOLD", "SILVER", "XAGUSD", "OIL", "WTICOUSD", "BRENTCRUSD", 
            "NATGAS", "COPPER"
        ]
        if symbol in commodities:
            return cls.COMMODITY
        
        # Common indices
        indices = [
            "SPX500", "NAS100", "US30", "UK100", "GER30", "JPN225", "AUS200", "ES", "NQ"
        ]
        if symbol in indices or symbol.startswith("^"):
            return cls.INDEX
        
        # Common ETFs
        if symbol.endswith("ETF") or len(symbol) <= 4 and symbol[0] in "XSVQ":
            return cls.ETF
        
        # Default
        return cls.UNKNOWN


class CorrelationType(Enum):
    """Types of correlations between assets."""
    POSITIVE = "positive"  # Assets move in the same direction
    NEGATIVE = "negative"  # Assets move in opposite directions
    UNCORRELATED = "uncorrelated"  # No significant correlation
    LEAD_LAG = "lead_lag"  # One asset leads the other
    REGIME_DEPENDENT = "regime_dependent"  # Correlation depends on market regime


class AssetCorrelation:
    """
    Represents a correlation between two assets.
    
    This class tracks the correlation characteristics, history,
    and metadata for a pair of assets.
    """
    
    def __init__(
        self,
        asset1: str,
        asset2: str,
        correlation_type: CorrelationType = CorrelationType.UNCORRELATED,
        correlation_value: float = 0.0,
        lookback_days: int = 30,
        asset1_class: Optional[AssetClass] = None,
        asset2_class: Optional[AssetClass] = None
    ):
        """
        Initialize an asset correlation.
        
        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            correlation_type: Type of correlation
            correlation_value: Correlation coefficient (-1 to 1)
            lookback_days: Number of days to look back for correlation analysis
            asset1_class: Asset class of first asset
            asset2_class: Asset class of second asset
        """
        self.asset1 = asset1
        self.asset2 = asset2
        self.correlation_type = correlation_type
        self.correlation_value = correlation_value
        self.lookback_days = lookback_days
        
        # Determine asset classes if not provided
        self.asset1_class = asset1_class or AssetClass.from_symbol(asset1)
        self.asset2_class = asset2_class or AssetClass.from_symbol(asset2)
        
        # Initialize history
        self.correlation_history: List[Dict[str, Any]] = []
        self.last_updated = datetime.now()
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        
        # Add initial correlation to history
        self._add_correlation_to_history(correlation_value)
    
    def _add_correlation_to_history(self, correlation_value: float) -> None:
        """
        Add a correlation value to history.
        
        Args:
            correlation_value: Correlation coefficient
        """
        entry = {
            "timestamp": datetime.now(),
            "correlation_value": correlation_value,
            "correlation_type": self._determine_correlation_type(correlation_value).value
        }
        
        self.correlation_history.append(entry)
        self.last_updated = entry["timestamp"]
    
    def _determine_correlation_type(self, correlation_value: float) -> CorrelationType:
        """
        Determine correlation type from correlation value.
        
        Args:
            correlation_value: Correlation coefficient
            
        Returns:
            CorrelationType enum value
        """
        if correlation_value > 0.5:
            return CorrelationType.POSITIVE
        elif correlation_value < -0.5:
            return CorrelationType.NEGATIVE
        else:
            return CorrelationType.UNCORRELATED
    
    def update_correlation(self, correlation_value: float) -> None:
        """
        Update the correlation with a new value.
        
        Args:
            correlation_value: New correlation coefficient
        """
        self.correlation_value = correlation_value
        self.correlation_type = self._determine_correlation_type(correlation_value)
        self._add_correlation_to_history(correlation_value)
    
    def get_average_correlation(self, days: Optional[int] = None) -> float:
        """
        Get the average correlation over a specified period.
        
        Args:
            days: Number of days to look back, or None for all history
            
        Returns:
            Average correlation value
        """
        if not self.correlation_history:
            return 0.0
        
        if days is None or days <= 0:
            # Return average of all history
            values = [entry["correlation_value"] for entry in self.correlation_history]
            return sum(values) / len(values)
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        recent_entries = [
            entry for entry in self.correlation_history
            if entry["timestamp"] >= cutoff
        ]
        
        if not recent_entries:
            return 0.0
        
        values = [entry["correlation_value"] for entry in recent_entries]
        return sum(values) / len(values)
    
    def is_significant(self, threshold: float = 0.7) -> bool:
        """
        Check if the correlation is significant.
        
        Args:
            threshold: Absolute correlation threshold for significance
            
        Returns:
            True if the correlation is significant
        """
        return abs(self.correlation_value) >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "correlation_type": self.correlation_type.value,
            "correlation_value": self.correlation_value,
            "lookback_days": self.lookback_days,
            "asset1_class": self.asset1_class.value,
            "asset2_class": self.asset2_class.value,
            "last_updated": self.last_updated.isoformat(),
            "correlation_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "correlation_value": entry["correlation_value"],
                    "correlation_type": entry["correlation_type"]
                }
                for entry in self.correlation_history
            ],
            "metadata": self.metadata
        }


class AssetCorrelationManager:
    """
    Manages correlations between multiple assets.
    
    This class is responsible for:
    1. Calculating and tracking correlations between assets
    2. Identifying correlation opportunities for trading
    3. Detecting changes in correlation patterns
    4. Publishing correlation events for strategy use
    """
    
    def __init__(
        self,
        monitored_assets: Optional[List[str]] = None,
        correlation_threshold: float = 0.7,
        lookback_days: int = 30,
        recalculation_interval_hours: int = 6
    ):
        """
        Initialize the asset correlation manager.
        
        Args:
            monitored_assets: List of assets to monitor
            correlation_threshold: Threshold for significant correlations
            lookback_days: Number of days to look back for correlation analysis
            recalculation_interval_hours: How often to recalculate correlations
        """
        self.monitored_assets = monitored_assets or []
        self.correlation_threshold = correlation_threshold
        self.lookback_days = lookback_days
        self.recalculation_interval_hours = recalculation_interval_hours
        
        # Storage for correlations
        self.correlations: Dict[str, AssetCorrelation] = {}
        
        # Storage for price data
        self.price_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Last calculation timestamp
        self.last_calculation = datetime.now() - timedelta(hours=recalculation_interval_hours)
        
        # Get event bus
        self.event_bus = EventBus()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"AssetCorrelationManager initialized with {len(self.monitored_assets)} assets")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # Listen for price updates
        self.event_bus.subscribe(
            EventType.PRICE_UPDATE,
            self._handle_price_update
        )
        
        # Listen for external signals
        self.event_bus.subscribe(
            EventType.EXTERNAL_SIGNAL,
            self._handle_external_signal
        )
    
    def _handle_price_update(self, event: Event) -> None:
        """
        Handle a price update event.
        
        Args:
            event: Price update event
        """
        price_data = event.data
        if not price_data:
            return
        
        symbol = price_data.get("symbol")
        if not symbol or symbol not in self.monitored_assets:
            return
        
        # Extract price information
        timestamp = price_data.get("timestamp", datetime.now())
        price = price_data.get("price")
        if not price:
            return
        
        # Update price data
        if symbol not in self.price_data:
            self.price_data[symbol] = []
        
        entry = {
            "timestamp": timestamp,
            "price": price
        }
        
        self.price_data[symbol].append(entry)
        
        # Check if we need to recalculate correlations
        self._check_recalculation()
    
    def _handle_external_signal(self, event: Event) -> None:
        """
        Handle an external signal event.
        
        Args:
            event: External signal event
        """
        signal_data = event.data.get("signal", {})
        if not signal_data:
            return
        
        symbol = signal_data.get("symbol")
        if not symbol:
            return
        
        # Add to monitored assets if not already there
        if symbol not in self.monitored_assets:
            self.monitored_assets.append(symbol)
            logger.info(f"Added {symbol} to monitored assets")
    
    def _check_recalculation(self) -> None:
        """Check if correlations need to be recalculated."""
        now = datetime.now()
        hours_since_last = (now - self.last_calculation).total_seconds() / 3600
        
        if hours_since_last >= self.recalculation_interval_hours:
            self._calculate_correlations()
            self.last_calculation = now
    
    def _calculate_correlations(self) -> None:
        """Calculate correlations between monitored assets."""
        logger.info("Calculating asset correlations")
        
        # Check if we have enough assets
        if len(self.monitored_assets) < 2:
            logger.warning("Not enough assets to calculate correlations")
            return
        
        # Check if we have enough price data
        assets_with_data = [
            symbol for symbol in self.monitored_assets
            if symbol in self.price_data and len(self.price_data[symbol]) >= 10
        ]
        
        if len(assets_with_data) < 2:
            logger.warning("Not enough price data to calculate correlations")
            return
        
        # Calculate correlations for each pair
        for i in range(len(assets_with_data)):
            for j in range(i + 1, len(assets_with_data)):
                asset1 = assets_with_data[i]
                asset2 = assets_with_data[j]
                
                try:
                    # Calculate correlation
                    correlation_value = self._calculate_pair_correlation(asset1, asset2)
                    
                    # Update or create correlation
                    pair_key = self._get_pair_key(asset1, asset2)
                    
                    if pair_key in self.correlations:
                        self.correlations[pair_key].update_correlation(correlation_value)
                    else:
                        self.correlations[pair_key] = AssetCorrelation(
                            asset1=asset1,
                            asset2=asset2,
                            correlation_value=correlation_value,
                            lookback_days=self.lookback_days
                        )
                    
                    # Publish correlation update if significant
                    if abs(correlation_value) >= self.correlation_threshold:
                        self._publish_correlation_update(
                            self.correlations[pair_key]
                        )
                        
                except Exception as e:
                    logger.error(f"Error calculating correlation for {asset1}-{asset2}: {str(e)}")
    
    def _calculate_pair_correlation(self, asset1: str, asset2: str) -> float:
        """
        Calculate correlation between two assets.
        
        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            
        Returns:
            Correlation coefficient
        """
        # Get price data
        prices1 = self.price_data.get(asset1, [])
        prices2 = self.price_data.get(asset2, [])
        
        if not prices1 or not prices2:
            return 0.0
        
        # Create series for each asset
        df1 = pd.DataFrame(prices1)
        df2 = pd.DataFrame(prices2)
        
        # Set timestamps as index
        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
        df2['timestamp'] = pd.to_datetime(df2['timestamp'])
        
        df1.set_index('timestamp', inplace=True)
        df2.set_index('timestamp', inplace=True)
        
        # Resample to ensure consistent timeframes
        df1 = df1.resample('1H').last().dropna()
        df2 = df2.resample('1H').last().dropna()
        
        # Find common timeframes
        common_index = df1.index.intersection(df2.index)
        
        if len(common_index) < 10:
            # Not enough common data points
            return 0.0
        
        # Filter to common timeframes
        df1 = df1.loc[common_index]
        df2 = df2.loc[common_index]
        
        # Calculate percentage changes
        returns1 = df1['price'].pct_change().dropna()
        returns2 = df2['price'].pct_change().dropna()
        
        # Calculate correlation
        correlation = returns1.corr(returns2)
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _get_pair_key(self, asset1: str, asset2: str) -> str:
        """
        Get a unique key for an asset pair.
        
        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            
        Returns:
            Unique pair key
        """
        # Sort alphabetically for consistency
        sorted_assets = sorted([asset1, asset2])
        return f"{sorted_assets[0]}_{sorted_assets[1]}"
    
    def _publish_correlation_update(self, correlation: AssetCorrelation) -> None:
        """
        Publish a correlation update event.
        
        Args:
            correlation: Asset correlation
        """
        event = Event(
            event_type=EventType.CORRELATION_UPDATE,
            data=correlation.to_dict()
        )
        
        self.event_bus.publish(event)
        logger.info(f"Published correlation update for {correlation.asset1}-{correlation.asset2}: {correlation.correlation_value:.2f}")
    
    def get_correlations(
        self,
        asset: Optional[str] = None,
        asset_class: Optional[Union[str, AssetClass]] = None,
        min_threshold: Optional[float] = None
    ) -> List[AssetCorrelation]:
        """
        Get correlations filtered by asset, class, or threshold.
        
        Args:
            asset: Optional asset symbol filter
            asset_class: Optional asset class filter
            min_threshold: Optional minimum correlation threshold
            
        Returns:
            List of asset correlations
        """
        if not self.correlations:
            return []
        
        threshold = min_threshold or self.correlation_threshold
        
        # Convert asset_class to enum if needed
        ac = None
        if asset_class:
            if isinstance(asset_class, str):
                ac = AssetClass(asset_class.lower())
            else:
                ac = asset_class
        
        # Filter correlations
        filtered_correlations = []
        
        for corr in self.correlations.values():
            # Skip if below threshold
            if abs(corr.correlation_value) < threshold:
                continue
            
            # Filter by asset
            if asset and asset != corr.asset1 and asset != corr.asset2:
                continue
            
            # Filter by asset class
            if ac and corr.asset1_class != ac and corr.asset2_class != ac:
                continue
            
            filtered_correlations.append(corr)
        
        return filtered_correlations
    
    def get_correlation(self, asset1: str, asset2: str) -> Optional[AssetCorrelation]:
        """
        Get correlation between two specific assets.
        
        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            
        Returns:
            AssetCorrelation or None
        """
        pair_key = self._get_pair_key(asset1, asset2)
        return self.correlations.get(pair_key)
    
    def get_correlated_assets(
        self,
        asset: str,
        correlation_type: Optional[CorrelationType] = None,
        min_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Get assets correlated with the given asset.
        
        Args:
            asset: Asset symbol
            correlation_type: Optional correlation type filter
            min_threshold: Minimum correlation threshold
            
        Returns:
            List of (asset, correlation) tuples
        """
        correlations = self.get_correlations(asset=asset, min_threshold=min_threshold)
        
        results = []
        for corr in correlations:
            # Skip if type doesn't match
            if correlation_type and corr.correlation_type != correlation_type:
                continue
            
            # Get the other asset
            other_asset = corr.asset2 if corr.asset1 == asset else corr.asset1
            
            results.append((other_asset, corr.correlation_value))
        
        return results
    
    def get_trade_opportunities(
        self,
        min_threshold: float = 0.8,
        correlation_types: Optional[List[CorrelationType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get potential trade opportunities based on correlations.
        
        Args:
            min_threshold: Minimum correlation threshold
            correlation_types: Optional list of correlation types to include
            
        Returns:
            List of trade opportunity dictionaries
        """
        opportunities = []
        
        # Use default correlation types if not specified
        types = correlation_types or [CorrelationType.POSITIVE, CorrelationType.NEGATIVE]
        
        # Get high-correlation pairs
        high_correlations = [
            corr for corr in self.correlations.values()
            if abs(corr.correlation_value) >= min_threshold and corr.correlation_type in types
        ]
        
        # Generate trade opportunities
        for corr in high_correlations:
            opportunity = {
                "asset1": corr.asset1,
                "asset2": corr.asset2,
                "correlation_value": corr.correlation_value,
                "correlation_type": corr.correlation_type.value,
                "asset1_class": corr.asset1_class.value,
                "asset2_class": corr.asset2_class.value,
                "opportunity_type": "correlation",
                "timestamp": datetime.now().isoformat()
            }
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def clear_old_price_data(self, max_age_days: int = 60) -> int:
        """
        Clear price data older than a certain age.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of entries cleared
        """
        now = datetime.now()
        max_age = timedelta(days=max_age_days)
        count = 0
        
        for symbol in self.price_data:
            old_length = len(self.price_data[symbol])
            self.price_data[symbol] = [
                entry for entry in self.price_data[symbol]
                if now - entry["timestamp"] <= max_age
            ]
            count += old_length - len(self.price_data[symbol])
        
        return count
    
    def add_monitored_asset(self, asset: str) -> None:
        """
        Add an asset to the monitored list.
        
        Args:
            asset: Asset symbol
        """
        if asset not in self.monitored_assets:
            self.monitored_assets.append(asset)
            logger.info(f"Added {asset} to monitored assets")
    
    def remove_monitored_asset(self, asset: str) -> None:
        """
        Remove an asset from the monitored list.
        
        Args:
            asset: Asset symbol
        """
        if asset in self.monitored_assets:
            self.monitored_assets.remove(asset)
            logger.info(f"Removed {asset} from monitored assets")
            
            # Remove related correlations
            to_remove = []
            for pair_key in self.correlations:
                if asset in pair_key:
                    to_remove.append(pair_key)
            
            for key in to_remove:
                del self.correlations[key]
            
            # Remove price data
            if asset in self.price_data:
                del self.price_data[asset]


def create_asset_correlation_manager(
    config: Optional[Dict[str, Any]] = None
) -> AssetCorrelationManager:
    """
    Factory function to create an AssetCorrelationManager with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AssetCorrelationManager
    """
    if not config:
        return AssetCorrelationManager()
    
    # Extract configuration
    monitored_assets = config.get("monitored_assets", [])
    correlation_threshold = config.get("correlation_threshold", 0.7)
    lookback_days = config.get("lookback_days", 30)
    recalculation_interval = config.get("recalculation_interval_hours", 6)
    
    return AssetCorrelationManager(
        monitored_assets=monitored_assets,
        correlation_threshold=correlation_threshold,
        lookback_days=lookback_days,
        recalculation_interval_hours=recalculation_interval
    )


if __name__ == "__main__":
    # Example usage
    correlation_manager = AssetCorrelationManager(
        monitored_assets=["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "AAPL", "MSFT"],
        correlation_threshold=0.7,
        lookback_days=30
    )
    
    # Example price updates (in a real scenario, these would come from the event bus)
    from random import random
    
    # Simulate price data for testing
    for symbol in correlation_manager.monitored_assets:
        correlation_manager.price_data[symbol] = []
        
        # Generate 100 price points
        base_price = 100.0
        for i in range(100):
            timestamp = datetime.now() - timedelta(hours=i)
            
            # EURUSD and GBPUSD will be positively correlated
            if symbol == "EURUSD":
                price = base_price + (random() - 0.3) * 5
            elif symbol == "GBPUSD":
                price = base_price + (random() - 0.3) * 5
            # USDJPY will be negatively correlated with EURUSD
            elif symbol == "USDJPY":
                price = base_price - (random() - 0.7) * 5
            # Others will be random
            else:
                price = base_price + (random() - 0.5) * 10
            
            entry = {
                "timestamp": timestamp,
                "price": price
            }
            
            correlation_manager.price_data[symbol].append(entry)
    
    # Calculate correlations
    correlation_manager._calculate_correlations()
    
    # Print correlations
    for corr in correlation_manager.correlations.values():
        print(f"{corr.asset1}-{corr.asset2}: {corr.correlation_value:.2f} ({corr.correlation_type.value})")
