#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asset Indicator Suite - Asset-agnostic technical indicators for all market types.
Extends CryptoIndicatorSuite to support equities, forex, and other asset classes.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from trading_bot.data.crypto_indicators import CryptoIndicatorSuite, IndicatorConfig

# Setup logging
logger = logging.getLogger(__name__)

class AssetType(str, Enum):
    """Asset type enumeration."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    FUTURES = "futures"
    UNKNOWN = "unknown"


@dataclass
class AssetIndicatorConfig(IndicatorConfig):
    """
    Configuration for asset indicators, extending IndicatorConfig with asset-specific settings.
    """
    asset_type: AssetType = AssetType.CRYPTO
    is_24h_market: bool = True
    # Default parameters optimized for each asset type
    equity_volatility_factor: float = 1.0
    forex_volatility_factor: float = 0.8
    # Settings for options data
    include_options_data: bool = False
    options_volume_weight: float = 0.3
    # Data quality settings
    tradingview_priority: bool = True  # When True, prefer TradingView indicator values
    

class AssetIndicatorSuite:
    """
    Asset-agnostic technical indicator suite supporting all market types.
    Wraps the CryptoIndicatorSuite and adjusts parameters based on asset type.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], AssetIndicatorConfig]] = None):
        """
        Initialize the asset indicator suite.
        
        Args:
            config: Configuration for indicators (Dict or AssetIndicatorConfig)
        """
        # Convert dict to AssetIndicatorConfig if needed
        if isinstance(config, dict):
            self.config = AssetIndicatorConfig(**config)
        else:
            self.config = config or AssetIndicatorConfig()
            
        # Create appropriate base indicator suite based on asset type
        indicator_config = self._adapt_config_for_asset()
        self.indicator_suite = CryptoIndicatorSuite(indicator_config)
        
        # Store metadata about available indicators by asset type
        self.indicator_metadata = {}
        
        # Initialize TradingView indicator cache
        self.tradingview_indicators = {}
        
        logger.info(f"Initialized AssetIndicatorSuite for {self.config.asset_type.value}")
    
    def _adapt_config_for_asset(self) -> Dict[str, Any]:
        """
        Adapt configuration parameters based on asset type.
        
        Returns:
            Dictionary with adapted configuration
        """
        from trading_bot.data.crypto_indicators import IndicatorConfig
        # Start with base config
        base_config = {k: v for k, v in self.config.__dict__.items() 
                      if k not in ['asset_type', 'is_24h_market', 'equity_volatility_factor', 
                                  'forex_volatility_factor', 'include_options_data', 
                                  'options_volume_weight', 'tradingview_priority']}
        
        # Adjust parameters based on asset type
        asset_type = self.config.asset_type
        
        if asset_type == AssetType.EQUITY:
            # Adjust for equity markets
            base_config['volatility_lookback_window'] = 252  # Trading days in a year
            base_config['bollinger_dev'] = 2.0 * self.config.equity_volatility_factor
            base_config['default_length'] = 20  # Standard for equities
            
        elif asset_type == AssetType.FOREX:
            # Adjust for forex markets
            base_config['volatility_lookback_window'] = 30  # Month of data
            base_config['bollinger_dev'] = 1.8 * self.config.forex_volatility_factor
            base_config['default_length'] = 14  # Common for forex
            
        elif asset_type == AssetType.FUTURES:
            # Adjust for futures markets
            base_config['donchian_window'] = 55  # Longer for futures
            base_config['volume_profile_periods'] = 20  # Standard for futures
            
        # For crypto, we use the defaults from CryptoIndicatorSuite
        
        # Only keep keys that are valid fields for IndicatorConfig
        valid_fields = set(IndicatorConfig.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in base_config.items() if k in valid_fields}
        return filtered_config

    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Validate columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return df
        
        # Apply indicators with the underlying suite
        result = self.indicator_suite.add_all_indicators(df)
        
        # Apply asset-specific additional indicators if needed
        asset_type = self.config.asset_type
        
        # Add equity-specific indicators
        if asset_type == AssetType.EQUITY:
            result = self._add_equity_indicators(result)
            
        # Add forex-specific indicators
        elif asset_type == AssetType.FOREX:
            result = self._add_forex_indicators(result)
        
        # Add futures-specific indicators
        elif asset_type == AssetType.FUTURES:
            result = self._add_futures_indicators(result)
        
        # Merge with TradingView indicators if available and prioritized
        if self.config.tradingview_priority:
            result = self._merge_tradingview_indicators(result)
        
        return result
    
    def _add_equity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add equity-specific indicators."""
        # For now, just return the DataFrame
        # This can be expanded with equity-specific indicators later
        return df
    
    def _add_forex_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forex-specific indicators."""
        # For now, just return the DataFrame
        # This can be expanded with forex-specific indicators later
        return df
    
    def _add_futures_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add futures-specific indicators."""
        # For now, just return the DataFrame
        # This can be expanded with futures-specific indicators later
        return df
    
    def _merge_tradingview_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge TradingView indicators with calculated indicators.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            DataFrame with merged indicators
        """
        if not self.tradingview_indicators:
            return df
        
        result = df.copy()
        
        # Get the symbol from the DataFrame (if available)
        if 'symbol' in result.columns:
            symbol = result['symbol'].iloc[0]
        else:
            # If no symbol column, just use the first symbol in tradingview_indicators
            if self.tradingview_indicators:
                symbol = list(self.tradingview_indicators.keys())[0]
            else:
                return result
        
        # Get indicators for this symbol
        if symbol not in self.tradingview_indicators:
            return result
        
        tv_indicators = self.tradingview_indicators[symbol]
        
        # Add TradingView indicators as new columns or override existing ones
        for indicator, value in tv_indicators.items():
            indicator_col = f"tv_{indicator}" if indicator in result.columns else indicator
            
            # Set the indicator value for the last row (most recent)
            if not result.empty:
                result.loc[result.index[-1], indicator_col] = value
        
        return result
    
    def update_tradingview_indicators(self, symbol: str, indicators: Dict[str, Any]) -> None:
        """
        Update TradingView indicators for a symbol.
        
        Args:
            symbol: Symbol to update
            indicators: Dictionary of indicator values from TradingView
        """
        self.tradingview_indicators[symbol] = indicators
        logger.info(f"Updated TradingView indicators for {symbol}")
    
    def get_indicator_metadata(self, indicator: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for indicators.
        
        Args:
            indicator: Specific indicator to get metadata for, or None for all
            
        Returns:
            Dictionary with indicator metadata
        """
        return self.indicator_suite.get_indicator_metadata(indicator)
    
    def generate_signals(self, df: pd.DataFrame, 
                        strategy: str = 'combined') -> pd.DataFrame:
        """
        Generate trading signals from indicators.
        
        Args:
            df: DataFrame with indicator data
            strategy: Strategy to use for signal generation
            
        Returns:
            DataFrame with added signals
        """
        return self.indicator_suite.generate_signals(df, strategy)


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example data
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000, 200, 100)
    }
    
    # Ensure OHLC relationships are valid
    for i in range(len(data['open'])):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
        data['volume'][i] = abs(data['volume'][i])
    
    df = pd.DataFrame(data, index=dates)
    
    # Create indicator suite for equity
    config = AssetIndicatorConfig(
        asset_type=AssetType.EQUITY,
        default_length=20,
        enable_adaptive_parameters=True
    )
    indicator_suite = AssetIndicatorSuite(config)
    
    # Add indicators
    result = indicator_suite.add_all_indicators(df)
    
    # Generate signals
    signals = indicator_suite.generate_signals(result)
    
    # Print some results
    print(f"Generated indicators and signals for {len(df)} data points")
    print(f"Indicator columns: {len(result.columns)}")
    print(f"Signal columns: {[col for col in signals.columns if 'signal' in col]}") 