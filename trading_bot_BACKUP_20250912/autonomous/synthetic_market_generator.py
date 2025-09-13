#!/usr/bin/env python3
"""
Synthetic Market Generator

This module provides tools for generating synthetic market data for testing
trading strategies across different market regimes. It enables:

1. Generating realistic price movements with configurable patterns
2. Creating specific market regime scenarios (trending, mean-reverting, volatile)
3. Synthesizing correlated asset movements
4. Testing strategy performance against various market conditions

This is part 1 of the implementation focusing on the core framework.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegimeType(str, Enum):
    """Types of market regimes for synthetic data generation."""
    BULLISH = "bullish"           # Steadily rising market
    BEARISH = "bearish"           # Steadily falling market
    SIDEWAYS = "sideways"         # Range-bound, low-volatility market
    VOLATILE = "volatile"         # High-volatility, unpredictable moves
    TRENDING = "trending"         # Strong directional movement
    MEAN_REVERTING = "mean_reverting"  # Price returns to mean
    REGIME_CHANGE = "regime_change"    # Transition between regimes
    CRASH = "crash"               # Sharp market decline
    RECOVERY = "recovery"         # Post-crash recovery
    SECTOR_ROTATION = "sector_rotation"  # Changing sector leadership
    CUSTOM = "custom"             # User-defined custom regime


class PriceSeriesGenerator:
    """
    Base class for generating synthetic price series.
    
    This class provides the foundation for creating synthetic price
    data with various patterns and characteristics.
    """
    
    def __init__(
        self,
        base_price: float = 100.0,
        volatility: float = 0.01,
        drift: float = 0.0001,
        seed: Optional[int] = None
    ):
        """
        Initialize the price series generator.
        
        Args:
            base_price: Starting price for the synthetic series
            volatility: Daily volatility (standard deviation of returns)
            drift: Daily drift (mean of returns)
            seed: Random seed for reproducibility
        """
        self.base_price = base_price
        self.volatility = volatility
        self.drift = drift
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def generate_random_walk(
        self,
        days: int = 252,
        include_weekends: bool = False
    ) -> pd.DataFrame:
        """
        Generate a basic random walk price series.
        
        Args:
            days: Number of days to generate
            include_weekends: Whether to include weekend days
            
        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        # Generate daily returns
        daily_returns = np.random.normal(
            self.drift, self.volatility, days
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Calculate price series
        prices = self.base_price * cumulative_returns
        
        # Generate dates
        end_date = datetime.now()
        
        if include_weekends:
            # Simple date range including all days
            dates = [end_date - timedelta(days=i) for i in range(days)]
        else:
            # Skip weekends
            dates = []
            current_date = end_date
            while len(dates) < days:
                if current_date.weekday() < 5:  # Monday=0, Sunday=6
                    dates.append(current_date)
                current_date = current_date - timedelta(days=1)
        
        # Reverse to get ascending order
        dates = dates[::-1]
        
        # Create price DataFrame with OHLCV structure
        df = pd.DataFrame(index=dates)
        
        # Close prices are directly from our simulation
        df['close'] = prices
        
        # Generate other OHLCV columns
        # Here we use a simple model where:
        # - Open is close from previous day with small random offset
        # - High is max of open and close plus random positive offset
        # - Low is min of open and close minus random positive offset
        # - Volume correlates with volatility
        
        # Offset close to get open (with mean-reversion tendency)
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(
            -0.2 * daily_returns, 0.3 * self.volatility, days
        ))
        df.loc[df.index[0], 'open'] = self.base_price
        
        # Calculate high and low from open and close
        for i in range(len(df)):
            min_price = min(df['open'].iloc[i], df['close'].iloc[i])
            max_price = max(df['open'].iloc[i], df['close'].iloc[i])
            
            # High is above the max of open and close
            high_offset = abs(np.random.normal(0, self.volatility * max_price))
            df.loc[df.index[i], 'high'] = max_price + high_offset
            
            # Low is below the min of open and close
            low_offset = abs(np.random.normal(0, self.volatility * min_price))
            df.loc[df.index[i], 'low'] = min_price - low_offset
        
        # Generate volume (higher in more volatile periods)
        avg_volume = 1000000  # Base volume level
        daily_volatility = np.abs(df['close'].pct_change())
        
        # Volume correlates with volatility but has its own randomness
        df['volume'] = avg_volume * (1 + 5 * daily_volatility) * np.random.lognormal(
            0, 0.5, days
        )
        df.loc[df.index[0], 'volume'] = avg_volume
        
        return df
    
    def apply_trend(
        self,
        df: pd.DataFrame,
        trend_strength: float = 0.001,
        trend_direction: int = 1  # 1 for up, -1 for down
    ) -> pd.DataFrame:
        """
        Apply a trend to an existing price series.
        
        Args:
            df: Input price DataFrame
            trend_strength: Strength of the trend
            trend_direction: Direction of trend (1=up, -1=down)
            
        Returns:
            DataFrame with trend applied
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Generate trend component
        days = len(df)
        trend = np.linspace(0, days * trend_strength, days) * trend_direction
        
        # Apply trend to prices
        for col in ['open', 'high', 'low', 'close']:
            if col in result.columns:
                # Apply exponential trend factor
                result[col] = result[col] * np.exp(trend)
        
        return result
    
    def apply_mean_reversion(
        self,
        df: pd.DataFrame,
        reversion_strength: float = 0.2,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Apply mean reversion to an existing price series.
        
        Args:
            df: Input price DataFrame
            reversion_strength: Strength of reversion to the mean
            window: Window for calculating the moving average
            
        Returns:
            DataFrame with mean reversion applied
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate moving average
        ma = result['close'].rolling(window=window, min_periods=1).mean()
        
        # Calculate percent distance from MA
        distance = (result['close'] - ma) / ma
        
        # Apply mean reversion: prices far from MA get pulled back
        reversion_factor = 1 - (reversion_strength * distance)
        
        # Apply to all price columns
        for col in ['open', 'high', 'low', 'close']:
            if col in result.columns:
                result[col] = result[col] * reversion_factor
        
        return result
    
    def apply_volatility_regime(
        self,
        df: pd.DataFrame,
        volatility_factor: float = 2.0,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply increased or decreased volatility to a section of the price series.
        
        Args:
            df: Input price DataFrame
            volatility_factor: Multiplier for volatility (>1 = increase, <1 = decrease)
            window_start: Starting index for volatility change (None = start of series)
            window_end: Ending index for volatility change (None = end of series)
            
        Returns:
            DataFrame with modified volatility
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Set default window if not specified
        if window_start is None:
            window_start = 0
        if window_end is None:
            window_end = len(df)
        
        # Get the subset to modify
        subset = result.iloc[window_start:window_end]
        
        # Calculate returns of closing prices
        returns = subset['close'].pct_change().fillna(0)
        
        # Calculate new returns with modified volatility
        new_returns = returns * volatility_factor
        
        # Recalculate prices from new returns
        base_price = subset['close'].iloc[0]
        new_closes = base_price * np.cumprod(1 + new_returns)
        
        # Apply new closing prices
        result.loc[subset.index, 'close'] = new_closes
        
        # Adjust other price columns to maintain relative relationships
        for i in range(len(subset)):
            if i > 0:  # Skip first row as it's already correct
                idx = subset.index[i]
                
                # Calculate ratio of new close to old close
                price_ratio = result.loc[idx, 'close'] / df.loc[idx, 'close']
                
                # Apply same ratio to other price columns
                for col in ['open', 'high', 'low']:
                    if col in result.columns:
                        result.loc[idx, col] = df.loc[idx, col] * price_ratio
        
        return result


class SyntheticMarketGenerator:
    """
    Generates synthetic market data for testing trading strategies.
    
    This class provides methods to create various market scenarios
    and regime patterns for strategy evaluation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic market generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.generator = PriceSeriesGenerator(seed=seed)
    
    def generate_regime_scenario(
        self,
        regime: MarketRegimeType,
        days: int = 252,
        base_price: float = 100.0,
        volatility: float = 0.01,
        include_weekends: bool = False
    ) -> pd.DataFrame:
        """
        Generate a specific market regime scenario.
        
        Args:
            regime: Type of market regime to generate
            days: Number of days to generate
            base_price: Starting price
            volatility: Base volatility level
            include_weekends: Whether to include weekend days
            
        Returns:
            DataFrame with market data for the specified regime
        """
        # Update generator with parameters
        self.generator = PriceSeriesGenerator(
            base_price=base_price,
            volatility=volatility,
            seed=self.seed
        )
        
        # Start with a basic random walk
        df = self.generator.generate_random_walk(
            days=days,
            include_weekends=include_weekends
        )
        
        # Apply regime-specific modifications
        if regime == MarketRegimeType.BULLISH:
            # Strong upward trend
            df = self.generator.apply_trend(
                df, trend_strength=0.001, trend_direction=1
            )
        
        elif regime == MarketRegimeType.BEARISH:
            # Strong downward trend
            df = self.generator.apply_trend(
                df, trend_strength=0.001, trend_direction=-1
            )
        
        elif regime == MarketRegimeType.SIDEWAYS:
            # Mean-reverting with low volatility
            df = self.generator.apply_mean_reversion(
                df, reversion_strength=0.3, window=10
            )
            df = self.generator.apply_volatility_regime(
                df, volatility_factor=0.5
            )
        
        elif regime == MarketRegimeType.VOLATILE:
            # High volatility
            df = self.generator.apply_volatility_regime(
                df, volatility_factor=2.5
            )
        
        elif regime == MarketRegimeType.TRENDING:
            # Random direction trend but strong
            direction = 1 if np.random.random() > 0.5 else -1
            df = self.generator.apply_trend(
                df, trend_strength=0.002, trend_direction=direction
            )
        
        elif regime == MarketRegimeType.MEAN_REVERTING:
            # Strong mean reversion
            df = self.generator.apply_mean_reversion(
                df, reversion_strength=0.4, window=15
            )
        
        elif regime == MarketRegimeType.CRASH:
            # Normal market followed by sharp crash
            crash_start = int(days * 0.7)  # Crash in last 30% of series
            
            # Apply uptrend before crash
            df = self.generator.apply_trend(
                df, trend_strength=0.0005, trend_direction=1
            )
            
            # Apply crash
            crash_series = pd.Series(
                index=range(crash_start, days),
                data=np.linspace(1.0, 0.6, days - crash_start)  # 40% drop
            )
            
            # Apply crash to prices
            for i in range(crash_start, days):
                crash_factor = crash_series.iloc[i - crash_start]
                df.iloc[i, df.columns.get_indexer(['open', 'high', 'low', 'close'])] *= crash_factor
        
        elif regime == MarketRegimeType.RECOVERY:
            # Initial drop followed by recovery
            recovery_start = int(days * 0.3)  # Recovery after first 30%
            
            # Apply downtrend before recovery
            df = self.generator.apply_trend(
                df, trend_strength=0.001, trend_direction=-1
            )
            
            # Overwrite with recovery trend
            recovery_df = df.iloc[recovery_start:].copy()
            recovery_df = self.generator.apply_trend(
                recovery_df, trend_strength=0.002, trend_direction=1
            )
            
            # Combine the dataframes
            df.iloc[recovery_start:] = recovery_df
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Save generated market data to CSV file.
        
        Args:
            df: DataFrame with market data
            filename: Output filename
        """
        df.to_csv(filename)
        logger.info(f"Saved synthetic market data to {filename}")
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load market data from CSV file.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with market data
        """
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        logger.info(f"Loaded synthetic market data from {filename}")
        return df


# Singleton instance
_synthetic_market_generator = None


def get_synthetic_market_generator() -> SyntheticMarketGenerator:
    """
    Get singleton instance of SyntheticMarketGenerator.
    
    Returns:
        SyntheticMarketGenerator instance
    """
    global _synthetic_market_generator
    
    if _synthetic_market_generator is None:
        _synthetic_market_generator = SyntheticMarketGenerator()
    
    return _synthetic_market_generator


if __name__ == "__main__":
    # Simple example usage
    generator = get_synthetic_market_generator()
    
    # Generate a bullish scenario
    bullish_data = generator.generate_regime_scenario(
        MarketRegimeType.BULLISH,
        days=252,
        base_price=100.0,
        volatility=0.015
    )
    
    print(f"Generated {len(bullish_data)} days of bullish market data")
    print(f"Starting price: ${bullish_data['close'].iloc[0]:.2f}")
    print(f"Ending price: ${bullish_data['close'].iloc[-1]:.2f}")
    print(f"Return: {(bullish_data['close'].iloc[-1] / bullish_data['close'].iloc[0] - 1) * 100:.2f}%")
