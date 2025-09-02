"""
Market Data Generator

Generates synthetic market data for different market regimes (bull, bear, sideways)
for backtesting purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class MarketRegimeType(Enum):
    """Market regime types for synthetic data generation"""
    BULL = "bull"           # Upward trending market
    BEAR = "bear"           # Downward trending market
    SIDEWAYS = "sideways"   # Range-bound market
    VOLATILE = "volatile"   # High volatility market
    BREAKOUT = "breakout"   # Ranging then breakout
    REVERSAL = "reversal"   # Trend reversal
    CRASH = "crash"         # Market crash
    RECOVERY = "recovery"   # Market recovery after crash
    MIXED = "mixed"         # Mixed conditions

class MarketDataGenerator:
    """
    Generates synthetic price data for backtesting across different market regimes.
    """
    
    def __init__(self):
        """Initialize the market data generator"""
        self.symbols = []
        self.cached_data = {}  # Symbol -> regime -> DataFrame
        
    def generate_regime_data(self, 
                           symbol: str,
                           regime: MarketRegimeType,
                           days: int = 252,
                           start_date: Optional[datetime] = None,
                           initial_price: float = 100.0,
                           volatility: float = 1.0) -> pd.DataFrame:
        """
        Generate synthetic price data for a specific market regime.
        
        Args:
            symbol: Trading symbol
            regime: Market regime type
            days: Number of trading days
            start_date: Starting date (defaults to today - days)
            initial_price: Starting price
            volatility: Volatility multiplier
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set start date if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
            
        # Generate date range
        dates = []
        current_date = start_date
        # Skip weekends for more realistic data
        for _ in range(days):
            while current_date.weekday() > 4:  # Skip Saturday (5) and Sunday (6)
                current_date += timedelta(days=1)
            dates.append(current_date)
            current_date += timedelta(days=1)
            
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        
        # Base parameters for price movements
        close_prices = [initial_price]
        
        # Generate prices based on regime
        if regime == MarketRegimeType.BULL:
            # Bull market: Upward trend with noise
            drift = 0.0005 * volatility  # Daily upward drift
            daily_vol = 0.008 * volatility
            
            for _ in range(1, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift, daily_vol)))
                
        elif regime == MarketRegimeType.BEAR:
            # Bear market: Downward trend with noise
            drift = -0.0005 * volatility
            daily_vol = 0.010 * volatility
            
            for _ in range(1, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift, daily_vol)))
                
        elif regime == MarketRegimeType.SIDEWAYS:
            # Sideways market: Mean-reverting around initial price
            mean_rev_strength = 0.05 * volatility
            daily_vol = 0.006 * volatility
            
            for _ in range(1, days):
                # Mean reversion component pulls price back toward initial_price
                mean_rev = mean_rev_strength * (initial_price - close_prices[-1]) / initial_price
                close_prices.append(close_prices[-1] * (1 + mean_rev + np.random.normal(0, daily_vol)))
                
        elif regime == MarketRegimeType.VOLATILE:
            # Volatile market: Higher noise, regime shifts
            daily_vol = 0.018 * volatility
            
            for i in range(1, days):
                # Occasionally shift drift direction
                if i % 30 == 0:  # Shift every ~30 days
                    drift = np.random.choice([-0.001, 0.001]) * volatility
                else:
                    drift = 0
                    
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift, daily_vol)))
                
        elif regime == MarketRegimeType.BREAKOUT:
            # Ranging then breakout
            # First 70% of days are sideways
            sideways_days = int(days * 0.7)
            mean_rev_strength = 0.05 * volatility
            daily_vol = 0.006 * volatility
            
            for i in range(1, sideways_days):
                mean_rev = mean_rev_strength * (initial_price - close_prices[-1]) / initial_price
                close_prices.append(close_prices[-1] * (1 + mean_rev + np.random.normal(0, daily_vol)))
            
            # Breakout day
            close_prices.append(close_prices[-1] * (1 + np.random.choice([1, -1]) * 0.03 * volatility))
            
            # Trending after breakout
            breakout_dir = 1 if close_prices[-1] > close_prices[-2] else -1
            drift = breakout_dir * 0.0006 * volatility
            
            for _ in range(sideways_days + 1, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift, daily_vol)))
                
        elif regime == MarketRegimeType.REVERSAL:
            # Trend reversal: First half trending one way, second half the other
            half_days = days // 2
            
            # First half - uptrend
            drift1 = 0.0006 * volatility
            daily_vol = 0.009 * volatility
            
            for _ in range(1, half_days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift1, daily_vol)))
            
            # Reversal point - more volatile
            close_prices.append(close_prices[-1] * (1 - 0.02 * volatility))
            
            # Second half - downtrend
            drift2 = -0.0007 * volatility
            
            for _ in range(half_days + 1, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift2, daily_vol)))
                
        elif regime == MarketRegimeType.CRASH:
            # Market crash: Initial stability, then sharp decline, followed by high volatility
            pre_crash_days = int(days * 0.6)
            crash_days = int(days * 0.1)
            post_crash_days = days - pre_crash_days - crash_days
            
            # Pre-crash: Slight uptrend or sideways
            drift = 0.0003 * volatility
            daily_vol = 0.007 * volatility
            
            for _ in range(1, pre_crash_days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift, daily_vol)))
            
            # Crash period: Sharp decline
            crash_drift = -0.015 * volatility
            crash_vol = 0.025 * volatility
            
            for _ in range(pre_crash_days, pre_crash_days + crash_days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(crash_drift, crash_vol)))
            
            # Post-crash: High volatility with slight downward bias
            post_drift = -0.0005 * volatility
            post_vol = 0.020 * volatility
            
            for _ in range(pre_crash_days + crash_days, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(post_drift, post_vol)))
                
        elif regime == MarketRegimeType.RECOVERY:
            # Market recovery: Initial sharp decline, followed by gradual recovery
            decline_days = int(days * 0.2)
            recovery_days = days - decline_days
            
            # Initial decline
            decline_drift = -0.008 * volatility
            decline_vol = 0.015 * volatility
            
            for _ in range(1, decline_days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(decline_drift, decline_vol)))
            
            # Recovery period: Gradual uptrend with decreasing volatility
            recovery_drift = 0.0007 * volatility
            
            for i in range(decline_days, days):
                # Volatility decreases as recovery progresses
                progress = (i - decline_days) / recovery_days
                current_vol = (0.018 - 0.008 * progress) * volatility
                close_prices.append(close_prices[-1] * (1 + np.random.normal(recovery_drift, current_vol)))
                
        elif regime == MarketRegimeType.MIXED:
            # Mixed market: Alternating regimes
            segment_length = days // 4  # Four different segments
            
            # First segment: Bullish
            drift1 = 0.0006 * volatility
            vol1 = 0.008 * volatility
            
            for _ in range(1, segment_length):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift1, vol1)))
            
            # Second segment: Sideways
            mean_rev_strength = 0.05 * volatility
            vol2 = 0.006 * volatility
            sideways_center = close_prices[-1]
            
            for _ in range(segment_length, 2 * segment_length):
                mean_rev = mean_rev_strength * (sideways_center - close_prices[-1]) / sideways_center
                close_prices.append(close_prices[-1] * (1 + mean_rev + np.random.normal(0, vol2)))
            
            # Third segment: Bearish
            drift3 = -0.0006 * volatility
            vol3 = 0.010 * volatility
            
            for _ in range(2 * segment_length, 3 * segment_length):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift3, vol3)))
            
            # Fourth segment: Volatile/Recovery
            drift4 = 0.0004 * volatility
            vol4 = 0.015 * volatility
            
            for _ in range(3 * segment_length, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(drift4, vol4)))
        
        else:
            # Default: Random walk with slight upward bias
            for _ in range(1, days):
                close_prices.append(close_prices[-1] * (1 + np.random.normal(0.0001, 0.01 * volatility)))
        
        # Generate OHLCV data
        df = pd.DataFrame()
        df['date'] = date_strs
        df['close'] = close_prices
        
        # Generate realistic open/high/low values based on close
        daily_range_pct = 0.015 * volatility  # 1.5% average daily range scaled by volatility
        
        # Open prices: Previous close with small gap
        open_prices = [close_prices[0]]  # First open matches first close
        open_prices.extend([
            prev_close * (1 + np.random.normal(0, 0.003 * volatility))
            for prev_close in close_prices[:-1]
        ])
        
        # High and low based on close and open
        highs = []
        lows = []
        
        for i in range(days):
            # Base range as percentage of price
            day_range = close_prices[i] * daily_range_pct
            
            # Determine if close > open (up day) or close < open (down day)
            is_up_day = close_prices[i] >= open_prices[i]
            
            if is_up_day:
                high = close_prices[i] + np.random.uniform(0, 0.6) * day_range
                low = open_prices[i] - np.random.uniform(0, 0.6) * day_range
            else:
                high = open_prices[i] + np.random.uniform(0, 0.6) * day_range
                low = close_prices[i] - np.random.uniform(0, 0.6) * day_range
                
            # Ensure low <= open, close <= high
            low = min(low, open_prices[i], close_prices[i])
            high = max(high, open_prices[i], close_prices[i])
            
            highs.append(high)
            lows.append(low)
        
        # Volume profile depends on regime and price action
        volumes = []
        base_volume = 1000000
        
        for i in range(days):
            is_up_day = close_prices[i] >= open_prices[i]
            price_change = abs(close_prices[i] - open_prices[i]) / open_prices[i]
            
            # Volume increases with volatility and price change
            vol_factor = 1.0 + 5.0 * price_change
            
            # Regime-specific volume patterns
            if regime == MarketRegimeType.CRASH and i >= int(days * 0.6) and i < int(days * 0.7):
                # Higher volume during crash
                vol_factor *= 3.0
            elif regime == MarketRegimeType.BREAKOUT and i == int(days * 0.7):
                # Breakout volume spike
                vol_factor *= 4.0
            elif regime == MarketRegimeType.REVERSAL and i == days // 2:
                # Reversal volume spike
                vol_factor *= 2.5
                
            # Random component
            vol_factor *= np.random.uniform(0.7, 1.3)
            
            # Up days tend to have higher volume in bull markets, down days in bear markets
            if regime == MarketRegimeType.BULL and is_up_day:
                vol_factor *= 1.2
            elif regime == MarketRegimeType.BEAR and not is_up_day:
                vol_factor *= 1.2
                
            volume = int(base_volume * vol_factor)
            volumes.append(volume)
        
        # Combine into dataframe
        df['open'] = open_prices
        df['high'] = highs
        df['low'] = lows
        df['close'] = close_prices
        df['volume'] = volumes
        df['symbol'] = symbol
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Store in cache
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            
        if symbol not in self.cached_data:
            self.cached_data[symbol] = {}
            
        regime_key = regime.value
        self.cached_data[symbol][regime_key] = df
        
        logger.info(f"Generated {days} days of {regime.value} market data for {symbol}")
        return df
    
    def generate_multi_symbol_data(self, 
                                 symbols: List[str],
                                 regime: MarketRegimeType,
                                 days: int = 252,
                                 correlation: float = 0.7,
                                 start_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate correlated data for multiple symbols in the same regime.
        
        Args:
            symbols: List of symbols
            regime: Market regime type
            days: Number of trading days
            correlation: Base correlation between symbols
            start_date: Starting date
            
        Returns:
            Dict mapping symbols to DataFrames
        """
        results = {}
        
        # Generate base data for first symbol
        base_data = self.generate_regime_data(symbols[0], regime, days, start_date)
        results[symbols[0]] = base_data
        
        # Generate correlated data for remaining symbols
        for i, symbol in enumerate(symbols[1:], 1):
            # Each symbol gets slightly different parameters
            # to create realistic differences while maintaining correlation
            volatility = np.random.uniform(0.8, 1.2)
            
            # Generate random walk component for this symbol
            symbol_data = self.generate_regime_data(
                symbol=symbol,
                regime=regime,
                days=days,
                start_date=start_date,
                volatility=volatility
            )
            
            # Get returns from base and current symbol
            base_returns = base_data['close'].pct_change().fillna(0).values
            symbol_returns = symbol_data['close'].pct_change().fillna(0).values
            
            # Create correlated returns
            # r_correlated = correlation * r_base + (1-correlation) * r_symbol
            correlated_returns = correlation * base_returns + (1 - correlation) * symbol_returns
            
            # Reconstruct prices from correlated returns
            initial_price = symbol_data['close'].iloc[0]
            correlated_prices = [initial_price]
            
            for r in correlated_returns[1:]:
                next_price = correlated_prices[-1] * (1 + r)
                correlated_prices.append(next_price)
                
            # Update symbol dataframe with correlated prices
            symbol_data['close'] = correlated_prices
            
            # Adjust open, high, low based on new close prices
            for j in range(1, len(symbol_data)):
                prev_close = symbol_data['close'].iloc[j-1]
                curr_close = symbol_data['close'].iloc[j]
                
                # Open is based on previous close
                open_price = prev_close * (1 + np.random.normal(0, 0.003 * volatility))
                
                # Determine daily range
                daily_range = curr_close * 0.015 * volatility
                
                # Determine high and low
                is_up_day = curr_close >= open_price
                
                if is_up_day:
                    high = curr_close + np.random.uniform(0, 0.6) * daily_range
                    low = open_price - np.random.uniform(0, 0.6) * daily_range
                else:
                    high = open_price + np.random.uniform(0, 0.6) * daily_range
                    low = curr_close - np.random.uniform(0, 0.6) * daily_range
                
                # Ensure constraints
                low = min(low, open_price, curr_close)
                high = max(high, open_price, curr_close)
                
                symbol_data.at[j, 'open'] = open_price
                symbol_data.at[j, 'high'] = high
                symbol_data.at[j, 'low'] = low
            
            results[symbol] = symbol_data
        
        return results
    
    def generate_multi_regime_sequence(self,
                                     symbol: str,
                                     regime_sequence: List[Tuple[MarketRegimeType, int]],
                                     start_date: Optional[datetime] = None,
                                     initial_price: float = 100.0) -> pd.DataFrame:
        """
        Generate data with a sequence of different market regimes.
        
        Args:
            symbol: Trading symbol
            regime_sequence: List of (regime, days) tuples
            start_date: Starting date
            initial_price: Starting price
            
        Returns:
            DataFrame with the complete sequence
        """
        if start_date is None:
            total_days = sum(days for _, days in regime_sequence)
            start_date = datetime.now() - timedelta(days=total_days)
        
        current_date = start_date
        current_price = initial_price
        all_dfs = []
        
        for regime, days in regime_sequence:
            # Generate data for this regime segment
            df = self.generate_regime_data(
                symbol=symbol,
                regime=regime,
                days=days,
                start_date=current_date,
                initial_price=current_price,
                volatility=1.0
            )
            
            all_dfs.append(df)
            
            # Update start date and price for next segment
            current_date = df['date'].iloc[-1] + timedelta(days=1)
            current_price = df['close'].iloc[-1]
        
        # Concatenate all segments
        result = pd.concat(all_dfs, ignore_index=True)
        
        # Add regime labels for analysis
        result['regime'] = None
        start_idx = 0
        
        for i, (regime, days) in enumerate(regime_sequence):
            end_idx = start_idx + days
            result.loc[start_idx:end_idx-1, 'regime'] = regime.value
            start_idx = end_idx
        
        return result
    
    def get_cached_data(self, 
                      symbol: str, 
                      regime: MarketRegimeType) -> Optional[pd.DataFrame]:
        """Retrieve cached data for a symbol and regime if available"""
        if symbol in self.cached_data and regime.value in self.cached_data[symbol]:
            return self.cached_data[symbol][regime.value]
        return None
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cached_data = {}
        logger.info("Cleared market data cache")
