#!/usr/bin/env python3
"""
Synthetic Market Generator - Creates diverse market conditions for strategy evolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

class SyntheticMarketGenerator:
    """
    Generates synthetic market data with different characteristics.
    
    Features:
    - Create bull, bear, sideways, volatile markets
    - Generate market regime shifts
    - Add realistic noise patterns
    - Create correlation between assets
    """
    
    def __init__(self, output_dir: str = "synthetic_markets"):
        """
        Initialize the synthetic market generator.
        
        Args:
            output_dir: Directory to store generated data
        """
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_base_trend(self, 
                           days: int = 252, 
                           start_price: float = 100.0,
                           trend_type: str = "random",
                           trend_strength: float = 0.1) -> np.ndarray:
        """
        Generate base price trend.
        
        Args:
            days: Number of trading days
            start_price: Starting price
            trend_type: Type of trend ('bull', 'bear', 'sideways', 'random')
            trend_strength: Strength of the trend (0.0 to 1.0)
            
        Returns:
            Array of prices
        """
        # Set drift based on trend type
        if trend_type == "bull":
            drift = trend_strength * 0.1  # Daily percentage gain
        elif trend_type == "bear":
            drift = -trend_strength * 0.1  # Daily percentage loss
        elif trend_type == "sideways":
            drift = 0.0
        else:  # random
            drift = np.random.normal(0, trend_strength * 0.05)
        
        # Generate random walk with drift
        daily_returns = np.random.normal(drift, 0.01, days)
        
        # Cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Calculate prices
        prices = start_price * cumulative_returns
        
        return prices
    
    def add_volatility(self, 
                      prices: np.ndarray, 
                      volatility_level: str = "medium",
                      volatility_clustering: bool = True) -> np.ndarray:
        """
        Add volatility to price series.
        
        Args:
            prices: Base price array
            volatility_level: Level of volatility ('low', 'medium', 'high', 'extreme')
            volatility_clustering: Whether to cluster volatility (GARCH-like effect)
            
        Returns:
            Prices with added volatility
        """
        # Set volatility parameters based on level
        if volatility_level == "low":
            base_vol = 0.005
            vol_of_vol = 0.001
        elif volatility_level == "medium":
            base_vol = 0.01
            vol_of_vol = 0.003
        elif volatility_level == "high":
            base_vol = 0.02
            vol_of_vol = 0.005
        else:  # extreme
            base_vol = 0.03
            vol_of_vol = 0.01
        
        days = len(prices)
        
        if volatility_clustering:
            # Generate GARCH-like volatility clustering
            volatility = np.zeros(days)
            volatility[0] = base_vol
            
            # Parameters for volatility persistence
            alpha = 0.1  # ARCH parameter
            beta = 0.8   # GARCH parameter
            omega = base_vol * (1 - alpha - beta)  # Long-run variance
            
            for i in range(1, days):
                # Random shock
                shock = np.random.normal(0, vol_of_vol)
                
                # Update volatility based on previous volatility and random shock
                volatility[i] = (omega + 
                               alpha * (volatility[i-1] * np.random.normal(0, 1))**2 + 
                               beta * volatility[i-1] + 
                               shock)
                
                # Ensure volatility stays positive
                volatility[i] = max(0.001, volatility[i])
        else:
            # Simple random volatility without clustering
            volatility = np.random.gamma(
                base_vol / vol_of_vol, 
                vol_of_vol, 
                days
            )
        
        # Apply volatility to prices
        daily_shocks = np.random.normal(0, volatility, days)
        return prices * (1 + daily_shocks)
    
    def add_regime_shifts(self, 
                         prices: np.ndarray, 
                         num_shifts: int = 2,
                         shift_magnitude: str = "medium") -> np.ndarray:
        """
        Add market regime shifts to price series.
        
        Args:
            prices: Base price array
            num_shifts: Number of regime shifts
            shift_magnitude: Magnitude of the shifts ('small', 'medium', 'large')
            
        Returns:
            Prices with regime shifts
        """
        days = len(prices)
        
        # Set shift parameters based on magnitude
        if shift_magnitude == "small":
            max_shift = 0.05  # 5% maximum shift
        elif shift_magnitude == "medium":
            max_shift = 0.1   # 10% maximum shift
        else:  # large
            max_shift = 0.2   # 20% maximum shift
        
        # Generate random shift points
        shift_points = np.sort(np.random.choice(
            range(10, days-10), 
            size=num_shifts, 
            replace=False
        ))
        
        # Generate shift magnitudes
        shift_values = np.random.uniform(-max_shift, max_shift, num_shifts)
        
        # Apply shifts
        new_prices = prices.copy()
        
        for i, point in enumerate(shift_points):
            shift_value = shift_values[i]
            new_prices[point:] *= (1 + shift_value)
        
        return new_prices
    
    def add_seasonality(self, 
                       prices: np.ndarray, 
                       seasonality_type: str = "daily",
                       amplitude: float = 0.01) -> np.ndarray:
        """
        Add seasonality patterns to price series.
        
        Args:
            prices: Base price array
            seasonality_type: Type of seasonality ('daily', 'weekly', 'monthly')
            amplitude: Amplitude of seasonal effect
            
        Returns:
            Prices with seasonality
        """
        days = len(prices)
        
        # Set seasonality period based on type
        if seasonality_type == "daily":
            period = 1
        elif seasonality_type == "weekly":
            period = 5  # Trading days per week
        else:  # monthly
            period = 21  # Approximate trading days per month
        
        # Generate seasonal pattern
        t = np.arange(days)
        seasonality = amplitude * np.sin(2 * np.pi * t / period)
        
        # Apply seasonality
        new_prices = prices * (1 + seasonality)
        
        return new_prices
    
    def generate_ohlcv(self, close_prices: np.ndarray, volatility_level: str = "medium") -> pd.DataFrame:
        """
        Generate OHLCV data from close prices.
        
        Args:
            close_prices: Array of close prices
            volatility_level: Level of intraday volatility
            
        Returns:
            DataFrame with OHLCV data
        """
        days = len(close_prices)
        
        # Set intraday volatility based on level
        if volatility_level == "low":
            intraday_vol = 0.005
        elif volatility_level == "medium":
            intraday_vol = 0.01
        elif volatility_level == "high":
            intraday_vol = 0.02
        else:  # extreme
            intraday_vol = 0.03
        
        # Create date range
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=days)
        date_range = pd.date_range(start=start_date, periods=days, freq='B')
        
        # Initialize OHLCV data
        data = {
            'date': date_range,
            'close': close_prices
        }
        
        # Generate open, high, low
        open_prices = np.zeros(days)
        high_prices = np.zeros(days)
        low_prices = np.zeros(days)
        volume = np.zeros(days)
        
        # First day initialization
        open_prices[0] = close_prices[0] * (1 + np.random.normal(0, intraday_vol/2))
        high_prices[0] = max(open_prices[0], close_prices[0]) * (1 + abs(np.random.normal(0, intraday_vol)))
        low_prices[0] = min(open_prices[0], close_prices[0]) * (1 - abs(np.random.normal(0, intraday_vol)))
        
        # Generate subsequent days
        for i in range(1, days):
            # Open price is related to previous close
            open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, intraday_vol/2))
            
            # High and low prices
            high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + abs(np.random.normal(0, intraday_vol)))
            low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - abs(np.random.normal(0, intraday_vol)))
            
            # Ensure high >= open, close >= low
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Generate volume (higher on volatile days)
        for i in range(days):
            price_range = high_prices[i] - low_prices[i]
            volume[i] = 1000000 * (1 + 5 * price_range / close_prices[i])
            # Add random noise to volume
            volume[i] *= np.random.uniform(0.7, 1.3)
        
        # Add to data dictionary
        data['open'] = open_prices
        data['high'] = high_prices
        data['low'] = low_prices
        data['volume'] = volume.astype(int)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    def generate_bull_market(self, 
                           days: int = 252, 
                           volatility: str = "medium",
                           regime_shifts: int = 1) -> pd.DataFrame:
        """
        Generate a bull market scenario.
        
        Args:
            days: Number of trading days
            volatility: Volatility level
            regime_shifts: Number of regime shifts
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate base trend (strong upward)
        prices = self.generate_base_trend(
            days=days,
            trend_type="bull",
            trend_strength=0.8
        )
        
        # Add volatility
        prices = self.add_volatility(
            prices=prices,
            volatility_level=volatility,
            volatility_clustering=True
        )
        
        # Add minor regime shifts
        if regime_shifts > 0:
            prices = self.add_regime_shifts(
                prices=prices,
                num_shifts=regime_shifts,
                shift_magnitude="small"
            )
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(prices, volatility_level=volatility)
        
        # Save to file
        filename = f"bull_market_{days}d_{volatility}_vol_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated bull market scenario: {filepath}")
        
        return ohlcv
    
    def generate_bear_market(self, 
                           days: int = 252, 
                           volatility: str = "high",
                           crash_day: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a bear market scenario, optionally with a market crash.
        
        Args:
            days: Number of trading days
            volatility: Volatility level
            crash_day: Day of market crash (None for no crash)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate base trend (downward)
        prices = self.generate_base_trend(
            days=days,
            trend_type="bear",
            trend_strength=0.6
        )
        
        # Add high volatility
        prices = self.add_volatility(
            prices=prices,
            volatility_level=volatility,
            volatility_clustering=True
        )
        
        # Add market crash if specified
        if crash_day is not None and 0 < crash_day < days:
            # Ensure crash_day is at least a few days in
            crash_day = max(5, min(crash_day, days-5))
            
            # Apply crash (15-30% drop)
            crash_pct = np.random.uniform(0.15, 0.3)
            prices[crash_day:] *= (1 - crash_pct)
            
            # Add post-crash volatility
            post_crash = prices[crash_day:].copy()
            post_crash = self.add_volatility(
                prices=post_crash,
                volatility_level="extreme",
                volatility_clustering=True
            )
            prices[crash_day:] = post_crash
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(prices, volatility_level=volatility)
        
        # Save to file
        crash_str = f"_crash_{crash_day}" if crash_day else ""
        filename = f"bear_market_{days}d_{volatility}_vol{crash_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated bear market scenario: {filepath}")
        
        return ohlcv
    
    def generate_sideways_market(self, 
                              days: int = 252, 
                              volatility: str = "low",
                              range_width: float = 0.05) -> pd.DataFrame:
        """
        Generate a sideways (range-bound) market scenario.
        
        Args:
            days: Number of trading days
            volatility: Volatility level
            range_width: Width of the trading range as % of starting price
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate flat trend with oscillations
        prices = self.generate_base_trend(
            days=days,
            trend_type="sideways",
            trend_strength=0.05  # Very slight trend
        )
        
        # Add volatility
        prices = self.add_volatility(
            prices=prices,
            volatility_level=volatility,
            volatility_clustering=True
        )
        
        # Add oscillating pattern to create range
        t = np.arange(days)
        oscillation = np.sin(t * (2*np.pi / (days/4))) * (range_width/2)  # Multiple cycles
        prices = prices * (1 + oscillation)
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(prices, volatility_level=volatility)
        
        # Save to file
        filename = f"sideways_market_{days}d_{volatility}_vol_range{int(range_width*100)}pct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated sideways market scenario: {filepath}")
        
        return ohlcv
    
    def generate_volatile_market(self, 
                              days: int = 252, 
                              base_trend: str = "none",
                              volatility: str = "extreme") -> pd.DataFrame:
        """
        Generate a highly volatile market scenario.
        
        Args:
            days: Number of trading days
            base_trend: Underlying trend direction ('none', 'bull', 'bear')
            volatility: Volatility level
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set trend parameters
        if base_trend == "bull":
            trend_type = "bull"
            trend_strength = 0.3
        elif base_trend == "bear":
            trend_type = "bear"
            trend_strength = 0.3
        else:  # no trend
            trend_type = "sideways"
            trend_strength = 0.05
        
        # Generate base prices
        prices = self.generate_base_trend(
            days=days,
            trend_type=trend_type,
            trend_strength=trend_strength
        )
        
        # Add extreme volatility
        prices = self.add_volatility(
            prices=prices,
            volatility_level=volatility,
            volatility_clustering=True
        )
        
        # Add multiple regime shifts
        prices = self.add_regime_shifts(
            prices=prices,
            num_shifts=5,  # Multiple shifts
            shift_magnitude="large"
        )
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(prices, volatility_level=volatility)
        
        # Save to file
        filename = f"volatile_market_{base_trend}_{days}d_{volatility}_vol_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated volatile market scenario: {filepath}")
        
        return ohlcv
    
    def generate_trend_reversal(self, 
                             days: int = 252, 
                             reversal_point: Optional[int] = None,
                             initial_trend: str = "bull",
                             volatility: str = "high") -> pd.DataFrame:
        """
        Generate a market with a trend reversal.
        
        Args:
            days: Number of trading days
            reversal_point: Day of trend reversal (None for random)
            initial_trend: Initial trend direction ('bull' or 'bear')
            volatility: Volatility level
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set reversal point if not specified
        if reversal_point is None:
            # Somewhere in the middle third of the time period
            third = days // 3
            reversal_point = np.random.randint(third, 2*third)
        
        # Ensure reversal_point is within bounds
        reversal_point = max(10, min(reversal_point, days-10))
        
        # Generate pre-reversal trend
        if initial_trend == "bull":
            pre_trend_type = "bull"
            post_trend_type = "bear"
            trend_strength = 0.7
        else:  # bear initial trend
            pre_trend_type = "bear"
            post_trend_type = "bull"
            trend_strength = 0.7
        
        # Generate first segment
        prices_pre = self.generate_base_trend(
            days=reversal_point,
            trend_type=pre_trend_type,
            trend_strength=trend_strength
        )
        
        # Add volatility to first segment
        prices_pre = self.add_volatility(
            prices=prices_pre,
            volatility_level=volatility,
            volatility_clustering=True
        )
        
        # Generate second segment
        remaining_days = days - reversal_point
        prices_post = self.generate_base_trend(
            days=remaining_days,
            start_price=prices_pre[-1],
            trend_type=post_trend_type,
            trend_strength=trend_strength
        )
        
        # Add increased volatility to second segment (typical during reversals)
        post_vol = "high" if volatility != "extreme" else "extreme"
        prices_post = self.add_volatility(
            prices=prices_post,
            volatility_level=post_vol,
            volatility_clustering=True
        )
        
        # Combine segments
        prices = np.concatenate([prices_pre, prices_post])
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(prices, volatility_level=volatility)
        
        # Save to file
        filename = f"trend_reversal_{initial_trend}_to_{post_trend_type}_{days}d_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated trend reversal scenario: {filepath}")
        
        return ohlcv
    
    def generate_multiple_scenarios(self, count: int = 5, days: int = 252) -> List[pd.DataFrame]:
        """
        Generate multiple diverse market scenarios.
        
        Args:
            count: Number of scenarios to generate
            days: Number of trading days for each scenario
            
        Returns:
            List of DataFrames with OHLCV data
        """
        scenarios = []
        
        # Generate a mix of different scenario types
        scenario_types = [
            "bull", "bear", "sideways", "volatile", "trend_reversal"
        ]
        
        # Generate scenarios
        for i in range(count):
            # Select scenario type (with more weight to common scenarios)
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]  # Bull and bear markets more common
            scenario_type = np.random.choice(scenario_types, p=weights)
            
            # Generate scenario based on type
            if scenario_type == "bull":
                vol = np.random.choice(["low", "medium", "high"])
                df = self.generate_bull_market(days=days, volatility=vol)
            elif scenario_type == "bear":
                vol = np.random.choice(["medium", "high"])
                df = self.generate_bear_market(days=days, volatility=vol)
            elif scenario_type == "sideways":
                range_width = np.random.uniform(0.03, 0.08)
                df = self.generate_sideways_market(days=days, range_width=range_width)
            elif scenario_type == "volatile":
                base = np.random.choice(["none", "bull", "bear"])
                df = self.generate_volatile_market(days=days, base_trend=base)
            else:  # trend_reversal
                initial = np.random.choice(["bull", "bear"])
                df = self.generate_trend_reversal(days=days, initial_trend=initial)
            
            scenarios.append(df)
        
        return scenarios


    def generate_flash_crash(self, 
                          days: int = 252, 
                          crash_day: Optional[int] = None,
                          crash_magnitude: float = 0.15,
                          recovery_time: int = 20) -> pd.DataFrame:
        """
        Generate a market scenario with a flash crash.
        
        Args:
            days: Number of trading days
            crash_day: Day of the flash crash (None for random)
            crash_magnitude: Magnitude of the crash as percentage (0.0 to 1.0)
            recovery_time: Number of days for partial recovery
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set crash day if not specified
        if crash_day is None:
            # Somewhere in the middle half of the time period
            quarter = days // 4
            crash_day = np.random.randint(quarter, 3*quarter)
        
        # Generate base trend (slight upward bias)
        prices = self.generate_base_trend(
            days=days,
            trend_type="bull",
            trend_strength=0.3
        )
        
        # Add normal volatility
        prices = self.add_volatility(
            prices=prices,
            volatility_level="medium",
            volatility_clustering=True
        )
        
        # Ensure crash_day is within bounds
        crash_day = max(10, min(crash_day, days-recovery_time-5))
        
        # Create flash crash
        # 1. Steep one-day decline
        prices[crash_day] *= (1 - crash_magnitude)
        
        # 2. Add extreme volatility during recovery
        recovery_segment = prices[crash_day:crash_day+recovery_time].copy()
        recovery_segment = self.add_volatility(
            prices=recovery_segment,
            volatility_level="extreme",
            volatility_clustering=True
        )
        
        # 3. Model partial recovery (60-80% of crash)
        recovery_pct = np.random.uniform(0.6, 0.8)
        final_recovery = prices[crash_day-1] * (1 - crash_magnitude * (1 - recovery_pct))
        
        # Linear recovery trajectory with noise
        start_price = prices[crash_day]
        price_diff = final_recovery - start_price
        for i in range(recovery_time):
            recovery_pct = i / (recovery_time - 1) if recovery_time > 1 else 1
            target_price = start_price + (price_diff * recovery_pct)
            # Add noise around target
            noise = np.random.normal(0, 0.01)
            recovery_segment[i] = target_price * (1 + noise)
        
        # Replace original prices with recovery segment
        prices[crash_day:crash_day+recovery_time] = recovery_segment
        
        # Generate OHLCV data with higher intraday vol on crash day
        ohlcv = self.generate_ohlcv(prices, volatility_level="high")
        
        # Modify crash day specifically for higher range
        crash_idx = ohlcv.index[crash_day]
        ohlcv.loc[crash_idx, 'low'] = ohlcv.loc[crash_idx, 'close'] * (1 - crash_magnitude * 1.1)
        
        # Save to file
        filename = f"flash_crash_{days}d_{int(crash_magnitude*100)}pct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated flash crash scenario: {filepath}")
        
        return ohlcv
    
    def generate_liquidity_crisis(self, 
                              days: int = 252, 
                              crisis_start: Optional[int] = None,
                              crisis_duration: int = 40,
                              severity: str = "high") -> pd.DataFrame:
        """
        Generate a market scenario with a liquidity crisis (sustained drawdown with high volatility).
        
        Args:
            days: Number of trading days
            crisis_start: Day when crisis starts (None for random)
            crisis_duration: Duration of the crisis in days
            severity: Severity of the crisis ('medium', 'high', 'extreme')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set crisis parameters
        if crisis_start is None:
            # Start in first half of time period
            crisis_start = np.random.randint(30, days//2)
        
        # Ensure parameters are within bounds
        crisis_start = max(20, min(crisis_start, days-crisis_duration-10))
        crisis_duration = min(crisis_duration, days-crisis_start-5)
        
        # Set severity parameters
        if severity == "medium":
            decline_rate = 0.003  # Daily decline rate
            vol_level = "high"
            gap_down_prob = 0.2   # Probability of gap downs
            gap_down_size = 0.03  # Typical gap down size
        elif severity == "high":
            decline_rate = 0.005
            vol_level = "high"
            gap_down_prob = 0.3
            gap_down_size = 0.05
        else:  # extreme
            decline_rate = 0.007
            vol_level = "extreme"
            gap_down_prob = 0.4
            gap_down_size = 0.07
        
        # Generate three segments: pre-crisis, crisis, post-crisis
        # 1. Pre-crisis: slight uptrend
        pre_crisis = self.generate_base_trend(
            days=crisis_start,
            trend_type="bull",
            trend_strength=0.4
        )
        pre_crisis = self.add_volatility(
            prices=pre_crisis,
            volatility_level="low",
            volatility_clustering=True
        )
        
        # 2. Crisis: declining with high volatility
        crisis_base = np.cumprod(
            1 - np.random.normal(decline_rate, decline_rate/2, crisis_duration)
        )
        crisis_segment = pre_crisis[-1] * crisis_base
        
        # Add gap downs
        for i in range(1, len(crisis_segment)):
            if np.random.random() < gap_down_prob:
                # Create a gap down
                gap_size = np.random.normal(gap_down_size, gap_down_size/3)
                gap_size = max(0.01, gap_size)  # Ensure positive gap
                crisis_segment[i:] *= (1 - gap_size)
        
        # Add high volatility
        crisis_segment = self.add_volatility(
            prices=crisis_segment,
            volatility_level=vol_level,
            volatility_clustering=True
        )
        
        # 3. Post-crisis: recovery
        post_crisis_days = days - crisis_start - crisis_duration
        post_crisis = self.generate_base_trend(
            days=post_crisis_days,
            start_price=crisis_segment[-1],
            trend_type="bull",
            trend_strength=0.5
        )
        post_crisis = self.add_volatility(
            prices=post_crisis,
            volatility_level="medium",
            volatility_clustering=True
        )
        
        # Combine all segments
        prices = np.concatenate([pre_crisis, crisis_segment, post_crisis])
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(prices)
        
        # During crisis, increase volume (liquidity drying up and panic selling)
        crisis_indices = ohlcv.index[crisis_start:crisis_start+crisis_duration]
        for idx in crisis_indices:
            # Increase volume by 50%-150%
            volume_multiplier = np.random.uniform(1.5, 2.5)
            ohlcv.loc[idx, 'volume'] = int(ohlcv.loc[idx, 'volume'] * volume_multiplier)
        
        # Save to file
        filename = f"liquidity_crisis_{days}d_{severity}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        ohlcv.to_csv(filepath)
        
        print(f"Generated liquidity crisis scenario: {filepath}")
        
        return ohlcv
    
    def generate_sector_rotation(self, 
                              days: int = 252, 
                              sectors: int = 4,
                              rotation_frequency: str = "medium") -> List[pd.DataFrame]:
        """
        Generate multiple correlated datasets to simulate sector rotation.
        
        Args:
            days: Number of trading days
            sectors: Number of sectors to generate
            rotation_frequency: How often rotations occur ('low', 'medium', 'high')
            
        Returns:
            List of DataFrames with OHLCV data for each sector
        """
        # Set rotation parameters
        if rotation_frequency == "low":
            # Infrequent rotations (1-2 per year)
            min_period = days // 3
            max_period = days // 2
        elif rotation_frequency == "medium":
            # Moderate rotations (3-4 per year)
            min_period = days // 6
            max_period = days // 3
        else:  # high
            # Frequent rotations
            min_period = days // 12
            max_period = days // 4
        
        # Generate market base trend (overall market)
        market_base = self.generate_base_trend(
            days=days,
            trend_type="bull",
            trend_strength=0.3
        )
        market_base = self.add_volatility(
            prices=market_base,
            volatility_level="medium",
            volatility_clustering=True
        )
        
        # Generate sector datasets
        sector_data = []
        sector_names = [f"sector_{i+1}" for i in range(sectors)]
        
        # First, generate base price for each sector with correlation to market
        sector_bases = []
        for i in range(sectors):
            # Correlation with market (0.5 to 0.9)
            market_correlation = np.random.uniform(0.5, 0.9)
            
            # Create correlated noise
            noise = np.random.normal(0, 0.01, days)
            sector_return = market_correlation * (market_base / market_base[0] - 1) + (1 - market_correlation) * np.cumsum(noise)
            
            # Create base prices (start at 100)
            base = 100 * (1 + sector_return)
            sector_bases.append(base)
        
        # Create rotation periods
        current_day = 0
        periods = []
        
        while current_day < days:
            period_length = np.random.randint(min_period, max_period+1)
            if current_day + period_length > days:
                period_length = days - current_day
            
            # Randomly select 1-2 outperforming sectors and 1-2 underperforming sectors
            outperform_count = min(np.random.randint(1, 3), sectors // 2)
            underperform_count = min(np.random.randint(1, 3), sectors // 2)
            
            outperformers = np.random.choice(range(sectors), outperform_count, replace=False)
            remaining = [i for i in range(sectors) if i not in outperformers]
            underperformers = np.random.choice(remaining, underperform_count, replace=False)
            
            periods.append({
                "start": current_day,
                "end": current_day + period_length,
                "outperformers": outperformers,
                "underperformers": underperformers
            })
            
            current_day += period_length
        
        # Apply rotation effects
        for period in periods:
            start, end = period["start"], period["end"]
            
            # Apply outperformance (10-30% annualized)
            for idx in period["outperformers"]:
                outperform_rate = np.random.uniform(0.1, 0.3) / 252  # Daily rate
                multiplier = np.cumprod(1 + np.random.normal(outperform_rate, outperform_rate/3, end-start))
                sector_bases[idx][start:end] *= multiplier
            
            # Apply underperformance (-5% to -20% annualized)
            for idx in period["underperformers"]:
                underperform_rate = np.random.uniform(0.05, 0.2) / 252  # Daily rate
                multiplier = np.cumprod(1 - np.random.normal(underperform_rate, underperform_rate/3, end-start))
                sector_bases[idx][start:end] *= multiplier
        
        # Generate full OHLCV data for each sector
        for i, base in enumerate(sector_bases):
            # Add specific volatility
            sector_vol = np.random.choice(["low", "medium", "high"])
            ohlcv = self.generate_ohlcv(base, volatility_level=sector_vol)
            
            # Save to file
            filename = f"sector_rotation_{sector_names[i]}_{days}d_{rotation_frequency}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            ohlcv.to_csv(filepath)
            
            sector_data.append(ohlcv)
        
        print(f"Generated sector rotation scenario with {sectors} sectors: {self.output_dir}")
        
        return sector_data
    
    def generate_advanced_scenarios(self, count: int = 5, days: int = 252) -> List[pd.DataFrame]:
        """
        Generate a mix of advanced market scenarios for comprehensive strategy testing.
        
        Args:
            count: Number of scenarios to generate
            days: Number of trading days for each scenario
            
        Returns:
            List of DataFrames with OHLCV data
        """
        scenarios = []
        
        # Define scenario types
        advanced_types = [
            "flash_crash",
            "liquidity_crisis",
            "sector_rotation",  # Will create multiple datasets
            "bull",
            "bear",
            "volatile",
            "trend_reversal"
        ]
        
        # Generate scenarios with weights favoring advanced scenarios
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.075, 0.075]  # Higher weights for advanced scenarios
        
        for i in range(count):
            scenario_type = np.random.choice(advanced_types, p=weights)
            
            if scenario_type == "flash_crash":
                crash_magnitude = np.random.uniform(0.08, 0.20)  # 8-20% crash
                df = self.generate_flash_crash(
                    days=days,
                    crash_magnitude=crash_magnitude
                )
                scenarios.append(df)
                
            elif scenario_type == "liquidity_crisis":
                severity = np.random.choice(["medium", "high", "extreme"])
                df = self.generate_liquidity_crisis(
                    days=days,
                    severity=severity
                )
                scenarios.append(df)
                
            elif scenario_type == "sector_rotation":
                # Will create 3-5 sector datasets
                sector_count = np.random.randint(3, 6)
                frequency = np.random.choice(["low", "medium", "high"])
                sector_dfs = self.generate_sector_rotation(
                    days=days,
                    sectors=sector_count,
                    rotation_frequency=frequency
                )
                # Add just one randomly selected sector
                scenarios.append(np.random.choice(sector_dfs))
                
            elif scenario_type == "bull":
                vol = np.random.choice(["low", "medium", "high"])
                df = self.generate_bull_market(days=days, volatility=vol)
                scenarios.append(df)
                
            elif scenario_type == "bear":
                vol = np.random.choice(["medium", "high", "extreme"])
                crash_day = np.random.randint(days//3, 2*days//3) if np.random.random() < 0.5 else None
                df = self.generate_bear_market(days=days, volatility=vol, crash_day=crash_day)
                scenarios.append(df)
                
            elif scenario_type == "volatile":
                base = np.random.choice(["none", "bull", "bear"])
                df = self.generate_volatile_market(days=days, base_trend=base)
                scenarios.append(df)
                
            else:  # trend_reversal
                initial = np.random.choice(["bull", "bear"])
                df = self.generate_trend_reversal(days=days, initial_trend=initial)
                scenarios.append(df)
        
        return scenarios


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic market data")
    parser.add_argument(
        "--type", 
        type=str, 
        default="mix",
        choices=["bull", "bear", "sideways", "volatile", "reversal", "flash_crash", 
                "liquidity_crisis", "sector_rotation", "advanced", "mix"],
        help="Type of market scenario to generate"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=252,
        help="Number of trading days"
    )
    parser.add_argument(
        "--count", 
        type=int, 
        default=5,
        help="Number of scenarios to generate (for 'mix' or 'advanced' type)"
    )
    parser.add_argument(
        "--volatility", 
        type=str, 
        default="medium",
        choices=["low", "medium", "high", "extreme"],
        help="Volatility level"
    )
    parser.add_argument(
        "--sectors",
        type=int,
        default=4,
        help="Number of sectors for sector rotation scenario"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="synthetic_markets",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticMarketGenerator(output_dir=args.output)
    
    # Generate scenarios based on type
    if args.type == "bull":
        generator.generate_bull_market(
            days=args.days, 
            volatility=args.volatility
        )
    elif args.type == "bear":
        generator.generate_bear_market(
            days=args.days, 
            volatility=args.volatility
        )
    elif args.type == "sideways":
        generator.generate_sideways_market(
            days=args.days, 
            volatility=args.volatility
        )
    elif args.type == "volatile":
        generator.generate_volatile_market(
            days=args.days, 
            volatility=args.volatility
        )
    elif args.type == "reversal":
        initial = np.random.choice(["bull", "bear"])
        generator.generate_trend_reversal(
            days=args.days, 
            initial_trend=initial,
            volatility=args.volatility
        )
    elif args.type == "flash_crash":
        generator.generate_flash_crash(
            days=args.days,
            crash_magnitude=np.random.uniform(0.1, 0.2)
        )
    elif args.type == "liquidity_crisis":
        generator.generate_liquidity_crisis(
            days=args.days,
            severity=args.volatility
        )
    elif args.type == "sector_rotation":
        generator.generate_sector_rotation(
            days=args.days,
            sectors=args.sectors,
            rotation_frequency=args.volatility
        )
    elif args.type == "advanced":
        scenarios = generator.generate_advanced_scenarios(
            count=args.count,
            days=args.days
        )
        print(f"Generated {len(scenarios)} advanced market scenarios")
    else:  # mix
        scenarios = generator.generate_multiple_scenarios(
            count=args.count,
            days=args.days
        )
        print(f"Generated {len(scenarios)} diverse market scenarios")
        
    print("Done!")

