"""
Cryptocurrency Technical Indicators Suite

This module provides specialized technical indicators optimized for cryptocurrency markets,
with support for both traditional indicators and crypto-specific analytics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union, List, Tuple, Callable, Any, Set
from dataclasses import dataclass, field
from functools import lru_cache
import warnings
import time
from datetime import datetime
from types import SimpleNamespace

# Try to import TA-Lib if available, otherwise use TA
try:
    import talib
    USE_TALIB = True
except ImportError:
    import ta
    USE_TALIB = False
    warnings.warn("TA-Lib not found, using TA-Lib alternative. Consider installing TA-Lib for better performance.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    # General settings
    default_length: int = 14
    enable_adaptive_parameters: bool = False
    cache_calculations: bool = True
    max_cache_size: int = 64  # LRU cache size for expensive calculations
    
    # Basic indicator parameters
    fast_rsi_window: int = 10
    donchian_window: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_dev: float = 2.0
    tsi_long_window: int = 25
    tsi_short_window: int = 13
    volume_profile_buckets: int = 10
    
    # Regime detection
    regime_lookback: int = 90
    mean_reversion_window: int = 20
    
    # Signal generation
    require_confirmation: bool = True
    min_signal_strength: int = 60
    use_multi_timeframe: bool = False
    
    # Advanced features
    detect_anomalies: bool = True
    outlier_std_threshold: float = 3.0
    apply_smoothing: bool = False
    noise_filter_level: float = 0.1
    min_volume_percentile: int = 25  # Minimum volume percentile for valid signals
    
    # Crypto-specific
    account_for_funding: bool = True
    detect_whale_activity: bool = True
    whale_volume_multiplier: float = 3.0  # Multiple of average volume to detect whale activity


class CryptoIndicatorSuite:
    """Suite of technical indicators optimized for cryptocurrency markets."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the indicator suite.
        
        Args:
            config: Configuration options for indicators
        """
        user_config = config or {}
        self.config = IndicatorConfig(**user_config)
        self._indicator_metadata = {}
        self._calculation_time = {}
        self._dependencies = self._build_dependencies()
        
        # Configure caching if enabled
        if self.config.cache_calculations:
            # Apply cache decorator to expensive methods
            self._calculate_tsi = lru_cache(maxsize=self.config.max_cache_size)(self._calculate_tsi)
            self._calculate_historical_volatility = lru_cache(maxsize=self.config.max_cache_size)(self._calculate_historical_volatility)
            self._add_supertrend = lru_cache(maxsize=self.config.max_cache_size)(self._add_supertrend)

    def _calculate_historical_volatility(self, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate annualized historical volatility of a price series.
        Args:
            close: Series of closing prices
            window: Lookback window for volatility (default 20)
        Returns:
            Annualized volatility as a pandas Series (in percent)
        """
        log_returns = np.log(close / close.shift(1))
        rolling_std = log_returns.rolling(window=window, min_periods=window).std()
        hist_vol = rolling_std * np.sqrt(252) * 100  # Annualize (252 trading days)
        return hist_vol

    
    def _build_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build dependency graph for indicators to optimize calculation order.
        
        Returns:
            Dictionary mapping indicator categories to their dependencies
        """
        dependencies = {
            'volume': set(),  # No dependencies
            'momentum': set(),  # No dependencies
            'volatility': set(),  # No dependencies
            'crypto_specific': {'volatility'},  # Depends on volatility indicators (e.g., NATR)
            'trend': {'volatility'},  # Depends on volatility (e.g., ATR for SuperTrend)
            'signals': {'momentum', 'volatility', 'trend', 'crypto_specific'}  # Depends on all indicator types
        }
        return dependencies
    
    def add_all_indicators(self, df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
        """
        Add all relevant indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data (must contain 'open', 'high', 'low', 'close', 'volume' columns)
            show_progress: Whether to print progress information
            
        Returns:
            DataFrame with added indicators
        
        Raises:
            ValueError: If required columns are missing
        """
        self._validate_dataframe(df)
        
        # Record original shape for validation
        original_shape = df.shape
        
        # Check for data anomalies if enabled
        if self.config.detect_anomalies:
            df = self._detect_and_handle_anomalies(df)
        
        # Reset calculation times
        self._calculation_time = {}
        
        # Apply indicators by category, respecting dependencies
        categories = {
            'volume': self.add_volume_indicators,
            'momentum': self.add_momentum_indicators,
            'volatility': self.add_volatility_indicators,
            'crypto_specific': self.add_crypto_specific_indicators,
            'trend': self.add_trend_indicators
        }
        
        # Process categories in dependency order
        for category, func in self._sorted_by_dependencies(categories):
            start_time = time.time()
            if show_progress:
                print(f"Calculating {category} indicators...")
            
            # Apply the indicator function
            df = func(df)
            
            # Record calculation time
            self._calculation_time[category] = time.time() - start_time
            
            if show_progress:
                print(f"  ✓ Done in {self._calculation_time[category]:.2f} seconds")
        
        # Verify no rows were lost during processing
        assert df.shape[0] == original_shape[0], "Row count changed during indicator calculation"
        
        if show_progress:
            print(f"All indicators calculated successfully. Total indicators: {len(df.columns) - 5}")
        
        return df
    
    def _sorted_by_dependencies(self, categories: Dict[str, Callable]) -> List[Tuple[str, Callable]]:
        """
        Sort categories by their dependencies for efficient calculation order.
        
        Args:
            categories: Dictionary of category names to indicator functions
            
        Returns:
            Sorted list of (category, function) tuples
        """
        # Track processed categories
        processed = set()
        result = []
        
        def process_category(category):
            # Skip if already processed
            if category in processed:
                return
            
            # Process dependencies first
            for dependency in self._dependencies.get(category, set()):
                if dependency in categories:
                    process_category(dependency)
            
            # Add this category
            processed.add(category)
            result.append((category, categories[category]))
        
        # Process all categories
        for category in categories:
            process_category(category)
        
        return result
        
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Check for NaN values in key columns
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            nan_info = ", ".join([f"{col}: {count}" for col, count in nan_counts.items() if count > 0])
            logger.warning(f"NaN values detected in OHLCV data: {nan_info}. Some indicators may return incomplete results.")
        
        # Check for data quality issues
        if (df['high'] < df['low']).any():
            logger.error("Data quality issue: 'high' values less than 'low' values detected")
            
        if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            logger.error("Data quality issue: 'close' values outside high-low range detected")
            
        # Check for zero or negative values
        if (df['volume'] <= 0).any():
            zero_volume_count = (df['volume'] <= 0).sum()
            logger.warning(f"Data quality issue: {zero_volume_count} rows with zero or negative volume detected")
            
        # Check for insufficient data
        min_data_points = max(
            self.config.default_length * 2,
            self.config.macd_slow * 2,
            100  # Reasonable minimum for statistical calculations
        )
        
        if len(df) < min_data_points:
            logger.warning(f"Insufficient data: at least {min_data_points} rows recommended for reliable indicators")
    
    def _detect_and_handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle anomalies in the data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with anomalies handled
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Handle extreme price anomalies (likely data errors)
        # Calculate rolling statistics
        rolling_median = result['close'].rolling(window=20, min_periods=5).median()
        rolling_std = result['close'].rolling(window=20, min_periods=5).std()
        
        # Define outlier threshold
        threshold = self.config.outlier_std_threshold
        
        # Identify outliers
        upper_bound = rolling_median + (rolling_std * threshold)
        lower_bound = rolling_median - (rolling_std * threshold)
        
        # Check for price outliers
        price_outliers = ((result['close'] > upper_bound) | (result['close'] < lower_bound))
        outlier_count = price_outliers.sum()
        
        if outlier_count > 0:
            logger.warning(f"Detected {outlier_count} potential price outliers")
            
            # Flag outliers
            result['is_price_outlier'] = price_outliers
            
            # Optionally replace extreme outliers with the median value
            # Note: We're not replacing by default as it could hide important events
            if self.config.apply_smoothing:
                logger.info("Smoothing outliers...")
                result.loc[price_outliers, 'close'] = rolling_median[price_outliers]
        
        # Check for volume anomalies (very large spikes)
        rolling_vol_median = result['volume'].rolling(window=20, min_periods=5).median()
        rolling_vol_std = result['volume'].rolling(window=20, min_periods=5).std()
        
        volume_upper_bound = rolling_vol_median + (rolling_vol_std * threshold)
        volume_outliers = (result['volume'] > volume_upper_bound)
        
        if volume_outliers.sum() > 0:
            # Mark potential whale activity
            result['potential_whale_activity'] = volume_outliers & (result['volume'] > rolling_vol_median * self.config.whale_volume_multiplier)
            
            # Flag volumes that are likely data errors (extremely high)
            extreme_volume = result['volume'] > rolling_vol_median * 10
            result['extreme_volume'] = extreme_volume
            
            if extreme_volume.sum() > 0 and self.config.apply_smoothing:
                logger.info(f"Smoothing {extreme_volume.sum()} extreme volume values")
                result.loc[extreme_volume, 'volume'] = rolling_vol_median[extreme_volume]
        
        return result
    
    def register_indicator_metadata(self, name: str, metadata: Dict) -> None:
        """
        Register metadata for an indicator to provide additional context.
        
        Args:
            name: Indicator column name
            metadata: Dictionary with metadata like:
                - description: Text description of the indicator
                - interpretation: How to interpret the values
                - range: Expected range of values (e.g., [0, 100] for RSI)
                - references: Links or citations for further reading
        """
        self._indicator_metadata[name] = metadata
    
    def get_indicator_metadata(self, name: Optional[str] = None) -> Dict:
        """
        Get metadata for indicators.
        
        Args:
            name: Specific indicator name, or None for all metadata
            
        Returns:
            Dictionary with indicator metadata
        """
        if name:
            return self._indicator_metadata.get(name, {})
        return self._indicator_metadata
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # On-Balance Volume (OBV)
        if USE_TALIB:
            result['obv'] = talib.OBV(result['close'], result['volume'])
        else:
            result['obv'] = ta.volume.on_balance_volume(result['close'], result['volume'])
        
        self.register_indicator_metadata('obv', {
            'description': 'On-Balance Volume',
            'interpretation': 'Cumulative indicator using volume flow to predict price changes',
            'references': 'https://www.investopedia.com/terms/o/onbalancevolume.asp'
        })
        
        # Normalize OBV for easier interpretation
        obv_mean = result['obv'].rolling(window=50).mean()
        obv_std = result['obv'].rolling(window=50).std()
        result['obv_normalized'] = (result['obv'] - obv_mean) / obv_std
        
        self.register_indicator_metadata('obv_normalized', {
            'description': 'Normalized On-Balance Volume',
            'interpretation': 'OBV normalized to standard deviations from mean',
            'range': [-3, 3],
            'references': 'Statistical normalization of OBV'
        })
        
        # Volume Weighted Average Price (VWAP)
        # Only calculate if we have a datetime index
        if isinstance(result.index, pd.DatetimeIndex):
            try:
                if USE_TALIB:
                    # TA-Lib doesn't have VWAP, so we'll use our own implementation
                    day_labels = result.index.date
                    result['vwap'] = np.nan
                    
                    for day in pd.unique(day_labels):
                        mask = (day_labels == day)
                        if mask.sum() > 0:
                            cumulative_vol = result.loc[mask, 'volume'].cumsum()
                            cumulative_vol_price = (result.loc[mask, 'volume'] * ((result.loc[mask, 'high'] + result.loc[mask, 'low'] + result.loc[mask, 'close']) / 3)).cumsum()
                            result.loc[mask, 'vwap'] = cumulative_vol_price / cumulative_vol
                else:
                    result['vwap'] = ta.volume.volume_weighted_average_price(
                        high=result['high'], 
                        low=result['low'], 
                        close=result['close'], 
                        volume=result['volume']
                    )
                
                self.register_indicator_metadata('vwap', {
                    'description': 'Volume Weighted Average Price',
                    'interpretation': 'Average price weighted by volume, used as support/resistance',
                    'references': 'https://www.investopedia.com/terms/v/vwap.asp'
                })
                
                # VWAP-based bands
                if not result['vwap'].isna().all():
                    # Calculate standard deviation of price from VWAP
                    vwap_dev = ((result['close'] - result['vwap']) ** 2).rolling(window=20).mean().pow(0.5)
                    result['vwap_upper'] = result['vwap'] + (vwap_dev * 2)
                    result['vwap_lower'] = result['vwap'] - (vwap_dev * 2)
                    
                    self.register_indicator_metadata('vwap_upper', {
                        'description': 'VWAP Upper Band',
                        'interpretation': 'Upper deviation band from VWAP, potential resistance',
                        'references': 'VWAP Bands strategy'
                    })
            except Exception as e:
                logger.warning(f"Failed to calculate VWAP: {e}")
                result['vwap'] = np.nan
        
        # Money Flow Index (MFI)
        if USE_TALIB:
            result['mfi'] = talib.MFI(
                result['high'], 
                result['low'], 
                result['close'], 
                result['volume'], 
                timeperiod=self.config.default_length
            )
        else:
            result['mfi'] = ta.volume.money_flow_index(
                high=result['high'],
                low=result['low'],
                close=result['close'],
                volume=result['volume'],
                window=self.config.default_length
            )
            
        self.register_indicator_metadata('mfi', {
            'description': 'Money Flow Index',
            'interpretation': 'Volume-weighted RSI, values above 80 overbought, below 20 oversold',
            'range': [0, 100],
            'references': 'https://www.investopedia.com/terms/m/mfi.asp'
        })
        
        # MFI divergence with price
        result['mfi_divergence'] = self._calculate_divergence(result['close'], result['mfi'], bullish_threshold=20, bearish_threshold=80)
        
        self.register_indicator_metadata('mfi_divergence', {
            'description': 'MFI Divergence',
            'interpretation': '1 for bullish divergence, -1 for bearish divergence, 0 for no divergence',
            'range': [-1, 0, 1],
            'references': 'Technical divergence analysis'
        })
        
        # Chaikin Money Flow (CMF)
        if not USE_TALIB:
            result['cmf'] = ta.volume.chaikin_money_flow(
                high=result['high'],
                low=result['low'],
                close=result['close'],
                volume=result['volume'],
                window=self.config.default_length
            )
            
            self.register_indicator_metadata('cmf', {
                'description': 'Chaikin Money Flow',
                'interpretation': 'Measures buying/selling pressure, values above 0 indicate accumulation',
                'range': [-1, 1],
                'references': 'https://www.investopedia.com/terms/c/chaikinmoneyflow.asp'
            })
        
        # Volume Force Index (detects smart money movement)
        result['force_index'] = result['close'].diff() * result['volume']
        result['force_index_ema13'] = result['force_index'].ewm(span=13, adjust=False).mean()
        
        self.register_indicator_metadata('force_index_ema13', {
            'description': 'Force Index (13-period EMA)',
            'interpretation': 'Measures power behind price moves, sign changes can signal reversal',
            'references': 'Dr. Alexander Elder'
        })
        
        # Relative Volume
        window = 20
        result['relative_volume'] = result['volume'] / result['volume'].rolling(window=window).mean()
        
        self.register_indicator_metadata('relative_volume', {
            'description': 'Relative Volume',
            'interpretation': 'Current volume relative to average, values > 1 indicate higher than average volume',
            'references': 'Volume analysis'
        })
        
        # Add new Volume Price Trend (VPT) indicator
        result['vpt'] = self._calculate_vpt(result)
        
        self.register_indicator_metadata('vpt', {
            'description': 'Volume Price Trend',
            'interpretation': 'Cumulative volume-price trend, trend shows buying/selling pressure',
            'references': 'https://www.investopedia.com/terms/v/vpt.asp'
        })
        
        # Volume Reversal Detection
        result['volume_reversal'] = self._detect_volume_climax(result)
        
        self.register_indicator_metadata('volume_reversal', {
            'description': 'Volume Reversal Signal',
            'interpretation': '1 for volume climax up, -1 for volume climax down, 0 for no signal',
            'range': [-1, 0, 1],
            'references': 'Volume climax analysis'
        })
        
        return result
    
    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Price Trend (VPT) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VPT values
        """
        close = df['close']
        volume = df['volume']
        
        # Calculate percentage price change
        price_change_pct = close.pct_change()
        
        # First element is NaN, so we'll start with 0
        vpt = pd.Series(0, index=close.index)
        
        # Calculate VPT
        for i in range(1, len(close)):
            vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * price_change_pct.iloc[i]
        
        return vpt
    
    def _detect_volume_climax(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Detect volume climax events (potential reversal points).
        
        Args:
            df: DataFrame with OHLCV data
            window: Lookback window
            
        Returns:
            Series with volume climax signals (1 for up, -1 for down, 0 for none)
        """
        # Define high volume threshold (top 5% of recent volume)
        high_volume = df['volume'] > df['volume'].rolling(window).quantile(0.95)
        
        # Define price and volume conditions
        up_day = df['close'] > df['open']
        down_day = df['close'] < df['open']
        
        # Check for price exhaustion (gap up followed by close near low or gap down followed by close near high)
        gap_up = df['low'] > df['high'].shift(1)
        gap_down = df['high'] < df['low'].shift(1)
        
        close_near_low = (df['close'] - df['low']) < (df['high'] - df['low']) * 0.25
        close_near_high = (df['high'] - df['close']) < (df['high'] - df['low']) * 0.25
        
        # Volume climax conditions
        upside_climax = high_volume & up_day & gap_up & close_near_low
        downside_climax = high_volume & down_day & gap_down & close_near_high
        
        # Create signal series
        signal = pd.Series(0, index=df.index)
        signal.loc[upside_climax] = 1
        signal.loc[downside_climax] = -1
        
        return signal
    
    def _calculate_divergence(self, price: pd.Series, indicator: pd.Series, 
                             bullish_threshold: float = 30, bearish_threshold: float = 70,
                             window: int = 10) -> pd.Series:
        """
        Detect divergence between price and indicator.
        
        Args:
            price: Price series
            indicator: Indicator series
            bullish_threshold: Threshold for bullish divergence (oversold)
            bearish_threshold: Threshold for bearish divergence (overbought)
            window: Lookback window for extrema
            
        Returns:
            Series with divergence signals (1 for bullish, -1 for bearish, 0 for none)
        """
        # Initialize result
        divergence = pd.Series(0, index=price.index)
        
        # Find local price extrema
        price_highs = price.rolling(window=window, center=True).apply(
            lambda x: 1 if (x.iloc[window//2] == max(x)) else 0
        )
        
        price_lows = price.rolling(window=window, center=True).apply(
            lambda x: 1 if (x.iloc[window//2] == min(x)) else 0
        )
        
        # Find local indicator extrema
        indicator_highs = indicator.rolling(window=window, center=True).apply(
            lambda x: 1 if (x.iloc[window//2] == max(x)) else 0
        )
        
        indicator_lows = indicator.rolling(window=window, center=True).apply(
            lambda x: 1 if (x.iloc[window//2] == min(x)) else 0
        )
        
        # Check for bullish divergence (lower price lows but higher indicator lows)
        for i in range(window, len(price) - window):
            if price_lows.iloc[i] == 1 and indicator.iloc[i] < bullish_threshold:
                # Look back for another price low
                for j in range(i-window, i):
                    if price_lows.iloc[j] == 1:
                        if (price.iloc[i] < price.iloc[j]) and (indicator.iloc[i] > indicator.iloc[j]):
                            divergence.iloc[i] = 1
                        break
        
        # Check for bearish divergence (higher price highs but lower indicator highs)
        for i in range(window, len(price) - window):
            if price_highs.iloc[i] == 1 and indicator.iloc[i] > bearish_threshold:
                # Look back for another price high
                for j in range(i-window, i):
                    if price_highs.iloc[j] == 1:
                        if (price.iloc[i] > price.iloc[j]) and (indicator.iloc[i] < indicator.iloc[j]):
                            divergence.iloc[i] = -1
                        break
        
        return divergence
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators optimized for crypto markets.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added momentum indicators
        """
        result = df.copy()
        # ... (existing code for momentum indicators)
        # Calculate TSI (True Strength Index)
        result['tsi'] = self._calculate_tsi(result['close'])
        self.register_indicator_metadata('tsi', {
            'description': 'True Strength Index',
            'interpretation': 'Momentum oscillator, values above 25 bullish, below -25 bearish',
            'range': [-100, 100],
            'references': 'https://www.investopedia.com/terms/t/tsi.asp'
        })
        return result

    def _calculate_tsi(self, close: pd.Series, r: int = None, s: int = None) -> pd.Series:
        """
        Calculate the True Strength Index (TSI) for a price series.
        Args:
            close: Series of closing prices
            r: Long EMA window (default from config if None)
            s: Short EMA window (default from config if None)
        Returns:
            TSI values as a pandas Series
        """
        r = r if r is not None else self.config.tsi_long_window
        s = s if s is not None else self.config.tsi_short_window
        diff = close.diff()
        abs_diff = diff.abs()
        double_smoothed_diff = diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
        double_smoothed_abs = abs_diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
        tsi = 100 * (double_smoothed_diff / double_smoothed_abs)
        return tsi

        """
        Add momentum indicators optimized for crypto markets.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added momentum indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Determine optimal RSI parameters if adaptive config is enabled
        if self.config.enable_adaptive_parameters:
            fast_rsi_window = self._optimize_rsi_period(result)
            logger.info(f"Using adaptive RSI period: {fast_rsi_window}")
        else:
            fast_rsi_window = self.config.fast_rsi_window
        
        # RSI with standard and faster settings
        if USE_TALIB:
            result['rsi'] = talib.RSI(result['close'], timeperiod=self.config.default_length)
            result['fast_rsi'] = talib.RSI(result['close'], timeperiod=fast_rsi_window)
        else:
            result['rsi'] = ta.momentum.rsi(result['close'], window=self.config.default_length)
            result['fast_rsi'] = ta.momentum.rsi(result['close'], window=fast_rsi_window)
            
        self.register_indicator_metadata('rsi', {
            'description': 'Relative Strength Index',
            'interpretation': 'Values above 70 considered overbought, below 30 oversold',
            'range': [0, 100],
            'references': 'https://www.investopedia.com/terms/r/rsi.asp'
        })
        
        self.register_indicator_metadata('fast_rsi', {
            'description': 'Fast Relative Strength Index',
            'interpretation': 'More responsive RSI for volatile markets',
            'range': [0, 100],
            'references': 'https://www.investopedia.com/terms/r/rsi.asp'
        })
        
        # Add RSI divergence detection
        result['rsi_divergence'] = self._calculate_divergence(result['close'], result['rsi'])
        
        self.register_indicator_metadata('rsi_divergence', {
            'description': 'RSI Divergence',
            'interpretation': '1 for bullish divergence, -1 for bearish divergence, 0 for no divergence',
            'range': [-1, 0, 1],
            'references': 'https://www.investopedia.com/terms/d/divergence.asp'
        })
        
        # Add RSI with smoothing - better for noisy crypto markets
        alpha = 0.5  # Smoothing factor (adjust based on market noise)
        result['smooth_rsi'] = result['rsi'].ewm(alpha=alpha, adjust=False).mean()
        
        self.register_indicator_metadata('smooth_rsi', {
            'description': 'Smoothed RSI',
            'interpretation': 'RSI with noise reduction, same levels as regular RSI',
            'range': [0, 100],
            'references': 'Noise-filtered RSI variant'
        })
        
        # MACD
        if USE_TALIB:
            macd, signal, hist = talib.MACD(
                result['close'], 
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            result['macd'] = macd
            result['macd_signal'] = signal
            result['macd_hist'] = hist
        else:
            macd = ta.trend.MACD(
                close=result['close'],
                window_slow=self.config.macd_slow,
                window_fast=self.config.macd_fast,
                window_sign=self.config.macd_signal
            )
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_hist'] = macd.macd_diff()
            
        self.register_indicator_metadata('macd', {
            'description': 'MACD Line',
            'interpretation': 'Trend and momentum indicator, crossovers with signal line generate signals',
            'references': 'https://www.investopedia.com/terms/m/macd.asp'
        })
        
        self.register_indicator_metadata('macd_signal', {
            'description': 'MACD Signal Line',
            'interpretation': '9-period EMA of MACD line',
            'references': 'https://www.investopedia.com/terms/m/macd.asp'
        })
        
        self.register_indicator_metadata('macd_hist', {
            'description': 'MACD Histogram',
            'interpretation': 'MACD line minus signal line, shows momentum shifts',
            'references': 'https://www.investopedia.com/terms/m/macd.asp'
        })
        
        # Normalized MACD (for cross-asset comparison)
        if not result['macd'].isna().all():
            # Scale MACD by price
            result['norm_macd'] = result['macd'] / result['close'] * 100
            
            self.register_indicator_metadata('norm_macd', {
                'description': 'Normalized MACD',
                'interpretation': 'MACD as percentage of price, allows comparison across assets',
                'references': 'Normalized indicators'
            })
        
        # MACD Histogram zero-line crossovers (momentum shifts)
        result['macd_cross'] = np.where(
            (result['macd_hist'] > 0) & (result['macd_hist'].shift(1) <= 0), 1,
            np.where((result['macd_hist'] < 0) & (result['macd_hist'].shift(1) >= 0), -1, 0)
        )
        
        self.register_indicator_metadata('macd_cross', {
            'description': 'MACD Histogram Zero Crossover',
            'interpretation': '1 for bullish cross, -1 for bearish cross, 0 for no cross',
            'range': [-1, 0, 1],
            'references': 'MACD histogram analysis'
        })
        
        # Add TSI (True Strength Index)
        result['tsi'] = self._calculate_tsi(
            result['close'], 
            long_window=self.config.tsi_long_window, 
            short_window=self.config.tsi_short_window
        )
        
        self.register_indicator_metadata('tsi', {
            'description': 'True Strength Index',
            'interpretation': 'Double-smoothed momentum oscillator, crossover at 0 signals trend change',
            'range': [-100, 100],
            'references': 'https://www.investopedia.com/terms/t/tsi.asp'
        })
        
        # Add custom KST (Know Sure Thing) oscillator for longer-term momentum
        result = self._add_kst(result)
        
        # Add CCI (Commodity Channel Index) for momentum timing
        if USE_TALIB:
            result['cci'] = talib.CCI(
                result['high'], 
                result['low'], 
                result['close'], 
                timeperiod=20
            )
        else:
            result['cci'] = ta.trend.cci(
                high=result['high'],
                low=result['low'],
                close=result['close'],
                window=20,
                constant=0.015
            )
            
        self.register_indicator_metadata('cci', {
            'description': 'Commodity Channel Index',
            'interpretation': 'Values above 100 indicate overbought, below -100 oversold',
            'references': 'https://www.investopedia.com/terms/c/commoditychannelindex.asp'
        })
        
        # Add momentum rank indicator (percentile of current momentum vs. historical)
        momentum = result['close'].pct_change(10)
        result['momentum_percentile'] = momentum.rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        self.register_indicator_metadata('momentum_percentile', {
            'description': 'Momentum Percentile',
            'interpretation': 'Percentile rank of current momentum vs. historical',
            'range': [0, 100],
            'references': 'Statistical momentum analysis'
        })
        
        return result
    
    def _optimize_rsi_period(self, df: pd.DataFrame, min_period: int = 5, max_period: int = 21) -> int:
        """
        Optimize RSI period based on recent volatility.
        
        Args:
            df: DataFrame with OHLCV data
            min_period: Minimum RSI period to consider
            max_period: Maximum RSI period to consider
            
        Returns:
            Optimized RSI period
        """
        # Calculate volatility
        if 'natr' in df.columns:
            # Use NATR if already calculated
            recent_vol = df['natr'].iloc[-20:].mean()
        else:
            # Calculate recent volatility
            returns = df['close'].pct_change().iloc[-40:]
            recent_vol = returns.std() * 100
        
        # Scale the RSI period based on volatility
        # Higher volatility -> shorter period for responsiveness
        # Lower volatility -> longer period for fewer false signals
        
        # Normalize volatility to 0-1 range (assuming typical crypto vol range of 1-10%)
        normalized_vol = min(max(recent_vol - 1, 0) / 9, 1)
        
        # Invert and scale to get period (high vol = lower period)
        period_range = max_period - min_period
        period = max_period - int(normalized_vol * period_range)
        
        return period
    
    def _add_kst(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add KST (Know Sure Thing) oscillator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with KST oscillator
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Parameters
        rocma1 = 10
        rocma2 = 15
        rocma3 = 20
        rocma4 = 30
        
        # Rate of Change calculations
        roc1 = result['close'].pct_change(rocma1) * 100
        roc2 = result['close'].pct_change(rocma2) * 100
        roc3 = result['close'].pct_change(rocma3) * 100
        roc4 = result['close'].pct_change(rocma4) * 100
        
        # Moving averages of ROC values
        ma1 = roc1.rolling(10).mean()
        ma2 = roc2.rolling(10).mean()
        ma3 = roc3.rolling(10).mean()
        ma4 = roc4.rolling(15).mean()
        
        # KST calculation
        result['kst'] = (ma1 * 1) + (ma2 * 2) + (ma3 * 3) + (ma4 * 4)
        
        # Signal line (9-period moving average of KST)
        result['kst_signal'] = result['kst'].rolling(9).mean()
        
        # KST histogram
        result['kst_hist'] = result['kst'] - result['kst_signal']
        
        self.register_indicator_metadata('kst', {
            'description': 'Know Sure Thing (KST) Oscillator',
            'interpretation': 'Longer-term momentum oscillator, uptrend when above signal line',
            'references': 'Martin Pring'
        })
        
        self.register_indicator_metadata('kst_signal', {
            'description': 'KST Signal Line',
            'interpretation': '9-period MA of KST, crossovers generate signals',
            'references': 'Martin Pring'
        })
        
        return result
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators optimized for crypto markets.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volatility indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate ATR (Average True Range)
        if USE_TALIB:
            result['atr'] = talib.ATR(
                result['high'], 
                result['low'], 
                result['close'], 
                timeperiod=self.config.default_length
            )
        else:
            result['atr'] = ta.volatility.average_true_range(
                high=result['high'],
                low=result['low'],
                close=result['close'],
                window=self.config.default_length
            )
            
        self.register_indicator_metadata('atr', {
            'description': 'Average True Range',
            'interpretation': 'Measures volatility, higher values indicate higher volatility',
            'references': 'https://www.investopedia.com/terms/a/atr.asp'
        })
        
        # Calculate NATR (Normalized ATR)
        if not result['atr'].isna().all() and not (result['close'] == 0).any():
            result['natr'] = result['atr'] / result['close'] * 100
            
            self.register_indicator_metadata('natr', {
                'description': 'Normalized Average True Range',
                'interpretation': 'ATR as percentage of price, allows comparison across assets',
                'references': 'Normalized indicators'
            })
        
        # Calculate Bollinger Bands
        if USE_TALIB:
            upper, middle, lower = talib.BBANDS(
                result['close'], 
                timeperiod=self.config.default_length,
                nbdevup=self.config.bbands_dev,
                nbdevdn=self.config.bbands_dev,
                matype=0  # Simple Moving Average
            )
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower
        else:
            bb = ta.volatility.BollingerBands(
                close=result['close'],
                window=self.config.default_length,
                window_dev=self.config.bbands_dev
            )
            result['bb_upper'] = bb.bollinger_hband()
            result['bb_middle'] = bb.bollinger_mavg()
            result['bb_lower'] = bb.bollinger_lband()
            
        self.register_indicator_metadata('bb_upper', {
            'description': 'Bollinger Band Upper',
            'interpretation': 'Upper volatility band, price reaching here suggests overbought',
            'references': 'https://www.investopedia.com/terms/b/bollingerbands.asp'
        })
        
        self.register_indicator_metadata('bb_middle', {
            'description': 'Bollinger Band Middle',
            'interpretation': 'Middle band (SMA), center of volatility channel',
            'references': 'https://www.investopedia.com/terms/b/bollingerbands.asp'
        })
        
        self.register_indicator_metadata('bb_lower', {
            'description': 'Bollinger Band Lower',
            'interpretation': 'Lower volatility band, price reaching here suggests oversold',
            'references': 'https://www.investopedia.com/terms/b/bollingerbands.asp'
        })
        
        # Add Bollinger Band Width
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        self.register_indicator_metadata('bb_width', {
            'description': 'Bollinger Band Width',
            'interpretation': 'Measures volatility, narrow width often precedes volatility expansion',
            'references': 'https://www.investopedia.com/terms/b/bollingerbands.asp'
        })
        
        # Add Bollinger Band Percent B (%B)
        result['bb_pct_b'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        self.register_indicator_metadata('bb_pct_b', {
            'description': 'Bollinger Band %B',
            'interpretation': 'Position within Bollinger Bands (0-1), >1 overbought, <0 oversold',
            'range': [0, 1],
            'references': 'https://www.investopedia.com/terms/b/bollingerbands.asp'
        })
        
        # Calculate Keltner Channels (using ATR)
        ema20 = result['close'].ewm(span=20, adjust=False).mean()
        result['keltner_middle'] = ema20
        result['keltner_upper'] = ema20 + (result['atr'] * 2)
        result['keltner_lower'] = ema20 - (result['atr'] * 2)
        
        self.register_indicator_metadata('keltner_middle', {
            'description': 'Keltner Channel Middle',
            'interpretation': 'Middle line of Keltner Channel (EMA)',
            'references': 'https://www.investopedia.com/terms/k/keltnerchannel.asp'
        })
        
        self.register_indicator_metadata('keltner_upper', {
            'description': 'Keltner Channel Upper',
            'interpretation': 'Upper volatility band, price above here suggests strong trend',
            'references': 'https://www.investopedia.com/terms/k/keltnerchannel.asp'
        })
        
        self.register_indicator_metadata('keltner_lower', {
            'description': 'Keltner Channel Lower',
            'interpretation': 'Lower volatility band, price below here suggests strong downtrend',
            'references': 'https://www.investopedia.com/terms/k/keltnerchannel.asp'
        })
        
        # Add Squeeze Momentum Indicator (BB vs KC)
        # Squeeze is on when Bollinger Bands are inside Keltner Channels
        result['squeeze_on'] = (result['bb_lower'] > result['keltner_lower']) & (result['bb_upper'] < result['keltner_upper'])
        
        self.register_indicator_metadata('squeeze_on', {
            'description': 'Bollinger/Keltner Squeeze',
            'interpretation': 'True when volatility is compressed, often precedes large moves',
            'references': 'John Carter, Mastering the Trade'
        })
        
        # Calculate historical volatility (standard deviation of log returns)
        returns = np.log(result['close'] / result['close'].shift(1))
        result['hist_vol'] = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized
        
        self.register_indicator_metadata('hist_vol', {
            'description': 'Historical Volatility (Annualized)',
            'interpretation': 'Annualized standard deviation of returns, higher values indicate higher volatility',
            'references': 'Statistical volatility measure'
        })
        
        # Add adaptive volatility bands
        result = self._add_adaptive_volatility_bands(result)
        
        # Add volatility regime classification
        result = self._add_volatility_regime(result)
        
        # Add volatility breakout detection
        result = self._add_volatility_breakout(result)
        
        # Add implied volatility estimation (if funding data available)
        if self.config.account_for_funding and 'funding_rate' in result.columns:
            result = self._estimate_implied_volatility(result)
        
        # Add volatility term structure (if multiple timeframes available)
        if hasattr(self, 'multi_timeframe_data') and self.multi_timeframe_data is not None:
            result = self._add_volatility_term_structure(result)
        
        # Add forward volatility prediction
        result['predicted_vol'] = self._predict_forward_volatility(result)
        
        self.register_indicator_metadata('predicted_vol', {
            'description': 'Predicted Forward Volatility',
            'interpretation': 'GARCH-based prediction of volatility for next period',
            'references': 'Statistical volatility forecasting'
        })
        
        return result
    
    def _add_adaptive_volatility_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add adaptive volatility bands that adjust based on market conditions.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with adaptive volatility bands
        """
        result = df.copy()
        
        # Detect if in trend or range
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        adx = ta.trend.adx(result['high'], result['low'], result['close'], window=14) if not USE_TALIB else pd.Series(talib.ADX(result['high'], result['low'], result['close'], timeperiod=14), index=result.index)
        
        # Determine if in trend (ADX > 25) or range
        is_trending = adx > 25
        
        # Base multiplier on trend/range and recent volatility
        vol_z_score = (result['hist_vol'] - result['hist_vol'].rolling(90).mean()) / result['hist_vol'].rolling(90).std()
        
        # Adjust multiplier: higher in trends, lower in ranges, adjusted by recent volatility
        adaptive_mult = np.where(
            is_trending,
            2.5 + (0.5 * vol_z_score),  # Higher multiplier in trends
            1.5 + (0.3 * vol_z_score)   # Lower multiplier in ranges
        )
        
        # Ensure multiplier is within reasonable bounds
        adaptive_mult = np.clip(adaptive_mult, 1.0, 4.0)
        
        # Calculate moving average (using EMA for more responsiveness)
        ema = typical_price.ewm(span=20, adjust=False).mean()
        
        # Calculate and store adaptive volatility bands
        atr = result['atr']
        result['adaptive_upper'] = ema + (atr * adaptive_mult)
        result['adaptive_lower'] = ema - (atr * adaptive_mult)
        
        self.register_indicator_metadata('adaptive_upper', {
            'description': 'Adaptive Volatility Upper Band',
            'interpretation': 'Dynamic upper band that adjusts based on market regime and volatility',
            'references': 'Custom adaptive volatility indicator'
        })
        
        self.register_indicator_metadata('adaptive_lower', {
            'description': 'Adaptive Volatility Lower Band',
            'interpretation': 'Dynamic lower band that adjusts based on market regime and volatility',
            'references': 'Custom adaptive volatility indicator'
        })
        
        return result
    
    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market into volatility regimes.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility regime classification
        """
        result = df.copy()
        
        # Use NATR for normalized volatility comparison
        if 'natr' in result.columns:
            # Calculate long-term (3-month) statistics
            vol_mean = result['natr'].rolling(window=90).mean()
            vol_std = result['natr'].rolling(window=90).std()
            
            # Calculate z-score of current volatility
            vol_z = (result['natr'] - vol_mean) / vol_std
            
            # Classify regime based on z-score
            # Low volatility: z < -0.5
            # Normal volatility: -0.5 <= z <= 0.5
            # High volatility: 0.5 < z <= 2
            # Extreme volatility: z > 2
            conditions = [
                (vol_z < -0.5),
                (vol_z >= -0.5) & (vol_z <= 0.5),
                (vol_z > 0.5) & (vol_z <= 2),
                (vol_z > 2)
            ]
            choices = ['low', 'normal', 'high', 'extreme']
            
            # Create regime classifications
            result['vol_regime'] = np.select(conditions, choices, default='normal')
            
            # Also create a numeric version for calculations
            numeric_choices = [0, 1, 2, 3]  # low, normal, high, extreme
            result['vol_regime_numeric'] = np.select(conditions, numeric_choices, default=1)
            
            self.register_indicator_metadata('vol_regime', {
                'description': 'Volatility Regime',
                'interpretation': 'Classification of current volatility (low, normal, high, extreme)',
                'references': 'Statistical volatility classification'
            })
        
        return result
    
    def _add_volatility_breakout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volatility breakouts.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility breakout signals
        """
        result = df.copy()
        
        # Calculate changes in volatility
        vol_change = result['natr'].pct_change(5)
        
        # Define breakout thresholds
        vol_expansion_threshold = 0.5  # 50% increase in volatility
        vol_contraction_threshold = -0.3  # 30% decrease in volatility
        
        # Generate breakout signals
        result['vol_breakout'] = 0
        result.loc[vol_change > vol_expansion_threshold, 'vol_breakout'] = 1
        result.loc[vol_change < vol_contraction_threshold, 'vol_breakout'] = -1
        
        self.register_indicator_metadata('vol_breakout', {
            'description': 'Volatility Breakout',
            'interpretation': '1 for expansion, -1 for contraction, 0 for no significant change',
            'range': [-1, 0, 1],
            'references': 'Volatility breakout analysis'
        })
        
        return result
    
    def _estimate_implied_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate implied volatility using funding rates.
        Only works for perpetual futures with funding rates.
        
        Args:
            df: DataFrame with OHLCV and funding data
            
        Returns:
            DataFrame with implied volatility estimate
        """
        result = df.copy()
        
        # Skip if no funding rate available
        if 'funding_rate' not in result.columns:
            return result
        
        # Calculate annualized funding rate (assuming 3 funding events per day)
        annualized_funding = result['funding_rate'].abs() * 3 * 365
        
        # Basic model: implied vol has correlation with funding rate
        # High funding often correlates with higher implied volatility
        result['implied_vol_estimate'] = (annualized_funding * 0.5) + (result['hist_vol'] * 0.5)
        
        self.register_indicator_metadata('implied_vol_estimate', {
            'description': 'Estimated Implied Volatility',
            'interpretation': 'Rough estimate of implied volatility based on funding rates and historical vol',
            'references': 'Crypto-specific volatility estimation'
        })
        
        return result
    
    def _predict_forward_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict forward volatility using a simple GARCH-like approach.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with predicted volatility
        """
        try:
            # Use historical volatility as base
            hist_vol = df['hist_vol'].copy()
            
            # Fill missing values to avoid issues
            hist_vol = hist_vol.fillna(method='ffill')
            
            # Simple GARCH(1,1)-like model for volatility prediction
            # σ²ₜ = ω + α·rₜ₋₁² + β·σ²ₜ₋₁
            
            # Constants (traditionally estimated, using typical values here)
            omega = 0.000002  # Long-term volatility component
            alpha = 0.05      # Weight of recent returns
            beta = 0.90       # Persistence of volatility
            
            # Calculate log returns for GARCH
            returns = np.log(df['close'] / df['close'].shift(1))
            
            # Initialize predicted variance
            var_pred = pd.Series(index=df.index)
            var_pred.iloc[0] = (hist_vol.iloc[0] / 100) ** 2  # Convert % to decimal
            
            # Calculate GARCH predictions
            for t in range(1, len(df)):
                if np.isnan(returns.iloc[t-1]):
                    # If return is NaN, use previous variance
                    var_pred.iloc[t] = var_pred.iloc[t-1]
                else:
                    # GARCH update equation
                    var_pred.iloc[t] = omega + alpha * returns.iloc[t-1]**2 + beta * var_pred.iloc[t-1]
            
            # Convert variance to volatility (annualized %)
            vol_pred = np.sqrt(var_pred) * np.sqrt(252) * 100
            
            return vol_pred
            
        except Exception as e:
            logger.warning(f"Error predicting forward volatility: {e}")
            # Return historical volatility as fallback
            return df['hist_vol'].copy()
    
    def add_crypto_specific_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cryptocurrency-specific indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with crypto-specific indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Detect volatility regime
        if 'natr' in result.columns:
            result['volatility_regime'] = np.where(
                result['natr'] > result['natr'].rolling(self.config.regime_lookback).mean() * 1.5, 'high',
                np.where(result['natr'] < result['natr'].rolling(self.config.regime_lookback).mean() * 0.5, 'low', 'normal')
            )
            self.register_indicator_metadata('volatility_regime', {
                'description': 'Volatility Regime',
                'interpretation': 'Classifies market as high/normal/low volatility regime',
                'range': ['high', 'normal', 'low'],
                'references': 'https://www.investopedia.com/terms/v/volatility.asp'
            })
        
        # Mean reversion indicator
        mean_reversion_window = self.config.mean_reversion_window
        result['price_distance'] = (result['close'] - result['close'].rolling(mean_reversion_window).mean()) / \
                              result['close'].rolling(mean_reversion_window).std()
        
        self.register_indicator_metadata('price_distance', {
            'description': 'Price Distance',
            'interpretation': 'Standardized distance from moving average in standard deviations',
            'references': 'https://www.investopedia.com/terms/m/meanreversion.asp'
        })
        
        # Mean reversion signal
        result['mean_reversion_signal'] = np.where(
            result['price_distance'] < -2, 'buy',
            np.where(result['price_distance'] > 2, 'sell', 'neutral')
        )
        
        self.register_indicator_metadata('mean_reversion_signal', {
            'description': 'Mean Reversion Signal',
            'interpretation': 'Trading signal based on extreme deviations from mean',
            'range': ['buy', 'neutral', 'sell'],
            'references': 'https://www.investopedia.com/terms/m/meanreversion.asp'
        })
        
        # Heikin-Ashi candles (smoother price action for trending markets)
        result = self._add_heikin_ashi(result)
        
        # Add custom trading range identifier
        result = self._add_trading_range_identifier(result)
        
        # Historical Volatility
        result['hist_volatility'] = self._calculate_historical_volatility(result['close'])
        
        self.register_indicator_metadata('hist_volatility', {
            'description': 'Historical Volatility (20-day)',
            'interpretation': 'Annualized standard deviation of daily returns',
            'references': 'https://www.investopedia.com/terms/h/historicalvolatility.asp'
        })
        
        return result
    
    def _add_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Heikin-Ashi candles.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Heikin-Ashi data
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        result['ha_close'] = (result['open'] + result['high'] + result['low'] + result['close']) / 4
        
        # Initialize HA Open with the first candle's values
        ha_open = [result['open'].iloc[0]]
        
        # Calculate HA Open for the rest of the candles
        for i in range(1, len(result)):
            ha_open.append((ha_open[-1] + result['ha_close'].iloc[i-1]) / 2)
        
        result['ha_open'] = ha_open
        result['ha_high'] = result[['high', 'ha_open', 'ha_close']].max(axis=1)
        result['ha_low'] = result[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        self.register_indicator_metadata('ha_close', {
            'description': 'Heikin-Ashi Close',
            'interpretation': 'Smoother price representation, useful for trend identification',
            'references': 'https://www.investopedia.com/trading/heikin-ashi-better-candlestick/'
        })
        
        return result
    
    def _add_trading_range_identifier(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Add trading range identification.
        
        Args:
            df: DataFrame with OHLCV data
            window: Analysis window
            
        Returns:
            DataFrame with trading range features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate high-low range as percentage
        result['hl_range_pct'] = (result['high'] - result['low']) / result['close'] * 100
        
        # Calculate average range
        result['avg_range_pct'] = result['hl_range_pct'].rolling(window=window).mean()
        
        # Calculate price change over window
        result['window_change_pct'] = (result['close'] / result['close'].shift(window) - 1) * 100
        
        # Identify trading range (small absolute change but normal volatility)
        result['in_trading_range'] = ((result['window_change_pct'].abs() < result['avg_range_pct'] * 0.5) & 
                                     (result['hl_range_pct'] > result['hl_range_pct'].rolling(window).mean() * 0.5))
        
        self.register_indicator_metadata('in_trading_range', {
            'description': 'Trading Range Indicator',
            'interpretation': 'Boolean showing potential trading range (sideways price action)',
            'range': [0, 1],
            'references': 'https://www.investopedia.com/terms/t/tradingrange.asp'
        })
        
        return result
    
    def add_funding_indicators(self, df: pd.DataFrame, funding_data: List[Dict]) -> pd.DataFrame:
        """
        Add funding rate indicators for futures from external funding data.
        
        Args:
            df: DataFrame with OHLCV data
            funding_data: List of dictionaries with funding rate data
                Each dict should have 'timestamp' and 'rate' keys
            
        Returns:
            DataFrame with funding indicators
        """
        if funding_data is None or len(funding_data) == 0:
            logger.warning("No funding data provided")
            return df
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Create a DataFrame from funding data
        funding_df = pd.DataFrame(funding_data)
        
        # Convert timestamp to datetime if it's in milliseconds
        if 'timestamp' in funding_df.columns:
            if funding_df['timestamp'].iloc[0] > 1e10:  # Likely milliseconds
                funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
            else:
                funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='s')
            
            funding_df.set_index('timestamp', inplace=True)
        
        # Check if the df index is datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, funding data alignment may be inaccurate")
            if 'date' in result.columns:
                if not isinstance(result['date'].iloc[0], pd.Timestamp):
                    result['date'] = pd.to_datetime(result['date'])
                temp_index = result['date']
            else:
                logger.warning("Cannot add funding indicators: no datetime index or 'date' column found")
                return result
        else:
            temp_index = result.index
        
        # Determine the frequency of df
        try:
            freq = pd.infer_freq(temp_index)
            if freq:
                # Resample to match OHLCV data timeframe
                resampled = funding_df.resample(freq).last()
                
                # Merge with main dataframe
                if isinstance(result.index, pd.DatetimeIndex):
                    result['funding_rate'] = resampled['rate']
                else:
                    funding_series = pd.Series(index=temp_index, data=np.nan)
                    for idx in range(len(temp_index)):
                        dt = temp_index.iloc[idx]
                        if dt in resampled.index:
                            funding_series.iloc[idx] = resampled.loc[dt, 'rate']
                    result['funding_rate'] = funding_series
            else:
                # No frequency detected, try direct mapping
                logger.warning("Could not detect frequency, using direct mapping")
                result['funding_rate'] = np.nan
                for idx, row in result.iterrows():
                    if isinstance(result.index, pd.DatetimeIndex):
                        lookup_dt = idx
                    else:
                        lookup_dt = row['date']
                    
                    # Find closest funding timestamp within threshold (e.g., 1 hour)
                    closest = funding_df.index[
                        (funding_df.index >= lookup_dt - pd.Timedelta(hours=1)) &
                        (funding_df.index <= lookup_dt + pd.Timedelta(hours=1))
                    ]
                    
                    if len(closest) > 0:
                        nearest = min(closest, key=lambda x: abs(x - lookup_dt))
                        result.loc[idx, 'funding_rate'] = funding_df.loc[nearest, 'rate']
        except Exception as e:
            logger.error(f"Failed to process funding data: {e}")
            result['funding_rate'] = np.nan
        
        # Add funding indicators if funding_rate was successfully added
        if 'funding_rate' in result.columns and not result['funding_rate'].isna().all():
            # Calculate funding rate moving average
            result['funding_rate_ma'] = result['funding_rate'].rolling(8).mean()
            
            # Calculate z-score for funding rate
            result['funding_z_score'] = (result['funding_rate'] - result['funding_rate_ma']) / \
                                       result['funding_rate'].rolling(8).std()
            
            # Funding arbitrage potential
            result['funding_arb_potential'] = np.where(
                result['funding_z_score'] > 2, 'short',
                np.where(result['funding_z_score'] < -2, 'long', 'neutral')
            )
            
            # Calculate cumulative funding
            result['cumulative_funding'] = result['funding_rate'].cumsum()
            
            self.register_indicator_metadata('funding_z_score', {
                'description': 'Funding Rate Z-Score',
                'interpretation': 'Standardized funding rate, extreme values may indicate arbitrage opportunity',
                'references': 'https://www.tradingview.com/script/3YAhQA1D-Funding-Rate-Crypto-Perpetual-Futures/'
            })
            
            self.register_indicator_metadata('funding_arb_potential', {
                'description': 'Funding Arbitrage Potential',
                'interpretation': 'Potential arbitrage direction based on extreme funding rates',
                'range': ['long', 'neutral', 'short'],
                'references': 'https://www.tradingview.com/script/3YAhQA1D-Funding-Rate-Crypto-Perpetual-Futures/'
            })
        
        return result
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Moving Averages
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()
        result['sma_200'] = result['close'].rolling(window=200).mean()
        
        # Calculate EMAs
        result['ema_20'] = result['close'].ewm(span=20, adjust=False).mean()
        result['ema_50'] = result['close'].ewm(span=50, adjust=False).mean()
        result['ema_200'] = result['close'].ewm(span=200, adjust=False).mean()
        
        # Moving Average Crossovers
        result['golden_cross'] = np.where(
            (result['sma_50'] > result['sma_200']) & (result['sma_50'].shift(1) <= result['sma_200'].shift(1)),
            1, 0
        )
        
        result['death_cross'] = np.where(
            (result['sma_50'] < result['sma_200']) & (result['sma_50'].shift(1) >= result['sma_200'].shift(1)),
            1, 0
        )
        
        # Register metadata
        self.register_indicator_metadata('golden_cross', {
            'description': 'Golden Cross',
            'interpretation': 'Binary signal (1=yes) when 50-day SMA crosses above 200-day SMA',
            'range': [0, 1],
            'references': 'https://www.investopedia.com/terms/g/goldencross.asp'
        })
        
        self.register_indicator_metadata('death_cross', {
            'description': 'Death Cross',
            'interpretation': 'Binary signal (1=yes) when 50-day SMA crosses below 200-day SMA',
            'range': [0, 1],
            'references': 'https://www.investopedia.com/terms/d/deathcross.asp'
        })
        
        # Add SuperTrend indicator
        result = self._add_supertrend(result)
        
        return result
    
    def _add_supertrend(self, df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Add SuperTrend indicator.
        
        Args:
            df: DataFrame with OHLCV data
            atr_period: Period for ATR calculation
            multiplier: Multiplier for ATR
            
        Returns:
            DataFrame with SuperTrend indicator
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate ATR if not already present
        if 'atr' not in result.columns:
            if USE_TALIB:
                result['atr'] = talib.ATR(
                    result['high'],
                    result['low'],
                    result['close'],
                    timeperiod=atr_period
                )
            else:
                result['atr'] = ta.volatility.average_true_range(
                    high=result['high'],
                    low=result['low'],
                    close=result['close'],
                    window=atr_period
                )
        
        # Calculate basic upper and lower bands
        result['basic_upper_band'] = ((result['high'] + result['low']) / 2) + (multiplier * result['atr'])
        result['basic_lower_band'] = ((result['high'] + result['low']) / 2) - (multiplier * result['atr'])
        
        # Initialize final bands and trend
        result['final_upper_band'] = 0.0
        result['final_lower_band'] = 0.0
        result['supertrend'] = 0.0
        
        # Initialize with first period values
        result.loc[0, 'final_upper_band'] = result.loc[0, 'basic_upper_band']
        result.loc[0, 'final_lower_band'] = result.loc[0, 'basic_lower_band']
        result.loc[0, 'supertrend'] = result.loc[0, 'final_lower_band']
        
        # Calculate SuperTrend
        for i in range(1, len(result)):
            # Calculate upper band
            if (result.loc[i, 'basic_upper_band'] < result.loc[i-1, 'final_upper_band'] or 
                result.loc[i-1, 'close'] > result.loc[i-1, 'final_upper_band']):
                result.loc[i, 'final_upper_band'] = result.loc[i, 'basic_upper_band']
            else:
                result.loc[i, 'final_upper_band'] = result.loc[i-1, 'final_upper_band']
                
            # Calculate lower band
            if (result.loc[i, 'basic_lower_band'] > result.loc[i-1, 'final_lower_band'] or 
                result.loc[i-1, 'close'] < result.loc[i-1, 'final_lower_band']):
                result.loc[i, 'final_lower_band'] = result.loc[i, 'basic_lower_band']
            else:
                result.loc[i, 'final_lower_band'] = result.loc[i-1, 'final_lower_band']
                
            # Calculate SuperTrend
            if (result.loc[i-1, 'supertrend'] == result.loc[i-1, 'final_upper_band'] and 
                result.loc[i, 'close'] <= result.loc[i, 'final_upper_band']):
                result.loc[i, 'supertrend'] = result.loc[i, 'final_upper_band']
            elif (result.loc[i-1, 'supertrend'] == result.loc[i-1, 'final_upper_band'] and 
                  result.loc[i, 'close'] > result.loc[i, 'final_upper_band']):
                result.loc[i, 'supertrend'] = result.loc[i, 'final_lower_band']
            elif (result.loc[i-1, 'supertrend'] == result.loc[i-1, 'final_lower_band'] and 
                  result.loc[i, 'close'] >= result.loc[i, 'final_lower_band']):
                result.loc[i, 'supertrend'] = result.loc[i, 'final_lower_band']
            elif (result.loc[i-1, 'supertrend'] == result.loc[i-1, 'final_lower_band'] and 
                  result.loc[i, 'close'] < result.loc[i, 'final_lower_band']):
                result.loc[i, 'supertrend'] = result.loc[i, 'final_upper_band']
        
        # Create trend direction (1 for uptrend, -1 for downtrend)
        result['supertrend_direction'] = np.where(result['close'] > result['supertrend'], 1, -1)
        
        # Clean up intermediate columns
        result = result.drop(['basic_upper_band', 'basic_lower_band'], axis=1)
        
        self.register_indicator_metadata('supertrend', {
            'description': 'SuperTrend Line',
            'interpretation': 'Trend following indicator that acts as dynamic support/resistance',
            'references': 'https://www.tradingview.com/support/solutions/43000565435-supertrend/'
        })
        
        self.register_indicator_metadata('supertrend_direction', {
            'description': 'SuperTrend Direction',
            'interpretation': '1 for uptrend, -1 for downtrend',
            'range': [-1, 1],
            'references': 'https://www.tradingview.com/support/solutions/43000565435-supertrend/'
        })
        
        return result
        
    def generate_signals(self, df: pd.DataFrame, strategy: str = 'combined') -> pd.DataFrame:
        """
        Generate trading signals based on indicators.
        
        Args:
            df: DataFrame with indicators
            strategy: Strategy to use for signal generation
                Options: 'combined', 'trend_following', 'mean_reversion', 'volatility_breakout'
            
        Returns:
            DataFrame with signals added
        """
        if strategy == 'trend_following':
            df = self._generate_trend_following_signals(df)
        elif strategy == 'mean_reversion':
            df = self._generate_mean_reversion_signals(df)
        elif strategy == 'volatility_breakout':
            df = self._generate_volatility_breakout_signals(df)
        else:  # combined
            df = self._generate_combined_signals(df)
            
        return df
    
    def _generate_trend_following_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following signals."""
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize signal column
        result['trend_signal'] = 0  # 0 for no signal, 1 for buy, -1 for sell
        
        # Check for required indicators
        required = ['macd', 'macd_signal', 'supertrend_direction']
        missing = [col for col in required if col not in result.columns]
        if missing:
            logger.warning(f"Missing required indicators for trend signals: {missing}")
            return result
        
        # MACD crossover
        buy_macd = (result['macd'] > result['macd_signal']) & (result['macd'].shift(1) <= result['macd_signal'].shift(1))
        sell_macd = (result['macd'] < result['macd_signal']) & (result['macd'].shift(1) >= result['macd_signal'].shift(1))
        
        # SuperTrend direction
        uptrend = result['supertrend_direction'] == 1
        downtrend = result['supertrend_direction'] == -1
        
        # Generate signals
        result['trend_signal'] = np.where(
            (buy_macd & uptrend), 1,
            np.where((sell_macd & downtrend), -1, 0)
        )
        
        self.register_indicator_metadata('trend_signal', {
            'description': 'Trend Following Signal',
            'interpretation': '1 for buy signal, -1 for sell signal, 0 for no signal',
            'range': [-1, 0, 1],
            'references': 'Custom trend following strategy'
        })
        
        return result
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize signal column
        result['reversion_signal'] = 0  # 0 for no signal, 1 for buy, -1 for sell
        
        # Check for required indicators
        required = ['rsi', 'bollinger_pct_b']
        missing = [col for col in required if col not in result.columns]
        if missing:
            logger.warning(f"Missing required indicators for mean reversion signals: {missing}")
            return result
        
        # Oversold/overbought conditions
        oversold = (result['rsi'] < 30) & (result['bollinger_pct_b'] < 0.05)
        overbought = (result['rsi'] > 70) & (result['bollinger_pct_b'] > 0.95)
        
        # Generate signals
        result['reversion_signal'] = np.where(
            oversold, 1,
            np.where(overbought, -1, 0)
        )
        
        self.register_indicator_metadata('reversion_signal', {
            'description': 'Mean Reversion Signal',
            'interpretation': '1 for buy signal, -1 for sell signal, 0 for no signal',
            'range': [-1, 0, 1],
            'references': 'Custom mean reversion strategy'
        })
        
        return result
    
    def _generate_volatility_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals."""
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize signal column
        result['breakout_signal'] = 0  # 0 for no signal, 1 for buy, -1 for sell
        
        # Check for required indicators
        required = ['bollinger_width', 'atr']
        missing = [col for col in required if col not in result.columns]
        if missing:
            logger.warning(f"Missing required indicators for breakout signals: {missing}")
            return result
        
        # Volatility contraction followed by expansion
        # Volatility contraction (Bollinger Band squeeze)
        result['vol_contraction'] = result['bollinger_width'] < result['bollinger_width'].rolling(20).mean() * 0.85
        
        # Volatility expansion (ATR increases)
        result['vol_expansion'] = result['atr'] > result['atr'].shift(1) * 1.1
        
        # Delayed contraction (for signal generation after contraction)
        result['vol_contraction_delay'] = result['vol_contraction'].shift(3)
        
        # Price movement for direction
        result['price_up'] = result['close'] > result['close'].shift(1)
        result['price_down'] = result['close'] < result['close'].shift(1)
        
        # Signal conditions - breakout after contraction
        result['breakout_signal'] = np.where(
            (result['vol_contraction_delay'] & result['vol_expansion'] & result['price_up']), 1,
            np.where((result['vol_contraction_delay'] & result['vol_expansion'] & result['price_down']), -1, 0)
        )
        
        self.register_indicator_metadata('breakout_signal', {
            'description': 'Volatility Breakout Signal',
            'interpretation': '1 for buy signal, -1 for sell signal, 0 for no signal',
            'range': [-1, 0, 1],
            'references': 'Custom volatility breakout strategy'
        })
        
        return result
    
    def _generate_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate combined signals from all strategies."""
        # Apply individual strategies
        result = self._generate_trend_following_signals(df)
        result = self._generate_mean_reversion_signals(result)
        result = self._generate_volatility_breakout_signals(result)
        
        # Combined signal - prioritize with trend following > volatility breakout > mean reversion
        result['combined_signal'] = 0
        
        # First, use trend following signals
        result.loc[result['trend_signal'] != 0, 'combined_signal'] = result.loc[result['trend_signal'] != 0, 'trend_signal']
        
        # Then, for rows without trend signals, use breakout signals
        no_trend_signal = result['combined_signal'] == 0
        result.loc[no_trend_signal & (result['breakout_signal'] != 0), 'combined_signal'] = \
            result.loc[no_trend_signal & (result['breakout_signal'] != 0), 'breakout_signal']
        
        # Finally, for rows without trend or breakout signals, use mean reversion signals
        no_signal_yet = result['combined_signal'] == 0
        result.loc[no_signal_yet & (result['reversion_signal'] != 0), 'combined_signal'] = \
            result.loc[no_signal_yet & (result['reversion_signal'] != 0), 'reversion_signal']
        
        self.register_indicator_metadata('combined_signal', {
            'description': 'Combined Trading Signal',
            'interpretation': '1 for buy signal, -1 for sell signal, 0 for no signal',
            'range': [-1, 0, 1],
            'references': 'Prioritized combination of trend, breakout, and mean reversion strategies'
        })
        
        return result 
    
    def add_cross_asset_correlations(self, df: pd.DataFrame, reference_assets: dict = None) -> pd.DataFrame:
        """
        Add correlation indicators against key reference assets.
        
        Args:
            df: DataFrame with OHLCV data for the target asset
            reference_assets: Dict of {asset_name: dataframe} with OHLCV data for reference assets
                             If None, will attempt to use internal market data
        
        Returns:
            DataFrame with added correlation indicators
        """
        result = df.copy()
        
        # Skip if we don't have reference assets
        if reference_assets is None and not hasattr(self, 'market_data'):
            logger.warning("No reference assets provided for correlation analysis")
            return result
            
        # Use provided reference assets or internal market data
        ref_assets = reference_assets or self.market_data
        
        if not ref_assets:
            return result
            
        # Define key reference assets for crypto
        key_assets = {
            'BTC': 'Bitcoin as crypto market leader',
            'ETH': 'Ethereum as smart contract leader',
            'TOTAL': 'Total crypto market cap',
            'DXY': 'US Dollar Index',
            'SPX': 'S&P 500 index'
        }
        
        # Calculate rolling correlation coefficients
        for asset_name, asset_df in ref_assets.items():
            # Skip if not a key asset (to avoid excessive calculations)
            if asset_name not in key_assets and not asset_name.endswith('USDT'):
                continue
                
            # Skip if insufficient data
            if len(asset_df) < 30:
                continue
                
            # Get reference price series (prefer close price)
            if 'close' in asset_df.columns:
                ref_price = asset_df['close']
            else:
                continue
                
            # Ensure index alignment
            if ref_price.index[0] != result.index[0]:
                # Attempt to align indexes
                ref_price = ref_price.reindex(result.index, method='ffill')
                
            # Calculate correlations over multiple timeframes
            for window in [10, 30, 90]:
                # Calculate rolling correlation
                col_name = f'corr_{asset_name}_{window}d'
                result[col_name] = result['close'].rolling(window=window).corr(ref_price)
                
                # Register metadata for the indicator
                self.register_indicator_metadata(col_name, {
                    'description': f'{window}-day Correlation with {asset_name}',
                    'interpretation': f'Correlation coefficient with {asset_name} ({key_assets.get(asset_name, "reference asset")})',
                    'range': [-1, 1],
                    'references': 'Cross-asset correlation analysis'
                })
        
        # Generate a correlation regime indicator for Bitcoin
        if 'corr_BTC_30d' in result.columns:
            # Define correlation regimes
            conditions = [
                (result['corr_BTC_30d'] > 0.7),
                (result['corr_BTC_30d'] > 0.3) & (result['corr_BTC_30d'] <= 0.7),
                (result['corr_BTC_30d'] >= -0.3) & (result['corr_BTC_30d'] <= 0.3),
                (result['corr_BTC_30d'] < -0.3) & (result['corr_BTC_30d'] >= -0.7),
                (result['corr_BTC_30d'] < -0.7)
            ]
            choices = ['very_high', 'high', 'neutral', 'low', 'very_low']
            
            result['btc_corr_regime'] = np.select(conditions, choices, default='neutral')
            
            self.register_indicator_metadata('btc_corr_regime', {
                'description': 'Bitcoin Correlation Regime',
                'interpretation': 'Categorizes the current correlation with Bitcoin',
                'references': 'Cross-asset correlation regimes'
            })
        
        return result
    
    def add_onchain_metrics(self, df: pd.DataFrame, onchain_data: dict = None) -> pd.DataFrame:
        """
        Add on-chain metrics for crypto analysis.
        
        Args:
            df: DataFrame with OHLCV data
            onchain_data: Dict of {metric_name: series} with on-chain data
                         If None, will skip on-chain analysis
        
        Returns:
            DataFrame with added on-chain metrics
        """
        result = df.copy()
        
        # Skip if we don't have on-chain data
        if onchain_data is None:
            logger.debug("No on-chain data provided for analysis")
            return result
            
        # Add the raw on-chain metrics first
        for metric_name, data_series in onchain_data.items():
            # Skip if insufficient data
            if len(data_series) < 10:
                continue
                
            # Align index with price data
            aligned_data = data_series.reindex(result.index, method='ffill')
            
            # Add the raw metric
            metric_col = f'onchain_{metric_name}'
            result[metric_col] = aligned_data
            
            # Register basic metadata
            self.register_indicator_metadata(metric_col, {
                'description': f'On-chain {metric_name}',
                'interpretation': f'Blockchain metric: {metric_name}',
                'references': 'On-chain analysis'
            })
        
        # Process common metrics if available
        # 1. NVT Ratio (Network Value to Transactions Ratio)
        if 'onchain_txn_volume_usd' in result.columns and 'close' in result.columns and 'circulating_supply' in result.columns:
            result['onchain_nvt'] = (result['close'] * result['circulating_supply']) / result['onchain_txn_volume_usd']
            
            self.register_indicator_metadata('onchain_nvt', {
                'description': 'Network Value to Transactions Ratio',
                'interpretation': 'Similar to P/E ratio for blockchains, higher values may indicate overvaluation',
                'references': 'Willy Woo, Chris Burniske on NVT'
            })
            
            # NVT Signal (using moving average of transaction volume)
            if len(result) > 90:
                smooth_txn_volume = result['onchain_txn_volume_usd'].rolling(90).mean()
                result['onchain_nvt_signal'] = (result['close'] * result['circulating_supply']) / smooth_txn_volume
                
                self.register_indicator_metadata('onchain_nvt_signal', {
                    'description': 'NVT Signal',
                    'interpretation': 'Smoothed NVT ratio developed by Willy Woo',
                    'references': 'https://woobull.com/introducing-nvt-signal-a-new-indicator-of-bitcoin-bubble-tops/'
                })
        
        # 2. SOPR (Spent Output Profit Ratio)
        if 'onchain_sopr' in result.columns:
            # Calculate SOPR moving averages
            result['onchain_sopr_ma7'] = result['onchain_sopr'].rolling(7).mean()
            result['onchain_sopr_ma30'] = result['onchain_sopr'].rolling(30).mean()
            
            # SOPR oscillator (comparing short-term to long-term)
            result['onchain_sopr_oscillator'] = result['onchain_sopr_ma7'] / result['onchain_sopr_ma30']
            
            self.register_indicator_metadata('onchain_sopr_oscillator', {
                'description': 'SOPR Oscillator',
                'interpretation': 'Values > 1 may indicate profit-taking, < 1 may indicate accumulation',
                'references': 'Glassnode SOPR analysis'
            })
        
        # 3. MVRV (Market Value to Realized Value)
        if 'onchain_realized_value' in result.columns and 'close' in result.columns and 'circulating_supply' in result.columns:
            market_cap = result['close'] * result['circulating_supply']
            result['onchain_mvrv'] = market_cap / result['onchain_realized_value']
            
            self.register_indicator_metadata('onchain_mvrv', {
                'description': 'Market Value to Realized Value Ratio',
                'interpretation': 'Values > 3.5 often indicate market tops, < 1 often indicate bottoms',
                'references': 'Murad Mahmudov and David Puell on MVRV'
            })
        
        # 4. Active Addresses Sentiment
        if 'onchain_active_addresses' in result.columns:
            # Calculate z-score of active addresses (normalized for network growth)
            if len(result) > 90:
                active_ma = result['onchain_active_addresses'].rolling(90).mean()
                active_std = result['onchain_active_addresses'].rolling(90).std()
                result['onchain_active_zscore'] = (result['onchain_active_addresses'] - active_ma) / active_std
                
                self.register_indicator_metadata('onchain_active_zscore', {
                    'description': 'Active Addresses Z-Score',
                    'interpretation': 'Normalized active addresses, values > 2 may indicate increased adoption/interest',
                    'references': 'On-chain analysis of network usage'
                })
        
        # 5. HODL Waves (if available)
        hodl_columns = [col for col in result.columns if col.startswith('onchain_hodl_')]
        if hodl_columns:
            # Calculate HODL ratio (long-term vs short-term holders)
            long_term_cols = [col for col in hodl_columns if '1y_plus' in col or '2y_plus' in col]
            short_term_cols = [col for col in hodl_columns if '3m_minus' in col or '6m_minus' in col]
            
            if long_term_cols and short_term_cols:
                result['onchain_hodl_ratio'] = result[long_term_cols].sum(axis=1) / result[short_term_cols].sum(axis=1)
                
                self.register_indicator_metadata('onchain_hodl_ratio', {
                    'description': 'HODL Ratio',
                    'interpretation': 'Ratio of long-term holders to short-term holders, higher values often indicate accumulation phases',
                    'references': 'UTXO age analysis'
                })
        
        return result
    
    def add_market_sentiment(self, df: pd.DataFrame, sentiment_data: dict = None) -> pd.DataFrame:
        """
        Add market sentiment indicators from external sources.
        
        Args:
            df: DataFrame with OHLCV data
            sentiment_data: Dict of {source: series} with sentiment data
                           If None, will skip sentiment analysis
        
        Returns:
            DataFrame with added sentiment indicators
        """
        result = df.copy()
        
        # Skip if we don't have sentiment data
        if sentiment_data is None:
            logger.debug("No sentiment data provided for analysis")
            return result
        
        # Add raw sentiment metrics
        for source, data_series in sentiment_data.items():
            # Skip if insufficient data
            if len(data_series) < 10:
                continue
                
            # Align index with price data
            aligned_data = data_series.reindex(result.index, method='ffill')
            
            # Add sentiment metric
            col_name = f'sentiment_{source}'
            result[col_name] = aligned_data
            
            # Register metadata
            self.register_indicator_metadata(col_name, {
                'description': f'Market Sentiment from {source}',
                'interpretation': 'Higher values indicate more positive sentiment',
                'references': f'Sentiment analysis from {source}'
            })
        
        # Calculate composite sentiment if we have multiple sources
        sentiment_cols = [col for col in result.columns if col.startswith('sentiment_')]
        if len(sentiment_cols) >= 2:
            # Normalize each sentiment to 0-1 range
            normalized_sentiments = pd.DataFrame()
            for col in sentiment_cols:
                # Skip if all NaN
                if result[col].isna().all():
                    continue
                    
                min_val = result[col].min()
                max_val = result[col].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    normalized_sentiments[col] = (result[col] - min_val) / (max_val - min_val)
                else:
                    normalized_sentiments[col] = 0.5  # Neutral if no variation
            
            # Calculate composite sentiment (simple average)
            if not normalized_sentiments.empty:
                result['sentiment_composite'] = normalized_sentiments.mean(axis=1)
                
                self.register_indicator_metadata('sentiment_composite', {
                    'description': 'Composite Market Sentiment',
                    'interpretation': 'Normalized and aggregated sentiment from multiple sources (0-1)',
                    'range': [0, 1],
                    'references': 'Aggregated sentiment analysis'
                })
                
                # Convert to z-score for standardized view
                if len(result) > 30:
                    sentiment_mean = result['sentiment_composite'].rolling(30).mean()
                    sentiment_std = result['sentiment_composite'].rolling(30).std()
                    
                    # Avoid division by zero
                    valid_std = sentiment_std > 0
                    result.loc[valid_std, 'sentiment_zscore'] = (
                        (result.loc[valid_std, 'sentiment_composite'] - 
                         sentiment_mean[valid_std]) / sentiment_std[valid_std]
                    )
                    
                    self.register_indicator_metadata('sentiment_zscore', {
                        'description': 'Sentiment Z-Score',
                        'interpretation': 'Standardized sentiment relative to recent history',
                        'references': 'Statistical sentiment analysis'
                    })
                    
                # Detect sentiment divergences with price
                if len(result) > 14:
                    # Calculate price momentum
                    price_mom = result['close'].pct_change(14)
                    # Calculate sentiment momentum
                    sentiment_mom = result['sentiment_composite'].diff(14)
                    
                    # Bullish divergence: price down, sentiment up
                    result['sentiment_bull_div'] = (price_mom < -0.05) & (sentiment_mom > 0.1)
                    
                    # Bearish divergence: price up, sentiment down
                    result['sentiment_bear_div'] = (price_mom > 0.05) & (sentiment_mom < -0.1)
                    
                    self.register_indicator_metadata('sentiment_bull_div', {
                        'description': 'Bullish Sentiment Divergence',
                        'interpretation': 'True when price falls but sentiment rises, potential bullish signal',
                        'references': 'Sentiment divergence analysis'
                    })
                    
                    self.register_indicator_metadata('sentiment_bear_div', {
                        'description': 'Bearish Sentiment Divergence',
                        'interpretation': 'True when price rises but sentiment falls, potential bearish signal',
                        'references': 'Sentiment divergence analysis'
                    })
        
        # Add Fear & Greed specific analysis if available
        if 'sentiment_fear_greed' in result.columns:
            # Convert Fear & Greed to regime classification
            conditions = [
                (result['sentiment_fear_greed'] <= 20),  # Extreme Fear
                (result['sentiment_fear_greed'] > 20) & (result['sentiment_fear_greed'] <= 40),  # Fear
                (result['sentiment_fear_greed'] > 40) & (result['sentiment_fear_greed'] <= 60),  # Neutral
                (result['sentiment_fear_greed'] > 60) & (result['sentiment_fear_greed'] <= 80),  # Greed
                (result['sentiment_fear_greed'] > 80)  # Extreme Greed
            ]
            choices = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
            
            result['fear_greed_regime'] = np.select(conditions, choices, default='neutral')
            
            self.register_indicator_metadata('fear_greed_regime', {
                'description': 'Fear & Greed Regime',
                'interpretation': 'Market sentiment classification based on Fear & Greed index',
                'references': 'Crypto Fear & Greed Index'
            })
            
            # Create contrarian signal from extreme readings
            extreme_fear = result['sentiment_fear_greed'] <= 20
            extreme_greed = result['sentiment_fear_greed'] >= 80
            
            # Signal: 1 for bullish (extreme fear), -1 for bearish (extreme greed), 0 for neutral
            result['fear_greed_signal'] = 0
            result.loc[extreme_fear, 'fear_greed_signal'] = 1
            result.loc[extreme_greed, 'fear_greed_signal'] = -1
            
            self.register_indicator_metadata('fear_greed_signal', {
                'description': 'Fear & Greed Contrarian Signal',
                'interpretation': '1 for bullish (extreme fear), -1 for bearish (extreme greed), 0 for neutral',
                'range': [-1, 0, 1],
                'references': 'Contrarian sentiment analysis'
            })
        
        # Add social media analysis if available
        if 'sentiment_social' in result.columns:
            # Calculate rate of change
            result['sentiment_social_roc'] = result['sentiment_social'].pct_change(3)
            
            self.register_indicator_metadata('sentiment_social_roc', {
                'description': 'Social Sentiment Rate of Change',
                'interpretation': 'Measures how quickly social sentiment is shifting',
                'references': 'Social media sentiment analytics'
            })
            
            # Detect social sentiment spikes (potential mania or capitulation)
            if len(result) > 30:
                social_mean = result['sentiment_social'].rolling(30).mean()
                social_std = result['sentiment_social'].rolling(30).std()
                
                # Calculate z-score of social sentiment
                valid_std = social_std > 0
                result.loc[valid_std, 'sentiment_social_zscore'] = (
                    (result.loc[valid_std, 'sentiment_social'] - 
                     social_mean[valid_std]) / social_std[valid_std]
                )
                
                # Detect extreme readings
                result['social_mania'] = result['sentiment_social_zscore'] > 2.5
                result['social_capitulation'] = result['sentiment_social_zscore'] < -2.5
                
                self.register_indicator_metadata('social_mania', {
                    'description': 'Social Media Mania',
                    'interpretation': 'True when social sentiment is extremely positive',
                    'references': 'Social media sentiment extremes'
                })
                
                self.register_indicator_metadata('social_capitulation', {
                    'description': 'Social Media Capitulation',
                    'interpretation': 'True when social sentiment is extremely negative',
                    'references': 'Social media sentiment extremes'
                })
        
        return result
    
    def process_indicators(self, df: pd.DataFrame, reference_assets: dict = None, 
                          onchain_data: dict = None, sentiment_data: dict = None) -> pd.DataFrame:
        """
        Process all indicators for crypto analysis.
        
        Args:
            df: DataFrame with OHLCV data
            reference_assets: Dict of reference assets for cross-asset correlation
            onchain_data: Dict of on-chain metrics
            sentiment_data: Dict of sentiment metrics
            
        Returns:
            DataFrame with all indicators
        """
        logger.info("Processing crypto indicators")
        
        # Create copy to avoid modifying original
        result = df.copy()
        
        # Add basic price indicators
        result = self.add_price_indicators(result)
        
        # Add volume indicators
        result = self.add_volume_indicators(result)
        
        # Add momentum indicators
        result = self.add_momentum_indicators(result)
        
        # Add volatility indicators
        result = self.add_volatility_indicators(result)
        
        # Add cross-asset correlations if reference data available
        if reference_assets is not None:
            result = self.add_cross_asset_correlations(result, reference_assets)
            
        # Add on-chain metrics if data available
        if onchain_data is not None:
            result = self.add_onchain_metrics(result, onchain_data)
            
        # Add sentiment analysis if data available
        if sentiment_data is not None:
            result = self.add_market_sentiment(result, sentiment_data)
        
        # Add crypto-specific indicators
        result = self.add_crypto_specific_indicators(result)
        
        # Calculate aggregate crypto health score
        result = self.calculate_crypto_health_score(result)
        
        logger.info(f"Processed {len(self.indicator_metadata)} crypto indicators")
        return result
    
    def calculate_crypto_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate an aggregate health score for crypto assets based on available indicators.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with added health score
        """
        result = df.copy()
        
        # Define categories and weights for the health score
        score_components = {
            'trend': {
                'weight': 0.25,
                'indicators': {
                    'trend_strength': 2.0,       # Trend strength (e.g., ADX)
                    'ema_alignment': 1.5,        # EMA alignment (short above long)
                    'price_above_ma200': 1.0,    # Price above 200-day MA
                }
            },
            'momentum': {
                'weight': 0.20,
                'indicators': {
                    'rsi': 1.0,                 # RSI in bullish range (>50)
                    'macd_histogram': 1.5,      # MACD histogram positive
                    'momentum_percentile': 2.0  # Momentum percentile rank
                }
            },
            'volatility': {
                'weight': 0.15,
                'indicators': {
                    'vol_regime_numeric': 1.0,  # Volatility regime (lower is better)
                    'squeeze_on': 0.5,          # Volatility squeeze (potential energy)
                    'bb_pct_b': 1.0            # Position within Bollinger Bands
                }
            },
            'volume': {
                'weight': 0.15,
                'indicators': {
                    'volume_trend': 1.0,        # Volume trend positive
                    'obv_norm': 1.5,            # OBV normalized positive 
                    'mfi': 1.0                  # MFI indicator
                }
            },
            'market_context': {
                'weight': 0.25,
                'indicators': {
                    'btc_corr_regime': 1.0,     # BTC correlation regime
                    'fear_greed_signal': 2.0,   # Fear & Greed contrarian signal
                    'onchain_mvrv': 1.5,        # MVRV ratio
                    'onchain_hodl_ratio': 1.0,  # HODL ratio
                    'onchain_active_zscore': 1.0 # Active addresses z-score
                }
            }
        }
        
        # Initialize score columns
        for category in score_components:
            result[f'score_{category}'] = 0.0
        
        # Calculate scores for each category
        for category, category_data in score_components.items():
            category_score = 0.0
            total_weights = 0.0
            
            for indicator, weight in category_data['indicators'].items():
                # Skip if indicator is not available
                if indicator not in result.columns:
                    continue
                    
                # Get the indicator value and normalize
                values = result[indicator].copy()
                
                # Skip if all values are NaN
                if values.isna().all():
                    continue
                
                # Process based on indicator type
                if indicator == 'trend_strength':  # ADX-like
                    # Higher is better, normalize 0-100 to 0-1
                    normalized = np.clip(values / 50, 0, 1)
                
                elif indicator == 'ema_alignment':  # Binary
                    # 1 if short above long, -1 if long above short
                    normalized = (values > 0).astype(float)
                
                elif indicator == 'price_above_ma200':  # Binary
                    # 1 if price above MA200, 0 otherwise
                    normalized = values.astype(float)
                
                elif indicator == 'rsi':  # 0-100 scale
                    # Centered around 50, higher is more bullish
                    normalized = np.clip((values - 30) / 40, 0, 1)
                
                elif indicator == 'macd_histogram':  # Can be positive or negative
                    # Convert to -1 to 1 range based on histogram sign and relative size
                    max_abs = np.max(np.abs(values))
                    if max_abs > 0:
                        normalized = values / max_abs / 2 + 0.5  # Center at 0.5
                    else:
                        normalized = 0.5  # Neutral if no signal
                
                elif indicator == 'momentum_percentile':  # Already 0-100
                    # Higher is better
                    normalized = values / 100
                
                elif indicator == 'vol_regime_numeric':  # 0-3 scale (low to extreme)
                    # Lower volatility is better (0-1), inverse the scale
                    normalized = 1 - (values / 3)
                
                elif indicator == 'squeeze_on':  # Boolean
                    # 1 if squeeze is on (potential energy building)
                    normalized = values.astype(float)
                
                elif indicator == 'bb_pct_b':  # 0-1 scale
                    # Middle of band is best (0.5)
                    normalized = 1 - np.abs(values - 0.5) * 2
                
                elif indicator == 'volume_trend':  # Can be positive or negative
                    # Positive volume trend is better
                    normalized = np.clip((values + 1) / 2, 0, 1)  # Scale -1 to 1 into 0 to 1
                
                elif indicator == 'obv_norm':  # Normalized to mean=0, std=1
                    # Higher is better, convert to 0-1 scale
                    normalized = np.clip((values + 2) / 4, 0, 1)  # Scale -2 to 2 into 0 to 1
                
                elif indicator == 'mfi':  # 0-100 scale
                    # Similar to RSI
                    normalized = np.clip((values - 30) / 40, 0, 1)
                
                elif indicator == 'btc_corr_regime':  # String categories
                    # Map correlation regimes to scores
                    regime_map = {
                        'very_high': 0.8,  # High correlation with BTC is generally good in bull markets
                        'high': 0.7,
                        'neutral': 0.5,
                        'low': 0.3,
                        'very_low': 0.2
                    }
                    normalized = values.map(regime_map).fillna(0.5)
                
                elif indicator == 'fear_greed_signal':  # -1 to 1 scale
                    # 1 for bullish, -1 for bearish, 0 neutral
                    normalized = (values + 1) / 2  # Scale -1 to 1 into 0 to 1
                
                elif indicator == 'onchain_mvrv':  # MVRV ratio
                    # Optimal range 1-2.5, extreme values are worse
                    mask_low = values < 1
                    mask_optimal = (values >= 1) & (values <= 2.5)
                    mask_high = values > 2.5
                    
                    normalized = pd.Series(index=values.index, data=0.5)  # Default neutral
                    normalized[mask_low] = values[mask_low]  # Linear scaling 0-1
                    normalized[mask_optimal] = 1.0  # Optimal range
                    normalized[mask_high] = np.clip(1 - (values[mask_high] - 2.5) / 2, 0, 1)  # Scale down as values increase
                
                elif indicator == 'onchain_hodl_ratio':  # Higher is generally better
                    # Normalize to 0-1 using percentile rank
                    normalized = values.rank(pct=True)
                
                elif indicator == 'onchain_active_zscore':  # Z-score
                    # Higher values indicate more network activity
                    normalized = np.clip((values + 3) / 6, 0, 1)  # Scale -3 to 3 into 0 to 1
                
                else:
                    # Default: use percentile rank
                    normalized = values.rank(pct=True)
                
                # Add weighted score
                category_score += normalized * weight
                total_weights += weight
            
            # Calculate weighted average if we have data
            if total_weights > 0:
                result[f'score_{category}'] = category_score / total_weights
        
        # Calculate overall health score
        result['crypto_health_score'] = 0.0
        total_weight = 0.0
        
        # Combine category scores
        for category, category_data in score_components.items():
            score_col = f'score_{category}'
            if score_col in result.columns and not result[score_col].isna().all():
                result['crypto_health_score'] += result[score_col] * category_data['weight']
                total_weight += category_data['weight']
        
        # Normalize the final score if we have data
        if total_weight > 0:
            result['crypto_health_score'] = result['crypto_health_score'] / total_weight
            
            # Scale to 0-100 for easier interpretation
            result['crypto_health_score'] = result['crypto_health_score'] * 100
            
            self.register_indicator_metadata('crypto_health_score', {
                'description': 'Crypto Asset Health Score',
                'interpretation': 'Composite health rating from 0-100, higher is healthier',
                'range': [0, 100],
                'references': 'Aggregate technical, on-chain, and sentiment analysis'
            })
        
        return result