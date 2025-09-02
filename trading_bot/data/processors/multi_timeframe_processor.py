#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiTimeframeProcessor - Process and combine data from multiple timeframes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import time
from collections import defaultdict

from trading_bot.data.processors.base_processor import DataProcessor

logger = logging.getLogger("MultiTimeframeProcessor")

class MultiTimeframeProcessor(DataProcessor):
    """
    Processor for handling and combining data from multiple timeframes.
    
    This processor can:
    - Resample data to different timeframes
    - Calculate indicators across timeframes
    - Align data from different timeframes for analysis
    - Combine signals from multiple timeframes
    """
    
    def __init__(self, name: str = "MultiTimeframeProcessor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize MultiTimeframeProcessor.
        
        Args:
            name: Processor name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Timeframes to process
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        
        # Base timeframe (lowest granularity)
        self.base_timeframe = self.config.get('base_timeframe', '1m')
        
        # Store dataframes for each timeframe
        self.data_by_timeframe = {}
        
        # Combined data (aligned to base timeframe)
        self.combined_data = None
        
        # Last update time for each timeframe
        self.last_update_times = {}
        
        # Technical indicators to calculate for each timeframe
        self.indicators = self.config.get('indicators', [
            'rsi', 'macd', 'bollinger', 'atr', 'sma', 'ema', 'stoch', 'adx'
        ])
        
        # Mapping of pandas frequency strings to time units
        self.freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W',
            '1M': '1M'
        }
        
        # Initialize TA-lib if available
        try:
            import talib
            self.talib_available = True
        except ImportError:
            self.talib_available = False
            logger.warning("TA-lib not available, using pandas for technical indicators")
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Weight for each timeframe when combining signals
        self.timeframe_weights = self.config.get('timeframe_weights', {
            '1m': 0.05,
            '5m': 0.10,
            '15m': 0.15,
            '1h': 0.20,
            '4h': 0.25,
            '1d': 0.25
        })
        
        # Normalize weights if not already normalized
        total_weight = sum(self.timeframe_weights.values())
        if abs(total_weight - 1.0) > 0.001:  # Allow small rounding error
            self.timeframe_weights = {tf: weight / total_weight 
                                      for tf, weight in self.timeframe_weights.items()}
        
        # Indicator parameters
        self.indicator_params = self.config.get('indicator_params', {
            'rsi': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'atr': {'period': 14},
            'sma': {'periods': [10, 20, 50, 200]},
            'ema': {'periods': [9, 21, 55, 200]},
            'stoch': {'k_period': 14, 'd_period': 3, 'slowing': 3},
            'adx': {'period': 14}
        })
        
        # Enable/disable specific features
        self.enable_alignment = self.config.get('enable_alignment', True)
        self.enable_signal_combination = self.config.get('enable_signal_combination', True)
        self.calculate_custom_indicators = self.config.get('calculate_custom_indicators', True)
    
    def process(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Process data for multiple timeframes.
        
        Args:
            data: DataFrame with OHLCV data or dict of DataFrames by timeframe
            
        Returns:
            Processed and combined DataFrame
        """
        # Process data by timeframe
        if isinstance(data, dict):
            # Data already separated by timeframe
            for timeframe, df in data.items():
                if timeframe in self.timeframes:
                    # Process each timeframe
                    processed_df = self._process_single_timeframe(df, timeframe)
                    self.data_by_timeframe[timeframe] = processed_df
                    self.last_update_times[timeframe] = datetime.now()
        else:
            # Assume base timeframe data, create other timeframes by resampling
            if data.empty:
                logger.warning("Empty data provided to MultiTimeframeProcessor")
                return pd.DataFrame()
                
            # Store original data as base timeframe
            self.data_by_timeframe[self.base_timeframe] = self._process_single_timeframe(
                data, self.base_timeframe
            )
            self.last_update_times[self.base_timeframe] = datetime.now()
            
            # Create other timeframes by resampling
            for timeframe in self.timeframes:
                if timeframe != self.base_timeframe:
                    resampled = self._resample_data(data, timeframe)
                    if not resampled.empty:
                        self.data_by_timeframe[timeframe] = self._process_single_timeframe(
                            resampled, timeframe
                        )
                        self.last_update_times[timeframe] = datetime.now()
        
        # Combine data from all timeframes if alignment is enabled
        if self.enable_alignment and self.data_by_timeframe:
            self.combined_data = self._align_timeframes()
            
            # Add combined signals if enabled
            if self.enable_signal_combination:
                self.combined_data = self._combine_signals(self.combined_data)
                
            return self.combined_data
        
        # Otherwise return only base timeframe data
        return self.data_by_timeframe.get(self.base_timeframe, pd.DataFrame())
    
    def _process_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Process data for a single timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe identifier (e.g., '1m', '1h', '1d')
            
        Returns:
            Processed DataFrame with indicators for this timeframe
        """
        # Ensure DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logger.error(f"Missing required columns for {timeframe}: {missing}")
            return df
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Make sure the index is datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            if 'timestamp' in result.columns:
                result.set_index('timestamp', inplace=True)
            else:
                logger.warning(f"No datetime index or timestamp column for {timeframe}")
                # Create a simple index if needed
                result.index = pd.date_range(
                    start=datetime.now() - timedelta(days=len(result)),
                    periods=len(result),
                    freq=self.freq_map.get(timeframe, '1min')
                )
        
        # Calculate technical indicators
        result = self._add_technical_indicators(result, timeframe)
        
        # Add timeframe identifier suffix to column names
        if timeframe != self.base_timeframe:
            # Don't rename OHLCV columns
            non_ohlcv_cols = [col for col in result.columns if col not in required_columns]
            
            # Rename columns with timeframe suffix
            rename_dict = {col: f"{col}_{timeframe}" for col in non_ohlcv_cols}
            result.rename(columns=rename_dict, inplace=True)
        
        return result
    
    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        # Ensure we have datetime index
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            else:
                logger.warning("No datetime index or timestamp column, cannot resample")
                return pd.DataFrame()
        
        # Get pandas frequency string
        freq = self.freq_map.get(timeframe)
        if not freq:
            logger.error(f"Unknown timeframe: {timeframe}")
            return pd.DataFrame()
        
        # Resample OHLCV data
        try:
            resampled = data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Drop rows with NaN values (incomplete bars)
            resampled.dropna(inplace=True)
            
            return resampled
        except Exception as e:
            logger.error(f"Error resampling to {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe identifier
            
        Returns:
            DataFrame with technical indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure we have enough data for indicators
        if len(result) < 10:
            logger.warning(f"Not enough data for indicators in {timeframe} timeframe")
            return result
        
        # Calculate indicators based on configuration
        for indicator in self.indicators:
            try:
                if indicator == 'rsi' and 'rsi' in self.indicator_params:
                    period = self.indicator_params['rsi']['period']
                    result = self._add_rsi(result, period)
                
                elif indicator == 'macd' and 'macd' in self.indicator_params:
                    params = self.indicator_params['macd']
                    result = self._add_macd(
                        result, 
                        params['fast_period'], 
                        params['slow_period'], 
                        params['signal_period']
                    )
                
                elif indicator == 'bollinger' and 'bollinger' in self.indicator_params:
                    params = self.indicator_params['bollinger']
                    result = self._add_bollinger_bands(result, params['period'], params['std_dev'])
                
                elif indicator == 'atr' and 'atr' in self.indicator_params:
                    period = self.indicator_params['atr']['period']
                    result = self._add_atr(result, period)
                
                elif indicator == 'sma' and 'sma' in self.indicator_params:
                    periods = self.indicator_params['sma']['periods']
                    result = self._add_sma(result, periods)
                
                elif indicator == 'ema' and 'ema' in self.indicator_params:
                    periods = self.indicator_params['ema']['periods']
                    result = self._add_ema(result, periods)
                
                elif indicator == 'stoch' and 'stoch' in self.indicator_params:
                    params = self.indicator_params['stoch']
                    result = self._add_stochastic(
                        result, 
                        params['k_period'], 
                        params['d_period'], 
                        params['slowing']
                    )
                
                elif indicator == 'adx' and 'adx' in self.indicator_params:
                    period = self.indicator_params['adx']['period']
                    result = self._add_adx(result, period)
            
            except Exception as e:
                logger.error(f"Error calculating {indicator} for {timeframe}: {str(e)}")
        
        # Add custom combined indicators if enabled
        if self.calculate_custom_indicators:
            result = self._add_custom_indicators(result)
        
        return result
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            DataFrame with RSI
        """
        result = df.copy()
        
        try:
            if self.talib_available:
                import talib
                result['rsi'] = talib.RSI(result['close'], timeperiod=period)
            else:
                # Calculate price changes
                delta = result['close'].diff()
                
                # Separate gains and losses
                gain = delta.copy()
                loss = delta.copy()
                gain[gain < 0] = 0
                loss[loss > 0] = 0
                loss = abs(loss)
                
                # Calculate average gain and loss
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                result['rsi'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
        
        return result
    
    def _add_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                signal_period: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD
        """
        result = df.copy()
        
        try:
            if self.talib_available:
                import talib
                result['macd'], result['macd_signal'], result['macd_hist'] = talib.MACD(
                    result['close'], 
                    fastperiod=fast_period, 
                    slowperiod=slow_period, 
                    signalperiod=signal_period
                )
            else:
                # Calculate EMAs
                fast_ema = result['close'].ewm(span=fast_period, adjust=False).mean()
                slow_ema = result['close'].ewm(span=slow_period, adjust=False).mean()
                
                # Calculate MACD line
                result['macd'] = fast_ema - slow_ema
                
                # Calculate signal line
                result['macd_signal'] = result['macd'].ewm(span=signal_period, adjust=False).mean()
                
                # Calculate histogram
                result['macd_hist'] = result['macd'] - result['macd_signal']
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
        
        return result
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands
        """
        result = df.copy()
        
        try:
            if self.talib_available:
                import talib
                result['bb_upper'], result['bb_middle'], result['bb_lower'] = talib.BBANDS(
                    result['close'], 
                    timeperiod=period, 
                    nbdevup=std_dev, 
                    nbdevdn=std_dev
                )
            else:
                # Calculate middle band (SMA)
                result['bb_middle'] = result['close'].rolling(window=period).mean()
                
                # Calculate standard deviation
                result['bb_std'] = result['close'].rolling(window=period).std()
                
                # Calculate upper and lower bands
                result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * std_dev)
                result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * std_dev)
                
                # Drop the std column
                result.drop(columns=['bb_std'], inplace=True)
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        
        return result
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            DataFrame with ATR
        """
        result = df.copy()
        
        try:
            if self.talib_available:
                import talib
                result['atr'] = talib.ATR(
                    result['high'], 
                    result['low'], 
                    result['close'], 
                    timeperiod=period
                )
            else:
                # Calculate true range
                result['tr'] = np.maximum(
                    result['high'] - result['low'],
                    np.maximum(
                        abs(result['high'] - result['close'].shift(1)),
                        abs(result['low'] - result['close'].shift(1))
                    )
                )
                
                # Calculate ATR
                result['atr'] = result['tr'].rolling(window=period).mean()
                
                # Drop the TR column
                result.drop(columns=['tr'], inplace=True)
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
        
        return result
    
    def _add_sma(self, df: pd.DataFrame, periods: List[int] = [10, 20, 50, 200]) -> pd.DataFrame:
        """
        Add Simple Moving Averages to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of SMA periods
            
        Returns:
            DataFrame with SMAs
        """
        result = df.copy()
        
        for period in periods:
            try:
                if self.talib_available:
                    import talib
                    result[f'sma_{period}'] = talib.SMA(result['close'], timeperiod=period)
                else:
                    result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
            except Exception as e:
                logger.error(f"Error calculating SMA_{period}: {str(e)}")
        
        return result
    
    def _add_ema(self, df: pd.DataFrame, periods: List[int] = [9, 21, 55, 200]) -> pd.DataFrame:
        """
        Add Exponential Moving Averages to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of EMA periods
            
        Returns:
            DataFrame with EMAs
        """
        result = df.copy()
        
        for period in periods:
            try:
                if self.talib_available:
                    import talib
                    result[f'ema_{period}'] = talib.EMA(result['close'], timeperiod=period)
                else:
                    result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
            except Exception as e:
                logger.error(f"Error calculating EMA_{period}: {str(e)}")
        
        return result
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, 
                      slowing: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            slowing: Slowing period
            
        Returns:
            DataFrame with Stochastic Oscillator
        """
        result = df.copy()
        
        try:
            if self.talib_available:
                import talib
                result['stoch_k'], result['stoch_d'] = talib.STOCH(
                    result['high'], 
                    result['low'], 
                    result['close'], 
                    fastk_period=k_period, 
                    slowk_period=slowing, 
                    slowk_matype=0, 
                    slowd_period=d_period, 
                    slowd_matype=0
                )
            else:
                # Calculate %K
                result['stoch_k'] = 100 * ((result['close'] - result['low'].rolling(k_period).min()) / 
                                         (result['high'].rolling(k_period).max() - 
                                          result['low'].rolling(k_period).min()))
                
                # Apply slowing
                if slowing > 1:
                    result['stoch_k'] = result['stoch_k'].rolling(slowing).mean()
                
                # Calculate %D
                result['stoch_d'] = result['stoch_k'].rolling(d_period).mean()
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
        
        return result
    
    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: ADX period
            
        Returns:
            DataFrame with ADX
        """
        result = df.copy()
        
        try:
            if self.talib_available:
                import talib
                result['adx'] = talib.ADX(
                    result['high'], 
                    result['low'], 
                    result['close'], 
                    timeperiod=period
                )
                result['plus_di'] = talib.PLUS_DI(
                    result['high'], 
                    result['low'], 
                    result['close'], 
                    timeperiod=period
                )
                result['minus_di'] = talib.MINUS_DI(
                    result['high'], 
                    result['low'], 
                    result['close'], 
                    timeperiod=period
                )
            else:
                # This is a simplified implementation
                # Calculate True Range
                result['tr'] = np.maximum(
                    result['high'] - result['low'],
                    np.maximum(
                        abs(result['high'] - result['close'].shift(1)),
                        abs(result['low'] - result['close'].shift(1))
                    )
                )
                
                # Calculate Directional Movement
                result['up_move'] = result['high'] - result['high'].shift(1)
                result['down_move'] = result['low'].shift(1) - result['low']
                
                # Calculate +DM and -DM
                result['plus_dm'] = np.where(
                    (result['up_move'] > result['down_move']) & (result['up_move'] > 0),
                    result['up_move'],
                    0
                )
                result['minus_dm'] = np.where(
                    (result['down_move'] > result['up_move']) & (result['down_move'] > 0),
                    result['down_move'],
                    0
                )
                
                # Calculate smoothed TR, +DM, and -DM
                result['atr'] = result['tr'].rolling(window=period).mean()
                result['plus_dm_smooth'] = result['plus_dm'].rolling(window=period).mean()
                result['minus_dm_smooth'] = result['minus_dm'].rolling(window=period).mean()
                
                # Calculate +DI and -DI
                result['plus_di'] = 100 * result['plus_dm_smooth'] / result['atr']
                result['minus_di'] = 100 * result['minus_dm_smooth'] / result['atr']
                
                # Calculate DX
                result['dx'] = 100 * abs(result['plus_di'] - result['minus_di']) / \
                             (result['plus_di'] + result['minus_di'])
                
                # Calculate ADX
                result['adx'] = result['dx'].rolling(window=period).mean()
                
                # Drop intermediate columns
                result.drop(columns=['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm',
                                   'plus_dm_smooth', 'minus_dm_smooth', 'dx'], inplace=True)
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
        
        return result
    
    def _add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom combined indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data and basic indicators
            
        Returns:
            DataFrame with custom indicators
        """
        result = df.copy()
        
        try:
            # Add multi-timeframe trend score if available
            if 'ema_9' in result.columns and 'ema_21' in result.columns and 'ema_55' in result.columns:
                # Calculate trend score based on EMA alignment
                result['trend_score'] = 0
                
                # EMA9 > EMA21 > EMA55 = Strong uptrend (+2)
                result.loc[(result['ema_9'] > result['ema_21']) & 
                          (result['ema_21'] > result['ema_55']), 'trend_score'] = 2
                
                # EMA9 > EMA21 but not EMA21 > EMA55 = Weak uptrend (+1)
                result.loc[(result['ema_9'] > result['ema_21']) & 
                          (result['ema_21'] <= result['ema_55']), 'trend_score'] = 1
                
                # EMA9 < EMA21 but not EMA21 < EMA55 = Weak downtrend (-1)
                result.loc[(result['ema_9'] < result['ema_21']) & 
                          (result['ema_21'] >= result['ema_55']), 'trend_score'] = -1
                
                # EMA9 < EMA21 < EMA55 = Strong downtrend (-2)
                result.loc[(result['ema_9'] < result['ema_21']) & 
                          (result['ema_21'] < result['ema_55']), 'trend_score'] = -2
            
            # Add volatility state if available
            if 'atr' in result.columns:
                # Calculate ATR percentage of price
                result['atr_pct'] = (result['atr'] / result['close']) * 100
                
                # Calculate normalized ATR (z-score)
                mean_atr = result['atr_pct'].rolling(window=50).mean()
                std_atr = result['atr_pct'].rolling(window=50).std()
                result['atr_z'] = (result['atr_pct'] - mean_atr) / std_atr
                
                # Categorize volatility
                result['volatility_state'] = pd.cut(
                    result['atr_z'],
                    bins=[-float('inf'), -1.0, -0.5, 0.5, 1.0, float('inf')],
                    labels=['Very Low', 'Low', 'Normal', 'High', 'Very High']
                )
            
            # Add momentum state if available
            if 'rsi' in result.columns and 'macd' in result.columns:
                # Combined momentum indicator
                result['momentum_score'] = 0
                
                # Add RSI component (-2 to +2)
                result.loc[result['rsi'] < 30, 'momentum_score'] += -2  # Oversold
                result.loc[(result['rsi'] >= 30) & (result['rsi'] < 45), 'momentum_score'] += -1  # Weak oversold
                result.loc[(result['rsi'] > 55) & (result['rsi'] <= 70), 'momentum_score'] += 1  # Weak overbought
                result.loc[result['rsi'] > 70, 'momentum_score'] += 2  # Overbought
                
                # Add MACD component (-1 to +1)
                if 'macd_hist' in result.columns:
                    result.loc[result['macd_hist'] > 0, 'momentum_score'] += 1  # Positive momentum
                    result.loc[result['macd_hist'] < 0, 'momentum_score'] += -1  # Negative momentum
                
                # Categorize momentum
                result['momentum_state'] = pd.cut(
                    result['momentum_score'],
                    bins=[-float('inf'), -2, -1, 1, 2, float('inf')],
                    labels=['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive']
                )
            
            # Add support/resistance detection
            if 'bb_upper' in result.columns and 'bb_lower' in result.columns:
                # Price position within Bollinger Bands
                result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
                
                # Signal when price reaches band extremes
                result['bb_signal'] = 0
                result.loc[result['bb_position'] >= 0.95, 'bb_signal'] = -1  # Overbought near upper band
                result.loc[result['bb_position'] <= 0.05, 'bb_signal'] = 1   # Oversold near lower band
        
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {str(e)}")
        
        return result
    
    def _align_timeframes(self) -> pd.DataFrame:
        """
        Align data from different timeframes to the base timeframe.
        
        Returns:
            DataFrame with aligned data from all timeframes
        """
        # Check if we have base timeframe data
        if self.base_timeframe not in self.data_by_timeframe:
            logger.error(f"Base timeframe {self.base_timeframe} data not available")
            return pd.DataFrame()
        
        # Start with base timeframe data
        result = self.data_by_timeframe[self.base_timeframe].copy()
        
        # Add data from other timeframes
        for timeframe, df in self.data_by_timeframe.items():
            if timeframe == self.base_timeframe:
                continue
                
            # For each higher timeframe, forward fill values
            # This simulates having the higher timeframe data available at each base timeframe tick
            columns_to_add = [col for col in df.columns if f"_{timeframe}" in col]
            
            if not columns_to_add:
                logger.warning(f"No columns with {timeframe} suffix found")
                continue
            
            try:
                # Reindex to match base timeframe
                reindexed = df[columns_to_add].reindex(result.index, method='ffill')
                
                # Add to result
                for col in columns_to_add:
                    result[col] = reindexed[col]
            except Exception as e:
                logger.error(f"Error aligning {timeframe} data: {str(e)}")
        
        return result
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine signals from multiple timeframes into consolidated signals.
        
        Args:
            df: DataFrame with aligned multi-timeframe data
            
        Returns:
            DataFrame with combined signals
        """
        result = df.copy()
        
        try:
            # Combine trend signals if available
            trend_cols = [col for col in result.columns if 'trend_score' in col]
            
            if trend_cols:
                # Initialize combined trend score
                result['combined_trend_score'] = 0
                
                # Weight and sum trend scores from all timeframes
                for col in trend_cols:
                    if col == 'trend_score':  # Base timeframe
                        timeframe = self.base_timeframe
                    else:
                        # Extract timeframe from column name
                        timeframe = col.split('_')[-1]
                    
                    # Apply weight for this timeframe
                    weight = self.timeframe_weights.get(timeframe, 0.1)
                    result['combined_trend_score'] += result[col] * weight
                
                # Categorize combined trend
                result['trend_strength'] = pd.cut(
                    result['combined_trend_score'],
                    bins=[-float('inf'), -1.5, -0.5, 0.5, 1.5, float('inf')],
                    labels=['Strong Downtrend', 'Downtrend', 'Neutral', 'Uptrend', 'Strong Uptrend']
                )
            
            # Combine momentum signals if available
            momentum_cols = [col for col in result.columns if 'momentum_score' in col]
            
            if momentum_cols:
                # Initialize combined momentum score
                result['combined_momentum_score'] = 0
                
                # Weight and sum momentum scores from all timeframes
                for col in momentum_cols:
                    if col == 'momentum_score':  # Base timeframe
                        timeframe = self.base_timeframe
                    else:
                        # Extract timeframe from column name
                        timeframe = col.split('_')[-1]
                    
                    # Apply weight for this timeframe
                    weight = self.timeframe_weights.get(timeframe, 0.1)
                    result['combined_momentum_score'] += result[col] * weight
                
                # Categorize combined momentum
                result['momentum_strength'] = pd.cut(
                    result['combined_momentum_score'],
                    bins=[-float('inf'), -1.5, -0.5, 0.5, 1.5, float('inf')],
                    labels=['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive']
                )
            
            # Create overall trading signal
            if 'combined_trend_score' in result.columns and 'combined_momentum_score' in result.columns:
                # Combined signal (-1 to +1)
                # Weight trend more heavily than momentum
                result['combined_signal'] = (result['combined_trend_score'] * 0.6 + 
                                          result['combined_momentum_score'] * 0.4) / 3
                
                # Clip to range [-1, 1]
                result['combined_signal'] = result['combined_signal'].clip(-1, 1)
                
                # Generate trade direction
                result['trade_direction'] = np.where(
                    result['combined_signal'] > 0.3, 'BUY',
                    np.where(
                        result['combined_signal'] < -0.3, 'SELL',
                        'NEUTRAL'
                    )
                )
        
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
        
        return result
    
    def get_current_signals(self) -> Dict[str, Any]:
        """
        Get current trading signals from the multi-timeframe analysis.
        
        Returns:
            Dictionary with current trading signals
        """
        if self.combined_data is None or self.combined_data.empty:
            return {'status': 'No data available'}
        
        # Get the last row of data
        last_row = self.combined_data.iloc[-1].to_dict()
        
        # Create signal summary
        signals = {
            'timestamp': last_row.get('timestamp', self.combined_data.index[-1]),
            'close': last_row.get('close'),
            'trend': last_row.get('trend_strength'),
            'momentum': last_row.get('momentum_strength'),
            'volatility': last_row.get('volatility_state'),
            'combined_signal': last_row.get('combined_signal'),
            'trade_direction': last_row.get('trade_direction')
        }
        
        # Add timeframe-specific signals if available
        for timeframe in self.timeframes:
            if timeframe == self.base_timeframe:
                key_prefix = ''
            else:
                key_prefix = f'_{timeframe}'
                
            # Collect indicators for this timeframe
            tf_indicators = {}
            
            # RSI
            rsi_key = f'rsi{key_prefix}'
            if rsi_key in last_row:
                tf_indicators['rsi'] = last_row[rsi_key]
            
            # MACD
            macd_key = f'macd{key_prefix}'
            if macd_key in last_row:
                tf_indicators['macd'] = last_row[macd_key]
                tf_indicators['macd_signal'] = last_row.get(f'macd_signal{key_prefix}')
                tf_indicators['macd_hist'] = last_row.get(f'macd_hist{key_prefix}')
            
            # Add to signals if indicators were found
            if tf_indicators:
                signals[f'{timeframe}_indicators'] = tf_indicators
        
        return signals
    
    def get_timeframe_weights(self) -> Dict[str, float]:
        """
        Get current weights for each timeframe.
        
        Returns:
            Dictionary mapping timeframes to their weights
        """
        return self.timeframe_weights
    
    def set_timeframe_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for each timeframe.
        
        Args:
            weights: Dictionary mapping timeframes to weights
        """
        # Validate timeframes
        invalid_timeframes = [tf for tf in weights if tf not in self.timeframes]
        if invalid_timeframes:
            logger.warning(f"Invalid timeframes: {invalid_timeframes}")
            
        # Filter valid timeframes
        valid_weights = {tf: weight for tf, weight in weights.items() if tf in self.timeframes}
        
        # Normalize weights
        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            normalized_weights = {tf: weight / total_weight for tf, weight in valid_weights.items()}
            
            # Update weights
            self.timeframe_weights.update(normalized_weights)
            logger.info(f"Updated timeframe weights: {self.timeframe_weights}")
        else:
            logger.warning("Invalid weights (sum must be > 0)")
    
    def update_indicator_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update parameters for technical indicators.
        
        Args:
            new_params: Dictionary with updated parameters
        """
        for indicator, params in new_params.items():
            if indicator in self.indicator_params:
                self.indicator_params[indicator].update(params)
                logger.info(f"Updated {indicator} parameters: {self.indicator_params[indicator]}")
            else:
                logger.warning(f"Unknown indicator: {indicator}")
    
    def __str__(self) -> str:
        """String representation of the MultiTimeframeProcessor."""
        tf_str = ", ".join(self.timeframes)
        indicators_str = ", ".join(self.indicators)
        return f"MultiTimeframeProcessor (Timeframes: {tf_str}, Indicators: {indicators_str})" 