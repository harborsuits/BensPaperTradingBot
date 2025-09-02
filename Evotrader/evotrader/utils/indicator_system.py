"""
Enhanced technical indicator system for EvoTrader.

This module provides a robust, class-based approach to technical indicators that:
1. Associates indicators with specific symbols
2. Works with both list-based and DataFrame data
3. Handles None values gracefully
4. Provides a consistent interface for all indicators
5. Supports efficient updates and vectorized operations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Import the existing robust indicator calculations
from .robust_indicators import (
    safe_sma, safe_ema, safe_rsi, safe_macd, safe_bollinger_bands,
    safe_atr, safe_stochastic
)

logger = logging.getLogger(__name__)


class Indicator(ABC):
    """Base class for all technical indicators."""
    
    def __init__(self, symbol: str, params: Dict[str, Any]):
        """
        Initialize indicator with symbol and parameters.
        
        Args:
            symbol: Trading symbol this indicator is tracking
            params: Configuration parameters for the indicator
        """
        self.symbol = symbol
        self.params = params
        self.values: List[Optional[float]] = []
        self.is_ready = False
        self.min_data_points = 1  # Override in subclasses
    
    @abstractmethod
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """
        Process a new price candle and update the indicator value.
        
        Args:
            candle: Price candle with OHLCV data
            
        Returns:
            Latest indicator value or None if not enough data
        """
        pass
    
    def update_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Update indicator with a pandas DataFrame of historical data.
        
        Args:
            df: DataFrame with price data (must include indicator's price_field)
        """
        for _, row in df.iterrows():
            candle = row.to_dict()
            self.update(candle)
    
    def get_values(self) -> List[Optional[float]]:
        """Get all calculated indicator values."""
        return self.values
    
    def get_last_value(self) -> Optional[float]:
        """Get the most recent indicator value."""
        return self.values[-1] if self.values else None
    
    def get_value_at(self, idx: int) -> Optional[float]:
        """Get indicator value at specific index."""
        if 0 <= idx < len(self.values):
            return self.values[idx]
        return None
    
    def get_last_n_values(self, n: int) -> List[Optional[float]]:
        """Get the last n indicator values."""
        return self.values[-n:] if len(self.values) >= n else self.values
    
    def reset(self) -> None:
        """Reset indicator state."""
        self.values = []
        self.is_ready = False


class SMA(Indicator):
    """Simple Moving Average indicator."""
    
    def __init__(self, symbol: str, period: int = 20, price_field: str = 'close'):
        """
        Initialize SMA indicator.
        
        Args:
            symbol: Trading symbol
            period: SMA period
            price_field: Price field to use (close, open, high, low)
        """
        super().__init__(symbol, {'period': period, 'price_field': price_field})
        self.price_history: List[Optional[float]] = []
        self.min_data_points = period
    
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """Update SMA with new price data."""
        if self.params['price_field'] not in candle:
            logger.warning(f"Price field {self.params['price_field']} not found in candle for {self.symbol}")
            self.values.append(None)
            return None
        
        # Get price and handle None values
        price = candle[self.params['price_field']]
        self.price_history.append(price)
        
        # Calculate SMA if we have enough data
        if len(self.price_history) >= self.params['period']:
            # Use our robust SMA calculation
            sma_values = safe_sma(self.price_history, self.params['period'])
            sma_value = sma_values[-1] if sma_values else None
            
            self.values.append(sma_value)
            self.is_ready = True
            return sma_value
        
        self.values.append(None)
        return None


class EMA(Indicator):
    """Exponential Moving Average indicator."""
    
    def __init__(self, symbol: str, period: int = 20, price_field: str = 'close', smoothing: float = 2.0):
        """
        Initialize EMA indicator.
        
        Args:
            symbol: Trading symbol
            period: EMA period
            price_field: Price field to use (close, open, high, low)
            smoothing: Smoothing factor (default 2.0)
        """
        super().__init__(symbol, {
            'period': period, 
            'price_field': price_field,
            'smoothing': smoothing
        })
        self.price_history: List[Optional[float]] = []
        self.min_data_points = period
    
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """Update EMA with new price data."""
        if self.params['price_field'] not in candle:
            logger.warning(f"Price field {self.params['price_field']} not found in candle for {self.symbol}")
            self.values.append(None)
            return None
        
        # Get price and handle None values
        price = candle[self.params['price_field']]
        self.price_history.append(price)
        
        # Calculate EMA if we have enough data
        if len(self.price_history) >= self.params['period']:
            # Use our robust EMA calculation
            ema_values = safe_ema(
                self.price_history, 
                self.params['period'],
                self.params['smoothing']
            )
            ema_value = ema_values[-1] if ema_values else None
            
            self.values.append(ema_value)
            self.is_ready = True
            return ema_value
        
        self.values.append(None)
        return None


class RSI(Indicator):
    """Relative Strength Index indicator."""
    
    def __init__(self, symbol: str, period: int = 14, price_field: str = 'close'):
        """
        Initialize RSI indicator.
        
        Args:
            symbol: Trading symbol
            period: RSI period
            price_field: Price field to use (close, open, high, low)
        """
        super().__init__(symbol, {'period': period, 'price_field': price_field})
        self.price_history: List[Optional[float]] = []
        self.min_data_points = period + 1  # Need period+1 for calculating price changes
    
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """Update RSI with new price data."""
        if self.params['price_field'] not in candle:
            logger.warning(f"Price field {self.params['price_field']} not found in candle for {self.symbol}")
            self.values.append(None)
            return None
        
        # Get price and handle None values
        price = candle[self.params['price_field']]
        self.price_history.append(price)
        
        # Calculate RSI if we have enough data
        if len(self.price_history) >= self.params['period'] + 1:
            # Use our robust RSI calculation
            rsi_values = safe_rsi(self.price_history, self.params['period'])
            rsi_value = rsi_values[-1] if rsi_values else None
            
            self.values.append(rsi_value)
            self.is_ready = True
            return rsi_value
        
        self.values.append(None)
        return None


class MACD(Indicator):
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(
        self, 
        symbol: str, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9,
        price_field: str = 'close'
    ):
        """
        Initialize MACD indicator.
        
        Args:
            symbol: Trading symbol
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_field: Price field to use (close, open, high, low)
        """
        super().__init__(symbol, {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'price_field': price_field
        })
        self.price_history: List[Optional[float]] = []
        self.macd_line: List[Optional[float]] = []
        self.signal_line: List[Optional[float]] = []
        self.histogram: List[Optional[float]] = []
        self.min_data_points = slow_period + signal_period
    
    def update(self, candle: Dict[str, Any]) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """
        Update MACD with new price data.
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram) values
        """
        if self.params['price_field'] not in candle:
            logger.warning(f"Price field {self.params['price_field']} not found in candle for {self.symbol}")
            self.values.append(None)
            self.macd_line.append(None)
            self.signal_line.append(None)
            self.histogram.append(None)
            return None
        
        # Get price and handle None values
        price = candle[self.params['price_field']]
        self.price_history.append(price)
        
        # Calculate MACD if we have enough data
        if len(self.price_history) >= self.min_data_points:
            # Use our robust MACD calculation
            macd_values, signal_values, hist_values = safe_macd(
                self.price_history,
                self.params['fast_period'],
                self.params['slow_period'],
                self.params['signal_period']
            )
            
            macd_value = macd_values[-1] if macd_values else None
            signal_value = signal_values[-1] if signal_values else None
            hist_value = hist_values[-1] if hist_values else None
            
            self.macd_line.append(macd_value)
            self.signal_line.append(signal_value)
            self.histogram.append(hist_value)
            
            # Store the MACD line as the primary value
            self.values.append(macd_value)
            self.is_ready = True
            
            return macd_value, signal_value, hist_value
        
        self.macd_line.append(None)
        self.signal_line.append(None)
        self.histogram.append(None)
        self.values.append(None)
        return None
    
    def get_macd_line(self) -> List[Optional[float]]:
        """Get MACD line values."""
        return self.macd_line
    
    def get_signal_line(self) -> List[Optional[float]]:
        """Get signal line values."""
        return self.signal_line
    
    def get_histogram(self) -> List[Optional[float]]:
        """Get histogram values."""
        return self.histogram
    
    def get_last_complete_value(self) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """Get the most recent complete MACD values."""
        if not self.is_ready:
            return None
        
        return (self.macd_line[-1], self.signal_line[-1], self.histogram[-1])


class BollingerBands(Indicator):
    """Bollinger Bands indicator."""
    
    def __init__(
        self, 
        symbol: str, 
        period: int = 20, 
        std_dev: float = 2.0,
        price_field: str = 'close'
    ):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            symbol: Trading symbol
            period: Period for moving average
            std_dev: Standard deviation multiplier
            price_field: Price field to use (close, open, high, low)
        """
        super().__init__(symbol, {
            'period': period,
            'std_dev': std_dev,
            'price_field': price_field
        })
        self.price_history: List[Optional[float]] = []
        self.middle_band: List[Optional[float]] = []
        self.upper_band: List[Optional[float]] = []
        self.lower_band: List[Optional[float]] = []
        self.bandwidth: List[Optional[float]] = []
        self.percent_b: List[Optional[float]] = []
        self.min_data_points = period
    
    def update(self, candle: Dict[str, Any]) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """
        Update Bollinger Bands with new price data.
        
        Returns:
            Tuple of (Middle Band, Upper Band, Lower Band) values
        """
        if self.params['price_field'] not in candle:
            logger.warning(f"Price field {self.params['price_field']} not found in candle for {self.symbol}")
            self._append_none_values()
            return None
        
        # Get price and handle None values
        price = candle[self.params['price_field']]
        self.price_history.append(price)
        
        # Calculate Bollinger Bands if we have enough data
        if len(self.price_history) >= self.params['period']:
            # Use our robust Bollinger Bands calculation
            mid, upper, lower = safe_bollinger_bands(
                self.price_history,
                self.params['period'],
                self.params['std_dev']
            )
            
            mid_value = mid[-1] if mid else None
            upper_value = upper[-1] if upper else None
            lower_value = lower[-1] if lower else None
            
            self.middle_band.append(mid_value)
            self.upper_band.append(upper_value)
            self.lower_band.append(lower_value)
            
            # Calculate bandwidth and %B
            if (mid_value is not None and upper_value is not None and 
                lower_value is not None and price is not None):
                
                if mid_value > 0:
                    bandwidth = (upper_value - lower_value) / mid_value
                else:
                    bandwidth = 0
                    
                if (upper_value - lower_value) > 0:
                    percent_b = (price - lower_value) / (upper_value - lower_value)
                else:
                    percent_b = 0.5
                
                self.bandwidth.append(bandwidth)
                self.percent_b.append(percent_b)
            else:
                self.bandwidth.append(None)
                self.percent_b.append(None)
            
            # Store the middle band as the primary value
            self.values.append(mid_value)
            self.is_ready = True
            
            return mid_value, upper_value, lower_value
        
        self._append_none_values()
        return None
    
    def _append_none_values(self):
        """Append None values to all indicator arrays."""
        self.middle_band.append(None)
        self.upper_band.append(None)
        self.lower_band.append(None)
        self.bandwidth.append(None)
        self.percent_b.append(None)
        self.values.append(None)
    
    def get_middle_band(self) -> List[Optional[float]]:
        """Get middle band values."""
        return self.middle_band
    
    def get_upper_band(self) -> List[Optional[float]]:
        """Get upper band values."""
        return self.upper_band
    
    def get_lower_band(self) -> List[Optional[float]]:
        """Get lower band values."""
        return self.lower_band
    
    def get_bandwidth(self) -> List[Optional[float]]:
        """Get bandwidth values."""
        return self.bandwidth
    
    def get_percent_b(self) -> List[Optional[float]]:
        """Get %B values."""
        return self.percent_b
    
    def get_last_complete_value(self) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """Get the most recent complete Bollinger Bands values."""
        if not self.is_ready:
            return None
        
        return (self.middle_band[-1], self.upper_band[-1], self.lower_band[-1])


class ATR(Indicator):
    """Average True Range indicator."""
    
    def __init__(self, symbol: str, period: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            symbol: Trading symbol
            period: ATR period
        """
        super().__init__(symbol, {'period': period})
        self.high_history: List[Optional[float]] = []
        self.low_history: List[Optional[float]] = []
        self.close_history: List[Optional[float]] = []
        self.min_data_points = period + 1  # Need previous close
    
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """Update ATR with new price data."""
        required_fields = ['high', 'low', 'close']
        if not all(field in candle for field in required_fields):
            missing = [f for f in required_fields if f not in candle]
            logger.warning(f"Required fields {missing} not found in candle for {self.symbol}")
            self.values.append(None)
            return None
        
        # Get price data and handle None values
        high, low, close = candle['high'], candle['low'], candle['close']
        
        if high is None or low is None or close is None:
            self.values.append(None)
            return None
        
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Calculate ATR if we have enough data
        if len(self.high_history) >= self.params['period'] + 1:
            # Use our robust ATR calculation
            atr_values = safe_atr(
                self.high_history,
                self.low_history,
                self.close_history,
                self.params['period']
            )
            
            atr_value = atr_values[-1] if atr_values else None
            
            self.values.append(atr_value)
            self.is_ready = True
            return atr_value
        
        self.values.append(None)
        return None


class Stochastic(Indicator):
    """Stochastic Oscillator indicator."""
    
    def __init__(
        self, 
        symbol: str, 
        k_period: int = 14, 
        d_period: int = 3
    ):
        """
        Initialize Stochastic Oscillator indicator.
        
        Args:
            symbol: Trading symbol
            k_period: %K period
            d_period: %D period (moving average of %K)
        """
        super().__init__(symbol, {'k_period': k_period, 'd_period': d_period})
        self.high_history: List[Optional[float]] = []
        self.low_history: List[Optional[float]] = []
        self.close_history: List[Optional[float]] = []
        self.k_values: List[Optional[float]] = []
        self.d_values: List[Optional[float]] = []
        self.min_data_points = k_period
    
    def update(self, candle: Dict[str, Any]) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """
        Update Stochastic Oscillator with new price data.
        
        Returns:
            Tuple of (%K, %D) values
        """
        required_fields = ['high', 'low', 'close']
        if not all(field in candle for field in required_fields):
            missing = [f for f in required_fields if f not in candle]
            logger.warning(f"Required fields {missing} not found in candle for {self.symbol}")
            self._append_none_values()
            return None
        
        # Get price data and handle None values
        high, low, close = candle['high'], candle['low'], candle['close']
        
        if high is None or low is None or close is None:
            self._append_none_values()
            return None
        
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Calculate Stochastic if we have enough data
        if len(self.high_history) >= self.params['k_period']:
            # Use our robust Stochastic calculation
            k_values, d_values = safe_stochastic(
                self.high_history,
                self.low_history,
                self.close_history,
                self.params['k_period'],
                self.params['d_period']
            )
            
            k_value = k_values[-1] if k_values else None
            d_value = d_values[-1] if d_values else None
            
            self.k_values.append(k_value)
            self.d_values.append(d_value)
            
            # Store %K as the primary value
            self.values.append(k_value)
            self.is_ready = True
            
            return k_value, d_value
        
        self._append_none_values()
        return None
    
    def _append_none_values(self):
        """Append None values to all indicator arrays."""
        self.k_values.append(None)
        self.d_values.append(None)
        self.values.append(None)
    
    def get_k_values(self) -> List[Optional[float]]:
        """Get %K values."""
        return self.k_values
    
    def get_d_values(self) -> List[Optional[float]]:
        """Get %D values."""
        return self.d_values
    
    def get_last_complete_value(self) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Get the most recent complete Stochastic values."""
        if not self.is_ready:
            return None
        
        return (self.k_values[-1], self.d_values[-1])


class IndicatorFactory:
    """Factory class for creating technical indicators."""
    
    @staticmethod
    def create(indicator_type: str, symbol: str, params: Dict[str, Any]) -> Optional[Indicator]:
        """
        Create a technical indicator instance.
        
        Args:
            indicator_type: Type of indicator to create
            symbol: Trading symbol
            params: Configuration parameters
            
        Returns:
            Indicator instance or None if type is invalid
        """
        indicator_type = indicator_type.lower()
        
        if indicator_type == 'sma':
            return SMA(
                symbol, 
                params.get('period', 20), 
                params.get('price_field', 'close')
            )
        elif indicator_type == 'ema':
            return EMA(
                symbol, 
                params.get('period', 20), 
                params.get('price_field', 'close'),
                params.get('smoothing', 2.0)
            )
        elif indicator_type == 'rsi':
            return RSI(
                symbol, 
                params.get('period', 14), 
                params.get('price_field', 'close')
            )
        elif indicator_type == 'macd':
            return MACD(
                symbol,
                params.get('fast_period', 12),
                params.get('slow_period', 26),
                params.get('signal_period', 9),
                params.get('price_field', 'close')
            )
        elif indicator_type == 'bollinger_bands' or indicator_type == 'bb':
            return BollingerBands(
                symbol,
                params.get('period', 20),
                params.get('std_dev', 2.0),
                params.get('price_field', 'close')
            )
        elif indicator_type == 'atr':
            return ATR(
                symbol,
                params.get('period', 14)
            )
        elif indicator_type == 'stochastic':
            return Stochastic(
                symbol,
                params.get('k_period', 14),
                params.get('d_period', 3)
            )
        else:
            logger.warning(f"Unknown indicator type: {indicator_type}")
            return None


class VectorizedIndicators:
    """Class for vectorized indicator calculations using pandas."""
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int, price_col: str = 'close') -> pd.Series:
        """
        Calculate SMA for an entire DataFrame.
        
        Args:
            df: DataFrame with price data
            period: SMA period
            price_col: Column name for price data
            
        Returns:
            pandas Series with SMA values
        """
        return df[price_col].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, price_col: str = 'close') -> pd.Series:
        """
        Calculate EMA for an entire DataFrame.
        
        Args:
            df: DataFrame with price data
            period: EMA period
            price_col: Column name for price data
            
        Returns:
            pandas Series with EMA values
        """
        return df[price_col].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """
        Calculate RSI for an entire DataFrame.
        
        Args:
            df: DataFrame with price data
            period: RSI period
            price_col: Column name for price data
            
        Returns:
            pandas Series with RSI values
        """
        # Calculate price changes
        delta = df[price_col].diff()
        
        # Create gain (positive) and loss (negative) Series
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0,
        price_col: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for an entire DataFrame.
        
        Args:
            df: DataFrame with price data
            period: Period for moving average
            std_dev: Standard deviation multiplier
            price_col: Column name for price data
            
        Returns:
            Tuple of (middle, upper, lower) bands as pandas Series
        """
        # Calculate middle band (SMA)
        middle = df[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = df[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)
        
        return middle, upper, lower
    
    @staticmethod
    def calculate_for_multiple_symbols(
        df_dict: Dict[str, pd.DataFrame],
        indicator_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate an indicator for multiple symbols.
        
        Args:
            df_dict: Dictionary mapping symbols to price DataFrames
            indicator_func: Function to calculate indicator
            **kwargs: Arguments to pass to indicator function
            
        Returns:
            Dictionary mapping symbols to calculated indicators
        """
        results = {}
        
        for symbol, df in df_dict.items():
            results[symbol] = indicator_func(df, **kwargs)
            
        return results
