"""
Enhanced data provider that integrates with the class-based indicator system.
This builds upon the SequentialDataProvider's functionality but uses the new
object-oriented indicator classes for improved robustness and performance.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import copy
import pandas as pd

from ..data.market_data_provider import MarketDataProvider
from ..data.sequential_data_provider import SequentialDataProvider
from ..utils.indicator_system import (
    IndicatorFactory, SMA, EMA, RSI, MACD, BollingerBands, ATR, Stochastic
)

logger = logging.getLogger(__name__)


class EnhancedDataProvider(SequentialDataProvider):
    """
    Enhanced data provider that uses the class-based indicator system.
    Inherits from SequentialDataProvider to maintain compatibility.
    """
    
    def __init__(self, provider: MarketDataProvider, lookback_window: int = 100):
        """
        Initialize the enhanced data provider.
        
        Args:
            provider: The underlying market data provider
            lookback_window: Number of historical data points to maintain
        """
        super().__init__(provider, lookback_window)
        self.indicators: Dict[str, Dict[str, Any]] = {}
        
    def initialize(self) -> None:
        """Initialize the data provider and set up indicators."""
        super().initialize()
        
        # Initialize indicators for each symbol
        for symbol in self.symbols:
            self.indicators[symbol] = self._create_indicators(symbol)
            
    def _create_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Create indicator instances for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicator instances
        """
        indicators = {}
        
        # Simple Moving Averages
        for period in [5, 8, 10, 20, 21, 50, 100, 200]:
            indicators[f'sma_{period}'] = IndicatorFactory.create(
                'sma', symbol, {'period': period}
            )
            
        # Exponential Moving Averages
        for period in [8, 12, 21, 26, 50, 200]:
            indicators[f'ema_{period}'] = IndicatorFactory.create(
                'ema', symbol, {'period': period}
            )
            
        # RSI
        indicators['rsi_14'] = IndicatorFactory.create(
            'rsi', symbol, {'period': 14}
        )
        indicators['rsi_21'] = IndicatorFactory.create(
            'rsi', symbol, {'period': 21}
        )
        
        # MACD
        indicators['macd'] = IndicatorFactory.create(
            'macd', symbol, {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            }
        )
        
        # Bollinger Bands
        indicators['bollinger_bands'] = IndicatorFactory.create(
            'bollinger_bands', symbol, {
                'period': 20,
                'std_dev': 2.0
            }
        )
        
        # ATR
        indicators['atr'] = IndicatorFactory.create(
            'atr', symbol, {'period': 14}
        )
        
        # Stochastic
        indicators['stochastic'] = IndicatorFactory.create(
            'stochastic', symbol, {
                'k_period': 14,
                'd_period': 3
            }
        )
        
        return indicators
    
    def _update_day(self, day_index: int) -> None:
        """
        Update the data for a specific day.
        
        Args:
            day_index: Index of the day to update
        """
        super()._update_day(day_index)
        
        # Update indicators with the latest data
        for symbol in self.symbols:
            if symbol in self.data_by_day[day_index]:
                self._update_indicators(symbol, self.data_by_day[day_index][symbol])
    
    def _update_indicators(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Update indicators for a symbol with the latest data.
        
        Args:
            symbol: Trading symbol
            data: Latest market data for the symbol
        """
        if symbol not in self.indicators:
            logger.warning(f"No indicators found for symbol: {symbol}")
            return
            
        # Create a candle from the data
        candle = self._create_candle_from_data(data)
        
        # Update each indicator
        for name, indicator in self.indicators[symbol].items():
            if indicator:
                try:
                    indicator.update(candle)
                except Exception as e:
                    logger.warning(f"Error updating indicator {name} for {symbol}: {str(e)}")
    
    def _create_candle_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a candle dictionary from market data.
        
        Args:
            data: Market data
            
        Returns:
            Candle dictionary with OHLCV data
        """
        # Get price history
        price_history = data.get('history', {})
        prices = price_history.get('prices', [])
        highs = price_history.get('highs', [])
        lows = price_history.get('lows', [])
        
        candle = {
            'timestamp': data.get('timestamp', 0),
            'close': prices[-1] if prices else None,
            'high': highs[-1] if highs else None,
            'low': lows[-1] if lows else None,
            'volume': data.get('volume', 0)
        }
        
        # Include open price if available
        if 'open' in data:
            candle['open'] = data['open']
        elif len(prices) > 1:
            # Use previous close as open
            candle['open'] = prices[-2]
        else:
            candle['open'] = candle['close']
            
        return candle
    
    def _calculate_indicators(self, data: Dict[str, Any]) -> None:
        """
        Calculate technical indicators for the current data point.
        This method enhances the SequentialDataProvider's indicator calculation
        by using the class-based indicators. It maintains the same output format
        for backward compatibility.
        
        Args:
            data: Market data dictionary to augment with indicators
        """
        try:
            symbol = data.get('symbol', '')
            
            if symbol not in self.indicators:
                logger.warning(f"No indicators initialized for symbol: {symbol}")
                return
                
            symbol_indicators = self.indicators[symbol]
            
            # Get basic price data
            prices = data['history']['prices']
            
            # Exit early if not enough price data
            if not prices or len(prices) < 2:
                logger.warning("Not enough price data to calculate indicators")
                return
                
            # Always calculate basic stats
            data['price'] = prices[-1] if prices else None
            
            # Calculate price change
            if len(prices) >= 2 and prices[-1] is not None and prices[-2] is not None:
                data['price_change'] = prices[-1] - prices[-2]
                if prices[-2] > 0:
                    data['price_change_pct'] = (data['price_change'] / prices[-2]) * 100
            
            # Add SMA values
            for period in [5, 8, 10, 20, 21, 50, 100, 200]:
                sma_key = f'sma_{period}'
                if sma_key in symbol_indicators and symbol_indicators[sma_key]:
                    sma = symbol_indicators[sma_key]
                    if sma.is_ready:
                        data[sma_key] = sma.get_last_value()
                        if len(sma.get_values()) > 1:
                            data[f'{sma_key}_prev'] = sma.get_value_at(-2)
            
            # Add EMA values
            for period in [8, 12, 21, 26, 50, 200]:
                ema_key = f'ema_{period}'
                if ema_key in symbol_indicators and symbol_indicators[ema_key]:
                    ema = symbol_indicators[ema_key]
                    if ema.is_ready:
                        data[ema_key] = ema.get_last_value()
                        if len(ema.get_values()) > 1:
                            data[f'{ema_key}_prev'] = ema.get_value_at(-2)
            
            # Calculate MA crossovers
            self._calculate_ma_crossovers(data, symbol_indicators)
            
            # Add RSI values
            if 'rsi_14' in symbol_indicators and symbol_indicators['rsi_14']:
                rsi = symbol_indicators['rsi_14']
                if rsi.is_ready:
                    data['rsi_14'] = rsi.get_last_value()
                    
                    # RSI signals
                    if data['rsi_14'] is not None:
                        data['rsi_overbought'] = data['rsi_14'] > 70
                        data['rsi_oversold'] = data['rsi_14'] < 30
            
            if 'rsi_21' in symbol_indicators and symbol_indicators['rsi_21']:
                rsi21 = symbol_indicators['rsi_21']
                if rsi21.is_ready:
                    data['rsi_21'] = rsi21.get_last_value()
            
            # Add MACD values
            if 'macd' in symbol_indicators and symbol_indicators['macd']:
                macd = symbol_indicators['macd']
                if macd.is_ready:
                    data['macd_line'] = macd.get_last_value()
                    data['macd_signal'] = macd.signal_line[-1] if macd.signal_line else None
                    data['macd_histogram'] = macd.histogram[-1] if macd.histogram else None
                    
                    # MACD crossover signals
                    if (len(macd.macd_line) > 1 and len(macd.signal_line) > 1):
                        macd_current = macd.macd_line[-1]
                        signal_current = macd.signal_line[-1]
                        macd_prev = macd.macd_line[-2]
                        signal_prev = macd.signal_line[-2]
                        
                        if all(v is not None for v in [macd_current, signal_current, macd_prev, signal_prev]):
                            current_diff = macd_current - signal_current
                            prev_diff = macd_prev - signal_prev
                            
                            data['macd_crossover'] = (current_diff * prev_diff) <= 0
                            data['macd_bullish'] = current_diff > 0 and prev_diff <= 0
                            data['macd_bearish'] = current_diff < 0 and prev_diff >= 0
            
            # Add Bollinger Bands values
            if 'bollinger_bands' in symbol_indicators and symbol_indicators['bollinger_bands']:
                bb = symbol_indicators['bollinger_bands']
                if bb.is_ready:
                    data['bb_middle'] = bb.middle_band[-1] if bb.middle_band else None
                    data['bb_upper'] = bb.upper_band[-1] if bb.upper_band else None
                    data['bb_lower'] = bb.lower_band[-1] if bb.lower_band else None
                    data['bb_width'] = bb.bandwidth[-1] if bb.bandwidth else None
                    data['bb_pct_b'] = bb.percent_b[-1] if bb.percent_b else None
                    
                    # Bollinger Band signals
                    if data['bb_width'] is not None:
                        data['bb_squeeze'] = data['bb_width'] < 0.1
                    
                    if all(v is not None for v in [data.get('price'), data.get('bb_upper'), data.get('bb_lower')]):
                        data['bb_breakout_up'] = data['price'] > data['bb_upper']
                        data['bb_breakout_down'] = data['price'] < data['bb_lower']
            
            # Add ATR value
            if 'atr' in symbol_indicators and symbol_indicators['atr']:
                atr = symbol_indicators['atr']
                if atr.is_ready:
                    data['atr_14'] = atr.get_last_value()
            
            # Add Stochastic values
            if 'stochastic' in symbol_indicators and symbol_indicators['stochastic']:
                stoch = symbol_indicators['stochastic']
                if stoch.is_ready:
                    data['stoch_k'] = stoch.k_values[-1] if stoch.k_values else None
                    data['stoch_d'] = stoch.d_values[-1] if stoch.d_values else None
                    
                    # Stochastic signals
                    if data['stoch_k'] is not None:
                        data['stoch_overbought'] = data['stoch_k'] > 80
                        data['stoch_oversold'] = data['stoch_k'] < 20
                    
                    # Stochastic crossover
                    if (len(stoch.k_values) > 1 and len(stoch.d_values) > 1):
                        k_current = stoch.k_values[-1]
                        d_current = stoch.d_values[-1]
                        k_prev = stoch.k_values[-2]
                        d_prev = stoch.d_values[-2]
                        
                        if all(v is not None for v in [k_current, d_current, k_prev, d_prev]):
                            current_diff = k_current - d_current
                            prev_diff = k_prev - d_prev
                            
                            data['stoch_crossover'] = (current_diff * prev_diff) <= 0
                            data['stoch_bullish'] = current_diff > 0 and prev_diff <= 0
                            data['stoch_bearish'] = current_diff < 0 and prev_diff >= 0
            
            logger.debug(f"Successfully calculated enhanced indicators for {symbol}")
                            
        except Exception as e:
            logger.warning(f"Error calculating enhanced indicators: {str(e)}")
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
    
    def _calculate_ma_crossovers(self, data: Dict[str, Any], indicators: Dict[str, Any]) -> None:
        """
        Calculate moving average crossovers.
        
        Args:
            data: Market data to update with crossover signals
            indicators: Dictionary of indicator instances
        """
        # SMA crossovers
        for fast, slow in [(5, 20), (8, 21), (10, 50), (50, 200)]:
            fast_key = f'sma_{fast}'
            slow_key = f'sma_{slow}'
            
            if fast_key in data and slow_key in data:
                if data[fast_key] is not None and data[slow_key] is not None:
                    # Current position
                    data[f'sma_{fast}_{slow}_position'] = 1 if data[fast_key] > data[slow_key] else -1
                    
                    # Crossover detection
                    fast_prev = f'{fast_key}_prev'
                    slow_prev = f'{slow_key}_prev'
                    
                    if fast_prev in data and slow_prev in data:
                        if data[fast_prev] is not None and data[slow_prev] is not None:
                            current_diff = data[fast_key] - data[slow_key]
                            prev_diff = data[fast_prev] - data[slow_prev]
                            
                            data[f'sma_{fast}_{slow}_crossover'] = (current_diff * prev_diff) <= 0
                            data[f'sma_{fast}_{slow}_bullish'] = current_diff > 0 and prev_diff <= 0
                            data[f'sma_{fast}_{slow}_bearish'] = current_diff < 0 and prev_diff >= 0
        
        # EMA crossovers
        for fast, slow in [(8, 21), (12, 26)]:
            fast_key = f'ema_{fast}'
            slow_key = f'ema_{slow}'
            
            if fast_key in data and slow_key in data:
                if data[fast_key] is not None and data[slow_key] is not None:
                    # Current position
                    data[f'ema_{fast}_{slow}_position'] = 1 if data[fast_key] > data[slow_key] else -1
                    
                    # Crossover detection
                    fast_prev = f'{fast_key}_prev'
                    slow_prev = f'{slow_key}_prev'
                    
                    if fast_prev in data and slow_prev in data:
                        if data[fast_prev] is not None and data[slow_prev] is not None:
                            current_diff = data[fast_key] - data[slow_key]
                            prev_diff = data[fast_prev] - data[slow_prev]
                            
                            data[f'ema_{fast}_{slow}_crossover'] = (current_diff * prev_diff) <= 0
                            data[f'ema_{fast}_{slow}_bullish'] = current_diff > 0 and prev_diff <= 0
                            data[f'ema_{fast}_{slow}_bearish'] = current_diff < 0 and prev_diff >= 0
