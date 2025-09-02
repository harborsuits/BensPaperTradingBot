"""Sequential Market Data Provider implementation to prevent lookahead bias."""

import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

from .market_data_provider import MarketDataProvider


class SequentialDataProvider(MarketDataProvider):
    """
    Provides market data sequentially to prevent lookahead bias.
    
    The SequentialDataProvider ensures strategies can only access historical
    data that would have been available at the point of decision making,
    preventing any form of lookahead bias or "future peeking."
    
    Features:
    - Maintains a sliding window of historical data
    - Only reveals data up to the current point in time
    - Supports on-demand indicator calculation
    - Enforces strict forward-only progression
    """
    
    def __init__(self, data_source: Union[Dict, pd.DataFrame, MarketDataProvider], 
                 lookback_window: int = 100,
                 seed: Optional[int] = None,
                 calculate_indicators: bool = True):
        """
        Initialize the sequential data provider.
        
        Args:
            data_source: The source of market data - can be a dictionary, DataFrame, 
                         or another MarketDataProvider
            lookback_window: Maximum number of historical bars to maintain
            seed: Random seed for reproducibility
            calculate_indicators: Whether to calculate indicators on each tick
        """
        self.logger = logging.getLogger(__name__)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize data storage
        self.data = None
        self.current_index = 0
        self.lookback_window = lookback_window
        self.calculate_indicators = calculate_indicators
        
        # Determine data source type and load data
        if isinstance(data_source, dict):
            self.data = data_source
        elif isinstance(data_source, pd.DataFrame):
            self.data = self._dataframe_to_dict(data_source)
        elif isinstance(data_source, MarketDataProvider):
            # This allows wrapping another provider like RandomWalkDataProvider
            self.underlying_provider = data_source
            # For providers we need to load data progressively
            self.data = {}
            self.max_days = 0
        else:
            raise ValueError("Unsupported data source type. Must be dict, DataFrame, or MarketDataProvider")
            
        # Setup state
        self.is_initialized = False
        self.current_day = 0
        self.history = {}  # Stores the sliding window of historical data
        
        self.logger.info(f"SequentialDataProvider initialized with lookback_window={lookback_window}")
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert a pandas DataFrame to our internal data format."""
        data_dict = {}
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['default']
        
        for symbol in symbols:
            if 'symbol' in df.columns:
                symbol_data = df[df['symbol'] == symbol]
            else:
                symbol_data = df
                
            # Convert to our data format
            data_dict[symbol] = {
                'prices': symbol_data['close'].tolist() if 'close' in symbol_data else [],
                'highs': symbol_data['high'].tolist() if 'high' in symbol_data else [],
                'lows': symbol_data['low'].tolist() if 'low' in symbol_data else [],
                'volumes': symbol_data['volume'].tolist() if 'volume' in symbol_data else [],
                'timestamps': symbol_data['timestamp'].tolist() if 'timestamp' in symbol_data else [],
            }
            
        return data_dict
        
    def initialize(self, max_days: int = 30) -> None:
        """Initialize the data provider with the maximum number of days to simulate.
        
        Args:
            max_days: Maximum number of days to provide data for
        """
        # Add extra lookback days for warm-up period
        self.max_simulation_days = max_days
        max_days_with_warmup = max_days + self.lookback_window
        if hasattr(self, 'underlying_provider'):
            self.max_days = max_days_with_warmup
            self.logger.info(f"Initializing sequential data provider from {self.underlying_provider.__class__.__name__}")
            # Pre-fetch lookback window plus extra days for warm-up
            for day in range(self.lookback_window):
                if day < max_days:
                    try:
                        daily_data = self.underlying_provider.get_data(day)
                        self.logger.debug(f"Day {day}: Got {len(daily_data)} symbols from underlying provider")
                        
                        if not self.data:
                            # Initialize with structure from first day
                            for symbol in daily_data.keys():
                                self.data[symbol] = {
                                    'prices': [],
                                    'highs': [],
                                    'lows': [],
                                    'volumes': [],
                                    'timestamps': [],
                                }
                                self.history[symbol] = {
                                    'prices': [],
                                    'highs': [],
                                    'lows': [],
                                    'volumes': [],
                                    'timestamps': [],
                                }
                            self.logger.info(f"Initialized data structure with {len(self.data)} symbols")
                        
                        # Append data for this day
                        for symbol, symbol_data in daily_data.items():
                            if symbol in self.data:
                                # For RandomWalkDataProvider, ensure we get the price
                                price = symbol_data.get('price')
                                if price is None and 'close' in symbol_data:
                                    price = symbol_data.get('close')
                                    
                                self.data[symbol]['prices'].append(price)
                                self.data[symbol]['highs'].append(symbol_data.get('high', price))
                                self.data[symbol]['lows'].append(symbol_data.get('low', price))
                                self.data[symbol]['volumes'].append(symbol_data.get('volume', 0))
                                self.data[symbol]['timestamps'].append(symbol_data.get('timestamp', day))
                                
                                # Also update history for warm-up
                                self.history[symbol]['prices'].append(price)
                                self.history[symbol]['highs'].append(symbol_data.get('high', price))
                                self.history[symbol]['lows'].append(symbol_data.get('low', price))
                                self.history[symbol]['volumes'].append(symbol_data.get('volume', 0))
                                self.history[symbol]['timestamps'].append(symbol_data.get('timestamp', day))
                    except Exception as e:
                        self.logger.error(f"Error fetching day {day}: {str(e)}")
                        
            # Verify data was loaded properly
            if len(self.data) == 0:
                self.logger.error("Failed to load any data from underlying provider!")
            else:
                sample_symbol = list(self.data.keys())[0]
                self.logger.info(f"Loaded {len(self.data[sample_symbol]['prices'])} data points for {sample_symbol}")
                
            self.current_index = 0
        else:
            # For static data sources, just initialize history with lookback window
            for symbol in self.data.keys():
                self.history[symbol] = {
                    'prices': self.data[symbol]['prices'][:self.lookback_window],
                    'highs': self.data[symbol]['highs'][:self.lookback_window],
                    'lows': self.data[symbol]['lows'][:self.lookback_window],
                    'volumes': self.data[symbol]['volumes'][:self.lookback_window],
                    'timestamps': self.data[symbol]['timestamps'][:self.lookback_window] if 'timestamps' in self.data[symbol] else list(range(self.lookback_window)),
                }
                
            self.current_index = self.lookback_window
            
        self.is_initialized = True
        self.logger.info(f"SequentialDataProvider initialized with {len(self.data)} symbols")
        
    def get_data(self, day: int) -> Dict[str, Any]:
        """
        Get market data for the specified day.
        
        For sequential data provider, day parameter is used differently:
        - First call initializes with day=0
        - Subsequent calls must use incremental days (1, 2, 3...)
        - Attempting to go backwards will raise an error
        
        Args:
            day: The day index (must be sequential)
            
        Returns:
            Market data dictionary with only data visible up to this point
        """
        if not self.is_initialized:
            self.initialize(max_days=day+100)  # Initialize with some buffer
            
        # Validate day is sequential
        if day < self.current_day:
            raise ValueError(f"Cannot go backwards in time: requested day {day}, current day {self.current_day}")
        elif day > self.current_day + 1:
            self.logger.warning(f"Skipping ahead from day {self.current_day} to {day}")
            
        self.current_day = day
        
        # Handle fetching more data from underlying provider if needed
        if hasattr(self, 'underlying_provider'):
            try:
                # Fetch a new day's worth of data from underlying provider
                daily_data = self.underlying_provider.get_data(day)
                self.logger.debug(f"Day {day}: Fetched {len(daily_data)} symbols from underlying provider")
                
                # Append new data to our historical datasets
                for symbol, symbol_data in daily_data.items():
                    if symbol in self.data:
                        # For RandomWalkDataProvider, ensure we get the price
                        price = symbol_data.get('price')
                        if price is None and 'close' in symbol_data:
                            price = symbol_data.get('close')
                            
                        # Only append if this is actually new data
                        if day >= len(self.data[symbol]['prices']):
                            self.data[symbol]['prices'].append(price)
                            self.data[symbol]['highs'].append(symbol_data.get('high', price))
                            self.data[symbol]['lows'].append(symbol_data.get('low', price))
                            self.data[symbol]['volumes'].append(symbol_data.get('volume', 0))
                            self.data[symbol]['timestamps'].append(symbol_data.get('timestamp', day))
                        
                            # Update sliding window for history
                            self.history[symbol]['prices'].append(price)
                            self.history[symbol]['highs'].append(symbol_data.get('high', price))
                            self.history[symbol]['lows'].append(symbol_data.get('low', price))
                            self.history[symbol]['volumes'].append(symbol_data.get('volume', 0))
                            
                            # Maintain lookback window size
                            if len(self.history[symbol]['prices']) > self.lookback_window:
                                self.history[symbol]['prices'] = self.history[symbol]['prices'][-self.lookback_window:]
                                self.history[symbol]['highs'] = self.history[symbol]['highs'][-self.lookback_window:]
                                self.history[symbol]['lows'] = self.history[symbol]['lows'][-self.lookback_window:]
                                self.history[symbol]['volumes'] = self.history[symbol]['volumes'][-self.lookback_window:]
            except Exception as e:
                self.logger.error(f"Error fetching data for day {day} from underlying provider: {str(e)}")
        
        # Prepare the current visible data
        result = {}
        
        # In debug mode, log extra information
        debug_enabled = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        
        # Ensure we have data before proceeding
        if not self.data or len(self.data) == 0:
            self.logger.error("No data available in the sequential data provider")
            return {}
            
        for symbol, symbol_data in self.data.items():
            # Make sure we have data for this day
            if day < len(symbol_data['prices']):
                # Current day's data point
                current_price = symbol_data['prices'][day]
                current_high = symbol_data.get('highs', [])[day] if symbol_data.get('highs') and day < len(symbol_data.get('highs', [])) else current_price
                current_low = symbol_data.get('lows', [])[day] if symbol_data.get('lows') and day < len(symbol_data.get('lows', [])) else current_price
                current_volume = symbol_data.get('volumes', [])[day] if symbol_data.get('volumes') and day < len(symbol_data.get('volumes', [])) else 0
                
                # Only update history if we haven't already (since we may have updated it earlier in this method)
                if len(self.history[symbol]['prices']) <= day:
                    self.history[symbol]['prices'].append(current_price)
                    self.history[symbol]['highs'].append(current_high)
                    self.history[symbol]['lows'].append(current_low)
                    self.history[symbol]['volumes'].append(current_volume)
                    
                    # Maintain lookback window size
                    if len(self.history[symbol]['prices']) > self.lookback_window:
                        self.history[symbol]['prices'] = self.history[symbol]['prices'][-self.lookback_window:]
                        self.history[symbol]['highs'] = self.history[symbol]['highs'][-self.lookback_window:]
                        self.history[symbol]['lows'] = self.history[symbol]['lows'][-self.lookback_window:]
                        self.history[symbol]['volumes'] = self.history[symbol]['volumes'][-self.lookback_window:]
                
                # Create market data with visible history
                result[symbol] = {
                    'price': current_price,
                    'high': current_high,
                    'low': current_low,
                    'volume': current_volume,
                    'day': day,  # Include the day for strategies that need it
                    'history': {
                        'prices': self.history[symbol]['prices'],
                        'highs': self.history[symbol]['highs'],
                        'lows': self.history[symbol]['lows'],
                        'volumes': self.history[symbol]['volumes'],
                    }
                }
                
                # Calculate indicators if enabled
                if self.calculate_indicators:
                    self._calculate_indicators(result[symbol])
                    
        # Log diagnostic information
        if debug_enabled and len(result) > 0:
            sample_symbol = list(result.keys())[0]
            self.logger.debug(f"Day {day}: Returning data for {len(result)} symbols. Sample price for {sample_symbol}: {result[sample_symbol].get('price')}")
            self.logger.debug(f"History length for {sample_symbol}: {len(result[sample_symbol]['history']['prices'])}")
        
        # Advance to next data point for next call
        self.current_index += 1
        
        return result
        
    def _calculate_indicators(self, data: Dict[str, Any]) -> None:
        """
        Calculate technical indicators for the current data point.
        
        Args:
            data: Market data dictionary to augment with indicators
        """
        try:
            # Use the robust indicator calculations that handle None values properly
            from ..utils.robust_indicators import (
                safe_sma, safe_ema, safe_rsi, safe_macd, safe_bollinger_bands, 
                safe_atr, safe_stochastic, is_bullish_crossover, is_bearish_crossover
            )
            
            prices = data['history']['prices']
            highs = data['history']['highs']
            lows = data['history']['lows']
            
            # Exit early if not enough price data
            if not prices or len(prices) < 2:
                self.logger.warning("Not enough price data to calculate indicators")
                return
                
            # Always calculate basic stats regardless of data length
            data['price'] = prices[-1] if prices else None
            
            # Handle None values in price change calculation
            if len(prices) >= 2 and prices[-1] is not None and prices[-2] is not None:
                data['price_change'] = prices[-1] - prices[-2]
                if prices[-2] > 0:
                    data['price_change_pct'] = (data['price_change'] / prices[-2]) * 100
            
            # Calculate common period moving averages using robust calculation
            sma_periods = [5, 8, 10, 20, 21, 50, 100, 200]
            for period in sma_periods:
                if len(prices) >= period:
                    # Use safe SMA calculation that handles None values
                    full_sma = safe_sma(prices, period)
                    
                    # Only set values if calculation succeeded
                    if full_sma and full_sma[-1] is not None:
                        data[f'sma_{period}'] = full_sma[-1]  # Current value
                        
                        # Store previous value if available
                        if len(full_sma) > 1 and full_sma[-2] is not None:
                            data[f'sma_{period}_prev'] = full_sma[-2]
            
            # Calculate EMA indicators using robust calculation
            ema_periods = [8, 12, 21, 26, 50, 200]
            for period in ema_periods:
                if len(prices) >= period:
                    # Use safe EMA calculation that handles None values
                    full_ema = safe_ema(prices, period)
                    
                    # Only set values if calculation succeeded
                    if full_ema and full_ema[-1] is not None:
                        data[f'ema_{period}'] = full_ema[-1]  # Current value
                        
                        # Store previous value if available
                        if len(full_ema) > 1 and full_ema[-2] is not None:
                            data[f'ema_{period}_prev'] = full_ema[-2]
            
            # Pre-calculate crossover conditions for common SMA combinations
            for fast, slow in [(5, 20), (8, 21), (10, 50), (50, 200)]:
                fast_key, slow_key = f'sma_{fast}', f'sma_{slow}'
                if fast_key in data and slow_key in data:
                    # Current position
                    data[f'sma_{fast}_{slow}_position'] = 1 if data[fast_key] > data[slow_key] else -1
                    
                    # Check for crossovers if previous values exist
                    fast_prev_key, slow_prev_key = f'sma_{fast}_prev', f'sma_{slow}_prev'
                    if fast_prev_key in data and slow_prev_key in data:
                        # Use helper functions to check for crossovers safely
                        data[f'sma_{fast}_{slow}_bullish'] = is_bullish_crossover(
                            data[fast_key], data[slow_key], 
                            data[fast_prev_key], data[slow_prev_key]
                        )
                        data[f'sma_{fast}_{slow}_bearish'] = is_bearish_crossover(
                            data[fast_key], data[slow_key], 
                            data[fast_prev_key], data[slow_prev_key]
                        )
                        data[f'sma_{fast}_{slow}_crossover'] = (
                            data[f'sma_{fast}_{slow}_bullish'] or 
                            data[f'sma_{fast}_{slow}_bearish']
                        )
            
            # Pre-calculate crossover conditions for common EMA combinations
            for fast, slow in [(8, 21), (12, 26)]:
                fast_key, slow_key = f'ema_{fast}', f'ema_{slow}'
                if fast_key in data and slow_key in data:
                    # Current position
                    data[f'ema_{fast}_{slow}_position'] = 1 if data[fast_key] > data[slow_key] else -1
                    
                    # Check for crossovers if previous values exist
                    fast_prev_key, slow_prev_key = f'ema_{fast}_prev', f'ema_{slow}_prev'
                    if fast_prev_key in data and slow_prev_key in data:
                        # Use helper functions to check for crossovers safely
                        data[f'ema_{fast}_{slow}_bullish'] = is_bullish_crossover(
                            data[fast_key], data[slow_key], 
                            data[fast_prev_key], data[slow_prev_key]
                        )
                        data[f'ema_{fast}_{slow}_bearish'] = is_bearish_crossover(
                            data[fast_key], data[slow_key], 
                            data[fast_prev_key], data[slow_prev_key]
                        )
                        data[f'ema_{fast}_{slow}_crossover'] = (
                            data[f'ema_{fast}_{slow}_bullish'] or 
                            data[f'ema_{fast}_{slow}_bearish']
                        )
            
            # Calculate advanced indicators with robust implementations
            if len(prices) >= 14:
                # MACD with safe calculation
                if len(prices) >= 26:
                    macd_line, signal, hist = safe_macd(prices)
                    
                    # Store results if valid
                    if macd_line and signal and hist and macd_line[-1] is not None and signal[-1] is not None:
                        data['macd_line'] = macd_line[-1]
                        data['macd_signal'] = signal[-1]
                        data['macd_histogram'] = hist[-1] if hist[-1] is not None else 0
                        
                        # Calculate crossovers if previous values available
                        if (len(macd_line) > 1 and len(signal) > 1 and 
                            macd_line[-2] is not None and signal[-2] is not None):
                            
                            # Bullish: MACD line crosses above signal
                            data['macd_bullish'] = is_bullish_crossover(
                                macd_line[-1], signal[-1],
                                macd_line[-2], signal[-2]
                            )
                            
                            # Bearish: MACD line crosses below signal
                            data['macd_bearish'] = is_bearish_crossover(
                                macd_line[-1], signal[-1],
                                macd_line[-2], signal[-2]
                            )
                            
                            # Any crossover
                            data['macd_crossover'] = data['macd_bullish'] or data['macd_bearish']
                
                # RSI with safe calculation
                rsi_values = safe_rsi(prices, 14)
                if rsi_values and rsi_values[-1] is not None:
                    data['rsi_14'] = rsi_values[-1]
                    
                    # RSI signals
                    data['rsi_overbought'] = data['rsi_14'] > 70
                    data['rsi_oversold'] = data['rsi_14'] < 30
                
                # RSI 21 with safe calculation
                if len(prices) >= 21:
                    rsi_21_values = safe_rsi(prices, 21)
                    if rsi_21_values and rsi_21_values[-1] is not None:
                        data['rsi_21'] = rsi_21_values[-1]
                
                # Bollinger Bands with safe calculation
                if len(prices) >= 20:
                    mid, upper, lower = safe_bollinger_bands(prices, 20, 2.0)
                    
                    # Store results if valid
                    if (mid and upper and lower and 
                        mid[-1] is not None and upper[-1] is not None and lower[-1] is not None):
                        
                        data['bb_middle'] = mid[-1]
                        data['bb_upper'] = upper[-1]
                        data['bb_lower'] = lower[-1]
                        
                        # Calculate width and percent B
                        if mid[-1] > 0:
                            data['bb_width'] = (upper[-1] - lower[-1]) / mid[-1]
                        else:
                            data['bb_width'] = 0
                        
                        if (upper[-1] - lower[-1]) > 0 and prices[-1] is not None:
                            data['bb_pct_b'] = (prices[-1] - lower[-1]) / (upper[-1] - lower[-1])
                        else:
                            data['bb_pct_b'] = 0.5
                        
                        # Bollinger Band trading signals
                        data['bb_squeeze'] = data['bb_width'] < 0.1  # Narrow bands
                        if prices[-1] is not None:
                            data['bb_breakout_up'] = prices[-1] > upper[-1]  # Price above upper band
                            data['bb_breakout_down'] = prices[-1] < lower[-1]  # Price below lower band
                
                # ATR with safe calculation
                if len(highs) >= 14 and len(lows) >= 14:
                    atr_values = safe_atr(highs, lows, prices, 14)
                    if atr_values and atr_values[-1] is not None:
                        data['atr_14'] = atr_values[-1]
                
                # Stochastic with safe calculation
                if len(highs) >= 14 and len(lows) >= 14:
                    k_values, d_values = safe_stochastic(highs, lows, prices)
                    
                    # Store results if valid
                    if (k_values and d_values and 
                        k_values[-1] is not None and d_values[-1] is not None):
                        
                        data['stoch_k'] = k_values[-1]
                        data['stoch_d'] = d_values[-1]
                        
                        # Stochastic signals
                        data['stoch_overbought'] = k_values[-1] > 80
                        data['stoch_oversold'] = k_values[-1] < 20
                        
                        # Stochastic crossovers
                        if (len(k_values) > 1 and len(d_values) > 1 and 
                            k_values[-2] is not None and d_values[-2] is not None):
                            
                            # Bullish: %K crosses above %D
                            data['stoch_bullish'] = is_bullish_crossover(
                                k_values[-1], d_values[-1],
                                k_values[-2], d_values[-2]
                            )
                            
                            # Bearish: %K crosses below %D
                            data['stoch_bearish'] = is_bearish_crossover(
                                k_values[-1], d_values[-1],
                                k_values[-2], d_values[-2]
                            )
                            
                            # Any crossover
                            data['stoch_crossover'] = data['stoch_bullish'] or data['stoch_bearish']
                            
            # Log success for debugging
            self.logger.debug(f"Successfully calculated indicators for data point")
                            
        except Exception as e:
            self.logger.warning(f"Error calculating indicators: {str(e)}")
            import traceback
            self.logger.debug(f"Error details: {traceback.format_exc()}")

    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols.
        
        Returns:
            List of symbol strings
        """
        if hasattr(self, 'underlying_provider'):
            return self.underlying_provider.get_symbols()
        return list(self.data.keys())
        
    def reset(self) -> None:
        """Reset the provider to the initial state."""
        self.current_index = self.lookback_window if self.is_initialized else 0
        self.current_day = 0
        self.history = {}
        self.is_initialized = False
