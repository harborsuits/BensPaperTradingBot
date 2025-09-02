"""
Signal Generator Components

Implementation of various signal generation components for the modular strategy system.
These components analyze market data and generate trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from trading_bot.strategies.base_strategy import SignalType
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, SignalGeneratorComponent
)

logger = logging.getLogger(__name__)

class MovingAverageSignalGenerator(SignalGeneratorComponent):
    """
    Generates signals based on moving average crossovers.
    """
    
    def __init__(self, 
                component_id: Optional[str] = None,
                fast_period: int = 20,
                slow_period: int = 50,
                signal_threshold: float = 0.0,
                price_column: str = 'close'):
        """
        Initialize moving average crossover signal generator
        
        Args:
            component_id: Unique component ID
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            signal_threshold: Threshold for signal generation (% difference)
            price_column: Price column to use for calculations
        """
        super().__init__(component_id)
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_threshold': signal_threshold,
            'price_column': price_column
        }
        self.description = f"MA Crossover ({fast_period}/{slow_period})"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals based on moving average crossovers
        
        Args:
            data: Dictionary of market data by symbol
            context: Processing context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {}
        
        # Get parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        threshold = self.parameters['signal_threshold']
        price_col = self.parameters['price_column']
        
        for symbol, df in data.items():
            if len(df) < slow_period:
                logger.debug(f"Not enough data for {symbol}, need at least {slow_period} bars")
                continue
                
            # Calculate moving averages
            if 'ma_fast' not in df.columns or 'ma_slow' not in df.columns:
                df['ma_fast'] = df[price_col].rolling(window=fast_period).mean()
                df['ma_slow'] = df[price_col].rolling(window=slow_period).mean()
            
            # Get last valid values
            last_idx = df.index[-1]
            fast_ma = df.at[last_idx, 'ma_fast']
            slow_ma = df.at[last_idx, 'ma_slow']
            
            # Skip if NaN
            if pd.isna(fast_ma) or pd.isna(slow_ma):
                continue
            
            # Calculate difference
            diff_pct = (fast_ma / slow_ma - 1) * 100
            
            # Generate signal
            if diff_pct > threshold:
                signals[symbol] = SignalType.LONG
            elif diff_pct < -threshold:
                signals[symbol] = SignalType.SHORT
            else:
                signals[symbol] = SignalType.FLAT
                
            # For position scaling
            prev_idx = df.index[-2] if len(df) > 1 else None
            if prev_idx is not None:
                prev_diff_pct = (df.at[prev_idx, 'ma_fast'] / df.at[prev_idx, 'ma_slow'] - 1) * 100
                
                # Strengthen signal if divergence is increasing
                if signals[symbol] == SignalType.LONG and diff_pct > prev_diff_pct:
                    signals[symbol] = SignalType.SCALE_UP
                elif signals[symbol] == SignalType.SHORT and diff_pct < prev_diff_pct:
                    signals[symbol] = SignalType.SCALE_DOWN
        
        return signals

class RSISignalGenerator(SignalGeneratorComponent):
    """
    Generates signals based on Relative Strength Index (RSI).
    """
    
    def __init__(self, 
                component_id: Optional[str] = None,
                period: int = 14,
                overbought: float = 70.0,
                oversold: float = 30.0,
                price_column: str = 'close'):
        """
        Initialize RSI signal generator
        
        Args:
            component_id: Unique component ID
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
            price_column: Price column to use for calculations
        """
        super().__init__(component_id)
        self.parameters = {
            'period': period,
            'overbought': overbought,
            'oversold': oversold,
            'price_column': price_column
        }
        self.description = f"RSI ({period})"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals based on RSI values
        
        Args:
            data: Dictionary of market data by symbol
            context: Processing context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {}
        
        # Get parameters
        period = self.parameters['period']
        overbought = self.parameters['overbought']
        oversold = self.parameters['oversold']
        price_col = self.parameters['price_column']
        
        for symbol, df in data.items():
            if len(df) < period + 1:
                logger.debug(f"Not enough data for {symbol}, need at least {period+1} bars")
                continue
                
            # Calculate RSI if not already calculated
            if 'rsi' not in df.columns:
                # Calculate price changes
                delta = df[price_col].diff()
                
                # Separate gains and losses
                gain = delta.copy()
                loss = delta.copy()
                gain[gain < 0] = 0
                loss[loss > 0] = 0
                loss = -loss  # Make loss positive
                
                # Calculate average gain and loss
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # Get last RSI value
            last_idx = df.index[-1]
            rsi = df.at[last_idx, 'rsi']
            
            # Skip if NaN
            if pd.isna(rsi):
                continue
            
            # Generate signal
            if rsi > overbought:
                signals[symbol] = SignalType.SHORT
            elif rsi < oversold:
                signals[symbol] = SignalType.LONG
            else:
                signals[symbol] = SignalType.FLAT
                
            # For position scaling
            prev_idx = df.index[-2] if len(df) > 1 else None
            if prev_idx is not None:
                prev_rsi = df.at[prev_idx, 'rsi']
                
                # Strengthen signal if RSI is crossing thresholds
                if signals[symbol] == SignalType.LONG and prev_rsi < rsi:
                    signals[symbol] = SignalType.SCALE_UP
                elif signals[symbol] == SignalType.SHORT and prev_rsi > rsi:
                    signals[symbol] = SignalType.SCALE_DOWN
        
        return signals

class BollingerBandSignalGenerator(SignalGeneratorComponent):
    """
    Generates signals based on Bollinger Bands.
    """
    
    def __init__(self, 
                component_id: Optional[str] = None,
                period: int = 20,
                std_dev: float = 2.0,
                price_column: str = 'close'):
        """
        Initialize Bollinger Band signal generator
        
        Args:
            component_id: Unique component ID
            period: Bollinger Band period
            std_dev: Number of standard deviations
            price_column: Price column to use for calculations
        """
        super().__init__(component_id)
        self.parameters = {
            'period': period,
            'std_dev': std_dev,
            'price_column': price_column
        }
        self.description = f"Bollinger Bands ({period}, {std_dev}σ)"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            data: Dictionary of market data by symbol
            context: Processing context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {}
        
        # Get parameters
        period = self.parameters['period']
        std_dev = self.parameters['std_dev']
        price_col = self.parameters['price_column']
        
        for symbol, df in data.items():
            if len(df) < period:
                logger.debug(f"Not enough data for {symbol}, need at least {period} bars")
                continue
                
            # Calculate Bollinger Bands if not already calculated
            if 'bb_middle' not in df.columns:
                # Calculate middle band (SMA)
                df['bb_middle'] = df[price_col].rolling(window=period).mean()
                
                # Calculate standard deviation
                std = df[price_col].rolling(window=period).std()
                
                # Calculate upper and lower bands
                df['bb_upper'] = df['bb_middle'] + (std * std_dev)
                df['bb_lower'] = df['bb_middle'] - (std * std_dev)
                
                # Calculate bandwidth
                df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                
                # Calculate %B (position within bands)
                df['bb_b'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Get last values
            last_idx = df.index[-1]
            price = df.at[last_idx, price_col]
            upper = df.at[last_idx, 'bb_upper']
            lower = df.at[last_idx, 'bb_lower']
            middle = df.at[last_idx, 'bb_middle']
            
            # Skip if NaN
            if pd.isna(price) or pd.isna(upper) or pd.isna(lower) or pd.isna(middle):
                continue
            
            # Generate signal
            if price > upper:
                signals[symbol] = SignalType.SHORT  # Price above upper band (potential reversal)
            elif price < lower:
                signals[symbol] = SignalType.LONG   # Price below lower band (potential reversal)
            else:
                # Calculate position within bands
                position = (price - lower) / (upper - lower)
                
                if position > 0.8:  # Near upper band
                    signals[symbol] = SignalType.SHORT
                elif position < 0.2:  # Near lower band
                    signals[symbol] = SignalType.LONG
                else:
                    signals[symbol] = SignalType.FLAT
        
        return signals

class MacdSignalGenerator(SignalGeneratorComponent):
    """
    Generates signals based on MACD (Moving Average Convergence Divergence).
    """
    
    def __init__(self, 
                component_id: Optional[str] = None,
                fast_period: int = 12,
                slow_period: int = 26,
                signal_period: int = 9,
                price_column: str = 'close'):
        """
        Initialize MACD signal generator
        
        Args:
            component_id: Unique component ID
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_column: Price column to use for calculations
        """
        super().__init__(component_id)
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'price_column': price_column
        }
        self.description = f"MACD ({fast_period}/{slow_period}/{signal_period})"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals based on MACD
        
        Args:
            data: Dictionary of market data by symbol
            context: Processing context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {}
        
        # Get parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        price_col = self.parameters['price_column']
        
        for symbol, df in data.items():
            if len(df) < slow_period + signal_period:
                logger.debug(f"Not enough data for {symbol}, need at least {slow_period + signal_period} bars")
                continue
                
            # Calculate MACD if not already calculated
            if 'macd' not in df.columns:
                # Calculate EMAs
                ema_fast = df[price_col].ewm(span=fast_period, adjust=False).mean()
                ema_slow = df[price_col].ewm(span=slow_period, adjust=False).mean()
                
                # Calculate MACD line
                df['macd'] = ema_fast - ema_slow
                
                # Calculate signal line
                df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                
                # Calculate histogram
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Get last values
            last_idx = df.index[-1]
            macd = df.at[last_idx, 'macd']
            signal = df.at[last_idx, 'macd_signal']
            hist = df.at[last_idx, 'macd_hist']
            
            # Skip if NaN
            if pd.isna(macd) or pd.isna(signal):
                continue
            
            # Generate signal
            if macd > signal:
                signals[symbol] = SignalType.LONG
            elif macd < signal:
                signals[symbol] = SignalType.SHORT
            else:
                signals[symbol] = SignalType.FLAT
                
            # Check for momentum
            if len(df) > 1:
                prev_idx = df.index[-2]
                prev_hist = df.at[prev_idx, 'macd_hist']
                
                # Strengthen signal if histogram is expanding
                if signals[symbol] == SignalType.LONG and hist > prev_hist:
                    signals[symbol] = SignalType.SCALE_UP
                elif signals[symbol] == SignalType.SHORT and hist < prev_hist:
                    signals[symbol] = SignalType.SCALE_DOWN
        
        return signals

class ATRBreakoutSignalGenerator(SignalGeneratorComponent):
    """
    Generates signals based on price breakouts normalized by ATR.
    """
    
    def __init__(self, 
                component_id: Optional[str] = None,
                atr_period: int = 14,
                breakout_period: int = 20,
                breakout_multiplier: float = 1.5):
        """
        Initialize ATR breakout signal generator
        
        Args:
            component_id: Unique component ID
            atr_period: ATR calculation period
            breakout_period: Lookback period for high/low
            breakout_multiplier: ATR multiplier for breakout threshold
        """
        super().__init__(component_id)
        self.parameters = {
            'atr_period': atr_period,
            'breakout_period': breakout_period,
            'breakout_multiplier': breakout_multiplier
        }
        self.description = f"ATR Breakout ({breakout_period}, {atr_period})"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals based on ATR breakouts
        
        Args:
            data: Dictionary of market data by symbol
            context: Processing context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {}
        
        # Get parameters
        atr_period = self.parameters['atr_period']
        breakout_period = self.parameters['breakout_period']
        multiplier = self.parameters['breakout_multiplier']
        
        for symbol, df in data.items():
            if len(df) < max(atr_period, breakout_period) + 1:
                logger.debug(f"Not enough data for {symbol}, need at least {max(atr_period, breakout_period) + 1} bars")
                continue
                
            # Calculate ATR if not already calculated
            if 'atr' not in df.columns:
                # Calculate True Range
                high_low = df['high'] - df['low']
                high_close_prev = abs(df['high'] - df['close'].shift(1))
                low_close_prev = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                
                # Calculate ATR
                df['atr'] = tr.rolling(window=atr_period).mean()
            
            # Get last values
            last_idx = df.index[-1]
            last_price = df.at[last_idx, 'close']
            last_atr = df.at[last_idx, 'atr']
            
            # Skip if NaN
            if pd.isna(last_atr):
                continue
            
            # Calculate breakout levels
            lookback_high = df['high'].rolling(window=breakout_period).max().shift(1)
            lookback_low = df['low'].rolling(window=breakout_period).min().shift(1)
            
            last_high = lookback_high.iloc[-1]
            last_low = lookback_low.iloc[-1]
            
            # Add ATR buffer to breakout levels
            breakout_buffer = last_atr * multiplier
            upper_breakout = last_high + breakout_buffer
            lower_breakout = last_low - breakout_buffer
            
            # Generate signal
            if last_price > upper_breakout:
                signals[symbol] = SignalType.LONG
            elif last_price < lower_breakout:
                signals[symbol] = SignalType.SHORT
            else:
                signals[symbol] = SignalType.FLAT
        
        return signals

class CompositeSignalGenerator(SignalGeneratorComponent):
    """
    Combines signals from multiple generators with weighted voting.
    """
    
    def __init__(self, 
                component_id: Optional[str] = None,
                generators: List[Tuple[SignalGeneratorComponent, float]] = None,
                threshold: float = 0.6):
        """
        Initialize composite signal generator
        
        Args:
            component_id: Unique component ID
            generators: List of (generator, weight) tuples
            threshold: Threshold for signal generation (between 0 and 1)
        """
        super().__init__(component_id)
        self.generators = generators or []
        self.parameters = {
            'threshold': threshold
        }
        
        generator_names = [g[0].description for g in self.generators]
        self.description = f"Composite ({', '.join(generator_names)})"
    
    def add_generator(self, generator: SignalGeneratorComponent, weight: float = 1.0) -> None:
        """
        Add a signal generator with specified weight
        
        Args:
            generator: Signal generator component
            weight: Weight for this generator
        """
        self.generators.append((generator, weight))
        
        # Update description
        generator_names = [g[0].description for g in self.generators]
        self.description = f"Composite ({', '.join(generator_names)})"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals by combining multiple generators
        
        Args:
            data: Dictionary of market data by symbol
            context: Processing context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        if not self.generators:
            return {}
        
        # Initialize tallies for each symbol
        symbols = list(data.keys())
        tallies = {symbol: 0.0 for symbol in symbols}
        total_weights = {symbol: 0.0 for symbol in symbols}
        
        # Process each generator
        for generator, weight in self.generators:
            generator_signals = generator.generate_signals(data, context)
            
            # Update tallies based on signals
            for symbol, signal in generator_signals.items():
                if symbol not in tallies:
                    continue
                    
                # Convert signal to numeric value (-1 to 1)
                if signal == SignalType.LONG or signal == SignalType.SCALE_UP:
                    value = 1.0
                elif signal == SignalType.SHORT or signal == SignalType.SCALE_DOWN:
                    value = -1.0
                else:
                    value = 0.0
                
                # Update tally with weighted value
                tallies[symbol] += value * weight
                total_weights[symbol] += weight
        
        # Calculate final signals based on tallies
        threshold = self.parameters['threshold']
        signals = {}
        
        for symbol, tally in tallies.items():
            if total_weights[symbol] == 0:
                continue
                
            # Normalize tally by total weight
            normalized_tally = tally / total_weights[symbol]
            
            # Generate signal based on normalized tally and threshold
            if normalized_tally >= threshold:
                signals[symbol] = SignalType.LONG
            elif normalized_tally <= -threshold:
                signals[symbol] = SignalType.SHORT
            else:
                signals[symbol] = SignalType.FLAT
        
        return signals
