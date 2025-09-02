"""
WeightedAvgPeak Strategy

This module implements the WeightedAvgPeak strategy from ilcardella's TradingBot,
adapted to work within our strategy framework with signal confidence scoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WeightedAvgPeakStrategy:
    """
    WeightedAvgPeak strategy implementation
    
    This strategy uses weighted averages and peak detection to identify
    potential entry and exit points for trades.
    """
    
    def __init__(self, name="Weighted Average Peak", parameters=None):
        """
        Initialize the strategy
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.params = parameters or {}
        
        # Configure strategy parameters
        self.price_target_pct = self.params.get('price_target_pct', 2.0)
        self.stop_loss_pct = self.params.get('stop_loss_pct', 1.5)
        self.max_intraday_multiplier = self.params.get('max_intraday_multiplier', 2.0)
        
        # Timeframes for analysis
        self.base_timeframe = self.params.get('base_timeframe', '1d')
        self.intraday_timeframe = self.params.get('intraday_timeframe', '1h')
        
        # Weighted average configuration
        self.volume_factor = self.params.get('volume_factor', 0.3)
        self.price_factor = self.params.get('price_factor', 0.7)
        
        # Moving averages
        self.short_ma = self.params.get('short_ma', 20)
        self.long_ma = self.params.get('long_ma', 50)
        self.signal_ma = self.params.get('signal_ma', 9)
        
        # Peak detection parameters
        self.peak_detection_periods = self.params.get('peak_detection_periods', 5)
        self.peak_detection_threshold = self.params.get('peak_detection_threshold', 0.5)
        
        logger.info(f"Initialized {self.name} strategy")
    
    @classmethod
    def is_available(cls):
        """Check if strategy is available"""
        return True
    
    def generate_signals(self, data, **kwargs) -> Dict[str, Any]:
        """
        Generate trading signals based on the weighted average peak strategy
        
        Args:
            data: DataFrame with OHLCV price data
            **kwargs: Additional parameters
            
        Returns:
            Dict: Signal information
        """
        # Clone the data to avoid modifying the original
        df = data.copy()
        
        if df.empty:
            logger.warning("Empty data provided to strategy")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'target': None,
                'stop': None,
                'reason': 'Empty data'
            }
        
        # Verify we have the minimum required data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns in data. Required: {required_columns}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'target': None,
                'stop': None,
                'reason': 'Missing required data'
            }
        
        # Add technical indicators
        df = self._add_indicators(df)
        
        # Get current market conditions
        current_conditions = self._analyze_market_conditions(df)
        
        # Signal generation based on strategy rules
        signal = self._generate_signal(df, current_conditions)
        
        # Calculate proper risk parameters
        if signal['action'] == 'buy':
            current_price = df['close'].iloc[-1]
            signal['target'] = current_price * (1 + self.price_target_pct / 100)
            signal['stop'] = current_price * (1 - self.stop_loss_pct / 100)
        elif signal['action'] == 'sell':
            current_price = df['close'].iloc[-1]
            signal['target'] = current_price * (1 - self.price_target_pct / 100)
            signal['stop'] = current_price * (1 + self.stop_loss_pct / 100)
        
        logger.debug(f"Generated signal: {signal['action']} with confidence {signal['confidence']:.2f}")
        return signal
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with added indicators
        """
        # Verify we have enough data
        if len(df) < self.long_ma + 10:
            logger.warning(f"Not enough data for indicators, need at least {self.long_ma + 10} periods")
            return df
        
        # Add simple moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_ma).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_ma).mean()
        
        # Add exponential moving averages
        df['ema_short'] = df['close'].ewm(span=self.short_ma, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_ma, adjust=False).mean()
        
        # Calculate MACD
        df['macd'] = df['ema_short'] - df['ema_long']
        df['macd_signal'] = df['macd'].ewm(span=self.signal_ma, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Weighted Average Price-Volume
        df['weighted_avg'] = (df['close'] * self.price_factor) + (df['volume'] * self.volume_factor)
        df['weighted_avg'] = df['weighted_avg'] / (df['weighted_avg'].rolling(window=self.short_ma).mean())
        
        # Detect peaks and troughs
        df['is_peak'] = self._detect_peaks(df['weighted_avg'], up=True)
        df['is_trough'] = self._detect_peaks(df['weighted_avg'], up=False)
        
        # Volatility
        df['atr'] = self._calculate_atr(df)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: Price DataFrame
            window: ATR window
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _detect_peaks(self, series: pd.Series, up: bool = True) -> pd.Series:
        """
        Detect peaks or troughs in a series
        
        Args:
            series: Data series
            up: If True, detect peaks; if False, detect troughs
            
        Returns:
            Boolean series where True indicates a peak/trough
        """
        # Handle NaN values
        series = series.fillna(method='ffill')
        
        # Initialize result series
        is_extreme = pd.Series(False, index=series.index)
        
        # Need at least 2*n+1 periods
        if len(series) < 2 * self.peak_detection_periods + 1:
            return is_extreme
        
        # Detect peaks or troughs
        for i in range(self.peak_detection_periods, len(series) - self.peak_detection_periods):
            left = series.iloc[i - self.peak_detection_periods:i]
            right = series.iloc[i + 1:i + self.peak_detection_periods + 1]
            
            if up:
                # Peak detection
                if (series.iloc[i] > left.max()) and (series.iloc[i] > right.max()):
                    is_extreme.iloc[i] = True
            else:
                # Trough detection
                if (series.iloc[i] < left.min()) and (series.iloc[i] < right.min()):
                    is_extreme.iloc[i] = True
        
        return is_extreme
    
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market conditions
        
        Args:
            df: Price DataFrame with indicators
            
        Returns:
            Dict with market condition analysis
        """
        if len(df) < 2:
            return {'trend': 'unknown', 'volatility': 'unknown', 'momentum': 'unknown'}
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Determine trend
        if latest['sma_short'] > latest['sma_long'] and latest['close'] > latest['sma_short']:
            trend = 'strong_up'
        elif latest['sma_short'] > latest['sma_long']:
            trend = 'up'
        elif latest['sma_short'] < latest['sma_long'] and latest['close'] < latest['sma_short']:
            trend = 'strong_down'
        elif latest['sma_short'] < latest['sma_long']:
            trend = 'down'
        else:
            trend = 'sideways'
        
        # Determine volatility
        atr_percent = latest['atr'] / latest['close'] * 100 if latest['close'] > 0 else 0
        if atr_percent > 2.5:
            volatility = 'high'
        elif atr_percent > 1.5:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # Determine momentum
        if latest['macd'] > 0 and latest['macd'] > latest['macd_signal']:
            momentum = 'strong_up'
        elif latest['macd'] > 0:
            momentum = 'up'
        elif latest['macd'] < 0 and latest['macd'] < latest['macd_signal']:
            momentum = 'strong_down'
        elif latest['macd'] < 0:
            momentum = 'down'
        else:
            momentum = 'neutral'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'momentum': momentum,
            'rsi': latest['rsi'],
            'weighted_avg': latest['weighted_avg'],
            'is_peak': latest['is_peak'],
            'is_trough': latest['is_trough']
        }
    
    def _generate_signal(self, df: pd.DataFrame, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on strategy rules
        
        Args:
            df: Price DataFrame with indicators
            conditions: Current market conditions
            
        Returns:
            Dict with signal information
        """
        # Default signal
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'target': None,
            'stop': None,
            'reason': 'No signal'
        }
        
        # Need enough data
        if len(df) < self.long_ma + 10:
            signal['reason'] = 'Insufficient data'
            return signal
        
        # Get recent data for analysis
        recent = df.iloc[-10:]
        latest = df.iloc[-1]
        
        # Buy signal conditions
        buy_conditions = [
            conditions['trend'] in ['up', 'strong_up'],
            latest['close'] > latest['sma_short'],
            latest['macd'] > latest['macd_signal'],
            latest['is_trough'],
            not latest['is_peak'],
            latest['rsi'] > 30 and latest['rsi'] < 70
        ]
        
        buy_confidence = sum(buy_conditions) / len(buy_conditions)
        
        # Additional confidence factors for buy
        if conditions['trend'] == 'strong_up':
            buy_confidence += 0.1
        if conditions['momentum'] == 'strong_up':
            buy_confidence += 0.1
        if 30 <= latest['rsi'] <= 40:
            buy_confidence += 0.1  # Potential bounce from oversold
        
        # Sell signal conditions
        sell_conditions = [
            conditions['trend'] in ['down', 'strong_down'],
            latest['close'] < latest['sma_short'],
            latest['macd'] < latest['macd_signal'],
            latest['is_peak'],
            not latest['is_trough'],
            latest['rsi'] < 70 and latest['rsi'] > 30
        ]
        
        sell_confidence = sum(sell_conditions) / len(sell_conditions)
        
        # Additional confidence factors for sell
        if conditions['trend'] == 'strong_down':
            sell_confidence += 0.1
        if conditions['momentum'] == 'strong_down':
            sell_confidence += 0.1
        if 60 <= latest['rsi'] <= 70:
            sell_confidence += 0.1  # Potential drop from overbought
        
        # Determine final signal based on confidence
        if buy_confidence >= 0.6 and buy_confidence > sell_confidence:
            signal['action'] = 'buy'
            signal['confidence'] = buy_confidence
            signal['reason'] = f"Strong buy signal (trend: {conditions['trend']}, momentum: {conditions['momentum']})"
        elif sell_confidence >= 0.6 and sell_confidence > buy_confidence:
            signal['action'] = 'sell'
            signal['confidence'] = sell_confidence
            signal['reason'] = f"Strong sell signal (trend: {conditions['trend']}, momentum: {conditions['momentum']})"
        else:
            # Not enough confidence for a trade
            signal['confidence'] = max(buy_confidence, sell_confidence)
            signal['reason'] = "No clear signal"
        
        return signal
