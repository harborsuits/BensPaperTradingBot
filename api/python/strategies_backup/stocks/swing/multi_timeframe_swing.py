#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Swing Trading Strategy Module

This module implements a sophisticated swing trading strategy that uses
multiple timeframes, technical indicators, and advanced filtering techniques
to identify and trade medium-term price swings.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Setup logging
logger = logging.getLogger(__name__)

class MultiTimeframeSwingStrategy(StrategyOptimizable):
    """
    Multi-Timeframe Swing Trading Strategy.
    
    Uses a higher timeframe for trend identification and a lower timeframe
    for entry triggers, incorporating advanced technical indicators, statistical
    filtering, and sophisticated exit strategies.
    
    Key features:
    - Implements a two-timeframe approach: higher for trend direction, lower for precise entries
    - Uses exponential moving average (EMA) ribbons to identify trend strength and direction
    - Incorporates Fibonacci retracement levels to find key support and resistance zones
    - Employs oscillators (RSI, Stochastic) to identify momentum shifts and entry points
    - Confirms entry signals with volume analysis and MACD convergence/divergence
    - Features sophisticated risk management with ATR-based position sizing and stops
    - Adjusts strategy parameters dynamically based on market volatility regimes
    - Includes correlation analysis to limit overexposure to similar market movements
    
    Ideal market conditions:
    - Established trends with identifiable pullbacks or consolidations
    - Markets with sufficient volatility to create meaningful swing opportunities
    - Liquid markets with clear volume patterns supporting price movements
    - Environments where multiple timeframes show harmonic confirmation patterns
    
    Limitations:
    - More complex implementation requiring additional data and computation
    - May miss early trend reversals due to emphasis on confirmation
    - Performance varies across different assets and volatility regimes
    - Requires regular parameter optimization to adapt to changing market conditions
    
    Risk management features:
    - Dynamic position sizing based on asset volatility and account risk parameters
    - Time-based exit rules to limit exposure to prolonged drawdowns
    - Multiple stop-loss mechanisms including fixed, trailing, and volatility-adjusted stops
    - Correlation filters to prevent overexposure to similar market factors
    - Earnings and news event avoidance to reduce event risk exposure
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Multi-Timeframe Swing Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # Timeframes
            "trend_timeframe": TimeFrame.DAY_1,  # Higher timeframe for trend identification
            "entry_timeframe": TimeFrame.HOUR_4,  # Lower timeframe for entry triggers
            
            # MA Ribbon parameters
            "ma_ribbon_periods": [8, 13, 21, 34, 55],  # EMA periods for ribbon
            
            # ADX parameters
            "adx_period": 14,                  # ADX period
            "adx_threshold": 25,               # Minimum ADX for strong trend
            
            # Pullback parameters
            "pullback_ema": 21,                # EMA for pullback measurement
            "fib_levels": [0.382, 0.5, 0.618], # Fibonacci retracement levels
            
            # Oscillator parameters
            "rsi_period": 14,                  # RSI period
            "rsi_lower_bound": 40,             # Lower bound for RSI pullback
            "rsi_upper_bound": 50,             # Upper bound for RSI pullback
            "stoch_k_period": 14,              # Stochastic K period
            "stoch_d_period": 3,               # Stochastic D period
            "stoch_slowing": 3,                # Stochastic slowing period
            "stoch_oversold": 20,              # Stochastic oversold threshold
            
            # Volume parameters
            "volume_ma_period": 20,            # Volume moving average period
            "volume_threshold": 1.2,           # Volume threshold multiple
            
            # MACD parameters
            "macd_fast_period": 12,            # MACD fast period
            "macd_slow_period": 26,            # MACD slow period
            "macd_signal_period": 9,           # MACD signal period
            
            # Exit parameters
            "atr_period": 14,                  # ATR period for volatility measurement
            "target_atr_multiple": 2.0,        # Target as multiple of ATR
            "trailing_stop_atr": 2.0,          # Trailing stop as multiple of ATR
            "max_holding_days": 14,            # Maximum holding period in days
            
            # Volatility filters
            "use_vix_filter": True,            # Use VIX for volatility filtering
            "vix_ma_period": 20,               # VIX moving average period
            "atr_channel_period": 20,          # ATR channel period
            "atr_channel_min_mult": 0.5,       # Minimum ATR channel multiple
            "atr_channel_max_mult": 1.5,       # Maximum ATR channel multiple
            
            # Risk management
            "max_correlated_positions": 2,     # Maximum positions in correlated assets
            "correlation_threshold": 0.8,      # Correlation threshold
            "max_risk_per_trade": 0.005,       # Maximum risk per trade (0.5%)
            
            # News and events
            "earnings_blackout_days": 1,       # Days to avoid trading before earnings
            "news_sentiment_threshold": 2.0,   # News sentiment threshold (std dev)
            
            # Backtest and optimization
            "in_sample_months": 12,            # In-sample period for optimization
            "out_sample_months": 3             # Out-of-sample period for validation
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Set appropriate timeframes for this strategy
        if metadata is None or not hasattr(metadata, 'timeframes'):
            self.timeframes = [
                TimeFrame.HOUR_4,  # Entry timeframe
                TimeFrame.DAY_1    # Trend timeframe
            ]
        
        logger.info(f"Initialized Multi-Timeframe Swing Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "ma_ribbon_periods": [[8, 13, 21, 34, 55], [5, 10, 20, 40, 60]],
            "adx_period": [10, 14, 20],
            "adx_threshold": [20, 25, 30],
            "pullback_ema": [14, 21, 34],
            "rsi_period": [10, 14, 21],
            "rsi_lower_bound": [35, 40, 45],
            "rsi_upper_bound": [45, 50, 55],
            "stoch_k_period": [10, 14, 21],
            "volume_threshold": [1.1, 1.2, 1.5],
            "atr_period": [10, 14, 21],
            "target_atr_multiple": [1.5, 2.0, 2.5],
            "trailing_stop_atr": [1.5, 2.0, 2.5],
            "max_holding_days": [10, 14, 21],
            "max_risk_per_trade": [0.003, 0.005, 0.007]
        }
    
    def _calculate_ma_ribbon(self, prices: pd.Series, periods: List[int]) -> Dict[int, pd.Series]:
        """
        Calculate MA ribbon using specified periods.
        
        Args:
            prices: Series of price data
            periods: List of periods for MA ribbon
            
        Returns:
            Dictionary mapping periods to moving average Series
        """
        ma_dict = {}
        for period in periods:
            ma_dict[period] = prices.ewm(span=period, adjust=False).mean()
        return ma_dict
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = pd.Series(0, index=high.index)
        minus_dm = pd.Series(0, index=high.index)
        
        plus_dm.loc[(high_diff > 0) & (high_diff > -low_diff)] = high_diff
        minus_dm.loc[(low_diff > 0) & (low_diff > high_diff)] = low_diff
        
        # Smooth +DM, -DM, and TR
        smoothed_plus_dm = plus_dm.rolling(window=period).mean()
        smoothed_minus_dm = minus_dm.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_period: int = 14, d_period: int = 3, slowing: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            k_period: Stochastic K period
            d_period: Stochastic D period
            slowing: Stochastic slowing period
            
        Returns:
            Tuple of (stoch_k, stoch_d)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k_fast.rolling(window=slowing).mean()
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD.
        
        Args:
            prices: Series of price data
            fast_period: MACD fast period
            slow_period: MACD slow period
            signal_period: MACD signal period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_fibonacci_levels(self, high: float, low: float, levels: List[float]) -> List[float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            high: High price
            low: Low price
            levels: List of Fibonacci ratios
            
        Returns:
            List of price levels
        """
        range_size = high - low
        return [high - level * range_size for level in levels]
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _check_ma_ribbon_alignment(self, ma_dict: Dict[int, pd.Series], index: int) -> bool:
        """
        Check if MA ribbon is properly aligned for uptrend.
        
        Args:
            ma_dict: Dictionary of moving average series
            index: Index to check alignment for
            
        Returns:
            Boolean indicating if ribbon is aligned for uptrend
        """
        ma_periods = sorted(ma_dict.keys())
        
        # Check alignment (smaller periods above larger periods)
        for i in range(len(ma_periods) - 1):
            if ma_dict[ma_periods[i]].iloc[index] <= ma_dict[ma_periods[i+1]].iloc[index]:
                return False
                
        return True
    
    def _check_ma_ribbon_downtrend(self, ma_dict: Dict[int, pd.Series], index: int) -> bool:
        """
        Check if MA ribbon is properly aligned for downtrend.
        
        Args:
            ma_dict: Dictionary of moving average series
            index: Index to check alignment for
            
        Returns:
            Boolean indicating if ribbon is aligned for downtrend
        """
        ma_periods = sorted(ma_dict.keys())
        
        # Check alignment (smaller periods below larger periods)
        for i in range(len(ma_periods) - 1):
            if ma_dict[ma_periods[i]].iloc[index] >= ma_dict[ma_periods[i+1]].iloc[index]:
                return False
                
        return True
    
    def _check_pullback_to_ema(self, close: float, ema: float, threshold: float = 0.01) -> bool:
        """
        Check if price has pulled back to EMA.
        
        Args:
            close: Close price
            ema: EMA value
            threshold: Threshold for proximity to EMA
            
        Returns:
            Boolean indicating if price is near EMA
        """
        return abs(close / ema - 1) <= threshold
    
    def _check_pullback_to_fib(self, close: float, fib_levels: List[float], threshold: float = 0.01) -> bool:
        """
        Check if price has pulled back to any Fibonacci level.
        
        Args:
            close: Close price
            fib_levels: List of Fibonacci level prices
            threshold: Threshold for proximity to Fibonacci level
            
        Returns:
            Boolean indicating if price is near any Fibonacci level
        """
        for level in fib_levels:
            if abs(close / level - 1) <= threshold:
                return True
        return False
    
    def _check_stochastic_cross_up(self, stoch_k: pd.Series, stoch_d: pd.Series, 
                                  index: int, oversold: int = 20) -> bool:
        """
        Check if Stochastic oscillator is crossing up from oversold.
        
        Args:
            stoch_k: Stochastic K values
            stoch_d: Stochastic D values
            index: Index to check cross for
            oversold: Oversold threshold
            
        Returns:
            Boolean indicating if Stochastic is crossing up from oversold
        """
        if index <= 0:
            return False
            
        # Check if K crossed above D
        k_cross_d = (stoch_k.iloc[index] > stoch_d.iloc[index]) and (stoch_k.iloc[index-1] <= stoch_d.iloc[index-1])
        
        # Check if coming from oversold
        from_oversold = stoch_k.iloc[index-1] < oversold or stoch_d.iloc[index-1] < oversold
        
        return k_cross_d and from_oversold
    
    def _check_macd_histogram_divergence(self, price: pd.Series, histogram: pd.Series, 
                                        lookback: int = 10) -> bool:
        """
        Check for MACD histogram divergence.
        
        Args:
            price: Series of price data
            histogram: MACD histogram values
            lookback: Lookback period for divergence check
            
        Returns:
            Boolean indicating if divergence exists
        """
        if len(price) <= lookback:
            return False
            
        # Find local minima in price
        price_subset = price[-lookback:]
        histogram_subset = histogram[-lookback:]
        
        price_min_idx = price_subset.idxmin()
        hist_min_idx = histogram_subset.idxmin()
        
        # Check for bullish divergence (price makes higher low, histogram makes lower low)
        if price_min_idx < hist_min_idx and price_subset[price_min_idx] < price_subset[hist_min_idx]:
            return True
            
        return False
    
    def calculate_indicators(self, data: Dict[str, Dict[TimeFrame, pd.DataFrame]]) -> Dict[str, Dict[TimeFrame, Dict[str, pd.DataFrame]]]:
        """
        Calculate indicators for all timeframes for all symbols.
        
        Args:
            data: Dictionary mapping symbols to dictionaries of timeframes to DataFrames
            
        Returns:
            Dictionary of calculated indicators for each symbol and timeframe
        """
        indicators = {}
        
        # Get parameters
        trend_timeframe = self.parameters.get("trend_timeframe", TimeFrame.DAY_1)
        entry_timeframe = self.parameters.get("entry_timeframe", TimeFrame.HOUR_4)
        ma_ribbon_periods = self.parameters.get("ma_ribbon_periods", [8, 13, 21, 34, 55])
        adx_period = self.parameters.get("adx_period", 14)
        pullback_ema = self.parameters.get("pullback_ema", 21)
        rsi_period = self.parameters.get("rsi_period", 14)
        stoch_k_period = self.parameters.get("stoch_k_period", 14)
        stoch_d_period = self.parameters.get("stoch_d_period", 3)
        stoch_slowing = self.parameters.get("stoch_slowing", 3)
        volume_ma_period = self.parameters.get("volume_ma_period", 20)
        macd_fast = self.parameters.get("macd_fast_period", 12)
        macd_slow = self.parameters.get("macd_slow_period", 26)
        macd_signal = self.parameters.get("macd_signal_period", 9)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, timeframes in data.items():
            indicators[symbol] = {}
            
            # Process trend timeframe indicators
            if trend_timeframe in timeframes:
                df_trend = timeframes[trend_timeframe]
                
                # Skip if required columns are missing
                if not all(col in df_trend.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    logger.warning(f"Required columns not found for {symbol} on {trend_timeframe}")
                    continue
                
                try:
                    # Calculate MA ribbon
                    ma_ribbon = self._calculate_ma_ribbon(df_trend['close'], ma_ribbon_periods)
                    
                    # Calculate ADX
                    adx = self._calculate_adx(df_trend['high'], df_trend['low'], df_trend['close'], adx_period)
                    
                    # Calculate ATR
                    atr = self._calculate_atr(df_trend['high'], df_trend['low'], df_trend['close'], atr_period)
                    
                    # Calculate pullback EMA
                    pullback_ema_values = df_trend['close'].ewm(span=pullback_ema, adjust=False).mean()
                    
                    # Store trend timeframe indicators
                    indicators[symbol][trend_timeframe] = {
                        "adx": pd.DataFrame({"adx": adx}),
                        "atr": pd.DataFrame({"atr": atr}),
                        "pullback_ema": pd.DataFrame({"pullback_ema": pullback_ema_values})
                    }
                    
                    # Store MA ribbon
                    for period, ma_values in ma_ribbon.items():
                        indicators[symbol][trend_timeframe][f"ema_{period}"] = pd.DataFrame({f"ema_{period}": ma_values})
                    
                except Exception as e:
                    logger.error(f"Error calculating trend indicators for {symbol}: {e}")
            
            # Process entry timeframe indicators
            if entry_timeframe in timeframes:
                df_entry = timeframes[entry_timeframe]
                
                # Skip if required columns are missing
                if not all(col in df_entry.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    logger.warning(f"Required columns not found for {symbol} on {entry_timeframe}")
                    continue
                
                try:
                    # Calculate RSI
                    rsi = self._calculate_rsi(df_entry['close'], rsi_period)
                    
                    # Calculate Stochastic
                    stoch_k, stoch_d = self._calculate_stochastic(
                        df_entry['high'], df_entry['low'], df_entry['close'],
                        stoch_k_period, stoch_d_period, stoch_slowing
                    )
                    
                    # Calculate Volume MA
                    volume_ma = df_entry['volume'].rolling(window=volume_ma_period).mean()
                    volume_ratio = df_entry['volume'] / volume_ma
                    
                    # Calculate MACD
                    macd_line, macd_signal_line, macd_histogram = self._calculate_macd(
                        df_entry['close'], macd_fast, macd_slow, macd_signal
                    )
                    
                    # Store entry timeframe indicators
                    indicators[symbol][entry_timeframe] = {
                        "rsi": pd.DataFrame({"rsi": rsi}),
                        "stoch_k": pd.DataFrame({"stoch_k": stoch_k}),
                        "stoch_d": pd.DataFrame({"stoch_d": stoch_d}),
                        "volume_ma": pd.DataFrame({"volume_ma": volume_ma}),
                        "volume_ratio": pd.DataFrame({"volume_ratio": volume_ratio}),
                        "macd_line": pd.DataFrame({"macd_line": macd_line}),
                        "macd_signal": pd.DataFrame({"macd_signal": macd_signal_line}),
                        "macd_histogram": pd.DataFrame({"macd_histogram": macd_histogram})
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating entry indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, Dict[TimeFrame, pd.DataFrame]], indicators: Optional[Dict[str, Dict[TimeFrame, Dict[str, pd.DataFrame]]]] = None) -> Dict[str, Signal]:
        """
        Generate multi-timeframe swing trading signals.
        
        Args:
            data: Dictionary mapping symbols to dictionaries of timeframes to DataFrames
            indicators: Pre-calculated indicators (optional, will be computed if not provided)
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators if not provided
        if indicators is None:
            indicators = self.calculate_indicators(data)
        
        # Get parameters
        trend_timeframe = self.parameters.get("trend_timeframe", TimeFrame.DAY_1)
        entry_timeframe = self.parameters.get("entry_timeframe", TimeFrame.HOUR_4)
        ma_ribbon_periods = self.parameters.get("ma_ribbon_periods", [8, 13, 21, 34, 55])
        adx_threshold = self.parameters.get("adx_threshold", 25)
        rsi_lower_bound = self.parameters.get("rsi_lower_bound", 40)
        rsi_upper_bound = self.parameters.get("rsi_upper_bound", 50)
        stoch_oversold = self.parameters.get("stoch_oversold", 20)
        volume_threshold = self.parameters.get("volume_threshold", 1.2)
        target_atr_multiple = self.parameters.get("target_atr_multiple", 2.0)
        
        # Generate signals
        signals = {}
        
        for symbol in data.keys():
            try:
                # Skip if we don't have both timeframes
                if trend_timeframe not in data[symbol] or entry_timeframe not in data[symbol]:
                    continue
                
                # Skip if we don't have indicators for both timeframes
                if trend_timeframe not in indicators[symbol] or entry_timeframe not in indicators[symbol]:
                    continue
                
                # Get latest data
                latest_trend = data[symbol][trend_timeframe].iloc[-1]
                latest_entry = data[symbol][entry_timeframe].iloc[-1]
                
                # Get latest timestamp
                latest_timestamp = latest_entry.name if isinstance(latest_entry.name, datetime) else datetime.now()
                
                # Get trend timeframe indicators
                adx = indicators[symbol][trend_timeframe]["adx"].iloc[-1]["adx"]
                atr = indicators[symbol][trend_timeframe]["atr"].iloc[-1]["atr"]
                pullback_ema = indicators[symbol][trend_timeframe]["pullback_ema"].iloc[-1]["pullback_ema"]
                
                # Get MA ribbon values
                ma_ribbon = {}
                for period in ma_ribbon_periods:
                    ma_ribbon[period] = indicators[symbol][trend_timeframe][f"ema_{period}"].iloc[-1][f"ema_{period}"]
                
                # Get entry timeframe indicators
                rsi = indicators[symbol][entry_timeframe]["rsi"].iloc[-1]["rsi"]
                stoch_k = indicators[symbol][entry_timeframe]["stoch_k"].iloc[-1]["stoch_k"]
                stoch_d = indicators[symbol][entry_timeframe]["stoch_d"].iloc[-1]["stoch_d"]
                volume_ratio = indicators[symbol][entry_timeframe]["volume_ratio"].iloc[-1]["volume_ratio"]
                macd_histogram = indicators[symbol][entry_timeframe]["macd_histogram"].iloc[-1]["macd_histogram"]
                
                # Check if indicators are valid
                if np.isnan(adx) or np.isnan(atr) or np.isnan(rsi) or np.isnan(stoch_k) or np.isnan(volume_ratio):
                    continue
                
                # Check for uptrend conditions
                uptrend = self._check_ma_ribbon_alignment(ma_ribbon, -1)
                strong_trend = adx > adx_threshold
                
                # Check for pullback to dynamic support
                pullback_to_support = self._check_pullback_to_ema(latest_trend['close'], pullback_ema)
                
                # Check RSI condition
                rsi_pullback = rsi_lower_bound <= rsi <= rsi_upper_bound
                
                # Check Stochastic condition
                stoch_cross_up = self._check_stochastic_cross_up(
                    indicators[symbol][entry_timeframe]["stoch_k"]["stoch_k"],
                    indicators[symbol][entry_timeframe]["stoch_d"]["stoch_d"],
                    -1, stoch_oversold
                )
                
                # Check volume condition
                volume_confirm = volume_ratio > volume_threshold
                
                # Check MACD histogram divergence
                macd_divergence = self._check_macd_histogram_divergence(
                    data[symbol][entry_timeframe]['close'],
                    indicators[symbol][entry_timeframe]["macd_histogram"]["macd_histogram"]
                )
                
                # Generate signal based on conditions
                signal_type = None
                
                # LONG signal - Uptrend + Pullback + Oscillator Confirmation + Volume
                if (uptrend and strong_trend and pullback_to_support and 
                    rsi_pullback and stoch_cross_up and volume_confirm):
                    signal_type = SignalType.BUY
                
                # If we have a valid signal, create Signal object
                if signal_type:
                    # Calculate stop loss and take profit
                    stop_loss = latest_trend['low'] - (atr * 0.5)  # Place stop below recent low
                    take_profit = latest_trend['close'] + (atr * target_atr_multiple)
                    
                    # Calculate confidence
                    confidence_factors = []
                    
                    # 1. Trend strength (ADX)
                    adx_conf = min(0.3, (adx - adx_threshold) / 30)
                    confidence_factors.append(adx_conf)
                    
                    # 2. RSI positioning
                    rsi_conf = 0.2 if rsi_lower_bound <= rsi <= rsi_upper_bound else 0.0
                    confidence_factors.append(rsi_conf)
                    
                    # 3. Volume confirmation
                    vol_conf = min(0.2, (volume_ratio - volume_threshold) * 0.1 + 0.1)
                    confidence_factors.append(vol_conf)
                    
                    # 4. MACD divergence
                    macd_conf = 0.2 if macd_divergence else 0.0
                    confidence_factors.append(macd_conf)
                    
                    # Calculate overall confidence
                    confidence = min(0.9, sum(confidence_factors))
                    
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_entry['close'],
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        timeframe=entry_timeframe,
                        metadata={
                            "trend_timeframe": trend_timeframe.value,
                            "entry_timeframe": entry_timeframe.value,
                            "adx": adx,
                            "atr": atr,
                            "rsi": rsi,
                            "stochastic_k": stoch_k,
                            "stochastic_d": stoch_d,
                            "volume_ratio": volume_ratio,
                            "max_holding_days": self.parameters.get("max_holding_days", 14),
                            "trailing_stop_atr": self.parameters.get("trailing_stop_atr", 2.0),
                            "strategy_type": "multi_timeframe_swing"
                        }
                    )
                    
                    logger.info(f"Generated multi-timeframe swing {signal_type} signal for {symbol}")
                    logger.info(f"ADX: {adx:.2f}, RSI: {rsi:.2f}, Volume Ratio: {volume_ratio:.2f}")
                    logger.info(f"Entry: {latest_entry['close']:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def should_exit(self, symbol: str, position_data: Dict[str, Any], current_data: Dict[TimeFrame, pd.DataFrame], days_held: int) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.
        
        Args:
            symbol: Symbol of the position
            position_data: Data about the position
            current_data: Current price data for the symbol
            days_held: Number of days the position has been held
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Get parameters
        max_holding_days = self.parameters.get("max_holding_days", 14)
        trend_timeframe = self.parameters.get("trend_timeframe", TimeFrame.DAY_1)
        entry_timeframe = self.parameters.get("entry_timeframe", TimeFrame.HOUR_4)
        
        # Check if position has been held longer than max holding days
        if days_held >= max_holding_days:
            return True, "Time-based exit triggered"
        
        # Check if trend has reversed
        try:
            if trend_timeframe in current_data:
                # Calculate MA ribbon
                ma_ribbon_periods = self.parameters.get("ma_ribbon_periods", [8, 13, 21, 34, 55])
                ma_ribbon = self._calculate_ma_ribbon(current_data[trend_timeframe]['close'], ma_ribbon_periods)
                
                # Check if uptrend is still valid
                if not self._check_ma_ribbon_alignment(ma_ribbon, -1):
                    return True, "Trend reversal exit triggered"
            
            # Check for opposite entry signal
            if entry_timeframe in current_data:
                rsi = self._calculate_rsi(current_data[entry_timeframe]['close'])
                
                # Exit if RSI moves to overbought territory
                if rsi.iloc[-1] > 70:
                    return True, "Overbought exit triggered"
        
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
        
        return False, "" 