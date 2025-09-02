#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Swing Trading Strategy Module

This module implements a swing trading strategy for stocks, designed to capture
medium-term price movements over periods of days to weeks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from trading_bot.strategies.base.stock_base import StockBaseStrategy
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime

logger = logging.getLogger(__name__)

class StockSwingTradingStrategy(StockBaseStrategy):
    """
    Stock Swing Trading Strategy designed to capture medium-term price swings.
    
    This strategy combines technical indicators (moving averages, RSI, MACD) 
    with support/resistance levels to identify potential swing trade opportunities
    that typically last 2-10 days.
    
    It extends the StockBaseStrategy to incorporate stock-specific functionality
    like sector analysis and market breadth.
    
    Key features:
    - Identifies and trades price swings during ongoing trends or market transitions
    - Uses multiple timeframe analysis to confirm trend direction and strength
    - Focuses on pullbacks to support in uptrends and resistance tests in downtrends
    - Incorporates volume confirmation to validate price movements
    - Implements dynamic position sizing based on ATR and price volatility
    - Adjusts entry/exit criteria based on detected market regime
    
    Ideal market conditions:
    - Trending markets with identifiable pullbacks
    - Markets with defined support/resistance levels
    - Moderate volatility environments allowing for price swings
    - Liquid markets with sufficient volume to validate signals
    
    Limitations:
    - May generate false signals in highly volatile or sideways markets
    - Performance dependent on accurate support/resistance identification
    - Less effective in markets with constant gap openings
    - Requires proper risk management to account for overnight price changes
    
    Risk management:
    - Uses ATR-based stop losses to adapt to individual stock volatility
    - Implements dynamic take-profit targets based on price objectives
    - Features trailing stops to lock in profits during strong moves
    - Limits position sizes based on account risk parameters
    """
    
    # Default parameters specific to stock swing trading
    DEFAULT_SWING_PARAMS = {
        # Technical indicator parameters
        "fast_ma_period": 20,
        "slow_ma_period": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        
        # Swing trading specific parameters
        "min_swing_days": 2,
        "max_swing_days": 10,
        "pullback_threshold": 0.03,  # 3% pullback considered for entry
        "trend_strength_threshold": 0.05,  # 5% for trend confirmation
        
        # Volume and liquidity filters
        "volume_filter": True,
        "min_volume_percentile": 60,
        
        # Risk management parameters
        "atr_period": 14,
        "stop_loss_atr_multiple": 2.0,
        "take_profit_atr_multiple": 3.0,
        "trailing_stop": True,
        "trailing_stop_activation_percent": 1.5,
        
        # Sector-specific parameters
        "sector_strength_filter": True,  # Whether to filter by sector strength
        "min_sector_strength_percentile": 60,  # Minimum sector strength percentile
        
        # Market regime parameters
        "adjust_for_market_regime": True,  # Whether to adjust signals for market regime
    }
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Stock Swing Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_SWING_PARAMS)
            metadata: Strategy metadata
        """
        # Start with default stock parameters from base class
        stock_params = self.DEFAULT_STOCK_PARAMS.copy()
        
        # Update with swing-specific parameters
        stock_params.update(self.DEFAULT_SWING_PARAMS)
        
        # Override with provided parameters
        if parameters:
            stock_params.update(parameters)
        
        # Initialize the parent class with the merged parameters
        super().__init__(name=name, parameters=stock_params, metadata=metadata)
        
        # Swing-specific member variables
        self.sector_performance = {}  # Track sector performance for relative strength
        
        logger.info(f"Initialized Stock Swing Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        # Include both stock base parameters and swing-specific parameters
        param_space = {
            # Stock universe parameters
            "min_stock_price": [5.0, 10.0, 20.0],
            "min_avg_volume": [50000, 100000, 500000],
            
            # Sector parameters
            "sector_filter": [None, "Technology", "Healthcare", "Consumer Cyclical"],
            "sector_strength_filter": [True, False],
            "min_sector_strength_percentile": [50, 60, 70],
            
            # Swing-specific parameters
            "fast_ma_period": [10, 15, 20, 25],
            "slow_ma_period": [40, 50, 60, 70],
            "rsi_period": [10, 14, 21],
            "rsi_overbought": [65, 70, 75, 80],
            "rsi_oversold": [20, 25, 30, 35],
            "macd_fast": [8, 12, 16],
            "macd_slow": [22, 26, 30],
            "macd_signal": [7, 9, 11],
            "min_swing_days": [1, 2, 3],
            "max_swing_days": [7, 10, 14],
            "pullback_threshold": [0.02, 0.03, 0.04, 0.05],
            "trend_strength_threshold": [0.03, 0.05, 0.07],
            "volume_filter": [True, False],
            "min_volume_percentile": [50, 60, 70],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0, 5.0],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [1.0, 1.5, 2.0, 2.5]
        }
        
        return param_space
    
    def _find_support_resistance(self, highs: pd.Series, lows: pd.Series, closes: pd.Series,
                                volume: pd.Series, lookback: int = 50, 
                                min_distance: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        Find significant support and resistance levels.
        
        Args:
            highs: Series of high prices
            lows: Series of low prices
            closes: Series of close prices
            volume: Series of volume data
            lookback: Period to look back
            min_distance: Minimum price distance (as %) for unique levels
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if len(closes) < lookback:
            return [], []
        
        # Use only lookback period
        recent_highs = highs[-lookback:].values
        recent_lows = lows[-lookback:].values
        recent_closes = closes[-lookback:].values
        recent_volume = volume[-lookback:].values if volume is not None else np.ones_like(recent_closes)
        
        # Initialize levels
        support_levels = []
        resistance_levels = []
        
        # Find local minima/maxima with volume confirmation
        for i in range(2, len(recent_closes)-2):
            # Check for potential resistance (local high)
            if (recent_closes[i] > recent_closes[i-1] and 
                recent_closes[i] > recent_closes[i-2] and
                recent_closes[i] > recent_closes[i+1] and
                recent_closes[i] > recent_closes[i+2] and
                recent_volume[i] > np.mean(recent_volume[i-5:i+5])):
                
                # New potential resistance level
                new_level = recent_closes[i]
                
                # Check if it's far enough from existing resistance levels
                is_unique = True
                for level in resistance_levels:
                    if abs(new_level/level - 1) < min_distance:
                        is_unique = False
                        break
                
                if is_unique:
                    resistance_levels.append(new_level)
            
            # Check for potential support (local low)
            if (recent_closes[i] < recent_closes[i-1] and 
                recent_closes[i] < recent_closes[i-2] and
                recent_closes[i] < recent_closes[i+1] and
                recent_closes[i] < recent_closes[i+2] and
                recent_volume[i] > np.mean(recent_volume[i-5:i+5])):
                
                # New potential support level
                new_level = recent_closes[i]
                
                # Check if it's far enough from existing support levels
                is_unique = True
                for level in support_levels:
                    if abs(new_level/level - 1) < min_distance:
                        is_unique = False
                        break
                
                if is_unique:
                    support_levels.append(new_level)
        
        # Sort levels
        support_levels.sort()
        resistance_levels.sort()
        
        return support_levels, resistance_levels
    
    def calculate_swing_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate swing trading specific indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        fast_ma = self.parameters.get("fast_ma_period", 20)
        slow_ma = self.parameters.get("slow_ma_period", 50)
        rsi_period = self.parameters.get("rsi_period", 14)
        macd_fast = self.parameters.get("macd_fast", 12)
        macd_slow = self.parameters.get("macd_slow", 26)
        macd_signal = self.parameters.get("macd_signal", 9)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate common stock indicators using the base class method
                # This gets standard indicators like MAs, RSI, MACD, BBands
                stock_indicators = super().calculate_stock_indicators(df)
                
                # Calculate additional swing-specific indicators
                
                # Calculate ATR for stop loss/take profit if not already done
                if "atr" not in stock_indicators:
                    high_low = df['high'] - df['low']
                    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                    atr = tr.rolling(window=atr_period).mean()
                    stock_indicators["atr"] = pd.DataFrame({"atr": atr})
                
                # Calculate support and resistance levels
                volume = df['volume'] if 'volume' in df.columns else None
                support_levels, resistance_levels = self._find_support_resistance(
                    df['high'], df['low'], df['close'], volume
                )
                
                # Calculate pullbacks from recent swing highs/lows
                recent_high = df['high'].rolling(window=20).max()
                recent_low = df['low'].rolling(window=20).min()
                
                # Pullback from high (for bullish setups)
                pullback_from_high = (recent_high - df['close']) / recent_high
                
                # Pullback from low (for bearish setups)
                pullback_from_low = (df['close'] - recent_low) / df['close']
                
                # Calculate volume indicators if volume data is available
                if 'volume' in df.columns and "volume_percentile" not in stock_indicators:
                    volume_percentile = df['volume'].rolling(window=20).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                    )
                    stock_indicators["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
                # Add swing-specific indicators to the result
                stock_indicators["pullback_from_high"] = pd.DataFrame({"pullback_from_high": pullback_from_high})
                stock_indicators["pullback_from_low"] = pd.DataFrame({"pullback_from_low": pullback_from_low})
                stock_indicators["support_levels"] = support_levels
                stock_indicators["resistance_levels"] = resistance_levels
                
                # Store all indicators for this symbol
                indicators[symbol] = stock_indicators
                
            except Exception as e:
                logger.error(f"Error calculating swing indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate swing trading signals based on indicator combinations.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Apply stock-specific filters from the base class
        filtered_data = self.filter_universe(data)
        
        # Get parameters
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        pullback_threshold = self.parameters.get("pullback_threshold", 0.03)
        trend_strength = self.parameters.get("trend_strength_threshold", 0.05)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 60)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 3.0)
        trailing_stop = self.parameters.get("trailing_stop", True)
        trailing_activation = self.parameters.get("trailing_stop_activation_percent", 1.5)
        
        # Check sector strength if enabled
        use_sector_filter = self.parameters.get("sector_strength_filter", True)
        if use_sector_filter:
            # Calculate sector strength if we have sector data
            self.sector_performance = self.check_sector_rotation(filtered_data)
        
        # Calculate swing indicators
        indicators = self.calculate_swing_indicators(filtered_data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Skip symbols with insufficient data
                if symbol not in filtered_data or filtered_data[symbol].empty:
                    continue
                
                # Get latest data
                latest_data = filtered_data[symbol].iloc[-1]
                prev_data = filtered_data[symbol].iloc[-2] if len(filtered_data[symbol]) > 1 else None
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get sector if available
                sector = latest_data.get('sector', None)
                
                # Skip if sector filter is enabled and sector is weak
                if use_sector_filter and sector and sector in self.sector_performance:
                    min_sector_pctl = self.parameters.get("min_sector_strength_percentile", 60)
                    sector_strength_pctl = self.sector_performance.get(sector, 0) * 100
                    if sector_strength_pctl < min_sector_pctl:
                        continue
                
                # Get latest indicator values
                latest_fast_ma = symbol_indicators["ma_20"].iloc[-1]["ma_20"]  # Using the key from stock base indicators
                latest_slow_ma = symbol_indicators["ma_50"].iloc[-1]["ma_50"]  # Using the key from stock base indicators
                latest_rsi = symbol_indicators["rsi"].iloc[-1]["rsi"]
                
                # Get MACD values using the keys that match what's actually in the indicators
                macd_keys = [k for k in symbol_indicators.keys() if 'macd' in k.lower()]
                if 'macd' in symbol_indicators:
                    latest_macd = symbol_indicators["macd"].iloc[-1]["macd_line"]
                    latest_macd_signal = symbol_indicators["macd"].iloc[-1]["signal_line"] 
                    latest_macd_hist = symbol_indicators["macd"].iloc[-1]["macd_hist"]
                    prev_macd_hist = symbol_indicators["macd"].iloc[-2]["macd_hist"] if len(symbol_indicators["macd"]) > 1 else 0
                else:
                    # Fall back to individual MACD components
                    latest_macd = symbol_indicators.get("macd_line", pd.DataFrame()).iloc[-1]["macd_line"] if "macd_line" in symbol_indicators else 0
                    latest_macd_signal = symbol_indicators.get("macd_signal", pd.DataFrame()).iloc[-1]["macd_signal"] if "macd_signal" in symbol_indicators else 0
                    latest_macd_hist = symbol_indicators.get("macd_histogram", pd.DataFrame()).iloc[-1]["macd_histogram"] if "macd_histogram" in symbol_indicators else 0
                    prev_macd_hist = symbol_indicators.get("macd_histogram", pd.DataFrame()).iloc[-2]["macd_histogram"] if "macd_histogram" in symbol_indicators and len(symbol_indicators["macd_histogram"]) > 1 else 0
                
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                latest_pullback_high = symbol_indicators["pullback_from_high"].iloc[-1]["pullback_from_high"]
                latest_pullback_low = symbol_indicators["pullback_from_low"].iloc[-1]["pullback_from_low"]
                
                # Check volume filter if enabled
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                # Get support/resistance levels
                support_levels = symbol_indicators.get("support_levels", [])
                resistance_levels = symbol_indicators.get("resistance_levels", [])
                
                # Find nearest support/resistance
                nearest_support = max([level for level in support_levels if level < latest_price], default=None)
                nearest_resistance = min([level for level in resistance_levels if level > latest_price], default=None)
                
                # Generate signal based on swing trading conditions
                signal_type = None
                confidence = 0.0
                stop_loss = None
                take_profit = None
                
                # BULLISH SWING SETUP
                if (latest_fast_ma > latest_slow_ma and  # Uptrend confirmed by MA crossover
                    latest_price > latest_slow_ma * (1 + trend_strength) and  # Price well above slow MA
                    latest_pullback_high >= pullback_threshold and  # Recent pullback from high
                    latest_rsi > 40 and latest_rsi < 60 and  # RSI in neutral zone after reset
                    latest_macd_hist > 0 and latest_macd_hist > prev_macd_hist and  # MACD histogram rising
                    volume_ok):
                    
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on multiple factors
                    # 1. Trend strength
                    trend_confidence = min(0.3, (latest_price / latest_slow_ma - 1) * 5)
                    
                    # 2. Pullback quality
                    pullback_confidence = min(0.2, latest_pullback_high * 5)
                    
                    # 3. MACD momentum
                    macd_momentum = (latest_macd_hist - prev_macd_hist) / abs(prev_macd_hist) if prev_macd_hist != 0 else 0
                    macd_confidence = min(0.2, max(0.1, macd_momentum))
                    
                    # 4. Support strength
                    support_confidence = 0.0
                    if nearest_support is not None:
                        support_proximity = (latest_price - nearest_support) / latest_price
                        if support_proximity < 0.05:  # Price close to support
                            support_confidence = min(0.2, 0.2 - support_proximity * 2)
                    
                    # 5. Volume
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.1, vol_pct / 1000)
                    
                    # 6. Sector strength (if available)
                    sector_confidence = 0.0
                    if sector and sector in self.sector_performance:
                        sector_confidence = min(0.1, self.sector_performance[sector] * 10)
                    
                    confidence = min(0.9, trend_confidence + pullback_confidence + macd_confidence + 
                                   support_confidence + volume_confidence + sector_confidence)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    if nearest_support is not None and nearest_support > stop_loss:
                        # Use support level if it's tighter than ATR-based stop
                        stop_loss = nearest_support * 0.99
                    
                    take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                    if nearest_resistance is not None:
                        # Target nearest resistance for take profit
                        take_profit = min(take_profit, nearest_resistance)
                
                # BEARISH SWING SETUP
                elif (latest_fast_ma < latest_slow_ma and  # Downtrend confirmed by MA crossover
                      latest_price < latest_slow_ma * (1 - trend_strength) and  # Price well below slow MA
                      latest_pullback_low >= pullback_threshold and  # Recent pullback from low
                      latest_rsi < 60 and latest_rsi > 40 and  # RSI in neutral zone after reset
                      latest_macd_hist < 0 and latest_macd_hist < prev_macd_hist and  # MACD histogram falling
                      volume_ok):
                    
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on multiple factors
                    # 1. Trend strength
                    trend_confidence = min(0.3, (1 - latest_price / latest_slow_ma) * 5)
                    
                    # 2. Pullback quality
                    pullback_confidence = min(0.2, latest_pullback_low * 5)
                    
                    # 3. MACD momentum
                    macd_momentum = (prev_macd_hist - latest_macd_hist) / abs(prev_macd_hist) if prev_macd_hist != 0 else 0
                    macd_confidence = min(0.2, max(0.1, macd_momentum))
                    
                    # 4. Resistance strength
                    resistance_confidence = 0.0
                    if nearest_resistance is not None:
                        resistance_proximity = (nearest_resistance - latest_price) / latest_price
                        if resistance_proximity < 0.05:  # Price close to resistance
                            resistance_confidence = min(0.2, 0.2 - resistance_proximity * 2)
                    
                    # 5. Volume
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.1, vol_pct / 1000)
                    
                    # 6. Sector strength (for bearish, we want weak sectors)
                    sector_confidence = 0.0
                    if sector and sector in self.sector_performance:
                        # For bearish signals, negative sector performance is good
                        sector_confidence = min(0.1, (1 - self.sector_performance[sector]) * 10)
                    
                    confidence = min(0.9, trend_confidence + pullback_confidence + macd_confidence + 
                                   resistance_confidence + volume_confidence + sector_confidence)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                    if nearest_resistance is not None and nearest_resistance < stop_loss:
                        # Use resistance level if it's tighter than ATR-based stop
                        stop_loss = nearest_resistance * 1.01
                    
                    take_profit = latest_price - (latest_atr * take_profit_atr_multiple)
                    if nearest_support is not None:
                        # Target nearest support for take profit
                        take_profit = max(take_profit, nearest_support)
                
                # Create signal if we have a valid signal type
                if signal_type:
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_price,
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            "fast_ma": latest_fast_ma,
                            "slow_ma": latest_slow_ma,
                            "rsi": latest_rsi,
                            "macd_histogram": latest_macd_hist,
                            "atr": latest_atr,
                            "sector": sector,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "stock_swing_trading"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        # Adjust signals for market regime if enabled
        if self.parameters.get("adjust_for_market_regime", True) and signals:
            # Determine current market regime (simplified implementation)
            market_regime = self._determine_market_regime(filtered_data)
            
            # Adjust signals based on market regime
            signals = self.adjust_for_market_regime(signals, market_regime)
        
        return signals
    
    def _determine_market_regime(self, data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """
        Determine the current market regime based on index data.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Current market regime
        """
        # Look for SPY or other market index in the data
        index_symbol = None
        for symbol in ['SPY', 'QQQ', 'IWM', '^GSPC', '^DJI', '^IXIC']:
            if symbol in data:
                index_symbol = symbol
                break
        
        # Default to unknown regime if no index found
        if not index_symbol:
            return MarketRegime.UNKNOWN
        
        # Get index data
        index_data = data[index_symbol]
        
        # Calculate indicators for regime detection
        ma_20 = index_data['close'].rolling(window=20).mean()
        ma_50 = index_data['close'].rolling(window=50).mean()
        ma_200 = index_data['close'].rolling(window=200).mean()
        
        # Calculate recent volatility (20-day standard deviation)
        volatility = index_data['close'].pct_change().rolling(window=20).std()
        
        # Get latest values
        latest_close = index_data['close'].iloc[-1]
        latest_ma_20 = ma_20.iloc[-1]
        latest_ma_50 = ma_50.iloc[-1]
        latest_ma_200 = ma_200.iloc[-1]
        latest_volatility = volatility.iloc[-1]
        
        # Historical volatility for comparison
        avg_volatility = volatility.mean()
        high_volatility_threshold = avg_volatility * 1.5
        low_volatility_threshold = avg_volatility * 0.5
        
        # Determine regime based on moving averages and volatility
        if latest_close > latest_ma_50 and latest_ma_50 > latest_ma_200:
            # Bullish trend
            if latest_volatility > high_volatility_threshold:
                return MarketRegime.HIGH_VOLATILITY
            elif latest_volatility < low_volatility_threshold:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.BULL_TREND
        elif latest_close < latest_ma_50 and latest_ma_50 < latest_ma_200:
            # Bearish trend
            return MarketRegime.BEAR_TREND
        else:
            # Mixed signals - probably consolidation
            return MarketRegime.CONSOLIDATION 