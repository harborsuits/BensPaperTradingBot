#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend Following Strategy Template

This module implements a comprehensive trend following strategy template
that can be optimized and extended for specific trading approaches.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import talib

from trading_bot.strategies.strategy_template import (
    StrategyOptimizable,
    StrategyMetadata,
    Signal,
    SignalType,
    TimeFrame,
    MarketRegime
)

logger = logging.getLogger(__name__)

class TrendFollowingTemplate(StrategyOptimizable):
    """
    Trend Following Strategy Template that identifies and follows market trends
    in the direction of the prevailing momentum.
    
    Supported indicators include:
    - Moving Average Crossovers
    - MACD (Moving Average Convergence Divergence)
    - ADX (Average Directional Index)
    - Parabolic SAR
    - Linear Regression Slopes
    - Donchian Channels
    
    This template can be customized and extended for specific trend following approaches.
    """
    
    def __init__(
        self,
        name: str = "trend_following",
        parameters: Dict[str, Any] = None,
        metadata: Optional[StrategyMetadata] = None
    ):
        """
        Initialize trend following strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            metadata: Strategy metadata
        """
        # Default parameters for trend following
        default_params = {
            # General parameters
            "lookback_period": 50,
            "trend_strength_threshold": 25,  # ADX threshold
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 3.0,
            "trailing_stop_pct": 0.05,  # 5% trailing stop
            "risk_per_trade": 0.01,
            
            # Moving average parameters
            "use_ma_crossover": True,
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "ma_type": "ema",  # 'sma', 'ema', 'wma'
            
            # MACD parameters
            "use_macd": True,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            
            # ADX parameters
            "use_adx": True,
            "adx_period": 14,
            
            # Parabolic SAR parameters
            "use_sar": False,
            "sar_acceleration": 0.02,
            "sar_maximum": 0.2,
            
            # Donchian Channel parameters
            "use_donchian": False,
            "donchian_period": 20,
            
            # Confirmation parameters
            "require_volume_confirmation": False,
            "volume_confirmation_threshold": 1.3,
            "min_indicators_for_signal": 2,
            "require_adx_filter": True
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = StrategyMetadata(
                name=name,
                version="1.0.0",
                description="Trend following strategy that identifies and trades with the prevailing trend",
                author="Trading Bot",
                timeframes=[
                    TimeFrame.HOUR_1,
                    TimeFrame.HOUR_4,
                    TimeFrame.DAY_1,
                    TimeFrame.WEEK_1
                ],
                asset_classes=["stocks", "forex", "crypto", "futures", "etf"],
                tags=["trend_following", "momentum", "moving_average", "macd", "adx"]
            )
        
        # Initialize parent class
        super().__init__(name, default_params, metadata)
        
        logger.info(f"Initialized trend following strategy: {name}")
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate trend following indicators for the given data.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            Dictionary of symbol -> indicators dictionary
        """
        indicators = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            lookback = max(
                self.parameters["lookback_period"],
                self.parameters["slow_ma_period"] * 2
            )
            if len(df) < lookback:
                continue
            
            symbol_indicators = {}
            
            # Extract price and volume data
            close_prices = df['close'].values
            high_prices = df['high'].values if 'high' in df else close_prices
            low_prices = df['low'].values if 'low' in df else close_prices
            volumes = df['volume'].values if 'volume' in df else np.ones_like(close_prices)
            
            # Calculate moving averages
            if self.parameters["use_ma_crossover"]:
                try:
                    fast_period = self.parameters["fast_ma_period"]
                    slow_period = self.parameters["slow_ma_period"]
                    ma_type = self.parameters["ma_type"].lower()
                    
                    if ma_type == "ema":
                        fast_ma = talib.EMA(close_prices, timeperiod=fast_period)
                        slow_ma = talib.EMA(close_prices, timeperiod=slow_period)
                    elif ma_type == "wma":
                        fast_ma = talib.WMA(close_prices, timeperiod=fast_period)
                        slow_ma = talib.WMA(close_prices, timeperiod=slow_period)
                    else:  # default to SMA
                        fast_ma = talib.SMA(close_prices, timeperiod=fast_period)
                        slow_ma = talib.SMA(close_prices, timeperiod=slow_period)
                    
                    symbol_indicators['fast_ma'] = pd.Series(fast_ma, index=df.index)
                    symbol_indicators['slow_ma'] = pd.Series(slow_ma, index=df.index)
                    
                    # Calculate moving average crossover signal
                    ma_diff = fast_ma - slow_ma
                    symbol_indicators['ma_diff'] = pd.Series(ma_diff, index=df.index)
                    
                    # Calculate rate of change of the difference
                    ma_diff_roc = np.zeros_like(ma_diff)
                    ma_diff_roc[1:] = (ma_diff[1:] - ma_diff[:-1]) / np.abs(ma_diff[:-1] + 1e-10)
                    symbol_indicators['ma_diff_roc'] = pd.Series(ma_diff_roc, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating moving averages for {symbol}: {e}")
            
            # Calculate MACD
            if self.parameters["use_macd"]:
                try:
                    macd, macd_signal, macd_hist = talib.MACD(
                        close_prices,
                        fastperiod=self.parameters["macd_fast_period"],
                        slowperiod=self.parameters["macd_slow_period"],
                        signalperiod=self.parameters["macd_signal_period"]
                    )
                    
                    symbol_indicators['macd'] = pd.Series(macd, index=df.index)
                    symbol_indicators['macd_signal'] = pd.Series(macd_signal, index=df.index)
                    symbol_indicators['macd_hist'] = pd.Series(macd_hist, index=df.index)
                    
                    # Calculate MACD momentum (rate of change of histogram)
                    macd_hist_roc = np.zeros_like(macd_hist)
                    macd_hist_roc[1:] = (macd_hist[1:] - macd_hist[:-1])
                    symbol_indicators['macd_momentum'] = pd.Series(macd_hist_roc, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating MACD for {symbol}: {e}")
            
            # Calculate ADX (Average Directional Index)
            if self.parameters["use_adx"]:
                try:
                    adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.parameters["adx_period"])
                    plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.parameters["adx_period"])
                    minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=self.parameters["adx_period"])
                    
                    symbol_indicators['adx'] = pd.Series(adx, index=df.index)
                    symbol_indicators['plus_di'] = pd.Series(plus_di, index=df.index)
                    symbol_indicators['minus_di'] = pd.Series(minus_di, index=df.index)
                    
                    # Calculate ADX trend direction and strength
                    di_diff = plus_di - minus_di
                    symbol_indicators['di_diff'] = pd.Series(di_diff, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating ADX for {symbol}: {e}")
            
            # Calculate Parabolic SAR
            if self.parameters["use_sar"]:
                try:
                    sar = talib.SAR(
                        high_prices, 
                        low_prices, 
                        acceleration=self.parameters["sar_acceleration"],
                        maximum=self.parameters["sar_maximum"]
                    )
                    
                    symbol_indicators['sar'] = pd.Series(sar, index=df.index)
                    
                    # Calculate SAR position (above or below price)
                    sar_position = np.where(sar < close_prices, 1, -1)
                    symbol_indicators['sar_position'] = pd.Series(sar_position, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating Parabolic SAR for {symbol}: {e}")
            
            # Calculate Donchian Channels
            if self.parameters["use_donchian"]:
                try:
                    n = self.parameters["donchian_period"]
                    
                    # Calculate upper and lower bands
                    upper = pd.Series(high_prices).rolling(window=n).max()
                    lower = pd.Series(low_prices).rolling(window=n).min()
                    middle = (upper + lower) / 2
                    
                    symbol_indicators['donchian_upper'] = pd.Series(upper.values, index=df.index)
                    symbol_indicators['donchian_lower'] = pd.Series(lower.values, index=df.index)
                    symbol_indicators['donchian_middle'] = pd.Series(middle.values, index=df.index)
                    
                    # Calculate breakout signals
                    prev_upper = np.roll(upper.values, 1)
                    prev_lower = np.roll(lower.values, 1)
                    
                    upper_breakout = (close_prices > prev_upper).astype(int)
                    lower_breakout = (close_prices < prev_lower).astype(int)
                    
                    symbol_indicators['donchian_upper_breakout'] = pd.Series(upper_breakout, index=df.index)
                    symbol_indicators['donchian_lower_breakout'] = pd.Series(lower_breakout, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating Donchian Channels for {symbol}: {e}")
            
            # Calculate ATR for position sizing and stop loss
            try:
                atr = talib.ATR(
                    high_prices, 
                    low_prices, 
                    close_prices,
                    timeperiod=14
                )
                symbol_indicators['atr'] = pd.Series(atr, index=df.index)
                symbol_indicators['atr_percent'] = pd.Series(atr / close_prices, index=df.index)
            except Exception as e:
                logger.error(f"Error calculating ATR for {symbol}: {e}")
                # Fallback simple volatility measure
                close_returns = pd.Series(close_prices).pct_change()
                symbol_indicators['volatility'] = close_returns.rolling(window=14).std()
            
            # Store indicators for this symbol
            indicators[symbol] = symbol_indicators
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Signal]:
        """
        Generate trend following trading signals for the given data.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/indicator data
            context: Additional context information (market regime, etc.)
            
        Returns:
            Dictionary of symbol -> Signal objects
        """
        signals = {}
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Get current market regime if available
        market_regime = context.get('market_regime', MarketRegime.UNKNOWN) if context else MarketRegime.UNKNOWN
        
        for symbol, symbol_indicators in indicators.items():
            # Skip if not enough indicators available
            min_indicators = self.parameters["min_indicators_for_signal"]
            available_indicators = sum([
                int(self.parameters["use_ma_crossover"] and 'fast_ma' in symbol_indicators),
                int(self.parameters["use_macd"] and 'macd' in symbol_indicators),
                int(self.parameters["use_adx"] and 'adx' in symbol_indicators),
                int(self.parameters["use_sar"] and 'sar' in symbol_indicators),
                int(self.parameters["use_donchian"] and 'donchian_upper' in symbol_indicators)
            ])
            
            if available_indicators < min_indicators:
                continue
            
            df = data[symbol]
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            # Initialize signal flags
            bullish_signals = 0
            bearish_signals = 0
            total_potential_signals = 0
            
            # If ADX filtering is required, check first
            adx_trend_confirmed = True
            if self.parameters["require_adx_filter"] and self.parameters["use_adx"]:
                if 'adx' in symbol_indicators:
                    adx = symbol_indicators['adx'].iloc[-1]
                    adx_threshold = self.parameters["trend_strength_threshold"]
                    
                    adx_trend_confirmed = adx > adx_threshold
                    
                    if not adx_trend_confirmed:
                        # Skip this symbol if ADX filter is required but not met
                        continue
            
            # Check moving average crossover
            if 'ma_diff' in symbol_indicators and self.parameters["use_ma_crossover"]:
                total_potential_signals += 1
                ma_diff = symbol_indicators['ma_diff'].iloc[-1]
                ma_diff_prev = symbol_indicators['ma_diff'].iloc[-2] if len(symbol_indicators['ma_diff']) > 1 else 0
                
                # Bullish when fast MA > slow MA
                if ma_diff > 0:
                    bullish_signals += 1
                    # Add strength if crossover just happened
                    if ma_diff_prev <= 0:
                        bullish_signals += 0.5
                # Bearish when fast MA < slow MA
                elif ma_diff < 0:
                    bearish_signals += 1
                    # Add strength if crossover just happened
                    if ma_diff_prev >= 0:
                        bearish_signals += 0.5
                
                # Consider slope of MA difference for confirmation
                if 'ma_diff_roc' in symbol_indicators:
                    ma_diff_roc = symbol_indicators['ma_diff_roc'].iloc[-1]
                    if ma_diff > 0 and ma_diff_roc > 0:
                        bullish_signals += 0.2  # Strengthening bullish trend
                    elif ma_diff < 0 and ma_diff_roc < 0:
                        bearish_signals += 0.2  # Strengthening bearish trend
            
            # Check MACD
            if 'macd' in symbol_indicators and 'macd_signal' in symbol_indicators and self.parameters["use_macd"]:
                total_potential_signals += 1
                macd = symbol_indicators['macd'].iloc[-1]
                macd_signal = symbol_indicators['macd_signal'].iloc[-1]
                macd_hist = symbol_indicators['macd_hist'].iloc[-1]
                
                # Bullish when MACD > Signal line
                if macd > macd_signal:
                    bullish_signals += 1
                    # Add strength based on histogram value
                    if macd_hist > 0:
                        bullish_signals += 0.2
                # Bearish when MACD < Signal line
                elif macd < macd_signal:
                    bearish_signals += 1
                    # Add strength based on histogram value
                    if macd_hist < 0:
                        bearish_signals += 0.2
                
                # Check for crossovers
                if len(symbol_indicators['macd']) > 1 and len(symbol_indicators['macd_signal']) > 1:
                    macd_prev = symbol_indicators['macd'].iloc[-2]
                    macd_signal_prev = symbol_indicators['macd_signal'].iloc[-2]
                    
                    # Bullish crossover: MACD crosses above signal line
                    if macd > macd_signal and macd_prev <= macd_signal_prev:
                        bullish_signals += 0.5
                    # Bearish crossover: MACD crosses below signal line
                    elif macd < macd_signal and macd_prev >= macd_signal_prev:
                        bearish_signals += 0.5
                
                # Consider MACD momentum
                if 'macd_momentum' in symbol_indicators:
                    macd_momentum = symbol_indicators['macd_momentum'].iloc[-1]
                    if macd_momentum > 0:
                        bullish_signals += 0.2  # Increasing momentum
                    elif macd_momentum < 0:
                        bearish_signals += 0.2  # Decreasing momentum
            
            # Check ADX directional movement
            if 'plus_di' in symbol_indicators and 'minus_di' in symbol_indicators and self.parameters["use_adx"]:
                total_potential_signals += 1
                plus_di = symbol_indicators['plus_di'].iloc[-1]
                minus_di = symbol_indicators['minus_di'].iloc[-1]
                
                # Bullish when +DI > -DI
                if plus_di > minus_di:
                    bullish_signals += 1
                    # Add strength based on the difference
                    if plus_di > minus_di * 1.5:
                        bullish_signals += 0.3
                # Bearish when -DI > +DI
                elif minus_di > plus_di:
                    bearish_signals += 1
                    # Add strength based on the difference
                    if minus_di > plus_di * 1.5:
                        bearish_signals += 0.3
                
                # Check for crossovers
                if len(symbol_indicators['plus_di']) > 1 and len(symbol_indicators['minus_di']) > 1:
                    plus_di_prev = symbol_indicators['plus_di'].iloc[-2]
                    minus_di_prev = symbol_indicators['minus_di'].iloc[-2]
                    
                    # Bullish crossover: +DI crosses above -DI
                    if plus_di > minus_di and plus_di_prev <= minus_di_prev:
                        bullish_signals += 0.5
                    # Bearish crossover: -DI crosses above +DI
                    elif minus_di > plus_di and minus_di_prev <= plus_di_prev:
                        bearish_signals += 0.5
            
            # Check Parabolic SAR
            if 'sar_position' in symbol_indicators and self.parameters["use_sar"]:
                total_potential_signals += 1
                sar_position = symbol_indicators['sar_position'].iloc[-1]
                
                # Bullish when price > SAR (SAR below price)
                if sar_position > 0:
                    bullish_signals += 1
                # Bearish when price < SAR (SAR above price)
                else:
                    bearish_signals += 1
                
                # Check for crossovers
                if len(symbol_indicators['sar_position']) > 1:
                    sar_position_prev = symbol_indicators['sar_position'].iloc[-2]
                    
                    # Bullish crossover: SAR moves below price
                    if sar_position > 0 and sar_position_prev <= 0:
                        bullish_signals += 0.5
                    # Bearish crossover: SAR moves above price
                    elif sar_position < 0 and sar_position_prev >= 0:
                        bearish_signals += 0.5
            
            # Check Donchian Channels
            if 'donchian_upper_breakout' in symbol_indicators and 'donchian_lower_breakout' in symbol_indicators and self.parameters["use_donchian"]:
                total_potential_signals += 1
                upper_breakout = symbol_indicators['donchian_upper_breakout'].iloc[-1]
                lower_breakout = symbol_indicators['donchian_lower_breakout'].iloc[-1]
                
                # Bullish on upper breakout
                if upper_breakout > 0:
                    bullish_signals += 1
                # Bearish on lower breakout
                elif lower_breakout > 0:
                    bearish_signals += 1
            
            # Check volume confirmation if required
            if self.parameters["require_volume_confirmation"] and 'volume' in df:
                volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                
                volume_confirmed = volume > avg_volume * self.parameters["volume_confirmation_threshold"]
                
                if not volume_confirmed:
                    # Reduce signal strength if volume confirmation is required but not present
                    bullish_signals *= 0.7
                    bearish_signals *= 0.7
            
            # Determine signal type
            signal_type = None
            signal_strength = 0
            min_signals_required = min(
                self.parameters["min_indicators_for_signal"],
                total_potential_signals
            )
            
            if bullish_signals >= min_signals_required and bullish_signals > bearish_signals:
                signal_type = SignalType.BUY
                signal_strength = bullish_signals / total_potential_signals
            elif bearish_signals >= min_signals_required and bearish_signals > bullish_signals:
                signal_type = SignalType.SELL
                signal_strength = bearish_signals / total_potential_signals
            
            # Generate signal if threshold met
            if signal_type:
                # Adjust confidence based on market regime
                regime_adjustment = self._get_regime_adjustment(market_regime, signal_type)
                confidence = min(0.95, signal_strength * regime_adjustment)
                
                # Calculate stop loss and take profit
                stop_loss = None
                take_profit = None
                
                if 'atr' in symbol_indicators:
                    atr = symbol_indicators['atr'].iloc[-1]
                    
                    if signal_type == SignalType.BUY:
                        stop_loss = current_price - atr * self.parameters["stop_loss_atr_multiple"]
                        take_profit = current_price + atr * self.parameters["take_profit_atr_multiple"]
                    else:  # SELL
                        stop_loss = current_price + atr * self.parameters["stop_loss_atr_multiple"]
                        take_profit = current_price - atr * self.parameters["take_profit_atr_multiple"]
                
                # Create signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    timestamp=current_time,
                    price=current_price,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_pct=self.parameters["trailing_stop_pct"],
                    metadata={
                        "strategy": self.name,
                        "indicators": {
                            "bullish_count": bullish_signals,
                            "bearish_count": bearish_signals,
                            "total_indicators": total_potential_signals,
                            "trend_strength": symbol_indicators.get('adx', pd.Series([0])).iloc[-1] 
                                if 'adx' in symbol_indicators else 0
                        }
                    }
                )
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get the parameter space for optimization.
        
        Returns:
            Dictionary of parameter names and possible values
        """
        return {
            # General parameters
            "trend_strength_threshold": [20, 25, 30, 35],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [2.0, 2.5, 3.0, 4.0],
            "trailing_stop_pct": [0.03, 0.05, 0.08, 0.1],
            "risk_per_trade": [0.005, 0.01, 0.015, 0.02],
            
            # Moving average parameters
            "fast_ma_period": [10, 15, 20, 25],
            "slow_ma_period": [40, 50, 60, 70],
            "ma_type": ["sma", "ema", "wma"],
            
            # MACD parameters
            "macd_fast_period": [8, 12, 16],
            "macd_slow_period": [21, 26, 30],
            "macd_signal_period": [7, 9, 11],
            
            # ADX parameters
            "adx_period": [10, 14, 20],
            
            # Donchian Channel parameters
            "donchian_period": [15, 20, 25],
            
            # Confirmation parameters
            "min_indicators_for_signal": [1, 2, 3],
            "volume_confirmation_threshold": [1.2, 1.5, 2.0]
        }
    
    def get_regime_compatibility(self, regime: MarketRegime) -> float:
        """
        Get compatibility score for this strategy in the given market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Compatibility score (0-1, higher is better)
        """
        # Trend following strategies typically work best in trending markets
        compatibility_map = {
            MarketRegime.TRENDING: 0.9,   # Best in trending markets
            MarketRegime.BULLISH: 0.8,    # Good in bullish markets
            MarketRegime.BEARISH: 0.7,    # Good in bearish markets
            MarketRegime.VOLATILE: 0.6,   # Can work in volatile markets if trends develop
            MarketRegime.NEUTRAL: 0.4,    # Less effective in neutral markets
            MarketRegime.RANGE_BOUND: 0.3,# Poor in range-bound markets
            MarketRegime.UNKNOWN: 0.5     # Moderate in unknown regimes
        }
        
        return compatibility_map.get(regime, 0.5)
    
    def _get_regime_adjustment(self, regime: MarketRegime, signal_type: SignalType) -> float:
        """
        Get adjustment factor for signal confidence based on market regime.
        
        Args:
            regime: Current market regime
            signal_type: Signal type (BUY/SELL)
            
        Returns:
            Adjustment factor (0-1.5)
        """
        # Default adjustment - no change
        adjustment = 1.0
        
        # Adjust based on regime
        if regime == MarketRegime.TRENDING:
            # Ideal for trend following
            adjustment = 1.3
        elif regime == MarketRegime.RANGE_BOUND:
            # Reduce confidence in range-bound markets
            adjustment = 0.6
        elif regime == MarketRegime.VOLATILE:
            # Can still work in volatile markets
            adjustment = 0.8
        elif regime == MarketRegime.BULLISH and signal_type == SignalType.BUY:
            # Increase confidence for longs in bullish market
            adjustment = 1.2
        elif regime == MarketRegime.BEARISH and signal_type == SignalType.SELL:
            # Increase confidence for shorts in bearish market
            adjustment = 1.2
        
        return adjustment
    
    def detect_stock_characteristics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Detect stock-specific characteristics relevant for trend following.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            Dictionary of symbol -> characteristics dictionary
        """
        characteristics = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            if len(df) < 100:  # Need enough data for meaningful statistics
                continue
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate autocorrelation (momentum tendency)
            autocorr = returns.autocorr(lag=1)
            
            # Calculate beta to a market benchmark if available
            beta = 1.0  # Default value
            if 'market_returns' in df:
                # Calculate beta using covariance and variance
                market_returns = df['market_returns'].dropna()
                if len(market_returns) > 20:
                    aligned_returns = returns.iloc[-len(market_returns):]
                    beta = np.cov(aligned_returns, market_returns)[0, 1] / np.var(market_returns)
            
            # Calculate average true range (ATR) as percentage of price
            atr_pct = None
            if 'high' in df and 'low' in df:
                high = df['high']
                low = df['low']
                close = df['close']
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                atr_pct = (atr / close).mean()
            
            # Categorize trend following suitability
            trend_strength = "unknown"
            
            if autocorr > 0.1:
                trend_strength = "strong"  # Strong momentum/trending tendencies
            elif autocorr > 0:
                trend_strength = "moderate"  # Moderate momentum tendencies
            elif autocorr > -0.1:
                trend_strength = "weak"  # Weak momentum tendencies
            else:
                trend_strength = "negative"  # Mean reversion tendencies
            
            # Store characteristics
            characteristics[symbol] = {
                "volatility": volatility,
                "autocorrelation": autocorr,
                "beta": beta,
                "atr_percentage": atr_pct,
                "trend_strength": trend_strength,
                "suitable_for_strategy": trend_strength in ["strong", "moderate"]
            }
        
        return characteristics 