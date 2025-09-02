#!/usr/bin/env python3
"""
Multi-Factor Trading Strategy

This module implements a multi-factor trading strategy that combines:
1. Technical indicators (moving averages, RSI, MACD, Bollinger Bands)
2. Volatility regime detection and adaptation
3. Market trend analysis
4. Adaptive position sizing based on conviction and risk
5. Dynamic stop-loss and take-profit levels

It's designed to work with the trading simulator and parameter optimizer.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import talib
from dataclasses import dataclass

from trading_bot.risk_manager import RiskManager
from trading_bot.data_providers.base_provider import DataProvider

# Configure logging
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()

class MarketRegime(Enum):
    """Market regime classifications"""
    LOW_VOLATILITY = auto()
    NORMAL = auto()
    HIGH_VOLATILITY = auto()
    EXTREME_VOLATILITY = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGE_BOUND = auto()

@dataclass
class TradingSignal:
    """Container for trading signals with conviction level"""
    type: SignalType
    conviction: float  # 0.0 to 1.0
    indicator: str
    timestamp: pd.Timestamp
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

class MultiFactor:
    """
    Multi-factor trading strategy combining multiple technical indicators,
    market regime detection, and adaptive position sizing.
    """
    
    def __init__(
        self,
        symbol: str,
        data_provider: DataProvider,
        risk_manager: Optional[RiskManager] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multi-factor strategy.
        
        Args:
            symbol: The trading symbol
            data_provider: Data provider for market data
            risk_manager: Optional risk manager for position sizing
            params: Optional strategy parameters
        """
        self.symbol = symbol
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        
        # Default parameters (can be overridden by params)
        self.default_params = {
            # Moving average parameters
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "ema_period": 13,
            
            # RSI parameters
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            
            # MACD parameters
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            
            # Bollinger Bands parameters
            "bb_period": 20,
            "bb_std_dev": 2.0,
            
            # Volatility parameters
            "atr_period": 14,
            "volatility_lookback": 20,
            
            # Position sizing parameters
            "base_position_size": 0.02,  # % of portfolio
            "max_position_size": 0.05,   # % of portfolio
            
            # Signal weighting
            "ma_signal_weight": 0.25,
            "rsi_signal_weight": 0.20,
            "macd_signal_weight": 0.25,
            "bb_signal_weight": 0.20,
            "volatility_signal_weight": 0.10,
            
            # Stop-loss and take-profit multipliers
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 3.0,
            
            # Trailing stop parameters
            "use_trailing_stop": True,
            "trailing_stop_activation": 0.02,  # % profit to activate
            "trailing_stop_distance": 0.015,   # % below price
            
            # Regime-specific adjustments
            "high_vol_size_adj": 0.5,      # Reduce position size in high vol
            "low_vol_size_adj": 1.2,       # Increase position size in low vol
            "regime_detection_period": 30,  # Days for regime detection
            
            # Trend detection
            "adx_period": 14,
            "adx_threshold": 25,           # ADX threshold for trend detection
            
            # Entry/exit thresholds
            "minimum_conviction": 0.6,     # Minimum conviction for entry
            "exit_conviction": 0.3,        # Exit when conviction falls below
        }
        
        # Update with provided parameters if any
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
            
        # Initialize strategy state
        self.current_position = 0
        self.current_market_regime = MarketRegime.NORMAL
        self.last_signal = None
        self.indicators_data = {}
        self.historical_signals = []
        self.signals_cache = {}
        
        logger.info(f"Initialized MultiFactor strategy for {symbol}")
    
    def generate_signals(self, timestamp) -> List[Dict]:
        """
        Generate trading signals for the given timestamp.
        
        Args:
            timestamp: The timestamp to generate signals for
            
        Returns:
            List of signal dictionaries with action, price, etc.
        """
        # Get historical data for analysis
        historical_data = self._get_historical_data(timestamp)
        
        if historical_data.empty:
            logger.warning(f"No historical data available for {self.symbol} at {timestamp}")
            return []
            
        # Calculate technical indicators
        self._calculate_indicators(historical_data)
        
        # Detect market regime
        self._detect_market_regime(historical_data)
        
        # Generate individual signals
        ma_signal = self._generate_ma_signal(historical_data, timestamp)
        rsi_signal = self._generate_rsi_signal(historical_data, timestamp)
        macd_signal = self._generate_macd_signal(historical_data, timestamp)
        bb_signal = self._generate_bollinger_signal(historical_data, timestamp)
        volatility_signal = self._generate_volatility_signal(historical_data, timestamp)
        
        # Combine signals with weights
        combined_signal = self._combine_signals(
            [ma_signal, rsi_signal, macd_signal, bb_signal, volatility_signal],
            [
                self.params["ma_signal_weight"],
                self.params["rsi_signal_weight"],
                self.params["macd_signal_weight"],
                self.params["bb_signal_weight"],
                self.params["volatility_signal_weight"]
            ],
            timestamp
        )
        
        # Convert to trading decision
        signals = []
        
        # Get current price
        current_price = historical_data.iloc[-1]["close"]
        
        # Check if we have a significant signal
        if combined_signal:
            self.last_signal = combined_signal
            
            # Calculate ATR for stop-loss and take-profit
            atr = self.indicators_data.get("atr", historical_data["close"].std())
            
            # Calculate stop-loss and take-profit
            stop_loss = None
            take_profit = None
            
            if combined_signal.type == SignalType.BULLISH:
                stop_loss = current_price - (atr * self.params["stop_loss_atr_mult"])
                take_profit = current_price + (atr * self.params["take_profit_atr_mult"])
            elif combined_signal.type == SignalType.BEARISH:
                stop_loss = current_price + (atr * self.params["stop_loss_atr_mult"])
                take_profit = current_price - (atr * self.params["take_profit_atr_mult"])
                
            # Check conviction threshold
            position_size = self._calculate_position_size(combined_signal, current_price)
            
            # Generate trading action if conviction is high enough
            if combined_signal.conviction >= self.params["minimum_conviction"]:
                # Entry signal
                if combined_signal.type == SignalType.BULLISH and self.current_position <= 0:
                    signals.append({
                        "action": "BUY",
                        "price": current_price,
                        "quantity": position_size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "reason": f"Bullish signal with {combined_signal.conviction:.2f} conviction",
                        "regime": self.current_market_regime.name
                    })
                    self.current_position = position_size
                # Exit signal
                elif combined_signal.type == SignalType.BEARISH and self.current_position > 0:
                    signals.append({
                        "action": "SELL",
                        "price": current_price,
                        "quantity": self.current_position,
                        "reason": f"Bearish signal with {combined_signal.conviction:.2f} conviction",
                        "regime": self.current_market_regime.name
                    })
                    self.current_position = 0
                    
            # Exit on low conviction
            elif combined_signal.conviction < self.params["exit_conviction"] and self.current_position > 0:
                signals.append({
                    "action": "SELL",
                    "price": current_price,
                    "quantity": self.current_position,
                    "reason": f"Low conviction ({combined_signal.conviction:.2f})",
                    "regime": self.current_market_regime.name
                })
                self.current_position = 0
                
        # Check for stop-loss or take-profit if in a position
        elif self.current_position > 0 and self.last_signal:
            # Calculate if stop-loss or take-profit were hit
            if self.last_signal.stop_loss and current_price <= self.last_signal.stop_loss:
                signals.append({
                    "action": "SELL",
                    "price": current_price,
                    "quantity": self.current_position,
                    "reason": "Stop-loss hit",
                    "regime": self.current_market_regime.name
                })
                self.current_position = 0
                
            elif self.last_signal.take_profit and current_price >= self.last_signal.take_profit:
                signals.append({
                    "action": "SELL",
                    "price": current_price,
                    "quantity": self.current_position,
                    "reason": "Take-profit hit",
                    "regime": self.current_market_regime.name
                })
                self.current_position = 0
                
        # Store signal for historical analysis
        if combined_signal:
            self.historical_signals.append(combined_signal)
            
        return signals
    
    def _get_historical_data(self, timestamp) -> pd.DataFrame:
        """Get historical data for analysis up to the given timestamp"""
        try:
            # Determine how much history we need for the longest indicator
            max_period = max(
                self.params["slow_ma_period"],
                self.params["rsi_period"],
                self.params["macd_slow_period"] + self.params["macd_signal_period"],
                self.params["bb_period"],
                self.params["atr_period"],
                self.params["regime_detection_period"]
            )
            
            # Add buffer for safety
            lookback_days = max(30, max_period * 2)
            
            # Convert timestamp to datetime if it's not already
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)
                
            # Calculate start date
            if timestamp.tzinfo:
                start_date = timestamp - pd.Timedelta(days=lookback_days)
                end_date = timestamp
            else:
                start_date = timestamp.replace(tzinfo=None) - pd.Timedelta(days=lookback_days)
                end_date = timestamp.replace(tzinfo=None)
            
            # Get data from provider
            data = self.data_provider.get_historical_data(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"  # Using daily data for strategy
            )
            
            # Make sure data is properly sorted
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, data: pd.DataFrame):
        """Calculate all technical indicators used by the strategy"""
        if data.empty:
            return
            
        try:
            # Moving averages
            data['fast_ma'] = data['close'].rolling(window=self.params["fast_ma_period"]).mean()
            data['slow_ma'] = data['close'].rolling(window=self.params["slow_ma_period"]).mean()
            data['ema'] = data['close'].ewm(span=self.params["ema_period"]).mean()
            
            # RSI
            data['rsi'] = talib.RSI(data['close'].values, timeperiod=self.params["rsi_period"])
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'].values,
                fastperiod=self.params["macd_fast_period"],
                slowperiod=self.params["macd_slow_period"],
                signalperiod=self.params["macd_signal_period"]
            )
            data['macd'] = macd
            data['macd_signal'] = macd_signal
            data['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=self.params["bb_period"],
                nbdevup=self.params["bb_std_dev"],
                nbdevdn=self.params["bb_std_dev"]
            )
            data['bb_upper'] = upper
            data['bb_middle'] = middle
            data['bb_lower'] = lower
            
            # ATR for volatility
            data['atr'] = talib.ATR(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod=self.params["atr_period"]
            )
            
            # ADX for trend strength
            data['adx'] = talib.ADX(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod=self.params["adx_period"]
            )
            
            # Calculate volatility
            data['daily_returns'] = data['close'].pct_change()
            data['volatility'] = data['daily_returns'].rolling(window=self.params["volatility_lookback"]).std()
            
            # Store important indicators in a dictionary for easy access
            self.indicators_data = {
                'fast_ma': data['fast_ma'].iloc[-1],
                'slow_ma': data['slow_ma'].iloc[-1],
                'ema': data['ema'].iloc[-1],
                'rsi': data['rsi'].iloc[-1],
                'macd': data['macd'].iloc[-1],
                'macd_signal': data['macd_signal'].iloc[-1],
                'macd_hist': data['macd_hist'].iloc[-1],
                'bb_upper': data['bb_upper'].iloc[-1],
                'bb_middle': data['bb_middle'].iloc[-1],
                'bb_lower': data['bb_lower'].iloc[-1],
                'atr': data['atr'].iloc[-1],
                'adx': data['adx'].iloc[-1],
                'volatility': data['volatility'].iloc[-1],
            }
            
            # Store the processed dataframe for further analysis
            self.processed_data = data
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {self.symbol}: {str(e)}")
    
    def _detect_market_regime(self, data: pd.DataFrame):
        """Detect the current market regime based on volatility and trend"""
        try:
            if data.empty or 'volatility' not in data.columns or 'adx' not in data.columns:
                self.current_market_regime = MarketRegime.NORMAL
                return
                
            # Get recent data
            recent_data = data.iloc[-self.params["regime_detection_period"]:]
            
            # Calculate average volatility over the detection period
            avg_volatility = recent_data['volatility'].mean()
            current_volatility = recent_data['volatility'].iloc[-1]
            
            # Calculate volatility ratio (current vs average)
            volatility_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
            
            # Get ADX for trend identification
            adx = recent_data['adx'].iloc[-1]
            
            # Classify market regime
            if volatility_ratio > 2.0:
                self.current_market_regime = MarketRegime.EXTREME_VOLATILITY
            elif volatility_ratio > 1.5:
                self.current_market_regime = MarketRegime.HIGH_VOLATILITY
            elif volatility_ratio < 0.7:
                self.current_market_regime = MarketRegime.LOW_VOLATILITY
            else:
                # Check for trending or range-bound
                if adx > self.params["adx_threshold"]:
                    # Check trend direction
                    if recent_data['close'].iloc[-1] > recent_data['close'].iloc[-5]:
                        self.current_market_regime = MarketRegime.TRENDING_UP
                    else:
                        self.current_market_regime = MarketRegime.TRENDING_DOWN
                else:
                    self.current_market_regime = MarketRegime.RANGE_BOUND
                    
            logger.debug(f"Detected market regime for {self.symbol}: {self.current_market_regime.name}")
            
        except Exception as e:
            logger.error(f"Error detecting market regime for {self.symbol}: {str(e)}")
            self.current_market_regime = MarketRegime.NORMAL
    
    def _generate_ma_signal(self, data: pd.DataFrame, timestamp) -> Optional[TradingSignal]:
        """Generate signal based on moving average crossovers"""
        try:
            if data.empty or 'fast_ma' not in data.columns or 'slow_ma' not in data.columns:
                return None
                
            # Get recent data points
            current_close = data['close'].iloc[-1]
            current_fast_ma = data['fast_ma'].iloc[-1]
            current_slow_ma = data['slow_ma'].iloc[-1]
            
            # Get previous data points
            prev_fast_ma = data['fast_ma'].iloc[-2]
            prev_slow_ma = data['slow_ma'].iloc[-2]
            
            # Check for crossover
            bullish_crossover = prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma
            bearish_crossover = prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma
            
            # Calculate signal strength based on distance between MAs
            ma_diff_pct = abs(current_fast_ma - current_slow_ma) / current_slow_ma
            strength = min(1.0, ma_diff_pct * 20)  # Scale the difference
            
            if bullish_crossover:
                return TradingSignal(
                    type=SignalType.BULLISH,
                    conviction=strength,
                    indicator="MA Crossover",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
            elif bearish_crossover:
                return TradingSignal(
                    type=SignalType.BEARISH,
                    conviction=strength,
                    indicator="MA Crossover",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            # Check for price crossing MA
            price_above_ema = current_close > data['ema'].iloc[-1]
            prev_price_above_ema = data['close'].iloc[-2] > data['ema'].iloc[-2]
            
            ema_crossover_up = not prev_price_above_ema and price_above_ema
            ema_crossover_down = prev_price_above_ema and not price_above_ema
            
            if ema_crossover_up:
                return TradingSignal(
                    type=SignalType.BULLISH,
                    conviction=0.7,  # EMA crossovers have medium-high conviction
                    indicator="EMA Crossover",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
            elif ema_crossover_down:
                return TradingSignal(
                    type=SignalType.BEARISH,
                    conviction=0.7,
                    indicator="EMA Crossover",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            # If in a trending market, check if price is above/below MAs
            if self.current_market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                if current_close > current_fast_ma > current_slow_ma:
                    return TradingSignal(
                        type=SignalType.BULLISH,
                        conviction=0.6,
                        indicator="Trend Following",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
                elif current_close < current_fast_ma < current_slow_ma:
                    return TradingSignal(
                        type=SignalType.BEARISH,
                        conviction=0.6,
                        indicator="Trend Following",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating MA signal for {self.symbol}: {str(e)}")
            return None
    
    def _generate_rsi_signal(self, data: pd.DataFrame, timestamp) -> Optional[TradingSignal]:
        """Generate signal based on RSI indicator"""
        try:
            if data.empty or 'rsi' not in data.columns:
                return None
                
            current_close = data['close'].iloc[-1]
            current_rsi = data['rsi'].iloc[-1]
            prev_rsi = data['rsi'].iloc[-2]
            
            # Check for overbought/oversold conditions
            overbought = current_rsi > self.params["rsi_overbought"]
            oversold = current_rsi < self.params["rsi_oversold"]
            
            # Check for RSI divergence
            price_higher = data['close'].iloc[-1] > data['close'].iloc[-5]
            rsi_higher = current_rsi > data['rsi'].iloc[-5]
            
            # Calculate conviction based on distance from threshold
            if oversold:
                # More oversold = higher conviction
                conviction = min(1.0, (self.params["rsi_oversold"] - current_rsi) / 10 + 0.6)
                
                # RSI turning up from oversold is a bullish signal
                rsi_turning_up = current_rsi > prev_rsi
                
                if rsi_turning_up:
                    return TradingSignal(
                        type=SignalType.BULLISH,
                        conviction=conviction,
                        indicator="RSI Oversold",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
            
            if overbought:
                # More overbought = higher conviction
                conviction = min(1.0, (current_rsi - self.params["rsi_overbought"]) / 10 + 0.6)
                
                # RSI turning down from overbought is a bearish signal
                rsi_turning_down = current_rsi < prev_rsi
                
                if rsi_turning_down:
                    return TradingSignal(
                        type=SignalType.BEARISH,
                        conviction=conviction,
                        indicator="RSI Overbought",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
            
            # Check for bullish divergence: price making lower lows but RSI making higher lows
            if not price_higher and rsi_higher:
                return TradingSignal(
                    type=SignalType.BULLISH,
                    conviction=0.7,
                    indicator="RSI Bullish Divergence",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            # Check for bearish divergence: price making higher highs but RSI making lower highs
            if price_higher and not rsi_higher:
                return TradingSignal(
                    type=SignalType.BEARISH,
                    conviction=0.7,
                    indicator="RSI Bearish Divergence",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating RSI signal for {self.symbol}: {str(e)}")
            return None
    
    def _generate_macd_signal(self, data: pd.DataFrame, timestamp) -> Optional[TradingSignal]:
        """Generate signal based on MACD indicator"""
        try:
            if data.empty or 'macd' not in data.columns or 'macd_signal' not in data.columns:
                return None
                
            current_close = data['close'].iloc[-1]
            current_macd = data['macd'].iloc[-1]
            current_signal = data['macd_signal'].iloc[-1]
            current_hist = data['macd_hist'].iloc[-1]
            
            prev_macd = data['macd'].iloc[-2]
            prev_signal = data['macd_signal'].iloc[-2]
            prev_hist = data['macd_hist'].iloc[-2]
            
            # Check for signal line crossover
            bullish_crossover = prev_macd <= prev_signal and current_macd > current_signal
            bearish_crossover = prev_macd >= prev_signal and current_macd < current_signal
            
            # Calculate conviction based on histogram strength
            hist_strength = abs(current_hist) / (abs(current_macd) + 0.0001)  # Avoid division by zero
            conviction = min(0.9, 0.5 + hist_strength)
            
            if bullish_crossover:
                return TradingSignal(
                    type=SignalType.BULLISH,
                    conviction=conviction,
                    indicator="MACD Crossover",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            if bearish_crossover:
                return TradingSignal(
                    type=SignalType.BEARISH,
                    conviction=conviction,
                    indicator="MACD Crossover",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            # Check for histogram reversal
            bullish_reversal = prev_hist < 0 and current_hist > prev_hist
            bearish_reversal = prev_hist > 0 and current_hist < prev_hist
            
            # Only consider strong reversals
            if bullish_reversal and current_macd < 0 and abs(current_hist - prev_hist) / abs(prev_hist) > 0.2:
                return TradingSignal(
                    type=SignalType.BULLISH,
                    conviction=0.6,
                    indicator="MACD Histogram Reversal",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            if bearish_reversal and current_macd > 0 and abs(current_hist - prev_hist) / abs(prev_hist) > 0.2:
                return TradingSignal(
                    type=SignalType.BEARISH,
                    conviction=0.6,
                    indicator="MACD Histogram Reversal",
                    timestamp=pd.Timestamp(timestamp),
                    price=current_close
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating MACD signal for {self.symbol}: {str(e)}")
            return None
    
    def _generate_bollinger_signal(self, data: pd.DataFrame, timestamp) -> Optional[TradingSignal]:
        """Generate signal based on Bollinger Bands"""
        try:
            if data.empty or 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
                return None
                
            current_close = data['close'].iloc[-1]
            upper_band = data['bb_upper'].iloc[-1]
            middle_band = data['bb_middle'].iloc[-1]
            lower_band = data['bb_lower'].iloc[-1]
            
            # Calculate percentage distance from bands
            upper_dist_pct = (upper_band - current_close) / current_close
            lower_dist_pct = (current_close - lower_band) / current_close
            
            # Check if price is testing bands
            testing_upper = current_close > upper_band * 0.985
            testing_lower = current_close < lower_band * 1.015
            
            # Calculate band width for volatility
            band_width = (upper_band - lower_band) / middle_band
            
            # Check for different signals based on market regime
            if self.current_market_regime in [MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY]:
                # In range-bound markets, price tends to revert from bands
                if testing_upper:
                    return TradingSignal(
                        type=SignalType.BEARISH,
                        conviction=0.7,
                        indicator="BB Upper Band Reversion",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
                elif testing_lower:
                    return TradingSignal(
                        type=SignalType.BULLISH,
                        conviction=0.7,
                        indicator="BB Lower Band Reversion",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
            
            elif self.current_market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # In trending markets, price can ride the bands
                # Look for pullback to middle band in a trend
                prev_close = data['close'].iloc[-2]
                
                if self.current_market_regime == MarketRegime.TRENDING_UP:
                    pullback_to_middle = prev_close > current_close and abs(current_close - middle_band) / middle_band < 0.01
                    if pullback_to_middle:
                        return TradingSignal(
                            type=SignalType.BULLISH,
                            conviction=0.65,
                            indicator="BB Trend Continuation",
                            timestamp=pd.Timestamp(timestamp),
                            price=current_close
                        )
                else:  # TRENDING_DOWN
                    pullback_to_middle = prev_close < current_close and abs(current_close - middle_band) / middle_band < 0.01
                    if pullback_to_middle:
                        return TradingSignal(
                            type=SignalType.BEARISH,
                            conviction=0.65,
                            indicator="BB Trend Continuation",
                            timestamp=pd.Timestamp(timestamp),
                            price=current_close
                        )
            
            # Check for Bollinger Band squeeze (low volatility)
            if band_width < 0.03:
                # A very tight squeeze often precedes a big move
                # But direction is unclear, so conviction is lower
                
                # Look for a small breakout to confirm direction
                for i in range(1, 4):
                    # Check last few days for a mini-breakout
                    if i < len(data):
                        if data['close'].iloc[-i] > data['bb_upper'].iloc[-i]:
                            return TradingSignal(
                                type=SignalType.BULLISH,
                                conviction=0.6,
                                indicator="BB Squeeze Breakout",
                                timestamp=pd.Timestamp(timestamp),
                                price=current_close
                            )
                        elif data['close'].iloc[-i] < data['bb_lower'].iloc[-i]:
                            return TradingSignal(
                                type=SignalType.BEARISH,
                                conviction=0.6,
                                indicator="BB Squeeze Breakout",
                                timestamp=pd.Timestamp(timestamp),
                                price=current_close
                            )
                            
            return None
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signal for {self.symbol}: {str(e)}")
            return None
    
    def _generate_volatility_signal(self, data: pd.DataFrame, timestamp) -> Optional[TradingSignal]:
        """Generate signal based on volatility patterns"""
        try:
            if data.empty or 'atr' not in data.columns or 'volatility' not in data.columns:
                return None
                
            current_close = data['close'].iloc[-1]
            current_atr = data['atr'].iloc[-1]
            current_volatility = data['volatility'].iloc[-1]
            
            # Get historical volatility
            hist_vol = data['volatility'].iloc[-self.params["volatility_lookback"]:].mean()
            vol_ratio = current_volatility / hist_vol if hist_vol > 0 else 1.0
            
            # Detect volatility contractions (often precede breakouts)
            is_vol_contracting = True
            for i in range(1, 5):
                if i < len(data) and data['volatility'].iloc[-i] < data['volatility'].iloc[-i-1]:
                    continue
                is_vol_contracting = False
                break
                
            # Detect volatility expansions (often signal trend changes)
            is_vol_expanding = vol_ratio > 1.5
            
            # Volatility contraction followed by price move
            if is_vol_contracting:
                # Look for early price movement to determine direction
                price_change = (current_close - data['close'].iloc[-5]) / data['close'].iloc[-5]
                
                if price_change > 0.01:  # Price moving up after volatility contraction
                    return TradingSignal(
                        type=SignalType.BULLISH,
                        conviction=0.6,
                        indicator="Volatility Contraction",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
                elif price_change < -0.01:  # Price moving down after volatility contraction
                    return TradingSignal(
                        type=SignalType.BEARISH,
                        conviction=0.6,
                        indicator="Volatility Contraction",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
            
            # Extreme volatility often leads to mean reversion
            if vol_ratio > 2.5:
                # In extremely high volatility, prepare for mean reversion
                # Check recent price direction
                recent_trend = (current_close - data['close'].iloc[-3]) / data['close'].iloc[-3]
                
                if recent_trend > 0.03:  # Sharp up move with high volatility
                    return TradingSignal(
                        type=SignalType.BEARISH,
                        conviction=0.55,
                        indicator="Volatility Mean Reversion",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
                elif recent_trend < -0.03:  # Sharp down move with high volatility
                    return TradingSignal(
                        type=SignalType.BULLISH,
                        conviction=0.55,
                        indicator="Volatility Mean Reversion",
                        timestamp=pd.Timestamp(timestamp),
                        price=current_close
                    )
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating volatility signal for {self.symbol}: {str(e)}")
            return None
    
    def _combine_signals(self, signals: List[Optional[TradingSignal]], weights: List[float], timestamp) -> Optional[TradingSignal]:
        """
        Combine multiple signals into a single consensus signal.
        
        Args:
            signals: List of signals from different indicators
            weights: Weights for each signal type
            timestamp: Current timestamp
            
        Returns:
            Combined signal or None if no clear signal
        """
        if not signals or all(s is None for s in signals):
            return None
            
        # Filter out None signals
        valid_signals = [(s, w) for s, w in zip(signals, weights) if s is not None]
        
        if not valid_signals:
            return None
            
        # Count bullish and bearish signals with their weights
        bullish_conviction = 0.0
        bearish_conviction = 0.0
        neutral_conviction = 0.0
        total_weight = 0.0
        
        for signal, weight in valid_signals:
            if signal.type == SignalType.BULLISH:
                bullish_conviction += signal.conviction * weight
            elif signal.type == SignalType.BEARISH:
                bearish_conviction += signal.conviction * weight
            else:
                neutral_conviction += signal.conviction * weight
            total_weight += weight
        
        # Normalize by sum of weights
        if total_weight > 0:
            bullish_conviction /= total_weight
            bearish_conviction /= total_weight
            neutral_conviction /= total_weight
            
        # Determine the dominant signal
        max_conviction = max(bullish_conviction, bearish_conviction, neutral_conviction)
        
        if max_conviction < 0.1:  # No strong signal
            return None
            
        # Get current price from the first valid signal
        current_price = valid_signals[0][0].price
        
        # Adjust conviction based on market regime
        regime_adjustment = 1.0
        if self.current_market_regime == MarketRegime.HIGH_VOLATILITY:
            regime_adjustment = 0.8  # Reduce conviction in high volatility
        elif self.current_market_regime == MarketRegime.EXTREME_VOLATILITY:
            regime_adjustment = 0.7  # Reduce conviction even more in extreme volatility
        elif self.current_market_regime == MarketRegime.LOW_VOLATILITY:
            regime_adjustment = 1.1  # Increase conviction in low volatility
            
        # Create the combined signal
        if bullish_conviction == max_conviction and bullish_conviction > 0.3:
            # Get ATR for stop-loss and take-profit if available
            atr = self.indicators_data.get('atr', current_price * 0.02)
            
            return TradingSignal(
                type=SignalType.BULLISH,
                conviction=bullish_conviction * regime_adjustment,
                indicator="Combined Signal",
                timestamp=pd.Timestamp(timestamp),
                price=current_price,
                stop_loss=current_price - (atr * self.params["stop_loss_atr_mult"]),
                take_profit=current_price + (atr * self.params["take_profit_atr_mult"])
            )
            
        elif bearish_conviction == max_conviction and bearish_conviction > 0.3:
            # Get ATR for stop-loss and take-profit if available
            atr = self.indicators_data.get('atr', current_price * 0.02)
            
            return TradingSignal(
                type=SignalType.BEARISH,
                conviction=bearish_conviction * regime_adjustment,
                indicator="Combined Signal",
                timestamp=pd.Timestamp(timestamp),
                price=current_price,
                stop_loss=current_price + (atr * self.params["stop_loss_atr_mult"]),
                take_profit=current_price - (atr * self.params["take_profit_atr_mult"])
            )
            
        return None
    
    def _calculate_position_size(self, signal: TradingSignal, price: float) -> float:
        """
        Calculate position size based on signal conviction and risk parameters.
        
        Args:
            signal: The trading signal
            price: Current price
            
        Returns:
            Position size in number of shares
        """
        if not signal:
            return 0.0
            
        # Start with base position size
        position_size_pct = self.params["base_position_size"]
        
        # Adjust based on conviction
        conviction_adj = signal.conviction / 0.8  # Scale so that 0.8 conviction = 100% of base size
        position_size_pct *= min(1.2, conviction_adj)  # Cap at 120% of base size
        
        # Adjust based on market regime
        if self.current_market_regime == MarketRegime.HIGH_VOLATILITY:
            position_size_pct *= self.params["high_vol_size_adj"]
        elif self.current_market_regime == MarketRegime.EXTREME_VOLATILITY:
            position_size_pct *= self.params["high_vol_size_adj"] * 0.7  # Further reduce in extreme volatility
        elif self.current_market_regime == MarketRegime.LOW_VOLATILITY:
            position_size_pct *= self.params["low_vol_size_adj"]
            
        # Cap at maximum position size
        position_size_pct = min(position_size_pct, self.params["max_position_size"])
        
        # If we have a risk manager, use it for position sizing
        if self.risk_manager:
            try:
                # Get portfolio value
                portfolio_value = self.risk_manager.get_portfolio_value()
                
                # Calculate dollar amount to invest
                dollar_amount = portfolio_value * position_size_pct
                
                # Calculate number of shares
                shares = dollar_amount / price
                
                # Let risk manager adjust this
                adjusted_shares = self.risk_manager.calculate_position_size(
                    self.symbol, 
                    shares, 
                    signal.stop_loss if signal.type == SignalType.BULLISH else None
                )
                
                return adjusted_shares
            except Exception as e:
                logger.error(f"Error calculating position size with risk manager: {str(e)}")
                
        # Fallback if no risk manager or error
        # Assume a default portfolio value of 100,000
        portfolio_value = 100000.0
        dollar_amount = portfolio_value * position_size_pct
        
        return dollar_amount / price 