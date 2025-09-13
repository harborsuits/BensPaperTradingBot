#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Reversion Strategy Template

This module implements a comprehensive mean reversion strategy template
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

class MeanReversionTemplate(StrategyOptimizable):
    """
    Mean Reversion Strategy Template that identifies overbought and oversold conditions
    and generates signals when price is expected to revert to the mean.
    
    Supported indicators include:
    - RSI (Relative Strength Index)
    - Bollinger Bands
    - Stochastic Oscillator
    - Price deviations from moving averages
    - Statistical Z-scores
    
    This template can be customized and extended for specific mean reversion approaches.
    """
    
    def __init__(
        self,
        name: str = "mean_reversion",
        parameters: Dict[str, Any] = None,
        metadata: Optional[StrategyMetadata] = None
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            metadata: Strategy metadata
        """
        # Default parameters for mean reversion
        default_params = {
            # General parameters
            "lookback_period": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 2.0,
            "max_holding_period": 10,
            "risk_per_trade": 0.01,
            
            # Indicator parameters
            "use_rsi": True,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            
            "use_bollinger": True,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            
            "use_stochastic": False,
            "stochastic_k_period": 14,
            "stochastic_d_period": 3,
            "stochastic_slowing": 3,
            "stochastic_upper": 80,
            "stochastic_lower": 20,
            
            "use_zscore": True,
            "zscore_period": 20,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            
            # Confirmation parameters
            "require_volume_confirmation": False,
            "volume_confirmation_threshold": 1.5,
            "require_multiple_indicators": True,
            "min_indicators_for_signal": 2
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = StrategyMetadata(
                name=name,
                version="1.0.0",
                description="Mean reversion strategy that trades overbought and oversold conditions",
                author="Trading Bot",
                timeframes=[
                    TimeFrame.MINUTE_15,
                    TimeFrame.MINUTE_30,
                    TimeFrame.HOUR_1,
                    TimeFrame.HOUR_4,
                    TimeFrame.DAY_1
                ],
                asset_classes=["stocks", "forex", "crypto", "etf"],
                tags=["mean_reversion", "oscillator", "overbought", "oversold"]
            )
        
        # Initialize parent class
        super().__init__(name, default_params, metadata)
        
        logger.info(f"Initialized mean reversion strategy: {name}")
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate mean reversion indicators for the given data.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            Dictionary of symbol -> indicators dictionary
        """
        indicators = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            if len(df) < self.parameters["lookback_period"]:
                continue
            
            symbol_indicators = {}
            
            # Extract price and volume data
            close_prices = df['close'].values
            high_prices = df['high'].values if 'high' in df else close_prices
            low_prices = df['low'].values if 'low' in df else close_prices
            volumes = df['volume'].values if 'volume' in df else np.ones_like(close_prices)
            
            # Calculate RSI
            if self.parameters["use_rsi"]:
                try:
                    rsi_values = talib.RSI(close_prices, timeperiod=self.parameters["rsi_period"])
                    symbol_indicators['rsi'] = pd.Series(rsi_values, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating RSI for {symbol}: {e}")
                    # Fallback calculation if talib fails
                    rsi_values = self._calculate_rsi(close_prices, self.parameters["rsi_period"])
                    symbol_indicators['rsi'] = pd.Series(rsi_values, index=df.index)
            
            # Calculate Bollinger Bands
            if self.parameters["use_bollinger"]:
                try:
                    upper, middle, lower = talib.BBANDS(
                        close_prices,
                        timeperiod=self.parameters["bollinger_period"],
                        nbdevup=self.parameters["bollinger_std"],
                        nbdevdn=self.parameters["bollinger_std"]
                    )
                    
                    symbol_indicators['bollinger_upper'] = pd.Series(upper, index=df.index)
                    symbol_indicators['bollinger_middle'] = pd.Series(middle, index=df.index)
                    symbol_indicators['bollinger_lower'] = pd.Series(lower, index=df.index)
                    
                    # Calculate Bollinger Band width
                    bb_width = (upper - lower) / middle
                    symbol_indicators['bollinger_width'] = pd.Series(bb_width, index=df.index)
                    
                    # Calculate percent B
                    percent_b = (close_prices - lower) / (upper - lower)
                    symbol_indicators['percent_b'] = pd.Series(percent_b, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
            
            # Calculate Stochastic Oscillator
            if self.parameters["use_stochastic"]:
                try:
                    slowk, slowd = talib.STOCH(
                        high_prices, low_prices, close_prices,
                        fastk_period=self.parameters["stochastic_k_period"],
                        slowk_period=self.parameters["stochastic_slowing"],
                        slowk_matype=0,
                        slowd_period=self.parameters["stochastic_d_period"],
                        slowd_matype=0
                    )
                    
                    symbol_indicators['stochastic_k'] = pd.Series(slowk, index=df.index)
                    symbol_indicators['stochastic_d'] = pd.Series(slowd, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating Stochastic Oscillator for {symbol}: {e}")
            
            # Calculate Z-Score
            if self.parameters["use_zscore"]:
                try:
                    # Calculate rolling mean and standard deviation
                    period = self.parameters["zscore_period"]
                    rolling_mean = df['close'].rolling(window=period).mean()
                    rolling_std = df['close'].rolling(window=period).std()
                    
                    # Calculate Z-Score
                    zscore = (df['close'] - rolling_mean) / rolling_std
                    symbol_indicators['zscore'] = zscore
                except Exception as e:
                    logger.error(f"Error calculating Z-Score for {symbol}: {e}")
            
            # Calculate ATR for position sizing and stop loss
            try:
                atr = talib.ATR(
                    high_prices, low_prices, close_prices,
                    timeperiod=14
                )
                symbol_indicators['atr'] = pd.Series(atr, index=df.index)
                symbol_indicators['atr_percent'] = symbol_indicators['atr'] / df['close']
            except Exception as e:
                logger.error(f"Error calculating ATR for {symbol}: {e}")
                # Fallback simple volatility measure
                close_returns = df['close'].pct_change()
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
        Generate mean reversion trading signals for the given data.
        
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
            available_indicators = len(symbol_indicators)
            if available_indicators < min_indicators and self.parameters["require_multiple_indicators"]:
                continue
            
            df = data[symbol]
            current_idx = len(df) - 1
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            # Initialize signal flags
            bullish_signals = 0
            bearish_signals = 0
            total_potential_signals = 0
            
            # Check RSI
            if 'rsi' in symbol_indicators and self.parameters["use_rsi"]:
                total_potential_signals += 1
                rsi = symbol_indicators['rsi'].iloc[-1]
                
                if rsi < self.parameters["rsi_oversold"]:
                    bullish_signals += 1
                elif rsi > self.parameters["rsi_overbought"]:
                    bearish_signals += 1
            
            # Check Bollinger Bands
            if 'bollinger_lower' in symbol_indicators and 'bollinger_upper' in symbol_indicators and self.parameters["use_bollinger"]:
                total_potential_signals += 1
                lower_band = symbol_indicators['bollinger_lower'].iloc[-1]
                upper_band = symbol_indicators['bollinger_upper'].iloc[-1]
                
                if current_price < lower_band:
                    bullish_signals += 1
                elif current_price > upper_band:
                    bearish_signals += 1
                
                # Additional check with percent B if available
                if 'percent_b' in symbol_indicators:
                    percent_b = symbol_indicators['percent_b'].iloc[-1]
                    
                    # Extreme values of percent B (close to 0 or 1)
                    if not np.isnan(percent_b):
                        if percent_b < 0.05:  # Very oversold
                            bullish_signals += 0.5
                        elif percent_b > 0.95:  # Very overbought
                            bearish_signals += 0.5
            
            # Check Stochastic
            if 'stochastic_k' in symbol_indicators and 'stochastic_d' in symbol_indicators and self.parameters["use_stochastic"]:
                total_potential_signals += 1
                k = symbol_indicators['stochastic_k'].iloc[-1]
                d = symbol_indicators['stochastic_d'].iloc[-1]
                
                # Check for oversold/overbought
                if k < self.parameters["stochastic_lower"] and d < self.parameters["stochastic_lower"]:
                    bullish_signals += 1
                elif k > self.parameters["stochastic_upper"] and d > self.parameters["stochastic_upper"]:
                    bearish_signals += 1
                
                # Check for crossovers
                if current_idx > 0:
                    prev_k = symbol_indicators['stochastic_k'].iloc[-2]
                    prev_d = symbol_indicators['stochastic_d'].iloc[-2]
                    
                    # K crosses above D (bullish)
                    if prev_k < prev_d and k > d and k < 30:
                        bullish_signals += 0.5
                    
                    # K crosses below D (bearish)
                    elif prev_k > prev_d and k < d and k > 70:
                        bearish_signals += 0.5
            
            # Check Z-Score
            if 'zscore' in symbol_indicators and self.parameters["use_zscore"]:
                total_potential_signals += 1
                zscore = symbol_indicators['zscore'].iloc[-1]
                
                zscore_entry = self.parameters["zscore_entry"]
                
                if not np.isnan(zscore):
                    if zscore < -zscore_entry:  # Oversold
                        bullish_signals += 1
                    elif zscore > zscore_entry:  # Overbought
                        bearish_signals += 1
            
            # Check volume confirmation if required
            if self.parameters["require_volume_confirmation"] and 'volume' in df:
                avg_volume = df['volume'].rolling(window=10).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                
                volume_confirmed = current_volume > avg_volume * self.parameters["volume_confirmation_threshold"]
                
                if not volume_confirmed:
                    # Reduce signal strength if volume confirmation is required but not present
                    bullish_signals *= 0.5
                    bearish_signals *= 0.5
            
            # Determine signal type
            signal_type = None
            signal_strength = 0
            min_signals_required = min(
                self.parameters["min_indicators_for_signal"],
                total_potential_signals
            )
            
            if bullish_signals >= min_signals_required:
                signal_type = SignalType.BUY
                signal_strength = bullish_signals / total_potential_signals
            elif bearish_signals >= min_signals_required:
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
                    trailing_stop_pct=0.02,  # Default 2% trailing stop
                    metadata={
                        "strategy": self.name,
                        "indicators": {
                            "bullish_count": bullish_signals,
                            "bearish_count": bearish_signals,
                            "total_indicators": total_potential_signals
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
            "lookback_period": [10, 20, 30, 50],
            "entry_threshold": [1.5, 2.0, 2.5, 3.0],
            "exit_threshold": [0.3, 0.5, 0.8, 1.0],
            "stop_loss_atr_multiple": [1.0, 1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [1.0, 1.5, 2.0, 2.5, 3.0],
            "max_holding_period": [5, 10, 15, 20],
            "risk_per_trade": [0.005, 0.01, 0.015, 0.02],
            
            # Indicator parameters
            "rsi_period": [7, 10, 14, 21],
            "rsi_overbought": [65, 70, 75, 80],
            "rsi_oversold": [20, 25, 30, 35],
            
            "bollinger_period": [10, 20, 30],
            "bollinger_std": [1.5, 2.0, 2.5, 3.0],
            
            "stochastic_k_period": [9, 14, 21],
            "stochastic_d_period": [3, 5, 7],
            
            "zscore_period": [10, 20, 30],
            "zscore_entry": [1.5, 2.0, 2.5, 3.0],
            "zscore_exit": [0.3, 0.5, 0.8, 1.0],
            
            # Configuration parameters
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
        # Mean reversion strategies typically work best in range-bound markets
        compatibility_map = {
            MarketRegime.RANGE_BOUND: 0.9,   # Best in range-bound markets
            MarketRegime.NEUTRAL: 0.8,       # Good in neutral markets
            MarketRegime.VOLATILE: 0.6,      # Can work in volatile markets
            MarketRegime.BULLISH: 0.4,       # Less effective in strong trends
            MarketRegime.BEARISH: 0.4,       # Less effective in strong trends
            MarketRegime.TRENDING: 0.3,      # Poor in trending markets
            MarketRegime.UNKNOWN: 0.5        # Moderate in unknown regimes
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
        if regime == MarketRegime.RANGE_BOUND:
            # Ideal for mean reversion
            adjustment = 1.3
        elif regime == MarketRegime.TRENDING:
            # Reduce confidence in trending markets
            adjustment = 0.7
        elif regime == MarketRegime.VOLATILE:
            # Can work in volatile markets but with caution
            adjustment = 0.9
        elif regime == MarketRegime.BULLISH and signal_type == SignalType.SELL:
            # Reduce confidence for shorts in bullish market
            adjustment = 0.6
        elif regime == MarketRegime.BEARISH and signal_type == SignalType.BUY:
            # Reduce confidence for longs in bearish market
            adjustment = 0.6
        
        return adjustment
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate RSI manually (fallback if talib fails).
        
        Args:
            prices: Array of prices
            period: RSI period
            
        Returns:
            Array of RSI values
        """
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        
        # Initialize gains and losses
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            # Handle divide by zero
            rs_values = np.ones_like(prices) * 100
            return rs_values
        
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period+1] = 100. - (100. / (1. + rs))
        
        # Calculate RSI for the rest of the data
        for i in range(period+1, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period-1) + upval) / period
            down = (down * (period-1) + downval) / period
            
            rs = up / down if down != 0 else 999
            rsi[i] = 100. - (100. / (1. + rs))
        
        return rsi
    
    def detect_stock_characteristics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Detect stock-specific characteristics relevant for mean reversion.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            Dictionary of symbol -> characteristics dictionary
        """
        characteristics = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            if len(df) < 50:  # Need enough data for meaningful statistics
                continue
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate autocorrelation (mean reversion tendency)
            autocorr = returns.autocorr(lag=1)
            
            # Calculate half-life of mean reversion
            half_life = self._calculate_half_life(returns)
            
            # Categorize mean reversion strength
            if autocorr < -0.3:
                mr_strength = "strong"
            elif autocorr < -0.1:
                mr_strength = "moderate"
            elif autocorr < 0:
                mr_strength = "weak"
            else:
                mr_strength = "none"  # Positive autocorrelation suggests momentum, not mean reversion
            
            # Store characteristics
            characteristics[symbol] = {
                "volatility": volatility,
                "autocorrelation": autocorr,
                "half_life": half_life,
                "mean_reversion_strength": mr_strength,
                "suitable_for_strategy": mr_strength in ["strong", "moderate"]
            }
        
        return characteristics
    
    def _calculate_half_life(self, returns: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            returns: Series of returns
            
        Returns:
            Half-life in periods
        """
        # Lag the returns
        lag_returns = returns.shift(1).dropna()
        returns = returns.iloc[1:]  # Align with lagged returns
        
        # Linear regression: r_t = b * r_{t-1} + a
        y = returns.values
        x = lag_returns.values.reshape(-1, 1)
        
        # Add constant for intercept
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        
        # Solve using normal equations
        try:
            beta = np.linalg.inv(x.T @ x) @ x.T @ y
            b = beta[1]  # Coefficient on the lagged return
            
            # Calculate half-life: ln(0.5) / ln(|b|)
            if b < 0 and b > -1:  # Valid range for mean reversion
                half_life = np.log(0.5) / np.log(abs(b))
                return half_life
            else:
                return float('inf')  # No mean reversion
        except:
            return float('inf')  # Error in calculation 