#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Strategy Module

This module implements a momentum-based trading strategy that capitalizes on the 
continuance of existing price trends in the market.

The momentum strategy is based on the empirical observation that assets which have 
performed well (or poorly) in the recent past tend to continue performing well (or poorly) 
in the near future. This phenomenon, often called price momentum, has been documented 
across various asset classes and time periods.

Key concepts implemented in this strategy:
1. Relative Strength Index (RSI) to identify overbought/oversold conditions
2. Rate of Change (ROC) to measure price momentum
3. Average Directional Index (ADX) to gauge trend strength
4. Volatility-adjusted momentum to normalize across different assets
5. Dynamic stop-loss and take-profit levels using Average True Range (ATR)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.base.stock_base import StockBaseStrategy
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime

logger = logging.getLogger(__name__)

class MomentumStrategy(StockBaseStrategy):
    """
    Momentum Trading Strategy
    
    This strategy identifies and trades stocks displaying strong momentum characteristics,
    entering positions in the direction of the established trend while using technical 
    indicators to filter for high-probability momentum continuation setups.
    
    Key characteristics:
    - Trend-following approach that aims to capture directional price movements
    - Uses multiple technical indicators for signal confirmation and filtering
    - Incorporates volatility adjustments to normalize momentum across assets
    - Implements dynamic position sizing and risk management based on ATR
    - Can operate in both single-stock analysis or cross-sectional ranking mode
    
    Key indicators and their roles:
    - Rate of Change (ROC): Primary momentum measure showing the velocity of price change
    - Relative Strength Index (RSI): Identifies overbought/oversold conditions and momentum divergence
    - Average Directional Index (ADX): Confirms trend strength to avoid range-bound markets
    - Plus/Minus Directional Indicators (+DI/-DI): Determines trend direction alongside ADX
    - Volatility-adjusted momentum: Normalizes momentum signals across assets with different volatility profiles
    - Volume analysis: Confirms price movements with proportional volume support
    
    Signal generation methodology:
    - BUY signals occur when RSI rebounds from oversold conditions with rising momentum and ADX confirmation
    - SELL signals trigger from overbought RSI readings, declining momentum, or reversal patterns
    - Confidence scoring incorporates multiple indicator alignment and trend strength metrics
    - Position sizing scales with signal strength, volatility, and account risk parameters
    
    Ideal market conditions:
    - Strong trending markets (particularly bullish environments)
    - Lower volatility periods allowing trends to develop
    - Liquid markets with sufficient trading volume
    - Markets with sector/industry rotation providing new momentum opportunities
    
    Limitations:
    - Underperforms in choppy, sideways, or mean-reverting markets
    - Subject to momentum crashes during sudden market regime changes
    - May have delayed entries as confirmation signals develop
    - Performance varies significantly across different market regimes
    """
    
    # Default stock parameters
    DEFAULT_STOCK_PARAMS = {
        "min_price": 5.0,              # Minimum stock price for filtering
        "min_volume": 100000,          # Minimum average daily volume
        "min_market_cap": 100000000,   # Minimum market cap ($100M)
        "max_price_to_earnings": 100,  # Maximum P/E ratio
        "universe": [],                # Default empty for all stocks
        "exclude_list": [],            # Stocks to exclude
        "sector_weights": {},          # Equal sector weighting by default
        "volatility_cap": 0.03,        # Maximum daily volatility 
        "beta_range": [0.5, 2.0],      # Beta range for filtering
    }
    
    # Default parameters specific to momentum trading
    DEFAULT_MOMENTUM_PARAMS = {
        # Momentum calculation parameters
        "lookback_period": 14,           # Period for calculating momentum indicators (typically 10-30 days)
        "overbought": 70,                # RSI threshold to identify overbought conditions (70-80)
        "oversold": 30,                  # RSI threshold to identify oversold conditions (20-30)
        
        # Trend identification parameters
        "adx_threshold": 25,             # Minimum ADX value to confirm trend strength (20-30)
        "trend_strength_threshold": 0.05, # Minimum price change % to confirm trend
        
        # Signal generation parameters
        "use_volatility_adjustment": True, # Whether to normalize momentum by volatility
        "cross_sectional": False,         # Whether to rank stocks against each other
        "signal_threshold": 0.0,          # Minimum signal strength to generate a trade
        "volatility_lookback": 20,        # Period for volatility calculation
        
        # Risk management parameters
        "stop_loss_atr_multiple": 2.0,    # Stop loss distance in ATR units
        "take_profit_atr_multiple": 3.0,  # Take profit distance in ATR units
    }
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the momentum strategy with configurable parameters.
        
        Args:
            name: Strategy name 
            parameters: Strategy parameters (will be merged with default parameters)
            metadata: Strategy metadata for tracking and identification
        """
        # Start with default stock parameters
        momentum_params = self.DEFAULT_STOCK_PARAMS.copy()
        
        # Add momentum-specific parameters
        momentum_params.update(self.DEFAULT_MOMENTUM_PARAMS)
        
        # Override with provided parameters
        if parameters:
            momentum_params.update(parameters)
            
        # Add strategy name to params
        momentum_params['strategy_name'] = name
        
        # Initialize the parent class with the combined parameters
        super().__init__(momentum_params)
        
        logger.info(f"Initialized momentum strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Defines the range of possible values for each parameter that can be
        optimized during strategy backtesting, walk-forward analysis, or
        parameter tuning.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "lookback_period": [5, 10, 14, 20, 30],
            "overbought": [65, 70, 75, 80],
            "oversold": [20, 25, 30, 35],
            "adx_threshold": [20, 25, 30],
            "trend_strength_threshold": [0.03, 0.05, 0.07],
            "use_volatility_adjustment": [True, False],
            "cross_sectional": [True, False],
            "signal_threshold": [0.0, 0.1, 0.2],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0, 5.0],
        }
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate momentum indicators for all symbols.
        
        Computes a comprehensive set of technical indicators used to identify momentum
        characteristics, trend strength, and potential entry and exit signals. These
        indicators form the basis for the signal generation logic.
        
        Key indicators calculated:
        - Price momentum: Percentage change over lookback period
        - Rate of Change (ROC): Normalized price change
        - Relative Strength Index (RSI): Oscillator identifying overbought/oversold conditions
        - Average Directional Index (ADX): Measure of trend strength
        - Average True Range (ATR): Volatility measure used for position sizing
        - Volatility-adjusted momentum: Momentum normalized by historical volatility
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol, organized by indicator type
        """
        indicators = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            if len(df) < self.parameters["lookback_period"]:
                continue
                
            try:
                # Calculate price momentum (close price change over lookback period)
                lookback = self.parameters["lookback_period"]
                momentum = df['close'].pct_change(lookback)
                
                # Calculate Rate of Change (ROC)
                roc = (df['close'] / df['close'].shift(lookback) - 1) * 100
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=lookback).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=lookback).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Calculate Average Directional Index (ADX) for trend strength
                high_diff = df['high'].diff()
                low_diff = df['low'].diff().abs()
                
                plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
                minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
                
                tr = pd.DataFrame({
                    'hl': df['high'] - df['low'],
                    'hc': (df['high'] - df['close'].shift()).abs(),
                    'lc': (df['low'] - df['close'].shift()).abs()
                }).max(axis=1)
                
                atr = tr.rolling(window=lookback).mean()
                
                plus_di = 100 * (plus_dm.rolling(window=lookback).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(window=lookback).mean() / atr)
                
                dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
                adx = dx.rolling(window=lookback).mean()
                
                # Prepare volatility-adjusted momentum if enabled
                if self.parameters["use_volatility_adjustment"]:
                    volatility = df['close'].pct_change().rolling(
                        window=self.parameters["volatility_lookback"]).std() * np.sqrt(252)
                    # Add small constant to avoid division by zero
                    vol_adj_momentum = momentum / (volatility + 1e-8)
                else:
                    vol_adj_momentum = momentum
                
                # Store indicators
                indicators[symbol] = {
                    "momentum": pd.DataFrame({"momentum": momentum}),
                    "roc": pd.DataFrame({"roc": roc}),
                    "rsi": pd.DataFrame({"rsi": rsi}),
                    "adx": pd.DataFrame({"adx": adx}),
                    "atr": pd.DataFrame({"atr": atr}),
                    "vol_adj_momentum": pd.DataFrame({"vol_adj_momentum": vol_adj_momentum}),
                    "plus_di": pd.DataFrame({"plus_di": plus_di}),
                    "minus_di": pd.DataFrame({"minus_di": minus_di})
                }
                
            except Exception as e:
                logger.error(f"Error calculating momentum indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals based on momentum indicators.
        
        Analyzes the calculated indicators to identify high-probability momentum
        trading opportunities. Signals are generated based on multiple confirmation
        factors, with signal confidence determined by the strength and alignment
        of various indicators.
        
        Signal generation logic:
        - BUY signals: When RSI rebounds from oversold conditions with strong trend
          confirmation from ADX and positive momentum
        - SELL signals: When RSI reaches overbought levels, momentum turns negative,
          or momentum shows signs of weakening after a strong uptrend
        
        Each signal includes:
        - Entry price
        - Stop-loss and take-profit levels calculated using ATR
        - Signal confidence based on indicator alignment
        - Metadata with indicator values for logging and analysis
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects for execution
        """
        # Apply stock-specific filters from the base class
        filtered_data = self.filter_universe(data)
        
        # Calculate indicators
        indicators = self.calculate_indicators(filtered_data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data
                latest_data = filtered_data[symbol].iloc[-1]
                prev_data = filtered_data[symbol].iloc[-2] if len(filtered_data[symbol]) > 1 else None
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get indicator values
                latest_momentum = symbol_indicators["momentum"].iloc[-1]["momentum"]
                latest_roc = symbol_indicators["roc"].iloc[-1]["roc"]
                latest_rsi = symbol_indicators["rsi"].iloc[-1]["rsi"]
                latest_adx = symbol_indicators["adx"].iloc[-1]["adx"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Previous values for trend analysis
                prev_rsi = symbol_indicators["rsi"].iloc[-2]["rsi"] if len(symbol_indicators["rsi"]) > 1 else 50
                prev_momentum = symbol_indicators["momentum"].iloc[-2]["momentum"] if len(symbol_indicators["momentum"]) > 1 else 0
                
                # Get parameters
                lookback = self.parameters["lookback_period"]
                overbought = self.parameters["overbought"]
                oversold = self.parameters["oversold"]
                adx_threshold = self.parameters["adx_threshold"]
                stop_loss_atr_multiple = self.parameters["stop_loss_atr_multiple"]
                take_profit_atr_multiple = self.parameters["take_profit_atr_multiple"]
                
                # Generate signal based on momentum conditions
                signal_type = None
                confidence = 0.0
                
                # BUY conditions:
                # 1. Strong upward momentum (positive ROC)
                # 2. RSI was oversold but is now increasing
                # 3. ADX indicates strong trend
                if (latest_roc > 0 and
                    latest_rsi > oversold and
                    prev_rsi <= oversold and
                    latest_adx > adx_threshold):
                    
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on multiple factors
                    # 1. Momentum strength
                    momentum_confidence = min(0.3, abs(latest_momentum) * 5)
                    
                    # 2. Trend strength via ADX
                    trend_confidence = min(0.3, latest_adx / 100)
                    
                    # 3. RSI confirmation
                    rsi_confirmation = min(0.2, (latest_rsi - oversold) / 20)
                    
                    # 4. Volume confirmation (if available)
                    volume_confidence = 0.0
                    if 'volume' in filtered_data[symbol].columns:
                        avg_volume = filtered_data[symbol]['volume'].rolling(window=20).mean().iloc[-1]
                        if filtered_data[symbol]['volume'].iloc[-1] > avg_volume:
                            volume_confidence = 0.2
                    
                    confidence = min(0.9, momentum_confidence + trend_confidence + rsi_confirmation + volume_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                
                # SELL conditions:
                # 1. Momentum turns negative
                # 2. RSI reaches overbought territory
                # 3. Momentum weakening after being strong
                elif (latest_roc < 0 or
                      latest_rsi >= overbought or
                      (prev_momentum > latest_momentum and latest_momentum > 0 and latest_rsi > 60)):
                    
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence for sell signal
                    # 1. Negative momentum strength
                    momentum_confidence = min(0.3, abs(latest_momentum) * 5) if latest_momentum < 0 else 0.1
                    
                    # 2. Overbought confirmation
                    overbought_confirmation = min(0.3, (latest_rsi - 50) / 30) if latest_rsi > 50 else 0.1
                    
                    # 3. Trend weakening
                    trend_weakening = min(0.2, (prev_momentum - latest_momentum) * 10) if prev_momentum > latest_momentum else 0.1
                    
                    # 4. Volume confirmation (if available)
                    volume_confidence = 0.0
                    if 'volume' in filtered_data[symbol].columns:
                        avg_volume = filtered_data[symbol]['volume'].rolling(window=20).mean().iloc[-1]
                        if filtered_data[symbol]['volume'].iloc[-1] > avg_volume:
                            volume_confidence = 0.2
                    
                    confidence = min(0.9, momentum_confidence + overbought_confirmation + trend_weakening + volume_confidence)
                    
                    # Calculate stop loss and take profit for short position
                    stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_price - (latest_atr * take_profit_atr_multiple)
                
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
                        timeframe=TimeFrame.DAY_1,
                        metadata={
                            "strategy_type": "momentum",
                            "lookback_period": lookback,
                            "rsi": latest_rsi,
                            "adx": latest_adx,
                            "momentum": latest_momentum,
                            "roc": latest_roc,
                            "atr": latest_atr
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating momentum signal for {symbol}: {e}")
        
        return signals
    
    def _calculate_performance_score(self, signals: Dict[str, Signal], 
                                   data: Dict[str, Any]) -> float:
        """
        Calculate a performance score for the generated signals.
        
        Evaluates the expected performance of generated signals based on historical 
        forward returns. This is used primarily for strategy optimization and 
        parameter tuning during the development and backtesting phase.
        
        The performance calculation simulates the outcomes of the generated signals
        using known forward price data and computes a Sharpe-like ratio to balance
        returns against consistency.
        
        Args:
            signals: Generated trading signals
            data: Historical price data including forward periods
            
        Returns:
            Performance score (higher is better) based on risk-adjusted returns
        """
        if not signals:
            return 0.0
            
        # This is a simplified performance calculation
        # In a real implementation, you would simulate trades and calculate metrics
            
        # Assume we have forward returns in the data (for backtest evaluation)
        forward_returns = {}
        for symbol, signal in signals.items():
            if symbol not in data:
                continue
                
            signal_date_idx = None
            for i, date in enumerate(data[symbol].index):
                if date >= signal.timestamp:
                    signal_date_idx = i
                    break
                    
            if signal_date_idx is None or signal_date_idx >= len(data[symbol]) - 5:
                continue
                
            # Get 5-day forward return
            entry_price = signal.price
            exit_price = data[symbol]['close'].iloc[signal_date_idx + 5]
            
            # Calculate return based on signal type
            if signal.signal_type == SignalType.BUY:
                ret = (exit_price / entry_price) - 1
            else:  # SELL signal
                ret = 1 - (exit_price / entry_price)
                
            forward_returns[symbol] = ret * signal.confidence
        
        if not forward_returns:
            return 0.0
            
        # Average return across all signals
        avg_return = sum(forward_returns.values()) / len(forward_returns)
        
        # Simple Sharpe ratio (no risk-free rate)
        if len(forward_returns) > 1:
            std_dev = np.std(list(forward_returns.values()))
            if std_dev > 0:
                sharpe = avg_return / std_dev
            else:
                sharpe = avg_return if avg_return > 0 else 0
        else:
            sharpe = avg_return if avg_return > 0 else 0
        
        return sharpe
    
    def regime_compatibility(self, regime: MarketRegime) -> float:
        """
        Get compatibility score for this strategy in the given market regime.
        
        Momentum strategies perform differently across various market conditions.
        This method provides a quantitative measure of how well the strategy is
        expected to perform in each market regime.
        
        The compatibility scores reflect empirical evidence that momentum strategies:
        - Perform best in trending markets, especially bull markets
        - Struggle in choppy, range-bound markets
        - Can experience significant drawdowns during regime shifts
        - May underperform during periods of extreme volatility
        
        Args:
            regime: Current market regime classification
            
        Returns:
            Compatibility score (0-1, higher indicates better compatibility)
        """
        # Regime compatibility scores
        compatibility = {
            MarketRegime.BULL_TREND: 0.9,      # Excellent in bull trends
            MarketRegime.BEAR_TREND: 0.3,      # Poor in bear trends
            MarketRegime.CONSOLIDATION: 0.2,   # Poor in sideways markets
            MarketRegime.HIGH_VOLATILITY: 0.3, # Below average in high volatility
            MarketRegime.LOW_VOLATILITY: 0.6,  # Good in low volatility
            MarketRegime.UNKNOWN: 0.5          # Neutral in unknown regime
        }
        
        return compatibility.get(regime, 0.5) 