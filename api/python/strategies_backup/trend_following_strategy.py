#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend-Following Strategy Module

This module implements a trend-following trading strategy that identifies and capitalizes
on persistent directional price movements in financial markets.

The trend-following strategy is based on the empirical observation that financial markets
often exhibit momentum, where price movements tend to persist in the same direction for
extended periods. This approach aims to identify these trends early, ride them during their
duration, and exit when they show signs of reversal.

Key concepts implemented in this strategy:
1. Moving Average Crossovers to identify trend direction and changes
2. Trend strength measurement to filter for significant trends
3. Signal line smoothing to reduce whipsaws and false signals
4. Volatility filtering via Average True Range (ATR)
5. Position sizing and risk management based on market conditions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TrendFollowingStrategy:
    """
    Trend-Following Trading Strategy
    
    This strategy identifies and capitalizes on persistent directional price movements,
    entering positions in the direction of established trends and exiting when the trend
    shows signs of weakening or reversal.
    
    Key characteristics:
    - Identifies trends using moving average crossovers and trend direction
    - Measures trend strength to filter out weak or nascent trends
    - Smooths signals using a signal line to reduce noise and false triggers
    - Incorporates volatility filters to adapt to changing market conditions
    - Implements dynamic position sizing and risk management
    
    Ideal market conditions:
    - Strongly trending markets with sustained directional price movements
    - Markets with clear cyclical patterns or strong fundamental drivers
    - Lower noise environments where trends can develop clearly
    - Liquid markets that allow for efficient entry and exit
    
    Limitations:
    - Underperforms in choppy, range-bound, or sideways markets
    - Can experience significant drawdowns during trend reversals
    - May suffer from late entries and exits due to signal lag
    - Susceptible to whipsaws in volatile markets with frequent direction changes
    """
    
    def __init__(
        self,
        short_ma_period: int = 20,
        long_ma_period: int = 50,
        signal_ma_period: int = 10,
        trend_strength_threshold: float = 0.05,
        use_atr_filter: bool = True,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        name: str = "trend_following"
    ):
        """
        Initialize the trend following strategy with configurable parameters.
        
        Args:
            short_ma_period: Period for short moving average (faster-responding)
            long_ma_period: Period for long moving average (slower, more stable)
            signal_ma_period: Period for signal moving average (smoothing filter)
            trend_strength_threshold: Minimum trend strength to generate signals (as % of price)
            use_atr_filter: Whether to use ATR for volatility filtering
            atr_period: Period for ATR calculation (typically 14-21 days)
            atr_multiplier: Multiplier for ATR (risk sizing factor)
            name: Strategy name for identification and logging
        """
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.signal_ma_period = signal_ma_period
        self.trend_strength_threshold = trend_strength_threshold
        self.use_atr_filter = use_atr_filter
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.name = name
        
        # Performance tracking
        self.signals = {}
        self.performance = {}
        
        logger.info(f"Initialized {self.name} strategy with MA periods: {short_ma_period}/{long_ma_period}")
    
    def calculate_indicators(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate trend following indicators for the given price data.
        
        Computes a suite of technical indicators used to identify trends, their strength,
        and potential entry and exit points. These indicators form the basis for the
        trend-following signal generation logic.
        
        Key indicators calculated:
        - Short-term Moving Average: Responsive to recent price changes
        - Long-term Moving Average: Establishes the baseline trend
        - Trend Direction: Differential between short and long MAs
        - Trend Strength: Normalized trend direction as percentage of price
        - Signal Line: Smoothed trend direction to filter out noise
        - ATR: Volatility measure used for position sizing and filtering
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            
        Returns:
            Dictionary of calculated indicators, organized by indicator type
        """
        indicators = {}
        
        # Calculate moving averages
        indicators['short_ma'] = prices.rolling(self.short_ma_period).mean()
        indicators['long_ma'] = prices.rolling(self.long_ma_period).mean()
        
        # Calculate trend direction (short_ma - long_ma)
        indicators['trend_direction'] = indicators['short_ma'] - indicators['long_ma']
        
        # Calculate trend strength (as percentage of price)
        indicators['trend_strength'] = indicators['trend_direction'] / prices
        
        # Calculate signal line (smoothed trend direction)
        indicators['signal_line'] = indicators['trend_direction'].rolling(self.signal_ma_period).mean()
        
        # Calculate ATR if needed
        if self.use_atr_filter:
            # Calculate daily ranges
            high_low_range = prices.rolling(2).max() - prices.rolling(2).min()
            
            # Simple approximation of ATR using high-low range
            indicators['atr'] = high_low_range.rolling(self.atr_period).mean()
            
            # ATR as percentage of price
            indicators['atr_pct'] = indicators['atr'] / prices
        
        return indicators
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trade signals based on trend following indicators.
        
        Analyzes the calculated indicators to identify high-probability trend-following
        trade opportunities. Signals are generated when trends are confirmed by multiple
        factors and filtered based on trend strength and volatility conditions.
        
        Signal generation logic:
        - LONG signals: When short MA crosses above long MA, trend strength exceeds threshold,
          and trend direction is stronger than the signal line (acceleration)
        - SHORT signals: When short MA crosses below long MA, trend strength exceeds threshold
          in the negative direction, and trend direction is below the signal line
        - Additional filtering based on ATR to avoid trading in extremely low or high
          volatility conditions
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            market_data: Additional market data for contextual analysis (optional)
            
        Returns:
            DataFrame of trade signals (1=buy, -1=sell, 0=neutral) for each asset at each time point
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Calculate indicators
        indicators = self.calculate_indicators(prices)
        
        # Generate signals based on trend direction and strength
        trend_direction = indicators['trend_direction']
        trend_strength = indicators['trend_strength']
        signal_line = indicators['signal_line']
        
        # Long signal: trend direction positive and strong enough
        long_condition = (
            (trend_direction > 0) & 
            (trend_strength > self.trend_strength_threshold) &
            (trend_direction > signal_line)
        )
        signals[long_condition] = 1
        
        # Short signal: trend direction negative and strong enough
        short_condition = (
            (trend_direction < 0) & 
            (trend_strength < -self.trend_strength_threshold) &
            (trend_direction < signal_line)
        )
        signals[short_condition] = -1
        
        # Apply ATR filter if enabled
        if self.use_atr_filter and 'atr_pct' in indicators:
            atr_pct = indicators['atr_pct']
            
            # Only take trades if ATR is reasonable (not too high or too low)
            min_atr_threshold = 0.005  # 0.5% daily range minimum
            max_atr_threshold = 0.03   # 3% daily range maximum
            
            # Filter out signals where volatility is too low or too high
            invalid_atr = (atr_pct < min_atr_threshold) | (atr_pct > max_atr_threshold)
            signals[invalid_atr] = 0
        
        # Store last signals for reference
        self.signals = signals.iloc[-1].to_dict()
        
        return signals
    
    def optimize_parameters(
        self, 
        prices: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on historical performance.
        
        Conducts a systematic search for the best combination of strategy parameters by
        testing multiple configurations against historical price data. The optimization
        evaluates performance using risk-adjusted metrics such as the Sharpe ratio.
        
        Optimization process:
        1. Tests combinations of moving average periods and threshold values
        2. Generates signals for each parameter set using historical data
        3. Simulates trading performance for each configuration
        4. Computes risk-adjusted return metrics (annualized return, volatility, Sharpe ratio)
        5. Selects the parameter set with the highest Sharpe ratio
        
        Args:
            prices: DataFrame of asset prices for backtesting
            market_data: Additional market data for contextual analysis (optional)
            
        Returns:
            Dictionary of optimized parameters and their performance metrics
        """
        # Simple optimization example - test a few parameter combinations
        best_sharpe = -np.inf
        best_params = {}
        
        # Test combinations of moving average periods
        short_ma_options = [10, 20, 30]
        long_ma_options = [50, 100, 200]
        threshold_options = [0.02, 0.05, 0.1]
        
        # Use last 1 year of data for optimization
        test_prices = prices.iloc[-252:]
        
        for short_ma in short_ma_options:
            for long_ma in long_ma_options:
                for threshold in threshold_options:
                    # Skip invalid combinations
                    if short_ma >= long_ma:
                        continue
                    
                    # Create test strategy with these parameters
                    test_strategy = TrendFollowingStrategy(
                        short_ma_period=short_ma,
                        long_ma_period=long_ma,
                        trend_strength_threshold=threshold
                    )
                    
                    # Generate signals
                    signals = test_strategy.generate_signals(test_prices)
                    
                    # Simulate returns (simplified)
                    shifted_signals = signals.shift(1).fillna(0)
                    returns = test_prices.pct_change() * shifted_signals
                    
                    # Calculate performance
                    portfolio_returns = returns.mean(axis=1)
                    
                    annualized_return = portfolio_returns.mean() * 252
                    volatility = portfolio_returns.std() * np.sqrt(252)
                    sharpe = annualized_return / volatility if volatility > 0 else 0
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            "short_ma_period": short_ma,
                            "long_ma_period": long_ma,
                            "trend_strength_threshold": threshold,
                            "sharpe_ratio": sharpe,
                            "annualized_return": annualized_return,
                            "volatility": volatility
                        }
        
        logger.info(f"Optimized {self.name} strategy parameters: {best_params}")
        return best_params
    
    def update_performance(
        self, 
        prices: pd.DataFrame, 
        signals: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Update strategy performance metrics based on historical or current data.
        
        Calculates a comprehensive set of performance metrics to evaluate the strategy's
        effectiveness and risk characteristics. These metrics can be used for strategy
        comparison, monitoring, and reporting.
        
        Performance metrics include:
        - Total return: Cumulative return over the entire period
        - Annualized return: Return normalized to a yearly basis
        - Volatility: Standard deviation of returns (annualized)
        - Sharpe ratio: Risk-adjusted return measure
        - Maximum drawdown: Largest peak-to-trough decline
        - Win rate: Percentage of positive return periods
        
        Args:
            prices: DataFrame of asset prices
            signals: Signals used (if None, generates new signals)
            
        Returns:
            Dictionary of performance metrics with standardized keys
        """
        if signals is None:
            signals = self.generate_signals(prices)
        
        # Simulate returns (signals are applied to next day's returns)
        shifted_signals = signals.shift(1).fillna(0)
        asset_returns = prices.pct_change()
        strategy_returns = asset_returns * shifted_signals
        
        # Calculate portfolio returns (equal weight across assets with signals)
        portfolio_returns = strategy_returns.mean(axis=1)
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = (portfolio_returns > 0).mean()
        
        # Store performance metrics
        self.performance = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "last_updated": datetime.now().isoformat()
        }
        
        return self.performance
    
    def regime_compatibility(self, regime: str) -> float:
        """
        Get compatibility score for this strategy in the given market regime.
        
        Trend-following strategies perform differently across various market conditions.
        This method provides a quantitative measure of how well the strategy is expected
        to perform in each market regime.
        
        The compatibility scores reflect empirical evidence that trend-following strategies:
        - Perform best in strong directional markets (bull or bear trends)
        - Struggle in choppy, sideways, or mean-reverting markets
        - Can adapt to different volatility environments with appropriate filters
        - May underperform during rapid regime shifts or market reversals
        
        Args:
            regime: Market regime classification string
            
        Returns:
            Compatibility score (0-2, higher indicates better compatibility)
        """
        # Regime compatibility scores
        compatibility = {
            "bull": 1.5,     # Very good in bullish markets
            "bear": 1.2,     # Good in bearish markets
            "sideways": 0.5, # Poor in sideways markets
            "high_vol": 0.7, # Below average in high volatility
            "low_vol": 1.0,  # Average in low volatility
            "crisis": 1.0,   # Average in crisis (can adapt)
            "unknown": 1.0   # Neutral in unknown regime
        }
        
        return compatibility.get(regime.lower(), 1.0)
    
    def get_current_signals(self) -> Dict[str, int]:
        """
        Get the most recent trading signals for all assets.
        
        Returns:
            Dictionary mapping asset symbols to their current signal values
            (1=buy, -1=sell, 0=neutral)
        """
        return self.signals
    
    def get_performance(self) -> Dict[str, float]:
        """
        Get current performance metrics for the strategy.
        
        Returns:
            Dictionary of performance metrics including returns, volatility,
            Sharpe ratio, and maximum drawdown
        """
        return self.performance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to a dictionary representation for serialization.
        
        Creates a complete representation of the strategy including all parameters,
        current signals, and performance metrics, suitable for storage, 
        transmission, or reconstruction.
        
        Returns:
            Dictionary representation of the strategy with all relevant attributes
        """
        return {
            "name": self.name,
            "type": "trend_following",
            "short_ma_period": self.short_ma_period,
            "long_ma_period": self.long_ma_period,
            "signal_ma_period": self.signal_ma_period,
            "trend_strength_threshold": self.trend_strength_threshold,
            "use_atr_filter": self.use_atr_filter,
            "performance": self.performance,
            "signals": self.signals
        } 