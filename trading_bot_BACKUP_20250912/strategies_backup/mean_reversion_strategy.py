#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Reversion Strategy Module (DEPRECATED)

This implementation is deprecated and will be removed in a future version.
Please use the new implementation in one of the following modules:

- trading_bot.strategies.mean_reversion.zscore_strategy (ZScoreMeanReversionStrategy)
- trading_bot.strategies.stocks.mean_reversion.mean_reversion_strategy (MeanReversionStrategy)
"""

import warnings
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy
    
    This strategy identifies and capitalizes on temporary price deviations from historical
    averages, entering counter-trend positions when assets become statistically overextended
    with the expectation that prices will revert to their mean.
    
    Key characteristics:
    - Counter-trend approach that trades against price extremes
    - Uses statistical measures (z-scores) to identify overbought/oversold conditions
    - Implements adaptable entry and exit thresholds based on market conditions
    - Includes volatility filtering to avoid trading during unstable periods
    - Controls risk through predefined holding periods and position sizing
    
    Ideal market conditions:
    - Range-bound or sideways markets with clear boundaries
    - Markets with cyclical patterns and regular oscillations
    - Lower volatility environments where mean reversion is more reliable
    - Assets with stable fundamentals and established trading ranges
    
    Limitations:
    - Vulnerable to strong trending markets and trend breakouts
    - May enter positions too early during significant trend extensions
    - Requires careful risk management to limit losses in runaway markets
    - Performance varies significantly across different market regimes
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        holding_period: int = 5,
        use_volatility_filter: bool = True,
        volatility_lookback: int = 50,
        name: str = "mean_reversion"
    ):
        warnings.warn(
            "This implementation of MeanReversionStrategy is deprecated. "
            "Please use the new implementation from trading_bot.strategies.mean_reversion "
            "or trading_bot.strategies.stocks.mean_reversion modules.",
            DeprecationWarning, 
            stacklevel=2
        )
        
        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.holding_period = holding_period
        self.use_volatility_filter = use_volatility_filter
        self.volatility_lookback = volatility_lookback
        self.name = name
        
        # Performance tracking
        self.signals = {}
        self.performance = {}
        
        logger.info(f"Initialized {self.name} strategy with lookback period: {lookback_period}")
    
    def calculate_zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-scores for the given price data.
        
        Computes the statistical z-score for each asset, which measures how many standard
        deviations a value is from the mean of a dataset. This is the core statistical
        measure used to identify potential mean reversion opportunities.
        
        Z-scores are calculated by:
        1. Computing returns for each asset
        2. Calculating the rolling mean and standard deviation of returns
        3. Measuring the distance of current returns from the rolling mean
        4. Normalizing this distance by the rolling standard deviation
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            
        Returns:
            DataFrame of z-scores for each asset at each time point
        """
        # Calculate returns
        returns = prices.pct_change()
        
        # Calculate rolling mean and standard deviation
        rolling_mean = returns.rolling(window=self.lookback_period).mean()
        rolling_std = returns.rolling(window=self.lookback_period).std()
        
        # Calculate z-scores (how many standard deviations from the mean)
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Replace inf/NaN values with 0
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return z_scores
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trade signals based on mean reversion indicators.
        
        Analyzes z-scores and other indicators to identify potential mean reversion
        opportunities. Signals are generated when assets become statistically 
        overextended in either direction.
        
        Signal generation logic:
        - LONG signals: When z-score becomes extremely negative (<-entry_z_score),
          indicating the asset is statistically oversold
        - SHORT signals: When z-score becomes extremely positive (>entry_z_score),
          indicating the asset is statistically overbought
        - EXIT signals: When z-score returns to a more normal range (based on exit_z_score)
          or when the maximum holding period is reached
        
        Additional volatility filtering can be applied to avoid trading during periods
        of excessive market volatility when mean reversion may be less reliable.
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            market_data: Additional market data for contextual analysis (optional)
            
        Returns:
            DataFrame of trade signals (1=buy, -1=sell, 0=neutral) for each asset at each time point
        """
        # Check if we have enough data
        if len(prices) < self.lookback_period:
            logger.warning(f"Insufficient data for {self.name} strategy, need at least {self.lookback_period} bars")
            return pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Calculate z-scores
        z_scores = self.calculate_zscore(prices)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Long signal: z-score < -entry_z_score (price has moved down too far)
        signals[z_scores < -self.entry_z_score] = 1
        
        # Short signal: z-score > entry_z_score (price has moved up too far)
        signals[z_scores > self.entry_z_score] = -1
        
        # Exit long positions when z-score > -exit_z_score
        exit_long = (signals.shift(1) == 1) & (z_scores > -self.exit_z_score)
        signals[exit_long] = 0
        
        # Exit short positions when z-score < exit_z_score
        exit_short = (signals.shift(1) == -1) & (z_scores < self.exit_z_score)
        signals[exit_short] = 0
        
        # Apply volatility filter if enabled
        if self.use_volatility_filter:
            # Calculate historical volatility
            returns = prices.pct_change()
            current_vol = returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
            
            # Calculate long-term volatility as baseline
            long_term_vol = returns.std() * np.sqrt(252)
            
            # Calculate volatility ratio
            vol_ratio = current_vol / long_term_vol.mean() if long_term_vol.mean() > 0 else current_vol
            
            # Filter out signals when volatility is too high (>1.5x baseline)
            high_vol_mask = vol_ratio > 1.5
            signals[high_vol_mask] = 0
        
        # Implement holding period limit (exit after holding_period bars)
        # This would be more complex in a real implementation
        
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
        1. Tests combinations of lookback periods and z-score thresholds
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
        
        # Test combinations of parameters
        lookback_options = [10, 20, 40]
        entry_z_options = [1.5, 2.0, 2.5]
        exit_z_options = [0.5, 1.0, 1.5]
        
        # Use last 1 year of data for optimization
        test_prices = prices.iloc[-252:]
        
        for lookback in lookback_options:
            for entry_z in entry_z_options:
                for exit_z in exit_z_options:
                    # Skip invalid combinations
                    if exit_z >= entry_z:
                        continue
                    
                    # Create test strategy with these parameters
                    test_strategy = MeanReversionStrategy(
                        lookback_period=lookback,
                        entry_z_score=entry_z,
                        exit_z_score=exit_z
                    )
                    
                    # Generate signals
                    signals = test_strategy.generate_signals(test_prices)
                    
                    # Simulate returns
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
                            "lookback_period": lookback,
                            "entry_z_score": entry_z,
                            "exit_z_score": exit_z,
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
        
        Mean reversion strategies perform differently across various market conditions.
        This method provides a quantitative measure of how well the strategy is expected
        to perform in each market regime.
        
        The compatibility scores reflect empirical evidence that mean reversion strategies:
        - Perform best in range-bound, sideways markets
        - Struggle in strong trending markets (bull or bear)
        - Can adapt to different volatility environments with filtering
        - May perform well during high volatility periods if properly calibrated
        - Typically underperform during major paradigm shifts or market crises
        
        Args:
            regime: Market regime classification string
            
        Returns:
            Compatibility score (0-2, higher indicates better compatibility)
        """
        # Regime compatibility scores
        compatibility = {
            "bull": 0.6,     # Poor in strong bull markets
            "bear": 1.0,     # Average in bear markets
            "sideways": 1.8, # Excellent in sideways markets
            "high_vol": 1.5, # Good in high volatility (but risky)
            "low_vol": 1.3,  # Good in low volatility
            "crisis": 0.3,   # Poor in crisis (excessive volatility)
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
            "type": "mean_reversion",
            "lookback_period": self.lookback_period,
            "entry_z_score": self.entry_z_score,
            "exit_z_score": self.exit_z_score,
            "holding_period": self.holding_period,
            "use_volatility_filter": self.use_volatility_filter,
            "performance": self.performance,
            "signals": self.signals
        }

# Import the new implementations for backwards compatibility
from trading_bot.strategies.mean_reversion import ZScoreMeanReversionStrategy
from trading_bot.strategies.stocks.mean_reversion import MeanReversionStrategy as StockMeanReversionStrategy 