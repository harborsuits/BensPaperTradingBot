#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Strategy - Captures continued price movement by buying assets that have
shown strong recent performance and selling those that have underperformed.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MomentumStrategy:
    """
    Momentum trading strategy implementation that identifies and trades based on
    price momentum and trend strength.
    """
    
    def __init__(self, lookback_period: int = 14, overbought: int = 70, oversold: int = 30):
        """
        Initialize the momentum strategy with configurable parameters.
        
        Args:
            lookback_period: Period for calculating momentum indicators
            overbought: RSI threshold for overbought condition
            oversold: RSI threshold for oversold condition
        """
        self.name = "Momentum"
        self.lookback_period = lookback_period
        self.overbought = overbought
        self.oversold = oversold
        self.description = "Trades based on price momentum and trend strength"
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trade signals based on momentum indicators.
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with added momentum indicators and trade signals
        """
        if len(data) < self.lookback_period:
            return pd.DataFrame()
        
        # Calculate price momentum (close price change over lookback period)
        data = data.copy()
        data['momentum'] = data['close'].pct_change(self.lookback_period)
        
        # Calculate Rate of Change (ROC)
        data['roc'] = (data['close'] / data['close'].shift(self.lookback_period) - 1) * 100
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.lookback_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.lookback_period).mean()
        
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Average Directional Index (ADX) for trend strength
        high_diff = data['high'].diff()
        low_diff = data['low'].diff().abs()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.DataFrame({
            'hl': data['high'] - data['low'],
            'hc': (data['high'] - data['close'].shift()).abs(),
            'lc': (data['low'] - data['close'].shift()).abs()
        }).max(axis=1)
        
        atr = tr.rolling(window=self.lookback_period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=self.lookback_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.lookback_period).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        data['adx'] = dx.rolling(window=self.lookback_period).mean()
        
        # Generate signals
        data['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Buy conditions:
        # 1. Strong upward momentum (positive ROC)
        # 2. RSI was oversold but is now increasing
        # 3. ADX indicates strong trend (> 25)
        data.loc[(data['roc'] > 0) & 
                 (data['rsi'] > self.oversold) & 
                 (data['rsi'].shift(1) <= self.oversold) &
                 (data['adx'] > 25), 'signal'] = 1
        
        # Sell conditions:
        # 1. Momentum turns negative
        # 2. RSI reaches overbought territory
        # 3. Price momentum weakening
        data.loc[(data['roc'] < 0) | 
                 (data['rsi'] >= self.overbought) |
                 ((data['momentum'].shift(1) > data['momentum']) & 
                  (data['momentum'] > 0) & 
                  (data['rsi'] > 60)), 'signal'] = -1
        
        return data
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters"""
        return {
            "lookback_period": self.lookback_period,
            "overbought": self.overbought,
            "oversold": self.oversold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set strategy parameters"""
        if 'lookback_period' in params:
            self.lookback_period = params['lookback_period']
        if 'overbought' in params:
            self.overbought = params['overbought']
        if 'oversold' in params:
            self.oversold = params['oversold']
            
    def optimize(self, data: pd.DataFrame, 
                 param_grid: Optional[Dict[str, List[Any]]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize strategy parameters based on historical data.
        
        Args:
            data: Historical price data
            param_grid: Dictionary of parameter names and possible values
            
        Returns:
            Tuple of (best parameters, performance metrics)
        """
        if param_grid is None:
            param_grid = {
                'lookback_period': [5, 10, 14, 20, 30],
                'overbought': [65, 70, 75, 80],
                'oversold': [20, 25, 30, 35]
            }
        
        best_params = {}
        best_performance = {
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'win_rate': 0,
            'max_drawdown': 100
        }
        
        # Grid search through parameters
        # In a real implementation, this would be more sophisticated
        for lookback in param_grid['lookback_period']:
            for overbought in param_grid['overbought']:
                for oversold in param_grid['oversold']:
                    # Skip invalid combinations
                    if oversold >= overbought:
                        continue
                        
                    # Set parameters and generate signals
                    self.set_parameters({
                        'lookback_period': lookback,
                        'overbought': overbought,
                        'oversold': oversold
                    })
                    
                    result = self.generate_signals(data)
                    metrics = self._calculate_performance(result)
                    
                    # Update best parameters if performance is better
                    if metrics['sharpe_ratio'] > best_performance['sharpe_ratio']:
                        best_performance = metrics
                        best_params = self.get_parameters()
        
        return best_params, best_performance
    
    def _calculate_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the strategy"""
        if 'signal' not in data.columns or len(data) == 0:
            return {
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'win_rate': 0,
                'max_drawdown': 100
            }
        
        # Calculate daily returns based on signals
        data['position'] = data['signal'].shift(1).fillna(0)
        data['returns'] = data['close'].pct_change() * data['position']
        
        # Calculate metrics
        total_return = data['returns'].sum()
        volatility = data['returns'].std() * np.sqrt(252)  # Annualized
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        
        # Calculate win rate and profit factor
        winning_trades = data[data['returns'] > 0]['returns'].sum()
        losing_trades = abs(data[data['returns'] < 0]['returns'].sum())
        win_count = len(data[data['returns'] > 0])
        total_trades = len(data[data['returns'] != 0])
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        profit_factor = winning_trades / losing_trades if losing_trades > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + data['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = abs(drawdown.min()) * 100
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }

    def calculate_momentum(
        self, 
        prices: pd.DataFrame, 
        lookback: int = None
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators for given price data.
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            lookback: Lookback period (uses all periods in self.lookback_periods if None)
            
        Returns:
            DataFrame of momentum indicators
        """
        if lookback is None:
            # Calculate for all lookback periods and average
            momentum_indicators = []
            
            for period in self.lookback_periods:
                momentum = prices.pct_change(period)
                momentum_indicators.append(momentum)
            
            # Average across all lookback periods
            momentum = pd.concat(momentum_indicators).groupby(level=0).mean()
        else:
            # Calculate for specific lookback period
            momentum = prices.pct_change(lookback)
        
        if self.volatility_adjust and momentum.shape[0] > self.volatility_lookback:
            # Adjust momentum by volatility
            volatility = prices.pct_change().rolling(self.volatility_lookback).std() * np.sqrt(252)
            # Add small constant to avoid division by zero
            momentum = momentum / (volatility + 1e-8)
        
        if self.cross_sectional:
            # Cross-sectional momentum (rank assets relative to each other)
            momentum = momentum.rank(axis=1, pct=True) - 0.5
        
        return momentum
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trade signals based on momentum indicators.
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            market_data: Additional market data (optional)
            
        Returns:
            DataFrame of trade signals (1=buy, -1=sell, 0=neutral)
        """
        # Calculate momentum indicators
        momentum = self.calculate_momentum(prices)
        
        # Generate signals based on momentum and threshold
        signals = pd.DataFrame(0, index=momentum.index, columns=momentum.columns)
        
        # Long positions for momentum > threshold
        signals[momentum > self.signal_threshold] = 1
        
        # Short positions for momentum < -threshold (if threshold is positive)
        if self.signal_threshold > 0:
            signals[momentum < -self.signal_threshold] = -1
        else:
            # If threshold is 0 or negative, short the bottom half
            signals[momentum < 0] = -1
        
        # Store signals for performance tracking
        self.signals = signals.iloc[-1].to_dict()
        
        return signals
    
    def optimize_parameters(
        self, 
        prices: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on historical performance.
        
        Args:
            prices: DataFrame of asset prices
            market_data: Additional market data (optional)
            
        Returns:
            Dict of optimized parameters
        """
        # Simple optimization example - test a few parameter combinations
        best_sharpe = -np.inf
        best_params = {}
        
        # Test combinations of lookback periods and thresholds
        lookback_options = [[20, 60, 120], [5, 20, 60], [10, 30, 90]]
        threshold_options = [0.0, 0.1, 0.2]
        volatility_adjust_options = [True, False]
        
        # Use last 1 year of data for optimization
        test_prices = prices.iloc[-252:]
        
        for lookback in lookback_options:
            for threshold in threshold_options:
                for vol_adjust in volatility_adjust_options:
                    # Create test strategy with these parameters
                    test_strategy = MomentumStrategy(
                        lookback_periods=lookback,
                        signal_threshold=threshold,
                        volatility_adjust=vol_adjust
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
                            "lookback_periods": lookback,
                            "signal_threshold": threshold,
                            "volatility_adjust": vol_adjust,
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
        Update strategy performance metrics.
        
        Args:
            prices: DataFrame of asset prices
            signals: Signals used (if None, generates new signals)
            
        Returns:
            Dict of performance metrics
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
        
        Args:
            regime: Market regime
            
        Returns:
            Compatibility score (0-2, higher is better)
        """
        # Regime compatibility scores
        compatibility = {
            "bull": 1.8,     # Excellent in bullish markets
            "bear": 0.5,     # Poor in bearish markets
            "sideways": 0.4, # Poor in sideways markets
            "high_vol": 0.6, # Below average in high volatility
            "low_vol": 1.2,  # Good in low volatility
            "crisis": 0.2,   # Very poor in crisis
            "unknown": 1.0   # Neutral in unknown regime
        }
        
        return compatibility.get(regime.lower(), 1.0)
    
    def get_current_signals(self) -> Dict[str, int]:
        """
        Get the most recent signals.
        
        Returns:
            Dict of assets and their signals
        """
        return self.signals
    
    def get_performance(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dict of performance metrics
        """
        return self.performance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dict representation.
        
        Returns:
            Dict representation of strategy
        """
        return {
            "name": self.name,
            "type": "momentum",
            "lookback_periods": self.lookback_period,
            "signal_threshold": self.signal_threshold,
            "volatility_adjust": self.volatility_adjust,
            "cross_sectional": self.cross_sectional,
            "performance": self.performance,
            "signals": self.signals
        } 