"""
Market regime detection for EvoTrader.

This module identifies different market conditions (trending, ranging, volatile)
to help strategies adapt their parameters dynamically to the current market state.
Inspired by FreqTrade's adaptive modeling capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class MarketRegimeType(Enum):
    """Enumeration of market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    Detects current market regime based on price action and volatility.
    
    This class uses various technical indicators and statistical methods to
    identify the current market regime, allowing strategies to adapt their
    parameters accordingly.
    """
    
    def __init__(self, 
                 lookback_periods: int = 100,
                 volatility_window: int = 20,
                 trend_threshold: float = 0.3,
                 volatility_threshold: float = 1.5,
                 range_threshold: float = 0.2,
                 breakout_threshold: float = 2.0):
        """
        Initialize the market regime detector.
        
        Args:
            lookback_periods: Number of periods to use for regime detection
            volatility_window: Window size for volatility calculations
            trend_threshold: Threshold for trend identification (correlation)
            volatility_threshold: Threshold for volatile market detection
            range_threshold: Threshold for ranging market detection
            breakout_threshold: Threshold for breakout detection
        """
        self.lookback_periods = lookback_periods
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.range_threshold = range_threshold
        self.breakout_threshold = breakout_threshold
        
        # Track regimes over time
        self.regime_history: Dict[str, List[Tuple[int, MarketRegimeType]]] = defaultdict(list)
        self.regime_counts: Dict[str, Dict[MarketRegimeType, int]] = defaultdict(lambda: defaultdict(int))
    
    def detect_regime(self, 
                      symbol: str, 
                      prices: List[float],
                      highs: Optional[List[float]] = None,
                      lows: Optional[List[float]] = None,
                      volumes: Optional[List[float]] = None,
                      day_index: Optional[int] = None) -> MarketRegimeType:
        """
        Identify the market regime for a given symbol and price series.
        
        Args:
            symbol: Trading symbol
            prices: List of closing prices
            highs: Optional list of high prices
            lows: Optional list of low prices
            volumes: Optional list of volumes
            day_index: Optional current day index for record keeping
            
        Returns:
            Detected market regime type
        """
        # Ensure sufficient data
        if len(prices) < self.lookback_periods:
            logger.debug(f"Insufficient data for regime detection: {len(prices)} < {self.lookback_periods}")
            return MarketRegimeType.UNKNOWN
        
        # Use only the most recent lookback_periods
        prices = prices[-self.lookback_periods:]
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Measure trend strength using linear regression
        regime = self._measure_trend_strength(prices, returns)
        
        # Check for volatility if we have highs and lows
        if highs and lows:
            highs = highs[-self.lookback_periods:]
            lows = lows[-self.lookback_periods:]
            
            # Measure volatility
            volatility_regime = self._measure_volatility(highs, lows, prices)
            
            # If the market is very volatile, override the trend regime
            if volatility_regime == MarketRegimeType.VOLATILE:
                regime = volatility_regime
            # Check for breakouts
            elif volatility_regime == MarketRegimeType.BREAKOUT:
                regime = volatility_regime
        
        # Check for ranging market
        if regime not in [MarketRegimeType.VOLATILE, MarketRegimeType.BREAKOUT]:
            range_regime = self._check_for_range(prices)
            if range_regime == MarketRegimeType.RANGING:
                regime = range_regime
        
        # Record regime
        if day_index is not None:
            self.regime_history[symbol].append((day_index, regime))
            self.regime_counts[symbol][regime] += 1
        
        return regime
    
    def _measure_trend_strength(self, 
                              prices: List[float], 
                              returns: List[float]) -> MarketRegimeType:
        """
        Measure trend strength using linear regression and return correlation.
        
        Args:
            prices: List of prices
            returns: List of price returns
            
        Returns:
            Market regime type based on trend strength
        """
        # Create a time axis
        time_axis = np.arange(len(prices))
        
        # Calculate linear regression
        slope, intercept = np.polyfit(time_axis, prices, 1)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(time_axis, prices)[0, 1]
        
        # Calculate average return
        avg_return = np.mean(returns)
        
        # Determine trend direction and strength
        if correlation > self.trend_threshold and avg_return > 0:
            return MarketRegimeType.TRENDING_UP
        elif correlation < -self.trend_threshold and avg_return < 0:
            return MarketRegimeType.TRENDING_DOWN
        else:
            return MarketRegimeType.NORMAL
    
    def _measure_volatility(self, 
                          highs: List[float], 
                          lows: List[float], 
                          closes: List[float]) -> MarketRegimeType:
        """
        Measure market volatility using high-low ranges.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Market regime type based on volatility
        """
        # Calculate true range
        tr_values = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr = max(tr1, tr2, tr3)
            tr_values.append(tr)
        
        # Calculate average true range for recent window
        recent_tr = tr_values[-self.volatility_window:]
        if not recent_tr:
            return MarketRegimeType.NORMAL
            
        atr = np.mean(recent_tr)
        
        # Calculate historical ATR for baseline
        if len(tr_values) > self.volatility_window * 2:
            historical_tr = tr_values[-(self.volatility_window * 2):-self.volatility_window]
            historical_atr = np.mean(historical_tr)
            
            # Volatility ratio
            volatility_ratio = atr / max(0.0001, historical_atr)
            
            # Check for high volatility
            if volatility_ratio > self.volatility_threshold:
                return MarketRegimeType.VOLATILE
                
            # Check for breakout
            if volatility_ratio > self.breakout_threshold:
                return MarketRegimeType.BREAKOUT
        
        return MarketRegimeType.NORMAL
    
    def _check_for_range(self, prices: List[float]) -> MarketRegimeType:
        """
        Check if the market is in a range.
        
        Args:
            prices: List of prices
            
        Returns:
            Market regime type based on range analysis
        """
        # Calculate the percent difference between max and min prices
        price_range = max(prices) - min(prices)
        avg_price = np.mean(prices)
        range_percent = price_range / avg_price
        
        # Check if the range is narrow
        if range_percent < self.range_threshold:
            return MarketRegimeType.RANGING
        
        return MarketRegimeType.NORMAL
    
    def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """
        Get statistics about regimes for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of regime statistics
        """
        if symbol not in self.regime_counts:
            return {}
        
        total_counts = sum(self.regime_counts[symbol].values())
        
        # Calculate percentage for each regime
        regime_percentages = {
            regime.value: (count / total_counts) * 100
            for regime, count in self.regime_counts[symbol].items()
        }
        
        # Calculate regime transitions
        regime_transitions = self._calculate_regime_transitions(symbol)
        
        return {
            'total_observations': total_counts,
            'regime_counts': {r.value: c for r, c in self.regime_counts[symbol].items()},
            'regime_percentages': regime_percentages,
            'regime_transitions': regime_transitions
        }
    
    def _calculate_regime_transitions(self, symbol: str) -> Dict[str, int]:
        """
        Calculate transitions between regimes.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of regime transition counts
        """
        transitions = defaultdict(int)
        
        history = self.regime_history.get(symbol, [])
        if len(history) < 2:
            return {}
        
        # Count transitions
        for i in range(1, len(history)):
            prev_regime = history[i-1][1]
            curr_regime = history[i][1]
            
            if prev_regime != curr_regime:
                transition_key = f"{prev_regime.value}_to_{curr_regime.value}"
                transitions[transition_key] += 1
        
        return dict(transitions)
    
    def get_optimal_parameters(self, 
                             regime: MarketRegimeType, 
                             strategy_type: str) -> Dict[str, Any]:
        """
        Get optimal strategy parameters for the current market regime.
        
        Args:
            regime: Current market regime
            strategy_type: Type of strategy
            
        Returns:
            Dictionary of optimal parameters
        """
        # This would normally be based on backtest results or machine learning
        # Here we'll use a simple heuristic approach
        
        if strategy_type == "trend_following":
            # Trend-following strategies
            if regime == MarketRegimeType.TRENDING_UP:
                return {
                    'fast_ma_period': 8,        # More responsive
                    'medium_ma_period': 21,
                    'slow_ma_period': 50,
                    'stop_loss_pct': 3.0,
                    'take_profit_pct': 5.0,
                    'risk_percent': 2.0
                }
            elif regime == MarketRegimeType.TRENDING_DOWN:
                return {
                    'fast_ma_period': 8,
                    'medium_ma_period': 21,
                    'slow_ma_period': 50,
                    'stop_loss_pct': 2.0,        # Tighter stop loss
                    'take_profit_pct': 4.0,
                    'risk_percent': 1.5          # More conservative
                }
            elif regime == MarketRegimeType.RANGING:
                return {
                    'fast_ma_period': 12,        # Slower to avoid whipsaws
                    'medium_ma_period': 30,
                    'slow_ma_period': 60,
                    'stop_loss_pct': 2.5,
                    'take_profit_pct': 2.5,      # Lower profit target
                    'risk_percent': 1.0          # More conservative
                }
            elif regime == MarketRegimeType.VOLATILE:
                return {
                    'fast_ma_period': 13,        # Slower to filter volatility
                    'medium_ma_period': 33,
                    'slow_ma_period': 65,
                    'stop_loss_pct': 4.0,        # Wider stop loss
                    'take_profit_pct': 6.0,
                    'risk_percent': 1.0          # More conservative
                }
            elif regime == MarketRegimeType.BREAKOUT:
                return {
                    'fast_ma_period': 5,         # More responsive
                    'medium_ma_period': 15,
                    'slow_ma_period': 40,
                    'stop_loss_pct': 3.5,
                    'take_profit_pct': 7.0,      # Higher profit target
                    'risk_percent': 2.5          # More aggressive
                }
        
        elif strategy_type == "mean_reversion":
            # Mean-reversion strategies (like RSI)
            if regime == MarketRegimeType.TRENDING_UP:
                return {
                    'rsi_period': 21,            # Longer to avoid premature reversals
                    'oversold_threshold': 25,    # More extreme
                    'overbought_threshold': 75,
                    'stop_loss_pct': 3.0,
                    'take_profit_pct': 2.5,
                    'risk_percent': 1.5
                }
            elif regime == MarketRegimeType.TRENDING_DOWN:
                return {
                    'rsi_period': 21,
                    'oversold_threshold': 25,
                    'overbought_threshold': 75,
                    'stop_loss_pct': 2.5,
                    'take_profit_pct': 2.0,
                    'risk_percent': 1.0
                }
            elif regime == MarketRegimeType.RANGING:
                return {
                    'rsi_period': 14,            # Standard settings work well in ranges
                    'oversold_threshold': 30,
                    'overbought_threshold': 70,
                    'stop_loss_pct': 2.0,
                    'take_profit_pct': 2.0,
                    'risk_percent': 2.0
                }
            elif regime == MarketRegimeType.VOLATILE:
                return {
                    'rsi_period': 21,            # Longer to filter volatility
                    'oversold_threshold': 20,    # More extreme
                    'overbought_threshold': 80,
                    'stop_loss_pct': 4.0,        # Wider stop loss
                    'take_profit_pct': 3.0,
                    'risk_percent': 1.0
                }
            elif regime == MarketRegimeType.BREAKOUT:
                return {
                    'rsi_period': 21,            # Don't trade counter-trend on breakouts
                    'oversold_threshold': 15,    # Very extreme values only
                    'overbought_threshold': 85,
                    'stop_loss_pct': 3.5,
                    'take_profit_pct': 2.5,
                    'risk_percent': 1.0
                }
        
        # Default parameters if no match
        return {
            'fast_ma_period': 10,
            'medium_ma_period': 25,
            'slow_ma_period': 50,
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'stop_loss_pct': 2.5,
            'take_profit_pct': 3.0,
            'risk_percent': 1.5
        }


class AdaptiveStrategy:
    """
    Mixin class for strategies to adapt to market regimes.
    
    This class provides methods for strategies to adjust their
    parameters based on detected market regimes.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the adaptive strategy mixin."""
        # Call the parent's __init__ if it exists
        super().__init__(*args, **kwargs)
        
        # Create market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Track current regimes
        self.current_regimes = {}
        
        # Track regime-specific performance
        self.regime_performance = {}
        
        # Adaptive parameters
        self.enable_adaptation = True
        self.adaptation_lookback = 100
    
    def detect_market_regime(self, 
                            symbol: str, 
                            market_data: Dict[str, Any],
                            current_day: int) -> MarketRegimeType:
        """
        Detect the market regime for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol
            current_day: Current simulation day
            
        Returns:
            Detected market regime
        """
        # Extract price data
        prices = market_data.get('close', [])
        if not prices and 'price' in market_data:
            # Some data providers might use 'price' instead of 'close'
            prices = market_data.get('price', [])
        
        # Extract highs and lows if available
        highs = market_data.get('high', None)
        lows = market_data.get('low', None)
        
        # Extract volumes if available
        volumes = market_data.get('volume', None)
        
        # Detect regime
        regime = self.regime_detector.detect_regime(
            symbol, 
            prices, 
            highs, 
            lows, 
            volumes, 
            current_day
        )
        
        # Update current regime
        self.current_regimes[symbol] = regime
        
        return regime
    
    def adapt_parameters(self, 
                        symbol: str, 
                        market_data: Dict[str, Any],
                        current_day: int) -> None:
        """
        Adapt strategy parameters based on market regime.
        
        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol
            current_day: Current simulation day
        """
        if not self.enable_adaptation:
            return
            
        # Detect regime
        regime = self.detect_market_regime(symbol, market_data, current_day)
        
        # Get optimal parameters for this regime and strategy type
        strategy_type = getattr(self, 'strategy_type', self.__class__.__name__)
        optimal_params = self.regime_detector.get_optimal_parameters(regime, strategy_type)
        
        # Apply parameters if the strategy has parameters attribute
        if hasattr(self, 'parameters'):
            # Only update parameters that exist in the strategy
            for param, value in optimal_params.items():
                if param in self.parameters:
                    # Update parameter
                    self.parameters[param] = value
        
        # Apply parameters to indicators if the strategy has indicators
        if hasattr(self, 'indicators') and symbol in self.indicators:
            for ind_name, indicator in self.indicators[symbol].items():
                # Apply relevant parameters to each indicator
                if hasattr(indicator, 'params'):
                    for param, value in optimal_params.items():
                        # Match parameters by name (e.g., period, overbought_threshold)
                        if param in indicator.params:
                            indicator.params[param] = value
                        # Handle special cases like rsi_period -> period for RSI indicator
                        elif ind_name.lower() in param.lower() and param.endswith('_period'):
                            if 'period' in indicator.params:
                                indicator.params['period'] = value
    
    def update_regime_performance(self, 
                                 symbol: str, 
                                 performance_data: Dict[str, float]) -> None:
        """
        Update performance tracking by market regime.
        
        Args:
            symbol: Trading symbol
            performance_data: Dictionary of performance metrics
        """
        # Skip if no regime information
        if symbol not in self.current_regimes:
            return
            
        regime = self.current_regimes[symbol]
        
        # Initialize regime performance tracking if needed
        if symbol not in self.regime_performance:
            self.regime_performance[symbol] = {}
            
        if regime not in self.regime_performance[symbol]:
            self.regime_performance[symbol][regime] = []
        
        # Add performance data
        self.regime_performance[symbol][regime].append(performance_data)
        
        # Trim to keep only the most recent data
        if len(self.regime_performance[symbol][regime]) > self.adaptation_lookback:
            self.regime_performance[symbol][regime] = self.regime_performance[symbol][regime][-self.adaptation_lookback:]
    
    def get_regime_performance_stats(self, 
                                   symbol: str, 
                                   regime: Optional[MarketRegimeType] = None) -> Dict[str, Any]:
        """
        Get performance statistics by market regime.
        
        Args:
            symbol: Trading symbol
            regime: Optional specific regime to get stats for
            
        Returns:
            Dictionary of performance statistics
        """
        # Skip if no performance data
        if symbol not in self.regime_performance:
            return {}
            
        # If regime is specified, return stats for that regime only
        if regime and regime in self.regime_performance[symbol]:
            return self._calculate_performance_stats(self.regime_performance[symbol][regime])
        
        # Otherwise return stats for all regimes
        stats = {}
        for r, performance_list in self.regime_performance[symbol].items():
            stats[r.value] = self._calculate_performance_stats(performance_list)
            
        return stats
    
    def _calculate_performance_stats(self, 
                                   performance_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate summary statistics from a list of performance metrics.
        
        Args:
            performance_list: List of performance dictionaries
            
        Returns:
            Dictionary of summary statistics
        """
        if not performance_list:
            return {}
            
        # Extract key metrics
        profits = [p.get('profit', 0) for p in performance_list]
        win_rates = [p.get('win_rate', 0) for p in performance_list]
        drawdowns = [p.get('max_drawdown', 0) for p in performance_list]
        sharpes = [p.get('sharpe_ratio', 0) for p in performance_list]
        
        # Calculate averages
        avg_profit = np.mean(profits) if profits else 0
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        avg_drawdown = np.mean(drawdowns) if drawdowns else 0
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        
        return {
            'avg_profit': avg_profit,
            'avg_win_rate': avg_win_rate,
            'avg_drawdown': avg_drawdown,
            'avg_sharpe': avg_sharpe,
            'sample_size': len(performance_list)
        }
