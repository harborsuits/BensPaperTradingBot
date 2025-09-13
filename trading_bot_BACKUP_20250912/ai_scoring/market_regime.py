#!/usr/bin/env python3
"""
Market Regime Classifier System

This module provides tools to classify market regimes based on various indicators
and analyze historical market conditions for strategy performance evaluation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """
    Classifies market conditions into distinct regimes based on technical indicators
    and other market data.
    """
    
    def __init__(self, market_data_provider):
        """
        Initialize the market regime classifier.
        
        Args:
            market_data_provider: Provider for market data
        """
        self.market_data = market_data_provider
        self.regimes = ["bullish", "bearish", "neutral", "volatile"]
        self.cache = {}  # Cache for regime classifications
        
    def classify_regime(self, date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Classify the market regime for a specific date or current date.
        
        Args:
            date: Date to classify (None for current date)
            
        Returns:
            Dictionary with regime classification and metrics
        """
        # Check cache first
        cache_key = date.isoformat() if date else "current"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Get market data
        data = self.market_data.get_market_data(date)
        if not data:
            logger.warning(f"No market data available for {date or 'current date'}")
            return {
                "primary_regime": "unknown",
                "traits": [],
                "metrics": {}
            }
        
        # Extract key metrics
        try:
            spy_close = data.get('spy_close', 0)
            spy_sma20 = data.get('spy_sma20', 0)
            spy_sma50 = data.get('spy_sma50', 0)
            vix = data.get('vix', 20)
            atr = data.get('atr', 0)  # Average True Range
            adx = data.get('adx', 0)  # Average Directional Index
            breadth = data.get('advance_decline_ratio', 1.0)
            
            # Primary regime classification
            if spy_close > spy_sma20 and spy_close > spy_sma50 and vix < 18 and adx > 20 and breadth > 1.5:
                primary_regime = "bullish"
            elif spy_close < spy_sma20 and spy_close < spy_sma50 and vix > 25 and breadth < 0.7:
                primary_regime = "bearish"
            elif vix > 30 and (atr > data.get('atr_20_avg', atr * 0.8) * 1.5 if 'atr_20_avg' in data else True):
                primary_regime = "volatile"
            else:
                primary_regime = "neutral"
                
            # Secondary characteristics
            regime_traits = self._calculate_regime_traits(data)
            
            # Create result
            result = {
                "primary_regime": primary_regime,
                "traits": regime_traits,
                "metrics": {
                    "vix": vix,
                    "breadth": breadth,
                    "trend_strength": adx,
                    "volatility": atr,
                    "price_to_sma20": (spy_close / spy_sma20 - 1) * 100 if spy_sma20 > 0 else 0,
                    "price_to_sma50": (spy_close / spy_sma50 - 1) * 100 if spy_sma50 > 0 else 0
                }
            }
            
            # Cache the result
            self.cache[cache_key] = result.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return {
                "primary_regime": "unknown",
                "traits": [],
                "metrics": {}
            }
    
    def _calculate_regime_traits(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract additional market traits beyond basic regime.
        
        Args:
            data: Market data dictionary
            
        Returns:
            List of trait strings
        """
        traits = []
        
        # Trend characteristics
        if data.get('macd', 0) > 0:
            traits.append("rising_trend")
        elif data.get('macd', 0) < 0:
            traits.append("falling_trend")
            
        # Volatility characteristics
        if data.get('vix', 20) < 15:
            traits.append("low_volatility")
        elif data.get('vix', 20) > 30:
            traits.append("high_volatility")
            
        # Breadth characteristics
        if data.get('advance_decline_ratio', 1.0) > 2:
            traits.append("broad_strength")
        elif data.get('advance_decline_ratio', 1.0) < 0.5:
            traits.append("broad_weakness")
        
        # Sector rotation characteristics
        if data.get('sector_deviation', 0) > 5:
            traits.append("high_sector_dispersion")
            
        # Momentum characteristics
        if data.get('rsi', 50) > 70:
            traits.append("overbought")
        elif data.get('rsi', 50) < 30:
            traits.append("oversold")
            
        # Volume characteristics
        if data.get('volume_ratio', 1.0) > 1.5:
            traits.append("high_volume")
        elif data.get('volume_ratio', 1.0) < 0.7:
            traits.append("low_volume")
            
        # Trend strength
        if data.get('adx', 0) > 30:
            traits.append("strong_trend")
        elif data.get('adx', 0) < 15:
            traits.append("weak_trend")
            
        # Market sentiment (if available)
        sentiment = data.get('market_sentiment', '')
        if sentiment == 'bullish':
            traits.append("bullish_sentiment")
        elif sentiment == 'bearish':
            traits.append("bearish_sentiment")
        elif sentiment == 'extreme_bullish':
            traits.append("extreme_bullish_sentiment")
        elif sentiment == 'extreme_bearish':
            traits.append("extreme_bearish_sentiment")
            
        return traits
        
    def historical_regime_tagging(self, start_date: datetime.date, 
                                 end_date: datetime.date) -> Dict[datetime.date, Dict[str, Any]]:
        """
        Tag all trading days in a date range with regime labels.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary mapping dates to regime data
        """
        # Get trading days in the range
        dates = self.market_data.get_trading_days(start_date, end_date)
        
        regime_data = {}
        for date in dates:
            regime_data[date] = self.classify_regime(date)
            
        logger.info(f"Tagged {len(regime_data)} days with market regimes")
        return regime_data
        
    def get_regime_distribution(self, start_date: datetime.date, 
                               end_date: datetime.date) -> Dict[str, float]:
        """
        Calculate the distribution of market regimes over a time period.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary mapping regimes to their frequency (percentage)
        """
        # Get regime data for the period
        regime_data = self.historical_regime_tagging(start_date, end_date)
        
        # Count regimes
        regime_counts = {regime: 0 for regime in self.regimes}
        for date, data in regime_data.items():
            regime = data.get("primary_regime")
            if regime in regime_counts:
                regime_counts[regime] += 1
            
        # Calculate percentages
        total_days = len(regime_data)
        if total_days > 0:
            regime_distribution = {
                regime: (count / total_days) * 100
                for regime, count in regime_counts.items()
            }
        else:
            regime_distribution = {regime: 0 for regime in self.regimes}
            
        return regime_distribution
        
    def get_regime_transitions(self, start_date: datetime.date, 
                              end_date: datetime.date) -> Dict[Tuple[str, str], int]:
        """
        Analyze regime transitions over a time period.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary mapping regime transitions to their count
        """
        # Get regime data for the period
        regime_data = self.historical_regime_tagging(start_date, end_date)
        
        # Sort dates
        sorted_dates = sorted(regime_data.keys())
        
        # Count transitions
        transitions = {}
        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i-1]
            curr_date = sorted_dates[i]
            
            prev_regime = regime_data[prev_date].get("primary_regime")
            curr_regime = regime_data[curr_date].get("primary_regime")
            
            if prev_regime != curr_regime:
                transition = (prev_regime, curr_regime)
                transitions[transition] = transitions.get(transition, 0) + 1
                
        return transitions
        
    def get_average_regime_duration(self, start_date: datetime.date, 
                                  end_date: datetime.date) -> Dict[str, float]:
        """
        Calculate the average duration of each market regime.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary mapping regimes to their average duration in days
        """
        # Get regime data for the period
        regime_data = self.historical_regime_tagging(start_date, end_date)
        
        # Sort dates
        sorted_dates = sorted(regime_data.keys())
        
        # Calculate durations
        durations = {regime: [] for regime in self.regimes}
        
        current_regime = None
        regime_start = None
        
        for date in sorted_dates:
            regime = regime_data[date].get("primary_regime")
            
            if regime != current_regime:
                # End of a regime period
                if current_regime and regime_start:
                    duration = (date - regime_start).days
                    if duration > 0:  # Ensure valid duration
                        durations[current_regime].append(duration)
                
                # Start of a new regime period
                current_regime = regime
                regime_start = date
        
        # Handle the last regime period
        if current_regime and regime_start and sorted_dates:
            duration = (sorted_dates[-1] - regime_start).days + 1
            if duration > 0:
                durations[current_regime].append(duration)
        
        # Calculate averages
        average_durations = {}
        for regime, regime_durations in durations.items():
            if regime_durations:
                average_durations[regime] = sum(regime_durations) / len(regime_durations)
            else:
                average_durations[regime] = 0
                
        return average_durations
        
    def clear_cache(self) -> None:
        """Clear the regime classification cache."""
        self.cache = {}
        logger.info("Cleared regime classification cache")


class HistoricalPerformanceAnalyzer:
    """
    Analyzes strategy performance across different market regimes based on historical data.
    """
    
    def __init__(self, trade_journal, regime_classifier: MarketRegimeClassifier):
        """
        Initialize the performance analyzer.
        
        Args:
            trade_journal: Trade journal for accessing historical trades
            regime_classifier: Market regime classifier
        """
        self.trade_journal = trade_journal
        self.regime_classifier = regime_classifier
        
    def analyze_performance_by_regime(self, strategy_names: List[str], 
                                     lookback_days: int = 365) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze how each strategy performed in different market regimes.
        
        Args:
            strategy_names: List of strategy names to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Nested dictionary mapping regimes to strategies to performance metrics
        """
        # Get historical regimes
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        historical_regimes = self.regime_classifier.historical_regime_tagging(start_date, end_date)
        
        # Get historical trades for each strategy
        performance_by_regime = {regime: {} for regime in self.regime_classifier.regimes}
        
        for strategy in strategy_names:
            logger.info(f"Analyzing performance for strategy: {strategy}")
            
            # Get trades for this strategy
            trades = self.trade_journal.get_historical_trades(strategy, start_date, end_date)
            if not trades:
                logger.warning(f"No historical trades found for strategy: {strategy}")
                continue
                
            # Group trades by regime
            regime_trades = {regime: [] for regime in self.regime_classifier.regimes}
            
            for trade in trades:
                trade_date = trade.entry_date
                if trade_date in historical_regimes:
                    regime = historical_regimes[trade_date].get("primary_regime")
                    regime_trades[regime].append(trade)
            
            # Calculate performance metrics for each regime
            for regime, trades_in_regime in regime_trades.items():
                if not trades_in_regime:
                    performance_by_regime[regime][strategy] = {
                        "win_rate": 0,
                        "avg_return": 0,
                        "sharpe": 0,
                        "max_drawdown": 0,
                        "sample_size": 0
                    }
                    continue
                
                # Calculate metrics
                returns = [trade.pnl for trade in trades_in_regime]
                wins = sum(1 for r in returns if r > 0)
                
                performance_by_regime[regime][strategy] = {
                    "win_rate": (wins / len(trades_in_regime)) * 100 if trades_in_regime else 0,
                    "avg_return": sum(returns) / len(returns) if returns else 0,
                    "sharpe": self._calculate_sharpe(returns),
                    "max_drawdown": self._calculate_max_drawdown(returns),
                    "sample_size": len(trades_in_regime)
                }
                
                logger.debug(f"Calculated {regime} performance for {strategy}: "
                           f"{performance_by_regime[regime][strategy]}")
        
        return performance_by_regime
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0) -> float:
        """
        Calculate Sharpe ratio from a list of returns.
        
        Args:
            returns: List of percentage returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = sum(returns) / len(returns)
        std_deviation = np.std(returns)
        
        if std_deviation == 0:
            return 0
            
        return (mean_return - risk_free_rate) / std_deviation
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from a list of returns.
        
        Args:
            returns: List of percentage returns
            
        Returns:
            Maximum drawdown as a percentage
        """
        if not returns:
            return 0
            
        # Convert returns to equity curve (starting at 100)
        equity = 100
        equity_curve = [equity]
        
        for r in returns:
            equity *= (1 + r/100)
            equity_curve.append(equity)
            
        # Calculate drawdown
        max_drawdown = 0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown * 100  # Return as percentage
    
    def get_regime_bias_adjustment(self, current_regime: str) -> Dict[str, float]:
        """
        Generate bias adjustment scores for strategies based on historical
        performance in the current market regime.
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Dictionary mapping strategy names to bias adjustments
        """
        all_strategies = self.trade_journal.get_strategy_names()
        performance_data = self.analyze_performance_by_regime(all_strategies)
        
        # Get performance in current regime
        regime_performance = performance_data.get(current_regime, {})
        
        # Calculate bias adjustments (normalized to range [-1, 1])
        bias_adjustments = {}
        
        # Extract metrics for normalization
        win_rates = [data.get("win_rate", 0) for data in regime_performance.values() if data]
        avg_returns = [data.get("avg_return", 0) for data in regime_performance.values() if data]
        sharpes = [data.get("sharpe", 0) for data in regime_performance.values() if data]
        
        if not win_rates or not avg_returns or not sharpes:
            logger.warning(f"Insufficient data for bias adjustment in {current_regime} regime")
            return {strategy: 0 for strategy in all_strategies}
        
        # Avoid division by zero
        win_rate_range = max(win_rates) - min(win_rates) if win_rates and max(win_rates) != min(win_rates) else 1
        avg_return_range = max(avg_returns) - min(avg_returns) if avg_returns and max(avg_returns) != min(avg_returns) else 1
        sharpe_range = max(sharpes) - min(sharpes) if sharpes and max(sharpes) != min(sharpes) else 1
        
        for strategy, metrics in regime_performance.items():
            if not metrics or metrics.get("sample_size", 0) < 5:
                # Not enough data for reliable adjustment
                bias_adjustments[strategy] = 0
                continue
                
            # Normalize metrics to [-1, 1] range
            normalized_win_rate = (metrics["win_rate"] - min(win_rates)) / win_rate_range * 2 - 1 if win_rate_range else 0
            normalized_return = (metrics["avg_return"] - min(avg_returns)) / avg_return_range * 2 - 1 if avg_return_range else 0
            normalized_sharpe = (metrics["sharpe"] - min(sharpes)) / sharpe_range * 2 - 1 if sharpe_range else 0
            
            # Weighted combination of metrics for final bias adjustment
            bias_adjustments[strategy] = (
                normalized_win_rate * 0.3 + 
                normalized_return * 0.4 + 
                normalized_sharpe * 0.3
            )
            
        return bias_adjustments
        
    def get_strategy_regime_report(self, strategy_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report of a strategy's performance across all regimes.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with detailed performance data
        """
        # Analyze performance across all regimes
        all_regime_performance = self.analyze_performance_by_regime([strategy_name])
        
        # Calculate overall statistics
        total_trades = sum(
            perf.get(strategy_name, {}).get("sample_size", 0) 
            for perf in all_regime_performance.values()
        )
        
        if total_trades == 0:
            return {
                "strategy": strategy_name,
                "total_trades": 0,
                "regime_data": {},
                "best_regime": None,
                "worst_regime": None
            }
            
        # Determine best and worst regimes
        best_regime = None
        best_return = -float('inf')
        worst_regime = None
        worst_return = float('inf')
        
        for regime, strategies in all_regime_performance.items():
            if strategy_name in strategies:
                regime_return = strategies[strategy_name].get("avg_return", 0)
                sample_size = strategies[strategy_name].get("sample_size", 0)
                
                # Only consider regimes with sufficient samples
                if sample_size >= 5:
                    if regime_return > best_return:
                        best_return = regime_return
                        best_regime = regime
                        
                    if regime_return < worst_return:
                        worst_return = regime_return
                        worst_regime = regime
        
        # Prepare the report
        report = {
            "strategy": strategy_name,
            "total_trades": total_trades,
            "regime_data": {
                regime: stats.get(strategy_name, {})
                for regime, stats in all_regime_performance.items()
                if strategy_name in stats
            },
            "best_regime": best_regime,
            "worst_regime": worst_regime
        }
        
        return report 