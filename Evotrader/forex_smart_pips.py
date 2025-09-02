#!/usr/bin/env python3
"""
Forex Smart Pip Analytics - Enhanced Pip Intelligence Module

This module provides advanced pip-based metrics capabilities:
- Volatility-adjusted pip targets
- Spread-aware pip analysis
- Risk-adjusted pip metrics
- Correlation-aware pip valuation
"""

import os
import sys
import yaml
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_smart_pips')


class SmartPipAnalyzer:
    """
    Enhanced pip-based analytics for Forex trading.
    Provides volatility-aware and risk-adjusted pip metrics.
    """
    
    def __init__(self, pair_manager=None, config: Dict[str, Any] = None):
        """
        Initialize the smart pip analyzer.
        
        Args:
            pair_manager: Forex pair manager instance
            config: Configuration dictionary
        """
        self.pair_manager = pair_manager
        self.config = config or {}
        
        # ATR history for volatility calculations
        self.atr_history = {}
        
        # Correlation data between pairs
        self.pair_correlations = {}
        
        # Load historical ATR data if available
        self._load_atr_history()
        
        # Load pair correlations if available
        self._load_pair_correlations()
        
        logger.info("Smart Pip Analyzer initialized")
    
    def _load_atr_history(self) -> None:
        """Load historical ATR data from file if available."""
        atr_file = self.config.get('atr_history_file', 'atr_history.json')
        
        if os.path.exists(atr_file):
            try:
                with open(atr_file, 'r') as f:
                    self.atr_history = json.load(f)
                logger.info(f"Loaded ATR history from {atr_file}")
            except Exception as e:
                logger.error(f"Error loading ATR history: {e}")
    
    def _load_pair_correlations(self) -> None:
        """Load pair correlation data from file if available."""
        corr_file = self.config.get('correlation_file', 'pair_correlations.json')
        
        if os.path.exists(corr_file):
            try:
                with open(corr_file, 'r') as f:
                    self.pair_correlations = json.load(f)
                logger.info(f"Loaded pair correlations from {corr_file}")
            except Exception as e:
                logger.error(f"Error loading pair correlations: {e}")
    
    def save_atr_history(self) -> None:
        """Save ATR history to file."""
        atr_file = self.config.get('atr_history_file', 'atr_history.json')
        
        try:
            with open(atr_file, 'w') as f:
                json.dump(self.atr_history, f, indent=2)
            logger.info(f"Saved ATR history to {atr_file}")
        except Exception as e:
            logger.error(f"Error saving ATR history: {e}")
    
    def save_pair_correlations(self) -> None:
        """Save pair correlations to file."""
        corr_file = self.config.get('correlation_file', 'pair_correlations.json')
        
        try:
            with open(corr_file, 'w') as f:
                json.dump(self.pair_correlations, f, indent=2)
            logger.info(f"Saved pair correlations to {corr_file}")
        except Exception as e:
            logger.error(f"Error saving pair correlations: {e}")
    
    def calculate_pip_target(self, pair: str, base_pips: float = 20.0, 
                            timeframe: str = '1h', 
                            data: Optional[pd.DataFrame] = None) -> float:
        """
        Dynamically adjust pip targets based on current volatility.
        
        Args:
            pair: Currency pair ('EURUSD', 'GBPJPY', etc.)
            base_pips: Base pip target value
            timeframe: Timeframe for volatility calculation
            data: Market data (optional)
            
        Returns:
            Volatility-adjusted pip target
        """
        # Get current and historical ATR values
        current_atr = self.get_current_atr(pair, timeframe, data)
        historical_atr = self.get_historical_atr(pair, timeframe)
        
        # Calculate adjustment factor
        if historical_atr > 0:
            adjustment_factor = current_atr / historical_atr
        else:
            adjustment_factor = 1.0
        
        # Apply floor and ceiling to adjustment factor
        adjustment_factor = max(min(adjustment_factor, 2.0), 0.5)
        
        # Calculate adjusted pip target
        adjusted_pip_target = base_pips * adjustment_factor
        
        logger.debug(f"Adjusted pip target for {pair} ({timeframe}): {adjusted_pip_target:.1f} pips (base: {base_pips:.1f})")
        return adjusted_pip_target
    
    def get_current_atr(self, pair: str, timeframe: str = '1h', 
                       data: Optional[pd.DataFrame] = None,
                       atr_periods: int = 14) -> float:
        """
        Calculate current Average True Range (ATR) in pips.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            data: Market data (optional)
            atr_periods: Periods for ATR calculation
            
        Returns:
            Current ATR in pips
        """
        # If data is provided, calculate from it
        if data is not None and len(data) > atr_periods:
            return self._calculate_atr_from_data(data, pair, atr_periods)
        
        # Use pair manager if available
        if self.pair_manager:
            try:
                atr_value = self.pair_manager.get_current_atr(pair, timeframe)
                if atr_value > 0:
                    return atr_value
            except Exception as e:
                logger.debug(f"Error getting current ATR from pair manager: {e}")
        
        # Fallback to recent history or default value
        pair_tf_key = f"{pair}_{timeframe}"
        if pair_tf_key in self.atr_history:
            atr_values = self.atr_history[pair_tf_key]
            if atr_values:
                return atr_values[-1]  # Most recent value
        
        # Fallback to typical values if no data available
        return self._get_typical_atr(pair, timeframe)
    
    def get_historical_atr(self, pair: str, timeframe: str = '1h', 
                          lookback_days: int = 30) -> float:
        """
        Get average historical ATR value over specified lookback period.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            lookback_days: Number of days to look back
            
        Returns:
            Average historical ATR
        """
        pair_tf_key = f"{pair}_{timeframe}"
        if pair_tf_key in self.atr_history:
            atr_values = self.atr_history[pair_tf_key]
            if atr_values:
                # Calculate average of last N days
                recent_history = atr_values[-min(lookback_days, len(atr_values)):]
                return sum(recent_history) / len(recent_history) if recent_history else self._get_typical_atr(pair, timeframe)
        
        # Fallback to typical values if no history available
        return self._get_typical_atr(pair, timeframe)
    
    def _calculate_atr_from_data(self, data: pd.DataFrame, pair: str, periods: int = 14) -> float:
        """
        Calculate ATR from OHLC data.
        
        Args:
            data: OHLC data
            pair: Currency pair (for pip conversion)
            periods: Periods for ATR calculation
            
        Returns:
            ATR value in pips
        """
        if len(data) <= periods or 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return self._get_typical_atr(pair)
        
        try:
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=periods).mean().iloc[-1]
            
            # Convert to pips
            pip_multiplier = self._get_pip_multiplier(pair)
            atr_pips = atr * pip_multiplier
            
            return atr_pips
            
        except Exception as e:
            logger.error(f"Error calculating ATR from data: {e}")
            return self._get_typical_atr(pair)
    
    def _get_typical_atr(self, pair: str, timeframe: str = '1h') -> float:
        """
        Get typical ATR value for a pair and timeframe when data is not available.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            
        Returns:
            Typical ATR value in pips
        """
        # Default typical ATR values in pips for different timeframes
        typical_values = {
            '1m': {'major': 1.0, 'minor': 1.5, 'exotic': 2.5, 'jpy': 10.0},
            '5m': {'major': 2.0, 'minor': 3.0, 'exotic': 5.0, 'jpy': 20.0},
            '15m': {'major': 3.5, 'minor': 5.0, 'exotic': 8.0, 'jpy': 35.0},
            '30m': {'major': 5.0, 'minor': 7.0, 'exotic': 12.0, 'jpy': 50.0},
            '1h': {'major': 7.0, 'minor': 10.0, 'exotic': 18.0, 'jpy': 70.0},
            '4h': {'major': 15.0, 'minor': 25.0, 'exotic': 40.0, 'jpy': 150.0},
            '1d': {'major': 40.0, 'minor': 60.0, 'exotic': 100.0, 'jpy': 400.0}
        }
        
        # Determine pair category
        if 'JPY' in pair:
            category = 'jpy'
        elif pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']:
            category = 'major'
        elif set(pair).intersection(['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF', 'JPY']):
            category = 'minor'
        else:
            category = 'exotic'
        
        # Get value for timeframe and category
        if timeframe in typical_values:
            return typical_values[timeframe][category]
        else:
            # Fallback to 1h if timeframe not found
            return typical_values['1h'][category]
    
    def update_atr_history(self, pair: str, timeframe: str, atr_value: float) -> None:
        """
        Update ATR history for a pair and timeframe.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            atr_value: ATR value to record
        """
        pair_tf_key = f"{pair}_{timeframe}"
        
        if pair_tf_key not in self.atr_history:
            self.atr_history[pair_tf_key] = []
        
        # Append new value
        self.atr_history[pair_tf_key].append(atr_value)
        
        # Trim history to last 90 days
        max_history = 90
        if len(self.atr_history[pair_tf_key]) > max_history:
            self.atr_history[pair_tf_key] = self.atr_history[pair_tf_key][-max_history:]
    
    def calculate_spread_cost_pips(self, pair: str, entry_spread: float = None, 
                                  exit_spread: float = None) -> float:
        """
        Calculate total spread cost in pips for a round-trip trade.
        
        Args:
            pair: Currency pair
            entry_spread: Entry spread in pips (if None, uses typical value)
            exit_spread: Exit spread in pips (if None, uses entry_spread)
            
        Returns:
            Total spread cost in pips
        """
        # Use provided spreads or get from pair manager
        if entry_spread is None:
            if self.pair_manager:
                pair_obj = self.pair_manager.get_pair(pair)
                if pair_obj:
                    spread_range = pair_obj.get_spread_range()
                    if spread_range and len(spread_range) == 2:
                        entry_spread = (spread_range[0] + spread_range[1]) / 2
        
        # Fallback to typical values if still None
        if entry_spread is None:
            if 'JPY' in pair:
                entry_spread = 2.0  # Typical for JPY pairs
            elif pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
                entry_spread = 1.0  # Typical for major pairs
            else:
                entry_spread = 2.5  # Typical for crosses and exotic pairs
        
        # Use entry spread for exit if not provided
        if exit_spread is None:
            exit_spread = entry_spread
        
        # Total round-trip cost
        total_spread_cost = entry_spread + exit_spread
        
        return total_spread_cost
    
    def calculate_risk_adjusted_pips(self, pip_profit: float, 
                                    pip_risk: float, 
                                    win_rate: float) -> Dict[str, float]:
        """
        Calculate risk-adjusted pip metrics like Sharpe and Sortino equivalents.
        
        Args:
            pip_profit: Average pip profit per winning trade
            pip_risk: Average pip loss per losing trade
            win_rate: Win rate as a decimal (0.0-1.0)
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        # Basic expectancy
        expectancy = (win_rate * pip_profit) - ((1 - win_rate) * pip_risk)
        
        # Pip Sharpe Ratio (expectancy / standard deviation)
        # Standard deviation of pip returns
        win_deviation = (pip_profit - expectancy) ** 2
        loss_deviation = (pip_risk + expectancy) ** 2
        std_dev = math.sqrt(win_rate * win_deviation + (1 - win_rate) * loss_deviation)
        
        # Avoid division by zero
        if std_dev > 0:
            pip_sharpe = expectancy / std_dev
        else:
            pip_sharpe = 0.0
        
        # Pip Sortino Ratio (expectancy / downside deviation)
        # Only negative deviations count
        downside_dev = math.sqrt((1 - win_rate) * loss_deviation)
        
        # Avoid division by zero
        if downside_dev > 0:
            pip_sortino = expectancy / downside_dev
        else:
            pip_sortino = 0.0
        
        # Pip Calmar Ratio (expectancy / max drawdown in pips)
        # Approximation: max drawdown ≈ 2-3 * average loss
        estimated_max_drawdown = 2.5 * pip_risk
        
        # Avoid division by zero
        if estimated_max_drawdown > 0:
            pip_calmar = expectancy / estimated_max_drawdown
        else:
            pip_calmar = 0.0
        
        return {
            'expectancy': expectancy,
            'pip_sharpe': pip_sharpe,
            'pip_sortino': pip_sortino,
            'pip_calmar': pip_calmar
        }
    
    def calculate_pair_correlation_adjusted_pips(self, 
                                               pair: str, 
                                               pips: float, 
                                               other_pairs: List[Tuple[str, float]] = None) -> float:
        """
        Adjust pip value based on correlation with other active pairs.
        
        Args:
            pair: Currency pair
            pips: Raw pip value
            other_pairs: List of (pair, pips) tuples for other active positions
            
        Returns:
            Correlation-adjusted pip value
        """
        if not other_pairs or not self.pair_correlations:
            return pips
        
        # Start with original pip value
        adjusted_pips = pips
        
        # Check correlation with each other active pair
        for other_pair, other_pips in other_pairs:
            # Skip same pair
            if other_pair == pair:
                continue
            
            # Get correlation between pairs
            correlation = self._get_pair_correlation(pair, other_pair)
            
            # Skip if correlation data not available
            if correlation is None:
                continue
            
            # Adjust pips based on correlation
            # If highly positively correlated, discount the pip value
            # If negatively correlated, boost the pip value
            if correlation > 0.7:
                # High positive correlation - discount pips
                adjustment_factor = 1.0 - ((correlation - 0.7) * 2.0)  # Linear discount up to 40% at correlation=0.9
                adjusted_pips *= adjustment_factor
            elif correlation < -0.5:
                # Negative correlation - boost pips
                adjustment_factor = 1.0 + (abs(correlation) - 0.5) * 0.4  # Linear boost up to 20% at correlation=-1.0
                adjusted_pips *= adjustment_factor
        
        return adjusted_pips
    
    def _get_pair_correlation(self, pair1: str, pair2: str) -> Optional[float]:
        """
        Get correlation between two currency pairs.
        
        Args:
            pair1: First currency pair
            pair2: Second currency pair
            
        Returns:
            Correlation coefficient or None if not available
        """
        # Try direct lookup
        pair_key = f"{pair1}_{pair2}"
        if pair_key in self.pair_correlations:
            return self.pair_correlations[pair_key]
        
        # Try reverse lookup
        reverse_key = f"{pair2}_{pair1}"
        if reverse_key in self.pair_correlations:
            return self.pair_correlations[reverse_key]
        
        # No correlation data available
        return None
    
    def update_pair_correlation(self, pair1: str, pair2: str, correlation: float) -> None:
        """
        Update correlation data between two currency pairs.
        
        Args:
            pair1: First currency pair
            pair2: Second currency pair
            correlation: Correlation coefficient (-1.0 to 1.0)
        """
        # Store with pairs in alphabetical order for consistency
        if pair1 > pair2:
            pair1, pair2 = pair2, pair1
        
        pair_key = f"{pair1}_{pair2}"
        self.pair_correlations[pair_key] = correlation
    
    def calculate_correlated_pairs(self, target_pair: str, 
                                  min_correlation: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find pairs highly correlated with the target pair.
        
        Args:
            target_pair: Currency pair to find correlations for
            min_correlation: Minimum absolute correlation coefficient
            
        Returns:
            List of dictionaries with pair and correlation data
        """
        correlated_pairs = []
        
        # Check all correlations
        for pair_key, correlation in self.pair_correlations.items():
            pair1, pair2 = pair_key.split('_')
            
            # Check if target pair is involved
            if pair1 == target_pair:
                if abs(correlation) >= min_correlation:
                    correlated_pairs.append({
                        'pair': pair2,
                        'correlation': correlation,
                        'relationship': 'positive' if correlation > 0 else 'negative'
                    })
            elif pair2 == target_pair:
                if abs(correlation) >= min_correlation:
                    correlated_pairs.append({
                        'pair': pair1,
                        'correlation': correlation,
                        'relationship': 'positive' if correlation > 0 else 'negative'
                    })
        
        # Sort by absolute correlation (highest first)
        correlated_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return correlated_pairs
    
    def _get_pip_multiplier(self, pair: str) -> float:
        """
        Get pip multiplier for converting price changes to pips.
        
        Args:
            pair: Currency pair
            
        Returns:
            Pip multiplier (e.g., 10000 for 4-decimal pairs, 100 for JPY pairs)
        """
        # Use pair manager if available
        if self.pair_manager:
            pair_obj = self.pair_manager.get_pair(pair)
            if pair_obj:
                return pair_obj.get_pip_multiplier()
        
        # Default multipliers
        if 'JPY' in pair:
            return 100  # JPY pairs have 2 decimal places
        else:
            return 10000  # Most pairs have 4 decimal places


# Test function
def test_smart_pips():
    """Test the smart pip analyzer functionality."""
    analyzer = SmartPipAnalyzer()
    
    # Test pip target adjustment
    adjusted_target = analyzer.calculate_pip_target('EURUSD', 20.0)
    print(f"Adjusted pip target for EURUSD: {adjusted_target:.1f} pips (base: 20.0)")
    
    # Test risk-adjusted metrics
    risk_metrics = analyzer.calculate_risk_adjusted_pips(30.0, 15.0, 0.55)
    print(f"Risk-adjusted pip metrics: {risk_metrics}")
    
    # Test correlation-adjusted pips
    analyzer.update_pair_correlation('EURUSD', 'GBPUSD', 0.85)
    analyzer.update_pair_correlation('EURUSD', 'USDCHF', -0.90)
    other_pairs = [('GBPUSD', 25.0), ('USDCHF', -15.0)]
    adjusted_pips = analyzer.calculate_pair_correlation_adjusted_pips('EURUSD', 20.0, other_pairs)
    print(f"Correlation-adjusted pips for EURUSD: {adjusted_pips:.1f} (original: 20.0)")
    
    return "Smart pip tests completed"


if __name__ == "__main__":
    test_smart_pips()
