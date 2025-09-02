#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap Trade Analyzer

This module provides specialized analytics for gap trading strategies, including:
- Gap identification and classification
- Gap statistics calculation
- Trade performance by gap type
- Historical gap behavior analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class GapTradeAnalyzer:
    """
    Specialized analyzer for gap trade statistics and performance metrics.
    
    This class provides tools to:
    - Identify and classify different types of gaps
    - Track gap fill statistics
    - Analyze gap performance by market conditions
    - Measure gap trade performance metrics
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the gap trade analyzer.
        
        Args:
            parameters: Configuration parameters
        """
        # Default parameters
        default_params = {
            'min_gap_percent': 1.0,
            'significant_gap_percent': 3.0,
            'lookback_days': 90,
            'gap_fill_lookback_days': 5,
            'categorize_gaps': True,
            'gap_categories': ['small', 'medium', 'large', 'extreme'],
            'gap_category_thresholds': [1.0, 3.0, 5.0], # % thresholds for categories
        }
        
        self.parameters = parameters or {}
        
        # Use defaults for missing parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Gap statistics storage
        self.gap_stats = {
            'total_gaps_detected': 0,
            'up_gaps': 0,
            'down_gaps': 0,
            'gap_fill_rate': 0.0,
            'avg_fill_time_days': 0.0,
            'gap_by_category': {
                'small': {'count': 0, 'fill_rate': 0.0, 'avg_move': 0.0},
                'medium': {'count': 0, 'fill_rate': 0.0, 'avg_move': 0.0},
                'large': {'count': 0, 'fill_rate': 0.0, 'avg_move': 0.0},
                'extreme': {'count': 0, 'fill_rate': 0.0, 'avg_move': 0.0},
            },
            'recent_gaps': [],
        }
        
        # Performance tracking
        self.performance = {
            'continuation_trades': {'count': 0, 'win_rate': 0.0, 'avg_return': 0.0},
            'fade_trades': {'count': 0, 'win_rate': 0.0, 'avg_return': 0.0},
            'by_size': {
                'small': {'trades': 0, 'win_rate': 0.0},
                'medium': {'trades': 0, 'win_rate': 0.0},
                'large': {'trades': 0, 'win_rate': 0.0},
                'extreme': {'trades': 0, 'win_rate': 0.0},
            },
            'by_sector': {},
        }
        
        logger.info(f"Gap Trade Analyzer initialized")
    
    def detect_gaps(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect gaps in historical price data.
        
        Args:
            data: Historical price data with OHLCV
            symbol: Symbol to analyze
            
        Returns:
            List of detected gaps with details
        """
        if len(data) < 2:
            return []
            
        min_gap_percent = self.parameters['min_gap_percent']
        gaps_detected = []
        
        # Check each bar for gaps
        for i in range(1, len(data)):
            prev_close = data['close'].iloc[i-1]
            curr_open = data['open'].iloc[i]
            
            # Calculate gap percentage
            gap_percent = ((curr_open / prev_close) - 1) * 100
            
            # Check if gap exceeds minimum threshold
            if abs(gap_percent) >= min_gap_percent:
                gap_direction = 'up' if gap_percent > 0 else 'down'
                gap_date = data.index[i]
                
                # Determine gap category
                gap_category = self._categorize_gap(gap_percent)
                
                # Calculate gap size in points
                gap_size = curr_open - prev_close
                
                # Create gap info dictionary
                gap_info = {
                    'symbol': symbol,
                    'date': gap_date,
                    'prev_close': prev_close,
                    'open': curr_open,
                    'gap_percent': gap_percent,
                    'gap_size': gap_size,
                    'gap_direction': gap_direction,
                    'gap_category': gap_category,
                    'filled': False,
                    'fill_date': None,
                    'fill_days': None,
                }
                
                # Check if the gap was filled within lookback period
                if i + self.parameters['gap_fill_lookback_days'] < len(data):
                    max_lookback = min(i + self.parameters['gap_fill_lookback_days'], len(data) - 1)
                    
                    if gap_direction == 'up':
                        # For up gaps, check if price falls back to the previous close
                        filled_indices = np.where(data['low'].iloc[i:max_lookback] <= prev_close)[0]
                        if len(filled_indices) > 0:
                            fill_index = filled_indices[0] + i
                            gap_info['filled'] = True
                            gap_info['fill_date'] = data.index[fill_index]
                            gap_info['fill_days'] = (data.index[fill_index] - gap_date).days
                    else:
                        # For down gaps, check if price rises back to the previous close
                        filled_indices = np.where(data['high'].iloc[i:max_lookback] >= prev_close)[0]
                        if len(filled_indices) > 0:
                            fill_index = filled_indices[0] + i
                            gap_info['filled'] = True
                            gap_info['fill_date'] = data.index[fill_index]
                            gap_info['fill_days'] = (data.index[fill_index] - gap_date).days
                
                gaps_detected.append(gap_info)
                
                # Update gap statistics
                self._update_gap_statistics(gap_info)
        
        return gaps_detected
    
    def _update_gap_statistics(self, gap_info: Dict[str, Any]) -> None:
        """
        Update internal gap statistics with new gap information.
        
        Args:
            gap_info: Information about a detected gap
        """
        # Update overall statistics
        self.gap_stats['total_gaps_detected'] += 1
        
        if gap_info['gap_direction'] == 'up':
            self.gap_stats['up_gaps'] += 1
        else:
            self.gap_stats['down_gaps'] += 1
        
        # Update category statistics
        category = gap_info['gap_category']
        self.gap_stats['gap_by_category'][category]['count'] += 1
        
        # Update fill statistics if we have fill information
        if gap_info.get('filled') is not None:
            # Calculate overall fill rate
            filled_gaps = [g for g in self.gap_stats['recent_gaps'] if g.get('filled', False)]
            total_gaps = len(self.gap_stats['recent_gaps']) + 1
            
            if total_gaps > 0:
                self.gap_stats['gap_fill_rate'] = len(filled_gaps) / total_gaps
            
            # Calculate average fill time
            fill_days = [g.get('fill_days', 0) for g in filled_gaps if g.get('fill_days') is not None]
            if fill_days:
                self.gap_stats['avg_fill_time_days'] = sum(fill_days) / len(fill_days)
        
        # Add to recent gaps list
        self.gap_stats['recent_gaps'].append(gap_info)
        
        # Keep list trimmed to lookback period
        max_gaps = self.parameters['lookback_days']
        if len(self.gap_stats['recent_gaps']) > max_gaps:
            self.gap_stats['recent_gaps'] = self.gap_stats['recent_gaps'][-max_gaps:]
    
    def _categorize_gap(self, gap_percent: float) -> str:
        """
        Categorize a gap based on its size.
        
        Args:
            gap_percent: Gap size as percentage
            
        Returns:
            Category name: 'small', 'medium', 'large', or 'extreme'
        """
        thresholds = self.parameters['gap_category_thresholds']
        categories = self.parameters['gap_categories']
        
        abs_gap = abs(gap_percent)
        
        for i, threshold in enumerate(thresholds):
            if abs_gap < threshold:
                return categories[i]
        
        return categories[-1]  # Return last category for gaps larger than all thresholds
    
    def update_trade_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update performance statistics with a completed trade result.
        
        Args:
            trade_result: Dictionary with trade details
        """
        # Extract trade details
        trade_type = trade_result.get('trade_type')  # 'continuation' or 'fade'
        gap_category = trade_result.get('gap_category')
        is_winning = trade_result.get('is_winning', False)
        return_pct = trade_result.get('return_pct', 0.0)
        sector = trade_result.get('sector', 'unknown')
        
        # Update statistics by trade type
        if trade_type == 'continuation':
            stats = self.performance['continuation_trades']
            stats['count'] += 1
            wins = stats['win_rate'] * (stats['count'] - 1)
            if is_winning:
                wins += 1
            stats['win_rate'] = wins / stats['count'] if stats['count'] > 0 else 0.0
            
            # Update average return
            old_total = stats['avg_return'] * (stats['count'] - 1)
            stats['avg_return'] = (old_total + return_pct) / stats['count'] if stats['count'] > 0 else 0.0
        elif trade_type == 'fade':
            stats = self.performance['fade_trades']
            stats['count'] += 1
            wins = stats['win_rate'] * (stats['count'] - 1)
            if is_winning:
                wins += 1
            stats['win_rate'] = wins / stats['count'] if stats['count'] > 0 else 0.0
            
            # Update average return
            old_total = stats['avg_return'] * (stats['count'] - 1)
            stats['avg_return'] = (old_total + return_pct) / stats['count'] if stats['count'] > 0 else 0.0
        
        # Update statistics by gap size
        if gap_category and gap_category in self.performance['by_size']:
            size_stats = self.performance['by_size'][gap_category]
            size_stats['trades'] += 1
            wins = size_stats['win_rate'] * (size_stats['trades'] - 1)
            if is_winning:
                wins += 1
            size_stats['win_rate'] = wins / size_stats['trades'] if size_stats['trades'] > 0 else 0.0
        
        # Update sector statistics
        if sector:
            if sector not in self.performance['by_sector']:
                self.performance['by_sector'][sector] = {'trades': 0, 'win_rate': 0.0, 'avg_return': 0.0}
            
            sector_stats = self.performance['by_sector'][sector]
            sector_stats['trades'] += 1
            wins = sector_stats['win_rate'] * (sector_stats['trades'] - 1)
            if is_winning:
                wins += 1
            sector_stats['win_rate'] = wins / sector_stats['trades'] if sector_stats['trades'] > 0 else 0.0
            
            # Update average return
            old_total = sector_stats['avg_return'] * (sector_stats['trades'] - 1)
            sector_stats['avg_return'] = (old_total + return_pct) / sector_stats['trades'] if sector_stats['trades'] > 0 else 0.0
    
    def get_gap_statistics(self) -> Dict[str, Any]:
        """
        Get current gap statistics.
        
        Returns:
            Dictionary with gap statistics
        """
        return self.gap_stats
    
    def get_trade_performance(self) -> Dict[str, Any]:
        """
        Get current trade performance statistics.
        
        Returns:
            Dictionary with trade performance metrics
        """
        return self.performance
    
    def analyze_best_gap_trading_approach(self, symbol: str = None) -> Dict[str, Any]:
        """
        Analyze historical data to determine optimal gap trading approach.
        
        Args:
            symbol: Optional specific symbol to analyze
            
        Returns:
            Dictionary with recommended trading approach
        """
        continuation_win_rate = self.performance['continuation_trades'].get('win_rate', 0.0)
        fade_win_rate = self.performance['fade_trades'].get('win_rate', 0.0)
        
        continuation_return = self.performance['continuation_trades'].get('avg_return', 0.0)
        fade_return = self.performance['fade_trades'].get('avg_return', 0.0)
        
        # Calculate preference score (weighted combination of win rate and return)
        continuation_score = (continuation_win_rate * 0.7) + (continuation_return / 10 * 0.3)
        fade_score = (fade_win_rate * 0.7) + (fade_return / 10 * 0.3)
        
        # Determine optimal approach
        if continuation_score > fade_score * 1.1:  # 10% better
            best_approach = 'continuation'
            preference = min(1.0, max(0.5, continuation_score / (continuation_score + fade_score)))
        elif fade_score > continuation_score * 1.1:  # 10% better
            best_approach = 'fade'
            preference = min(1.0, max(0.5, fade_score / (continuation_score + fade_score)))
        else:
            best_approach = 'both'
            preference = 0.5
        
        # Analyze by gap size
        best_size = None
        best_size_win_rate = 0.0
        
        for size, stats in self.performance['by_size'].items():
            if stats['trades'] >= 10 and stats['win_rate'] > best_size_win_rate:
                best_size = size
                best_size_win_rate = stats['win_rate']
        
        # Analyze by sector
        best_sector = None
        best_sector_win_rate = 0.0
        
        for sector, stats in self.performance['by_sector'].items():
            if stats['trades'] >= 10 and stats['win_rate'] > best_sector_win_rate:
                best_sector = sector
                best_sector_win_rate = stats['win_rate']
        
        # Form recommendations
        recommendations = {
            'recommended_approach': best_approach,
            'continuation_preference': preference,
            'best_gap_size': best_size,
            'best_sectors': best_sector,
            'continuation_stats': {
                'win_rate': continuation_win_rate,
                'avg_return': continuation_return
            },
            'fade_stats': {
                'win_rate': fade_win_rate,
                'avg_return': fade_return
            },
        }
        
        return recommendations
