#!/usr/bin/env python3
"""
Market Regime Classifier

This module provides functionality to classify market regimes based on
various market indicators, volatility measurements, and trend analysis.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """
    Classifies market regimes based on multiple indicators and market context.
    
    Regimes include:
    - bullish: Strong uptrend with low volatility
    - moderately_bullish: Uptrend with moderate volatility
    - neutral: No clear trend with low volatility
    - moderately_bearish: Downtrend with moderate volatility
    - bearish: Strong downtrend with high volatility
    - volatile: High volatility with no clear trend
    - sideways: Range-bound market with low volatility
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_provider = None,
        lookback_days: int = 60
    ):
        """
        Initialize the market regime classifier.
        
        Args:
            config_path: Path to configuration file
            data_provider: Market data provider instance
            lookback_days: Days to look back for regime classification
        """
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # Store data provider and settings
        self.data_provider = data_provider
        self.lookback_days = lookback_days
        
        # Define market indices to track
        self.market_indices = self.config.get('market_indices', ['SPY', 'QQQ', 'IWM', 'VIX'])
        
        # Define regime thresholds
        self.thresholds = self.config.get('regime_thresholds', {
            'vix_high': 25.0,
            'vix_very_high': 35.0,
            'trend_strong': 0.12,
            'trend_moderate': 0.05,
            'volatility_high': 0.015,
            'breadth_strong': 0.7,
            'breadth_weak': 0.3
        })
        
        logger.info("Market Regime Classifier initialized")
    
    def classify_regime(self) -> str:
        """
        Classify the current market regime.
        
        Returns:
            Market regime string
        """
        try:
            # Check if data provider is available
            if not self.data_provider:
                logger.warning("No data provider available, using fallback classification")
                return "neutral"  # Default to neutral regime
            
            # Collect market data
            market_data = self._collect_market_data()
            
            # Calculate regime indicators
            indicators = self._calculate_regime_indicators(market_data)
            
            # Determine regime based on indicators
            regime = self._determine_regime(indicators)
            
            logger.info(f"Market regime classified as: {regime}")
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {str(e)}")
            return "neutral"  # Default to neutral regime on error
    
    def _collect_market_data(self) -> Dict[str, Any]:
        """
        Collect market data required for regime classification.
        
        Returns:
            Dictionary of market data
        """
        market_data = {}
        
        # Get historical data for indices
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        
        try:
            # Get data from provider
            historical_data = self.data_provider.get_historical_data(
                symbols=self.market_indices,
                start_date=start_date,
                end_date=end_date,
                timeframe="day"
            )
            
            market_data['historical'] = historical_data
            
            # Get current market context
            current_data = self.data_provider.get_current_market_data(self.market_indices)
            market_data['current'] = current_data
            
            # Get market context if available
            if 'market_context' in current_data:
                market_data['context'] = current_data['market_context']
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            return {}
    
    def _calculate_regime_indicators(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate regime indicators from market data.
        
        Args:
            market_data: Dictionary of market data
            
        Returns:
            Dictionary of regime indicators
        """
        indicators = {}
        
        # Extract historical data
        historical = market_data.get('historical', {})
        
        # Get SPY data for trend and volatility calculations
        spy_data = historical.get('SPY', pd.DataFrame())
        
        if not spy_data.empty:
            # Calculate trend indicators
            spy_data['return'] = spy_data['close'].pct_change()
            
            # 20-day and 50-day moving averages
            spy_data['MA20'] = spy_data['close'].rolling(window=20).mean()
            spy_data['MA50'] = spy_data['close'].rolling(window=50).mean()
            
            # Calculate recent trend (slope of 20-day MA)
            recent_ma = spy_data['MA20'].dropna().tail(20)
            if len(recent_ma) >= 10:
                x = np.arange(len(recent_ma))
                y = recent_ma.values
                slope, _ = np.polyfit(x, y, 1)
                trend_20d = slope / recent_ma.iloc[0]
                indicators['trend_20d'] = trend_20d
            
            # Calculate longer trend (slope of 50-day MA)
            recent_ma = spy_data['MA50'].dropna().tail(50)
            if len(recent_ma) >= 25:
                x = np.arange(len(recent_ma))
                y = recent_ma.values
                slope, _ = np.polyfit(x, y, 1)
                trend_50d = slope / recent_ma.iloc[0]
                indicators['trend_50d'] = trend_50d
            
            # Calculate volatility (standard deviation of returns)
            volatility_20d = spy_data['return'].tail(20).std() * np.sqrt(252)  # Annualized
            indicators['volatility_20d'] = volatility_20d
            
            # Moving average relationship
            last_row = spy_data.iloc[-1]
            if not pd.isna(last_row['MA20']) and not pd.isna(last_row['MA50']):
                ma_relationship = last_row['MA20'] / last_row['MA50'] - 1
                indicators['ma_relationship'] = ma_relationship
        
        # Get VIX data for volatility regime
        vix_data = historical.get('VIX', pd.DataFrame())
        
        if not vix_data.empty:
            # Current VIX level
            current_vix = vix_data['close'].iloc[-1] if not vix_data.empty else None
            indicators['vix_current'] = current_vix
            
            # VIX percentile
            vix_percentile = np.percentile(vix_data['close'], 75) if not vix_data.empty else None
            indicators['vix_percentile'] = vix_percentile
            
            # VIX trend
            vix_change_5d = vix_data['close'].pct_change(5).iloc[-1] if len(vix_data) > 5 else None
            indicators['vix_change_5d'] = vix_change_5d
        
        # Market breadth indicators
        if 'context' in market_data:
            context = market_data['context']
            
            # Get sector performance as breadth indicator
            sector_perf = context.get('sector_performance', {})
            if sector_perf:
                # Count positive vs negative sectors
                pos_sectors = sum(1 for v in sector_perf.values() if v > 0)
                neg_sectors = sum(1 for v in sector_perf.values() if v < 0)
                total_sectors = len(sector_perf)
                
                if total_sectors > 0:
                    breadth_ratio = pos_sectors / total_sectors
                    indicators['breadth_ratio'] = breadth_ratio
            
            # If market regime already provided by data provider, use it as a hint
            if 'market_regime' in context:
                indicators['provider_regime'] = context['market_regime']
        
        return indicators
    
    def _determine_regime(self, indicators: Dict[str, float]) -> str:
        """
        Determine market regime based on calculated indicators.
        
        Args:
            indicators: Dictionary of regime indicators
            
        Returns:
            Market regime string
        """
        # If provider already has a regime classification, use it as a strong hint
        if 'provider_regime' in indicators:
            provider_regime = indicators['provider_regime']
            # Still perform our checks but give weight to provider's regime
            regime_score = self._calculate_regime_scores(indicators)
            
            # If provider regime is in top 2 scores, use it
            sorted_regimes = sorted(regime_score.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_regimes) >= 2:
                if provider_regime in [sorted_regimes[0][0], sorted_regimes[1][0]]:
                    return provider_regime
        
        # Get VIX level and thresholds
        vix_current = indicators.get('vix_current', 15.0)
        vix_high = self.thresholds.get('vix_high', 25.0)
        vix_very_high = self.thresholds.get('vix_very_high', 35.0)
        
        # Get trend indicators
        trend_20d = indicators.get('trend_20d', 0)
        trend_50d = indicators.get('trend_50d', 0)
        trend_strong = self.thresholds.get('trend_strong', 0.12)
        trend_moderate = self.thresholds.get('trend_moderate', 0.05)
        
        # Get volatility
        volatility_20d = indicators.get('volatility_20d', 0.01)
        volatility_high = self.thresholds.get('volatility_high', 0.015)
        
        # Get breadth
        breadth_ratio = indicators.get('breadth_ratio', 0.5)
        breadth_strong = self.thresholds.get('breadth_strong', 0.7)
        breadth_weak = self.thresholds.get('breadth_weak', 0.3)
        
        # Get MA relationship
        ma_relationship = indicators.get('ma_relationship', 0)
        
        # Determine regime based on combined indicators
        if vix_current > vix_very_high:
            # Very high VIX typically indicates a bearish or volatile regime
            if trend_20d < -trend_moderate and trend_50d < -trend_moderate:
                return "bearish"
            else:
                return "volatile"
        
        elif vix_current > vix_high:
            # High VIX with strong downtrend is moderately bearish
            if trend_20d < -trend_moderate:
                return "moderately_bearish"
            # High VIX with strong uptrend is volatile bullish
            elif trend_20d > trend_moderate:
                return "moderately_bullish"
            else:
                return "volatile"
        
        else:
            # Lower VIX - determine based on trend and breadth
            if trend_20d > trend_strong and trend_50d > trend_moderate and breadth_ratio > breadth_strong:
                return "bullish"
            elif trend_20d > trend_moderate and breadth_ratio > breadth_strong * 0.8:
                return "moderately_bullish"
            elif trend_20d < -trend_strong and trend_50d < -trend_moderate and breadth_ratio < breadth_weak:
                return "moderately_bearish"
            elif abs(trend_20d) < trend_moderate / 2 and abs(trend_50d) < trend_moderate / 2:
                return "sideways" if volatility_20d < volatility_high else "neutral"
            else:
                return "neutral"
    
    def _calculate_regime_scores(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate scores for each possible regime.
        
        Args:
            indicators: Dictionary of regime indicators
            
        Returns:
            Dictionary mapping regimes to scores
        """
        scores = {
            "bullish": 0,
            "moderately_bullish": 0,
            "neutral": 0,
            "moderately_bearish": 0,
            "bearish": 0,
            "volatile": 0,
            "sideways": 0
        }
        
        # Extract indicators
        vix_current = indicators.get('vix_current', 15.0)
        trend_20d = indicators.get('trend_20d', 0)
        trend_50d = indicators.get('trend_50d', 0)
        volatility_20d = indicators.get('volatility_20d', 0.01)
        breadth_ratio = indicators.get('breadth_ratio', 0.5)
        ma_relationship = indicators.get('ma_relationship', 0)
        
        # VIX contribution
        if vix_current > self.thresholds.get('vix_very_high', 35.0):
            scores["bearish"] += 2
            scores["volatile"] += 3
        elif vix_current > self.thresholds.get('vix_high', 25.0):
            scores["moderately_bearish"] += 2
            scores["volatile"] += 2
        else:
            scores["bullish"] += 1
            scores["moderately_bullish"] += 1
            scores["neutral"] += 1
            scores["sideways"] += 1
        
        # Trend contribution
        if trend_20d > self.thresholds.get('trend_strong', 0.12):
            scores["bullish"] += 3
            scores["moderately_bullish"] += 1
        elif trend_20d > self.thresholds.get('trend_moderate', 0.05):
            scores["bullish"] += 1
            scores["moderately_bullish"] += 3
        elif trend_20d < -self.thresholds.get('trend_strong', 0.12):
            scores["bearish"] += 3
            scores["moderately_bearish"] += 1
        elif trend_20d < -self.thresholds.get('trend_moderate', 0.05):
            scores["bearish"] += 1
            scores["moderately_bearish"] += 3
        else:
            scores["neutral"] += 2
            scores["sideways"] += 2
        
        # Longer trend contribution
        if trend_50d > self.thresholds.get('trend_moderate', 0.05):
            scores["bullish"] += 2
            scores["moderately_bullish"] += 1
        elif trend_50d < -self.thresholds.get('trend_moderate', 0.05):
            scores["bearish"] += 2
            scores["moderately_bearish"] += 1
        else:
            scores["neutral"] += 1
            scores["sideways"] += 1
        
        # Volatility contribution
        if volatility_20d > self.thresholds.get('volatility_high', 0.015):
            scores["volatile"] += 3
            scores["moderately_bearish"] += 1
            scores["moderately_bullish"] += 1
        else:
            scores["sideways"] += 2
            scores["neutral"] += 1
        
        # Breadth contribution
        if breadth_ratio > self.thresholds.get('breadth_strong', 0.7):
            scores["bullish"] += 2
            scores["moderately_bullish"] += 1
        elif breadth_ratio < self.thresholds.get('breadth_weak', 0.3):
            scores["bearish"] += 2
            scores["moderately_bearish"] += 1
        else:
            scores["neutral"] += 1
        
        # MA relationship contribution
        if ma_relationship > 0.02:
            scores["bullish"] += 1
            scores["moderately_bullish"] += 1
        elif ma_relationship < -0.02:
            scores["bearish"] += 1
            scores["moderately_bearish"] += 1
        else:
            scores["sideways"] += 1
        
        return scores
    
    def classify_historical_regimes(
        self,
        start_date: str,
        end_date: str,
        resolution: str = 'day'
    ) -> Dict[str, str]:
        """
        Classify market regimes for a historical period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            resolution: Resolution of classification ('day', 'week', 'month')
            
        Returns:
            Dictionary mapping dates to regime classifications
        """
        # Check if data provider is available
        if not self.data_provider:
            logger.warning("No data provider available for historical regime classification")
            return {}
        
        try:
            # Get historical data
            historical_data = self.data_provider.get_historical_data(
                symbols=self.market_indices,
                start_date=start_date,
                end_date=end_date,
                timeframe="day"
            )
            
            # Create date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Initialize results
            regime_map = {}
            
            # Group by resolution if needed
            if resolution == 'week':
                # Group by week
                week_groups = {d.isocalendar()[1]: d for d in date_range}
                evaluation_dates = list(week_groups.values())
            elif resolution == 'month':
                # Group by month
                month_groups = {d.month: d for d in date_range}
                evaluation_dates = list(month_groups.values())
            else:
                # Daily resolution
                evaluation_dates = date_range
            
            # Classify each date
            for date in evaluation_dates:
                date_str = date.strftime("%Y-%m-%d")
                
                # Create a temporary snapshot of data until this date
                snapshot_data = {}
                for symbol, df in historical_data.items():
                    snapshot_df = df[df['date'] <= date]
                    if not snapshot_df.empty:
                        snapshot_data[symbol] = snapshot_df
                
                # Skip if insufficient data
                if not all(symbol in snapshot_data for symbol in ['SPY', 'VIX']):
                    continue
                
                # Calculate indicators for this date
                indicators = self._calculate_historical_indicators(snapshot_data, date)
                
                # Determine regime
                regime = self._determine_regime(indicators)
                
                # Store result
                regime_map[date_str] = regime
            
            return regime_map
            
        except Exception as e:
            logger.error(f"Error in historical regime classification: {str(e)}")
            return {}
    
    def _calculate_historical_indicators(
        self,
        historical_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, float]:
        """
        Calculate regime indicators for a historical date.
        
        Args:
            historical_data: Dictionary of historical data by symbol
            date: Target date
            
        Returns:
            Dictionary of regime indicators
        """
        indicators = {}
        date_str = date.strftime("%Y-%m-%d")
        
        # Get SPY data
        spy_data = historical_data.get('SPY', pd.DataFrame())
        
        if not spy_data.empty:
            # Ensure data is sorted by date
            spy_data = spy_data.sort_values('date')
            
            # Calculate returns
            spy_data['return'] = spy_data['close'].pct_change()
            
            # Calculate moving averages
            spy_data['MA20'] = spy_data['close'].rolling(window=20).mean()
            spy_data['MA50'] = spy_data['close'].rolling(window=50).mean()
            
            # Filter to data up to the target date
            spy_data = spy_data[spy_data['date'] <= date]
            
            if len(spy_data) >= 50:  # Ensure enough data for calculations
                # Calculate trend indicators
                recent_ma20 = spy_data['MA20'].dropna().tail(20)
                if len(recent_ma20) >= 10:
                    x = np.arange(len(recent_ma20))
                    y = recent_ma20.values
                    slope, _ = np.polyfit(x, y, 1)
                    trend_20d = slope / recent_ma20.iloc[0]
                    indicators['trend_20d'] = trend_20d
                
                recent_ma50 = spy_data['MA50'].dropna().tail(50)
                if len(recent_ma50) >= 25:
                    x = np.arange(len(recent_ma50))
                    y = recent_ma50.values
                    slope, _ = np.polyfit(x, y, 1)
                    trend_50d = slope / recent_ma50.iloc[0]
                    indicators['trend_50d'] = trend_50d
                
                # Calculate volatility
                volatility_20d = spy_data['return'].tail(20).std() * np.sqrt(252)
                indicators['volatility_20d'] = volatility_20d
                
                # Moving average relationship
                last_row = spy_data.iloc[-1]
                if not pd.isna(last_row['MA20']) and not pd.isna(last_row['MA50']):
                    ma_relationship = last_row['MA20'] / last_row['MA50'] - 1
                    indicators['ma_relationship'] = ma_relationship
        
        # Get VIX data
        vix_data = historical_data.get('VIX', pd.DataFrame())
        
        if not vix_data.empty:
            # Ensure data is sorted by date
            vix_data = vix_data.sort_values('date')
            
            # Filter to data up to the target date
            vix_data = vix_data[vix_data['date'] <= date]
            
            if not vix_data.empty:
                # Current VIX level
                current_vix = vix_data['close'].iloc[-1]
                indicators['vix_current'] = current_vix
                
                if len(vix_data) >= 20:
                    # VIX percentile
                    vix_percentile = np.percentile(vix_data['close'], 75)
                    indicators['vix_percentile'] = vix_percentile
                    
                    # VIX trend
                    if len(vix_data) >= 5:
                        vix_change_5d = vix_data['close'].pct_change(5).iloc[-1]
                        indicators['vix_change_5d'] = vix_change_5d
        
        # Calculate breadth indicators
        breadth_ratio = 0.5  # Default value
        
        # Add other index data for breadth calculation if available
        indices = ['QQQ', 'IWM', 'DIA']
        returns = []
        
        for idx in indices:
            if idx in historical_data:
                idx_data = historical_data[idx]
                idx_data = idx_data.sort_values('date')
                idx_data = idx_data[idx_data['date'] <= date]
                
                if not idx_data.empty and len(idx_data) >= 2:
                    latest_close = idx_data['close'].iloc[-1]
                    prev_close = idx_data['close'].iloc[-2]
                    
                    if prev_close > 0:
                        daily_return = (latest_close / prev_close) - 1
                        returns.append(daily_return)
        
        # Calculate breadth ratio based on positive vs negative index returns
        if returns:
            positive_returns = sum(1 for r in returns if r > 0)
            breadth_ratio = positive_returns / len(returns)
        
        indicators['breadth_ratio'] = breadth_ratio
        
        return indicators 