#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Analysis Module for Complex Spread Strategies

This module provides volatility and market analysis functionality
specifically designed for complex options spread strategies like 
iron condors, butterflies, etc.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm, percentileofscore

logger = logging.getLogger(__name__)

class ComplexSpreadMarketAnalyzer:
    """
    Market analyzer for complex option spread strategies.
    
    This class provides specialized analysis for identifying optimal market conditions
    for complex spread strategies including iron condors, butterflies, and other
    multi-leg option spreads.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the market analyzer with strategy parameters.
        
        Args:
            parameters: Dictionary of configuration parameters
        """
        self.params = parameters or {}
        self.historical_volatility_cache = {}
        self.iv_percentile_cache = {}
        self.logger = logger
        
    def calculate_historical_volatility(self, price_data: pd.DataFrame, period: int = 20, 
                                        return_type: str = 'latest') -> Union[float, pd.Series]:
        """
        Calculate historical volatility from price data.
        
        Args:
            price_data: DataFrame with price data (must have 'close' column)
            period: Period for volatility calculation in trading days
            return_type: 'latest' for most recent value, 'series' for full series
            
        Returns:
            Either the latest volatility value or a series of volatility values
        """
        try:
            # Verify the data has required columns
            if not isinstance(price_data, pd.DataFrame) or 'close' not in price_data.columns:
                self.logger.error("Invalid price data format for volatility calculation")
                return 0.0 if return_type == 'latest' else pd.Series()
            
            # Calculate log returns
            log_returns = np.log(price_data['close'] / price_data['close'].shift(1))
            
            # Calculate rolling standard deviation
            rolling_std = log_returns.rolling(window=period).std()
            
            # Convert to annualized volatility (252 trading days)
            annualized_vol = rolling_std * np.sqrt(252)
            
            if return_type == 'latest':
                return annualized_vol.iloc[-1] if not pd.isna(annualized_vol.iloc[-1]) else 0.0
            else:
                return annualized_vol
                
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility: {e}")
            return 0.0 if return_type == 'latest' else pd.Series()
            
    def calculate_implied_volatility_surface(self, option_chain: Any, 
                                           current_price: float) -> Dict[str, Any]:
        """
        Calculate implied volatility surface from option chain data.
        
        Args:
            option_chain: Option chain data
            current_price: Current price of the underlying
            
        Returns:
            Dictionary with implied volatility data organized by strike and expiration
        """
        iv_surface = {
            'call': {},
            'put': {},
            'skew': {},
            'term_structure': {},
            'atm_iv': {}
        }
        
        try:
            # Check if we have valid option chain data
            if not option_chain:
                return iv_surface
                
            # Process different option chain formats
            if isinstance(option_chain, pd.DataFrame):
                return self._process_dataframe_option_chain(option_chain, current_price)
            elif isinstance(option_chain, dict):
                return self._process_dict_option_chain(option_chain, current_price)
            else:
                self.logger.warning(f"Unsupported option chain format: {type(option_chain)}")
                return iv_surface
                
        except Exception as e:
            self.logger.error(f"Error calculating IV surface: {e}")
            return iv_surface
            
    def _process_dataframe_option_chain(self, chain: pd.DataFrame, 
                                      current_price: float) -> Dict[str, Any]:
        """
        Process a DataFrame-formatted option chain.
        
        Args:
            chain: Option chain as DataFrame
            current_price: Current price of the underlying
            
        Returns:
            Dictionary with implied volatility data
        """
        iv_surface = {
            'call': {},
            'put': {},
            'skew': {},
            'term_structure': {},
            'atm_iv': {}
        }
        
        try:
            # Check if we have the required columns
            required_cols = ['strike', 'expiration', 'option_type', 'implied_volatility']
            if not all(col in chain.columns for col in required_cols):
                missing = [col for col in required_cols if col not in chain.columns]
                self.logger.warning(f"Missing columns in option chain: {missing}")
                return iv_surface
                
            # Group by expiration date
            for expiry, expiry_group in chain.groupby('expiration'):
                exp_key = str(expiry)
                
                # Process calls
                calls = expiry_group[expiry_group['option_type'].str.lower() == 'call']
                if not calls.empty:
                    iv_surface['call'][exp_key] = calls.set_index('strike')['implied_volatility'].to_dict()
                    
                # Process puts
                puts = expiry_group[expiry_group['option_type'].str.lower() == 'put']
                if not puts.empty:
                    iv_surface['put'][exp_key] = puts.set_index('strike')['implied_volatility'].to_dict()
                    
                # Find the closest strike to current price for ATM IV
                if not calls.empty and not puts.empty:
                    closest_strike = calls['strike'].iloc[(calls['strike'] - current_price).abs().argsort()[0]]
                    closest_calls = calls[calls['strike'] == closest_strike]
                    closest_puts = puts[puts['strike'] == closest_strike]
                    
                    if not closest_calls.empty and not closest_puts.empty:
                        call_iv = closest_calls['implied_volatility'].iloc[0]
                        put_iv = closest_puts['implied_volatility'].iloc[0]
                        iv_surface['atm_iv'][exp_key] = (call_iv + put_iv) / 2
                        
            # Calculate volatility skew
            for exp_key in iv_surface['call'].keys():
                if exp_key in iv_surface['put']:
                    call_skew = {}
                    for strike in iv_surface['call'][exp_key].keys():
                        if strike in iv_surface['put'][exp_key]:
                            call_iv = iv_surface['call'][exp_key][strike]
                            put_iv = iv_surface['put'][exp_key][strike]
                            skew = put_iv - call_iv
                            call_skew[strike] = skew
                    iv_surface['skew'][exp_key] = call_skew
                    
            # Calculate term structure using ATM IV
            atm_iv_list = [(exp, iv) for exp, iv in iv_surface['atm_iv'].items()]
            atm_iv_list.sort(key=lambda x: x[0])  # Sort by expiration
            
            if len(atm_iv_list) > 1:
                for i in range(len(atm_iv_list) - 1):
                    curr_exp, curr_iv = atm_iv_list[i]
                    next_exp, next_iv = atm_iv_list[i + 1]
                    term_diff = next_iv - curr_iv
                    iv_surface['term_structure'][f"{curr_exp}_to_{next_exp}"] = term_diff
                    
            return iv_surface
                
        except Exception as e:
            self.logger.error(f"Error processing DataFrame option chain: {e}")
            return iv_surface
            
    def _process_dict_option_chain(self, chain: Dict[str, Any], 
                                 current_price: float) -> Dict[str, Any]:
        """
        Process a dictionary-formatted option chain.
        
        Args:
            chain: Option chain as dictionary
            current_price: Current price of the underlying
            
        Returns:
            Dictionary with implied volatility data
        """
        # Placeholder for custom dictionary format processing
        # This would need to be customized based on your specific dictionary format
        self.logger.warning("Dictionary option chain format processing not implemented")
        return {
            'call': {},
            'put': {},
            'skew': {},
            'term_structure': {},
            'atm_iv': {}
        }
        
    def calculate_iv_percentile(self, symbol: str, current_iv: float, 
                              iv_history: Optional[pd.Series] = None) -> float:
        """
        Calculate IV percentile relative to historical IV.
        
        Args:
            symbol: Symbol for caching
            current_iv: Current implied volatility
            iv_history: Historical implied volatility series (optional)
            
        Returns:
            IV percentile (0-100)
        """
        # Check cache first
        cache_key = f"{symbol}_iv_percentile"
        if cache_key in self.iv_percentile_cache and iv_history is None:
            cache_entry = self.iv_percentile_cache[cache_key]
            # Use cache if less than 4 hours old
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < 14400:
                return cache_entry['value']
                
        try:
            # If we have historical IV data, use it to calculate percentile
            if iv_history is not None and len(iv_history) > 30:
                # Remove NaN values
                clean_history = iv_history.dropna()
                
                if len(clean_history) < 30:
                    self.logger.warning(f"Insufficient clean IV history for {symbol}")
                    return 50.0  # Default to middle percentile
                    
                percentile = percentileofscore(clean_history, current_iv)
                
                # Update cache
                self.iv_percentile_cache[cache_key] = {
                    'value': percentile,
                    'timestamp': datetime.now()
                }
                
                return percentile
            else:
                self.logger.warning(f"Insufficient IV history for {symbol}")
                return 50.0  # Default to middle percentile
                
        except Exception as e:
            self.logger.error(f"Error calculating IV percentile: {e}")
            return 50.0  # Default to middle percentile
            
    def identify_trading_range(self, price_data: pd.DataFrame, 
                             lookback_period: int = 20) -> Dict[str, Any]:
        """
        Identify the trading range for a symbol.
        
        Args:
            price_data: DataFrame with price data
            lookback_period: Period to analyze for range identification
            
        Returns:
            Dictionary with trading range data
        """
        range_data = {
            'is_range_bound': False,
            'upper_bound': 0,
            'lower_bound': 0,
            'range_width_pct': 0,
            'days_in_range': 0,
            'touch_points': 0,
            'confidence': 0
        }
        
        try:
            # Need sufficient data
            if not isinstance(price_data, pd.DataFrame) or len(price_data) < lookback_period:
                return range_data
                
            # Get the last n periods
            recent_data = price_data.iloc[-lookback_period:]
            
            # Calculate potential range bounds
            upper_bound = recent_data['high'].max()
            lower_bound = recent_data['low'].min()
            
            # Calculate range width as percentage
            mid_price = (upper_bound + lower_bound) / 2
            range_width_pct = (upper_bound - lower_bound) / mid_price
            
            # Calculate how many days price stayed within 90% of the range
            range_90pct_upper = upper_bound - (upper_bound - lower_bound) * 0.05
            range_90pct_lower = lower_bound + (upper_bound - lower_bound) * 0.05
            
            days_in_range = sum((recent_data['high'] <= range_90pct_upper) & 
                               (recent_data['low'] >= range_90pct_lower))
            
            # Calculate touch points (price approaching boundaries)
            upper_touches = sum(recent_data['high'] >= upper_bound * 0.98)
            lower_touches = sum(recent_data['low'] <= lower_bound * 1.02)
            touch_points = upper_touches + lower_touches
            
            # Determine if the stock is range-bound
            # Criteria: range width < 15%, spent >70% time in range, at least 3 touch points
            range_width_threshold = self.params.get('range_width_threshold', 0.15)
            range_time_threshold = self.params.get('range_time_threshold', 0.7)
            min_touch_points = self.params.get('min_touch_points', 3)
            
            is_range_bound = (
                range_width_pct < range_width_threshold and
                days_in_range / lookback_period >= range_time_threshold and
                touch_points >= min_touch_points
            )
            
            # Calculate confidence score
            confidence = min(
                1.0,
                (range_width_threshold - range_width_pct) / range_width_threshold * 0.4 +
                (days_in_range / lookback_period) * 0.4 +
                min(touch_points / min_touch_points, 1.0) * 0.2
            ) * 100
            
            # Populate result
            range_data.update({
                'is_range_bound': is_range_bound,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'range_width_pct': range_width_pct,
                'days_in_range': days_in_range,
                'touch_points': touch_points,
                'confidence': confidence
            })
            
            return range_data
            
        except Exception as e:
            self.logger.error(f"Error identifying trading range: {e}")
            return range_data
            
    def analyze_market_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market regime for suitability for complex spread strategies.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Dictionary with market regime analysis
        """
        regime_data = {
            'is_suitable': False,
            'regime': 'unknown',
            'volatility_regime': 'unknown',
            'trend_strength': 0,
            'range_strength': 0,
            'confidence': 0,
            'recommendation': 'neutral'
        }
        
        try:
            # Need sufficient data
            if not isinstance(price_data, pd.DataFrame) or len(price_data) < 60:
                return regime_data
                
            # Calculate volatility 
            hist_vol = self.calculate_historical_volatility(price_data, period=20)
            hist_vol_60 = self.calculate_historical_volatility(price_data, period=60)
            
            # Calculate trend metrics
            returns = price_data['close'].pct_change()
            rolling_std_20 = returns.rolling(window=20).std()
            
            # Calculate ADX-like trend strength (simplified)
            high_low_range = price_data['high'] - price_data['low']
            tr = pd.concat([
                high_low_range,
                abs(price_data['high'] - price_data['close'].shift(1)),
                abs(price_data['low'] - price_data['close'].shift(1))
            ], axis=1).max(axis=1)
            
            atr_14 = tr.rolling(window=14).mean()
            
            # Simple trend detection using moving averages
            ma_20 = price_data['close'].rolling(window=20).mean()
            ma_50 = price_data['close'].rolling(window=50).mean()
            
            # Trend direction and strength
            trend_direction = 1 if ma_20.iloc[-1] > ma_50.iloc[-1] else -1
            trend_strength = abs(ma_20.iloc[-1] - ma_50.iloc[-1]) / price_data['close'].iloc[-1]
            
            # Determine volatility regime
            vol_ratio = hist_vol / hist_vol_60
            
            if hist_vol < 0.15:
                vol_regime = 'low'
            elif hist_vol > 0.30:
                vol_regime = 'high'
            else:
                vol_regime = 'medium'
                
            # Check for expanding or contracting volatility
            if vol_ratio > 1.2:
                vol_regime += '_expanding'
            elif vol_ratio < 0.8:
                vol_regime += '_contracting'
            else:
                vol_regime += '_stable'
                
            # Get range analysis
            range_data = self.identify_trading_range(price_data)
            
            # Determine overall regime
            if range_data['is_range_bound'] and vol_regime.startswith('low') or vol_regime.startswith('medium'):
                regime = 'range_bound'
                regime_strength = range_data['confidence'] / 100
            elif trend_strength > 0.05 and not vol_regime.startswith('high'):
                regime = 'trending' + ('_up' if trend_direction > 0 else '_down')
                regime_strength = min(trend_strength * 10, 1.0)
            elif vol_regime.startswith('high'):
                regime = 'volatile'
                regime_strength = min(hist_vol * 2, 1.0)
            else:
                regime = 'neutral'
                regime_strength = 0.5
                
            # Determine strategy recommendations
            recommendation = 'neutral'
            
            # For iron condors: favor range-bound, low-to-medium volatility
            iron_condor_score = 0
            if regime == 'range_bound':
                iron_condor_score += 40
            if vol_regime.startswith('low') or vol_regime.startswith('medium_contracting'):
                iron_condor_score += 40
            if not vol_regime.startswith('high'):
                iron_condor_score += 20
                
            # For butterflies: favor range-bound, precise price targets
            butterfly_score = 0
            if regime == 'range_bound' and range_data['range_width_pct'] < 0.1:
                butterfly_score += 50
            if vol_regime.startswith('low'):
                butterfly_score += 30
            if range_data['touch_points'] >= 5:
                butterfly_score += 20
                
            # Determine confidence and recommendation
            if iron_condor_score >= 60 and iron_condor_score >= butterfly_score:
                recommendation = 'iron_condor'
                confidence = iron_condor_score
            elif butterfly_score >= 60:
                recommendation = 'butterfly'
                confidence = butterfly_score
            else:
                recommendation = 'neutral'
                confidence = max(iron_condor_score, butterfly_score)
                
            # Check if market is suitable for complex spreads
            is_suitable = confidence >= 60
            
            # Populate result
            regime_data.update({
                'is_suitable': is_suitable,
                'regime': regime,
                'volatility_regime': vol_regime,
                'trend_strength': trend_strength,
                'range_strength': range_data['confidence'] / 100,
                'confidence': confidence / 100,
                'recommendation': recommendation
            })
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {e}")
            return regime_data
            
    def detect_price_levels(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect important price levels for strike selection.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Dictionary with price level data
        """
        price_levels = {
            'support': [],
            'resistance': [],
            'value_area_high': 0,
            'value_area_low': 0,
            'pivot_points': {}
        }
        
        try:
            # Need sufficient data
            if not isinstance(price_data, pd.DataFrame) or len(price_data) < 60:
                return price_levels
                
            # Find support and resistance levels using recent highs and lows
            recent_highs = price_data['high'].rolling(window=5).max()
            recent_lows = price_data['low'].rolling(window=5).min()
            
            # Find local maxima
            resistance_candidates = []
            for i in range(5, len(recent_highs) - 5):
                if (recent_highs.iloc[i] > recent_highs.iloc[i-5:i].max() and 
                    recent_highs.iloc[i] > recent_highs.iloc[i+1:i+6].max()):
                    resistance_candidates.append(recent_highs.iloc[i])
                    
            # Find local minima
            support_candidates = []
            for i in range(5, len(recent_lows) - 5):
                if (recent_lows.iloc[i] < recent_lows.iloc[i-5:i].min() and 
                    recent_lows.iloc[i] < recent_lows.iloc[i+1:i+6].min()):
                    support_candidates.append(recent_lows.iloc[i])
                    
            # Cluster nearby levels to avoid duplication
            if resistance_candidates:
                resistance = self._cluster_price_levels(resistance_candidates)
                price_levels['resistance'] = resistance
                
            if support_candidates:
                support = self._cluster_price_levels(support_candidates)
                price_levels['support'] = support
                
            # Calculate value area (simplified approach)
            recent_closing = price_data['close'].iloc[-20:]
            price_levels['value_area_high'] = recent_closing.quantile(0.7)
            price_levels['value_area_low'] = recent_closing.quantile(0.3)
            
            # Calculate pivot points
            last_day = price_data.iloc[-1]
            high, low, close = last_day['high'], last_day['low'], last_day['close']
            
            # Standard pivot point
            pivot = (high + low + close) / 3
            
            # Support and resistance levels
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            
            price_levels['pivot_points'] = {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                's1': s1,
                's2': s2
            }
            
            return price_levels
            
        except Exception as e:
            self.logger.error(f"Error detecting price levels: {e}")
            return price_levels
            
    def _cluster_price_levels(self, levels: List[float], threshold_pct: float = 0.01) -> List[float]:
        """
        Cluster nearby price levels to avoid duplication.
        
        Args:
            levels: List of price levels
            threshold_pct: Threshold as percentage for clustering
            
        Returns:
            List of clustered price levels
        """
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Initialize clusters
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        # Cluster nearby levels
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            prev_level = current_cluster[-1]
            
            # If the level is close to the previous one, add to the current cluster
            if (current_level - prev_level) / prev_level <= threshold_pct:
                current_cluster.append(current_level)
            else:
                # Finalize the current cluster and start a new one
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [current_level]
                
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
            
        return clusters
