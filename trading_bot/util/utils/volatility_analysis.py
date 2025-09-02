#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Analysis Utilities

This module provides utilities for analyzing volatility surfaces,
term structures, and implied volatility environments.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_expiration_days(expiration_dates: List[Union[str, datetime]], base_date: Optional[datetime] = None) -> Dict[Union[str, datetime], int]:
    """
    Calculate days to expiration for a list of expiration dates.
    
    Args:
        expiration_dates: List of expiration dates
        base_date: Base date for calculation (defaults to today)
        
    Returns:
        Dictionary mapping expiration dates to days to expiration
    """
    try:
        if base_date is None:
            base_date = datetime.now().date()
        elif isinstance(base_date, datetime):
            base_date = base_date.date()
        
        result = {}
        
        for expiration in expiration_dates:
            if isinstance(expiration, str):
                # Parse date string (assuming format YYYY-MM-DD)
                try:
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Could not parse expiration date: {expiration}")
                    continue
            elif isinstance(expiration, datetime):
                exp_date = expiration.date()
            else:
                exp_date = expiration
            
            # Calculate days to expiration
            dte = (exp_date - base_date).days
            
            # Store in result
            result[expiration] = max(0, dte)
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating days to expiration: {e}")
        return {}

def analyze_volatility_surface(
    option_chains: Dict[Union[str, datetime], pd.DataFrame],
    underlying_price: float
) -> Dict[str, Any]:
    """
    Analyze the volatility surface across different expirations and strikes.
    
    Args:
        option_chains: Dictionary mapping expiration dates to option chain DataFrames
        underlying_price: Current price of the underlying
        
    Returns:
        Dictionary with volatility surface analysis
    """
    try:
        # Initialize result structure
        result = {
            'expiration_vols': {},     # IV by expiration
            'strike_vols': {},         # IV by strike
            'vol_surface': {},         # Full surface (IV by expiration and strike)
            'skew': {},                # Skew metrics by expiration
            'term_structure': {},      # Term structure metrics
            'summary': {}              # Summary statistics
        }
        
        # Get days to expiration for each expiration
        expirations_days = get_expiration_days(list(option_chains.keys()))
        
        # Sort expirations by days to expiration
        sorted_expirations = sorted(expirations_days.items(), key=lambda x: x[1])
        
        # Collect IV data across expirations and strikes
        for expiration, days in sorted_expirations:
            chain = option_chains[expiration]
            
            # Skip if no implied volatility data
            if 'implied_volatility' not in chain.columns or 'strike' not in chain.columns:
                logger.warning(f"Option chain for {expiration} missing implied volatility or strike data")
                continue
            
            # Get unique strikes
            strikes = sorted(chain['strike'].unique())
            
            # Filter by call/put if available
            calls = chain[chain['option_type'] == 'call'] if 'option_type' in chain.columns else chain
            puts = chain[chain['option_type'] == 'put'] if 'option_type' in chain.columns else pd.DataFrame()
            
            # Create volatility data for this expiration
            strikes_data = {}
            
            for strike in strikes:
                strike_data = {}
                
                # Get call IV at this strike
                call_data = calls[calls['strike'] == strike]
                if not call_data.empty and 'implied_volatility' in call_data.columns:
                    strike_data['call_iv'] = call_data['implied_volatility'].iloc[0]
                
                # Get put IV at this strike
                if not puts.empty:
                    put_data = puts[puts['strike'] == strike]
                    if not put_data.empty and 'implied_volatility' in put_data.columns:
                        strike_data['put_iv'] = put_data['implied_volatility'].iloc[0]
                
                # Store combined IV (average of call and put if both available)
                if 'call_iv' in strike_data and 'put_iv' in strike_data:
                    strike_data['avg_iv'] = (strike_data['call_iv'] + strike_data['put_iv']) / 2
                elif 'call_iv' in strike_data:
                    strike_data['avg_iv'] = strike_data['call_iv']
                elif 'put_iv' in strike_data:
                    strike_data['avg_iv'] = strike_data['put_iv']
                else:
                    continue
                
                strikes_data[strike] = strike_data
            
            # Skip if no valid volatility data
            if not strikes_data:
                continue
            
            # Store in result
            result['vol_surface'][expiration] = {
                'days': days,
                'strikes_data': strikes_data
            }
            
            # Calculate average IV for this expiration
            ivs = [data['avg_iv'] for data in strikes_data.values() if 'avg_iv' in data]
            if ivs:
                result['expiration_vols'][expiration] = {
                    'days': days,
                    'avg_iv': np.mean(ivs),
                    'min_iv': np.min(ivs),
                    'max_iv': np.max(ivs)
                }
            
            # Find ATM strike
            atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))
            
            # Calculate skew metrics
            skew_metrics = {}
            
            # 1. Put-Call Skew
            if 'call_iv' in strikes_data.get(atm_strike, {}) and 'put_iv' in strikes_data.get(atm_strike, {}):
                put_call_skew = strikes_data[atm_strike]['put_iv'] - strikes_data[atm_strike]['call_iv']
                skew_metrics['put_call_skew'] = put_call_skew
            
            # 2. 25-Delta Skew (simplified)
            # Find strikes approximately 5% OTM on each side
            lower_strike = max((s for s in strikes if s < underlying_price * 0.95), default=None)
            upper_strike = min((s for s in strikes if s > underlying_price * 1.05), default=None)
            
            if lower_strike and upper_strike:
                lower_iv = strikes_data.get(lower_strike, {}).get('avg_iv')
                upper_iv = strikes_data.get(upper_strike, {}).get('avg_iv')
                atm_iv = strikes_data.get(atm_strike, {}).get('avg_iv')
                
                if lower_iv and upper_iv and atm_iv:
                    wing_skew = (lower_iv - upper_iv)
                    skew_metrics['wing_skew'] = wing_skew
                    
                    # Skew to ATM
                    skew_metrics['downside_skew'] = (lower_iv - atm_iv)
                    skew_metrics['upside_skew'] = (upper_iv - atm_iv)
                    
                    # Smile curvature
                    skew_metrics['smile_curvature'] = (lower_iv + upper_iv) / 2 - atm_iv
            
            # Store skew metrics
            if skew_metrics:
                result['skew'][expiration] = {
                    'days': days,
                    'metrics': skew_metrics
                }
        
        # Analyze term structure
        if result['expiration_vols']:
            # Sort by days to expiration
            term_structure = sorted(
                [(exp, data['days'], data['avg_iv']) for exp, data in result['expiration_vols'].items()],
                key=lambda x: x[1]
            )
            
            # Extract data series
            expirations = [item[0] for item in term_structure]
            days = [item[1] for item in term_structure]
            ivs = [item[2] for item in term_structure]
            
            # Store term structure data
            result['term_structure']['data'] = {
                'expirations': expirations,
                'days': days,
                'ivs': ivs
            }
            
            # Calculate term structure metrics
            if len(days) >= 2:
                # Calculate slope between first two expirations
                slope_30d = (ivs[1] - ivs[0]) / (days[1] - days[0]) if days[1] != days[0] else 0
                
                # Calculate slope between first and last expiration
                slope_full = (ivs[-1] - ivs[0]) / (days[-1] - days[0]) if days[-1] != days[0] else 0
                
                # Determine if in contango or backwardation
                if slope_full > 0.0001:  # Small positive threshold
                    term_shape = 'contango'
                elif slope_full < -0.0001:  # Small negative threshold
                    term_shape = 'backwardation'
                else:
                    term_shape = 'flat'
                
                # Store metrics
                result['term_structure']['metrics'] = {
                    'slope_30d': slope_30d,
                    'slope_full': slope_full,
                    'term_shape': term_shape
                }
        
        # Calculate summary statistics
        if result['vol_surface']:
            # Collect all IVs
            all_ivs = []
            for exp_data in result['vol_surface'].values():
                for strike_data in exp_data['strikes_data'].values():
                    if 'avg_iv' in strike_data:
                        all_ivs.append(strike_data['avg_iv'])
            
            if all_ivs:
                result['summary'] = {
                    'avg_iv': np.mean(all_ivs),
                    'median_iv': np.median(all_ivs),
                    'min_iv': np.min(all_ivs),
                    'max_iv': np.max(all_ivs),
                    'std_iv': np.std(all_ivs)
                }
                
                # Add term structure summary if available
                if 'metrics' in result['term_structure']:
                    result['summary']['term_shape'] = result['term_structure']['metrics']['term_shape']
                
                # Add skew summary if available
                if result['skew']:
                    # Average wing skew across all expirations
                    wing_skews = [exp_data['metrics'].get('wing_skew') for exp_data in result['skew'].values()
                                 if 'wing_skew' in exp_data['metrics']]
                    
                    if wing_skews:
                        result['summary']['avg_wing_skew'] = np.mean(wing_skews)
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing volatility surface: {e}")
        return {'error': str(e)}

def select_strategy_based_on_vol_environment(
    vol_analysis: Dict[str, Any],
    iv_rank: float,
    historical_data: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Select optimal options strategy based on volatility environment.
    
    Args:
        vol_analysis: Output from analyze_volatility_surface
        iv_rank: Current IV Rank (0-100)
        historical_data: Historical price data (optional)
        
    Returns:
        Dictionary with strategy recommendations
    """
    try:
        recommendations = []
        
        # Extract key metrics from volatility analysis
        if 'summary' not in vol_analysis:
            logger.warning("Incomplete volatility analysis data")
            return {'error': 'Incomplete volatility analysis data'}
        
        summary = vol_analysis['summary']
        term_shape = summary.get('term_shape', 'unknown')
        avg_wing_skew = summary.get('avg_wing_skew', 0)
        
        # Determine market regime based on historical data
        market_regime = 'unknown'
        if historical_data is not None:
            # Simple trend detection
            if 'close' in historical_data.columns:
                prices = historical_data['close']
                if len(prices) >= 20:
                    sma20 = prices.rolling(20).mean()
                    sma50 = prices.rolling(50).mean() if len(prices) >= 50 else None
                    
                    if sma50 is not None:
                        if prices.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]:
                            market_regime = 'uptrend'
                        elif prices.iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1]:
                            market_regime = 'downtrend'
                        else:
                            market_regime = 'range_bound'
                    else:
                        market_regime = 'uptrend' if prices.iloc[-1] > sma20.iloc[-1] else 'downtrend'
        
        # Calendar Spread specific strategies
        cal_spread_score = 0
        cal_spread_rationale = []
        
        # 1. Term Structure - Contango is ideal for calendar spreads
        if term_shape == 'contango':
            cal_spread_score += 30
            cal_spread_rationale.append("Volatility term structure in contango (upward sloping)")
        elif term_shape == 'flat':
            cal_spread_score += 15
            cal_spread_rationale.append("Flat volatility term structure")
        else:  # backwardation
            cal_spread_score -= 10
            cal_spread_rationale.append("Volatility term structure in backwardation (downward sloping)")
        
        # 2. IV Rank - Mid-range IV rank is ideal
        if 30 <= iv_rank <= 60:
            cal_spread_score += 30
            cal_spread_rationale.append(f"Moderate IV rank ({iv_rank:.1f})")
        elif iv_rank > 60:
            cal_spread_score += 15
            cal_spread_rationale.append(f"Elevated IV rank ({iv_rank:.1f})")
        else:
            cal_spread_score += 5
            cal_spread_rationale.append(f"Low IV rank ({iv_rank:.1f})")
        
        # 3. Market Regime - Range-bound is ideal
        if market_regime == 'range_bound':
            cal_spread_score += 20
            cal_spread_rationale.append("Range-bound market")
        elif market_regime in ['uptrend', 'downtrend']:
            cal_spread_score += 10
            cal_spread_rationale.append(f"{market_regime.capitalize()} market")
        
        # 4. Skew - Low skew is better for standard calendar
        if abs(avg_wing_skew) < 0.05:
            cal_spread_score += 20
            cal_spread_rationale.append("Low volatility skew")
        elif abs(avg_wing_skew) < 0.1:
            cal_spread_score += 10
            cal_spread_rationale.append("Moderate volatility skew")
        else:
            cal_spread_score += 5
            cal_spread_rationale.append("High volatility skew")
        
        # Add calendar spread recommendation
        recommendations.append({
            'strategy': 'calendar_spread',
            'score': cal_spread_score,
            'max_score': 100,
            'rationale': cal_spread_rationale,
            'optimizations': get_calendar_spread_optimizations(vol_analysis, iv_rank, market_regime)
        })
        
        # Add other strategies for comparison
        if iv_rank > 60:
            # High IV - good for premium selling strategies
            recommendations.append({
                'strategy': 'iron_condor',
                'score': min(100, 60 + (iv_rank - 60) * 0.8),
                'max_score': 100,
                'rationale': [
                    f"High IV rank ({iv_rank:.1f})",
                    "Good environment for premium selling"
                ]
            })
        elif iv_rank < 30:
            # Low IV - good for long premium strategies
            recommendations.append({
                'strategy': 'long_straddle',
                'score': min(100, 60 + (30 - iv_rank) * 1.5),
                'max_score': 100,
                'rationale': [
                    f"Low IV rank ({iv_rank:.1f})",
                    "Good environment for buying premium"
                ]
            })
        
        # Sort recommendations by score
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        return {
            'recommendations': recommendations,
            'vol_environment': {
                'iv_rank': iv_rank,
                'term_structure': term_shape,
                'market_regime': market_regime
            }
        }
    
    except Exception as e:
        logger.error(f"Error selecting strategy based on vol environment: {e}")
        return {'error': str(e)}

def get_calendar_spread_optimizations(
    vol_analysis: Dict[str, Any],
    iv_rank: float,
    market_regime: str
) -> List[Dict[str, Any]]:
    """
    Get optimizations for calendar spread based on vol environment.
    
    Args:
        vol_analysis: Output from analyze_volatility_surface
        iv_rank: Current IV Rank (0-100)
        market_regime: Market regime string
        
    Returns:
        List of optimization recommendations
    """
    optimizations = []
    
    # Extract skew data if available
    skew_data = None
    if 'skew' in vol_analysis and vol_analysis['skew']:
        # Use first expiration for simplicity
        first_exp = next(iter(vol_analysis['skew']))
        skew_data = vol_analysis['skew'][first_exp]['metrics']
    
    # 1. Strike Selection
    if skew_data and 'wing_skew' in skew_data:
        wing_skew = skew_data['wing_skew']
        if wing_skew > 0.05:  # Puts more expensive than calls
            optimizations.append({
                'type': 'strike_selection',
                'recommendation': 'Consider higher strikes',
                'rationale': 'Put skew favors moving strikes higher',
                'implementation': 'Use +1 strike bias from ATM'
            })
        elif wing_skew < -0.05:  # Calls more expensive than puts
            optimizations.append({
                'type': 'strike_selection',
                'recommendation': 'Consider lower strikes',
                'rationale': 'Call skew favors moving strikes lower',
                'implementation': 'Use -1 strike bias from ATM'
            })
        else:
            optimizations.append({
                'type': 'strike_selection',
                'recommendation': 'Use ATM strikes',
                'rationale': 'Minimal skew suggests ATM optimal',
                'implementation': 'Use ATM strikes (0 bias)'
            })
    
    # 2. DTE Selection
    if 'term_structure' in vol_analysis and 'metrics' in vol_analysis['term_structure']:
        term_metrics = vol_analysis['term_structure']['metrics']
        if term_metrics['term_shape'] == 'contango' and term_metrics.get('slope_30d', 0) > 0.001:
            # Strong contango - wider DTE spread beneficial
            optimizations.append({
                'type': 'dte_selection',
                'recommendation': 'Use wider DTE spread',
                'rationale': 'Strong contango term structure',
                'implementation': 'Use short leg DTE 7-14, long leg DTE 60-90'
            })
        elif term_metrics['term_shape'] == 'backwardation':
            # Backwardation - narrower DTE spread
            optimizations.append({
                'type': 'dte_selection',
                'recommendation': 'Use narrower DTE spread',
                'rationale': 'Backwardation term structure',
                'implementation': 'Use short leg DTE 14-21, long leg DTE 30-45'
            })
        else:
            # Flat or mild contango
            optimizations.append({
                'type': 'dte_selection',
                'recommendation': 'Use standard DTE spread',
                'rationale': 'Normal term structure',
                'implementation': 'Use short leg DTE 14-21, long leg DTE 45-60'
            })
    
    # 3. Directional Bias
    if market_regime == 'uptrend':
        optimizations.append({
            'type': 'directional_bias',
            'recommendation': 'Consider bullish calendar spread',
            'rationale': 'Uptrending market',
            'implementation': 'Use slightly OTM calls or shift to call diagonal spread'
        })
    elif market_regime == 'downtrend':
        optimizations.append({
            'type': 'directional_bias',
            'recommendation': 'Consider bearish calendar spread',
            'rationale': 'Downtrending market',
            'implementation': 'Use slightly OTM puts or shift to put diagonal spread'
        })
    else:
        optimizations.append({
            'type': 'directional_bias',
            'recommendation': 'Use neutral calendar spread',
            'rationale': 'Range-bound or unclear market direction',
            'implementation': 'Use ATM options with equal upside/downside exposure'
        })
    
    # 4. IV Rank Considerations
    if iv_rank > 70:
        optimizations.append({
            'type': 'iv_considerations',
            'recommendation': 'Consider shorter-term spread',
            'rationale': 'High IV rank may revert to mean',
            'implementation': 'Reduce long leg DTE to 30-45 days'
        })
    elif iv_rank < 30:
        optimizations.append({
            'type': 'iv_considerations',
            'recommendation': 'Consider longer-term spread',
            'rationale': 'Low IV rank likely to expand',
            'implementation': 'Increase long leg DTE to 75-90 days'
        })
    
    return optimizations

def analyze_iv_term_structure(
    expirations_iv: Dict[Union[str, datetime], float],
    base_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Analyze the IV term structure curve.
    
    Args:
        expirations_iv: Dictionary mapping expiration dates to average IVs
        base_date: Base date for calculation (defaults to today)
        
    Returns:
        Dictionary with term structure analysis
    """
    try:
        if base_date is None:
            base_date = datetime.now().date()
        elif isinstance(base_date, datetime):
            base_date = base_date.date()
        
        # Calculate days to expiration
        data_points = []
        
        for expiration, iv in expirations_iv.items():
            if isinstance(expiration, str):
                try:
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Could not parse expiration date: {expiration}")
                    continue
            elif isinstance(expiration, datetime):
                exp_date = expiration.date()
            else:
                exp_date = expiration
            
            # Calculate days to expiration
            dte = max(0, (exp_date - base_date).days)
            
            data_points.append((expiration, dte, iv))
        
        # Sort by days to expiration
        data_points.sort(key=lambda x: x[1])
        
        # Extract data series
        expirations = [point[0] for point in data_points]
        days = [point[1] for point in data_points]
        ivs = [point[2] for point in data_points]
        
        # Calculate term structure metrics
        metrics = {}
        
        if len(days) >= 2:
            # Calculate average slope
            if days[-1] != days[0]:
                avg_slope = (ivs[-1] - ivs[0]) / (days[-1] - days[0])
                metrics['average_slope'] = avg_slope
            
            # Calculate near-term slope (first two points)
            if days[1] != days[0]:
                near_term_slope = (ivs[1] - ivs[0]) / (days[1] - days[0])
                metrics['near_term_slope'] = near_term_slope
            
            # Determine curve shape
            if len(ivs) >= 3:
                # Fit quadratic curve
                if len(set(days)) >= 3:  # Ensure unique x values
                    try:
                        coeffs = np.polyfit(days, ivs, 2)
                        metrics['quadratic_coeffs'] = coeffs.tolist()
                        
                        # Curvature direction (coefficient of x²)
                        metrics['curvature'] = 'convex' if coeffs[0] > 0 else 'concave'
                    except Exception as e:
                        logger.warning(f"Could not fit quadratic curve: {e}")
            
            # Categorize term structure
            if 'average_slope' in metrics:
                if metrics['average_slope'] > 0.0001:  # Small positive threshold
                    metrics['term_shape'] = 'contango'
                elif metrics['average_slope'] < -0.0001:  # Small negative threshold
                    metrics['term_shape'] = 'backwardation'
                else:
                    metrics['term_shape'] = 'flat'
        
        # Calculate steepness (max IV - min IV)
        if ivs:
            metrics['min_iv'] = min(ivs)
            metrics['max_iv'] = max(ivs)
            metrics['steepness'] = metrics['max_iv'] - metrics['min_iv']
        
        return {
            'expirations': expirations,
            'days': days,
            'ivs': ivs,
            'metrics': metrics
        }
    
    except Exception as e:
        logger.error(f"Error analyzing IV term structure: {e}")
        return {'error': str(e)} 