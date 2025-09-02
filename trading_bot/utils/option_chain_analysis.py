#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Chain Analysis Utilities

This module provides utilities for analyzing options chains,
filtering for liquidity, and selecting optimal strikes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def filter_option_chain_by_liquidity(
    option_chain: pd.DataFrame,
    min_volume: int = 100,
    min_open_interest: int = 500,
    max_spread_pct: float = 10.0,
    min_dollar_volume: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter option chain to include only liquid contracts.
    
    Args:
        option_chain: DataFrame with option chain data
        min_volume: Minimum trading volume
        min_open_interest: Minimum open interest
        max_spread_pct: Maximum bid-ask spread as percentage
        min_dollar_volume: Minimum dollar volume (price * volume * multiplier)
        
    Returns:
        Filtered option chain DataFrame
    """
    try:
        # Verify required columns exist
        required_columns = ['bid', 'ask', 'volume', 'open_interest']
        missing_columns = [col for col in required_columns if col not in option_chain.columns]
        
        if missing_columns:
            logger.warning(f"Option chain missing required columns: {missing_columns}")
            return option_chain
        
        # Make a copy to avoid modifying the original
        df = option_chain.copy()
        
        # Calculate bid-ask spread as percentage
        df['spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        
        # Avoid division by zero
        df['spread_pct'] = np.where(
            df['mid_price'] > 0,
            df['spread'] / df['mid_price'] * 100,
            float('inf')
        )
        
        # Calculate dollar volume if requested
        if min_dollar_volume is not None:
            contract_multiplier = 100  # Standard multiplier for equity options
            df['dollar_volume'] = df['volume'] * df['mid_price'] * contract_multiplier
        
        # Apply filters
        filtered = df[
            (df['volume'] >= min_volume) &
            (df['open_interest'] >= min_open_interest) &
            (df['spread_pct'] <= max_spread_pct)
        ]
        
        # Apply dollar volume filter if requested
        if min_dollar_volume is not None:
            filtered = filtered[filtered['dollar_volume'] >= min_dollar_volume]
        
        # Check if we filtered out too much
        if len(filtered) == 0:
            logger.warning("Liquidity filters too restrictive, returning empty DataFrame")
        else:
            logger.info(f"Filtered option chain from {len(df)} to {len(filtered)} contracts")
        
        return filtered
    
    except Exception as e:
        logger.error(f"Error filtering option chain by liquidity: {e}")
        return pd.DataFrame()

def apply_tiered_liquidity_filters(
    option_chain: pd.DataFrame,
    liquidity_tier: str = 'medium'
) -> pd.DataFrame:
    """
    Apply tiered liquidity filters based on strategy requirements.
    
    Args:
        option_chain: DataFrame with option chain data
        liquidity_tier: 'high', 'medium', or 'low'
        
    Returns:
        Filtered option chain DataFrame
    """
    try:
        # Define filter parameters for each tier
        tiers = {
            'high': {
                'min_volume': 500,
                'min_open_interest': 1000,
                'max_spread_pct': 5.0,
                'min_dollar_volume': 100000
            },
            'medium': {
                'min_volume': 200,
                'min_open_interest': 500,
                'max_spread_pct': 8.0,
                'min_dollar_volume': 50000
            },
            'low': {
                'min_volume': 50,
                'min_open_interest': 200,
                'max_spread_pct': 15.0,
                'min_dollar_volume': 10000
            }
        }
        
        # Default to medium tier if invalid tier specified
        if liquidity_tier not in tiers:
            logger.warning(f"Invalid liquidity tier '{liquidity_tier}', using 'medium'")
            liquidity_tier = 'medium'
        
        # Get filter parameters for the selected tier
        filters = tiers[liquidity_tier]
        
        # Apply filters
        return filter_option_chain_by_liquidity(
            option_chain,
            min_volume=filters['min_volume'],
            min_open_interest=filters['min_open_interest'],
            max_spread_pct=filters['max_spread_pct'],
            min_dollar_volume=filters['min_dollar_volume']
        )
    
    except Exception as e:
        logger.error(f"Error applying tiered liquidity filters: {e}")
        return pd.DataFrame()

def find_atm_strike(
    option_chain: pd.DataFrame,
    underlying_price: float
) -> Optional[float]:
    """
    Find the at-the-money (ATM) strike price.
    
    Args:
        option_chain: DataFrame with option chain data
        underlying_price: Current price of the underlying
        
    Returns:
        ATM strike price or None if not found
    """
    try:
        # Verify strike column exists
        if 'strike' not in option_chain.columns:
            logger.warning("Option chain missing 'strike' column")
            return None
        
        # Get unique strikes
        strikes = option_chain['strike'].unique()
        
        if len(strikes) == 0:
            logger.warning("No strikes found in option chain")
            return None
        
        # Find strike closest to current price
        atm_strike = strikes[np.abs(strikes - underlying_price).argmin()]
        
        return atm_strike
    
    except Exception as e:
        logger.error(f"Error finding ATM strike: {e}")
        return None

def find_strike_by_delta(
    option_chain: pd.DataFrame,
    target_delta: float,
    option_type: str = 'call'
) -> Optional[float]:
    """
    Find the strike price corresponding to a specific delta.
    
    Args:
        option_chain: DataFrame with option chain data
        target_delta: Target delta value
        option_type: 'call' or 'put'
        
    Returns:
        Strike price with closest delta or None if not found
    """
    try:
        # Verify required columns exist
        if 'delta' not in option_chain.columns or 'strike' not in option_chain.columns:
            logger.warning("Option chain missing 'delta' or 'strike' column")
            return None
        
        # Filter by option type if specified in the DataFrame
        if 'option_type' in option_chain.columns:
            filtered = option_chain[option_chain['option_type'] == option_type]
        else:
            filtered = option_chain
        
        if len(filtered) == 0:
            logger.warning(f"No options found for type {option_type}")
            return None
        
        # Find strike with delta closest to target
        closest_idx = (filtered['delta'] - target_delta).abs().idxmin()
        strike = filtered.loc[closest_idx, 'strike']
        
        return strike
    
    except Exception as e:
        logger.error(f"Error finding strike by delta: {e}")
        return None

def select_calendar_spread_strikes(
    option_chain: pd.DataFrame,
    underlying_price: float,
    iv_rank: float,
    direction: str = 'neutral',
    strike_selection_method: str = 'atm'
) -> List[Dict[str, Any]]:
    """
    Select optimal strikes for a calendar spread.
    
    Args:
        option_chain: DataFrame with option chain data
        underlying_price: Current price of the underlying
        iv_rank: IV Rank (0-100)
        direction: 'neutral', 'bullish', or 'bearish'
        strike_selection_method: 'atm', 'delta', or 'custom'
        
    Returns:
        List of recommended strikes with metadata
    """
    try:
        # Verify required columns
        if 'strike' not in option_chain.columns:
            logger.warning("Option chain missing 'strike' column")
            return []
        
        # Initialize recommendations list
        recommendations = []
        
        # Find ATM strike as baseline
        atm_strike = find_atm_strike(option_chain, underlying_price)
        
        if atm_strike is None:
            logger.warning("Could not find ATM strike")
            return []
        
        # Get unique strikes sorted
        unique_strikes = sorted(option_chain['strike'].unique())
        
        # Find index of ATM strike
        try:
            atm_index = unique_strikes.index(atm_strike)
        except ValueError:
            logger.warning("ATM strike not in unique strikes list")
            return []
        
        # Add strikes based on selection method and direction
        if strike_selection_method == 'atm':
            # Always include ATM strike
            recommendations.append({
                'strike': atm_strike,
                'description': 'At-the-money strike',
                'rationale': 'Maximum front-month theta with balanced directional exposure',
                'score': 100
            })
            
        elif strike_selection_method == 'delta' and 'delta' in option_chain.columns:
            # Select strikes based on delta targets
            if direction == 'neutral':
                # Look for ~50 delta (ATM)
                delta_target = 0.5 if 'call' in option_chain['option_type'].unique() else -0.5
                strike = find_strike_by_delta(option_chain, delta_target)
                if strike is not None:
                    recommendations.append({
                        'strike': strike,
                        'description': 'Delta-neutral strike',
                        'rationale': 'Balanced positive/negative delta exposure',
                        'score': 100
                    })
            
            elif direction == 'bullish':
                # Look for ~40 delta (slightly ITM calls or OTM puts)
                delta_targets = [0.4, 0.45, 0.55]
                for target in delta_targets:
                    strike = find_strike_by_delta(option_chain, target, 'call')
                    if strike is not None:
                        recommendations.append({
                            'strike': strike,
                            'description': f'~{target*100:.0f} delta bullish-bias strike',
                            'rationale': 'Positive delta exposure with good theta',
                            'score': 100 - abs(0.5 - target) * 40  # Score based on distance from 0.5
                        })
            
            elif direction == 'bearish':
                # Look for ~40 delta (slightly ITM puts or OTM calls)
                delta_targets = [-0.4, -0.45, -0.55]
                for target in delta_targets:
                    strike = find_strike_by_delta(option_chain, target, 'put')
                    if strike is not None:
                        recommendations.append({
                            'strike': strike,
                            'description': f'~{-target*100:.0f} delta bearish-bias strike',
                            'rationale': 'Negative delta exposure with good theta',
                            'score': 100 - abs(-0.5 - target) * 40  # Score based on distance from -0.5
                        })
        
        else:
            # Default custom selection based on strike offset
            # ATM
            recommendations.append({
                'strike': atm_strike,
                'description': 'At-the-money strike',
                'rationale': 'Maximum front-month theta with balanced directional exposure',
                'score': 100
            })
            
            # Direction-based recommendations
            if direction == 'bullish' and atm_index + 1 < len(unique_strikes):
                # One strike OTM for bullish bias
                strike = unique_strikes[atm_index + 1]
                recommendations.append({
                    'strike': strike,
                    'description': 'Bullish-bias strike (OTM)',
                    'rationale': 'Positive delta exposure with good theta',
                    'score': 95
                })
                
            elif direction == 'bearish' and atm_index > 0:
                # One strike OTM for bearish bias
                strike = unique_strikes[atm_index - 1]
                recommendations.append({
                    'strike': strike,
                    'description': 'Bearish-bias strike (OTM)',
                    'rationale': 'Negative delta exposure with good theta',
                    'score': 95
                })
        
        # Add high-volume strike if available
        if 'volume' in option_chain.columns:
            high_volume_strike = option_chain.loc[option_chain['volume'].idxmax(), 'strike']
            
            # Only add if not already in recommendations
            if high_volume_strike not in [r['strike'] for r in recommendations]:
                recommendations.append({
                    'strike': high_volume_strike,
                    'description': 'High-volume strike',
                    'rationale': 'Better liquidity for entry and exit',
                    'score': 80
                })
        
        # Sort by score (highest first)
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error selecting calendar spread strikes: {e}")
        return []

def find_most_liquid_expiration(
    expirations_data: Dict[str, pd.DataFrame],
    target_dte_range: Tuple[int, int]
) -> Optional[str]:
    """
    Find the most liquid expiration within a target DTE range.
    
    Args:
        expirations_data: Dictionary mapping expiration dates to option chain DataFrames
        target_dte_range: Tuple of (min_dte, max_dte)
        
    Returns:
        Most liquid expiration date or None if not found
    """
    try:
        # Filter expirations within target range
        filtered_expirations = {}
        
        for expiration, chain in expirations_data.items():
            # Parse expiration to get DTE
            if isinstance(expiration, str):
                # Assuming expiration is in format 'YYYY-MM-DD'
                # We'd need to calculate DTE based on current date
                # This is a placeholder - replace with actual calculation
                dte = 30  # Placeholder
            else:
                # Assuming dte column is already in the DataFrame
                if 'dte' in chain.columns:
                    dte = chain['dte'].iloc[0]
                else:
                    logger.warning(f"Cannot determine DTE for expiration {expiration}")
                    continue
            
            # Check if within range
            if target_dte_range[0] <= dte <= target_dte_range[1]:
                filtered_expirations[expiration] = chain
        
        if not filtered_expirations:
            logger.warning(f"No expirations found within DTE range {target_dte_range}")
            return None
        
        # Calculate liquidity score for each expiration
        liquidity_scores = {}
        
        for expiration, chain in filtered_expirations.items():
            # Check for required columns
            required_columns = ['volume', 'open_interest']
            if not all(col in chain.columns for col in required_columns):
                logger.warning(f"Missing volume or open interest data for {expiration}")
                liquidity_scores[expiration] = 0
                continue
            
            # Calculate liquidity metrics
            total_volume = chain['volume'].sum()
            total_open_interest = chain['open_interest'].sum()
            
            # Average spread percentage if available
            if 'bid' in chain.columns and 'ask' in chain.columns:
                chain['mid_price'] = (chain['ask'] + chain['bid']) / 2
                chain['spread_pct'] = (chain['ask'] - chain['bid']) / chain['mid_price'] * 100
                avg_spread = chain['spread_pct'].mean()
                # Tighter spreads are better (invert for scoring)
                spread_score = 100 / (1 + avg_spread)
            else:
                spread_score = 50  # Neutral score if data not available
            
            # Calculate overall liquidity score
            # Weight: 50% volume, 30% open interest, 20% spread
            liquidity_score = (
                0.5 * total_volume + 
                0.3 * total_open_interest + 
                0.2 * spread_score
            )
            
            liquidity_scores[expiration] = liquidity_score
        
        # Find expiration with highest liquidity score
        if not liquidity_scores:
            return None
            
        best_expiration = max(liquidity_scores, key=liquidity_scores.get)
        
        return best_expiration
    
    except Exception as e:
        logger.error(f"Error finding most liquid expiration: {e}")
        return None

def analyze_strike_selection_for_calendar_spread(
    option_chain: pd.DataFrame,
    underlying_price: float,
    iv_rank: float,
    volatility_skew: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Analyze strike selection for calendar spread considering volatility skew.
    
    Args:
        option_chain: DataFrame with option chain data
        underlying_price: Current price of the underlying
        iv_rank: IV Rank (0-100)
        volatility_skew: Dictionary with skew metrics
        
    Returns:
        Dictionary with strike analysis
    """
    try:
        # Get ATM strike
        atm_strike = find_atm_strike(option_chain, underlying_price)
        
        if atm_strike is None:
            return {'error': 'Could not find ATM strike'}
        
        # Get unique strikes sorted
        if 'strike' not in option_chain.columns:
            return {'error': 'Option chain missing strike column'}
            
        unique_strikes = sorted(option_chain['strike'].unique())
        
        if not unique_strikes:
            return {'error': 'No strikes in option chain'}
        
        # Calculate distance of each strike from ATM
        strikes_info = []
        
        for strike in unique_strikes:
            strike_info = {
                'strike': strike,
                'distance_from_atm': strike - atm_strike,
                'distance_pct': (strike - atm_strike) / atm_strike * 100,
                'is_atm': strike == atm_strike
            }
            
            # Add liquidity metrics if available
            strike_data = option_chain[option_chain['strike'] == strike]
            
            if 'volume' in strike_data.columns:
                strike_info['volume'] = strike_data['volume'].sum()
            
            if 'open_interest' in strike_data.columns:
                strike_info['open_interest'] = strike_data['open_interest'].sum()
            
            # Calculate spread metrics if available
            if all(col in strike_data.columns for col in ['bid', 'ask']):
                bid = strike_data['bid'].max()
                ask = strike_data['ask'].min()
                mid = (bid + ask) / 2
                strike_info['spread'] = ask - bid
                strike_info['spread_pct'] = (ask - bid) / mid * 100 if mid > 0 else float('inf')
            
            # Calculate implied volatility if available
            if 'implied_volatility' in strike_data.columns:
                strike_info['implied_volatility'] = strike_data['implied_volatility'].mean()
            
            strikes_info.append(strike_info)
        
        # Consider volatility skew if provided
        if volatility_skew:
            # Analyze if skew suggests shifting strikes
            if 'put_call_skew' in volatility_skew and volatility_skew['put_call_skew'] > 0.1:
                # Significant put skew (higher IV for downside strikes)
                skew_indication = 'Consider higher strikes due to expensive puts'
            elif 'put_call_skew' in volatility_skew and volatility_skew['put_call_skew'] < -0.1:
                # Significant call skew (higher IV for upside strikes)
                skew_indication = 'Consider lower strikes due to expensive calls'
            else:
                skew_indication = 'Relatively flat skew, ATM strike likely optimal'
            
            # Add skew indication to result
            skew_analysis = {
                'indication': skew_indication,
                'skew_data': volatility_skew
            }
        else:
            skew_analysis = {
                'indication': 'No skew data available',
                'skew_data': None
            }
        
        # IV rank considerations
        if iv_rank > 70:
            iv_recommendation = 'High IV rank suggests good premium selling opportunity'
        elif iv_rank < 30:
            iv_recommendation = 'Low IV rank may limit calendar spread profitability'
        else:
            iv_recommendation = 'Moderate IV rank is favorable for calendar spreads'
        
        # Compile and return result
        result = {
            'underlying_price': underlying_price,
            'atm_strike': atm_strike,
            'iv_rank': iv_rank,
            'iv_recommendation': iv_recommendation,
            'strikes_info': strikes_info,
            'skew_analysis': skew_analysis
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing strike selection: {e}")
        return {'error': str(e)} 