#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta Decay Analysis Utilities

This module provides utilities for analyzing theta decay curves
and calculating calendar spread returns based on theta decay.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.utils.options_pricing import black_scholes, calculate_greeks

logger = logging.getLogger(__name__)

def calculate_theta_decay_curve(S: float, K: float, T: float, r: float, sigma: float, 
                               option_type: str = 'call', 
                               days_step: int = 1) -> Dict[str, List[float]]:
    """
    Calculate theta decay curve for an option over time.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        days_step: Step size in days for the curve
        
    Returns:
        Dictionary with days, theta values, and option prices
    """
    try:
        # Convert T to days
        total_days = int(T * 365)
        
        if total_days <= 0:
            logger.warning("Option already expired")
            return {'days': [], 'theta': [], 'price': []}
        
        # Create days array
        days = list(range(total_days, 0, -days_step))
        if days[-1] != 1:  # Ensure last day is included
            days.append(1)
        
        # Calculate theta and price for each day
        theta_values = []
        option_prices = []
        
        for day in days:
            time_to_expiry = day / 365.0  # Convert days to years
            
            # Calculate option price and Greeks
            price = black_scholes(S, K, time_to_expiry, r, sigma, option_type)
            greeks = calculate_greeks(S, K, time_to_expiry, r, sigma, option_type)
            
            theta_values.append(greeks['theta'])
            option_prices.append(price)
        
        return {
            'days': days,
            'theta': theta_values,
            'price': option_prices
        }
    
    except Exception as e:
        logger.error(f"Error calculating theta decay curve: {e}")
        return {'days': [], 'theta': [], 'price': []}

def calculate_calendar_spread_metrics(S: float, K: float, r: float, sigma: float,
                                     short_leg_dte: int, long_leg_dte: int,
                                     option_type: str = 'call') -> Dict[str, Any]:
    """
    Calculate key metrics for a calendar spread.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate
        sigma: Volatility
        short_leg_dte: Days to expiration for short leg
        long_leg_dte: Days to expiration for long leg
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with calendar spread metrics
    """
    try:
        # Convert DTE to years
        short_T = short_leg_dte / 365.0
        long_T = long_leg_dte / 365.0
        
        # Calculate option prices
        short_price = black_scholes(S, K, short_T, r, sigma, option_type)
        long_price = black_scholes(S, K, long_T, r, sigma, option_type)
        
        # Calculate Greeks
        short_greeks = calculate_greeks(S, K, short_T, r, sigma, option_type)
        long_greeks = calculate_greeks(S, K, long_T, r, sigma, option_type)
        
        # Calculate spread metrics
        net_debit = long_price - short_price
        net_theta = -short_greeks['theta'] + long_greeks['theta']  # Negative for short, positive for long
        net_delta = long_greeks['delta'] - short_greeks['delta']
        net_gamma = long_greeks['gamma'] - short_greeks['gamma']
        net_vega = long_greeks['vega'] - short_greeks['vega']
        
        # Calculate theta decay advantage
        theta_decay_advantage = net_theta / net_debit * 100 if net_debit > 0 else 0
        
        # Calculate days to theoretical max profit
        # This is a simplification - in reality the max profit depends on volatility changes too
        if net_theta > 0:
            days_to_max_profit = short_leg_dte
        else:
            days_to_max_profit = 0
        
        # Calculate max potential return based on historical theta capture
        typical_theta_capture = 0.7  # Typical to capture about 70% of theoretical theta
        max_potential_return = net_theta * short_leg_dte * typical_theta_capture / net_debit * 100
        
        return {
            'short_leg_price': short_price,
            'long_leg_price': long_price,
            'net_debit': net_debit,
            'net_theta': net_theta,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_vega': net_vega,
            'theta_decay_advantage': theta_decay_advantage,
            'days_to_max_profit': days_to_max_profit,
            'max_potential_return': max_potential_return
        }
    
    except Exception as e:
        logger.error(f"Error calculating calendar spread metrics: {e}")
        return {
            'short_leg_price': 0.0,
            'long_leg_price': 0.0,
            'net_debit': 0.0,
            'net_theta': 0.0,
            'net_delta': 0.0,
            'net_gamma': 0.0,
            'net_vega': 0.0,
            'theta_decay_advantage': 0.0,
            'days_to_max_profit': 0,
            'max_potential_return': 0.0
        }

def find_optimal_calendar_spread_dte(S: float, K: float, r: float, sigma: float,
                                    min_short_dte: int = 7, max_short_dte: int = 45,
                                    min_long_dte: int = 30, max_long_dte: int = 120,
                                    option_type: str = 'call') -> Dict[str, Any]:
    """
    Find optimal DTE combination for a calendar spread.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate
        sigma: Volatility
        min_short_dte: Minimum short leg DTE to consider
        max_short_dte: Maximum short leg DTE to consider
        min_long_dte: Minimum long leg DTE to consider
        max_long_dte: Maximum long leg DTE to consider
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with optimal DTE combination and metrics
    """
    try:
        best_combination = None
        best_theta_advantage = 0
        
        # Iterate through possible DTE combinations
        for short_dte in range(min_short_dte, max_short_dte + 1, 7):  # Weekly steps
            for long_dte in range(min_long_dte, max_long_dte + 1, 7):  # Weekly steps
                # Ensure long DTE is greater than short DTE
                if long_dte <= short_dte:
                    continue
                
                # Calculate metrics for this combination
                metrics = calculate_calendar_spread_metrics(
                    S, K, r, sigma, short_dte, long_dte, option_type
                )
                
                # Check if this combination has better theta advantage
                if metrics['theta_decay_advantage'] > best_theta_advantage:
                    best_theta_advantage = metrics['theta_decay_advantage']
                    best_combination = {
                        'short_leg_dte': short_dte,
                        'long_leg_dte': long_dte,
                        'metrics': metrics
                    }
        
        if best_combination is None:
            logger.warning("Could not find optimal calendar spread DTE combination")
            return {
                'short_leg_dte': min_short_dte,
                'long_leg_dte': min_long_dte,
                'metrics': {}
            }
        
        return best_combination
    
    except Exception as e:
        logger.error(f"Error finding optimal calendar spread DTE: {e}")
        return {
            'short_leg_dte': min_short_dte,
            'long_leg_dte': min_long_dte,
            'metrics': {}
        }

def project_calendar_spread_performance(S: float, K: float, r: float, sigma: float,
                                       short_leg_dte: int, long_leg_dte: int,
                                       option_type: str = 'call',
                                       days_step: int = 1) -> Dict[str, List[float]]:
    """
    Project performance of a calendar spread over time.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate
        sigma: Volatility
        short_leg_dte: Days to expiration for short leg
        long_leg_dte: Days to expiration for long leg
        option_type: 'call' or 'put'
        days_step: Step size in days for the projection
        
    Returns:
        Dictionary with days and projected P&L
    """
    try:
        # Calculate initial prices
        short_T = short_leg_dte / 365.0
        long_T = long_leg_dte / 365.0
        
        initial_short_price = black_scholes(S, K, short_T, r, sigma, option_type)
        initial_long_price = black_scholes(S, K, long_T, r, sigma, option_type)
        initial_net_debit = initial_long_price - initial_short_price
        
        # Create list of days
        days = list(range(0, short_leg_dte + 1, days_step))
        
        # Track P&L over time
        pnl = []
        pnl_pct = []
        net_values = []
        
        for day in days:
            remaining_short_dte = max(0, short_leg_dte - day)
            remaining_long_dte = max(0, long_leg_dte - day)
            
            short_t = remaining_short_dte / 365.0
            long_t = remaining_long_dte / 365.0
            
            # Calculate current prices
            short_price = black_scholes(S, K, short_t, r, sigma, option_type)
            long_price = black_scholes(S, K, long_t, r, sigma, option_type)
            
            # Calculate current spread value and P&L
            current_value = long_price - short_price
            current_pnl = current_value - initial_net_debit
            current_pnl_pct = (current_pnl / initial_net_debit) * 100 if initial_net_debit > 0 else 0
            
            net_values.append(current_value)
            pnl.append(current_pnl)
            pnl_pct.append(current_pnl_pct)
        
        return {
            'days': days,
            'net_values': net_values,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'initial_net_debit': initial_net_debit
        }
    
    except Exception as e:
        logger.error(f"Error projecting calendar spread performance: {e}")
        return {
            'days': [],
            'net_values': [],
            'pnl': [],
            'pnl_pct': [],
            'initial_net_debit': 0.0
        }

def analyze_calendar_spread_risk_profile(S: float, K: float, r: float, sigma: float,
                                        short_leg_dte: int, long_leg_dte: int,
                                        option_type: str = 'call',
                                        price_range_pct: float = 0.1) -> Dict[str, Any]:
    """
    Analyze the risk profile of a calendar spread at expiration of the front month.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate
        sigma: Volatility
        short_leg_dte: Days to expiration for short leg
        long_leg_dte: Days to expiration for long leg
        option_type: 'call' or 'put'
        price_range_pct: Price range to analyze as percentage of current price
        
    Returns:
        Dictionary with risk profile metrics
    """
    try:
        # Calculate initial prices
        short_T = short_leg_dte / 365.0
        long_T = long_leg_dte / 365.0
        
        initial_short_price = black_scholes(S, K, short_T, r, sigma, option_type)
        initial_long_price = black_scholes(S, K, long_T, r, sigma, option_type)
        initial_net_debit = initial_long_price - initial_short_price
        
        # Define price range to analyze
        min_price = S * (1 - price_range_pct)
        max_price = S * (1 + price_range_pct)
        price_steps = 41  # Number of price points to analyze
        prices = np.linspace(min_price, max_price, price_steps)
        
        # Calculate spread value at expiration of front month for each price point
        values = []
        pnls = []
        pnl_pcts = []
        
        remaining_long_dte = long_leg_dte - short_leg_dte
        remaining_long_T = remaining_long_dte / 365.0
        
        for price in prices:
            # At expiration of front month, short option is worth its intrinsic value
            if option_type.lower() == 'call':
                short_value = max(0, price - K)
            else:
                short_value = max(0, K - price)
            
            # Long option is valued using Black-Scholes
            long_value = black_scholes(price, K, remaining_long_T, r, sigma, option_type)
            
            # Calculate spread value and P&L
            spread_value = long_value - short_value
            pnl = spread_value - initial_net_debit
            pnl_pct = (pnl / initial_net_debit) * 100 if initial_net_debit > 0 else 0
            
            values.append(spread_value)
            pnls.append(pnl)
            pnl_pcts.append(pnl_pct)
        
        # Calculate max profit and max loss within the analyzed range
        max_profit = max(pnls)
        max_loss = min(pnls)
        
        # Find price at max profit
        max_profit_index = pnls.index(max_profit)
        price_at_max_profit = prices[max_profit_index]
        
        # Calculate breakeven points
        breakeven_prices = []
        for i in range(1, len(pnls)):
            if (pnls[i-1] < 0 and pnls[i] >= 0) or (pnls[i-1] >= 0 and pnls[i] < 0):
                # Linear interpolation to find more precise breakeven
                price1, price2 = prices[i-1], prices[i]
                pnl1, pnl2 = pnls[i-1], pnls[i]
                
                if pnl2 - pnl1 != 0:
                    breakeven = price1 + (price2 - price1) * (-pnl1) / (pnl2 - pnl1)
                    breakeven_prices.append(breakeven)
        
        return {
            'initial_net_debit': initial_net_debit,
            'prices': prices.tolist(),
            'values': values,
            'pnl': pnls,
            'pnl_pct': pnl_pcts,
            'max_profit': max_profit,
            'max_profit_pct': (max_profit / initial_net_debit) * 100 if initial_net_debit > 0 else 0,
            'price_at_max_profit': price_at_max_profit,
            'max_loss': max_loss,
            'max_loss_pct': (max_loss / initial_net_debit) * 100 if initial_net_debit > 0 else 0,
            'breakeven_prices': breakeven_prices
        }
    
    except Exception as e:
        logger.error(f"Error analyzing calendar spread risk profile: {e}")
        return {
            'initial_net_debit': 0.0,
            'prices': [],
            'values': [],
            'pnl': [],
            'pnl_pct': [],
            'max_profit': 0.0,
            'max_profit_pct': 0.0,
            'price_at_max_profit': 0.0,
            'max_loss': 0.0,
            'max_loss_pct': 0.0,
            'breakeven_prices': []
        } 