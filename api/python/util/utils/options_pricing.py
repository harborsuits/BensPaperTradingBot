#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Pricing Models

This module provides implementations of various options pricing models
including Black-Scholes and Binomial Tree, along with Greeks calculations.
"""

import logging
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str = 'call') -> float:
    """
    Calculate option price using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annual, decimal)
        sigma: Volatility (annual, decimal)
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    try:
        # Handle special cases
        if T <= 0:
            # Option is expired
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Calculate d1 and d2 parameters
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price based on type
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # Put option
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    except Exception as e:
        logger.error(f"Error in Black-Scholes calculation: {e}")
        return 0.0

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate option Greeks using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annual, decimal)
        sigma: Volatility (annual, decimal)
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary containing delta, gamma, theta, vega, rho
    """
    try:
        # Handle special cases
        if T <= 0:
            # Option is expired, all Greeks are zero
            return {
                'delta': 0.0, 
                'gamma': 0.0, 
                'theta': 0.0, 
                'vega': 0.0, 
                'rho': 0.0
            }
        
        # Calculate d1 and d2 parameters
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Greeks
        
        # Delta - rate of change of option price with respect to stock price
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma - rate of change of delta with respect to stock price (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta - rate of change of option price with respect to time (per day)
        if option_type.lower() == 'call':
            theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to daily theta
        theta = theta / 365.0
        
        # Vega - rate of change of option price with respect to volatility (per 1% change)
        vega = S * np.sqrt(T) * norm.pdf(d1) / 100
        
        # Rho - rate of change of option price with respect to interest rate (per 1% change)
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    except Exception as e:
        logger.error(f"Error in Greeks calculation: {e}")
        return {
            'delta': 0.0, 
            'gamma': 0.0, 
            'theta': 0.0, 
            'vega': 0.0, 
            'rho': 0.0
        }

def binomial_tree(S: float, K: float, T: float, r: float, sigma: float, 
                 steps: int, option_type: str = 'call', 
                 american: bool = True) -> float:
    """
    Price an option using the binomial tree model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annual, decimal)
        sigma: Volatility (annual, decimal)
        steps: Number of time steps in the tree
        option_type: 'call' or 'put'
        american: True for American options, False for European
        
    Returns:
        Option price
    """
    try:
        # Handle special cases
        if T <= 0:
            # Option is expired
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        if steps <= 0:
            logger.warning("Binomial tree requires at least 1 step. Using 1 step.")
            steps = 1
        
        # Time step
        dt = T / steps
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize stock price tree
        stock_tree = np.zeros((steps + 1, steps + 1))
        
        # Initialize option price tree
        option_tree = np.zeros((steps + 1, steps + 1))
        
        # Build stock price tree
        for i in range(steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Calculate option value at each leaf node
        if option_type.lower() == 'call':
            option_tree[:, steps] = np.maximum(0, stock_tree[:, steps] - K)
        else:
            option_tree[:, steps] = np.maximum(0, K - stock_tree[:, steps])
        
        # Work backwards through the tree
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                # Option value if held
                option_held = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                
                if american:
                    # Option value if exercised
                    if option_type.lower() == 'call':
                        option_exercised = max(0, stock_tree[j, i] - K)
                    else:
                        option_exercised = max(0, K - stock_tree[j, i])
                    
                    # Choose the maximum value (exercise or hold)
                    option_tree[j, i] = max(option_held, option_exercised)
                else:
                    option_tree[j, i] = option_held
        
        return option_tree[0, 0]
    
    except Exception as e:
        logger.error(f"Error in binomial tree calculation: {e}")
        return 0.0

def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, 
                      option_type: str = 'call', precision: float = 0.0001) -> Optional[float]:
    """
    Calculate implied volatility using bisection method.
    
    Args:
        market_price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annual, decimal)
        option_type: 'call' or 'put'
        precision: Desired precision for IV
        
    Returns:
        Implied volatility or None if calculation fails
    """
    try:
        # Handle special cases
        if T <= 0:
            return None
        
        # Calculate intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        # If market price is less than intrinsic value (arbitrage), return None
        if market_price < intrinsic:
            logger.warning(f"Market price ({market_price}) below intrinsic value ({intrinsic})")
            return None
        
        # Define bounds for IV search (1% to 500%)
        low = 0.01
        high = 5.0
        
        # Check if price is within theoretical bounds
        low_price = black_scholes(S, K, T, r, low, option_type)
        high_price = black_scholes(S, K, T, r, high, option_type)
        
        if market_price < low_price or market_price > high_price:
            logger.warning(f"Market price ({market_price}) outside theoretical bounds: [{low_price}, {high_price}]")
            return None
        
        # Binary search for implied volatility
        iterations = 0
        max_iterations = 100
        
        while high - low > precision and iterations < max_iterations:
            mid = (low + high) / 2
            mid_price = black_scholes(S, K, T, r, mid, option_type)
            
            if mid_price < market_price:
                low = mid
            else:
                high = mid
            
            iterations += 1
        
        if iterations >= max_iterations:
            logger.warning(f"Implied volatility calculation reached maximum iterations")
        
        return (low + high) / 2
    
    except Exception as e:
        logger.error(f"Error calculating implied volatility: {e}")
        return None

def newton_implied_volatility(market_price: float, S: float, K: float, T: float, r: float, 
                             option_type: str = 'call', 
                             initial_guess: float = 0.3,
                             max_iterations: int = 100,
                             precision: float = 0.00001) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        market_price: Market price of the option
        S, K, T, r: Standard Black-Scholes parameters
        option_type: 'call' or 'put'
        initial_guess: Initial volatility guess
        max_iterations: Maximum number of iterations
        precision: Desired precision for IV
        
    Returns:
        Implied volatility or None if calculation fails
    """
    try:
        # Handle special cases
        if T <= 0:
            return None
        
        sigma = initial_guess
        iterations = 0
        
        while iterations < max_iterations:
            # Calculate option price and vega at current sigma
            price = black_scholes(S, K, T, r, sigma, option_type)
            greeks = calculate_greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert to per 1.0 change in vol
            
            # Calculate price difference
            price_diff = price - market_price
            
            # Check if we've reached desired precision
            if abs(price_diff) < precision:
                return sigma
            
            # Check for near-zero vega to avoid division by zero
            if abs(vega) < 1e-10:
                logger.warning("Vega too small, switching to bisection method")
                return implied_volatility(market_price, S, K, T, r, option_type, precision)
            
            # Update sigma using Newton-Raphson formula
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays within reasonable bounds (1% to 500%)
            sigma = max(0.01, min(5.0, sigma))
            
            iterations += 1
        
        logger.warning(f"Newton method reached maximum iterations ({max_iterations})")
        
        # Fall back to bisection method if Newton method didn't converge
        return implied_volatility(market_price, S, K, T, r, option_type, precision)
    
    except Exception as e:
        logger.error(f"Error in Newton implied volatility calculation: {e}")
        # Fall back to bisection method
        return implied_volatility(market_price, S, K, T, r, option_type, precision)

def calculate_iv_rank(current_iv: float, historical_iv: List[float]) -> float:
    """
    Calculate IV Rank (0-100) based on historical volatility.
    
    Args:
        current_iv: Current implied volatility
        historical_iv: List of historical IV values
        
    Returns:
        IV Rank as percentage (0-100)
    """
    try:
        # Handle invalid inputs
        if not historical_iv or len(historical_iv) < 2:
            logger.warning("Insufficient historical IV data, returning default IV rank of 50")
            return 50.0
        
        # Calculate min and max IV over the historical period
        iv_min = min(historical_iv)
        iv_max = max(historical_iv)
        
        # Avoid division by zero
        if iv_max == iv_min:
            return 50.0
        
        # Calculate IV Rank: (current - min) / (max - min)
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        
        # Ensure result is within 0-100 range
        return max(0, min(100, iv_rank))
    
    except Exception as e:
        logger.error(f"Error calculating IV rank: {e}")
        return 50.0  # Default to middle value

def calculate_iv_percentile(current_iv: float, historical_iv: List[float]) -> float:
    """
    Calculate IV Percentile (0-100) based on historical volatility.
    
    Args:
        current_iv: Current implied volatility
        historical_iv: List of historical IV values
        
    Returns:
        IV Percentile (0-100)
    """
    try:
        # Handle invalid inputs
        if not historical_iv:
            logger.warning("No historical IV data, returning default IV percentile of 50")
            return 50.0
        
        # Count how many historical values are below current IV
        values_below = sum(1 for iv in historical_iv if iv < current_iv)
        
        # Calculate percentile
        percentile = (values_below / len(historical_iv)) * 100
        
        return percentile
    
    except Exception as e:
        logger.error(f"Error calculating IV percentile: {e}")
        return 50.0  # Default to middle value 