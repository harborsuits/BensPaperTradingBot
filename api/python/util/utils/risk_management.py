#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Utilities

This module provides utilities for options risk management,
position sizing, and portfolio Greek calculations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

from trading_bot.utils.options_pricing import calculate_greeks

logger = logging.getLogger(__name__)

def calculate_position_size(
    account_value: float,
    risk_per_trade_pct: float,
    max_loss_per_contract: float,
    min_contracts: int = 1,
    max_contracts: Optional[int] = None,
    contract_multiplier: int = 100
) -> int:
    """
    Calculate appropriate position size based on risk parameters.
    
    Args:
        account_value: Total account value
        risk_per_trade_pct: Percentage of account to risk per trade (e.g., 1.0 for 1%)
        max_loss_per_contract: Maximum loss per contract
        min_contracts: Minimum number of contracts
        max_contracts: Maximum number of contracts
        contract_multiplier: Option contract multiplier (typically 100)
        
    Returns:
        Number of contracts to trade
    """
    try:
        # Calculate maximum risk amount
        max_risk_amount = account_value * (risk_per_trade_pct / 100)
        
        # Calculate number of contracts
        if max_loss_per_contract <= 0:
            logger.warning("Invalid max loss per contract (must be positive)")
            return min_contracts
        
        num_contracts = int(max_risk_amount / max_loss_per_contract)
        
        # Apply minimum and maximum constraints
        num_contracts = max(min_contracts, num_contracts)
        
        if max_contracts is not None:
            num_contracts = min(max_contracts, num_contracts)
        
        return num_contracts
    
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return min_contracts

def calculate_calendar_spread_position_size(
    account_value: float,
    position_size_pct: float,
    net_debit: float,
    stop_loss_multiplier: float = 1.0,
    min_contracts: int = 1,
    max_contracts: Optional[int] = None,
    contract_multiplier: int = 100
) -> int:
    """
    Calculate position size for a calendar spread.
    
    Args:
        account_value: Total account value
        position_size_pct: Percentage of account to allocate per trade (e.g., 1.0 for 1%)
        net_debit: Net debit per spread
        stop_loss_multiplier: Multiplier for stop loss (e.g., 1.0 = 100% of net debit)
        min_contracts: Minimum number of contracts
        max_contracts: Maximum number of contracts
        contract_multiplier: Option contract multiplier (typically 100)
        
    Returns:
        Number of contracts to trade
    """
    try:
        # Calculate max loss per spread
        max_loss_per_spread = net_debit * stop_loss_multiplier
        
        # Calculate max loss per contract
        max_loss_per_contract = max_loss_per_spread * contract_multiplier
        
        # Call position size calculator
        return calculate_position_size(
            account_value,
            position_size_pct,
            max_loss_per_contract,
            min_contracts,
            max_contracts,
            contract_multiplier
        )
    
    except Exception as e:
        logger.error(f"Error calculating calendar spread position size: {e}")
        return min_contracts

def calculate_portfolio_greeks(
    positions: List[Dict[str, Any]],
    account_value: float
) -> Dict[str, float]:
    """
    Calculate portfolio Greek exposures.
    
    Args:
        positions: List of position dictionaries with contract details
        account_value: Total account value
        
    Returns:
        Dictionary with portfolio Greeks and risk metrics
    """
    try:
        # Initialize portfolio Greeks
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        # Track position values
        total_position_value = 0.0
        max_single_position_value = 0.0
        
        # Process each position
        for position in positions:
            # Extract position details
            quantity = position.get('quantity', 0)
            contract_multiplier = position.get('contract_multiplier', 100)
            direction = 1 if position.get('direction', 'long') == 'long' else -1
            position_type = position.get('position_type', '')
            
            # Extract contract details
            contract = position.get('contract', {})
            
            if not contract:
                logger.warning(f"Missing contract details for position: {position.get('id', 'unknown')}")
                continue
            
            # Calculate position value
            position_value = contract.get('price', 0) * quantity * contract_multiplier
            total_position_value += position_value
            max_single_position_value = max(max_single_position_value, position_value)
            
            # Special handling for spreads (calendar, vertical, etc.)
            if position_type in ['calendar_spread', 'vertical_spread', 'diagonal_spread']:
                # Process each leg separately
                legs = position.get('legs', [])
                
                if not legs:
                    logger.warning(f"Missing legs for spread position: {position.get('id', 'unknown')}")
                    continue
                
                for leg in legs:
                    leg_quantity = leg.get('quantity', quantity)
                    leg_direction = 1 if leg.get('direction', 'long') == 'long' else -1
                    leg_contract = leg.get('contract', {})
                    
                    if not leg_contract:
                        continue
                    
                    # Attempt to calculate Greeks if not provided
                    leg_greeks = leg_contract.get('greeks', {})
                    
                    if not leg_greeks:
                        # Try to calculate Greeks
                        S = leg_contract.get('underlying_price', 0)
                        K = leg_contract.get('strike', 0)
                        T = leg_contract.get('time_to_expiry', 0)
                        r = leg_contract.get('risk_free_rate', 0.04)
                        sigma = leg_contract.get('implied_volatility', 0.3)
                        option_type = leg_contract.get('option_type', 'call')
                        
                        if all([S, K, T, sigma]):
                            leg_greeks = calculate_greeks(S, K, T, r, sigma, option_type)
                    
                    # Add leg contribution to portfolio Greeks
                    for greek, value in leg_greeks.items():
                        if greek in portfolio_greeks:
                            portfolio_greeks[greek] += value * leg_quantity * contract_multiplier * leg_direction
            
            else:
                # Standard single-leg position
                position_greeks = contract.get('greeks', {})
                
                if not position_greeks:
                    # Try to calculate Greeks
                    S = contract.get('underlying_price', 0)
                    K = contract.get('strike', 0)
                    T = contract.get('time_to_expiry', 0)
                    r = contract.get('risk_free_rate', 0.04)
                    sigma = contract.get('implied_volatility', 0.3)
                    option_type = contract.get('option_type', 'call')
                    
                    if all([S, K, T, sigma]):
                        position_greeks = calculate_greeks(S, K, T, r, sigma, option_type)
                
                # Add position contribution to portfolio Greeks
                for greek, value in position_greeks.items():
                    if greek in portfolio_greeks:
                        portfolio_greeks[greek] += value * quantity * contract_multiplier * direction
        
        # Calculate risk metrics
        risk_metrics = {}
        
        if account_value > 0:
            # Calculate normalized Greeks (per $100k of account value)
            normalized_scale = 100000 / account_value
            normalized_greeks = {
                f"normalized_{greek}": value * normalized_scale
                for greek, value in portfolio_greeks.items()
            }
            
            # Calculate concentration metrics
            if total_position_value > 0:
                risk_metrics['max_position_concentration'] = max_single_position_value / total_position_value * 100
            
            risk_metrics['total_exposure_pct'] = total_position_value / account_value * 100
            
            # Add normalized Greeks to risk metrics
            risk_metrics.update(normalized_greeks)
        
        # Combine results
        result = {
            **portfolio_greeks,
            **risk_metrics
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating portfolio Greeks: {e}")
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'error': str(e)
        }

def check_portfolio_risk_limits(
    portfolio_greeks: Dict[str, float],
    risk_limits: Dict[str, float]
) -> Tuple[bool, List[str]]:
    """
    Check if portfolio is within risk limits.
    
    Args:
        portfolio_greeks: Portfolio Greeks and risk metrics
        risk_limits: Risk limits for each metric
        
    Returns:
        Tuple of (within_limits, violation_messages)
    """
    try:
        violations = []
        
        # Check each limit
        for metric, limit in risk_limits.items():
            if metric in portfolio_greeks:
                value = portfolio_greeks[metric]
                
                # Different checks based on metric type
                if metric.startswith('normalized_delta'):
                    # Delta limits are typically absolute value
                    if abs(value) > limit:
                        violations.append(f"{metric} exposure ({value:.2f}) exceeds limit ({limit:.2f})")
                
                elif metric.startswith('normalized_gamma'):
                    # Gamma limits are typically absolute value
                    if abs(value) > limit:
                        violations.append(f"{metric} exposure ({value:.2f}) exceeds limit ({limit:.2f})")
                
                elif metric.startswith('normalized_theta'):
                    # Theta is typically negative for long options, positive for shorts
                    # For a positive theta target (credit strategies), we want theta >= limit
                    if limit > 0 and value < limit:
                        violations.append(f"{metric} exposure ({value:.2f}) below target ({limit:.2f})")
                    # For a negative theta limit (debit strategies), we want theta >= limit (less negative)
                    elif limit < 0 and value < limit:
                        violations.append(f"{metric} exposure ({value:.2f}) below limit ({limit:.2f})")
                
                elif metric.startswith('normalized_vega'):
                    # Vega limits are typically absolute value
                    if abs(value) > limit:
                        violations.append(f"{metric} exposure ({value:.2f}) exceeds limit ({limit:.2f})")
                
                elif metric == 'total_exposure_pct':
                    # Total exposure should be below limit
                    if value > limit:
                        violations.append(f"Total exposure ({value:.2f}%) exceeds limit ({limit:.2f}%)")
                
                elif metric == 'max_position_concentration':
                    # Max position concentration should be below limit
                    if value > limit:
                        violations.append(f"Position concentration ({value:.2f}%) exceeds limit ({limit:.2f}%)")
                
                else:
                    # Generic check - value should be below limit
                    if value > limit:
                        violations.append(f"{metric} ({value:.2f}) exceeds limit ({limit:.2f})")
        
        return len(violations) == 0, violations
    
    except Exception as e:
        logger.error(f"Error checking portfolio risk limits: {e}")
        return False, [f"Error checking risk limits: {e}"]

def get_default_risk_limits() -> Dict[str, float]:
    """
    Get default risk limits for a well-diversified options portfolio.
    
    Returns:
        Dictionary with default risk limits
    """
    return {
        'normalized_delta': 500,   # Max $500 delta per $100k (0.5% of account)
        'normalized_gamma': 50,    # Max $50 gamma per $100k per 1% move
        'normalized_theta': 30,    # Min $30 theta per $100k per day (for income strategies)
        'normalized_vega': 200,    # Max $200 vega per $100k per 1% vol change
        'normalized_rho': 50,      # Max $50 rho per $100k per 1% rate change
        'total_exposure_pct': 50,  # Max 50% of account in options
        'max_position_concentration': 20  # Max 20% of options in a single position
    }

def get_calendar_spread_risk_limits() -> Dict[str, float]:
    """
    Get risk limits specifically for calendar spread strategies.
    
    Returns:
        Dictionary with risk limits for calendar spreads
    """
    return {
        'normalized_delta': 300,   # Lower delta limit for neutral strategy
        'normalized_gamma': 30,    # Lower gamma limit
        'normalized_theta': 40,    # Higher theta target (income strategy)
        'normalized_vega': 300,    # Higher vega limit (calendar spreads are vega-positive)
        'normalized_rho': 30,      # Lower rho limit
        'total_exposure_pct': 40,  # Max 40% of account in calendar spreads
        'max_position_concentration': 15  # Max 15% in a single calendar spread
    }

def should_adjust_position(
    position: Dict[str, Any],
    current_market_data: Dict[str, Any],
    adjustment_rules: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Determine if a position should be adjusted based on rules.
    
    Args:
        position: Current position data
        current_market_data: Current market data
        adjustment_rules: Rules for position adjustment
        
    Returns:
        Tuple of (should_adjust, reason, adjustment_details)
    """
    try:
        should_adjust = False
        reason = ""
        adjustment_details = {}
        
        # Extract position details
        position_type = position.get('position_type', '')
        
        # Only process calendar spreads
        if position_type != 'calendar_spread':
            return False, "Not a calendar spread position", {}
        
        # Extract calendar spread details
        entry_price = position.get('entry_price', 0)
        current_price = current_market_data.get('current_price', 0)
        entry_iv = position.get('entry_iv', 0)
        current_iv = current_market_data.get('current_iv', 0)
        
        # Check if price data is available
        if not entry_price or not current_price:
            return False, "Insufficient price data", {}
        
        # Calculate P&L
        pnl_pct = (current_price - entry_price) / abs(entry_price) * 100
        
        # Extract adjustment rules
        profit_target_pct = adjustment_rules.get('profit_target_pct', 50)
        stop_loss_pct = adjustment_rules.get('stop_loss_pct', 100)
        iv_change_threshold_pct = adjustment_rules.get('iv_change_threshold_pct', 20)
        
        # Check profit target
        if pnl_pct >= profit_target_pct:
            should_adjust = True
            reason = f"Profit target reached: {pnl_pct:.2f}% >= {profit_target_pct:.2f}%"
            adjustment_details = {
                'action': 'close_position',
                'reason': reason
            }
        
        # Check stop loss
        elif pnl_pct <= -stop_loss_pct:
            should_adjust = True
            reason = f"Stop loss triggered: {pnl_pct:.2f}% <= -{stop_loss_pct:.2f}%"
            adjustment_details = {
                'action': 'close_position',
                'reason': reason
            }
        
        # Check IV change
        elif entry_iv and current_iv:
            iv_change_pct = (current_iv - entry_iv) / entry_iv * 100
            
            if abs(iv_change_pct) >= iv_change_threshold_pct:
                should_adjust = True
                
                if iv_change_pct > 0:
                    reason = f"IV expanded significantly: {iv_change_pct:.2f}% >= {iv_change_threshold_pct:.2f}%"
                    adjustment_details = {
                        'action': 'roll_front_leg',
                        'reason': reason
                    }
                else:
                    reason = f"IV contracted significantly: {iv_change_pct:.2f}% <= -{iv_change_threshold_pct:.2f}%"
                    adjustment_details = {
                        'action': 'close_position',
                        'reason': reason
                    }
        
        # Check DTE
        short_leg_dte = current_market_data.get('short_leg_dte', 0)
        roll_trigger_dte = adjustment_rules.get('roll_trigger_dte', 7)
        
        if short_leg_dte <= roll_trigger_dte:
            should_adjust = True
            reason = f"Time-based roll trigger: {short_leg_dte} DTE <= {roll_trigger_dte} DTE"
            adjustment_details = {
                'action': 'roll_front_leg',
                'reason': reason
            }
        
        return should_adjust, reason, adjustment_details
    
    except Exception as e:
        logger.error(f"Error checking position adjustment: {e}")
        return False, f"Error: {e}", {}

def position_size_by_risk(
    price: float,
    account_value: float,
    risk_percent: float,
    stop_price: Optional[float] = None,
    max_position_percent: float = 15.0,
    min_position_size: int = 1
) -> int:
    """
    Calculate position size based on risk percentage of account value.
    
    Args:
        price: Current price of the asset
        account_value: Total account value in dollars
        risk_percent: Percentage of account to risk (e.g., 1.0 for 1%)
        stop_price: Price level for stop loss (if None, uses 7% from entry price)
        max_position_percent: Maximum percentage of account for any position
        min_position_size: Minimum position size to return
        
    Returns:
        Recommended position size in number of shares/contracts
    """
    try:
        # Validate inputs
        if price <= 0:
            logger.warning("Invalid price, must be positive")
            return min_position_size
            
        if account_value <= 0:
            logger.warning("Invalid account value, must be positive")
            return min_position_size
            
        if risk_percent <= 0 or risk_percent > 100:
            logger.warning("Invalid risk percentage, must be between 0 and 100")
            risk_percent = 1.0  # Default to 1%
        
        # Calculate risk amount in dollars
        risk_amount = account_value * (risk_percent / 100)
        
        # Calculate dollar risk per share
        if stop_price is not None and stop_price > 0:
            # If stop price is provided, use it
            if price > stop_price:
                # Long position
                dollar_risk_per_share = price - stop_price
            else:
                # Short position
                dollar_risk_per_share = stop_price - price
        else:
            # Use default 7% risk if no stop price provided
            dollar_risk_per_share = price * 0.07
            
        # Avoid division by zero
        if dollar_risk_per_share <= 0:
            logger.warning("Invalid dollar risk per share, defaulting to 1% of price")
            dollar_risk_per_share = price * 0.01
            
        # Calculate position size based on risk
        position_size = int(risk_amount / dollar_risk_per_share)
        
        # Check against maximum position size
        max_position_size = int((account_value * max_position_percent / 100) / price)
        position_size = min(position_size, max_position_size)
        
        # Ensure minimum position size
        position_size = max(position_size, min_position_size)
        
        return position_size
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return min_position_size

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate the maximum drawdown from an equity curve.
    
    Args:
        equity_curve: List of equity values over time
        
    Returns:
        Maximum drawdown as a percentage (0-100)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
        
    # Calculate drawdowns
    peak = equity_curve[0]
    max_drawdown = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
            
    return max_drawdown

def calculate_sharpe_ratio(
    returns: List[float], 
    risk_free_rate: float = 0.0, 
    annualization_factor: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a list of returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Factor to annualize (252 for daily, 12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
        
    # Convert to numpy array for easier calculations
    returns_array = np.array(returns)
    
    # Calculate average return
    avg_return = returns_array.mean()
    
    # Calculate return standard deviation
    std_return = returns_array.std()
    
    if std_return == 0:
        return 0.0
        
    # Convert risk-free rate to period rate
    period_risk_free = risk_free_rate / annualization_factor
    
    # Calculate Sharpe ratio
    sharpe = (avg_return - period_risk_free) / std_return
    
    # Annualize
    sharpe_annualized = sharpe * np.sqrt(annualization_factor)
    
    return sharpe_annualized

def calculate_position_exposure(positions: Dict[str, Any], account_value: float) -> Dict[str, Any]:
    """
    Calculate exposure metrics for current positions.
    
    Args:
        positions: Dictionary mapping symbols to position details
        account_value: Total account value
        
    Returns:
        Dictionary with exposure metrics
    """
    if not positions or account_value <= 0:
        return {
            "total_exposure": 0.0,
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "net_exposure": 0.0,
            "gross_exposure": 0.0,
            "largest_position_pct": 0.0,
            "position_count": 0
        }
        
    # Calculate exposures
    long_exposure = 0.0
    short_exposure = 0.0
    position_values = []
    
    for symbol, position in positions.items():
        position_value = abs(position.get("market_value", 0.0))
        position_values.append(position_value)
        
        if position.get("quantity", 0) > 0:
            long_exposure += position_value
        else:
            short_exposure += position_value
            
    total_exposure = long_exposure + short_exposure
    net_exposure = long_exposure - short_exposure
    
    # Calculate metrics
    gross_exposure_pct = (total_exposure / account_value) * 100 if account_value > 0 else 0
    net_exposure_pct = (net_exposure / account_value) * 100 if account_value > 0 else 0
    largest_position_pct = (max(position_values, default=0) / account_value) * 100 if account_value > 0 else 0
    
    return {
        "total_exposure": total_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "net_exposure": net_exposure,
        "gross_exposure_pct": gross_exposure_pct,
        "net_exposure_pct": net_exposure_pct,
        "largest_position_pct": largest_position_pct,
        "position_count": len(positions)
    } 