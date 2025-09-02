import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                           annualization_factor: int = 252) -> float:
    """
    Calculate Sortino Ratio - measures return relative to downside volatility.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annualized risk-free rate (default: 0)
        annualization_factor: Factor for annualizing returns (252 for daily returns)
        
    Returns:
        Sortino ratio (float)
    """
    # Annualize the returns
    avg_return = np.mean(returns) * annualization_factor
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = returns[returns < 0]
    
    # If no negative returns, return a high value with a warning
    if len(negative_returns) == 0:
        return float('inf')  # Perfect Sortino ratio (no downside)
    
    # Calculate downside deviation and annualize
    downside_deviation = np.std(negative_returns) * np.sqrt(annualization_factor)
    
    # Calculate Sortino ratio
    sortino_ratio = (avg_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio


def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float, 
                          annualization_factor: int = 252) -> float:
    """
    Calculate Calmar Ratio - measures return relative to maximum drawdown.
    
    Args:
        returns: Array of period returns
        max_drawdown: Maximum drawdown as a positive decimal (e.g. 0.25 for 25% drawdown)
        annualization_factor: Factor for annualizing returns (252 for daily returns)
        
    Returns:
        Calmar ratio (float)
    """
    # Annualize the returns
    avg_return = np.mean(returns) * annualization_factor
    
    # Ensure max_drawdown is positive (convert from negative percentage if needed)
    max_drawdown_abs = abs(max_drawdown)
    
    # Handle case where max_drawdown is zero or near-zero
    if max_drawdown_abs < 1e-10:
        return float('inf')  # Perfect Calmar ratio (no drawdown)
    
    # Calculate Calmar ratio
    calmar_ratio = avg_return / max_drawdown_abs
    
    return calmar_ratio


def calculate_max_consecutive_losses(returns: np.ndarray) -> int:
    """
    Calculate the maximum number of consecutive negative returns.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Maximum consecutive losses (int)
    """
    # Initialize variables
    current_streak = 0
    max_streak = 0
    
    # Iterate through returns
    for ret in returns:
        if ret < 0:
            # Negative return, increment streak
            current_streak += 1
            # Update max streak if current streak is longer
            max_streak = max(max_streak, current_streak)
        else:
            # Positive return, reset streak
            current_streak = 0
    
    return max_streak


def calculate_max_consecutive_wins(returns: np.ndarray) -> int:
    """
    Calculate the maximum number of consecutive positive returns.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Maximum consecutive wins (int)
    """
    # Initialize variables
    current_streak = 0
    max_streak = 0
    
    # Iterate through returns
    for ret in returns:
        if ret > 0:
            # Positive return, increment streak
            current_streak += 1
            # Update max streak if current streak is longer
            max_streak = max(max_streak, current_streak)
        else:
            # Negative or zero return, reset streak
            current_streak = 0
    
    return max_streak


def calculate_downside_deviation(returns: np.ndarray, 
                                target_return: float = 0.0, 
                                annualization_factor: int = 252) -> float:
    """
    Calculate downside deviation - standard deviation of returns below target.
    
    Args:
        returns: Array of period returns
        target_return: Target return threshold (default: 0)
        annualization_factor: Factor for annualizing (252 for daily returns)
        
    Returns:
        Downside deviation (float)
    """
    # Get returns below target
    downside_returns = returns[returns < target_return]
    
    # If no returns below target, return 0
    if len(downside_returns) == 0:
        return 0.0
    
    # Calculate deviation from target for returns below target
    deviations = target_return - downside_returns
    
    # Calculate downside deviation and annualize
    downside_dev = np.sqrt(np.mean(np.square(deviations))) * np.sqrt(annualization_factor)
    
    return downside_dev


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) for a given confidence level.
    
    Args:
        returns: Array of period returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        Value at Risk (float) - a negative number representing the loss
    """
    # Calculate the percentile corresponding to the confidence level
    var = np.percentile(returns, 100 * (1 - confidence_level))
    
    return var


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.
    
    Args:
        returns: Array of period returns
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
        
    Returns:
        Conditional Value at Risk (float) - a negative number representing the average loss
    """
    # Calculate VaR at the given confidence level
    var = calculate_var(returns, confidence_level)
    
    # Filter for returns below VaR
    tail_returns = returns[returns <= var]
    
    # If no returns in the tail, return VaR
    if len(tail_returns) == 0:
        return var
    
    # Calculate CVaR (average of returns in the tail)
    cvar = np.mean(tail_returns)
    
    return cvar


def calculate_omega_ratio(returns: np.ndarray, 
                         threshold_return: float = 0.0, 
                         annualization_factor: int = 252) -> float:
    """
    Calculate Omega ratio - probability-weighted ratio of gains vs. losses.
    
    Args:
        returns: Array of period returns
        threshold_return: Threshold return for considering gain vs loss
        annualization_factor: Factor for annualizing (252 for daily returns)
        
    Returns:
        Omega ratio (float)
    """
    # Separate returns above and below threshold
    returns_above = returns[returns > threshold_return] - threshold_return
    returns_below = threshold_return - returns[returns < threshold_return]
    
    # Calculate sum of returns above and below threshold
    sum_above = np.sum(returns_above)
    sum_below = np.sum(returns_below)
    
    # Handle case where no returns are below threshold
    if sum_below == 0:
        return float('inf')  # Perfect Omega ratio
    
    # Calculate Omega ratio
    omega_ratio = sum_above / sum_below
    
    return omega_ratio


def calculate_gain_to_pain_ratio(returns: np.ndarray) -> float:
    """
    Calculate Gain to Pain ratio - sum of returns divided by absolute sum of negative returns.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Gain to Pain ratio (float)
    """
    # Calculate sum of all returns
    sum_returns = np.sum(returns)
    
    # Calculate absolute sum of negative returns
    abs_sum_negative = np.abs(np.sum(returns[returns < 0]))
    
    # Handle case where no negative returns
    if abs_sum_negative == 0:
        return float('inf')  # Perfect gain-to-pain ratio
    
    # Calculate Gain to Pain ratio
    gain_to_pain = sum_returns / abs_sum_negative
    
    return gain_to_pain


def calculate_ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index - measure of downside risk that considers depth and duration of drawdowns.
    
    Args:
        equity_curve: Array of cumulative equity values
        
    Returns:
        Ulcer Index (float)
    """
    # Calculate running maximum of equity curve
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate percentage drawdowns
    drawdowns = (equity_curve - running_max) / running_max
    
    # Square the drawdowns
    squared_drawdowns = np.square(drawdowns)
    
    # Calculate Ulcer Index (square root of average squared drawdown)
    ulcer_index = np.sqrt(np.mean(squared_drawdowns))
    
    return ulcer_index


def calculate_drawdowns(equity_curve: np.ndarray) -> Dict:
    """
    Calculate drawdown statistics including maximum drawdown, drawdown duration, and recovery time.
    
    Args:
        equity_curve: Array of cumulative equity values
        
    Returns:
        Dictionary with drawdown statistics
    """
    # Initialize result dictionary
    result = {}
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdowns
    drawdowns = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown and its index
    max_drawdown = np.min(drawdowns)
    max_drawdown_idx = np.argmin(drawdowns)
    
    # Find peak index (where the drawdown started)
    peak_idx = np.where(equity_curve[:max_drawdown_idx] == running_max[max_drawdown_idx])[0][-1]
    
    # Find recovery index (where equity gets back to the previous peak)
    try:
        recovery_idx = np.where(equity_curve[max_drawdown_idx:] >= equity_curve[peak_idx])[0][0] + max_drawdown_idx
    except IndexError:
        # If no recovery, set to end of series
        recovery_idx = len(equity_curve) - 1
    
    # Calculate drawdown duration (peak to trough)
    drawdown_duration = max_drawdown_idx - peak_idx
    
    # Calculate recovery time (trough to recovery)
    recovery_time = recovery_idx - max_drawdown_idx if recovery_idx < len(equity_curve) - 1 else None
    
    # Calculate total underwater time (peak to recovery)
    underwater_time = recovery_idx - peak_idx if recovery_idx < len(equity_curve) - 1 else len(equity_curve) - peak_idx
    
    # Store results
    result['max_drawdown'] = max_drawdown
    result['max_drawdown_idx'] = max_drawdown_idx
    result['peak_idx'] = peak_idx
    result['recovery_idx'] = recovery_idx if recovery_idx < len(equity_curve) - 1 else None
    result['drawdown_duration'] = drawdown_duration
    result['recovery_time'] = recovery_time
    result['underwater_time'] = underwater_time
    result['drawdown_series'] = drawdowns
    
    # Identify all significant drawdowns (greater than 5%)
    significant_drawdowns = []
    
    # Track current drawdown
    in_drawdown = False
    current_peak = 0
    current_trough = 0
    current_depth = 0
    
    for i in range(1, len(equity_curve)):
        # If not in drawdown and equity drops from peak
        if not in_drawdown and equity_curve[i] < running_max[i-1]:
            in_drawdown = True
            current_peak = i - 1
            current_trough = i
            current_depth = (equity_curve[i] - running_max[i-1]) / running_max[i-1]
        
        # If in drawdown
        elif in_drawdown:
            # Update trough and depth if drawdown gets deeper
            if equity_curve[i] < equity_curve[current_trough]:
                current_trough = i
                current_depth = (equity_curve[i] - equity_curve[current_peak]) / equity_curve[current_peak]
            
            # Check if drawdown is over (equity reaches a new peak)
            if equity_curve[i] >= equity_curve[current_peak]:
                in_drawdown = False
                
                # Only record significant drawdowns (e.g., > 5%)
                if abs(current_depth) >= 0.05:
                    significant_drawdowns.append({
                        'start_idx': current_peak,
                        'trough_idx': current_trough,
                        'end_idx': i,
                        'depth': current_depth,
                        'duration': current_trough - current_peak,
                        'recovery': i - current_trough
                    })
    
    # If still in drawdown at the end
    if in_drawdown and abs(current_depth) >= 0.05:
        significant_drawdowns.append({
            'start_idx': current_peak,
            'trough_idx': current_trough,
            'end_idx': None,
            'depth': current_depth,
            'duration': current_trough - current_peak,
            'recovery': None
        })
    
    # Store significant drawdowns
    result['significant_drawdowns'] = significant_drawdowns
    
    return result


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate - percentage of positive returns.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Win rate as a percentage (float)
    """
    # Count positive returns
    positive_returns = np.sum(returns > 0)
    
    # Calculate win rate
    win_rate = (positive_returns / len(returns)) * 100 if len(returns) > 0 else 0
    
    return win_rate


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor - ratio of gross profits to gross losses.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Profit factor (float)
    """
    # Separate positive and negative returns
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    # Calculate gross profit and loss
    gross_profit = np.sum(positive_returns)
    gross_loss = np.abs(np.sum(negative_returns))
    
    # Calculate profit factor (handle division by zero)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return profit_factor


def calculate_average_win_loss_ratio(returns: np.ndarray) -> float:
    """
    Calculate average win/loss ratio - average win divided by average loss.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Average win/loss ratio (float)
    """
    # Separate positive and negative returns
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    # Calculate average win and loss
    avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
    avg_loss = np.abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
    
    # Calculate win/loss ratio (handle division by zero)
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    return win_loss_ratio


def calculate_comprehensive_metrics(returns: np.ndarray, 
                                   equity_curve: np.ndarray, 
                                   risk_free_rate: float = 0.0,
                                   annualization_factor: int = 252) -> Dict:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Args:
        returns: Array of period returns
        equity_curve: Array of cumulative equity values
        risk_free_rate: Annualized risk-free rate
        annualization_factor: Factor for annualizing returns
        
    Returns:
        Dictionary with all performance metrics
    """
    # Initialize results dictionary
    metrics = {}
    
    # Basic return metrics
    metrics['total_return'] = (equity_curve[-1] / equity_curve[0]) - 1
    metrics['annualized_return'] = np.mean(returns) * annualization_factor
    
    # Volatility
    metrics['volatility'] = np.std(returns) * np.sqrt(annualization_factor)
    
    # Traditional risk-adjusted metrics
    metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['volatility'] if metrics['volatility'] > 0 else float('inf')
    
    # Advanced risk-adjusted metrics
    metrics['downside_deviation'] = calculate_downside_deviation(returns, 0, annualization_factor)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, annualization_factor)
    
    # Drawdown analysis
    drawdown_metrics = calculate_drawdowns(equity_curve)
    metrics['max_drawdown'] = drawdown_metrics['max_drawdown']
    metrics['drawdown_duration'] = drawdown_metrics['drawdown_duration']
    metrics['recovery_time'] = drawdown_metrics['recovery_time']
    metrics['underwater_time'] = drawdown_metrics['underwater_time']
    metrics['significant_drawdowns'] = drawdown_metrics['significant_drawdowns']
    metrics['drawdown_series'] = drawdown_metrics['drawdown_series']
    
    # Calmar ratio
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, abs(metrics['max_drawdown']), annualization_factor)
    
    # Win/loss metrics
    metrics['win_rate'] = calculate_win_rate(returns)
    metrics['profit_factor'] = calculate_profit_factor(returns)
    metrics['win_loss_ratio'] = calculate_average_win_loss_ratio(returns)
    metrics['max_consecutive_wins'] = calculate_max_consecutive_wins(returns)
    metrics['max_consecutive_losses'] = calculate_max_consecutive_losses(returns)
    
    # Additional risk metrics
    metrics['var_95'] = calculate_var(returns, 0.95)
    metrics['cvar_95'] = calculate_cvar(returns, 0.95)
    metrics['ulcer_index'] = calculate_ulcer_index(equity_curve)
    metrics['omega_ratio'] = calculate_omega_ratio(returns, 0, annualization_factor)
    metrics['gain_to_pain_ratio'] = calculate_gain_to_pain_ratio(returns)
    
    return metrics


def format_performance_report(metrics: Dict) -> Dict:
    """
    Format comprehensive performance metrics into a structured report.
    
    Args:
        metrics: Dictionary with performance metrics
        
    Returns:
        Structured performance report
    """
    # Initialize report sections
    report = {
        'summary': {},
        'risk_metrics': {},
        'risk_adjusted_metrics': {},
        'trade_metrics': {},
        'drawdown_analysis': {},
        'advanced_metrics': {}
    }
    
    # Fill summary section
    report['summary'] = {
        'total_return_pct': metrics['total_return'] * 100,
        'annualized_return_pct': metrics['annualized_return'] * 100,
        'volatility_pct': metrics['volatility'] * 100
    }
    
    # Fill risk metrics section
    report['risk_metrics'] = {
        'max_drawdown_pct': metrics['max_drawdown'] * 100,
        'var_95_pct': metrics['var_95'] * 100,
        'cvar_95_pct': metrics['cvar_95'] * 100,
        'downside_deviation_pct': metrics['downside_deviation'] * 100,
        'ulcer_index': metrics['ulcer_index']
    }
    
    # Fill risk-adjusted metrics section
    report['risk_adjusted_metrics'] = {
        'sharpe_ratio': metrics['sharpe_ratio'],
        'sortino_ratio': metrics['sortino_ratio'],
        'calmar_ratio': metrics['calmar_ratio'],
        'omega_ratio': metrics['omega_ratio'],
        'gain_to_pain_ratio': metrics['gain_to_pain_ratio']
    }
    
    # Fill trade metrics section
    report['trade_metrics'] = {
        'win_rate_pct': metrics['win_rate'],
        'profit_factor': metrics['profit_factor'],
        'win_loss_ratio': metrics['win_loss_ratio'],
        'max_consecutive_wins': metrics['max_consecutive_wins'],
        'max_consecutive_losses': metrics['max_consecutive_losses']
    }
    
    # Fill drawdown analysis section
    report['drawdown_analysis'] = {
        'max_drawdown_pct': metrics['max_drawdown'] * 100,
        'drawdown_duration_days': metrics['drawdown_duration'],
        'recovery_time_days': metrics['recovery_time'],
        'underwater_time_days': metrics['underwater_time']
    }
    
    # Add significant drawdowns with formatted dates if timestamps are available
    if 'significant_drawdowns' in metrics and metrics['significant_drawdowns']:
        report['drawdown_analysis']['significant_drawdowns'] = metrics['significant_drawdowns']
    
    return report 