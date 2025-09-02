import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

def calculate_metrics(returns: Union[pd.Series, List[float], np.ndarray], 
                      risk_free_rate: float = 0.03,
                      benchmark_returns: Optional[Union[pd.Series, List[float], np.ndarray]] = None) -> Dict[str, Any]:
    """
    Calculate common performance metrics for a return series.
    
    Args:
        returns: Daily return series (percentage or decimal)
        risk_free_rate: Annualized risk-free rate (default 3%)
        benchmark_returns: Optional benchmark return series for relative metrics
        
    Returns:
        Dictionary containing performance metrics
    """
    # Ensure returns are in pandas Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Convert returns to decimal if in percentage
    if returns.mean() > 0.1:  # Rough heuristic to detect percentage vs decimal
        returns = returns / 100
    
    if benchmark_returns is not None and not isinstance(benchmark_returns, pd.Series):
        benchmark_returns = pd.Series(benchmark_returns)
        if benchmark_returns.mean() > 0.1:  # Same heuristic
            benchmark_returns = benchmark_returns / 100
    
    # Calculate daily risk-free rate 
    daily_risk_free = risk_free_rate / 252
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk metrics
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Drawdown analysis
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index / previous_peaks - 1)
    max_drawdown = drawdowns.min()
    max_drawdown_duration = _get_max_drawdown_duration(wealth_index)
    
    # Win/loss metrics
    win_rate = (returns > 0).mean()
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
    
    # Relative metrics if benchmark is provided
    benchmark_metrics = {}
    if benchmark_returns is not None:
        # Calculate excess returns
        if len(benchmark_returns) == len(returns):
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            beta = _calculate_beta(returns, benchmark_returns)
            alpha = _calculate_alpha(returns, benchmark_returns, risk_free_rate, beta)
            
            benchmark_metrics = {
                'alpha': alpha,
                'beta': beta,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'up_capture': _calculate_up_capture(returns, benchmark_returns),
                'down_capture': _calculate_down_capture(returns, benchmark_returns)
            }
    
    # Combine all metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': abs(annualized_return / max_drawdown) if max_drawdown != 0 else float('inf')
    }
    
    # Add benchmark metrics if available
    if benchmark_metrics:
        metrics.update(benchmark_metrics)
    
    return metrics

def _get_max_drawdown_duration(wealth_index: pd.Series) -> int:
    """
    Calculate the duration of the maximum drawdown period.
    
    Args:
        wealth_index: Cumulative wealth index
        
    Returns:
        Duration in days
    """
    # Calculate drawdown
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index / previous_peaks - 1)
    
    # Find max drawdown period
    is_drawdown = drawdowns < 0
    
    # If no drawdowns, return 0
    if not is_drawdown.any():
        return 0
    
    # Initialize tracking variables
    current_dd_start = None
    max_dd_duration = 0
    
    # Find drawdown periods
    for date, in_dd in is_drawdown.items():
        if in_dd:
            if current_dd_start is None:
                current_dd_start = date
        else:
            if current_dd_start is not None:
                dd_duration = (date - current_dd_start).days
                max_dd_duration = max(max_dd_duration, dd_duration)
                current_dd_start = None
    
    # Check if we're still in a drawdown at the end of the series
    if current_dd_start is not None:
        dd_duration = (is_drawdown.index[-1] - current_dd_start).days
        max_dd_duration = max(max_dd_duration, dd_duration)
    
    return max_dd_duration

def _calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate beta (sensitivity to market).
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Beta value
    """
    covariance = returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    return covariance / benchmark_variance if benchmark_variance > 0 else 1.0

def _calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series, 
                     risk_free_rate: float, beta: float) -> float:
    """
    Calculate alpha (excess return over what would be predicted by beta).
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annualized risk-free rate
        beta: Calculated beta
        
    Returns:
        Alpha value (annualized)
    """
    daily_rf = risk_free_rate / 252
    alpha = returns.mean() - (daily_rf + beta * (benchmark_returns.mean() - daily_rf))
    return alpha * 252  # Annualize

def _calculate_up_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate upside capture ratio.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Upside capture ratio
    """
    up_markets = benchmark_returns > 0
    if up_markets.sum() == 0:
        return 1.0
    
    return returns[up_markets].mean() / benchmark_returns[up_markets].mean() if benchmark_returns[up_markets].mean() != 0 else float('inf')

def _calculate_down_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate downside capture ratio.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Downside capture ratio
    """
    down_markets = benchmark_returns < 0
    if down_markets.sum() == 0:
        return 0.0
    
    return returns[down_markets].mean() / benchmark_returns[down_markets].mean() if benchmark_returns[down_markets].mean() != 0 else 0.0

def calculate_rolling_metrics(returns: pd.Series, window: int = 60, 
                             risk_free_rate: float = 0.03) -> pd.DataFrame:
    """
    Calculate rolling performance metrics over a window of days.
    
    Args:
        returns: Daily return series
        window: Rolling window size in days
        risk_free_rate: Annualized risk-free rate
        
    Returns:
        DataFrame with rolling metrics
    """
    if len(returns) < window:
        raise ValueError(f"Returns series length ({len(returns)}) is less than window size ({window})")
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=returns.index[window-1:])
    
    # Rolling return
    rolling_return = returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1)
    results['rolling_return'] = rolling_return
    
    # Rolling volatility (annualized)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    results['rolling_volatility'] = rolling_vol
    
    # Rolling Sharpe
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    rolling_sharpe = excess_returns.rolling(window=window).mean() / returns.rolling(window=window).std()
    results['rolling_sharpe'] = rolling_sharpe * np.sqrt(252)
    
    # Rolling max drawdown
    def calc_max_dd(x):
        cum_returns = (1 + x).cumprod()
        return (cum_returns / cum_returns.cummax() - 1).min()
    
    results['rolling_max_drawdown'] = returns.rolling(window=window).apply(calc_max_dd)
    
    # Rolling win rate
    results['rolling_win_rate'] = returns.rolling(window=window).apply(lambda x: (x > 0).mean())
    
    return results

def calculate_regime_performance(returns: pd.Series, regime_data: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics by market regime.
    
    Args:
        returns: Daily return series
        regime_data: Series with regime labels (e.g., 'bullish', 'bearish', etc.)
        
    Returns:
        Dictionary with performance metrics for each regime
    """
    if len(returns) != len(regime_data):
        raise ValueError("Returns and regime data must have the same length")
    
    # Ensure indexes match
    if isinstance(returns.index, pd.DatetimeIndex) and isinstance(regime_data.index, pd.DatetimeIndex):
        common_dates = returns.index.intersection(regime_data.index)
        returns = returns.loc[common_dates]
        regime_data = regime_data.loc[common_dates]
    
    # Get unique regimes
    regimes = regime_data.unique()
    
    # Calculate metrics for each regime
    regime_metrics = {}
    for regime in regimes:
        regime_returns = returns[regime_data == regime]
        if len(regime_returns) > 0:
            metrics = calculate_metrics(regime_returns)
            regime_metrics[regime] = metrics
    
    return regime_metrics 