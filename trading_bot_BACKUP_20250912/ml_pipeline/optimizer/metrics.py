"""
Strategy Metrics Module

Provides comprehensive metrics for evaluating trading strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
import math
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class StrategyMetrics:
    """
    Comprehensive metrics calculator for trading strategies
    
    Calculates various risk-adjusted returns, drawdown analysis,
    trade quality metrics, and robustness measures.
    """
    
    def __init__(self):
        """Initialize the metrics calculator"""
        pass
    
    def calculate_all_metrics(self, 
                             trades: List[Dict], 
                             equity_curve: List[Dict], 
                             symbol: str, 
                             final_value: float, 
                             initial_value: float) -> Dict[str, float]:
        """
        Calculate all metrics for a strategy
        
        Args:
            trades: List of trade dictionaries
            equity_curve: List of equity point dictionaries
            symbol: Symbol being analyzed
            final_value: Final portfolio value
            initial_value: Initial portfolio value
            
        Returns:
            Dict with all calculated metrics
        """
        # Basic metrics
        metrics = {
            'symbol': symbol,
            'total_trades': len(trades),
            'total_profit': final_value / initial_value - 1,
            'ending_portfolio_value': final_value
        }
        
        # Add risk metrics
        risk_metrics = self.calculate_risk_metrics(trades, equity_curve)
        metrics.update(risk_metrics)
        
        # Add trade metrics
        trade_metrics = self.calculate_trade_metrics(trades)
        metrics.update(trade_metrics)
        
        # Add robustness metrics if enough trades
        if len(trades) >= 30:
            robustness_metrics = self.calculate_robustness_metrics(trades)
            metrics.update(robustness_metrics)
        
        return metrics
    
    def calculate_risk_metrics(self, trades: List[Dict], equity_curve: List[Dict]) -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics
        
        Args:
            trades: List of trade dictionaries
            equity_curve: List of equity point dictionaries
            
        Returns:
            Dict with risk metrics
        """
        metrics = {}
        
        # Create equity curve dataframe
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate drawdown
            if not equity_df.empty:
                equity_df['peak'] = equity_df['portfolio_value'].cummax()
                equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
                metrics['max_drawdown'] = equity_df['drawdown'].min()
                
                # Calculate average drawdown
                negative_drawdowns = equity_df[equity_df['drawdown'] < 0]['drawdown']
                metrics['avg_drawdown'] = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
                
                # Calculate pain index (average of absolute drawdowns)
                metrics['pain_index'] = equity_df['drawdown'].abs().mean()
                
                # Calculate pain ratio (return / pain index)
                if metrics['pain_index'] > 0 and len(equity_df) > 1:
                    total_return = (equity_df['portfolio_value'].iloc[-1] / equity_df['portfolio_value'].iloc[0]) - 1
                    metrics['pain_ratio'] = total_return / metrics['pain_index']
                else:
                    metrics['pain_ratio'] = 0
                
                # Calculate ulcer index (square root of mean squared drawdown)
                metrics['ulcer_index'] = np.sqrt((equity_df['drawdown'] ** 2).mean())
                
                # Calculate martin ratio (return / ulcer index)
                if metrics['ulcer_index'] > 0 and len(equity_df) > 1:
                    total_return = (equity_df['portfolio_value'].iloc[-1] / equity_df['portfolio_value'].iloc[0]) - 1
                    metrics['martin_ratio'] = total_return / metrics['ulcer_index']
                else:
                    metrics['martin_ratio'] = 0
                
                # Calculate drawdown duration
                equity_df['in_drawdown'] = equity_df['drawdown'] < 0
                if equity_df['in_drawdown'].any():
                    drawdown_groups = (equity_df['in_drawdown'].diff() != 0).cumsum()
                    drawdown_periods = equity_df.groupby(drawdown_groups)
                    
                    # Find the maximum drawdown duration
                    max_duration = 0
                    for name, group in drawdown_periods:
                        if group['in_drawdown'].any():
                            duration = len(group)
                            max_duration = max(max_duration, duration)
                    
                    metrics['max_drawdown_duration'] = max_duration
                else:
                    metrics['max_drawdown_duration'] = 0
                
                # Calculate underwater percentage
                metrics['underwater_percent'] = (equity_df['in_drawdown'].sum() / len(equity_df)) if len(equity_df) > 0 else 0
                
                # Calculate Sharpe ratio (assuming daily data)
                if len(equity_df) > 1:
                    equity_df['return'] = equity_df['portfolio_value'].pct_change().fillna(0)
                    
                    # Annualize - assuming daily data
                    annualization_factor = 252  # Trading days in a year
                    
                    # Sharpe ratio
                    returns_mean = equity_df['return'].mean()
                    returns_std = equity_df['return'].std()
                    if returns_std > 0:
                        sharpe = returns_mean / returns_std * np.sqrt(annualization_factor)
                        metrics['sharpe_ratio'] = sharpe
                    else:
                        metrics['sharpe_ratio'] = float('inf') if returns_mean > 0 else float('-inf')
                    
                    # Sortino ratio (downside deviation only)
                    negative_returns = equity_df['return'][equity_df['return'] < 0]
                    if len(negative_returns) > 0 and negative_returns.std() > 0:
                        sortino = returns_mean / negative_returns.std() * np.sqrt(annualization_factor)
                        metrics['sortino_ratio'] = sortino
                    else:
                        metrics['sortino_ratio'] = float('inf') if returns_mean > 0 else float('-inf')
                    
                    # Calmar ratio (return / max drawdown)
                    if metrics.get('max_drawdown', 0) != 0:
                        calmar = (returns_mean * annualization_factor) / abs(metrics['max_drawdown'])
                        metrics['calmar_ratio'] = calmar
                    else:
                        metrics['calmar_ratio'] = float('inf') if returns_mean > 0 else float('-inf')
                    
                    # Omega ratio (probability weighted ratio of gains versus losses)
                    threshold = 0  # Can be adjusted
                    if len(equity_df['return']) > 0:
                        gains = equity_df['return'][equity_df['return'] > threshold]
                        losses = equity_df['return'][equity_df['return'] <= threshold]
                        
                        if len(losses) > 0 and abs(losses.sum()) > 0:
                            omega = gains.sum() / abs(losses.sum()) if len(gains) > 0 else 0
                            metrics['omega_ratio'] = omega
                        else:
                            metrics['omega_ratio'] = float('inf') if len(gains) > 0 else 0
                    else:
                        metrics['omega_ratio'] = 0
            else:
                # Default values if equity curve is empty
                metrics.update({
                    'max_drawdown': 0,
                    'max_drawdown_duration': 0,
                    'underwater_percent': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'calmar_ratio': 0,
                    'omega_ratio': 0
                })
        
        return metrics
    
    def calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate trade quality metrics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dict with trade metrics
        """
        metrics = {}
        
        if not trades:
            return metrics
        
        # Calculate win rate and profit metrics
        winning_trades = [trade for trade in trades if trade.get('pnl', 0) > 0]
        losing_trades = [trade for trade in trades if trade.get('pnl', 0) <= 0]
        
        metrics['win_rate'] = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate average profit/loss
        if winning_trades:
            metrics['avg_profit'] = sum(trade.get('pnl', 0) for trade in winning_trades) / len(winning_trades)
                if trade['pnl_pct'] > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            metrics['max_consecutive_wins'] = max_consecutive_wins
            metrics['max_consecutive_losses'] = max_consecutive_losses
        else:
            # Default values if no trades
            metrics.update({
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'avg_holding_period_hours': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'kelly_criterion': 0.0,
                'sqn': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            })
        
        return metrics
    
    def calculate_robustness_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate robustness metrics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dict with robustness metrics
        """
        metrics = {}
        
        if len(trades) >= 30:  # Need enough trades for meaningful statistics
            # Monte Carlo simulation
            n_simulations = 1000
            mc_results = self._run_monte_carlo(trades, n_simulations)
            
            metrics['mc_5_percentile_return'] = mc_results['percentiles'][5]
            metrics['mc_50_percentile_return'] = mc_results['percentiles'][50]
            metrics['mc_95_percentile_return'] = mc_results['percentiles'][95]
            metrics['mc_worst_return'] = mc_results['min_return']
            metrics['mc_best_return'] = mc_results['max_return']
            
            # Statistical significance
            pnl_values = [trade['pnl_pct'] for trade in trades]
            t_stat, p_value = stats.ttest_1samp(pnl_values, 0)
            
            metrics['t_statistic'] = t_stat
            metrics['p_value'] = p_value
            metrics['statistically_significant'] = p_value < 0.05
            
            # Strategy consistency
            equity_curve = [trades[i]['portfolio_value'] for i in range(len(trades))]
            rolling_returns = []
            
            window_size = min(20, len(trades) // 3)  # Use smaller window if not enough trades
            if window_size > 0:
                for i in range(len(equity_curve) - window_size):
                    window_return = equity_curve[i + window_size] / equity_curve[i] - 1
                    rolling_returns.append(window_return)
                
                if rolling_returns:
                    metrics['rolling_returns_std'] = np.std(rolling_returns)
                    metrics['consistency_score'] = 1 / (1 + metrics['rolling_returns_std']) if metrics['rolling_returns_std'] > 0 else 1.0
                else:
                    metrics['rolling_returns_std'] = 0.0
                    metrics['consistency_score'] = 0.0
            else:
                metrics['rolling_returns_std'] = 0.0
                metrics['consistency_score'] = 0.0
        
        return metrics
    
    def _run_monte_carlo(self, trades: List[Dict], n_simulations: int) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on trade results
        
        Args:
            trades: List of trade dictionaries
            n_simulations: Number of simulations to run
            
        Returns:
            Dict with Monte Carlo results
        """
        pnl_values = [trade['pnl_pct'] for trade in trades]
        
        # Run simulations
        final_returns = []
        for _ in range(n_simulations):
            # Randomly sample trades with replacement
            sampled_pnls = np.random.choice(pnl_values, size=len(pnl_values), replace=True)
            
            # Calculate compounded return
            final_value = 1.0
            for pnl in sampled_pnls:
                final_value *= (1 + pnl/100)
            
            final_returns.append(final_value - 1)
        
        # Calculate percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[p] = np.percentile(final_returns, p)
        
        return {
            'percentiles': percentiles,
            'min_return': min(final_returns),
            'max_return': max(final_returns),
            'mean_return': np.mean(final_returns),
            'std_return': np.std(final_returns)
        }
    
    def combine_metrics(self, symbol_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine metrics from multiple symbols
        
        Args:
            symbol_results: List of per-symbol metric dictionaries
            
        Returns:
            Dict with combined metrics
        """
        if not symbol_results:
            return {}
        
        # Calculate portfolio level metrics
        combined = {}
        
        # Sum portfolio values
        total_portfolio_value = sum(result.get('ending_portfolio_value', 0) for result in symbol_results)
        initial_value = 1000 * len(symbol_results)
        
        # Calculate total profit
        combined['total_profit'] = (total_portfolio_value / initial_value) - 1
        
        # Average risk metrics
        for metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio', 'sqn', 'consistency_score']:
            values = [result.get(metric, 0) for result in symbol_results]
            combined[metric] = np.mean(values) if values else 0
        
        # Weighted average of trade metrics by number of trades
        weighted_metrics = ['win_rate', 'expectancy', 'profit_factor', 'kelly_criterion']
        for metric in weighted_metrics:
            total_weight = sum(result.get('total_trades', 0) for result in symbol_results)
            if total_weight > 0:
                weighted_sum = sum(
                    result.get(metric, 0) * result.get('total_trades', 0) 
                    for result in symbol_results
                )
                combined[metric] = weighted_sum / total_weight
            else:
                combined[metric] = 0
        
        # Max drawdown is the worst across all symbols
        max_drawdowns = [result.get('max_drawdown', 0) for result in symbol_results]
        combined['max_drawdown'] = min(max_drawdowns) if max_drawdowns else 0
        
        # Total trades
        combined['total_trades'] = sum(result.get('total_trades', 0) for result in symbol_results)
        
        # Number of symbols traded
        combined['symbols_traded'] = len(symbol_results)
        
        # Statistical significance - combined p-value using Fisher's method
        p_values = [result.get('p_value', 1.0) for result in symbol_results if 'p_value' in result]
        if p_values:
            # Fisher's method for combining p-values
            chi_square = -2 * sum(np.log(p) for p in p_values)
            combined_p = 1 - stats.chi2.cdf(chi_square, 2 * len(p_values))
            combined['combined_p_value'] = combined_p
            combined['combined_statistical_significance'] = combined_p < 0.05
        
        return combined
