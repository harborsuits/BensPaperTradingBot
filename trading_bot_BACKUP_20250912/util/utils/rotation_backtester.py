"""
Strategy Rotation Backtester

This module provides backtesting capabilities for the strategy rotation system.
It simulates how the rotation would have performed over historical market data.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from trading_bot.utils.market_context_fetcher import MarketContextFetcher
from trading_bot.ai_scoring.integrated_strategy_rotator import IntegratedStrategyRotator

class RotationBacktester:
    """Backtests the strategy rotation system using historical market data"""
    
    def __init__(self, 
                 strategies: List[str],
                 initial_allocations: Dict[str, float],
                 start_date: str,
                 end_date: Optional[str] = None,
                 initial_capital: float = 100000.0):
        """
        Initialize the backtester
        
        Args:
            strategies: List of strategies to test
            initial_allocations: Initial allocation percentages
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD), defaults to current date
            initial_capital: Initial capital amount
        """
        self.strategies = strategies
        self.initial_allocations = initial_allocations
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        self.initial_capital = initial_capital
        
        # Initialize the market fetcher
        self.market_fetcher = MarketContextFetcher(use_mock=False)
        
        # For storing backtest results
        self.backtest_results = {
            "daily_values": [],
            "allocation_history": [],
            "metrics": {}
        }
    
    def run_backtest(self, 
                    rotation_interval_days: int = 7,
                    use_optimization: bool = True,
                    use_dynamic_constraints: bool = True) -> Dict[str, Any]:
        """
        Run the backtest simulation
        
        Args:
            rotation_interval_days: Number of days between rotation checks
            use_optimization: Whether to use performance optimization
            use_dynamic_constraints: Whether to use dynamic constraints
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Starting backtest from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        # Initialize rotator
        rotator = IntegratedStrategyRotator(
            strategies=self.strategies,
            initial_allocations=self.initial_allocations,
            portfolio_value=self.initial_capital,
            use_mock=False,  # Use historical data, not mock
            config_path=None  # Use default config
        )
        
        # Generate dates for the backtest
        current_date = self.start_date
        dates = []
        while current_date <= self.end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Simulate strategy performance (this would use actual historical returns in a real system)
        # For demo purposes, we'll generate synthetic returns based on some assumptions
        strategy_returns = self._generate_synthetic_returns(dates)
        
        # Run the simulation
        portfolio_values = []
        allocations_history = []
        current_allocations = self.initial_allocations.copy()
        portfolio_value = self.initial_capital
        
        last_rotation_date = None
        
        for date in dates:
            # Check if it's time for rotation
            should_rotate = (
                last_rotation_date is None or 
                (date - last_rotation_date).days >= rotation_interval_days
            )
            
            if should_rotate:
                try:
                    # Get historical market context for this date
                    market_context = self._get_historical_market_context(date)
                    
                    # Update rotator portfolio value
                    rotator.update_portfolio_value(portfolio_value)
                    
                    # Perform rotation
                    rotation_result = rotator.rotate_strategies(market_context=market_context, force_rotation=True)
                    
                    if rotation_result.get("rotated", False):
                        current_allocations = rotation_result.get("new_allocations", {})
                        allocations_history.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "allocations": current_allocations.copy(),
                            "regime": rotation_result.get("regime", "unknown"),
                            "portfolio_value": portfolio_value
                        })
                        last_rotation_date = date
                except Exception as e:
                    print(f"Error rotating on {date.strftime('%Y-%m-%d')}: {str(e)}")
            
            # Calculate portfolio return for the day
            daily_return = 0
            for strategy, allocation in current_allocations.items():
                if strategy in strategy_returns:
                    strategy_daily_return = strategy_returns[strategy].get(date.strftime("%Y-%m-%d"), 0)
                    daily_return += (allocation / 100) * strategy_daily_return
            
            # Update portfolio value
            portfolio_value *= (1 + daily_return)
            
            # Record portfolio value
            portfolio_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": portfolio_value
            })
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_values)
        
        # Store results
        self.backtest_results = {
            "daily_values": portfolio_values,
            "allocation_history": allocations_history,
            "metrics": metrics,
            "settings": {
                "rotation_interval_days": rotation_interval_days,
                "use_optimization": use_optimization,
                "use_dynamic_constraints": use_dynamic_constraints
            }
        }
        
        return self.backtest_results
    
    def _get_historical_market_context(self, date: datetime) -> Dict[str, Any]:
        """
        Get historical market context for a specific date
        
        Args:
            date: Date to get market context for
            
        Returns:
            Market context data
        """
        # In a real implementation, this would fetch actual historical market data
        # For now, we'll use a mock context that simulates different market regimes
        
        # Define a basic cycle of market regimes for demonstration
        year_fraction = (date.timetuple().tm_yday / 366)
        
        # Cycle through regimes
        if year_fraction < 0.25:
            regime = "bullish"
        elif year_fraction < 0.5:
            regime = "sideways"
        elif year_fraction < 0.75:
            regime = "volatile"
        else:
            regime = "bearish"
        
        # Get a mock context with the appropriate regime
        return self.market_fetcher.get_mock_market_context(scenario=regime)
    
    def _generate_synthetic_returns(self, dates: List[datetime]) -> Dict[str, Dict[str, float]]:
        """
        Generate synthetic daily returns for each strategy
        
        Args:
            dates: List of dates for the simulation
            
        Returns:
            Dictionary mapping strategies to their daily returns by date
        """
        # Strategy characteristics (mean return, volatility, and regime sensitivity)
        strategy_profiles = {
            "momentum": {"mean": 0.06/252, "vol": 0.16/np.sqrt(252), "regime_sensitivity": {
                "bullish": 1.5, "bearish": 0.5, "volatile": 0.7, "sideways": 0.8
            }},
            "mean_reversion": {"mean": 0.05/252, "vol": 0.12/np.sqrt(252), "regime_sensitivity": {
                "bullish": 0.7, "bearish": 0.9, "volatile": 1.3, "sideways": 1.5
            }},
            "trend_following": {"mean": 0.08/252, "vol": 0.18/np.sqrt(252), "regime_sensitivity": {
                "bullish": 1.6, "bearish": 1.2, "volatile": 1.0, "sideways": 0.5
            }},
            "breakout_swing": {"mean": 0.09/252, "vol": 0.22/np.sqrt(252), "regime_sensitivity": {
                "bullish": 1.2, "bearish": 0.6, "volatile": 1.7, "sideways": 0.4
            }},
            "volatility_breakout": {"mean": 0.07/252, "vol": 0.20/np.sqrt(252), "regime_sensitivity": {
                "bullish": 0.6, "bearish": 1.0, "volatile": 2.0, "sideways": 0.5
            }},
            "option_spreads": {"mean": 0.04/252, "vol": 0.10/np.sqrt(252), "regime_sensitivity": {
                "bullish": 0.9, "bearish": 1.1, "volatile": 1.4, "sideways": 1.2
            }}
        }
        
        # Generate returns for each strategy
        returns = {}
        for strategy, profile in strategy_profiles.items():
            strategy_returns = {}
            
            # Generate correlated random returns
            np.random.seed(hash(strategy) % 10000)  # Different seed per strategy
            daily_returns = np.random.normal(profile["mean"], profile["vol"], len(dates))
            
            # Apply regime effects
            for i, date in enumerate(dates):
                # Get regime for this date
                regime = self._get_date_regime(date)
                
                # Apply regime sensitivity
                regime_factor = profile["regime_sensitivity"].get(regime, 1.0)
                daily_returns[i] *= regime_factor
                
                # Store return for this date
                strategy_returns[date.strftime("%Y-%m-%d")] = daily_returns[i]
            
            returns[strategy] = strategy_returns
        
        return returns
    
    def _get_date_regime(self, date: datetime) -> str:
        """Determine the market regime for a date based on a simple cycle"""
        year_fraction = (date.timetuple().tm_yday / 366)
        
        if year_fraction < 0.25:
            return "bullish"
        elif year_fraction < 0.5:
            return "sideways"
        elif year_fraction < 0.75:
            return "volatile"
        else:
            return "bearish"
    
    def _calculate_performance_metrics(self, portfolio_values: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest
        
        Args:
            portfolio_values: List of daily portfolio values
            
        Returns:
            Dictionary of performance metrics
        """
        if not portfolio_values:
            return {}
        
        # Convert to pandas series for easier calculation
        values = pd.Series({item["date"]: item["value"] for item in portfolio_values})
        
        # Calculate returns
        returns = values.pct_change().dropna()
        
        # Total return
        start_value = portfolio_values[0]["value"]
        end_value = portfolio_values[-1]["value"]
        total_return = (end_value / start_value) - 1
        
        # Annualized return
        days = len(portfolio_values)
        annualized_return = (1 + total_return) ** (365 / days) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }
    
    def plot_performance(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot the backtest performance
        
        Args:
            save_path: Path to save the plot image (if None, displays plot)
            
        Returns:
            Matplotlib Figure object if save_path is provided, None otherwise
        """
        if not self.backtest_results["daily_values"]:
            print("No backtest data to plot")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio value
        dates = [item["date"] for item in self.backtest_results["daily_values"]]
        values = [item["value"] for item in self.backtest_results["daily_values"]]
        
        ax1.plot(pd.to_datetime(dates), values, label="Portfolio Value")
        ax1.set_title("Strategy Rotation Backtest Performance")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True)
        ax1.legend()
        
        # Format x-axis
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
        fig.autofmt_xdate()
        
        # Plot allocation changes
        if self.backtest_results["allocation_history"]:
            rotation_dates = [item["date"] for item in self.backtest_results["allocation_history"]]
            rotation_values = [item["portfolio_value"] for item in self.backtest_results["allocation_history"]]
            ax1.scatter(pd.to_datetime(rotation_dates), rotation_values, color='red', s=50, label="Rotation Points")
            ax1.legend()
            
            # Create a stacked area chart of allocations
            allocations_df = pd.DataFrame(index=pd.to_datetime(rotation_dates))
            
            for strategy in self.strategies:
                allocations_df[strategy] = [
                    item["allocations"].get(strategy, 0) 
                    for item in self.backtest_results["allocation_history"]
                ]
            
            # Forward fill to have allocations for all dates
            full_date_range = pd.date_range(start=self.start_date, end=self.end_date)
            allocations_df = allocations_df.reindex(full_date_range).fillna(method='ffill')
            
            # Plot stacked area chart
            ax2.stackplot(allocations_df.index, 
                         [allocations_df[s] for s in self.strategies],
                         labels=self.strategies, alpha=0.7)
            
            ax2.set_title("Strategy Allocations Over Time")
            ax2.set_ylabel("Allocation %")
            ax2.set_ylim(0, 100)
            ax2.grid(True)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        # Add performance metrics as text
        metrics = self.backtest_results["metrics"]
        metrics_text = (
            f"Total Return: {metrics.get('total_return', 0):.2%}  "
            f"Ann. Return: {metrics.get('annualized_return', 0):.2%}  "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  "
            f"Max DD: {metrics.get('max_drawdown', 0):.2%}  "
        )
        fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return fig
        else:
            plt.show()
            return None
        
    def export_results_to_csv(self, base_path: str = "./backtest_results") -> Tuple[str, str]:
        """
        Export backtest results to CSV files
        
        Args:
            base_path: Base directory to save files
            
        Returns:
            Tuple of paths to the saved files
        """
        os.makedirs(base_path, exist_ok=True)
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export daily values
        daily_path = f"{base_path}/backtest_values_{timestamp}.csv"
        pd.DataFrame(self.backtest_results["daily_values"]).to_csv(daily_path, index=False)
        
        # Export allocation history
        alloc_path = f"{base_path}/backtest_allocations_{timestamp}.csv"
        
        # Flatten allocation history for CSV export
        flat_allocations = []
        for entry in self.backtest_results["allocation_history"]:
            row = {"date": entry["date"], "regime": entry["regime"]}
            for strategy, allocation in entry["allocations"].items():
                row[f"{strategy}_allocation"] = allocation
            flat_allocations.append(row)
        
        pd.DataFrame(flat_allocations).to_csv(alloc_path, index=False)
        
        return daily_path, alloc_path 