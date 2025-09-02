#!/usr/bin/env python
"""
BensBot Autonomous Trading Pipeline Demo
This script demonstrates the autonomous strategy selection and rotation pipeline
with minimal UI clutter - just the core functionality.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path if needed
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create results directory
os.makedirs("demo_results", exist_ok=True)

class AutomatedTradingPipeline:
    """Simplified demonstration of the autonomous trading pipeline."""
    
    def __init__(self):
        """Initialize the demonstration pipeline."""
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        self.strategies = ["momentum", "mean_reversion", "trend_following", "breakout", "volatility"]
        self.start_date = datetime.now() - timedelta(days=365)
        self.end_date = datetime.now()
        self.initial_capital = 100000.0
        
        # Strategy performance characteristics (in a real system, this comes from backtesting)
        self.strategy_performance = {
            "momentum": {"sharpe": 1.2, "volatility": 0.15, "correlation": {"SPY": 0.65}},
            "mean_reversion": {"sharpe": 0.9, "volatility": 0.12, "correlation": {"SPY": 0.3}},
            "trend_following": {"sharpe": 1.4, "volatility": 0.22, "correlation": {"SPY": 0.7}},
            "breakout": {"sharpe": 1.1, "volatility": 0.18, "correlation": {"SPY": 0.6}},
            "volatility": {"sharpe": 1.0, "volatility": 0.25, "correlation": {"SPY": 0.2}}
        }
        
        # Market regime data (in a real system, this is detected from market data)
        self.market_regimes = self._generate_market_regimes()
        
        # Strategy allocations will be determined by the autonomous system
        self.strategy_allocations = {}
        self.performance_data = None
    
    def _generate_market_regimes(self):
        """Generate simulated market regime data."""
        # In a real system, this would be determined from market indicators
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='W-FRI')
        
        # Randomly assign regimes: bull, bear, sideways, volatile
        regimes = np.random.choice(
            ["bull", "bear", "sideways", "volatile"], 
            size=len(date_range),
            p=[0.4, 0.2, 0.3, 0.1]  # Probabilities of each regime
        )
        
        # Create a dataframe with dates and regimes
        return pd.DataFrame({
            'date': date_range,
            'regime': regimes
        })
    
    def _select_optimal_strategies(self, date, regime, max_strategies=2):
        """
        Autonomously select the optimal strategies based on market regime.
        
        In the real system, this uses machine learning and performance analysis.
        This is a simplified version for demonstration.
        """
        logger.info(f"Selecting optimal strategies for {date.strftime('%Y-%m-%d')}, regime: {regime}")
        
        # Strategy preferences based on regime
        regime_preferences = {
            "bull": ["momentum", "trend_following", "breakout"],
            "bear": ["mean_reversion", "volatility"],
            "sideways": ["mean_reversion", "volatility"],
            "volatile": ["volatility", "breakout"]
        }
        
        # Preferred strategies for this regime
        preferred = regime_preferences[regime]
        
        # For demonstration, select top strategies based on Sharpe ratio
        # In real system, would be more sophisticated with ML-based selection
        candidates = [(s, self.strategy_performance[s]["sharpe"]) for s in self.strategies]
        
        # Sort by Sharpe, with bonus for preferred strategies
        sorted_strategies = sorted(
            candidates,
            key=lambda x: x[1] + (0.5 if x[0] in preferred else 0),
            reverse=True
        )
        
        # Select top strategies
        selected = [s[0] for s in sorted_strategies[:max_strategies]]
        
        logger.info(f"Selected strategies: {selected}")
        
        # Record selection rationale (for transparency)
        self.strategy_allocations[date.strftime('%Y-%m-%d')] = {
            "strategies": selected,
            "regime": regime,
            "rationale": f"Selected based on {regime} market regime and Sharpe ratio"
        }
        
        return selected
    
    def _simulate_strategy_performance(self, strategy, start_date, end_date, market_regime):
        """
        Simulate the performance of a strategy over a time period.
        
        In the real system, this would be actual backtest results. 
        This is a simplified simulation for demonstration.
        """
        # Number of trading days
        days = (end_date - start_date).days
        trading_days = min(days, 5 * (days // 7))  # Approximate trading days
        
        # Base mean and std based on strategy characteristics
        strategy_sharpe = self.strategy_performance[strategy]["sharpe"]
        strategy_vol = self.strategy_performance[strategy]["volatility"]
        
        # Adjust returns based on market regime
        regime_adjustments = {
            "bull": {"mean_adj": 0.03, "vol_adj": -0.05},
            "bear": {"mean_adj": -0.03, "vol_adj": 0.1},
            "sideways": {"mean_adj": 0.0, "vol_adj": -0.02},
            "volatile": {"mean_adj": 0.0, "vol_adj": 0.15}
        }
        
        adj = regime_adjustments[market_regime]
        
        # Generate returns
        mean = (strategy_sharpe * strategy_vol / 16) + adj["mean_adj"]  # Annualized to daily
        vol = (strategy_vol / 16) * (1 + adj["vol_adj"])  # Annualized to daily
        
        # Generate slightly autocorrelated returns for realism
        base_returns = np.random.normal(mean, vol, trading_days)
        autocorr = 0.2  # Slight autocorrelation for realism
        for i in range(1, len(base_returns)):
            base_returns[i] = (1-autocorr) * base_returns[i] + autocorr * base_returns[i-1]
        
        # Create date range and returns series
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')[:trading_days]
        returns = pd.Series(base_returns, index=date_range)
        
        return returns
    
    def run_autonomous_pipeline(self):
        """Run the autonomous trading pipeline demonstration."""
        logger.info("Starting autonomous trading pipeline demonstration")
        
        # We'll store all strategy returns and portfolio performance
        all_returns = {}
        portfolio_returns = []
        portfolio_value = pd.Series([self.initial_capital], index=[self.start_date])
        current_value = self.initial_capital
        
        # Simulate weekly rebalancing and strategy rotation
        for i in range(len(self.market_regimes) - 1):
            current_date = self.market_regimes.iloc[i]['date']
            next_date = self.market_regimes.iloc[i+1]['date']
            current_regime = self.market_regimes.iloc[i]['regime']
            
            # Autonomous strategy selection
            selected_strategies = self._select_optimal_strategies(
                current_date, current_regime
            )
            
            # Simulate performance of each strategy
            strategy_returns = {}
            for strategy in selected_strategies:
                returns = self._simulate_strategy_performance(
                    strategy, current_date, next_date, current_regime
                )
                strategy_returns[strategy] = returns
                
                # Store for overall comparison
                if strategy not in all_returns:
                    all_returns[strategy] = []
                all_returns[strategy].append(returns)
            
            # Equal weight allocation
            if strategy_returns:
                # Combine strategy returns
                combined_returns = pd.concat(strategy_returns.values()).groupby(level=0).mean()
                portfolio_returns.append(combined_returns)
                
                # Update portfolio value
                period_return = (1 + combined_returns).prod() - 1
                current_value *= (1 + period_return)
                # Use pd.concat instead of append which is deprecated
                portfolio_value = pd.concat([
                    portfolio_value,
                    pd.Series([current_value], index=[next_date])
                ])
        
        # Combine all portfolio returns into a single series
        all_portfolio_returns = pd.concat(portfolio_returns)
        
        # Calculate portfolio performance metrics
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        days = (self.end_date - self.start_date).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        volatility = all_portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdowns
        portfolio_cum_returns = (1 + all_portfolio_returns).cumprod()
        running_max = portfolio_cum_returns.cummax()
        drawdowns = (portfolio_cum_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Store performance data
        self.performance_data = {
            "portfolio_value": portfolio_value,
            "portfolio_returns": all_portfolio_returns,
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "strategy_allocations": self.strategy_allocations
        }
        
        logger.info(f"Autonomous pipeline complete. Final portfolio value: ${portfolio_value.iloc[-1]:,.2f}")
        return self.performance_data
    
    def generate_report(self):
        """Generate a report on the autonomous trading pipeline results."""
        if not self.performance_data:
            raise ValueError("Must run the pipeline first")
        
        # Create report file
        with open("demo_results/autonomous_pipeline_report.txt", "w") as f:
            f.write("======== BENSBOT AUTONOMOUS TRADING PIPELINE REPORT ========\n\n")
            
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"Final Portfolio Value: ${self.performance_data['portfolio_value'].iloc[-1]:,.2f}\n")
            f.write(f"Total Return: {self.performance_data['total_return']:.2%}\n")
            f.write(f"Annualized Return: {self.performance_data['annual_return']:.2%}\n")
            f.write(f"Volatility: {self.performance_data['volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {self.performance_data['sharpe_ratio']:.2f}\n")
            f.write(f"Maximum Drawdown: {self.performance_data['max_drawdown']:.2%}\n\n")
            
            f.write("== AUTONOMOUS STRATEGY SELECTION DECISIONS ==\n\n")
            for date, allocation in self.strategy_allocations.items():
                f.write(f"Period Starting {date}:\n")
                f.write(f"  Market Regime: {allocation['regime']}\n")
                f.write(f"  Selected Strategies: {', '.join(allocation['strategies'])}\n")
                f.write(f"  Rationale: {allocation['rationale']}\n\n")
        
        logger.info("Generated report at demo_results/autonomous_pipeline_report.txt")
    
    def generate_visualizations(self):
        """Generate visualizations of the autonomous trading pipeline results."""
        if not self.performance_data:
            raise ValueError("Must run the pipeline first")
        
        # Plot portfolio value
        plt.figure(figsize=(12, 6))
        self.performance_data['portfolio_value'].plot()
        plt.title('Portfolio Value - Autonomous Strategy Selection', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(alpha=0.3)
        plt.savefig('demo_results/portfolio_value.png', dpi=300, bbox_inches='tight')
        
        # Plot strategy allocations over time
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Color coding for market regimes
        regime_colors = {
            'bull': 'green',
            'bear': 'red',
            'sideways': 'blue',
            'volatile': 'purple'
        }
        
        # Plot portfolio value
        portfolio_value = self.performance_data['portfolio_value']
        ax.plot(portfolio_value.index, portfolio_value.values, 'k-', linewidth=2, label='Portfolio Value')
        
        # Mark regime changes and strategy selections
        last_y = portfolio_value.iloc[0]
        for date, allocation in self.strategy_allocations.items():
            dt = pd.to_datetime(date)
            if dt in portfolio_value.index:
                y = portfolio_value.loc[dt]
                regime = allocation['regime']
                strategies = allocation['strategies']
                
                # Draw line marking regime change
                ax.axvline(x=dt, color=regime_colors[regime], linestyle='--', alpha=0.7)
                
                # Annotate with strategies
                ax.annotate(f"{regime}: {', '.join(strategies)}", 
                           xy=(dt, y),
                           xytext=(10, 0),
                           textcoords="offset points",
                           rotation=90,
                           fontsize=10,
                           color=regime_colors[regime],
                           va='bottom')
                
                last_y = y
        
        # Add legend for market regimes
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, lw=2, linestyle='--', label=regime) 
                          for regime, color in regime_colors.items()]
        legend_elements.append(Line2D([0], [0], color='k', lw=2, label='Portfolio Value'))
        ax.legend(handles=legend_elements)
        
        plt.title('Autonomous Strategy Selection and Market Regimes', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo_results/autonomous_strategy_rotation.png', dpi=300, bbox_inches='tight')
        
        logger.info("Generated visualizations in demo_results directory")


if __name__ == "__main__":
    # Run the autonomous pipeline demonstration
    pipeline = AutomatedTradingPipeline()
    results = pipeline.run_autonomous_pipeline()
    
    # Generate report and visualizations
    pipeline.generate_report()
    pipeline.generate_visualizations()
    
    # Display summary results in terminal
    print("\n========== AUTONOMOUS TRADING PIPELINE RESULTS ==========")
    print(f"Initial Capital: ${pipeline.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
    print("=========================================================")
    
    print("\nStrategy Selections by Market Regime:")
    for date, allocation in sorted(pipeline.strategy_allocations.items()):
        print(f"  {date} ({allocation['regime']}): {', '.join(allocation['strategies'])}")
    
    print("\nReport and visualizations saved to demo_results directory")
    print(" - demo_results/autonomous_pipeline_report.txt")
    print(" - demo_results/portfolio_value.png")
    print(" - demo_results/autonomous_strategy_rotation.png")
