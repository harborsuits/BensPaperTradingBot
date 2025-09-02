import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

from trading_bot.ai_scoring.strategy_rotator import StrategyRotator
from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
from trading_bot.utils.market_context_fetcher import MarketContextFetcher
from trading_bot.utils.performance_metrics import calculate_metrics
from trading_bot.backtesting.unified_backtester import UnifiedBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    """
    Walk-Forward Backtester implements a robust walk-forward optimization and testing framework.
    
    The walk-forward process involves:
    1. Dividing the full time series into multiple segments
    2. For each segment, using previous data as in-sample (IS) for optimization
    3. Using the current segment as out-of-sample (OOS) for validation
    4. Moving forward through all segments to build a complete OOS performance track record
    
    This approach helps prevent overfitting by continuously validating strategies
    on unseen data and provides a more realistic assessment of strategy performance.
    """
    
    def __init__(
        self,
        strategies: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        is_window_size: int = 180,  # Days for in-sample optimization
        oos_window_size: int = 60,   # Days for out-of-sample testing
        anchor_mode: str = "rolling", # "rolling" or "expanding"
        data_path: str = 'data/backtest_data',
        results_path: str = 'data/backtest_results',
        initial_capital: float = 100000.0,
        rebalance_frequency: str = 'weekly',
        benchmark_symbol: str = 'SPY',
        use_mock: bool = False,
        risk_free_rate: float = 0.03,
    ):
        """
        Initialize the WalkForwardBacktester with configuration parameters.
        
        Args:
            strategies: List of strategy names to include in the backtest
            start_date: Start date for the backtest in 'YYYY-MM-DD' format
            end_date: End date for the backtest in 'YYYY-MM-DD' format
            is_window_size: Size of in-sample window in days
            oos_window_size: Size of out-of-sample window in days
            anchor_mode: "rolling" uses a moving window of is_window_size days,
                         "expanding" uses all data from start up to the current point
            data_path: Directory containing historical data for strategies and market
            results_path: Directory to save backtest results
            initial_capital: Starting capital for the backtest
            rebalance_frequency: How often to rebalance/rotate strategies
            benchmark_symbol: Symbol to use as benchmark (e.g., 'SPY')
            use_mock: Whether to use mock data when market data is unavailable
            risk_free_rate: Annual risk-free rate for performance calculations
        """
        self.strategies = strategies or [
            'trend_following', 'momentum', 'mean_reversion', 
            'breakout_swing', 'volatility_breakout', 'option_spreads'
        ]
        
        # Initialize dates
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=365)
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        
        # Window sizes
        self.is_window_size = is_window_size
        self.oos_window_size = oos_window_size
        self.anchor_mode = anchor_mode
        
        # Configuration
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.benchmark_symbol = benchmark_symbol
        self.use_mock = use_mock
        self.risk_free_rate = risk_free_rate
        
        # Create directories if they don't exist
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.results_path.mkdir(exist_ok=True, parents=True)
        
        # Results storage
        self.segment_results = []
        self.optimized_parameters = []
        self.oos_performance = []
        self.consolidated_metrics = {}
        
        logger.info(f"Initialized WalkForwardBacktester with {len(self.strategies)} strategies")
        logger.info(f"Backtest period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Walk-forward windows: {is_window_size} days in-sample, {oos_window_size} days out-of-sample")
        logger.info(f"Anchor mode: {anchor_mode}")
    
    def generate_time_segments(self) -> List[Dict[str, datetime]]:
        """
        Generate time segments for walk-forward analysis.
        
        Returns:
            List of dictionaries with is_start, is_end, oos_start, oos_end dates
        """
        segments = []
        
        # Calculate total backtest duration
        total_days = (self.end_date - self.start_date).days
        
        # Ensure we have enough data for at least one segment
        if total_days < (self.is_window_size + self.oos_window_size):
            logger.error(f"Insufficient data for walk-forward testing. Need at least "
                        f"{self.is_window_size + self.oos_window_size} days, but only have {total_days} days.")
            return []
        
        # Generate segments
        current_date = self.start_date
        segment_num = 1
        
        while current_date + timedelta(days=self.is_window_size + self.oos_window_size) <= self.end_date:
            # In-sample period
            is_start = current_date
            is_end = current_date + timedelta(days=self.is_window_size)
            
            # Out-of-sample period
            oos_start = is_end + timedelta(days=1)
            oos_end = oos_start + timedelta(days=self.oos_window_size - 1)
            
            segment = {
                "segment_num": segment_num,
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end
            }
            
            segments.append(segment)
            
            # Move to next segment
            if self.anchor_mode == "rolling":
                # Rolling window: move both in-sample and out-of-sample windows forward
                current_date = current_date + timedelta(days=self.oos_window_size)
            else:  # expanding window
                # Expanding window: keep the same start date, only move the out-of-sample window
                current_date = self.start_date
                # Increase in-sample window size
                self.is_window_size += self.oos_window_size
            
            segment_num += 1
        
        logger.info(f"Generated {len(segments)} time segments for walk-forward analysis")
        return segments
    
    def load_data(self):
        """
        Load historical data for all strategies and market.
        """
        # Create a temporary backester instance to load data
        temp_backtester = UnifiedBacktester(
            initial_capital=self.initial_capital,
            strategies=self.strategies,
            start_date=self.start_date.strftime('%Y-%m-%d'),
            end_date=self.end_date.strftime('%Y-%m-%d'),
            rebalance_frequency=self.rebalance_frequency,
            data_path=str(self.data_path),
            results_path=str(self.results_path),
            benchmark_symbol=self.benchmark_symbol,
            use_mock=self.use_mock,
            risk_free_rate=self.risk_free_rate
        )
        
        # Load all necessary data
        temp_backtester.load_strategy_data()
        temp_backtester.load_market_data()
        temp_backtester.load_regime_data()
        temp_backtester.load_benchmark_data()
        
        # Store data for our use
        self.strategy_data = temp_backtester.strategy_data
        self.market_data = temp_backtester.market_data
        self.regime_data = temp_backtester.regime_data
        self.benchmark_data = temp_backtester.benchmark_data
        
        logger.info("Data loaded successfully")
    
    def optimize_parameters(self, is_segment: Dict[str, datetime]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using in-sample data.
        
        Args:
            is_segment: Dictionary with in-sample start and end dates
            
        Returns:
            Dictionary of optimized parameters for each strategy
        """
        is_start = is_segment["is_start"].strftime('%Y-%m-%d')
        is_end = is_segment["is_end"].strftime('%Y-%m-%d')
        
        logger.info(f"Optimizing parameters using in-sample data from {is_start} to {is_end}")
        
        # Create a backtest instance for the in-sample period
        is_backtester = UnifiedBacktester(
            initial_capital=self.initial_capital,
            strategies=self.strategies,
            start_date=is_start,
            end_date=is_end,
            rebalance_frequency=self.rebalance_frequency,
            data_path=str(self.data_path),
            results_path=str(self.results_path),
            benchmark_symbol=self.benchmark_symbol,
            use_mock=self.use_mock,
            risk_free_rate=self.risk_free_rate
        )
        
        # Use pre-loaded data
        is_backtester.strategy_data = self.strategy_data
        is_backtester.market_data = self.market_data
        is_backtester.regime_data = self.regime_data
        is_backtester.benchmark_data = self.benchmark_data
        
        # Grid search for optimal parameters
        # This is where you would implement a grid search or optimization algorithm
        # For demonstration, we'll use a simplified approach focusing on allocation optimization
        
        # Create parameter grid with different allocation strategies
        parameter_grid = [
            {"name": "equal_weight", "allocations": {s: 100/len(self.strategies) for s in self.strategies}},
            {"name": "momentum_heavy", "allocations": self._create_biased_allocation("momentum", 0.4)},
            {"name": "trend_following_heavy", "allocations": self._create_biased_allocation("trend_following", 0.4)},
            {"name": "mean_reversion_heavy", "allocations": self._create_biased_allocation("mean_reversion", 0.4)},
            {"name": "volatility_heavy", "allocations": self._create_biased_allocation("volatility_breakout", 0.4)}
        ]
        
        # Test each parameter set
        best_sharpe = -float('inf')
        best_parameters = None
        
        for params in parameter_grid:
            # Set initial allocations
            is_backtester.initial_allocations = params["allocations"]
            
            # Run backtest with these parameters
            results = is_backtester.run_backtest()
            
            # Store results with parameters
            param_results = {
                "parameters": params,
                "results": results
            }
            
            # Keep track of the best parameter set
            if results["sharpe_ratio"] > best_sharpe:
                best_sharpe = results["sharpe_ratio"]
                best_parameters = params.copy()
        
        logger.info(f"Optimized parameters: {best_parameters}")
        return best_parameters
    
    def _create_biased_allocation(self, primary_strategy: str, primary_weight: float) -> Dict[str, float]:
        """
        Create an allocation dictionary with bias towards a primary strategy.
        
        Args:
            primary_strategy: Strategy to give higher weight
            primary_weight: Weight to give the primary strategy (0-1)
            
        Returns:
            Dictionary of allocations
        """
        remaining_weight = 1.0 - primary_weight
        other_strategies = [s for s in self.strategies if s != primary_strategy]
        
        if not other_strategies:
            return {primary_strategy: 100.0}
        
        allocations = {s: (remaining_weight / len(other_strategies)) * 100 for s in other_strategies}
        allocations[primary_strategy] = primary_weight * 100
        
        return allocations
    
    def run_oos_test(self, oos_segment: Dict[str, datetime], optimized_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run out-of-sample test using optimized parameters.
        
        Args:
            oos_segment: Dictionary with out-of-sample start and end dates
            optimized_params: Optimized parameters from in-sample optimization
            
        Returns:
            Dictionary with OOS test results
        """
        oos_start = oos_segment["oos_start"].strftime('%Y-%m-%d')
        oos_end = oos_segment["oos_end"].strftime('%Y-%m-%d')
        
        logger.info(f"Running OOS test from {oos_start} to {oos_end} with optimized parameters")
        
        # Create a backtest instance for the out-of-sample period
        oos_backtester = UnifiedBacktester(
            initial_capital=self.initial_capital,
            strategies=self.strategies,
            start_date=oos_start,
            end_date=oos_end,
            rebalance_frequency=self.rebalance_frequency,
            data_path=str(self.data_path),
            results_path=str(self.results_path),
            benchmark_symbol=self.benchmark_symbol,
            use_mock=self.use_mock,
            risk_free_rate=self.risk_free_rate
        )
        
        # Use pre-loaded data
        oos_backtester.strategy_data = self.strategy_data
        oos_backtester.market_data = self.market_data
        oos_backtester.regime_data = self.regime_data
        oos_backtester.benchmark_data = self.benchmark_data
        
        # Set initial allocations based on optimized parameters
        if "allocations" in optimized_params:
            oos_backtester.initial_allocations = optimized_params["allocations"]
        
        # Run backtest with optimized parameters
        results = oos_backtester.run_backtest()
        
        # Add segment information to results
        results["segment_info"] = {
            "start_date": oos_start,
            "end_date": oos_end,
            "days": (oos_segment["oos_end"] - oos_segment["oos_start"]).days
        }
        
        return results
    
    def run_walk_forward_backtest(self) -> Dict[str, Any]:
        """
        Run the complete walk-forward optimization and backtest process.
        
        Returns:
            Dictionary with consolidated results
        """
        logger.info("Starting walk-forward backtest")
        
        # Load data
        self.load_data()
        
        # Generate time segments
        segments = self.generate_time_segments()
        
        if not segments:
            logger.error("No valid segments generated. Aborting backtest.")
            return {"error": "No valid segments"}
        
        # Process each segment
        full_oos_portfolio_values = []
        full_oos_dates = []
        
        for segment in segments:
            logger.info(f"Processing segment {segment['segment_num']}")
            
            # Optimize parameters on in-sample data
            optimized_params = self.optimize_parameters(segment)
            self.optimized_parameters.append({
                "segment": segment["segment_num"],
                "is_period": f"{segment['is_start'].strftime('%Y-%m-%d')} to {segment['is_end'].strftime('%Y-%m-%d')}",
                "parameters": optimized_params
            })
            
            # Run out-of-sample test with optimized parameters
            oos_results = self.run_oos_test(segment, optimized_params)
            
            # Store OOS results
            self.oos_performance.append({
                "segment": segment["segment_num"],
                "oos_period": f"{segment['oos_start'].strftime('%Y-%m-%d')} to {segment['oos_end'].strftime('%Y-%m-%d')}",
                "results": oos_results
            })
            
            # Collect portfolio values for full OOS equity curve
            if hasattr(oos_results, 'portfolio_df'):
                dates = oos_results.portfolio_df.index.tolist()
                values = oos_results.portfolio_df['portfolio_value'].tolist()
                
                full_oos_dates.extend(dates)
                full_oos_portfolio_values.extend(values)
        
        # Create consolidated OOS equity curve
        if full_oos_dates and full_oos_portfolio_values:
            oos_equity_df = pd.DataFrame({
                'date': full_oos_dates,
                'portfolio_value': full_oos_portfolio_values
            }).sort_values('date').drop_duplicates('date')
            
            # Calculate consolidated performance metrics
            consolidated_metrics = self._calculate_consolidated_metrics(oos_equity_df)
            self.consolidated_metrics = consolidated_metrics
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info("Walk-forward backtest completed")
        
        return summary
    
    def _calculate_consolidated_metrics(self, oos_equity_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate consolidated performance metrics across all OOS periods.
        
        Args:
            oos_equity_df: DataFrame with date and portfolio_value columns
            
        Returns:
            Dictionary with consolidated performance metrics
        """
        # Calculate returns
        oos_equity_df['daily_return'] = oos_equity_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        returns = oos_equity_df['daily_return'].dropna().values
        
        metrics = calculate_metrics(
            returns=returns,
            risk_free_rate=self.risk_free_rate
        )
        
        # Add additional metrics
        initial_value = oos_equity_df['portfolio_value'].iloc[0]
        final_value = oos_equity_df['portfolio_value'].iloc[-1]
        
        metrics['total_days'] = len(oos_equity_df)
        metrics['initial_value'] = initial_value
        metrics['final_value'] = final_value
        metrics['total_return_dollar'] = final_value - initial_value
        
        return metrics
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of walk-forward backtest results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.oos_performance:
            return {"error": "No out-of-sample results available"}
        
        # Extract key metrics from each OOS segment
        segment_metrics = []
        
        for perf in self.oos_performance:
            segment_num = perf["segment"]
            results = perf["results"]
            
            metrics = {
                "segment": segment_num,
                "period": perf["oos_period"],
                "total_return": results.get("total_return_pct", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "max_drawdown": results.get("max_drawdown_pct", 0),
                "volatility": results.get("volatility_pct", 0),
                "win_rate": results.get("win_rate_pct", 0)
            }
            
            segment_metrics.append(metrics)
        
        # Calculate average metrics across segments
        avg_metrics = {}
        
        for key in ["total_return", "sharpe_ratio", "max_drawdown", "volatility", "win_rate"]:
            avg_metrics[f"avg_{key}"] = np.mean([s[key] for s in segment_metrics])
        
        # Compile summary
        summary = {
            "walk_forward_efficiency": self._calculate_efficiency(),
            "total_segments": len(self.oos_performance),
            "segment_metrics": segment_metrics,
            "average_metrics": avg_metrics,
            "consolidated_metrics": self.consolidated_metrics,
            "configuration": {
                "strategies": self.strategies,
                "is_window_size": self.is_window_size,
                "oos_window_size": self.oos_window_size,
                "anchor_mode": self.anchor_mode,
                "rebalance_frequency": self.rebalance_frequency
            }
        }
        
        return summary
    
    def _calculate_efficiency(self) -> float:
        """
        Calculate walk-forward efficiency (OOS performance / IS performance).
        
        Returns:
            Efficiency ratio
        """
        # Extract average IS and OOS returns
        if not self.optimized_parameters or not self.oos_performance:
            return 0.0
        
        # This is a simplified efficiency calculation
        # In a complete implementation, we would compare expected IS returns
        # with actual OOS returns for each segment
        
        is_returns = []
        oos_returns = []
        
        for i in range(len(self.optimized_parameters)):
            # Find corresponding OOS result
            segment = self.optimized_parameters[i]["segment"]
            
            # Find IS performance (best parameter performance)
            # In a real implementation, this would be extracted from optimization results
            is_return = 0.05  # Placeholder - would be extracted from actual IS results
            is_returns.append(is_return)
            
            # Find OOS performance
            for perf in self.oos_performance:
                if perf["segment"] == segment:
                    oos_return = perf["results"].get("total_return_pct", 0) / 100.0
                    oos_returns.append(oos_return)
                    break
        
        # Calculate average returns
        avg_is_return = np.mean(is_returns) if is_returns else 0
        avg_oos_return = np.mean(oos_returns) if oos_returns else 0
        
        # Calculate efficiency
        if avg_is_return == 0:
            return 0.0
        
        efficiency = avg_oos_return / avg_is_return
        return efficiency
    
    def plot_oos_equity_curve(self, save_path: Optional[str] = None):
        """
        Plot equity curve for combined OOS periods.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.oos_performance:
            logger.warning("No OOS performance data available for plotting")
            return
        
        # Collect all OOS portfolio values
        all_dates = []
        all_values = []
        segment_boundaries = []
        
        for perf in self.oos_performance:
            results = perf["results"]
            
            if hasattr(results, 'portfolio_df'):
                dates = results.portfolio_df.index.tolist()
                values = results.portfolio_df['portfolio_value'].tolist()
                
                # Mark segment boundary
                if all_dates:
                    segment_boundaries.append(len(all_dates))
                
                all_dates.extend(dates)
                all_values.extend(values)
        
        if not all_dates:
            logger.warning("No portfolio values available for plotting")
            return
        
        # Create combined equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(all_dates, all_values, label='OOS Performance')
        
        # Mark segment boundaries
        for boundary in segment_boundaries:
            plt.axvline(x=all_dates[boundary], color='r', linestyle='--', alpha=0.3)
        
        plt.title('Walk-Forward Backtest: Out-of-Sample Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_parameter_stability(self, save_path: Optional[str] = None):
        """
        Plot parameter stability across segments.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.optimized_parameters:
            logger.warning("No optimized parameters available for plotting")
            return
        
        # Extract parameter data
        segments = [p["segment"] for p in self.optimized_parameters]
        
        # For allocation-based parameters, extract allocation for each strategy
        strategy_allocations = {strategy: [] for strategy in self.strategies}
        
        for params in self.optimized_parameters:
            if "parameters" in params and "allocations" in params["parameters"]:
                for strategy in self.strategies:
                    allocation = params["parameters"]["allocations"].get(strategy, 0)
                    strategy_allocations[strategy].append(allocation)
        
        # Plot allocations over time
        plt.figure(figsize=(12, 6))
        
        for strategy, allocations in strategy_allocations.items():
            if len(allocations) == len(segments):
                plt.plot(segments, allocations, marker='o', label=strategy)
        
        plt.title('Parameter Stability: Strategy Allocations Across Segments')
        plt.xlabel('Segment')
        plt.ylabel('Allocation (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_results(self, filename: str = None) -> str:
        """
        Save walk-forward backtest results to file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved results file
        """
        if not self.oos_performance:
            logger.warning("No results to save")
            return ""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"walk_forward_{timestamp}.json"
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.results_path / filename
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return "" 