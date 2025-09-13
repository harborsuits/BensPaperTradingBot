import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkComparison:
    """
    Benchmark comparison module for evaluating strategy performance against market benchmarks.
    
    This module provides:
    1. Performance comparison against multiple benchmarks
    2. Risk-adjusted return metrics relative to benchmarks
    3. Rolling correlation and beta analysis
    4. Regime-specific benchmark comparison
    5. Visualization tools for benchmark analysis
    """
    
    def __init__(
        self,
        portfolio_returns: Union[pd.Series, pd.DataFrame],
        benchmark_data_path: str = 'data/benchmarks',
        risk_free_rate: float = 0.03,
        frequency: str = 'D',  # 'D' for daily, 'W' for weekly, 'M' for monthly
        include_standard_benchmarks: bool = True
    ):
        """
        Initialize the benchmark comparison module.
        
        Args:
            portfolio_returns: Series or DataFrame of portfolio returns
            benchmark_data_path: Path to benchmark data files
            risk_free_rate: Annual risk-free rate (e.g., 0.03 for 3%)
            frequency: Data frequency ('D', 'W', 'M')
            include_standard_benchmarks: Whether to include standard benchmarks (SPY, AGG, etc.)
        """
        # Convert portfolio returns to Series if DataFrame
        if isinstance(portfolio_returns, pd.DataFrame) and 'daily_return' in portfolio_returns.columns:
            self.portfolio_returns = portfolio_returns['daily_return']
            self.portfolio_dates = portfolio_returns.index if portfolio_returns.index.name == 'date' else portfolio_returns['date']
        else:
            self.portfolio_returns = portfolio_returns
            self.portfolio_dates = portfolio_returns.index
        
        self.benchmark_data_path = Path(benchmark_data_path)
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        
        # Adjust risk-free rate for data frequency
        self.periodic_risk_free = self._adjust_risk_free_rate(risk_free_rate, frequency)
        
        # Load benchmark data
        self.benchmarks = {}
        if include_standard_benchmarks:
            self._load_standard_benchmarks()
        
        logger.info(f"Initialized BenchmarkComparison with {len(self.portfolio_returns)} data points")
    
    def _adjust_risk_free_rate(self, annual_rate: float, frequency: str) -> float:
        """
        Adjust the annual risk-free rate to match the data frequency.
        
        Args:
            annual_rate: Annual risk-free rate
            frequency: Data frequency
            
        Returns:
            Adjusted risk-free rate for the specified frequency
        """
        if frequency == 'D':
            # Daily rate (approximate)
            return annual_rate / 252
        elif frequency == 'W':
            # Weekly rate
            return annual_rate / 52
        elif frequency == 'M':
            # Monthly rate
            return annual_rate / 12
        else:
            # Default to daily
            return annual_rate / 252
    
    def _load_standard_benchmarks(self) -> None:
        """
        Load standard benchmark data.
        """
        # List of standard benchmarks
        standard_benchmarks = {
            'spy': 'S&P 500 ETF',
            'agg': 'US Aggregate Bond ETF',
            'qqq': 'Nasdaq 100 ETF',
            'iwm': 'Russell 2000 ETF',
            'gld': 'Gold ETF',
            'veu': 'International Equity ETF',
            'tlt': 'Long-Term Treasury ETF'
        }
        
        # Load each benchmark
        for symbol, name in standard_benchmarks.items():
            file_path = self.benchmark_data_path / f"{symbol}.csv"
            try:
                if file_path.exists():
                    self.load_benchmark(symbol, file_path, name)
                else:
                    logger.warning(f"Benchmark file {file_path} not found")
            except Exception as e:
                logger.error(f"Error loading benchmark {symbol}: {e}")
    
    def load_benchmark(
        self,
        benchmark_id: str,
        data_path: Union[str, Path],
        name: str = None,
        return_column: str = 'daily_return'
    ) -> bool:
        """
        Load benchmark data from a file.
        
        Args:
            benchmark_id: Unique identifier for the benchmark
            data_path: Path to the benchmark data file
            name: Descriptive name for the benchmark
            return_column: Column name for returns in the data file
            
        Returns:
            Success flag
        """
        try:
            # Load benchmark data
            data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            
            # Extract returns
            if return_column in data.columns:
                returns = data[return_column]
            elif 'close' in data.columns:
                # Calculate returns from prices
                returns = data['close'].pct_change().dropna()
            else:
                logger.error(f"No return or price data found in {data_path}")
                return False
            
            # Store benchmark
            self.benchmarks[benchmark_id] = {
                'name': name or benchmark_id,
                'returns': returns,
                'data': data
            }
            
            logger.info(f"Loaded benchmark {benchmark_id} with {len(returns)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error loading benchmark {benchmark_id} from {data_path}: {e}")
            return False
    
    def add_custom_benchmark(
        self,
        benchmark_id: str,
        returns: pd.Series,
        name: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a custom benchmark series.
        
        Args:
            benchmark_id: Unique identifier for the benchmark
            returns: Series of benchmark returns
            name: Descriptive name for the benchmark
            metadata: Additional metadata for the benchmark
        """
        if benchmark_id in self.benchmarks:
            logger.warning(f"Overwriting existing benchmark {benchmark_id}")
        
        self.benchmarks[benchmark_id] = {
            'name': name or benchmark_id,
            'returns': returns,
            'metadata': metadata or {}
        }
        
        logger.info(f"Added custom benchmark {benchmark_id} with {len(returns)} data points")
    
    def align_data(self, benchmark_id: str) -> pd.DataFrame:
        """
        Align portfolio and benchmark data on shared dates.
        
        Args:
            benchmark_id: Benchmark identifier
            
        Returns:
            DataFrame with aligned portfolio and benchmark returns
        """
        if benchmark_id not in self.benchmarks:
            logger.error(f"Benchmark {benchmark_id} not found")
            return pd.DataFrame()
        
        # Get benchmark returns
        benchmark_returns = self.benchmarks[benchmark_id]['returns']
        
        # Create DataFrame with portfolio returns
        aligned_data = pd.DataFrame({'portfolio': self.portfolio_returns})
        
        # Add benchmark returns
        aligned_data[benchmark_id] = benchmark_returns
        
        # Ensure data is aligned on dates
        aligned_data = aligned_data.dropna()
        
        return aligned_data
    
    def calculate_relative_metrics(
        self,
        benchmark_id: str
    ) -> Dict[str, float]:
        """
        Calculate performance metrics relative to a benchmark.
        
        Args:
            benchmark_id: Benchmark identifier
            
        Returns:
            Dictionary of performance metrics
        """
        if benchmark_id not in self.benchmarks:
            logger.error(f"Benchmark {benchmark_id} not found")
            return {}
        
        # Align data
        aligned_data = self.align_data(benchmark_id)
        
        if aligned_data.empty:
            logger.warning(f"No aligned data found for benchmark {benchmark_id}")
            return {}
        
        # Extract returns
        portfolio_returns = aligned_data['portfolio']
        benchmark_returns = aligned_data[benchmark_id]
        
        # Calculate excess returns
        portfolio_excess = portfolio_returns - self.periodic_risk_free
        benchmark_excess = benchmark_returns - self.periodic_risk_free
        
        # Calculate beta
        cov_matrix = aligned_data.cov()
        beta = cov_matrix.loc['portfolio', benchmark_id] / cov_matrix.loc[benchmark_id, benchmark_id]
        
        # Calculate alpha (Jensen's alpha)
        expected_return = self.periodic_risk_free + beta * (benchmark_returns.mean() - self.periodic_risk_free)
        alpha = portfolio_returns.mean() - expected_return
        
        # Annualize alpha based on frequency
        if self.frequency == 'D':
            alpha_annualized = alpha * 252
        elif self.frequency == 'W':
            alpha_annualized = alpha * 52
        elif self.frequency == 'M':
            alpha_annualized = alpha * 12
        else:
            alpha_annualized = alpha * 252
        
        # Calculate tracking error
        tracking_difference = portfolio_returns - benchmark_returns
        tracking_error = tracking_difference.std()
        
        # Calculate information ratio
        information_ratio = tracking_difference.mean() / tracking_error if tracking_error > 0 else 0
        
        # Calculate up/down capture
        up_benchmark = benchmark_returns > 0
        down_benchmark = benchmark_returns < 0
        
        up_capture = (portfolio_returns[up_benchmark].mean() / benchmark_returns[up_benchmark].mean()) * 100 if up_benchmark.any() else 0
        down_capture = (portfolio_returns[down_benchmark].mean() / benchmark_returns[down_benchmark].mean()) * 100 if down_benchmark.any() else 0
        
        # Calculate correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        return {
            'alpha': alpha,
            'alpha_annualized': alpha_annualized,
            'beta': beta,
            'r_squared': r_squared,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture
        }
    
    def calculate_rolling_metrics(
        self,
        benchmark_id: str,
        window: int = 90,
        min_periods: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics relative to a benchmark.
        
        Args:
            benchmark_id: Benchmark identifier
            window: Rolling window size
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with rolling metrics
        """
        if benchmark_id not in self.benchmarks:
            logger.error(f"Benchmark {benchmark_id} not found")
            return pd.DataFrame()
        
        # Align data
        aligned_data = self.align_data(benchmark_id)
        
        if aligned_data.empty:
            logger.warning(f"No aligned data found for benchmark {benchmark_id}")
            return pd.DataFrame()
        
        # Calculate rolling correlation
        rolling_corr = aligned_data['portfolio'].rolling(window=window, min_periods=min_periods).corr(aligned_data[benchmark_id])
        
        # Calculate rolling beta
        def rolling_beta(x):
            if len(x) < min_periods:
                return np.nan
            cov = np.cov(x['portfolio'], x[benchmark_id])
            if cov[1, 1] == 0:
                return np.nan
            return cov[0, 1] / cov[1, 1]
        
        rolling_beta = aligned_data.rolling(window=window, min_periods=min_periods).apply(rolling_beta)
        
        # Calculate rolling alpha
        def rolling_alpha(x):
            if len(x) < min_periods:
                return np.nan
            portfolio_mean = x['portfolio'].mean()
            benchmark_mean = x[benchmark_id].mean()
            beta = np.cov(x['portfolio'], x[benchmark_id])[0, 1] / np.var(x[benchmark_id])
            return portfolio_mean - self.periodic_risk_free - beta * (benchmark_mean - self.periodic_risk_free)
        
        rolling_alpha = aligned_data.rolling(window=window, min_periods=min_periods).apply(rolling_alpha)
        
        # Combine results
        rolling_metrics = pd.DataFrame({
            'correlation': rolling_corr,
            'beta': rolling_beta,
            'alpha': rolling_alpha
        }, index=aligned_data.index)
        
        return rolling_metrics
    
    def plot_cumulative_comparison(
        self,
        benchmark_ids: List[str] = None,
        title: str = "Cumulative Return Comparison",
        figsize: tuple = (12, 6),
        save_path: str = None
    ) -> None:
        """
        Plot cumulative returns for portfolio and benchmarks.
        
        Args:
            benchmark_ids: List of benchmark identifiers (None for all)
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot image
        """
        if not benchmark_ids:
            benchmark_ids = list(self.benchmarks.keys())
        
        # Filter to existing benchmarks
        benchmark_ids = [bid for bid in benchmark_ids if bid in self.benchmarks]
        
        if not benchmark_ids:
            logger.warning("No valid benchmarks to plot")
            return
        
        # Prepare data
        aligned_all = pd.DataFrame({'portfolio': self.portfolio_returns})
        
        for bid in benchmark_ids:
            benchmark_returns = self.benchmarks[bid]['returns']
            aligned_all[bid] = benchmark_returns
        
        # Drop rows with NaN values
        aligned_all = aligned_all.dropna()
        
        if aligned_all.empty:
            logger.warning("No aligned data found for plotting")
            return
        
        # Calculate cumulative returns
        cumulative_returns = (1 + aligned_all).cumprod()
        
        # Plot
        plt.figure(figsize=figsize)
        plt.plot(cumulative_returns.index, cumulative_returns['portfolio'], label='Portfolio', linewidth=2)
        
        for bid in benchmark_ids:
            benchmark_name = self.benchmarks[bid]['name']
            plt.plot(cumulative_returns.index, cumulative_returns[bid], label=benchmark_name, alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_rolling_correlation(
        self,
        benchmark_ids: List[str] = None,
        window: int = 90,
        title: str = "Rolling Correlation",
        figsize: tuple = (12, 6),
        save_path: str = None
    ) -> None:
        """
        Plot rolling correlation with benchmarks.
        
        Args:
            benchmark_ids: List of benchmark identifiers (None for all)
            window: Rolling window size
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot image
        """
        if not benchmark_ids:
            benchmark_ids = list(self.benchmarks.keys())
        
        # Filter to existing benchmarks
        benchmark_ids = [bid for bid in benchmark_ids if bid in self.benchmarks]
        
        if not benchmark_ids:
            logger.warning("No valid benchmarks to plot")
            return
        
        # Calculate rolling correlations
        all_corrs = pd.DataFrame()
        
        for bid in benchmark_ids:
            aligned_data = self.align_data(bid)
            if not aligned_data.empty:
                rolling_corr = aligned_data['portfolio'].rolling(window=window).corr(aligned_data[bid])
                all_corrs[bid] = rolling_corr
        
        if all_corrs.empty:
            logger.warning("No correlation data to plot")
            return
        
        # Plot
        plt.figure(figsize=figsize)
        
        for bid in all_corrs.columns:
            benchmark_name = self.benchmarks[bid]['name']
            plt.plot(all_corrs.index, all_corrs[bid], label=benchmark_name)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_rolling_beta(
        self,
        benchmark_ids: List[str] = None,
        window: int = 90,
        title: str = "Rolling Beta",
        figsize: tuple = (12, 6),
        save_path: str = None
    ) -> None:
        """
        Plot rolling beta relative to benchmarks.
        
        Args:
            benchmark_ids: List of benchmark identifiers (None for all)
            window: Rolling window size
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot image
        """
        if not benchmark_ids:
            benchmark_ids = list(self.benchmarks.keys())[:3]  # Limit to first 3 by default
        
        # Filter to existing benchmarks
        benchmark_ids = [bid for bid in benchmark_ids if bid in self.benchmarks]
        
        if not benchmark_ids:
            logger.warning("No valid benchmarks to plot")
            return
        
        # Calculate rolling betas
        all_betas = pd.DataFrame()
        
        for bid in benchmark_ids:
            rolling_metrics = self.calculate_rolling_metrics(bid, window=window)
            if not rolling_metrics.empty:
                all_betas[bid] = rolling_metrics['beta']
        
        if all_betas.empty:
            logger.warning("No beta data to plot")
            return
        
        # Plot
        plt.figure(figsize=figsize)
        
        for bid in all_betas.columns:
            benchmark_name = self.benchmarks[bid]['name']
            plt.plot(all_betas.index, all_betas[bid], label=benchmark_name)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Beta')
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def generate_benchmark_report(
        self,
        benchmark_ids: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a comprehensive benchmark comparison report.
        
        Args:
            benchmark_ids: List of benchmark identifiers (None for all)
            
        Returns:
            Dictionary with benchmark comparison results
        """
        if not benchmark_ids:
            benchmark_ids = list(self.benchmarks.keys())
        
        # Filter to existing benchmarks
        benchmark_ids = [bid for bid in benchmark_ids if bid in self.benchmarks]
        
        if not benchmark_ids:
            logger.warning("No valid benchmarks for report")
            return {}
        
        # Calculate metrics for each benchmark
        benchmark_metrics = {}
        
        for bid in benchmark_ids:
            metrics = self.calculate_relative_metrics(bid)
            if metrics:
                benchmark_name = self.benchmarks[bid]['name']
                benchmark_metrics[bid] = {
                    'name': benchmark_name,
                    'metrics': metrics
                }
        
        # Calculate portfolio metrics
        portfolio_returns = self.portfolio_returns.dropna()
        
        if not portfolio_returns.empty:
            mean_return = portfolio_returns.mean()
            annualized_return = self._annualize_return(mean_return)
            volatility = portfolio_returns.std()
            annualized_volatility = self._annualize_volatility(volatility)
            
            sharpe = (mean_return - self.periodic_risk_free) / volatility if volatility > 0 else 0
            annualized_sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            portfolio_metrics = {
                'mean_return': mean_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe,
                'annualized_sharpe': annualized_sharpe
            }
        else:
            portfolio_metrics = {}
        
        # Combine results
        report = {
            'portfolio': portfolio_metrics,
            'benchmarks': benchmark_metrics
        }
        
        return report
    
    def _annualize_return(self, periodic_return: float) -> float:
        """
        Annualize a periodic return.
        
        Args:
            periodic_return: Return for the current frequency
            
        Returns:
            Annualized return
        """
        if self.frequency == 'D':
            return (1 + periodic_return) ** 252 - 1
        elif self.frequency == 'W':
            return (1 + periodic_return) ** 52 - 1
        elif self.frequency == 'M':
            return (1 + periodic_return) ** 12 - 1
        else:
            return (1 + periodic_return) ** 252 - 1
    
    def _annualize_volatility(self, periodic_vol: float) -> float:
        """
        Annualize periodic volatility.
        
        Args:
            periodic_vol: Volatility for the current frequency
            
        Returns:
            Annualized volatility
        """
        if self.frequency == 'D':
            return periodic_vol * np.sqrt(252)
        elif self.frequency == 'W':
            return periodic_vol * np.sqrt(52)
        elif self.frequency == 'M':
            return periodic_vol * np.sqrt(12)
        else:
            return periodic_vol * np.sqrt(252)
    
    def get_regime_specific_comparison(
        self,
        benchmark_id: str,
        regime_data: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate benchmark comparison metrics specific to market regimes.
        
        Args:
            benchmark_id: Benchmark identifier
            regime_data: Series with regime labels indexed by date
            
        Returns:
            Dictionary with regime-specific comparison metrics
        """
        if benchmark_id not in self.benchmarks:
            logger.error(f"Benchmark {benchmark_id} not found")
            return {}
        
        # Align data
        aligned_data = self.align_data(benchmark_id)
        
        if aligned_data.empty:
            logger.warning(f"No aligned data found for benchmark {benchmark_id}")
            return {}
        
        # Add regime data
        if aligned_data.index.name == 'date':
            aligned_data['regime'] = regime_data
        else:
            aligned_data['regime'] = regime_data.reindex(aligned_data.index)
        
        # Drop rows with missing regime data
        aligned_data = aligned_data.dropna()
        
        if aligned_data.empty:
            logger.warning("No data with regime labels")
            return {}
        
        # Calculate metrics for each regime
        regime_metrics = {}
        
        for regime, group in aligned_data.groupby('regime'):
            # Skip if not enough data
            if len(group) < 20:
                continue
            
            # Extract returns
            portfolio_returns = group['portfolio']
            benchmark_returns = group[benchmark_id]
            
            # Calculate excess returns
            portfolio_excess = portfolio_returns - self.periodic_risk_free
            benchmark_excess = benchmark_returns - self.periodic_risk_free
            
            # Calculate beta
            cov_matrix = group[['portfolio', benchmark_id]].cov()
            beta = cov_matrix.loc['portfolio', benchmark_id] / cov_matrix.loc[benchmark_id, benchmark_id]
            
            # Calculate alpha
            expected_return = self.periodic_risk_free + beta * (benchmark_returns.mean() - self.periodic_risk_free)
            alpha = portfolio_returns.mean() - expected_return
            
            # Calculate tracking error
            tracking_difference = portfolio_returns - benchmark_returns
            tracking_error = tracking_difference.std()
            
            # Calculate information ratio
            information_ratio = tracking_difference.mean() / tracking_error if tracking_error > 0 else 0
            
            # Calculate correlation
            correlation = portfolio_returns.corr(benchmark_returns)
            
            # Store metrics for this regime
            regime_metrics[regime] = {
                'observations': len(group),
                'portfolio_return': portfolio_returns.mean(),
                'benchmark_return': benchmark_returns.mean(),
                'alpha': alpha,
                'beta': beta,
                'correlation': correlation,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
        
        return {'regime_comparison': regime_metrics} 