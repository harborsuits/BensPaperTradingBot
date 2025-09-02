"""
Benchmark analysis module for comparing strategy performance against market benchmarks.

This module allows backtested strategies to be compared against various market benchmarks
like S&P 500, Russell 2000, or sector ETFs to evaluate relative performance.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

class BenchmarkAnalyzer:
    """
    Analyzes strategy performance against market benchmarks.
    
    This class provides functionality to:
    1. Load benchmark data
    2. Align strategy and benchmark returns
    3. Calculate relative performance metrics
    4. Generate benchmark comparisons and visualizations
    """
    
    def __init__(
        self,
        strategy_returns: pd.Series = None,
        benchmark_data: Dict[str, pd.DataFrame] = None,
        default_benchmark: str = "SPY",
        risk_free_rate: float = 0.02,
        base_currency: str = "USD"
    ):
        """
        Initialize the benchmark analyzer.
        
        Args:
            strategy_returns: Optional pandas Series of strategy returns (indexed by date)
            benchmark_data: Optional dictionary of benchmark dataframes
            default_benchmark: Default benchmark symbol to use (e.g., "SPY")
            risk_free_rate: Annual risk-free rate (decimal)
            base_currency: Base currency for calculations
        """
        self.strategy_returns = strategy_returns
        self.benchmark_data = benchmark_data or {}
        self.default_benchmark = default_benchmark
        self.risk_free_rate = risk_free_rate
        self.base_currency = base_currency
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Performance metrics
        self.metrics = {}
        
    def load_benchmark_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_provider = None
    ) -> None:
        """
        Load benchmark data for the specified symbols and date range.
        
        Args:
            symbols: List of benchmark symbols to load
            start_date: Start date for benchmark data
            end_date: End date for benchmark data
            data_provider: Optional data provider to use for data loading
        """
        if data_provider is None:
            logger.warning("No data provider specified, benchmark data must be added manually")
            return
            
        for symbol in symbols:
            try:
                # Load data from the provider (adjust based on actual provider interface)
                benchmark_df = data_provider.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )
                
                # Calculate daily returns
                if 'close' in benchmark_df.columns:
                    benchmark_df['return'] = benchmark_df['close'].pct_change()
                    self.benchmark_data[symbol] = benchmark_df
                else:
                    logger.error(f"No close price data for benchmark {symbol}")
            except Exception as e:
                logger.error(f"Error loading benchmark data for {symbol}: {str(e)}")
    
    def set_strategy_returns(self, returns: pd.Series) -> None:
        """
        Set strategy returns.
        
        Args:
            returns: Pandas Series of strategy returns (indexed by date)
        """
        self.strategy_returns = returns
        
    def add_benchmark(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Add benchmark data manually.
        
        Args:
            symbol: Benchmark symbol
            data: DataFrame with benchmark data (must include 'close' column)
        """
        if 'close' not in data.columns:
            raise ValueError("Benchmark data must include 'close' column")
            
        # Calculate returns if needed
        if 'return' not in data.columns:
            data['return'] = data['close'].pct_change()
            
        self.benchmark_data[symbol] = data
    
    def analyze_relative_performance(
        self,
        benchmark_symbol: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics comparing strategy to benchmark.
        
        Args:
            benchmark_symbol: Benchmark symbol to compare against (uses default if None)
            
        Returns:
            Dictionary of relative performance metrics
        """
        if self.strategy_returns is None:
            raise ValueError("Strategy returns not set")
            
        symbol = benchmark_symbol or self.default_benchmark
        if symbol not in self.benchmark_data:
            raise ValueError(f"Benchmark {symbol} not found in available benchmarks")
            
        benchmark_df = self.benchmark_data[symbol]
        benchmark_returns = benchmark_df['return']
        
        # Align dates
        common_dates = self.strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between strategy and benchmark")
            
        strategy_returns = self.strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized
        
        # Beta calculation
        covariance = np.cov(strategy_returns.fillna(0), benchmark_returns.fillna(0))[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
        
        # Alpha calculation (CAPM)
        risk_free_daily = pd.Series(self.daily_risk_free_rate, index=common_dates)
        strategy_excess = strategy_returns - risk_free_daily
        benchmark_excess = benchmark_returns - risk_free_daily
        
        expected_return = risk_free_daily + beta * benchmark_excess
        alpha = (strategy_returns - expected_return).mean() * 252  # Annualized
        
        # Information ratio
        information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252) if tracking_error != 0 else 0
        
        # Up/Down capture
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        up_capture = 0
        if up_market.any():
            up_capture = (
                strategy_returns[up_market].mean() / 
                benchmark_returns[up_market].mean()
            ) if benchmark_returns[up_market].mean() != 0 else 0
            
        down_capture = 0
        if down_market.any():
            down_capture = (
                strategy_returns[down_market].mean() / 
                benchmark_returns[down_market].mean()
            ) if benchmark_returns[down_market].mean() != 0 else 0
        
        # Calculate correlation
        correlation = strategy_returns.corr(benchmark_returns)
        
        # R-squared
        r_squared = correlation ** 2
        
        # Calculate cumulative performance
        cum_strategy = (1 + strategy_returns).cumprod() - 1
        cum_benchmark = (1 + benchmark_returns).cumprod() - 1
        
        # Outperformance frequency
        outperformance_freq = (strategy_returns > benchmark_returns).mean()
        
        # Maximum drawdown comparison
        strategy_drawdown = self._calculate_max_drawdown(strategy_returns)
        benchmark_drawdown = self._calculate_max_drawdown(benchmark_returns)
        
        metrics = {
            "alpha": alpha,
            "beta": beta,
            "correlation": correlation,
            "r_squared": r_squared,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "up_capture": up_capture,
            "down_capture": down_capture,
            "outperformance_frequency": outperformance_freq,
            "strategy_max_drawdown": strategy_drawdown,
            "benchmark_max_drawdown": benchmark_drawdown,
            "excess_return": excess_returns.mean() * 252,  # Annualized
            "total_return_strategy": cum_strategy.iloc[-1] if len(cum_strategy) > 0 else 0,
            "total_return_benchmark": cum_benchmark.iloc[-1] if len(cum_benchmark) > 0 else 0
        }
        
        self.metrics[symbol] = metrics
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from a series of returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown as a positive decimal
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        return abs(drawdown.min())
    
    def plot_relative_performance(
        self,
        benchmark_symbol: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot relative performance of strategy vs benchmark.
        
        Args:
            benchmark_symbol: Benchmark symbol to compare against
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.strategy_returns is None:
            raise ValueError("Strategy returns not set")
            
        symbol = benchmark_symbol or self.default_benchmark
        if symbol not in self.benchmark_data:
            raise ValueError(f"Benchmark {symbol} not found in available benchmarks")
            
        # Calculate metrics if not already done
        if symbol not in self.metrics:
            self.analyze_relative_performance(symbol)
            
        benchmark_df = self.benchmark_data[symbol]
        benchmark_returns = benchmark_df['return']
        
        # Align dates
        common_dates = self.strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = self.strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate cumulative returns
        cum_strategy = (1 + strategy_returns).cumprod() - 1
        cum_benchmark = (1 + benchmark_returns).cumprod() - 1
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Main plot - cumulative returns
        plt.subplot(2, 2, 1)
        plt.plot(cum_strategy.index, cum_strategy * 100, label=f'Strategy')
        plt.plot(cum_benchmark.index, cum_benchmark * 100, label=f'{symbol}')
        plt.title('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True)
        
        # Rolling correlation
        plt.subplot(2, 2, 2)
        rolling_corr = strategy_returns.rolling(window=60).corr(benchmark_returns)
        plt.plot(rolling_corr.index, rolling_corr)
        plt.title('Rolling 60-Day Correlation')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        # Rolling alpha
        plt.subplot(2, 2, 3)
        # Calculate rolling alpha (using window=60)
        window = 60
        rolling_alpha = pd.Series(index=strategy_returns.index[window-1:])
        
        for i in range(window-1, len(strategy_returns)):
            window_strategy = strategy_returns.iloc[i-window+1:i+1]
            window_benchmark = benchmark_returns.iloc[i-window+1:i+1]
            
            # Calculate beta for window
            cov = np.cov(window_strategy, window_benchmark)[0, 1]
            beta = cov / window_benchmark.var() if window_benchmark.var() != 0 else 1.0
            
            # Calculate alpha
            window_rf = pd.Series(self.daily_risk_free_rate, index=window_strategy.index)
            expected_return = window_rf + beta * (window_benchmark - window_rf)
            alpha = (window_strategy - expected_return).mean() * 252
            
            rolling_alpha.iloc[i-(window-1)] = alpha
            
        plt.plot(rolling_alpha.index, rolling_alpha * 100)  # Convert to percentage
        plt.title('Rolling 60-Day Alpha (%)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        # Scatter plot of returns
        plt.subplot(2, 2, 4)
        plt.scatter(benchmark_returns * 100, strategy_returns * 100, alpha=0.5)
        plt.xlabel(f'{symbol} Daily Returns (%)')
        plt.ylabel('Strategy Daily Returns (%)')
        
        # Add regression line
        slope, intercept, _, _, _ = stats.linregress(
            benchmark_returns.fillna(0) * 100, 
            strategy_returns.fillna(0) * 100
        )
        x_range = np.linspace(benchmark_returns.min() * 100, benchmark_returns.max() * 100, 100)
        plt.plot(x_range, intercept + slope * x_range, 'r', alpha=0.7)
        plt.title(f'Returns Regression (β={slope:.2f})')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return plt.gcf()
    
    def plot_drawdown_analysis(
        self,
        benchmark_symbol: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot drawdown analysis of strategy vs benchmark.
        
        Args:
            benchmark_symbol: Benchmark symbol to compare against
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.strategy_returns is None:
            raise ValueError("Strategy returns not set")
            
        symbol = benchmark_symbol or self.default_benchmark
        if symbol not in self.benchmark_data:
            raise ValueError(f"Benchmark {symbol} not found in available benchmarks")
            
        benchmark_df = self.benchmark_data[symbol]
        benchmark_returns = benchmark_df['return']
        
        # Align dates
        common_dates = self.strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = self.strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate drawdowns
        strategy_cum = (1 + strategy_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        strategy_drawdown = 1 - strategy_cum / strategy_cum.cummax()
        benchmark_drawdown = 1 - benchmark_cum / benchmark_cum.cummax()
        
        # Create plot
        plt.figure(figsize=figsize)
        
        plt.subplot(2, 1, 1)
        plt.plot(strategy_drawdown.index, strategy_drawdown * 100, label='Strategy')
        plt.title('Strategy Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(benchmark_drawdown.index, benchmark_drawdown * 100, label=symbol)
        plt.title(f'{symbol} Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return plt.gcf()
    
    def generate_benchmark_report(
        self,
        benchmark_symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive benchmark comparison report.
        
        Args:
            benchmark_symbol: Benchmark symbol to compare against
            
        Returns:
            Dictionary with benchmark comparison metrics and analysis
        """
        symbol = benchmark_symbol or self.default_benchmark
        
        # Calculate metrics if not already done
        if symbol not in self.metrics:
            self.analyze_relative_performance(symbol)
            
        metrics = self.metrics[symbol]
        
        # Summarize results in a report
        report = {
            "benchmark_symbol": symbol,
            "comparison_metrics": metrics,
            "interpretation": {
                "alpha_interpretation": self._interpret_alpha(metrics["alpha"]),
                "beta_interpretation": self._interpret_beta(metrics["beta"]),
                "correlation_interpretation": self._interpret_correlation(metrics["correlation"]),
                "capture_interpretation": self._interpret_capture(metrics["up_capture"], metrics["down_capture"]),
                "drawdown_interpretation": self._interpret_drawdown(
                    metrics["strategy_max_drawdown"], 
                    metrics["benchmark_max_drawdown"]
                ),
                "overall_assessment": self._overall_assessment(metrics)
            }
        }
        
        return report
    
    def _interpret_alpha(self, alpha: float) -> str:
        """Generate interpretation for alpha value."""
        if alpha > 0.05:
            return "Strong positive alpha - strategy significantly outperforms the benchmark on a risk-adjusted basis"
        elif alpha > 0.02:
            return "Positive alpha - strategy moderately outperforms the benchmark on a risk-adjusted basis"
        elif alpha > -0.02:
            return "Neutral alpha - strategy performs similarly to the benchmark on a risk-adjusted basis"
        elif alpha > -0.05:
            return "Negative alpha - strategy slightly underperforms the benchmark on a risk-adjusted basis"
        else:
            return "Strong negative alpha - strategy significantly underperforms the benchmark on a risk-adjusted basis"
    
    def _interpret_beta(self, beta: float) -> str:
        """Generate interpretation for beta value."""
        if beta > 1.2:
            return "High beta - strategy is significantly more volatile than the benchmark"
        elif beta > 1.0:
            return "Above average beta - strategy is somewhat more volatile than the benchmark"
        elif beta > 0.8:
            return "Average beta - strategy has similar volatility to the benchmark"
        elif beta > 0.5:
            return "Below average beta - strategy is somewhat less volatile than the benchmark"
        else:
            return "Low beta - strategy is significantly less volatile than the benchmark"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Generate interpretation for correlation value."""
        if abs(correlation) > 0.8:
            return "Very high correlation - strategy moves very similarly to the benchmark"
        elif abs(correlation) > 0.6:
            return "High correlation - strategy moves similarly to the benchmark"
        elif abs(correlation) > 0.4:
            return "Moderate correlation - strategy has some similar movements to the benchmark"
        elif abs(correlation) > 0.2:
            return "Low correlation - strategy moves somewhat independently from the benchmark"
        else:
            return "Very low correlation - strategy moves independently from the benchmark"
    
    def _interpret_capture(self, up_capture: float, down_capture: float) -> str:
        """Generate interpretation for up/down capture ratios."""
        if up_capture > 1.0 and down_capture < 1.0:
            return "Ideal capture - strategy captures more upside and less downside than the benchmark"
        elif up_capture > 1.0 and down_capture > 1.0:
            return "High volatility capture - strategy amplifies both upside and downside movements"
        elif up_capture < 1.0 and down_capture < 1.0:
            return "Low volatility capture - strategy dampens both upside and downside movements"
        elif up_capture < 1.0 and down_capture > 1.0:
            return "Poor capture - strategy captures less upside and more downside than the benchmark"
        else:
            return "Neutral capture - strategy performs similarly to the benchmark in both up and down markets"
    
    def _interpret_drawdown(self, strategy_dd: float, benchmark_dd: float) -> str:
        """Generate interpretation for drawdown comparison."""
        ratio = strategy_dd / benchmark_dd if benchmark_dd > 0 else float('inf')
        
        if ratio < 0.7:
            return "Excellent drawdown protection - strategy experiences significantly smaller drawdowns"
        elif ratio < 0.9:
            return "Good drawdown protection - strategy experiences smaller drawdowns than the benchmark"
        elif ratio < 1.1:
            return "Similar drawdowns - strategy and benchmark have comparable maximum drawdowns"
        elif ratio < 1.3:
            return "Higher drawdowns - strategy experiences somewhat larger drawdowns than the benchmark"
        else:
            return "Much higher drawdowns - strategy experiences significantly larger drawdowns than the benchmark"
    
    def _overall_assessment(self, metrics: Dict[str, float]) -> str:
        """Generate overall assessment based on metrics."""
        # Simplistic scoring system
        score = 0
        
        # Alpha contribution (heavily weighted)
        if metrics["alpha"] > 0.05: score += 3
        elif metrics["alpha"] > 0.02: score += 2
        elif metrics["alpha"] > 0: score += 1
        elif metrics["alpha"] > -0.02: score += 0
        else: score -= 1
        
        # Information ratio contribution
        if metrics["information_ratio"] > 1.0: score += 2
        elif metrics["information_ratio"] > 0.5: score += 1
        elif metrics["information_ratio"] > 0: score += 0.5
        else: score -= 0.5
        
        # Capture ratio contribution
        if metrics["up_capture"] > 1.0 and metrics["down_capture"] < 1.0: score += 2
        elif metrics["up_capture"] > metrics["down_capture"]: score += 1
        elif metrics["up_capture"] < metrics["down_capture"]: score -= 1
        
        # Drawdown contribution
        dd_ratio = metrics["strategy_max_drawdown"] / metrics["benchmark_max_drawdown"] if metrics["benchmark_max_drawdown"] > 0 else 2.0
        if dd_ratio < 0.8: score += 2
        elif dd_ratio < 1.0: score += 1
        elif dd_ratio > 1.2: score -= 1
        
        # Generate assessment based on score
        if score >= 5:
            return "Outstanding performance - strategy strongly outperforms the benchmark with better risk characteristics"
        elif score >= 3:
            return "Good performance - strategy outperforms the benchmark with favorable risk-adjusted returns"
        elif score >= 1:
            return "Slightly favorable - strategy shows some advantages over the benchmark"
        elif score >= -1:
            return "Similar to benchmark - strategy performs similarly to the benchmark overall"
        else:
            return "Underperformance - strategy does not offer advantages over the benchmark" 