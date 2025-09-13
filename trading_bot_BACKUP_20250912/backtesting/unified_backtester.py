import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Import risk components
from trading_bot.risk import RiskManager, RiskMonitor
from trading_bot.strategy.strategy_rotator import StrategyRotator

# Import typed settings if available
try:
    from trading_bot.config.typed_settings import (
        load_config as typed_load_config,
        BacktestSettings,
        RiskSettings,
        TradingBotSettings
    )
    from trading_bot.config.migration_utils import get_config_from_legacy_path
    TYPED_SETTINGS_AVAILABLE = True
except ImportError:
    TYPED_SETTINGS_AVAILABLE = False
from trading_bot.backtesting.plotting import (
    plot_equity_curve, plot_drawdowns, plot_monthly_returns, 
    plot_rolling_metrics, plot_returns_distribution, 
    plot_correlation_matrix, plot_strategy_allocations,
    plot_regime_analysis, create_performance_dashboard
)

# Initialize logger
logger = logging.getLogger(__name__)

class UnifiedBacktester:
    """
    Unified backtesting system for strategy rotation.
    
    This class provides backtesting capabilities for the strategy rotation system,
    simulating trading over a historical period and calculating performance metrics.
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000.0,
        strategies: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        rebalance_frequency: str = "weekly",
        benchmark_symbol: str = "SPY",
        data_dir: str = "data",
        results_path: str = "data/backtest_results",
        use_mock: bool = False,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        trading_cost_pct: float = 0.1,  # 0.1% trading cost
        config_path: Optional[str] = None,
        settings: Optional[Union[BacktestSettings, TradingBotSettings]] = None,
        **kwargs
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for the backtest
            strategies: List of strategy names to include
            start_date: Start date for the backtest (YYYY-MM-DD)
            end_date: End date for the backtest (YYYY-MM-DD)
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
            benchmark_symbol: Symbol for benchmark comparison
            data_dir: Directory for data files
            results_path: Path to save backtest results
            use_mock: Whether to use mock data for strategies
            risk_free_rate: Annual risk-free rate (decimal form)
            trading_cost_pct: Trading cost as percentage
            **kwargs: Additional keyword arguments
        """
        # Load settings from typed settings system if available
        self.typed_settings = None
        self.backtest_settings = None
        self.risk_settings = None
        
        # Try to load settings from provided settings object or config path
        if TYPED_SETTINGS_AVAILABLE:
            if settings:
                # Extract settings based on what was provided
                if hasattr(settings, 'backtest') and hasattr(settings, 'risk'):
                    # Full TradingBotSettings provided
                    self.typed_settings = settings
                    self.backtest_settings = settings.backtest
                    self.risk_settings = settings.risk
                elif hasattr(settings, 'default_symbols'):
                    # Just BacktestSettings provided
                    self.backtest_settings = settings
                    
                    # Try to load risk settings from config path
                    if config_path:
                        try:
                            full_config = typed_load_config(config_path)
                            self.risk_settings = full_config.risk
                        except Exception as e:
                            logger.warning(f"Could not load risk settings from config: {e}")
            elif config_path:
                # Try to load from config path
                try:
                    full_config = typed_load_config(config_path)
                    self.typed_settings = full_config
                    self.backtest_settings = full_config.backtest
                    self.risk_settings = full_config.risk
                    logger.info(f"Loaded backtest settings from {config_path}")
                except Exception as e:
                    logger.warning(f"Could not load typed settings from {config_path}: {e}")
        
        # Apply settings if available, otherwise use constructor parameters
        if self.backtest_settings:
            self.initial_capital = self.backtest_settings.initial_capital
            self.trading_cost_pct = self.backtest_settings.commission_per_trade
            start_date = start_date or self.backtest_settings.default_start_date
            end_date = end_date or self.backtest_settings.default_end_date
            strategies = strategies or self.backtest_settings.default_symbols
            data_dir = data_dir or "data"
            
            # Apply other settings from typed backtest configuration
            self.slippage_pct = self.backtest_settings.slippage_pct
            self.data_source = self.backtest_settings.data_source
        else:
            # Use constructor parameters
            self.initial_capital = initial_capital
            self.trading_cost_pct = trading_cost_pct
            self.slippage_pct = kwargs.get("slippage_pct", 0.0005)
            self.data_source = kwargs.get("data_source", "local")
        
        # Set strategies list
        self.strategies = strategies or ["trend_following", "momentum", "mean_reversion"]
        
        # Set date range (default to last 1 year if not specified)
        today = datetime.now()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else today
        
        if start_date:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = today - timedelta(days=365)  # Default to 1 year
        
        self.benchmark_symbol = benchmark_symbol
        self.data_dir = Path(data_dir)
        self.results_path = results_path
        self.use_mock = use_mock
        self.risk_free_rate = risk_free_rate
        
        # Set rebalance frequency
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_days = self._get_rebalance_days(rebalance_frequency)
        
        # Initial allocations (equal weight by default)
        self.initial_allocations = kwargs.get("initial_allocations", {})
        if not self.initial_allocations:
            allocation = 100.0 / len(self.strategies) if self.strategies else 0
            self.initial_allocations = {strategy: allocation for strategy in self.strategies}
        
        # Initialize strategy rotator with risk management
        rotator_kwargs = {
            "initial_capital": self.initial_capital,
            "trading_cost_pct": self.trading_cost_pct,
            "risk_config": kwargs.get("risk_config", self._load_risk_config()),
            "enable_risk_management": kwargs.get("enable_risk_management", True),
            "enable_circuit_breakers": kwargs.get("enable_circuit_breakers", True),
            "enable_dynamic_sizing": kwargs.get("enable_dynamic_sizing", True)
        }
        
        self.strategy_rotator = StrategyRotator(
            strategies=self.strategies,
            initial_allocations=self.initial_allocations,
            **rotator_kwargs
        )
        
        # Initialize risk management components
        risk_config = kwargs.get("risk_config", self._load_risk_config())
        self.risk_manager = RiskManager(risk_config)
        self.risk_monitor = RiskMonitor(risk_config)
        
        # Risk management settings
        self.enable_risk_management = kwargs.get("enable_risk_management", True)
        
        # Initialize data containers
        self.strategy_data = {}
        self.market_data = pd.DataFrame()
        self.regime_data = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        
        # Performance tracking
        self.portfolio_history = []
        self.allocation_history = []
        self.debug_data = []
        self.trades = []  # List to track all trades
        self.total_costs = 0.0  # Track total trading costs
        
        # Trade execution parameters
        self.min_trade_value = kwargs.get("min_trade_value", 100.0)  # Minimum $ value for a trade
        
        # Debug mode
        self.debug_mode = kwargs.get("debug_mode", False)
        
        # Order execution settings
        self.order_settings = kwargs.get("order_settings", {
            "order_type": "market",               # Options: market, limit, vwap, close
            "slippage_model": "percentage",       # Options: percentage, fixed, volume_based, volatility_based
            "slippage_value": 0.1,                # Default 0.1% slippage
            "limit_offset_pct": 0.2,              # For limit orders: 0.2% away from market
            "enable_market_impact": False,        # Whether to model market impact
            "market_impact_constant": 0.1         # Constant for market impact calculation
        })
    
    def _load_risk_config(self) -> Dict[str, Any]:
        """Load risk management configuration."""
        if self.risk_settings:
            # Use typed risk settings if available
            return {
                "max_drawdown": self.risk_settings.portfolio_stop_loss_pct or 0.20,
                "max_allocation": self.risk_settings.max_position_pct or 0.25,
                "max_portfolio_risk": self.risk_settings.max_portfolio_risk or 0.20,
                "max_correlated_positions": self.risk_settings.max_correlated_positions or 3,
                "target_volatility": 0.15,  # Default target volatility
                "correlation_threshold": self.risk_settings.correlation_threshold or 0.7,
                "enable_portfolio_stop_loss": self.risk_settings.enable_portfolio_stop_loss, 
                "enable_position_stop_loss": self.risk_settings.enable_position_stop_loss,
            }
        else:
            # Default risk configuration
            return {
                "max_drawdown": 0.20,            # Maximum 20% drawdown
                "max_allocation": 0.25,         # Maximum 25% to any one strategy
                "target_volatility": 0.15,       # Target 15% annualized volatility
                "max_portfolio_risk": 0.20,      # Maximum 20% portfolio risk
                "correlation_threshold": 0.7,    # Correlation threshold
                "enable_portfolio_stop_loss": True,
                "enable_position_stop_loss": True,
            }
        
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Strategies: {', '.join(self.strategies)}")
        logger.info(f"Data source: {self.data_source}")
        logger.info(f"Trading cost: {self.trading_cost_pct:.2%}")
        logger.info(f"Slippage: {self.slippage_pct:.4%}")
        
        # Load required data
        self.load_strategy_data()
        self.load_market_data()
        self.load_regime_data()
        self.load_benchmark_data()
        
        # Reset performance tracking
        self.portfolio_history = []
        self.allocation_history = []
        self.trades = []
        self.total_costs = 0.0
        
        # Initialize portfolio with initial allocations
        initial_positions = {}
        for strategy, allocation in self.initial_allocations.items():
            position_value = (allocation / 100.0) * self.initial_capital
            initial_positions[strategy] = position_value
        
        # Initialize portfolio history
        self.portfolio_history.append({
            'date': self.start_date,
            'capital': self.initial_capital,
            'positions': initial_positions.copy(),
            'daily_return': 0.0
        })
        
        # Record initial allocations
        self._record_allocations(self.start_date, self.initial_allocations)
        
        # Generate simulation dates (business days only)
        simulation_dates = pd.date_range(
            start=self.start_date + timedelta(days=1),  # Start day after initialization
            end=self.end_date,
            freq='B'  # Business days
        )
        
        if self.debug_mode:
            logger.info(f"Running backtest with {len(simulation_dates)} trading days")
            logger.info(f"Initial portfolio: {initial_positions}")
        
        # Run simulation for each day
        for current_date in simulation_dates:
            if self.debug_mode and (current_date.day == 1 or current_date == simulation_dates[-1]):
                logger.info(f"Simulating {current_date.strftime('%Y-%m-%d')}")
            
            try:
                self.step(current_date)
            except Exception as e:
                logger.error(f"Error in simulation step for {current_date}: {str(e)}")
                if self.debug_mode:
                    import traceback
                    logger.error(traceback.format_exc())
        
        # Process results
        self._process_backtest_results()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        # Generate performance report
        performance_report = self.generate_performance_report(metrics)
        
        # Add trade info to results
        metrics['trade_count'] = len(self.trades)
        metrics['total_costs'] = self.total_costs
        metrics['trades'] = self.trades
        metrics['performance_report'] = performance_report
        
        # Log summary
        logger.info(f"Backtest completed. Final capital: ${metrics['final_capital']:,.2f}")
        logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Number of trades: {len(self.trades)}")
        logger.info(f"Trading costs: ${self.total_costs:.2f}")
        
        return metrics

    def plot_allocation_history(self, save_path: Optional[str] = None):
        """
        Plot strategy allocations over time.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Check if we have allocation history
        if not hasattr(self, 'allocation_history') or len(self.allocation_history) == 0:
            logger.warning("No allocation history available")
            return None, None
        
        # Convert allocation history to DataFrame if needed
        if isinstance(self.allocation_history, list):
            allocation_df = pd.DataFrame(self.allocation_history)
            allocation_df.set_index('date', inplace=True)
        else:
            allocation_df = self.allocation_history
        
        # Plot strategy allocations
        fig, ax = plot_strategy_allocations(
            allocations=allocation_df,
            title="Strategy Allocations Over Time",
            figsize=(12, 8),
            save_path=save_path
        )
        
        return fig, ax

    def plot_regime_performance(self, save_path: Optional[str] = None):
        """
        Plot performance by market regime.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Check if we have regime data
        if not hasattr(self, 'regime_history') or len(self.regime_history) == 0:
            logger.warning("No market regime data available")
            return None, None
        
        # Convert regime history to DataFrame if needed
        if isinstance(self.regime_history, list):
            regime_df = pd.DataFrame(self.regime_history)
            regime_df.set_index('date', inplace=True)
        else:
            regime_df = self.regime_history
        
        # Make sure we have portfolio data
        if not hasattr(self, 'portfolio_df'):
            self._calculate_performance_metrics()
        
        # Get daily returns Series
        returns = pd.Series(
            data=self.portfolio_df['daily_return'].values,
            index=self.portfolio_df.index,
            name='Returns'
        )
        
        # Map regime data to returns
        regime_returns = {}
        regime_column = 'regime' if 'regime' in regime_df.columns else next(iter(regime_df.columns))
        
        # Get all unique regimes
        all_regimes = regime_df[regime_column].unique()
        
        # Calculate returns by regime
        for regime in all_regimes:
            # Get dates for this regime
            regime_dates = regime_df[regime_df[regime_column] == regime].index
            
            # Get returns for these dates
            regime_return_series = returns.loc[returns.index.isin(regime_dates)]
            
            if not regime_return_series.empty:
                regime_returns[f"Regime {regime}"] = regime_return_series
        
        # Plot regime analysis
        fig, ax = plot_regime_analysis(
            regime_returns=regime_returns,
            title="Performance by Market Regime",
            figsize=(12, 10),
            save_path=save_path
        )
        
        return fig, ax

    def create_performance_dashboard(self, save_path: Optional[str] = None):
        """
        Create a comprehensive performance dashboard with multiple plots.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate metrics if not done already
        if not hasattr(self, 'portfolio_df') or not hasattr(self, 'performance_metrics'):
            self._calculate_performance_metrics()
            self.calculate_advanced_metrics()
        
        # Create dashboard
        fig = create_performance_dashboard(
            equity_curve=pd.Series(
                data=self.portfolio_df['capital'].values,
                index=self.portfolio_df.index,
                name='Portfolio'
            ),
            returns=pd.Series(
                data=self.portfolio_df['daily_return'].values,
                index=self.portfolio_df.index,
                name='Returns'
            ),
            drawdowns=self.calculate_drawdowns()['drawdown'],
            metrics=self.performance_metrics,
            advanced_metrics=self.advanced_metrics if hasattr(self, 'advanced_metrics') else None,
            trade_history=self.trade_history if hasattr(self, 'trade_history') else None,
            title=f"Trading Performance Dashboard: {self.strategy_name}",
            figsize=(16, 12),
            save_path=save_path
        )
        
        return fig

    def save_performance_report(self, output_dir: str = None):
        """
        Generate and save a complete performance report with visualizations.
        
        Args:
            output_dir: Directory to save the report (default: results/reports/<strategy_name>)
            
        Returns:
            Dictionary with paths to saved files
        """
        # Set default output directory if not provided
        if output_dir is None:
            strategy_name_safe = re.sub(r'[^\w\-_]', '_', self.strategy_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/reports/{strategy_name_safe}_{timestamp}"
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store saved file paths
        saved_files = {}
        
        # Calculate metrics if needed
        if not hasattr(self, 'performance_metrics'):
            self._calculate_performance_metrics()
            self.calculate_advanced_metrics()
        
        # Generate performance report
        report_path = self.generate_performance_report(os.path.join(output_dir, "performance_report.html"))
        saved_files['performance_report'] = report_path
        
        # Save portfolio performance plot
        fig, _ = self.plot_portfolio_performance(save_path=os.path.join(output_dir, "portfolio_performance.png"))
        if fig is not None:
            plt.close(fig)
            saved_files['portfolio_performance'] = os.path.join(output_dir, "portfolio_performance.png")
        
        # Save drawdowns plot
        fig, _ = self.plot_drawdowns(save_path=os.path.join(output_dir, "drawdowns.png"))
        if fig is not None:
            plt.close(fig)
            saved_files['drawdowns'] = os.path.join(output_dir, "drawdowns.png")
        
        # Save monthly returns plot
        fig, _ = self.plot_monthly_returns(save_path=os.path.join(output_dir, "monthly_returns.png"))
        if fig is not None:
            plt.close(fig)
            saved_files['monthly_returns'] = os.path.join(output_dir, "monthly_returns.png")
        
        # Save rolling metrics plot
        fig, _ = self.plot_rolling_risk_metrics(save_path=os.path.join(output_dir, "rolling_metrics.png"))
        if fig is not None:
            plt.close(fig)
            saved_files['rolling_metrics'] = os.path.join(output_dir, "rolling_metrics.png")
        
        # Save returns distribution plot
        fig, _ = self.plot_returns_distribution(save_path=os.path.join(output_dir, "returns_distribution.png"))
        if fig is not None:
            plt.close(fig)
            saved_files['returns_distribution'] = os.path.join(output_dir, "returns_distribution.png")
        
        # Save strategy correlations plot if applicable
        if hasattr(self, 'strategy_returns'):
            fig, _ = self.plot_strategy_correlations(save_path=os.path.join(output_dir, "strategy_correlations.png"))
            if fig is not None:
                plt.close(fig)
                saved_files['strategy_correlations'] = os.path.join(output_dir, "strategy_correlations.png")
        
        # Save allocation history plot if applicable
        if hasattr(self, 'allocation_history') and len(self.allocation_history) > 0:
            fig, _ = self.plot_allocation_history(save_path=os.path.join(output_dir, "allocation_history.png"))
            if fig is not None:
                plt.close(fig)
                saved_files['allocation_history'] = os.path.join(output_dir, "allocation_history.png")
        
        # Save regime performance plot if applicable
        if hasattr(self, 'regime_history') and len(self.regime_history) > 0:
            fig, _ = self.plot_regime_performance(save_path=os.path.join(output_dir, "regime_performance.png"))
            if fig is not None:
                plt.close(fig)
                saved_files['regime_performance'] = os.path.join(output_dir, "regime_performance.png")
        
        # Save performance dashboard
        fig = self.create_performance_dashboard(save_path=os.path.join(output_dir, "performance_dashboard.png"))
        if fig is not None:
            plt.close(fig)
            saved_files['performance_dashboard'] = os.path.join(output_dir, "performance_dashboard.png")
        
        # Save metrics as JSON
        metrics_combined = {
            **self.performance_metrics,
            **(self.advanced_metrics if hasattr(self, 'advanced_metrics') else {})
        }
        
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_combined, f, indent=4, default=str)
        
        saved_files['metrics_json'] = os.path.join(output_dir, "metrics.json")
        
        # Log success message
        logger.info(f"Performance report saved to {output_dir}")
        
        return saved_files

    def plot_portfolio_performance(self, benchmark: bool = True, save_path: Optional[str] = None):
        """
        Plot the portfolio equity curve with optional benchmark comparison.
        
        Args:
            benchmark: Whether to include benchmark in the plot
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Check if we have portfolio history
        if not hasattr(self, 'portfolio_history') or len(self.portfolio_history) == 0:
            logger.warning("No portfolio history available")
            return None, None
        
        # Convert portfolio history to DataFrame if not done already
        if not hasattr(self, 'portfolio_df'):
            self._calculate_performance_metrics()
        
        # Create equity curve Series
        equity_curve = pd.Series(
            data=self.portfolio_df['capital'].values,
            index=self.portfolio_df.index,
            name='Portfolio'
        )
        
        # Get benchmark data if requested and available
        benchmark_series = None
        if benchmark and not self.benchmark_data.empty:
            # Normalize benchmark to match portfolio initial value
            benchmark_values = self.benchmark_data['close'].values
            initial_ratio = equity_curve.iloc[0] / benchmark_values[0] if len(benchmark_values) > 0 else 1
            normalized_benchmark = benchmark_values * initial_ratio
            
            benchmark_series = pd.Series(
                data=normalized_benchmark,
                index=self.benchmark_data.index,
                name=self.benchmark_symbol
            )
        
        # Plot equity curve
        fig, ax = plot_equity_curve(
            equity_curve=equity_curve,
            benchmark=benchmark_series,
            drawdowns=self.calculate_drawdowns()['drawdown'],
            title="Portfolio Performance",
            figsize=(12, 8),
            save_path=save_path
        )
        
        return fig, ax

    def plot_drawdowns(self, top_n: int = 5, save_path: Optional[str] = None):
        """
        Plot portfolio drawdowns over time.
        
        Args:
            top_n: Number of largest drawdowns to highlight
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Calculate drawdowns if not done already
        drawdown_info = self.calculate_drawdowns()
        
        # Plot drawdown curve
        fig, ax = plot_drawdowns(
            drawdowns=drawdown_info['drawdown'],
            equity_curve=pd.Series(
                data=self.portfolio_df['capital'].values,
                index=self.portfolio_df.index
            ),
            drawdown_table=drawdown_info['drawdown_periods'].head(top_n),
            title="Portfolio Drawdowns",
            figsize=(12, 8),
            save_path=save_path
        )
        
        return fig, ax

    def plot_rolling_risk_metrics(self, window: int = 63, save_path: Optional[str] = None):
        """
        Plot rolling risk-adjusted metrics over time.
        
        Args:
            window: Rolling window size in days
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Check if we have portfolio history
        if not hasattr(self, 'portfolio_df'):
            self._calculate_performance_metrics()
        
        # Get daily returns
        returns = pd.Series(
            data=self.portfolio_df['daily_return'].values,
            index=self.portfolio_df.index,
            name='Returns'
        )
        
        # Calculate rolling metrics
        rolling_sharpe = (returns.rolling(window=window).mean() * 252) / (returns.rolling(window=window).std() * np.sqrt(252))
        
        # Calculate rolling Sortino
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = 0
        rolling_sortino = (returns.rolling(window=window).mean() * 252) / (negative_returns.rolling(window=window).std() * np.sqrt(252) + 1e-10)
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # annualized percentage
        
        # Plot rolling metrics
        fig, ax = plot_rolling_metrics(
            metrics={
                'Sharpe Ratio': rolling_sharpe,
                'Sortino Ratio': rolling_sortino,
                'Volatility (%)': rolling_vol
            },
            title=f"Rolling {window}-Day Risk Metrics",
            figsize=(12, 8),
            save_path=save_path
        )
        
        return fig, ax

    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """
        Plot a heatmap of monthly returns.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Check if we have portfolio history
        if not hasattr(self, 'portfolio_df'):
            self._calculate_performance_metrics()
        
        # Get daily returns
        returns = pd.Series(
            data=self.portfolio_df['daily_return'].values,
            index=self.portfolio_df.index,
            name='Returns'
        )
        
        # Calculate monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create monthly returns table
        monthly_table = monthly_returns.to_frame().pivot_table(
            values='Returns',
            index=monthly_returns.index.month,
            columns=monthly_returns.index.year,
            aggfunc='sum'
        )
        
        # Convert month numbers to names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_table.index = [month_names[m] for m in monthly_table.index]
        
        # Plot monthly returns heatmap
        fig, ax = plot_monthly_returns(
            monthly_returns=monthly_table,
            title="Monthly Returns (%)",
            figsize=(12, 8),
            save_path=save_path
        )
        
        return fig, ax

    def plot_strategy_correlations(self, save_path: Optional[str] = None):
        """
        Plot a correlation matrix of strategy returns.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Get strategy correlations
        corr_matrix = self.analyze_strategy_correlations()
        
        if corr_matrix is None or corr_matrix.empty:
            logger.warning("No strategy correlation data available")
            return None, None
        
        # Plot correlation matrix
        fig, ax = plot_correlation_matrix(
            correlation_matrix=corr_matrix,
            title="Strategy Return Correlations",
            figsize=(10, 8),
            save_path=save_path
        )
        
        return fig, ax

    def plot_returns_distribution(self, save_path: Optional[str] = None):
        """
        Plot the distribution of returns with overlays for VaR and normal distribution.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure and axes
        """
        # Check if we have portfolio history
        if not hasattr(self, 'portfolio_df'):
            self._calculate_performance_metrics()
        
        # Get daily returns
        returns = pd.Series(
            data=self.portfolio_df['daily_return'].values,
            index=self.portfolio_df.index,
            name='Returns'
        )
        
        # Calculate VaR and CVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Plot returns distribution
        fig, ax = plot_returns_distribution(
            returns=returns,
            var_threshold=var_95,
            cvar_value=cvar_95,
            confidence_level=95,
            title="Daily Returns Distribution",
            figsize=(12, 8),
            save_path=save_path
        )
        
        return fig, ax

# Example usage
if __name__ == "__main__":
    # Try to use typed settings if available
    if TYPED_SETTINGS_AVAILABLE:
        try:
            # Load configuration from the canonical config file
            settings = typed_load_config("/Users/bendickinson/Desktop/Trading/trading_bot/config/config.yaml")
            
            # Initialize backtester with typed settings
            backtester = UnifiedBacktester(settings=settings)
            logger.info("Using typed settings configuration")
        except Exception as e:
            logger.warning(f"Could not load typed settings: {e}")
            logger.warning("Falling back to default configuration")
            
            # Initialize backtester with default settings
            backtester = UnifiedBacktester(
                initial_capital=100000.0,
                start_date="2022-01-01",
                end_date="2022-12-31",
                rebalance_frequency="monthly",
                use_mock=True  # Use mock data for example
            )
    else:
        # Initialize backtester with default settings
        backtester = UnifiedBacktester(
            initial_capital=100000.0,
            start_date="2022-01-01",
            end_date="2022-12-31",
            rebalance_frequency="monthly",
            use_mock=True  # Use mock data for example
        )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Display results
    print("\nBacktest Performance Summary:")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['annual_return_pct']:.2f}%")
    print(f"Volatility: {results['volatility_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")
    
    # Plot results
    backtester.plot_portfolio_performance()
    backtester.plot_strategy_allocations()
    
    # Save results
    backtester.save_results() 