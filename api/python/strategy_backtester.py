import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import datetime
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
import warnings
import pickle

from trading_bot.trading_strategy import TradingStrategy

@dataclass
class Trade:
    """Represents a single trade in the backtest"""
    symbol: str
    entry_date: datetime.datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    position_size: float
    exit_date: Optional[datetime.datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: str = "open"  # 'open' or 'closed'
    exit_reason: Optional[str] = None
    trade_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"{self.symbol}_{self.entry_date.strftime('%Y%m%d%H%M%S')}"
        
    def close(self, exit_date: datetime.datetime, exit_price: float, exit_reason: str = "signal"):
        """Close the trade with the given exit price and date"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = "closed"
        
        # Calculate P&L
        if self.direction == "long":
            self.pnl = (self.exit_price - self.entry_price) * self.position_size
            self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.position_size
            self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price * 100
        
        return self.pnl


class StrategyBacktester:
    """
    A backtesting engine for evaluating trading strategies with historical data.
    
    This class allows backesting of trading strategies on historical price data,
    calculates performance metrics, and generates performance visualizations.
    """
    
    def __init__(self, 
                 strategy_class: type,
                 strategy_params: Dict = None,
                 initial_capital: float = 100000.0,
                 commission: float = 0.0,
                 slippage: float = 0.0,
                 data_dir: str = "data",
                 output_dir: str = "backtest_results"):
        """
        Initialize the backtester with strategy and test parameters.
        
        Args:
            strategy_class: The TradingStrategy class to backtest
            strategy_params: Parameters to pass to the strategy
            initial_capital: Starting capital for the backtest
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (percentage)
            data_dir: Directory containing historical price data
            output_dir: Directory for saving backtest results
        """
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Setup directories
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize strategy (will be properly set up during backtest)
        self.strategy = None
        
        # Initialize containers
        self.historical_data = {}  # Symbol -> DataFrame
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.positions = {}  # Current positions: symbol -> Trade
        
        # Performance metrics
        self.metrics = {}
        
        # Set up logging
        self.logger = logging.getLogger("strategy_backtester")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = "daily") -> Dict[str, pd.DataFrame]:
        """
        Load historical price data for the given symbols and date range.
        
        Args:
            symbols: List of symbols to load data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe ('daily', 'hourly', etc.)
            
        Returns:
            Dictionary mapping symbols to their historical data DataFrames
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        self.historical_data = {}
        
        for symbol in symbols:
            # Construct file path based on symbol and timeframe
            file_path = self.data_dir / f"{symbol}_{timeframe}.csv"
            
            if not file_path.exists():
                self.logger.warning(f"Data file for {symbol} not found at {file_path}")
                continue
                
            # Load data
            df = pd.read_csv(file_path)
            
            # Ensure datetime column
            date_col = next((col for col in ['date', 'datetime', 'Date', 'Datetime'] if col in df.columns), None)
            if date_col:
                df['date'] = pd.to_datetime(df[date_col])
            else:
                self.logger.warning(f"No date column found in data for {symbol}")
                continue
                
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Rename columns to standardized format if needed
            col_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns}, inplace=True)
            
            # Set index to date
            df.set_index('date', inplace=True)
            
            # Store data
            self.historical_data[symbol] = df
            
            self.logger.info(f"Loaded {len(df)} rows of {timeframe} data for {symbol}")
            
        return self.historical_data
    
    def run_backtest(self) -> Dict:
        """
        Run the backtest using the loaded data and strategy.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.historical_data:
            self.logger.error("No historical data loaded. Call load_data() first.")
            return {}
            
        symbols = list(self.historical_data.keys())
        if not symbols:
            self.logger.error("No symbols in historical data.")
            return {}
            
        # Initialize strategy
        self.strategy = self.strategy_class(
            symbols=symbols,
            **self.strategy_params
        )
        
        # Initialize tracking variables
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.daily_returns = []
        self.current_capital = self.initial_capital
        
        # Determine common date range across all symbols
        common_dates = set.intersection(*[set(data.index) for data in self.historical_data.values()])
        common_dates = sorted(common_dates)
        
        if not common_dates:
            self.logger.error("No common dates found across all symbols.")
            return {}
            
        # Prepare equity curve with initial capital
        self.equity_curve = [(pd.to_datetime(common_dates[0]), self.initial_capital)]
        
        # Main backtest loop
        self.logger.info(f"Starting backtest with {len(common_dates)} trading days")
        for current_date in common_dates:
            # Get current day's data for all symbols
            current_data = {symbol: data.loc[[current_date]] for symbol, data in self.historical_data.items()}
            
            # Check for open positions and handle stops/take profits
            self._process_open_positions(current_date, current_data)
            
            # Generate signals for the current day
            signals = self._generate_signals_for_date(current_date, current_data)
            
            # Process signals
            self._process_signals(current_date, signals, current_data)
            
            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(current_date, current_data)
            self.equity_curve.append((current_date, portfolio_value))
            
            # Calculate daily return
            if len(self.equity_curve) >= 2:
                daily_return = (portfolio_value - self.equity_curve[-2][1]) / self.equity_curve[-2][1]
                self.daily_returns.append((current_date, daily_return))
        
        # Convert lists to DataFrames
        self._create_equity_and_return_series()
        
        # Calculate performance metrics
        self.metrics = self._calculate_performance_metrics()
        
        self.logger.info(f"Backtest completed with {len(self.trades)} trades")
        self.logger.info(f"Final portfolio value: ${self.equity_curve[-1][1]:.2f}")
        
        return self.metrics
    
    def _process_open_positions(self, current_date, current_data):
        """Process open positions for the current date"""
        for symbol, trade in list(self.positions.items()):
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close'].iloc[0]
            
            # Apply stop loss or take profit if configured
            # (This is a simplified example - actual implementation would depend on strategy)
            if hasattr(self.strategy, 'should_exit') and self.strategy.should_exit(
                symbol=symbol,
                current_price=current_price,
                trade=trade,
                current_date=current_date
            ):
                self._close_position(symbol, current_date, current_price, "exit_signal")
    
    def _generate_signals_for_date(self, current_date, current_data):
        """Generate trading signals for the current date"""
        # Create a dictionary of historical data up to the current date for each symbol
        historical_data_to_date = {}
        for symbol, full_data in self.historical_data.items():
            historical_data_to_date[symbol] = full_data.loc[:current_date]
        
        # Generate signals using the strategy
        try:
            signals = self.strategy.generate_signals(
                historical_data=historical_data_to_date,
                current_date=current_date
            )
            return signals
        except Exception as e:
            self.logger.error(f"Error generating signals for {current_date}: {str(e)}")
            return {}
    
    def _process_signals(self, current_date, signals, current_data):
        """Process trading signals for the current date"""
        for symbol, signal in signals.items():
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close'].iloc[0]
            
            # Apply slippage to the price
            if signal['action'] == 'buy':
                adjusted_price = current_price * (1 + self.slippage / 100)
            elif signal['action'] == 'sell':
                adjusted_price = current_price * (1 - self.slippage / 100)
            else:
                adjusted_price = current_price
            
            # Process the signal
            if signal['action'] == 'buy' and symbol not in self.positions:
                self._open_position(symbol, current_date, adjusted_price, 'long', signal)
            elif signal['action'] == 'sell' and symbol in self.positions:
                self._close_position(symbol, current_date, adjusted_price, 'sell_signal')
            elif signal['action'] == 'short' and symbol not in self.positions:
                self._open_position(symbol, current_date, adjusted_price, 'short', signal)
            elif signal['action'] == 'cover' and symbol in self.positions and self.positions[symbol].direction == 'short':
                self._close_position(symbol, current_date, adjusted_price, 'cover_signal')
    
    def _open_position(self, symbol, date, price, direction, signal):
        """Open a new position"""
        # Calculate position size based on available capital
        position_size = self._calculate_position_size(symbol, price, signal.get('risk_percent', 2.0))
        
        # Apply commission
        commission_cost = price * position_size * (self.commission / 100)
        self.current_capital -= commission_cost
        
        # Create trade object
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            direction=direction,
            position_size=position_size,
            metadata=signal.get('metadata', {})
        )
        
        # Add to positions
        self.positions[symbol] = trade
        
        # Update capital
        self.current_capital -= (price * position_size)
        
        self.logger.debug(f"Opened {direction} position in {symbol} at {price:.2f}, size: {position_size:.2f}")
    
    def _close_position(self, symbol, date, price, reason):
        """Close an existing position"""
        if symbol not in self.positions:
            return
            
        trade = self.positions[symbol]
        
        # Apply commission
        commission_cost = price * trade.position_size * (self.commission / 100)
        self.current_capital -= commission_cost
        
        # Close the trade
        pnl = trade.close(date, price, reason)
        
        # Add capital from closed position
        self.current_capital += (price * trade.position_size)
        
        # Add trade to history
        self.trades.append(trade)
        
        # Remove from active positions
        del self.positions[symbol]
        
        self.logger.debug(f"Closed {trade.direction} position in {symbol} at {price:.2f}, P&L: {pnl:.2f}")
    
    def _calculate_position_size(self, symbol, price, risk_percent):
        """Calculate position size based on risk percentage"""
        # Simple position sizing based on percentage of current capital
        risk_amount = self.current_capital * (risk_percent / 100)
        position_size = risk_amount / price
        return position_size
    
    def _calculate_portfolio_value(self, current_date, current_data):
        """Calculate the current portfolio value including cash and open positions"""
        portfolio_value = self.current_capital
        
        for symbol, trade in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol]['close'].iloc[0]
                position_value = current_price * trade.position_size
                
                if trade.direction == 'long':
                    portfolio_value += position_value
                else:  # short
                    # For shorts, we've already set aside the capital
                    unrealized_pnl = (trade.entry_price - current_price) * trade.position_size
                    portfolio_value += unrealized_pnl
        
        return portfolio_value
    
    def _create_equity_and_return_series(self):
        """Convert equity curve and daily returns to pandas Series"""
        equity_dates, equity_values = zip(*self.equity_curve)
        self.equity_series = pd.Series(equity_values, index=equity_dates)
        
        if self.daily_returns:
            return_dates, return_values = zip(*self.daily_returns)
            self.return_series = pd.Series(return_values, index=return_dates)
        else:
            self.return_series = pd.Series()
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results"""
        if not hasattr(self, 'equity_series') or len(self.equity_series) < 2:
            return {}
            
        metrics = {}
        
        # Basic metrics
        metrics['start_date'] = str(self.equity_series.index[0].date())
        metrics['end_date'] = str(self.equity_series.index[-1].date())
        metrics['total_days'] = len(self.equity_series)
        
        metrics['starting_capital'] = self.initial_capital
        metrics['ending_capital'] = self.equity_series.iloc[-1]
        metrics['total_return'] = (metrics['ending_capital'] / self.initial_capital - 1) * 100
        metrics['total_return_annualized'] = self._annualized_return(metrics['total_return'] / 100) * 100
        
        # Calculate daily, monthly, yearly returns
        if hasattr(self, 'return_series') and len(self.return_series) > 0:
            metrics['daily_returns_mean'] = self.return_series.mean() * 100
            metrics['daily_returns_std'] = self.return_series.std() * 100
            
            # Monthly returns
            monthly_returns = self.equity_series.resample('M').last().pct_change()
            metrics['monthly_returns_mean'] = monthly_returns.mean() * 100
            metrics['monthly_returns_std'] = monthly_returns.std() * 100
            
            # Yearly returns
            yearly_returns = self.equity_series.resample('Y').last().pct_change()
            metrics['yearly_returns_mean'] = yearly_returns.mean() * 100
            
            # Risk metrics
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(self.return_series)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(self.return_series)
            
            # Drawdown analysis
            max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(self.equity_series)
            metrics['max_drawdown'] = max_drawdown * 100
            metrics['max_drawdown_duration'] = max_drawdown_duration
        
        # Trade statistics
        if self.trades:
            win_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
            loss_trades = [t for t in self.trades if t.pnl and t.pnl <= 0]
            
            metrics['total_trades'] = len(self.trades)
            metrics['win_rate'] = len(win_trades) / len(self.trades) * 100 if self.trades else 0
            
            if win_trades:
                metrics['avg_win'] = sum(t.pnl for t in win_trades) / len(win_trades)
                metrics['avg_win_percent'] = sum(t.pnl_percent for t in win_trades) / len(win_trades)
            else:
                metrics['avg_win'] = 0
                metrics['avg_win_percent'] = 0
                
            if loss_trades:
                metrics['avg_loss'] = sum(t.pnl for t in loss_trades) / len(loss_trades)
                metrics['avg_loss_percent'] = sum(t.pnl_percent for t in loss_trades) / len(loss_trades)
            else:
                metrics['avg_loss'] = 0
                metrics['avg_loss_percent'] = 0
            
            if metrics['avg_loss'] != 0:
                metrics['profit_factor'] = abs(sum(t.pnl for t in win_trades) / sum(t.pnl for t in loss_trades)) if loss_trades else float('inf')
                metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
            else:
                metrics['profit_factor'] = float('inf')
                metrics['win_loss_ratio'] = float('inf')
            
            # Calculate average trade duration
            durations = []
            for trade in self.trades:
                if trade.exit_date and trade.entry_date:
                    duration = (trade.exit_date - trade.entry_date).total_seconds() / (24 * 60 * 60)  # in days
                    durations.append(duration)
            
            metrics['avg_trade_duration_days'] = sum(durations) / len(durations) if durations else 0
        
        return metrics
    
    def _annualized_return(self, total_return: float) -> float:
        """Calculate annualized return from total return"""
        if not hasattr(self, 'equity_series') or len(self.equity_series) < 2:
            return 0
            
        years = (self.equity_series.index[-1] - self.equity_series.index[0]).days / 365.25
        if years == 0:
            return 0
            
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration"""
        # Calculate the running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown in percentage terms
        drawdown = (equity_curve - running_max) / running_max
        
        # Find the maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find when the maximum drawdown occurred
        max_dd_idx = drawdown.idxmin()
        
        # Find the start of the drawdown period (previous peak)
        drawdown_start = equity_curve[:max_dd_idx].idxmax()
        
        # Find the end of the drawdown period (recovery to previous peak)
        try:
            # Find the first point after max_dd_idx where equity >= the value at drawdown_start
            post_dd = equity_curve[max_dd_idx:]
            recovery_mask = post_dd >= equity_curve[drawdown_start]
            
            if recovery_mask.any():
                drawdown_end = post_dd[recovery_mask].index[0]
            else:
                drawdown_end = equity_curve.index[-1]  # Never recovered
                
            # Calculate duration in days
            drawdown_duration = (drawdown_end - drawdown_start).days
        except Exception:
            # If there's any error in calculating duration, return 0
            drawdown_duration = 0
        
        return max_drawdown, drawdown_duration
    
    def plot_equity_curve(self, figsize=(12, 6), save_path=None):
        """Plot the equity curve"""
        if not hasattr(self, 'equity_series') or len(self.equity_series) < 2:
            self.logger.warning("Not enough data to plot equity curve")
            return
            
        plt.figure(figsize=figsize)
        plt.plot(self.equity_series, label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
    
    def plot_drawdowns(self, figsize=(12, 6), save_path=None):
        """Plot drawdowns over time"""
        if not hasattr(self, 'equity_series') or len(self.equity_series) < 2:
            self.logger.warning("Not enough data to plot drawdowns")
            return
            
        # Calculate the running maximum
        running_max = self.equity_series.cummax()
        
        # Calculate drawdown in percentage terms
        drawdown = (self.equity_series - running_max) / running_max * 100
        
        plt.figure(figsize=figsize)
        plt.plot(drawdown, color='red')
        plt.title('Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Drawdown chart saved to {save_path}")
        else:
            plt.show()
    
    def plot_monthly_returns(self, figsize=(12, 8), save_path=None):
        """Plot monthly returns heatmap"""
        if not hasattr(self, 'equity_series') or len(self.equity_series) < 30:  # Need at least a month
            self.logger.warning("Not enough data to plot monthly returns")
            return
            
        # Calculate monthly returns
        monthly_returns = self.equity_series.resample('M').last().pct_change() * 100
        
        # Create a pivot table for the heatmap: rows=years, columns=months
        monthly_returns.index = monthly_returns.index.to_period('M')
        returns_table = monthly_returns.to_frame('returns')
        returns_table['year'] = returns_table.index.year
        returns_table['month'] = returns_table.index.month
        
        pivot_table = returns_table.pivot('year', 'month', 'returns')
        
        # Plot the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                   linewidths=1, cbar_kws={"label": "Monthly Return (%)"})
        
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Format month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12) + 0.5, months)
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Monthly returns heatmap saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_analysis(self, figsize=(18, 10), save_path=None):
        """Plot trade analysis charts"""
        if not self.trades:
            self.logger.warning("No trades to analyze")
            return
            
        # Create a DataFrame of trades
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'direction': t.direction,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'duration': (t.exit_date - t.entry_date).total_seconds() / (24 * 60 * 60) if t.exit_date else 0
            }
            for t in self.trades if t.exit_date and t.pnl is not None
        ])
        
        if trades_df.empty:
            self.logger.warning("No closed trades to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. P&L Distribution
        axes[0, 0].hist(trades_df['pnl'], bins=20, color='green', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--')
        axes[0, 0].set_title('P&L Distribution')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Number of Trades')
        
        # 2. Cumulative P&L
        cumulative_pnl = trades_df['pnl'].cumsum()
        axes[0, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, marker='o', markersize=3)
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L ($)')
        axes[0, 1].grid(True)
        
        # 3. P&L by Symbol
        if len(trades_df['symbol'].unique()) > 1:
            symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values()
            symbol_pnl.plot(kind='barh', ax=axes[1, 0], color='skyblue')
            axes[1, 0].set_title('P&L by Symbol')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Symbol')
            axes[1, 0].grid(True, axis='x')
        else:
            axes[1, 0].text(0.5, 0.5, 'Only one symbol traded', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('P&L by Symbol')
        
        # 4. Trade Duration vs P&L
        axes[1, 1].scatter(trades_df['duration'], trades_df['pnl_percent'], 
                         alpha=0.7, c=trades_df['pnl'].apply(lambda x: 'green' if x > 0 else 'red'))
        axes[1, 1].set_title('Trade Duration vs P&L')
        axes[1, 1].set_xlabel('Duration (days)')
        axes[1, 1].set_ylabel('P&L (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Trade analysis saved to {save_path}")
        else:
            plt.show()
    
    def save_backtest_results(self, filename=None):
        """Save backtest results to a file"""
        if not hasattr(self, 'metrics') or not self.metrics:
            self.logger.warning("No metrics to save")
            return
            
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.__class__.__name__
            filename = f"{strategy_name}_backtest_{timestamp}.json"
        
        # Prepare results dictionary
        results = {
            'strategy': self.strategy.__class__.__name__,
            'strategy_params': self.strategy_params,
            'test_parameters': {
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'symbols': list(self.historical_data.keys()),
                'start_date': str(min(data.index[0] for data in self.historical_data.values())),
                'end_date': str(max(data.index[-1] for data in self.historical_data.values())),
            },
            'metrics': self.metrics,
            'trades_summary': {
                'total_trades': len(self.trades),
                'winning_trades': sum(1 for t in self.trades if t.pnl and t.pnl > 0),
                'losing_trades': sum(1 for t in self.trades if t.pnl and t.pnl <= 0),
            }
        }
        
        # Save as JSON
        file_path = self.output_dir / filename
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Backtest results saved to {file_path}")
        
        return file_path
    
    def save_trades_to_csv(self, filename=None):
        """Save trade details to a CSV file"""
        if not self.trades:
            self.logger.warning("No trades to save")
            return
            
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.__class__.__name__
            filename = f"{strategy_name}_trades_{timestamp}.csv"
        
        # Create DataFrame from trades
        trades_data = []
        for trade in self.trades:
            trade_dict = {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'entry_date': trade.entry_date,
                'entry_price': trade.entry_price,
                'exit_date': trade.exit_date,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
                'status': trade.status,
            }
            trades_data.append(trade_dict)
        
        trades_df = pd.DataFrame(trades_data)
        
        # Save to CSV
        file_path = self.output_dir / filename
        trades_df.to_csv(file_path, index=False)
        
        self.logger.info(f"Trade details saved to {file_path}")
        
        return file_path
    
    def compare_strategies(self, other_backtesters: List['StrategyBacktester'], metrics_to_compare=None):
        """
        Compare backtest results against other strategy backtests
        
        Args:
            other_backtesters: List of other StrategyBacktester instances
            metrics_to_compare: List of metric names to compare
            
        Returns:
            DataFrame with comparison results
        """
        if not hasattr(self, 'metrics') or not self.metrics:
            self.logger.warning("Current strategy has no metrics. Run backtest first.")
            return None
            
        all_backtesters = [self] + other_backtesters
        
        # Default metrics to compare if none provided
        if metrics_to_compare is None:
            metrics_to_compare = [
                'total_return', 'total_return_annualized', 'sharpe_ratio', 
                'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor'
            ]
        
        # Extract metrics from all backtesters
        comparison_data = {}
        for bt in all_backtesters:
            if not hasattr(bt, 'metrics') or not bt.metrics:
                self.logger.warning(f"Strategy {bt.strategy.__class__.__name__} has no metrics")
                continue
                
            strategy_name = bt.strategy.__class__.__name__
            comparison_data[strategy_name] = {
                metric: bt.metrics.get(metric, None) for metric in metrics_to_compare
            }
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df.T  # Transpose for better readability

# Example usage
if __name__ == "__main__":
    # This is a simplified example
    from trading_bot.trading_strategy import MovingAverageCrossover
    
    # Initialize backtester
    backtester = StrategyBacktester(
        strategy_class=MovingAverageCrossover,
        strategy_params={
            'fast_period': 10,
            'slow_period': 30
        },
        initial_capital=100000,
        commission=0.1,  # 0.1% commission per trade
        slippage=0.05    # 0.05% slippage per trade
    )
    
    # Load data
    backtester.load_data(
        symbols=['AAPL', 'MSFT', 'GOOG'],
        start_date='2020-01-01',
        end_date='2021-12-31',
        timeframe='daily'
    )
    
    # Run backtest
    metrics = backtester.run_backtest()
    
    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot results
    backtester.plot_equity_curve()
    backtester.plot_drawdowns()
    backtester.plot_monthly_returns()
    backtester.plot_trade_analysis()
    
    # Save results
    backtester.save_backtest_results()
    backtester.save_trades_to_csv() 