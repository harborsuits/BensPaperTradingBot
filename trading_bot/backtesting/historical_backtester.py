"""
Historical Backtesting Engine

This module provides a comprehensive backtesting engine that can replay
trading strategies across historical market data to evaluate performance.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Import internal components
from trading_bot.ai_scoring.regime_aware_strategy_prioritizer import RegimeAwareStrategyPrioritizer
from trading_bot.ai_scoring.trade_executor import TradeExecutor
from trading_bot.data.market_data_provider import create_data_provider

# Configure logging
logger = logging.getLogger(__name__)

class HistoricalBacktester:
    """
    Historical backtesting engine for trading strategies.
    
    This engine allows running strategies over historical data to evaluate performance.
    It integrates with the RegimeAwareStrategyPrioritizer and TradeExecutor components.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_provider_type: str = 'alpaca',
        data_provider_kwargs: Optional[Dict] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize the historical backtester.
        
        Args:
            config_path: Path to configuration file
            data_provider_type: Type of data provider ('alpaca', 'tradier', 'polygon')
            data_provider_kwargs: Additional parameters for data provider
            initial_capital: Initial capital for backtesting
        """
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize data provider
        data_provider_kwargs = data_provider_kwargs or {}
        self.data_provider = create_data_provider(
            data_provider_type,
            config_path=config_path,
            **data_provider_kwargs
        )
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.commission_rate = self.config.get('commission_rate', 0.001)  # Default 0.1%
        self.slippage = self.config.get('slippage', 0.001)  # Default 0.1%
        
        # Backtesting state
        self.results = None
        self.metrics = None
        self.signals_history = []
        self.trades_history = []
        self.portfolio_history = []
        
        logger.info(f"Historical backtester initialized with {data_provider_type} data provider")
    
    def run_backtest(
        self,
        symbols: List[str],
        strategies: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = 'day',
        rebalance_frequency: str = 'week',
        regime_lookback_days: int = 60,
        signal_generator: Optional[Callable] = None,
        use_regime_aware: bool = True,
        cache_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a historical backtest over the specified period.
        
        Args:
            symbols: List of trading symbols to include
            strategies: List of strategies to rotate between
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe ('day', 'hour', etc.)
            rebalance_frequency: How often to rebalance ('day', 'week', 'month')
            regime_lookback_days: Days to look back for regime classification
            signal_generator: Optional custom signal generator function
            use_regime_aware: Whether to use regime-aware strategy prioritization
            cache_dir: Directory to cache API responses
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date} with {len(symbols)} symbols and {len(strategies)} strategies")
        
        # Get historical data
        historical_data = self.data_provider.get_historical_data(
            symbols=symbols, 
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        if not historical_data:
            logger.error("Failed to retrieve historical data for backtest")
            return {"error": "Failed to retrieve historical data"}
        
        # Create DatetimeIndex for the backtest
        all_dates = set()
        for symbol, df in historical_data.items():
            all_dates.update(df['date'].dt.date)
        
        # Sort dates and convert to datetime
        backtest_dates = sorted(all_dates)
        backtest_dates = [datetime.combine(date, datetime.min.time()) for date in backtest_dates]
        
        # Generate rebalance dates based on frequency
        rebalance_dates = self._generate_rebalance_dates(backtest_dates, rebalance_frequency)
        
        # Initialize strategy prioritizer
        strategy_prioritizer = RegimeAwareStrategyPrioritizer(
            strategies=strategies,
            use_mock=False,
            regime_lookback_days=regime_lookback_days,
            cache_dir=cache_dir
        )
        
        # Initialize trade executor
        trade_executor = TradeExecutor(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage=self.slippage
        )
        
        # Initialize tracking variables
        current_allocations = {strategy: 0 for strategy in strategies}
        daily_returns = []
        strategy_allocations_history = []
        
        # Run backtest day by day
        for i, current_date in enumerate(backtest_dates):
            date_str = current_date.strftime('%Y-%m-%d')
            logger.debug(f"Processing date: {date_str} ({i+1}/{len(backtest_dates)})")
            
            # Get market data for current date
            market_data = self._get_market_data_for_date(historical_data, current_date)
            
            # Skip if no data for this date
            if not market_data:
                logger.debug(f"No market data available for {date_str}, skipping")
                continue
            
            # Check if rebalancing is needed
            is_rebalance_day = current_date in rebalance_dates
            
            if is_rebalance_day:
                logger.debug(f"Rebalance day: {date_str}")
                
                # Calculate market context
                market_context = self._calculate_market_context(market_data, historical_data, current_date, regime_lookback_days)
                
                # Get strategy allocations
                if use_regime_aware:
                    prioritization = strategy_prioritizer.prioritize_strategies(market_context=market_context)
                    new_allocations = prioritization["allocations"]
                else:
                    # Equal weighting if not using regime-aware
                    weight = 100.0 / len(strategies)
                    new_allocations = {strategy: weight for strategy in strategies}
                
                # Generate trading signals
                if signal_generator:
                    # Use custom signal generator if provided
                    signals = signal_generator(market_data, market_context, new_allocations)
                else:
                    # Otherwise use default signal generator
                    signals = self._generate_signals(market_data, market_context, new_allocations)
                
                # Track signals
                for signal in signals:
                    signal['date'] = date_str
                    self.signals_history.append(signal)
                
                # Execute allocation changes
                execution_results = trade_executor.execute_allocations(
                    new_allocations=new_allocations,
                    current_allocations=current_allocations,
                    market_data=market_data,
                    signals=signals,
                    date=date_str
                )
                
                # Track allocations
                strategy_allocations_history.append({
                    'date': date_str,
                    'allocations': new_allocations.copy()
                })
                
                # Update current allocations
                current_allocations = trade_executor.get_current_allocations(market_data)
            else:
                # On non-rebalance days, just manage existing positions
                _ = trade_executor.manage_positions(market_data, date=date_str)
            
            # Track portfolio value
            portfolio_value = trade_executor.get_portfolio_value(market_data)
            portfolio_record = {
                'date': date_str,
                'value': portfolio_value,
                'cash': trade_executor.cash,
                'positions': len(trade_executor.positions)
            }
            self.portfolio_history.append(portfolio_record)
            
            # Calculate daily return
            if i > 0:
                prev_value = self.portfolio_history[-2]['value']
                if prev_value > 0:
                    daily_return = (portfolio_value / prev_value) - 1.0
                    daily_returns.append(daily_return)
                    portfolio_record['daily_return'] = daily_return
        
        # Calculate performance metrics
        self.metrics = self._calculate_performance_metrics(daily_returns)
        
        # Save results
        self.results = {
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio_history[-1]['value'] if self.portfolio_history else self.initial_capital,
            'start_date': start_date,
            'end_date': end_date,
            'symbols': symbols,
            'strategies': strategies,
            'portfolio_history': self.portfolio_history,
            'metrics': self.metrics,
            'trades': trade_executor.transactions,
            'closed_positions': trade_executor.closed_positions,
            'signals_history': self.signals_history,
            'strategy_allocations': strategy_allocations_history
        }
        
        logger.info(f"Backtest completed: {self.metrics['total_return']*100:.2f}% total return, Sharpe: {self.metrics['sharpe_ratio']:.2f}")
        
        return self.results
    
    def _generate_rebalance_dates(self, dates: List[datetime], frequency: str) -> List[datetime]:
        """
        Generate rebalance dates based on the specified frequency.
        
        Args:
            dates: List of datetime objects
            frequency: Rebalance frequency ('day', 'week', 'month', etc.)
            
        Returns:
            List of datetime objects for rebalance dates
        """
        if frequency == 'day':
            return dates  # Rebalance every day
        
        rebalance_dates = []
        current_week = None
        current_month = None
        current_quarter = None
        
        for date in dates:
            if frequency == 'week':
                # Rebalance on first trading day of each week
                if date.isocalendar()[1] != current_week:
                    current_week = date.isocalendar()[1]
                    rebalance_dates.append(date)
            elif frequency == 'month':
                # Rebalance on first trading day of each month
                if date.month != current_month:
                    current_month = date.month
                    rebalance_dates.append(date)
            elif frequency == 'quarter':
                # Rebalance on first trading day of each quarter
                quarter = (date.month - 1) // 3 + 1
                if quarter != current_quarter:
                    current_quarter = quarter
                    rebalance_dates.append(date)
        
        return rebalance_dates
    
    def _get_market_data_for_date(self, historical_data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, Any]:
        """
        Extract market data for a specific date from historical data.
        
        Args:
            historical_data: Dictionary of historical data by symbol
            date: Target datetime
            
        Returns:
            Market data for the specified date
        """
        date_str = date.strftime('%Y-%m-%d')
        market_data = {}
        
        for symbol, df in historical_data.items():
            # Get data for the specific date
            day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
            
            if not day_data.empty:
                # Use last row if multiple entries exist
                row = day_data.iloc[-1]
                
                market_data[symbol] = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'price': row['close']  # Use close as current price
                }
        
        return market_data
    
    def _calculate_market_context(
        self, 
        market_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        lookback_days: int
    ) -> Dict[str, Any]:
        """
        Calculate market context data for regime classification.
        
        Args:
            market_data: Current market data
            historical_data: Full historical data
            current_date: Current datetime
            lookback_days: Number of days to look back
            
        Returns:
            Market context dictionary
        """
        # Start with basic market context
        market_context = {
            'date': current_date.strftime('%Y-%m-%d'),
            'symbols': list(market_data.keys())
        }
        
        # Try to infer market regime from index symbols if available
        indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']
        present_indices = [idx for idx in indices if idx in market_data]
        
        # Calculate sector performance if sector ETFs available
        sector_etfs = {
            'Technology': 'XLK',
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Healthcare': 'XLV',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE'
        }
        
        sector_performance = {}
        for sector, etf in sector_etfs.items():
            if etf in market_data:
                data = market_data[etf]
                daily_change = ((data['close'] / data['open']) - 1) * 100
                sector_performance[sector] = daily_change
        
        # Add to market context
        market_context['sector_performance'] = sector_performance
        
        # Try to infer market regime
        market_context['market_regime'] = self._infer_market_regime(market_data)
        
        # Calculate volatility if VIX present, otherwise estimate from SPY if available
        if 'VIX' in market_data:
            market_context['volatility_index'] = market_data['VIX']['close']
        elif 'SPY' in historical_data:
            # Calculate historical volatility from SPY
            end_date = current_date
            start_date = end_date - timedelta(days=lookback_days)
            
            spy_data = historical_data['SPY']
            spy_data['date'] = pd.to_datetime(spy_data['date'])
            mask = (spy_data['date'] >= start_date) & (spy_data['date'] <= end_date)
            recent_spy = spy_data.loc[mask]
            
            if len(recent_spy) > 5:
                # Calculate returns
                recent_spy['return'] = recent_spy['close'].pct_change()
                volatility = recent_spy['return'].std() * np.sqrt(252) * 100  # Annualized
                market_context['volatility_index'] = volatility
        
        return market_context
    
    def _infer_market_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Infer market regime based on market data.
        
        Args:
            market_data: Dictionary with market data
            
        Returns:
            Inferred market regime string
        """
        # Default regime
        regime = 'neutral'
        
        # Extract VIX and SPY data if available
        vix = market_data.get('VIX', {}).get('close', 15.0)
        spy = market_data.get('SPY', {})
        spy_close = spy.get('close', 0)
        spy_open = spy.get('open', spy_close)
        
        # Calculate SPY daily change
        spy_change = ((spy_close / spy_open) - 1) * 100 if spy_open > 0 else 0
        
        # Determine regime based on VIX and SPY performance
        if vix > 30:
            if spy_change < -1.0:
                regime = 'bearish'
            elif spy_change > 1.0:
                regime = 'volatile'
            else:
                regime = 'moderately_bearish'
        elif vix > 20:
            if spy_change < -0.5:
                regime = 'moderately_bearish'
            elif spy_change > 0.5:
                regime = 'moderately_bullish'
            else:
                regime = 'neutral'
        else:  # Low volatility
            if spy_change > 0.5:
                regime = 'bullish'
            elif spy_change < -0.5:
                regime = 'moderately_bearish'
            else:
                regime = 'sideways'
        
        return regime
    
    def _generate_signals(
        self, 
        market_data: Dict[str, Any], 
        market_context: Dict[str, Any],
        strategy_allocations: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data and strategy allocations.
        
        Args:
            market_data: Current market data
            market_context: Market context data
            strategy_allocations: Strategy allocation percentages
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get top strategies (allocation > 5%)
        top_strategies = {s: a for s, a in strategy_allocations.items() if a > 5}
        if not top_strategies:
            return signals
        
        # Get available symbols with price data
        available_symbols = [s for s in market_data.keys() if s not in ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']]
        if not available_symbols:
            return signals
        
        # Generate signals based on strategies
        for strategy, allocation in top_strategies.items():
            # Number of signals to generate based on allocation
            signal_count = min(3, max(1, int(allocation / 10)))
            
            # Randomly select symbols for this strategy's signals
            np.random.seed(int(allocation + hash(strategy) % 10000))
            selected_symbols = np.random.choice(
                available_symbols, 
                size=min(signal_count, len(available_symbols)), 
                replace=False
            )
            
            for symbol in selected_symbols:
                # Get current price
                current_price = market_data[symbol]['close']
                
                # Determine direction based on strategy and market regime
                regime = market_context.get('market_regime', 'neutral')
                
                if strategy in ['momentum', 'trend_following'] and regime in ['bullish', 'moderately_bullish']:
                    direction = 'long'
                elif strategy in ['mean_reversion'] and regime in ['sideways']:
                    direction = 'long' if np.random.random() > 0.5 else 'short'
                elif strategy in ['volatility_breakout']:
                    direction = 'long' if np.random.random() > 0.4 else 'short'
                else:
                    direction = 'long' if np.random.random() > 0.3 else 'short'
                
                # Calculate stop loss and take profit
                if direction == 'long':
                    stop_loss = current_price * (1 - np.random.uniform(0.03, 0.08))
                    take_profit = current_price * (1 + np.random.uniform(0.05, 0.15))
                else:  # short
                    stop_loss = current_price * (1 + np.random.uniform(0.03, 0.08))
                    take_profit = current_price * (1 - np.random.uniform(0.05, 0.15))
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'signal_type': strategy,
                    'direction': direction,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': np.random.uniform(0.6, 0.95),
                    'trailing_stop_distance': current_price * 0.05
                }
                
                signals.append(signal)
        
        return signals
    
    def _calculate_performance_metrics(self, daily_returns: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics from daily returns.
        
        Args:
            daily_returns: List of daily returns
            
        Returns:
            Dictionary of performance metrics
        """
        if not daily_returns:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Convert to numpy array
        returns = np.array(daily_returns)
        
        # Calculate metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        max_drawdown = drawdown.max()
        
        # Calculate win rate from closed positions
        if hasattr(self, 'results') and self.results:
            positions = self.results.get('closed_positions', [])
            if positions:
                winning_trades = [p for p in positions if p.get('pnl', 0) > 0]
                win_rate = len(winning_trades) / len(positions)
                
                # Profit factor
                profit = sum(p.get('pnl', 0) for p in winning_trades)
                loss = abs(sum(p.get('pnl', 0) for p in positions if p.get('pnl', 0) <= 0))
                profit_factor = profit / loss if loss > 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def save_results(self, output_dir: str = 'backtest_results') -> Dict[str, str]:
        """
        Save backtest results to CSV and JSON files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        if not self.results:
            logger.warning("No backtest results to save")
            return {}
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save portfolio history
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_file = output_path / f"portfolio_history_{timestamp}.csv"
        portfolio_df.to_csv(portfolio_file, index=False)
        
        # Save trades
        trades_df = pd.DataFrame(self.results['trades'])
        trades_file = output_path / f"trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        
        # Save closed positions
        positions_df = pd.DataFrame(self.results['closed_positions'])
        positions_file = output_path / f"positions_{timestamp}.csv"
        positions_df.to_csv(positions_file, index=False)
        
        # Save signals
        signals_df = pd.DataFrame(self.signals_history)
        signals_file = output_path / f"signals_{timestamp}.csv"
        signals_df.to_csv(signals_file, index=False)
        
        # Save strategy allocations
        allocations_df = pd.DataFrame([
            {'date': alloc['date'], **{k: v for k, v in alloc['allocations'].items()}}
            for alloc in self.results['strategy_allocations']
        ])
        allocations_file = output_path / f"strategy_allocations_{timestamp}.csv"
        allocations_df.to_csv(allocations_file, index=False)
        
        # Save summary metrics
        summary = {
            'backtest_info': {
                'start_date': self.results['start_date'],
                'end_date': self.results['end_date'],
                'initial_capital': self.results['initial_capital'],
                'final_value': self.results['final_value'],
                'symbols': self.results['symbols'],
                'strategies': self.results['strategies']
            },
            'metrics': self.metrics
        }
        
        summary_file = output_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Save full results
        results_file = output_path / f"full_results_{timestamp}.json"
        
        # Create a copy of results with manageable size for JSON serialization
        results_to_save = self.results.copy()
        
        # Ensure all values are JSON serializable
        for key in ['portfolio_history', 'trades', 'closed_positions', 'signals_history', 'strategy_allocations']:
            if key in results_to_save:
                # Limit the number of items to save (for large backtests)
                max_items = 10000
                if len(results_to_save[key]) > max_items:
                    results_to_save[key] = results_to_save[key][:max_items]
                
                # Convert DataFrame items to dict if needed
                results_to_save[key] = [
                    {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                     for k, v in item.items()}
                    for item in results_to_save[key]
                ]
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        logger.info(f"Backtest results saved to {output_dir}")
        
        return {
            'portfolio_file': str(portfolio_file),
            'trades_file': str(trades_file),
            'positions_file': str(positions_file),
            'signals_file': str(signals_file),
            'allocations_file': str(allocations_file),
            'summary_file': str(summary_file),
            'results_file': str(results_file)
        }
    
    def plot_performance(self, output_file: Optional[str] = None):
        """
        Generate performance plots for the backtest.
        
        Args:
            output_file: Optional file path to save the plot
            
        Returns:
            matplotlib.figure.Figure object
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            if not self.results:
                logger.warning("No backtest results to plot")
                return None
            
            # Create portfolio value DataFrame
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 2]})
            
            # Plot 1: Portfolio Value
            axes[0].plot(portfolio_df['date'], portfolio_df['value'], 'b-', linewidth=2)
            axes[0].set_title('Portfolio Value', fontsize=14)
            axes[0].set_ylabel('Value ($)', fontsize=12)
            axes[0].grid(True)
            axes[0].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Plot 2: Daily Returns
            if 'daily_return' in portfolio_df.columns:
                axes[1].plot(portfolio_df['date'], portfolio_df['daily_return'] * 100, 'g-', linewidth=1)
                axes[1].axhline(y=0, color='r', linestyle='-', linewidth=0.5)
                axes[1].set_title('Daily Returns', fontsize=14)
                axes[1].set_ylabel('Return (%)', fontsize=12)
                axes[1].grid(True)
                axes[1].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Plot 3: Strategy Allocations
            if 'strategy_allocations' in self.results:
                # Create DataFrame from strategy allocations
                allocations_data = []
                for alloc in self.results['strategy_allocations']:
                    allocations_data.append({
                        'date': pd.to_datetime(alloc['date']),
                        **alloc['allocations']
                    })
                
                alloc_df = pd.DataFrame(allocations_data)
                alloc_df.set_index('date', inplace=True)
                
                # Plot stacked area chart
                alloc_df.plot.area(ax=axes[2], stacked=True, colormap='viridis')
                axes[2].set_title('Strategy Allocations', fontsize=14)
                axes[2].set_ylabel('Allocation (%)', fontsize=12)
                axes[2].grid(True)
                axes[2].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                axes[2].legend(loc='upper left', fontsize=10)
            
            plt.tight_layout()
            
            # Save if output file provided
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plot saved to {output_file}")
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib required for plotting. Install with: pip install matplotlib")
            return None
        except Exception as e:
            logger.error(f"Error generating performance plot: {str(e)}")
            return None 