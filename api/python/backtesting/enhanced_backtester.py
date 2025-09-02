"""
Enhanced Backtester

This module provides a comprehensive backtesting framework integrating strategy rotation,
signal generation, trade execution, and performance analysis.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("enhanced_backtester")

# Import components
from trading_bot.ai_scoring.integrated_strategy_rotator import IntegratedStrategyRotator
from trading_bot.utils.enhanced_trade_executor import EnhancedTradeExecutor, calculate_position_sizes
from trading_bot.utils.market_context_fetcher import MarketContextFetcher

# Import strategies
from trading_bot.strategies.trend_following import TrendFollowingStrategy
from trading_bot.strategies.momentum import MomentumStrategy
from trading_bot.strategies.mean_reversion import MeanReversionStrategy
from trading_bot.strategies.breakout_swing import BreakoutSwingStrategy
from trading_bot.strategies.volatility_breakout import VolatilityBreakoutStrategy
from trading_bot.strategies.option_spreads import OptionSpreadsStrategy

class EnhancedBacktester:
    """Enhanced backtester with proper signal generation and trade execution"""
    
    def __init__(self, config):
        """
        Initialize the enhanced backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')
        self.strategies = config.get('strategies', [])
        
        logger.info(f"Initializing enhanced backtester with {len(self.strategies)} strategies")
        
        # Initialize strategy instances
        self.strategy_instances = self._initialize_strategies()
        
        # Initialize trade executor
        self.trade_executor = EnhancedTradeExecutor(
            initial_capital=self.initial_capital,
            commission_rate=config.get('commission_rate', 0.001),
            slippage=config.get('slippage', 0.001)
        )
        
        # Initialize strategy rotator
        self.strategy_rotator = IntegratedStrategyRotator(
            strategies=self.strategies,
            initial_allocations=config.get('initial_allocations', {}),
            use_mock=config.get('use_mock', True)
        )
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Results storage
        self.portfolio_values = []
        self.allocation_history = []
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info("Enhanced backtester initialized successfully")
    
    def run_backtest(self):
        """
        Run backtest with strategy rotation and proper trade execution.
        
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest")
        
        # Trading dates
        trading_dates = sorted(self.market_data.keys())
        if not trading_dates:
            logger.error("No market data available")
            return {"error": "No market data available"}
        
        # Current allocations
        current_allocations = self.strategy_rotator.get_current_allocations()
        
        # Track initial allocation
        self.allocation_history.append({
            'date': trading_dates[0],
            'allocations': current_allocations.copy()
        })
        
        # Current portfolio value
        current_portfolio_value = self.initial_capital
        
        # Track daily portfolio values
        self.portfolio_values.append({
            'date': trading_dates[0],
            'value': current_portfolio_value,
            'daily_return': 0.0
        })
        
        # Iterate through trading days
        for date_idx, date in enumerate(trading_dates):
            # Skip first day (already processed)
            if date_idx == 0:
                continue
            
            logger.info(f"Processing date: {date} ({date_idx}/{len(trading_dates)-1})")
            
            # Get market data for current date
            daily_data = self.market_data[date]
            
            # Generate market context (weekly to reduce computation)
            if date_idx % 5 == 0 or date_idx == 1:
                logger.info(f"Generating market context for date: {date}")
                market_context = self._generate_market_context(date, daily_data)
            
            # Check if we need to rebalance today
            should_rebalance = self._should_rebalance(date, date_idx)
            
            if should_rebalance:
                logger.info(f"Rebalancing on date: {date}")
                # Perform strategy rotation
                rotation_result = self.strategy_rotator.rotate_strategies(
                    market_context=market_context,
                    force_rotation=True
                )
                
                # Get new allocations
                new_allocations = self.strategy_rotator.get_current_allocations()
                
                # Track allocation change
                self.allocation_history.append({
                    'date': date,
                    'allocations': new_allocations.copy()
                })
                
                # Update current allocations
                current_allocations = new_allocations
            
            # Generate signals from all strategies
            all_signals = []
            for strategy_name, strategy in self.strategy_instances.items():
                signals = strategy.generate_signals(daily_data, market_context)
                all_signals.extend(signals)
            
            # Calculate position sizes based on current allocations
            current_portfolio_value = self.trade_executor.portfolio_values[-1]['value'] if self.trade_executor.portfolio_values else self.initial_capital
            position_sizes = calculate_position_sizes(all_signals, current_portfolio_value, current_allocations)
            
            # Execute signals with calculated position sizes
            executed_trades = self.trade_executor.execute_signals(all_signals, daily_data, current_portfolio_value)
            
            # Manage existing positions (stop losses, take profits)
            closed_trades = self.trade_executor.manage_positions(daily_data)
            
            # Update trade history
            self.trade_history.extend(executed_trades)
            self.trade_history.extend(closed_trades)
            
            # Get updated portfolio value
            updated_portfolio_value = self.trade_executor.portfolio_values[-1]['value']
            
            # Calculate daily return
            daily_return = (updated_portfolio_value / current_portfolio_value) - 1.0
            
            # Track portfolio value
            self.portfolio_values.append({
                'date': date,
                'value': updated_portfolio_value,
                'daily_return': daily_return
            })
            
            # Update current portfolio value
            current_portfolio_value = updated_portfolio_value
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        logger.info("Backtest completed successfully")
        
        return {
            'summary': self.performance_metrics,
            'portfolio_values': self.portfolio_values,
            'allocation_history': self.allocation_history,
            'trade_history': self.trade_history
        }
    
    def _initialize_strategies(self):
        """
        Initialize strategy instances based on configuration.
        
        Returns:
            Dictionary of strategy instances
        """
        strategy_instances = {}
        
        for strategy_name in self.strategies:
            if strategy_name == 'trend_following':
                strategy_instances[strategy_name] = TrendFollowingStrategy()
            elif strategy_name == 'momentum':
                strategy_instances[strategy_name] = MomentumStrategy()
            elif strategy_name == 'mean_reversion':
                strategy_instances[strategy_name] = MeanReversionStrategy()
            elif strategy_name == 'breakout_swing':
                strategy_instances[strategy_name] = BreakoutSwingStrategy()
            elif strategy_name == 'volatility_breakout':
                strategy_instances[strategy_name] = VolatilityBreakoutStrategy()
            elif strategy_name == 'option_spreads':
                strategy_instances[strategy_name] = OptionSpreadsStrategy()
        
        logger.info(f"Initialized {len(strategy_instances)} strategy instances")
        return strategy_instances
    
    def _load_market_data(self):
        """
        Load market data for backtest period.
        
        Returns:
            Dictionary of market data by date
        """
        # This would connect to your data loading logic
        # Return a dictionary of {date: {symbol: {OHLCV data}}}
        # For now, return a placeholder structure
        
        logger.info("Loading market data (placeholder)")
        
        # This is just a placeholder. In a real implementation, you would load
        # actual market data from your data source (CSV, API, database, etc.)
        return {}
    
    def _generate_market_context(self, date, daily_data):
        """
        Generate market context for current date.
        
        Args:
            date: Current date
            daily_data: Market data for current date
            
        Returns:
            Market context
        """
        # In a real implementation, you would generate actual market context
        # based on market data, technical indicators, economic data, etc.
        
        # For now, return a placeholder
        context = {
            'market_regime': 'bullish',
            'trend_strength': 0.7,
            'volatility_index': 15.0
        }
        
        logger.info(f"Generated market context for {date}: {context['market_regime']}")
        return context
    
    def _should_rebalance(self, date, date_idx):
        """
        Determine if we should rebalance on this date.
        
        Args:
            date: Current date
            date_idx: Index of current date
            
        Returns:
            Boolean indicating whether to rebalance
        """
        # Get rebalance frequency from config
        rebalance_frequency = self.config.get('rebalance_frequency', 'monthly')
        
        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d') if isinstance(date, str) else date
        
        # Rebalance logic based on frequency
        if rebalance_frequency == 'daily':
            return True
        elif rebalance_frequency == 'weekly':
            return date_obj.weekday() == 0  # Monday
        elif rebalance_frequency == 'monthly':
            return date_obj.day == 1  # First day of month
        else:
            return date_idx % 20 == 0  # Every 20 trading days
    
    def _calculate_performance_metrics(self):
        """Calculate backtest performance metrics"""
        if not self.portfolio_values:
            logger.warning("No portfolio values to calculate performance metrics")
            return
        
        # Extract values and returns
        values = [entry['value'] for entry in self.portfolio_values]
        returns = [entry['daily_return'] for entry in self.portfolio_values if 'daily_return' in entry]
        
        # Calculate metrics
        initial_value = values[0]
        final_value = values[-1]
        total_return = (final_value / initial_value) - 1.0
        
        # Annualized return
        trading_days = len(values) - 1
        annual_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% risk-free rate
        daily_rfr = risk_free_rate / 252
        excess_returns = [r - daily_rfr for r in returns]
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if returns and np.std(excess_returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = 0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate
        winning_trades = [t for t in self.trade_history if 'pnl' in t and t['pnl'] > 0]
        total_closed_trades = [t for t in self.trade_history if 'pnl' in t]
        win_rate = len(winning_trades) / len(total_closed_trades) if total_closed_trades else 0
        
        # Store metrics
        self.performance_metrics = {
            'initial_capital': initial_value,
            'final_capital': final_value,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate * 100,
            'backtest_days': trading_days,
            'num_trades': len(total_closed_trades),
            'strategies': self.strategies,
            'rebalance_frequency': self.config.get('rebalance_frequency', 'monthly')
        }
        
        logger.info(f"Performance metrics calculated: {self.performance_metrics}")
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save plot
            
        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Create DataFrame for portfolio values
            portfolio_df = pd.DataFrame(self.portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Plot portfolio value over time
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(portfolio_df.index, portfolio_df['value'])
            plt.title('Portfolio Value Over Time')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            
            # Plot drawdown
            peak = portfolio_df['value'].cummax()
            drawdown = (peak - portfolio_df['value']) / peak
            
            plt.subplot(2, 1, 2)
            plt.plot(portfolio_df.index, drawdown * 100)
            plt.title('Portfolio Drawdown')
            plt.ylabel('Drawdown (%)')
            plt.xlabel('Date')
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.error("matplotlib and pandas required for plotting")
    
    def export_results_to_csv(self, portfolio_path=None, trades_path=None, allocations_path=None):
        """
        Export backtest results to CSV.
        
        Args:
            portfolio_path: Path to save portfolio values
            trades_path: Path to save trade history
            allocations_path: Path to save allocation history
            
        Returns:
            Tuple of paths where results were saved
        """
        try:
            import pandas as pd
            
            # Generate default paths if not provided
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if not portfolio_path:
                portfolio_path = f"backtest_portfolio_{timestamp}.csv"
            if not trades_path:
                trades_path = f"backtest_trades_{timestamp}.csv"
            if not allocations_path:
                allocations_path = f"backtest_allocations_{timestamp}.csv"
            
            # Export portfolio values
            portfolio_df = pd.DataFrame(self.portfolio_values)
            portfolio_df.to_csv(portfolio_path, index=False)
            logger.info(f"Portfolio values exported to {portfolio_path}")
            
            # Export trade history
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Trade history exported to {trades_path}")
            
            # Export allocation history
            allocations_df = pd.DataFrame([
                {'date': entry['date'], **entry['allocations']}
                for entry in self.allocation_history
            ])
            allocations_df.to_csv(allocations_path, index=False)
            logger.info(f"Allocation history exported to {allocations_path}")
            
            return portfolio_path, trades_path, allocations_path
            
        except ImportError:
            logger.error("pandas required for exporting to CSV")
            return None, None, None 