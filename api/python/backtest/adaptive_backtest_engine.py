"""
Adaptive Strategy Backtest Engine

Core engine for backtesting adaptive trading strategies across different market regimes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import json

from trading_bot.backtest.market_data_generator import MarketDataGenerator, MarketRegimeType
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.analytics.performance_tracker import PerformanceTracker
from trading_bot.analytics.market_regime_detector import MarketRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)

class AdaptiveBacktestEngine:
    """
    Engine for backtesting the adaptive strategy controller.
    
    Simulates trading with the adaptive strategy controller using
    historical or synthetic market data across different market regimes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the backtest engine.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize market data generator
        self.market_data_generator = MarketDataGenerator()
        
        # Output directory for results
        self.output_dir = self.config.get('output_dir', './backtest_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.performance_metrics = {}
        self.allocation_history = {}
        self.position_size_history = {}
        self.trade_history = {}
        self.equity_curve = {}
        
        logger.info("Initialized AdaptiveBacktestEngine")
    
    def run_backtest(self, 
                   controller_config: Dict[str, Any],
                   market_data: Dict[str, pd.DataFrame],
                   strategies: Dict[str, Dict[str, Any]],
                   simulation_days: int = 252,
                   trade_frequency: int = 5,
                   initial_equity: float = 10000.0,
                   name: str = "backtest") -> Dict[str, Any]:
        """
        Run a single backtest with the given configuration.
        
        Args:
            controller_config: Configuration for the adaptive strategy controller
            market_data: Dictionary mapping symbols to market data DataFrames
            strategies: Dictionary mapping strategy IDs to strategy metadata
            simulation_days: Number of days to simulate
            trade_frequency: Average number of days between trades
            initial_equity: Initial equity amount
            name: Name for this backtest run
            
        Returns:
            Dictionary with backtest results
        """
        # Set up the controller with the given configuration
        controller_config['initial_equity'] = initial_equity
        controller = AdaptiveStrategyController(config=controller_config)
        
        # Register strategies
        for strategy_id, metadata in strategies.items():
            controller.register_strategy(strategy_id, metadata)
        
        # Initialize results tracking
        equity_curve = [initial_equity]
        allocation_history = {strategy_id: [] for strategy_id in strategies}
        position_size_history = {strategy_id: [] for strategy_id in strategies}
        trade_history = {strategy_id: [] for strategy_id in strategies}
        
        # Prepare market data
        symbols = list(market_data.keys())
        start_date = None
        end_date = None
        
        # Find common date range across all symbols
        for symbol, data in market_data.items():
            if start_date is None or data['date'].min() > start_date:
                start_date = data['date'].min()
            if end_date is None or data['date'].max() < end_date:
                end_date = data['date'].max()
        
        # Ensure all dataframes have the same date range
        for symbol, data in market_data.items():
            market_data[symbol] = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # Sort data by date
        for symbol, data in market_data.items():
            market_data[symbol] = data.sort_values('date')
        
        # Get dates for simulation
        simulation_dates = pd.date_range(start=start_date, periods=min(simulation_days, len(market_data[symbols[0]])), freq='B')
        
        # Main simulation loop
        logger.info(f"Starting backtest '{name}' with {len(simulation_dates)} trading days")
        
        for day_idx, current_date in enumerate(simulation_dates):
            # Update market data in controller
            for symbol, data in market_data.items():
                # Get data up to current date
                current_data = data[data['date'] <= current_date].copy()
                
                if not current_data.empty:
                    # Update controller with latest data
                    controller.update_market_data(symbol, current_data)
            
            # Get current market regimes
            regimes = controller.get_market_regimes()
            
            # Generate trades for this day based on trade frequency
            # On average, each strategy trades every 'trade_frequency' days
            for strategy_id in strategies:
                # Skip if strategy inactive
                if not controller.is_strategy_active(strategy_id):
                    continue
                
                # Determine if we should generate a trade today
                # Use deterministic approach based on day index and strategy ID
                # to ensure reproducibility
                strategy_seed = sum(ord(c) for c in strategy_id) 
                if (day_idx + strategy_seed) % trade_frequency == 0:
                    # Select a random symbol for this strategy
                    strategy_symbols = strategies[strategy_id]['symbols']
                    symbol = strategy_symbols[day_idx % len(strategy_symbols)]
                    
                    # Get current data for this symbol
                    symbol_data = market_data[symbol]
                    current_price = symbol_data[symbol_data['date'] <= current_date]['close'].iloc[-1]
                    
                    # Generate a trade
                    # In a real backtest we'd have actual entry/exit logic
                    # Here we generate a simple trade with random outcome
                    
                    # Get position size
                    stop_loss = current_price * 0.98  # Simple 2% stop loss
                    position_info = controller.get_position_size(
                        strategy_id=strategy_id,
                        symbol=symbol,
                        entry_price=current_price,
                        stop_loss=stop_loss
                    )
                    
                    # Record position size
                    position_size_history[strategy_id].append({
                        'date': current_date,
                        'symbol': symbol,
                        'size': position_info['size'],
                        'notional': position_info['notional']
                    })
                    
                    # Simulate trade outcome
                    # Use regime to influence win rate - better results in favorable regimes
                    regime_data = controller.market_regime_detector.get_current_regime(symbol)
                    strategy_type = strategies[strategy_id]['category']
                    
                    base_win_rate = 0.5  # 50% base win rate
                    
                    # Adjust win rate based on regime suitability
                    if regime_data:
                        suitability = controller.market_regime_detector.get_strategy_suitability(symbol)
                        if strategy_type in suitability:
                            # Higher suitability = higher win rate 
                            # Scale from 0.3 (unsuitable) to 0.7 (highly suitable)
                            base_win_rate = 0.3 + 0.4 * suitability[strategy_type]
                    
                    # Determine if trade is a winner
                    is_winner = np.random.random() < base_win_rate
                    
                    # Calculate profit/loss
                    if is_winner:
                        pnl_pct = np.random.uniform(0.01, 0.03)
                    else:
                        pnl_pct = -np.random.uniform(0.01, 0.02)
                        
                    exit_price = current_price * (1 + pnl_pct)
                    quantity = position_info['size']
                    pnl_amount = (exit_price - current_price) * quantity
                    
                    # Create trade data
                    trade_data = {
                        'entry_time': current_date.isoformat(),
                        'exit_time': (current_date + timedelta(days=1)).isoformat(),
                        'symbol': symbol,
                        'direction': 'long',
                        'entry_price': current_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'pnl': pnl_amount,
                        'pnl_pct': pnl_pct,
                        'fees': quantity * current_price * 0.001,  # 0.1% trading fee
                        'slippage': 0.001 * current_price  # 0.1% slippage
                    }
                    
                    # Record trade
                    controller.record_trade_result(strategy_id, trade_data)
                    trade_history[strategy_id].append(trade_data)
            
            # Update equity curve
            equity = initial_equity + sum(
                sum(trade['pnl'] for trade in trades)
                for trades in trade_history.values()
            )
            controller.update_equity(equity)
            equity_curve.append(equity)
            
            # Record allocations
            allocations = controller.get_all_allocations()
            for strategy_id, allocation in allocations.items():
                allocation_history[strategy_id].append({
                    'date': current_date,
                    'allocation': allocation
                })
            
            # Log progress every 50 days
            if day_idx % 50 == 0 or day_idx == len(simulation_dates) - 1:
                logger.info(f"Backtest '{name}' - Day {day_idx+1}/{len(simulation_dates)} - Equity: ${equity:.2f}")
        
        # Calculate performance metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity / initial_equity) - 1
        
        # Convert equity curve to numpy array for calculations
        equity_array = np.array(equity_curve)
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calculate key metrics
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        max_drawdown = 0
        peak = equity_array[0]
        
        for value in equity_array:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Store results
        results = {
            'name': name,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'allocation_history': allocation_history,
            'position_size_history': position_size_history,
            'trade_history': trade_history,
            'simulation_days': len(simulation_dates),
            'controller_config': controller_config,
            'strategies': strategies
        }
        
        # Store in instance
        self.performance_metrics[name] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        self.equity_curve[name] = equity_curve
        self.allocation_history[name] = allocation_history
        self.trade_history[name] = trade_history
        
        # Save results to disk
        self._save_results(results, name)
        
        logger.info(f"Completed backtest '{name}' - Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], name: str) -> None:
        """Save backtest results to disk"""
        # Create directory for this backtest
        backtest_dir = os.path.join(self.output_dir, name)
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Save summary metrics
        summary = {
            'name': name,
            'initial_equity': results['initial_equity'],
            'final_equity': results['final_equity'],
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'simulation_days': results['simulation_days'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(backtest_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'day': range(len(results['equity_curve'])),
            'equity': results['equity_curve']
        })
        equity_df.to_csv(os.path.join(backtest_dir, 'equity_curve.csv'), index=False)
        
        # Save allocation history
        for strategy_id, allocations in results['allocation_history'].items():
            if allocations:
                alloc_df = pd.DataFrame(allocations)
                alloc_df.to_csv(os.path.join(backtest_dir, f'allocation_{strategy_id}.csv'), index=False)
        
        # Save trade history
        for strategy_id, trades in results['trade_history'].items():
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_csv(os.path.join(backtest_dir, f'trades_{strategy_id}.csv'), index=False)
        
        # Generate and save equity curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(results['equity_curve'])
        plt.title(f'Equity Curve - {name}')
        plt.xlabel('Trading Day')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig(os.path.join(backtest_dir, 'equity_curve.png'))
        plt.close()
        
        logger.info(f"Saved backtest results to {backtest_dir}")
