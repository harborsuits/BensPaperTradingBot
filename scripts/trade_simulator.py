#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Trade Simulator

This module implements an advanced trade simulator with support for
trailing stops, dynamic TP/SL ratios, and comprehensive performance metrics.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradeSimulator")

class TradeSimulator:
    """
    Advanced trade simulator with support for trailing stops,
    dynamic TP/SL ratios, and performance analysis.
    """
    
    def __init__(self, data, initial_balance=5000.0, name="Simulator"):
        """
        Initialize the trade simulator.
        
        Args:
            data: Dictionary of symbol -> DataFrame with market data
            initial_balance: Starting account balance
            name: Name of simulator for results tracking
        """
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.name = name
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        self.events = []
        
        # Performance tracking
        self.drawdowns = []
        self.peak_balance = initial_balance
        self.current_drawdown = 0
        
        # Flags for tracking state
        self.in_drawdown = False
        
    def reset(self):
        """Reset the simulator to initial state."""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        self.events = []
        self.drawdowns = []
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0
        self.in_drawdown = False
    
    def execute_trade(self, 
                     symbol: str, 
                     entry_time: datetime, 
                     direction: str, 
                     position_size: float, 
                     risk_amount: float, 
                     stop_loss_pips: float = 20,
                     take_profit_pips: float = 40,
                     max_bars: int = 100,
                     strategy_id: str = None,
                     strategy_name: str = None,
                     use_trailing_stop: bool = False,
                     trailing_activation: float = 0.5,
                     trailing_distance: float = 0.5,
                     tp_sl_ratio: float = 2.0,
                     entry_conditions: Dict = None,
                     market_regime: str = "unknown"):
        """
        Execute a simulated trade with enhanced exit strategies.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            direction: 'buy' or 'sell'
            position_size: Position size in lots
            risk_amount: Amount risked on the trade
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips (can be dynamically calculated)
            max_bars: Maximum bars to hold the trade
            strategy_id: Strategy ID
            strategy_name: Strategy name
            use_trailing_stop: Whether to use trailing stops
            trailing_activation: % of profit target to activate trailing stop
            trailing_distance: % of stop loss for trailing distance
            tp_sl_ratio: Take profit to stop loss ratio for dynamic calculation
            entry_conditions: Dictionary of entry conditions for pattern detection
            market_regime: Current market regime
            
        Returns:
            Trade result dictionary
        """
        if symbol not in self.data:
            logger.warning(f"Symbol {symbol} not found in data")
            return None
        
        # Find entry bar
        df = self.data[symbol]
        entry_idx = -1
        
        for i, time_idx in enumerate(df.index):
            if time_idx >= entry_time:
                entry_idx = i
                break
        
        if entry_idx == -1 or entry_idx >= len(df) - 1:
            logger.warning(f"Entry time {entry_time} not found in data")
            return None
        
        # Entry price
        entry_price = df.iloc[entry_idx]['Close']
        
        # Calculate pip value (simplified for major forex pairs)
        pip_value = 0.0001  # Standard pip for forex
        
        # For JPY pairs, adjust pip value
        if symbol.endswith('JPY'):
            pip_value = 0.01
        
        # Dynamically calculate take profit based on tp_sl_ratio if provided
        if tp_sl_ratio > 0:
            take_profit_pips = stop_loss_pips * tp_sl_ratio
        
        # Calculate stop loss and take profit levels
        if direction == 'buy':
            stop_loss = entry_price - (stop_loss_pips * pip_value)
            take_profit = entry_price + (take_profit_pips * pip_value)
        else:  # 'sell'
            stop_loss = entry_price + (stop_loss_pips * pip_value)
            take_profit = entry_price - (take_profit_pips * pip_value)
        
        # Calculate trailing stop variables
        trailing_stop = stop_loss
        trailing_activation_level = entry_price + (take_profit_pips * pip_value * trailing_activation) if direction == 'buy' else \
                                  entry_price - (take_profit_pips * pip_value * trailing_activation)
        trailing_stop_distance = stop_loss_pips * pip_value * trailing_distance
        
        highest_price = entry_price if direction == 'buy' else entry_price
        lowest_price = entry_price if direction == 'sell' else entry_price
        trailing_activated = False
        
        # Track trade through subsequent bars
        exit_idx = -1
        exit_price = entry_price
        exit_reason = 'max_bars'
        
        # Record trade start
        self.equity_curve.append({
            'time': df.index[entry_idx],
            'balance': self.balance,
            'symbol': symbol,
            'action': f'ENTRY {direction.upper()}',
            'price': entry_price
        })
        
        # Record trade event
        self.events.append({
            'time': df.index[entry_idx],
            'type': 'entry',
            'symbol': symbol,
            'direction': direction,
            'price': entry_price,
            'balance': self.balance,
            'strategy': strategy_name
        })
        
        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            bar = df.iloc[i]
            
            # Update highest/lowest prices for trailing stops
            if use_trailing_stop:
                if direction == 'buy':
                    # For buy trades, update highest price
                    if bar['High'] > highest_price:
                        highest_price = bar['High']
                        # Check for trailing stop activation
                        if not trailing_activated and highest_price >= trailing_activation_level:
                            trailing_activated = True
                            # Log activation
                            self.events.append({
                                'time': df.index[i],
                                'type': 'trailing_activated',
                                'symbol': symbol,
                                'price': highest_price,
                                'balance': self.balance
                            })
                        
                        # Move trailing stop if activated
                        if trailing_activated:
                            new_stop = highest_price - trailing_stop_distance
                            # Only move the stop up, never down
                            if new_stop > trailing_stop:
                                trailing_stop = new_stop
                else:  # 'sell'
                    # For sell trades, update lowest price
                    if bar['Low'] < lowest_price:
                        lowest_price = bar['Low']
                        # Check for trailing stop activation
                        if not trailing_activated and lowest_price <= trailing_activation_level:
                            trailing_activated = True
                            # Log activation
                            self.events.append({
                                'time': df.index[i],
                                'type': 'trailing_activated',
                                'symbol': symbol,
                                'price': lowest_price,
                                'balance': self.balance
                            })
                        
                        # Move trailing stop if activated
                        if trailing_activated:
                            new_stop = lowest_price + trailing_stop_distance
                            # Only move the stop down, never up
                            if new_stop < trailing_stop or not trailing_activated:
                                trailing_stop = new_stop
            
            # Check for stop loss (use trailing stop if activated)
            if direction == 'buy':
                stop_level = trailing_stop if trailing_activated else stop_loss
                if bar['Low'] <= stop_level:
                    exit_price = stop_level
                    exit_idx = i
                    exit_reason = 'trailing_stop' if trailing_activated else 'stop_loss'
                    break
            else:  # 'sell'
                stop_level = trailing_stop if trailing_activated else stop_loss
                if bar['High'] >= stop_level:
                    exit_price = stop_level
                    exit_idx = i
                    exit_reason = 'trailing_stop' if trailing_activated else 'stop_loss'
                    break
            
            # Check for take profit
            if direction == 'buy':
                if bar['High'] >= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
            else:  # 'sell'
                if bar['Low'] <= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
        
        # If we haven't exited yet, exit at the last processed bar
        if exit_idx == -1:
            exit_idx = min(entry_idx + max_bars, len(df) - 1)
            exit_price = df.iloc[exit_idx]['Close']
            exit_reason = 'max_bars'
        
        # Calculate profit/loss in pips
        if direction == 'buy':
            pips_gained = (exit_price - entry_price) / pip_value
        else:  # 'sell'
            pips_gained = (entry_price - exit_price) / pip_value
        
        # Calculate monetary PnL
        if exit_reason == 'stop_loss':
            pnl = -risk_amount
        elif exit_reason == 'trailing_stop':
            # Calculate based on actual exit price
            pnl = pips_gained * (risk_amount / stop_loss_pips)
        elif exit_reason == 'take_profit':
            pnl = risk_amount * tp_sl_ratio
        else:
            # Calculate based on actual pips gained/lost
            risk_per_pip = risk_amount / stop_loss_pips
            pnl = pips_gained * risk_per_pip
        
        # Update balance
        self.balance += pnl
        
        # Update peak balance and drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            # If we were in drawdown, record it and reset
            if self.in_drawdown:
                self.in_drawdown = False
                self.current_drawdown = 0
        else:
            # Calculate current drawdown
            dd_pct = (self.peak_balance - self.balance) / self.peak_balance * 100
            if dd_pct > self.current_drawdown:
                self.current_drawdown = dd_pct
                # Record in drawdown list if significant
                if self.current_drawdown > 5:  # Only track drawdowns over 5%
                    self.drawdowns.append({
                        'start_time': df.index[entry_idx],
                        'current_time': df.index[exit_idx],
                        'peak_balance': self.peak_balance,
                        'current_balance': self.balance,
                        'drawdown_pct': self.current_drawdown
                    })
                    self.in_drawdown = True
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': df.index[exit_idx],
            'exit_price': exit_price,
            'direction': direction,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': (risk_amount / (self.balance - pnl)) * 100,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'tp_sl_ratio': tp_sl_ratio,
            'use_trailing_stop': use_trailing_stop,
            'trailing_activated': trailing_activated,
            'pnl': pnl,
            'pips': pips_gained,
            'exit_reason': exit_reason,
            'bars_held': exit_idx - entry_idx,
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'balance_after': self.balance,
            'market_regime': market_regime,
            'entry_conditions': entry_conditions or {}
        }
        
        # Add to trade history
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append({
            'time': df.index[exit_idx],
            'balance': self.balance,
            'symbol': symbol,
            'action': f'EXIT {exit_reason.upper()}',
            'price': exit_price,
            'pnl': pnl
        })
        
        # Record trade event
        self.events.append({
            'time': df.index[exit_idx],
            'type': 'exit',
            'symbol': symbol,
            'reason': exit_reason,
            'price': exit_price,
            'pnl': pnl,
            'balance': self.balance,
            'bars_held': exit_idx - entry_idx
        })
        
        return trade
    
    def get_results(self):
        """
        Get comprehensive performance results.
        
        Returns:
            Dict with performance metrics and statistics
        """
        if not self.trades:
            return {
                'name': self.name,
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'total_trades': 0
            }
        
        # Calculate performance metrics
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        losses = sum(1 for t in self.trades if t['pnl'] <= 0)
        
        profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        loss = sum(abs(t['pnl']) for t in self.trades if t['pnl'] <= 0)
        
        win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0
        profit_factor = profit / loss if loss > 0 else float('inf')
        
        # Calculate max drawdown
        peak = self.initial_balance
        drawdown = 0
        max_drawdown = 0
        
        for trade in self.trades:
            if trade['balance_after'] > peak:
                peak = trade['balance_after']
            else:
                drawdown = (peak - trade['balance_after']) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate more advanced metrics
        expected_return = ((self.balance / self.initial_balance) ** (1/len(self.trades)) - 1) * 100 if len(self.trades) > 0 else 0
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Performance by market regime
        regime_performance = self._calculate_regime_performance()
        
        # Exit performance
        exit_performance = self._calculate_exit_performance()
        
        # Calculate trade statistics
        avg_win = profit / wins if wins > 0 else 0
        avg_loss = loss / losses if losses > 0 else 0
        avg_bars_held = sum(t['bars_held'] for t in self.trades) / len(self.trades) if len(self.trades) > 0 else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            'name': self.name,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': (self.balance / self.initial_balance - 1) * 100,
            'total_trades': len(self.trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars_held,
            'risk_reward_ratio': risk_reward,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'regime_performance': regime_performance,
            'exit_performance': exit_performance,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 3:
            return 0
        
        # Get balance at regular intervals
        balances = [point['balance'] for point in self.equity_curve if 'balance' in point]
        
        # Calculate returns
        returns = [balances[i] / balances[i-1] - 1 for i in range(1, len(balances))]
        
        # Calculate Sharpe ratio
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
            
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_regime_performance(self):
        """Calculate performance by market regime."""
        regimes = {}
        
        for trade in self.trades:
            regime = trade['market_regime']
            if regime not in regimes:
                regimes[regime] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0,
                    'loss': 0,
                    'win_rate': 0,
                    'profit_factor': 0
                }
            
            regimes[regime]['trades'] += 1
            if trade['pnl'] > 0:
                regimes[regime]['wins'] += 1
                regimes[regime]['profit'] += trade['pnl']
            else:
                regimes[regime]['losses'] += 1
                regimes[regime]['loss'] += abs(trade['pnl'])
        
        # Calculate metrics for each regime
        for regime in regimes:
            stats = regimes[regime]
            stats['win_rate'] = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            stats['profit_factor'] = stats['profit'] / stats['loss'] if stats['loss'] > 0 else float('inf')
        
        return regimes
    
    def _calculate_exit_performance(self):
        """Calculate performance by exit reason."""
        exits = {}
        
        for trade in self.trades:
            reason = trade['exit_reason']
            if reason not in exits:
                exits[reason] = {
                    'count': 0,
                    'profit': 0,
                    'avg_profit': 0,
                    'avg_bars': 0
                }
            
            exits[reason]['count'] += 1
            exits[reason]['profit'] += trade['pnl']
            exits[reason]['avg_bars'] += trade['bars_held']
        
        # Calculate averages
        for reason in exits:
            exits[reason]['avg_profit'] = exits[reason]['profit'] / exits[reason]['count'] if exits[reason]['count'] > 0 else 0
            exits[reason]['avg_bars'] = exits[reason]['avg_bars'] / exits[reason]['count'] if exits[reason]['count'] > 0 else 0
        
        return exits
    
    def plot_results(self, save_path=None):
        """
        Plot equity curve and performance metrics.
        
        Args:
            save_path: Path to save the plot, if None, display the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create equity curve data
            equity_data = pd.DataFrame(self.equity_curve)
            if len(equity_data) == 0:
                logger.warning("No equity data to plot")
                return
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            times = [point['time'] for point in self.equity_curve if 'balance' in point]
            balances = [point['balance'] for point in self.equity_curve if 'balance' in point]
            
            ax1.plot(times, balances, label=f'{self.name} Equity Curve')
            ax1.set_title(f'{self.name} Performance: {self.balance:.2f} ({(self.balance/self.initial_balance - 1)*100:.2f}%)')
            ax1.set_ylabel('Account Balance')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdowns
            drawdown_series = []
            peak = self.initial_balance
            for point in self.equity_curve:
                if 'balance' in point:
                    balance = point['balance']
                    if balance > peak:
                        peak = balance
                    
                    dd = (peak - balance) / peak * 100
                    drawdown_series.append((point['time'], dd))
            
            if drawdown_series:
                ax2.fill_between([t for t, d in drawdown_series], 0, [d for t, d in drawdown_series], 
                                 color='red', alpha=0.3)
                ax2.set_ylabel('Drawdown %')
                ax2.set_title('Drawdown')
                ax2.grid(True)
                ax2.invert_yaxis()  # Invert to show drawdowns as going down
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            
    def generate_report(self, save_path=None):
        """
        Generate comprehensive performance report.
        
        Args:
            save_path: Path to save the report as HTML, if None, return HTML string
        """
        results = self.get_results()
        
        html = f"""
        <html>
        <head>
            <title>{self.name} Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
                .metric {{ width: 30%; margin-bottom: 20px; background: #f5f5f5; padding: 10px; border-radius: 5px; }}
                .metric h3 {{ margin-top: 0; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                table, th, td {{ border: 1px solid #ddd; }}
                th, td {{ padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart {{ margin-top: 30px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{self.name} Trading Performance Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <div class="metric">
                        <h3>Account Performance</h3>
                        <p>Initial Balance: ${results['initial_balance']:.2f}</p>
                        <p>Final Balance: <span class="{('good' if results['final_balance'] >= results['initial_balance'] else 'bad')}">${results['final_balance']:.2f}</span></p>
                        <p>Total Return: <span class="{('good' if results['total_return'] >= 0 else 'bad')}">{results['total_return']:.2f}%</span></p>
                        <p>Max Drawdown: <span class="{('good' if results['max_drawdown'] < 20 else 'bad')}">{results['max_drawdown']:.2f}%</span></p>
                    </div>
                    
                    <div class="metric">
                        <h3>Trading Statistics</h3>
                        <p>Total Trades: {results['total_trades']}</p>
                        <p>Win Rate: <span class="{('good' if results['win_rate'] >= 50 else 'bad')}">{results['win_rate']:.2f}%</span></p>
                        <p>Profit Factor: <span class="{('good' if results['profit_factor'] >= 1.5 else 'bad')}">{results['profit_factor']:.2f}</span></p>
                        <p>Risk/Reward Ratio: <span class="{('good' if results['risk_reward_ratio'] >= 1 else 'bad')}">{results['risk_reward_ratio']:.2f}</span></p>
                    </div>
                    
                    <div class="metric">
                        <h3>Trade Metrics</h3>
                        <p>Average Win: ${results['avg_win']:.2f}</p>
                        <p>Average Loss: ${results['avg_loss']:.2f}</p>
                        <p>Average Holding Period: {results['avg_bars_held']:.1f} bars</p>
                        <p>Sharpe Ratio: {results['sharpe_ratio']:.2f}</p>
                    </div>
                </div>
                
                <h2>Performance by Market Regime</h2>
                <table>
                    <tr>
                        <th>Regime</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                        <th>Net Profit</th>
                    </tr>
        """
        
        # Add regime performance rows
        for regime, stats in results['regime_performance'].items():
            html += f"""
                    <tr>
                        <td>{regime}</td>
                        <td>{stats['trades']}</td>
                        <td>{stats['win_rate']:.2f}%</td>
                        <td>{stats['profit_factor']:.2f}</td>
                        <td>${stats['profit'] - stats['loss']:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <h2>Performance by Exit Type</h2>
                <table>
                    <tr>
                        <th>Exit Reason</th>
                        <th>Count</th>
                        <th>Average Profit</th>
                        <th>Average Bars Held</th>
                        <th>Total Profit</th>
                    </tr>
        """
        
        # Add exit performance rows
        for reason, stats in results['exit_performance'].items():
            html += f"""
                    <tr>
                        <td>{reason}</td>
                        <td>{stats['count']}</td>
                        <td>${stats['avg_profit']:.2f}</td>
                        <td>{stats['avg_bars']:.1f}</td>
                        <td>${stats['profit']:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <div class="chart">
                    <h2>Equity Curve</h2>
                    <p>See equity curve plot for visual representation of performance.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            logger.info(f"Report saved to {save_path}")
            return save_path
        else:
            return html
