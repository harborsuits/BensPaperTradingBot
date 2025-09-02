"""
Performance Metrics

This module calculates trading performance metrics and generates performance reports.
"""

import logging
import sqlite3
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculates trading performance metrics and generates performance reports.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize the performance metrics calculator.
        
        Args:
            database_path: Path to SQLite database with trade records
        """
        self.database_path = database_path
        
        # Cache for performance data
        self.performance_cache = {}
        self.last_cache_update = datetime.now() - timedelta(hours=1)  # Force initial update
        self.cache_ttl = 300  # 5 minutes
    
    def get_strategy_performance(self, strategy_id: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: ID of the strategy to analyze
            period: Optional time period filter ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            Dict with performance metrics
        """
        try:
            # Check cache first if not day period (which changes frequently)
            cache_key = f"{strategy_id}_{period}"
            if period != 'day' and cache_key in self.performance_cache:
                cache_age = (datetime.now() - self.last_cache_update).total_seconds()
                if cache_age < self.cache_ttl:
                    return self.performance_cache[cache_key]
            
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Determine date range based on period
                date_filter = ""
                if period == 'day':
                    date_filter = "AND date(exit_time) = date('now')"
                elif period == 'week':
                    date_filter = "AND date(exit_time) >= date('now', '-7 days')"
                elif period == 'month':
                    date_filter = "AND date(exit_time) >= date('now', '-1 month')"
                elif period == 'year':
                    date_filter = "AND date(exit_time) >= date('now', '-1 year')"
                
                # Get all closed trades for this strategy in the period
                cursor.execute(f"""
                SELECT * FROM trades 
                WHERE strategy_id = ? AND status = 'closed' {date_filter}
                ORDER BY exit_time DESC
                """, (strategy_id,))
                
                trades = [dict(row) for row in cursor.fetchall()]
                
                if not trades:
                    # No trades for this strategy in this period
                    empty_metrics = {
                        'strategy_id': strategy_id,
                        'period': period or 'all',
                        'total_trades': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'total_pnl': 0,
                        'average_pnl': 0,
                        'average_win': 0,
                        'average_loss': 0,
                        'largest_win': 0,
                        'largest_loss': 0,
                        'expectancy': 0,
                        'avg_holding_time': 0
                    }
                    
                    # Cache the results
                    self.performance_cache[cache_key] = empty_metrics
                    self.last_cache_update = datetime.now()
                    
                    return empty_metrics
                
                # Calculate metrics
                pnls = [t.get('realized_pnl', 0) for t in trades]
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]
                
                win_count = len(wins)
                loss_count = len(losses)
                total_trades = len(trades)
                
                win_rate = win_count / total_trades if total_trades > 0 else 0
                total_pnl = sum(pnls)
                average_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                average_win = sum(wins) / win_count if win_count > 0 else 0
                average_loss = sum(losses) / loss_count if loss_count > 0 else 0
                
                profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
                
                largest_win = max(wins) if wins else 0
                largest_loss = min(losses) if losses else 0
                
                # Calculate expectancy
                expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss)) if total_trades > 0 else 0
                
                # Calculate average holding time
                holding_times = []
                for trade in trades:
                    if trade.get('entry_time') and trade.get('exit_time'):
                        try:
                            entry_time = datetime.fromisoformat(trade['entry_time'])
                            exit_time = datetime.fromisoformat(trade['exit_time'])
                            holding_time = (exit_time - entry_time).total_seconds() / 3600  # hours
                            holding_times.append(holding_time)
                        except (ValueError, TypeError):
                            pass
                
                avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
                
                # Compile metrics
                metrics = {
                    'strategy_id': strategy_id,
                    'period': period or 'all',
                    'total_trades': total_trades,
                    'win_count': win_count,
                    'loss_count': loss_count,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_pnl': total_pnl,
                    'average_pnl': average_pnl,
                    'average_win': average_win,
                    'average_loss': average_loss,
                    'largest_win': largest_win,
                    'largest_loss': largest_loss,
                    'expectancy': expectancy,
                    'avg_holding_time': avg_holding_time
                }
                
                # Cache the results
                self.performance_cache[cache_key] = metrics
                self.last_cache_update = datetime.now()
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {str(e)}")
            return {
                'strategy_id': strategy_id,
                'period': period or 'all',
                'total_trades': 0,
                'error': str(e)
            }
    
    def generate_performance_report(self, period: str = "all") -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            period: Report period ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            Dict with performance report data
        """
        try:
            # Determine date range for the period
            date_filter = ""
            if period == 'day':
                date_filter = "WHERE date(exit_time) = date('now')"
            elif period == 'week':
                date_filter = "WHERE date(exit_time) >= date('now', '-7 days')"
            elif period == 'month':
                date_filter = "WHERE date(exit_time) >= date('now', '-1 month')"
            elif period == 'year':
                date_filter = "WHERE date(exit_time) >= date('now', '-1 year')"
            
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get total realized P&L
                cursor.execute(f"""
                SELECT SUM(realized_pnl) as total_pnl
                FROM trades 
                WHERE status = 'closed' {date_filter.replace('WHERE', 'AND') if date_filter else ''}
                """)
                
                result = cursor.fetchone()
                total_realized_pnl = result['total_pnl'] if result and result['total_pnl'] is not None else 0
                
                # Get all closed trades in the period
                cursor.execute(f"""
                SELECT * FROM trades 
                WHERE status = 'closed' {date_filter.replace('WHERE', 'AND') if date_filter else ''}
                ORDER BY exit_time DESC
                """)
                
                trades = [dict(row) for row in cursor.fetchall()]
                
                # Get all strategy IDs
                cursor.execute(f"""
                SELECT DISTINCT strategy_id FROM trades 
                WHERE status = 'closed' {date_filter.replace('WHERE', 'AND') if date_filter else ''}
                """)
                
                strategy_ids = [row['strategy_id'] for row in cursor.fetchall() if row['strategy_id']]
                
                # Calculate overall metrics
                total_trades = len(trades)
                pnls = [t.get('realized_pnl', 0) for t in trades]
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]
                
                win_count = len(wins)
                loss_count = len(losses)
                
                win_rate = win_count / total_trades if total_trades > 0 else 0
                profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
                
                # Get strategy performance for each strategy
                strategy_performances = {}
                for strategy_id in strategy_ids:
                    strategy_performances[strategy_id] = self.get_strategy_performance(strategy_id, period)
                
                # Compile the report
                report = {
                    'period': period,
                    'total_realized_pnl': total_realized_pnl,
                    'total_trades': total_trades,
                    'win_count': win_count,
                    'loss_count': loss_count,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'strategies': strategy_performances,
                    'trade_summary': trades[:10]  # Include 10 most recent trades
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {
                'period': period,
                'error': str(e)
            }
    
    def generate_tax_report(self, year: int) -> Dict[str, Any]:
        """
        Generate tax report for specified year.
        
        Args:
            year: Tax year
            
        Returns:
            Dict with tax report data
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all closed trades for the year
                cursor.execute("""
                SELECT * FROM trades 
                WHERE status = 'closed' AND strftime('%Y', exit_time) = ?
                ORDER BY exit_time
                """, (str(year),))
                
                trades = [dict(row) for row in cursor.fetchall()]
                
                if not trades:
                    return {
                        'year': year,
                        'total_trades': 0,
                        'total_pnl': 0,
                        'asset_classes': {},
                        'trades': []
                    }
                
                # Categorize trades by asset class
                asset_classes = {}
                for trade in trades:
                    asset_class = trade.get('asset_class', 'unknown')
                    if asset_class not in asset_classes:
                        asset_classes[asset_class] = {
                            'trades': [],
                            'realized_pnl': 0,
                            'commissions': 0
                        }
                    
                    asset_classes[asset_class]['trades'].append(trade)
                    asset_classes[asset_class]['realized_pnl'] += trade.get('realized_pnl', 0)
                    asset_classes[asset_class]['commissions'] += trade.get('commission', 0)
                
                # Calculate totals
                total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
                total_commissions = sum(t.get('commission', 0) for t in trades)
                
                # Compile report
                report = {
                    'year': year,
                    'total_trades': len(trades),
                    'total_pnl': total_pnl,
                    'total_commissions': total_commissions,
                    'net_pnl': total_pnl - total_commissions,
                    'asset_classes': {},
                    'monthly_summary': {}
                }
                
                # Add asset class summaries
                for asset_class, data in asset_classes.items():
                    report['asset_classes'][asset_class] = {
                        'trade_count': len(data['trades']),
                        'realized_pnl': data['realized_pnl'],
                        'commissions': data['commissions'],
                        'net_pnl': data['realized_pnl'] - data['commissions']
                    }
                
                # Add monthly summary
                monthly_data = {}
                for trade in trades:
                    exit_time = trade.get('exit_time', '')
                    if exit_time:
                        try:
                            month = datetime.fromisoformat(exit_time).strftime('%Y-%m')
                            if month not in monthly_data:
                                monthly_data[month] = {
                                    'trade_count': 0,
                                    'realized_pnl': 0,
                                    'commissions': 0
                                }
                                
                            monthly_data[month]['trade_count'] += 1
                            monthly_data[month]['realized_pnl'] += trade.get('realized_pnl', 0)
                            monthly_data[month]['commissions'] += trade.get('commission', 0)
                        except (ValueError, TypeError):
                            pass
                
                # Format monthly data
                for month, data in monthly_data.items():
                    report['monthly_summary'][month] = {
                        'trade_count': data['trade_count'],
                        'realized_pnl': data['realized_pnl'],
                        'commissions': data['commissions'],
                        'net_pnl': data['realized_pnl'] - data['commissions']
                    }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating tax report: {str(e)}")
            return {
                'year': year,
                'error': str(e)
            }
    
    def calculate_drawdown(self, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate maximum drawdown over the specified period.
        
        Args:
            period: Optional time period filter ('month', 'year', 'all')
            
        Returns:
            Dict with drawdown information
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Determine date range based on period
                date_filter = ""
                if period == 'month':
                    date_filter = "WHERE date(exit_time) >= date('now', '-1 month')"
                elif period == 'year':
                    date_filter = "WHERE date(exit_time) >= date('now', '-1 year')"
                
                # Get all trades ordered by exit time
                cursor.execute(f"""
                SELECT exit_time, realized_pnl FROM trades 
                WHERE status = 'closed' {date_filter}
                ORDER BY exit_time
                """)
                
                trades = [dict(row) for row in cursor.fetchall()]
                
                if not trades:
                    return {
                        'period': period or 'all',
                        'max_drawdown': 0,
                        'max_drawdown_pct': 0,
                        'current_drawdown': 0,
                        'current_drawdown_pct': 0
                    }
                
                # Calculate cumulative P&L and find drawdowns
                cumulative_pnl = 0
                peak = 0
                max_drawdown = 0
                current_drawdown = 0
                max_drawdown_pct = 0
                
                daily_pnl = {}
                
                for trade in trades:
                    exit_date = trade.get('exit_time', '')[:10]  # Get date part
                    pnl = trade.get('realized_pnl', 0)
                    
                    if exit_date not in daily_pnl:
                        daily_pnl[exit_date] = 0
                    
                    daily_pnl[exit_date] += pnl
                
                # Sort by date and calculate drawdown
                sorted_dates = sorted(daily_pnl.keys())
                for date in sorted_dates:
                    cumulative_pnl += daily_pnl[date]
                    
                    if cumulative_pnl > peak:
                        peak = cumulative_pnl
                    
                    drawdown = peak - cumulative_pnl
                    drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
                    
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        max_drawdown_pct = drawdown_pct
                    
                    # Keep track of current drawdown
                    if date == sorted_dates[-1]:
                        current_drawdown = drawdown
                        current_drawdown_pct = drawdown_pct
                
                return {
                    'period': period or 'all',
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown_pct,
                    'current_drawdown': current_drawdown,
                    'current_drawdown_pct': current_drawdown_pct
                }
                
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return {
                'period': period or 'all',
                'error': str(e)
            }
    
    def calculate_sharpe_ratio(self, period: Optional[str] = None) -> float:
        """
        Calculate Sharpe ratio for the specified period.
        
        Args:
            period: Optional time period filter ('month', 'year', 'all')
            
        Returns:
            Sharpe ratio
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Determine date range based on period
                date_filter = ""
                if period == 'month':
                    date_filter = "WHERE date(exit_time) >= date('now', '-1 month')"
                elif period == 'year':
                    date_filter = "WHERE date(exit_time) >= date('now', '-1 year')"
                
                # Get daily P&L
                cursor.execute(f"""
                SELECT date(exit_time) as date, SUM(realized_pnl) as daily_pnl
                FROM trades
                WHERE status = 'closed' {date_filter}
                GROUP BY date(exit_time)
                ORDER BY date
                """)
                
                results = cursor.fetchall()
                
                if not results:
                    return 0.0
                
                # Calculate daily returns
                daily_returns = [row[1] for row in results]
                
                # Calculate Sharpe ratio
                avg_return = sum(daily_returns) / len(daily_returns)
                std_dev = np.std(daily_returns) if len(daily_returns) > 1 else 1.0
                
                # Annualize (assuming daily returns)
                risk_free_rate = 0.02  # 2% annual risk-free rate
                daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
                
                sharpe_ratio = ((avg_return - daily_risk_free) / std_dev) * np.sqrt(252) if std_dev > 0 else 0
                
                return float(sharpe_ratio)
                
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
