#!/usr/bin/env python3
"""
Forex Pip Logger

This module provides pip-based trade tracking and performance analysis for forex trading.
It replaces percentage-based metrics with pip calculations for more accurate forex
performance measurement.
"""

import pandas as pd
import numpy as np
import datetime
import logging
import os
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_pip_logger')


class ForexPipLogger:
    """
    Forex Pip Logger for tracking trade performance in pips.
    
    Core features:
    - Records trades with pip-based metrics
    - Tracks spread costs for entry/exit
    - Categorizes trades by session
    """
    
    def __init__(self, pair_manager=None):
        """
        Initialize the Forex Pip Logger.
        
        Args:
            pair_manager: Optional Forex Pair Manager instance for pip calculations
        """
        self.pair_manager = pair_manager
        self.trades = []
        self.session_info = {}
        
        # BenBot integration status
        self.benbot_enabled = False
        self.benbot_endpoint = None
        
        logger.info("Forex Pip Logger initialized")
    
    def enable_benbot(self, endpoint: str) -> None:
        """Enable BenBot integration for reporting."""
        self.benbot_enabled = True
        self.benbot_endpoint = endpoint
        logger.info(f"BenBot integration enabled at {endpoint}")
    
    def log_trade(self, 
                 pair: str,
                 entry_time: datetime.datetime,
                 exit_time: datetime.datetime,
                 direction: int,
                 entry_price: float,
                 exit_price: float,
                 lot_size: float,
                 stop_loss_pips: Optional[float] = None,
                 take_profit_pips: Optional[float] = None,
                 entry_spread_pips: Optional[float] = None,
                 exit_spread_pips: Optional[float] = None,
                 session: Optional[str] = None,
                 strategy_name: Optional[str] = None,
                 trade_id: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Log a forex trade with pip-based metrics.
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            entry_time: Trade entry datetime
            exit_time: Trade exit datetime
            direction: 1 for long, -1 for short
            entry_price: Entry price
            exit_price: Exit price
            lot_size: Position size in lots
            stop_loss_pips: Stop loss in pips (optional)
            take_profit_pips: Take profit in pips (optional)
            entry_spread_pips: Entry spread in pips (optional)
            exit_spread_pips: Exit spread in pips (optional)
            session: Trading session (e.g., 'London', 'NewYork', 'Asia')
            strategy_name: Name of the strategy (optional)
            trade_id: Unique trade identifier (optional)
            tags: List of tags for the trade (optional)
            
        Returns:
            Dictionary with trade details including pip metrics
        """
        # Generate trade ID if not provided
        if not trade_id:
            trade_id = f"{pair}_{entry_time.strftime('%Y%m%d%H%M%S')}"
            
        # Calculate pips based on direction and pair
        pip_multiplier = 10000  # Default for most pairs (4 decimal places)
        if 'JPY' in pair:
            pip_multiplier = 100  # JPY pairs have 2 decimal places
            
        # Use pair manager if available for more accurate pip calculations
        if self.pair_manager:
            pair_obj = self.pair_manager.get_pair(pair)
            if pair_obj:
                pip_multiplier = pair_obj.get_pip_multiplier()
        
        # Calculate pip profit/loss
        price_diff = exit_price - entry_price
        pip_diff = price_diff * pip_multiplier
        pips = pip_diff * direction  # Positive if profitable in given direction
        
        # Calculate spread cost in pips if provided
        spread_cost_pips = 0
        if entry_spread_pips is not None:
            spread_cost_pips += entry_spread_pips
        if exit_spread_pips is not None:
            spread_cost_pips += exit_spread_pips
            
        # Calculate net pips after spread
        net_pips = pips - spread_cost_pips
        
        # Calculate pip value based on lot size
        pip_value = 10.0  # Default $10 per pip for 1 standard lot
        if self.pair_manager:
            pair_obj = self.pair_manager.get_pair(pair)
            if pair_obj:
                pip_value = pair_obj.calculate_pip_value(lot_size)
        else:
            pip_value = 10.0 * lot_size
        
        # Calculate profit/loss in account currency
        profit_loss = net_pips * pip_value
        
        # Determine trade result
        result = "win" if net_pips > 0 else "loss" if net_pips < 0 else "breakeven"
        
        # Calculate risk-reward metrics if stop loss and take profit provided
        risk_reward_ratio = None
        if stop_loss_pips is not None and take_profit_pips is not None and stop_loss_pips > 0:
            risk_reward_ratio = take_profit_pips / stop_loss_pips
        
        # Detect session if not provided and session info available
        if not session and hasattr(self, 'session_manager'):
            active_sessions = self.session_manager.get_active_sessions(entry_time)
            if active_sessions:
                session = active_sessions[0]
                # If in overlap, use the highest priority session
                if len(active_sessions) > 1:
                    for s in ["London", "NewYork", "Tokyo", "Sydney"]:
                        if s in active_sessions:
                            session = s
                            break
        
        # Create trade record
        trade = {
            'trade_id': trade_id,
            'pair': pair,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'lot_size': lot_size,
            'pips': pips,
            'spread_cost_pips': spread_cost_pips,
            'net_pips': net_pips,
            'pip_value': pip_value,
            'profit_loss': profit_loss,
            'result': result,
            'duration_minutes': (exit_time - entry_time).total_seconds() / 60,
            'session': session,
            'strategy_name': strategy_name,
            'tags': tags or [],
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'risk_reward_ratio': risk_reward_ratio,
            'entry_spread_pips': entry_spread_pips,
            'exit_spread_pips': exit_spread_pips
        }
        
        # Add trade to log
        self.trades.append(trade)
        logger.info(f"Logged {result} trade: {pair} {direction} {net_pips:.1f} pips")
        
        # Report to BenBot if enabled
        if self.benbot_enabled:
            self._report_trade_to_benbot(trade)
            
        return trade
    
    def _report_trade_to_benbot(self, trade: Dict[str, Any]) -> bool:
        """
        Report trade to BenBot for integration.
        
        Args:
            trade: Trade details dictionary
            
        Returns:
            True if successfully reported, False otherwise
        """
        if not self.benbot_enabled or not self.benbot_endpoint:
            return False
            
        try:
            import requests
            
            # Simplified payload for BenBot
            payload = {
                'source': 'EvoTrader',
                'module': 'PipLogger',
                'trade_id': trade['trade_id'],
                'pair': trade['pair'],
                'entry_time': trade['entry_time'].isoformat(),
                'exit_time': trade['exit_time'].isoformat(),
                'direction': trade['direction'],
                'net_pips': trade['net_pips'],
                'spread_cost_pips': trade['spread_cost_pips'],
                'profit_loss': trade['profit_loss'],
                'session': trade['session'],
                'strategy_name': trade['strategy_name'],
                'risk_reward_ratio': trade['risk_reward_ratio'],
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{self.benbot_endpoint.rstrip('/')}/trade", 
                json=payload, 
                timeout=2
            )
            
            success = response.status_code == 200
            if not success:
                logger.warning(f"Failed to report trade to BenBot: {response.status_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error reporting to BenBot: {e}")
            return False
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Convert trades list to a pandas DataFrame.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'trade_id', 'pair', 'entry_time', 'exit_time', 'direction',
                'pips', 'net_pips', 'profit_loss', 'session', 'strategy_name'
            ])
            
        # Create DataFrame from trades list
        df = pd.DataFrame(self.trades)
        
        # Convert datetime columns
        for col in ['entry_time', 'exit_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        # Set trade_id as index
        if 'trade_id' in df.columns:
            df.set_index('trade_id', inplace=True)
            
        return df
    
    def get_trades_by_session(self, session: str) -> List[Dict[str, Any]]:
        """
        Get trades filtered by trading session.
        
        Args:
            session: Session name to filter by
            
        Returns:
            List of trade dictionaries
        """
        return [t for t in self.trades if t.get('session') == session]
    
    def get_trades_by_pair(self, pair: str) -> List[Dict[str, Any]]:
        """
        Get trades filtered by currency pair.
        
        Args:
            pair: Currency pair to filter by
            
        Returns:
            List of trade dictionaries
        """
        return [t for t in self.trades if t.get('pair') == pair]
    
    def get_trades_by_strategy(self, strategy_name: str) -> List[Dict[str, Any]]:
        """
        Get trades filtered by strategy name.
        
        Args:
            strategy_name: Strategy name to filter by
            
        Returns:
            List of trade dictionaries
        """
        return [t for t in self.trades if t.get('strategy_name') == strategy_name]
    
    def get_performance_metrics(self, 
                               trades: Optional[List[Dict[str, Any]]] = None, 
                               include_spreads: bool = True) -> Dict[str, Any]:
        """
        Calculate performance metrics for a set of trades.
        
        Args:
            trades: List of trades to analyze (default: all trades)
            include_spreads: Whether to include spread costs in calculations
            
        Returns:
            Dictionary with performance metrics
        """
        # Use all trades if not specified
        if trades is None:
            trades = self.trades
            
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pips': 0.0,
                'total_net_pips': 0.0,
                'avg_pips_per_trade': 0.0,
                'profit_factor': 0.0,
                'avg_winner_pips': 0.0,
                'avg_loser_pips': 0.0,
                'largest_winner_pips': 0.0,
                'largest_loser_pips': 0.0,
                'total_profit_loss': 0.0,
                'avg_trade_duration_minutes': 0.0,
                'total_spread_cost_pips': 0.0,
                'avg_spread_cost_pips': 0.0
            }
        
        # Extract pips values based on spread inclusion setting
        pip_values = [t['net_pips'] if include_spreads else t['pips'] for t in trades]
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if (t['net_pips'] if include_spreads else t['pips']) > 0]
        losing_trades = [t for t in trades if (t['net_pips'] if include_spreads else t['pips']) < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_pips = sum(t['pips'] for t in trades)
        total_net_pips = sum(t['net_pips'] for t in trades)
        avg_pips_per_trade = total_net_pips / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor (gross win / gross loss)
        gross_win_pips = sum((t['net_pips'] if include_spreads else t['pips']) for t in winning_trades)
        gross_loss_pips = abs(sum((t['net_pips'] if include_spreads else t['pips']) for t in losing_trades))
        profit_factor = gross_win_pips / gross_loss_pips if gross_loss_pips > 0 else float('inf')
        
        # Calculate average winners and losers
        avg_winner_pips = gross_win_pips / len(winning_trades) if winning_trades else 0.0
        avg_loser_pips = gross_loss_pips / len(losing_trades) if losing_trades else 0.0
        
        # Find largest winner and loser
        largest_winner_pips = max((t['net_pips'] if include_spreads else t['pips']) for t in winning_trades) if winning_trades else 0.0
        largest_loser_pips = min((t['net_pips'] if include_spreads else t['pips']) for t in losing_trades) if losing_trades else 0.0
        
        # Calculate monetary profit/loss
        total_profit_loss = sum(t['profit_loss'] for t in trades)
        
        # Calculate average trade duration
        durations = [t.get('duration_minutes', 0) for t in trades]
        avg_trade_duration_minutes = sum(durations) / len(durations) if durations else 0.0
        
        # Calculate spread costs
        total_spread_cost_pips = sum(t.get('spread_cost_pips', 0) for t in trades)
        avg_spread_cost_pips = total_spread_cost_pips / total_trades if total_trades > 0 else 0.0
        
        # Compile metrics
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'total_net_pips': total_net_pips,
            'avg_pips_per_trade': avg_pips_per_trade,
            'profit_factor': profit_factor,
            'avg_winner_pips': avg_winner_pips,
            'avg_loser_pips': avg_loser_pips,
            'largest_winner_pips': largest_winner_pips,
            'largest_loser_pips': largest_loser_pips,
            'total_profit_loss': total_profit_loss,
            'avg_trade_duration_minutes': avg_trade_duration_minutes,
            'total_spread_cost_pips': total_spread_cost_pips,
            'avg_spread_cost_pips': avg_spread_cost_pips
        }
        
        return metrics
    
    def get_session_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics segmented by trading session.
        
        Returns:
            Dictionary with session-specific metrics
        """
        # Get all sessions
        sessions = set(t.get('session') for t in self.trades if t.get('session'))
        
        # Calculate metrics for each session
        session_metrics = {}
        for session in sessions:
            session_trades = self.get_trades_by_session(session)
            session_metrics[session] = self.get_performance_metrics(session_trades)
            
        return session_metrics
    
    def get_pair_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics segmented by currency pair.
        
        Returns:
            Dictionary with pair-specific metrics
        """
        # Get all pairs
        pairs = set(t.get('pair') for t in self.trades if t.get('pair'))
        
        # Calculate metrics for each pair
        pair_metrics = {}
        for pair in pairs:
            pair_trades = self.get_trades_by_pair(pair)
            pair_metrics[pair] = self.get_performance_metrics(pair_trades)
            
        return pair_metrics
    
    def get_equity_curve_data(self) -> Dict[str, List[float]]:
        """
        Generate equity curve data from trades.
        
        Returns:
            Dictionary with dates and equity values
        """
        if not self.trades:
            return {'dates': [], 'equity': [], 'equity_pips': []}
        
        # Sort trades by exit time
        sorted_trades = sorted(self.trades, key=lambda t: t['exit_time'])
        
        # Initialize data
        dates = [sorted_trades[0]['entry_time']]
        equity = [100.0]  # Starting equity (base 100)
        equity_pips = [0.0]  # Cumulative pips
        
        # Build equity curve
        for trade in sorted_trades:
            dates.append(trade['exit_time'])
            
            # Add pip/monetary results
            equity.append(equity[-1] * (1 + trade['profit_loss'] / 100))
            equity_pips.append(equity_pips[-1] + trade['net_pips'])
        
        return {
            'dates': dates,
            'equity': equity,
            'equity_pips': equity_pips
        }
    
    def save_trades_to_file(self, filepath: str) -> bool:
        """
        Save trades data to a JSON file.
        
        Args:
            filepath: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert trades to serializable format
            serializable_trades = []
            for trade in self.trades:
                trade_copy = trade.copy()
                
                # Convert datetime objects to strings
                if 'entry_time' in trade_copy:
                    trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
                if 'exit_time' in trade_copy:
                    trade_copy['exit_time'] = trade_copy['exit_time'].isoformat()
                
                serializable_trades.append(trade_copy)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(serializable_trades, f, indent=2)
                
            logger.info(f"Saved {len(serializable_trades)} trades to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trades to file: {e}")
            return False
    
    def load_trades_from_file(self, filepath: str) -> bool:
        """
        Load trades data from a JSON file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                trades_data = json.load(f)
            
            # Convert strings back to datetime objects
            for trade in trades_data:
                if 'entry_time' in trade:
                    trade['entry_time'] = datetime.datetime.fromisoformat(trade['entry_time'])
                if 'exit_time' in trade:
                    trade['exit_time'] = datetime.datetime.fromisoformat(trade['exit_time'])
            
            # Set trades
            self.trades = trades_data
            logger.info(f"Loaded {len(trades_data)} trades from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading trades from file: {e}")
            return False
    
    def generate_pip_report(self, 
                          include_sessions: bool = True,
                          include_pairs: bool = True) -> str:
        """
        Generate a text report of pip-based performance.
        
        Args:
            include_sessions: Whether to include session breakdown
            include_pairs: Whether to include pair breakdown
            
        Returns:
            Formatted report string
        """
        # Get overall metrics
        metrics = self.get_performance_metrics()
        
        # Build report
        report = "=== FOREX PIP-BASED PERFORMANCE REPORT ===\n\n"
        
        # Overall metrics
        report += "== OVERALL METRICS ==\n"
        report += f"Total Trades: {metrics['total_trades']}\n"
        report += f"Win Rate: {metrics['win_rate']:.2%}\n"
        report += f"Total Net Pips: {metrics['total_net_pips']:.1f}\n"
        report += f"Average Pips/Trade: {metrics['avg_pips_per_trade']:.1f}\n"
        report += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        report += f"Total P&L: ${metrics['total_profit_loss']:.2f}\n"
        report += f"Average Winner: {metrics['avg_winner_pips']:.1f} pips\n"
        report += f"Average Loser: {metrics['avg_loser_pips']:.1f} pips\n"
        report += f"Largest Winner: {metrics['largest_winner_pips']:.1f} pips\n"
        report += f"Largest Loser: {metrics['largest_loser_pips']:.1f} pips\n"
        report += f"Total Spread Cost: {metrics['total_spread_cost_pips']:.1f} pips\n"
        report += f"Average Spread Cost: {metrics['avg_spread_cost_pips']:.1f} pips/trade\n\n"
        
        # Session breakdown
        if include_sessions:
            report += "== SESSION PERFORMANCE ==\n"
            session_metrics = self.get_session_performance()
            
            for session, s_metrics in session_metrics.items():
                if not session:
                    session = "Unknown"
                    
                report += f"--- {session} Session ---\n"
                report += f"Trades: {s_metrics['total_trades']}\n"
                report += f"Win Rate: {s_metrics['win_rate']:.2%}\n"
                report += f"Net Pips: {s_metrics['total_net_pips']:.1f}\n"
                report += f"Avg Pips/Trade: {s_metrics['avg_pips_per_trade']:.1f}\n"
                report += f"Profit Factor: {s_metrics['profit_factor']:.2f}\n\n"
        
        # Pair breakdown
        if include_pairs:
            report += "== PAIR PERFORMANCE ==\n"
            pair_metrics = self.get_pair_performance()
            
            for pair, p_metrics in pair_metrics.items():
                report += f"--- {pair} ---\n"
                report += f"Trades: {p_metrics['total_trades']}\n"
                report += f"Win Rate: {p_metrics['win_rate']:.2%}\n"
                report += f"Net Pips: {p_metrics['total_net_pips']:.1f}\n"
                report += f"Avg Pips/Trade: {p_metrics['avg_pips_per_trade']:.1f}\n"
                report += f"Profit Factor: {p_metrics['profit_factor']:.2f}\n"
                report += f"Spread Cost: {p_metrics['total_spread_cost_pips']:.1f} pips\n\n"
                
        return report
    
    def plot_pip_curve(self, 
                     save_path: Optional[str] = None, 
                     show_equity: bool = True,
                     show_pips: bool = True) -> None:
        """
        Plot pip-based equity curve.
        
        Args:
            save_path: Path to save the plot (optional)
            show_equity: Whether to show monetary equity curve
            show_pips: Whether to show pip-based curve
        """
        if not self.trades:
            logger.warning("No trades to plot")
            return
            
        # Get equity curve data
        curve_data = self.get_equity_curve_data()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot monetary equity if requested
        if show_equity:
            plt.subplot(2, 1, 1) if show_pips else plt.subplot(1, 1, 1)
            plt.plot(curve_data['dates'], curve_data['equity'], label='Equity')
            plt.title('Account Equity')
            plt.grid(True)
            plt.legend()
        
        # Plot pip equity if requested
        if show_pips:
            plt.subplot(2, 1, 2) if show_equity else plt.subplot(1, 1, 1)
            plt.plot(curve_data['dates'], curve_data['equity_pips'], label='Cumulative Pips', color='green')
            plt.title('Cumulative Pips')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                plt.savefig(save_path)
                logger.info(f"Saved pip curve plot to {save_path}")
            except Exception as e:
                logger.error(f"Error saving plot: {e}")
        
        plt.close()


# Module execution
if __name__ == "__main__":
    import argparse
    from tabulate import tabulate
    
    parser = argparse.ArgumentParser(description="Forex Pip Logger")
    
    parser.add_argument(
        "--load", 
        type=str,
        help="Load trades from JSON file"
    )
    
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate performance report"
    )
    
    parser.add_argument(
        "--session", 
        type=str,
        help="Filter report by session"
    )
    
    parser.add_argument(
        "--pair", 
        type=str,
        help="Filter report by currency pair"
    )
    
    parser.add_argument(
        "--plot", 
        type=str,
        help="Generate and save pip curve plot to file"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    pip_logger = ForexPipLogger()
    
    # Load trades if specified
    if args.load and os.path.exists(args.load):
        pip_logger.load_trades_from_file(args.load)
    
    # Generate report
    if args.report:
        if args.session:
            trades = pip_logger.get_trades_by_session(args.session)
            metrics = pip_logger.get_performance_metrics(trades)
            print(f"\nPerformance Report for {args.session} Session:")
        elif args.pair:
            trades = pip_logger.get_trades_by_pair(args.pair.upper())
            metrics = pip_logger.get_performance_metrics(trades)
            print(f"\nPerformance Report for {args.pair.upper()}:")
        else:
            print(pip_logger.generate_pip_report())
            # Exit early since full report is generated
            exit(0)
            
        # Print filtered report
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Net Pips: {metrics['total_net_pips']:.1f}")
        print(f"Average Pips/Trade: {metrics['avg_pips_per_trade']:.1f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total P&L: ${metrics['total_profit_loss']:.2f}")
        print(f"Total Spread Cost: {metrics['total_spread_cost_pips']:.1f} pips")
    
    # Generate plot
    if args.plot:
        pip_logger.plot_pip_curve(save_path=args.plot)
        print(f"Pip curve plot saved to {args.plot}")
        
    # Default behavior - show sample usage
    if not (args.load or args.report or args.plot):
        print("Forex Pip Logger - Sample Usage")
        print("==============================")
        print("This module provides pip-based trade tracking for forex.")
        print("\nExample trade logging:")
        print("  logger = ForexPipLogger()")
        print("  logger.log_trade(")
        print("      pair='EURUSD',")
        print("      entry_time=datetime.datetime.now(),")
        print("      exit_time=datetime.datetime.now() + datetime.timedelta(hours=2),")
        print("      direction=1,  # 1 for long, -1 for short")
        print("      entry_price=1.10250,")
        print("      exit_price=1.10350,")
        print("      lot_size=0.1,")
        print("      session='London'")
        print("  )")
        print("\nUse --help to see all available commands")
