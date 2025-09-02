#!/usr/bin/env python3
"""
Session Performance Tracker

Tracks and analyzes trading strategy performance by session (London, New York, Asia).
Extends the strategy registry with session-specific metrics and visualizations.
Integrates with BenBot for session-aware trading decisions.
"""

import os
import pandas as pd
import numpy as np
import datetime
import logging
import json
import requests
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

# Import internal components
from session_performance_db import SessionPerformanceDB
from forex_session_manager import ForexSessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('session_performance_tracker')


class SessionPerformanceTracker:
    """
    Tracks and analyzes trading strategy performance by session.
    
    Features:
    - Extends strategy registry with session-specific metrics
    - Identifies optimal trading sessions for each strategy
    - Visualizes session performance with customizable charts
    - Integrates with BenBot for session-aware trading decisions
    """
    
    def __init__(self, 
                db_path: str = 'session_performance.db',
                benbot_endpoint: Optional[str] = None):
        """
        Initialize the session performance tracker.
        
        Args:
            db_path: Path to the database file
            benbot_endpoint: BenBot API endpoint for integration (optional)
        """
        # Initialize components
        self.db = SessionPerformanceDB(db_path)
        self.session_manager = ForexSessionManager()
        
        # BenBot integration
        self.benbot_endpoint = benbot_endpoint
        self.benbot_available = False
        
        if benbot_endpoint:
            self.benbot_available = self._check_benbot_connection()
            if self.benbot_available:
                logger.info(f"BenBot integration enabled at {benbot_endpoint}")
            else:
                logger.warning("BenBot integration unavailable - running in standalone mode")
        
        logger.info("Session Performance Tracker initialized")
    
    def _check_benbot_connection(self) -> bool:
        """Check if BenBot integration is available."""
        if not self.benbot_endpoint:
            return False
        
        try:
            # Simple ping to BenBot
            endpoint = f"{self.benbot_endpoint.rstrip('/')}/ping"
            response = requests.get(endpoint, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def process_trades(self, 
                      strategy_id: str, 
                      strategy_name: str,
                      trades_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Process trades and update session-specific performance metrics.
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_name: Strategy name
            trades_df: DataFrame with trade data
            
        Returns:
            Dictionary with session performance metrics
        """
        # Ensure trades DataFrame has session information
        if 'session' not in trades_df.columns:
            trades_df = self._add_session_info(trades_df)
        
        # Group trades by session
        session_trades = trades_df.groupby('session')
        
        # Process each session
        session_metrics = {}
        for session, group in session_trades:
            # Skip if no session
            if not session or pd.isna(session):
                continue
                
            # Calculate session-specific metrics
            metrics = self._calculate_metrics(group)
            
            # Add to result dictionary
            session_metrics[session] = metrics
            
            # Update database
            self.db.update_session_performance(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                session=session,
                metrics=metrics
            )
            
            logger.info(f"Updated {session} metrics for {strategy_name} - {len(group)} trades")
        
        # Report to BenBot if available
        if self.benbot_available:
            self._report_to_benbot(strategy_id, strategy_name, session_metrics)
        
        return session_metrics
    
    def _add_session_info(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session information to trades DataFrame.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            DataFrame with added session column
        """
        # Create a copy to avoid modifying the original
        df = trades_df.copy()
        
        # Ensure we have datetime index or entry_time column
        if not pd.api.types.is_datetime64_dtype(df.index):
            if 'entry_time' in df.columns:
                if not pd.api.types.is_datetime64_dtype(df['entry_time']):
                    df['entry_time'] = pd.to_datetime(df['entry_time'])
                timestamp_col = 'entry_time'
            else:
                logger.warning("No datetime index or entry_time column found")
                df['session'] = None
                return df
        else:
            timestamp_col = None  # Use index
        
        # Add session column
        df['session'] = None
        
        # Process each row
        for idx in df.index:
            # Get timestamp
            if timestamp_col:
                timestamp = df.loc[idx, timestamp_col]
            else:
                timestamp = idx
            
            # Get active sessions
            active_sessions = self.session_manager.get_active_sessions(timestamp)
            
            # Assign session (prioritize major sessions)
            if active_sessions:
                for preferred in ['London', 'NewYork', 'Tokyo', 'Sydney']:
                    if preferred in active_sessions:
                        df.loc[idx, 'session'] = preferred
                        break
                        
                # If no preferred session matched, use the first one
                if df.loc[idx, 'session'] is None:
                    df.loc[idx, 'session'] = active_sessions[0]
        
        return df
    
    def _calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from trades.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Dictionary with performance metrics
        """
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pips': 0.0,
                'profit_factor': 0.0,
                'avg_pips_per_trade': 0.0,
                'max_drawdown': 0.0
            }
        
        # Determine profit column to use (pips preferred)
        if 'net_pips' in trades_df.columns:
            profit_col = 'net_pips'
        elif 'pips' in trades_df.columns:
            profit_col = 'pips'
        elif 'profit_loss' in trades_df.columns:
            profit_col = 'profit_loss'
        else:
            logger.warning("No profit column found in trades DataFrame")
            return {
                'total_trades': len(trades_df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pips': 0.0,
                'profit_factor': 0.0,
                'avg_pips_per_trade': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df[profit_col] > 0])
        losing_trades = len(trades_df[trades_df[profit_col] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate pip metrics if available
        if profit_col in ['net_pips', 'pips']:
            total_pips = trades_df[profit_col].sum()
            
            # Calculate profit factor
            gross_win = trades_df[trades_df[profit_col] > 0][profit_col].sum()
            gross_loss = abs(trades_df[trades_df[profit_col] < 0][profit_col].sum())
            profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
            
            avg_pips_per_trade = total_pips / total_trades
            
        else:
            # Use monetary profit
            total_pips = 0.0
            profit_factor = 0.0
            avg_pips_per_trade = 0.0
        
        # Calculate drawdown
        if 'equity' in trades_df.columns or 'equity_curve' in trades_df.columns:
            equity_col = 'equity' if 'equity' in trades_df.columns else 'equity_curve'
            # Calculate maximum drawdown
            equity = trades_df[equity_col].values
            max_drawdown = self._calculate_max_drawdown(equity)
        else:
            # Calculate rolling drawdown from trade results
            max_drawdown = 0.0
        
        # Calculate additional metrics if data available
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pips': total_pips if profit_col in ['net_pips', 'pips'] else 0.0,
            'profit_factor': profit_factor,
            'avg_pips_per_trade': avg_pips_per_trade,
            'max_drawdown': max_drawdown
        }
        
        # Add Sharpe ratio if daily returns available
        if 'daily_returns' in trades_df.columns:
            daily_returns = trades_df['daily_returns'].dropna().values
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe_ratio
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            Maximum drawdown as a decimal (not percentage)
        """
        if len(equity_curve) == 0:
            return 0.0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (running_max - equity_curve) / running_max
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
    def _report_to_benbot(self, 
                        strategy_id: str, 
                        strategy_name: str,
                        session_metrics: Dict[str, Dict[str, Any]]) -> bool:
        """
        Report session performance metrics to BenBot.
        
        Args:
            strategy_id: Strategy ID
            strategy_name: Strategy name
            session_metrics: Dictionary with session metrics
            
        Returns:
            True if successfully reported, False otherwise
        """
        if not self.benbot_available or not self.benbot_endpoint:
            return False
            
        try:
            # Get strategy metadata
            metadata = self.db.get_strategy_metadata(strategy_id)
            
            # Create payload
            payload = {
                'source': 'EvoTrader',
                'module': 'SessionPerformanceTracker',
                'strategy_id': strategy_id,
                'strategy_name': strategy_name,
                'optimal_session': metadata.get('optimal_session') if metadata else None,
                'session_metrics': session_metrics,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Send to BenBot
            endpoint = f"{self.benbot_endpoint.rstrip('/')}/session_performance"
            response = requests.post(endpoint, json=payload, timeout=5)
            
            success = response.status_code == 200
            if not success:
                logger.warning(f"Failed to report to BenBot: {response.status_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error reporting to BenBot: {e}")
            return False
    
    def get_session_performance(self, 
                              strategy_id: str,
                              session: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get session performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            session: Specific session to get (optional)
            
        Returns:
            List of session performance dictionaries
        """
        return self.db.get_session_performance(strategy_id, session)
    
    def get_optimal_session(self, strategy_id: str) -> Optional[str]:
        """
        Get the optimal trading session for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Optimal session name or None if unknown
        """
        metadata = self.db.get_strategy_metadata(strategy_id)
        if metadata:
            return metadata.get('optimal_session')
        return None
    
    def get_strategies_for_session(self, session: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get strategies optimized for a specific trading session.
        
        Args:
            session: Session name
            limit: Maximum number of strategies to return
            
        Returns:
            List of strategy dictionaries
        """
        strategies = self.db.get_strategies_by_session(session)
        return strategies[:limit] if limit else strategies
    
    def check_benbot_session_directive(self, 
                                     strategy_id: str, 
                                     current_session: str) -> Dict[str, Any]:
        """
        Check if BenBot has specific directives for the strategy in this session.
        
        Args:
            strategy_id: Strategy ID
            current_session: Current trading session
            
        Returns:
            Dictionary with BenBot directive information
        """
        if not self.benbot_available or not self.benbot_endpoint:
            # Default response when BenBot is unavailable
            return {
                'trade_allowed': True,
                'reason': 'BenBot unavailable, using local decision',
                'override': False
            }
            
        try:
            # Request directive from BenBot
            endpoint = f"{self.benbot_endpoint.rstrip('/')}/session_directive"
            
            payload = {
                'source': 'EvoTrader',
                'module': 'SessionPerformanceTracker',
                'strategy_id': strategy_id,
                'current_session': current_session,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            response = requests.post(endpoint, json=payload, timeout=2)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get BenBot directive: {response.status_code}")
                # Default to permissive response
                return {
                    'trade_allowed': True,
                    'reason': f"BenBot returned error {response.status_code}",
                    'override': False
                }
                
        except Exception as e:
            logger.error(f"Error checking BenBot directive: {e}")
            # Default to permissive response on error
            return {
                'trade_allowed': True,
                'reason': f"Error connecting to BenBot: {str(e)}",
                'override': False
            }
    
    def plot_session_performance(self, 
                               strategy_id: str,
                               save_path: Optional[str] = None) -> None:
        """
        Generate a visualization of strategy performance by session.
        
        Args:
            strategy_id: Strategy ID
            save_path: Path to save the visualization (optional)
        """
        # Get strategy info
        metadata = self.db.get_strategy_metadata(strategy_id)
        if not metadata:
            logger.warning(f"Strategy not found: {strategy_id}")
            return
            
        # Get session performance
        performances = self.db.get_session_performance(strategy_id)
        if not performances:
            logger.warning(f"No session performance data for strategy: {strategy_id}")
            return
        
        # Extract data for plotting
        sessions = []
        win_rates = []
        profit_factors = []
        avg_pips = []
        trade_counts = []
        
        for perf in performances:
            sessions.append(perf['session'])
            win_rates.append(perf['win_rate'] * 100)  # Convert to percentage
            profit_factors.append(perf['profit_factor'])
            avg_pips.append(perf['avg_pips_per_trade'])
            trade_counts.append(perf['total_trades'])
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot win rates
        plt.subplot(2, 2, 1)
        plt.bar(sessions, win_rates, color='blue', alpha=0.7)
        plt.title('Win Rate by Session')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Plot profit factors
        plt.subplot(2, 2, 2)
        plt.bar(sessions, profit_factors, color='green', alpha=0.7)
        plt.title('Profit Factor by Session')
        plt.ylabel('Profit Factor')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Plot average pips
        plt.subplot(2, 2, 3)
        plt.bar(sessions, avg_pips, color='orange', alpha=0.7)
        plt.title('Average Pips per Trade by Session')
        plt.ylabel('Pips')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Plot trade counts
        plt.subplot(2, 2, 4)
        plt.bar(sessions, trade_counts, color='purple', alpha=0.7)
        plt.title('Trade Count by Session')
        plt.ylabel('Number of Trades')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add overall title
        plt.suptitle(f"Session Performance: {metadata['strategy_name']}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save or show
        if save_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Session performance plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving plot: {e}")
                
        plt.close()
    
    def generate_session_report(self, strategy_id: str) -> str:
        """
        Generate a text report of session performance.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Formatted report string
        """
        # Get strategy info
        metadata = self.db.get_strategy_metadata(strategy_id)
        if not metadata:
            return f"Strategy not found: {strategy_id}"
            
        # Get session performance
        performances = self.db.get_session_performance(strategy_id)
        if not performances:
            return f"No session performance data for strategy: {metadata['strategy_name']}"
        
        # Build report
        report = f"=== SESSION PERFORMANCE REPORT ===\n"
        report += f"Strategy: {metadata['strategy_name']} ({strategy_id})\n"
        report += f"Type: {metadata.get('strategy_type', 'Unknown')}\n"
        report += f"Optimal Session: {metadata.get('optimal_session', 'Unknown')}\n\n"
        
        # Add performance by session
        report += "Performance by Session:\n"
        report += "------------------------\n\n"
        
        for perf in sorted(performances, key=lambda p: p['profit_factor'] * p['win_rate'], reverse=True):
            report += f"== {perf['session']} Session ==\n"
            report += f"Trades: {perf['total_trades']}\n"
            report += f"Win Rate: {perf['win_rate']:.2%}\n"
            report += f"Profit Factor: {perf['profit_factor']:.2f}\n"
            report += f"Total Pips: {perf['total_pips']:.1f}\n"
            report += f"Avg Pips/Trade: {perf['avg_pips_per_trade']:.1f}\n"
            report += f"Max Drawdown: {perf['max_drawdown']:.2%}\n"
            if 'sharpe_ratio' in perf and perf['sharpe_ratio'] is not None:
                report += f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}\n"
            report += "\n"
        
        # Add recommendations
        report += "Recommendations:\n"
        report += "---------------\n"
        if metadata.get('optimal_session'):
            report += f"- Primary trading session should be {metadata['optimal_session']}\n"
            
            # Find second best session if available
            if len(performances) > 1:
                sorted_perfs = sorted(performances, key=lambda p: p['profit_factor'] * p['win_rate'], reverse=True)
                if sorted_perfs[0]['session'] == metadata['optimal_session'] and len(sorted_perfs) > 1:
                    report += f"- Secondary session could be {sorted_perfs[1]['session']}\n"
        
        return report


# Module execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Session Performance Tracker")
    
    parser.add_argument(
        "--db", 
        type=str,
        default="session_performance.db",
        help="Database file path"
    )
    
    parser.add_argument(
        "--strategy", 
        type=str,
        help="Show session performance for a specific strategy"
    )
    
    parser.add_argument(
        "--session", 
        type=str,
        help="Filter by session or show strategies for a session"
    )
    
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate detailed report for the specified strategy"
    )
    
    parser.add_argument(
        "--plot", 
        type=str,
        help="Generate and save session performance plot to specified path"
    )
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = SessionPerformanceTracker(args.db)
    
    # Process commands
    if args.strategy:
        if args.report:
            # Generate report
            report = tracker.generate_session_report(args.strategy)
            print(report)
        elif args.plot:
            # Generate plot
            tracker.plot_session_performance(args.strategy, args.plot)
            print(f"Session performance plot saved to {args.plot}")
        else:
            # Show basic info
            metadata = tracker.db.get_strategy_metadata(args.strategy)
            if metadata:
                print(f"\nStrategy: {metadata['strategy_name']} ({args.strategy})")
                print(f"Optimal Session: {metadata.get('optimal_session', 'Unknown')}")
                
                # Show session performance
                if args.session:
                    performances = tracker.get_session_performance(args.strategy, args.session)
                    print(f"\nPerformance in {args.session} session:")
                else:
                    performances = tracker.get_session_performance(args.strategy)
                    print("\nPerformance by session:")
                
                for perf in performances:
                    print(f"- {perf['session']}:")
                    print(f"  Trades: {perf['total_trades']}")
                    print(f"  Win Rate: {perf['win_rate']:.2%}")
                    print(f"  Profit Factor: {perf['profit_factor']:.2f}")
                    print(f"  Avg Pips/Trade: {perf['avg_pips_per_trade']:.1f}")
            else:
                print(f"Strategy not found: {args.strategy}")
    
    # Show strategies for a session
    elif args.session:
        strategies = tracker.get_strategies_for_session(args.session)
        print(f"\nTop Strategies for {args.session} Session:")
        
        for i, strat in enumerate(strategies, 1):
            print(f"{i}. {strat['strategy_name']} (ID: {strat['strategy_id']})")
            print(f"   Win Rate: {strat['win_rate']:.2%}")
            print(f"   Profit Factor: {strat['profit_factor']:.2f}")
            print(f"   Total Trades: {strat['total_trades']}")
            print()
    
    # Default behavior - show help
    if not (args.strategy or (args.session and not args.strategy)):
        print("Session Performance Tracker")
        print("==========================")
        print("This tool tracks and analyzes strategy performance by trading session.")
        print("\nExamples:")
        print("  --strategy STRAT_ID --report  : Generate full session report")
        print("  --strategy STRAT_ID --plot plot.png : Create session performance visualization")
        print("  --session London : List top strategies for London session")
        print("\nUse --help to see all available commands")
