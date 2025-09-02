#!/usr/bin/env python3
"""
Forex Session Manager

This module provides tools for working with global forex trading sessions,
including session detection, overlap identification, and optimal trading window selection.
It supports filtering trades and analyses by session and provides session-aware metrics.
"""

import datetime
import pandas as pd
import numpy as np
import pytz
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_session_manager')


class ForexSessionManager:
    """
    Manages forex trading sessions with timezone-aware calculations.
    
    Features:
    - Identifies current active sessions
    - Detects session overlaps (high liquidity periods)
    - Labels data by session
    - Provides session-specific statistics
    - Tracks session performance metrics
    """
    
    # Default session definitions (UTC times)
    DEFAULT_SESSIONS = {
        'Sydney': {'start': '21:00', 'end': '06:00'},
        'Tokyo': {'start': '23:00', 'end': '08:00'},
        'London': {'start': '07:00', 'end': '16:00'},
        'NewYork': {'start': '12:00', 'end': '21:00'}
    }
    
    # Session group definitions
    SESSION_GROUPS = {
        'Asia': ['Sydney', 'Tokyo'],
        'Europe': ['London'],
        'Americas': ['NewYork']
    }
    
    # Traditional forex session overlaps (high liquidity periods)
    KEY_OVERLAPS = {
        'Tokyo-London': {'start': '07:00', 'end': '08:00', 'description': 'Asian/European overlap'},
        'London-NewYork': {'start': '12:00', 'end': '16:00', 'description': 'European/American overlap'}
    }
    
    def __init__(self, custom_sessions: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize the forex session manager.
        
        Args:
            custom_sessions: Optional custom session definitions
                Format: {'SessionName': {'start': 'HH:MM', 'end': 'HH:MM'}, ...}
        """
        self.sessions = custom_sessions if custom_sessions else self.DEFAULT_SESSIONS
        self._parse_session_times()
        logger.info(f"Forex Session Manager initialized with {len(self.sessions)} sessions")
    
    def _parse_session_times(self) -> None:
        """Parse session start/end strings into datetime.time objects."""
        for session, times in self.sessions.items():
            try:
                start_hour, start_min = map(int, times['start'].split(':'))
                end_hour, end_min = map(int, times['end'].split(':'))
                
                self.sessions[session]['start_time'] = datetime.time(start_hour, start_min)
                self.sessions[session]['end_time'] = datetime.time(end_hour, end_min)
                
            except Exception as e:
                logger.error(f"Error parsing session times for {session}: {e}")
                # Set default times (midnight)
                self.sessions[session]['start_time'] = datetime.time(0, 0)
                self.sessions[session]['end_time'] = datetime.time(0, 0)
    
    def get_active_sessions(self, timestamp: Optional[Union[datetime.datetime, pd.Timestamp]] = None) -> List[str]:
        """
        Get active sessions for the given timestamp.
        
        Args:
            timestamp: The datetime to check (default: current time)
            
        Returns:
            List of active session names
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(pytz.UTC)
        elif not timestamp.tzinfo:
            # Assume UTC if no timezone is specified
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
            
        current_time = timestamp.time()
        active_sessions = []
        
        for session, times in self.sessions.items():
            start_time = times['start_time']
            end_time = times['end_time']
            
            # Handle sessions crossing midnight
            if start_time > end_time:
                if current_time >= start_time or current_time < end_time:
                    active_sessions.append(session)
            else:
                if start_time <= current_time < end_time:
                    active_sessions.append(session)
        
        return active_sessions
    
    def get_active_session_groups(self, timestamp: Optional[Union[datetime.datetime, pd.Timestamp]] = None) -> List[str]:
        """
        Get active session groups for the given timestamp.
        
        Args:
            timestamp: The datetime to check (default: current time)
            
        Returns:
            List of active session group names
        """
        active_sessions = self.get_active_sessions(timestamp)
        active_groups = []
        
        for group, sessions in self.SESSION_GROUPS.items():
            if any(session in active_sessions for session in sessions):
                active_groups.append(group)
                
        return active_groups
    
    def is_session_overlap(self, timestamp: Optional[Union[datetime.datetime, pd.Timestamp]] = None) -> bool:
        """
        Check if current time is in a session overlap period.
        
        Args:
            timestamp: The datetime to check (default: current time)
            
        Returns:
            True if in overlap period, False otherwise
        """
        active_groups = self.get_active_session_groups(timestamp)
        return len(active_groups) > 1
    
    def get_session_overlap_info(self, timestamp: Optional[Union[datetime.datetime, pd.Timestamp]] = None) -> Dict[str, Any]:
        """
        Get detailed overlap information for the given timestamp.
        
        Args:
            timestamp: The datetime to check (default: current time)
            
        Returns:
            Dictionary with overlap details
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(pytz.UTC)
        elif not timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
            
        current_time = timestamp.time()
        overlap_info = {'is_overlap': False, 'sessions': [], 'description': ''}
        
        # Check for predefined key overlaps
        for name, overlap in self.KEY_OVERLAPS.items():
            start_hour, start_min = map(int, overlap['start'].split(':'))
            end_hour, end_min = map(int, overlap['end'].split(':'))
            
            start_time = datetime.time(start_hour, start_min)
            end_time = datetime.time(end_hour, end_min)
            
            # Handle overlaps crossing midnight
            if start_time > end_time:
                if current_time >= start_time or current_time < end_time:
                    overlap_info['is_overlap'] = True
                    overlap_info['sessions'] = name.split('-')
                    overlap_info['description'] = overlap['description']
                    return overlap_info
            else:
                if start_time <= current_time < end_time:
                    overlap_info['is_overlap'] = True
                    overlap_info['sessions'] = name.split('-')
                    overlap_info['description'] = overlap['description']
                    return overlap_info
        
        # Generic overlap detection
        active_sessions = self.get_active_sessions(timestamp)
        if len(active_sessions) > 1:
            overlap_info['is_overlap'] = True
            overlap_info['sessions'] = active_sessions
            overlap_info['description'] = f"Overlap between {' and '.join(active_sessions)}"
            
        return overlap_info

    def label_dataframe_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session labels to a DataFrame.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with additional session columns
        """
        # Create copy to avoid modifying the original
        result = df.copy()
        
        # Create session indicator columns
        for session in self.sessions:
            result[f'session_{session}'] = 0
            
        for overlap in self.KEY_OVERLAPS:
            result[f'overlap_{overlap}'] = 0
        
        # Add session group columns
        for group in self.SESSION_GROUPS:
            result[f'session_group_{group}'] = 0
        
        # Add generic overlap indicator
        result['is_session_overlap'] = 0
        
        # Process each row
        for idx, timestamp in enumerate(result.index):
            # Label individual sessions
            active_sessions = self.get_active_sessions(timestamp)
            for session in active_sessions:
                result.loc[timestamp, f'session_{session}'] = 1
            
            # Label session groups
            active_groups = self.get_active_session_groups(timestamp)
            for group in active_groups:
                result.loc[timestamp, f'session_group_{group}'] = 1
            
            # Label overlaps
            overlap_info = self.get_session_overlap_info(timestamp)
            if overlap_info['is_overlap']:
                result.loc[timestamp, 'is_session_overlap'] = 1
                
                # Label specific overlaps
                for overlap in self.KEY_OVERLAPS:
                    sessions = overlap.split('-')
                    if all(session in active_sessions for session in sessions):
                        result.loc[timestamp, f'overlap_{overlap}'] = 1
        
        return result
    
    def filter_by_session(self, df: pd.DataFrame, session: str) -> pd.DataFrame:
        """
        Filter DataFrame to only include rows from a specific session.
        
        Args:
            df: DataFrame with datetime index
            session: Session name to filter by
            
        Returns:
            Filtered DataFrame
        """
        # First ensure session labels exist
        if f'session_{session}' not in df.columns:
            df = self.label_dataframe_sessions(df)
        
        # Filter rows
        return df[df[f'session_{session}'] == 1]
    
    def filter_by_overlap(self, df: pd.DataFrame, overlap: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame to only include rows during session overlaps.
        
        Args:
            df: DataFrame with datetime index
            overlap: Specific overlap to filter by (default: any overlap)
            
        Returns:
            Filtered DataFrame
        """
        # First ensure labels exist
        if 'is_session_overlap' not in df.columns:
            df = self.label_dataframe_sessions(df)
        
        # Filter by specific overlap or any overlap
        if overlap and f'overlap_{overlap}' in df.columns:
            return df[df[f'overlap_{overlap}'] == 1]
        else:
            return df[df['is_session_overlap'] == 1]
    
    def get_session_statistics(self, trades_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate trading statistics grouped by session.
        
        Args:
            trades_df: DataFrame with trade data and datetime index
            
        Returns:
            Dictionary with session stats
        """
        # Ensure session labels exist
        trades = self.label_dataframe_sessions(trades_df)
        
        session_stats = {}
        
        # Process each session
        for session in self.sessions:
            session_trades = trades[trades[f'session_{session}'] == 1]
            
            if len(session_trades) > 0 and 'pnl' in session_trades.columns:
                # Calculate statistics
                win_rate = (session_trades['pnl'] > 0).mean() * 100
                avg_win = session_trades[session_trades['pnl'] > 0]['pnl'].mean() if any(session_trades['pnl'] > 0) else 0
                avg_loss = session_trades[session_trades['pnl'] < 0]['pnl'].mean() if any(session_trades['pnl'] < 0) else 0
                profit_factor = abs(session_trades[session_trades['pnl'] > 0]['pnl'].sum() / 
                                   session_trades[session_trades['pnl'] < 0]['pnl'].sum()) if session_trades[session_trades['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
                
                # Store stats
                session_stats[session] = {
                    'trade_count': len(session_trades),
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'total_pnl': session_trades['pnl'].sum(),
                    'avg_pnl': session_trades['pnl'].mean()
                }
                
                # Add pip metrics if available
                if 'pips' in session_trades.columns:
                    session_stats[session]['total_pips'] = session_trades['pips'].sum()
                    session_stats[session]['avg_pips'] = session_trades['pips'].mean()
            else:
                # No trades in this session
                session_stats[session] = {
                    'trade_count': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
                
                if 'pips' in trades.columns:
                    session_stats[session]['total_pips'] = 0
                    session_stats[session]['avg_pips'] = 0
        
        return session_stats
    
    def get_session_windows(self, pair: str = None) -> Dict[str, Dict[str, datetime.time]]:
        """
        Get optimal trading windows for specific forex pairs.
        
        Args:
            pair: Optional forex pair to get specialized windows for
            
        Returns:
            Dictionary of trading windows with start/end times
        """
        # Base windows from sessions
        windows = {
            'london_open': {
                'start': self.sessions['London']['start_time'],
                'end': datetime.time((self.sessions['London']['start_time'].hour + 1) % 24, self.sessions['London']['start_time'].minute),
                'description': 'London session opening hour'
            },
            'ny_open': {
                'start': self.sessions['NewYork']['start_time'],
                'end': datetime.time((self.sessions['NewYork']['start_time'].hour + 1) % 24, self.sessions['NewYork']['start_time'].minute),
                'description': 'New York session opening hour'
            },
            'london_ny_overlap': {
                'start': self.KEY_OVERLAPS['London-NewYork']['start'].split(':')[0],
                'end': self.KEY_OVERLAPS['London-NewYork']['end'].split(':')[0],
                'description': 'London-New York overlap (highest liquidity)'
            },
            'asia_open': {
                'start': self.sessions['Tokyo']['start_time'],
                'end': datetime.time((self.sessions['Tokyo']['start_time'].hour + 1) % 24, self.sessions['Tokyo']['start_time'].minute),
                'description': 'Tokyo session opening hour'
            }
        }
        
        # Pair-specific optimizations
        if pair:
            pair = pair.upper()
            
            # EUR/USD and GBP/USD - European pairs
            if pair in ['EURUSD', 'GBPUSD', 'EURGBP']:
                # European pairs often move most during London session
                windows['optimal'] = {
                    'start': self.sessions['London']['start_time'],
                    'end': datetime.time(16, 0),  # End of London session
                    'description': f'Optimal window for {pair} (London and London-NY overlap)'
                }
            
            # USD/JPY and AUD/USD - Asian influence
            elif pair in ['USDJPY', 'AUDUSD', 'NZDUSD']:
                # These pairs often have good movement during Asia and into London
                windows['optimal'] = {
                    'start': self.sessions['Tokyo']['start_time'],
                    'end': datetime.time(10, 0),  # Mid-London session
                    'description': f'Optimal window for {pair} (Tokyo and into London)'
                }
                
            # USD/CAD - Americas focus
            elif pair == 'USDCAD':
                # CAD pairs often move most during NY session
                windows['optimal'] = {
                    'start': datetime.time(12, 0),  # NY open
                    'end': datetime.time(17, 0),  # Mid NY session
                    'description': f'Optimal window for {pair} (NY session)'
                }
            
            else:
                # Default to London-NY overlap for other pairs
                windows['optimal'] = {
                    'start': datetime.time(12, 0),
                    'end': datetime.time(16, 0),
                    'description': f'Default optimal window for {pair} (London-NY overlap)'
                }
        
        return windows
    
    def is_optimal_trading_window(self, timestamp: Union[datetime.datetime, pd.Timestamp], 
                                 pair: str = None) -> bool:
        """
        Check if current time is in an optimal trading window for the given pair.
        
        Args:
            timestamp: The datetime to check
            pair: Optional forex pair to use specialized windows
            
        Returns:
            True if in optimal window, False otherwise
        """
        if not timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
            
        windows = self.get_session_windows(pair)
        current_time = timestamp.time()
        
        if 'optimal' in windows:
            window = windows['optimal']
            start_time = window['start']
            end_time = window['end']
            
            # Handle windows crossing midnight
            if start_time > end_time:
                return current_time >= start_time or current_time < end_time
            else:
                return start_time <= current_time < end_time
        else:
            # Default to checking if in any session
            return len(self.get_active_sessions(timestamp)) > 0
    
    def get_next_session_change(self, timestamp: Optional[Union[datetime.datetime, pd.Timestamp]] = None) -> Dict[str, Any]:
        """
        Find the next session transition from the given timestamp.
        
        Args:
            timestamp: The datetime to check (default: current time)
            
        Returns:
            Dictionary with next session change details
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(pytz.UTC)
        elif not timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
            
        current_time = timestamp.time()
        current_date = timestamp.date()
        
        next_change = {
            'event': None,
            'session': None,
            'timestamp': None,
            'seconds_until': float('inf')
        }
        
        # Check for next session start or end
        for session, times in self.sessions.items():
            start_time = times['start_time']
            end_time = times['end_time']
            
            # Calculate time until session start
            if current_time < start_time or (start_time > end_time and current_time >= end_time):
                # Session starts later today
                start_timestamp = datetime.datetime.combine(current_date, start_time).replace(tzinfo=pytz.UTC)
                
                # If current time is past the start time, session starts tomorrow
                if current_time > start_time:
                    start_timestamp += datetime.timedelta(days=1)
                
                seconds_until = (start_timestamp - timestamp).total_seconds()
                
                if seconds_until < next_change['seconds_until']:
                    next_change['event'] = 'start'
                    next_change['session'] = session
                    next_change['timestamp'] = start_timestamp
                    next_change['seconds_until'] = seconds_until
            
            # Calculate time until session end
            if (start_time < end_time and start_time <= current_time < end_time) or \
               (start_time > end_time and (current_time >= start_time or current_time < end_time)):
                # Session is active, calculate time until end
                end_timestamp = datetime.datetime.combine(current_date, end_time).replace(tzinfo=pytz.UTC)
                
                # If end time is earlier than start time, session ends tomorrow
                if end_time < start_time and current_time >= start_time:
                    end_timestamp += datetime.timedelta(days=1)
                
                seconds_until = (end_timestamp - timestamp).total_seconds()
                
                if seconds_until < next_change['seconds_until']:
                    next_change['event'] = 'end'
                    next_change['session'] = session
                    next_change['timestamp'] = end_timestamp
                    next_change['seconds_until'] = seconds_until
        
        return next_change
    
    def format_session_stats(self, session_stats: Dict[str, Dict[str, float]]) -> str:
        """
        Format session statistics for display.
        
        Args:
            session_stats: Dictionary with session statistics
            
        Returns:
            Formatted string for display
        """
        output = "===== FOREX SESSION STATISTICS =====\n\n"
        
        for session, stats in session_stats.items():
            output += f"--- {session} Session ---\n"
            output += f"Trades: {stats['trade_count']}\n"
            
            if stats['trade_count'] > 0:
                output += f"Win Rate: {stats['win_rate']:.2f}%\n"
                output += f"Profit Factor: {stats['profit_factor']:.2f}\n"
                output += f"Total P&L: {stats['total_pnl']:.2f}\n"
                
                if 'total_pips' in stats:
                    output += f"Total Pips: {stats['total_pips']:.2f}\n"
                    output += f"Avg Pips/Trade: {stats['avg_pips']:.2f}\n"
                
                output += f"Avg Win: {stats['avg_win']:.2f}\n"
                output += f"Avg Loss: {stats['avg_loss']:.2f}\n"
            
            output += "\n"
        
        return output


# Module execution
if __name__ == "__main__":
    import argparse
    import json
    from tabulate import tabulate
    
    parser = argparse.ArgumentParser(description="Forex Session Manager")
    
    parser.add_argument(
        "--current", 
        action="store_true",
        help="Show current active sessions"
    )
    
    parser.add_argument(
        "--stats", 
        type=str,
        help="Calculate session statistics from trade CSV file"
    )
    
    parser.add_argument(
        "--next-change", 
        action="store_true",
        help="Show next session transition"
    )
    
    parser.add_argument(
        "--pair", 
        type=str,
        help="Forex pair for optimized windows (e.g., EURUSD)"
    )
    
    args = parser.parse_args()
    
    session_manager = ForexSessionManager()
    
    if args.current:
        now = datetime.datetime.now(pytz.UTC)
        active_sessions = session_manager.get_active_sessions(now)
        active_groups = session_manager.get_active_session_groups(now)
        overlap_info = session_manager.get_session_overlap_info(now)
        
        print(f"Current time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active Sessions: {', '.join(active_sessions) if active_sessions else 'None'}")
        print(f"Active Groups: {', '.join(active_groups) if active_groups else 'None'}")
        print(f"Session Overlap: {'Yes - ' + overlap_info['description'] if overlap_info['is_overlap'] else 'No'}")
        
        if args.pair:
            is_optimal = session_manager.is_optimal_trading_window(now, args.pair)
            print(f"Optimal Trading Window for {args.pair}: {'Yes' if is_optimal else 'No'}")
    
    if args.next_change:
        now = datetime.datetime.now(pytz.UTC)
        next_change = session_manager.get_next_session_change(now)
        
        if next_change['event']:
            time_until = datetime.timedelta(seconds=next_change['seconds_until'])
            print(f"Next change: {next_change['session']} session {next_change['event']}")
            print(f"Time: {next_change['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Time until: {time_until}")
        else:
            print("No upcoming session changes found")
    
    if args.stats and args.stats.endswith('.csv'):
        try:
            trades_df = pd.read_csv(args.stats, parse_dates=['entry_time', 'exit_time'])
            trades_df.set_index('entry_time', inplace=True)
            
            session_stats = session_manager.get_session_statistics(trades_df)
            
            # Create tabular output
            table_data = []
            headers = ['Session', 'Trades', 'Win Rate', 'Profit Factor', 'Total P&L']
            
            if 'total_pips' in list(session_stats.values())[0]:
                headers.extend(['Total Pips', 'Avg Pips'])
            
            for session, stats in session_stats.items():
                row = [
                    session,
                    stats['trade_count'],
                    f"{stats['win_rate']:.2f}%" if stats['trade_count'] > 0 else "N/A",
                    f"{stats['profit_factor']:.2f}" if stats['trade_count'] > 0 else "N/A",
                    f"{stats['total_pnl']:.2f}" if stats['trade_count'] > 0 else "N/A"
                ]
                
                if 'total_pips' in stats:
                    row.extend([
                        f"{stats['total_pips']:.2f}" if stats['trade_count'] > 0 else "N/A",
                        f"{stats['avg_pips']:.2f}" if stats['trade_count'] > 0 else "N/A"
                    ])
                
                table_data.append(row)
            
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            
        except Exception as e:
            print(f"Error processing trades file: {e}")
            
    # If no arguments provided, show available sessions
    if not (args.current or args.next_change or args.stats):
        print("Available Forex Sessions:")
        for session, times in session_manager.sessions.items():
            print(f"- {session}: {times['start']} - {times['end']} UTC")
            
        print("\nSession Groups:")
        for group, sessions in session_manager.SESSION_GROUPS.items():
            print(f"- {group}: {', '.join(sessions)}")
            
        print("\nKey Trading Windows:")
        for name, window in session_manager.KEY_OVERLAPS.items():
            print(f"- {name}: {window['start']} - {window['end']} UTC ({window['description']})")
            
        if args.pair:
            pair = args.pair.upper()
            windows = session_manager.get_session_windows(pair)
            print(f"\nTrading Windows for {pair}:")
            for name, window in windows.items():
                if isinstance(window['start'], datetime.time):
                    start_str = window['start'].strftime('%H:%M')
                    end_str = window['end'].strftime('%H:%M')
                else:
                    start_str = window['start']
                    end_str = window['end']
                print(f"- {name}: {start_str} - {end_str} UTC ({window['description']})")
