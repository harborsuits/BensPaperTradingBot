#!/usr/bin/env python3
"""
Session Performance Database

Core component of the SessionPerformanceTracker that stores and retrieves
session-specific performance metrics for trading strategies.
"""

import sqlite3
import os
import json
import logging
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('session_performance_db')


class SessionPerformanceDB:
    """
    Database for storing and retrieving session-specific performance metrics.
    
    This is a core component of the SessionPerformanceTracker that handles:
    - Storing strategy performance by session
    - Retrieving session-specific metrics
    - Finding optimal sessions for strategies
    """
    
    def __init__(self, db_path: str = 'session_performance.db'):
        """
        Initialize the session performance database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
        logger.info(f"Session Performance DB initialized at {db_path}")
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create session performance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                session TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pips REAL NOT NULL,
                profit_factor REAL NOT NULL,
                avg_pips_per_trade REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL,
                last_updated TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(strategy_id, session)
            )
            ''')
            
            # Create strategy metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_metadata (
                strategy_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT,
                optimal_session TEXT,
                session_scores TEXT,
                creation_date TEXT,
                last_updated TEXT NOT NULL,
                params TEXT,
                tags TEXT,
                is_active BOOLEAN DEFAULT 1
            )
            ''')
            
            conn.commit()
            logger.info("Database tables created/verified")
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            
        finally:
            if conn:
                conn.close()
    
    def add_strategy(self, 
                    strategy_id: str, 
                    strategy_name: str, 
                    strategy_type: Optional[str] = None,
                    params: Optional[Dict[str, Any]] = None,
                    tags: Optional[List[str]] = None) -> bool:
        """
        Add a new strategy to the database.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_name: Name of the strategy
            strategy_type: Type of strategy (optional)
            params: Strategy parameters (optional)
            tags: Strategy tags (optional)
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.datetime.now().isoformat()
            
            # Serialize params and tags
            params_json = json.dumps(params) if params else None
            tags_json = json.dumps(tags) if tags else None
            
            # Check if strategy already exists
            cursor.execute(
                "SELECT strategy_id FROM strategy_metadata WHERE strategy_id = ?", 
                (strategy_id,)
            )
            exists = cursor.fetchone()
            
            if exists:
                # Update existing strategy
                cursor.execute('''
                UPDATE strategy_metadata
                SET strategy_name = ?, strategy_type = ?, last_updated = ?,
                    params = ?, tags = ?
                WHERE strategy_id = ?
                ''', (
                    strategy_name, strategy_type, now,
                    params_json, tags_json, strategy_id
                ))
            else:
                # Insert new strategy
                cursor.execute('''
                INSERT INTO strategy_metadata
                (strategy_id, strategy_name, strategy_type, creation_date, 
                 last_updated, params, tags, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ''', (
                    strategy_id, strategy_name, strategy_type, now,
                    now, params_json, tags_json
                ))
            
            conn.commit()
            logger.info(f"{'Updated' if exists else 'Added'} strategy: {strategy_name} ({strategy_id})")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error adding strategy: {e}")
            return False
            
        finally:
            if conn:
                conn.close()
    
    def update_session_performance(self,
                                  strategy_id: str,
                                  strategy_name: str,
                                  session: str,
                                  metrics: Dict[str, Any]) -> bool:
        """
        Update session-specific performance metrics for a strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_name: Name of the strategy
            session: Trading session name (e.g., 'London', 'NewYork', 'Asia')
            metrics: Dictionary of performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.datetime.now().isoformat()
            
            # Extract required metrics
            total_trades = metrics.get('total_trades', 0)
            winning_trades = metrics.get('winning_trades', 0)
            losing_trades = metrics.get('losing_trades', 0)
            win_rate = metrics.get('win_rate', 0.0)
            total_pips = metrics.get('total_pips', 0.0)
            profit_factor = metrics.get('profit_factor', 0.0)
            avg_pips_per_trade = metrics.get('avg_pips_per_trade', 0.0)
            max_drawdown = metrics.get('max_drawdown', 0.0)
            sharpe_ratio = metrics.get('sharpe_ratio', None)
            
            # Additional metadata as JSON
            metadata = {k: v for k, v in metrics.items() if k not in [
                'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
                'total_pips', 'profit_factor', 'avg_pips_per_trade',
                'max_drawdown', 'sharpe_ratio'
            ]}
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Check if record exists
            cursor.execute(
                "SELECT id FROM session_performance WHERE strategy_id = ? AND session = ?", 
                (strategy_id, session)
            )
            exists = cursor.fetchone()
            
            if exists:
                # Update existing record
                cursor.execute('''
                UPDATE session_performance
                SET strategy_name = ?, total_trades = ?, winning_trades = ?,
                    losing_trades = ?, win_rate = ?, total_pips = ?,
                    profit_factor = ?, avg_pips_per_trade = ?, max_drawdown = ?,
                    sharpe_ratio = ?, last_updated = ?, metadata = ?
                WHERE strategy_id = ? AND session = ?
                ''', (
                    strategy_name, total_trades, winning_trades, losing_trades,
                    win_rate, total_pips, profit_factor, avg_pips_per_trade,
                    max_drawdown, sharpe_ratio, now, metadata_json,
                    strategy_id, session
                ))
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO session_performance
                (strategy_id, strategy_name, session, total_trades, winning_trades,
                 losing_trades, win_rate, total_pips, profit_factor, 
                 avg_pips_per_trade, max_drawdown, sharpe_ratio, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy_id, strategy_name, session, total_trades, winning_trades,
                    losing_trades, win_rate, total_pips, profit_factor,
                    avg_pips_per_trade, max_drawdown, sharpe_ratio, now, metadata_json
                ))
            
            # Make sure strategy exists in metadata table
            self.add_strategy(strategy_id, strategy_name)
            
            # Update optimal session in metadata
            self._update_optimal_session(strategy_id, conn)
            
            conn.commit()
            logger.info(f"Updated {session} session performance for {strategy_name}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error updating session performance: {e}")
            return False
            
        finally:
            if conn:
                conn.close()
    
    def _update_optimal_session(self, strategy_id: str, conn: sqlite3.Connection) -> None:
        """
        Update the optimal session for a strategy based on performance metrics.
        
        Args:
            strategy_id: Strategy ID
            conn: Database connection
        """
        try:
            cursor = conn.cursor()
            
            # Get session performance metrics
            cursor.execute('''
            SELECT session, win_rate, profit_factor, total_trades
            FROM session_performance
            WHERE strategy_id = ? AND total_trades >= 20
            ''', (strategy_id,))
            
            sessions = cursor.fetchall()
            
            if not sessions:
                return
            
            # Calculate session scores
            session_scores = {}
            for session, win_rate, profit_factor, total_trades in sessions:
                # Simple scoring formula: (win_rate * profit_factor) adjusted by trade count
                trade_weight = min(1.0, total_trades / 100.0)  # Cap at 100 trades
                score = (win_rate * profit_factor) * trade_weight
                session_scores[session] = score
            
            # Find optimal session
            if session_scores:
                optimal_session = max(session_scores, key=session_scores.get)
                
                # Update strategy metadata
                cursor.execute('''
                UPDATE strategy_metadata
                SET optimal_session = ?, session_scores = ?
                WHERE strategy_id = ?
                ''', (optimal_session, json.dumps(session_scores), strategy_id))
                
                logger.info(f"Updated optimal session for {strategy_id} to {optimal_session}")
            
        except sqlite3.Error as e:
            logger.error(f"Error updating optimal session: {e}")
    
    def get_session_performance(self, 
                              strategy_id: str, 
                              session: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get session performance for a strategy.
        
        Args:
            strategy_id: Strategy ID
            session: Specific session to get (optional, if None returns all sessions)
            
        Returns:
            List of performance metric dictionaries
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            # Query with or without session filter
            if session:
                cursor.execute('''
                SELECT * FROM session_performance
                WHERE strategy_id = ? AND session = ?
                ''', (strategy_id, session))
            else:
                cursor.execute('''
                SELECT * FROM session_performance
                WHERE strategy_id = ?
                ORDER BY session
                ''', (strategy_id,))
            
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            result = []
            for row in rows:
                data = dict(row)
                
                # Parse metadata JSON
                if data.get('metadata'):
                    try:
                        data['metadata'] = json.loads(data['metadata'])
                    except:
                        data['metadata'] = {}
                
                result.append(data)
            
            return result
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting session performance: {e}")
            return []
            
        finally:
            if conn:
                conn.close()
    
    def get_strategy_metadata(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Strategy metadata dictionary or None if not found
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM strategy_metadata
            WHERE strategy_id = ?
            ''', (strategy_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
                
            # Convert to dictionary
            data = dict(row)
            
            # Parse JSON fields
            for field in ['params', 'tags', 'session_scores']:
                if data.get(field):
                    try:
                        data[field] = json.loads(data[field])
                    except:
                        data[field] = None
            
            return data
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting strategy metadata: {e}")
            return None
            
        finally:
            if conn:
                conn.close()
    
    def get_all_strategies(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all strategies.
        
        Args:
            active_only: If True, return only active strategies
            
        Returns:
            List of strategy metadata dictionaries
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query with or without active filter
            if active_only:
                cursor.execute('''
                SELECT * FROM strategy_metadata
                WHERE is_active = 1
                ORDER BY strategy_name
                ''')
            else:
                cursor.execute('''
                SELECT * FROM strategy_metadata
                ORDER BY strategy_name
                ''')
            
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            result = []
            for row in rows:
                data = dict(row)
                
                # Parse JSON fields
                for field in ['params', 'tags', 'session_scores']:
                    if data.get(field):
                        try:
                            data[field] = json.loads(data[field])
                        except:
                            data[field] = None
                
                result.append(data)
            
            return result
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting all strategies: {e}")
            return []
            
        finally:
            if conn:
                conn.close()
    
    def get_strategies_by_session(self, session: str) -> List[Dict[str, Any]]:
        """
        Get strategies that perform well in a specific session.
        
        Args:
            session: Session name
            
        Returns:
            List of strategy metadata dictionaries, sorted by performance
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Join metadata with performance for the specified session
            cursor.execute('''
            SELECT sm.*, sp.win_rate, sp.profit_factor, sp.total_pips, 
                   sp.avg_pips_per_trade, sp.total_trades
            FROM strategy_metadata sm
            JOIN session_performance sp ON sm.strategy_id = sp.strategy_id
            WHERE sm.is_active = 1 AND sp.session = ?
            ORDER BY sp.profit_factor * sp.win_rate DESC
            ''', (session,))
            
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            result = []
            for row in rows:
                data = dict(row)
                
                # Parse JSON fields
                for field in ['params', 'tags', 'session_scores']:
                    if data.get(field):
                        try:
                            data[field] = json.loads(data[field])
                        except:
                            data[field] = None
                
                result.append(data)
            
            return result
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting strategies by session: {e}")
            return []
            
        finally:
            if conn:
                conn.close()
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Deactivate a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE strategy_metadata
            SET is_active = 0, last_updated = ?
            WHERE strategy_id = ?
            ''', (datetime.datetime.now().isoformat(), strategy_id))
            
            conn.commit()
            rows_affected = cursor.rowcount
            
            if rows_affected > 0:
                logger.info(f"Deactivated strategy: {strategy_id}")
                return True
            else:
                logger.warning(f"Strategy not found for deactivation: {strategy_id}")
                return False
            
        except sqlite3.Error as e:
            logger.error(f"Database error deactivating strategy: {e}")
            return False
            
        finally:
            if conn:
                conn.close()
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of performance by session across all strategies.
        
        Returns:
            Dictionary with session summaries
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get summary by session
            cursor.execute('''
            SELECT session, 
                   COUNT(*) as strategy_count,
                   AVG(win_rate) as avg_win_rate,
                   AVG(profit_factor) as avg_profit_factor,
                   AVG(total_pips) as avg_total_pips,
                   AVG(avg_pips_per_trade) as avg_pips_per_trade,
                   SUM(total_trades) as total_trades
            FROM session_performance
            GROUP BY session
            ''')
            
            rows = cursor.fetchall()
            
            # Convert to dictionary by session
            result = {}
            for row in rows:
                data = dict(row)
                session = data.pop('session')
                result[session] = data
            
            return result
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting performance summary: {e}")
            return {}
            
        finally:
            if conn:
                conn.close()


# Module execution
if __name__ == "__main__":
    import argparse
    from tabulate import tabulate
    
    parser = argparse.ArgumentParser(description="Session Performance Database")
    
    parser.add_argument(
        "--db", 
        type=str,
        default="session_performance.db",
        help="Database file path"
    )
    
    parser.add_argument(
        "--list-strategies", 
        action="store_true",
        help="List all strategies"
    )
    
    parser.add_argument(
        "--strategy", 
        type=str,
        help="Show details for a specific strategy"
    )
    
    parser.add_argument(
        "--session", 
        type=str,
        help="Filter by session or show strategies for a session"
    )
    
    parser.add_argument(
        "--summary", 
        action="store_true",
        help="Show performance summary by session"
    )
    
    args = parser.parse_args()
    
    # Initialize database
    db = SessionPerformanceDB(args.db)
    
    # Show strategy details
    if args.strategy:
        metadata = db.get_strategy_metadata(args.strategy)
        
        if metadata:
            print(f"\nStrategy: {metadata['strategy_name']} ({metadata['strategy_id']})")
            print(f"Type: {metadata.get('strategy_type', 'Unknown')}")
            print(f"Optimal Session: {metadata.get('optimal_session', 'Unknown')}")
            print(f"Active: {'Yes' if metadata.get('is_active') else 'No'}")
            print(f"Created: {metadata.get('creation_date', 'Unknown')}")
            print(f"Last Updated: {metadata.get('last_updated', 'Unknown')}")
            
            # Show session scores if available
            if metadata.get('session_scores'):
                print("\nSession Scores:")
                for session, score in metadata['session_scores'].items():
                    print(f"  {session}: {score:.4f}")
            
            # Show session performance
            performances = db.get_session_performance(args.strategy)
            
            if performances:
                print("\nSession Performance:")
                table_data = []
                
                for perf in performances:
                    table_data.append([
                        perf['session'],
                        perf['total_trades'],
                        f"{perf['win_rate']:.2%}",
                        f"{perf['profit_factor']:.2f}",
                        f"{perf['total_pips']:.1f}",
                        f"{perf['avg_pips_per_trade']:.1f}",
                        f"{perf['max_drawdown']:.2%}" if perf['max_drawdown'] else "N/A"
                    ])
                
                print(tabulate(
                    table_data,
                    headers=["Session", "Trades", "Win Rate", "Profit Factor", 
                             "Total Pips", "Avg Pips", "Max DD"],
                    tablefmt="grid"
                ))
            else:
                print("\nNo performance data available for this strategy")
        else:
            print(f"Strategy not found: {args.strategy}")
    
    # List all strategies
    elif args.list_strategies:
        # If session specified, list strategies for that session
        if args.session:
            strategies = db.get_strategies_by_session(args.session)
            print(f"\nStrategies optimized for {args.session} session:")
        else:
            strategies = db.get_all_strategies()
            print("\nAll active strategies:")
        
        if strategies:
            table_data = []
            
            for strat in strategies:
                table_data.append([
                    strat['strategy_id'],
                    strat['strategy_name'],
                    strat.get('strategy_type', 'Unknown'),
                    strat.get('optimal_session', 'Unknown'),
                    strat.get('win_rate', 'N/A') if 'win_rate' in strat else 'N/A',
                    strat.get('profit_factor', 'N/A') if 'profit_factor' in strat else 'N/A',
                    strat.get('total_trades', 'N/A') if 'total_trades' in strat else 'N/A'
                ])
            
            headers = ["ID", "Name", "Type", "Optimal Session"]
            if args.session:
                headers.extend(["Win Rate", "Profit Factor", "Trades"])
                
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print("No strategies found")
    
    # Show performance summary
    elif args.summary:
        summary = db.get_performance_summary()
        
        if summary:
            print("\nPerformance Summary by Session:")
            table_data = []
            
            for session, metrics in summary.items():
                table_data.append([
                    session,
                    metrics['strategy_count'],
                    metrics['total_trades'],
                    f"{metrics['avg_win_rate']:.2%}",
                    f"{metrics['avg_profit_factor']:.2f}",
                    f"{metrics['avg_total_pips']:.1f}",
                    f"{metrics['avg_pips_per_trade']:.1f}"
                ])
            
            print(tabulate(
                table_data,
                headers=["Session", "Strategies", "Total Trades", "Avg Win Rate", 
                         "Avg PF", "Avg Total Pips", "Avg Pips/Trade"],
                tablefmt="grid"
            ))
        else:
            print("No performance data available")
    
    # Default output
    if not (args.strategy or args.list_strategies or args.summary):
        print("Session Performance Database")
        print("==========================")
        print("This tool manages session-specific performance metrics for trading strategies.")
        print("\nUse --help to see available commands")
