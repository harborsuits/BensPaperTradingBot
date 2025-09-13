#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Database for BensBot Trading System

A lightweight, persistent database for tracking trading strategy performance,
trade history, and context-related performance data. Uses SQLite for simplicity
and portability with no server dependencies.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "performance")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "performance.db")

class PerformanceDB:
    """
    Lightweight database for tracking trading performance.
    Uses SQLite for simplicity with no server dependencies.
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the performance database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.connect()
        self.create_tables()
        
        logger.info(f"Performance database initialized at {db_path}")
    
    def connect(self) -> bool:
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Configure to return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """Create database tables if they don't exist."""
        try:
            # Daily trades table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                success INTEGER,
                account_type TEXT NOT NULL,
                trade_context TEXT,
                market_regime TEXT,
                notes TEXT
            )
            ''')
            
            # Strategy stats table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_stats (
                strategy_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                avg_win REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                sharpe_ratio REAL,
                win_rate REAL DEFAULT 0.0,
                asset_class TEXT,
                strategy_type TEXT,
                timeframe TEXT,
                config_json TEXT
            )
            ''')
            
            # Context performance table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_type TEXT NOT NULL,
                context_value TEXT NOT NULL,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                UNIQUE(context_type, context_value)
            )
            ''')
            
            # Daily summary table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                profitable_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                best_strategy TEXT,
                worst_strategy TEXT,
                best_pnl REAL DEFAULT 0.0,
                worst_pnl REAL DEFAULT 0.0,
                market_regime TEXT,
                notes TEXT
            )
            ''')
            
            # Commit changes
            self.conn.commit()
            
            logger.info("Database tables created/verified")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            self.conn.rollback()
            return False
    
    def add_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Add a new trade to the database.
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            Inserted row ID or -1 on failure
        """
        try:
            # Ensure required fields are present
            required_fields = ['date', 'symbol', 'strategy_id', 'entry_time', 
                               'entry_price', 'quantity', 'account_type']
            
            for field in required_fields:
                if field not in trade_data:
                    logger.error(f"Missing required field: {field}")
                    return -1
            
            # Convert trade context to JSON if it's a dictionary
            if 'trade_context' in trade_data and isinstance(trade_data['trade_context'], dict):
                trade_data['trade_context'] = json.dumps(trade_data['trade_context'])
            
            # Convert success to integer (0/1) if it's boolean
            if 'success' in trade_data and isinstance(trade_data['success'], bool):
                trade_data['success'] = 1 if trade_data['success'] else 0
            
            # Prepare SQL query
            columns = ', '.join(trade_data.keys())
            placeholders = ', '.join(['?'] * len(trade_data))
            
            sql = f"INSERT INTO daily_trades ({columns}) VALUES ({placeholders})"
            
            # Execute query
            self.cursor.execute(sql, list(trade_data.values()))
            self.conn.commit()
            
            # If this is a completed trade (has exit info), update strategy stats
            if 'exit_price' in trade_data and trade_data['exit_price'] is not None:
                self.update_strategy_stats(trade_data)
                
                # Also update context performance if context is provided
                if 'market_regime' in trade_data and trade_data['market_regime']:
                    self.update_context_performance('market_regime', trade_data['market_regime'], 
                                                   trade_data.get('success', None), 
                                                   trade_data.get('pnl', 0.0))
            
            return self.cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to add trade: {str(e)}")
            self.conn.rollback()
            return -1
    
    def update_trade(self, trade_id: int, exit_data: Dict[str, Any]) -> bool:
        """
        Update a trade with exit information.
        
        Args:
            trade_id: ID of the trade to update
            exit_data: Dictionary containing exit information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current trade data
            self.cursor.execute("SELECT * FROM daily_trades WHERE id = ?", (trade_id,))
            trade = dict(self.cursor.fetchone())
            
            # Update with exit information
            for key, value in exit_data.items():
                trade[key] = value
            
            # Build update SQL
            update_fields = [f"{key} = ?" for key in exit_data.keys()]
            sql = f"UPDATE daily_trades SET {', '.join(update_fields)} WHERE id = ?"
            
            # Execute update
            values = list(exit_data.values()) + [trade_id]
            self.cursor.execute(sql, values)
            self.conn.commit()
            
            # Update strategy stats with the complete trade info
            self.update_strategy_stats(trade)
            
            # Update context performance if applicable
            if 'market_regime' in trade and trade['market_regime']:
                self.update_context_performance('market_regime', trade['market_regime'], 
                                               trade.get('success', None), 
                                               trade.get('pnl', 0.0))
            
            return True
        except Exception as e:
            logger.error(f"Failed to update trade: {str(e)}")
            self.conn.rollback()
            return False
    
    def update_strategy_stats(self, trade_data: Dict[str, Any]) -> bool:
        """
        Update strategy statistics based on a completed trade.
        
        Args:
            trade_data: Complete trade data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            strategy_id = trade_data['strategy_id']
            
            # Check if strategy exists
            self.cursor.execute("SELECT * FROM strategy_stats WHERE strategy_id = ?", (strategy_id,))
            strategy = self.cursor.fetchone()
            
            # Get trade success and PnL
            success = trade_data.get('success')
            pnl = trade_data.get('pnl', 0.0)
            
            # Set current time
            now = datetime.now().isoformat()
            
            if strategy:
                # Strategy exists, update stats
                strategy_dict = dict(strategy)
                
                # Update win/loss counters
                if success == 1:
                    win_count = strategy_dict['win_count'] + 1
                    loss_count = strategy_dict['loss_count']
                    
                    # Update average win
                    avg_win = ((strategy_dict['avg_win'] * strategy_dict['win_count']) + pnl) / win_count
                    avg_loss = strategy_dict['avg_loss']
                elif success == 0:
                    win_count = strategy_dict['win_count']
                    loss_count = strategy_dict['loss_count'] + 1
                    
                    # Update average loss
                    avg_win = strategy_dict['avg_win']
                    avg_loss = ((strategy_dict['avg_loss'] * strategy_dict['loss_count']) + pnl) / loss_count
                else:
                    win_count = strategy_dict['win_count']
                    loss_count = strategy_dict['loss_count']
                    avg_win = strategy_dict['avg_win']
                    avg_loss = strategy_dict['avg_loss']
                
                # Calculate win rate
                total_trades = win_count + loss_count
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
                
                # Update strategy stats
                self.cursor.execute("""
                UPDATE strategy_stats SET 
                    win_count = ?, 
                    loss_count = ?, 
                    total_pnl = total_pnl + ?,
                    last_updated = ?,
                    avg_win = ?,
                    avg_loss = ?,
                    win_rate = ?
                WHERE strategy_id = ?
                """, (win_count, loss_count, pnl, now, avg_win, avg_loss, win_rate, strategy_id))
            else:
                # Strategy doesn't exist, create new entry
                strategy_name = trade_data.get('strategy_name', strategy_id)
                
                if success == 1:
                    win_count = 1
                    loss_count = 0
                    avg_win = pnl
                    avg_loss = 0.0
                elif success == 0:
                    win_count = 0
                    loss_count = 1
                    avg_win = 0.0
                    avg_loss = pnl
                else:
                    win_count = 0
                    loss_count = 0
                    avg_win = 0.0
                    avg_loss = 0.0
                
                # Calculate win rate
                total_trades = win_count + loss_count
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
                
                # Insert new strategy stats
                self.cursor.execute("""
                INSERT INTO strategy_stats (
                    strategy_id, strategy_name, win_count, loss_count, 
                    total_pnl, last_updated, avg_win, avg_loss, win_rate,
                    asset_class, strategy_type, timeframe
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id, strategy_name, win_count, loss_count, 
                    pnl, now, avg_win, avg_loss, win_rate,
                    trade_data.get('asset_class'), trade_data.get('strategy_type'),
                    trade_data.get('timeframe')
                ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update strategy stats: {str(e)}")
            self.conn.rollback()
            return False
    
    def update_context_performance(self, context_type: str, context_value: str, 
                                  success: Optional[int], pnl: float) -> bool:
        """
        Update performance statistics for a specific context.
        
        Args:
            context_type: Type of context (e.g., market_regime, time_of_day)
            context_value: Value of the context (e.g., bullish, morning)
            success: Whether the trade was successful (1=win, 0=loss, None=unknown)
            pnl: Profit/loss from the trade
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if context exists
            self.cursor.execute(
                "SELECT * FROM context_performance WHERE context_type = ? AND context_value = ?", 
                (context_type, context_value)
            )
            context = self.cursor.fetchone()
            
            # Set current time
            now = datetime.now().isoformat()
            
            if context:
                # Context exists, update stats
                context_dict = dict(context)
                
                # Update win/loss counters
                if success == 1:
                    win_count = context_dict['win_count'] + 1
                    loss_count = context_dict['loss_count']
                elif success == 0:
                    win_count = context_dict['win_count']
                    loss_count = context_dict['loss_count'] + 1
                else:
                    win_count = context_dict['win_count']
                    loss_count = context_dict['loss_count']
                
                # Calculate win rate
                total_trades = win_count + loss_count
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
                
                # Update context performance
                self.cursor.execute("""
                UPDATE context_performance SET 
                    win_count = ?, 
                    loss_count = ?, 
                    total_pnl = total_pnl + ?,
                    win_rate = ?,
                    last_updated = ?
                WHERE context_type = ? AND context_value = ?
                """, (win_count, loss_count, pnl, win_rate, now, context_type, context_value))
            else:
                # Context doesn't exist, create new entry
                if success == 1:
                    win_count = 1
                    loss_count = 0
                elif success == 0:
                    win_count = 0
                    loss_count = 1
                else:
                    win_count = 0
                    loss_count = 0
                
                # Calculate win rate
                total_trades = win_count + loss_count
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
                
                # Insert new context performance
                self.cursor.execute("""
                INSERT INTO context_performance (
                    context_type, context_value, win_count, loss_count, 
                    total_pnl, win_rate, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    context_type, context_value, win_count, loss_count, 
                    pnl, win_rate, now
                ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update context performance: {str(e)}")
            self.conn.rollback()
            return False
    
    def add_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Add or update a daily trading summary.
        
        Args:
            summary_data: Dictionary containing daily summary information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            date_str = summary_data.get('date')
            if not date_str:
                date_str = datetime.now().strftime('%Y-%m-%d')
                summary_data['date'] = date_str
            
            # Check if summary exists for this date
            self.cursor.execute("SELECT * FROM daily_summary WHERE date = ?", (date_str,))
            existing = self.cursor.fetchone()
            
            if existing:
                # Update existing summary
                update_fields = [f"{key} = ?" for key in summary_data.keys()]
                sql = f"UPDATE daily_summary SET {', '.join(update_fields)} WHERE date = ?"
                
                values = list(summary_data.values()) + [date_str]
                self.cursor.execute(sql, values)
            else:
                # Insert new summary
                columns = ', '.join(summary_data.keys())
                placeholders = ', '.join(['?'] * len(summary_data))
                
                sql = f"INSERT INTO daily_summary ({columns}) VALUES ({placeholders})"
                
                self.cursor.execute(sql, list(summary_data.values()))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add daily summary: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_trades(self, filters: Dict[str, Any] = None, 
                  limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get trades from the database with optional filtering.
        
        Args:
            filters: Dictionary of filters (column=value)
            limit: Maximum number of trades to return
            offset: Offset for pagination
            
        Returns:
            List of trade dictionaries
        """
        try:
            sql = "SELECT * FROM daily_trades"
            params = []
            
            # Add filters if provided
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
            
            # Add limit and offset
            sql += " ORDER BY date DESC, entry_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            self.cursor.execute(sql, params)
            
            # Convert rows to dictionaries
            result = [dict(row) for row in self.cursor.fetchall()]
            
            # Parse JSON fields
            for trade in result:
                if 'trade_context' in trade and trade['trade_context']:
                    try:
                        trade['trade_context'] = json.loads(trade['trade_context'])
                    except:
                        pass
            
            return result
        except Exception as e:
            logger.error(f"Failed to get trades: {str(e)}")
            return []
    
    def get_strategy_stats(self, strategy_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get statistics for a strategy or all strategies.
        
        Args:
            strategy_id: Optional ID of a specific strategy
            
        Returns:
            Dictionary of strategy stats or list of all strategy stats
        """
        try:
            if strategy_id:
                # Get stats for a specific strategy
                self.cursor.execute("SELECT * FROM strategy_stats WHERE strategy_id = ?", (strategy_id,))
                row = self.cursor.fetchone()
                
                if row:
                    result = dict(row)
                    
                    # Parse JSON fields
                    if 'config_json' in result and result['config_json']:
                        try:
                            result['config_json'] = json.loads(result['config_json'])
                        except:
                            pass
                    
                    return result
                else:
                    return {}
            else:
                # Get stats for all strategies
                self.cursor.execute("SELECT * FROM strategy_stats ORDER BY total_pnl DESC")
                
                # Convert rows to dictionaries
                result = [dict(row) for row in self.cursor.fetchall()]
                
                # Parse JSON fields
                for strategy in result:
                    if 'config_json' in strategy and strategy['config_json']:
                        try:
                            strategy['config_json'] = json.loads(strategy['config_json'])
                        except:
                            pass
                
                return result
        except Exception as e:
            logger.error(f"Failed to get strategy stats: {str(e)}")
            return [] if strategy_id is None else {}
    
    def get_context_performance(self, context_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get performance statistics for contexts.
        
        Args:
            context_type: Optional type of context to filter by
            
        Returns:
            List of context performance dictionaries
        """
        try:
            if context_type:
                # Get performance for a specific context type
                self.cursor.execute(
                    "SELECT * FROM context_performance WHERE context_type = ? ORDER BY win_rate DESC, total_pnl DESC", 
                    (context_type,)
                )
            else:
                # Get performance for all contexts
                self.cursor.execute(
                    "SELECT * FROM context_performance ORDER BY context_type, win_rate DESC, total_pnl DESC"
                )
            
            # Convert rows to dictionaries
            result = [dict(row) for row in self.cursor.fetchall()]
            
            return result
        except Exception as e:
            logger.error(f"Failed to get context performance: {str(e)}")
            return []
    
    def get_daily_summary(self, date_str: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get daily summary for a specific date or all dates.
        
        Args:
            date_str: Optional date string in YYYY-MM-DD format
            
        Returns:
            Dictionary of daily summary or list of all daily summaries
        """
        try:
            if date_str:
                # Get summary for a specific date
                self.cursor.execute("SELECT * FROM daily_summary WHERE date = ?", (date_str,))
                row = self.cursor.fetchone()
                
                return dict(row) if row else {}
            else:
                # Get summary for all dates
                self.cursor.execute("SELECT * FROM daily_summary ORDER BY date DESC")
                
                # Convert rows to dictionaries
                result = [dict(row) for row in self.cursor.fetchall()]
                
                return result
        except Exception as e:
            logger.error(f"Failed to get daily summary: {str(e)}")
            return [] if date_str is None else {}
    
    def get_performance_dataframe(self, table: str, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Get a pandas DataFrame from a database table with optional filtering.
        
        Args:
            table: Name of the table to query
            filters: Optional dictionary of filters
            
        Returns:
            Pandas DataFrame with the query results
        """
        try:
            sql = f"SELECT * FROM {table}"
            params = []
            
            # Add filters if provided
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
            
            # Execute query
            return pd.read_sql_query(sql, self.conn, params=params)
        except Exception as e:
            logger.error(f"Failed to get DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# Factory function to get a database connection
def connect_performance_db(db_path: str = DEFAULT_DB_PATH) -> PerformanceDB:
    """
    Connect to the performance database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        PerformanceDB instance
    """
    return PerformanceDB(db_path=db_path)

# For testing and demonstration
if __name__ == "__main__":
    db = connect_performance_db()
    
    # Sample trade data
    sample_trade = {
        'date': date.today().isoformat(),
        'symbol': 'AAPL',
        'strategy_id': 'macd_crossover',
        'strategy_name': 'MACD Crossover Strategy',
        'entry_time': datetime.now().isoformat(),
        'entry_price': 150.25,
        'quantity': 10,
        'account_type': 'paper',
        'market_regime': 'bullish',
        'asset_class': 'stocks',
        'strategy_type': 'momentum',
        'timeframe': '1h'
    }
    
    # Add the trade
    trade_id = db.add_trade(sample_trade)
    print(f"Added trade with ID: {trade_id}")
    
    # Complete the trade with exit information
    exit_data = {
        'exit_time': datetime.now().isoformat(),
        'exit_price': 152.50,
        'pnl': 22.50,  # 10 shares * $2.25 profit per share
        'success': 1
    }
    
    # Update the trade
    db.update_trade(trade_id, exit_data)
    print("Updated trade with exit information")
    
    # Add a daily summary
    summary = {
        'date': date.today().isoformat(),
        'total_trades': 5,
        'profitable_trades': 3,
        'total_pnl': 125.75,
        'best_strategy': 'macd_crossover',
        'worst_strategy': 'rsi_divergence',
        'best_pnl': 85.50,
        'worst_pnl': -35.25,
        'market_regime': 'bullish',
        'notes': 'Good trading day with positive market momentum'
    }
    
    db.add_daily_summary(summary)
    print("Added daily summary")
    
    # Query and display some results
    print("\nStrategy Statistics:")
    stats = db.get_strategy_stats()
    for strategy in stats:
        print(f"- {strategy['strategy_name']}: Win Rate: {strategy['win_rate']}%, PnL: ${strategy['total_pnl']}")
    
    print("\nContext Performance:")
    contexts = db.get_context_performance()
    for context in contexts:
        print(f"- {context['context_type']} - {context['context_value']}: Win Rate: {context['win_rate']}%, PnL: ${context['total_pnl']}")
    
    # Close the connection
    db.close()
