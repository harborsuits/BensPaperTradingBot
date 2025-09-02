"""
Trade Accounting System

This module provides comprehensive trade tracking, P&L calculation, and performance
reporting for the trading system. It interfaces with the PositionManager for accurate
position tracking and reconciliation.

Features:
- Trade execution logging with full details
- Realized and unrealized P&L tracking
- Strategy-level accounting
- Period-based performance metrics
- Tax reporting capabilities
"""

import logging
import os
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid

# Import position manager for position access
from trading_bot.position.position_manager import PositionManager

logger = logging.getLogger(__name__)

class TradeAccounting:
    """
    Comprehensive trade accounting system that tracks all trades, calculates P&L,
    and generates performance reports.
    """
    
    def __init__(self, 
                 database_path: Optional[str] = None, 
                 position_manager: Optional[PositionManager] = None,
                 log_dir: str = "logs"):
        """
        Initialize the trade accounting system.
        
        Args:
            database_path: Path to SQLite database for trade records
            position_manager: Optional position manager for position access
            log_dir: Directory for log files
        """
        # Store configuration
        self.database_path = database_path or os.path.join(log_dir, "trades.db")
        self.position_manager = position_manager
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Cache for performance metrics
        self.performance_cache = {}
        self.last_cache_update = datetime.now() - timedelta(hours=1)  # Force initial update
        self.cache_ttl = 300  # 5 minutes
        
        logger.info(f"Trade Accounting initialized with database at {self.database_path}")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Trades table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    strategy_id TEXT,
                    symbol TEXT NOT NULL,
                    broker_id TEXT,
                    asset_class TEXT,
                    direction TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    commission REAL,
                    slippage REAL,
                    realized_pnl REAL,
                    pnl_percent REAL,
                    status TEXT NOT NULL,
                    exit_reason TEXT,
                    tags TEXT,
                    metadata TEXT
                )
                ''')
                
                # Daily performance table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    starting_balance REAL,
                    ending_balance REAL,
                    net_pnl REAL,
                    trades_closed INTEGER,
                    win_count INTEGER,
                    loss_count INTEGER,
                    total_commissions REAL
                )
                ''')
                
                # Strategy performance table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_id TEXT,
                    date TEXT,
                    trades_closed INTEGER,
                    win_count INTEGER,
                    loss_count INTEGER,
                    net_pnl REAL,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    PRIMARY KEY (strategy_id, date)
                )
                ''')
                
                # Transaction log for audit trail
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS transaction_log (
                    transaction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    trade_id TEXT,
                    action TEXT NOT NULL,
                    description TEXT,
                    data TEXT
                )
                ''')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a trade execution to the database.
        
        Args:
            trade_data: Dict with trade details including:
                - trade_id (optional): Unique ID (generated if not provided)
                - timestamp: Trade timestamp (ISO format)
                - strategy_id: Strategy that generated the trade
                - symbol: Trading symbol
                - broker_id: Broker identifier
                - asset_class: Asset class (e.g., 'forex', 'equity')
                - direction: Trade direction ('long' or 'short')
                - quantity: Trade size
                - entry_price: Entry price
                - exit_price (optional): Exit price if closed
                - entry_time: Entry time (ISO format)
                - exit_time (optional): Exit time if closed
                - commission (optional): Commission paid
                - slippage (optional): Slippage experienced
                - realized_pnl (optional): Realized P&L if closed
                - pnl_percent (optional): P&L as percentage if closed
                - status: Trade status ('open', 'closed', 'cancelled')
                - exit_reason (optional): Reason for exit
                - tags (optional): Tags for filtering
                - metadata (optional): Additional trade metadata
                
        Returns:
            trade_id: ID of the logged trade
        """
        try:
            # Generate ID if not provided
            trade_id = trade_data.get('trade_id', str(uuid.uuid4()))
            trade_data['trade_id'] = trade_id
            
            # Set defaults
            timestamp = trade_data.get('timestamp', datetime.now().isoformat())
            
            # Set status default
            if 'status' not in trade_data:
                trade_data['status'] = 'open' if 'exit_price' not in trade_data else 'closed'
            
            # Convert any dictionaries to JSON strings
            metadata = trade_data.get('metadata')
            if metadata and isinstance(metadata, dict):
                trade_data['metadata'] = json.dumps(metadata)
                
            tags = trade_data.get('tags')
            if tags:
                if isinstance(tags, list):
                    trade_data['tags'] = ','.join(tags)
                elif isinstance(tags, dict):
                    trade_data['tags'] = json.dumps(tags)
            
            # Insert into database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Construct columns and placeholders for dynamic insertion
                columns = []
                placeholders = []
                values = []
                
                for key, value in trade_data.items():
                    if value is not None:
                        columns.append(key)
                        placeholders.append('?')
                        values.append(value)
                
                query = f'''
                INSERT OR REPLACE INTO trades ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                '''
                
                cursor.execute(query, values)
                
                # Log the transaction
                transaction_data = {
                    'timestamp': timestamp,
                    'trade_id': trade_id,
                    'action': 'log_trade',
                    'description': f"Logged trade for {trade_data.get('symbol')}",
                    'data': json.dumps(trade_data)
                }
                
                self._log_transaction(cursor, transaction_data)
                
                conn.commit()
            
            logger.info(f"Logged trade {trade_id} for {trade_data.get('symbol')}")
            
            # If the position manager is available, update position
            if self.position_manager and trade_data.get('status') == 'open':
                try:
                    position_data = {
                        'symbol': trade_data.get('symbol'),
                        'broker_id': trade_data.get('broker_id', 'unknown'),
                        'quantity': trade_data.get('quantity'),
                        'entry_price': trade_data.get('entry_price'),
                        'entry_date': trade_data.get('entry_time'),
                        'direction': trade_data.get('direction'),
                        'strategy_id': trade_data.get('strategy_id'),
                        'status': 'open',
                        'trade_id': trade_id
                    }
                    self.position_manager.add_position(position_data)
                except Exception as e:
                    logger.error(f"Error updating position manager: {str(e)}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            raise
    
    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade record.
        
        Args:
            trade_id: ID of the trade to update
            update_data: Data to update
            
        Returns:
            bool: Success status
        """
        try:
            # Convert any dictionaries to JSON strings
            metadata = update_data.get('metadata')
            if metadata and isinstance(metadata, dict):
                update_data['metadata'] = json.dumps(metadata)
                
            tags = update_data.get('tags')
            if tags:
                if isinstance(tags, list):
                    update_data['tags'] = ','.join(tags)
                elif isinstance(tags, dict):
                    update_data['tags'] = json.dumps(tags)
            
            # Update database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Construct SET clause for dynamic update
                set_clause = []
                values = []
                
                for key, value in update_data.items():
                    set_clause.append(f"{key} = ?")
                    values.append(value)
                
                # Add trade_id to values
                values.append(trade_id)
                
                query = f'''
                UPDATE trades
                SET {', '.join(set_clause)}
                WHERE trade_id = ?
                '''
                
                cursor.execute(query, values)
                
                # Log the transaction
                transaction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'trade_id': trade_id,
                    'action': 'update_trade',
                    'description': f"Updated trade {trade_id}",
                    'data': json.dumps(update_data)
                }
                
                self._log_transaction(cursor, transaction_data)
                
                conn.commit()
                
                # Check if any rows were affected
                if cursor.rowcount == 0:
                    logger.warning(f"Trade {trade_id} not found for update")
                    return False
            
            logger.info(f"Updated trade {trade_id}")
            
            # If this is marking a trade as closed and position manager is available
            if update_data.get('status') == 'closed' and self.position_manager:
                try:
                    # Get trade details
                    trade = self.get_trade(trade_id)
                    if trade:
                        position_id = self.position_manager._generate_position_id(
                            trade.get('symbol', ''),
                            trade.get('broker_id', 'unknown')
                        )
                        
                        close_data = {
                            'price': update_data.get('exit_price'),
                            'timestamp': update_data.get('exit_time', datetime.now().isoformat()),
                            'pnl': update_data.get('realized_pnl'),
                            'pnl_percent': update_data.get('pnl_percent')
                        }
                        
                        self.position_manager.close_position(position_id, close_data)
                except Exception as e:
                    logger.error(f"Error updating position manager: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating trade: {str(e)}")
            raise
    
    def close_trade(self, trade_id: str, exit_data: Dict[str, Any]) -> bool:
        """
        Close an open trade.
        
        Args:
            trade_id: ID of the trade to close
            exit_data: Dict with exit details:
                - exit_price: Exit price
                - exit_time: Exit time (defaults to now)
                - commission: Additional commission (optional)
                - slippage: Slippage experienced (optional)
                - exit_reason: Reason for exit (optional)
                
        Returns:
            bool: Success status
        """
        try:
            # Get the trade details
            trade = self.get_trade(trade_id)
            if not trade:
                logger.error(f"Trade {trade_id} not found")
                return False
            
            if trade.get('status') == 'closed':
                logger.warning(f"Trade {trade_id} is already closed")
                return False
            
            # Calculate realized P&L
            entry_price = trade.get('entry_price', 0)
            exit_price = exit_data.get('exit_price', 0)
            quantity = trade.get('quantity', 0)
            direction = trade.get('direction', 'long').lower()
            
            # Calculate P&L based on direction
            if direction == 'long':
                realized_pnl = (exit_price - entry_price) * quantity
            else:  # short
                realized_pnl = (entry_price - exit_price) * quantity
            
            # Subtract commission if provided
            commission = exit_data.get('commission', 0)
            realized_pnl -= commission
            
            # Calculate P&L percent
            if entry_price > 0:
                pnl_percent = (realized_pnl / (entry_price * quantity)) * 100
            else:
                pnl_percent = 0
            
            # Prepare update data
            update_data = {
                'exit_price': exit_price,
                'exit_time': exit_data.get('exit_time', datetime.now().isoformat()),
                'realized_pnl': realized_pnl,
                'pnl_percent': pnl_percent,
                'status': 'closed',
                'exit_reason': exit_data.get('exit_reason')
            }
            
            # Add commission if provided
            if commission:
                update_data['commission'] = trade.get('commission', 0) + commission
            
            # Add slippage if provided
            if 'slippage' in exit_data:
                update_data['slippage'] = exit_data.get('slippage')
            
            # Update the trade
            success = self.update_trade(trade_id, update_data)
            
            if success:
                # Update daily and strategy performance
                self._update_performance_metrics(trade_id)
                
                logger.info(f"Closed trade {trade_id} with P&L: {realized_pnl:.2f} ({pnl_percent:.2f}%)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            raise
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific trade.
        
        Args:
            trade_id: ID of the trade to retrieve
            
        Returns:
            Dict with trade details or None if not found
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
                row = cursor.fetchone()
                
                if row:
                    trade = dict(row)
                    
                    # Parse JSON fields
                    if trade.get('metadata'):
                        try:
                            trade['metadata'] = json.loads(trade['metadata'])
                        except:
                            pass
                    
                    if trade.get('tags') and ',' in trade['tags']:
                        trade['tags'] = trade['tags'].split(',')
                    
                    return trade
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving trade {trade_id}: {str(e)}")
            return None
    
    def get_trades(self, 
                  status: Optional[str] = None,
                  symbol: Optional[str] = None,
                  strategy_id: Optional[str] = None,
                  broker_id: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trades with optional filtering.
        
        Args:
            status: Filter by status ('open', 'closed')
            symbol: Filter by symbol
            strategy_id: Filter by strategy
            broker_id: Filter by broker
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of trades to return
            
        Returns:
            List of trades matching criteria
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build query with filters
                query = 'SELECT * FROM trades'
                conditions = []
                params = []
                
                if status:
                    conditions.append('status = ?')
                    params.append(status)
                
                if symbol:
                    conditions.append('symbol = ?')
                    params.append(symbol)
                
                if strategy_id:
                    conditions.append('strategy_id = ?')
                    params.append(strategy_id)
                
                if broker_id:
                    conditions.append('broker_id = ?')
                    params.append(broker_id)
                
                if start_date:
                    conditions.append('entry_time >= ?')
                    params.append(start_date)
                
                if end_date:
                    conditions.append('entry_time <= ?')
                    params.append(end_date)
                
                # Add WHERE clause if we have conditions
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                # Add order by and limit
                query += ' ORDER BY entry_time DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trade = dict(row)
                    
                    # Parse JSON fields
                    if trade.get('metadata'):
                        try:
                            trade['metadata'] = json.loads(trade['metadata'])
                        except:
                            pass
                    
                    if trade.get('tags') and ',' in trade['tags']:
                        trade['tags'] = trade['tags'].split(',')
                    
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Error retrieving trades: {str(e)}")
            return []
