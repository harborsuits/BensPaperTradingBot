"""
Database models for the trading system

This module defines the database models for persistent storage of strategy allocations,
performance metrics, and rotation history.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

class AllocationDatabase:
    """Database for storing strategy allocations and rotation history"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the allocation database
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            # Default location is in the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "strategy_allocations.db")
            
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for current allocations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS current_allocations (
            strategy TEXT PRIMARY KEY,
            allocation REAL,
            updated_at TEXT
        )
        ''')
        
        # Create table for allocation history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS allocation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            market_regime TEXT,
            allocations TEXT,
            reasoning TEXT
        )
        ''')
        
        # Create table for performance metrics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            strategy TEXT,
            timestamp TEXT,
            returns REAL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            volatility REAL,
            win_rate REAL,
            profit_factor REAL,
            PRIMARY KEY (strategy, timestamp)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_current_allocations(self, allocations: Dict[str, float]) -> bool:
        """
        Save current strategy allocations to the database
        
        Args:
            allocations: Dictionary mapping strategy names to allocation percentages
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First delete existing allocations
            cursor.execute("DELETE FROM current_allocations")
            
            # Insert new allocations
            now = datetime.now().isoformat()
            for strategy, allocation in allocations.items():
                cursor.execute(
                    "INSERT INTO current_allocations (strategy, allocation, updated_at) VALUES (?, ?, ?)",
                    (strategy, allocation, now)
                )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving allocations: {str(e)}")
            return False
    
    def get_current_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations from the database
        
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT strategy, allocation FROM current_allocations")
            results = cursor.fetchall()
            
            conn.close()
            
            return {row[0]: row[1] for row in results}
            
        except Exception as e:
            print(f"Error retrieving allocations: {str(e)}")
            return {}
    
    def add_allocation_history(self, 
                              market_regime: str,
                              allocations: Dict[str, float],
                              reasoning: str = "") -> bool:
        """
        Add an entry to the allocation history
        
        Args:
            market_regime: Current market regime
            allocations: New strategy allocations
            reasoning: Reasoning for the allocation change
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            allocations_json = json.dumps(allocations)
            
            cursor.execute(
                "INSERT INTO allocation_history (timestamp, market_regime, allocations, reasoning) VALUES (?, ?, ?, ?)",
                (now, market_regime, allocations_json, reasoning)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving allocation history: {str(e)}")
            return False
    
    def get_allocation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent allocation history
        
        Args:
            limit: Maximum number of history entries to retrieve
            
        Returns:
            List of allocation history entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM allocation_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                entry = dict(row)
                entry['allocations'] = json.loads(entry['allocations'])
                history.append(entry)
                
            return history
            
        except Exception as e:
            print(f"Error retrieving allocation history: {str(e)}")
            return []
    
    def save_performance_metrics(self, 
                               strategy: str,
                               metrics: Dict[str, float]) -> bool:
        """
        Save performance metrics for a strategy
        
        Args:
            strategy: Strategy name
            metrics: Dictionary of performance metrics
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute(
                """
                INSERT INTO performance_metrics 
                (strategy, timestamp, returns, sharpe_ratio, sortino_ratio, 
                max_drawdown, volatility, win_rate, profit_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy, 
                    now,
                    metrics.get('returns', 0.0),
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('sortino_ratio', 0.0),
                    metrics.get('max_drawdown', 0.0),
                    metrics.get('volatility', 0.0),
                    metrics.get('win_rate', 0.0),
                    metrics.get('profit_factor', 1.0)
                )
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving performance metrics: {str(e)}")
            return False
    
    def get_latest_performance_metrics(self, strategy: str) -> Dict[str, float]:
        """
        Get the latest performance metrics for a strategy
        
        Args:
            strategy: Strategy name
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM performance_metrics WHERE strategy = ? ORDER BY timestamp DESC LIMIT 1",
                (strategy,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                metrics = dict(result)
                return {
                    'returns': metrics.get('returns', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'sortino_ratio': metrics.get('sortino_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'volatility': metrics.get('volatility', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'profit_factor': metrics.get('profit_factor', 1.0)
                }
            
            return {}
            
        except Exception as e:
            print(f"Error retrieving performance metrics: {str(e)}")
            return {} 