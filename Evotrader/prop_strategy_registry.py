#!/usr/bin/env python3
"""
Proprietary Strategy Registry

This module implements a registry system for tracking strategies through their
lifecycle in a proprietary trading firm context. It manages strategy metadata,
status changes, and performance history.
"""

import os
import json
import sqlite3
import datetime
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prop_strategy_registry')


class PropStrategyRegistry:
    """
    Registry for tracking proprietary trading strategies through their lifecycle.
    
    Tracks metadata, performance, and status changes for strategies optimized
    for funded trading account criteria.
    """
    
    # Strategy lifecycle status definitions
    STATUS_EVOLVED = "evolved"           # Initial state after evolution
    STATUS_VALIDATED = "validated"       # Passed backtest validation
    STATUS_FUNDABLE = "fundable"         # Meets all prop firm requirements in backtest
    STATUS_UNDER_TEST = "under_test"     # Currently undergoing forward testing
    STATUS_PAPER_TESTING = "paper_testing" # Paper trading phase
    STATUS_PROMOTED = "promoted"         # Passed forward testing, ready for live
    STATUS_LIVE = "live"                 # Currently trading live
    STATUS_RETIRED = "retired"           # No longer in use
    STATUS_REJECTED = "rejected"         # Failed to meet requirements
    
    def __init__(self, registry_path: str = "./prop_strategy_registry", 
                 risk_profile_path: Optional[str] = None):
        """
        Initialize the strategy registry.
        
        Args:
            registry_path: Directory for registry storage
            risk_profile_path: Path to risk profile YAML
        """
        self.registry_path = registry_path
        self.db_path = os.path.join(registry_path, "registry.db")
        
        # Create registry directory if it doesn't exist
        os.makedirs(registry_path, exist_ok=True)
        
        # Load risk profile if provided
        self.risk_profile = {}
        if risk_profile_path and os.path.exists(risk_profile_path):
            with open(risk_profile_path, 'r') as f:
                self.risk_profile = yaml.safe_load(f)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for strategy storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create strategies table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id TEXT PRIMARY KEY,
            name TEXT,
            strategy_type TEXT,
            status TEXT,
            creation_date TEXT,
            last_updated TEXT,
            parameters TEXT,
            fingerprint TEXT,
            tags TEXT
        )
        ''')
        
        # Create performance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id TEXT PRIMARY KEY,
            strategy_id TEXT,
            scenario TEXT,
            date TEXT,
            metrics TEXT,
            score REAL,
            passes_evaluation INTEGER,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
        ''')
        
        # Create status_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            old_status TEXT,
            new_status TEXT,
            change_date TEXT,
            reason TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
        ''')
        
        # Create parameters_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameters_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            parameters TEXT,
            change_date TEXT,
            reason TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
        ''')
        
        # Create live metrics table for storing real-world performance
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS live_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            sharpe_ratio REAL,
            max_drawdown REAL,
            total_return REAL,
            win_rate REAL,
            trade_count INTEGER,
            avg_trade_duration TEXT,
            last_updated TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
        ''')
        
        # Create performance delta table for tracking backtest vs live differences
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_delta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            sharpe_delta REAL,
            drawdown_delta REAL,
            return_delta REAL,
            win_rate_delta REAL,
            confidence_score REAL,
            recommendation TEXT,
            calculation_date TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_strategy_id(self, strategy_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a strategy.
        
        Args:
            strategy_type: Type of strategy
            parameters: Strategy parameters
            
        Returns:
            Unique ID string
        """
        # Convert parameters to sorted string representation
        param_str = json.dumps(parameters, sort_keys=True)
        
        # Create hash from strategy type and parameters
        hash_input = f"{strategy_type}:{param_str}"
        strategy_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Use timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        return f"{strategy_type}_{timestamp}_{strategy_hash[:8]}"
    
    def _generate_fingerprint(self, strategy_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a fingerprint for strategy similarity comparison.
        
        Args:
            strategy_type: Type of strategy
            parameters: Strategy parameters
            
        Returns:
            Strategy fingerprint
        """
        # For numeric parameters, round to reduce sensitivity to small changes
        processed_params = {}
        
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Round numbers to reduce sensitivity
                processed_params[key] = round(value, 2)
            else:
                processed_params[key] = value
        
        # Convert to string and hash
        param_str = json.dumps(processed_params, sort_keys=True)
        fingerprint = hashlib.sha256(f"{strategy_type}:{param_str}".encode()).hexdigest()
        
        return fingerprint
    
    def register_strategy(self, 
                         strategy_type: str,
                         parameters: Dict[str, Any],
                         name: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> str:
        """
        Register a new strategy in the registry.
        
        Args:
            strategy_type: Type of strategy
            parameters: Strategy parameters
            name: Optional name for the strategy
            tags: Optional tags for categorization
            
        Returns:
            Strategy ID
        """
        # Generate ID and fingerprint
        strategy_id = self._generate_strategy_id(strategy_type, parameters)
        fingerprint = self._generate_fingerprint(strategy_type, parameters)
        
        # Auto-generate name if not provided
        if name is None:
            name = f"{strategy_type}_{datetime.datetime.now().strftime('%Y%m%d')}"
        
        # Default to empty list for tags
        if tags is None:
            tags = []
        
        # Current date and time
        now = datetime.datetime.now().isoformat()
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert strategy
        cursor.execute('''
        INSERT INTO strategies 
        (id, name, strategy_type, status, creation_date, last_updated, parameters, fingerprint, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id,
            name,
            strategy_type,
            self.STATUS_EVOLVED,  # Initial status
            now,
            now,
            json.dumps(parameters),
            fingerprint,
            json.dumps(tags)
        ))
        
        # Record initial parameters history
        cursor.execute('''
        INSERT INTO parameters_history
        (strategy_id, parameters, change_date, reason)
        VALUES (?, ?, ?, ?)
        ''', (
            strategy_id,
            json.dumps(parameters),
            now,
            "Initial creation"
        ))
        
        # Record initial status history
        cursor.execute('''
        INSERT INTO status_history
        (strategy_id, old_status, new_status, change_date, reason)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            strategy_id,
            None,
            self.STATUS_EVOLVED,
            now,
            "Initial creation"
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Registered new strategy: {strategy_id} ({name})")
        
        return strategy_id
    
    def update_strategy_status(self, 
                              strategy_id: str,
                              new_status: str,
                              reason: str) -> bool:
        """
        Update the status of a strategy.
        
        Args:
            strategy_id: ID of the strategy
            new_status: New status to set
            reason: Reason for status change
            
        Returns:
            True if successful, False otherwise
        """
        # Validate status
        valid_statuses = [
            self.STATUS_EVOLVED,
            self.STATUS_VALIDATED,
            self.STATUS_FUNDABLE,
            self.STATUS_UNDER_TEST,
            self.STATUS_PROMOTED,
            self.STATUS_LIVE,
            self.STATUS_RETIRED,
            self.STATUS_REJECTED
        ]
        
        if new_status not in valid_statuses:
            logger.error(f"Invalid status: {new_status}")
            return False
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current status
        cursor.execute("SELECT status FROM strategies WHERE id = ?", (strategy_id,))
        result = cursor.fetchone()
        
        if result is None:
            logger.error(f"Strategy not found: {strategy_id}")
            conn.close()
            return False
        
        old_status = result[0]
        
        # Update status
        now = datetime.datetime.now().isoformat()
        
        cursor.execute('''
        UPDATE strategies
        SET status = ?, last_updated = ?
        WHERE id = ?
        ''', (new_status, now, strategy_id))
        
        # Record status change
        cursor.execute('''
        INSERT INTO status_history
        (strategy_id, old_status, new_status, change_date, reason)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            strategy_id,
            old_status,
            new_status,
            now,
            reason
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated strategy {strategy_id} status: {old_status} → {new_status} ({reason})")
        
        return True
    
    def update_strategy_parameters(self,
                                  strategy_id: str,
                                  parameters: Dict[str, Any],
                                  reason: str) -> bool:
        """
        Update parameters for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            parameters: New parameters
            reason: Reason for parameter change
            
        Returns:
            True if successful, False otherwise
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if strategy exists
        cursor.execute("SELECT strategy_type FROM strategies WHERE id = ?", (strategy_id,))
        result = cursor.fetchone()
        
        if result is None:
            logger.error(f"Strategy not found: {strategy_id}")
            conn.close()
            return False
        
        strategy_type = result[0]
        
        # Generate new fingerprint
        fingerprint = self._generate_fingerprint(strategy_type, parameters)
        
        # Update parameters
        now = datetime.datetime.now().isoformat()
        
        cursor.execute('''
        UPDATE strategies
        SET parameters = ?, fingerprint = ?, last_updated = ?
        WHERE id = ?
        ''', (json.dumps(parameters), fingerprint, now, strategy_id))
        
        # Record parameter change
        cursor.execute('''
        INSERT INTO parameters_history
        (strategy_id, parameters, change_date, reason)
        VALUES (?, ?, ?, ?)
        ''', (
            strategy_id,
            json.dumps(parameters),
            now,
            reason
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated parameters for strategy {strategy_id} ({reason})")
        
        return True
    
    def record_performance(self,
                          strategy_id: str,
                          scenario: str,
                          metrics: Dict[str, Any],
                          score: float,
                          passes_evaluation: bool) -> bool:
        """
        Record performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            scenario: Testing scenario (e.g., 'backtest_bull', 'forward_test')
            metrics: Performance metrics
            score: Overall score
            passes_evaluation: Whether strategy passes evaluation
            
        Returns:
            True if successful, False otherwise
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if strategy exists
        cursor.execute("SELECT id FROM strategies WHERE id = ?", (strategy_id,))
        
        if cursor.fetchone() is None:
            logger.error(f"Strategy not found: {strategy_id}")
            conn.close()
            return False
        
        # Generate performance ID
        perf_id = f"{strategy_id}_{scenario}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Insert performance record
        cursor.execute('''
        INSERT INTO performance
        (id, strategy_id, scenario, date, metrics, score, passes_evaluation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            perf_id,
            strategy_id,
            scenario,
            datetime.datetime.now().isoformat(),
            json.dumps(metrics),
            score,
            1 if passes_evaluation else 0
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded performance for strategy {strategy_id} in scenario {scenario}")
        
        return True
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get strategy details by ID.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy details or None if not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Get strategy
        cursor.execute('''
        SELECT id, name, strategy_type, status, creation_date, last_updated, parameters, fingerprint, tags
        FROM strategies
        WHERE id = ?
        ''', (strategy_id,))
        
        result = cursor.fetchone()
        
        if result is None:
            conn.close()
            return None
        
        # Convert row to dictionary
        strategy = dict(result)
        
        # Parse JSON fields
        strategy['parameters'] = json.loads(strategy['parameters'])
        strategy['tags'] = json.loads(strategy['tags'])
        
        # Get performance history
        cursor.execute('''
        SELECT id, scenario, date, metrics, score, passes_evaluation
        FROM performance
        WHERE strategy_id = ?
        ORDER BY date DESC
        ''', (strategy_id,))
        
        performance = []
        for row in cursor.fetchall():
            perf = dict(row)
            perf['metrics'] = json.loads(perf['metrics'])
            perf['passes_evaluation'] = bool(perf['passes_evaluation'])
            performance.append(perf)
        
        strategy['performance_history'] = performance
        
        # Get status history
        cursor.execute('''
        SELECT old_status, new_status, change_date, reason
        FROM status_history
        WHERE strategy_id = ?
        ORDER BY change_date DESC
        ''', (strategy_id,))
        
        strategy['status_history'] = [dict(row) for row in cursor.fetchall()]
        
        # Get parameters history
        cursor.execute('''
        SELECT parameters, change_date, reason
        FROM parameters_history
        WHERE strategy_id = ?
        ORDER BY change_date DESC
        ''', (strategy_id,))
        
        params_history = []
        for row in cursor.fetchall():
            params = dict(row)
            params['parameters'] = json.loads(params['parameters'])
            params_history.append(params)
        
        strategy['parameters_history'] = params_history
        
        conn.close()
        
        return strategy
    
    def get_strategies_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all strategies with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching strategies
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get strategies
        cursor.execute('''
        SELECT id, name, strategy_type, status, creation_date, last_updated, fingerprint, tags
        FROM strategies
        WHERE status = ?
        ORDER BY last_updated DESC
        ''', (status,))
        
        strategies = []
        for row in cursor.fetchall():
            strategy = dict(row)
            strategy['tags'] = json.loads(strategy['tags'])
            strategies.append(strategy)
        
        conn.close()
        
        return strategies
    
    def find_similar_strategies(self, 
                               strategy_id: str, 
                               threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find strategies similar to the given strategy.
        
        Args:
            strategy_id: ID of the reference strategy
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar strategies with similarity scores
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get reference strategy
        cursor.execute('''
        SELECT strategy_type, parameters, fingerprint
        FROM strategies
        WHERE id = ?
        ''', (strategy_id,))
        
        ref_strategy = cursor.fetchone()
        
        if ref_strategy is None:
            conn.close()
            return []
        
        ref_type = ref_strategy['strategy_type']
        ref_params = json.loads(ref_strategy['parameters'])
        ref_fingerprint = ref_strategy['fingerprint']
        
        # Get all strategies of the same type
        cursor.execute('''
        SELECT id, name, status, parameters, fingerprint
        FROM strategies
        WHERE strategy_type = ? AND id != ?
        ''', (ref_type, strategy_id))
        
        similar_strategies = []
        
        for row in cursor.fetchall():
            strategy = dict(row)
            fingerprint = strategy['fingerprint']
            
            # Calculate similarity based on fingerprint
            # Simple version: count matching characters in fingerprint
            matching_chars = sum(1 for a, b in zip(ref_fingerprint, fingerprint) if a == b)
            similarity = matching_chars / len(ref_fingerprint)
            
            if similarity >= threshold:
                strategy['parameters'] = json.loads(strategy['parameters'])
                strategy['similarity'] = similarity
                similar_strategies.append(strategy)
        
        # Sort by similarity (descending)
        similar_strategies.sort(key=lambda s: s['similarity'], reverse=True)
        
        conn.close()
        
        return similar_strategies
    
    def get_best_strategies(self, 
                           scenario: str,
                           min_score: float = 0.0,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the best performing strategies for a specific scenario.
        
        Args:
            scenario: Scenario to filter by
            min_score: Minimum score threshold
            limit: Maximum number of strategies to return
            
        Returns:
            List of top strategies with their performance
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get top strategies
        cursor.execute('''
        SELECT s.id, s.name, s.strategy_type, s.status, p.score, p.passes_evaluation, p.date
        FROM strategies s
        JOIN performance p ON s.id = p.strategy_id
        WHERE p.scenario = ? AND p.score >= ?
        ORDER BY p.score DESC
        LIMIT ?
        ''', (scenario, min_score, limit))
        
        strategies = []
        for row in cursor.fetchall():
            strategy = dict(row)
            strategy['passes_evaluation'] = bool(strategy['passes_evaluation'])
            strategies.append(strategy)
        
        conn.close()
        
        return strategies
        
    def update_live_metrics(self, strategy_id: str, live_metrics: Dict[str, Any]) -> bool:
        """
        Update live performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            live_metrics: Dictionary of live performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_id not in self.get_all_strategy_ids():
            logger.error(f"Strategy {strategy_id} not found in registry")
            return False
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if strategy already has live metrics
        cursor.execute('''
        SELECT id FROM live_metrics WHERE strategy_id = ?
        ''', (strategy_id,))
        
        existing = cursor.fetchone()
        now = datetime.datetime.now().isoformat()
        
        # Extract metrics
        metrics = {
            'sharpe_ratio': live_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': live_metrics.get('max_drawdown', 0.0),
            'total_return': live_metrics.get('total_return', 0.0),
            'win_rate': live_metrics.get('win_rate', 0.0),
            'trade_count': live_metrics.get('trade_count', 0),
            'avg_trade_duration': live_metrics.get('avg_trade_duration', '0h'),
            'last_updated': now
        }
        
        if existing:
            # Update existing record
            cursor.execute('''
            UPDATE live_metrics
            SET sharpe_ratio = ?, max_drawdown = ?, total_return = ?, 
                win_rate = ?, trade_count = ?, avg_trade_duration = ?, last_updated = ?
            WHERE strategy_id = ?
            ''', (
                metrics['sharpe_ratio'], metrics['max_drawdown'], metrics['total_return'],
                metrics['win_rate'], metrics['trade_count'], metrics['avg_trade_duration'],
                metrics['last_updated'], strategy_id
            ))
        else:
            # Insert new record
            cursor.execute('''
            INSERT INTO live_metrics
            (strategy_id, sharpe_ratio, max_drawdown, total_return, win_rate, 
             trade_count, avg_trade_duration, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_id, metrics['sharpe_ratio'], metrics['max_drawdown'], 
                metrics['total_return'], metrics['win_rate'], metrics['trade_count'],
                metrics['avg_trade_duration'], metrics['last_updated']
            ))
        
        # Update strategy's last_updated timestamp
        cursor.execute('''
        UPDATE strategies
        SET last_updated = ?
        WHERE id = ?
        ''', (now, strategy_id))
        
        conn.commit()
        conn.close()
        
        # Calculate and store delta metrics
        self.calculate_and_store_delta(strategy_id)
        
        logger.info(f"Updated live metrics for strategy {strategy_id}")
        return True
        
    def get_live_metrics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get live performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of live metrics or None if not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get live metrics
        cursor.execute('''
        SELECT sharpe_ratio, max_drawdown, total_return, win_rate, 
               trade_count, avg_trade_duration, last_updated
        FROM live_metrics
        WHERE strategy_id = ?
        ''', (strategy_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        
        return None
        
    def get_backtest_metrics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest backtest metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of backtest metrics or None if not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get latest backtest metrics
        cursor.execute('''
        SELECT metrics, score
        FROM performance
        WHERE strategy_id = ? AND scenario LIKE 'backtest%'
        ORDER BY date DESC
        LIMIT 1
        ''', (strategy_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            metrics = json.loads(row['metrics'])
            metrics['score'] = row['score']
            return metrics
        
        return None
        
    def calculate_and_store_delta(self, strategy_id: str) -> bool:
        """
        Calculate and store the delta between live and backtest metrics.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            True if successful, False otherwise
        """
        # Get live and backtest metrics
        live = self.get_live_metrics(strategy_id)
        backtest = self.get_backtest_metrics(strategy_id)
        
        if not live or not backtest:
            logger.warning(f"Missing metrics for {strategy_id}")
            return False
        
        # Calculate deltas
        deltas = {}
        deltas['sharpe_delta'] = live.get('sharpe_ratio', 0) - backtest.get('sharpe_ratio', 0)
        deltas['drawdown_delta'] = live.get('max_drawdown', 0) - backtest.get('max_drawdown', 0)
        deltas['return_delta'] = live.get('total_return', 0) - backtest.get('total_return', 0)
        deltas['win_rate_delta'] = live.get('win_rate', 0) - backtest.get('win_rate', 0)
        
        # Calculate confidence score (simplified version)
        # Negative values mean live performance is worse than backtest
        # Calculate confidence on scale of 0-1 where 1 means perfect match or better
        sharpe_weight = 0.4
        drawdown_weight = 0.3
        return_weight = 0.2
        win_rate_weight = 0.1
        
        # Convert deltas to confidence components (0-1 scale)
        sharpe_conf = max(0, min(1, 1 - abs(deltas['sharpe_delta']) / max(backtest.get('sharpe_ratio', 1), 0.1)))
        # For drawdown, negative delta is good (less drawdown in live)
        drawdown_conf = max(0, min(1, 1 - max(0, deltas['drawdown_delta']) / max(backtest.get('max_drawdown', 0.05), 0.01)))
        return_conf = max(0, min(1, 1 - abs(deltas['return_delta']) / max(abs(backtest.get('total_return', 0.1)), 0.01)))
        win_rate_conf = max(0, min(1, 1 - abs(deltas['win_rate_delta']) / max(backtest.get('win_rate', 0.5), 0.1)))
        
        # Calculate weighted confidence score
        confidence = (
            sharpe_weight * sharpe_conf +
            drawdown_weight * drawdown_conf +
            return_weight * return_conf +
            win_rate_weight * win_rate_conf
        )
        
        deltas['confidence_score'] = confidence
        
        # Determine recommendation
        if confidence > 0.8 and live.get('trade_count', 0) >= 20:
            deltas['recommendation'] = 'promote'
        elif confidence < 0.3 and live.get('trade_count', 0) >= 10:
            deltas['recommendation'] = 'demote'
        elif confidence < 0.2 and live.get('trade_count', 0) >= 5:
            deltas['recommendation'] = 'terminate'
        else:
            deltas['recommendation'] = 'maintain'
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if delta metrics already exist
        cursor.execute('''
        SELECT id FROM performance_delta WHERE strategy_id = ?
        ''', (strategy_id,))
        
        existing = cursor.fetchone()
        now = datetime.datetime.now().isoformat()
        
        if existing:
            # Update existing record
            cursor.execute('''
            UPDATE performance_delta
            SET sharpe_delta = ?, drawdown_delta = ?, return_delta = ?, 
                win_rate_delta = ?, confidence_score = ?, recommendation = ?,
                calculation_date = ?
            WHERE strategy_id = ?
            ''', (
                deltas['sharpe_delta'], deltas['drawdown_delta'], deltas['return_delta'],
                deltas['win_rate_delta'], deltas['confidence_score'], deltas['recommendation'],
                now, strategy_id
            ))
        else:
            # Insert new record
            cursor.execute('''
            INSERT INTO performance_delta
            (strategy_id, sharpe_delta, drawdown_delta, return_delta, win_rate_delta,
             confidence_score, recommendation, calculation_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_id, deltas['sharpe_delta'], deltas['drawdown_delta'],
                deltas['return_delta'], deltas['win_rate_delta'], deltas['confidence_score'],
                deltas['recommendation'], now
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Calculated delta metrics for strategy {strategy_id} (confidence: {confidence:.2f})")
        return True
        
    def get_performance_delta(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the performance delta metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of delta metrics or None if not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get delta metrics
        cursor.execute('''
        SELECT sharpe_delta, drawdown_delta, return_delta, win_rate_delta,
               confidence_score, recommendation, calculation_date
        FROM performance_delta
        WHERE strategy_id = ?
        ''', (strategy_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        
        return None
        
    def get_active_strategies(self) -> List[str]:
        """
        Get all active strategies (paper or live).
        
        Returns:
            List of strategy IDs
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get active strategies
        cursor.execute('''
        SELECT id FROM strategies
        WHERE status IN (?, ?, ?)
        ''', (self.STATUS_UNDER_TEST, self.STATUS_PAPER_TESTING, self.STATUS_LIVE))
        
        strategy_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return strategy_ids
        
    def get_all_strategy_ids(self) -> List[str]:
        """
        Get all strategy IDs in the registry.
        
        Returns:
            List of strategy IDs
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all strategy IDs
        cursor.execute('SELECT id FROM strategies')
        strategy_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return strategy_ids
        
    def get_confidence_ranking(self) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by confidence score (live vs backtest).
        
        Returns:
            List of (strategy_id, confidence_score) tuples, sorted by confidence
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get strategies with confidence scores
        cursor.execute('''
        SELECT s.id, s.name, pd.confidence_score
        FROM strategies s
        JOIN performance_delta pd ON s.id = pd.strategy_id
        ORDER BY pd.confidence_score DESC
        ''')
        
        strategies = [(row[0], row[2]) for row in cursor.fetchall()]
        conn.close()
        
        return strategies
        
    def get_promotion_candidates(self, min_confidence: float = 0.7, min_trades: int = 20) -> List[Dict[str, Any]]:
        """
        Get strategies that are candidates for promotion to live.
        
        Args:
            min_confidence: Minimum confidence score threshold
            min_trades: Minimum number of trades required
            
        Returns:
            List of promotion candidate dictionaries
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get promotion candidates
        cursor.execute('''
        SELECT s.id, s.name, s.strategy_type, pd.confidence_score, lm.trade_count,
               lm.sharpe_ratio, lm.last_updated
        FROM strategies s
        JOIN performance_delta pd ON s.id = pd.strategy_id
        JOIN live_metrics lm ON s.id = lm.strategy_id
        WHERE s.status = ? AND pd.confidence_score >= ? AND lm.trade_count >= ?
        ORDER BY pd.confidence_score DESC
        ''', (self.STATUS_PAPER_TESTING, min_confidence, min_trades))
        
        candidates = []
        for row in cursor.fetchall():
            candidate = dict(row)
            # Calculate days in paper testing
            days_in_paper = self._days_in_status(row['id'], self.STATUS_PAPER_TESTING)
            candidate['days_in_paper'] = days_in_paper
            candidates.append(candidate)
        
        conn.close()
        
        return candidates
        
    def get_demotion_candidates(self, max_confidence: float = 0.3, min_trades: int = 10) -> List[Dict[str, Any]]:
        """
        Get strategies that are candidates for demotion or termination.
        
        Args:
            max_confidence: Maximum confidence score threshold
            min_trades: Minimum number of trades required
            
        Returns:
            List of demotion candidate dictionaries
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get demotion candidates
        cursor.execute('''
        SELECT s.id, s.name, s.strategy_type, s.status, pd.confidence_score, pd.recommendation,
               lm.trade_count, lm.sharpe_ratio, lm.last_updated
        FROM strategies s
        JOIN performance_delta pd ON s.id = pd.strategy_id
        JOIN live_metrics lm ON s.id = lm.strategy_id
        WHERE s.status IN (?, ?) AND pd.confidence_score <= ? AND lm.trade_count >= ?
        ORDER BY pd.confidence_score ASC
        ''', (self.STATUS_PAPER_TESTING, self.STATUS_LIVE, max_confidence, min_trades))
        
        candidates = []
        for row in cursor.fetchall():
            candidates.append(dict(row))
        
        conn.close()
        
        return candidates
        
    def _days_in_status(self, strategy_id: str, status: str) -> int:
        """
        Calculate how many days a strategy has been in a given status.
        
        Args:
            strategy_id: ID of the strategy
            status: Status to check for
            
        Returns:
            Number of days in the status
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the most recent status change to this status
        cursor.execute('''
        SELECT change_date
        FROM status_history
        WHERE strategy_id = ? AND new_status = ?
        ORDER BY change_date DESC
        LIMIT 1
        ''', (strategy_id, status))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return 0
        
        try:
            change_date = datetime.datetime.fromisoformat(row['change_date'])
            days = (datetime.datetime.now() - change_date).days
            return max(0, days)
        except (ValueError, TypeError):
            return 0
