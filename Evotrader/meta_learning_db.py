#!/usr/bin/env python3
"""
Meta-Learning Database

Database structure for storing and analyzing strategy performance patterns
across different market regimes, to guide future evolution.
"""

import os
import json
import sqlite3
import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meta_learning_db')

class MetaLearningDB:
    """
    Database for storing and retrieving meta-learning data.
    
    Tracks historical performance patterns across multiple strategy evolutions
    and market regimes to inform future evolutions.
    """
    
    def __init__(self, db_path="./meta_learning/meta_db.sqlite"):
        """
        Initialize the meta-learning database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database tables
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for meta-learning storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create strategy_results table - main table for all strategy results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategy_results (
            id TEXT PRIMARY KEY,
            strategy_id TEXT,
            timestamp TEXT,
            strategy_type TEXT,
            indicators TEXT,  -- JSON array
            parameters TEXT,  -- JSON object
            market_regime TEXT,
            volatility TEXT,
            asset_class TEXT,
            symbol TEXT,
            backtest_metrics TEXT,  -- JSON object
            live_metrics TEXT,  -- JSON object
            consistency_metrics TEXT,  -- JSON object
            status TEXT,
            session_performance TEXT  -- JSON object
        )
        ''')
        
        # Create strategy_type_success table - aggregated success by strategy type
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategy_type_success (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_type TEXT,
            market_regime TEXT,
            success_count INTEGER,
            failure_count INTEGER,
            success_rate REAL,
            avg_confidence_score REAL,
            last_updated TEXT
        )
        ''')
        
        # Create indicator_effectiveness table - tracks effectiveness of indicators
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicator_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator TEXT,
            market_regime TEXT,
            success_count INTEGER,
            failure_count INTEGER,
            effectiveness_score REAL,
            last_updated TEXT
        )
        ''')
        
        # Create parameter_zones table - tracks successful parameter ranges
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameter_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_type TEXT,
            parameter_name TEXT,
            min_value REAL,
            max_value REAL,
            avg_value REAL,
            success_count INTEGER,
            last_updated TEXT
        )
        ''')
        
        # Create indicator_combinations table - tracks effective combinations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicator_combinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicators TEXT,  -- JSON array of sorted indicators
            market_regime TEXT,
            success_count INTEGER,
            failure_count INTEGER,
            effectiveness_score REAL,
            last_updated TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Meta-learning database initialized at {self.db_path}")
    
    def log_strategy_result(self, 
                           strategy_data: Dict[str, Any], 
                           performance_data: Dict[str, Any], 
                           market_conditions: Dict[str, Any]) -> bool:
        """
        Log the result of a strategy after paper/live testing.
        
        Args:
            strategy_data: Strategy details (id, type, indicators, params)
            performance_data: Performance metrics (backtest, live, consistency)
            market_conditions: Market conditions when strategy was used
            
        Returns:
            Success status
        """
        try:
            # Generate record ID
            record_id = f"{strategy_data['strategy_id']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert into strategy_results
            cursor.execute('''
            INSERT INTO strategy_results
            (id, strategy_id, timestamp, strategy_type, indicators, parameters,
             market_regime, volatility, asset_class, symbol,
             backtest_metrics, live_metrics, consistency_metrics, status, session_performance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                strategy_data.get('strategy_id'),
                datetime.datetime.now().isoformat(),
                strategy_data.get('type'),
                json.dumps(strategy_data.get('indicators', [])),
                json.dumps(strategy_data.get('parameters', {})),
                market_conditions.get('regime'),
                market_conditions.get('volatility'),
                market_conditions.get('asset_class'),
                market_conditions.get('symbol'),
                json.dumps(performance_data.get('backtest', {})),
                json.dumps(performance_data.get('live', {})),
                json.dumps(performance_data.get('consistency', {})),
                performance_data.get('status'),
                json.dumps(performance_data.get('session_performance', {}))
            ))
            
            # Update aggregated tables
            self._update_strategy_type_success(
                cursor, 
                strategy_data.get('type'), 
                market_conditions.get('regime'),
                performance_data
            )
            
            self._update_indicator_effectiveness(
                cursor,
                strategy_data.get('indicators', []),
                market_conditions.get('regime'),
                performance_data
            )
            
            self._update_parameter_zones(
                cursor,
                strategy_data.get('type'),
                strategy_data.get('parameters', {}),
                performance_data
            )
            
            self._update_indicator_combinations(
                cursor,
                strategy_data.get('indicators', []),
                market_conditions.get('regime'),
                performance_data
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged strategy result for {strategy_data.get('strategy_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging strategy result: {e}")
            return False
            
    def _update_strategy_type_success(self, cursor, strategy_type, market_regime, performance_data):
        """Update strategy type success rates."""
        if not strategy_type or not market_regime:
            return
            
        # Determine if strategy was successful
        consistency = performance_data.get('consistency', {})
        confidence_score = consistency.get('confidence_score', 0)
        is_successful = confidence_score > 0.7 and performance_data.get('status') in ['promoted', 'live']
        
        # Check if record exists
        cursor.execute('''
        SELECT success_count, failure_count, avg_confidence_score
        FROM strategy_type_success
        WHERE strategy_type = ? AND market_regime = ?
        ''', (strategy_type, market_regime))
        
        row = cursor.fetchone()
        
        if row:
            # Update existing record
            success_count = row[0] + (1 if is_successful else 0)
            failure_count = row[1] + (0 if is_successful else 1)
            total_count = success_count + failure_count
            
            # Update average confidence score
            old_avg = row[2]
            new_avg = ((old_avg * (total_count - 1)) + confidence_score) / total_count
            
            # Calculate success rate
            success_rate = success_count / total_count if total_count > 0 else 0
            
            cursor.execute('''
            UPDATE strategy_type_success
            SET success_count = ?, failure_count = ?, success_rate = ?, 
                avg_confidence_score = ?, last_updated = ?
            WHERE strategy_type = ? AND market_regime = ?
            ''', (
                success_count, failure_count, success_rate, 
                new_avg, datetime.datetime.now().isoformat(),
                strategy_type, market_regime
            ))
        else:
            # Insert new record
            success_count = 1 if is_successful else 0
            failure_count = 0 if is_successful else 1
            success_rate = 1.0 if is_successful else 0.0
            
            cursor.execute('''
            INSERT INTO strategy_type_success
            (strategy_type, market_regime, success_count, failure_count, 
             success_rate, avg_confidence_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_type, market_regime, success_count, failure_count,
                success_rate, confidence_score, datetime.datetime.now().isoformat()
            ))
    
    def _update_indicator_effectiveness(self, cursor, indicators, market_regime, performance_data):
        """Update indicator effectiveness metrics."""
        if not indicators or not market_regime:
            return
            
        # Determine if strategy was successful
        consistency = performance_data.get('consistency', {})
        confidence_score = consistency.get('confidence_score', 0)
        is_successful = confidence_score > 0.7 and performance_data.get('status') in ['promoted', 'live']
        
        # Update each indicator
        for indicator in indicators:
            # Check if record exists
            cursor.execute('''
            SELECT success_count, failure_count, effectiveness_score
            FROM indicator_effectiveness
            WHERE indicator = ? AND market_regime = ?
            ''', (indicator, market_regime))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing record
                success_count = row[0] + (1 if is_successful else 0)
                failure_count = row[1] + (0 if is_successful else 1)
                total_count = success_count + failure_count
                
                # Calculate effectiveness score
                effectiveness = success_count / total_count if total_count > 0 else 0
                
                cursor.execute('''
                UPDATE indicator_effectiveness
                SET success_count = ?, failure_count = ?, 
                    effectiveness_score = ?, last_updated = ?
                WHERE indicator = ? AND market_regime = ?
                ''', (
                    success_count, failure_count, effectiveness,
                    datetime.datetime.now().isoformat(),
                    indicator, market_regime
                ))
            else:
                # Insert new record
                success_count = 1 if is_successful else 0
                failure_count = 0 if is_successful else 1
                effectiveness = 1.0 if is_successful else 0.0
                
                cursor.execute('''
                INSERT INTO indicator_effectiveness
                (indicator, market_regime, success_count, failure_count, 
                 effectiveness_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    indicator, market_regime, success_count, failure_count,
                    effectiveness, datetime.datetime.now().isoformat()
                ))
    
    def _update_parameter_zones(self, cursor, strategy_type, parameters, performance_data):
        """Update parameter zones (successful ranges)."""
        if not strategy_type or not parameters:
            return
            
        # Only track parameters for successful strategies
        consistency = performance_data.get('consistency', {})
        confidence_score = consistency.get('confidence_score', 0)
        if confidence_score <= 0.7 or performance_data.get('status') not in ['promoted', 'live']:
            return
            
        # Update each numeric parameter
        for param_name, param_value in parameters.items():
            if not isinstance(param_value, (int, float)):
                continue
                
            # Check if record exists
            cursor.execute('''
            SELECT min_value, max_value, avg_value, success_count
            FROM parameter_zones
            WHERE strategy_type = ? AND parameter_name = ?
            ''', (strategy_type, param_name))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing record
                min_value = min(row[0], param_value)
                max_value = max(row[1], param_value)
                
                # Update running average
                old_avg = row[2]
                old_count = row[3]
                new_count = old_count + 1
                new_avg = ((old_avg * old_count) + param_value) / new_count
                
                cursor.execute('''
                UPDATE parameter_zones
                SET min_value = ?, max_value = ?, avg_value = ?, 
                    success_count = ?, last_updated = ?
                WHERE strategy_type = ? AND parameter_name = ?
                ''', (
                    min_value, max_value, new_avg, new_count,
                    datetime.datetime.now().isoformat(),
                    strategy_type, param_name
                ))
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO parameter_zones
                (strategy_type, parameter_name, min_value, max_value, 
                 avg_value, success_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy_type, param_name, param_value, param_value,
                    param_value, 1, datetime.datetime.now().isoformat()
                ))
    
    def _update_indicator_combinations(self, cursor, indicators, market_regime, performance_data):
        """Update indicator combination effectiveness."""
        if len(indicators) < 2 or not market_regime:
            return
            
        # Determine if strategy was successful
        consistency = performance_data.get('consistency', {})
        confidence_score = consistency.get('confidence_score', 0)
        is_successful = confidence_score > 0.7 and performance_data.get('status') in ['promoted', 'live']
        
        # Sort indicators to create consistent key
        sorted_indicators = sorted(indicators)
        indicators_key = json.dumps(sorted_indicators)
        
        # Check if record exists
        cursor.execute('''
        SELECT success_count, failure_count, effectiveness_score
        FROM indicator_combinations
        WHERE indicators = ? AND market_regime = ?
        ''', (indicators_key, market_regime))
        
        row = cursor.fetchone()
        
        if row:
            # Update existing record
            success_count = row[0] + (1 if is_successful else 0)
            failure_count = row[1] + (0 if is_successful else 1)
            total_count = success_count + failure_count
            
            # Calculate effectiveness score
            effectiveness = success_count / total_count if total_count > 0 else 0
            
            cursor.execute('''
            UPDATE indicator_combinations
            SET success_count = ?, failure_count = ?, 
                effectiveness_score = ?, last_updated = ?
            WHERE indicators = ? AND market_regime = ?
            ''', (
                success_count, failure_count, effectiveness,
                datetime.datetime.now().isoformat(),
                indicators_key, market_regime
            ))
        else:
            # Insert new record
            success_count = 1 if is_successful else 0
            failure_count = 0 if is_successful else 1
            effectiveness = 1.0 if is_successful else 0.0
            
            cursor.execute('''
            INSERT INTO indicator_combinations
            (indicators, market_regime, success_count, failure_count, 
             effectiveness_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                indicators_key, market_regime, success_count, failure_count,
                effectiveness, datetime.datetime.now().isoformat()
            ))
    
    def get_strategy_type_success_rates(self, market_regime=None):
        """
        Get success rates for different strategy types.
        
        Args:
            market_regime: Optional market regime to filter by
            
        Returns:
            DataFrame with strategy type success rates
        """
        conn = sqlite3.connect(self.db_path)
        
        if market_regime:
            query = '''
            SELECT strategy_type, market_regime, success_count, failure_count, 
                   success_rate, avg_confidence_score
            FROM strategy_type_success
            WHERE market_regime = ?
            ORDER BY success_rate DESC
            '''
            df = pd.read_sql_query(query, conn, params=(market_regime,))
        else:
            query = '''
            SELECT strategy_type, market_regime, success_count, failure_count, 
                   success_rate, avg_confidence_score
            FROM strategy_type_success
            ORDER BY success_rate DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_successful_parameter_ranges(self, strategy_type, parameter=None):
        """
        Get the successful ranges for parameters.
        
        Args:
            strategy_type: Type of strategy
            parameter: Optional specific parameter to get
            
        Returns:
            DataFrame with parameter ranges
        """
        conn = sqlite3.connect(self.db_path)
        
        if parameter:
            query = '''
            SELECT parameter_name, min_value, max_value, avg_value, success_count
            FROM parameter_zones
            WHERE strategy_type = ? AND parameter_name = ?
            ORDER BY success_count DESC
            '''
            df = pd.read_sql_query(query, conn, params=(strategy_type, parameter))
        else:
            query = '''
            SELECT parameter_name, min_value, max_value, avg_value, success_count
            FROM parameter_zones
            WHERE strategy_type = ?
            ORDER BY parameter_name
            '''
            df = pd.read_sql_query(query, conn, params=(strategy_type,))
        
        conn.close()
        return df
        
    def get_indicator_effectiveness(self, market_regime=None):
        """
        Get the effectiveness of different indicators.
        
        Args:
            market_regime: Optional market regime to filter by
            
        Returns:
            DataFrame with indicator effectiveness
        """
        conn = sqlite3.connect(self.db_path)
        
        if market_regime:
            query = '''
            SELECT indicator, market_regime, success_count, failure_count, effectiveness_score
            FROM indicator_effectiveness
            WHERE market_regime = ?
            ORDER BY effectiveness_score DESC
            '''
            df = pd.read_sql_query(query, conn, params=(market_regime,))
        else:
            query = '''
            SELECT indicator, market_regime, success_count, failure_count, effectiveness_score
            FROM indicator_effectiveness
            ORDER BY effectiveness_score DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_indicator_combinations(self, market_regime=None):
        """
        Get the effectiveness of indicator combinations.
        
        Args:
            market_regime: Optional market regime to filter by
            
        Returns:
            DataFrame with indicator combination effectiveness
        """
        conn = sqlite3.connect(self.db_path)
        
        if market_regime:
            query = '''
            SELECT indicators, market_regime, success_count, failure_count, effectiveness_score
            FROM indicator_combinations
            WHERE market_regime = ?
            ORDER BY effectiveness_score DESC
            '''
            df = pd.read_sql_query(query, conn, params=(market_regime,))
        else:
            query = '''
            SELECT indicators, market_regime, success_count, failure_count, effectiveness_score
            FROM indicator_combinations
            ORDER BY effectiveness_score DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Parse indicators JSON
        df['indicators_list'] = df['indicators'].apply(json.loads)
        
        return df
    
    def get_performance_consistency(self, strategy_type=None):
        """
        Get the backtest-to-live consistency for strategies.
        
        Args:
            strategy_type: Optional strategy type to filter by
            
        Returns:
            DataFrame with consistency metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        if strategy_type:
            query = '''
            SELECT strategy_id, strategy_type, timestamp, 
                   json_extract(consistency_metrics, '$.confidence_score') as confidence_score,
                   json_extract(consistency_metrics, '$.sharpe_delta') as sharpe_delta,
                   json_extract(consistency_metrics, '$.drawdown_delta') as drawdown_delta,
                   json_extract(consistency_metrics, '$.return_delta') as return_delta,
                   json_extract(consistency_metrics, '$.win_rate_delta') as win_rate_delta,
                   status
            FROM strategy_results
            WHERE strategy_type = ?
            ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(strategy_type,))
        else:
            query = '''
            SELECT strategy_id, strategy_type, timestamp, 
                   json_extract(consistency_metrics, '$.confidence_score') as confidence_score,
                   json_extract(consistency_metrics, '$.sharpe_delta') as sharpe_delta,
                   json_extract(consistency_metrics, '$.drawdown_delta') as drawdown_delta,
                   json_extract(consistency_metrics, '$.return_delta') as return_delta,
                   json_extract(consistency_metrics, '$.win_rate_delta') as win_rate_delta,
                   status
            FROM strategy_results
            ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_evolution_bias_config(self, market_regime=None):
        """
        Generate a bias configuration for evolution based on historical data.
        
        Args:
            market_regime: Optional market regime to filter by
            
        Returns:
            Dictionary with bias configuration
        """
        # Get top strategy types
        strategy_types_df = self.get_strategy_type_success_rates(market_regime)
        
        # Get top indicators
        indicators_df = self.get_indicator_effectiveness(market_regime)
        
        # Get top indicator combinations
        combinations_df = self.get_indicator_combinations(market_regime)
        
        # Build bias configuration
        bias_config = {
            'generated_at': datetime.datetime.now().isoformat(),
            'market_regime': market_regime,
            'strategy_types': {},
            'indicators': {},
            'indicator_combinations': [],
            'parameter_ranges': {}
        }
        
        # Add strategy type biases (only include types with at least 2 samples)
        if not strategy_types_df.empty:
            for _, row in strategy_types_df.iterrows():
                if row['success_count'] + row['failure_count'] >= 2:
                    bias_config['strategy_types'][row['strategy_type']] = {
                        'success_rate': float(row['success_rate']),
                        'sample_count': int(row['success_count'] + row['failure_count']),
                        'confidence': float(row['avg_confidence_score'])
                    }
        
        # Add indicator biases (only include indicators with at least 3 samples)
        if not indicators_df.empty:
            for _, row in indicators_df.iterrows():
                total_samples = row['success_count'] + row['failure_count']
                if total_samples >= 3:
                    bias_config['indicators'][row['indicator']] = {
                        'effectiveness': float(row['effectiveness_score']),
                        'sample_count': int(total_samples)
                    }
        
        # Add top 10 indicator combinations
        if not combinations_df.empty:
            for _, row in combinations_df.head(10).iterrows():
                total_samples = row['success_count'] + row['failure_count']
                if total_samples >= 2:
                    bias_config['indicator_combinations'].append({
                        'indicators': row['indicators_list'],
                        'effectiveness': float(row['effectiveness_score']),
                        'sample_count': int(total_samples)
                    })
        
        # Add parameter ranges for top strategy types
        for strategy_type in bias_config['strategy_types']:
            params_df = self.get_successful_parameter_ranges(strategy_type)
            
            if not params_df.empty:
                param_dict = {}
                for _, row in params_df.iterrows():
                    if row['success_count'] >= 3:
                        param_dict[row['parameter_name']] = {
                            'min': float(row['min_value']),
                            'max': float(row['max_value']),
                            'avg': float(row['avg_value']),
                            'sample_count': int(row['success_count'])
                        }
                
                if param_dict:
                    bias_config['parameter_ranges'][strategy_type] = param_dict
        
        return bias_config


if __name__ == "__main__":
    # Test functionality
    db = MetaLearningDB()
    print("Meta-learning database initialized successfully.")
