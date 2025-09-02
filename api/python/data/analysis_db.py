#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database module for storing trading analysis results and selection history.

This module provides a SQLite-based database implementation for persistently storing:
1. Analysis results (technical, fundamental, sentiment, etc.)
2. Stock selection history
3. Strategy selection history
4. Performance metrics from analysis
5. Market regime classifications and analysis context
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class AnalysisDatabase:
    """Database for storing analysis results and selection history"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the analysis database
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            # Default location is in the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "analysis_results.db")
            
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for stock analysis results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            score REAL,
            rank INTEGER,
            recommendation TEXT,
            metrics TEXT,
            analysis_details TEXT,
            model_version TEXT,
            UNIQUE(symbol, timestamp, analysis_type)
        )
        ''')
        
        # Create table for stock selection history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_selection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            selection_criteria TEXT NOT NULL,
            symbols TEXT NOT NULL,
            weights TEXT,
            market_regime TEXT,
            reasoning TEXT,
            performance_snapshot TEXT
        )
        ''')
        
        # Create table for strategy selection history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategy_selection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            selected_strategy TEXT NOT NULL,
            market_regime TEXT,
            confidence_score REAL,
            reasoning TEXT,
            parameters TEXT
        )
        ''')
        
        # Create table for market regime analysis
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL,
            indicators TEXT,
            description TEXT
        )
        ''')
        
        # Create table for sentiment analysis results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            sentiment_score REAL,
            sentiment_label TEXT,
            volume INTEGER,
            key_phrases TEXT,
            source_details TEXT,
            UNIQUE(symbol, timestamp, source)
        )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_analysis_symbol ON stock_analysis(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_analysis_timestamp ON stock_analysis(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_analysis(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment_analysis(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info(f"Analysis database initialized at {self.db_path}")

    def save_stock_analysis(self, 
                          symbol: str,
                          analysis_type: str,
                          score: float,
                          rank: Optional[int] = None,
                          recommendation: Optional[str] = None,
                          metrics: Optional[Dict[str, Any]] = None,
                          analysis_details: Optional[Dict[str, Any]] = None,
                          model_version: Optional[str] = None) -> bool:
        """
        Save stock analysis results to the database
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis (technical, fundamental, ml, etc.)
            score: Analysis score (normalized value)
            rank: Rank among analyzed stocks (optional)
            recommendation: Trading recommendation (buy, sell, hold)
            metrics: Dictionary of analysis metrics
            analysis_details: Detailed analysis information
            model_version: Version of the analysis model used
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            metrics_json = json.dumps(metrics or {})
            details_json = json.dumps(analysis_details or {})
            
            cursor.execute(
                '''
                INSERT OR REPLACE INTO stock_analysis 
                (symbol, timestamp, analysis_type, score, rank, recommendation, 
                metrics, analysis_details, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    symbol, 
                    now,
                    analysis_type,
                    score,
                    rank,
                    recommendation,
                    metrics_json,
                    details_json,
                    model_version
                )
            )
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved {analysis_type} analysis for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stock analysis: {str(e)}")
            return False
    
    def get_latest_stock_analysis(self,
                                symbol: str,
                                analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the latest analysis for a stock
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis to retrieve (optional)
            
        Returns:
            List of analysis results dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if analysis_type:
                cursor.execute(
                    '''
                    SELECT * FROM stock_analysis 
                    WHERE symbol = ? AND analysis_type = ?
                    ORDER BY timestamp DESC LIMIT 1
                    ''',
                    (symbol, analysis_type)
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM stock_analysis 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    ''',
                    (symbol,)
                )
            
            results = cursor.fetchall()
            conn.close()
            
            analysis_results = []
            for row in results:
                result = dict(row)
                result['metrics'] = json.loads(result['metrics'])
                result['analysis_details'] = json.loads(result['analysis_details'])
                analysis_results.append(result)
                
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error retrieving stock analysis: {str(e)}")
            return []
    
    def save_stock_selection(self,
                           selection_criteria: str,
                           symbols: List[str],
                           weights: Optional[Dict[str, float]] = None,
                           market_regime: Optional[str] = None,
                           reasoning: Optional[str] = None,
                           performance_snapshot: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save stock selection history to the database
        
        Args:
            selection_criteria: Criteria used for selection
            symbols: List of selected stock symbols
            weights: Dictionary mapping symbols to allocation weights
            market_regime: Current market regime
            reasoning: Explanation for the selection
            performance_snapshot: Snapshot of portfolio performance metrics
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            symbols_json = json.dumps(symbols)
            weights_json = json.dumps(weights or {})
            performance_json = json.dumps(performance_snapshot or {})
            
            cursor.execute(
                '''
                INSERT INTO stock_selection_history 
                (timestamp, selection_criteria, symbols, weights, market_regime, 
                reasoning, performance_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    now,
                    selection_criteria,
                    symbols_json,
                    weights_json,
                    market_regime,
                    reasoning,
                    performance_json
                )
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Saved stock selection with {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stock selection: {str(e)}")
            return False
    
    def get_stock_selection_history(self, 
                                  limit: int = 10,
                                  selection_criteria: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get stock selection history
        
        Args:
            limit: Maximum number of history entries to retrieve
            selection_criteria: Filter by selection criteria
            
        Returns:
            List of selection history entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if selection_criteria:
                cursor.execute(
                    '''
                    SELECT * FROM stock_selection_history
                    WHERE selection_criteria = ?
                    ORDER BY timestamp DESC LIMIT ?
                    ''',
                    (selection_criteria, limit)
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM stock_selection_history
                    ORDER BY timestamp DESC LIMIT ?
                    ''',
                    (limit,)
                )
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                entry = dict(row)
                entry['symbols'] = json.loads(entry['symbols'])
                entry['weights'] = json.loads(entry['weights'])
                entry['performance_snapshot'] = json.loads(entry['performance_snapshot'])
                history.append(entry)
                
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving stock selection history: {str(e)}")
            return []
    
    def save_strategy_selection(self,
                              selected_strategy: str,
                              market_regime: Optional[str] = None,
                              confidence_score: Optional[float] = None,
                              reasoning: Optional[str] = None,
                              parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save strategy selection to the database
        
        Args:
            selected_strategy: Name of the selected strategy
            market_regime: Current market regime
            confidence_score: Confidence score for the selection
            reasoning: Explanation for the selection
            parameters: Strategy parameters
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            parameters_json = json.dumps(parameters or {})
            
            cursor.execute(
                '''
                INSERT INTO strategy_selection_history 
                (timestamp, selected_strategy, market_regime, confidence_score, 
                reasoning, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    now,
                    selected_strategy,
                    market_regime,
                    confidence_score,
                    reasoning,
                    parameters_json
                )
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Saved strategy selection: {selected_strategy}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy selection: {str(e)}")
            return False
    
    def get_strategy_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get strategy selection history
        
        Args:
            limit: Maximum number of history entries to retrieve
            
        Returns:
            List of strategy selection history entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                SELECT * FROM strategy_selection_history
                ORDER BY timestamp DESC LIMIT ?
                ''',
                (limit,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                entry = dict(row)
                entry['parameters'] = json.loads(entry['parameters'])
                history.append(entry)
                
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving strategy selection history: {str(e)}")
            return []
    
    def save_market_regime(self,
                         regime: str,
                         confidence: float,
                         indicators: Dict[str, Any],
                         description: Optional[str] = None) -> bool:
        """
        Save market regime analysis to the database
        
        Args:
            regime: Identified market regime
            confidence: Confidence score
            indicators: Dictionary of indicator values used
            description: Description of the market regime
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            indicators_json = json.dumps(indicators)
            
            cursor.execute(
                '''
                INSERT INTO market_regime_history 
                (timestamp, regime, confidence, indicators, description)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    now,
                    regime,
                    confidence,
                    indicators_json,
                    description
                )
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Saved market regime analysis: {regime}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving market regime: {str(e)}")
            return False
    
    def get_market_regime_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get market regime history
        
        Args:
            limit: Maximum number of history entries to retrieve
            
        Returns:
            List of market regime history entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                SELECT * FROM market_regime_history
                ORDER BY timestamp DESC LIMIT ?
                ''',
                (limit,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                entry = dict(row)
                entry['indicators'] = json.loads(entry['indicators'])
                history.append(entry)
                
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving market regime history: {str(e)}")
            return []
    
    def save_sentiment_analysis(self,
                              symbol: str,
                              source: str,
                              sentiment_score: float,
                              sentiment_label: Optional[str] = None,
                              volume: Optional[int] = None,
                              key_phrases: Optional[List[str]] = None,
                              source_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save sentiment analysis results to the database
        
        Args:
            symbol: Stock symbol
            source: Source of sentiment data (twitter, news, etc.)
            sentiment_score: Normalized sentiment score
            sentiment_label: Sentiment classification label
            volume: Volume of sentiment data points
            key_phrases: List of key phrases or topics
            source_details: Additional details about the source
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            key_phrases_json = json.dumps(key_phrases or [])
            source_details_json = json.dumps(source_details or {})
            
            cursor.execute(
                '''
                INSERT OR REPLACE INTO sentiment_analysis 
                (symbol, timestamp, source, sentiment_score, sentiment_label, 
                volume, key_phrases, source_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    symbol,
                    now,
                    source,
                    sentiment_score,
                    sentiment_label,
                    volume,
                    key_phrases_json,
                    source_details_json
                )
            )
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved sentiment analysis for {symbol} from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving sentiment analysis: {str(e)}")
            return False
    
    def get_sentiment_analysis(self,
                             symbol: str,
                             source: Optional[str] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get sentiment analysis for a symbol
        
        Args:
            symbol: Stock symbol
            source: Filter by source (optional)
            limit: Maximum number of results to retrieve
            
        Returns:
            List of sentiment analysis results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if source:
                cursor.execute(
                    '''
                    SELECT * FROM sentiment_analysis
                    WHERE symbol = ? AND source = ?
                    ORDER BY timestamp DESC LIMIT ?
                    ''',
                    (symbol, source, limit)
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM sentiment_analysis
                    WHERE symbol = ?
                    ORDER BY timestamp DESC LIMIT ?
                    ''',
                    (symbol, limit)
                )
            
            results = cursor.fetchall()
            conn.close()
            
            sentiment_results = []
            for row in results:
                result = dict(row)
                result['key_phrases'] = json.loads(result['key_phrases'])
                result['source_details'] = json.loads(result['source_details'])
                sentiment_results.append(result)
                
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error retrieving sentiment analysis: {str(e)}")
            return []

    def get_symbols_with_analysis(self, 
                               analysis_type: Optional[str] = None,
                               since: Optional[str] = None) -> List[str]:
        """
        Get list of all symbols that have analysis data
        
        Args:
            analysis_type: Filter by analysis type (optional)
            since: Only include analyses since this timestamp (ISO format)
            
        Returns:
            List of symbols
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if analysis_type and since:
                cursor.execute(
                    '''
                    SELECT DISTINCT symbol FROM stock_analysis
                    WHERE analysis_type = ? AND timestamp >= ?
                    ''',
                    (analysis_type, since)
                )
            elif analysis_type:
                cursor.execute(
                    '''
                    SELECT DISTINCT symbol FROM stock_analysis
                    WHERE analysis_type = ?
                    ''',
                    (analysis_type,)
                )
            elif since:
                cursor.execute(
                    '''
                    SELECT DISTINCT symbol FROM stock_analysis
                    WHERE timestamp >= ?
                    ''',
                    (since,)
                )
            else:
                cursor.execute('SELECT DISTINCT symbol FROM stock_analysis')
            
            results = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving symbols with analysis: {str(e)}")
            return []

    def purge_old_data(self, days_to_keep: int = 30) -> bool:
        """
        Purge data older than the specified number of days
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - datetime.timedelta(days=days_to_keep)).isoformat()
            
            # Delete old data from all tables
            tables = [
                'stock_analysis',
                'stock_selection_history',
                'strategy_selection_history',
                'market_regime_history',
                'sentiment_analysis'
            ]
            
            for table in tables:
                cursor.execute(
                    f'''
                    DELETE FROM {table}
                    WHERE timestamp < ?
                    ''',
                    (cutoff_date,)
                )
            
            conn.commit()
            conn.close()
            logger.info(f"Purged data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            logger.error(f"Error purging old data: {str(e)}")
            return False 