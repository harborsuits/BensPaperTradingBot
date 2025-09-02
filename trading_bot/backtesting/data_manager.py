#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data manager for backtesting with machine learning.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import sqlite3

from trading_bot.indicators.factory import IndicatorFactory

logger = logging.getLogger(__name__)

class BacktestDataManager:
    """
    Data manager for backtesting with machine learning.
    
    This class handles loading, processing, and preparing market data for
    machine learning models in backtesting.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the BacktestDataManager.
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        self.indicator_factory = IndicatorFactory()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def load_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        custom_data_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load market data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data
            custom_data_path: Path to custom data file
        
        Returns:
            DataFrame with market data
        """
        if custom_data_path and os.path.exists(custom_data_path):
            # Load from custom path
            df = pd.read_csv(custom_data_path)
            
            # Convert date column to datetime
            date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.rename(columns={date_col: 'datetime'})
            elif 'datetime' not in df.columns:
                raise ValueError("DataFrame must have a datetime column")
            
            # Filter by date range
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            
        else:
            # Construct standard data path
            filename = f"{symbol}_{timeframe}.csv"
            data_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load data
            df = pd.read_csv(data_path)
            
            # Convert date column to datetime
            date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.rename(columns={date_col: 'datetime'})
            elif 'datetime' not in df.columns:
                raise ValueError("DataFrame must have a datetime column")
            
            # Filter by date range
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        return df
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with market data
            indicators: List of indicators to add
                Each indicator is a dict with:
                - name: Name of the indicator
                - params: Parameters for the indicator
        
        Returns:
            DataFrame with added indicators
        """
        df_copy = df.copy()
        
        for indicator_config in indicators:
            indicator_name = indicator_config.get('name')
            indicator_params = indicator_config.get('params', {})
            
            try:
                # Get indicator instance from factory
                indicator = self.indicator_factory.create_indicator(indicator_name, **indicator_params)
                
                # Calculate indicator values
                indicator_df = indicator.calculate(df_copy)
                
                # Merge with original data
                df_copy = pd.concat([df_copy, indicator_df], axis=1)
                
                # Remove duplicated columns
                df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]
                
            except Exception as e:
                logger.error(f"Error adding indicator {indicator_name}: {str(e)}")
        
        return df_copy
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_type: str = 'direction',
        lookahead: int = 1,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for machine learning.
        
        Args:
            df: DataFrame with market data
            target_type: Type of target variable
                - 'direction': Binary classification (up/down)
                - 'return': Regression (future return)
                - 'classification': Multi-class based on threshold
            lookahead: Number of periods to look ahead
            threshold: Threshold for classification targets
        
        Returns:
            DataFrame with target variable
        """
        df_copy = df.copy()
        
        if 'close' not in df_copy.columns:
            raise ValueError("DataFrame must have a 'close' column")
        
        # Calculate future price
        df_copy['future_price'] = df_copy['close'].shift(-lookahead)
        
        # Calculate price change and return
        df_copy['price_change'] = df_copy['future_price'] - df_copy['close']
        df_copy['return'] = df_copy['price_change'] / df_copy['close']
        
        # Create target based on type
        if target_type == 'direction':
            # Binary classification (1 for up, 0 for down)
            df_copy['target'] = (df_copy['price_change'] > threshold).astype(int)
            
        elif target_type == 'return':
            # Regression target
            df_copy['target'] = df_copy['return']
            
        elif target_type == 'classification':
            # Multi-class classification
            conditions = [
                (df_copy['return'] > threshold),
                (df_copy['return'] < -threshold),
                ((df_copy['return'] >= -threshold) & (df_copy['return'] <= threshold))
            ]
            choices = [1, -1, 0]  # 1 for up, -1 for down, 0 for neutral
            df_copy['target'] = np.select(conditions, choices, default=0)
            
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        # Remove rows with NaN targets
        df_copy = df_copy.dropna(subset=['target'])
        
        return df_copy
    
    def split_data(
        self,
        df: pd.DataFrame,
        method: str = 'time',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame with market data
            method: Method for splitting data
                - 'time': Chronological split
                - 'random': Random split
                - 'walk_forward': Walk-forward validation
            test_size: Size of the test set
            val_size: Size of the validation set
            random_state: Random state for reproducibility
            n_splits: Number of splits for walk-forward validation
        
        Returns:
            Dictionary with split indices and DataFrames
        """
        # Get feature columns (exclude datetime and target)
        feature_cols = [col for col in df.columns if col not in ['datetime', 'target', 'future_price', 'price_change', 'return']]
        
        # Convert to numpy arrays
        X = df[feature_cols].values
        y = df['target'].values
        
        if method == 'time':
            # Chronological split
            train_size = 1 - test_size - val_size
            total_samples = len(df)
            
            train_end = int(total_samples * train_size)
            val_end = int(total_samples * (train_size + val_size))
            
            train_idx = list(range(0, train_end))
            val_idx = list(range(train_end, val_end))
            test_idx = list(range(val_end, total_samples))
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
        elif method == 'random':
            # Random split
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
            )
            
            # Get indices for train, val, test
            train_idx = []
            val_idx = []
            test_idx = []
            
            for i in range(len(df)):
                if any(np.array_equal(X[i], x) for x in X_train):
                    train_idx.append(i)
                elif any(np.array_equal(X[i], x) for x in X_val):
                    val_idx.append(i)
                elif any(np.array_equal(X[i], x) for x in X_test):
                    test_idx.append(i)
            
        elif method == 'walk_forward':
            # Walk-forward validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Get the last split for final evaluation
            train_idx = []
            test_idx = []
            
            for train_index, test_index in tscv.split(X):
                train_idx = train_index
                test_idx = test_index
            
            # Create validation set from the end of training
            val_size_samples = int(len(train_idx) * val_size)
            val_idx = train_idx[-val_size_samples:]
            train_idx = train_idx[:-val_size_samples]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
        else:
            raise ValueError(f"Unsupported split method: {method}")
        
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols
        }
    
    def scale_features(
        self,
        split_data: Dict[str, Any],
        method: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Scale features for machine learning.
        
        Args:
            split_data: Dictionary with split data
            method: Method for scaling features
                - 'standard': StandardScaler
                - 'minmax': MinMaxScaler
                - 'robust': RobustScaler
                - None: No scaling
        
        Returns:
            Dictionary with scaled data
        """
        result = split_data.copy()
        
        if method is None or method.lower() == 'none':
            # No scaling
            return result
        
        # Get scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        # Fit scaler on training data
        scaler.fit(result['X_train'])
        
        # Transform all sets
        result['X_train'] = scaler.transform(result['X_train'])
        result['X_val'] = scaler.transform(result['X_val'])
        result['X_test'] = scaler.transform(result['X_test'])
        
        # Store scaler for future use
        result['scaler'] = scaler
        
        return result
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df: DataFrame with market data
            method: Method for handling missing values
                - 'drop': Drop rows with missing values
                - 'forward': Forward fill
                - 'backward': Backward fill
                - 'zero': Fill with zeros
                - 'mean': Fill with column mean
        
        Returns:
            DataFrame with handled missing values
        """
        df_copy = df.copy()
        
        if method == 'drop':
            df_copy = df_copy.dropna()
            
        elif method == 'forward':
            df_copy = df_copy.fillna(method='ffill')
            
        elif method == 'backward':
            df_copy = df_copy.fillna(method='bfill')
            
        elif method == 'zero':
            df_copy = df_copy.fillna(0)
            
        elif method == 'mean':
            for col in df_copy.columns:
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                else:
                    # For non-numeric columns, use forward fill
                    df_copy[col] = df_copy[col].fillna(method='ffill')
            
        else:
            raise ValueError(f"Unsupported missing value handling method: {method}")
        
        return df_copy
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        indicators: List[Dict[str, Any]] = [],
        target_type: str = 'direction',
        target_lookahead: int = 1,
        target_threshold: float = 0.0,
        split_method: str = 'time',
        test_size: float = 0.2,
        val_size: float = 0.1,
        scaling_method: str = 'standard',
        missing_value_method: str = 'drop',
        custom_features: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Prepare data for machine learning.
        
        Args:
            df: DataFrame with market data
            indicators: List of indicators to add
            target_type: Type of target variable
            target_lookahead: Number of periods to look ahead
            target_threshold: Threshold for classification targets
            split_method: Method for splitting data
            test_size: Size of the test set
            val_size: Size of the validation set
            scaling_method: Method for scaling features
            missing_value_method: Method for handling missing values
            custom_features: Configuration for custom features
        
        Returns:
            Dictionary with prepared data
        """
        # Store original data
        original_data = df.copy()
        
        # Step 1: Add technical indicators
        logger.info("Adding technical indicators")
        df_indicators = self.add_technical_indicators(df, indicators)
        
        # Step 2: Add custom features if specified
        if custom_features:
            logger.info("Adding custom features")
            # Implementation would depend on the structure of custom_features
            # For now, just pass through
            df_features = df_indicators
        else:
            df_features = df_indicators
        
        # Step 3: Create target variable
        logger.info(f"Creating target variable (type: {target_type}, lookahead: {target_lookahead})")
        df_target = self.create_target_variable(
            df_features,
            target_type=target_type,
            lookahead=target_lookahead,
            threshold=target_threshold
        )
        
        # Step 4: Handle missing values
        logger.info(f"Handling missing values (method: {missing_value_method})")
        df_clean = self.handle_missing_values(df_target, method=missing_value_method)
        
        # Step 5: Split data
        logger.info(f"Splitting data (method: {split_method})")
        split_result = self.split_data(
            df_clean,
            method=split_method,
            test_size=test_size,
            val_size=val_size
        )
        
        # Step 6: Scale features
        logger.info(f"Scaling features (method: {scaling_method})")
        scaled_result = self.scale_features(split_result, method=scaling_method)
        
        # Combine results
        result = {
            "original_data": original_data,
            "processed_data": df_clean,
            "train_indices": scaled_result['train_idx'],
            "val_indices": scaled_result['val_idx'],
            "test_indices": scaled_result['test_idx'],
            "X_train": scaled_result['X_train'],
            "y_train": scaled_result['y_train'],
            "X_val": scaled_result['X_val'],
            "y_val": scaled_result['y_val'],
            "X_test": scaled_result['X_test'],
            "y_test": scaled_result['y_test'],
            "feature_names": scaled_result['feature_cols'],
            "target_type": target_type,
            "scaler": scaled_result.get('scaler')
        }
        
        return result
    
    def save_processed_data(
        self,
        data_result: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Save processed data to disk.
        
        Args:
            data_result: Dictionary with processed data
            filename: Name for the saved file
        
        Returns:
            Path to the saved file
        """
        # Create folder if it doesn't exist
        processed_dir = os.path.join(self.data_dir, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Create save path
        save_path = os.path.join(processed_dir, filename)
        
        # Prepare data for saving
        save_data = {
            "processed_data": data_result["processed_data"].to_dict(orient='records'),
            "train_indices": data_result["train_indices"],
            "val_indices": data_result["val_indices"],
            "test_indices": data_result["test_indices"],
            "feature_names": data_result["feature_names"],
            "target_type": data_result["target_type"]
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        return save_path
    
    def load_processed_data(
        self,
        filename: str
    ) -> Dict[str, Any]:
        """
        Load processed data from disk.
        
        Args:
            filename: Name of the file to load
        
        Returns:
            Dictionary with processed data
        """
        # Create load path
        load_path = os.path.join(self.data_dir, "processed", filename)
        
        # Check if file exists
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Processed data file not found: {load_path}")
        
        # Load from file
        with open(load_path, 'r') as f:
            load_data = json.load(f)
        
        # Convert to DataFrame
        processed_data = pd.DataFrame(load_data["processed_data"])
        
        # Get indices
        train_indices = load_data["train_indices"]
        val_indices = load_data["val_indices"]
        test_indices = load_data["test_indices"]
        
        # Extract features and target
        feature_names = load_data["feature_names"]
        X = processed_data[feature_names].values
        y = processed_data['target'].values
        
        # Create result dictionary
        result = {
            "processed_data": processed_data,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "X_train": X[train_indices],
            "y_train": y[train_indices],
            "X_val": X[val_indices],
            "y_val": y[val_indices],
            "X_test": X[test_indices],
            "y_test": y[test_indices],
            "feature_names": feature_names,
            "target_type": load_data["target_type"]
        }
        
        return result

class DataManager:
    """
    Responsible for collecting, storing and retrieving backtest data.
    Supports both JSON file storage and SQLite database.
    """
    
    def __init__(
        self, 
        save_path: str = "data/backtest_history.json",
        use_sqlite: bool = False,
        db_path: str = "data/backtest_history.db"
    ):
        """
        Initialize the data manager.
        
        Args:
            save_path: Path to save the JSON data
            use_sqlite: Whether to use SQLite for storage
            db_path: Path to SQLite database file
        """
        self.save_path = save_path
        self.use_sqlite = use_sqlite
        self.db_path = db_path
        self.data = []
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if use_sqlite:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._initialize_db()
        
        logger.info(f"Initialized DataManager with save_path={save_path}, use_sqlite={use_sqlite}")
    
    def _initialize_db(self):
        """Initialize SQLite database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                symbol TEXT,
                type TEXT,
                quantity REAL,
                price REAL,
                pnl REAL,
                fees REAL,
                slippage REAL,
                market_context TEXT,
                execution_metrics TEXT
            )
            ''')
            
            # Create signals table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                symbol TEXT,
                signal_type TEXT,
                strength REAL,
                direction TEXT,
                confidence REAL,
                market_context TEXT
            )
            ''')
            
            # Create portfolio_snapshots table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_value REAL,
                cash_value REAL,
                holdings TEXT,
                daily_return REAL,
                daily_volatility REAL,
                drawdown REAL
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Initialized SQLite database tables")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def log_trade(self, trade_info: Dict[str, Any]):
        """
        Log a trade execution.
        
        Args:
            trade_info: Dictionary containing trade information
        """
        # Add timestamp if not provided
        if 'timestamp' not in trade_info:
            trade_info['timestamp'] = datetime.now().isoformat()
        
        if self.use_sqlite:
            self._log_trade_to_db(trade_info)
        else:
            self.data.append({'type': 'trade', 'data': trade_info})
        
        logger.debug(f"Logged trade: {trade_info.get('symbol')} {trade_info.get('type')}")
    
    def _log_trade_to_db(self, trade_info: Dict[str, Any]):
        """Log trade to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert dict objects to JSON strings
            market_context = json.dumps(trade_info.get('market_context', {}))
            execution_metrics = json.dumps(trade_info.get('execution_metrics', {}))
            
            cursor.execute('''
            INSERT INTO trades (
                timestamp, strategy, symbol, type, quantity, price, pnl, fees, slippage, market_context, execution_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_info.get('timestamp'),
                trade_info.get('strategy'),
                trade_info.get('symbol'),
                trade_info.get('type'),
                trade_info.get('quantity'),
                trade_info.get('price'),
                trade_info.get('pnl'),
                trade_info.get('fees'),
                trade_info.get('slippage'),
                market_context,
                execution_metrics
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging trade to database: {str(e)}")
    
    def log_signal(self, signal_info: Dict[str, Any]):
        """
        Log a strategy signal.
        
        Args:
            signal_info: Dictionary containing signal information
        """
        # Add timestamp if not provided
        if 'timestamp' not in signal_info:
            signal_info['timestamp'] = datetime.now().isoformat()
        
        if self.use_sqlite:
            self._log_signal_to_db(signal_info)
        else:
            self.data.append({'type': 'signal', 'data': signal_info})
        
        logger.debug(f"Logged signal: {signal_info.get('symbol')} {signal_info.get('signal_type')}")
    
    def _log_signal_to_db(self, signal_info: Dict[str, Any]):
        """Log signal to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert dict objects to JSON strings
            market_context = json.dumps(signal_info.get('market_context', {}))
            
            cursor.execute('''
            INSERT INTO signals (
                timestamp, strategy, symbol, signal_type, strength, direction, confidence, market_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_info.get('timestamp'),
                signal_info.get('strategy'),
                signal_info.get('symbol'),
                signal_info.get('signal_type'),
                signal_info.get('strength'),
                signal_info.get('direction'),
                signal_info.get('confidence'),
                market_context
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging signal to database: {str(e)}")
    
    def log_portfolio_snapshot(self, snapshot_info: Dict[str, Any]):
        """
        Log a portfolio snapshot.
        
        Args:
            snapshot_info: Dictionary containing portfolio snapshot information
        """
        # Add timestamp if not provided
        if 'timestamp' not in snapshot_info:
            snapshot_info['timestamp'] = datetime.now().isoformat()
        
        if self.use_sqlite:
            self._log_portfolio_snapshot_to_db(snapshot_info)
        else:
            self.data.append({'type': 'portfolio_snapshot', 'data': snapshot_info})
        
        logger.debug(f"Logged portfolio snapshot: {snapshot_info.get('timestamp')}")
    
    def _log_portfolio_snapshot_to_db(self, snapshot_info: Dict[str, Any]):
        """Log portfolio snapshot to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert holdings dict to JSON string
            holdings = json.dumps(snapshot_info.get('holdings', {}))
            
            cursor.execute('''
            INSERT INTO portfolio_snapshots (
                timestamp, total_value, cash_value, holdings, daily_return, daily_volatility, drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot_info.get('timestamp'),
                snapshot_info.get('total_value'),
                snapshot_info.get('cash_value'),
                holdings,
                snapshot_info.get('daily_return'),
                snapshot_info.get('daily_volatility'),
                snapshot_info.get('drawdown')
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging portfolio snapshot to database: {str(e)}")
    
    def log_custom_data(self, data_type: str, data: Dict[str, Any]):
        """
        Log custom data.
        
        Args:
            data_type: Type of data
            data: Dictionary containing data
        """
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        if self.use_sqlite:
            logger.warning("Custom data logging not supported with SQLite. Use JSON storage instead.")
        
        self.data.append({'type': data_type, 'data': data})
        logger.debug(f"Logged custom data: {data_type}")
    
    def save(self):
        """Save data to file or database"""
        if not self.use_sqlite:
            try:
                with open(self.save_path, "w") as f:
                    json.dump(self.data, f, indent=2)
                logger.info(f"Saved data to {self.save_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving data: {str(e)}")
                return False
        return True  # For SQLite, data is saved on each log operation
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Returns:
            List of data entries
        """
        if not self.use_sqlite:
            if os.path.exists(self.save_path):
                try:
                    with open(self.save_path, "r") as f:
                        self.data = json.load(f)
                    logger.info(f"Loaded data from {self.save_path}")
                    return self.data
                except Exception as e:
                    logger.error(f"Error loading data: {str(e)}")
                    return []
            else:
                logger.warning(f"Data file {self.save_path} does not exist")
                return []
        else:
            return self._load_from_db()
    
    def _load_from_db(self) -> List[Dict[str, Any]]:
        """Load data from SQLite database"""
        result = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Load trades
            cursor.execute("SELECT * FROM trades")
            trades = cursor.fetchall()
            for trade in trades:
                trade_dict = dict(trade)
                # Parse JSON strings
                trade_dict['market_context'] = json.loads(trade_dict['market_context'])
                trade_dict['execution_metrics'] = json.loads(trade_dict['execution_metrics'])
                result.append({'type': 'trade', 'data': trade_dict})
            
            # Load signals
            cursor.execute("SELECT * FROM signals")
            signals = cursor.fetchall()
            for signal in signals:
                signal_dict = dict(signal)
                # Parse JSON strings
                signal_dict['market_context'] = json.loads(signal_dict['market_context'])
                result.append({'type': 'signal', 'data': signal_dict})
            
            # Load portfolio snapshots
            cursor.execute("SELECT * FROM portfolio_snapshots")
            snapshots = cursor.fetchall()
            for snapshot in snapshots:
                snapshot_dict = dict(snapshot)
                # Parse JSON strings
                snapshot_dict['holdings'] = json.loads(snapshot_dict['holdings'])
                result.append({'type': 'portfolio_snapshot', 'data': snapshot_dict})
            
            conn.close()
            logger.info(f"Loaded {len(result)} records from database")
            
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
        
        return result
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get all trade data.
        
        Returns:
            List of trade entries
        """
        if not self.use_sqlite:
            return [entry['data'] for entry in self.data if entry['type'] == 'trade']
        else:
            return self._get_trades_from_db()
    
    def _get_trades_from_db(self) -> List[Dict[str, Any]]:
        """Get trades from SQLite database"""
        result = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM trades")
            trades = cursor.fetchall()
            for trade in trades:
                trade_dict = dict(trade)
                # Parse JSON strings
                trade_dict['market_context'] = json.loads(trade_dict['market_context'])
                trade_dict['execution_metrics'] = json.loads(trade_dict['execution_metrics'])
                result.append(trade_dict)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting trades from database: {str(e)}")
        
        return result
    
    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Get all signal data.
        
        Returns:
            List of signal entries
        """
        if not self.use_sqlite:
            return [entry['data'] for entry in self.data if entry['type'] == 'signal']
        else:
            return self._get_signals_from_db()
    
    def _get_signals_from_db(self) -> List[Dict[str, Any]]:
        """Get signals from SQLite database"""
        result = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM signals")
            signals = cursor.fetchall()
            for signal in signals:
                signal_dict = dict(signal)
                # Parse JSON strings
                signal_dict['market_context'] = json.loads(signal_dict['market_context'])
                result.append(signal_dict)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting signals from database: {str(e)}")
        
        return result
    
    def get_portfolio_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get all portfolio snapshot data.
        
        Returns:
            List of portfolio snapshot entries
        """
        if not self.use_sqlite:
            return [entry['data'] for entry in self.data if entry['type'] == 'portfolio_snapshot']
        else:
            return self._get_portfolio_snapshots_from_db()
    
    def _get_portfolio_snapshots_from_db(self) -> List[Dict[str, Any]]:
        """Get portfolio snapshots from SQLite database"""
        result = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM portfolio_snapshots")
            snapshots = cursor.fetchall()
            for snapshot in snapshots:
                snapshot_dict = dict(snapshot)
                # Parse JSON strings
                snapshot_dict['holdings'] = json.loads(snapshot_dict['holdings'])
                result.append(snapshot_dict)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting portfolio snapshots from database: {str(e)}")
        
        return result
    
    def get_data_as_dataframe(self, data_type: str = None) -> pd.DataFrame:
        """
        Get data as a pandas DataFrame.
        
        Args:
            data_type: Type of data to get (trade, signal, portfolio_snapshot, or None for all)
            
        Returns:
            DataFrame containing requested data
        """
        try:
            if not self.use_sqlite:
                if data_type:
                    filtered_data = [entry['data'] for entry in self.data if entry['type'] == data_type]
                    return pd.DataFrame(filtered_data)
                else:
                    # For all data, we need to restructure a bit
                    all_data = []
                    for entry in self.data:
                        entry_data = entry['data'].copy()
                        entry_data['data_type'] = entry['type']
                        all_data.append(entry_data)
                    return pd.DataFrame(all_data)
            else:
                if data_type == 'trade':
                    return pd.DataFrame(self._get_trades_from_db())
                elif data_type == 'signal':
                    return pd.DataFrame(self._get_signals_from_db())
                elif data_type == 'portfolio_snapshot':
                    return pd.DataFrame(self._get_portfolio_snapshots_from_db())
                else:
                    # Combine all data
                    trades_df = pd.DataFrame(self._get_trades_from_db())
                    if not trades_df.empty:
                        trades_df['data_type'] = 'trade'
                    
                    signals_df = pd.DataFrame(self._get_signals_from_db())
                    if not signals_df.empty:
                        signals_df['data_type'] = 'signal'
                    
                    snapshots_df = pd.DataFrame(self._get_portfolio_snapshots_from_db())
                    if not snapshots_df.empty:
                        snapshots_df['data_type'] = 'portfolio_snapshot'
                    
                    # Concatenate if any dataframes exist
                    dfs = [df for df in [trades_df, signals_df, snapshots_df] if not df.empty]
                    if dfs:
                        return pd.concat(dfs, ignore_index=True)
                    else:
                        return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def clear(self):
        """Clear all data"""
        self.data = []
        
        if self.use_sqlite:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM trades")
                cursor.execute("DELETE FROM signals")
                cursor.execute("DELETE FROM portfolio_snapshots")
                
                conn.commit()
                conn.close()
                logger.info("Cleared all data from database")
            except Exception as e:
                logger.error(f"Error clearing database: {str(e)}")
        
        logger.info("Cleared all data") 