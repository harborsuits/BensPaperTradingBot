#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Storage Module

This module provides functionality for persisting and retrieving historical market data.
"""

import os
import logging
import pandas as pd
import json
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class DataStorage:
    """
    Data storage for persisting market data.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize data storage.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.market_data_dir = self.base_dir / "market_data"
        self.option_data_dir = self.base_dir / "option_data"
        
        # Create directories if they don't exist
        self.market_data_dir.mkdir(parents=True, exist_ok=True)
        self.option_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data storage initialized with base directory: {self.base_dir}")
    
    def save_market_data(self, symbol: str, data: pd.DataFrame, 
                        data_type: str = "ohlcv") -> bool:
        """
        Save market data to storage.
        
        Args:
            symbol: Symbol for the data
            data: DataFrame with market data
            data_type: Type of data (ohlcv, indicators, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data.empty:
                logger.warning(f"Cannot save empty data for {symbol}")
                return False
            
            # Create directory for symbol if it doesn't exist
            symbol_dir = self.market_data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Create file path
            file_path = symbol_dir / f"{data_type}.parquet"
            
            # Save data to parquet format (efficient for time series data)
            data.to_parquet(file_path)
            
            logger.info(f"Saved {data_type} data for {symbol} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving market data for {symbol}: {e}")
            return False
    
    def load_market_data(self, symbol: str, data_type: str = "ohlcv") -> Optional[pd.DataFrame]:
        """
        Load market data from storage.
        
        Args:
            symbol: Symbol to load data for
            data_type: Type of data to load
            
        Returns:
            DataFrame with market data or None if not found
        """
        try:
            # Create file path
            file_path = self.market_data_dir / symbol / f"{data_type}.parquet"
            
            # Check if file exists
            if not file_path.exists():
                logger.info(f"No {data_type} data found for {symbol}")
                return None
            
            # Load data
            data = pd.read_parquet(file_path)
            
            logger.info(f"Loaded {data_type} data for {symbol} from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading market data for {symbol}: {e}")
            return None
    
    def save_option_data(self, symbol: str, expiration_date: str, 
                        data: Dict[str, Any]) -> bool:
        """
        Save option data to storage.
        
        Args:
            symbol: Symbol for the options
            expiration_date: Expiration date string
            data: Option data (potentially complex with DataFrames)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory for symbol if it doesn't exist
            symbol_dir = self.option_data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Create file path with sanitized expiration date
            safe_date = expiration_date.replace('-', '_')
            file_path = symbol_dir / f"{safe_date}.pkl"
            
            # Save data using pickle (to preserve DataFrame objects)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save a metadata file with timestamp for reference
            metadata = {
                'symbol': symbol,
                'expiration': expiration_date,
                'saved_at': datetime.now().isoformat(),
                'format': 'pickle'
            }
            
            meta_path = symbol_dir / f"{safe_date}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Saved option data for {symbol} expiration {expiration_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving option data for {symbol} expiration {expiration_date}: {e}")
            return False
    
    def load_option_data(self, symbol: str, expiration_date: str) -> Optional[Dict[str, Any]]:
        """
        Load option data from storage.
        
        Args:
            symbol: Symbol to load options for
            expiration_date: Expiration date string
            
        Returns:
            Option data or None if not found
        """
        try:
            # Create file path with sanitized expiration date
            safe_date = expiration_date.replace('-', '_')
            file_path = self.option_data_dir / symbol / f"{safe_date}.pkl"
            
            # Check if file exists
            if not file_path.exists():
                logger.info(f"No option data found for {symbol} expiration {expiration_date}")
                return None
            
            # Load data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded option data for {symbol} expiration {expiration_date}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading option data for {symbol} expiration {expiration_date}: {e}")
            return None
    
    def list_available_data(self, data_type: str = "market") -> Dict[str, List[str]]:
        """
        List available data in storage.
        
        Args:
            data_type: Type of data to list ("market" or "option")
            
        Returns:
            Dictionary mapping symbols to available data types or dates
        """
        result = {}
        
        try:
            # Determine directory to search
            if data_type == "market":
                base_dir = self.market_data_dir
            elif data_type == "option":
                base_dir = self.option_data_dir
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return result
            
            # Check if directory exists
            if not base_dir.exists():
                return result
            
            # List all symbols (directories)
            for symbol_dir in base_dir.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name
                    
                    # List available data
                    if data_type == "market":
                        # List available data types (files)
                        data_types = []
                        for file_path in symbol_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.parquet':
                                data_types.append(file_path.stem)
                        
                        result[symbol] = data_types
                    
                    else:  # option data
                        # List available expiration dates
                        dates = []
                        for file_path in symbol_dir.iterdir():
                            if file_path.is_file() and file_path.suffix == '.pkl' and not file_path.name.endswith('_meta.json'):
                                # Convert filename back to date format
                                date_str = file_path.stem.replace('_', '-')
                                dates.append(date_str)
                        
                        result[symbol] = dates
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing available {data_type} data: {e}")
            return result
    
    def delete_market_data(self, symbol: str, data_type: Optional[str] = None) -> bool:
        """
        Delete market data from storage.
        
        Args:
            symbol: Symbol to delete data for
            data_type: Specific data type to delete or None for all
            
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol_dir = self.market_data_dir / symbol
            
            # Check if symbol directory exists
            if not symbol_dir.exists():
                logger.warning(f"No data found for {symbol}")
                return False
            
            # Delete specific data type
            if data_type is not None:
                file_path = symbol_dir / f"{data_type}.parquet"
                
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted {data_type} data for {symbol}")
                    return True
                else:
                    logger.warning(f"No {data_type} data found for {symbol}")
                    return False
            
            # Delete all data for symbol
            else:
                # Delete all files in directory
                for file_path in symbol_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                
                # Remove directory
                symbol_dir.rmdir()
                
                logger.info(f"Deleted all market data for {symbol}")
                return True
            
        except Exception as e:
            logger.error(f"Error deleting market data for {symbol}: {e}")
            return False 