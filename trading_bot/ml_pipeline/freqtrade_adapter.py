"""
FreqTrade Strategy Adapter

Provides utilities for importing and adapting FreqTrade strategies for use
in the multi-model prediction pipeline.
"""

import os
import sys
import inspect
import importlib.util
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default FreqTrade strategy interface for type hints
class IFreqtradeStrategy:
    """Interface defining expected FreqTrade strategy methods"""
    minimal_roi: Dict[str, float]
    stoploss: float
    timeframe: str
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Add indicators to dataframe"""
        return dataframe
    
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Add buy signals to dataframe"""
        return dataframe
    
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Add sell signals to dataframe"""
        return dataframe

class FreqTradeStrategyAdapter:
    """
    Adapter for FreqTrade strategies to be used in the multi-model prediction pipeline
    
    This adapter allows importing FreqTrade strategies from Python files or modules
    and using them as signal generators in our multi-model prediction system.
    """
    
    def __init__(self, 
                strategy_path: Optional[str] = None, 
                strategy_class: Optional[Type] = None,
                strategy_config: Optional[Dict[str, Any]] = None):
        """
        Initialize FreqTrade strategy adapter
        
        Args:
            strategy_path: Path to FreqTrade strategy file or module
            strategy_class: Directly provided strategy class
            strategy_config: Optional configuration for the strategy
        """
        self.strategy_path = strategy_path
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config or {}
        self.strategy_instance = None
        self.strategy_name = ""
        
        # Load strategy if path or class is provided
        if strategy_class:
            self._load_from_class(strategy_class)
        elif strategy_path:
            self._load_from_path(strategy_path)
    
    def _load_from_path(self, path: str):
        """
        Load strategy from file path
        
        Args:
            path: Path to FreqTrade strategy file
        """
        try:
            # Get module name from file path
            module_name = os.path.basename(path).replace('.py', '')
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module spec from {path}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find strategy class in module
            strategy_class = None
            for name, obj in inspect.getmembers(module):
                # Skip if not a class
                if not inspect.isclass(obj):
                    continue
                
                # Check if class has expected methods
                if (hasattr(obj, 'populate_indicators') and 
                    hasattr(obj, 'populate_buy_trend') and 
                    hasattr(obj, 'populate_sell_trend')):
                    strategy_class = obj
                    break
            
            if strategy_class is None:
                raise ImportError(f"No FreqTrade strategy class found in {path}")
                
            # Initialize strategy
            self.strategy_class = strategy_class
            self.strategy_instance = strategy_class()
            self.strategy_name = strategy_class.__name__
            
            logger.info(f"Loaded FreqTrade strategy: {self.strategy_name} from {path}")
            
        except Exception as e:
            logger.error(f"Error loading FreqTrade strategy from {path}: {e}")
            raise
    
    def _load_from_class(self, strategy_class: Type):
        """
        Load strategy from class
        
        Args:
            strategy_class: FreqTrade strategy class
        """
        try:
            # Validate strategy class
            if not (hasattr(strategy_class, 'populate_indicators') and 
                   hasattr(strategy_class, 'populate_buy_trend') and 
                   hasattr(strategy_class, 'populate_sell_trend')):
                raise ValueError("Class does not implement FreqTrade strategy interface")
                
            # Initialize strategy
            self.strategy_class = strategy_class
            self.strategy_instance = strategy_class()
            self.strategy_name = strategy_class.__name__
            
            logger.info(f"Loaded FreqTrade strategy: {self.strategy_name}")
            
        except Exception as e:
            logger.error(f"Error loading FreqTrade strategy class: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame, pair: str = "BTC/USDT") -> pd.DataFrame:
        """
        Prepare dataframe for FreqTrade strategy
        
        Args:
            data: OHLCV dataframe
            pair: Trading pair
            
        Returns:
            Prepared dataframe in FreqTrade format
        """
        if self.strategy_instance is None:
            raise ValueError("No strategy loaded")
            
        try:
            # FreqTrade expects specific column names
            ft_data = data.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in ft_data.columns:
                    # Try case insensitive match
                    for data_col in ft_data.columns:
                        if data_col.lower() == col:
                            ft_data[col] = ft_data[data_col]
                            break
                    else:
                        raise ValueError(f"Required column '{col}' not found in data")
            
            # Ensure datetime index
            if not isinstance(ft_data.index, pd.DatetimeIndex):
                # Try to find a datetime column
                date_cols = [col for col in ft_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    ft_data.set_index(date_cols[0], inplace=True)
                else:
                    # Use existing index but convert to datetime
                    ft_data.index = pd.to_datetime(ft_data.index)
            
            # Prepare metadata dict as expected by FreqTrade
            metadata = {
                'pair': pair,
            }
            
            # Populate indicators
            ft_data = self.strategy_instance.populate_indicators(ft_data, metadata)
            
            return ft_data
            
        except Exception as e:
            logger.error(f"Error preparing data for FreqTrade strategy: {e}")
            raise
    
    def generate_signals(self, data: pd.DataFrame, pair: str = "BTC/USDT") -> pd.DataFrame:
        """
        Generate trading signals using FreqTrade strategy
        
        Args:
            data: OHLCV dataframe
            pair: Trading pair
            
        Returns:
            Dataframe with buy/sell signals
        """
        if self.strategy_instance is None:
            raise ValueError("No strategy loaded")
            
        try:
            # Prepare data for strategy
            ft_data = self.prepare_data(data, pair)
            
            # Prepare metadata dict as expected by FreqTrade
            metadata = {
                'pair': pair,
            }
            
            # Generate buy signals
            ft_data = self.strategy_instance.populate_buy_trend(ft_data, metadata)
            
            # Generate sell signals
            ft_data = self.strategy_instance.populate_sell_trend(ft_data, metadata)
            
            return ft_data
            
        except Exception as e:
            logger.error(f"Error generating signals with FreqTrade strategy: {e}")
            raise
    
    def get_prediction(self, data: pd.DataFrame, pair: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Get prediction from FreqTrade strategy
        
        Args:
            data: OHLCV dataframe
            pair: Trading pair
            
        Returns:
            Dictionary with prediction result
        """
        try:
            # Generate signals
            result_df = self.generate_signals(data, pair)
            
            # Get last row for current prediction
            last_row = result_df.iloc[-1]
            
            # Extract buy/sell signals
            buy_signal = False
            sell_signal = False
            
            # Check for standard FreqTrade columns
            if 'buy' in last_row:
                buy_signal = bool(last_row['buy'])
            elif 'buy_tag' in last_row and not pd.isna(last_row['buy_tag']):
                buy_signal = True
                
            if 'sell' in last_row:
                sell_signal = bool(last_row['sell'])
            elif 'sell_tag' in last_row and not pd.isna(last_row['sell_tag']):
                sell_signal = True
            
            # Convert to our signal format
            if buy_signal and not sell_signal:
                signal = "buy"
                strength = 1.0
            elif sell_signal and not buy_signal:
                signal = "sell"
                strength = -1.0
            else:
                signal = "neutral"
                strength = 0.0
            
            # Extract any additional metrics if available
            confidence = 0.7  # Default confidence for FreqTrade strategies
            metrics = {}
            
            # Extract any numeric indicators that might represent confidence
            for col in result_df.columns:
                if (col.endswith('_prob') or col.endswith('_confidence') or col.endswith('_strength')) and isinstance(last_row[col], (int, float)):
                    metrics[col] = float(last_row[col])
                    # Use as confidence if between 0-1
                    if 0 <= last_row[col] <= 1:
                        confidence = float(last_row[col])
            
            return {
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'raw_buy': buy_signal,
                'raw_sell': sell_signal,
                'metrics': metrics,
                'timestamp': last_row.name if isinstance(last_row.name, datetime) else datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction from FreqTrade strategy: {e}")
            return {
                'signal': 'neutral',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Get strategy parameters
        
        Returns:
            Dictionary with strategy parameters
        """
        if self.strategy_instance is None:
            return {}
            
        params = {
            'name': self.strategy_name,
            'type': 'freqtrade'
        }
        
        # Extract common FreqTrade parameters
        for param in ['minimal_roi', 'stoploss', 'timeframe', 'trailing_stop', 'process_only_new_candles']:
            if hasattr(self.strategy_instance, param):
                params[param] = getattr(self.strategy_instance, param)
        
        return params

class FreqTradeStrategyRegistry:
    """
    Registry for FreqTrade strategies
    
    Manages a collection of FreqTrade strategies and provides
    methods for using them in the multi-model prediction pipeline.
    """
    
    def __init__(self, strategies_dir: Optional[str] = None):
        """
        Initialize FreqTrade strategy registry
        
        Args:
            strategies_dir: Optional directory to scan for strategy files
        """
        self.strategies = {}
        self.strategies_dir = strategies_dir
        
        # Scan directory if provided
        if strategies_dir and os.path.exists(strategies_dir):
            self.scan_strategies_dir()
    
    def scan_strategies_dir(self):
        """Scan directory for FreqTrade strategy files"""
        try:
            if not self.strategies_dir or not os.path.exists(self.strategies_dir):
                logger.warning("No strategies directory provided or does not exist")
                return
                
            for filename in os.listdir(self.strategies_dir):
                if filename.endswith('.py'):
                    filepath = os.path.join(self.strategies_dir, filename)
                    try:
                        # Try to load as strategy
                        adapter = FreqTradeStrategyAdapter(strategy_path=filepath)
                        self.strategies[adapter.strategy_name] = adapter
                        logger.info(f"Registered FreqTrade strategy: {adapter.strategy_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {filename} as FreqTrade strategy: {e}")
                        
            logger.info(f"Loaded {len(self.strategies)} FreqTrade strategies from {self.strategies_dir}")
            
        except Exception as e:
            logger.error(f"Error scanning strategies directory: {e}")
    
    def register_strategy(self, 
                         name: Optional[str] = None, 
                         strategy_path: Optional[str] = None,
                         strategy_class: Optional[Type] = None,
                         strategy_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a FreqTrade strategy
        
        Args:
            name: Optional name for the strategy
            strategy_path: Path to strategy file
            strategy_class: Strategy class
            strategy_config: Optional configuration
            
        Returns:
            Name of registered strategy
        """
        if not strategy_path and not strategy_class:
            raise ValueError("Either strategy_path or strategy_class must be provided")
            
        try:
            # Create adapter
            adapter = FreqTradeStrategyAdapter(
                strategy_path=strategy_path, 
                strategy_class=strategy_class,
                strategy_config=strategy_config
            )
            
            # Use provided name or get from strategy
            strategy_name = name or adapter.strategy_name
            
            # Register strategy
            self.strategies[strategy_name] = adapter
            
            logger.info(f"Registered FreqTrade strategy: {strategy_name}")
            
            return strategy_name
            
        except Exception as e:
            logger.error(f"Error registering FreqTrade strategy: {e}")
            raise
    
    def get_strategy(self, name: str) -> Optional[FreqTradeStrategyAdapter]:
        """
        Get FreqTrade strategy by name
        
        Args:
            name: Strategy name
            
        Returns:
            FreqTradeStrategyAdapter or None if not found
        """
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategies
        
        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())
    
    def get_predictions(self, data: pd.DataFrame, pair: str = "BTC/USDT") -> Dict[str, Dict[str, Any]]:
        """
        Get predictions from all registered strategies
        
        Args:
            data: OHLCV dataframe
            pair: Trading pair
            
        Returns:
            Dictionary of strategy name -> prediction
        """
        predictions = {}
        
        for name, adapter in self.strategies.items():
            try:
                prediction = adapter.get_prediction(data, pair)
                predictions[name] = prediction
            except Exception as e:
                logger.error(f"Error getting prediction from strategy {name}: {e}")
                predictions[name] = {
                    'signal': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return predictions

# Utility Functions

def convert_ohlcv_to_freqtrade(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert generic OHLCV data to FreqTrade format
    
    Args:
        data: OHLCV dataframe
        
    Returns:
        Dataframe in FreqTrade format
    """
    ft_data = data.copy()
    
    # Standardize column names (lowercase)
    ft_data.columns = [col.lower() for col in ft_data.columns]
    
    # Rename common columns to FreqTrade expected names
    rename_map = {
        'timestamp': 'date',
        'time': 'date',
        'datetime': 'date',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }
    
    for old, new in rename_map.items():
        if old in ft_data.columns and new not in ft_data.columns:
            ft_data[new] = ft_data[old]
    
    # Ensure date is index if present
    if 'date' in ft_data.columns:
        ft_data.set_index('date', inplace=True)
    
    # Ensure datetime index
    if not isinstance(ft_data.index, pd.DatetimeIndex):
        ft_data.index = pd.to_datetime(ft_data.index)
    
    return ft_data

def import_freqtrade_strategy(path: str) -> FreqTradeStrategyAdapter:
    """
    Import a FreqTrade strategy from file
    
    Args:
        path: Path to strategy file
        
    Returns:
        FreqTradeStrategyAdapter instance
    """
    return FreqTradeStrategyAdapter(strategy_path=path)
