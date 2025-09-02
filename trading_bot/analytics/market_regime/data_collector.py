"""
Market Regime Data Collector

This module provides tools for collecting and storing market data and performance
metrics to improve regime classification over time.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime, timedelta
import threading
import time

# Import local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.core.event_bus import EventBus, Event

logger = logging.getLogger(__name__)

class RegimeDataCollector:
    """
    Collects and stores market data, regime classifications, and performance metrics
    to improve regime detection and parameter optimization over time.
    
    Features:
    - Saves OHLCV data with regime labels
    - Tracks regime transition performance
    - Captures feature importance for classification
    - Prepares datasets for ML model training
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        data_dir: str = "data/market_regime",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data collector.
        
        Args:
            event_bus: System event bus for event subscription
            data_dir: Directory to store collected data
            config: Configuration parameters
        """
        self.event_bus = event_bus
        self.data_dir = data_dir
        self.config = config or {}
        
        # Create necessary directories
        os.makedirs(os.path.join(self.data_dir, "regime_data"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "performance"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "transitions"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "features"), exist_ok=True)
        
        # Collection state
        self.collecting = False
        self._collection_thread = None
        self._stop_event = threading.Event()
        
        # Collection settings
        self.collection_interval = self.config.get("collection_interval_seconds", 3600)  # 1 hour
        self.retention_days = self.config.get("data_retention_days", 365)  # 1 year
        self.symbols_to_collect = self.config.get("symbols_to_collect", [])
        self.timeframes_to_collect = self.config.get("timeframes_to_collect", ["1d", "4h", "1h"])
        
        # Data cache
        self.regime_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.performance_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Regime Data Collector initialized")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for system events."""
        try:
            # Register for regime change events
            self.event_bus.register("market_regime_change", self._handle_regime_change)
            
            # Register for trade events
            self.event_bus.register("trade_closed", self._handle_trade_closed)
            
            # Register for strategy adaptation events
            self.event_bus.register("strategy_adapted", self._handle_strategy_adapted)
            
            # Register for price update events
            self.event_bus.register("price_update", self._handle_price_update)
            
            logger.info("Registered data collector event handlers")
            
        except Exception as e:
            logger.error(f"Error registering event handlers: {str(e)}")
    
    def start_collection(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Start the data collection process.
        
        Args:
            symbols: Optional list of symbols to collect data for
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                if self.collecting:
                    logger.warning("Data collection already running")
                    return True
                
                logger.info("Starting regime data collection")
                
                # Update symbols if provided
                if symbols:
                    self.symbols_to_collect = symbols
                
                # Start collection thread
                self._stop_event.clear()
                self._collection_thread = threading.Thread(
                    target=self._collection_loop,
                    name="RegimeDataCollectionThread",
                    daemon=True
                )
                self._collection_thread.start()
                
                self.collecting = True
                logger.info(f"Regime data collection started for {len(self.symbols_to_collect)} symbols")
                return True
                
            except Exception as e:
                logger.error(f"Error starting data collection: {str(e)}")
                return False
    
    def stop_collection(self) -> bool:
        """
        Stop the data collection process.
        
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                if not self.collecting:
                    return True
                
                logger.info("Stopping regime data collection")
                
                # Stop collection thread
                self._stop_event.set()
                
                if self._collection_thread and self._collection_thread.is_alive():
                    self._collection_thread.join(timeout=10)
                
                self.collecting = False
                
                # Save any remaining data
                self._save_all_data()
                
                logger.info("Regime data collection stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping data collection: {str(e)}")
                return False
    
    def _collection_loop(self) -> None:
        """Background thread for periodic data collection."""
        last_save_time = datetime.now()
        
        try:
            while not self._stop_event.is_set():
                try:
                    current_time = datetime.now()
                    
                    # Save data periodically
                    if (current_time - last_save_time).total_seconds() >= self.collection_interval:
                        self._save_all_data()
                        last_save_time = current_time
                    
                    # Sleep until next collection
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in collection loop: {str(e)}")
                    time.sleep(300)  # Sleep longer on error
            
        except Exception as e:
            logger.error(f"Fatal error in collection thread: {str(e)}")
        finally:
            logger.info("Data collection thread exiting")
    
    def _save_all_data(self) -> None:
        """Save all collected data to disk."""
        with self._lock:
            try:
                # Save regime data
                for symbol, timeframe_data in self.regime_data_cache.items():
                    for timeframe, df in timeframe_data.items():
                        self._save_regime_data(symbol, timeframe, df)
                
                # Save performance data
                for strategy_id, regime_data in self.performance_cache.items():
                    for regime_str, performance_list in regime_data.items():
                        self._save_performance_data(strategy_id, regime_str, performance_list)
                
                # Clean up old data
                self._cleanup_old_data()
                
                logger.info("Saved all collected data")
                
            except Exception as e:
                logger.error(f"Error saving data: {str(e)}")
    
    def _save_regime_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """
        Save regime data for a symbol and timeframe.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            df: DataFrame with OHLCV and regime data
        """
        try:
            if df.empty:
                return
            
            # Create path
            file_path = os.path.join(
                self.data_dir, "regime_data", 
                f"{symbol}_{timeframe}_regime_data.csv"
            )
            
            # Save to CSV
            df.to_csv(file_path, index=True)
            
            logger.debug(f"Saved regime data for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error saving regime data: {str(e)}")
    
    def _save_performance_data(self, strategy_id: str, regime_str: str, performance_list: List[Dict[str, Any]]) -> None:
        """
        Save performance data for a strategy and regime.
        
        Args:
            strategy_id: Strategy identifier
            regime_str: Regime type as string
            performance_list: List of performance records
        """
        try:
            if not performance_list:
                return
            
            # Create path
            file_path = os.path.join(
                self.data_dir, "performance", 
                f"{strategy_id}_{regime_str}_performance.json"
            )
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(performance_list, f, indent=2)
            
            logger.debug(f"Saved performance data for {strategy_id} in {regime_str}")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data files based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # TODO: Implement cleanup of old data files
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def _handle_regime_change(self, event: Event) -> None:
        """
        Handle market regime change event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            new_regime = data.get('new_regime')
            confidence = data.get('confidence', 0.0)
            features = data.get('features', {})
            
            if not symbol or not timeframe or not new_regime:
                return
            
            # Convert string to enum if needed
            if isinstance(new_regime, str):
                try:
                    new_regime = MarketRegimeType(new_regime)
                except ValueError:
                    logger.warning(f"Unknown regime type: {new_regime}")
                    return
            
            # Record regime transition
            self._record_regime_transition(
                symbol, timeframe, new_regime, confidence, features
            )
            
        except Exception as e:
            logger.error(f"Error handling regime change event: {str(e)}")
    
    def _handle_trade_closed(self, event: Event) -> None:
        """
        Handle trade closed event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            trade = data.get('trade')
            
            if not trade:
                return
            
            # Record trade performance
            self._record_trade_performance(trade)
            
        except Exception as e:
            logger.error(f"Error handling trade closed event: {str(e)}")
    
    def _handle_strategy_adapted(self, event: Event) -> None:
        """
        Handle strategy adaptation event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            strategy_id = data.get('strategy_id')
            regime = data.get('regime')
            parameters = data.get('parameters')
            
            if not strategy_id or not regime or not parameters:
                return
            
            # Record strategy adaptation
            self._record_strategy_adaptation(strategy_id, regime, parameters)
            
        except Exception as e:
            logger.error(f"Error handling strategy adaptation event: {str(e)}")
    
    def _handle_price_update(self, event: Event) -> None:
        """
        Handle price update event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            ohlcv = data.get('ohlcv')
            
            if not symbol or not timeframe or not ohlcv:
                return
            
            # Only collect data for configured symbols and timeframes
            if symbol not in self.symbols_to_collect or timeframe not in self.timeframes_to_collect:
                return
            
            # Record OHLCV data
            self._record_ohlcv_data(symbol, timeframe, ohlcv)
            
        except Exception as e:
            logger.error(f"Error handling price update event: {str(e)}")
    
    def _record_regime_transition(
        self, symbol: str, timeframe: str, 
        regime: MarketRegimeType, confidence: float,
        features: Dict[str, Any]
    ) -> None:
        """
        Record a regime transition.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            regime: New regime type
            confidence: Confidence level
            features: Feature values
        """
        try:
            # Create transition record
            transition = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'regime': regime.value,
                'confidence': confidence,
                'features': features
            }
            
            # Save to transitions directory
            file_path = os.path.join(
                self.data_dir, "transitions", 
                f"{symbol}_{timeframe}_transitions.json"
            )
            
            # Append to file
            transitions = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        transitions = json.load(f)
                except Exception:
                    # Start with empty list if file is corrupt
                    transitions = []
            
            transitions.append(transition)
            
            # Limit size
            max_transitions = self.config.get("max_transitions", 1000)
            if len(transitions) > max_transitions:
                transitions = transitions[-max_transitions:]
            
            # Save
            with open(file_path, 'w') as f:
                json.dump(transitions, f, indent=2)
            
            # Also record feature importance
            self._record_feature_importance(symbol, timeframe, regime, features)
            
        except Exception as e:
            logger.error(f"Error recording regime transition: {str(e)}")
    
    def _record_trade_performance(self, trade: Dict[str, Any]) -> None:
        """
        Record trade performance.
        
        Args:
            trade: Trade data
        """
        try:
            strategy_id = trade.get('strategy_id')
            symbol = trade.get('symbol')
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            profit_loss = trade.get('realized_pnl')
            
            if not strategy_id or not symbol or not entry_time or not exit_time or profit_loss is None:
                return
            
            # Determine regime at entry and exit
            # Note: This would require looking up the regime at those specific times
            # For now, we'll just use a placeholder
            
            # Create performance record
            performance = {
                'trade_id': trade.get('trade_id'),
                'strategy_id': strategy_id,
                'symbol': symbol,
                'entry_time': entry_time.isoformat() if hasattr(entry_time, 'isoformat') else entry_time,
                'exit_time': exit_time.isoformat() if hasattr(exit_time, 'isoformat') else exit_time,
                'profit_loss': profit_loss,
                'return_pct': trade.get('return_pct'),
                'position_size': trade.get('position_size'),
                'side': trade.get('side'),
                'regime': 'unknown'  # Placeholder
            }
            
            # Save to performance cache
            if strategy_id not in self.performance_cache:
                self.performance_cache[strategy_id] = {}
            
            regime_str = performance['regime']
            if regime_str not in self.performance_cache[strategy_id]:
                self.performance_cache[strategy_id][regime_str] = []
            
            self.performance_cache[strategy_id][regime_str].append(performance)
            
        except Exception as e:
            logger.error(f"Error recording trade performance: {str(e)}")
    
    def _record_strategy_adaptation(
        self, strategy_id: str, regime: str, parameters: Dict[str, Any]
    ) -> None:
        """
        Record strategy adaptation.
        
        Args:
            strategy_id: Strategy identifier
            regime: Regime type
            parameters: Adapted parameters
        """
        try:
            # Create adaptation record
            adaptation = {
                'timestamp': datetime.now().isoformat(),
                'strategy_id': strategy_id,
                'regime': regime,
                'parameters': parameters
            }
            
            # Save to file
            file_path = os.path.join(
                self.data_dir, "adaptations", 
                f"{strategy_id}_adaptations.json"
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Append to file
            adaptations = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        adaptations = json.load(f)
                except Exception:
                    # Start with empty list if file is corrupt
                    adaptations = []
            
            adaptations.append(adaptation)
            
            # Limit size
            max_adaptations = self.config.get("max_adaptations", 1000)
            if len(adaptations) > max_adaptations:
                adaptations = adaptations[-max_adaptations:]
            
            # Save
            with open(file_path, 'w') as f:
                json.dump(adaptations, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error recording strategy adaptation: {str(e)}")
    
    def _record_ohlcv_data(
        self, symbol: str, timeframe: str, ohlcv: Dict[str, Any]
    ) -> None:
        """
        Record OHLCV data with regime label.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            ohlcv: OHLCV data
        """
        try:
            # Initialize cache for this symbol and timeframe
            if symbol not in self.regime_data_cache:
                self.regime_data_cache[symbol] = {}
            
            if timeframe not in self.regime_data_cache[symbol]:
                self.regime_data_cache[symbol][timeframe] = pd.DataFrame()
            
            # Create DataFrame with a single row
            timestamp = ohlcv.get('timestamp', datetime.now())
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            
            # Create or update dataframe
            df_cache = self.regime_data_cache[symbol][timeframe]
            
            new_row = pd.DataFrame({
                'open': [ohlcv.get('open')],
                'high': [ohlcv.get('high')],
                'low': [ohlcv.get('low')],
                'close': [ohlcv.get('close')],
                'volume': [ohlcv.get('volume')],
                'regime': [ohlcv.get('regime', 'unknown')]
            }, index=[timestamp])
            
            # Append row
            self.regime_data_cache[symbol][timeframe] = pd.concat([df_cache, new_row])
            
            # Sort by index (timestamp)
            self.regime_data_cache[symbol][timeframe].sort_index(inplace=True)
            
            # Limit size
            max_rows = self.config.get("max_ohlcv_rows", 10000)
            if len(self.regime_data_cache[symbol][timeframe]) > max_rows:
                self.regime_data_cache[symbol][timeframe] = self.regime_data_cache[symbol][timeframe].iloc[-max_rows:]
            
        except Exception as e:
            logger.error(f"Error recording OHLCV data: {str(e)}")
    
    def _record_feature_importance(
        self, symbol: str, timeframe: str, regime: MarketRegimeType, features: Dict[str, Any]
    ) -> None:
        """
        Record feature importance.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            regime: Regime type
            features: Feature values
        """
        try:
            # Create importance record
            importance = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'regime': regime.value,
                'features': features
            }
            
            # Save to file
            file_path = os.path.join(
                self.data_dir, "features", 
                f"{symbol}_{timeframe}_features.json"
            )
            
            # Append to file
            feature_records = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        feature_records = json.load(f)
                except Exception:
                    # Start with empty list if file is corrupt
                    feature_records = []
            
            feature_records.append(importance)
            
            # Limit size
            max_records = self.config.get("max_feature_records", 1000)
            if len(feature_records) > max_records:
                feature_records = feature_records[-max_records:]
            
            # Save
            with open(file_path, 'w') as f:
                json.dump(feature_records, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error recording feature importance: {str(e)}")
    
    def export_training_dataset(
        self, symbols: Optional[List[str]] = None, 
        timeframes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Export a dataset for training ML models.
        
        Args:
            symbols: List of symbols to include (defaults to all collected)
            timeframes: List of timeframes to include (defaults to all collected)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dict mapping symbols to timeframes to DataFrames
        """
        try:
            # Use all collected symbols and timeframes if not specified
            if not symbols:
                symbols = list(self.regime_data_cache.keys())
            
            result = {}
            
            # Loop through symbols and timeframes
            for symbol in symbols:
                if symbol not in self.regime_data_cache:
                    continue
                
                result[symbol] = {}
                
                timeframes_to_export = timeframes or list(self.regime_data_cache[symbol].keys())
                
                for timeframe in timeframes_to_export:
                    if timeframe not in self.regime_data_cache[symbol]:
                        continue
                    
                    # Get dataframe
                    df = self.regime_data_cache[symbol][timeframe].copy()
                    
                    # Apply date filters
                    if start_date:
                        df = df[df.index >= start_date]
                    
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    # Save to result
                    result[symbol][timeframe] = df
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting training dataset: {str(e)}")
            return {}
    
    def generate_ml_features(
        self, training_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate ML features from OHLCV data.
        
        Args:
            training_data: Dict mapping symbols to timeframes to DataFrames
            
        Returns:
            Dict mapping symbols to timeframes to feature DataFrames
        """
        try:
            # Generate features for each symbol and timeframe
            feature_data = {}
            
            for symbol, timeframe_data in training_data.items():
                feature_data[symbol] = {}
                
                for timeframe, df in timeframe_data.items():
                    # Generate features
                    features_df = self._calculate_features(df)
                    
                    # Store in result
                    feature_data[symbol][timeframe] = features_df
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error generating ML features: {str(e)}")
            return {}
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        try:
            # Make a copy to avoid modifying original
            features_df = df.copy()
            
            # Add basic features
            # Returns
            features_df['returns'] = features_df['close'].pct_change()
            
            # Moving averages
            features_df['ma_20'] = features_df['close'].rolling(20).mean()
            features_df['ma_50'] = features_df['close'].rolling(50).mean()
            
            # Relative to moving average
            features_df['close_to_ma20'] = features_df['close'] / features_df['ma_20'] - 1
            features_df['close_to_ma50'] = features_df['close'] / features_df['ma_50'] - 1
            
            # Volatility
            features_df['volatility_20'] = features_df['returns'].rolling(20).std()
            
            # Trading range
            features_df['trading_range'] = (features_df['high'] - features_df['low']) / features_df['close']
            
            # Volume change
            features_df['volume_change'] = features_df['volume'].pct_change()
            
            # Drop NaN values
            features_df.dropna(inplace=True)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            return df

# Helper function to create and initialize the data collector
def create_regime_data_collector(
    event_bus: EventBus,
    data_dir: str = "data/market_regime",
    config: Optional[Dict[str, Any]] = None
) -> RegimeDataCollector:
    """
    Create and initialize a regime data collector.
    
    Args:
        event_bus: System event bus
        data_dir: Directory to store collected data
        config: Optional configuration
        
    Returns:
        Initialized RegimeDataCollector
    """
    collector = RegimeDataCollector(event_bus, data_dir, config)
    
    # Start collection if configured to auto-start
    if config and config.get("auto_start", True):
        symbols = config.get("symbols_to_collect", [])
        collector.start_collection(symbols)
    
    return collector
