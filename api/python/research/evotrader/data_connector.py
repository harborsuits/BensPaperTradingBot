"""
Data Connector for BensBot-EvoTrader Integration

This module provides data synchronization between BensBot and EvoTrader systems,
enabling market data, indicators, and analysis to flow between the systems.
"""

import os
import json
import logging
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger("benbot.research.evotrader.data")

class EvoTraderDataConnector:
    """
    Handles data synchronization between BensBot and EvoTrader systems.
    
    This connector manages:
    - Market data transfer (OHLCV data for forex and crypto)
    - Indicator data synchronization
    - Results and metrics flow between systems
    - Training/testing data split and management
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 cache_dir: str = None):
        """
        Initialize the data connector.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching data
        """
        self.config = config or {}
        self.cache_dir = cache_dir or "evotrader_data_cache"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories
        self.forex_dir = os.path.join(self.cache_dir, "forex")
        self.crypto_dir = os.path.join(self.cache_dir, "crypto")
        self.indicators_dir = os.path.join(self.cache_dir, "indicators")
        self.results_dir = os.path.join(self.cache_dir, "results")
        
        for directory in [self.forex_dir, self.crypto_dir, self.indicators_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Data connector initialized with cache at {self.cache_dir}")
        
    def get_forex_data(self, 
                      symbol: str, 
                      timeframe: str, 
                      start_date: str = None,
                      end_date: str = None,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Get forex market data from BensBot and prepare for EvoTrader.
        
        Args:
            symbol: Forex pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first if enabled
        if use_cache:
            cache_file = os.path.join(self.forex_dir, f"{symbol}_{timeframe}.csv")
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Apply date filters if present
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                    
                if not df.empty:
                    logger.info(f"Loaded cached forex data for {symbol} {timeframe} with {len(df)} rows")
                    return df
        
        # If cache not available or not to be used, fetch from BensBot
        try:
            # In a real implementation, this would fetch from BensBot's data API
            # For demo, we'll create synthetic data
            
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            if not end_date:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
            # Convert to datetime
            start_dt = datetime.datetime.fromisoformat(start_date)
            end_dt = datetime.datetime.fromisoformat(end_date)
            
            # Generate dates based on timeframe
            if timeframe == "1h":
                freq = "H"
            elif timeframe == "4h":
                freq = "4H"
            elif timeframe == "1d":
                freq = "D"
            else:
                freq = "D"  # Default to daily
                
            # Generate date range
            dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)
            
            # Create DataFrame with synthetic data
            # This is just for demo purposes - in a real implementation,
            # we would fetch actual data from BensBot's database
            import numpy as np
            base_price = 1.1000 if symbol == "EURUSD" else 1.3000
            
            df = pd.DataFrame(index=dates)
            df['open'] = base_price + np.random.normal(0, 0.002, len(dates))
            df['high'] = df['open'] + abs(np.random.normal(0, 0.001, len(dates)))
            df['low'] = df['open'] - abs(np.random.normal(0, 0.001, len(dates)))
            df['close'] = df['open'] + np.random.normal(0, 0.002, len(dates))
            df['volume'] = np.random.randint(100, 1000, len(dates))
            
            # Cache the data
            os.makedirs(self.forex_dir, exist_ok=True)
            df.to_csv(os.path.join(self.forex_dir, f"{symbol}_{timeframe}.csv"))
            
            logger.info(f"Generated synthetic forex data for {symbol} {timeframe} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return pd.DataFrame()
    
    def get_crypto_data(self, 
                       symbol: str, 
                       timeframe: str, 
                       start_date: str = None,
                       end_date: str = None,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Get cryptocurrency market data from BensBot and prepare for EvoTrader.
        
        Args:
            symbol: Crypto pair (e.g., "BTC/USD")
            timeframe: Timeframe (e.g., "15m", "1h", "4h", "1d")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first if enabled
        if use_cache:
            # Convert symbol name for filename (replace / with _)
            symbol_filename = symbol.replace("/", "_")
            cache_file = os.path.join(self.crypto_dir, f"{symbol_filename}_{timeframe}.csv")
            
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Apply date filters if present
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                    
                if not df.empty:
                    logger.info(f"Loaded cached crypto data for {symbol} {timeframe} with {len(df)} rows")
                    return df
        
        # If cache not available or not to be used, fetch from BensBot
        try:
            # In a real implementation, this would fetch from BensBot's data API
            # For demo, we'll create synthetic data
            
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            if not end_date:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
            # Convert to datetime
            start_dt = datetime.datetime.fromisoformat(start_date)
            end_dt = datetime.datetime.fromisoformat(end_date)
            
            # Generate dates based on timeframe
            if timeframe == "15m":
                freq = "15min"
            elif timeframe == "1h":
                freq = "H"
            elif timeframe == "4h":
                freq = "4H"
            elif timeframe == "1d":
                freq = "D"
            else:
                freq = "H"  # Default to hourly
                
            # Generate date range
            dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)
            
            # Create DataFrame with synthetic data
            import numpy as np
            
            # Set base price based on symbol
            if "BTC" in symbol:
                base_price = 40000
                volatility = 1000
            elif "ETH" in symbol:
                base_price = 2500
                volatility = 100
            elif "SOL" in symbol:
                base_price = 100
                volatility = 5
            else:
                base_price = 50
                volatility = 2
            
            # Generate synthetic price data with realistic trends
            time_steps = np.arange(len(dates))
            trend = np.sin(time_steps / (len(time_steps) / 3)) * volatility * 2
            noise = np.random.normal(0, volatility, len(dates))
            
            df = pd.DataFrame(index=dates)
            df['open'] = base_price + trend + noise
            df['high'] = df['open'] + abs(np.random.normal(0, volatility/10, len(dates)))
            df['low'] = df['open'] - abs(np.random.normal(0, volatility/10, len(dates)))
            df['close'] = df['open'] + np.random.normal(0, volatility/5, len(dates))
            df['volume'] = np.random.randint(10, 100, len(dates)) * 100
            
            # Ensure values are positive
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].clip(lower=0)
            
            # Ensure high >= open, close, low
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            # Ensure low <= open, close, high
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
            # Cache the data
            symbol_filename = symbol.replace("/", "_")
            os.makedirs(self.crypto_dir, exist_ok=True)
            df.to_csv(os.path.join(self.crypto_dir, f"{symbol_filename}_{timeframe}.csv"))
            
            logger.info(f"Generated synthetic crypto data for {symbol} {timeframe} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return pd.DataFrame()
    
    def get_indicator_data(self,
                          symbol: str,
                          timeframe: str,
                          indicator_name: str,
                          parameters: Dict[str, Any] = None,
                          asset_class: str = "forex",
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Get technical indicator data for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The time interval
            indicator_name: Name of the indicator
            parameters: Indicator parameters
            asset_class: "forex" or "crypto"
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with indicator values
        """
        # Create parameter string for cache key
        param_str = ""
        if parameters:
            param_str = "_".join([f"{k}_{v}" for k, v in sorted(parameters.items())])
        
        # Check cache first if enabled
        if use_cache:
            symbol_filename = symbol.replace("/", "_")
            cache_file = os.path.join(
                self.indicators_dir, 
                f"{asset_class}_{symbol_filename}_{timeframe}_{indicator_name}_{param_str}.csv"
            )
            
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached indicator data for {indicator_name} on {symbol} {timeframe}")
                return df
        
        # Get price data
        if asset_class == "forex":
            price_data = self.get_forex_data(symbol, timeframe)
        else:
            price_data = self.get_crypto_data(symbol, timeframe)
            
        if price_data.empty:
            logger.error(f"No price data available for {symbol} {timeframe}")
            return pd.DataFrame()
            
        # Calculate indicator
        try:
            # In a real implementation, we would use talib or another library
            # For demo, we'll generate some basic indicators
            import numpy as np
            
            result_df = pd.DataFrame(index=price_data.index)
            
            if indicator_name.lower() == "sma":
                # Simple Moving Average
                period = parameters.get("period", 14)
                result_df["value"] = price_data["close"].rolling(window=period).mean()
                
            elif indicator_name.lower() == "ema":
                # Exponential Moving Average
                period = parameters.get("period", 14)
                result_df["value"] = price_data["close"].ewm(span=period, adjust=False).mean()
                
            elif indicator_name.lower() == "rsi":
                # Relative Strength Index
                period = parameters.get("period", 14)
                delta = price_data["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                result_df["value"] = 100 - (100 / (1 + rs))
                
            elif indicator_name.lower() == "macd":
                # MACD
                fast_period = parameters.get("fast_period", 12)
                slow_period = parameters.get("slow_period", 26)
                signal_period = parameters.get("signal_period", 9)
                
                fast_ema = price_data["close"].ewm(span=fast_period, adjust=False).mean()
                slow_ema = price_data["close"].ewm(span=slow_period, adjust=False).mean()
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                
                result_df["macd"] = macd_line
                result_df["signal"] = signal_line
                result_df["histogram"] = macd_line - signal_line
                
            else:
                # Default to a simple moving average
                period = parameters.get("period", 14)
                result_df["value"] = price_data["close"].rolling(window=period).mean()
            
            # Cache the result
            symbol_filename = symbol.replace("/", "_")
            os.makedirs(self.indicators_dir, exist_ok=True)
            
            cache_file = os.path.join(
                self.indicators_dir, 
                f"{asset_class}_{symbol_filename}_{timeframe}_{indicator_name}_{param_str}.csv"
            )
            result_df.to_csv(cache_file)
            
            logger.info(f"Calculated {indicator_name} for {symbol} {timeframe}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating indicator {indicator_name}: {e}")
            return pd.DataFrame()
    
    def save_strategy_results(self,
                             strategy_id: str,
                             results: Dict[str, Any],
                             asset_class: str = "forex"):
        """
        Save strategy backtest or live results for future analysis.
        
        Args:
            strategy_id: Unique strategy identifier
            results: Dictionary containing strategy results
            asset_class: "forex" or "crypto"
        """
        try:
            # Create results directory
            results_dir = os.path.join(self.results_dir, asset_class)
            os.makedirs(results_dir, exist_ok=True)
            
            # Add timestamp
            results["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save to file
            results_file = os.path.join(results_dir, f"{strategy_id}.json")
            
            # If file exists, load existing results and append
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    existing_results = json.load(f)
                
                # Check if it's a list or a single result
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
                
                # Append new results
                existing_results.append(results)
                
                with open(results_file, "w") as f:
                    json.dump(existing_results, f, indent=2)
            else:
                # Save as a new file
                with open(results_file, "w") as f:
                    json.dump([results], f, indent=2)
            
            logger.info(f"Saved results for strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"Error saving strategy results: {e}")
    
    def get_strategy_results(self,
                            strategy_id: str = None,
                            asset_class: str = "forex") -> List[Dict[str, Any]]:
        """
        Get strategy results for analysis.
        
        Args:
            strategy_id: Specific strategy ID or None for all
            asset_class: "forex" or "crypto"
            
        Returns:
            List of result dictionaries
        """
        results = []
        results_dir = os.path.join(self.results_dir, asset_class)
        
        if not os.path.exists(results_dir):
            logger.warning(f"No results directory found for {asset_class}")
            return results
            
        try:
            # If specific strategy requested
            if strategy_id:
                results_file = os.path.join(results_dir, f"{strategy_id}.json")
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        strategy_results = json.load(f)
                        
                    # Convert to list if not already
                    if not isinstance(strategy_results, list):
                        strategy_results = [strategy_results]
                        
                    return strategy_results
                else:
                    logger.warning(f"No results found for strategy {strategy_id}")
                    return []
            
            # If all strategies requested
            for filename in os.listdir(results_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(results_dir, filename), "r") as f:
                        strategy_results = json.load(f)
                    
                    # Convert to list if not already
                    if not isinstance(strategy_results, list):
                        strategy_results = [strategy_results]
                        
                    results.extend(strategy_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting strategy results: {e}")
            return []
    
    def clear_cache(self, asset_class: str = None, data_type: str = None):
        """
        Clear data cache.
        
        Args:
            asset_class: "forex", "crypto", or None for all
            data_type: "market", "indicators", "results", or None for all
        """
        try:
            dirs_to_clear = []
            
            # Determine which directories to clear
            if asset_class and data_type:
                if asset_class == "forex" and data_type == "market":
                    dirs_to_clear.append(self.forex_dir)
                elif asset_class == "crypto" and data_type == "market":
                    dirs_to_clear.append(self.crypto_dir)
                elif data_type == "indicators":
                    # Clear only specific asset class indicators
                    for filename in os.listdir(self.indicators_dir):
                        if filename.startswith(f"{asset_class}_"):
                            os.remove(os.path.join(self.indicators_dir, filename))
                    logger.info(f"Cleared {asset_class} indicators cache")
                elif data_type == "results":
                    results_dir = os.path.join(self.results_dir, asset_class)
                    if os.path.exists(results_dir):
                        dirs_to_clear.append(results_dir)
            
            elif asset_class:
                if asset_class == "forex":
                    dirs_to_clear.append(self.forex_dir)
                elif asset_class == "crypto":
                    dirs_to_clear.append(self.crypto_dir)
                
                # Clear asset-specific indicators
                for filename in os.listdir(self.indicators_dir):
                    if filename.startswith(f"{asset_class}_"):
                        os.remove(os.path.join(self.indicators_dir, filename))
                
                # Clear asset-specific results
                results_dir = os.path.join(self.results_dir, asset_class)
                if os.path.exists(results_dir):
                    dirs_to_clear.append(results_dir)
            
            elif data_type:
                if data_type == "market":
                    dirs_to_clear.extend([self.forex_dir, self.crypto_dir])
                elif data_type == "indicators":
                    dirs_to_clear.append(self.indicators_dir)
                elif data_type == "results":
                    dirs_to_clear.append(self.results_dir)
            
            else:
                # Clear all cache
                dirs_to_clear.extend([
                    self.forex_dir, 
                    self.crypto_dir, 
                    self.indicators_dir, 
                    self.results_dir
                ])
            
            # Clear the directories
            for directory in dirs_to_clear:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
