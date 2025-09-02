#!/usr/bin/env python3
"""
BensBot-EvoTrader Data Format Adapter

Converts data formats between BensBot and EvoTrader systems without modifying either system.
Acts as a translation layer for seamless interoperability.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_format_adapter')

class DataFormatAdapter:
    """
    Translates data formats between BensBot and EvoTrader without modifying either system.
    
    This adapter handles:
    1. Market data format conversion
    2. Order format translation
    3. Performance metrics standardization
    4. Strategy parameter mapping
    """
    
    def __init__(self):
        """Initialize the data format adapter."""
        logger.info("Initializing data format adapter")
    
    def benbot_to_evotrader_market_data(self, 
                                       benbot_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert BensBot market data format to EvoTrader format.
        
        Args:
            benbot_data: Market data in BensBot format
            
        Returns:
            Market data in EvoTrader format
        """
        try:
            logger.debug("Converting BensBot market data to EvoTrader format")
            
            # Convert to DataFrame if necessary
            if isinstance(benbot_data, dict):
                # Try to determine the format
                if 'ohlcv' in benbot_data:
                    # BensBot may store as a dictionary with OHLCV arrays
                    df = pd.DataFrame(benbot_data['ohlcv'], 
                                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                elif all(key in benbot_data for key in ['open', 'high', 'low', 'close']):
                    # Dictionary with separate OHLC arrays
                    df = pd.DataFrame({
                        'open': benbot_data['open'],
                        'high': benbot_data['high'],
                        'low': benbot_data['low'],
                        'close': benbot_data['close'],
                        'volume': benbot_data.get('volume', np.zeros_like(benbot_data['close']))
                    })
                    if 'timestamp' in benbot_data:
                        df['timestamp'] = pd.to_datetime(benbot_data['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                else:
                    logger.warning("Unknown BensBot data format, returning as-is")
                    return pd.DataFrame(benbot_data)
            else:
                df = benbot_data.copy()
            
            # Ensure consistent column naming
            column_mapping = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            df.rename(columns=column_mapping, inplace=True, errors='ignore')
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in data")
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Add volume if not present
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Convert index to datetime if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                else:
                    # Create a synthetic index if needed
                    df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='1min')
            
            logger.debug("Successfully converted BensBot market data to EvoTrader format")
            return df
            
        except Exception as e:
            logger.error(f"Error converting BensBot market data: {e}")
            # Return original data if conversion fails
            if isinstance(benbot_data, pd.DataFrame):
                return benbot_data
            else:
                return pd.DataFrame(benbot_data)
    
    def evotrader_to_benbot_market_data(self, 
                                       evotrader_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert EvoTrader market data format to BensBot format.
        
        Args:
            evotrader_data: Market data in EvoTrader format
            
        Returns:
            Market data in BensBot format
        """
        try:
            logger.debug("Converting EvoTrader market data to BensBot format")
            
            df = evotrader_data.copy()
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("EvoTrader data index is not DatetimeIndex, creating synthetic timestamps")
                df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='1min')
            
            # Ensure required columns exist with correct names
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }
            
            df.rename(columns=column_mapping, inplace=True, errors='ignore')
            
            # Format for BensBot
            benbot_data = {
                'timestamp': df.index.astype(np.int64) // 10**6,  # Convert to milliseconds
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'volume': df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
            }
            
            logger.debug("Successfully converted EvoTrader market data to BensBot format")
            return benbot_data
            
        except Exception as e:
            logger.error(f"Error converting EvoTrader market data: {e}")
            # Return a minimal compatible format if conversion fails
            return {
                'timestamp': evotrader_data.index.astype(np.int64) // 10**6,
                'open': evotrader_data['open'].values if 'open' in evotrader_data.columns else [],
                'high': evotrader_data['high'].values if 'high' in evotrader_data.columns else [],
                'low': evotrader_data['low'].values if 'low' in evotrader_data.columns else [],
                'close': evotrader_data['close'].values if 'close' in evotrader_data.columns else [],
                'volume': evotrader_data['volume'].values if 'volume' in evotrader_data.columns else []
            }
    
    def benbot_to_evotrader_strategy_params(self, 
                                          benbot_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert BensBot strategy parameters to EvoTrader format.
        
        Args:
            benbot_params: Strategy parameters in BensBot format
            
        Returns:
            Strategy parameters in EvoTrader format
        """
        try:
            logger.debug("Converting BensBot strategy parameters to EvoTrader format")
            
            # Parameter name mapping (BensBot → EvoTrader)
            param_mapping = {
                # Standard indicators
                'weekly_ema_period': 'ema_period',
                'daily_sma_fast_period': 'fast_ma_period',
                'daily_sma_slow_period': 'slow_ma_period',
                'adx_period': 'adx_period',
                'macd_fast': 'macd_fast',
                'macd_slow': 'macd_slow',
                'macd_signal': 'macd_signal',
                'rsi_period': 'rsi_period',
                'rsi_overbought': 'overbought_threshold',
                'rsi_oversold': 'oversold_threshold',
                'bollinger_period': 'bollinger_period',
                'bollinger_std': 'bollinger_std_dev',
                'atr_period': 'atr_period',
                
                # Risk management
                'risk_percent': 'risk_per_trade',
                'max_exposure_percent': 'max_exposure',
                'stop_loss_atr_multiple': 'stop_loss_atr_multiple',
                'profit_target_atr_multiple': 'take_profit_atr_multiple',
                'trailing_stop_atr_multiple': 'trailing_stop_atr',
                
                # Timeframes
                'min_holding_weeks': 'min_hold_time',
                'max_holding_weeks': 'max_hold_time'
            }
            
            # Convert parameters
            evotrader_params = {}
            
            for benbot_name, benbot_value in benbot_params.items():
                # Use mapping if available, otherwise keep original name
                evotrader_name = param_mapping.get(benbot_name, benbot_name)
                evotrader_params[evotrader_name] = benbot_value
            
            logger.debug("Successfully converted BensBot strategy parameters to EvoTrader format")
            return evotrader_params
            
        except Exception as e:
            logger.error(f"Error converting BensBot strategy parameters: {e}")
            # Return original parameters if conversion fails
            return benbot_params
    
    def evotrader_to_benbot_strategy_params(self, 
                                          evotrader_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert EvoTrader strategy parameters to BensBot format.
        
        Args:
            evotrader_params: Strategy parameters in EvoTrader format
            
        Returns:
            Strategy parameters in BensBot format
        """
        try:
            logger.debug("Converting EvoTrader strategy parameters to BensBot format")
            
            # Parameter name mapping (EvoTrader → BensBot)
            param_mapping = {
                # Standard indicators
                'ema_period': 'weekly_ema_period',
                'fast_ma_period': 'daily_sma_fast_period',
                'slow_ma_period': 'daily_sma_slow_period',
                'adx_period': 'adx_period',
                'macd_fast': 'macd_fast',
                'macd_slow': 'macd_slow',
                'macd_signal': 'macd_signal',
                'rsi_period': 'rsi_period',
                'overbought_threshold': 'rsi_overbought',
                'oversold_threshold': 'rsi_oversold',
                'bollinger_period': 'bollinger_period',
                'bollinger_std_dev': 'bollinger_std',
                'atr_period': 'atr_period',
                
                # Risk management
                'risk_per_trade': 'risk_percent',
                'max_exposure': 'max_exposure_percent',
                'stop_loss_atr_multiple': 'stop_loss_atr_multiple',
                'take_profit_atr_multiple': 'profit_target_atr_multiple',
                'trailing_stop_atr': 'trailing_stop_atr_multiple',
                
                # Timeframes
                'min_hold_time': 'min_holding_weeks',
                'max_hold_time': 'max_holding_weeks'
            }
            
            # Convert parameters
            benbot_params = {}
            
            for evotrader_name, evotrader_value in evotrader_params.items():
                # Use mapping if available, otherwise keep original name
                benbot_name = param_mapping.get(evotrader_name, evotrader_name)
                benbot_params[benbot_name] = evotrader_value
            
            logger.debug("Successfully converted EvoTrader strategy parameters to BensBot format")
            return benbot_params
            
        except Exception as e:
            logger.error(f"Error converting EvoTrader strategy parameters: {e}")
            # Return original parameters if conversion fails
            return evotrader_params
    
    def benbot_to_evotrader_performance(self,
                                       benbot_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert BensBot strategy performance metrics to EvoTrader format.
        
        Args:
            benbot_performance: Performance metrics in BensBot format
            
        Returns:
            Performance metrics in EvoTrader format
        """
        try:
            logger.debug("Converting BensBot performance metrics to EvoTrader format")
            
            # Metric name mapping (BensBot → EvoTrader)
            metric_mapping = {
                'totalReturn': 'total_return',
                'annualizedReturn': 'annualized_return',
                'sharpeRatio': 'sharpe_ratio',
                'sortino': 'sortino_ratio',
                'maxDrawdown': 'max_drawdown',
                'winRate': 'win_rate',
                'totalTrades': 'total_trades',
                'profitableTradesCount': 'winning_trades',
                'lossingTradesCount': 'losing_trades',
                'averageProfitLoss': 'avg_trade_pnl',
                'averageHoldingPeriod': 'avg_hold_time',
                'calmarRatio': 'calmar_ratio'
            }
            
            # Convert metrics
            evotrader_performance = {}
            
            for benbot_name, benbot_value in benbot_performance.items():
                # Use mapping if available, otherwise keep original name
                evotrader_name = metric_mapping.get(benbot_name, benbot_name)
                evotrader_performance[evotrader_name] = benbot_value
            
            # Calculate any missing derived metrics if possible
            if 'total_trades' in evotrader_performance and 'winning_trades' in evotrader_performance:
                if 'win_rate' not in evotrader_performance:
                    total = evotrader_performance['total_trades']
                    winning = evotrader_performance['winning_trades']
                    if total > 0:
                        evotrader_performance['win_rate'] = winning / total
            
            logger.debug("Successfully converted BensBot performance metrics to EvoTrader format")
            return evotrader_performance
            
        except Exception as e:
            logger.error(f"Error converting BensBot performance metrics: {e}")
            # Return original metrics if conversion fails
            return benbot_performance
    
    def evotrader_to_benbot_performance(self,
                                       evotrader_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert EvoTrader strategy performance metrics to BensBot format.
        
        Args:
            evotrader_performance: Performance metrics in EvoTrader format
            
        Returns:
            Performance metrics in BensBot format
        """
        try:
            logger.debug("Converting EvoTrader performance metrics to BensBot format")
            
            # Metric name mapping (EvoTrader → BensBot)
            metric_mapping = {
                'total_return': 'totalReturn',
                'annualized_return': 'annualizedReturn',
                'sharpe_ratio': 'sharpeRatio',
                'sortino_ratio': 'sortino',
                'max_drawdown': 'maxDrawdown',
                'win_rate': 'winRate',
                'total_trades': 'totalTrades',
                'winning_trades': 'profitableTradesCount',
                'losing_trades': 'lossingTradesCount',
                'avg_trade_pnl': 'averageProfitLoss',
                'avg_hold_time': 'averageHoldingPeriod',
                'calmar_ratio': 'calmarRatio'
            }
            
            # Convert metrics
            benbot_performance = {}
            
            for evotrader_name, evotrader_value in evotrader_performance.items():
                # Use mapping if available, otherwise keep original name
                benbot_name = metric_mapping.get(evotrader_name, evotrader_name)
                benbot_performance[benbot_name] = evotrader_value
            
            logger.debug("Successfully converted EvoTrader performance metrics to BensBot format")
            return benbot_performance
            
        except Exception as e:
            logger.error(f"Error converting EvoTrader performance metrics: {e}")
            # Return original metrics if conversion fails
            return evotrader_performance
    
    def benbot_to_evotrader_order(self, 
                                 benbot_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert BensBot order format to EvoTrader format.
        
        Args:
            benbot_order: Order in BensBot format
            
        Returns:
            Order in EvoTrader format
        """
        try:
            logger.debug("Converting BensBot order to EvoTrader format")
            
            # Field name mapping (BensBot → EvoTrader)
            field_mapping = {
                'symbol': 'symbol',
                'side': 'direction',
                'type': 'order_type',
                'quantity': 'size',
                'price': 'price',
                'stopPrice': 'stop_price',
                'timeInForce': 'time_in_force',
                'orderId': 'order_id',
                'status': 'status'
            }
            
            # Direction mapping
            direction_mapping = {
                'BUY': 'long',
                'SELL': 'short',
                'buy': 'long',
                'sell': 'short'
            }
            
            # Order type mapping
            order_type_mapping = {
                'MARKET': 'market',
                'LIMIT': 'limit',
                'STOP': 'stop',
                'STOP_LIMIT': 'stop_limit',
                'market': 'market',
                'limit': 'limit',
                'stop': 'stop',
                'stop_limit': 'stop_limit'
            }
            
            # Convert order
            evotrader_order = {}
            
            for benbot_field, benbot_value in benbot_order.items():
                # Use mapping if available, otherwise keep original name
                evotrader_field = field_mapping.get(benbot_field, benbot_field)
                
                # Apply value mappings for specific fields
                if benbot_field == 'side':
                    evotrader_order[evotrader_field] = direction_mapping.get(benbot_value, benbot_value)
                elif benbot_field == 'type':
                    evotrader_order[evotrader_field] = order_type_mapping.get(benbot_value, benbot_value)
                else:
                    evotrader_order[evotrader_field] = benbot_value
            
            logger.debug("Successfully converted BensBot order to EvoTrader format")
            return evotrader_order
            
        except Exception as e:
            logger.error(f"Error converting BensBot order: {e}")
            # Return original order if conversion fails
            return benbot_order
    
    def evotrader_to_benbot_order(self, 
                                 evotrader_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert EvoTrader order format to BensBot format.
        
        Args:
            evotrader_order: Order in EvoTrader format
            
        Returns:
            Order in BensBot format
        """
        try:
            logger.debug("Converting EvoTrader order to BensBot format")
            
            # Field name mapping (EvoTrader → BensBot)
            field_mapping = {
                'symbol': 'symbol',
                'direction': 'side',
                'order_type': 'type',
                'size': 'quantity',
                'price': 'price',
                'stop_price': 'stopPrice',
                'time_in_force': 'timeInForce',
                'order_id': 'orderId',
                'status': 'status'
            }
            
            # Direction mapping
            direction_mapping = {
                'long': 'BUY',
                'short': 'SELL',
                'buy': 'BUY',
                'sell': 'SELL'
            }
            
            # Order type mapping
            order_type_mapping = {
                'market': 'MARKET',
                'limit': 'LIMIT',
                'stop': 'STOP',
                'stop_limit': 'STOP_LIMIT'
            }
            
            # Convert order
            benbot_order = {}
            
            for evotrader_field, evotrader_value in evotrader_order.items():
                # Use mapping if available, otherwise keep original name
                benbot_field = field_mapping.get(evotrader_field, evotrader_field)
                
                # Apply value mappings for specific fields
                if evotrader_field == 'direction':
                    benbot_order[benbot_field] = direction_mapping.get(evotrader_value, evotrader_value)
                elif evotrader_field == 'order_type':
                    benbot_order[benbot_field] = order_type_mapping.get(evotrader_value, evotrader_value)
                else:
                    benbot_order[benbot_field] = evotrader_value
            
            logger.debug("Successfully converted EvoTrader order to BensBot format")
            return benbot_order
            
        except Exception as e:
            logger.error(f"Error converting EvoTrader order: {e}")
            # Return original order if conversion fails
            return evotrader_order


if __name__ == "__main__":
    # Example usage
    adapter = DataFormatAdapter()
    
    # Create example data
    example_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [103.0, 104.0, 105.0, 106.0, 107.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range(start='2023-01-01', periods=5, freq='1d'))
    
    # Test conversions
    benbot_format = adapter.evotrader_to_benbot_market_data(example_data)
    print("BensBot Format:")
    for key, value in benbot_format.items():
        print(f"{key}: {value[:3]}...")
    
    evotrader_format = adapter.benbot_to_evotrader_market_data(benbot_format)
    print("\nEvoTrader Format (converted back):")
    print(evotrader_format.head(3))
