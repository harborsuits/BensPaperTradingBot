#!/usr/bin/env python3
"""
BensBot Strategy Importer

Imports strategies from BensBot and adapts them to work with EvoTrader's meta-learning system.
"""

import os
import sys
import inspect
import importlib.util
import logging
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union, Type

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import EvoTrader components
from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'strategy_importer.log')),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger('strategy_importer')

class BenBotStrategyAdapter:
    """
    Adapter that converts BensBot strategies to EvoTrader compatible strategies.
    
    This class:
    1. Imports BensBot strategy classes
    2. Extracts parameters and metadata
    3. Converts them to EvoTrader format
    4. Applies meta-learning optimizations
    """
    
    # Strategy type mapping from BensBot to EvoTrader
    STRATEGY_TYPE_MAPPING = {
        "TrendTradingStrategy": "trend_following",
        "RangeTradingStrategy": "mean_reversion",
        "BreakoutTradingStrategy": "breakout",
        "MomentumTradingStrategy": "momentum",
        "SwingTradingStrategy": "swing_trading",
        "ScalpingStrategy": "scalping",
        "ReversalTradingStrategy": "reversal"
    }
    
    # Parameter mapping from BensBot to EvoTrader
    PARAMETER_MAPPING = {
        # EMA/SMA
        "weekly_ema_period": "ema_period",
        "daily_sma_fast_period": "fast_ma_period",
        "daily_sma_slow_period": "slow_ma_period",
        
        # MACD
        "macd_fast": "macd_fast",
        "macd_slow": "macd_slow",
        "macd_signal": "macd_signal",
        
        # RSI
        "rsi_period": "rsi_period",
        "rsi_overbought": "overbought_threshold",
        "rsi_oversold": "oversold_threshold",
        
        # Bollinger Bands
        "bollinger_period": "bollinger_period",
        "bollinger_std": "bollinger_std_dev",
        
        # ATR
        "atr_period": "atr_period",
        
        # Risk and position sizing
        "risk_percent": "risk_per_trade",
        "stop_loss_atr_multiple": "stop_loss_atr_multiple",
        "profit_target_atr_multiple": "take_profit_atr_multiple"
    }
    
    def __init__(self, meta_db_path: str = None):
        """
        Initialize the strategy adapter.
        
        Args:
            meta_db_path: Path to meta-learning database
        """
        self.meta_db = None
        if meta_db_path:
            try:
                self.meta_db = MetaLearningDB(db_path=meta_db_path)
                logger.info("Connected to meta-learning database")
            except Exception as e:
                logger.error(f"Failed to connect to meta-learning database: {e}")
        
        self.regime_detector = MarketRegimeDetector()
        
        # Track imported strategies
        self.imported_strategies = {}
    
    def import_strategy_from_path(self, 
                                 file_path: str, 
                                 strategy_name: str = None) -> Dict[str, Any]:
        """
        Import a strategy from a file path.
        
        Args:
            file_path: Path to the strategy file
            strategy_name: Optional name of the strategy class to import
            
        Returns:
            Strategy information dictionary
        """
        try:
            logger.info(f"Importing strategy from: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Strategy file not found: {file_path}")
                return {}
            
            # Import module
            module_name = os.path.basename(file_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find strategy class
            strategy_class = None
            
            if strategy_name:
                # Look for specific strategy
                if hasattr(module, strategy_name):
                    strategy_class = getattr(module, strategy_name)
            else:
                # Find first class that looks like a strategy
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        (name.endswith('Strategy') or 'Trading' in name) and
                        name != 'StrategyTemplate' and
                        name != 'StrategyOptimizable'):
                        strategy_class = obj
                        strategy_name = name
                        break
            
            if not strategy_class:
                logger.error(f"No strategy class found in: {file_path}")
                return {}
            
            # Create a temporary instance to extract information
            try:
                strategy_instance = strategy_class(name=f"Imported_{strategy_name}")
            except Exception as e:
                logger.error(f"Failed to instantiate strategy {strategy_name}: {e}")
                
                # Try with minimum required arguments
                try:
                    # Inspect __init__ to find required arguments
                    init_params = inspect.signature(strategy_class.__init__).parameters
                    args = {}
                    
                    for name, param in init_params.items():
                        if name == 'self':
                            continue
                        
                        if param.default == inspect.Parameter.empty:
                            # Required parameter
                            if name == 'name':
                                args[name] = f"Imported_{strategy_name}"
                            else:
                                args[name] = None
                    
                    strategy_instance = strategy_class(**args)
                except Exception as inner_e:
                    logger.error(f"Failed to instantiate strategy {strategy_name} with minimum args: {inner_e}")
                    return {}
            
            # Extract parameters
            parameters = {}
            
            # Try to get default parameters
            if hasattr(strategy_instance, 'default_parameters'):
                parameters = strategy_instance.default_parameters
            elif hasattr(strategy_class, 'default_params'):
                parameters = strategy_class.default_params
            
            # Try to access parameters as a property or attribute
            if not parameters and hasattr(strategy_instance, 'parameters'):
                params_attr = getattr(strategy_instance, 'parameters')
                if callable(params_attr):
                    parameters = params_attr()
                else:
                    parameters = params_attr
            
            # If still no parameters, look for parameter-like attributes
            if not parameters:
                for attr_name in dir(strategy_instance):
                    if attr_name.startswith('_'):
                        continue
                    
                    if attr_name.endswith('_period') or attr_name.endswith('_threshold') or attr_name.endswith('_multiple'):
                        attr_value = getattr(strategy_instance, attr_name)
                        if isinstance(attr_value, (int, float)):
                            parameters[attr_name] = attr_value
            
            # Extract metadata
            metadata = {}
            
            if hasattr(strategy_instance, 'metadata'):
                metadata_attr = getattr(strategy_instance, 'metadata')
                if callable(metadata_attr):
                    metadata = metadata_attr()
                else:
                    metadata = metadata_attr
            
            # Determine strategy type
            strategy_type = self.STRATEGY_TYPE_MAPPING.get(
                strategy_name,
                "unknown"  # Default type if not in mapping
            )
            
            # Determine supported timeframes
            supported_timeframes = []
            if 'supported_timeframes' in metadata:
                supported_timeframes = metadata['supported_timeframes']
            elif hasattr(strategy_instance, 'supported_timeframes'):
                supported_timeframes = strategy_instance.supported_timeframes
            elif 'Scalping' in strategy_name:
                supported_timeframes = ['1m', '5m', '15m']
            elif 'Day' in strategy_name:
                supported_timeframes = ['15m', '30m', '1h', '4h']
            elif 'Swing' in strategy_name:
                supported_timeframes = ['1h', '4h', 'D']
            else:
                supported_timeframes = ['5m', '15m', '30m', '1h', '4h', 'D']
            
            # Determine preferred market regimes
            preferred_regimes = []
            if 'preferred_regimes' in metadata:
                preferred_regimes = metadata['preferred_regimes']
            elif hasattr(strategy_instance, 'preferred_regimes'):
                preferred_regimes = strategy_instance.preferred_regimes
            elif 'Trend' in strategy_name:
                preferred_regimes = ['bullish', 'bearish']
            elif 'Range' in strategy_name:
                preferred_regimes = ['ranging', 'choppy']
            elif 'Breakout' in strategy_name:
                preferred_regimes = ['volatile_bullish', 'volatile_bearish']
            elif 'Reversal' in strategy_name:
                preferred_regimes = ['ranging', 'choppy']
            else:
                preferred_regimes = ['bullish', 'bearish', 'ranging']
            
            # Create EvoTrader-compatible strategy info
            strategy_info = {
                'strategy_id': f"imported_{strategy_name.lower()}",
                'strategy_name': strategy_name,
                'strategy_type': strategy_type,
                'file_path': file_path,
                'parameters': self._map_parameters(parameters),
                'original_parameters': parameters,
                'metadata': {
                    'source': 'benbot',
                    'supported_timeframes': supported_timeframes,
                    'preferred_regimes': preferred_regimes,
                    'original_metadata': metadata
                },
                'strategy_class': strategy_class
            }
            
            # Add to imported strategies
            self.imported_strategies[strategy_info['strategy_id']] = strategy_info
            
            logger.info(f"Successfully imported strategy: {strategy_name} as {strategy_info['strategy_id']}")
            return strategy_info
            
        except Exception as e:
            logger.error(f"Failed to import strategy from {file_path}: {e}")
            return {}
    
    def import_strategy_from_directory(self, 
                                      directory_path: str, 
                                      recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Import all strategies from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            
        Returns:
            List of strategy information dictionaries
        """
        try:
            logger.info(f"Importing strategies from: {directory_path}")
            
            # Check if directory exists
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return []
            
            # Find Python files
            strategy_files = []
            
            if recursive:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            strategy_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory_path):
                    if file.endswith('.py') and not file.startswith('__'):
                        strategy_files.append(os.path.join(directory_path, file))
            
            # Import strategies
            imported_strategies = []
            
            for file_path in strategy_files:
                strategy_info = self.import_strategy_from_path(file_path)
                if strategy_info:
                    imported_strategies.append(strategy_info)
            
            logger.info(f"Imported {len(imported_strategies)} strategies from {directory_path}")
            return imported_strategies
            
        except Exception as e:
            logger.error(f"Failed to import strategies from {directory_path}: {e}")
            return []
    
    def apply_meta_learning(self, 
                           strategy_id: str, 
                           market_regime: str = None,
                           price_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply meta-learning optimizations to a strategy.
        
        Args:
            strategy_id: ID of the imported strategy
            market_regime: Current market regime (optional)
            price_data: Recent price data for regime detection (optional)
            
        Returns:
            Optimized strategy information
        """
        if not self.meta_db:
            logger.warning("Meta-learning database not connected, cannot apply optimizations")
            return self.imported_strategies.get(strategy_id, {})
        
        try:
            logger.info(f"Applying meta-learning optimizations to strategy: {strategy_id}")
            
            # Check if strategy exists
            if strategy_id not in self.imported_strategies:
                logger.error(f"Strategy not found: {strategy_id}")
                return {}
            
            strategy_info = self.imported_strategies[strategy_id].copy()
            
            # Detect market regime if not provided
            if not market_regime and price_data:
                regime_info = self.regime_detector.detect_regime(price_data)
                market_regime = regime_info['regime']
                logger.info(f"Detected market regime: {market_regime}")
            elif not market_regime:
                market_regime = "unknown"
                logger.warning("No market regime specified or price data provided")
            
            # Get strategy type
            strategy_type = strategy_info['strategy_type']
            
            # Query meta-learning database for insights
            regime_insights = self.meta_db.get_regime_insights(market_regime)
            
            # Extract relevant insights for this strategy type
            strategy_type_insights = {}
            if 'strategy_type_performance' in regime_insights:
                for s_type, performance in regime_insights['strategy_type_performance'].items():
                    if s_type == strategy_type:
                        strategy_type_insights = performance
                        break
            
            # Extract parameter insights
            parameter_insights = {}
            if 'parameter_clusters' in regime_insights:
                for param, clusters in regime_insights['parameter_clusters'].items():
                    if param in strategy_info['parameters']:
                        # Find best cluster
                        best_cluster = max(clusters, key=lambda c: c.get('performance', 0)) if clusters else None
                        if best_cluster:
                            parameter_insights[param] = best_cluster
            
            # Apply insights to optimize parameters
            optimized_parameters = strategy_info['parameters'].copy()
            
            for param, value in optimized_parameters.items():
                # Check if we have insights for this parameter
                if param in parameter_insights:
                    cluster = parameter_insights[param]
                    
                    # Use cluster center as optimized value
                    if 'center' in cluster:
                        # Apply partial optimization (blend original and optimized)
                        original_value = float(value)
                        optimized_value = float(cluster['center'])
                        
                        # Use 70% weight for optimized value
                        blended_value = original_value * 0.3 + optimized_value * 0.7
                        
                        # Round to original precision
                        if isinstance(value, int):
                            optimized_parameters[param] = int(round(blended_value))
                        else:
                            optimized_parameters[param] = round(blended_value, 2)
                        
                        logger.info(f"Optimized parameter {param}: {value} -> {optimized_parameters[param]}")
            
            # Update strategy info
            strategy_info['parameters'] = optimized_parameters
            strategy_info['metadata']['market_regime'] = market_regime
            strategy_info['metadata']['meta_learning_applied'] = True
            strategy_info['metadata']['optimization_timestamp'] = self.meta_db.get_timestamp()
            
            # Update imported strategies
            self.imported_strategies[strategy_id] = strategy_info
            
            logger.info(f"Successfully applied meta-learning optimizations to {strategy_id}")
            return strategy_info
            
        except Exception as e:
            logger.error(f"Failed to apply meta-learning optimizations to {strategy_id}: {e}")
            return self.imported_strategies.get(strategy_id, {})
    
    def create_evotrader_strategy_file(self, 
                                      strategy_id: str, 
                                      output_dir: str = None) -> str:
        """
        Create an EvoTrader-compatible strategy file from a BensBot strategy.
        
        Args:
            strategy_id: ID of the imported strategy
            output_dir: Directory to save the file
            
        Returns:
            Path to the created file
        """
        try:
            logger.info(f"Creating EvoTrader strategy file for: {strategy_id}")
            
            # Check if strategy exists
            if strategy_id not in self.imported_strategies:
                logger.error(f"Strategy not found: {strategy_id}")
                return ""
            
            strategy_info = self.imported_strategies[strategy_id]
            
            # Determine output directory
            if not output_dir:
                output_dir = os.path.join(project_root, 'evotrader', 'strategies', 'imported')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create file name
            file_name = f"{strategy_id.lower()}.py"
            file_path = os.path.join(output_dir, file_name)
            
            # Generate code
            code = self._generate_evotrader_strategy_code(strategy_info)
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(code)
            
            logger.info(f"Created EvoTrader strategy file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to create EvoTrader strategy file for {strategy_id}: {e}")
            return ""
    
    def _map_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map BensBot parameters to EvoTrader parameters."""
        mapped_params = {}
        
        for param, value in parameters.items():
            # Use mapping if available
            mapped_param = self.PARAMETER_MAPPING.get(param, param)
            mapped_params[mapped_param] = value
        
        return mapped_params
    
    def _generate_evotrader_strategy_code(self, strategy_info: Dict[str, Any]) -> str:
        """Generate EvoTrader-compatible strategy code."""
        strategy_id = strategy_info['strategy_id']
        strategy_name = strategy_info['strategy_name']
        strategy_type = strategy_info['strategy_type']
        parameters = strategy_info['parameters']
        metadata = strategy_info['metadata']
        
        # Start with template
        code = f"""#!/usr/bin/env python3
\"\"\"
{strategy_name}

Imported from BensBot and adapted for EvoTrader.
Type: {strategy_type}
\"\"\"

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

# Parameters
"""
        
        # Add parameters
        for param, value in parameters.items():
            code += f"{param} = {value}\n"
        
        # Add initialization function
        code += """
def initialize(context):
    \"\"\"Initialize the strategy.\"\"\"
"""
        for param, value in parameters.items():
            code += f"    context.{param} = {value}\n"
        
        # Add metadata
        code += f"""
    # Metadata
    context.strategy_id = "{strategy_id}"
    context.strategy_type = "{strategy_type}"
    context.preferred_regimes = {metadata.get('preferred_regimes', [])}
    context.supported_timeframes = {metadata.get('supported_timeframes', [])}
    context.meta_learning_applied = {metadata.get('meta_learning_applied', False)}
"""
        
        # Add calculation function based on strategy type
        if strategy_type == "trend_following":
            code += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Trend following strategy logic
    close = data['close']
    
    # Calculate moving averages
    if hasattr(context, 'fast_ma_period') and hasattr(context, 'slow_ma_period'):
        fast_ma = data.get('ema', close.rolling(window=context.fast_ma_period).mean())
        slow_ma = close.rolling(window=context.slow_ma_period).mean()
        
        # Generate signal
        if fast_ma[-1] > slow_ma[-1] and fast_ma[-2] <= slow_ma[-2]:
            return 1  # Buy signal
        elif fast_ma[-1] < slow_ma[-1] and fast_ma[-2] >= slow_ma[-2]:
            return -1  # Sell signal
    
    # Use MACD if available
    elif 'macd' in data and 'macd_signal' in data:
        macd = data['macd']
        macd_signal = data['macd_signal']
        
        if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
            return 1  # Buy signal
        elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
            return -1  # Sell signal
    
    return 0  # No signal
"""
        elif strategy_type == "mean_reversion":
            code += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Mean reversion strategy logic
    close = data['close']
    
    # Use RSI if available
    if 'rsi' in data:
        rsi = data['rsi']
        oversold = getattr(context, 'oversold_threshold', 30)
        overbought = getattr(context, 'overbought_threshold', 70)
        
        # Generate signal
        if rsi[-1] < oversold:
            return 1  # Buy signal (oversold)
        elif rsi[-1] > overbought:
            return -1  # Sell signal (overbought)
    
    # Use Bollinger Bands if available
    elif 'bollinger_upper' in data and 'bollinger_lower' in data:
        upper = data['bollinger_upper']
        lower = data['bollinger_lower']
        
        if close[-1] < lower[-1]:
            return 1  # Buy signal (below lower band)
        elif close[-1] > upper[-1]:
            return -1  # Sell signal (above upper band)
    
    return 0  # No signal
"""
        elif strategy_type == "breakout":
            code += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Breakout strategy logic
    close = data['close']
    high = data['high']
    low = data['low']
    
    # Calculate recent highs and lows
    lookback = getattr(context, 'breakout_period', 20)
    period_high = high.rolling(window=lookback).max()
    period_low = low.rolling(window=lookback).min()
    
    # Generate signal
    if close[-1] > period_high[-2]:
        return 1  # Buy signal (breakout above resistance)
    elif close[-1] < period_low[-2]:
        return -1  # Sell signal (breakdown below support)
    
    return 0  # No signal
"""
        else:
            # Generic template for other strategy types
            code += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Generic strategy logic
    close = data['close']
    
    # This is a placeholder for actual strategy logic
    # In a real implementation, you would use the appropriate indicators
    # based on the strategy type
    
    return 0  # No signal by default
"""
        
        return code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BensBot Strategy Importer")
    parser.add_argument("--source", type=str, required=True, help="Path to BensBot strategy file or directory")
    parser.add_argument("--output", type=str, help="Output directory for EvoTrader strategy files")
    parser.add_argument("--meta-db", type=str, help="Path to meta-learning database")
    parser.add_argument("--recursive", action="store_true", help="Search directory recursively")
    
    args = parser.parse_args()
    
    # Create importer
    importer = BenBotStrategyAdapter(meta_db_path=args.meta_db)
    
    # Import strategies
    if os.path.isdir(args.source):
        imported = importer.import_strategy_from_directory(args.source, recursive=args.recursive)
    else:
        imported = [importer.import_strategy_from_path(args.source)]
    
    # Create EvoTrader strategy files
    for strategy_info in imported:
        if strategy_info:
            strategy_id = strategy_info['strategy_id']
            
            # Apply meta-learning optimizations if database is available
            if args.meta_db:
                strategy_info = importer.apply_meta_learning(strategy_id)
            
            # Create strategy file
            file_path = importer.create_evotrader_strategy_file(strategy_id, output_dir=args.output)
            
            if file_path:
                print(f"Created EvoTrader strategy file: {file_path}")
            else:
                print(f"Failed to create EvoTrader strategy file for: {strategy_id}")
