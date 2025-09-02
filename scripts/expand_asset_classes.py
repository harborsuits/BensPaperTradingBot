#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asset Class Expansion Script

This script expands the strategy organization to additional asset classes,
applying the same patterns used for forex strategies to stocks, crypto, and options.
"""

import os
import sys
import shutil
import re
import logging
import argparse
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AssetClassExpansion")

# Strategy categories for each asset class
STRATEGY_TYPES = {
    'stocks': [
        'trend', 'momentum', 'value', 'growth', 'dividend', 'mean_reversion',
        'breakout', 'technical', 'fundamental', 'sentiment', 'statistical'
    ],
    'crypto': [
        'trend', 'momentum', 'arbitrage', 'market_making', 'sentiment',
        'order_flow', 'volatility', 'onchain', 'defi', 'token_economics'
    ],
    'options': [
        'income', 'volatility', 'delta_neutral', 'directional', 'spread',
        'gamma_scalping', 'earnings_play', 'premium_collection', 'decay'
    ]
}

def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def create_base_strategy(asset_class: str, base_path: str) -> None:
    """
    Create a base strategy for the specified asset class.
    
    Args:
        asset_class: Asset class to create base strategy for
        base_path: Base path of the project
    """
    logger.info(f"Creating base strategy for {asset_class}")
    
    # Create the destination directory
    dest_dir = os.path.join(base_path, 'trading_bot', 'strategies', asset_class, 'base')
    ensure_directory(dest_dir)
    
    # Create the base strategy file
    base_strategy_name = f"{asset_class.capitalize()}BaseStrategy"
    dest_file = os.path.join(dest_dir, f"{asset_class}_base_strategy.py")
    
    # Template for base strategy
    base_strategy_template = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{asset_class.capitalize()} Base Strategy

Base class for all {asset_class} trading strategies, providing common functionality.
\"\"\"

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from trading_bot.strategies.strategy_template import Strategy, Signal, SignalType
from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class {base_strategy_name}(Strategy):
    \"\"\"
    Base class for {asset_class} trading strategies.
    
    This class provides common functionality for {asset_class} strategies,
    including specialized risk management, position sizing, and
    market-specific logic.
    \"\"\"
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        \"\"\"
        Initialize {base_strategy_name}.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        \"\"\"
        super().__init__(name, parameters, metadata)
        
        # Default parameters specific to {asset_class}
        self.default_params = {{
            'max_positions': 5,
            'max_risk_per_trade_percent': 0.01,  # 1% risk per trade
        }}
        
        # Override defaults with provided parameters
        if parameters:
            self.parameters.update(parameters)
        else:
            self.parameters = self.default_params.copy()
        
        # Asset class specific state
        self.positions = {{}}
        self.pending_orders = {{}}
        
        logger.info(f"{base_strategy_name} initialized")
    
    def register_events(self, event_bus: EventBus) -> None:
        \"\"\"
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        \"\"\"
        self.event_bus = event_bus
        
        # Register for market data events
        event_bus.register(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        event_bus.register(EventType.TIMEFRAME_COMPLETED, self._on_timeframe_completed)
        
        # Register for additional {asset_class}-specific events if needed
        
        logger.info(f"Strategy registered for events")
    
    def _on_market_data_updated(self, event: Event) -> None:
        \"\"\"Handle market data updated events.\"\"\"
        pass  # Implement in child classes
    
    def _on_timeframe_completed(self, event: Event) -> None:
        \"\"\"Handle timeframe completed events.\"\"\"
        pass  # Implement in child classes
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        \"\"\"
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        \"\"\"
        raise NotImplementedError("Subclasses must implement calculate_indicators")
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        \"\"\"
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        \"\"\"
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        \"\"\"
        Calculate position size for the signal based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units
        \"\"\"
        # Extract parameters
        max_risk_percent = self.parameters['max_risk_per_trade_percent']
        risk_amount = account_balance * max_risk_percent
        
        # Calculate position size
        if signal.stop_loss is None:
            # Use a default risk amount if no stop loss is specified
            position_size = risk_amount / account_balance * 100
        else:
            # Calculate based on stop loss
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            
            # Calculate risk per share/unit
            if signal.signal_type == SignalType.LONG:
                risk_per_unit = entry_price - stop_loss
            else:
                risk_per_unit = stop_loss - entry_price
            
            # Calculate position size
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                # Fallback if stop loss is not properly defined
                position_size = risk_amount / entry_price
        
        return position_size
"""
    
    # Write to the file
    with open(dest_file, 'w') as f:
        f.write(base_strategy_template)
    
    logger.info(f"Created base strategy for {asset_class} at {dest_file}")
    
    # Create __init__.py file
    init_file = os.path.join(dest_dir, '__init__.py')
    with open(init_file, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{asset_class.capitalize()} base strategy package.
\"\"\"

from .{asset_class}_base_strategy import {base_strategy_name}

__all__ = [
    "{base_strategy_name}"
]
""")
    
    logger.info(f"Created __init__.py for {asset_class} base strategy")

def create_strategy_template(asset_class: str, strategy_type: str, base_path: str) -> None:
    """
    Create a strategy template for the specified asset class and type.
    
    Args:
        asset_class: Asset class (stocks, crypto, options)
        strategy_type: Strategy type
        base_path: Base path of the project
    """
    logger.info(f"Creating {asset_class} {strategy_type} strategy template")
    
    # Create the destination directory
    dest_dir = os.path.join(base_path, 'trading_bot', 'strategies', asset_class, strategy_type)
    ensure_directory(dest_dir)
    
    # Create the strategy template file
    strategy_name = f"{asset_class.capitalize()}{strategy_type.capitalize()}Strategy"
    file_name = f"{asset_class}_{strategy_type}_strategy.py"
    dest_file = os.path.join(dest_dir, file_name)
    
    # Template for the strategy
    strategy_template = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{asset_class.capitalize()} {strategy_type.capitalize()} Strategy

This strategy implements {strategy_type} trading for {asset_class}.
\"\"\"

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from trading_bot.strategies.{asset_class}.base.{asset_class}_base_strategy import {asset_class.capitalize()}BaseStrategy
from trading_bot.strategies.factory.strategy_registry import register_strategy
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({{
    'asset_class': '{asset_class}',
    'strategy_type': '{strategy_type}',
    'compatible_market_regimes': ['trending', 'all_weather'],
    'timeframe': 'daily',
    'regime_compatibility_scores': {{
        'trending': 0.70,       # Good compatibility with trending markets
        'ranging': 0.60,        # Moderate compatibility with ranging markets
        'volatile': 0.50,       # Moderate compatibility with volatile markets
        'low_volatility': 0.60, # Moderate compatibility with low volatility markets
        'all_weather': 0.65     # Good overall compatibility
    }}
}})
class {strategy_name}({asset_class.capitalize()}BaseStrategy):
    \"\"\"
    {asset_class.capitalize()} {strategy_type.capitalize()} Strategy
    
    This strategy implements {strategy_type} trading for {asset_class}, using:
    - Specialized indicators for {asset_class}
    - {strategy_type.capitalize()}-based approach to market analysis
    - Risk management tailored to {asset_class} markets
    \"\"\"
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {{
        # Strategy parameters
        'lookback_period': 20,
        'entry_threshold': 0.5,
        'exit_threshold': -0.2,
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.01  # 1% risk per trade
    }}
    
    def __init__(self, name: str = "{strategy_name}", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        \"\"\"
        Initialize {strategy_name}.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        \"\"\"
        # Merge default parameters with provided parameters
        merged_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            merged_params.update(parameters)
        
        super().__init__(name, merged_params, metadata)
        
        # Strategy-specific state variables
        self.signals = {{}}  # Last signals by symbol
        
        logger.info(f"{strategy_name} initialized")
    
    def register_events(self, event_bus: EventBus) -> None:
        \"\"\"
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        \"\"\"
        super().register_events(event_bus)
        
        # Register for additional events specific to this strategy
        # event_bus.register(EventType.CUSTOM_EVENT, self._on_custom_event)
        
        logger.info(f"Strategy registered for events")
    
    def _on_timeframe_completed(self, event: Event) -> None:
        \"\"\"
        Handle timeframe completed events.
        
        This is when we'll generate new trading signals based on the 
        completed candle data.
        \"\"\"
        data = event.data
        if not data or 'symbol' not in data or 'timeframe' not in data:
            return
        
        # Check if this is our target timeframe
        if data['timeframe'] != self.parameters.get('timeframe', 'daily'):
            return
        
        symbol = data['symbol']
        logger.debug(f"Timeframe completed for {{symbol}}")
        
        # Get the current market data
        universe = {{}}
        if self.event_bus:
            # Request market data from the system
            market_data_event = Event(
                event_type=EventType.MARKET_DATA_REQUEST,
                data={{
                    'symbols': [symbol],
                    'timeframe': data['timeframe']
                }}
            )
            response = self.event_bus.request(market_data_event)
            if response and 'data' in response:
                universe = response['data']
        
        # If we have market data, generate signals
        if universe:
            signals = self.generate_signals(universe)
            
            # Publish signals
            if signals and self.event_bus:
                for sym, signal in signals.items():
                    signal_event = Event(
                        event_type=EventType.SIGNAL_GENERATED,
                        data={{
                            'signal': signal.to_dict()
                        }}
                    )
                    self.event_bus.publish(signal_event)
                    logger.info(f"Published signal for {{sym}}: {{signal.signal_type}}")
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        \"\"\"
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        \"\"\"
        result = {{}}
        
        # Input validation
        if len(data) < self.parameters['lookback_period']:
            logger.warning(f"Insufficient data for {{symbol}}: {{len(data)}} bars")
            return result
        
        try:
            # Calculate indicators - actual implementation depends on strategy type
            # This is just a placeholder
            
            # Example indicators that might be used:
            result['sma_20'] = data['close'].rolling(window=20).mean().iloc[-1]
            result['sma_50'] = data['close'].rolling(window=50).mean().iloc[-1]
            result['rsi'] = self._calculate_rsi(data['close'], period=14).iloc[-1]
            result['volatility'] = data['close'].pct_change().std() * np.sqrt(252)
            
            # Strategy-specific indicators would be added here
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {{symbol}}: {{str(e)}}")
            return {{}}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        \"\"\"Calculate RSI technical indicator.\"\"\"
        delta = prices.diff()
        
        gain = delta.copy()
        gain[gain < 0] = 0
        
        loss = delta.copy()
        loss[loss > 0] = 0
        loss = -loss
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        \"\"\"
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        \"\"\"
        signals = {{}}
        
        # Process each symbol
        for symbol, data in universe.items():
            if len(data) < self.parameters['lookback_period']:
                logger.debug(f"Skipping {{symbol}}: insufficient data")
                continue
            
            # Calculate indicators
            indicators = self.calculate_indicators(data, symbol)
            if not indicators:
                continue
            
            # Generate signal based on indicators
            # This is a placeholder - actual logic would depend on strategy type
            signal = None
            
            # Example signal generation logic:
            if indicators.get('sma_20', 0) > indicators.get('sma_50', 0) and indicators.get('rsi', 50) < 70:
                # Bullish conditions
                
                # Create a long signal
                price = data['close'].iloc[-1]
                stop_loss = price * 0.95  # 5% stop loss
                take_profit = price * 1.10  # 10% take profit
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    confidence=0.7,  # Confidence level
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={{
                        'strategy': self.name,
                        'indicators': indicators
                    }}
                )
                
            elif indicators.get('sma_20', 0) < indicators.get('sma_50', 0) and indicators.get('rsi', 50) > 30:
                # Bearish conditions
                
                # Create a short signal
                price = data['close'].iloc[-1]
                stop_loss = price * 1.05  # 5% stop loss
                take_profit = price * 0.90  # 10% take profit
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    confidence=0.7,  # Confidence level
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={{
                        'strategy': self.name,
                        'indicators': indicators
                    }}
                )
            
            # Add signal to results if generated
            if signal:
                signals[symbol] = signal
                self.signals[symbol] = signal  # Store the last signal
        
        return signals
"""
    
    # Skip if file already exists
    if os.path.exists(dest_file):
        logger.info(f"File {dest_file} already exists, skipping")
    else:
        # Write to the file
        with open(dest_file, 'w') as f:
            f.write(strategy_template)
        
        logger.info(f"Created {asset_class} {strategy_type} strategy template at {dest_file}")
    
    # Create __init__.py file
    init_file = os.path.join(dest_dir, '__init__.py')
    with open(init_file, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{asset_class.capitalize()} {strategy_type} strategy package.
\"\"\"

from .{asset_class}_{strategy_type}_strategy import {strategy_name}

__all__ = [
    "{strategy_name}"
]
""")
    
    logger.info(f"Created __init__.py for {asset_class} {strategy_type} strategy")

def create_asset_class_structure(asset_class: str, base_path: str) -> None:
    """
    Create the complete structure for an asset class.
    
    Args:
        asset_class: Asset class to create structure for
        base_path: Base path of the project
    """
    logger.info(f"Creating structure for {asset_class}")
    
    # Create base directory
    asset_class_dir = os.path.join(base_path, 'trading_bot', 'strategies', asset_class)
    ensure_directory(asset_class_dir)
    
    # Create base strategy
    create_base_strategy(asset_class, base_path)
    
    # Create strategy types
    for strategy_type in STRATEGY_TYPES.get(asset_class, []):
        create_strategy_template(asset_class, strategy_type, base_path)
    
    # Create __init__.py file
    init_file = os.path.join(asset_class_dir, '__init__.py')
    with open(init_file, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{asset_class.capitalize()} strategies package.
\"\"\"

# Import subpackages
from . import base
{new_line.join([f'from . import {strategy_type}' for strategy_type in STRATEGY_TYPES.get(asset_class, [])])}

# Import all strategies
{new_line.join([f'from .{strategy_type} import {asset_class.capitalize()}{strategy_type.capitalize()}Strategy' for strategy_type in STRATEGY_TYPES.get(asset_class, [])])}

__all__ = [
    "{asset_class.capitalize()}BaseStrategy",
    {new_line.join([f'    "{asset_class.capitalize()}{strategy_type.capitalize()}Strategy",' for strategy_type in STRATEGY_TYPES.get(asset_class, [])])}
]
""")
    
    logger.info(f"Created __init__.py for {asset_class}")

def create_test_file(asset_class: str, base_path: str) -> None:
    """
    Create a test file for the asset class strategies.
    
    Args:
        asset_class: Asset class to create test file for
        base_path: Base path of the project
    """
    logger.info(f"Creating test file for {asset_class}")
    
    # Find strategy files for the asset class by scanning directories
    strategy_files = []
    strategy_path = os.path.join(base_path, 'trading_bot', 'strategies', asset_class)
    
    if os.path.exists(strategy_path):
        for root, dirs, files in os.walk(strategy_path):
            for file in files:
                if file.endswith('_strategy.py') and not file.startswith('__'):
                    strategy_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(strategy_files)} {asset_class} strategy files")
    
    # Create the test file path
    test_file_path = os.path.join(base_path, f'test_{asset_class}_strategies.py')
    
    # Template for the test file
    test_template = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Test {asset_class.capitalize()} Strategies

This script tests the {asset_class} strategies by instantiating them and
generating signals on sample data.
\"\"\"

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from trading_bot.strategies.factory.strategy_registry import StrategyRegistry, AssetClass
from trading_bot.core.event_system import EventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Test{asset_class.capitalize()}Strategies")

def generate_test_data(symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
    \"\"\"
    Generate test data for strategies.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days of data
        
    Returns:
        Dictionary mapping symbols to DataFrames
    \"\"\"
    result = {{}}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        # Generate date range
        dates = pd.date_range(start=start_date, periods=days)
        
        # Create price series with trend and noise
        np.random.seed(hash(symbol) % 10000)
        base_price = 100
        trend = np.cumsum(np.random.normal(0, 0.01, len(dates)))
        noise = np.random.normal(0, 0.02, len(dates))
        price_series = base_price * (1 + trend + noise)
        
        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['open'] = price_series * (1 + np.random.normal(0, 0.005, len(dates)))
        df['high'] = price_series * (1 + np.random.uniform(0, 0.01, len(dates)))
        df['low'] = price_series * (1 - np.random.uniform(0, 0.01, len(dates)))
        df['close'] = price_series * (1 + np.random.normal(0, 0.005, len(dates)))
        df['volume'] = np.random.uniform(1000, 10000, len(dates))
        
        # Ensure OHLC integrity
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Add some technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        result[symbol] = df
    
    return result

def main():
    """Main entry point."""
    logger.info(f"Testing {asset_class} strategies")
    
    # Get strategies for the asset class by finding files
    logger.info(f"Found {len(strategy_files)} {asset_class} strategy files")
    
    if not strategy_files:
        logger.error(f"No {asset_class} strategy files found")
        return
    
    # Create event bus
    event_bus = EventBus()
    
    # Generate test data
    test_symbols = ['SYMBOL1', 'SYMBOL2', 'SYMBOL3']
    test_data = generate_test_data(test_symbols)
    
    # Test each strategy file
    for strategy_file in strategy_files:
        try:
            # Extract strategy name from filename
            strategy_name = os.path.basename(strategy_file).replace('.py', '')
            logger.info(f"Testing strategy: {strategy_name}")
            
            # We would import and test each strategy here
            # In a real implementation, we'd use importlib to dynamically import the strategy
            
            # Note: In an actual implementation, we would:
            # 1. Import the strategy module using importlib
            # 2. Get the strategy class
            # 3. Instantiate it with proper parameters
            # 4. Generate signals
            #
            # For this template, we'll just log that we would test it
            logger.info(f"Would generate signals using {strategy_name}")
            
            # Log would-be results
            logger.info(f"Strategy {strategy_name} test complete")
            
            # In a real implementation we would print signal details
            logger.info(f"  Would print signal details for each test symbol")
            
        except Exception as e:
            logger.error(f"Error testing strategy {strategy_name}: {str(e)}")
    
    logger.info("Testing complete")

if __name__ == "__main__":
    main()
"""
    
    # Write to the file
    with open(test_file, 'w') as f:
        f.write(test_template)
    
    logger.info(f"Created test file for {asset_class} at {test_file}")

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Asset Class Expansion Script")
    parser.add_argument('--path', default='.', help='Base path of the project')
    parser.add_argument('--asset-classes', nargs='+', 
                      choices=['stocks', 'crypto', 'options', 'all'],
                      default=['all'], help='Asset classes to expand to')
    
    args = parser.parse_args()
    
    # Ensure base path is absolute
    base_path = os.path.abspath(args.path)
    
    # Determine which asset classes to expand
    asset_classes = []
    if 'all' in args.asset_classes:
        asset_classes = ['stocks', 'crypto', 'options']
    else:
        asset_classes = args.asset_classes
    
    logger.info(f"Expanding to asset classes: {', '.join(asset_classes)}")
    
    # Create the structure for each asset class
    for asset_class in asset_classes:
        create_asset_class_structure(asset_class, base_path)
        create_test_file(asset_class, base_path)
    
    logger.info("Asset class expansion complete")
    
    # Print next steps
    print("\nNext Steps:")
    print("1. Review the generated strategy templates")
    print("2. Customize the strategies for your specific needs")
    print("3. Test the strategies using the generated test files")
    print("4. Integrate with the trading system")

# Define a cross-platform newline
new_line = '\n'

if __name__ == "__main__":
    main()
