#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase-Enabled Moving Average Strategy Example

This example shows how to use the CryptoMAStrategy with Coinbase market data.
It demonstrates how to initialize the strategy with Coinbase as the broker
and how to run it in read-only mode for testing.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import strategy components
from trading_bot.strategies_new.crypto.ma_crossover.crypto_ma_crossover_strategy import CryptoMAStrategy
from trading_bot.strategies_new.crypto.mixins.crypto_coinbase_mixin import CoinbaseAPIMixin
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoSession
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.brokers.coinbase_cloud_broker import CoinbaseCloudBroker
from trading_bot.brokers.coinbase_cloud_client import CoinbaseCloudBrokerageClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinbaseMAStrategy(CoinbaseAPIMixin, CryptoMAStrategy):
    """
    Moving Average strategy specifically enhanced for Coinbase integration.
    
    This class inherits from both CoinbaseAPIMixin and CryptoMAStrategy to combine
    the MA strategy logic with Coinbase-specific data handling.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize with Coinbase as the broker
        kwargs['broker_name'] = 'coinbase'
        super().__init__(*args, **kwargs)


def initialize_coinbase_broker():
    """Initialize the Coinbase Cloud broker with the correct API credentials"""
    
    # BenbotReal credentials
    api_key_name = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
    private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
    
    # Initialize broker in read-only mode for safety
    broker = CoinbaseCloudBroker(api_key_name=api_key_name, private_key=private_key, sandbox=False)
    
    # Wrap in the brokerage client for standardized interface
    return CoinbaseCloudBrokerageClient(broker)

def run_strategy():
    """Run the Coinbase MA strategy example"""
    print("Coinbase Moving Average Strategy Example")
    print("========================================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize broker
    broker = initialize_coinbase_broker()
    
    # Test broker connection
    print("\nTesting Coinbase broker connection...")
    connection_status = broker.check_connection()
    print(f"Connection status: {connection_status}")
    
    # Initialize session and data pipeline
    session = CryptoSession(
        symbol="BTC-USD",
        timeframe="H1",
        exchange="Coinbase",
        broker=broker
    )
    
    data_pipeline = DataPipeline()
    
    # Strategy parameters
    strategy_params = {
        "fast_ma_type": "EMA",
        "slow_ma_type": "EMA",
        "fast_ma_period": 9,
        "slow_ma_period": 21,
        "use_volume_confirmation": True,
        "volume_ma_period": 20,
        "use_atr_for_stops": True,
        "atr_period": 14,
        "risk_per_trade": 0.01,
    }
    
    # Initialize strategy
    strategy = CoinbaseMAStrategy(session, data_pipeline, parameters=strategy_params)
    
    # Fetch historical data from Coinbase
    print("\nFetching historical data for BTC-USD from Coinbase...")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # Get 30 days of data
    
    historical_data = strategy.get_market_data(
        symbol="BTC-USD",
        timeframe="H1",
        start=start_time,
        end=end_time
    )
    
    if historical_data.empty:
        print("❌ Failed to fetch historical data")
        return
    
    print(f"✅ Retrieved {len(historical_data)} candles of historical data")
    print("\nRecent data:")
    print(historical_data.tail(5))
    
    # Calculate indicators
    print("\nCalculating indicators...")
    indicators = strategy.calculate_indicators(historical_data)
    
    print("Indicators calculated:")
    for key, value in indicators.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            print(f"- {key}: {type(value)}, last value: {value.iloc[-1]}")
        else:
            print(f"- {key}: {type(value)}")
    
    # Generate signals
    print("\nGenerating trading signals...")
    signals = strategy.generate_signals(historical_data, indicators)
    
    print("Signals generated:")
    for key, value in signals.items():
        print(f"- {key}: {value}")
    
    # Calculate position size example
    if signals.get('entry_signal'):
        print("\nCalculating position size for entry signal...")
        direction = 'long' if signals.get('direction') == 'buy' else 'short'
        position_size = strategy.calculate_position_size(direction, historical_data, indicators)
        print(f"Recommended position size: {position_size} BTC")
    
    # Simulate strategy on recent data
    print("\nSimulating strategy performance on recent data...")
    recent_data = historical_data[-100:]  # Last 100 candles
    
    # Track trades
    trades = []
    
    # Simple backtest implementation
    in_position = False
    position_direction = None
    entry_price = 0
    entry_time = None
    
    for i in range(len(recent_data)):
        if i < max(strategy_params['fast_ma_period'], strategy_params['slow_ma_period']):
            continue
            
        # Get the current candle
        current_data = recent_data.iloc[:i+1]
        
        # Calculate indicators for the current data
        current_indicators = strategy.calculate_indicators(current_data)
        
        # Generate signals
        current_signals = strategy.generate_signals(current_data, current_indicators)
        
        current_price = current_data.iloc[-1]['close']
        current_time = current_data.index[-1]
        
        # Process signals
        if not in_position and current_signals.get('entry_signal'):
            in_position = True
            position_direction = current_signals.get('direction')
            entry_price = current_price
            entry_time = current_time
            
            print(f"ENTRY: {position_direction} at {entry_price} on {entry_time}")
            
        elif in_position and current_signals.get('exit_signal'):
            # Calculate profit/loss
            pnl = 0
            if position_direction == 'buy':
                pnl = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) / entry_price
                
            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': current_price,
                'direction': position_direction,
                'pnl': pnl
            })
            
            print(f"EXIT: {position_direction} at {current_price} on {current_time}, PNL: {pnl:.2%}")
            
            in_position = False
            position_direction = None
            entry_price = 0
            entry_time = None
    
    # Print trade summary
    if trades:
        print("\nTrade Summary:")
        print(f"Total trades: {len(trades)}")
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades):.2%})")
        
        total_pnl = sum(t['pnl'] for t in trades)
        print(f"Total PnL: {total_pnl:.2%}")
        
        avg_pnl = total_pnl / len(trades)
        print(f"Average PnL per trade: {avg_pnl:.2%}")
    else:
        print("\nNo trades were generated during the simulation period.")
    
    print("\nStrategy simulation complete!")
    
    print("\n--- Current Market Conditions ---")
    current_price = historical_data.iloc[-1]['close']
    current_time = historical_data.index[-1]
    fast_ma = indicators['fast_ma'].iloc[-1]
    slow_ma = indicators['slow_ma'].iloc[-1]
    
    print(f"BTC-USD price: ${current_price} as of {current_time}")
    print(f"Fast MA ({strategy_params['fast_ma_period']}): ${fast_ma:.2f}")
    print(f"Slow MA ({strategy_params['slow_ma_period']}): ${slow_ma:.2f}")
    
    if fast_ma > slow_ma:
        print("Current trend: BULLISH (Fast MA above Slow MA)")
    else:
        print("Current trend: BEARISH (Fast MA below Slow MA)")
        
    if signals.get('entry_signal'):
        print(f"Current signal: {signals.get('direction', 'unknown').upper()}")
    else:
        print("Current signal: NONE")
    
    print("\nNext steps:")
    print("1. Integrate this strategy with your trading dashboard")
    print("2. Configure position sizing to your specific risk parameters")
    print("3. Set up alerts for new signals")
    print("4. Test in read-only mode before enabling live trading")

if __name__ == "__main__":
    run_strategy()
