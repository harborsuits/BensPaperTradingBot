#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Crypto Strategy Example

This example shows how to use the Coinbase broker integration with
the account-aware crypto trading strategies.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.brokers.coinbase_broker_client import CoinbaseBrokerageClient
from trading_bot.brokers.broker_registry import get_broker_registry
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoSession
from trading_bot.strategies_new.crypto.scalping.crypto_scalping_strategy import CryptoScalpingStrategy
from trading_bot.strategies_new.crypto.mixins.crypto_account_aware_mixin import CryptoAccountAwareMixin
from trading_bot.strategies_new.crypto.mixins.defi_strategy_mixin import DeFiStrategyMixin
from trading_bot.strategies_new.crypto.defi.yield_farming_strategy import YieldFarmingStrategy
from trading_bot.strategies_new.crypto.defi.on_chain_analysis_strategy import OnChainAnalysisStrategy
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.constants import TimeFrame
from trading_bot.core.event_bus import EventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the Coinbase crypto strategy example."""
    # ------------------------------------
    # Step 1: Set up your Coinbase API credentials
    # ------------------------------------
    # Replace these with your actual Coinbase API credentials
    # For security, you should use environment variables instead
    coinbase_config = {
        'api_key': os.environ.get('COINBASE_API_KEY', 'your_api_key_here'),
        'api_secret': os.environ.get('COINBASE_API_SECRET', 'your_api_secret_here'),
        'passphrase': os.environ.get('COINBASE_PASSPHRASE', None),  # Optional for Advanced API
        'sandbox': False  # Set to True for testing with sandbox environment
    }
    
    # ------------------------------------
    # Step 2: Create Coinbase broker client
    # ------------------------------------
    try:
        coinbase_client = CoinbaseBrokerageClient(**coinbase_config)
        
        # Check connection
        connection_status = coinbase_client.check_connection()
        logger.info(f"Coinbase connection status: {connection_status}")
        
        # Register with broker registry for global access
        broker_registry = get_broker_registry()
        broker_registry.register_broker('coinbase', coinbase_client)
        
        logger.info("Coinbase broker client registered successfully")
    except Exception as e:
        logger.error(f"Error setting up Coinbase client: {str(e)}")
        return
    
    # ------------------------------------
    # Step 3: Set up the event bus
    # ------------------------------------
    event_bus = EventBus()
    
    # ------------------------------------
    # Step 4: Create a data pipeline
    # ------------------------------------
    data_pipeline = DataPipeline()
    
    # ------------------------------------
    # Step 5: Create trading session and strategy
    # ------------------------------------
    # Create a session for BTC-USD with 1-hour timeframe
    session = CryptoSession(
        symbol="BTC-USD",
        timeframe=TimeFrame.HOUR_1,
        exchange="Coinbase",
        quote_currency="USD"
    )
    
    # Create strategy parameters
    strategy_params = {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "ma_short": 9,
        "ma_long": 21,
        "risk_per_trade": 0.01,  # 1% risk per trade
        "max_trades_per_day": 5
    }
    
    # Create a crypto strategy with account awareness
    # Here we're demonstrating with a scalping strategy
    class AccountAwareCryptoStrategy(CryptoScalpingStrategy, CryptoAccountAwareMixin):
        """Crypto scalping strategy with account awareness."""
        
        def __init__(self, session, data_pipeline, parameters=None):
            CryptoScalpingStrategy.__init__(self, session, data_pipeline, parameters)
            CryptoAccountAwareMixin.__init__(self)
            
        def _check_for_trade_opportunities(self):
            """Override to add account awareness checks."""
            # First check account limits and risk
            risk_ok, reason = self.check_risk_limits()
            if not risk_ok:
                logger.warning(f"Risk check failed: {reason}")
                return
                
            # Call the original method to check for trading opportunities
            super()._check_for_trade_opportunities()
            
        def _open_position(self, direction, size):
            """Override to add account validation before opening positions."""
            # Validate order against account constraints
            is_valid, reason = self.validate_order(
                symbol=self.session.symbol,
                order_type="market",
                side=direction,
                amount=size
            )
            
            if not is_valid:
                logger.warning(f"Order validation failed: {reason}")
                return
                
            # Use account-aware position sizing
            account_size = size if size > 0 else self.calculate_position_size(
                symbol=self.session.symbol,
                risk_per_trade_pct=self.parameters.get('risk_per_trade', 0.01),
                stop_loss_pct=0.02  # 2% stop loss
            )
            
            # Call the original method with the validated size
            super()._open_position(direction, account_size)
    
    # Create the account-aware strategy
    strategy = AccountAwareCryptoStrategy(
        session=session,
        data_pipeline=data_pipeline,
        parameters=strategy_params
    )
    
    # Register for events
    strategy.register_for_events(event_bus)
    
    # ------------------------------------
    # Step 6: Initialize with account data
    # ------------------------------------
    try:
        # Get account information from Coinbase
        account_info = coinbase_client.get_account_info()
        logger.info(f"Account Balance: ${account_info.get('total_value', 0):.2f}")
        
        # Feed account data to the strategy
        strategy.update_account_info({
            'balances': account_info.get('balances', {}),
            'positions': coinbase_client.get_positions(),
            'lending_positions': [],
            'borrowing_positions': [],
            'staking_positions': [],
            'lp_positions': []
        })
        
        # Get current positions
        positions = coinbase_client.get_positions()
        if positions:
            logger.info(f"Current positions: {len(positions)}")
            for position in positions:
                logger.info(f"  {position['symbol']}: {position['quantity']} @ ${position['current_price']:.2f}")
    except Exception as e:
        logger.error(f"Error fetching account data: {str(e)}")
    
    # ------------------------------------
    # Step 7: Get market data
    # ------------------------------------
    try:
        # Get historical data for strategy initialization
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Get 30 days of data
        
        historical_data = coinbase_client.get_bars(
            symbol="BTC-USD",
            timeframe="1h",
            start=start_date,
            end=end_date
        )
        
        # Convert to DataFrame for the strategy
        import pandas as pd
        df = pd.DataFrame(historical_data)
        
        # Process through the data pipeline
        df = data_pipeline.process(df)
        
        logger.info(f"Loaded {len(df)} historical bars for BTC-USD")
        
        # Feed data to the strategy
        strategy.on_data(df)
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
    
    # ------------------------------------
    # Step 8: Real-time market data setup
    # ------------------------------------
    def publish_market_data():
        """Fetch and publish market data periodically."""
        while True:
            try:
                # Get current market data
                quote = coinbase_client.get_quote("BTC-USD")
                
                # Publish market data event
                event_bus.publish({
                    'type': 'MARKET_DATA',
                    'data': {
                        'symbol': "BTC-USD",
                        'timestamp': datetime.now().isoformat(),
                        'open': quote.get('last', 0),
                        'high': quote.get('last', 0),
                        'low': quote.get('last', 0),
                        'close': quote.get('last', 0),
                        'volume': quote.get('volume', 0)
                    }
                })
                
                logger.info(f"BTC-USD: ${quote.get('last', 0):.2f}")
                
                # Wait before fetching again
                time.sleep(60)  # 1-minute interval
                
            except Exception as e:
                logger.error(f"Error fetching market data: {str(e)}")
                time.sleep(10)  # Wait on error
    
    # ------------------------------------
    # Step 9: DeFi strategy example
    # ------------------------------------
    def setup_defi_strategy():
        """Set up a DeFi yield farming strategy as another example."""
        # Create a session for ETH-USD
        defi_session = CryptoSession(
            symbol="ETH-USD",
            timeframe=TimeFrame.HOUR_4,
            exchange="Coinbase",
            quote_currency="USD"
        )
        
        # Create strategy parameters
        defi_params = {
            "rebalance_frequency_hours": 24,
            "min_apy_threshold": 5.0,
            "risk_profile": "medium",
            "max_gas_per_tx_usd": 50.0
        }
        
        # Create the yield farming strategy
        yield_strategy = YieldFarmingStrategy(
            session=defi_session,
            data_pipeline=data_pipeline,
            parameters=defi_params
        )
        
        # Register for events
        yield_strategy.register_for_events(event_bus)
        
        # Initialize with account data
        yield_strategy.update_account_info({
            'balances': account_info.get('balances', {}),
            'positions': coinbase_client.get_positions(),
            'lending_positions': [],
            'borrowing_positions': [],
            'staking_positions': [],
            'lp_positions': []
        })
        
        logger.info("DeFi yield farming strategy initialized")
        return yield_strategy
    
    # Set up the DeFi strategy (uncomment to use)
    # defi_strategy = setup_defi_strategy()
    
    # ------------------------------------
    # Step 10: Run the strategy (demo only)
    # ------------------------------------
    logger.info("Starting strategy demo (press Ctrl+C to exit)...")
    
    try:
        # In a real implementation, this would run in a separate thread
        # For this demo, we'll just fetch data a few times
        for _ in range(5):
            # Get current market data
            quote = coinbase_client.get_quote("BTC-USD")
            
            # Publish market data event
            event_bus.publish({
                'type': 'MARKET_DATA',
                'data': {
                    'symbol': "BTC-USD",
                    'timestamp': datetime.now().isoformat(),
                    'open': quote.get('last', 0),
                    'high': quote.get('last', 0),
                    'low': quote.get('last', 0),
                    'close': quote.get('last', 0),
                    'volume': quote.get('volume', 0)
                }
            })
            
            logger.info(f"BTC-USD: ${quote.get('last', 0):.2f}")
            
            # Wait before fetching again
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Strategy demo stopped by user")
    except Exception as e:
        logger.error(f"Error in strategy execution: {str(e)}")
    finally:
        # Clean up
        broker_registry.disconnect_all()
        logger.info("Strategy demo completed")

if __name__ == "__main__":
    main()
