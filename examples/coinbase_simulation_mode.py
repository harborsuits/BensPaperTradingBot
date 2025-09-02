#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase API Simulation Mode

This script demonstrates a complete paper trading simulation using the Coinbase API
for market data while simulating all trading activity.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.brokers.coinbase_broker_client import CoinbaseBrokerageClient
from trading_bot.brokers.broker_registry import get_broker_registry
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoSession
from trading_bot.strategies_new.crypto.defi.yield_farming_strategy import YieldFarmingStrategy
from trading_bot.core.constants import TimeFrame
from trading_bot.core.event_bus import EventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinbaseSimulationClient:
    """
    A Coinbase client simulator that uses real market data but simulates trading.
    """
    
    def __init__(self, real_client, initial_balance=10000.0):
        """
        Initialize the simulation client.
        
        Args:
            real_client: The actual CoinbaseBrokerageClient for market data
            initial_balance: Starting USD balance for simulation
        """
        self.real_client = real_client
        self.positions = {}
        self.orders = []
        self.order_id_counter = 1
        self.transaction_history = []
        
        # Initialize with starting balances
        self.balances = {
            'USD': Decimal(str(initial_balance)),
            'BTC': Decimal('0'),
            'ETH': Decimal('0'),
            'SOL': Decimal('0'),
            'LINK': Decimal('0'),
            'AVAX': Decimal('0')
        }
        
        # Track current prices
        self.current_prices = {}
        
        # Simulation configuration
        self.slippage_pct = 0.001  # 0.1% slippage
        self.maker_fee_pct = 0.0015  # 0.15% maker fee
        self.taker_fee_pct = 0.0025  # 0.25% taker fee
        
        # Portfolio value history
        self.portfolio_history = []
        self.last_update_time = datetime.now()
        
        # Update initial portfolio value
        self._update_portfolio_value()
        
    def _update_portfolio_value(self):
        """Calculate and record current portfolio value."""
        total_value = self.balances['USD']
        
        # Add value of all crypto positions
        for symbol, amount in self.balances.items():
            if symbol != 'USD' and amount > 0:
                # Get current price for this asset
                try:
                    price = self.get_latest_price(f"{symbol}-USD")
                    total_value += amount * Decimal(str(price))
                except Exception as e:
                    logger.warning(f"Couldn't get price for {symbol}: {e}")
        
        # Record in history
        current_time = datetime.now()
        self.portfolio_history.append({
            'timestamp': current_time,
            'value': float(total_value)
        })
        self.last_update_time = current_time
        
        return float(total_value)
        
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol, with caching."""
        # If we have a recent price (last 60 seconds), use it
        if symbol in self.current_prices:
            price_data = self.current_prices[symbol]
            if datetime.now().timestamp() - price_data['timestamp'] < 60:
                return price_data['price']
        
        # Otherwise fetch a new price
        try:
            quote = self.real_client.get_quote(symbol)
            price = float(quote.get('last', 0))
            self.current_prices[symbol] = {
                'price': price,
                'timestamp': datetime.now().timestamp()
            }
            return price
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            # If we have any price at all, return it
            if symbol in self.current_prices:
                return self.current_prices[symbol]['price']
            return 0
    
    # Methods that use real data from the Coinbase API
    def get_quote(self, symbol):
        """Get real market quote from Coinbase."""
        return self.real_client.get_quote(symbol)
    
    def get_bars(self, symbol, timeframe, start, end):
        """Get real historical bars from Coinbase."""
        return self.real_client.get_bars(symbol, timeframe, start, end)
    
    def check_connection(self):
        """Check connection to Coinbase API."""
        return self.real_client.check_connection()
    
    # Simulated trading methods
    def place_order(self, symbol, side, quantity, order_type, price=None):
        """
        Simulate placing an order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: 'buy' or 'sell'
            quantity: Amount to buy/sell
            order_type: 'market' or 'limit'
            price: Limit price (for limit orders)
            
        Returns:
            Simulated order response
        """
        # Extract the base currency from the symbol
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        
        # Get the current price
        current_price = self.get_latest_price(symbol)
        
        # Calculate order value
        if order_type == 'market':
            execution_price = current_price
            
            # Apply slippage
            if side == 'buy':
                execution_price *= (1 + self.slippage_pct)
            else:
                execution_price *= (1 - self.slippage_pct)
                
            # Apply fees
            fee_pct = self.taker_fee_pct
        else:  # limit order
            execution_price = float(price)
            fee_pct = self.maker_fee_pct
        
        # Calculate total value and fees
        total_value = float(quantity) * execution_price
        fee_amount = total_value * fee_pct
        
        # Check if we have enough balance
        if side == 'buy':
            if self.balances[quote_currency] < Decimal(str(total_value + fee_amount)):
                return {
                    'status': 'rejected',
                    'message': f'Insufficient {quote_currency} balance'
                }
        else:  # sell
            if self.balances[base_currency] < Decimal(str(quantity)):
                return {
                    'status': 'rejected',
                    'message': f'Insufficient {base_currency} balance'
                }
        
        # Create order object
        order_id = f"sim-{self.order_id_counter}"
        self.order_id_counter += 1
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'type': order_type,
            'price': execution_price,
            'status': 'filled' if order_type == 'market' else 'open',
            'created_at': datetime.now().isoformat(),
            'filled_at': datetime.now().isoformat() if order_type == 'market' else None,
            'fee': fee_amount
        }
        
        self.orders.append(order)
        
        # Execute market orders immediately
        if order_type == 'market':
            self._execute_order(order)
        
        return {
            'status': 'success',
            'order_id': order_id,
            'message': f'{side.capitalize()} {quantity} {base_currency} at {execution_price} {quote_currency}'
        }
    
    def _execute_order(self, order):
        """
        Execute a simulated order by updating balances.
        
        Args:
            order: Order dict to execute
        """
        symbol = order['symbol']
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        
        total_value = order['quantity'] * order['price']
        fee = order['fee']
        
        # Update balances
        if order['side'] == 'buy':
            # Deduct quote currency (e.g., USD)
            self.balances[quote_currency] -= Decimal(str(total_value + fee))
            # Add base currency (e.g., BTC)
            self.balances[base_currency] += Decimal(str(order['quantity']))
        else:  # sell
            # Add quote currency (e.g., USD)
            self.balances[quote_currency] += Decimal(str(total_value - fee))
            # Deduct base currency (e.g., BTC)
            self.balances[base_currency] -= Decimal(str(order['quantity']))
        
        # Record transaction
        transaction = {
            'order_id': order['id'],
            'type': f"{order['side']}_{order['type']}",
            'symbol': symbol,
            'quantity': order['quantity'],
            'price': order['price'],
            'value': total_value,
            'fee': fee,
            'timestamp': datetime.now().isoformat()
        }
        self.transaction_history.append(transaction)
        
        # Update current positions
        self._update_positions()
        
        # Update portfolio value
        self._update_portfolio_value()
        
        return True
    
    def _update_positions(self):
        """Update the current positions based on balances."""
        self.positions = {}
        
        for currency, amount in self.balances.items():
            if currency != 'USD' and amount > 0:
                symbol = f"{currency}-USD"
                price = self.get_latest_price(symbol)
                
                self.positions[currency] = {
                    'symbol': symbol,
                    'quantity': float(amount),
                    'current_price': price,
                    'value_usd': float(amount) * price,
                    'last_updated': datetime.now().isoformat()
                }
    
    # Account information methods
    def get_account_info(self):
        """Get simulated account information."""
        self._update_positions()
        portfolio_value = self._update_portfolio_value()
        
        return {
            'id': 'simulation-account',
            'total_value': portfolio_value,
            'cash_value': float(self.balances['USD']),
            'balances': {k: float(v) for k, v in self.balances.items()},
            'margin_used': 0,
            'margin_available': 0,
            'status': 'active',
            'last_updated': datetime.now().isoformat()
        }
    
    def get_positions(self):
        """Get current simulated positions."""
        self._update_positions()
        return list(self.positions.values())
    
    def get_orders(self, status=None):
        """Get simulated orders with optional status filter."""
        if status:
            return [order for order in self.orders if order['status'] == status]
        return self.orders
    
    def cancel_order(self, order_id):
        """Cancel a simulated order."""
        for i, order in enumerate(self.orders):
            if order['id'] == order_id and order['status'] == 'open':
                self.orders[i]['status'] = 'canceled'
                return {'status': 'success', 'message': f'Order {order_id} canceled'}
        
        return {'status': 'error', 'message': f'Order {order_id} not found or not open'}
    
    def get_performance(self, timeframe='daily'):
        """
        Get performance metrics for the simulation.
        
        Args:
            timeframe: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Performance metrics
        """
        if not self.portfolio_history:
            return {
                'return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'values': []
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample based on timeframe
        if timeframe == 'daily':
            df = df.resample('D').last().fillna(method='ffill')
        elif timeframe == 'weekly':
            df = df.resample('W').last().fillna(method='ffill')
        elif timeframe == 'monthly':
            df = df.resample('M').last().fillna(method='ffill')
        
        # Calculate performance metrics
        initial_value = df['value'].iloc[0]
        final_value = df['value'].iloc[-1]
        return_pct = (final_value / initial_value - 1) * 100
        
        # Daily returns for Sharpe ratio
        df['daily_return'] = df['value'].pct_change()
        sharpe_ratio = 0
        if len(df) > 1:
            sharpe_ratio = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)
            
        # Maximum drawdown
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] / df['cummax'] - 1) * 100
        max_drawdown = df['drawdown'].min()
        
        return {
            'return_pct': return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'values': df['value'].tolist(),
            'timestamps': df.index.tolist()
        }
    
    def save_simulation_results(self, filename):
        """
        Save the simulation results to a file.
        
        Args:
            filename: Name of the file to save to
        """
        performance = self.get_performance()
        
        results = {
            'simulation_config': {
                'slippage_pct': self.slippage_pct,
                'maker_fee_pct': self.maker_fee_pct,
                'taker_fee_pct': self.taker_fee_pct
            },
            'final_balances': {k: float(v) for k, v in self.balances.items()},
            'positions': self.positions,
            'performance': performance,
            'transactions': self.transaction_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Simulation results saved to {filename}")


def run_simulation():
    """Run a simulation of a crypto strategy using real market data."""
    # Set up Coinbase API client (only for market data)
    coinbase_config = {
        'api_key': os.environ.get('COINBASE_API_KEY', 'your_api_key_here'),
        'api_secret': os.environ.get('COINBASE_API_SECRET', 'your_api_secret_here'),
        'passphrase': os.environ.get('COINBASE_PASSPHRASE', None)
    }
    
    try:
        # Create real client for market data
        real_client = CoinbaseBrokerageClient(**coinbase_config)
        
        # Verify connection
        connection_status = real_client.check_connection()
        if not connection_status.get('connected', False):
            logger.error("Could not connect to Coinbase API")
            return
            
        logger.info("Connected to Coinbase API successfully")
        
        # Create simulation client
        sim_client = CoinbaseSimulationClient(real_client, initial_balance=10000.0)
        
        # Register with broker registry for strategy use
        broker_registry = get_broker_registry()
        broker_registry.register_broker('coinbase_sim', sim_client)
        
        # Set up event bus
        event_bus = EventBus()
        
        # Create a crypto session for ETH-USD
        session = CryptoSession(
            symbol="ETH-USD",
            timeframe=TimeFrame.HOUR_1,
            exchange="Coinbase",
            quote_currency="USD"
        )
        
        # Create the yield farming strategy
        strategy_params = {
            "rebalance_frequency_hours": 24,
            "min_apy_threshold": 5.0,
            "risk_profile": "medium",
            "max_gas_per_tx_usd": 50.0
        }
        
        strategy = YieldFarmingStrategy(
            session=session,
            data_pipeline=None,  # We'll feed data directly
            parameters=strategy_params
        )
        
        # Register for events
        strategy.register_for_events(event_bus)
        
        # Get historical data for backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = sim_client.get_bars(
            symbol="ETH-USD",
            timeframe="1h",
            start=start_date,
            end=end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Initialize strategy with account data
        account_info = sim_client.get_account_info()
        strategy.update_account_info({
            'balances': account_info.get('balances', {}),
            'positions': sim_client.get_positions(),
            'lending_positions': [],
            'borrowing_positions': [],
            'staking_positions': [],
            'lp_positions': []
        })
        
        # Initialize with historical data
        strategy.on_data(df)
        
        # Run simulation
        logger.info("Starting simulation...")
        
        # Simulate 5 days of trading (one bar per hour)
        simulation_days = 5
        hours_per_day = 24
        total_bars = simulation_days * hours_per_day
        
        for i in range(total_bars):
            try:
                # Get current market data
                quote = sim_client.get_quote("ETH-USD")
                
                # Create a simulated bar
                current_time = datetime.now() + timedelta(hours=i)
                
                # Some randomness to price movement
                price_change_pct = np.random.normal(0, 0.005)  # 0.5% std dev
                last_price = float(quote.get('last', 1000))
                new_price = last_price * (1 + price_change_pct)
                
                bar = {
                    'symbol': "ETH-USD",
                    'timestamp': current_time.isoformat(),
                    'open': last_price,
                    'high': max(last_price, new_price),
                    'low': min(last_price, new_price),
                    'close': new_price,
                    'volume': np.random.randint(10, 100)
                }
                
                # Update simulated latest price
                sim_client.current_prices["ETH-USD"] = {
                    'price': new_price,
                    'timestamp': current_time.timestamp()
                }
                
                # Publish market data event
                event_bus.publish({
                    'type': 'MARKET_DATA',
                    'data': bar
                })
                
                # Update simulation portfolio value
                sim_client._update_portfolio_value()
                
                # Log every 6 hours
                if i % 6 == 0:
                    account = sim_client.get_account_info()
                    logger.info(f"Simulation hour {i}: Portfolio value ${account['total_value']:.2f}")
                
                # Wait a very short time to avoid consuming all CPU
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in simulation step {i}: {str(e)}")
        
        # Simulation complete
        final_account = sim_client.get_account_info()
        performance = sim_client.get_performance()
        
        logger.info("Simulation complete!")
        logger.info(f"Final portfolio value: ${final_account['total_value']:.2f}")
        logger.info(f"Return: {performance['return_pct']:.2f}%")
        logger.info(f"Max drawdown: {performance['max_drawdown_pct']:.2f}%")
        
        # Save simulation results
        sim_client.save_simulation_results("crypto_simulation_results.json")
        
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
    finally:
        # Clean up
        broker_registry.disconnect_all()
        logger.info("Simulation resources released")


if __name__ == "__main__":
    run_simulation()
