#!/usr/bin/env python3
"""
Data Integration Example

This example script shows how to integrate real-time market data processing
with market regime detection and trading strategy execution.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import json
from pprint import pprint

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import local modules
from trading_bot.data.real_time_data_processor import RealTimeDataManager
from trading_bot.optimization.advanced_market_regime_detector import AdvancedMarketRegimeDetector
from trading_bot.optimization.strategy_regime_rotator import StrategyRegimeRotator

# Import strategy classes (adapt this to your actual strategy implementations)
from trading_bot.strategy.moving_average_strategy import MovingAverageStrategy
from trading_bot.strategy.mean_reversion_strategy import MeanReversionStrategy
from trading_bot.strategy.momentum_strategy import MomentumStrategy
from trading_bot.strategy.volatility_strategy import VolatilityStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_integration.log')
    ]
)

logger = logging.getLogger(__name__)

class MockTradingEngine:
    """
    Mock trading engine for demonstration purposes.
    
    In a real implementation, this would connect to a broker API and execute trades.
    """
    
    def __init__(self, initial_capital=100000.0):
        """
        Initialize the mock trading engine.
        
        Args:
            initial_capital: Initial portfolio capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        
        self.logger = logging.getLogger(f"{__name__}.MockTradingEngine")
    
    def place_order(self, symbol, order_type, quantity, price=None, stop_price=None):
        """
        Place a mock order.
        
        Args:
            symbol: Trading symbol
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            quantity: Order quantity (negative for sell)
            price: Limit price (if applicable)
            stop_price: Stop price (if applicable)
            
        Returns:
            dict: Order details
        """
        order_id = len(self.orders) + 1
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'stop_price': stop_price,
            'status': 'submitted',
            'filled_quantity': 0,
            'filled_price': None,
            'timestamp': datetime.now()
        }
        
        self.orders.append(order)
        self.logger.info(f"Placed order: {order}")
        
        # Simulate immediate fill for market orders
        if order_type == 'market':
            self._fill_order(order_id, quantity, price or 100.0)  # Mock price if not provided
        
        return order
    
    def _fill_order(self, order_id, filled_quantity, filled_price):
        """
        Simulate filling an order.
        
        Args:
            order_id: Order ID
            filled_quantity: Quantity filled
            filled_price: Fill price
        """
        for order in self.orders:
            if order['order_id'] == order_id:
                # Update order
                order['status'] = 'filled'
                order['filled_quantity'] = filled_quantity
                order['filled_price'] = filled_price
                
                # Record trade
                trade = {
                    'order_id': order_id,
                    'symbol': order['symbol'],
                    'quantity': filled_quantity,
                    'price': filled_price,
                    'timestamp': datetime.now(),
                    'commission': 0.0  # Mock commission
                }
                
                self.trades.append(trade)
                
                # Update position
                symbol = order['symbol']
                if symbol not in self.positions:
                    self.positions[symbol] = 0
                
                self.positions[symbol] += filled_quantity
                
                # Update capital (simplified)
                self.current_capital -= filled_quantity * filled_price
                
                self.logger.info(f"Filled order {order_id}: {filled_quantity} @ {filled_price}")
                break
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
            dict: Current positions
        """
        return self.positions
    
    def get_portfolio_value(self, current_prices):
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            float: Portfolio value
        """
        positions_value = sum(
            qty * current_prices.get(symbol, 100.0)  # Use 100 as default price if not provided
            for symbol, qty in self.positions.items()
        )
        
        return self.current_capital + positions_value


class TradingSystem:
    """
    Integrated trading system that combines real-time data, regime detection, and trading.
    """
    
    def __init__(self, symbols, config=None):
        """
        Initialize the trading system.
        
        Args:
            symbols: List of symbols to trade
            config: Configuration dictionary
        """
        self.symbols = symbols
        self.config = config or {}
        
        # Initialize trading engine
        self.trading_engine = MockTradingEngine(
            initial_capital=self.config.get('initial_capital', 100000.0)
        )
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Initialize real-time data manager
        self.data_manager = RealTimeDataManager(
            symbols=symbols,
            config={
                'data_source': self.config.get('data_source', 'alpaca'),
                'alpaca_config': self.config.get('alpaca_config', {}),
                'timeframes': self.config.get('timeframes', ['1min', '5min', '15min', '1hour', '1day']),
                'use_market_regimes': True,
                'regime_config': self.config.get('regime_config', {}),
                'use_strategy_rotation': True,
                'strategies': self.strategies,
                'lookback_window': self.config.get('lookback_window', 60),
                'rebalance_frequency': self.config.get('rebalance_frequency', 'daily')
            }
        )
        
        # Register event handlers
        self.data_manager.on_regime_change = self._on_regime_change
        self.data_manager.on_bar_update = self._on_bar_update
        self.data_manager.on_strategy_update = self._on_strategy_update
        
        # Trading state
        self.is_trading = False
        self.trading_thread = None
        
        # Performance tracking
        self.portfolio_history = []
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.TradingSystem")
    
    def _initialize_strategies(self):
        """
        Initialize trading strategies.
        
        Returns:
            list: Strategy instances
        """
        # Create strategy instances (adapt parameters to your needs)
        strategies = [
            MovingAverageStrategy(
                name="MA_Trend",
                strategy_type="trend_following",
                fast_period=10,
                slow_period=30
            ),
            MeanReversionStrategy(
                name="Mean_Reversion",
                strategy_type="mean_reversion",
                lookback_period=20,
                z_score_threshold=2.0
            ),
            MomentumStrategy(
                name="Momentum",
                strategy_type="momentum",
                momentum_period=20,
                threshold=0.01
            ),
            VolatilityStrategy(
                name="Volatility_Breakout",
                strategy_type="volatility",
                atr_period=14,
                multiplier=2.0
            )
        ]
        
        return strategies
    
    def _on_regime_change(self, regime):
        """
        Handle market regime changes.
        
        Args:
            regime: New market regime
        """
        self.logger.info(f"Market regime changed to: {regime}")
        
        # Get regime information
        if hasattr(self.data_manager, 'market_regime_detector') and self.data_manager.market_regime_detector:
            # Get the last symbol with data
            symbol_data = None
            for symbol, data in self.data_manager.market_data_cache.items():
                if data is not None and not data.empty:
                    symbol_data = data
                    break
            
            if symbol_data is not None:
                # Get detailed regime info
                regime_info = self.data_manager.market_regime_detector.get_current_regime_info(symbol_data)
                self.logger.info(f"Regime info: {regime_info}")
    
    def _on_bar_update(self, data):
        """
        Handle new bar data.
        
        Args:
            data: Bar data
        """
        symbol = data.get('symbol')
        
        # Log less frequently to avoid flooding (e.g., only for 5-minute bars)
        timestamp = data.get('timestamp')
        if timestamp and timestamp.minute % 5 == 0 and timestamp.second == 0:
            self.logger.debug(f"New bar for {symbol}: {data}")
            
            # Update portfolio history once per 5 minutes
            self._update_portfolio_history()
    
    def _on_strategy_update(self, weights):
        """
        Handle strategy weight updates.
        
        Args:
            weights: New strategy weights
        """
        self.logger.info(f"Strategy weights updated: {weights}")
        
        # Execute trades based on new weights
        if self.is_trading:
            self._rebalance_portfolio(weights)
    
    def _update_portfolio_history(self):
        """Update portfolio history with current value."""
        # Get current prices
        current_prices = {}
        for symbol in self.symbols:
            # Get latest tick or bar close
            tick_data = self.data_manager.data_processor.get_latest_tick(symbol)
            if tick_data:
                current_prices[symbol] = tick_data.get('price', 100.0)  # Use 100 as default if price missing
        
        # Calculate portfolio value
        portfolio_value = self.trading_engine.get_portfolio_value(current_prices)
        
        # Add to history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': portfolio_value,
            'regime': self.data_manager.current_regime
        })
    
    def _rebalance_portfolio(self, target_weights):
        """
        Rebalance portfolio based on target weights.
        
        Args:
            target_weights: Target strategy weights
        """
        # Get current positions
        current_positions = self.trading_engine.get_positions()
        
        # Get current prices
        current_prices = {}
        for symbol in self.symbols:
            # Get latest tick or bar close
            tick_data = self.data_manager.data_processor.get_latest_tick(symbol)
            if tick_data:
                current_prices[symbol] = tick_data.get('price', 100.0)
        
        # Calculate portfolio value
        portfolio_value = self.trading_engine.get_portfolio_value(current_prices)
        
        # Calculate target position values
        strategy_signals = {}
        
        # Generate signals from each strategy
        for strategy in self.strategies:
            signals = {}
            for symbol in self.symbols:
                # Get latest bars
                bars = self.data_manager.get_latest_bars(symbol, '1day', 100)
                if not bars.empty:
                    # Calculate signal (-1 to 1)
                    if hasattr(strategy, 'calculate_signal'):
                        signal = strategy.calculate_signal(bars)
                        signals[symbol] = signal
            
            strategy_signals[strategy.name] = signals
        
        # Calculate combined signals based on strategy weights
        combined_signals = {}
        for symbol in self.symbols:
            combined_signal = 0.0
            for strategy_name, signals in strategy_signals.items():
                if symbol in signals:
                    strategy_weight = target_weights.get(strategy_name, 0.0)
                    combined_signal += signals[symbol] * strategy_weight
            
            combined_signals[symbol] = combined_signal
        
        # Calculate target positions
        target_positions = {}
        for symbol, signal in combined_signals.items():
            if symbol in current_prices:
                # Scale signal to position size (-1 to 1 -> target percentage of portfolio)
                target_percentage = signal * 0.95  # Max 95% of portfolio
                
                # Calculate target position value
                target_value = portfolio_value * target_percentage
                
                # Convert value to quantity
                price = current_prices[symbol]
                target_quantity = int(target_value / price)
                
                target_positions[symbol] = target_quantity
        
        # Execute trades to reach target positions
        for symbol, target_qty in target_positions.items():
            current_qty = current_positions.get(symbol, 0)
            
            # Calculate quantity to trade
            qty_difference = target_qty - current_qty
            
            # Execute trade if difference is significant
            if abs(qty_difference) >= 1:
                self.trading_engine.place_order(
                    symbol=symbol,
                    order_type='market',
                    quantity=qty_difference,
                    price=current_prices.get(symbol)
                )
    
    def start_trading(self):
        """Start the trading system."""
        if self.is_trading:
            self.logger.warning("Trading system is already running")
            return
        
        # Start real-time data processing
        self.data_manager.start()
        
        # Set trading flag
        self.is_trading = True
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("Trading system started")
    
    def stop_trading(self):
        """Stop the trading system."""
        if not self.is_trading:
            return
        
        # Clear trading flag
        self.is_trading = False
        
        # Stop data processing
        self.data_manager.stop()
        
        # Wait for trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5.0)
        
        self.logger.info("Trading system stopped")
    
    def _trading_loop(self):
        """Main trading loop."""
        while self.is_trading:
            try:
                # Check for rebalance opportunities every minute
                time.sleep(60)
                
                # Check if any strategies have updated weights
                current_weights = self.data_manager.get_strategy_weights()
                if current_weights:
                    self._rebalance_portfolio(current_weights)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)  # Avoid tight error loops
    
    def get_performance_summary(self):
        """
        Get trading performance summary.
        
        Returns:
            dict: Performance metrics
        """
        if not self.portfolio_history:
            return {"error": "No performance data available"}
        
        # Calculate performance metrics
        values = [entry['value'] for entry in self.portfolio_history]
        timestamps = [entry['timestamp'] for entry in self.portfolio_history]
        regimes = [entry['regime'] for entry in self.portfolio_history]
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'regime': regimes
        })
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        df['return'] = df['value'].pct_change()
        
        # Calculate metrics
        if len(df) > 1:
            total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
            volatility = df['return'].std() * np.sqrt(252 * 24 * 60 / 5)  # Annualized, assuming 5-min data
            sharpe = df['return'].mean() / df['return'].std() * np.sqrt(252 * 24 * 60 / 5) if df['return'].std() > 0 else 0
            
            # Calculate maximum drawdown
            df['cumulative_return'] = (1 + df['return']).cumprod()
            df['drawdown'] = df['cumulative_return'] / df['cumulative_return'].cummax() - 1
            max_drawdown = df['drawdown'].min()
            
            # Calculate regime performance
            regime_performance = {}
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                if len(regime_data) > 1:
                    regime_return = (regime_data['value'].iloc[-1] / regime_data['value'].iloc[0]) - 1
                    regime_performance[regime] = {
                        'return': regime_return,
                        'days': len(regime_data) * 5 / (24 * 60)  # Approximate days, assuming 5-min data
                    }
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'regime_performance': regime_performance
            }
        
        return {"error": "Insufficient data for performance calculation"}


# Example usage
if __name__ == "__main__":
    # Define symbols to trade
    symbols = ['SPY', 'QQQ', 'IWM', 'GLD']
    
    # Configuration
    config = {
        'data_source': 'alpaca',
        'alpaca_config': {
            'api_key': os.environ.get('ALPACA_API_KEY', 'YOUR_API_KEY'),
            'api_secret': os.environ.get('ALPACA_API_SECRET', 'YOUR_API_SECRET')
        },
        'initial_capital': 100000.0,
        'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
        'regime_config': {
            'model_path': 'models/market_regime_model.joblib'
        },
        'lookback_window': 60,
        'rebalance_frequency': 'daily'
    }
    
    # Create and start trading system
    trading_system = TradingSystem(symbols, config)
    
    try:
        logger.info("Starting trading system...")
        trading_system.start_trading()
        
        # Run for a specified duration or until interrupted
        duration_hours = 8
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        logger.info(f"Trading system will run until {end_time}")
        
        while datetime.now() < end_time:
            # Print performance every hour
            time.sleep(3600)
            performance = trading_system.get_performance_summary()
            logger.info(f"Current performance: {performance}")
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    except Exception as e:
        logger.error(f"Error in trading system: {str(e)}", exc_info=True)
    finally:
        # Stop trading and print summary
        logger.info("Stopping trading system...")
        trading_system.stop_trading()
        
        # Print final performance
        performance = trading_system.get_performance_summary()
        logger.info("Final performance summary:")
        pprint(performance) 