"""
Automated Trading Script Based on Optimization Results

Executes trades automatically based on strategy optimizer results,
handles regime detection, portfolio allocation, and risk management.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import schedule

# Import regime detector and optimizer
from trading_bot.ml_pipeline.ml_regime_detector import MLRegimeDetector
from trading_bot.ml_pipeline.portfolio_optimizer import PortfolioOptimizer
from trading_bot.brokers.trade_executor import TradeExecutor
from trading_bot.strategies.regime_specific_strategy import RegimeSpecificStrategy
from trading_bot.data_handlers.data_loader import DataLoader
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.triggers.regime_change_notifier import RegimeChangeNotifier

# Setup logger
logger = logging.getLogger(__name__)

class AutomatedTrader:
    """
    Automated trading system based on optimization results
    
    Executes trades based on optimized strategies, adapts to market regimes,
    and manages overall portfolio risk and exposure.
    """
    
    def __init__(self, config=None):
        """
        Initialize the automated trader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Trading parameters
        self.symbols = self.config.get('symbols', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'])
        self.strategies = self.config.get('strategies', ['hybrid', 'momentum', 'mean_reversion', 'trend_following'])
        self.max_positions = self.config.get('max_positions', 10)
        self.position_size_pct = self.config.get('position_size_pct', 0.05)  # 5% per position
        self.max_risk_pct = self.config.get('max_risk_pct', 0.01)  # 1% risk per trade
        self.check_interval_minutes = self.config.get('check_interval_minutes', 15)
        self.enable_auto_trading = self.config.get('enable_auto_trading', False)
        
        # Load API keys securely from config
        self.api_keys = self._load_api_keys()
        
        # Initialize components
        self.regime_detector = MLRegimeDetector(config=self.config.get('regime_detector', {}))
        self.portfolio_optimizer = PortfolioOptimizer(config=self.config.get('portfolio_optimizer', {}))
        self.data_loader = DataLoader(config=self.config.get('data_loader', {}))
        self.trade_executor = TradeExecutor(config=self.config.get('trade_executor', {}))
        self.regime_notifier = RegimeChangeNotifier(config=self.config.get('regime_notifier', {}))
        
        # Create regime-specific strategies
        self.strategies_by_regime = self._create_regime_specific_strategies()
        
        # State tracking
        self.current_regime = None
        self.current_positions = {}
        self.pending_orders = {}
        self.trading_active = False
        self.trading_thread = None
        self.last_check_time = None
        self.last_trade_time = None
        self.strategy_signals = {}
        
        # Performance tracking
        self.daily_returns = []
        self.trade_history = []
        
        logger.info("AutomatedTrader initialized")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from config securely
        
        Returns:
            Dictionary of API keys
        """
        # Default to empty dict
        api_keys = {}
        
        try:
            # Try to load from config file
            from trading_bot.config.config_loader import ConfigLoader
            
            config_loader = ConfigLoader()
            config = config_loader.load_config()
            
            if config and 'api_keys' in config:
                api_keys = config['api_keys']
                logger.info("Loaded API keys from config")
            else:
                logger.warning("No API keys found in config")
        
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
        
        return api_keys
    
    def _create_regime_specific_strategies(self) -> Dict[str, Any]:
        """
        Create regime-specific strategy instances
        
        Returns:
            Dictionary of regime -> strategy
        """
        # Define regimes
        regimes = ['bullish_trend', 'bearish_trend', 'volatile', 'ranging', 'consolidation']
        
        # Create strategies for each regime
        strategies_by_regime = {}
        
        for regime in regimes:
            try:
                # Create RegimeSpecificStrategy for each regime
                regime_config = {
                    'strategy_type': 'hybrid',  # Default to hybrid
                    'market_regime': {
                        'current_regime': regime
                    },
                    'results_dir': 'optimization_results'
                }
                
                # Create the strategy
                strategies_by_regime[regime] = RegimeSpecificStrategy(config=regime_config)
                logger.info(f"Created regime-specific strategy for {regime}")
            
            except Exception as e:
                logger.error(f"Error creating strategy for {regime}: {e}")
        
        return strategies_by_regime
    
    def start_trading(self):
        """Start automated trading in a background thread"""
        if self.trading_active:
            logger.warning("Automated trading already active")
            return
        
        # Set active flag
        self.trading_active = True
        
        # Start the trading thread
        def trading_loop():
            logger.info("Starting automated trading loop")
            
            # Setup schedule for regular checks
            schedule.every(self.check_interval_minutes).minutes.do(self._trading_cycle)
            
            # Run initial check
            self._trading_cycle()
            
            # Main loop
            while self.trading_active:
                # Run pending tasks
                schedule.run_pending()
                
                # Sleep for 1 second
                time.sleep(1)
        
        # Create and start the thread
        self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
        self.trading_thread.start()
        
        # Start regime monitoring
        self.regime_notifier.start_monitoring(self._get_market_data)
        
        logger.info(f"Automated trading started, checking every {self.check_interval_minutes} minutes")
    
    def stop_trading(self):
        """Stop automated trading"""
        if not self.trading_active:
            logger.warning("Automated trading not active")
            return
        
        # Clear schedule
        schedule.clear()
        
        # Set flag to stop thread
        self.trading_active = False
        
        # Stop regime monitoring
        self.regime_notifier.stop_monitoring()
        
        # Wait for thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=2.0)
            self.trading_thread = None
        
        logger.info("Automated trading stopped")
    
    def _trading_cycle(self):
        """Execute a full trading cycle"""
        try:
            # Update check time
            self.last_check_time = datetime.now()
            logger.info(f"Running trading cycle at {self.last_check_time}")
            
            # 1. Get market data
            market_data = self._get_market_data(self.symbols)
            if not market_data:
                logger.warning("No market data available, skipping trading cycle")
                return
            
            # 2. Detect current market regime
            regime_info = self.regime_detector.detect_regime(market_data.get('SPY', next(iter(market_data.values()))))
            self.current_regime = regime_info['regime']
            logger.info(f"Current market regime: {self.current_regime} (confidence: {regime_info['confidence']:.4f})")
            
            # 3. Get optimal allocation for current regime
            allocation = self.portfolio_optimizer.get_allocation_for_regime(self.current_regime)
            
            # 4. Apply allocation to get strategy weights
            if 'portfolio_weights' in allocation:
                strategy_weights = allocation['portfolio_weights']
                asset_allocations = allocation.get('asset_allocations', {})
                logger.info(f"Using {self.current_regime} regime allocation: {strategy_weights}")
            else:
                # Default to equal weights
                strategy_weights = {strategy: 1.0 / len(self.strategies) for strategy in self.strategies}
                asset_allocations = {}
                logger.warning(f"No allocation for {self.current_regime} regime, using equal weights")
            
            # 5. Get current portfolio state
            portfolio = self._get_portfolio_state()
            
            # 6. Generate trading signals from all strategies
            self._generate_trading_signals(market_data, strategy_weights)
            
            # 7. Execute trades if auto-trading is enabled
            if self.enable_auto_trading:
                self._execute_trades(portfolio)
                self.last_trade_time = datetime.now()
            else:
                logger.info("Auto-trading disabled, not executing trades")
            
            # 8. Update portfolio state
            self._update_portfolio_metrics(portfolio)
            
            # Log completion
            logger.info(f"Trading cycle completed in {(datetime.now() - self.last_check_time).total_seconds():.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _get_market_data(self, symbols: List[str], days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for the symbols
        
        Args:
            symbols: List of symbols to get data for
            days: Number of days of historical data
            
        Returns:
            Dictionary of symbol -> DataFrame with market data
        """
        try:
            market_data = {}
            
            for symbol in symbols:
                # Load data
                data = self.data_loader.load_historical_data(
                    symbol=symbol,
                    timeframe='1d',
                    days=days
                )
                
                if data is not None and not data.empty:
                    market_data[symbol] = data
            
            logger.info(f"Loaded market data for {len(market_data)}/{len(symbols)} symbols")
            return market_data
        
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state from broker
        
        Returns:
            Dictionary with portfolio information
        """
        try:
            # Get account info from trade executor
            account_info = self.trade_executor.get_account_info()
            
            # Get positions
            positions = self.trade_executor.get_positions()
            
            # Calculate total value and allocation
            total_value = account_info.get('portfolio_value', 0)
            cash = account_info.get('cash', 0)
            
            # Calculate allocation percentages
            current_allocation = {}
            for position in positions:
                symbol = position.get('symbol', '')
                value = position.get('market_value', 0)
                
                if total_value > 0:
                    current_allocation[symbol] = value / total_value
            
            # Build portfolio state
            portfolio = {
                'total_value': total_value,
                'cash': cash,
                'positions': positions,
                'current_allocation': current_allocation,
                'timestamp': datetime.now()
            }
            
            # Store current positions for reference
            self.current_positions = {
                position['symbol']: position 
                for position in positions
            }
            
            logger.info(f"Portfolio: ${total_value:.2f} ({len(positions)} positions, ${cash:.2f} cash)")
            return portfolio
        
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return {
                'total_value': 0,
                'cash': 0,
                'positions': [],
                'current_allocation': {},
                'error': str(e)
            }
    
    def _generate_trading_signals(self, market_data: Dict[str, pd.DataFrame], 
                                 strategy_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate trading signals from all strategies
        
        Args:
            market_data: Market data by symbol
            strategy_weights: Strategy weight allocation
            
        Returns:
            Dictionary with combined trading signals
        """
        try:
            # Clear previous signals
            self.strategy_signals = {}
            combined_signals = {}
            
            # Get regime-specific strategy for current regime
            regime_strategy = self.strategies_by_regime.get(
                self.current_regime, 
                self.strategies_by_regime.get('ranging')  # Default to ranging as safest
            )
            
            # Generate signals for each symbol
            for symbol, data in market_data.items():
                symbol_signals = {}
                
                # First try regime-specific strategy
                if regime_strategy:
                    regime_signal = regime_strategy.generate_signals(data)
                    if regime_signal:
                        signal_value = regime_signal.get('signal', 0)
                        confidence = regime_signal.get('confidence', 0.5)
                        
                        # Store with high weight (regime-specific strategy is trusted more)
                        symbol_signals['regime_specific'] = {
                            'signal': signal_value,
                            'confidence': confidence,
                            'weight': 0.6  # Give higher weight to regime-specific strategy
                        }
                
                # Then add signals from individual strategies
                total_weight = 0.4  # Remaining weight for individual strategies
                
                for strategy_name, weight in strategy_weights.items():
                    # Skip if weight is zero
                    if weight <= 0:
                        continue
                    
                    # Create strategy instance
                    try:
                        strategy = StrategyFactory.create_strategy(strategy_name)
                        
                        # Generate signal
                        signal_result = strategy.generate_signals(data)
                        
                        if signal_result:
                            signal_value = signal_result.get('signal', 0)
                            confidence = signal_result.get('confidence', 0.5)
                            
                            # Store signal with weight
                            symbol_signals[strategy_name] = {
                                'signal': signal_value,
                                'confidence': confidence,
                                'weight': weight * total_weight
                            }
                    
                    except Exception as e:
                        logger.error(f"Error generating {strategy_name} signal for {symbol}: {e}")
                
                # Combine signals into a single recommendation
                if symbol_signals:
                    weighted_sum = 0
                    total_weights = 0
                    
                    for strategy_name, signal_info in symbol_signals.items():
                        weighted_sum += signal_info['signal'] * signal_info['weight'] * signal_info['confidence']
                        total_weights += signal_info['weight'] * signal_info['confidence']
                    
                    # Calculate combined signal
                    if total_weights > 0:
                        combined_signal = weighted_sum / total_weights
                    else:
                        combined_signal = 0
                    
                    # Store in combined signals
                    combined_signals[symbol] = {
                        'signal': combined_signal,
                        'components': symbol_signals,
                        'timestamp': datetime.now()
                    }
            
            # Store for reference
            self.strategy_signals = combined_signals
            
            logger.info(f"Generated signals for {len(combined_signals)} symbols")
            return combined_signals
        
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {}
    
    def _execute_trades(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trades based on signals and portfolio state
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dictionary with trade execution results
        """
        try:
            # Check if we have signals and portfolio info
            if not self.strategy_signals or not portfolio:
                logger.warning("No signals or portfolio info available")
                return {
                    'success': False,
                    'error': 'No signals or portfolio info available'
                }
            
            # Get portfolio value and cash
            portfolio_value = portfolio.get('total_value', 0)
            cash = portfolio.get('cash', 0)
            
            if portfolio_value <= 0:
                logger.warning("Invalid portfolio value")
                return {
                    'success': False,
                    'error': 'Invalid portfolio value'
                }
            
            # Prepare trade orders
            buy_orders = []
            sell_orders = []
            
            # Process each signal
            for symbol, signal_info in self.strategy_signals.items():
                signal = signal_info['signal']
                
                # Current position in this symbol
                position = self.current_positions.get(symbol, None)
                current_shares = position['quantity'] if position else 0
                
                # Calculate target position
                # Signal is between -1 and 1, representing -100% to +100% of position size
                target_allocation = max(0, signal) * self.position_size_pct
                target_value = portfolio_value * target_allocation
                
                # Get current price
                current_price = None
                if symbol in self.current_positions:
                    current_price = self.current_positions[symbol].get('current_price')
                
                if current_price is None or current_price <= 0:
                    # Get price from recent data
                    for sym, data in self._get_market_data([symbol], days=1).items():
                        if not data.empty:
                            current_price = data['close'].iloc[-1]
                            break
                
                if current_price is None or current_price <= 0:
                    logger.warning(f"Could not get price for {symbol}")
                    continue
                
                # Calculate target shares
                target_shares = int(target_value / current_price) if current_price > 0 else 0
                
                # Adjust for risk based on regime
                if self.current_regime == 'volatile':
                    # Reduce position size in volatile regimes
                    target_shares = int(target_shares * 0.7)
                elif self.current_regime == 'bearish_trend':
                    # Reduce position size in bearish regimes
                    target_shares = int(target_shares * 0.8)
                
                # Calculate shares to buy or sell
                shares_delta = target_shares - current_shares
                
                if shares_delta > 0:
                    # Buy order
                    buy_orders.append({
                        'symbol': symbol,
                        'quantity': shares_delta,
                        'order_type': 'market',
                        'signal': signal,
                        'regime': self.current_regime
                    })
                elif shares_delta < 0:
                    # Sell order
                    sell_orders.append({
                        'symbol': symbol,
                        'quantity': abs(shares_delta),
                        'order_type': 'market',
                        'signal': signal,
                        'regime': self.current_regime
                    })
            
            # Execute sell orders first to free up cash
            sell_results = []
            for order in sell_orders:
                try:
                    result = self.trade_executor.place_order(
                        symbol=order['symbol'],
                        order_type='sell',
                        quantity=order['quantity'],
                        order_subtype=order['order_type']
                    )
                    
                    order['result'] = result
                    sell_results.append(order)
                    
                    logger.info(f"Sold {order['quantity']} shares of {order['symbol']}")
                
                except Exception as e:
                    logger.error(f"Error selling {order['symbol']}: {e}")
                    order['error'] = str(e)
                    sell_results.append(order)
            
            # Execute buy orders with priority based on signal strength
            buy_orders.sort(key=lambda x: x['signal'], reverse=True)
            buy_results = []
            
            # Update cash after sells
            portfolio = self._get_portfolio_state()
            cash = portfolio.get('cash', 0)
            
            for order in buy_orders:
                try:
                    # Check if we have enough cash
                    if cash <= 0:
                        logger.warning("Out of cash, skipping remaining buy orders")
                        break
                    
                    # Calculate order value
                    symbol_price = None
                    for sym, data in self._get_market_data([order['symbol']], days=1).items():
                        if not data.empty:
                            symbol_price = data['close'].iloc[-1]
                            break
                    
                    if symbol_price is None:
                        logger.warning(f"Could not get price for {order['symbol']}")
                        continue
                    
                    order_value = order['quantity'] * symbol_price
                    
                    # Adjust quantity if not enough cash
                    if order_value > cash:
                        adjusted_quantity = int(cash / symbol_price) if symbol_price > 0 else 0
                        if adjusted_quantity <= 0:
                            logger.warning(f"Not enough cash for {order['symbol']}")
                            continue
                        
                        order['quantity'] = adjusted_quantity
                        order_value = adjusted_quantity * symbol_price
                    
                    # Place order
                    result = self.trade_executor.place_order(
                        symbol=order['symbol'],
                        order_type='buy',
                        quantity=order['quantity'],
                        order_subtype=order['order_type']
                    )
                    
                    order['result'] = result
                    buy_results.append(order)
                    
                    # Update cash
                    cash -= order_value
                    
                    logger.info(f"Bought {order['quantity']} shares of {order['symbol']}")
                
                except Exception as e:
                    logger.error(f"Error buying {order['symbol']}: {e}")
                    order['error'] = str(e)
                    buy_results.append(order)
            
            # Store results
            execution_results = {
                'success': True,
                'sell_orders': sell_results,
                'buy_orders': buy_results,
                'timestamp': datetime.now()
            }
            
            # Save to trade history
            self.trade_history.append(execution_results)
            
            # Log summary
            logger.info(f"Executed {len(sell_results)} sells and {len(buy_results)} buys")
            
            return execution_results
        
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_portfolio_metrics(self, portfolio: Dict[str, Any]):
        """
        Update portfolio performance metrics
        
        Args:
            portfolio: Current portfolio state
        """
        try:
            # Calculate daily return
            if not self.daily_returns:
                # First day, no return
                daily_return = 0
            else:
                # Get previous value
                prev_value = self.daily_returns[-1]['portfolio_value']
                curr_value = portfolio['total_value']
                
                # Calculate return
                if prev_value > 0:
                    daily_return = (curr_value / prev_value) - 1
                else:
                    daily_return = 0
            
            # Store daily return
            self.daily_returns.append({
                'date': datetime.now().date(),
                'portfolio_value': portfolio['total_value'],
                'return': daily_return,
                'regime': self.current_regime
            })
            
            # Trim history to last 90 days
            if len(self.daily_returns) > 90:
                self.daily_returns = self.daily_returns[-90:]
            
            # Calculate metrics
            if len(self.daily_returns) >= 5:
                returns = [day['return'] for day in self.daily_returns]
                
                # Calculate metrics
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                drawdown = min(0, (portfolio['total_value'] / max(day['portfolio_value'] for day in self.daily_returns)) - 1)
                
                logger.info(f"Portfolio metrics - Sharpe: {sharpe:.4f}, Drawdown: {drawdown:.4%}")
        
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current state of the automated trader
        
        Returns:
            Dictionary with current state
        """
        # Get portfolio
        portfolio = self._get_portfolio_state()
        
        # Build state report
        state = {
            'trading_active': self.trading_active,
            'current_regime': self.current_regime,
            'last_check_time': self.last_check_time,
            'last_trade_time': self.last_trade_time,
            'portfolio': portfolio,
            'signals': self.strategy_signals,
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'daily_returns': self.daily_returns[-30:] if self.daily_returns else []
        }
        
        return state
    
    def update_configuration(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update trader configuration
        
        Args:
            new_config: New configuration values
            
        Returns:
            Dict with update results
        """
        try:
            # Update fields that exist in config
            updated_fields = []
            
            for key, value in new_config.items():
                if key in self.config:
                    old_value = self.config[key]
                    self.config[key] = value
                    updated_fields.append({
                        'field': key,
                        'old_value': old_value,
                        'new_value': value
                    })
            
            # Special handling for some fields
            if 'enable_auto_trading' in new_config:
                self.enable_auto_trading = new_config['enable_auto_trading']
                logger.info(f"Auto-trading {'enabled' if self.enable_auto_trading else 'disabled'}")
            
            if 'check_interval_minutes' in new_config:
                # Update schedule if running
                if self.trading_active:
                    # Stop and restart with new interval
                    self.stop_trading()
                    self.check_interval_minutes = new_config['check_interval_minutes']
                    self.start_trading()
                else:
                    self.check_interval_minutes = new_config['check_interval_minutes']
            
            if 'symbols' in new_config:
                self.symbols = new_config['symbols']
            
            if 'strategies' in new_config:
                self.strategies = new_config['strategies']
            
            logger.info(f"Updated {len(updated_fields)} configuration fields")
            
            return {
                'success': True,
                'updated_fields': updated_fields
            }
        
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return {
                'success': False,
                'error': str(e)
            }
