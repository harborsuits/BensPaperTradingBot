"""Enhanced ChallengeBot implementation."""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple

from .trading_bot import TradingBot, Order, OrderType, OrderSide
from .strategy import Strategy, Signal, SignalType
from ..utils.logging import get_bot_logger


class ChallengeBot(TradingBot):
    """Concrete trading bot that participates in the evolution challenge.
    
    Features:
    - Wraps a trading strategy
    - Tracks performance metrics
    - Handles order execution
    - Logs detailed activity
    - Calculates fitness for evolution
    """

    def __init__(self, bot_id: str, strategy: Strategy, initial_balance: float = 1.0):
        """Initialize a challenge bot instance.
        
        Args:
            bot_id: Unique identifier for this bot
            strategy: Trading strategy to use
            initial_balance: Starting capital
        """
        super().__init__(bot_id)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.logger = get_bot_logger(bot_id)
        
        # Additional challenge-specific tracking
        self.generation: int = 0
        self.parent_ids: List[str] = []  # IDs of parent bots if this bot was created via evolution
        self.creation_timestamp = time.time()
        self.last_signal_time: Optional[float] = None
        self.total_trades: int = 0
        self.signals_generated: int = 0
        self.signals_executed: int = 0
        
        # Risk management
        self.max_drawdown_percent: float = 0.0
        self.last_balance_update = time.time()
        self.daily_returns: Dict[int, float] = {}  # day -> return
        
        # Evolution metrics
        self.fitness_score: float = 0.0
        self.survival_days: int = 0
        
    def initialize(self, initial_balance: float = None):
        """Prepare bot state before simulation.
        
        Args:
            initial_balance: Override for initial balance
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
            
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = {}
        self.open_orders = {}
        self.trades = []
        self.metrics = {
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0.0,
            "gross_profit": 0.0,  # Sum of all profitable trades
            "gross_loss": 0.0,    # Sum of all losing trades (positive value)
            "max_drawdown": 0.0,
            "max_equity": self.initial_balance,
            "daily_returns": {},
            "sharpe_ratio": 0.0,  # Default to 0 instead of None for easier calculations
        }
        
        self.logger.info(f"Bot initialized with balance: ${self.balance:.2f}")
        self.is_active = True

    def on_data(self, market_data: Dict[str, Any]) -> List[Order]:
        """Process market data and execute trading logic.
        
        Args:
            market_data: Current market data snapshot
            
        Returns:
            List of orders placed during this update
        """
        if not self.is_active:
            return []
            
        # Pre-process market data - normalize structure for strategies
        # This helps when using different data provider types (sequential vs backtest)
        processed_data = self._prepare_market_data(market_data)
        
        # Diagnostic logging - every 10 updates, log one symbol's data structure
        if hasattr(self, 'update_counter'):
            self.update_counter += 1
        else:
            self.update_counter = 0
            
        if self.update_counter % 10 == 0 and processed_data:
            # Log a sample of the market data structure to diagnose issues
            sample_symbol = list(processed_data.keys())[0]
            sample_data = processed_data[sample_symbol]
            self.logger.debug(f"[DIAGNOSTICS] Data for {sample_symbol}: Available keys: {list(sample_data.keys())}")
            
            # Check for key indicators needed for trading strategies
            for key in ['sma_10', 'sma_20', 'rsi_14', 'price']:
                self.logger.debug(f"[DIAGNOSTICS] Has {key}: {key in sample_data}, Value: {sample_data.get(key, 'N/A')}")
                
            # Log history length to check for sufficient data
            if 'history' in sample_data:
                self.logger.debug(f"[DIAGNOSTICS] History lengths: prices={len(sample_data['history'].get('prices', []))}, highs={len(sample_data['history'].get('highs', []))}")
        
        # Update equity and track daily returns if day changed
        self.update_equity(processed_data)
        current_day = int((time.time() - self.creation_timestamp) / (24 * 3600))
        if current_day not in self.daily_returns and current_day > 0:
            prev_day = current_day - 1
            if prev_day in self.daily_returns:
                day_return = (self.equity / self.metrics["daily_returns"].get(prev_day, self.initial_balance)) - 1.0
                self.daily_returns[current_day] = day_return
                self.metrics["daily_returns"][current_day] = self.equity
                self.survival_days = current_day
        
        # Generate signals from strategy
        try:
            # Log the strategy type for debugging
            self.logger.debug(f"[DIAGNOSTICS] Using strategy: {self.strategy.__class__.__name__} with parameters: {self.strategy.parameters}")
            
            signals = self.strategy.generate_signals(processed_data)
            
            # Log signals for debugging
            if signals:
                self.logger.debug(f"Generated {len(signals)} signals: {signals}")
            elif self.update_counter % 10 == 0:  # Only log periodically to avoid log spam
                self.logger.debug(f"[DIAGNOSTICS] No signals generated. Current market conditions not triggering strategy rules.")
                
            self.signals_generated += len(signals)
            self.last_signal_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            import traceback
            self.logger.error(f"[DIAGNOSTICS] Signal error details: {traceback.format_exc()}")
            signals = []
        
        # Convert signals to orders
        orders = []
        for signal in signals:
            try:
                order = self._create_order_from_signal(signal, processed_data)
                if order:
                    placed_order = self.place_order(order)
                    orders.append(placed_order)
                    self.signals_executed += 1
                    self.logger.info(f"Placed order from signal: {signal}")
                else:
                    self.logger.debug(f"[DIAGNOSTICS] Could not create order from signal: {signal}")
            except Exception as e:
                self.logger.error(f"Error processing signal: {str(e)}")
        
        # Handle immediate execution in simulation
        self._simulate_executions(processed_data)
        
        return orders
    
    def _prepare_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and normalize market data for strategy consumption.
        
        This ensures that strategies have consistent data access regardless of
        whether we're using sequential, backtest, or live data feeds.
        
        Args:
            market_data: Raw market data from data provider
            
        Returns:
            Processed market data with all necessary fields
        """
        processed_data = {}
        
        for symbol, data in market_data.items():
            processed_data[symbol] = data.copy()  # Make a copy to avoid modifying the original
            
            # Ensure history field exists
            if 'history' not in processed_data[symbol]:
                processed_data[symbol]['history'] = {}
                
            # Ensure all core history fields are present
            for field in ['prices', 'highs', 'lows', 'volumes']:
                if field not in processed_data[symbol].get('history', {}):
                    # Create empty array if field doesn't exist
                    processed_data[symbol]['history'][field] = []
            
            # Add current price to history if not already there
            if 'price' in processed_data[symbol] and 'prices' in processed_data[symbol]['history']:
                current_price = processed_data[symbol]['price']
                # Append current price to ensure the most recent data is always available
                if len(processed_data[symbol]['history']['prices']) == 0 or processed_data[symbol]['history']['prices'][-1] != current_price:
                    processed_data[symbol]['history']['prices'].append(current_price)
            
            # Same for high/low/volume if available
            if 'high' in processed_data[symbol] and 'highs' in processed_data[symbol]['history']:
                processed_data[symbol]['history']['highs'].append(processed_data[symbol]['high'])
                
            if 'low' in processed_data[symbol] and 'lows' in processed_data[symbol]['history']:
                processed_data[symbol]['history']['lows'].append(processed_data[symbol]['low'])
                
            if 'volume' in processed_data[symbol] and 'volumes' in processed_data[symbol]['history']:
                processed_data[symbol]['history']['volumes'].append(processed_data[symbol]['volume'])
                
            # Ensure required price field exists
            if 'price' not in processed_data[symbol] and 'history' in processed_data[symbol] and 'prices' in processed_data[symbol]['history'] and len(processed_data[symbol]['history']['prices']) > 0:
                processed_data[symbol]['price'] = processed_data[symbol]['history']['prices'][-1]
            
            # Calculate missing indicators if needed
            if 'history' in processed_data[symbol] and 'prices' in processed_data[symbol]['history'] and len(processed_data[symbol]['history']['prices']) >= 20:
                from ..utils.indicators import sma, ema, rsi
                
                prices = processed_data[symbol]['history']['prices']
                
                # Add basic SMAs that strategies commonly use
                if 'sma_10' not in processed_data[symbol] and len(prices) >= 10:
                    processed_data[symbol]['sma_10'] = sma(prices, 10)[-1]
                    
                if 'sma_20' not in processed_data[symbol] and len(prices) >= 20:
                    processed_data[symbol]['sma_20'] = sma(prices, 20)[-1]
                    
                if 'sma_50' not in processed_data[symbol] and len(prices) >= 50:
                    processed_data[symbol]['sma_50'] = sma(prices, 50)[-1]
                    
                # Add basic RSI that strategies commonly use
                if 'rsi_14' not in processed_data[symbol] and len(prices) >= 14:
                    processed_data[symbol]['rsi_14'] = rsi(prices, 14)[-1]
        
        return processed_data

    def _simulate_executions(self, market_data: Dict[str, Any]) -> None:
        """Simulate order executions based on current market data."""
        executed_order_ids = []
        
        for order_id, order in self.open_orders.items():
            if order.symbol not in market_data:
                continue
                
            current_price = market_data[order.symbol].get("price")
            if not current_price:
                continue
                
            # Simple market order execution
            if order.order_type == OrderType.MARKET:
                # Apply slippage
                slippage = 0.001  # 0.1% slippage
                fill_price = current_price * (1 + slippage) if order.side == OrderSide.BUY else current_price * (1 - slippage)
                
                # Apply fees
                fee_percent = 0.001  # 0.1% fee
                fee_amount = order.quantity * fill_price * fee_percent
                
                # Execute the order
                trade, pnl = self.handle_order_fill(order, fill_price, time.time(), fee_amount)
                executed_order_ids.append(order_id)
                self.total_trades += 1
                
                # Notify strategy
                self.strategy.on_order_filled({
                    "order": order,
                    "fill_price": fill_price,
                    "fee": fee_amount,
                    "pnl": pnl
                })
                
                self.logger.info(f"Executed order {order_id} at {fill_price:.6f}, PnL: {pnl:.6f}")
                
            # TODO: Implement limit, stop, etc. order types
        
        # Clean up executed orders
        for order_id in executed_order_ids:
            if order_id in self.open_orders:
                del self.open_orders[order_id]
    
    def _create_order_from_signal(self, signal: Signal, market_data: Dict[str, Any]) -> Optional[Order]:
        """Convert a strategy signal to an executable order."""
        try:
            self.logger.debug(f"[ORDER] Processing signal: {signal.signal_type} {signal.symbol} confidence={signal.confidence}")
            
            if signal.signal_type not in [SignalType.BUY, SignalType.SELL, SignalType.CLOSE]:
                self.logger.debug(f"[ORDER] Invalid signal type: {signal.signal_type}, ignoring")
                return None
                
            # Check if we have market data for this symbol
            if signal.symbol not in market_data:
                self.logger.warning(f"[ORDER] No market data for {signal.symbol}, cannot create order")
                return None
                
            if "price" not in market_data[signal.symbol]:
                self.logger.warning(f"[ORDER] No price data for {signal.symbol}, available keys: {list(market_data[signal.symbol].keys())}")
                return None
                
            current_price = market_data[signal.symbol]["price"]
            self.logger.debug(f"[ORDER] Current price for {signal.symbol}: {current_price}")
            
            # Determine order side
            if signal.signal_type == SignalType.BUY:
                side = OrderSide.BUY
            elif signal.signal_type == SignalType.SELL:
                side = OrderSide.SELL
            elif signal.signal_type == SignalType.CLOSE:
                # Close means opposite of current position
                position_size = self.get_position_size(signal.symbol)
                if position_size == 0:
                    self.logger.debug(f"[ORDER] Nothing to close for {signal.symbol}")
                    return None  # Nothing to close
                side = OrderSide.SELL if position_size > 0 else OrderSide.BUY
            else:
                return None
            
            # Determine quantity
            if signal.quantity is not None:
                quantity = signal.quantity
                self.logger.debug(f"[ORDER] Using explicit quantity from signal: {quantity}")
            elif signal.signal_type == SignalType.CLOSE:
                # Close full position
                quantity = abs(self.get_position_size(signal.symbol))
                self.logger.debug(f"[ORDER] Closing position with quantity: {quantity}")
            else:
                # Use risk percentage from signal params or strategy default
                risk_percent = signal.params.get("risk_percent", 3.0)  # More conservative default (3% instead of 10%)
                
                # Use at least a minimum amount for testing
                min_trade_value = 100.0  # Ensure we trade at least $100 worth
                
                # Calculate position size based on risk percentage of account
                max_position_value = max(min_trade_value, self.balance * (risk_percent / 100.0))
                quantity = max_position_value / current_price
                
                self.logger.debug(f"[ORDER] Calculated quantity {quantity} based on risk {risk_percent}% of balance ${self.balance:.2f}")
                
            # Basic risk management checks
            if side == OrderSide.BUY:
                # Check if we have enough balance
                cost = quantity * current_price
                if cost > self.balance:
                    self.logger.warning(f"[ORDER] Insufficient balance: need ${cost:.2f}, have ${self.balance:.2f}")
                    # Reduce quantity to match available balance (leave small buffer)
                    quantity = (self.balance * 0.95) / current_price  # Use 95% of available balance
                    self.logger.debug(f"[ORDER] Reduced quantity to {quantity} to fit available balance")
                    
                # Ensure minimum viable order size
                if quantity * current_price < 10.0:  # Minimum $10 order
                    if self.balance >= 10.0:
                        quantity = 10.0 / current_price
                        self.logger.debug(f"[ORDER] Increased to minimum viable quantity: {quantity}")
                    else:
                        self.logger.debug(f"[ORDER] Insufficient balance for minimum order size")
                        return None
        
            # Create the order
            order_type = OrderType.MARKET  # Default to market order
            price = None
            
            # Handle limit orders if price specified
            if signal.price is not None:
                order_type = OrderType.LIMIT
                price = signal.price
            
            # Ensure quantity is positive and not too small
            quantity = max(0.001, quantity)  # Ensure minimum quantity
            
            order = Order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            
            self.logger.info(f"[ORDER] Created {side} order for {quantity} {signal.symbol} at {order_type} price ${price if price else current_price:.2f}")
            return order
            
        except Exception as e:
            self.logger.error(f"[ORDER] Error creating order from signal: {str(e)}")
            import traceback
            self.logger.error(f"[ORDER] Error details: {traceback.format_exc()}")
            return None

    def finalize(self) -> Dict[str, Any]:
        """Calculate final metrics and return performance summary.
        
        Returns:
            Dict containing performance metrics
        """
        # Notify strategy of simulation end
        strategy_metrics = self.strategy.on_simulation_end()
        
        # Calculate Sharpe ratio if we have daily returns
        if len(self.daily_returns) > 0:
            import numpy as np
            returns = list(self.daily_returns.values())
            if len(returns) > 1 and np.std(returns) > 0:
                self.metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate final equity growth
        if self.initial_balance > 0:
            growth_factor = self.equity / self.initial_balance
            growth_percent = (growth_factor - 1.0) * 100.0
        else:
            growth_factor = 0
            growth_percent = 0
        
        # Calculate win rate
        total_closed_trades = self.metrics["win_count"] + self.metrics["loss_count"]
        win_rate = self.metrics["win_count"] / total_closed_trades if total_closed_trades > 0 else 0
        
        # Calculate advanced multi-factor fitness score
    
        # Base component - growth factor
        balance_score = growth_factor
        
        # Risk-adjusted returns (Sharpe Ratio)
        # Already calculated above, we can use it directly
        sharpe = self.metrics["sharpe_ratio"]
        normalized_sharpe = min(max(sharpe, 0), 3) / 3  # Normalize between 0-1, capping at 3
        
        # Drawdown penalty - more severe penalty for large drawdowns
        max_drawdown = self.metrics["max_drawdown"]
        drawdown_penalty = 1.0
        if max_drawdown > 0.4:  # Penalize for drawdowns > 40%
            drawdown_penalty = 0.5
        elif max_drawdown > 0.25:  # Minor penalty for drawdowns 25-40%
            drawdown_penalty = 0.8
        
        # Consistency bonus (reward steady growth)
        consistency = 1.0
        # Calculate profit factor if we have enough trades
        if self.metrics["win_count"] > 0 and self.metrics["loss_count"] > 0:
            profit_factor = (self.metrics["gross_profit"] / max(self.metrics["gross_loss"], 0.001))
            if profit_factor > 2.0 and win_rate > 0.5:
                consistency = 1.2
        
        # Combine all factors
        self.fitness_score = (balance_score * 0.4 + 
                              normalized_sharpe * 0.3 + 
                              drawdown_penalty * 0.2 + 
                              consistency * 0.1) * win_rate  # Win rate as multiplier
        
        # Compile final metrics
        final_metrics = {
            "bot_id": self.bot_id,
            "strategy_id": self.strategy.strategy_id,
            "strategy_type": self.strategy.__class__.__name__,
            "generation": self.generation,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "final_equity": self.equity,
            "growth_factor": growth_factor,
            "growth_percent": growth_percent,
            "total_trades": self.total_trades,
            "win_count": self.metrics["win_count"],
            "loss_count": self.metrics["loss_count"],
            "win_rate": win_rate,
            "max_drawdown": self.metrics["max_drawdown"],
            "sharpe_ratio": self.metrics["sharpe_ratio"],
            "signals_generated": self.signals_generated,
            "signals_executed": self.signals_executed,
            "fitness_score": self.fitness_score,
            **strategy_metrics  # Include strategy-specific metrics
        }
        
        self.logger.info(f"Bot finalized with equity: ${self.equity:.2f}, "  
                      f"growth: {growth_percent:.2f}%, fitness: {self.fitness_score:.4f}")
        
        return final_metrics
    
    def clone_with_mutation(self, new_bot_id: str = None, mutation_rate: float = 0.1) -> 'ChallengeBot':
        """Create a mutated clone of this bot for evolution.
        
        Args:
            new_bot_id: Optional ID for the new bot
            mutation_rate: How much to mutate parameters (0.0-1.0)
            
        Returns:
            A new bot with mutated strategy parameters
        """
        import random
        import copy
        import uuid
        
        if new_bot_id is None:
            new_bot_id = f"bot_{str(uuid.uuid4())[:8]}"
        
        # Clone the strategy with mutation
        mutated_strategy = copy.deepcopy(self.strategy)
        
        # Get parameter definitions to understand constraints
        param_defs = {p.name: p for p in self.strategy.get_parameters()}
        
        # Apply mutations to parameters
        for name, value in mutated_strategy.parameters.items():
            # Skip parameters that aren't mutable
            if name in param_defs and not param_defs[name].is_mutable:
                continue
                
            # Apply mutation based on parameter type
            if isinstance(value, (int, float)):
                # Get mutation constraints
                if name in param_defs:
                    min_val = param_defs[name].min_value
                    max_val = param_defs[name].max_value
                    mutation_factor = param_defs[name].mutation_factor
                else:
                    # Default constraints
                    min_val = value * 0.5 if value > 0 else value * 2.0
                    max_val = value * 2.0 if value > 0 else value * 0.5
                    mutation_factor = 0.2
                
                # Apply mutation
                if random.random() < mutation_rate:
                    # Determine the mutation size
                    max_change = (max_val - min_val) * mutation_factor
                    change = random.uniform(-max_change, max_change)
                    
                    # Apply the change and clamp to valid range
                    new_val = value + change
                    if min_val is not None:
                        new_val = max(min_val, new_val)
                    if max_val is not None:
                        new_val = min(max_val, new_val)
                        
                    # Round to int if the original was int
                    if isinstance(value, int):
                        new_val = int(round(new_val))
                        
                    mutated_strategy.parameters[name] = new_val
            
            elif isinstance(value, bool) and random.random() < mutation_rate:
                # Flip boolean values
                mutated_strategy.parameters[name] = not value
                
            elif isinstance(value, str) and random.random() < mutation_rate:
                # Handle string parameters (e.g., selecting from options)
                # This is a stub - would need to know valid options for each string parameter
                pass
        
        # Create new bot with the mutated strategy
        new_bot = ChallengeBot(new_bot_id, mutated_strategy, self.initial_balance)
        
        # Set up evolution tracking
        new_bot.generation = self.generation + 1
        new_bot.parent_ids = [self.bot_id]
        
        return new_bot
