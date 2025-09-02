import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import abc
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    LONG = 1        # Buy/long signal
    SHORT = -1      # Sell/short signal
    FLAT = 0        # No position/exit signal
    SCALE_UP = 2    # Increase position size
    SCALE_DOWN = -2 # Decrease position size

class Position:
    """Represents a trading position"""
    
    def __init__(
        self, 
        symbol: str, 
        direction: SignalType, 
        size: float, 
        entry_price: float,
        entry_time: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None
    ):
        """
        Initialize a position
        
        Args:
            symbol: Traded symbol
            direction: Long or short
            size: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            trailing_stop: Trailing stop percentage (optional)
        """
        self.symbol = symbol
        self.direction = direction
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        
        # Runtime state
        self.current_price = entry_price
        self.highest_price = entry_price if direction == SignalType.LONG else float('-inf')
        self.lowest_price = entry_price if direction == SignalType.SHORT else float('inf')
        self.pnl = 0.0
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
    
    def update(self, current_price: float, current_time: pd.Timestamp) -> Tuple[bool, str]:
        """
        Update position with current price
        
        Args:
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        self.current_price = current_price
        
        # Update high/low water marks
        if self.direction == SignalType.LONG:
            self.highest_price = max(self.highest_price, current_price)
            self.lowest_price = min(self.lowest_price, current_price)
        else:
            self.highest_price = max(self.highest_price, current_price)
            self.lowest_price = min(self.lowest_price, current_price)
        
        # Calculate PnL
        if self.direction == SignalType.LONG:
            self.pnl = (current_price / self.entry_price - 1) * self.size
        else:
            self.pnl = (self.entry_price / current_price - 1) * self.size
        
        # Check for exit conditions
        should_exit = False
        exit_reason = None
        
        # Stop loss
        if self.stop_loss is not None:
            if (self.direction == SignalType.LONG and current_price <= self.stop_loss) or \
               (self.direction == SignalType.SHORT and current_price >= self.stop_loss):
                should_exit = True
                exit_reason = "stop_loss"
        
        # Take profit
        if self.take_profit is not None:
            if (self.direction == SignalType.LONG and current_price >= self.take_profit) or \
               (self.direction == SignalType.SHORT and current_price <= self.take_profit):
                should_exit = True
                exit_reason = "take_profit"
        
        # Trailing stop
        if self.trailing_stop is not None:
            if self.direction == SignalType.LONG:
                trail_price = self.highest_price * (1 - self.trailing_stop)
                if current_price <= trail_price:
                    should_exit = True
                    exit_reason = "trailing_stop"
            else:
                trail_price = self.lowest_price * (1 + self.trailing_stop)
                if current_price >= trail_price:
                    should_exit = True
                    exit_reason = "trailing_stop"
        
        # If exiting, record exit details
        if should_exit:
            self.exit_price = current_price
            self.exit_time = current_time
            self.exit_reason = exit_reason
        
        return should_exit, exit_reason
    
    def __repr__(self) -> str:
        direction_str = "LONG" if self.direction == SignalType.LONG else "SHORT"
        return f"Position({self.symbol}, {direction_str}, size={self.size}, entry={self.entry_price:.4f}, pnl={self.pnl:.2f})"


class Strategy(abc.ABC):
    """Base class for all trading strategies"""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        parameters: Dict[str, Any] = None,
        min_history_bars: int = 20
    ):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            parameters: Strategy-specific parameters
            min_history_bars: Minimum number of bars required for strategy
        """
        self.name = name
        self.symbols = symbols
        self.parameters = parameters or {}
        self.min_history_bars = min_history_bars
        
        # Runtime state
        self.positions: Dict[str, Position] = {}
        self.historical_positions: List[Position] = []
        self.equity_curve: List[float] = []
        
        logger.info(f"Initialized strategy: {name} for symbols: {symbols}")
    
    @abc.abstractmethod
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame], 
        current_time: pd.Timestamp
    ) -> Dict[str, SignalType]:
        """
        Generate trading signals for each symbol
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            
        Returns:
            Dictionary of symbol -> signal type
        """
        pass
    
    def calculate_position_size(
        self,
        symbol: str,
        signal: SignalType,
        price: float,
        volatility: float,
        account_size: float
    ) -> float:
        """
        Calculate position size based on volatility and account size
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility (e.g., ATR or standard deviation)
            account_size: Current account size
            
        Returns:
            Position size (in units or contracts)
        """
        # Default implementation: risk 1% of account per trade
        # Strategies can override this for custom sizing logic
        risk_pct = self.parameters.get("risk_per_trade", 0.01)
        
        # If volatility is available, size based on it
        if volatility > 0:
            # For simplicity: position size = (account_size * risk_pct) / volatility
            pos_size = (account_size * risk_pct) / volatility
            
            # Apply maximum position size limit if specified
            max_pos_pct = self.parameters.get("max_position_size", 0.1)
            max_pos_size = account_size * max_pos_pct / price
            pos_size = min(pos_size, max_pos_size)
            
            return pos_size
        else:
            # If volatility not available, use a fixed percentage of account
            return (account_size * risk_pct) / price
    
    def calculate_stop_loss(
        self, 
        symbol: str, 
        signal: SignalType, 
        price: float, 
        volatility: float
    ) -> Optional[float]:
        """
        Calculate stop loss price
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility (e.g., ATR or standard deviation)
            
        Returns:
            Stop loss price or None if not using stop loss
        """
        # Default implementation: stop loss at 2 * ATR
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr", 2.0)
        
        if stop_loss_atr_multiple <= 0:
            return None
        
        if signal == SignalType.LONG:
            return price * (1 - stop_loss_atr_multiple * volatility / price)
        else:
            return price * (1 + stop_loss_atr_multiple * volatility / price)
    
    def calculate_take_profit(
        self, 
        symbol: str, 
        signal: SignalType, 
        price: float, 
        volatility: float
    ) -> Optional[float]:
        """
        Calculate take profit price
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility (e.g., ATR or standard deviation)
            
        Returns:
            Take profit price or None if not using take profit
        """
        # Default implementation: take profit at 3 * ATR
        take_profit_atr_multiple = self.parameters.get("take_profit_atr", 3.0)
        
        if take_profit_atr_multiple <= 0:
            return None
        
        if signal == SignalType.LONG:
            return price * (1 + take_profit_atr_multiple * volatility / price)
        else:
            return price * (1 - take_profit_atr_multiple * volatility / price)
    
    def update(
        self, 
        data: Dict[str, pd.DataFrame], 
        current_time: pd.Timestamp, 
        account_size: float
    ) -> Dict[str, Any]:
        """
        Update strategy with new data and generate orders
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            account_size: Current account size
            
        Returns:
            Dictionary with update results including any new orders
        """
        # Check if we have enough history
        for symbol, df in data.items():
            if len(df) < self.min_history_bars:
                logger.warning(f"Not enough history for {symbol}, need {self.min_history_bars} bars but got {len(df)}")
                return {"orders": []}
        
        # Update existing positions
        for symbol, position in list(self.positions.items()):
            current_price = data[symbol].iloc[-1]['close']
            
            # Update position with current price
            should_exit, exit_reason = position.update(current_price, current_time)
            
            # Check if position should be closed
            if should_exit:
                logger.info(f"Exiting position: {position} due to {exit_reason}")
                self.historical_positions.append(position)
                del self.positions[symbol]
        
        # Generate new signals
        signals = self.generate_signals(data, current_time)
        
        # Create orders based on signals
        orders = []
        
        for symbol, signal in signals.items():
            # Skip if no signal
            if signal == SignalType.FLAT:
                continue
            
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                existing_pos = self.positions[symbol]
                
                # If signal is in the opposite direction, close existing position
                if (existing_pos.direction == SignalType.LONG and signal == SignalType.SHORT) or \
                   (existing_pos.direction == SignalType.SHORT and signal == SignalType.LONG):
                    logger.info(f"Reversing position in {symbol}: {existing_pos.direction.name} to {signal.name}")
                    
                    # Add to historical positions
                    existing_pos.exit_price = data[symbol].iloc[-1]['close']
                    existing_pos.exit_time = current_time
                    existing_pos.exit_reason = "signal_reversal"
                    self.historical_positions.append(existing_pos)
                    
                    # Remove from active positions
                    del self.positions[symbol]
                    
                    # Create new order for the opposite direction
                    current_price = data[symbol].iloc[-1]['close']
                    
                    # Calculate volatility (using close prices std as a simple measure)
                    volatility = data[symbol]['close'].pct_change().std()
                    
                    # Calculate position size
                    size = self.calculate_position_size(symbol, signal, current_price, volatility, account_size)
                    
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(symbol, signal, current_price, volatility)
                    take_profit = self.calculate_take_profit(symbol, signal, current_price, volatility)
                    
                    # Create order
                    order = {
                        "symbol": symbol,
                        "direction": signal,
                        "size": size,
                        "price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "trailing_stop": self.parameters.get("trailing_stop", None),
                        "timestamp": current_time
                    }
                    orders.append(order)
                
                # Handle scaling in/out
                elif signal in [SignalType.SCALE_UP, SignalType.SCALE_DOWN]:
                    logger.info(f"Scaling {'up' if signal == SignalType.SCALE_UP else 'down'} position in {symbol}")
                    
                    current_price = data[symbol].iloc[-1]['close']
                    volatility = data[symbol]['close'].pct_change().std()
                    
                    # Calculate adjustment size (default: ±50% of original position)
                    scale_factor = 0.5 if signal == SignalType.SCALE_UP else -0.5
                    adjustment_size = existing_pos.size * scale_factor
                    
                    # Update position size
                    existing_pos.size += adjustment_size
                    
                    # Create scaling order
                    order = {
                        "symbol": symbol,
                        "direction": existing_pos.direction,  # Same direction as existing position
                        "size": adjustment_size,
                        "price": current_price,
                        "is_adjustment": True,
                        "timestamp": current_time
                    }
                    orders.append(order)
                
            # If we don't have a position and signal is LONG or SHORT, create a new position
            elif signal in [SignalType.LONG, SignalType.SHORT]:
                current_price = data[symbol].iloc[-1]['close']
                
                # Calculate volatility (using close prices std as a simple measure)
                volatility = data[symbol]['close'].pct_change().std()
                
                # Calculate position size
                size = self.calculate_position_size(symbol, signal, current_price, volatility, account_size)
                
                # Calculate stop loss and take profit
                stop_loss = self.calculate_stop_loss(symbol, signal, current_price, volatility)
                take_profit = self.calculate_take_profit(symbol, signal, current_price, volatility)
                
                # Create new position
                position = Position(
                    symbol=symbol,
                    direction=signal,
                    size=size,
                    entry_price=current_price,
                    entry_time=current_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=self.parameters.get("trailing_stop", None)
                )
                
                # Add to active positions
                self.positions[symbol] = position
                
                # Create order
                order = {
                    "symbol": symbol,
                    "direction": signal,
                    "size": size,
                    "price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop": self.parameters.get("trailing_stop", None),
                    "timestamp": current_time
                }
                orders.append(order)
        
        # Calculate current equity
        current_equity = account_size
        for position in self.positions.values():
            current_equity += position.pnl
        
        self.equity_curve.append(current_equity)
        
        return {
            "orders": orders,
            "positions": list(self.positions.values()),
            "equity": current_equity
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate strategy performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        # Need at least 2 equity points to calculate returns
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
        
        # Calculate return metrics
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        total_return = (equity[-1] / equity[0]) - 1
        
        # Annualized Sharpe ratio (assuming daily returns)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = 1 - equity / peak
        max_drawdown = drawdown.max()
        
        # Win rate (completed positions only)
        if len(self.historical_positions) > 0:
            winning_trades = sum(1 for pos in self.historical_positions if pos.pnl > 0)
            win_rate = winning_trades / len(self.historical_positions)
        else:
            win_rate = 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": self._calculate_sortino_ratio(returns),
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": self._calculate_profit_factor(),
            "num_trades": len(self.historical_positions)
        }
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (using daily returns)"""
        if len(returns) == 0:
            return 0.0
        
        # Downside returns
        downside_returns = returns[returns < 0]
        
        # Downside deviation
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        # Sortino ratio (annualized for daily returns)
        if downside_deviation > 0:
            return returns.mean() / downside_deviation * np.sqrt(252)
        else:
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if len(self.historical_positions) == 0:
            return 0.0
        
        gross_profit = sum(pos.pnl for pos in self.historical_positions if pos.pnl > 0)
        gross_loss = sum(abs(pos.pnl) for pos in self.historical_positions if pos.pnl < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.positions = {}
        self.historical_positions = []
        self.equity_curve = [] 