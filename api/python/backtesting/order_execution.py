import pandas as pd
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Supported order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderDirection(Enum):
    """Order direction"""
    BUY = "buy"
    SELL = "sell"

class OrderTimeInForce(Enum):
    """Time in force options"""
    DAY = "day"                # Valid for the day
    GTC = "gtc"                # Good till canceled
    IOC = "ioc"                # Immediate or cancel 
    FOK = "fok"                # Fill or kill

class DataFrequency(Enum):
    """Supported data frequencies for the simulator"""
    TICK = "tick"              # Tick-by-tick data
    SECOND = "second"          # Second-level data
    MINUTE = "minute"          # Minute-level data
    HOUR = "hour"              # Hour-level data
    DAY = "day"                # Daily OHLCV bars
    
@dataclass
class Order:
    """Order object for backtesting"""
    id: str
    strategy: str
    direction: OrderDirection
    quantity: float
    price: Optional[float] = None     # Limit price (if applicable)
    stop_price: Optional[float] = None  # Stop price (if applicable)
    order_type: OrderType = OrderType.MARKET
    time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    submitted_at: pd.Timestamp = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    fill_timestamps: List[pd.Timestamp] = None
    partial_fills: List[Dict[str, Any]] = None
    commission: float = 0.0
    slippage: float = 0.0
    expiration: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.fill_timestamps is None:
            self.fill_timestamps = []
        if self.partial_fills is None:
            self.partial_fills = []
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to be filled"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or canceled)"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def average_fill_price(self) -> float:
        """Calculate average fill price for the order"""
        if not self.partial_fills:
            return self.filled_price or 0.0
        
        total_value = sum(fill['price'] * fill['quantity'] for fill in self.partial_fills)
        total_quantity = sum(fill['quantity'] for fill in self.partial_fills)
        
        return total_value / total_quantity if total_quantity > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary format"""
        return {
            'id': self.id,
            'strategy': self.strategy,
            'direction': self.direction.value,
            'order_type': self.order_type.value,
            'time_in_force': self.time_in_force.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'submitted_at': self.submitted_at,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price if self.filled_price is not None else self.average_fill_price,
            'commission': self.commission,
            'slippage': self.slippage,
            'partial_fills': self.partial_fills
        }

class SlippageModel:
    """Base class for slippage models"""
    
    def calculate_slippage(self, order: Order, bar_data: pd.Series, market_data: pd.DataFrame = None) -> float:
        """
        Calculate slippage for an order
        
        Args:
            order: The order being executed
            bar_data: OHLCV data for the current bar
            market_data: Additional market data (if available)
            
        Returns:
            Slippage amount in percentage points
        """
        raise NotImplementedError("Subclasses must implement this method")

class FixedSlippageModel(SlippageModel):
    """Fixed slippage model (constant slippage regardless of order size)"""
    
    def __init__(self, slippage_pct: float = 0.05):
        """
        Initialize with fixed slippage percentage
        
        Args:
            slippage_pct: Slippage as percentage points (e.g., 0.05 = 5 basis points)
        """
        self.slippage_pct = slippage_pct
        
    def calculate_slippage(self, order: Order, bar_data: pd.Series, market_data: pd.DataFrame = None) -> float:
        """Apply fixed slippage regardless of order size"""
        # Direction-aware slippage (positive for buys, negative for sells)
        direction_multiplier = 1 if order.direction == OrderDirection.BUY else -1
        return self.slippage_pct * direction_multiplier
    
class VolumeBasedSlippageModel(SlippageModel):
    """Volume-based slippage model (slippage increases with order size relative to volume)"""
    
    def __init__(self, base_slippage_pct: float = 0.05, volume_impact_factor: float = 0.1):
        """
        Initialize with base slippage and volume impact factor
        
        Args:
            base_slippage_pct: Base slippage percentage (minimum slippage)
            volume_impact_factor: Factor to scale impact based on order size relative to volume
        """
        self.base_slippage_pct = base_slippage_pct
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(self, order: Order, bar_data: pd.Series, market_data: pd.DataFrame = None) -> float:
        """Calculate slippage based on order volume relative to bar volume"""
        # Get volume from bar data
        volume = bar_data.get('volume', 0)
        if volume <= 0:
            # If no volume data, use fixed slippage
            direction_multiplier = 1 if order.direction == OrderDirection.BUY else -1
            return self.base_slippage_pct * direction_multiplier
        
        # Calculate order's impact on volume
        volume_ratio = order.quantity / volume
        
        # Calculate slippage based on volume impact
        slippage = self.base_slippage_pct + (volume_ratio * self.volume_impact_factor)
        
        # Direction-aware slippage (positive for buys, negative for sells)
        direction_multiplier = 1 if order.direction == OrderDirection.BUY else -1
        return slippage * direction_multiplier

class VolatilityBasedSlippageModel(SlippageModel):
    """Volatility-based slippage model (slippage increases with market volatility)"""
    
    def __init__(self, base_slippage_pct: float = 0.05, volatility_window: int = 20,
                volatility_impact_factor: float = 2.0, volume_impact_factor: float = 0.1):
        """
        Initialize with parameters for volatility-based slippage
        
        Args:
            base_slippage_pct: Base slippage percentage
            volatility_window: Window for calculating recent volatility
            volatility_impact_factor: Factor to scale impact based on volatility
            volume_impact_factor: Factor to scale impact based on order size
        """
        self.base_slippage_pct = base_slippage_pct
        self.volatility_window = volatility_window
        self.volatility_impact_factor = volatility_impact_factor
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(self, order: Order, bar_data: pd.Series, market_data: pd.DataFrame = None) -> float:
        """Calculate slippage based on market volatility and order size"""
        # Starting with base slippage
        slippage = self.base_slippage_pct
        
        # Add volume-based component if volume data available
        volume = bar_data.get('volume', 0)
        if volume > 0:
            volume_ratio = order.quantity / volume
            slippage += volume_ratio * self.volume_impact_factor
        
        # Add volatility component if market data available
        if market_data is not None and len(market_data) >= self.volatility_window:
            # Calculate recent volatility (standard deviation of returns)
            recent_data = market_data.tail(self.volatility_window)
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Normalize volatility impact (expected range 0-3)
            normalized_volatility = min(3.0, volatility * 100)
            
            # Add volatility component to slippage
            slippage += (normalized_volatility / 100) * self.volatility_impact_factor
        
        # Direction-aware slippage (positive for buys, negative for sells)
        direction_multiplier = 1 if order.direction == OrderDirection.BUY else -1
        return slippage * direction_multiplier

class CommissionModel:
    """Base class for commission models"""
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """
        Calculate commission for an order or partial fill
        
        Args:
            order: The order being executed
            fill_price: Execution price for this fill
            fill_quantity: Quantity for this fill
            
        Returns:
            Commission amount in currency units
        """
        raise NotImplementedError("Subclasses must implement this method")

class PercentageCommissionModel(CommissionModel):
    """Commission as a percentage of trade value"""
    
    def __init__(self, commission_pct: float = 0.1, min_commission: float = 0.0):
        """
        Initialize with commission percentage
        
        Args:
            commission_pct: Commission as percentage points (e.g., 0.1 = 10 basis points)
            min_commission: Minimum commission per order
        """
        self.commission_pct = commission_pct / 100.0  # Convert to decimal
        self.min_commission = min_commission
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """Calculate commission as percentage of trade value"""
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_pct
        return max(commission, self.min_commission)

class FixedCommissionModel(CommissionModel):
    """Fixed commission per trade"""
    
    def __init__(self, commission_per_trade: float = 5.0):
        """
        Initialize with fixed commission
        
        Args:
            commission_per_trade: Fixed commission per trade (regardless of size)
        """
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """Return fixed commission per trade"""
        # For partial fills, we need to determine if this is the first fill
        # Only charge commission on first fill
        if order.filled_quantity == 0:
            return self.commission_per_trade
        return 0.0

class TieredCommissionModel(CommissionModel):
    """Tiered commission model based on trade value"""
    
    def __init__(self, tiers: List[Tuple[float, float]], min_commission: float = 0.0):
        """
        Initialize with commission tiers
        
        Args:
            tiers: List of (trade_value_threshold, commission_pct) tuples
            min_commission: Minimum commission per order
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.min_commission = min_commission
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """Calculate commission based on trade value tiers"""
        trade_value = fill_price * fill_quantity
        
        # Find applicable tier
        commission_pct = self.tiers[-1][1]  # Default to highest tier
        for threshold, pct in self.tiers:
            if trade_value <= threshold:
                commission_pct = pct
                break
        
        commission = trade_value * (commission_pct / 100.0)
        return max(commission, self.min_commission)

class OrderExecutionSimulator:
    """
    Advanced order execution simulator for backtesting
    
    This class simulates realistic order execution including:
    - Different order types (market, limit, stop, stop-limit)
    - Slippage models based on volume, volatility, etc.
    - Partial fills based on available liquidity
    - Time-in-force handling and order expiration
    """
    
    def __init__(
        self,
        data_frequency: DataFrequency = DataFrequency.DAY,
        slippage_model: SlippageModel = None,
        commission_model: CommissionModel = None,
        market_impact_model: Any = None,
        min_fill_ratio: float = 0.0,  # Minimum ratio to fill in a single bar (0=disable partial fills)
        liquidity_factor: float = 1.0,  # Scale factor for available liquidity
        realistic_market_hours: bool = True,  # Respect market hours
        fill_probability: float = 1.0,  # Probability of fill for limit orders when price is touched
        enable_random_behavior: bool = False,  # Add random realistic behaviors
        random_seed: int = None,  # Seed for reproducibility
        debug_mode: bool = False  # Enable detailed debugging
    ):
        """
        Initialize the order execution simulator
        
        Args:
            data_frequency: Frequency of market data (affects fill behavior)
            slippage_model: Model for calculating price slippage
            commission_model: Model for calculating commissions
            market_impact_model: Model for calculating market impact
            min_fill_ratio: Minimum ratio to fill in a single bar (0=no partial fills)
            liquidity_factor: Scale factor for available liquidity
            realistic_market_hours: Whether to respect market hours
            fill_probability: Probability of fill for limit orders at touch price
            enable_random_behavior: Add random realistic behaviors
            random_seed: Seed for reproducibility
            debug_mode: Enable detailed debugging
        """
        self.data_frequency = data_frequency
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.commission_model = commission_model or PercentageCommissionModel()
        self.market_impact_model = market_impact_model
        self.min_fill_ratio = min_fill_ratio
        self.liquidity_factor = liquidity_factor
        self.realistic_market_hours = realistic_market_hours
        self.fill_probability = fill_probability
        self.enable_random_behavior = enable_random_behavior
        self.debug_mode = debug_mode
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Order storage
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_id_counter = 0
        
        # Trade history
        self.trade_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized OrderExecutionSimulator with {data_frequency.value} data frequency")
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID"""
        self.order_id_counter += 1
        return f"order_{self.order_id_counter}"
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order to the execution simulator
        
        Args:
            order: The order to submit
            
        Returns:
            Order ID
        """
        if order.id is None:
            order.id = self._generate_order_id()
        
        if order.submitted_at is None:
            order.submitted_at = pd.Timestamp.now()
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.completed_orders[order.id] = order
            logger.warning(f"Order {order.id} rejected: Invalid order parameters")
            return order.id
        
        # Store the order as active
        self.active_orders[order.id] = order
        logger.debug(f"Order {order.id} submitted: {order.order_type.value} {order.direction.value} "
                    f"{order.quantity} @ {order.price}")
        
        return order.id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if order was canceled, False otherwise
        """
        if order_id not in self.active_orders:
            logger.warning(f"Attempt to cancel non-existent order {order_id}")
            return False
        
        order = self.active_orders.pop(order_id)
        
        # If order was partially filled, mark as canceled
        if order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.CANCELED
        
        # Move to completed orders
        self.completed_orders[order_id] = order
        logger.debug(f"Order {order_id} canceled")
        
        return True
    
    def _validate_order(self, order: Order) -> bool:
        """
        Validate order parameters
        
        Args:
            order: Order to validate
            
        Returns:
            True if order is valid, False otherwise
        """
        # Check for required fields based on order type
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.warning(f"Limit order requires a price: {order.id}")
            return False
        
        if order.order_type == OrderType.STOP and order.stop_price is None:
            logger.warning(f"Stop order requires a stop price: {order.id}")
            return False
        
        if order.order_type == OrderType.STOP_LIMIT and (order.price is None or order.stop_price is None):
            logger.warning(f"Stop-limit order requires both price and stop price: {order.id}")
            return False
        
        # Check for valid quantity
        if order.quantity <= 0:
            logger.warning(f"Order quantity must be positive: {order.id}")
            return False
        
        return True
    
    def process_bar(self, bar_data: Dict[str, pd.Series], timestamp: pd.Timestamp, market_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Process a single bar of market data and update order status
        
        Args:
            bar_data: Dictionary of strategy -> OHLCV data for the current bar
            timestamp: Timestamp for this bar
            market_data: Historical market data for context (optional)
            
        Returns:
            List of trades executed in this bar
        """
        if not self.active_orders:
            return []
        
        executed_trades = []
        orders_to_remove = []
        
        for order_id, order in self.active_orders.items():
            # Skip if strategy data not available
            if order.strategy not in bar_data:
                continue
            
            # Get bar data for this strategy
            strategy_bar = bar_data[order.strategy]
            
            # Check if order can be executed in this bar
            trades = self._process_order_for_bar(order, strategy_bar, timestamp, market_data)
            
            if trades:
                executed_trades.extend(trades)
                
                # Check if order is fully filled
                if order.status == OrderStatus.FILLED:
                    orders_to_remove.append(order_id)
            
            # Check for expired orders
            if (order.time_in_force == OrderTimeInForce.DAY and 
                timestamp.date() > order.submitted_at.date()):
                order.status = OrderStatus.EXPIRED
                orders_to_remove.append(order_id)
        
        # Remove completed orders from active list
        for order_id in orders_to_remove:
            order = self.active_orders.pop(order_id)
            self.completed_orders[order_id] = order
        
        # Record all trades
        self.trade_history.extend(executed_trades)
        
        return executed_trades
    
    def _process_order_for_bar(self, order: Order, bar_data: pd.Series, timestamp: pd.Timestamp, market_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Process a single order for the current bar
        
        Args:
            order: Order to process
            bar_data: OHLCV data for the current bar
            timestamp: Current timestamp
            market_data: Historical market data for context
            
        Returns:
            List of executed trades (if any)
        """
        trades = []
        
        # Handle different order types
        if order.order_type == OrderType.MARKET:
            trades = self._process_market_order(order, bar_data, timestamp, market_data)
        
        elif order.order_type == OrderType.LIMIT:
            trades = self._process_limit_order(order, bar_data, timestamp, market_data)
        
        elif order.order_type == OrderType.STOP:
            trades = self._process_stop_order(order, bar_data, timestamp, market_data)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            trades = self._process_stop_limit_order(order, bar_data, timestamp, market_data)
        
        return trades
    
    def _process_market_order(self, order: Order, bar_data: pd.Series, timestamp: pd.Timestamp, market_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Process a market order"""
        # Market orders execute immediately at the available price
        trades = []
        
        # Determine fill price based on data frequency
        if self.data_frequency == DataFrequency.DAY:
            # For daily bars, use average of open-high-low-close as approximation
            base_price = (bar_data['open'] + bar_data['high'] + bar_data['low'] + bar_data['close']) / 4
        else:
            # For intraday/tick data, use appropriate price
            base_price = bar_data.get('price', bar_data.get('close', 0))
        
        # Calculate available liquidity for this bar
        available_liquidity = self._estimate_available_liquidity(order, bar_data)
        
        # Determine quantity to fill in this bar
        fill_quantity = min(order.remaining_quantity, available_liquidity)
        
        if fill_quantity <= 0:
            return trades
        
        # Calculate slippage
        slippage_pct = self.slippage_model.calculate_slippage(order, bar_data, market_data)
        
        # Apply slippage to price
        fill_price = base_price * (1 + slippage_pct/100)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, fill_price, fill_quantity)
        
        # Record the fill
        fill_data = {
            'order_id': order.id,
            'strategy': order.strategy,
            'direction': order.direction.value,
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': timestamp,
            'slippage_pct': slippage_pct,
            'commission': commission
        }
        
        # Update order status
        self._update_order_with_fill(order, fill_quantity, fill_price, timestamp, commission, slippage_pct)
        
        trades.append(fill_data)
        logger.debug(f"Market order {order.id} filled: {fill_quantity} @ {fill_price} ({slippage_pct:.4f}% slippage)")
        
        return trades
    
    def _process_limit_order(self, order: Order, bar_data: pd.Series, timestamp: pd.Timestamp, market_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Process a limit order"""
        trades = []
        
        # Check if the limit price was reached
        if order.direction == OrderDirection.BUY and bar_data['low'] <= order.price:
            # Buy limit triggered if price drops to or below limit price
            self._handle_limit_order_execution(order, bar_data, timestamp, market_data, trades, is_buy=True)
        
        elif order.direction == OrderDirection.SELL and bar_data['high'] >= order.price:
            # Sell limit triggered if price rises to or above limit price
            self._handle_limit_order_execution(order, bar_data, timestamp, market_data, trades, is_buy=False)
        
        return trades
    
    def _process_stop_order(self, order: Order, bar_data: pd.Series, timestamp: pd.Timestamp, market_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Process a stop order"""
        trades = []
        
        # Check if stop price was reached
        if order.direction == OrderDirection.BUY and bar_data['high'] >= order.stop_price:
            # Buy stop triggered if price rises to or above stop price
            # Convert to market order once triggered
            logger.debug(f"Buy stop order {order.id} triggered at {order.stop_price}")
            
            # Use a temporary market order for execution
            market_order = Order(
                id=f"{order.id}_market",
                strategy=order.strategy,
                direction=order.direction,
                quantity=order.remaining_quantity,
                order_type=OrderType.MARKET,
                submitted_at=timestamp
            )
            
            # Execute as market order
            market_trades = self._process_market_order(market_order, bar_data, timestamp, market_data)
            
            # Update the original order with fill information
            if market_trades:
                for trade in market_trades:
                    self._update_order_with_fill(
                        order, 
                        trade['quantity'], 
                        trade['price'], 
                        timestamp,
                        trade['commission'],
                        trade['slippage_pct']
                    )
                    trade['order_id'] = order.id  # Use original order ID
                    trades.append(trade)
        
        elif order.direction == OrderDirection.SELL and bar_data['low'] <= order.stop_price:
            # Sell stop triggered if price drops to or below stop price
            logger.debug(f"Sell stop order {order.id} triggered at {order.stop_price}")
            
            # Use a temporary market order for execution
            market_order = Order(
                id=f"{order.id}_market",
                strategy=order.strategy,
                direction=order.direction,
                quantity=order.remaining_quantity,
                order_type=OrderType.MARKET,
                submitted_at=timestamp
            )
            
            # Execute as market order
            market_trades = self._process_market_order(market_order, bar_data, timestamp, market_data)
            
            # Update the original order with fill information
            if market_trades:
                for trade in market_trades:
                    self._update_order_with_fill(
                        order, 
                        trade['quantity'], 
                        trade['price'], 
                        timestamp,
                        trade['commission'],
                        trade['slippage_pct']
                    )
                    trade['order_id'] = order.id  # Use original order ID
                    trades.append(trade)
        
        return trades
    
    def _process_stop_limit_order(self, order: Order, bar_data: pd.Series, timestamp: pd.Timestamp, market_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Process a stop-limit order"""
        trades = []
        
        # First check if stop price is reached (to activate the limit order)
        if order.direction == OrderDirection.BUY and bar_data['high'] >= order.stop_price:
            # Buy stop-limit triggered if price rises to or above stop price
            logger.debug(f"Buy stop-limit order {order.id} stop price triggered at {order.stop_price}")
            
            # Now check if limit price is reached
            if bar_data['low'] <= order.price:
                # Limit price reached, execute order
                self._handle_limit_order_execution(order, bar_data, timestamp, market_data, trades, is_buy=True)
        
        elif order.direction == OrderDirection.SELL and bar_data['low'] <= order.stop_price:
            # Sell stop-limit triggered if price drops to or below stop price
            logger.debug(f"Sell stop-limit order {order.id} stop price triggered at {order.stop_price}")
            
            # Now check if limit price is reached
            if bar_data['high'] >= order.price:
                # Limit price reached, execute order
                self._handle_limit_order_execution(order, bar_data, timestamp, market_data, trades, is_buy=False)
        
        return trades
    
    def _handle_limit_order_execution(self, order: Order, bar_data: pd.Series, timestamp: pd.Timestamp, market_data: pd.DataFrame, trades: List[Dict[str, Any]], is_buy: bool) -> None:
        """Helper to handle limit order execution"""
        # Determine fill price (limit price with possible improvement)
        if is_buy:
            # For buys, the price might be better (lower) than limit price
            # Use average of limit price and low price as an approximation
            fill_price = min(order.price, (order.price + bar_data['low']) / 2)
        else:
            # For sells, the price might be better (higher) than limit price
            # Use average of limit price and high price as an approximation
            fill_price = max(order.price, (order.price + bar_data['high']) / 2)
        
        # Apply fill probability for limit orders
        if self.enable_random_behavior and self.fill_probability < 1.0:
            if np.random.random() > self.fill_probability:
                # Skip fill this time
                logger.debug(f"Limit order {order.id} price touched but not filled due to probability")
                return
        
        # Calculate available liquidity
        available_liquidity = self._estimate_available_liquidity(order, bar_data)
        
        # Determine quantity to fill
        fill_quantity = min(order.remaining_quantity, available_liquidity)
        
        if fill_quantity <= 0:
            return
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, fill_price, fill_quantity)
        
        # Calculate slippage (usually minimal for limit orders)
        slippage_pct = 0.0
        
        # Record the fill
        fill_data = {
            'order_id': order.id,
            'strategy': order.strategy,
            'direction': order.direction.value,
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': timestamp,
            'slippage_pct': slippage_pct,
            'commission': commission
        }
        
        # Update order status
        self._update_order_with_fill(order, fill_quantity, fill_price, timestamp, commission, slippage_pct)
        
        trades.append(fill_data)
        logger.debug(f"Limit order {order.id} filled: {fill_quantity} @ {fill_price}")
    
    def _update_order_with_fill(self, order: Order, fill_quantity: float, fill_price: float, timestamp: pd.Timestamp, commission: float, slippage_pct: float) -> None:
        """Update order with fill information"""
        # Record the partial fill
        order.partial_fills.append({
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': timestamp,
            'commission': commission,
            'slippage_pct': slippage_pct
        })
        
        # Update filled quantity
        order.filled_quantity += fill_quantity
        
        # Update order status
        if abs(order.filled_quantity - order.quantity) < 1e-10:  # Floating point comparison
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Update commission and slippage
        order.commission += commission
        
        # Calculate slippage
        order.slippage += slippage_pct
        
        # Update timestamps
        order.fill_timestamps.append(timestamp)
        
        # Update filled price (weighted average for multiple fills)
        if len(order.partial_fills) == 1:
            order.filled_price = fill_price
        else:
            # Calculate weighted average price
            total_value = sum(fill['price'] * fill['quantity'] for fill in order.partial_fills)
            total_quantity = sum(fill['quantity'] for fill in order.partial_fills)
            order.filled_price = total_value / total_quantity if total_quantity > 0 else 0.0
    
    def _estimate_available_liquidity(self, order: Order, bar_data: pd.Series) -> float:
        """
        Estimate available liquidity for this order based on bar data
        
        Args:
            order: The order being executed
            bar_data: OHLCV data for the current bar
            
        Returns:
            Estimated available quantity
        """
        # If partial fills disabled, return full quantity
        if self.min_fill_ratio <= 0:
            return order.remaining_quantity
        
        # Get volume from bar data
        volume = bar_data.get('volume', 0)
        
        if volume <= 0:
            # If no volume data available, use simple heuristic
            if self.data_frequency == DataFrequency.DAY:
                # On daily bars, assume we can fill the whole order
                return order.remaining_quantity
            else:
                # On intraday bars, use min fill ratio as a constraint
                return order.remaining_quantity * self.min_fill_ratio
        
        # Estimate available liquidity based on volume
        # Assume we can execute a percentage of the volume
        max_participation_rate = 0.1 * self.liquidity_factor  # 10% default, scaled by liquidity factor
        available_quantity = volume * max_participation_rate
        
        # Ensure we have at least min_fill_ratio of the remaining quantity
        min_quantity = order.remaining_quantity * self.min_fill_ratio
        
        return max(min_quantity, min(available_quantity, order.remaining_quantity))
    
    def get_active_orders(self) -> List[Order]:
        """Get list of all active orders"""
        return list(self.active_orders.values())
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID from either active or completed orders"""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        if order_id in self.completed_orders:
            return self.completed_orders[order_id]
        return None
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get full trade history"""
        return self.trade_history
    
    def reset(self) -> None:
        """Reset the simulator state"""
        self.active_orders = {}
        self.completed_orders = {}
        self.trade_history = []
        self.order_id_counter = 0
        
        logger.info("Order execution simulator reset") 