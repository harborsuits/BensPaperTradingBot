"""
Order Type Selector

This module provides functionality for selecting the appropriate order type,
time in force, and other order parameters based on the trading strategy,
market conditions, and broker capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime, time

from .brokerage_client import BrokerageClient, OrderType, OrderSide, TimeInForce

# Configure logging
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Market condition categories for order selection"""
    NORMAL = "normal"              # Standard market conditions
    HIGH_VOLATILITY = "high_vol"   # High volatility
    LOW_VOLATILITY = "low_vol"     # Low volatility
    OPENING = "opening"            # Market opening period
    CLOSING = "closing"            # Market closing period
    AFTER_HOURS = "after_hours"    # After-hours trading
    PRE_MARKET = "pre_market"      # Pre-market trading
    THIN_VOLUME = "thin_volume"    # Low trading volume
    EARNINGS = "earnings"          # During earnings announcements
    NEWS = "news"                  # During significant news events

class ExecutionSpeed(Enum):
    """Execution speed preference"""
    IMMEDIATE = "immediate"  # Fill as quickly as possible
    PASSIVE = "passive"      # Prioritize price over speed
    BALANCED = "balanced"    # Balance speed and price

class PriceAggression(Enum):
    """Price aggression for order placement"""
    AGGRESSIVE = "aggressive"  # Willing to give up price for certainty of execution
    NEUTRAL = "neutral"        # Standard price approach
    PASSIVE = "passive"        # Prioritize price over certainty of execution

class OrderSelector:
    """
    Class for selecting appropriate order parameters based on
    strategy requirements and market conditions.
    """
    
    def __init__(self, 
                broker_client: BrokerageClient,
                default_time_in_force: TimeInForce = TimeInForce.DAY,
                default_execution_speed: ExecutionSpeed = ExecutionSpeed.BALANCED,
                default_price_aggression: PriceAggression = PriceAggression.NEUTRAL):
        """
        Initialize the order selector.
        
        Args:
            broker_client: Brokerage client for checking supported order types
            default_time_in_force: Default time in force setting
            default_execution_speed: Default execution speed preference
            default_price_aggression: Default price aggression setting
        """
        self.broker = broker_client
        self.default_time_in_force = default_time_in_force
        self.default_execution_speed = default_execution_speed
        self.default_price_aggression = default_price_aggression
        
        # Initialize market hours
        self.market_open_time = time(9, 30)  # 9:30 AM ET
        self.market_close_time = time(16, 0)  # 4:00 PM ET
        self.pre_market_start = time(4, 0)    # 4:00 AM ET
        self.after_hours_end = time(20, 0)    # 8:00 PM ET
        
        # Get supported order types from broker
        self.supported_order_types = broker_client.get_supported_order_types()
        self.supported_time_in_force = broker_client.get_supported_time_in_force()
        
        logger.info(f"Initialized OrderSelector with {len(self.supported_order_types)} supported order types")
    
    def select_order_type(self, 
                        desired_type: Optional[OrderType] = None, 
                        market_condition: Optional[MarketCondition] = None,
                        execution_speed: Optional[ExecutionSpeed] = None,
                        price_aggression: Optional[PriceAggression] = None,
                        has_limit_price: bool = False,
                        has_stop_price: bool = False) -> OrderType:
        """
        Select the appropriate order type based on requirements and conditions.
        
        Args:
            desired_type: Preferred order type (if any)
            market_condition: Current market condition
            execution_speed: Desired execution speed
            price_aggression: Desired price aggression
            has_limit_price: Whether a limit price is available
            has_stop_price: Whether a stop price is available
            
        Returns:
            OrderType: Selected order type
        """
        # Use defaults if not provided
        if execution_speed is None:
            execution_speed = self.default_execution_speed
        
        if price_aggression is None:
            price_aggression = self.default_price_aggression
        
        # Detect current market condition if not provided
        if market_condition is None:
            market_condition = self._detect_market_condition()
        
        # If a specific order type is desired and supported, use it
        if desired_type is not None and desired_type in self.supported_order_types:
            # Validate the desired type has necessary price parameters
            if self._validate_order_type_parameters(desired_type, has_limit_price, has_stop_price):
                return desired_type
            else:
                logger.warning(f"Desired order type {desired_type} does not have required price parameters")
        
        # Select based on market condition and execution preferences
        selected_type = self._select_for_market_condition(
            market_condition, 
            execution_speed, 
            price_aggression,
            has_limit_price,
            has_stop_price
        )
        
        # If selected type is not supported or doesn't have required parameters,
        # fall back to a supported alternative
        if selected_type not in self.supported_order_types:
            selected_type = self._find_alternative_order_type(
                selected_type, 
                has_limit_price, 
                has_stop_price
            )
        
        logger.debug(f"Selected order type: {selected_type}")
        return selected_type
    
    def select_time_in_force(self, 
                           desired_tif: Optional[TimeInForce] = None,
                           market_condition: Optional[MarketCondition] = None,
                           order_type: Optional[OrderType] = None) -> TimeInForce:
        """
        Select the appropriate time in force based on requirements and conditions.
        
        Args:
            desired_tif: Preferred time in force (if any)
            market_condition: Current market condition
            order_type: Order type being used
            
        Returns:
            TimeInForce: Selected time in force
        """
        # Use default if not provided
        if desired_tif is None:
            desired_tif = self.default_time_in_force
        
        # Detect market condition if not provided
        if market_condition is None:
            market_condition = self._detect_market_condition()
        
        # If desired time in force is supported, use it
        if desired_tif in self.supported_time_in_force:
            return desired_tif
        
        # Special handling for after-hours and pre-market
        if market_condition in [MarketCondition.AFTER_HOURS, MarketCondition.PRE_MARKET]:
            # For extended hours, prefer GTC if available
            if TimeInForce.GTC in self.supported_time_in_force:
                return TimeInForce.GTC
        
        # Special handling for market close
        if market_condition == MarketCondition.CLOSING:
            # For market close, prefer IOC or FOK to ensure execution before close
            if TimeInForce.IOC in self.supported_time_in_force:
                return TimeInForce.IOC
            elif TimeInForce.FOK in self.supported_time_in_force:
                return TimeInForce.FOK
        
        # Default to DAY if available, otherwise first available option
        if TimeInForce.DAY in self.supported_time_in_force:
            return TimeInForce.DAY
        else:
            return next(iter(self.supported_time_in_force))
    
    def get_optimal_order_parameters(self,
                                   symbol: str,
                                   side: OrderSide,
                                   quantity: float,
                                   desired_entry: Optional[float] = None,
                                   stop_price: Optional[float] = None,
                                   market_condition: Optional[MarketCondition] = None,
                                   execution_speed: Optional[ExecutionSpeed] = None,
                                   price_aggression: Optional[PriceAggression] = None) -> Dict[str, Any]:
        """
        Get optimal order parameters for a trade.
        
        Args:
            symbol: Symbol to trade
            side: Order side
            quantity: Order quantity
            desired_entry: Desired entry price (if any)
            stop_price: Desired stop price (if any)
            market_condition: Current market condition
            execution_speed: Desired execution speed
            price_aggression: Desired price aggression
            
        Returns:
            Dict[str, Any]: Optimal order parameters
        """
        # Determine if we have prices
        has_limit_price = desired_entry is not None
        has_stop_price = stop_price is not None
        
        # Select order type
        order_type = self.select_order_type(
            market_condition=market_condition,
            execution_speed=execution_speed,
            price_aggression=price_aggression,
            has_limit_price=has_limit_price,
            has_stop_price=has_stop_price
        )
        
        # Select time in force
        time_in_force = self.select_time_in_force(
            market_condition=market_condition,
            order_type=order_type
        )
        
        # Adjust prices based on price aggression
        limit_price = None
        if has_limit_price:
            limit_price = self._adjust_price_for_aggression(
                desired_entry, 
                side, 
                price_aggression
            )
        
        # Prepare parameters
        params = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'time_in_force': time_in_force
        }
        
        # Add prices if available
        if limit_price is not None:
            params['limit_price'] = limit_price
        
        if stop_price is not None:
            params['stop_price'] = stop_price
        
        return params
    
    def _detect_market_condition(self) -> MarketCondition:
        """
        Detect current market condition based on time of day and market status.
        
        Returns:
            MarketCondition: Detected market condition
        """
        now = datetime.now().time()
        
        # Check if market is open
        try:
            is_open = self.broker.is_market_open()
        except Exception:
            # Default to assuming market is open during normal hours
            is_open = (
                now >= self.market_open_time and
                now < self.market_close_time
            )
        
        # Determine market condition based on time
        if is_open:
            # Market open conditions
            if now < self.market_open_time + time(0, 15):  # First 15 mins
                return MarketCondition.OPENING
            elif now > self.market_close_time - time(0, 15):  # Last 15 mins
                return MarketCondition.CLOSING
            else:
                return MarketCondition.NORMAL
        else:
            # Market closed conditions
            if now >= self.pre_market_start and now < self.market_open_time:
                return MarketCondition.PRE_MARKET
            elif now >= self.market_close_time and now < self.after_hours_end:
                return MarketCondition.AFTER_HOURS
            else:
                return MarketCondition.NORMAL
    
    def _validate_order_type_parameters(self, 
                                      order_type: OrderType, 
                                      has_limit_price: bool, 
                                      has_stop_price: bool) -> bool:
        """
        Validate that an order type has necessary price parameters.
        
        Args:
            order_type: Order type to validate
            has_limit_price: Whether a limit price is available
            has_stop_price: Whether a stop price is available
            
        Returns:
            bool: True if valid, False otherwise
        """
        if order_type == OrderType.LIMIT and not has_limit_price:
            return False
        elif order_type == OrderType.STOP and not has_stop_price:
            return False
        elif order_type == OrderType.STOP_LIMIT and (not has_limit_price or not has_stop_price):
            return False
        
        return True
    
    def _select_for_market_condition(self,
                                   market_condition: MarketCondition,
                                   execution_speed: ExecutionSpeed,
                                   price_aggression: PriceAggression,
                                   has_limit_price: bool,
                                   has_stop_price: bool) -> OrderType:
        """
        Select order type based on market condition and execution preferences.
        
        Args:
            market_condition: Current market condition
            execution_speed: Desired execution speed
            price_aggression: Desired price aggression
            has_limit_price: Whether a limit price is available
            has_stop_price: Whether a stop price is available
            
        Returns:
            OrderType: Selected order type
        """
        # High volatility conditions
        if market_condition == MarketCondition.HIGH_VOLATILITY:
            if execution_speed == ExecutionSpeed.IMMEDIATE:
                return OrderType.MARKET
            elif has_limit_price:
                return OrderType.LIMIT
        
        # Market opening/closing
        if market_condition in [MarketCondition.OPENING, MarketCondition.CLOSING]:
            if execution_speed == ExecutionSpeed.IMMEDIATE:
                return OrderType.MARKET
            elif has_limit_price and price_aggression != PriceAggression.AGGRESSIVE:
                return OrderType.LIMIT
            else:
                return OrderType.MARKET
        
        # Extended hours
        if market_condition in [MarketCondition.PRE_MARKET, MarketCondition.AFTER_HOURS]:
            # Extended hours typically requires limit orders
            if has_limit_price:
                return OrderType.LIMIT
            else:
                logger.warning("Extended hours trading typically requires limit orders")
                return OrderType.LIMIT  # Will need a price
        
        # Normal conditions
        if market_condition == MarketCondition.NORMAL:
            if execution_speed == ExecutionSpeed.IMMEDIATE:
                return OrderType.MARKET
            elif execution_speed == ExecutionSpeed.PASSIVE and has_limit_price:
                return OrderType.LIMIT
            elif has_stop_price and has_limit_price:
                return OrderType.STOP_LIMIT
            elif has_stop_price:
                return OrderType.STOP
            elif has_limit_price:
                return OrderType.LIMIT
        
        # Thin volume
        if market_condition == MarketCondition.THIN_VOLUME:
            if has_limit_price:
                return OrderType.LIMIT
            else:
                logger.warning("Thin volume trading is safer with limit orders")
                return OrderType.LIMIT  # Will need a price
        
        # Default to market order
        return OrderType.MARKET
    
    def _find_alternative_order_type(self,
                                   desired_type: OrderType,
                                   has_limit_price: bool,
                                   has_stop_price: bool) -> OrderType:
        """
        Find an alternative order type if desired type is not supported.
        
        Args:
            desired_type: Desired order type
            has_limit_price: Whether a limit price is available
            has_stop_price: Whether a stop price is available
            
        Returns:
            OrderType: Alternative order type
        """
        # Market order is always a fallback if supported
        if OrderType.MARKET in self.supported_order_types:
            market_fallback = OrderType.MARKET
        else:
            # If market orders aren't supported, use first available type
            market_fallback = next(iter(self.supported_order_types))
        
        # Find alternatives based on desired type
        if desired_type == OrderType.LIMIT:
            if has_limit_price and has_stop_price and OrderType.STOP_LIMIT in self.supported_order_types:
                return OrderType.STOP_LIMIT
            elif market_fallback:
                return market_fallback
        
        elif desired_type == OrderType.STOP:
            if has_limit_price and has_stop_price and OrderType.STOP_LIMIT in self.supported_order_types:
                return OrderType.STOP_LIMIT
            elif market_fallback:
                return market_fallback
        
        elif desired_type == OrderType.STOP_LIMIT:
            if has_stop_price and OrderType.STOP in self.supported_order_types:
                return OrderType.STOP
            elif has_limit_price and OrderType.LIMIT in self.supported_order_types:
                return OrderType.LIMIT
            elif market_fallback:
                return market_fallback
        
        elif desired_type == OrderType.TRAILING_STOP:
            if has_stop_price and OrderType.STOP in self.supported_order_types:
                return OrderType.STOP
            elif market_fallback:
                return market_fallback
        
        # Default to market order
        return market_fallback
    
    def _adjust_price_for_aggression(self, 
                                   base_price: float, 
                                   side: OrderSide, 
                                   aggression: PriceAggression) -> float:
        """
        Adjust price based on specified aggression level.
        
        Args:
            base_price: Base price
            side: Order side
            aggression: Price aggression level
            
        Returns:
            float: Adjusted price
        """
        # Default to no adjustment
        adjustment = 0.0
        
        # Adjust based on aggression
        if aggression == PriceAggression.AGGRESSIVE:
            # Aggressive pricing - willing to pay more or receive less
            if side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                # For buys, increase price to increase likelihood of fill
                adjustment = 0.01 * base_price  # Adjust by 1%
            else:
                # For sells, decrease price to increase likelihood of fill
                adjustment = -0.01 * base_price  # Adjust by -1%
        
        elif aggression == PriceAggression.PASSIVE:
            # Passive pricing - trying to get better execution price
            if side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                # For buys, decrease price to try to get better fill
                adjustment = -0.005 * base_price  # Adjust by -0.5%
            else:
                # For sells, increase price to try to get better fill
                adjustment = 0.005 * base_price  # Adjust by 0.5%
        
        # Apply adjustment
        adjusted_price = base_price + adjustment
        
        # Ensure minimum price increment
        adjusted_price = round(adjusted_price, 2)  # Round to 2 decimal places for most stocks
        
        return adjusted_price 