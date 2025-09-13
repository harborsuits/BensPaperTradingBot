#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Broker Router

Routes orders from strategies to the appropriate broker (paper or live)
based on the strategy's configuration.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.core.strategy_base import Strategy
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)


class StrategyBrokerRouter:
    """
    Routes orders from strategies to the appropriate broker (paper or live)
    based on each strategy's mode configuration.
    """
    
    def __init__(
        self,
        broker_manager: MultiBrokerManager,
        default_paper_broker_id: str = "paper",
        default_live_broker_id: Optional[str] = None
    ):
        """
        Initialize the strategy broker router.
        
        Args:
            broker_manager: Multi-broker manager instance
            default_paper_broker_id: Default broker ID for paper trading
            default_live_broker_id: Default broker ID for live trading
        """
        self._broker_manager = broker_manager
        self._default_paper_broker_id = default_paper_broker_id
        self._default_live_broker_id = default_live_broker_id
        
        # Mapping of strategy IDs to broker IDs
        self._strategy_broker_map: Dict[str, str] = {}
        
        # Mapping of strategy IDs to trading modes (paper/live)
        self._strategy_mode_map: Dict[str, str] = {}
        
        # Event bus for notifications
        self._event_bus = ServiceRegistry.get_instance().get_service(EventBus)
        
        logger.info("Strategy broker router initialized")
    
    def register_strategy(
        self,
        strategy: Strategy,
        mode: str = "paper",
        broker_id: Optional[str] = None
    ) -> None:
        """
        Register a strategy with a specific trading mode and broker.
        
        Args:
            strategy: Strategy instance to register
            mode: Trading mode ("paper" or "live")
            broker_id: Specific broker ID to use, or None for default
        """
        strategy_id = strategy.get_id()
        
        # Validate mode
        if mode not in ["paper", "live"]:
            logger.warning(f"Invalid mode '{mode}' for strategy '{strategy_id}', defaulting to 'paper'")
            mode = "paper"
        
        # Determine broker ID
        if broker_id is None:
            if mode == "paper":
                broker_id = self._default_paper_broker_id
            else:
                broker_id = self._default_live_broker_id
        
        # Validate broker exists
        if broker_id not in self._broker_manager.get_broker_ids():
            logger.error(f"Broker '{broker_id}' not found for strategy '{strategy_id}'")
            return
        
        # Register mappings
        self._strategy_broker_map[strategy_id] = broker_id
        self._strategy_mode_map[strategy_id] = mode
        
        # If paper mode, ensure the PAPER tag is added to the strategy's tags
        if mode == "paper" and hasattr(strategy, 'add_tag'):
            strategy.add_tag("PAPER")
        
        logger.info(f"Registered strategy '{strategy_id}' with {mode} mode using broker '{broker_id}'")
        
        # Publish registration event
        self._publish_registration_event(strategy_id, mode, broker_id)
    
    def register_strategy_from_config(
        self,
        strategy: Strategy,
        config: Dict[str, Any]
    ) -> None:
        """
        Register a strategy using configuration dictionary.
        
        Args:
            strategy: Strategy instance to register
            config: Strategy configuration dict
        """
        strategy_id = strategy.get_id()
        mode = config.get("mode", "paper").lower()
        broker_id = config.get("broker")
        
        self.register_strategy(strategy, mode, broker_id)
    
    def get_broker_for_strategy(self, strategy_id: str) -> Optional[BrokerInterface]:
        """
        Get the broker instance for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            BrokerInterface: Broker instance or None if not found
        """
        broker_id = self._strategy_broker_map.get(strategy_id)
        if not broker_id:
            logger.warning(f"No broker mapping found for strategy '{strategy_id}'")
            return None
        
        try:
            return self._broker_manager.get_broker(broker_id)
        except Exception as e:
            logger.error(f"Error getting broker '{broker_id}' for strategy '{strategy_id}': {str(e)}")
            return None
    
    def place_order_for_strategy(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        duration: str = "day",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place an order on behalf of a strategy.
        
        Args:
            strategy_id: Strategy ID
            symbol: Stock symbol
            side: Order side ('buy' or 'sell')
            quantity: Number of shares
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            duration: Order duration ('day', 'gtc')
            price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            tags: Optional tags for the order
            
        Returns:
            Dict: Order result with ID and status, or None if failed
        """
        # Get broker for this strategy
        broker = self.get_broker_for_strategy(strategy_id)
        if not broker:
            logger.error(f"No broker available for strategy '{strategy_id}'")
            return None
        
        # Get trading mode for this strategy
        mode = self._strategy_mode_map.get(strategy_id, "paper")
        
        # Create tags if none provided
        if tags is None:
            tags = []
        
        # Always add strategy ID to tags
        if strategy_id not in tags:
            tags.append(strategy_id)
        
        # Add PAPER tag for paper trading mode
        if mode == "paper" and "PAPER" not in tags:
            tags.append("PAPER")
        
        try:
            # Place order
            result = broker.place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                duration=duration,
                price=price,
                stop_price=stop_price,
                strategy_id=strategy_id,
                tags=tags
            )
            
            # Log with appropriate tag
            mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"
            logger.info(f"{mode_tag} Order placed for strategy '{strategy_id}': {side} {quantity} {symbol}")
            
            # Publish order event
            self._publish_order_event(
                strategy_id=strategy_id,
                mode=mode,
                order_result=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing order for strategy '{strategy_id}': {str(e)}")
            return None
    
    def cancel_order_for_strategy(
        self,
        strategy_id: str,
        order_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Cancel an order on behalf of a strategy.
        
        Args:
            strategy_id: Strategy ID
            order_id: Order ID to cancel
            
        Returns:
            Dict: Cancellation result or None if failed
        """
        # Get broker for this strategy
        broker = self.get_broker_for_strategy(strategy_id)
        if not broker:
            logger.error(f"No broker available for strategy '{strategy_id}'")
            return None
        
        try:
            # Cancel order
            result = broker.cancel_order(order_id)
            
            # Log cancellation
            mode = self._strategy_mode_map.get(strategy_id, "paper")
            mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"
            logger.info(f"{mode_tag} Order {order_id} canceled for strategy '{strategy_id}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error canceling order for strategy '{strategy_id}': {str(e)}")
            return None
    
    def get_order_status_for_strategy(
        self,
        strategy_id: str,
        order_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get order status on behalf of a strategy.
        
        Args:
            strategy_id: Strategy ID
            order_id: Order ID to check
            
        Returns:
            Dict: Order status or None if failed
        """
        # Get broker for this strategy
        broker = self.get_broker_for_strategy(strategy_id)
        if not broker:
            logger.error(f"No broker available for strategy '{strategy_id}'")
            return None
        
        try:
            return broker.get_order_status(order_id)
        except Exception as e:
            logger.error(f"Error getting order status for strategy '{strategy_id}': {str(e)}")
            return None
    
    def get_positions_for_strategy(
        self,
        strategy_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get positions on behalf of a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List[Dict]: Positions or None if failed
        """
        # Get broker for this strategy
        broker = self.get_broker_for_strategy(strategy_id)
        if not broker:
            logger.error(f"No broker available for strategy '{strategy_id}'")
            return None
        
        try:
            # Get all positions
            positions = broker.get_positions()
            
            # Filter to only include positions with this strategy's tag
            # This only works reliably with paper broker that tracks strategy IDs
            filtered_positions = []
            for position in positions:
                tags = position.get("tags", [])
                if strategy_id in tags:
                    filtered_positions.append(position)
            
            # If filtering found positions, return those
            if filtered_positions:
                return filtered_positions
            
            # Otherwise return all (broker doesn't support tagging)
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions for strategy '{strategy_id}': {str(e)}")
            return None
    
    def get_strategy_ids(self) -> List[str]:
        """
        Get all registered strategy IDs.
        
        Returns:
            List[str]: List of strategy IDs
        """
        return list(self._strategy_broker_map.keys())
    
    def get_strategy_mode(self, strategy_id: str) -> Optional[str]:
        """
        Get trading mode for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            str: 'paper' or 'live', or None if not found
        """
        return self._strategy_mode_map.get(strategy_id)
    
    def get_strategy_broker_id(self, strategy_id: str) -> Optional[str]:
        """
        Get broker ID for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            str: Broker ID or None if not found
        """
        return self._strategy_broker_map.get(strategy_id)
    
    def get_strategies_by_mode(self, mode: str) -> List[str]:
        """
        Get all strategy IDs with a specific mode.
        
        Args:
            mode: Trading mode ('paper' or 'live')
            
        Returns:
            List[str]: List of strategy IDs
        """
        return [
            strategy_id for strategy_id, strategy_mode 
            in self._strategy_mode_map.items() 
            if strategy_mode == mode
        ]
    
    def _publish_registration_event(self, strategy_id: str, mode: str, broker_id: str) -> None:
        """
        Publish strategy registration event.
        
        Args:
            strategy_id: Strategy ID
            mode: Trading mode
            broker_id: Broker ID
        """
        if not self._event_bus:
            return
        
        event_data = {
            "strategy_id": strategy_id,
            "mode": mode,
            "broker_id": broker_id,
            "timestamp": str(datetime.now())
        }
        
        self._event_bus.publish(Event(
            event_type=EventType.STRATEGY_REGISTERED,
            data=event_data
        ))
    
    def _publish_order_event(self, strategy_id: str, mode: str, order_result: Dict[str, Any]) -> None:
        """
        Publish order placement event.
        
        Args:
            strategy_id: Strategy ID
            mode: Trading mode
            order_result: Order result from broker
        """
        if not self._event_bus:
            return
        
        event_data = {
            "strategy_id": strategy_id,
            "mode": mode,
            "order": order_result,
            "timestamp": str(datetime.now())
        }
        
        self._event_bus.publish(Event(
            event_type=EventType.ORDER_PLACED,
            data=event_data
        ))
