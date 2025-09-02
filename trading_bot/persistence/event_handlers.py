#!/usr/bin/env python3
"""
Persistence Event Handlers

This module provides event handlers that persist trading events to the database.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import (
    OrderAcknowledged, OrderFilled, OrderPartialFill, OrderRejected, OrderCancelled,
    PositionUpdate, PortfolioEquityUpdate, EventType
)
from trading_bot.persistence.order_repository import OrderRepository
from trading_bot.persistence.fill_repository import FillRepository
from trading_bot.persistence.position_repository import PositionRepository, PositionModel
from trading_bot.persistence.pnl_repository import PnLRepository, PnLModel
from trading_bot.persistence.connection_manager import ConnectionManager


class PersistenceEventHandler:
    """Handles persistence of trading events to the database"""
    
    def __init__(self, connection_manager: ConnectionManager, event_bus: EventBus):
        """
        Initialize the persistence event handler.
        
        Args:
            connection_manager: Database connection manager
            event_bus: Event bus to subscribe to
        """
        self.connection_manager = connection_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Initialize repositories
        self.order_repo = OrderRepository(connection_manager)
        self.fill_repo = FillRepository(connection_manager)
        self.position_repo = PositionRepository(connection_manager)
        self.pnl_repo = PnLRepository(connection_manager)
        
        # Subscribe to events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events on the event bus"""
        # Order lifecycle events
        self.event_bus.subscribe(EventType.ORDER_ACKNOWLEDGED, self._handle_order_acknowledged)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self.event_bus.subscribe(EventType.ORDER_PARTIAL_FILL, self._handle_order_partial_fill)
        self.event_bus.subscribe(EventType.ORDER_REJECTED, self._handle_order_rejected)
        self.event_bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        
        # Position events
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        
        # Portfolio events
        self.event_bus.subscribe(EventType.PORTFOLIO_EQUITY_UPDATE, self._handle_portfolio_equity_update)
        
        self.logger.info("Persistence event handlers subscribed to event bus")
    
    def _handle_order_acknowledged(self, event: OrderAcknowledged) -> None:
        """
        Handle OrderAcknowledged event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting OrderAcknowledged: {event.order_id}")
            self.order_repo.update_from_event(event)
            
            # Register idempotent mapping if broker_order_id is available
            if hasattr(event, 'broker_order_id') and event.broker_order_id:
                self.order_repo.register_idempotent_mapping(
                    event.order_id, event.broker_order_id, event.broker
                )
                
        except Exception as e:
            self.logger.error(f"Error persisting OrderAcknowledged: {str(e)}")
    
    def _handle_order_filled(self, event: OrderFilled) -> None:
        """
        Handle OrderFilled event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting OrderFilled: {event.order_id}")
            
            # Update order status
            self.order_repo.update_from_event(event)
            
            # Record fill
            self.fill_repo.record_fill(event)
            
            # Update position
            if hasattr(event, 'symbol') and event.symbol:
                self.position_repo.update_from_fill(
                    symbol=event.symbol,
                    broker=event.broker,
                    price=event.avg_fill_price,
                    quantity=event.total_qty if event.side.lower() == 'buy' else -event.total_qty,
                    strategy=event.metadata.get('strategy') if event.metadata else None
                )
                
        except Exception as e:
            self.logger.error(f"Error persisting OrderFilled: {str(e)}")
    
    def _handle_order_partial_fill(self, event: OrderPartialFill) -> None:
        """
        Handle OrderPartialFill event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting OrderPartialFill: {event.order_id}")
            
            # Update order status
            self.order_repo.update_from_event(event)
            
            # Record fill
            self.fill_repo.record_fill(event)
            
            # Update position
            if hasattr(event, 'symbol') and event.symbol:
                self.position_repo.update_from_fill(
                    symbol=event.symbol,
                    broker=event.broker,
                    price=event.fill_price,
                    quantity=event.fill_qty if event.side.lower() == 'buy' else -event.fill_qty,
                    strategy=event.metadata.get('strategy') if event.metadata else None
                )
                
        except Exception as e:
            self.logger.error(f"Error persisting OrderPartialFill: {str(e)}")
    
    def _handle_order_rejected(self, event: OrderRejected) -> None:
        """
        Handle OrderRejected event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting OrderRejected: {event.order_id}")
            self.order_repo.update_from_event(event)
                
        except Exception as e:
            self.logger.error(f"Error persisting OrderRejected: {str(e)}")
    
    def _handle_order_cancelled(self, event: OrderCancelled) -> None:
        """
        Handle OrderCancelled event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting OrderCancelled: {event.order_id}")
            self.order_repo.update_from_event(event)
                
        except Exception as e:
            self.logger.error(f"Error persisting OrderCancelled: {str(e)}")
    
    def _handle_position_update(self, event: PositionUpdate) -> None:
        """
        Handle PositionUpdate event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting PositionUpdate: {event.symbol}")
            
            # Create or update position
            position = PositionModel(
                symbol=event.symbol,
                quantity=event.quantity,
                avg_cost=event.avg_cost,
                broker=event.broker,
                unrealized_pnl=event.unrealized_pnl if hasattr(event, 'unrealized_pnl') else None,
                strategy=event.metadata.get('strategy') if event.metadata else None
            )
            
            position_id = f"{event.broker}:{event.symbol}"
            
            # Find existing position
            existing = self.position_repo.find_by_position_id(position_id)
            
            if existing:
                # Update existing position
                position._id = existing._id
                position.open_date = existing.open_date
                position.realized_pnl = existing.realized_pnl
                
                if not position.unrealized_pnl and existing.unrealized_pnl:
                    position.unrealized_pnl = existing.unrealized_pnl
                
                self.position_repo.update_position(position_id, position)
            else:
                # Create new position
                self.position_repo.save_position(position)
                
        except Exception as e:
            self.logger.error(f"Error persisting PositionUpdate: {str(e)}")
    
    def _handle_portfolio_equity_update(self, event: PortfolioEquityUpdate) -> None:
        """
        Handle PortfolioEquityUpdate event.
        
        Args:
            event: The event to handle
        """
        try:
            self.logger.debug(f"Persisting PortfolioEquityUpdate")
            
            # Create PnL record
            pnl = PnLModel.from_portfolio_update(
                equity=event.total_equity,
                unrealized_pnl=event.unrealized_pnl,
                realized_pnl=event.realized_pnl,
                cash=event.cash_balance if hasattr(event, 'cash_balance') else None,
                broker=event.broker if hasattr(event, 'broker') else None
            )
            
            # Save snapshot
            self.pnl_repo.record_snapshot(pnl)
            
            # If it's end of day, save EOD snapshot as well
            now = datetime.now()
            if now.hour >= 16 and now.minute >= 0:  # After market close (4:00 PM)
                self.pnl_repo.record_eod_snapshot(pnl)
                
                # Sync positions to durable storage
                self.position_repo.sync_to_durable_storage()
                
        except Exception as e:
            self.logger.error(f"Error persisting PortfolioEquityUpdate: {str(e)}")


# Sync function for hourly sync to durable storage
def sync_hot_state_to_durable_storage(persistence_handler: PersistenceEventHandler) -> None:
    """
    Sync hot state (Redis) to durable storage (MongoDB).
    
    Args:
        persistence_handler: Persistence event handler
    """
    try:
        # Sync positions
        persistence_handler.position_repo.sync_to_durable_storage()
        
        # Log success
        persistence_handler.logger.info("Successfully synced hot state to durable storage")
        
    except Exception as e:
        persistence_handler.logger.error(f"Error syncing hot state to durable storage: {str(e)}")
