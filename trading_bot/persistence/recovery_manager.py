#!/usr/bin/env python3
"""
Recovery Manager

This module provides state recovery functionality for the trading system.
"""

import logging
from typing import Dict, List, Any, Optional, Set

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import EventType, OrderAcknowledged, OrderStatus
from trading_bot.persistence.order_repository import OrderRepository, OrderModel
from trading_bot.persistence.position_repository import PositionRepository, PositionModel
from trading_bot.persistence.fill_repository import FillRepository
from trading_bot.persistence.pnl_repository import PnLRepository
from trading_bot.persistence.idempotency import IdempotencyManager
from trading_bot.persistence.connection_manager import ConnectionManager


class RecoveryManager:
    """Manages recovery of trading state after restart/crash"""
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        event_bus: EventBus,
        order_repo: Optional[OrderRepository] = None,
        position_repo: Optional[PositionRepository] = None,
        fill_repo: Optional[FillRepository] = None,
        pnl_repo: Optional[PnLRepository] = None,
        idempotency_manager: Optional[IdempotencyManager] = None
    ):
        """
        Initialize the recovery manager.
        
        Args:
            connection_manager: Database connection manager
            event_bus: Event bus to publish recovery events
            order_repo: Optional order repository (created if not provided)
            position_repo: Optional position repository (created if not provided)
            fill_repo: Optional fill repository (created if not provided)
            pnl_repo: Optional PnL repository (created if not provided)
            idempotency_manager: Optional idempotency manager (created if not provided)
        """
        self.connection_manager = connection_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Initialize repositories if not provided
        self.order_repo = order_repo or OrderRepository(connection_manager)
        self.position_repo = position_repo or PositionRepository(connection_manager)
        self.fill_repo = fill_repo or FillRepository(connection_manager)
        self.pnl_repo = pnl_repo or PnLRepository(connection_manager)
        self.idempotency_manager = idempotency_manager or IdempotencyManager(connection_manager)
        
        # Keep track of recovered items
        self.recovered_orders: Set[str] = set()
        self.recovered_positions: Set[str] = set()
        
    def recover_full_state(self) -> Dict[str, Any]:
        """
        Recover full trading state from persistent storage.
        
        Returns:
            Summary of recovery with counts of recovered items
        """
        self.logger.info("Starting full state recovery...")
        
        # Reset recovery counters
        self.recovered_orders.clear()
        self.recovered_positions.clear()
        
        # Recover in sequence
        open_orders = self.recover_open_orders()
        positions = self.recover_positions()
        pnl = self.recover_latest_pnl()
        
        # Return recovery summary
        return {
            'open_orders_recovered': len(open_orders),
            'positions_recovered': len(positions),
            'pnl_recovered': pnl is not None,
            'order_ids': list(self.recovered_orders),
            'position_symbols': list(symbol for broker, symbol in self.recovered_positions)
        }
        
    def recover_open_orders(self) -> List[OrderModel]:
        """
        Recover open orders from persistent storage.
        
        Returns:
            List of recovered open orders
        """
        try:
            self.logger.info("Recovering open orders...")
            
            # Get open orders from repository
            open_orders = self.order_repo.fetch_open_orders()
            
            if not open_orders:
                self.logger.info("No open orders to recover")
                return []
                
            self.logger.info(f"Recovered {len(open_orders)} open orders")
            
            # Emit events for each open order
            for order in open_orders:
                # Track recovered order IDs
                self.recovered_orders.add(order.internal_id)
                
                # Create order acknowledged event for each open order
                # This will allow the system to re-establish broker polling for these orders
                event = OrderAcknowledged(
                    order_id=order.internal_id,
                    broker=order.broker,
                    symbol=order.symbol,
                    quantity=order.quantity,
                    side=order.side,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    status=OrderStatus.ACKNOWLEDGED,
                    broker_order_id=order.broker_order_id
                )
                
                # Set recovery flag on event
                if not hasattr(event, 'metadata') or not event.metadata:
                    event.metadata = {}
                event.metadata['recovered'] = True
                
                # Emit event
                self.event_bus.emit(EventType.ORDER_ACKNOWLEDGED, event)
                
            return open_orders
            
        except Exception as e:
            self.logger.error(f"Error recovering open orders: {str(e)}")
            return []
            
    def recover_positions(self) -> List[PositionModel]:
        """
        Recover positions from persistent storage.
        
        Returns:
            List of recovered positions
        """
        try:
            self.logger.info("Recovering positions...")
            
            # Get positions from repository
            positions = self.position_repo.find_non_zero_positions()
            
            if not positions:
                self.logger.info("No positions to recover")
                return []
                
            self.logger.info(f"Recovered {len(positions)} positions")
            
            # Load positions into Redis for hot-state caching
            self.position_repo.load_positions(positions)
            
            # Emit position update events for each position
            for position in positions:
                # Track recovered positions
                self.recovered_positions.add((position.broker, position.symbol))
                
                # Create position update event
                event = {
                    'type': EventType.POSITION_UPDATE,
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_cost': position.avg_cost,
                    'broker': position.broker,
                    'unrealized_pnl': position.unrealized_pnl,
                    'strategy': position.strategy,
                    'metadata': {
                        'recovered': True,
                        'position_id': position.position_id,
                        'open_date': position.open_date.isoformat() if position.open_date else None
                    }
                }
                
                # Emit event
                self.event_bus.emit(EventType.POSITION_UPDATE, event)
                
            return positions
            
        except Exception as e:
            self.logger.error(f"Error recovering positions: {str(e)}")
            return []
            
    def recover_latest_pnl(self) -> Optional[Dict[str, Any]]:
        """
        Recover latest P&L data from persistent storage.
        
        Returns:
            Latest P&L data if available
        """
        try:
            self.logger.info("Recovering latest P&L...")
            
            # Get latest P&L snapshot
            latest_pnl = self.pnl_repo.get_latest_snapshot()
            
            if not latest_pnl:
                self.logger.info("No P&L data to recover")
                return None
                
            self.logger.info(f"Recovered latest P&L from {latest_pnl.timestamp}")
            
            # Create portfolio equity update event
            event = {
                'type': EventType.PORTFOLIO_EQUITY_UPDATE,
                'total_equity': latest_pnl.total_equity,
                'unrealized_pnl': latest_pnl.unrealized_pnl,
                'realized_pnl': latest_pnl.realized_pnl,
                'cash_balance': latest_pnl.cash_balance,
                'broker': latest_pnl.broker,
                'metadata': {
                    'recovered': True,
                    'timestamp': latest_pnl.timestamp.isoformat()
                }
            }
            
            # Emit event
            self.event_bus.emit(EventType.PORTFOLIO_EQUITY_UPDATE, event)
            
            # Return P&L data as dict
            return {
                'total_equity': latest_pnl.total_equity,
                'unrealized_pnl': latest_pnl.unrealized_pnl,
                'realized_pnl': latest_pnl.realized_pnl,
                'timestamp': latest_pnl.timestamp.isoformat(),
                'drawdown': latest_pnl.drawdown,
                'drawdown_pct': latest_pnl.drawdown_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error recovering latest P&L: {str(e)}")
            return None
    
    def recover_idempotency_state(self) -> int:
        """
        Recover idempotency state to prevent duplicate operations.
        
        Returns:
            Number of pending operations recovered
        """
        try:
            self.logger.info("Recovering idempotency state...")
            
            # Initialize idempotency manager's cache
            self.idempotency_manager._init_cache()
            
            # Get pending operations
            pending_operations = self.idempotency_manager.find_pending_operations()
            
            self.logger.info(f"Recovered {len(pending_operations)} pending idempotent operations")
            
            return len(pending_operations)
            
        except Exception as e:
            self.logger.error(f"Error recovering idempotency state: {str(e)}")
            return 0
