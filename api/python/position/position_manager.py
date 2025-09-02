"""
Position Manager

This module handles position tracking and reconciliation between the internal
position tracking system and actual broker positions. It ensures that what
the trading bot thinks it has matches what actually exists at the broker.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json

# Import broker-related components
from trading_bot.brokers.broker_interface import BrokerInterface, Position
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.analytics.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

class PositionReconciliationError(Exception):
    """Exception raised when position reconciliation fails."""
    pass

class PositionManager:
    """
    Manages positions across multiple brokers and ensures reconciliation
    between internal position state and actual broker positions.
    """
    
    def __init__(self, 
                 broker_manager: MultiBrokerManager,
                 trade_logger: Optional[TradeLogger] = None,
                 reconciliation_interval: int = 60,  # seconds
                 auto_reconcile: bool = True,
                 position_history_file: Optional[str] = None):
        """
        Initialize the position manager.
        
        Args:
            broker_manager: Manager for multiple broker connections
            trade_logger: Optional logger for trade and position tracking
            reconciliation_interval: How often to reconcile positions (seconds)
            auto_reconcile: Whether to automatically reconcile positions
            position_history_file: Optional file to save position history
        """
        self.broker_manager = broker_manager
        self.trade_logger = trade_logger
        
        # Internal position tracking
        self.internal_positions: Dict[str, Dict[str, Any]] = {}  # Key: position_id (symbol-broker_id)
        
        # Reconciliation settings
        self.reconciliation_interval = reconciliation_interval
        self.auto_reconcile = auto_reconcile
        self.last_reconciliation = datetime.now()
        self.reconciliation_errors: List[Dict[str, Any]] = []
        
        # Thread for periodic reconciliation
        self.reconciliation_thread = None
        self.reconciliation_active = False
        
        # Position history
        self.position_history_file = position_history_file
        self.position_history: List[Dict[str, Any]] = []
        
        # For thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._load_internal_positions()
        
        logger.info("Position Manager initialized")
    
    def _load_internal_positions(self):
        """Load internal positions from trade logger if available."""
        if not self.trade_logger:
            logger.warning("No trade logger provided, internal positions will start empty")
            return
            
        try:
            positions = self.trade_logger.get_open_positions()
            for position in positions:
                position_id = self._generate_position_id(position.get('symbol', ''), 
                                                        position.get('broker_id', 'unknown'))
                self.internal_positions[position_id] = position
                
            logger.info(f"Loaded {len(positions)} internal positions from trade logger")
        except Exception as e:
            logger.error(f"Error loading internal positions: {str(e)}")
    
    def start_reconciliation_thread(self) -> bool:
        """
        Start background thread for position reconciliation.
        
        Returns:
            bool: Success status
        """
        if self.reconciliation_thread and self.reconciliation_thread.is_alive():
            logger.warning("Reconciliation thread already running")
            return False
            
        self.reconciliation_active = True
        self.reconciliation_thread = threading.Thread(
            target=self._reconciliation_loop,
            daemon=True
        )
        self.reconciliation_thread.start()
        logger.info("Position reconciliation thread started")
        return True
    
    def stop_reconciliation_thread(self) -> bool:
        """
        Stop the reconciliation thread.
        
        Returns:
            bool: Success status
        """
        if not self.reconciliation_thread or not self.reconciliation_thread.is_alive():
            logger.warning("No active reconciliation thread to stop")
            return False
            
        self.reconciliation_active = False
        self.reconciliation_thread.join(timeout=5.0)
        logger.info("Position reconciliation thread stopped")
        return True
    
    def _reconciliation_loop(self):
        """Background thread for periodic position reconciliation."""
        logger.info("Position reconciliation loop started")
        
        while self.reconciliation_active:
            try:
                self.reconcile_positions()
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {str(e)}")
                
            # Sleep for the interval
            time.sleep(self.reconciliation_interval)
    
    def reconcile_positions(self, broker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reconcile internal positions with broker positions.
        
        Args:
            broker_id: Optional specific broker to reconcile with
                      (reconciles all brokers if None)
                      
        Returns:
            Dict with reconciliation results
        """
        with self._lock:
            logger.info("Starting position reconciliation")
            self.last_reconciliation = datetime.now()
            
            # Get all broker positions
            broker_positions = {}
            
            if broker_id:
                # Reconcile with a specific broker
                try:
                    positions = self.broker_manager.brokers[broker_id].get_positions()
                    broker_positions[broker_id] = positions
                except Exception as e:
                    logger.error(f"Error getting positions from broker {broker_id}: {str(e)}")
                    return {"success": False, "error": str(e)}
            else:
                # Reconcile with all brokers
                try:
                    broker_positions = self.broker_manager.get_all_positions()
                except Exception as e:
                    logger.error(f"Error getting positions from all brokers: {str(e)}")
                    return {"success": False, "error": str(e)}
            
            # Flatten broker positions for easier comparison
            flat_broker_positions = {}
            for broker_id, positions in broker_positions.items():
                for position in positions:
                    position_id = self._generate_position_id(position.get('symbol', ''), broker_id)
                    flat_broker_positions[position_id] = position
            
            # Find discrepancies
            missing_from_broker = set(self.internal_positions.keys()) - set(flat_broker_positions.keys())
            missing_from_internal = set(flat_broker_positions.keys()) - set(self.internal_positions.keys())
            common_positions = set(self.internal_positions.keys()) & set(flat_broker_positions.keys())
            
            # Check for quantity mismatches in common positions
            quantity_mismatches = []
            for position_id in common_positions:
                internal_qty = self.internal_positions[position_id].get('quantity', 0)
                broker_qty = flat_broker_positions[position_id].get('quantity', 0)
                
                # Compare quantities with tolerance for floating point
                if abs(internal_qty - broker_qty) > 0.001:
                    quantity_mismatches.append({
                        'position_id': position_id,
                        'internal_quantity': internal_qty,
                        'broker_quantity': broker_qty
                    })
            
            # Log reconciliation results
            if not missing_from_broker and not missing_from_internal and not quantity_mismatches:
                logger.info("Position reconciliation successful - all positions match")
                return {
                    "success": True,
                    "timestamp": self.last_reconciliation.isoformat(),
                    "positions_reconciled": len(common_positions)
                }
            
            # Handle discrepancies if auto-reconcile is enabled
            if self.auto_reconcile:
                # Positions missing from internal tracking that exist at broker
                for position_id in missing_from_internal:
                    broker_position = flat_broker_positions[position_id]
                    logger.warning(f"Found position at broker that was missing internally: {position_id}")
                    self.internal_positions[position_id] = broker_position
                
                # Positions missing from broker that exist in internal tracking
                for position_id in missing_from_broker:
                    logger.warning(f"Position exists internally but not at broker: {position_id}")
                    # Mark as closed or adjust internal state
                    if self.trade_logger:
                        symbol, broker = self._parse_position_id(position_id)
                        self.trade_logger._update_positions({
                            "symbol": symbol,
                            "broker_id": broker,
                            "action": "close_position",
                            "timestamp": datetime.now().isoformat()
                        })
                    # Remove from internal tracking
                    self.internal_positions.pop(position_id, None)
                
                # Handle quantity mismatches
                for mismatch in quantity_mismatches:
                    position_id = mismatch['position_id']
                    broker_qty = mismatch['broker_quantity']
                    
                    logger.warning(f"Quantity mismatch for {position_id}: " +
                                  f"internal={mismatch['internal_quantity']}, broker={broker_qty}")
                    
                    # Update internal quantity to match broker
                    self.internal_positions[position_id]['quantity'] = broker_qty
            
            # Record reconciliation errors for later analysis
            if missing_from_broker or missing_from_internal or quantity_mismatches:
                error = {
                    "timestamp": self.last_reconciliation.isoformat(),
                    "missing_from_broker": list(missing_from_broker),
                    "missing_from_internal": list(missing_from_internal),
                    "quantity_mismatches": quantity_mismatches
                }
                self.reconciliation_errors.append(error)
                
                # Keep only the last 100 errors
                if len(self.reconciliation_errors) > 100:
                    self.reconciliation_errors = self.reconciliation_errors[-100:]
            
            return {
                "success": False if (missing_from_broker or missing_from_internal or quantity_mismatches) else True,
                "timestamp": self.last_reconciliation.isoformat(),
                "missing_from_broker": list(missing_from_broker),
                "missing_from_internal": list(missing_from_internal),
                "quantity_mismatches": quantity_mismatches,
                "auto_reconciled": self.auto_reconcile
            }
    
    def add_position(self, position_data: Dict[str, Any]) -> str:
        """
        Add a new position to internal tracking.
        
        Args:
            position_data: Position data including symbol, quantity, etc.
            
        Returns:
            position_id: ID of the newly added position
        """
        with self._lock:
            symbol = position_data.get('symbol')
            broker_id = position_data.get('broker_id', 'unknown')
            
            if not symbol:
                raise ValueError("Symbol is required for position tracking")
            
            position_id = self._generate_position_id(symbol, broker_id)
            
            # Check if position already exists
            if position_id in self.internal_positions:
                logger.warning(f"Position {position_id} already exists, updating instead")
                return self.update_position(position_id, position_data)
            
            # Add timestamp if not provided
            if 'entry_date' not in position_data:
                position_data['entry_date'] = datetime.now().isoformat()
            
            # Store position
            self.internal_positions[position_id] = position_data
            
            # Record in history
            self._record_position_history(position_id, 'opened', position_data)
            
            logger.info(f"Added new position: {position_id}")
            return position_id
    
    def update_position(self, position_id: str, position_data: Dict[str, Any]) -> str:
        """
        Update an existing position.
        
        Args:
            position_id: ID of the position to update
            position_data: New position data
            
        Returns:
            position_id: ID of the updated position
        """
        with self._lock:
            if position_id not in self.internal_positions:
                raise ValueError(f"Position {position_id} not found")
            
            # Update fields
            for key, value in position_data.items():
                self.internal_positions[position_id][key] = value
            
            # Record in history
            self._record_position_history(position_id, 'updated', self.internal_positions[position_id])
            
            logger.info(f"Updated position: {position_id}")
            return position_id
    
    def close_position(self, position_id: str, close_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            position_id: ID of the position to close
            close_data: Data about the close transaction
            
        Returns:
            Dict with close result
        """
        with self._lock:
            if position_id not in self.internal_positions:
                raise ValueError(f"Position {position_id} not found")
            
            position = self.internal_positions[position_id]
            
            # Update position with close data
            position['status'] = 'closed'
            position['exit_price'] = close_data.get('price')
            position['exit_date'] = close_data.get('timestamp', datetime.now().isoformat())
            
            # Calculate P&L if not provided
            if 'pnl' not in close_data and 'entry_price' in position and 'exit_price' in position:
                entry_price = float(position['entry_price'])
                exit_price = float(position['exit_price'])
                quantity = float(position['quantity'])
                
                # Direction matters for P&L calculation
                direction = position.get('direction', 'long').lower()
                if direction == 'long':
                    pnl = (exit_price - entry_price) * quantity
                else:  # short
                    pnl = (entry_price - exit_price) * quantity
                
                position['pnl'] = pnl
                
                # Calculate P&L percent
                if entry_price > 0:
                    position['pnl_percent'] = (pnl / (entry_price * quantity)) * 100
            else:
                position['pnl'] = close_data.get('pnl')
                position['pnl_percent'] = close_data.get('pnl_percent')
            
            # Record in history
            self._record_position_history(position_id, 'closed', position)
            
            # Remove from active positions
            closed_position = self.internal_positions.pop(position_id)
            
            logger.info(f"Closed position {position_id} with P&L: {position.get('pnl', 'unknown')}")
            return closed_position
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific position.
        
        Args:
            position_id: ID of the position to retrieve
            
        Returns:
            Position details or None if not found
        """
        return self.internal_positions.get(position_id)
    
    def get_positions(self, broker_id: Optional[str] = None, 
                     symbol: Optional[str] = None,
                     status: str = 'open') -> List[Dict[str, Any]]:
        """
        Get positions with optional filtering.
        
        Args:
            broker_id: Filter by broker
            symbol: Filter by symbol
            status: Filter by status ('open', 'closed', 'all')
            
        Returns:
            List of positions matching the criteria
        """
        positions = list(self.internal_positions.values())
        
        # Apply filters
        if broker_id:
            positions = [p for p in positions if p.get('broker_id') == broker_id]
            
        if symbol:
            positions = [p for p in positions if p.get('symbol') == symbol]
            
        if status != 'all':
            positions = [p for p in positions if p.get('status', 'open') == status]
            
        return positions
    
    def _generate_position_id(self, symbol: str, broker_id: str) -> str:
        """Generate a unique position ID from symbol and broker ID."""
        return f"{symbol.upper()}-{broker_id}"
    
    def _parse_position_id(self, position_id: str) -> Tuple[str, str]:
        """Parse a position ID into symbol and broker ID."""
        parts = position_id.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid position ID format: {position_id}")
        return parts[0], parts[1]
    
    def _record_position_history(self, position_id: str, action: str, position_data: Dict[str, Any]):
        """Record position action in history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'position_id': position_id,
            'action': action,
            'position_data': position_data.copy()
        }
        
        self.position_history.append(history_entry)
        
        # Keep history from growing too large (last 1000 entries)
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
            
        # Optionally save to file
        if self.position_history_file:
            try:
                with open(self.position_history_file, 'w') as f:
                    json.dump(self.position_history[-100:], f, indent=2)  # Save last 100 entries
            except Exception as e:
                logger.error(f"Error saving position history: {str(e)}")
    
    def get_reconciliation_errors(self) -> List[Dict[str, Any]]:
        """Get list of recent reconciliation errors."""
        return self.reconciliation_errors.copy()
    
    def get_position_history(self, position_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get position history with optional filtering.
        
        Args:
            position_id: Optional position ID to filter by
            
        Returns:
            List of position history entries
        """
        if position_id:
            return [entry for entry in self.position_history if entry['position_id'] == position_id]
        return self.position_history.copy()
