"""
Portfolio Tracker

Provides cross-broker position reconciliation, consolidated portfolio views,
and unified risk management across multiple broker accounts.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
import threading
import json
import copy
import pandas as pd
import numpy as np

from trading_bot.event_system import EventBus, Event
from trading_bot.brokers.broker_interface import (
    BrokerInterface, BrokerAccount, Position, Order, Quote,
    AssetType, OrderType, OrderStatus
)

# Configure logging
logger = logging.getLogger(__name__)


class PositionKey:
    """
    Unique identifier for a position across brokers
    
    Positions are uniquely identified by a combination of:
    - Symbol
    - Asset type
    - Option details (if applicable)
    - Account ID (optional, for separating accounts)
    """
    
    def __init__(self, symbol: str, asset_type: str,
               option_type: Optional[str] = None,
               option_expiry: Optional[str] = None,
               option_strike: Optional[float] = None,
               account_id: Optional[str] = None):
        """
        Initialize a position key
        
        Args:
            symbol: Asset symbol
            asset_type: Type of asset (stock, option, crypto, etc.)
            option_type: For options, the type (call, put)
            option_expiry: For options, the expiration date
            option_strike: For options, the strike price
            account_id: Optional account identifier
        """
        self.symbol = symbol.upper()
        self.asset_type = asset_type.lower()
        self.option_type = option_type.lower() if option_type else None
        self.option_expiry = option_expiry
        self.option_strike = option_strike
        self.account_id = account_id
    
    def __eq__(self, other: Any) -> bool:
        """Check if two position keys are equal"""
        if not isinstance(other, PositionKey):
            return False
        
        return (
            self.symbol == other.symbol and
            self.asset_type == other.asset_type and
            self.option_type == other.option_type and
            self.option_expiry == other.option_expiry and
            self.option_strike == other.option_strike and
            self.account_id == other.account_id
        )
    
    def __hash__(self) -> int:
        """Generate hash for position key"""
        return hash((
            self.symbol,
            self.asset_type,
            self.option_type,
            self.option_expiry,
            self.option_strike,
            self.account_id
        ))
    
    def __str__(self) -> str:
        """String representation of position key"""
        if self.asset_type == 'option':
            return f"{self.symbol} {self.option_type} {self.option_strike} {self.option_expiry}"
        else:
            return f"{self.symbol} ({self.asset_type})"
    
    @staticmethod
    def from_position(position: Dict[str, Any], account_id: Optional[str] = None) -> 'PositionKey':
        """
        Create a position key from a position dictionary
        
        Args:
            position: Position data
            account_id: Optional account identifier
            
        Returns:
            PositionKey: The position key
        """
        symbol = position.get('symbol', '').upper()
        asset_type = position.get('asset_type', 'stock').lower()
        
        option_type = None
        option_expiry = None
        option_strike = None
        
        if asset_type == 'option':
            option_type = position.get('option_type')
            option_expiry = position.get('expiration_date')
            option_strike = position.get('strike_price')
        
        return PositionKey(
            symbol=symbol,
            asset_type=asset_type,
            option_type=option_type,
            option_expiry=option_expiry,
            option_strike=option_strike,
            account_id=account_id or position.get('account_id')
        )


class ConsolidatedPosition:
    """
    Represents a position consolidated across multiple brokers
    """
    
    def __init__(self, position_key: PositionKey):
        """
        Initialize a consolidated position
        
        Args:
            position_key: The position key
        """
        self.key = position_key
        self.broker_positions = {}  # broker_id -> position data
        self.last_updated = datetime.now()
    
    def add_broker_position(self, broker_id: str, position: Dict[str, Any]) -> None:
        """
        Add or update a broker's position
        
        Args:
            broker_id: Broker identifier
            position: Position data
        """
        self.broker_positions[broker_id] = copy.deepcopy(position)
        self.last_updated = datetime.now()
    
    def remove_broker_position(self, broker_id: str) -> None:
        """
        Remove a broker's position
        
        Args:
            broker_id: Broker identifier
        """
        if broker_id in self.broker_positions:
            del self.broker_positions[broker_id]
            self.last_updated = datetime.now()
    
    def get_total_quantity(self) -> float:
        """
        Get total quantity across all brokers
        
        Returns:
            float: Total position quantity
        """
        return sum(pos.get('quantity', 0) for pos in self.broker_positions.values())
    
    def get_weighted_cost_basis(self) -> float:
        """
        Get weighted average cost basis across all brokers
        
        Returns:
            float: Weighted average cost basis
        """
        total_quantity = self.get_total_quantity()
        if total_quantity == 0:
            return 0.0
        
        weighted_sum = sum(
            pos.get('quantity', 0) * pos.get('cost_basis', 0)
            for pos in self.broker_positions.values()
        )
        
        return weighted_sum / total_quantity
    
    def get_total_market_value(self) -> float:
        """
        Get total market value across all brokers
        
        Returns:
            float: Total market value
        """
        return sum(pos.get('market_value', 0) for pos in self.broker_positions.values())
    
    def get_total_unrealized_pnl(self) -> float:
        """
        Get total unrealized P&L across all brokers
        
        Returns:
            float: Total unrealized P&L
        """
        return sum(pos.get('unrealized_pnl', 0) for pos in self.broker_positions.values())
    
    def get_consolidated_data(self) -> Dict[str, Any]:
        """
        Get consolidated position data
        
        Returns:
            Dict: Consolidated position data
        """
        total_quantity = self.get_total_quantity()
        weighted_cost_basis = self.get_weighted_cost_basis()
        total_market_value = self.get_total_market_value()
        total_unrealized_pnl = self.get_total_unrealized_pnl()
        
        return {
            'symbol': self.key.symbol,
            'asset_type': self.key.asset_type,
            'option_type': self.key.option_type,
            'option_expiry': self.key.option_expiry,
            'option_strike': self.key.option_strike,
            'account_id': self.key.account_id,
            'quantity': total_quantity,
            'cost_basis': weighted_cost_basis,
            'market_value': total_market_value,
            'unrealized_pnl': total_unrealized_pnl,
            'broker_count': len(self.broker_positions),
            'brokers': list(self.broker_positions.keys()),
            'last_updated': self.last_updated.isoformat()
        }


class PortfolioTracker:
    """
    Tracks and reconciles positions across multiple brokers
    
    Features:
    - Consolidated position view
    - Cross-broker risk calculation
    - Position reconciliation
    - Exposure monitoring
    """
    
    def __init__(self):
        """Initialize the portfolio tracker"""
        self.positions = {}  # PositionKey -> ConsolidatedPosition
        self.broker_accounts = {}  # broker_id -> account_info
        
        # Risk limits
        self.risk_limits = {
            'max_position_size': {},  # symbol -> max size
            'max_total_exposure': 0.0,  # max total exposure
            'max_sector_exposure': {},  # sector -> max exposure
            'max_asset_type_exposure': {},  # asset_type -> max exposure
        }
        
        # Event bus for notifications
        self.event_bus = EventBus()
        
        # For thread safety
        self._lock = threading.RLock()
        
        # Last update times
        self._last_position_update = {}  # broker_id -> last update time
        
        logger.info("Initialized PortfolioTracker")
    
    def update_positions(self, broker_id: str, positions: List[Dict[str, Any]]) -> None:
        """
        Update positions for a broker
        
        Args:
            broker_id: Broker identifier
            positions: List of position data
        """
        with self._lock:
            # Create a set of keys for new positions
            new_position_keys = set()
            
            # Process each position
            for position in positions:
                # Create position key
                key = PositionKey.from_position(position)
                new_position_keys.add(key)
                
                # Get or create consolidated position
                if key not in self.positions:
                    self.positions[key] = ConsolidatedPosition(key)
                
                # Update broker's position
                self.positions[key].add_broker_position(broker_id, position)
            
            # Remove positions that no longer exist for this broker
            for key, cons_pos in list(self.positions.items()):
                if broker_id in cons_pos.broker_positions and key not in new_position_keys:
                    cons_pos.remove_broker_position(broker_id)
                    
                    # Remove consolidated position if no brokers left
                    if not cons_pos.broker_positions:
                        del self.positions[key]
            
            # Update last update time
            self._last_position_update[broker_id] = datetime.now()
            
            # Check risk limits
            self._check_risk_limits()
            
            # Publish portfolio updated event
            self.event_bus.publish(Event(
                "portfolio_updated",
                {
                    "broker_id": broker_id,
                    "position_count": len(positions),
                    "timestamp": datetime.now().isoformat()
                }
            ))
    
    def update_account_info(self, broker_id: str, account_info: Dict[str, Any]) -> None:
        """
        Update account information for a broker
        
        Args:
            broker_id: Broker identifier
            account_info: Account information
        """
        with self._lock:
            self.broker_accounts[broker_id] = copy.deepcopy(account_info)
    
    def get_consolidated_positions(self) -> List[Dict[str, Any]]:
        """
        Get consolidated positions across all brokers
        
        Returns:
            List[Dict]: List of consolidated position data
        """
        with self._lock:
            return [pos.get_consolidated_data() for pos in self.positions.values()]
    
    def get_position_by_symbol(self, symbol: str, asset_type: str = 'stock') -> Optional[Dict[str, Any]]:
        """
        Get consolidated position for a symbol
        
        Args:
            symbol: Asset symbol
            asset_type: Asset type
            
        Returns:
            Optional[Dict]: Consolidated position data or None if not found
        """
        with self._lock:
            # Create position key (without option details)
            key = PositionKey(symbol=symbol, asset_type=asset_type)
            
            # Find matching positions
            matching_positions = []
            for pos_key, cons_pos in self.positions.items():
                if pos_key.symbol == key.symbol and pos_key.asset_type == key.asset_type:
                    matching_positions.append(cons_pos)
            
            if not matching_positions:
                return None
            
            # If only one position, return it
            if len(matching_positions) == 1:
                return matching_positions[0].get_consolidated_data()
            
            # If multiple positions (could be options with different strikes/expiries),
            # combine them - this is a simplification, handling options properly would
            # require more complex logic
            total_quantity = sum(pos.get_total_quantity() for pos in matching_positions)
            total_market_value = sum(pos.get_total_market_value() for pos in matching_positions)
            total_pnl = sum(pos.get_total_unrealized_pnl() for pos in matching_positions)
            
            return {
                'symbol': symbol,
                'asset_type': asset_type,
                'quantity': total_quantity,
                'market_value': total_market_value,
                'unrealized_pnl': total_pnl,
                'is_combined': True,
                'position_count': len(matching_positions)
            }
    
    def get_position_breakdown(self, symbol: str, asset_type: str = 'stock') -> Dict[str, Dict[str, Any]]:
        """
        Get position breakdown by broker
        
        Args:
            symbol: Asset symbol
            asset_type: Asset type
            
        Returns:
            Dict: Map of broker IDs to their position data
        """
        with self._lock:
            # Create position key (without option details)
            key = PositionKey(symbol=symbol, asset_type=asset_type)
            
            # Find matching positions
            breakdown = {}
            for pos_key, cons_pos in self.positions.items():
                if pos_key.symbol == key.symbol and pos_key.asset_type == key.asset_type:
                    for broker_id, pos_data in cons_pos.broker_positions.items():
                        if broker_id not in breakdown:
                            breakdown[broker_id] = []
                        breakdown[broker_id].append(copy.deepcopy(pos_data))
            
            return breakdown
    
    def get_total_portfolio_value(self) -> float:
        """
        Get total portfolio value across all brokers
        
        Returns:
            float: Total portfolio value
        """
        with self._lock:
            return sum(pos.get_total_market_value() for pos in self.positions.values())
    
    def get_total_portfolio_exposure(self) -> Dict[str, float]:
        """
        Get total portfolio exposure by asset type
        
        Returns:
            Dict: Map of asset types to exposure amounts
        """
        with self._lock:
            exposure = {}
            
            for pos in self.positions.values():
                asset_type = pos.key.asset_type
                market_value = pos.get_total_market_value()
                
                if asset_type not in exposure:
                    exposure[asset_type] = 0.0
                
                exposure[asset_type] += market_value
            
            return exposure
    
    def get_broker_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation by broker
        
        Returns:
            Dict: Map of broker IDs to portfolio percentage
        """
        with self._lock:
            total_value = self.get_total_portfolio_value()
            if total_value == 0:
                return {}
            
            broker_values = {}
            
            for pos in self.positions.values():
                for broker_id, pos_data in pos.broker_positions.items():
                    if broker_id not in broker_values:
                        broker_values[broker_id] = 0.0
                    
                    broker_values[broker_id] += pos_data.get('market_value', 0)
            
            return {
                broker_id: (value / total_value) * 100
                for broker_id, value in broker_values.items()
            }
    
    def set_risk_limit(self, limit_type: str, value: Any, key: Optional[str] = None) -> None:
        """
        Set a risk limit
        
        Args:
            limit_type: Type of limit (e.g., 'max_position_size', 'max_total_exposure')
            value: Limit value
            key: Optional key for specific limits (e.g., symbol, sector)
        """
        with self._lock:
            if limit_type == 'max_total_exposure':
                self.risk_limits['max_total_exposure'] = float(value)
            elif limit_type in ('max_position_size', 'max_sector_exposure', 'max_asset_type_exposure'):
                if key is None:
                    logger.error(f"Key required for risk limit type: {limit_type}")
                    return
                
                self.risk_limits[limit_type][key] = float(value)
            else:
                logger.error(f"Unknown risk limit type: {limit_type}")
                return
            
            # Check limits after update
            self._check_risk_limits()
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """
        Get all risk limits
        
        Returns:
            Dict: Risk limits
        """
        with self._lock:
            return copy.deepcopy(self.risk_limits)
    
    def get_broker_positions(self, broker_id: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific broker
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            List[Dict]: List of position data
        """
        with self._lock:
            broker_positions = []
            
            for pos in self.positions.values():
                if broker_id in pos.broker_positions:
                    broker_positions.append(pos.broker_positions[broker_id])
            
            return copy.deepcopy(broker_positions)
    
    def _check_risk_limits(self) -> None:
        """Check if any risk limits are breached and publish events if so"""
        # Check total exposure
        total_value = self.get_total_portfolio_value()
        max_total = self.risk_limits['max_total_exposure']
        
        if max_total > 0 and total_value > max_total:
            logger.warning(f"Risk limit breached: Total exposure {total_value:.2f} exceeds limit {max_total:.2f}")
            
            self.event_bus.publish(Event(
                "risk_limit_breach",
                {
                    "limit_type": "max_total_exposure",
                    "current_value": total_value,
                    "limit_value": max_total,
                    "percentage": (total_value / max_total) * 100,
                    "timestamp": datetime.now().isoformat()
                }
            ))
        
        # Check position size limits
        for pos in self.positions.values():
            symbol = pos.key.symbol
            market_value = pos.get_total_market_value()
            
            if symbol in self.risk_limits['max_position_size']:
                max_size = self.risk_limits['max_position_size'][symbol]
                
                if market_value > max_size:
                    logger.warning(f"Risk limit breached: Position {symbol} size {market_value:.2f} exceeds limit {max_size:.2f}")
                    
                    self.event_bus.publish(Event(
                        "risk_limit_breach",
                        {
                            "limit_type": "max_position_size",
                            "symbol": symbol,
                            "current_value": market_value,
                            "limit_value": max_size,
                            "percentage": (market_value / max_size) * 100,
                            "timestamp": datetime.now().isoformat()
                        }
                    ))
        
        # Check asset type exposure limits
        asset_exposure = self.get_total_portfolio_exposure()
        
        for asset_type, exposure in asset_exposure.items():
            if asset_type in self.risk_limits['max_asset_type_exposure']:
                max_exposure = self.risk_limits['max_asset_type_exposure'][asset_type]
                
                if exposure > max_exposure:
                    logger.warning(f"Risk limit breached: {asset_type} exposure {exposure:.2f} exceeds limit {max_exposure:.2f}")
                    
                    self.event_bus.publish(Event(
                        "risk_limit_breach",
                        {
                            "limit_type": "max_asset_type_exposure",
                            "asset_type": asset_type,
                            "current_value": exposure,
                            "limit_value": max_exposure,
                            "percentage": (exposure / max_exposure) * 100,
                            "timestamp": datetime.now().isoformat()
                        }
                    ))
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert consolidated positions to a pandas DataFrame
        
        Returns:
            pd.DataFrame: DataFrame of positions
        """
        with self._lock:
            positions = self.get_consolidated_positions()
            
            if not positions:
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=[
                    'symbol', 'asset_type', 'quantity', 'cost_basis',
                    'market_value', 'unrealized_pnl', 'broker_count'
                ])
            
            return pd.DataFrame(positions)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary
        
        Returns:
            Dict: Portfolio summary
        """
        with self._lock:
            positions = self.get_consolidated_positions()
            
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
            
            asset_types = {}
            for pos in positions:
                asset_type = pos.get('asset_type', 'unknown')
                if asset_type not in asset_types:
                    asset_types[asset_type] = 0.0
                asset_types[asset_type] += pos.get('market_value', 0)
            
            broker_allocation = self.get_broker_allocation()
            
            return {
                'position_count': len(positions),
                'total_value': total_value,
                'total_unrealized_pnl': total_pnl,
                'pnl_percentage': (total_pnl / total_value * 100) if total_value > 0 else 0.0,
                'asset_type_allocation': {
                    asset_type: (value / total_value * 100) if total_value > 0 else 0.0
                    for asset_type, value in asset_types.items()
                },
                'broker_allocation': broker_allocation,
                'broker_count': len(broker_allocation),
                'last_updated': datetime.now().isoformat()
            }
