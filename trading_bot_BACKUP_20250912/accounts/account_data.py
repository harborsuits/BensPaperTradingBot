"""
Account Data Module

This module provides classes and functions for managing account data,
including balances, positions, and account metrics.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AccountData:
    """
    Class for managing and providing access to account data.
    
    This class handles retrieval, storage, and analysis of account-related
    information, including balances, positions, and performance metrics.
    
    Attributes:
        account_id: Unique identifier for the account
        broker: Broker name or identifier
        initial_balance: Starting account balance
        current_balance: Current account balance
        positions: Dictionary of current positions
        cash_available: Available cash for trading
        margin_available: Available margin for trading
        buying_power: Total buying power (cash + margin)
    """
    
    def __init__(self, account_id: str = "", broker: str = "default", 
                initial_balance: float = 100000.0):
        """
        Initialize the AccountData object.
        
        Args:
            account_id: Unique identifier for the account
            broker: Broker name or identifier
            initial_balance: Starting account balance
        """
        self.account_id = account_id
        self.broker = broker
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.cash_available = initial_balance
        self.margin_available = 0.0
        self.buying_power = initial_balance
        self.transactions = []
        
        logger.info(f"Initialized AccountData for account {account_id} with {initial_balance} balance")
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get a summary of account information.
        
        Returns:
            Dictionary containing account summary data
        """
        return {
            'account_id': self.account_id,
            'broker': self.broker,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'cash_available': self.cash_available,
            'margin_available': self.margin_available,
            'buying_power': self.buying_power,
            'num_positions': len(self.positions),
            'portfolio_value': self.get_portfolio_value(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbols to position details
        """
        return self.positions
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a specific symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position details dictionary or None if not found
        """
        return self.positions.get(symbol)
    
    def add_position(self, symbol: str, quantity: int, price: float, 
                    position_type: str = "long", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new position or update an existing one.
        
        Args:
            symbol: Symbol for the position
            quantity: Number of shares or contracts
            price: Entry price per share/contract
            position_type: Type of position (long, short, option, etc.)
            metadata: Additional position data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            position_cost = quantity * price
            
            # Check if we have enough cash
            if position_cost > self.cash_available:
                logger.warning(f"Insufficient funds to add position in {symbol}")
                return False
                
            # Update or create position
            if symbol in self.positions:
                # Update existing position (average in)
                existing = self.positions[symbol]
                total_shares = existing['quantity'] + quantity
                total_cost = (existing['quantity'] * existing['price']) + (quantity * price)
                avg_price = total_cost / total_shares if total_shares != 0 else 0
                
                self.positions[symbol]['quantity'] = total_shares
                self.positions[symbol]['price'] = avg_price
                self.positions[symbol]['last_updated'] = datetime.now().isoformat()
                
                if metadata:
                    self.positions[symbol]['metadata'].update(metadata)
            else:
                # Create new position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'position_type': position_type,
                    'entry_date': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'metadata': metadata or {}
                }
                
            # Update cash and record transaction
            self.cash_available -= position_cost
            self.buying_power = self.cash_available + self.margin_available
            
            # Record transaction
            self.transactions.append({
                'type': 'buy',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'position_type': position_type
            })
            
            logger.info(f"Added position in {symbol}: {quantity} @ {price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position in {symbol}: {e}")
            return False
    
    def remove_position(self, symbol: str, quantity: Optional[int] = None, 
                       price: float = 0.0) -> bool:
        """
        Remove or reduce a position.
        
        Args:
            symbol: Symbol for the position to remove
            quantity: Number of shares/contracts to remove (None for all)
            price: Exit price per share/contract
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
                
            position = self.positions[symbol]
            
            # Default to closing the entire position
            if quantity is None:
                quantity = position['quantity']
                
            # Validate quantity
            if quantity > position['quantity']:
                logger.warning(f"Cannot remove {quantity} from position with only {position['quantity']} shares")
                return False
                
            # Calculate sale proceeds
            proceeds = quantity * price
            
            # Update position
            if quantity == position['quantity']:
                # Close entire position
                del self.positions[symbol]
            else:
                # Reduce position
                self.positions[symbol]['quantity'] -= quantity
                self.positions[symbol]['last_updated'] = datetime.now().isoformat()
            
            # Update cash and record transaction
            self.cash_available += proceeds
            self.buying_power = self.cash_available + self.margin_available
            
            # Record transaction
            self.transactions.append({
                'type': 'sell',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Removed {quantity} shares of {symbol} @ {price}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing position in {symbol}: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Returns:
            Total portfolio value
        """
        # In a real implementation, this would use current market prices
        # For now, just use the recorded position prices
        position_value = sum(
            pos['quantity'] * pos['price'] 
            for pos in self.positions.values()
        )
        
        return self.cash_available + position_value
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation as percentages.
        
        Returns:
            Dictionary mapping symbols to their allocation percentage
        """
        total_value = self.get_portfolio_value()
        
        if total_value <= 0:
            return {'cash': 100.0}
            
        allocation = {
            symbol: (pos['quantity'] * pos['price'] / total_value * 100)
            for symbol, pos in self.positions.items()
        }
        
        # Add cash allocation
        allocation['cash'] = self.cash_available / total_value * 100
        
        return allocation
    
    def get_transaction_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get transaction history, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter transactions
            
        Returns:
            List of transaction records
        """
        if symbol:
            return [t for t in self.transactions if t['symbol'] == symbol]
        return self.transactions 