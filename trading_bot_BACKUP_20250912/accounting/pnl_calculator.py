"""
P&L Calculator

This module provides functions for calculating realized and unrealized
profit/loss for trading positions.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import sqlite3

from trading_bot.position.position_manager import PositionManager

logger = logging.getLogger(__name__)

class PnLCalculator:
    """
    Calculates realized and unrealized profit/loss for trading positions.
    """
    
    def __init__(self, 
                 database_path: str,
                 position_manager: Optional[PositionManager] = None):
        """
        Initialize the P&L calculator.
        
        Args:
            database_path: Path to SQLite database with trade records
            position_manager: Optional position manager for accessing current positions
        """
        self.database_path = database_path
        self.position_manager = position_manager
    
    def calculate_realized_pnl(self, 
                              period: Optional[str] = None, 
                              strategy_id: Optional[str] = None, 
                              broker_id: Optional[str] = None) -> float:
        """
        Calculate realized P&L for closed trades with optional filtering.
        
        Args:
            period: Time period ('day', 'week', 'month', 'year', 'all')
            strategy_id: Filter by strategy
            broker_id: Filter by broker
            
        Returns:
            Total realized P&L for the period
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = 'SELECT SUM(realized_pnl) FROM trades WHERE status = "closed"'
                params = []
                
                # Add strategy filter
                if strategy_id:
                    query += ' AND strategy_id = ?'
                    params.append(strategy_id)
                
                # Add broker filter
                if broker_id:
                    query += ' AND broker_id = ?'
                    params.append(broker_id)
                
                # Add period filter
                if period:
                    if period == 'day':
                        query += ' AND date(exit_time) = date("now")'
                    elif period == 'week':
                        query += ' AND date(exit_time) >= date("now", "-7 days")'
                    elif period == 'month':
                        query += ' AND date(exit_time) >= date("now", "-1 month")'
                    elif period == 'year':
                        query += ' AND date(exit_time) >= date("now", "-1 year")'
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                # Return total or 0 if no trades
                total_pnl = result[0] if result and result[0] is not None else 0
                return float(total_pnl)
                
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {str(e)}")
            return 0.0
    
    def calculate_unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L for open positions.
        
        Uses position_manager to get current positions and
        fetches current market prices to determine unrealized value.
        
        Returns:
            Total unrealized P&L
        """
        if not self.position_manager:
            logger.warning("Cannot calculate unrealized P&L without position manager")
            return 0.0
            
        try:
            # Get all open positions
            positions = self.position_manager.get_positions(status='open')
            if not positions:
                return 0.0
            
            total_unrealized_pnl = 0.0
            
            for position in positions:
                symbol = position.get('symbol')
                broker_id = position.get('broker_id', 'unknown')
                quantity = float(position.get('quantity', 0))
                entry_price = float(position.get('entry_price', 0))
                direction = position.get('direction', 'long').lower()
                
                # Try to get current price from broker manager
                try:
                    if hasattr(self.position_manager, 'broker_manager'):
                        broker = self.position_manager.broker_manager.brokers.get(broker_id)
                        if broker:
                            quote = broker.get_quote(symbol)
                            current_price = float(quote.get('last', 0)) if quote else 0
                        else:
                            # Fallback to first available broker
                            for b_id, b in self.position_manager.broker_manager.brokers.items():
                                quote = b.get_quote(symbol)
                                if quote:
                                    current_price = float(quote.get('last', 0))
                                    break
                            else:
                                logger.warning(f"Could not get current price for {symbol}")
                                continue
                    else:
                        # No broker manager available, skip
                        logger.warning("No broker manager available to get current prices")
                        continue
                except Exception as e:
                    logger.error(f"Error getting current price for {symbol}: {str(e)}")
                    continue
                
                # Calculate unrealized P&L
                if direction == 'long':
                    position_pnl = (current_price - entry_price) * quantity
                else:  # short
                    position_pnl = (entry_price - current_price) * quantity
                
                total_unrealized_pnl += position_pnl
            
            return total_unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {str(e)}")
            return 0.0
    
    def calculate_position_pnl(self, position: Dict[str, Any]) -> float:
        """
        Calculate P&L for a single position.
        
        Args:
            position: Position data dictionary
            
        Returns:
            Position P&L (unrealized if open, realized if closed)
        """
        try:
            status = position.get('status', 'open')
            quantity = float(position.get('quantity', 0))
            entry_price = float(position.get('entry_price', 0))
            direction = position.get('direction', 'long').lower()
            
            if status == 'closed':
                # Realized P&L
                exit_price = float(position.get('exit_price', 0))
                
                if direction == 'long':
                    pnl = (exit_price - entry_price) * quantity
                else:  # short
                    pnl = (entry_price - exit_price) * quantity
                    
                # Subtract commission if available
                pnl -= float(position.get('commission', 0))
                
                return pnl
            else:
                # Unrealized P&L - need current price
                symbol = position.get('symbol')
                broker_id = position.get('broker_id', 'unknown')
                
                # Try to get current price
                if self.position_manager and hasattr(self.position_manager, 'broker_manager'):
                    try:
                        broker = self.position_manager.broker_manager.brokers.get(broker_id)
                        if broker:
                            quote = broker.get_quote(symbol)
                            current_price = float(quote.get('last', 0)) if quote else 0
                        else:
                            # No broker, use 0 price
                            logger.warning(f"No broker found for {broker_id}")
                            return 0.0
                            
                        if direction == 'long':
                            pnl = (current_price - entry_price) * quantity
                        else:  # short
                            pnl = (entry_price - current_price) * quantity
                            
                        return pnl
                    except Exception as e:
                        logger.error(f"Error calculating position P&L: {str(e)}")
                        return 0.0
                else:
                    # No position manager or broker manager
                    logger.warning("Cannot calculate unrealized P&L without broker access")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Error calculating position P&L: {str(e)}")
            return 0.0
            
    def calculate_daily_pnl(self, date: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate daily P&L for a specific date.
        
        Args:
            date: Date string in format 'YYYY-MM-DD' (defaults to today)
            
        Returns:
            Dict with realized and unrealized P&L
        """
        date_str = date or datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get realized P&L for the day
                cursor.execute("""
                SELECT SUM(realized_pnl) FROM trades 
                WHERE status = 'closed' AND date(exit_time) = ?
                """, (date_str,))
                
                result = cursor.fetchone()
                realized_pnl = result[0] if result and result[0] is not None else 0
                
                # Get unrealized P&L if date is today
                unrealized_pnl = 0.0
                if date_str == datetime.now().strftime('%Y-%m-%d'):
                    unrealized_pnl = self.calculate_unrealized_pnl()
                
                return {
                    'date': date_str,
                    'realized_pnl': realized_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'total_pnl': realized_pnl + unrealized_pnl
                }
                
        except Exception as e:
            logger.error(f"Error calculating daily P&L for {date_str}: {str(e)}")
            return {
                'date': date_str,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'total_pnl': 0.0,
                'error': str(e)
            }
