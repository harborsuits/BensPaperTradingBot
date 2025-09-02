"""
Core interfaces for the trading bot system.

This module defines abstract base classes and interfaces that form
the foundation of the trading system's architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class DataProvider(ABC):
    """Interface for data sources that provide market data."""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None,
                          interval: str = "1d") -> Dict[str, Any]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Trading symbol or ticker
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Time interval (e.g., '1d', '1h')
            
        Returns:
            Dictionary containing historical price data
        """
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading symbol or ticker
            
        Returns:
            Latest price as a float
        """
        pass
    
    @abstractmethod
    def get_multiple_symbols(self, symbols: List[str], interval: str = "1d") -> Dict[str, Dict[str, Any]]:
        """
        Get data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            interval: Time interval (e.g., '1d', '1h')
            
        Returns:
            Dictionary mapping symbols to their data
        """
        pass


class IndicatorInterface(ABC):
    """Interface for technical indicators and analytics."""
    
    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate indicator values.
        
        Args:
            data: Price/volume data to analyze
            
        Returns:
            Dictionary containing calculated indicator values
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get indicator parameters.
        
        Returns:
            Dictionary of current parameter values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set indicator parameters.
        
        Args:
            parameters: Dictionary of parameter values to set
        """
        pass


class StrategyInterface(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Dictionary containing trading signals
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of current parameter values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dictionary of parameter values to set
        """
        pass


class SignalInterface(ABC):
    """Interface for trading signals."""
    
    @abstractmethod
    def get_signal_type(self) -> str:
        """
        Get the type of the signal (buy, sell, etc.).
        
        Returns:
            Signal type as a string
        """
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """
        Get the confidence level of the signal.
        
        Returns:
            Confidence as a float (typically 0-1)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get additional signal metadata.
        
        Returns:
            Dictionary containing signal metadata
        """
        pass

class RiskManager(ABC):
    """Interface for risk management."""
    
    @abstractmethod
    def validate_signal(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a trading signal against risk management rules.
        
        Args:
            signal: Trading signal to validate
            portfolio: Current portfolio state
            
        Returns:
            Validated signal (possibly modified) or None if rejected
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> int:
        """
        Calculate appropriate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            Position size (number of contracts/shares)
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall portfolio risk metrics.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dictionary of risk metrics
        """
        pass

class OrderManager(ABC):
    """Interface for order management."""
    
    @abstractmethod
    def place_order(self, order: Dict[str, Any]) -> str:
        """
        Place an order with the broker.
        
        Args:
            order: Order details
            
        Returns:
            Order ID
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            Order status details
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Returns:
            List of open orders
        """
        pass

class PortfolioManager(ABC):
    """Interface for portfolio management."""
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dictionary of positions
        """
        pass
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.
        
        Returns:
            Portfolio value
        """
        pass
    
    @abstractmethod
    def update_portfolio(self, transaction: Dict[str, Any]) -> None:
        """
        Update portfolio based on a transaction.
        
        Args:
            transaction: Transaction details
        """
        pass
    
    @abstractmethod
    def get_portfolio_history(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get portfolio history for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Portfolio history data
        """
        pass

class NotificationManager(ABC):
    """Interface for sending notifications."""
    
    @abstractmethod
    def send_notification(self, message: str, level: str = "info", 
                        category: str = "general") -> bool:
        """
        Send a notification.
        
        Args:
            message: Notification message
            level: Notification level (info, warning, error)
            category: Notification category
            
        Returns:
            True if successful, False otherwise
        """
        pass 