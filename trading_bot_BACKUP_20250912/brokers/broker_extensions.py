"""
Broker Extensions

Defines interfaces and implementations for broker-specific features that extend
beyond the standard BrokerInterface. Each broker may have unique capabilities
that can be leveraged by strategies when available.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.models.broker_models import (
    AssetType, OrderType, TimeInForce, Quote, Bar
)

# Configure logging
logger = logging.getLogger(__name__)


class BrokerExtension(ABC):
    """Base class for all broker-specific extensions"""
    
    @abstractmethod
    def get_extension_name(self) -> str:
        """Get the name of this extension"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Set[str]:
        """
        Get the set of capabilities provided by this extension
        
        Returns:
            Set[str]: Set of capability IDs
        """
        pass


class StreamingDataExtension(BrokerExtension):
    """Extension for brokers that support real-time data streaming"""
    
    @abstractmethod
    def subscribe_to_quotes(self, symbols: List[str], callback: callable) -> bool:
        """
        Subscribe to real-time quote updates for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call with each update
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def subscribe_to_bars(self, symbols: List[str], timeframe: str, callback: callable) -> bool:
        """
        Subscribe to real-time bar updates for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            timeframe: Bar timeframe (e.g., "1M", "5M", "1H")
            callback: Function to call with each update
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def unsubscribe_all(self) -> bool:
        """
        Unsubscribe from all streaming data
        
        Returns:
            bool: Success status
        """
        pass
    
    def get_extension_name(self) -> str:
        return "StreamingDataExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"streaming_quotes", "streaming_bars"}


class AdvancedOptionsExtension(BrokerExtension):
    """Extension for brokers with advanced options trading capabilities"""
    
    @abstractmethod
    def get_option_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get the full option chain for a symbol
        
        Args:
            symbol: Underlying symbol
            expiration_date: Specific expiration date (optional)
            
        Returns:
            Dict: Option chain data
        """
        pass
    
    @abstractmethod
    def get_option_expiration_dates(self, symbol: str) -> List[datetime]:
        """
        Get available expiration dates for options on a symbol
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            List[datetime]: Available expiration dates
        """
        pass
    
    @abstractmethod
    def get_option_strikes(self, symbol: str, expiration_date: datetime) -> List[float]:
        """
        Get available strike prices for a symbol and expiration
        
        Args:
            symbol: Underlying symbol
            expiration_date: Option expiration date
            
        Returns:
            List[float]: Available strike prices
        """
        pass
    
    @abstractmethod
    def create_option_spread(self, 
                           symbol: str,
                           spread_type: str,  # e.g., "vertical", "iron_condor", "butterfly"
                           expiration_date: datetime,
                           width: float,      # Distance between strikes
                           is_bullish: bool,  # Direction of the spread
                           quantity: int,
                           limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a multi-leg option spread order
        
        Args:
            symbol: Underlying symbol
            spread_type: Type of spread to create
            expiration_date: Option expiration date
            width: Width between strikes
            is_bullish: True for bullish spread, False for bearish
            quantity: Number of spreads to trade
            limit_price: Optional limit price for the entire spread
            
        Returns:
            Dict: Order information
        """
        pass
    
    def get_extension_name(self) -> str:
        return "AdvancedOptionsExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"option_chains", "option_spreads", "option_analytics"}


class TechnicalIndicatorExtension(BrokerExtension):
    """Extension for brokers that provide technical indicators"""
    
    @abstractmethod
    def get_technical_indicator(self, 
                              symbol: str,
                              indicator: str,   # e.g., "SMA", "RSI", "MACD"
                              timeframe: str,
                              params: Dict[str, Any],
                              start: datetime,
                              end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get technical indicator values calculated by the broker
        
        Args:
            symbol: Symbol to calculate indicator for
            indicator: Indicator name
            timeframe: Bar timeframe for calculation
            params: Indicator-specific parameters
            start: Start date
            end: End date (default: now)
            
        Returns:
            DataFrame: Indicator values with timestamps
        """
        pass
    
    @abstractmethod
    def get_available_indicators(self) -> List[str]:
        """
        Get list of available technical indicators from this broker
        
        Returns:
            List[str]: Available indicator names
        """
        pass
    
    def get_extension_name(self) -> str:
        return "TechnicalIndicatorExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"technical_indicators"}


class CryptoExtension(BrokerExtension):
    """Extension for brokers with advanced cryptocurrency capabilities"""
    
    @abstractmethod
    def get_crypto_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get order book data for a cryptocurrency
        
        Args:
            symbol: Crypto symbol (e.g., "BTC/USD")
            depth: Depth of the order book to return
            
        Returns:
            Dict: Order book with bids and asks
        """
        pass
    
    @abstractmethod
    def get_crypto_trading_pairs(self) -> List[str]:
        """
        Get available cryptocurrency trading pairs
        
        Returns:
            List[str]: Available trading pairs
        """
        pass
    
    @abstractmethod
    def get_crypto_account_history(self, 
                                 symbol: Optional[str] = None,
                                 start: Optional[datetime] = None,
                                 end: Optional[datetime] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get crypto account history (deposits, withdrawals, transfers)
        
        Args:
            symbol: Optional symbol to filter by
            start: Start date
            end: End date
            limit: Maximum records to return
            
        Returns:
            List[Dict]: Account history records
        """
        pass
    
    def get_extension_name(self) -> str:
        return "CryptoExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"crypto_orderbook", "crypto_pairs", "crypto_history"}


class FuturesExtension(BrokerExtension):
    """Extension for brokers with advanced futures trading capabilities"""
    
    @abstractmethod
    def get_futures_contracts(self, root_symbol: str) -> List[Dict[str, Any]]:
        """
        Get available futures contracts for a symbol
        
        Args:
            root_symbol: Root futures symbol (e.g., "ES", "CL")
            
        Returns:
            List[Dict]: Available contracts with details
        """
        pass
    
    @abstractmethod
    def get_futures_margin_requirements(self, symbol: str) -> Dict[str, float]:
        """
        Get margin requirements for a futures contract
        
        Args:
            symbol: Full futures contract symbol
            
        Returns:
            Dict: Margin requirements (initial, maintenance)
        """
        pass
    
    @abstractmethod
    def get_futures_roll_dates(self, root_symbol: str) -> Dict[str, datetime]:
        """
        Get contract roll dates for a futures symbol
        
        Args:
            root_symbol: Root futures symbol
            
        Returns:
            Dict: Contract symbols mapped to roll dates
        """
        pass
    
    def get_extension_name(self) -> str:
        return "FuturesExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"futures_contracts", "futures_margins", "futures_roll_dates"}


class PortfolioAnalysisExtension(BrokerExtension):
    """Extension for brokers that provide portfolio analysis tools"""
    
    @abstractmethod
    def get_portfolio_risk_metrics(self) -> Dict[str, float]:
        """
        Get risk metrics for the current portfolio
        
        Returns:
            Dict: Risk metrics (beta, VaR, etc.)
        """
        pass
    
    @abstractmethod
    def get_position_performance(self, 
                               symbol: Optional[str] = None, 
                               timeframe: str = "1D",
                               start: Optional[datetime] = None,
                               end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get performance metrics for positions
        
        Args:
            symbol: Optional symbol to filter by
            timeframe: Analysis timeframe
            start: Start date
            end: End date
            
        Returns:
            DataFrame: Performance metrics
        """
        pass
    
    @abstractmethod
    def get_portfolio_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for current portfolio holdings
        
        Returns:
            DataFrame: Correlation matrix
        """
        pass
    
    def get_extension_name(self) -> str:
        return "PortfolioAnalysisExtension"
    
    def get_capabilities(self) -> Set[str]:
        return {"portfolio_risk", "position_performance", "portfolio_correlation"}
