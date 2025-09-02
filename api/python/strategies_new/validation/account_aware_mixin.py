#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Account Aware Mixin

This module provides a mixin class that adds robust account awareness functionality
to trading strategies, ensuring they properly check for:
- Account balance requirements
- Regulatory constraints (PDT rule, etc.)
- Position sizing based on available capital
- Per-broker trading limitations
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class TradingAccountType(Enum):
    """Enumeration of trading account types and their regulatory requirements"""
    CASH = "cash"
    MARGIN = "margin"
    IRA = "ira"
    ROTH_IRA = "roth_ira"
    CRYPTO = "crypto"  # Dedicated crypto account
    FOREX = "forex"    # Dedicated forex account

class AccountAwareMixin:
    """
    Mixin class that adds account awareness capabilities to trading strategies.
    
    This class provides methods to check account status, verify regulatory
    compliance, and ensure proper position sizing based on available capital
    and risk management rules.
    """
    
    def __init__(self):
        """Initialize the account aware mixin."""
        # This will be initialized by the strategy using this mixin
        self.account_status = {}
        self.trading_platforms = {}
        self.recent_day_trades = {}
        
        # Default regulatory parameters
        self.regulatory_params = {
            'pdt_rule_min_equity': 25000.0,  # $25K minimum for pattern day trading
            'enforce_pdt_rule': True,        # Default to enforce PDT rule
            'max_leverage': 4.0,             # Maximum leverage for margin accounts
            'day_trade_buying_power': 4.0,   # Day trading buying power multiplier
            'overnight_buying_power': 2.0,   # Overnight buying power multiplier
            'pdt_max_day_trades': 3,         # Max day trades in 5 business days (below 25K)
            'pdt_lookback_days': 5,          # Lookback period for PDT rule
        }
        
        # Platform-specific parameters
        self.platform_params = {
            'alpaca': {
                'supports_fractional_shares': True,
                'min_order_size': 1.0,
                'enabled_for_paper': True,
                'enabled_for_live': True
            },
            'tradier': {
                'supports_fractional_shares': False,
                'min_order_size': 1.0,
                'enabled_for_paper': True,
                'enabled_for_live': True
            },
            'etrade': {
                'supports_fractional_shares': True,
                'min_order_size': 1.0,
                'enabled_for_paper': False,  # E*TRADE might not be enabled for paper
                'enabled_for_live': True
            }
        }
    
    def check_account_status(self) -> Dict[str, Any]:
        """
        Check current account status and update internal state.
        
        Returns:
            Dictionary with account status information
        """
        try:
            # This would be implemented by the strategy class to query the actual account
            if hasattr(self, 'session') and hasattr(self.session, 'get_account_status'):
                self.account_status = self.session.get_account_status()
            else:
                logger.warning("Session doesn't support get_account_status method")
                
            return self.account_status
        except Exception as e:
            logger.error(f"Error checking account status: {str(e)}")
            return {}
    
    def get_account_equity(self) -> float:
        """
        Get current account equity.
        
        Returns:
            Current account equity as float
        """
        try:
            # Refresh account status if needed
            if not self.account_status:
                self.check_account_status()
                
            return float(self.account_status.get('equity', 0.0))
        except Exception as e:
            logger.error(f"Error getting account equity: {str(e)}")
            return 0.0
    
    def get_buying_power(self, day_trade: bool = False) -> float:
        """
        Get current buying power, considering day trade status if applicable.
        
        Args:
            day_trade: Whether this is for a day trade (uses different multiplier)
            
        Returns:
            Available buying power as float
        """
        try:
            # Refresh account status if needed
            if not self.account_status:
                self.check_account_status()
                
            if day_trade:
                # Use day trade buying power if provided by broker
                if 'day_trade_buying_power' in self.account_status:
                    return float(self.account_status.get('day_trade_buying_power', 0.0))
                
                # Otherwise calculate based on equity and multiplier
                equity = self.get_account_equity()
                if equity >= self.regulatory_params['pdt_rule_min_equity']:
                    return equity * self.regulatory_params['day_trade_buying_power']
                else:
                    # Below PDT threshold - use overnight buying power
                    return equity * self.regulatory_params['overnight_buying_power']
            else:
                # Regular (overnight) buying power
                if 'buying_power' in self.account_status:
                    return float(self.account_status.get('buying_power', 0.0))
                
                # Calculate if not provided
                equity = self.get_account_equity()
                return equity * self.regulatory_params['overnight_buying_power']
        except Exception as e:
            logger.error(f"Error getting buying power: {str(e)}")
            return 0.0
    
    def get_account_type(self) -> TradingAccountType:
        """
        Get the type of trading account.
        
        Returns:
            TradingAccountType enum value
        """
        try:
            if not self.account_status:
                self.check_account_status()
                
            account_type_str = self.account_status.get('account_type', '').lower()
            
            # Map broker-specific account type strings to our enum
            if 'cash' in account_type_str:
                return TradingAccountType.CASH
            elif 'margin' in account_type_str:
                return TradingAccountType.MARGIN
            elif 'ira' in account_type_str and 'roth' in account_type_str:
                return TradingAccountType.ROTH_IRA
            elif 'ira' in account_type_str:
                return TradingAccountType.IRA
            elif 'crypto' in account_type_str:
                return TradingAccountType.CRYPTO
            elif 'forex' in account_type_str:
                return TradingAccountType.FOREX
            else:
                # Default to cash account (most conservative)
                return TradingAccountType.CASH
                
        except Exception as e:
            logger.error(f"Error determining account type: {str(e)}")
            # Default to cash account for safety
            return TradingAccountType.CASH
    
    def check_pdt_rule_compliance(self, is_day_trade: bool = True) -> bool:
        """
        Check if the account complies with Pattern Day Trader (PDT) rule.
        
        Args:
            is_day_trade: Whether the proposed trade is a day trade
            
        Returns:
            Boolean indicating whether the trade complies with PDT rule
        """
        try:
            # Skip check if PDT rule enforcement is disabled or it's not a day trade
            if not self.regulatory_params['enforce_pdt_rule'] or not is_day_trade:
                return True
                
            # Crypto and forex accounts are exempt from PDT rule
            account_type = self.get_account_type()
            if account_type in [TradingAccountType.CRYPTO, TradingAccountType.FOREX]:
                return True
                
            # Get account equity
            equity = self.get_account_equity()
            
            # Check if equity meets minimum requirement
            if equity >= self.regulatory_params['pdt_rule_min_equity']:
                logger.debug(f"Account equity (${equity:.2f}) exceeds PDT minimum")
                return True
                
            # Below PDT threshold - check day trade count
            day_trade_count = self.get_day_trade_count()
            
            if day_trade_count >= self.regulatory_params['pdt_max_day_trades']:
                logger.warning(f"Account equity (${equity:.2f}) is below PDT rule minimum (${self.regulatory_params['pdt_rule_min_equity']:.2f})")
                logger.warning(f"Day trade count ({day_trade_count}) has reached maximum ({self.regulatory_params['pdt_max_day_trades']})")
                logger.warning("Day trading restricted due to Pattern Day Trader (PDT) rule")
                return False
                
            # Below PDT threshold but haven't reached max day trades yet
            logger.info(f"Account below PDT threshold but has {day_trade_count} day trades out of {self.regulatory_params['pdt_max_day_trades']} allowed")
            logger.info(f"This day trade will leave {self.regulatory_params['pdt_max_day_trades'] - day_trade_count - 1} remaining for this 5-day period")
            return True
                
        except Exception as e:
            logger.error(f"Error checking PDT rule compliance: {str(e)}")
            # Default to not allowing day trading if we encounter an error
            return False
    
    def get_day_trade_count(self) -> int:
        """
        Get the number of day trades in the lookback period.
        
        Returns:
            Number of day trades in the lookback period
        """
        try:
            # This would typically query the broker API for recent day trades
            if hasattr(self, 'session') and hasattr(self.session, 'get_day_trade_count'):
                return self.session.get_day_trade_count()
            
            # Fallback to our internal tracking
            # Clear old day trades outside the lookback period
            lookback_start = datetime.now() - timedelta(days=self.regulatory_params['pdt_lookback_days'])
            self.recent_day_trades = {
                date_str: count for date_str, count in self.recent_day_trades.items()
                if datetime.strptime(date_str, '%Y-%m-%d') >= lookback_start
            }
            
            # Count total day trades in the period
            return sum(self.recent_day_trades.values())
        except Exception as e:
            logger.error(f"Error getting day trade count: {str(e)}")
            # If we can't determine count, assume it's at limit for safety
            return self.regulatory_params['pdt_max_day_trades']
            
    def record_day_trade(self):
        """Record a day trade in our internal tracking."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            self.recent_day_trades[today] = self.recent_day_trades.get(today, 0) + 1
            logger.info(f"Recorded day trade, new count for today: {self.recent_day_trades[today]}")
        except Exception as e:
            logger.error(f"Error recording day trade: {str(e)}")
            
    def calculate_max_position_size(self, 
                                    price: float, 
                                    is_day_trade: bool = False, 
                                    risk_percent: float = None) -> Tuple[float, float]:
        """
        Calculate the maximum position size based on account equity and risk parameters.
        
        Args:
            price: Current price of the security
            is_day_trade: Whether this is a day trade
            risk_percent: Maximum risk as percentage of account (overrides strategy default)
            
        Returns:
            Tuple of (max_shares, max_notional)
        """
        try:
            if price <= 0:
                return 0, 0
                
            # Use strategy's risk percent if not specified
            if risk_percent is None:
                if hasattr(self, 'parameters') and 'max_risk_per_trade_pct' in self.parameters:
                    risk_percent = self.parameters['max_risk_per_trade_pct']
                else:
                    # Default to 2% if not specified
                    risk_percent = 2.0
                    
            # Get buying power based on trade type
            buying_power = self.get_buying_power(day_trade=is_day_trade)
            
            # Calculate max position size based on risk percentage
            equity = self.get_account_equity()
            max_risk_amount = equity * (risk_percent / 100.0)
            
            # Calculate max notional exposure
            # For day trades, we can use more leverage but must limit risk
            if is_day_trade:
                # More conservative of the two approaches
                max_notional_by_risk = max_risk_amount * 10  # Assuming 10% max loss
                max_notional_by_buying_power = buying_power
                max_notional = min(max_notional_by_risk, max_notional_by_buying_power)
            else:
                # For swing/position trades, use more conservative sizing
                max_notional_by_risk = max_risk_amount * 5  # Assuming 20% max loss
                max_notional_by_buying_power = buying_power
                max_notional = min(max_notional_by_risk, max_notional_by_buying_power)
            
            # Convert to shares
            max_shares = max_notional / price
            
            # Check if fractional shares are supported
            account_type = self.get_account_type()
            platform = self.account_status.get('platform', 'alpaca')
            
            if platform in self.platform_params and not self.platform_params[platform]['supports_fractional_shares']:
                # Round down to whole shares
                max_shares = int(max_shares)
                # Recalculate max notional based on whole shares
                max_notional = max_shares * price
                
            return max_shares, max_notional
            
        except Exception as e:
            logger.error(f"Error calculating max position size: {str(e)}")
            return 0, 0
    
    def can_trade_security(self, symbol: str, security_type: str = 'stock') -> bool:
        """
        Check if the account can trade the specified security type.
        
        Args:
            symbol: Security symbol
            security_type: Type of security (stock, option, crypto, forex)
            
        Returns:
            Boolean indicating whether the account can trade this security
        """
        try:
            account_type = self.get_account_type()
            
            # Check account type restrictions
            if account_type == TradingAccountType.IRA or account_type == TradingAccountType.ROTH_IRA:
                # IRAs typically have restrictions on certain security types
                if security_type in ['forex', 'futures']:
                    logger.warning(f"Cannot trade {security_type} in IRA account")
                    return False
            
            # Check if the platform supports this security type
            platform = self.account_status.get('platform', 'alpaca')
            
            # This would need to be expanded based on actual platform capabilities
            platform_security_support = {
                'alpaca': ['stock', 'crypto'],
                'tradier': ['stock', 'option'],
                'etrade': ['stock', 'option'],
                'oanda': ['forex'],
                'binance': ['crypto']
            }
            
            if platform in platform_security_support and security_type not in platform_security_support[platform]:
                logger.warning(f"Platform {platform} does not support {security_type} trading")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking security tradability: {str(e)}")
            return False
    
    def validate_trade_size(self, 
                           symbol: str, 
                           quantity: float, 
                           price: float, 
                           is_day_trade: bool = False) -> bool:
        """
        Validate if a proposed trade is within account limitations.
        
        Args:
            symbol: Security symbol
            quantity: Number of shares/contracts
            price: Current price per share/contract
            is_day_trade: Whether this is a day trade
            
        Returns:
            Boolean indicating whether the trade size is valid
        """
        try:
            # Calculate notional value
            notional_value = quantity * price
            
            # Check minimum order size
            platform = self.account_status.get('platform', 'alpaca')
            if platform in self.platform_params:
                min_order_size = self.platform_params[platform]['min_order_size']
                if notional_value < min_order_size:
                    logger.warning(f"Order size (${notional_value:.2f}) below minimum (${min_order_size:.2f})")
                    return False
            
            # Check buying power
            buying_power = self.get_buying_power(day_trade=is_day_trade)
            if notional_value > buying_power:
                logger.warning(f"Order size (${notional_value:.2f}) exceeds buying power (${buying_power:.2f})")
                return False
            
            # Check PDT rule for day trades
            if is_day_trade and not self.check_pdt_rule_compliance(is_day_trade=True):
                logger.warning("Order rejected due to PDT rule restrictions")
                return False
                
            # Check max position size based on risk management
            max_shares, max_notional = self.calculate_max_position_size(price, is_day_trade)
            if quantity > max_shares:
                logger.warning(f"Order quantity ({quantity}) exceeds max allowed ({max_shares:.2f}) based on risk management")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade size: {str(e)}")
            return False
