#!/usr/bin/env python3
"""
Tradier Broker Adapter - Margin Status Extension

This module extends the Tradier adapter with margin status reporting
capabilities for risk management.
"""

from typing import Dict, Any, Optional
import logging

class TradierMarginExtension:
    """
    Extension for Tradier adapter to support margin status reporting
    
    This mixin class adds margin status functionality to the Tradier adapter.
    """
    
    def get_margin_status(self) -> Dict[str, Any]:
        """
        Get margin account status from Tradier
        
        Returns:
            Dict containing margin status information
        """
        try:
            # Get account balances
            balances = self.get_account_balances()
            
            # Extract margin-related information
            account_id = self.account_id
            cash = balances.get('cash', 0.0)
            margin_used = balances.get('margin_balance', 0.0)
            buying_power = balances.get('day_trade_buying_power', balances.get('buying_power', 0.0))
            maintenance_requirement = balances.get('margin_requirement', 0.0)
            
            # If margin-specific fields aren't available, use what we have
            if margin_used == 0 and 'reg_t_balance' in balances:
                margin_used = max(0, -balances.get('reg_t_balance', 0.0))
            
            # Calculate actual maintenance margin level if not directly provided
            if maintenance_requirement == 0 and 'long_market_value' in balances:
                # Default maintenance requirement is 25% for stocks
                maintenance_requirement = balances.get('long_market_value', 0.0) * 0.25
            
            # Format the response according to the standard interface
            return {
                "account_id": account_id,
                "cash": cash,
                "margin_used": margin_used,
                "buying_power": buying_power,
                "maintenance_requirement": maintenance_requirement
            }
            
        except Exception as e:
            self.logger.error(f"Error getting margin status: {str(e)}")
            
            # Return default values if we fail to get actual data
            return {
                "account_id": self.account_id,
                "cash": 0.0,
                "margin_used": 0.0,
                "buying_power": 0.0,
                "maintenance_requirement": 0.0
            }
