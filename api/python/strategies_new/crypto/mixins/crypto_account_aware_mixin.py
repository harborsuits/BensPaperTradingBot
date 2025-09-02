#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Account Aware Mixin

This module provides a mixin class that makes crypto strategies account-aware,
ensuring they respect account balance, risk limits, and other constraints.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class CryptoAccountAwareMixin:
    """
    Mixin class to make crypto strategies account-aware.
    
    This mixin provides methods to:
    1. Check account balances and positions
    2. Enforce risk limits and position sizing based on account value
    3. Handle crypto-specific constraints like:
       - Network fees (gas costs)
       - Blockchain confirmation times
       - Exchange-specific constraints
       - Lending/borrowing positions
       - Staking and LP positions
    """
    
    def __init__(self, *args, **kwargs):
        # Call the parent's __init__ if it exists
        super().__init__(*args, **kwargs)
        
        # Initialize account tracking
        self.account_balances = {}  # Currency -> amount
        self.lending_positions = []  # Tracks lending positions
        self.borrowing_positions = []  # Tracks borrowing
        self.staking_positions = []  # Tracks staking
        self.lp_positions = []  # Tracks liquidity providing positions
        
        # Risk management parameters
        self.risk_params = {
            'max_position_size_pct': 0.05,  # Maximum 5% of portfolio in single position
            'max_exchange_exposure_pct': 0.25,  # Maximum 25% on a single exchange
            'max_asset_exposure_pct': 0.20,  # Maximum 20% in a single asset
            'min_remaining_balance_pct': 0.10,  # Keep at least 10% in reserve
            'max_leverage': 2.0,  # Maximum 2x leverage
            'max_gas_per_tx_usd': 50.0,  # Maximum $50 gas per transaction
            'max_slippage_pct': 0.01,  # Maximum 1% slippage
            'max_daily_loss_pct': 0.03,  # Maximum 3% daily loss
            'max_drawdown_pct': 0.15,  # Maximum 15% drawdown
        }
        
        # Tracking variables
        self.portfolio_value_history = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.starting_daily_value = 0.0
        self.last_update_day = datetime.now(timezone.utc).date()
        
    def update_account_info(self, account_data: Dict[str, Any]) -> None:
        """
        Update the account information from external source.
        
        Args:
            account_data: Dictionary with account information including balances,
                          positions, lending, borrowing, etc.
        """
        # Update basic balances
        self.account_balances = account_data.get('balances', {})
        
        # Update DeFi positions
        self.lending_positions = account_data.get('lending_positions', [])
        self.borrowing_positions = account_data.get('borrowing_positions', [])
        self.staking_positions = account_data.get('staking_positions', [])
        self.lp_positions = account_data.get('lp_positions', [])
        
        # Calculate total portfolio value
        portfolio_value = self._calculate_portfolio_value()
        
        # Track for drawdown calculations
        self.portfolio_value_history.append({
            'timestamp': datetime.now(timezone.utc),
            'value': portfolio_value
        })
        
        # Initialize daily tracking if needed
        current_day = datetime.now(timezone.utc).date()
        if not self.starting_daily_value or self.last_update_day != current_day:
            self.starting_daily_value = portfolio_value
            self.daily_pnl = 0.0
            self.last_update_day = current_day
        else:
            # Update daily P&L
            self.daily_pnl = portfolio_value - self.starting_daily_value
        
        # Update drawdown
        self._update_drawdown()
        
        logger.debug(f"Updated account info. Portfolio value: ${portfolio_value:.2f}")
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate the total portfolio value including all positions.
        
        Returns:
            Total portfolio value in USD
        """
        # Base calculation from spot holdings
        total_value = sum(
            balance * self._get_asset_price(asset) 
            for asset, balance in self.account_balances.items()
        )
        
        # Add value of lending positions
        for position in self.lending_positions:
            total_value += position.get('value_usd', 0)
        
        # Add value of staking positions
        for position in self.staking_positions:
            total_value += position.get('value_usd', 0)
        
        # Add value of LP positions
        for position in self.lp_positions:
            total_value += position.get('value_usd', 0)
        
        # Subtract borrowed amounts
        for position in self.borrowing_positions:
            total_value -= position.get('value_usd', 0)
        
        return total_value
    
    def _get_asset_price(self, asset: str) -> float:
        """
        Get the current price of an asset in USD.
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Current price in USD
        """
        # In a real implementation, this would fetch prices from market data
        # For now, we'll just return a placeholder
        placeholder_prices = {
            'BTC': 40000.0,
            'ETH': 2800.0,
            'USDT': 1.0,
            'USDC': 1.0,
            'DAI': 1.0
        }
        
        return placeholder_prices.get(asset, 0.0)
    
    def _update_drawdown(self) -> None:
        """Update drawdown calculations based on portfolio value history."""
        if not self.portfolio_value_history:
            return
            
        # Get the highest portfolio value so far
        peak_value = max(item['value'] for item in self.portfolio_value_history)
        current_value = self.portfolio_value_history[-1]['value']
        
        # Calculate current drawdown
        if peak_value > 0:
            self.current_drawdown = (peak_value - current_value) / peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def check_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if current positions and account state respect risk limits.
        
        Returns:
            Tuple of (limits_respected, reason_if_not)
        """
        # Check daily loss limit
        if self.daily_pnl < 0 and abs(self.daily_pnl) / self.starting_daily_value > self.risk_params['max_daily_loss_pct']:
            return False, f"Daily loss limit exceeded: {abs(self.daily_pnl) / self.starting_daily_value:.2%}"
        
        # Check drawdown limit
        if self.current_drawdown > self.risk_params['max_drawdown_pct']:
            return False, f"Max drawdown exceeded: {self.current_drawdown:.2%}"
        
        # Check exchange exposure
        exchange_exposure = self._calculate_exchange_exposure()
        for exchange, exposure in exchange_exposure.items():
            if exposure > self.risk_params['max_exchange_exposure_pct']:
                return False, f"Exchange exposure limit exceeded for {exchange}: {exposure:.2%}"
        
        # Check asset exposure
        asset_exposure = self._calculate_asset_exposure()
        for asset, exposure in asset_exposure.items():
            if exposure > self.risk_params['max_asset_exposure_pct']:
                return False, f"Asset exposure limit exceeded for {asset}: {exposure:.2%}"
        
        # All checks passed
        return True, ""
    
    def _calculate_exchange_exposure(self) -> Dict[str, float]:
        """
        Calculate the exposure to each exchange as percentage of portfolio.
        
        Returns:
            Dictionary mapping exchange name to exposure percentage
        """
        # In a real implementation, this would calculate actual exchange exposure
        # For now, return a placeholder
        return {'binance': 0.2, 'coinbase': 0.15, 'uniswap': 0.05}
    
    def _calculate_asset_exposure(self) -> Dict[str, float]:
        """
        Calculate the exposure to each asset as percentage of portfolio.
        
        Returns:
            Dictionary mapping asset symbol to exposure percentage
        """
        # In a real implementation, this would calculate actual asset exposure
        # For now, return a placeholder
        return {'BTC': 0.15, 'ETH': 0.12, 'USDT': 0.3}
    
    def validate_order(self, 
                      symbol: str, 
                      order_type: str, 
                      side: str, 
                      amount: float, 
                      price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate an order against account constraints.
        
        Args:
            symbol: Trading pair symbol
            order_type: Type of order (market, limit, etc.)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (for limit orders)
            
        Returns:
            Tuple of (is_valid, reason_if_not)
        """
        portfolio_value = self._calculate_portfolio_value()
        
        # Check if we're buying or selling the base currency
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol.split('-')[0]
        quote_currency = symbol.split('/')[1] if '/' in symbol else symbol.split('-')[1]
        
        # Calculate order value
        if price:
            order_value = amount * price
        else:
            # For market orders, use the current market price
            asset_price = self._get_asset_price(base_currency)
            order_value = amount * asset_price
        
        # Validate against max position size
        if order_value / portfolio_value > self.risk_params['max_position_size_pct']:
            return False, f"Order exceeds maximum position size: {order_value / portfolio_value:.2%}"
        
        # Validate for buying with sufficient balance
        if side == 'buy':
            quote_balance = self.account_balances.get(quote_currency, 0)
            if quote_balance < order_value:
                return False, f"Insufficient {quote_currency} balance: {quote_balance} < {order_value}"
            
            # Check minimum remaining balance
            remaining_balance = quote_balance - order_value
            if remaining_balance / portfolio_value < self.risk_params['min_remaining_balance_pct']:
                return False, f"Order would leave insufficient reserve: {remaining_balance / portfolio_value:.2%}"
        
        # Validate for selling with sufficient balance
        elif side == 'sell':
            base_balance = self.account_balances.get(base_currency, 0)
            if base_balance < amount:
                return False, f"Insufficient {base_currency} balance: {base_balance} < {amount}"
        
        # For DeFi transactions, check gas costs (Ethereum-based)
        if base_currency in ['ETH', 'WETH'] or quote_currency in ['ETH', 'WETH']:
            # Placeholder for estimating gas costs
            estimated_gas_usd = 30.0
            
            if estimated_gas_usd > self.risk_params['max_gas_per_tx_usd']:
                return False, f"Estimated gas cost too high: ${estimated_gas_usd:.2f}"
        
        # All validations passed
        return True, ""
    
    def calculate_position_size(self, 
                                symbol: str, 
                                risk_per_trade_pct: float, 
                                stop_loss_pct: float) -> float:
        """
        Calculate the position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            risk_per_trade_pct: Percentage of portfolio to risk on this trade
            stop_loss_pct: Stop loss distance as percentage
            
        Returns:
            Position size in base currency units
        """
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate risk amount in USD
        risk_amount = portfolio_value * risk_per_trade_pct
        
        # Get the base currency from the symbol
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol.split('-')[0]
        
        # Get current price of the asset
        current_price = self._get_asset_price(base_currency)
        
        # Calculate position size in units
        if stop_loss_pct > 0:
            position_size = risk_amount / (current_price * stop_loss_pct)
        else:
            # Fallback if no stop loss specified
            position_size = risk_amount / current_price
        
        # Cap position size based on max position size and available balance
        max_position_size = portfolio_value * self.risk_params['max_position_size_pct'] / current_price
        position_size = min(position_size, max_position_size)
        
        # Round to appropriate precision for the asset
        decimals = 8 if base_currency == 'BTC' else 6
        position_size = round(position_size, decimals)
        
        return position_size
    
    def estimate_transaction_costs(self, 
                                  symbol: str, 
                                  amount: float, 
                                  price: float, 
                                  is_defi: bool = False) -> Dict[str, float]:
        """
        Estimate transaction costs including exchange fees and gas costs.
        
        Args:
            symbol: Trading pair symbol
            amount: Order amount
            price: Asset price
            is_defi: Whether this is a DeFi transaction
            
        Returns:
            Dictionary with cost breakdown
        """
        # Basic calculation for centralized exchanges
        trade_value = amount * price
        exchange_fee = trade_value * 0.001  # Assume 0.1% fee
        
        costs = {
            'exchange_fee': exchange_fee,
            'slippage_estimate': trade_value * 0.001,  # Estimate 0.1% slippage
            'total': exchange_fee
        }
        
        # Add gas costs for DeFi transactions
        if is_defi:
            # ETH gas estimate (in USD)
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol.split('-')[0]
            
            # Different gas estimates based on network
            if base_currency in ['ETH', 'WETH'] or symbol.endswith('ETH'):
                # Ethereum mainnet
                gas_cost_usd = 25.0
            elif base_currency.startswith('ARB') or 'ARBITRUM' in symbol:
                # Arbitrum
                gas_cost_usd = 1.0
            elif base_currency.startswith('MATIC') or 'POLYGON' in symbol:
                # Polygon
                gas_cost_usd = 0.5
            else:
                # Default estimate
                gas_cost_usd = 5.0
                
            costs['gas_cost_usd'] = gas_cost_usd
            costs['total'] += gas_cost_usd
        
        return costs
    
    def get_available_balance(self, currency: str) -> float:
        """
        Get the available balance for a specific currency.
        
        Args:
            currency: Currency symbol
            
        Returns:
            Available balance
        """
        return self.account_balances.get(currency, 0.0)
    
    def check_lending_eligibility(self, asset: str, amount: float) -> Tuple[bool, str]:
        """
        Check if an asset is eligible for lending.
        
        Args:
            asset: Asset symbol
            amount: Amount to lend
            
        Returns:
            Tuple of (is_eligible, reason_if_not)
        """
        # Check if we have enough balance
        available_balance = self.get_available_balance(asset)
        if available_balance < amount:
            return False, f"Insufficient balance: {available_balance} < {amount}"
        
        # Check lending exposure
        total_lending = sum(p.get('value_usd', 0) for p in self.lending_positions if p.get('asset') == asset)
        asset_price = self._get_asset_price(asset)
        new_lending_value = total_lending + (amount * asset_price)
        
        # Limit lending to 50% of portfolio value for a single asset
        portfolio_value = self._calculate_portfolio_value()
        if new_lending_value / portfolio_value > 0.5:
            return False, f"Lending exposure too high: {new_lending_value / portfolio_value:.2%}"
        
        return True, ""
    
    def check_borrowing_eligibility(self, asset: str, amount: float) -> Tuple[bool, str]:
        """
        Check if borrowing an asset is within risk limits.
        
        Args:
            asset: Asset symbol
            amount: Amount to borrow
            
        Returns:
            Tuple of (is_eligible, reason_if_not)
        """
        # Calculate current borrowing value
        total_borrowing = sum(p.get('value_usd', 0) for p in self.borrowing_positions)
        asset_price = self._get_asset_price(asset)
        new_borrowing_value = total_borrowing + (amount * asset_price)
        
        # Calculate portfolio value and health factor
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate collateral value (assuming all non-borrowed assets can be collateral)
        collateral_value = portfolio_value + total_borrowing
        
        # Check borrowing capacity - limit to 40% of collateral value
        if new_borrowing_value / collateral_value > 0.4:
            return False, f"Borrowing capacity exceeded: {new_borrowing_value / collateral_value:.2%}"
        
        return True, ""
    
    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics."""
        self.starting_daily_value = self._calculate_portfolio_value()
        self.daily_pnl = 0.0
        self.last_update_day = datetime.now(timezone.utc).date()
