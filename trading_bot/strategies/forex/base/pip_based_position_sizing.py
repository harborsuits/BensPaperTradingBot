#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pip-Based Position Sizing for Forex

This module implements specialized pip-based position sizing for forex trades,
with risk management calculations specific to forex pairs and lot sizes.
"""

import logging
import math
import numpy as np
from typing import Dict, Optional, Any, Union, List, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class RiskProfile(Enum):
    """Risk profiles that can be applied based on market regime"""
    CONSERVATIVE = "conservative"  # Lower risk, smaller positions
    MODERATE = "moderate"        # Balanced risk-reward
    AGGRESSIVE = "aggressive"    # Higher risk, larger positions
    VOLATILITY_ADAPTIVE = "volatility_adaptive"  # Adjusts based on volatility
    TREND_FOLLOWING = "trend_following"  # More aggressive in strong trends
    MEAN_REVERSION = "mean_reversion"  # More aggressive in extended moves
    SECURITY_FOCUSED = "security_focused"  # Scales back risk as account grows


class PipBasedPositionSizing:
    """
    Advanced Pip-Based Position Sizing for Forex
    
    Features:
    - Fixed risk per trade (optional % of account)
    - Stop loss in pips
    - Volatility-based position size adjustment
    - Adaptive risk management based on market regime
    - Progressive risk scaling for security (reducing risk as account grows)
    - Multi-asset correlation awareness
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        'max_risk_per_trade_percent': 1.0,   # 1% risk per trade
        'min_position_size': 0.01,           # Minimum position size (micro lot)
        'max_position_size': 5.0,            # Maximum position size
        'max_position_value_percent': 15.0,  # Maximum position size as % of account
        'position_size_rounding': 0.01,      # Round to 0.01 lots
        'use_dynamic_lot_sizing': True,      # Dynamically adjust lot size based on volatility
        'atr_risk_multiplier': 1.5,          # ATR multiplier for dynamic sizing
        'lot_size_discount_above_threshold': 0.8,  # 20% discount for large accounts
        'account_size_threshold': 50000,     # Threshold for discount (USD)
        'pip_value': 0.0001,                 # Standard pip value for 4-digit pairs
        'jpy_pip_value': 0.01,               # Pip value for JPY pairs
        
        # Advanced parameters for adaptive risk management
        'enable_adaptive_risk': True,        # Enable adaptive risk adjustment
        'min_risk_percent': 0.5,             # Minimum risk percentage (floor)
        'max_risk_percent': 2.0,             # Maximum risk percentage (ceiling)
        'performance_lookback_trades': 20,   # Number of trades to consider for performance
        'regime_adjustment_factor': 0.8,     # How strongly regime affects sizing (0-1)
        'correlation_adjustment_enabled': True,  # Consider correlation in sizing
        'max_portfolio_risk_percent': 5.0,   # Maximum overall portfolio risk
        'drawdown_adjustment_factor': 0.5,   # How strongly drawdown affects sizing (0-1)
        'drawdown_threshold': 5.0            # Drawdown percentage to trigger reduction
    }
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Pip-Based Position Sizing.
        
        Args:
            parameters: Custom parameters to override defaults
        """
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
            
        logger.info(f"Pip-Based Position Sizing initialized with parameters: {self.parameters}")
        
        self.risk_per_trade = self.parameters['max_risk_per_trade_percent'] / 100.0  # Risk per trade (as decimal)
        self.base_lot_size = self.parameters['min_position_size']    # Base lot size
        self.max_lot_size = self.parameters['max_position_size']      # Maximum lot size
        self.volatility_adjustment = self.parameters['use_dynamic_lot_sizing']  # Adjust for vol?
        self.min_stop_pips = 10       # Minimum stop size in pips
        self.enable_progressive_risk = True  # Enable progressive risk scaling
        self.initial_capital = 500.0  # Initial starting capital for progressive scaling
        self.default_risk_profile = RiskProfile.MODERATE  # Default risk profile
        
        # Progressive risk scaling parameters - "Snowball or Fail" approach with balanced risk
        # Aggressive but sustainable approach until hitting $25,000 for day trading eligibility
        # Format: (account_balance, risk_percentage_of_account)
        self.use_absolute_thresholds = True  # Use dollar amounts instead of multipliers
        self.day_trading_threshold = 25000   # PDT rule threshold
        self.progressive_risk_thresholds = [
            # Aggressive but sustainable phase - initial capital growth
            (500, 0.25),     # Up to $500 - risk 25% of account (still aggressive but survivable)
            (1000, 0.22),    # Up to $1000 - risk 22% of account
            (2500, 0.20),    # Up to $2500 - risk 20% of account
            (5000, 0.18),    # Up to $5000 - risk 18% of account
            (7000, 0.16),    # Up to $7K - risk 16% of account
            
            # More moderate aggressive phase (still pushing for PDT)
            (10000, 0.15),   # Up to $10k - risk 15% of account
            (15000, 0.12),   # Up to $15k - risk 12% of account
            (20000, 0.10),   # Up to $20k - risk 10% of account
            (24999, 0.08),   # Up to $24,999 - risk 8% of account
            
            # After reaching day trading threshold, become more conservative
            (25000, 0.05),   # At PDT threshold - drop to 5% (protect achievement)
            (35000, 0.05),   # Up to $35k - risk 5% of account
            (50000, 0.04),   # Up to $50k - risk 4% of account
            (100000, 0.03),  # Up to $100k - risk 3% of account
            (250000, 0.025), # Up to $250k - risk 2.5% of account
            (500000, 0.02),  # Up to $500k - risk 2% of account
            (1000000, 0.015) # Over $1M - risk 1.5% of account
        ]
        
        # Start with a high risk approach initially
        self.initial_risk_high = True  # Flag to indicate we're using the high initial risk approach
        
        # Advanced parameters
        self.risk_adjustment_ratio = 1.0  # Default no adjustment
        
        # Risk adjustment by market regime (for MODERATE profile)
        self.market_regime_risk_ratios = {
            'trending': 1.2,    # More risk in trending markets
            'ranging': 0.8,     # Less risk in ranging markets
            'volatile': 0.6,    # Much less risk in volatile markets
            'breakout': 1.0,    # Normal risk in breakout markets
            'low_volatility': 1.3,  # More risk in low vol markets
            'unknown': 1.0      # Default for unknown regime
        }
        
        # Risk profiles define different approaches to position sizing
        self.risk_profiles = {
            RiskProfile.CONSERVATIVE: {
                'trending': 0.8,     # Still cautious in trends
                'ranging': 0.6,      # Very cautious in ranges
                'volatile': 0.4,     # Extremely cautious in volatility
                'breakout': 0.6,     # Cautious on breakouts
                'low_volatility': 0.9,  # Less cautious in low vol
                'unknown': 0.6       # Cautious by default
            },
            RiskProfile.MODERATE: self.market_regime_risk_ratios,  # Use default settings
            RiskProfile.AGGRESSIVE: {
                'trending': 1.5,     # Significantly more risk in trends
                'ranging': 1.0,      # Normal risk in ranges
                'volatile': 0.8,     # Slightly reduced in volatility
                'breakout': 1.4,     # More risk on breakouts
                'low_volatility': 1.6,  # Much more in low vol
                'unknown': 1.2       # Slightly elevated by default
            },
            RiskProfile.VOLATILITY_ADAPTIVE: {
                # This is dynamically calculated based on actual volatility
                # The baseline values are just starting points
                'trending': 1.0,
                'ranging': 1.0,
                'volatile': 0.5,
                'breakout': 0.8,
                'low_volatility': 1.5,
                'unknown': 1.0
            },
            RiskProfile.TREND_FOLLOWING: {
                'trending': 1.8,     # Much larger in confirmed trends
                'ranging': 0.5,      # Much smaller in ranges
                'volatile': 0.7,     # Reduced in volatility
                'breakout': 1.5,     # Higher on breakouts (potential new trends)
                'low_volatility': 1.2,  # Slightly higher in low vol
                'unknown': 1.0       # Normal by default
            },
            RiskProfile.MEAN_REVERSION: {
                'trending': 0.7,     # Lower in trends
                'ranging': 1.4,      # Higher in ranges
                'volatile': 1.0,     # Normal in volatility
                'breakout': 0.6,     # Lower on breakouts
                'low_volatility': 0.8,  # Lower in low vol
                'unknown': 1.0       # Normal by default
            },
            RiskProfile.SECURITY_FOCUSED: {
                # This uses progressive risk scaling regardless of regime
                # But we still adjust slightly based on regime
                'trending': 1.1,    
                'ranging': 0.9,
                'volatile': 0.6,
                'breakout': 0.8,
                'low_volatility': 1.2,
                'unknown': 1.0
            }
        }
    
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss_pips: float, 
                              account_balance: float,
                              account_currency: str = 'USD',
                              pair_exchange_rate: Optional[float] = None,
                              volatility_factor: Optional[float] = None,
                              risk_profile: Optional[RiskProfile] = None,
                              initial_account_balance: Optional[float] = None) -> float:
        """
        Calculate position size (lot size) based on risk parameters and stop loss in pips.
        
        Args:
            symbol: Forex pair symbol (e.g., 'EUR/USD')
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            account_balance: Account balance
            account_currency: Account currency (default: USD)
            pair_exchange_rate: Exchange rate for cross-currency calculations
            volatility_factor: Optional volatility adjustment (1.0 = normal, <1.0 = reduce for high volatility)
            risk_profile: Risk profile to use (optional)
            initial_account_balance: Initial account balance for progressive risk scaling
            
        Returns:
            Position size in lots
        """
        # Validate inputs
        if stop_loss_pips <= 0:
            logger.warning(f"Invalid stop loss pips: {stop_loss_pips}, using fallback of 10 pips")
            stop_loss_pips = 10.0
            
        # Calculate risk amount in account currency
        max_risk_percent = self.parameters['max_risk_per_trade_percent'] / 100.0
        risk_amount = account_balance * max_risk_percent
        
        # Apply discount for large accounts
        if account_balance > self.parameters['account_size_threshold']:
            logger.debug(f"Applying position size discount for large account: {account_balance}")
            risk_amount *= self.parameters['lot_size_discount_above_threshold']
        
        # Apply volatility adjustment if provided
        if volatility_factor is not None and self.parameters['use_dynamic_lot_sizing']:
            logger.debug(f"Applying volatility factor: {volatility_factor}")
            risk_amount *= volatility_factor
        
        # Calculate pip value in account currency
        pip_value_per_lot = self._calculate_pip_value(
            symbol, 
            entry_price, 
            account_currency, 
            pair_exchange_rate
        )
        
        # Calculate position size in lots
        if pip_value_per_lot > 0 and stop_loss_pips > 0:
            position_size = risk_amount / (pip_value_per_lot * stop_loss_pips)
        else:
            # Fallback calculation if we can't determine pip value
            logger.warning(f"Using fallback position sizing for {symbol}")
            position_size = risk_amount / (entry_price * 0.01)  # Assume 1% move
        
        # Apply min/max constraints
        position_size = max(self.parameters['min_position_size'], position_size)
        position_size = min(self.parameters['max_position_size'], position_size)
        
        # Check maximum position value constraint
        max_position_value = account_balance * (self.parameters['max_position_value_percent'] / 100.0)
        position_value = position_size * 100000 * entry_price  # Assuming standard lot = 100,000 units
        
        if position_value > max_position_value:
            constrained_position_size = max_position_value / (100000 * entry_price)
            logger.info(f"Position size constrained by max value: {position_size:.2f} -> {constrained_position_size:.2f}")
            position_size = constrained_position_size
        
        # Apply risk profile adjustments
        if risk_profile:
            regime_adjustment = self.risk_profiles[risk_profile]['unknown']
            position_size *= regime_adjustment
        
        # Apply progressive risk scaling if enabled
        if self.enable_progressive_risk:
            account_multiplier = account_balance / (initial_account_balance or self.initial_capital)
            progressive_risk_ratio = self._get_progressive_risk_ratio(account_multiplier)
            position_size *= progressive_risk_ratio
        
        # Round to the nearest increment
        rounding = self.parameters['position_size_rounding']
        position_size = self._round_lot_size(position_size, rounding)
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.2f} lots (risk: {risk_amount:.2f} {account_currency}, stop: {stop_loss_pips} pips)")
        
        return position_size
    
    def _get_progressive_risk_ratio(self, account_balance: float) -> float:
        """
        Get the progressive risk ratio based on the account balance.
        
        Args:
            account_balance: Current account balance in dollars
            
        Returns:
            Progressive risk ratio (as a percentage of account to risk)
        """
        # Find the appropriate threshold for the current account size
        for threshold, ratio in self.progressive_risk_thresholds:
            if account_balance <= threshold:
                return ratio
        
        # If we exceed all thresholds, use the last ratio
        return self.progressive_risk_thresholds[-1][1]
    
    def calculate_position_size_with_atr(self,
                                       symbol: str,
                                       entry_price: float,
                                       atr_value: float,
                                       account_balance: float,
                                       account_currency: str = 'USD',
                                       pair_exchange_rate: Optional[float] = None) -> float:
        """
        Calculate position size using ATR for dynamic stop loss distance.
        
        Args:
            symbol: Forex pair symbol
            entry_price: Entry price
            atr_value: Average True Range value
            account_balance: Account balance
            account_currency: Account currency
            pair_exchange_rate: Exchange rate for cross-currency calculations
            
        Returns:
            Position size in lots
        """
        # Determine pip value for this pair
        pip_value = self.parameters['pip_value']
        if 'JPY' in symbol:
            pip_value = self.parameters['jpy_pip_value']
            
        # Convert ATR to pips
        atr_pips = atr_value / pip_value
        
        # Use ATR to determine stop loss distance
        stop_loss_pips = atr_pips * self.parameters['atr_risk_multiplier']
        
        # Calculate volatility factor
        # Higher ATR = higher volatility = reduce position size
        baseline_atr = {
            'EUR/USD': 60,  # Baseline ATR in pips for major pairs
            'GBP/USD': 90,
            'USD/JPY': 50,
            'USD/CHF': 60,
            'AUD/USD': 60,
            'NZD/USD': 60,
            'USD/CAD': 60,
        }.get(symbol, 70)  # Default baseline for other pairs
        
        volatility_factor = min(1.0, baseline_atr / max(atr_pips, 1))
        
        # Calculate position size
        return self.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_pips=stop_loss_pips,
            account_balance=account_balance,
            account_currency=account_currency,
            pair_exchange_rate=pair_exchange_rate,
            volatility_factor=volatility_factor
        )
    
    def _calculate_pip_value(self, 
                           symbol: str, 
                           price: float, 
                           account_currency: str,
                           pair_exchange_rate: Optional[float] = None) -> float:
        """
        Calculate the value of one pip per standard lot in account currency.
        
        Args:
            symbol: Forex pair (e.g., 'EUR/USD')
            price: Current exchange rate
            account_currency: Account currency
            pair_exchange_rate: Exchange rate for cross-currency calculations
            
        Returns:
            Pip value per standard lot in account currency
        """
        # Extract base and quote currencies
        if '/' in symbol:
            base_currency, quote_currency = symbol.split('/')
        else:
            # Handle different formats like 'EURUSD'
            if len(symbol) >= 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
            else:
                logger.warning(f"Cannot parse symbol: {symbol}, using fallback pip value")
                return 10.0  # Fallback value
        
        # Determine pip value
        pip_value = self.parameters['pip_value']
        if 'JPY' in symbol:
            pip_value = self.parameters['jpy_pip_value']
        
        # Standard lot size
        standard_lot = 100000
        
        # Calculate pip value in quote currency
        pip_value_quote = standard_lot * pip_value
        
        # Convert to account currency if needed
        if quote_currency == account_currency:
            # Direct conversion - pip value is already in account currency
            return pip_value_quote
        elif base_currency == account_currency:
            # Inverse conversion
            return pip_value_quote / price
        else:
            # Cross conversion - need additional exchange rate
            if pair_exchange_rate is not None:
                return pip_value_quote * pair_exchange_rate
            else:
                # Fallback estimation
                logger.warning(f"No exchange rate provided for {symbol} to {account_currency} conversion, using estimate")
                # Approximate value for major pairs
                estimated_values = {
                    'EUR/USD': 10.0,
                    'GBP/USD': 10.0,
                    'USD/JPY': 9.0,
                    'AUD/USD': 10.0,
                    'NZD/USD': 10.0,
                    'USD/CAD': 8.0,
                    'USD/CHF': 10.0
                }
                return estimated_values.get(symbol, 10.0)  # Default estimate
    
    def _round_lot_size(self, lot_size: float, increment: float) -> float:
        """
        Round lot size to the specified increment.
        
        Args:
            lot_size: Calculated lot size
            increment: Rounding increment (e.g., 0.01 for micro lots)
            
        Returns:
            Rounded lot size
                                       correlation_matrix: Dict[str, Dict[str, float]] = None,
                                       current_drawdown: float = 0.0,
                                       asset_class: str = 'forex',
                                       account_currency: str = 'USD',
                                       risk_profile: Optional[RiskProfile] = None,
                                       initial_account_balance: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate position size with adaptive risk management based on market conditions and performance.
        Implements security-focused progressive risk scaling that reduces risk as account grows.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            account_balance: Current account balance
            recent_trades: List of recent trades for performance analysis
            market_regime: Current market regime
            volatility_state: Current volatility state
            current_positions: List of current open positions
            correlation_matrix: Correlation matrix for multi-asset management
            current_drawdown: Current drawdown percentage
            asset_class: Asset class ('forex', 'stocks', etc.)
            account_currency: Account currency (default: USD)
            risk_profile: Risk profile to use (optional)
            initial_account_balance: Initial account balance for progressive risk scaling
            
        Returns:
            Dictionary with position size details and explanations
        """
    def _round_lot_size(self, lot_size: float, increment: float) -> float:
        """Round lot size to the specified increment.
        
        Args:
            lot_size: Lot size to round
            increment: Increment to round to
            
        Returns:
            Rounded lot size
        """
        # Use Decimal for precise rounding
        lot_decimal = Decimal(str(lot_size))
        increment_decimal = Decimal(str(increment))
        
        # Calculate number of increments
        increments = lot_decimal / increment_decimal
        
        # Round down to the nearest increment
        rounded_increments = math.floor(increments)
        
        # Convert back to the actual lot size
        rounded_lot_size = Decimal(str(rounded_increments)) * increment_decimal
        
        return float(rounded_lot_size)
    
    def calculate_adaptive_position_size(self, 
                                       symbol: str,
                                       entry_price: float,
                                       stop_loss_pips: float,
                                       account_balance: float,
                                       recent_trades: List[Dict[str, Any]] = None,
                                       market_regime: str = 'unknown',
                                       volatility_state: str = 'medium',
                                       current_positions: List[Dict[str, Any]] = None,
                                       correlation_matrix: Dict[str, Dict[str, float]] = None,
                                       current_drawdown: float = 0.0,
                                       asset_class: str = 'forex',
                                       account_currency: str = 'USD',
                                       risk_profile: Optional[RiskProfile] = None,
                                       initial_account_balance: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate position size with adaptive risk management based on market conditions and performance.
        Implements security-focused progressive risk scaling that reduces risk as account grows.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            account_balance: Current account balance
            recent_trades: List of recent trades for performance analysis
            market_regime: Current market regime
            volatility_state: Current volatility state
            current_positions: List of current open positions
            correlation_matrix: Correlation matrix for multi-asset management
            current_drawdown: Current drawdown percentage
            asset_class: Asset class ('forex', 'stocks', etc.)
            account_currency: Account currency (default: USD)
            risk_profile: Risk profile to use (optional)
            initial_account_balance: Initial account balance for progressive risk scaling
            
        Returns:
            Dictionary with position size details and explanations
        """
        # Use provided risk profile or default
        active_risk_profile = risk_profile or self.default_risk_profile
        
        # Use provided initial account balance or default
        current_initial_balance = initial_account_balance or self.initial_capital
        
        # 1. Start with base position size calculation
        # Convert pips to price for forex/CFDs
        pip_value = self._get_pip_value(symbol, entry_price, asset_class)
        stop_loss_amount = stop_loss_pips * pip_value
        
        # For our high-risk security approach, we use the thresholds as direct percentages of account
        # to gradually reduce risk as the account grows based on absolute dollar amounts
        if self.enable_progressive_risk:
            # Get the appropriate risk percentage based on account balance
            risk_percentage = self._get_progressive_risk_ratio(account_balance)
            
            # This percentage is now a direct percentage of the account to risk
            # For example, 0.80 means we're willing to risk 80% of the account on this trade
            base_risk_amount = account_balance * risk_percentage
            
            # Find next threshold for explanation
            next_threshold = None
            for threshold, _ in self.progressive_risk_thresholds:
                if threshold > account_balance:
                    next_threshold = threshold
                    break
                    
            # Track the risk ratio for later reference
            progressive_risk_ratio = risk_percentage
            
            # Create explanation with clear milestone amounts
            if next_threshold:
                progressive_explanation = f"Risk security approach: Risking {risk_percentage*100:.1f}% of account (${base_risk_amount:.2f}). Next reduction at ${next_threshold:,}"
            else:
                progressive_explanation = f"Risk security approach: Risking {risk_percentage*100:.1f}% of account (${base_risk_amount:.2f}). Maximum security level."
        else:
            # If progressive risk is disabled, use standard risk percentage
            base_risk_amount = account_balance * self.risk_per_trade
            progressive_risk_ratio = 1.0
            progressive_explanation = "Progressive risk scaling: disabled"
            
        # 3. Apply regime-specific risk profile
        if market_regime in self.risk_profiles[active_risk_profile]:
            regime_adjustment = self.risk_profiles[active_risk_profile][market_regime]
        else:
            regime_adjustment = self.risk_profiles[active_risk_profile]['unknown']
        
        # Apply market regime adjustment
        risk_amount = base_risk_amount * regime_adjustment
        
        # 4. Apply adaptive risk adjustments based on recent performance
        if recent_trades:
            performance_adjustment = self.adjust_risk_based_on_performance(
                recent_trades, market_regime, volatility_state
            )
            risk_amount *= performance_adjustment
            performance_explanation = f"Performance adjustment: {performance_adjustment:.2f}x"
        else:
            performance_adjustment = 1.0
            performance_explanation = "No recent trades for performance adjustment"
        
        # 5. Calculate base position size from risk amount
        position_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else 0
        
        # 6. Adjust for portfolio exposure and correlation
        if current_positions and correlation_matrix:
            original_position = position_size
            position_size = self.adjust_for_portfolio_exposure(
                position_size, symbol, current_positions, correlation_matrix
            )
            correlation_adjustment = position_size / original_position if original_position > 0 else 1.0
            correlation_explanation = f"Correlation adjustment: {correlation_adjustment:.2f}x"
        else:
            correlation_explanation = "No correlation data available"
        
        # 7. Adjust for drawdown protection
        if current_drawdown > 0:
            original_position = position_size
            position_size = self.adjust_for_drawdown(position_size, current_drawdown)
            drawdown_adjustment = position_size / original_position if original_position > 0 else 1.0
            drawdown_explanation = f"Drawdown protection ({current_drawdown:.1f}%): {drawdown_adjustment:.2f}x"
        else:
            drawdown_explanation = "No drawdown adjustment needed"
        
        # 8. Convert to appropriate lot size for forex
        lot_size = self._convert_to_lots(position_size, symbol, asset_class)
        
        # Ensure lot size is within allowed range
        lot_size = max(self.base_lot_size, min(lot_size, self.max_lot_size))
        
        return {
            "position_size": position_size,
            "lot_size": lot_size,
            "risk_amount": risk_amount,
            "base_risk_percent": self.risk_per_trade * 100,
            "adjusted_risk_percent": (risk_amount / account_balance) * 100,
            "stop_loss_pips": stop_loss_pips,
            "stop_loss_amount": stop_loss_amount,
            "regime_adjustment": regime_adjustment,
            "progressive_risk_ratio": progressive_risk_ratio,
            "active_risk_profile": active_risk_profile.value,
            "explanations": {
                "base": f"Base risk: {self.risk_per_trade*100:.1f}% of ${account_balance:.2f}",
                "progressive": progressive_explanation,
                "profile": f"Risk profile: {active_risk_profile.value}",
                "regime": f"Market regime ({market_regime}): {regime_adjustment:.2f}x adjustment",
                "performance": performance_explanation,
                "correlation": correlation_explanation,
                "drawdown": drawdown_explanation,
                "final": f"Final position: {lot_size:.2f} lots with ${risk_amount:.2f} at risk ({(risk_amount/account_balance)*100:.2f}%)"
            }
        }
        base_position_size = self.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_pips=stop_loss_pips,
            account_balance=account_balance,
            account_currency='USD',  # Default, can be parameterized
            volatility_factor=1.0 if volatility_state == 'medium' else 
                            0.7 if volatility_state == 'high' else 1.2  # Simple volatility adjustment
        )
        
        # Adjust for correlation if data is available
        if current_positions and correlation_matrix:
            base_position_size = self.adjust_for_portfolio_exposure(
                base_position_size,
                symbol,
                current_positions,
                correlation_matrix
            )
        
        # Adjust for drawdown
        if current_drawdown > 0:
            base_position_size = self.adjust_for_drawdown(
                base_position_size,
                current_drawdown
            )
        
        return base_position_size
