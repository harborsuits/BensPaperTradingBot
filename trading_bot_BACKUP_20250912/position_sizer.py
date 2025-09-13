"""
Position Sizing Module

This module provides various position sizing strategies to calculate appropriate trade sizes
based on account equity, risk parameters, and market conditions.
"""

import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class PositionSizing(str, Enum):
    """Position sizing strategies"""
    FIXED_DOLLAR = "fixed_dollar"          # Fixed dollar amount per trade
    FIXED_RISK = "fixed_risk"              # Fixed percentage risk of account
    FIXED_PERCENTAGE = "fixed_percentage"  # Fixed percentage of account
    KELLY_CRITERION = "kelly_criterion"    # Kelly formula for optimal position sizing
    ATR_MULTIPLE = "atr_multiple"          # Position size based on ATR volatility
    DYNAMIC = "dynamic"                    # Dynamic position sizing based on multiple factors
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Volatility-based adjustment
    TIER_BASED = "tier_based"              # Tiered sizing based on conviction
    DRAWDOWN_ADJUSTED = "drawdown_adjusted"  # Adjust size based on account drawdown
    PRE_EARNINGS = "pre_earnings"          # Adjust size for earnings events
    VIX_ADJUSTED = "vix_adjusted"          # Adjust size based on VIX level

class PositionSizer:
    """
    Calculates optimal position sizes using various strategies.
    """
    
    def __init__(self, 
                account_size: float, 
                max_risk_percent: float = 1.0,
                min_risk_percent: float = 0.25,
                max_position_percent: float = 5.0,
                default_strategy: PositionSizing = PositionSizing.FIXED_RISK,
                performance_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the position sizer.
        
        Args:
            account_size: Total account equity
            max_risk_percent: Maximum risk percentage per trade (1.0 = 1%)
            min_risk_percent: Minimum risk percentage per trade (0.25 = 0.25%)
            max_position_percent: Maximum position size as percentage of account
            default_strategy: Default position sizing strategy
            performance_data: Historical performance data for Kelly calculations
        """
        self.account_size = account_size
        self.max_risk_percent = max_risk_percent
        self.min_risk_percent = min_risk_percent
        self.max_position_percent = max_position_percent
        self.default_strategy = default_strategy
        self.performance_data = performance_data or {}
        
        # Initial values for drawdown statistics
        self.initial_equity = account_size
        self.highest_equity = account_size
        self.current_drawdown_percent = 0.0
        
        # Drawdown adjustment thresholds and factors
        self.drawdown_thresholds = {
            5.0: 0.75,  # 5% drawdown -> reduce to 75% of normal size
            10.0: 0.5,  # 10% drawdown -> reduce to 50% of normal size
            15.0: 0.25  # 15% drawdown -> reduce to 25% of normal size
        }
        
        # Correlated position risk tracking
        self.correlated_positions = {}
        self.max_correlated_risk = 2.0  # Maximum risk across correlated assets (2R)

        # Volatility adjustment settings
        self.high_atr_threshold = 1.5  # ATR > 150% of 20-day average
        self.pre_earnings_reduction = 0.5  # 50% reduction before earnings
        self.vix_scale_factor = 20.0  # Scale by factor of (20/VIX)
        self.vix_min_adjustment = 0.4  # Minimum adjustment factor for VIX (caps at 60% reduction)
        self.vix_max_adjustment = 1.5  # Maximum adjustment factor for VIX (caps at 50% increase)
        
        logger.info(f"Position sizer initialized with account size: ${account_size:,.2f}, " 
                   f"max risk: {max_risk_percent}%, default strategy: {default_strategy}")
    
    def update_account_size(self, new_account_size: float) -> None:
        """
        Update the account size used for calculations and track drawdown.
        
        Args:
            new_account_size: Updated account equity
        """
        if new_account_size <= 0:
            logger.error(f"Invalid account size: {new_account_size}")
            return
            
        # Track historical equity high water mark
        self.highest_equity = max(self.highest_equity, new_account_size)
        
        # Calculate current drawdown percentage
        if self.highest_equity > 0:
            self.current_drawdown_percent = max(0, (self.highest_equity - new_account_size) / self.highest_equity * 100)
            logger.info(f"Current drawdown: {self.current_drawdown_percent:.2f}%")
        
        self.account_size = new_account_size
        logger.info(f"Account size updated to ${new_account_size:,.2f}")
    
    def update_performance_data(self, performance_data: Dict[str, Any]) -> None:
        """
        Update performance data used for Kelly criterion calculations.
        
        Args:
            performance_data: Dictionary with performance metrics
        """
        self.performance_data = performance_data
        
        # Update drawdown tracking if provided
        if 'current_equity' in performance_data:
            self.update_account_size(performance_data['current_equity'])
        
        if 'highest_equity' in performance_data:
            self.highest_equity = performance_data['highest_equity']
            
        logger.info("Performance data updated for position sizing")
    
    def add_correlated_position(self, correlation_group: str, symbol: str, risk_amount: float) -> None:
        """
        Add a correlated position to track for group risk management.
        
        Args:
            correlation_group: Group name for correlated assets (e.g., "tech_stocks", "energy_sector")
            symbol: Trading symbol
            risk_amount: Risk amount in currency units (1R equivalent)
        """
        if correlation_group not in self.correlated_positions:
            self.correlated_positions[correlation_group] = {}
            
        self.correlated_positions[correlation_group][symbol] = risk_amount
        
        # Log total risk in this correlation group
        total_group_risk = sum(self.correlated_positions[correlation_group].values())
        logger.info(f"Added {symbol} to {correlation_group} group. Total group risk: {total_group_risk:.2f}R")
    
    def remove_correlated_position(self, correlation_group: str, symbol: str) -> None:
        """
        Remove a position from correlated group tracking.
        
        Args:
            correlation_group: Group name for correlated assets
            symbol: Trading symbol to remove
        """
        if correlation_group in self.correlated_positions and symbol in self.correlated_positions[correlation_group]:
            del self.correlated_positions[correlation_group][symbol]
            logger.info(f"Removed {symbol} from {correlation_group} correlation group")
            
            # Clean up empty groups
            if not self.correlated_positions[correlation_group]:
                del self.correlated_positions[correlation_group]
    
    def check_correlation_risk(self, correlation_group: str, risk_amount: float) -> Dict[str, Any]:
        """
        Check if adding a new position would exceed correlation risk limits.
        
        Args:
            correlation_group: Group name for correlated assets
            risk_amount: Risk amount of the new position
            
        Returns:
            Dictionary with risk assessment details
        """
        if correlation_group not in self.correlated_positions:
            return {
                "within_limits": True,
                "total_risk": risk_amount,
                "max_risk": self.max_correlated_risk,
                "remaining_risk": self.max_correlated_risk - risk_amount
            }
            
        # Calculate current total risk in this group
        current_risk = sum(self.correlated_positions[correlation_group].values())
        new_total_risk = current_risk + risk_amount
        
        # Check if new position would exceed limits
        within_limits = new_total_risk <= self.max_correlated_risk
        
        return {
            "within_limits": within_limits,
            "total_risk": new_total_risk,
            "current_risk": current_risk,
            "max_risk": self.max_correlated_risk,
            "remaining_risk": max(0, self.max_correlated_risk - current_risk),
            "positions": list(self.correlated_positions[correlation_group].keys())
        }
    
    def calculate_position_size(self, 
                               entry_price: float,
                               stop_price: float,
                               risk_percent: Optional[float] = None,
                               strategy: Optional[PositionSizing] = None,
                               atr_value: Optional[float] = None,
                               avg_atr_value: Optional[float] = None, 
                               win_rate: Optional[float] = None,
                               reward_risk_ratio: Optional[float] = None,
                               conviction_level: Optional[float] = None,
                               market_volatility: Optional[float] = None,
                               setup_quality: Optional[str] = None,
                               correlation_group: Optional[str] = None,
                               ignore_drawdown: bool = False,
                               near_earnings: bool = False,
                               vix_value: Optional[float] = None,
                               sector_rotation: bool = False,
                               **kwargs) -> Dict[str, Any]:
        """
        Calculate position size based on selected strategy and parameters.
        
        Args:
            entry_price: Planned entry price
            stop_price: Planned stop loss price
            risk_percent: Risk percentage for this specific trade
            strategy: Position sizing strategy to use
            atr_value: Average True Range for volatility-based sizing
            avg_atr_value: Average of ATR over a lookback period (e.g., 20 days)
            win_rate: Historical win rate for Kelly calculations
            reward_risk_ratio: Reward:risk ratio for Kelly calculations
            conviction_level: Trade conviction (0.0-1.0) for tier-based sizing
            market_volatility: Current market volatility metric
            setup_quality: Quality rating of setup (A+, A, B, C)
            correlation_group: Group of correlated positions this belongs to
            ignore_drawdown: Whether to ignore drawdown adjustments
            near_earnings: Whether position is being entered before earnings
            vix_value: Current VIX index value for volatility-based sizing
            sector_rotation: Whether the position is in a sector experiencing rapid rotation
            **kwargs: Additional parameters for specific sizing methods
            
        Returns:
            Dictionary with position sizing details
        """
        # Validate inputs
        if entry_price <= 0:
            return {
                "position_size": 0,
                "error": "Invalid entry price"
            }
            
        # Use default strategy if none provided
        strategy = strategy or self.default_strategy
        
        # Use default risk percentage if none provided
        if risk_percent is None:
            risk_percent = self.max_risk_percent
        
        # Adjust risk percent based on setup quality
        risk_percent = self._adjust_for_setup_quality(risk_percent, setup_quality)
        
        # Perform volatility-based adjustments
        volatility_adjustments = {}
        
        # Check if ATR is high relative to average
        if atr_value is not None and avg_atr_value is not None and avg_atr_value > 0:
            atr_ratio = atr_value / avg_atr_value
            if atr_ratio > self.high_atr_threshold:
                atr_adjustment_factor = 1.0 / atr_ratio
                risk_percent *= atr_adjustment_factor
                volatility_adjustments["high_atr"] = {
                    "atr_ratio": atr_ratio,
                    "adjustment_factor": atr_adjustment_factor,
                    "description": f"High ATR ({atr_ratio:.2f}x avg) - reduced size to {atr_adjustment_factor:.2f}x"
                }
                logger.info(f"High ATR adjustment: {atr_ratio:.2f}x average → {atr_adjustment_factor:.2f}x risk")
        
        # Reduce size if near earnings
        if near_earnings:
            risk_percent *= self.pre_earnings_reduction
            volatility_adjustments["pre_earnings"] = {
                "adjustment_factor": self.pre_earnings_reduction,
                "description": f"Pre-earnings position - reduced to {self.pre_earnings_reduction:.2f}x size"
            }
            logger.info(f"Pre-earnings adjustment: reduced risk to {risk_percent:.2f}%")
        
        # Adjust for VIX level if provided
        if vix_value is not None and vix_value > 0:
            vix_adjustment = min(self.vix_max_adjustment, max(self.vix_min_adjustment, self.vix_scale_factor / vix_value))
            risk_percent *= vix_adjustment
            volatility_adjustments["vix"] = {
                "vix_value": vix_value,
                "adjustment_factor": vix_adjustment,
                "description": f"VIX at {vix_value:.1f} - adjusted size to {vix_adjustment:.2f}x"
            }
            logger.info(f"VIX adjustment: {vix_value:.1f} → {vix_adjustment:.2f}x risk")
        
        # Reduce size for sectors experiencing rapid rotation
        if sector_rotation:
            sector_adjustment = 0.75  # Reduce to 75% size during sector rotation
            risk_percent *= sector_adjustment
            volatility_adjustments["sector_rotation"] = {
                "adjustment_factor": sector_adjustment,
                "description": "Sector experiencing rapid rotation - reduced to 0.75x size"
            }
            logger.info(f"Sector rotation adjustment: reduced risk to {risk_percent:.2f}%")
        
        # Adjust for account drawdown if enabled
        if not ignore_drawdown and self.current_drawdown_percent > 0:
            risk_percent, drawdown_adjustment = self._apply_drawdown_adjustment(risk_percent)
        else:
            drawdown_adjustment = None
        
        # Check correlation risk if applicable
        correlation_check = None
        if correlation_group:
            # We need to calculate estimated risk amount first
            estimated_risk_amount = self.account_size * (risk_percent / 100)
            correlation_check = self.check_correlation_risk(correlation_group, estimated_risk_amount)
            
            # Automatically adjust to stay within correlation limits
            if not correlation_check["within_limits"]:
                remaining_risk = correlation_check["remaining_risk"]
                if remaining_risk > 0:
                    # Adjust risk percent to fit within remaining risk allowance
                    adjusted_risk_percent = (remaining_risk / self.account_size) * 100
                    logger.info(f"Adjusting risk from {risk_percent:.2f}% to {adjusted_risk_percent:.2f}% due to correlation limits")
                    risk_percent = adjusted_risk_percent
                else:
                    logger.warning(f"No remaining risk allowance in {correlation_group} group. Consider different position.")
        
        # Calculate position size based on strategy
        try:
            if strategy == PositionSizing.FIXED_DOLLAR:
                result = self._fixed_dollar_position(
                    dollar_amount=kwargs.get('dollar_amount')
                )
                
            elif strategy == PositionSizing.FIXED_RISK:
                result = self._fixed_risk_position(
                    entry_price, stop_price, risk_percent
                )
                
            elif strategy == PositionSizing.FIXED_PERCENTAGE:
                result = self._fixed_percentage_position(
                    entry_price, percentage=kwargs.get('percentage', risk_percent)
                )
                
            elif strategy == PositionSizing.KELLY_CRITERION:
                result = self._kelly_criterion_position(
                    entry_price, stop_price, win_rate, reward_risk_ratio,
                    fractional=kwargs.get('fractional', 0.5)
                )
                
            elif strategy == PositionSizing.ATR_MULTIPLE:
                result = self._atr_multiple_position(
                    entry_price, atr_value, risk_percent,
                    atr_multiple=kwargs.get('atr_multiple', 2.0)
                )
                
            elif strategy == PositionSizing.DYNAMIC:
                result = self._dynamic_position(
                    entry_price, stop_price, risk_percent, market_volatility, conviction_level
                )
                
            elif strategy == PositionSizing.VOLATILITY_ADJUSTED:
                result = self._volatility_adjusted_position(
                    entry_price, risk_percent, 
                    volatility=market_volatility or atr_value,
                    vol_factor=kwargs.get('vol_factor', 1.0)
                )
                
            elif strategy == PositionSizing.TIER_BASED:
                result = self._tier_based_position(
                    entry_price, conviction_level, risk_percent
                )
                
            elif strategy == PositionSizing.DRAWDOWN_ADJUSTED:
                result = self._drawdown_adjusted_position(
                    entry_price, stop_price, risk_percent
                )
            
            elif strategy == PositionSizing.PRE_EARNINGS:
                result = self._pre_earnings_position(
                    entry_price, stop_price, risk_percent
                )
                
            elif strategy == PositionSizing.VIX_ADJUSTED:
                result = self._vix_adjusted_position(
                    entry_price, stop_price, risk_percent, vix_value
                )
                
            else:
                logger.warning(f"Unknown position sizing strategy: {strategy}, using fixed risk")
                result = self._fixed_risk_position(entry_price, stop_price, risk_percent)
                
            # Add drawdown adjustment info if applied
            if drawdown_adjustment:
                result["drawdown_adjustment"] = drawdown_adjustment
                
            # Add correlation check results if performed
            if correlation_check:
                result["correlation_check"] = correlation_check
                
            # Add setup quality information if provided
            if setup_quality:
                result["setup_quality"] = setup_quality
                
            # Add volatility adjustments if any were applied
            if volatility_adjustments:
                result["volatility_adjustments"] = volatility_adjustments
                
            return result
                
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {
                "position_size": 0,
                "error": f"Calculation error: {str(e)}"
            }
    
    def _adjust_for_setup_quality(self, risk_percent: float, setup_quality: Optional[str]) -> float:
        """
        Adjust risk percentage based on setup quality rating.
        
        Args:
            risk_percent: Base risk percentage
            setup_quality: Quality rating (A+, A, B, C)
            
        Returns:
            Adjusted risk percentage
        """
        if not setup_quality:
            return risk_percent
            
        # Apply adjustment factors based on setup quality
        if setup_quality == "A+":
            # A+ setup - can increase to 1.5R
            adjustment = 1.5
            logger.info(f"A+ setup quality - increasing risk to {risk_percent * adjustment:.2f}%")
        elif setup_quality == "A":
            # A setup - standard 1R
            adjustment = 1.0
        elif setup_quality == "B":
            # B setup - reduce to 0.75R
            adjustment = 0.75
            logger.info(f"B setup quality - reducing risk to {risk_percent * adjustment:.2f}%")
        elif setup_quality == "C":
            # C setup - reduce to 0.5R
            adjustment = 0.5
            logger.info(f"C setup quality - reducing risk to {risk_percent * adjustment:.2f}%")
        else:
            adjustment = 1.0
            
        return risk_percent * adjustment
    
    def _apply_drawdown_adjustment(self, risk_percent: float) -> (float, Dict[str, Any]):
        """
        Apply position size adjustment based on account drawdown levels.
        
        Args:
            risk_percent: Original risk percentage
            
        Returns:
            Tuple of (adjusted risk percentage, adjustment details dictionary)
        """
        # Find the appropriate adjustment factor based on drawdown thresholds
        adjustment_factor = 1.0
        threshold_applied = 0.0
        
        # Look for the highest threshold that has been exceeded
        for threshold, factor in sorted(self.drawdown_thresholds.items(), reverse=True):
            if self.current_drawdown_percent >= threshold:
                adjustment_factor = factor
                threshold_applied = threshold
                break
                
        # Apply the adjustment
        adjusted_risk = risk_percent * adjustment_factor
        
        # Ensure we don't go below minimum risk percent
        adjusted_risk = max(adjusted_risk, self.min_risk_percent)
        
        # Create adjustment details
        adjustment_details = {
            "original_risk_percent": risk_percent,
            "adjusted_risk_percent": adjusted_risk,
            "adjustment_factor": adjustment_factor,
            "current_drawdown": self.current_drawdown_percent,
            "threshold_applied": threshold_applied
        }
        
        if adjustment_factor < 1.0:
            logger.info(f"Applied drawdown adjustment: {self.current_drawdown_percent:.2f}% drawdown → {adjustment_factor:.2f}x risk → {adjusted_risk:.2f}%")
        
        return adjusted_risk, adjustment_details
    
    def _drawdown_adjusted_position(self, 
                                   entry_price: float, 
                                   stop_price: float,
                                   risk_percent: float) -> Dict[str, Any]:
        """
        Calculate position size with automatic drawdown adjustment.
        
        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            risk_percent: Base risk percentage
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Apply drawdown adjustment to risk percent
        adjusted_risk, drawdown_details = self._apply_drawdown_adjustment(risk_percent)
        
        # Calculate position using fixed risk with adjusted percentage
        result = self._fixed_risk_position(entry_price, stop_price, adjusted_risk)
        
        # Add drawdown adjustment details to result
        result["strategy"] = PositionSizing.DRAWDOWN_ADJUSTED
        result["base_risk_percent"] = risk_percent
        result["drawdown_adjustment"] = drawdown_details
        
        return result
    
    def _pre_earnings_position(self,
                             entry_price: float, 
                             stop_price: float,
                             risk_percent: float) -> Dict[str, Any]:
        """
        Calculate position size for pre-earnings trades with reduced risk.
        
        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            risk_percent: Base risk percentage
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Reduce position size by 50% for pre-earnings trades
        adjusted_risk = risk_percent * self.pre_earnings_reduction
        
        # Calculate position using fixed risk with adjusted percentage
        result = self._fixed_risk_position(entry_price, stop_price, adjusted_risk)
        
        # Add adjustment details to result
        result["strategy"] = PositionSizing.PRE_EARNINGS
        result["base_risk_percent"] = risk_percent
        result["adjusted_risk_percent"] = adjusted_risk
        result["earnings_adjustment_factor"] = self.pre_earnings_reduction
        
        return result
    
    def _vix_adjusted_position(self,
                             entry_price: float, 
                             stop_price: float,
                             risk_percent: float,
                             vix_value: float) -> Dict[str, Any]:
        """
        Calculate position size adjusted based on VIX level.
        
        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            risk_percent: Base risk percentage
            vix_value: Current VIX index value
            
        Returns:
            Dictionary with position size and calculation details
        """
        if vix_value <= 0:
            logger.warning("Invalid VIX value provided")
            return self._fixed_risk_position(entry_price, stop_price, risk_percent)
        
        # Calculate adjustment factor (20/VIX)
        # Higher VIX = smaller position
        vix_adjustment = min(self.vix_max_adjustment, max(self.vix_min_adjustment, self.vix_scale_factor / vix_value))
        adjusted_risk = risk_percent * vix_adjustment
        
        # Calculate position using fixed risk with adjusted percentage
        result = self._fixed_risk_position(entry_price, stop_price, adjusted_risk)
        
        # Add VIX adjustment details to result
        result["strategy"] = PositionSizing.VIX_ADJUSTED
        result["base_risk_percent"] = risk_percent
        result["adjusted_risk_percent"] = adjusted_risk
        result["vix_value"] = vix_value
        result["vix_adjustment_factor"] = vix_adjustment
        
        return result
    
    def adjust_for_psych_risk(self, position_sizing: Dict[str, Any], risk_score: float) -> Dict[str, Any]:
        """
        Adjust position size based on psychological risk score.
        
        Args:
            position_sizing: Original position sizing result
            risk_score: Psychological risk score (higher = more risk)
            
        Returns:
            Adjusted position sizing
        """
        # Save original size for reference
        result = position_sizing.copy()
        
        if "position_size" not in result or result["position_size"] <= 0:
            return result
        
        original_size = result["position_size"]
        result["original_size"] = original_size
        
        # Calculate risk adjustment factor (0.25-1.0)
        # Higher risk score = lower adjustment factor (smaller position)
        if risk_score <= 3.0:  # Low risk
            adjustment = 1.0  # No reduction
        elif risk_score <= 5.0:  # Moderate risk
            adjustment = 0.75  # 25% reduction
        elif risk_score <= 7.5:  # High risk
            adjustment = 0.5  # 50% reduction
        else:  # Extreme risk
            adjustment = 0.25  # 75% reduction
        
        # Apply adjustment
        result["position_size"] = original_size * adjustment
        result["psych_adjustment"] = adjustment
        result["risk_score"] = risk_score
        
        logger.info(f"Position size adjusted for psychological risk: {risk_score:.1f} " 
                   f"→ {adjustment:.2f}x → {result['position_size']:.2f}")
        
        return result
    
    def _fixed_dollar_position(self, dollar_amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate position size based on fixed dollar amount.
        
        Args:
            dollar_amount: Dollar amount to risk (optional)
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Use 1% of account by default if no amount specified
        if dollar_amount is None:
            dollar_amount = self.account_size * (self.max_risk_percent / 100)
        
        # Apply max position size limit
        max_dollar_amount = self.account_size * (self.max_position_percent / 100)
        if dollar_amount > max_dollar_amount:
            dollar_amount = max_dollar_amount
            logger.info(f"Dollar amount reduced to maximum allowed: ${dollar_amount:,.2f}")
        
        return {
            "position_size": dollar_amount,
            "strategy": PositionSizing.FIXED_DOLLAR,
            "dollar_amount": dollar_amount,
            "account_size": self.account_size
        }
    
    def _fixed_risk_position(self, 
                           entry_price: float, 
                           stop_price: float, 
                           risk_percent: float) -> Dict[str, Any]:
        """
        Calculate position size based on fixed percentage risk.
        
        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            risk_percent: Percentage of account to risk
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Check if stop price is provided
        if stop_price <= 0:
            logger.warning("Stop price required for fixed risk calculation")
            return {
                "position_size": 0,
                "error": "Stop price required for fixed risk calculation"
            }
            
        # Calculate price difference for risk calculation
        price_difference = abs(entry_price - stop_price)
        
        if price_difference <= 0:
            logger.warning("Entry and stop price cannot be equal")
            return {
                "position_size": 0,
                "error": "Entry and stop price cannot be equal"
            }
            
        # Calculate dollar risk amount
        dollar_risk = self.account_size * (risk_percent / 100)
        
        # Calculate shares
        shares = dollar_risk / price_difference
        
        # Calculate position value
        position_value = shares * entry_price
        
        # Apply max position size limit
        max_position_value = self.account_size * (self.max_position_percent / 100)
        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price
            logger.info(f"Position size reduced to maximum allowed: ${position_value:,.2f}")
        
        return {
            "position_size": shares,
            "value": position_value,
            "risk_amount": dollar_risk,
            "risk_percent": risk_percent,
            "price_difference": price_difference,
            "strategy": PositionSizing.FIXED_RISK,
            "account_size": self.account_size
        }
    
    def _fixed_percentage_position(self, 
                                 entry_price: float,
                                 percentage: float) -> Dict[str, Any]:
        """
        Calculate position based on fixed percentage of account.
        
        Args:
            entry_price: Current price
            percentage: Account percentage to allocate
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Limit to maximum position percentage
        actual_percentage = min(percentage, self.max_position_percent)
        
        # Calculate position value
        position_value = self.account_size * (actual_percentage / 100)
        
        # Calculate shares
        shares = position_value / entry_price if entry_price > 0 else 0
        
        return {
            "position_size": shares,
            "value": position_value,
            "percentage": actual_percentage,
            "strategy": PositionSizing.FIXED_PERCENTAGE,
            "account_size": self.account_size
        }
    
    def _kelly_criterion_position(self, 
                                entry_price: float,
                                stop_price: float,
                                win_rate: Optional[float], 
                                reward_risk_ratio: Optional[float],
                                fractional: float = 0.5) -> Dict[str, Any]:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            entry_price: Current price
            stop_price: Stop loss price
            win_rate: Probability of winning (0.0-1.0)
            reward_risk_ratio: Average winner / Average loser
            fractional: Fraction of Kelly to use (safer)
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Use performance data if specific values not provided
        if win_rate is None and 'win_rate' in self.performance_data:
            win_rate = self.performance_data['win_rate']
        
        if reward_risk_ratio is None and 'reward_risk_ratio' in self.performance_data:
            reward_risk_ratio = self.performance_data['reward_risk_ratio']
            
        # Default values if still not available
        win_rate = win_rate or 0.5
        reward_risk_ratio = reward_risk_ratio or 1.0
        
        # Calculate Kelly percentage
        if win_rate >= 1.0 or win_rate <= 0.0:
            logger.warning(f"Invalid win rate for Kelly calculation: {win_rate}")
            kelly_pct = 0
        else:
            kelly_pct = win_rate - ((1 - win_rate) / reward_risk_ratio)
        
        # Apply fractional Kelly (safer approach)
        kelly_pct = kelly_pct * fractional
        
        # Calculate position size
        if kelly_pct <= 0:
            logger.warning("Negative Kelly result: trade has negative expectancy")
            return {
                "position_size": 0,
                "strategy": PositionSizing.KELLY_CRITERION,
                "kelly_percentage": 0,
                "error": "Negative expectancy trade"
            }
        
        # Calculate position value
        position_value = self.account_size * kelly_pct
        
        # Apply max position limit
        max_position_value = self.account_size * (self.max_position_percent / 100)
        if position_value > max_position_value:
            position_value = max_position_value
            logger.info(f"Kelly position size reduced to maximum allowed: ${position_value:,.2f}")
        
        # Calculate shares
        shares = position_value / entry_price if entry_price > 0 else 0
        
        # Calculate risk amount if stop loss provided
        risk_amount = 0
        if stop_price > 0:
            risk_per_share = abs(entry_price - stop_price)
            risk_amount = risk_per_share * shares
        
        return {
            "position_size": shares,
            "value": position_value,
            "kelly_percentage": kelly_pct * 100,
            "win_rate": win_rate,
            "reward_risk_ratio": reward_risk_ratio,
            "fractional": fractional,
            "risk_amount": risk_amount,
            "strategy": PositionSizing.KELLY_CRITERION,
            "account_size": self.account_size
        }
    
    def _atr_multiple_position(self, 
                             entry_price: float, 
                             atr_value: Optional[float], 
                             risk_percent: float,
                             atr_multiple: float = 2.0) -> Dict[str, Any]:
        """
        Calculate position size based on ATR (Average True Range) volatility.
        
        Args:
            entry_price: Current price
            atr_value: Average True Range value
            risk_percent: Risk percentage
            atr_multiple: Multiple of ATR to use for stop placement
            
        Returns:
            Dictionary with position size and calculation details
        """
        if atr_value is None or atr_value <= 0:
            logger.warning("Valid ATR value required for ATR-based sizing")
            return {
                "position_size": 0,
                "error": "Valid ATR value required"
            }
        
        # Calculate dollar risk amount
        dollar_risk = self.account_size * (risk_percent / 100)
        
        # Calculate risk per share based on ATR
        risk_per_share = atr_value * atr_multiple
        
        # Calculate shares and position size
        shares = dollar_risk / risk_per_share
        position_value = shares * entry_price
        
        # Apply max position size limit
        max_position_value = self.account_size * (self.max_position_percent / 100)
        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price
            logger.info(f"ATR position size reduced to maximum allowed: ${position_value:,.2f}")
        
        return {
            "position_size": shares,
            "value": position_value,
            "risk_amount": dollar_risk,
            "risk_percent": risk_percent,
            "atr_value": atr_value,
            "atr_multiple": atr_multiple,
            "risk_per_share": risk_per_share,
            "strategy": PositionSizing.ATR_MULTIPLE,
            "account_size": self.account_size
        }
    
    def _dynamic_position(self, 
                        entry_price: float, 
                        stop_price: float, 
                        risk_percent: float,
                        market_volatility: Optional[float] = None,
                        conviction_level: Optional[float] = None) -> Dict[str, Any]:
        """
        Dynamic position sizing based on multiple factors.
        
        Args:
            entry_price: Current price
            stop_price: Stop loss price
            risk_percent: Base risk percentage
            market_volatility: Market volatility metric (0.0-1.0, higher = more volatile)
            conviction_level: Trade conviction level (0.0-1.0, higher = more conviction)
            
        Returns:
            Dictionary with position size and calculation details
        """
        # Start with base risk percentage
        adjusted_risk_pct = risk_percent
        adjustments = {}
        
        # Adjust for market volatility (reduce size in high volatility)
        if market_volatility is not None:
            normalized_volatility = min(1.0, max(0.0, market_volatility))
            volatility_factor = 1.0 - (normalized_volatility * 0.5)  # Up to 50% reduction
            adjusted_risk_pct *= volatility_factor
            
            adjustments["volatility"] = {
                "factor": volatility_factor,
                "description": f"Adjusted for market volatility: {volatility_factor:.2f}x"
            }
        
        # Adjust for conviction level (increase size for high conviction)
        if conviction_level is not None:
            normalized_conviction = min(1.0, max(0.0, conviction_level))
            conviction_factor = 0.75 + (normalized_conviction * 0.5)  # 75-125%
            adjusted_risk_pct *= conviction_factor
            
            adjustments["conviction"] = {
                "factor": conviction_factor,
                "description": f"Adjusted for conviction level: {conviction_factor:.2f}x"
            }
        
        # Add win streak adjustment if available in performance data
        if 'current_streak_type' in self.performance_data and 'current_streak_count' in self.performance_data:
            streak_type = self.performance_data['current_streak_type']
            streak_count = self.performance_data['current_streak_count']
            
            if streak_type == 'win' and streak_count >= 3:
                # Increase size on win streaks (up to 25% increase)
                streak_factor = min(1.25, 1.0 + (streak_count * 0.05))
                adjusted_risk_pct *= streak_factor
                
                adjustments["win_streak"] = {
                    "factor": streak_factor,
                    "description": f"Adjusted for win streak ({streak_count}): {streak_factor:.2f}x"
                }
            elif streak_type == 'loss' and streak_count >= 2:
                # Decrease size on loss streaks (up to 50% reduction)
                streak_factor = max(0.5, 1.0 - (streak_count * 0.1))
                adjusted_risk_pct *= streak_factor
                
                adjustments["loss_streak"] = {
                    "factor": streak_factor,
                    "description": f"Adjusted for loss streak ({streak_count}): {streak_factor:.2f}x"
                }
        
        # Calculate position with adjusted risk percentage
        result = self._fixed_risk_position(entry_price, stop_price, adjusted_risk_pct)
        
        # Add dynamic adjustment information
        result["strategy"] = PositionSizing.DYNAMIC
        result["base_risk_percent"] = risk_percent
        result["adjusted_risk_percent"] = adjusted_risk_pct
        result["adjustments"] = adjustments
        
        return result
    
    def _volatility_adjusted_position(self, 
                                    entry_price: float, 
                                    risk_percent: float,
                                    volatility: Optional[float] = None,
                                    vol_factor: float = 1.0) -> Dict[str, Any]:
        """
        Position sizing adjusted for market volatility.
        
        Args:
            entry_price: Current price
            risk_percent: Risk percentage
            volatility: Volatility metric (ATR or VIX-like)
            vol_factor: Volatility adjustment factor
            
        Returns:
            Dictionary with position size and calculation details
        """
        if volatility is None or volatility <= 0:
            logger.warning("Invalid volatility value for volatility-adjusted sizing")
            return {
                "position_size": 0,
                "error": "Invalid volatility value"
            }
        
        # Calculate volatility adjustment (inverse relationship)
        # Higher volatility = smaller position
        vol_adjustment = 1.0 / (volatility ** vol_factor)
        
        # Normalize adjustment to prevent extreme values
        vol_adjustment = max(0.25, min(2.0, vol_adjustment))
        
        # Adjust risk percentage based on volatility
        adjusted_risk_pct = risk_percent * vol_adjustment
        
        # Calculate position size based on adjusted risk
        position_value = self.account_size * (adjusted_risk_pct / 100)
        
        # Apply max position size limit
        max_position_value = self.account_size * (self.max_position_percent / 100)
        if position_value > max_position_value:
            position_value = max_position_value
            logger.info(f"Position size reduced to maximum allowed: ${position_value:,.2f}")
        
        # Calculate shares
        shares = position_value / entry_price if entry_price > 0 else 0
        
        return {
            "position_size": shares,
            "value": position_value,
            "base_risk_percent": risk_percent,
            "adjusted_risk_percent": adjusted_risk_pct,
            "volatility": volatility,
            "vol_adjustment": vol_adjustment,
            "strategy": PositionSizing.VOLATILITY_ADJUSTED,
            "account_size": self.account_size
        }
    
    def _tier_based_position(self, 
                           entry_price: float,
                           conviction_level: Optional[float], 
                           base_risk_pct: float) -> Dict[str, Any]:
        """
        Tiered position sizing based on conviction level.
        
        Args:
            entry_price: Current price
            conviction_level: Trade conviction (0.0-1.0)
            base_risk_pct: Base risk percentage
            
        Returns:
            Dictionary with position size and calculation details
        """
        if conviction_level is None:
            logger.warning("No conviction level provided for tier-based sizing")
            return {
                "position_size": 0,
                "error": "Conviction level required for tier-based sizing"
            }
        
        # Normalize conviction to 0-1 scale
        normalized_conviction = min(1.0, max(0.0, conviction_level))
        
        # Define tier levels
        if normalized_conviction < 0.3:
            tier = "low"
            tier_factor = 0.5  # 50% of base risk
        elif normalized_conviction < 0.7:
            tier = "medium"
            tier_factor = 1.0  # 100% of base risk
        else:
            tier = "high"
            tier_factor = 1.5  # 150% of base risk
        
        # Calculate adjusted risk percentage
        adjusted_risk_pct = base_risk_pct * tier_factor
        
        # Calculate position size
        position_value = self.account_size * (adjusted_risk_pct / 100)
        
        # Apply max position size limit
        max_position_value = self.account_size * (self.max_position_percent / 100)
        if position_value > max_position_value:
            position_value = max_position_value
            logger.info(f"Tier-based position size reduced to maximum allowed: ${position_value:,.2f}")
        
        # Calculate shares
        shares = position_value / entry_price if entry_price > 0 else 0
        
        return {
            "position_size": shares,
            "value": position_value,
            "conviction_level": normalized_conviction,
            "tier": tier,
            "tier_factor": tier_factor,
            "base_risk_percent": base_risk_pct,
            "adjusted_risk_percent": adjusted_risk_pct,
            "strategy": PositionSizing.TIER_BASED,
            "account_size": self.account_size
        }
    
    def calculate_shares(self, position_size: float, price: float) -> int:
        """
        Calculate number of shares based on position size and price.
        
        Args:
            position_size: Position size in currency units
            price: Price per share
            
        Returns:
            Number of shares to trade
        """
        if price <= 0:
            return 0
            
        # Calculate raw share count
        shares = position_size / price
        
        # Round down to whole shares
        return math.floor(shares)
    
    def calculate_options_contracts(self, position_size: float, price_per_contract: float) -> int:
        """
        Calculate number of options contracts based on position size.
        
        Args:
            position_size: Position size in currency units
            price_per_contract: Price per options contract
            
        Returns:
            Number of contracts to trade
        """
        if price_per_contract <= 0:
            return 0
            
        # Calculate contracts
        contracts = position_size / price_per_contract
        
        # Round down to whole contracts
        return math.floor(contracts)
    
    def _round_decimal(self, amount: float, decimals: int = 2) -> float:
        """
        Round a decimal to the specified number of places.
        
        Args:
            amount: Amount to round
            decimals: Number of decimal places
            
        Returns:
            Rounded amount
        """
        if amount == 0:
            return 0
            
        multiplier = 10 ** decimals
        return float(Decimal(str(amount)).quantize(
            Decimal('0.1') ** decimals, rounding=ROUND_HALF_UP
        )) 