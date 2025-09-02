"""
Position Sizing Module

This module provides advanced position sizing strategies for trade management
with emphasis on intelligent risk management and capital preservation.
"""

import math
import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import os

logger = logging.getLogger(__name__)

class SizingMethod(str, Enum):
    """Enumeration of position sizing methods"""
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_SHARES = "fixed_shares"
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_RISK = "fixed_risk"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    DYNAMIC = "dynamic"
    SCALED = "scaled"


class PositionSizer:
    """
    Calculates position sizes for trades based on various sizing methods
    and risk management parameters.
    """
    
    def __init__(self,
                account_size: float,
                max_risk_percent: float = 1.0,
                max_position_percent: float = 5.0,
                default_method: str = "fixed_risk",
                journal_dir: str = "journal"):
        """
        Initialize the position sizer.
        
        Args:
            account_size: Current account size/capital in currency units
            max_risk_percent: Maximum risk per trade as a percentage (1.0 = 1%)
            max_position_percent: Maximum position size as percentage of account
            default_method: Default position sizing method
            journal_dir: Directory to store position sizing history
        """
        self.account_size = account_size
        self.max_risk_percent = max_risk_percent
        self.max_position_percent = max_position_percent
        self.default_method = default_method
        self.journal_dir = journal_dir
        
        # Create directory if it doesn't exist
        os.makedirs(journal_dir, exist_ok=True)
        
        # Load historical position sizing data
        self.position_history = self._load_position_history()
        
        logger.info(f"Position Sizer initialized with account size: ${account_size:,.2f}")
    
    def _load_position_history(self) -> List[Dict[str, Any]]:
        """
        Load position sizing history from journal file.
        
        Returns:
            List of position sizing history entries
        """
        history_file = os.path.join(self.journal_dir, "position_history.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading position history: {str(e)}")
                return []
        else:
            # Create empty history file
            with open(history_file, 'w') as f:
                json.dump([], f)
            return []
    
    def _save_position_history(self, position_data: Dict[str, Any]) -> None:
        """
        Save position sizing data to history.
        
        Args:
            position_data: Position sizing details to save
        """
        self.position_history.append(position_data)
        
        # Keep only the last 1000 entries
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
            
        # Save to file
        history_file = os.path.join(self.journal_dir, "position_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.position_history, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving position history: {str(e)}")
    
    def update_account_size(self, new_size: float) -> None:
        """
        Update the account size used for position calculations.
        
        Args:
            new_size: New account size in currency units
        """
        if new_size <= 0:
            logger.error(f"Invalid account size: {new_size}")
            return
            
        old_size = self.account_size
        self.account_size = new_size
        logger.info(f"Account size updated from ${old_size:,.2f} to ${new_size:,.2f}")
        
    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_price: float = 0.0,
                               risk_percent: Optional[float] = None,
                               method: Optional[str] = None,
                               atr_value: Optional[float] = None,
                               win_rate: Optional[float] = None,
                               avg_win_loss_ratio: Optional[float] = None,
                               psych_adjustment: Optional[float] = None,
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate position size based on selected method and parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_price: Planned stop loss price (required for risk-based sizing)
            risk_percent: Risk percentage override (default uses max_risk_percent)
            method: Position sizing method (uses default if None)
            atr_value: Average True Range for volatility-based sizing
            win_rate: Historical win rate for Kelly Criterion
            avg_win_loss_ratio: Ratio of average win to average loss
            psych_adjustment: Psychological adjustment factor (0.0-1.0, 1.0 = no adjustment)
            context: Additional context data
            
        Returns:
            Dictionary with position sizing details
        """
        # Validate inputs
        if entry_price <= 0:
            return self._error_result("Invalid entry price", symbol)
            
        # Default method if not specified
        method = method or self.default_method
        
        # Default risk_percent if not specified
        if risk_percent is None:
            risk_percent = self.max_risk_percent
        
        # Create result dictionary with common values
        result = {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "account_size": self.account_size,
            "method": method,
            "timestamp": self._get_timestamp()
        }
        
        # Different calculation methods
        try:
            if method == SizingMethod.FIXED_DOLLAR:
                self._calculate_fixed_dollar(result, risk_percent)
            elif method == SizingMethod.FIXED_SHARES:
                self._calculate_fixed_shares(result, context)
            elif method == SizingMethod.FIXED_PERCENTAGE:
                self._calculate_fixed_percentage(result, risk_percent)
            elif method == SizingMethod.FIXED_RISK:
                self._calculate_fixed_risk(result, risk_percent)
            elif method == SizingMethod.VOLATILITY_ADJUSTED:
                self._calculate_volatility_adjusted(result, atr_value, risk_percent)
            elif method == SizingMethod.KELLY_CRITERION:
                self._calculate_kelly(result, win_rate, avg_win_loss_ratio, risk_percent)
            elif method == SizingMethod.DYNAMIC:
                self._calculate_dynamic(result, context, risk_percent)
            elif method == SizingMethod.SCALED:
                self._calculate_scaled(result, context, risk_percent)
            else:
                # Default to fixed risk if method not recognized
                logger.warning(f"Unrecognized sizing method '{method}', using fixed risk")
                self._calculate_fixed_risk(result, risk_percent)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self._error_result(f"Calculation error: {str(e)}", symbol)
        
        # Apply psychological adjustment if provided
        if psych_adjustment is not None:
            self._apply_psych_adjustment(result, psych_adjustment)
        
        # Apply maximum position size constraint
        self._apply_max_position_constraint(result)
        
        # Round shares to appropriate precision
        self._round_shares(result)
        
        # Save position sizing history
        self._save_position_history(result.copy())
        
        return result
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _error_result(self, error_message: str, symbol: str) -> Dict[str, Any]:
        """Create an error result dictionary"""
        return {
            "symbol": symbol,
            "error": error_message,
            "position_size": 0,
            "position_value": 0,
            "timestamp": self._get_timestamp()
        }
    
    def _calculate_fixed_dollar(self, result: Dict[str, Any], risk_percent: float) -> None:
        """
        Calculate position size based on fixed dollar amount.
        
        Args:
            result: Result dictionary to update
            risk_percent: Risk percentage (used to determine dollar amount)
        """
        # Calculate dollar amount to risk
        dollar_amount = self.account_size * (risk_percent / 100)
        
        # Calculate shares
        shares = dollar_amount / result["entry_price"]
        
        # Update result
        result["position_size"] = shares
        result["position_value"] = shares * result["entry_price"]
        result["risk_amount"] = dollar_amount
        result["risk_percent"] = risk_percent
    
    def _calculate_fixed_shares(self, result: Dict[str, Any], context: Optional[Dict[str, Any]]) -> None:
        """
        Calculate position based on fixed number of shares.
        
        Args:
            result: Result dictionary to update
            context: Context containing number of shares
        """
        # Get shares from context, default to 1
        shares = 1
        if context and "shares" in context:
            shares = context["shares"]
            
        # Calculate position value and risk
        position_value = shares * result["entry_price"]
        risk_amount = 0
        
        if result["stop_price"] > 0:
            # Calculate risk if stop price is provided
            risk_amount = abs(result["entry_price"] - result["stop_price"]) * shares
            
        risk_percent = (risk_amount / self.account_size) * 100 if self.account_size > 0 else 0
        
        # Update result
        result["position_size"] = shares
        result["position_value"] = position_value
        result["risk_amount"] = risk_amount
        result["risk_percent"] = risk_percent
    
    def _calculate_fixed_percentage(self, result: Dict[str, Any], percentage: float) -> None:
        """
        Calculate position based on fixed percentage of account.
        
        Args:
            result: Result dictionary to update
            percentage: Account percentage to allocate (max is max_position_percent)
        """
        # Limit to maximum position percentage
        actual_percentage = min(percentage, self.max_position_percent)
        
        # Calculate position value
        position_value = self.account_size * (actual_percentage / 100)
        
        # Calculate shares
        shares = position_value / result["entry_price"] if result["entry_price"] > 0 else 0
        
        # Calculate risk if stop price is provided
        risk_amount = 0
        if result["stop_price"] > 0:
            risk_amount = abs(result["entry_price"] - result["stop_price"]) * shares
            
        risk_percent = (risk_amount / self.account_size) * 100 if self.account_size > 0 else 0
        
        # Update result
        result["position_size"] = shares
        result["position_value"] = position_value
        result["risk_amount"] = risk_amount
        result["risk_percent"] = risk_percent
        result["allocation_percent"] = actual_percentage
    
    def _calculate_fixed_risk(self, result: Dict[str, Any], risk_percent: float) -> None:
        """
        Calculate position size based on fixed risk percentage.
        
        Args:
            result: Result dictionary to update
            risk_percent: Percentage of account to risk
        """
        # Check if stop price is provided
        if result["stop_price"] <= 0:
            logger.warning("Stop price required for fixed risk calculation")
            result["position_size"] = 0
            result["position_value"] = 0
            result["risk_amount"] = 0
            result["risk_percent"] = 0
            result["error"] = "Stop price required for fixed risk calculation"
            return
            
        # Calculate price difference for risk calculation
        price_difference = abs(result["entry_price"] - result["stop_price"])
        
        if price_difference <= 0:
            logger.warning("Entry and stop price cannot be equal")
            result["position_size"] = 0
            result["position_value"] = 0
            result["risk_amount"] = 0
            result["risk_percent"] = 0
            result["error"] = "Entry and stop price cannot be equal"
            return
            
        # Calculate dollar risk amount
        dollar_risk = self.account_size * (risk_percent / 100)
        
        # Calculate shares
        shares = dollar_risk / price_difference
        
        # Calculate position value
        position_value = shares * result["entry_price"]
        
        # Update result
        result["position_size"] = shares
        result["position_value"] = position_value
        result["risk_amount"] = dollar_risk
        result["risk_percent"] = risk_percent
        result["price_difference"] = price_difference
    
    def _calculate_volatility_adjusted(self, 
                                      result: Dict[str, Any], 
                                      atr_value: Optional[float], 
                                      risk_percent: float) -> None:
        """
        Calculate position size based on volatility (ATR).
        
        Args:
            result: Result dictionary to update
            atr_value: Average True Range value
            risk_percent: Risk percentage
        """
        # Check if ATR is provided
        if not atr_value or atr_value <= 0:
            logger.warning("Valid ATR value required for volatility-adjusted sizing")
            # Fall back to fixed risk
            self._calculate_fixed_risk(result, risk_percent)
            result["note"] = "Fell back to fixed risk due to missing ATR"
            return
            
        # Calculate dollar risk amount
        dollar_risk = self.account_size * (risk_percent / 100)
        
        # Use ATR for risk calculation (common multiple is 2x ATR)
        atr_multiplier = 2.0
        risk_per_share = atr_value * atr_multiplier
        
        # Calculate shares
        shares = dollar_risk / risk_per_share
        
        # Calculate position value
        position_value = shares * result["entry_price"]
        
        # Update result
        result["position_size"] = shares
        result["position_value"] = position_value
        result["risk_amount"] = dollar_risk
        result["risk_percent"] = risk_percent
        result["atr_value"] = atr_value
        result["atr_multiplier"] = atr_multiplier
        result["risk_per_share"] = risk_per_share
    
    def _calculate_kelly(self, 
                        result: Dict[str, Any], 
                        win_rate: Optional[float], 
                        win_loss_ratio: Optional[float], 
                        risk_percent: float) -> None:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            result: Result dictionary to update
            win_rate: Historical win rate (0.0 to 1.0)
            win_loss_ratio: Ratio of average win to average loss
            risk_percent: Default risk percentage if Kelly cannot be calculated
        """
        # Check if required parameters are provided
        if not win_rate or not win_loss_ratio or win_rate <= 0 or win_loss_ratio <= 0:
            logger.warning("Win rate and win/loss ratio required for Kelly calculation")
            # Fall back to fixed risk
            self._calculate_fixed_risk(result, risk_percent)
            result["note"] = "Fell back to fixed risk due to missing Kelly parameters"
            return
            
        # Calculate Kelly percentage
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Common practice is to use half-Kelly for more conservative sizing
        half_kelly = kelly_pct / 2
        
        # Kelly can be negative for unfavorable bets
        if kelly_pct <= 0:
            logger.warning("Negative Kelly result, trade has negative expectancy")
            result["position_size"] = 0
            result["position_value"] = 0
            result["risk_amount"] = 0
            result["risk_percent"] = 0
            result["kelly_percentage"] = kelly_pct * 100
            result["half_kelly"] = half_kelly * 100
            result["error"] = "Negative expectancy trade"
            return
            
        # Limit kelly percentage to max position percentage
        usable_kelly = min(half_kelly, self.max_position_percent / 100)
        
        # Calculate position value
        position_value = self.account_size * usable_kelly
        
        # Calculate shares
        shares = position_value / result["entry_price"]
        
        # Calculate risk if stop price is provided
        risk_amount = 0
        risk_percent = 0
        if result["stop_price"] > 0:
            price_difference = abs(result["entry_price"] - result["stop_price"])
            risk_amount = price_difference * shares
            risk_percent = (risk_amount / self.account_size) * 100
            
        # Update result
        result["position_size"] = shares
        result["position_value"] = position_value
        result["risk_amount"] = risk_amount
        result["risk_percent"] = risk_percent
        result["kelly_percentage"] = kelly_pct * 100
        result["half_kelly"] = half_kelly * 100
        result["usable_kelly"] = usable_kelly * 100
    
    def _calculate_dynamic(self, 
                         result: Dict[str, Any], 
                         context: Optional[Dict[str, Any]], 
                         risk_percent: float) -> None:
        """
        Calculate position size using dynamic factors.
        
        Args:
            result: Result dictionary to update
            context: Context with dynamic sizing factors
            risk_percent: Base risk percentage
        """
        # Start with the base risk percentage
        adjusted_risk = risk_percent
        
        # Apply adjustment factors if context is provided
        if context:
            # Market volatility adjustment
            if "market_volatility" in context:
                vol = context["market_volatility"]
                if vol > 0.5:  # High volatility
                    adjusted_risk *= 0.7  # Reduce risk in volatile markets
                elif vol < 0.3:  # Low volatility
                    adjusted_risk *= 1.1  # Can increase risk slightly in calm markets
                    
            # Winning streak adjustment
            if "winning_streak" in context:
                streak = context["winning_streak"]
                if streak > 3:
                    # Gradually increase risk on winning streaks
                    adjusted_risk *= min(1.25, 1 + (streak * 0.05))
                    
            # Losing streak adjustment
            if "losing_streak" in context:
                streak = context["losing_streak"]
                if streak > 1:
                    # Sharply decrease risk on losing streaks
                    adjusted_risk *= max(0.5, 1 - (streak * 0.15))
                    
            # Setup quality adjustment
            if "setup_quality" in context:
                quality = context["setup_quality"]  # Expected to be 0.0-1.0
                if quality > 0.8:  # High quality setup
                    adjusted_risk *= 1.2
                elif quality < 0.5:  # Low quality setup
                    adjusted_risk *= 0.8
        
        # Calculate position using adjusted risk
        self._calculate_fixed_risk(result, adjusted_risk)
        
        # Add dynamic sizing info to result
        result["base_risk_percent"] = risk_percent
        result["adjusted_risk_percent"] = adjusted_risk
        result["adjustment_factors"] = context if context else {}
    
    def _calculate_scaled(self,
                        result: Dict[str, Any],
                        context: Optional[Dict[str, Any]],
                        risk_percent: float) -> None:
        """
        Calculate scaled position (averaging in/out).
        
        Args:
            result: Result dictionary to update
            context: Context with scaling parameters
            risk_percent: Base risk percentage
        """
        # Check if context has scaling parameters
        if not context or "scale_points" not in context or not context["scale_points"]:
            logger.warning("Scale points required for scaled position sizing")
            # Fall back to fixed risk
            self._calculate_fixed_risk(result, risk_percent)
            result["note"] = "Fell back to fixed risk due to missing scale parameters"
            return
            
        # Get scaling parameters
        scale_points = context["scale_points"]
        total_points = len(scale_points)
        
        if total_points < 2:
            logger.warning("At least 2 scale points required for scaled position sizing")
            self._calculate_fixed_risk(result, risk_percent)
            result["note"] = "Fell back to fixed risk due to insufficient scale points"
            return
            
        # Calculate total position size based on fixed risk
        self._calculate_fixed_risk(result, risk_percent)
        total_size = result["position_size"]
        
        # Distribute position across scale points
        distributed_sizes = []
        
        # Check if weights are provided
        if "scale_weights" in context and len(context["scale_weights"]) == total_points:
            weights = context["scale_weights"]
            weight_sum = sum(weights)
            
            for i, point in enumerate(scale_points):
                weight_pct = weights[i] / weight_sum
                size = total_size * weight_pct
                distributed_sizes.append({
                    "price": point,
                    "size": size,
                    "percentage": weight_pct * 100
                })
        else:
            # Equal distribution
            size_per_point = total_size / total_points
            for point in scale_points:
                distributed_sizes.append({
                    "price": point,
                    "size": size_per_point,
                    "percentage": 100 / total_points
                })
        
        # Update result
        result["total_position_size"] = total_size
        result["scale_points"] = distributed_sizes
        result["average_entry"] = sum(p["price"] * p["size"] for p in distributed_sizes) / total_size if total_size > 0 else 0
    
    def _apply_psych_adjustment(self, result: Dict[str, Any], psych_adjustment: float) -> None:
        """
        Apply psychological adjustment to position size.
        
        Args:
            result: Result dictionary to update
            psych_adjustment: Adjustment factor (0.0-1.0, 1.0 = no adjustment)
        """
        # Validate adjustment factor
        factor = min(max(psych_adjustment, 0.0), 1.0)
        
        # Store original values
        original_size = result["position_size"]
        original_value = result["position_value"]
        
        # Apply adjustment
        result["position_size"] *= factor
        result["position_value"] *= factor
        
        # If using risk-based sizing, update risk amount
        if "risk_amount" in result:
            result["risk_amount"] *= factor
            
        # Store adjustment info
        result["psych_adjustment_factor"] = factor
        result["original_position_size"] = original_size
        result["original_position_value"] = original_value
    
    def _apply_max_position_constraint(self, result: Dict[str, Any]) -> None:
        """
        Apply maximum position size constraint.
        
        Args:
            result: Result dictionary to update
        """
        # Skip if position size is zero or missing
        if "position_size" not in result or result["position_size"] <= 0:
            return
            
        # Calculate maximum position value
        max_position_value = self.account_size * (self.max_position_percent / 100)
        
        # Check if position exceeds maximum
        if result["position_value"] > max_position_value:
            # Store original values
            original_size = result["position_size"]
            original_value = result["position_value"]
            
            # Adjust to maximum
            adjustment_factor = max_position_value / original_value
            result["position_size"] *= adjustment_factor
            result["position_value"] = max_position_value
            
            # If using risk-based sizing, update risk amount
            if "risk_amount" in result:
                result["risk_amount"] *= adjustment_factor
                if "risk_percent" in result:
                    result["risk_percent"] = (result["risk_amount"] / self.account_size) * 100
                    
            # Store adjustment info
            result["max_size_adjustment_factor"] = adjustment_factor
            result["original_position_size"] = original_size
            result["original_position_value"] = original_value
    
    def _round_shares(self, result: Dict[str, Any]) -> None:
        """
        Round shares to appropriate precision.
        
        Args:
            result: Result dictionary to update
        """
        # Skip if position size is missing or zero
        if "position_size" not in result or result["position_size"] <= 0:
            return
            
        # Get price for determining rounding precision
        price = result["entry_price"]
        shares = result["position_size"]
        
        # Different rounding rules based on price
        if price >= 100:
            # Round to whole shares for high-priced stocks
            rounded_shares = math.floor(shares)
        elif price >= 10:
            # Round to 0.1 shares for medium-priced stocks
            rounded_shares = math.floor(shares * 10) / 10
        else:
            # Round to 0.01 shares for low-priced stocks
            rounded_shares = math.floor(shares * 100) / 100
            
        # Apply rounding
        if rounded_shares != shares:
            # Store original value
            result["pre_rounded_size"] = shares
            
            # Update with rounded value
            result["position_size"] = rounded_shares
            result["position_value"] = rounded_shares * price
            
            # If using risk-based sizing, update risk amount
            if "risk_amount" in result:
                if "price_difference" in result and result["price_difference"] > 0:
                    result["risk_amount"] = result["price_difference"] * rounded_shares
                elif "stop_price" in result and result["stop_price"] > 0:
                    result["risk_amount"] = abs(price - result["stop_price"]) * rounded_shares
                    
                if "risk_percent" in result:
                    result["risk_percent"] = (result["risk_amount"] / self.account_size) * 100
    
    def get_historical_sizing(self, 
                             symbol: Optional[str] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical position sizing data.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of historical position sizing entries
        """
        if symbol:
            # Filter by symbol
            filtered = [p for p in self.position_history if p.get("symbol") == symbol]
        else:
            # All symbols
            filtered = self.position_history.copy()
            
        # Sort by timestamp (newest first) and limit
        sorted_history = sorted(filtered, 
                               key=lambda x: x.get("timestamp", ""), 
                               reverse=True)
        
        return sorted_history[:limit]
    
    def analyze_position_history(self) -> Dict[str, Any]:
        """
        Analyze position sizing history for patterns and insights.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.position_history:
            return {"status": "insufficient_data", "message": "No position history available"}
            
        # Analyze by symbol
        symbol_stats = {}
        methods_used = {}
        
        for position in self.position_history:
            symbol = position.get("symbol", "unknown")
            method = position.get("method", "unknown")
            
            # Count methods used
            methods_used[method] = methods_used.get(method, 0) + 1
            
            # Skip entries with errors
            if "error" in position:
                continue
                
            # Initialize symbol stats if needed
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    "count": 0,
                    "avg_position_size": 0,
                    "avg_risk_percent": 0,
                    "total_position_value": 0,
                    "position_sizes": []
                }
                
            # Update stats
            stats = symbol_stats[symbol]
            stats["count"] += 1
            stats["total_position_value"] += position.get("position_value", 0)
            
            if "position_size" in position:
                stats["position_sizes"].append(position["position_size"])
                
            if "risk_percent" in position:
                stats["avg_risk_percent"] += position["risk_percent"]
        
        # Calculate averages
        for symbol, stats in symbol_stats.items():
            if stats["count"] > 0:
                stats["avg_position_size"] = sum(stats["position_sizes"]) / len(stats["position_sizes"]) if stats["position_sizes"] else 0
                stats["avg_risk_percent"] = stats["avg_risk_percent"] / stats["count"]
                stats["avg_position_value"] = stats["total_position_value"] / stats["count"]
        
        # Find most common method
        most_common_method = max(methods_used.items(), key=lambda x: x[1], default=(None, 0))
        
        return {
            "status": "success",
            "total_positions": len(self.position_history),
            "unique_symbols": len(symbol_stats),
            "symbol_stats": symbol_stats,
            "methods_used": methods_used,
            "most_common_method": most_common_method[0] if most_common_method[0] else None
        } 