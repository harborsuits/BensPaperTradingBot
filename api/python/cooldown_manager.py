"""
Cooldown Manager Module

This module implements a comprehensive system of cooldown triggers and safeguards
to protect trading accounts from emotional decision-making and excessive risk-taking.
It integrates with the position sizer to enforce restrictions when triggered.
"""

import logging
import datetime
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
import yaml
import os

# Configure logging
logger = logging.getLogger(__name__)

class CooldownLevel(str, Enum):
    """Cooldown severity levels"""
    NONE = "none"             # No cooldown active
    MILD = "mild"             # Mild restrictions (reduced size, extra checks)
    MODERATE = "moderate"     # Moderate restrictions (significant size reduction, timeouts)
    SEVERE = "severe"         # Severe restrictions (trading halt, system review)
    CRITICAL = "critical"     # Critical restrictions (extended trading suspension)

class TriggerCategory(str, Enum):
    """Categories of cooldown triggers"""
    LOSS = "loss"                 # Loss-based triggers
    BEHAVIOR = "behavior"         # Behavior-based triggers
    MARKET = "market"             # Market condition triggers
    WIN = "win"                   # Winning streak safeguards
    ADAPTIVE = "adaptive"         # Hybrid adaptive triggers
    ML = "ml"                     # Machine learning enhanced safeguards

class CooldownManager:
    """
    Manages trading cooldown periods and restrictions based on configurable triggers.
    Integrates with position sizer to enforce position size reductions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the cooldown manager.
        
        Args:
            config_path: Path to the cooldown triggers configuration file (YAML)
        """
        # Initialize state tracking
        self.active_cooldowns = {}
        self.current_level = CooldownLevel.NONE
        self.cooldown_history = []
        self.trade_history = []
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_trade_result = None
        self.daily_trades_count = 0
        self.daily_profit_loss = 0.0
        self.weekly_profit_loss = 0.0
        self.last_trade_time = None
        self.session_start_time = datetime.datetime.now()
        self.reached_daily_goal = False
        self.account_peak = 0.0
        self.current_drawdown_pct = 0.0
        self.position_sizer = None  # Will be linked later
        
        # Market state tracking
        self.vix_value = 0.0
        self.vix_change_pct = 0.0
        self.market_breadth = 0.0
        self.sector_correlations = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.enabled = True
        
        logger.info("Cooldown manager initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load cooldown trigger configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration settings
        """
        default_config = {
            "loss_triggers": {
                "consecutive_losses": {
                    "level_1": {
                        "trigger": "2 consecutive losses",
                        "actions": [
                            "Reduce next position size to 75% normal",
                            "Complete psychological checklist before next trade",
                            "15-minute timeout before new positions"
                        ],
                        "reset_condition": "1 winning trade"
                    },
                    "level_2": {
                        "trigger": "3 consecutive losses",
                        "actions": [
                            "Reduce next position size to 50% normal",
                            "Mandatory 30-minute break from screen",
                            "Tighten entry criteria (require A+ setups only)",
                            "Journal entry with loss analysis required"
                        ],
                        "reset_condition": "2 consecutive winning trades"
                    },
                    "level_3": {
                        "trigger": "4 consecutive losses",
                        "actions": [
                            "Pause trading for remainder of day",
                            "Complete full trading journal review",
                            "Strategy validation check required before resuming"
                        ],
                        "reset_condition": "New trading day following review"
                    }
                },
                "account_drawdown": {
                    "level_1": {
                        "trigger": "Daily drawdown exceeds 3% of account",
                        "actions": [
                            "Reduce all position sizes to 75% normal for remainder of day",
                            "30-minute timeout before new positions"
                        ],
                        "reset_condition": "New trading day"
                    },
                    "level_2": {
                        "trigger": "Daily drawdown exceeds 5% of account",
                        "actions": [
                            "No new positions for remainder of day",
                            "Complete loss analysis report"
                        ],
                        "reset_condition": "New trading day following review"
                    }
                }
            },
            "behavior_triggers": {
                "trade_frequency": {
                    "level_1": {
                        "trigger": "More than 3 trades in 2 hours",
                        "actions": [
                            "Minimum 30-minute cooling period between subsequent trades",
                            "Detailed entry justification required for each new trade"
                        ],
                        "reset_condition": "New trading day"
                    }
                }
            },
            "market_triggers": {
                "volatility_changes": {
                    "level_1": {
                        "trigger": "VIX increases by 10% intraday",
                        "actions": [
                            "Reduce position size to 75% normal",
                            "Widen stops by 15%"
                        ],
                        "reset_condition": "VIX stabilizes for 2 hours"
                    },
                    "level_2": {
                        "trigger": "VIX increases by 20% intraday",
                        "actions": [
                            "Reduce position size to 50% normal",
                            "No new counter-trend positions"
                        ],
                        "reset_condition": "VIX returns to within 10% of previous day's close"
                    }
                }
            }
        }
        
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded cooldown configuration from {config_path}")
                    return config
            else:
                logger.warning("Config path not provided or file not found, using default configuration")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}, using default configuration")
            return default_config
    
    def link_position_sizer(self, position_sizer) -> None:
        """
        Link the position sizer to enable position size adjustments.
        
        Args:
            position_sizer: Position sizer instance to link
        """
        self.position_sizer = position_sizer
        logger.info("Position sizer linked to cooldown manager")
    
    def update_account_metrics(self, 
                              account_size: float,
                              daily_pnl: float = 0.0,
                              weekly_pnl: float = 0.0,
                              vix_value: Optional[float] = None,
                              vix_change_pct: Optional[float] = None,
                              market_breadth: Optional[float] = None,
                              **market_data) -> None:
        """
        Update account and market metrics for trigger evaluation.
        
        Args:
            account_size: Current account size
            daily_pnl: Profit/loss for current day
            weekly_pnl: Profit/loss for current week
            vix_value: Current VIX value if available
            vix_change_pct: Percentage change in VIX
            market_breadth: Current market breadth ratio
            **market_data: Additional market metrics
        """
        # Update account peak if current size is higher
        if account_size > self.account_peak:
            self.account_peak = account_size
        
        # Calculate current drawdown
        if self.account_peak > 0:
            self.current_drawdown_pct = (self.account_peak - account_size) / self.account_peak * 100
        
        # Update P&L tracking
        self.daily_profit_loss = daily_pnl
        self.weekly_profit_loss = weekly_pnl
        
        # Update market metrics if provided
        if vix_value is not None:
            self.vix_value = vix_value
        if vix_change_pct is not None:
            self.vix_change_pct = vix_change_pct
        if market_breadth is not None:
            self.market_breadth = market_breadth
            
        # Store any additional market data
        for key, value in market_data.items():
            setattr(self, key, value)
            
        # Check for triggers based on updated metrics
        self._check_account_triggers(account_size)
        self._check_market_triggers()
        
        logger.info(f"Account metrics updated - Size: ${account_size:,.2f}, " 
                   f"Daily P&L: ${daily_pnl:,.2f}, Drawdown: {self.current_drawdown_pct:.2f}%")
    
    def register_trade(self, 
                      trade_result: str,
                      pnl_amount: float,
                      risk_amount: float,
                      r_multiple: float,
                      symbol: str,
                      setup_type: str,
                      entry_time: datetime.datetime,
                      exit_time: Optional[datetime.datetime] = None,
                      trade_duration: Optional[float] = None,
                      notes: str = "") -> Dict[str, Any]:
        """
        Register a completed trade and check for triggered cooldowns.
        
        Args:
            trade_result: 'win' or 'loss'
            pnl_amount: Profit/loss amount in currency
            risk_amount: Amount risked on the trade
            r_multiple: R-multiple outcome (e.g., 2R win, -1R loss)
            symbol: Trading symbol
            setup_type: Type of setup or strategy used
            entry_time: Trade entry time
            exit_time: Trade exit time (if available)
            trade_duration: Duration of trade in minutes/hours
            notes: Additional notes about the trade
            
        Returns:
            Dictionary with triggered cooldowns and restrictions
        """
        # Create trade record
        trade = {
            "result": trade_result,
            "pnl_amount": pnl_amount,
            "risk_amount": risk_amount,
            "r_multiple": r_multiple,
            "symbol": symbol,
            "setup_type": setup_type,
            "entry_time": entry_time,
            "exit_time": exit_time or datetime.datetime.now(),
            "trade_duration": trade_duration,
            "notes": notes
        }
        
        # Update trade history
        self.trade_history.append(trade)
        self.last_trade_time = exit_time or datetime.datetime.now()
        self.daily_trades_count += 1
        
        # Update streak counters
        if trade_result.lower() == "win":
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.last_trade_result = "win"
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_trade_result = "loss"
        
        # Check for triggers based on new trade
        triggered_cooldowns = self._check_trade_triggers(trade)
        
        logger.info(f"Trade registered: {symbol} {trade_result.upper()}, {r_multiple}R, {len(triggered_cooldowns)} cooldowns triggered")
        
        return {
            "trade_id": len(self.trade_history),
            "triggered_cooldowns": triggered_cooldowns,
            "current_cooldown_level": self.current_level,
            "trading_allowed": self.is_trading_allowed(),
            "position_size_adjustment": self.get_position_size_adjustment()
        }
    
    def _check_trade_triggers(self, trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if the trade triggers any cooldown conditions.
        
        Args:
            trade: Trade details dictionary
            
        Returns:
            List of triggered cooldowns
        """
        triggered = []
        
        # Check consecutive loss triggers
        if trade["result"].lower() == "loss":
            # Level 1: 2 consecutive losses
            if self.consecutive_losses == 2:
                cooldown = self._activate_cooldown(
                    "loss_triggers.consecutive_losses.level_1",
                    CooldownLevel.MILD,
                    expiration=self._get_timeout_expiration(minutes=15),
                    position_size_pct=75
                )
                triggered.append(cooldown)
                
            # Level 2: 3 consecutive losses
            elif self.consecutive_losses == 3:
                cooldown = self._activate_cooldown(
                    "loss_triggers.consecutive_losses.level_2",
                    CooldownLevel.MODERATE,
                    expiration=self._get_timeout_expiration(minutes=30),
                    position_size_pct=50,
                    require_setup_quality="A+"
                )
                triggered.append(cooldown)
                
            # Level 3: 4 consecutive losses
            elif self.consecutive_losses >= 4:
                cooldown = self._activate_cooldown(
                    "loss_triggers.consecutive_losses.level_3",
                    CooldownLevel.SEVERE,
                    expiration=self._get_end_of_day_expiration(),
                    position_size_pct=0  # No trading (0% size)
                )
                triggered.append(cooldown)
            
            # Check single loss magnitude triggers
            r_multiple = abs(trade["r_multiple"])
            
            if r_multiple >= 1.5 and r_multiple < 2.0:
                cooldown = self._activate_cooldown(
                    "loss_triggers.single_loss_magnitude.level_1",
                    CooldownLevel.MILD,
                    expiration=self._get_timeout_expiration(minutes=20),
                    position_size_pct=75
                )
                triggered.append(cooldown)
                
            elif r_multiple >= 2.0 and r_multiple < 3.0:
                cooldown = self._activate_cooldown(
                    "loss_triggers.single_loss_magnitude.level_2",
                    CooldownLevel.MODERATE,
                    expiration=self._get_timeout_expiration(minutes=60),
                    position_size_pct=50
                )
                triggered.append(cooldown)
                
            elif r_multiple >= 3.0:
                cooldown = self._activate_cooldown(
                    "loss_triggers.single_loss_magnitude.level_3",
                    CooldownLevel.SEVERE,
                    expiration=self._get_end_of_day_expiration(),
                    position_size_pct=0  # No trading (0% size)
                )
                triggered.append(cooldown)
        
        # Check winning streak triggers
        elif trade["result"].lower() == "win":
            if self.consecutive_wins == 3:
                cooldown = self._activate_cooldown(
                    "win_triggers.winning_streaks.level_1",
                    CooldownLevel.MILD,
                    expiration=None,  # Until a loss occurs
                    position_size_pct=100,  # No reduction, just extra checks
                    extra_confirmation=True
                )
                triggered.append(cooldown)
                
            elif self.consecutive_wins == 5:
                cooldown = self._activate_cooldown(
                    "win_triggers.winning_streaks.level_2",
                    CooldownLevel.MILD,
                    expiration=None,  # Until a loss occurs
                    max_position_cap=True,
                    review_required=True
                )
                triggered.append(cooldown)
        
        # Check behavioral triggers
        self._check_behavioral_triggers(trade, triggered)
        
        return triggered
    
    def _check_behavioral_triggers(self, trade: Dict[str, Any], triggered: List[Dict[str, Any]]) -> None:
        """
        Check for behavioral triggers related to trading frequency and patterns.
        
        Args:
            trade: Current trade details
            triggered: List to append triggered cooldowns to
        """
        # Check for overtrading (more than 3 trades in 2 hours)
        if len(self.trade_history) >= 3:
            recent_trades = self.trade_history[-3:]
            newest_time = recent_trades[-1]["entry_time"]
            oldest_time = recent_trades[0]["entry_time"]
            
            time_diff = (newest_time - oldest_time).total_seconds() / 3600  # in hours
            
            if time_diff <= 2.0:
                cooldown = self._activate_cooldown(
                    "behavior_triggers.trade_frequency.level_1",
                    CooldownLevel.MILD,
                    expiration=self._get_timeout_expiration(minutes=30),
                    min_time_between_trades=30,  # 30 minutes minimum between trades
                    detailed_justification=True
                )
                triggered.append(cooldown)
        
        # Check for potential revenge trading (entry within 10 minutes after a loss)
        if len(self.trade_history) >= 2:
            prev_trade = self.trade_history[-2]
            current_trade = self.trade_history[-1]
            
            if (prev_trade["result"].lower() == "loss" and 
                (current_trade["entry_time"] - prev_trade["exit_time"]).total_seconds() < 600):  # 10 minutes
                
                cooldown = self._activate_cooldown(
                    "behavior_triggers.emotional_indicators.level_3",
                    CooldownLevel.MODERATE,
                    expiration=self._get_timeout_expiration(minutes=60),
                    position_size_pct=50,
                    psych_assessment=True
                )
                triggered.append(cooldown)
    
    def _check_account_triggers(self, account_size: float) -> List[Dict[str, Any]]:
        """
        Check for account-based triggers like drawdown levels.
        
        Args:
            account_size: Current account size
            
        Returns:
            List of triggered cooldowns
        """
        triggered = []
        
        # Daily drawdown checks
        daily_drawdown_pct = -(self.daily_profit_loss / account_size) * 100 if self.daily_profit_loss < 0 else 0
        
        if daily_drawdown_pct >= 3.0 and daily_drawdown_pct < 5.0:
            cooldown = self._activate_cooldown(
                "loss_triggers.account_drawdown.level_1",
                CooldownLevel.MODERATE,
                expiration=self._get_end_of_day_expiration(),
                position_size_pct=75,
                timeout_minutes=30
            )
            triggered.append(cooldown)
            
        elif daily_drawdown_pct >= 5.0:
            cooldown = self._activate_cooldown(
                "loss_triggers.account_drawdown.level_2",
                CooldownLevel.SEVERE,
                expiration=self._get_end_of_day_expiration(),
                position_size_pct=0,  # No trading for rest of day
                report_required=True
            )
            triggered.append(cooldown)
        
        # Overall account drawdown checks
        if self.current_drawdown_pct >= 7.0 and self.current_drawdown_pct < 10.0:
            cooldown = self._activate_cooldown(
                "loss_triggers.account_drawdown.level_3",
                CooldownLevel.MODERATE,
                expiration=self._get_end_of_week_expiration(),
                max_risk_pct=0.5,  # Reduce to 0.5% risk per trade
                restricted_strategies=True
            )
            triggered.append(cooldown)
            
        elif self.current_drawdown_pct >= 10.0:
            cooldown = self._activate_cooldown(
                "loss_triggers.account_drawdown.level_4",
                CooldownLevel.CRITICAL,
                expiration=self._get_expiration_days(14),  # Two weeks
                max_risk_pct=0.25,  # Reduce to 0.25% risk per trade
                system_review=True
            )
            triggered.append(cooldown)
        
        return triggered
    
    def _check_market_triggers(self) -> List[Dict[str, Any]]:
        """
        Check for market condition based triggers.
        
        Returns:
            List of triggered cooldowns
        """
        triggered = []
        
        # VIX-based volatility triggers
        if self.vix_change_pct >= 10.0 and self.vix_change_pct < 20.0:
            cooldown = self._activate_cooldown(
                "market_triggers.volatility_changes.level_1",
                CooldownLevel.MILD,
                expiration=self._get_timeout_expiration(hours=2),
                position_size_pct=75,
                widen_stops_pct=15
            )
            triggered.append(cooldown)
            
        elif self.vix_change_pct >= 20.0:
            cooldown = self._activate_cooldown(
                "market_triggers.volatility_changes.level_2",
                CooldownLevel.MODERATE,
                expiration=self._get_market_condition_expiration("vix", 10.0),
                position_size_pct=50,
                no_counter_trend=True
            )
            triggered.append(cooldown)
        
        # Absolute VIX level triggers
        if self.vix_value >= 30.0 and self.vix_value < 40.0:
            cooldown = self._activate_cooldown(
                "market_triggers.volatility_changes.level_3",
                CooldownLevel.MODERATE,
                expiration=self._get_market_condition_expiration("vix_below", 25.0),
                position_size_pct=50,
                high_quality_only=True,
                min_reward_risk=3.0
            )
            triggered.append(cooldown)
            
        elif self.vix_value >= 40.0:
            cooldown = self._activate_cooldown(
                "market_triggers.volatility_changes.level_4",
                CooldownLevel.SEVERE,
                expiration=self._get_market_condition_expiration("vix_below_days", 30.0, 2),
                position_size_pct=25,
                hedged_only=True
            )
            triggered.append(cooldown)
        
        # Market breadth trigger
        if self.market_breadth < 0.5:
            # Check if this has been true for 3 consecutive days
            # This would require more historical data tracking than we have in this example
            # For now, just trigger based on current value
            cooldown = self._activate_cooldown(
                "market_triggers.market_breadth",
                CooldownLevel.MODERATE,
                expiration=self._get_market_condition_expiration("breadth_above", 1.0, 2),
                reduce_long_exposure=25,
                add_hedges=True
            )
            triggered.append(cooldown)
            
        return triggered
    
    def _activate_cooldown(self, 
                          trigger_id: str, 
                          level: CooldownLevel,
                          expiration: Optional[datetime.datetime] = None,
                          **parameters) -> Dict[str, Any]:
        """
        Activate a cooldown restriction.
        
        Args:
            trigger_id: Identifier for the trigger
            level: Cooldown severity level
            expiration: Expiration time for the cooldown
            **parameters: Additional parameters for the cooldown
            
        Returns:
            Cooldown details dictionary
        """
        # Create cooldown record
        cooldown = {
            "id": f"cooldown_{len(self.cooldown_history) + 1}",
            "trigger_id": trigger_id,
            "level": level,
            "start_time": datetime.datetime.now(),
            "expiration": expiration,
            "parameters": parameters,
            "status": "active"
        }
        
        # Store in active cooldowns
        self.active_cooldowns[cooldown["id"]] = cooldown
        
        # Add to history
        self.cooldown_history.append(cooldown)
        
        # Update current cooldown level (always use the most restrictive)
        self._update_cooldown_level()
        
        logger.info(f"Activated cooldown: {trigger_id}, level: {level}, expires: {expiration}")
        logger.info(f"Cooldown parameters: {parameters}")
        
        return cooldown
    
    def _update_cooldown_level(self) -> None:
        """Update the current cooldown level based on active cooldowns."""
        level_priority = {
            CooldownLevel.CRITICAL: 4,
            CooldownLevel.SEVERE: 3,
            CooldownLevel.MODERATE: 2,
            CooldownLevel.MILD: 1,
            CooldownLevel.NONE: 0
        }
        
        highest_level = CooldownLevel.NONE
        highest_priority = 0
        
        # Find the highest priority level among active cooldowns
        for cooldown in self.active_cooldowns.values():
            level = cooldown["level"]
            priority = level_priority.get(level, 0)
            
            if priority > highest_priority:
                highest_priority = priority
                highest_level = level
        
        self.current_level = highest_level
    
    def _get_timeout_expiration(self, minutes: int = 0, hours: int = 0) -> datetime.datetime:
        """
        Calculate expiration time for a timeout.
        
        Args:
            minutes: Minutes to timeout
            hours: Hours to timeout
            
        Returns:
            Expiration datetime
        """
        delta = datetime.timedelta(minutes=minutes, hours=hours)
        return datetime.datetime.now() + delta
    
    def _get_end_of_day_expiration(self) -> datetime.datetime:
        """
        Get expiration time for end of current trading day.
        
        Returns:
            Expiration datetime
        """
        now = datetime.datetime.now()
        end_of_day = datetime.datetime(now.year, now.month, now.day, 23, 59, 59)
        return end_of_day
    
    def _get_end_of_week_expiration(self) -> datetime.datetime:
        """
        Get expiration time for end of current trading week.
        
        Returns:
            Expiration datetime
        """
        now = datetime.datetime.now()
        days_to_friday = 4 - now.weekday() if now.weekday() < 5 else 11 - now.weekday()
        if days_to_friday < 0:
            days_to_friday += 7
        end_of_week = now + datetime.timedelta(days=days_to_friday, hours=23-now.hour, minutes=59-now.minute, seconds=59-now.second)
        return end_of_week
    
    def _get_expiration_days(self, days: int) -> datetime.datetime:
        """
        Get expiration time N days in the future.
        
        Args:
            days: Number of days until expiration
            
        Returns:
            Expiration datetime
        """
        return datetime.datetime.now() + datetime.timedelta(days=days)
    
    def _get_market_condition_expiration(self, 
                                        condition_type: str, 
                                        threshold: float,
                                        consecutive_days: int = 1) -> Optional[datetime.datetime]:
        """
        Get expiration based on market condition.
        
        Args:
            condition_type: Type of market condition ('vix', 'breadth_above', etc.)
            threshold: Threshold value for condition
            consecutive_days: Number of consecutive days required
            
        Returns:
            Expiration datetime or None for condition-based expiration
        """
        # For condition-based expirations, return None and handle in check_expirations
        # In a real implementation, you might want to store the condition details
        # For simplicity, just use a far future date
        return datetime.datetime.now() + datetime.timedelta(days=30)
    
    def check_expirations(self) -> List[str]:
        """
        Check and process expired cooldowns.
        
        Returns:
            List of expired cooldown IDs
        """
        expired = []
        now = datetime.datetime.now()
        
        for cooldown_id, cooldown in list(self.active_cooldowns.items()):
            # Check time-based expiration
            if cooldown["expiration"] and now >= cooldown["expiration"]:
                cooldown["status"] = "expired"
                expired.append(cooldown_id)
                del self.active_cooldowns[cooldown_id]
                logger.info(f"Cooldown expired: {cooldown_id}")
            
            # Check condition-based expiration (would need more complex logic in real implementation)
            # This is simplified here
            if "trigger_id" in cooldown and "consecutive_losses" in cooldown["trigger_id"]:
                if self.consecutive_wins >= 1:
                    cooldown["status"] = "reset"
                    expired.append(cooldown_id)
                    del self.active_cooldowns[cooldown_id]
                    logger.info(f"Cooldown reset by win: {cooldown_id}")
        
        # Update current level after processing expirations
        self._update_cooldown_level()
        
        return expired
    
    def get_position_size_adjustment(self) -> Dict[str, Any]:
        """
        Get the current position size adjustment based on active cooldowns.
        
        Returns:
            Dictionary with position size adjustment details
        """
        # Start with default (no adjustment)
        adjustment = {
            "position_size_pct": 100,  # 100% = no reduction
            "max_risk_pct": None,      # Use default risk percentage
            "max_position_cap": False, # No special cap
            "reason": "No active cooldowns"
        }
        
        if not self.active_cooldowns:
            return adjustment
        
        # Find the most restrictive position size adjustment
        min_position_size = 100
        min_risk_pct = None
        reasons = []
        
        for cooldown in self.active_cooldowns.values():
            params = cooldown["parameters"]
            
            # Check for position size restriction
            if "position_size_pct" in params and params["position_size_pct"] < min_position_size:
                min_position_size = params["position_size_pct"]
                reasons.append(f"{cooldown['trigger_id']}: {min_position_size}%")
            
            # Check for max risk restriction
            if "max_risk_pct" in params:
                if min_risk_pct is None or params["max_risk_pct"] < min_risk_pct:
                    min_risk_pct = params["max_risk_pct"]
                    reasons.append(f"{cooldown['trigger_id']}: {min_risk_pct}% max risk")
            
            # Check for position cap
            if "max_position_cap" in params and params["max_position_cap"]:
                adjustment["max_position_cap"] = True
                reasons.append(f"{cooldown['trigger_id']}: position cap")
        
        # Update adjustment with most restrictive values
        adjustment["position_size_pct"] = min_position_size
        if min_risk_pct is not None:
            adjustment["max_risk_pct"] = min_risk_pct
        
        if reasons:
            adjustment["reason"] = ", ".join(reasons)
        
        return adjustment
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed based on active cooldowns.
        
        Returns:
            Boolean indicating if trading is allowed
        """
        # If not enabled at all, trading is not allowed
        if not self.enabled:
            return False
        
        # Check each active cooldown
        for cooldown in self.active_cooldowns.values():
            params = cooldown["parameters"]
            
            # Position size of 0 means trading is halted
            if "position_size_pct" in params and params["position_size_pct"] == 0:
                logger.info(f"Trading halted due to cooldown: {cooldown['trigger_id']}")
                return False
            
            # Level 3 or higher cooldowns might halt trading
            if cooldown["level"] in [CooldownLevel.SEVERE, CooldownLevel.CRITICAL]:
                if "allow_trading" not in params or not params["allow_trading"]:
                    logger.info(f"Trading halted due to {cooldown['level']} cooldown: {cooldown['trigger_id']}")
                    return False
        
        return True
    
    def get_trade_restrictions(self) -> Dict[str, Any]:
        """
        Get all current trade restrictions and requirements.
        
        Returns:
            Dictionary with all active restrictions
        """
        restrictions = {
            "trading_allowed": self.is_trading_allowed(),
            "position_sizing": self.get_position_size_adjustment(),
            "cooldown_level": self.current_level,
            "active_cooldowns": len(self.active_cooldowns),
            "restrictions": {
                "setup_quality_minimum": None,
                "min_reward_risk": None,
                "restricted_strategies": False,
                "no_counter_trend": False,
                "min_time_between_trades": None,
                "widen_stops_pct": None,
                "timeout_minutes": None,
                "detailed_justification": False,
                "psych_assessment": False,
                "extra_confirmation": False,
                "system_review": False,
                "report_required": False,
                "hedged_only": False
            }
        }
        
        # Combine all active restrictions
        for cooldown in self.active_cooldowns.values():
            params = cooldown["parameters"]
            
            # Setup quality requirement
            if "require_setup_quality" in params:
                restrictions["restrictions"]["setup_quality_minimum"] = params["require_setup_quality"]
            
            # Minimum reward:risk requirement
            if "min_reward_risk" in params:
                current = restrictions["restrictions"]["min_reward_risk"]
                if current is None or params["min_reward_risk"] > current:
                    restrictions["restrictions"]["min_reward_risk"] = params["min_reward_risk"]
            
            # Strategy restrictions
            if "restricted_strategies" in params and params["restricted_strategies"]:
                restrictions["restrictions"]["restricted_strategies"] = True
            
            # Counter-trend prohibition
            if "no_counter_trend" in params and params["no_counter_trend"]:
                restrictions["restrictions"]["no_counter_trend"] = True
            
            # Time between trades
            if "min_time_between_trades" in params:
                current = restrictions["restrictions"]["min_time_between_trades"]
                if current is None or params["min_time_between_trades"] > current:
                    restrictions["restrictions"]["min_time_between_trades"] = params["min_time_between_trades"]
            
            # Stop adjustments
            if "widen_stops_pct" in params:
                current = restrictions["restrictions"]["widen_stops_pct"]
                if current is None or params["widen_stops_pct"] > current:
                    restrictions["restrictions"]["widen_stops_pct"] = params["widen_stops_pct"]
            
            # Timeout duration
            if "timeout_minutes" in params:
                current = restrictions["restrictions"]["timeout_minutes"]
                if current is None or params["timeout_minutes"] > current:
                    restrictions["restrictions"]["timeout_minutes"] = params["timeout_minutes"]
            
            # Various flags
            for flag in ["detailed_justification", "psych_assessment", "extra_confirmation", 
                         "system_review", "report_required", "hedged_only"]:
                if flag in params and params[flag]:
                    restrictions["restrictions"][flag] = True
        
        return restrictions
    
    def adjust_position_size(self, original_position_size: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply cooldown restrictions to a position size calculation.
        
        Args:
            original_position_size: Original position sizing result
            
        Returns:
            Adjusted position sizing result
        """
        if not self.enabled or not original_position_size:
            return original_position_size
        
        # Get current adjustments
        adjustments = self.get_position_size_adjustment()
        
        # Create a copy of the original result
        result = original_position_size.copy()
        
        # Store original values for reference
        result["original_size"] = result.get("position_size", 0)
        result["original_risk_percent"] = result.get("risk_percent", 0)
        
        # Apply position size percentage adjustment
        if "position_size" in result and adjustments["position_size_pct"] < 100:
            adjustment_factor = adjustments["position_size_pct"] / 100.0
            result["position_size"] = result["position_size"] * adjustment_factor
            result["cooldown_adjustment_factor"] = adjustment_factor
            result["cooldown_adjustment_reason"] = adjustments["reason"]
            
            # If there's a position value, adjust that too
            if "value" in result:
                result["value"] = result["value"] * adjustment_factor
        
        # Apply max risk percentage adjustment if specified
        if "risk_percent" in result and adjustments["max_risk_pct"] is not None:
            if result["risk_percent"] > adjustments["max_risk_pct"]:
                result["risk_percent"] = adjustments["max_risk_pct"]
                result["cooldown_max_risk_applied"] = True
                
                # Recalculate position size if we have the necessary information
                if all(k in result for k in ["account_size", "price_difference"]) and result["price_difference"] > 0:
                    dollar_risk = result["account_size"] * (result["risk_percent"] / 100)
                    result["position_size"] = dollar_risk / result["price_difference"]
                    result["value"] = result["position_size"] * result.get("entry_price", 0)
        
        # Add cooldown information to result
        result["cooldown_level"] = self.current_level
        result["cooldown_active"] = len(self.active_cooldowns) > 0
        result["trading_allowed"] = self.is_trading_allowed()
        
        return result
    
    def integrate_with_position_sizer(self):
        """
        Integrate the cooldown manager with the linked position sizer.
        This method adds a wrapper to the sizer's calculate_position_size method.
        """
        if not self.position_sizer:
            logger.warning("Cannot integrate with position sizer - no sizer linked")
            return False
        
        # Store reference to original method
        original_calculate = self.position_sizer.calculate_position_size
        
        # Define wrapper function
        def wrapped_calculate_position_size(*args, **kwargs):
            # Call original method
            result = original_calculate(*args, **kwargs)
            
            # Apply cooldown adjustments
            return self.adjust_position_size(result)
        
        # Replace method with wrapped version
        self.position_sizer.calculate_position_size = wrapped_calculate_position_size
        
        logger.info("Successfully integrated cooldown manager with position sizer")
        return True 