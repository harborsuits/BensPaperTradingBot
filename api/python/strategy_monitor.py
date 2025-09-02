"""
Strategy Monitoring System

This module provides automated monitoring of trading strategies, detecting underperformance
and triggering adaptive responses to protect capital and optimize returns.
Integrates with the CooldownManager for a complete risk management system.
"""

import logging
import datetime
import json
import statistics
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class StrategyStatus(str, Enum):
    """Strategy status indicators"""
    ACTIVE = "active"                # Strategy fully active
    REDUCED_SIZE = "reduced_size"    # Active with reduced position size
    SUSPENDED = "suspended"          # Temporarily not available
    UNDER_REVIEW = "under_review"    # Under performance review
    DISABLED = "disabled"            # Permanently disabled
    RECOVERY = "recovery"            # Active but in recovery mode

class MarketEnvironment(str, Enum):
    """Market environment classifications"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"
    UNCLEAR = "unclear"

class StrategyMonitor:
    """
    Monitors trading strategy performance, detects deterioration patterns,
    and automatically adapts parameters or triggers protective measures.
    Integrates with the CooldownManager to coordinate emotional and strategy safeguards.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the strategy monitor.
        
        Args:
            config_path: Path to strategy configuration file
        """
        # Initialize tracking data structures
        self.strategies = {}
        self.strategy_trade_history = {}
        self.strategy_triggers = {}
        self.active_adaptations = {}
        self.strategy_status = {}
        self.recovery_tracking = {}
        self.market_environment = MarketEnvironment.UNCLEAR
        
        # Micro-regime tracking
        self.current_volatility = "normal"  # low, normal, high
        self.current_liquidity = "normal"   # low, normal, high
        self.current_trend = "neutral"      # bullish, bearish, neutral
        
        # Market metrics
        self.vix_value = 0.0
        self.market_breadth = 0.0
        self.sector_correlations = {}
        self.adx_value = 0.0
        
        # Load strategy configuration
        self.config = self._load_config(config_path)
        
        # Initialize strategies from config
        self._initialize_strategies()
        
        # Integration points
        self.cooldown_manager = None
        self.enabled = True
        
        logger.info("Strategy monitor initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load strategy configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration with essential strategies
        default_config = {
            "strategies": {
                "breakout_swing": {
                    "triggers": [
                        "5 consecutive losing trades",
                        "average return < -2% over 10 trades",
                        "breakouts failing to hold above breakout level",
                        "success rate drops >20% below historical average",
                        "risk:reward ratio deteriorates below 1:1.5",
                        "decreasing volume confirmation on successful breakouts",
                        "increasing time to target achievement"
                    ],
                    "action": "Flag for review, reduce size by 50%, suppress alerts temporarily",
                    "analysis_prompt": "Breakout Swing has underperformed recently. Check for macro headwinds, low volume breakouts, or news interference.",
                    "historical_win_rate": 0.65,
                    "historical_avg_return": 1.8,
                    "historical_avg_rr_ratio": 2.1,
                    "typical_hold_time": 3.5,  # in days
                    "ideal_conditions": [
                        "ADX > 25",
                        "sector relative strength positive",
                        "index in established uptrend",
                        "decreasing correlation among sector components"
                    ],
                    "adverse_conditions": [
                        "narrow range days forming",
                        "declining overall market breadth",
                        "major resistance overhead",
                        "earnings season uncertainty"
                    ]
                },
                "theta_spread": {
                    "triggers": [
                        "IV crush observed on 3 or more spreads in same week",
                        "2 spreads breached short leg within 5 days",
                        "earnings-related setups underperforming",
                        "realized volatility exceeding implied volatility by >15%",
                        "multiple standard deviation moves in underlying within spread duration"
                    ],
                    "action": "Remove from rotation for 1 week, replace with calendar spreads if volatility remains compressed",
                    "analysis_prompt": "Theta spreads appear vulnerable to unexpected IV decay or mispriced earnings. Consider shifting to defined-date-neutral strategies.",
                    "historical_win_rate": 0.78,
                    "historical_avg_return": 1.2,
                    "historical_avg_rr_ratio": 1.5,
                    "typical_hold_time": 14.0,  # in days
                    "ideal_conditions": [
                        "low VIX overall",
                        "normal volatility term structure",
                        "range-bound price action in underlying",
                        "decreasing volume/volatility trend"
                    ],
                    "adverse_conditions": [
                        "pending binary events",
                        "volatility crush already occurred",
                        "unusual options flow detected",
                        "significant skew distortion"
                    ]
                },
                "vwap_bounce": {
                    "triggers": [
                        "Volume fails confirmation on entry 3x",
                        "ATR stop hit prematurely on 2 trades in 3 days",
                        "Market internals diverge from direction",
                        "Intraday sector rotation occurring during setup formation",
                        "Lower timeframe momentum divergence present"
                    ],
                    "action": "Limit to morning session only. Use with confirmation only. Log all failed bounces.",
                    "analysis_prompt": "VWAP bounces underperforming in current market conditions. Consider time-of-day restrictions and additional confirmation factors.",
                    "historical_win_rate": 0.62,
                    "historical_avg_return": 1.1,
                    "historical_avg_rr_ratio": 1.7,
                    "typical_hold_time": 0.25,  # in days (intraday)
                    "ideal_conditions": [
                        "low overnight gap",
                        "normal to slightly above average volume",
                        "clear trend direction established",
                        "orderly price action"
                    ],
                    "adverse_conditions": [
                        "opening drive strength",
                        "news-driven gapping action",
                        "unusual pre-market activity",
                        "TICK extremes at open"
                    ]
                }
            }
        }
        
        # Load from file if provided, else use default
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    logger.info(f"Loaded strategy configuration from {config_path}")
                    return loaded_config
            except Exception as e:
                logger.error(f"Error loading strategy config: {str(e)}, using default")
                return default_config
        else:
            return default_config
    
    def _initialize_strategies(self) -> None:
        """Initialize tracking data structures for each strategy."""
        for strategy_name, strategy_config in self.config["strategies"].items():
            # Initialize trade history
            self.strategy_trade_history[strategy_name] = []
            
            # Set initial status to active
            self.strategy_status[strategy_name] = StrategyStatus.ACTIVE
            
            # Initialize triggers tracking
            self.strategy_triggers[strategy_name] = {
                "consecutive_losses": 0,
                "recent_trades": [],
                "failed_breakouts": 0,
                "volume_confirmation_failures": 0,
                "current_win_rate": strategy_config.get("historical_win_rate", 0.5),
                "current_avg_return": strategy_config.get("historical_avg_return", 1.0),
                "current_avg_rr_ratio": strategy_config.get("historical_avg_rr_ratio", 1.5),
                "triggered_conditions": set(),
                "last_review_date": None
            }
            
            # Initialize recovery tracking
            self.recovery_tracking[strategy_name] = {
                "in_recovery": False,
                "recovery_start_date": None,
                "profitable_trades_in_recovery": 0,
                "trades_in_recovery": 0,
                "recovery_conditions_met": set(),
                "remaining_conditions": set()
            }
            
            logger.info(f"Initialized strategy monitoring for {strategy_name}")
    
    def link_cooldown_manager(self, cooldown_manager) -> None:
        """
        Link to the CooldownManager for integrated risk management.
        
        Args:
            cooldown_manager: CooldownManager instance
        """
        self.cooldown_manager = cooldown_manager
        logger.info("Cooldown manager linked to strategy monitor")
    
    def get_strategy_status(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get the current status and metrics for a strategy.
        
        Args:
            strategy_name: Name of strategy to check
            
        Returns:
            Dictionary with strategy status and metrics
        """
        if strategy_name not in self.strategy_status:
            return {"error": f"Strategy {strategy_name} not found"}
        
        # Build status response
        status = {
            "name": strategy_name,
            "status": self.strategy_status[strategy_name],
            "active_adaptations": self.active_adaptations.get(strategy_name, []),
            "metrics": {
                "win_rate": self.strategy_triggers[strategy_name]["current_win_rate"],
                "avg_return": self.strategy_triggers[strategy_name]["current_avg_return"],
                "avg_rr_ratio": self.strategy_triggers[strategy_name]["current_avg_rr_ratio"],
                "consecutive_losses": self.strategy_triggers[strategy_name]["consecutive_losses"],
                "triggered_conditions": list(self.strategy_triggers[strategy_name]["triggered_conditions"])
            },
            "historical_metrics": {
                "win_rate": self.config["strategies"][strategy_name].get("historical_win_rate", 0),
                "avg_return": self.config["strategies"][strategy_name].get("historical_avg_return", 0),
                "avg_rr_ratio": self.config["strategies"][strategy_name].get("historical_avg_rr_ratio", 0)
            },
            "trade_count": len(self.strategy_trade_history[strategy_name]),
            "recovery_status": self.recovery_tracking[strategy_name] if self.recovery_tracking[strategy_name]["in_recovery"] else None
        }
        
        # Add environment suitability analysis
        status["environment_suitability"] = self._check_environment_suitability(strategy_name)
        
        return status
    
    def register_trade(self, 
                       strategy_name: str,
                       trade_result: str, 
                       pnl_amount: float,
                       risk_amount: float,
                       r_multiple: float,
                       symbols: Union[str, List[str]],
                       setup_quality: str,
                       entry_time: datetime.datetime,
                       exit_time: Optional[datetime.datetime] = None,
                       hold_time: Optional[float] = None,
                       setup_notes: Optional[Dict[str, Any]] = None,
                       **trade_details) -> Dict[str, Any]:
        """
        Register a completed trade for a strategy and check for trigger conditions.
        
        Args:
            strategy_name: Name of the strategy used
            trade_result: 'win' or 'loss'
            pnl_amount: Profit/loss amount in currency
            risk_amount: Amount risked on the trade
            r_multiple: R-multiple outcome (e.g., 2R win, -1R loss)
            symbols: Trading symbol(s) used
            setup_quality: Quality rating of setup (A+, A, B, C)
            entry_time: Trade entry time
            exit_time: Trade exit time (if available)
            hold_time: Duration of trade in days
            setup_notes: Additional notes about specific setup factors
            **trade_details: Additional trade details
            
        Returns:
            Dictionary with strategy status and triggered adaptations
        """
        # Validate strategy
        if strategy_name not in self.strategy_status:
            return {"error": f"Strategy {strategy_name} not found"}
        
        # Create trade record
        if isinstance(symbols, str):
            symbols = [symbols]
            
        trade = {
            "strategy": strategy_name,
            "result": trade_result.lower(),
            "pnl_amount": pnl_amount,
            "risk_amount": risk_amount,
            "r_multiple": r_multiple,
            "symbols": symbols,
            "setup_quality": setup_quality,
            "entry_time": entry_time,
            "exit_time": exit_time or datetime.datetime.now(),
            "hold_time": hold_time or ((exit_time or datetime.datetime.now()) - entry_time).total_seconds() / 86400,  # in days
            "setup_notes": setup_notes or {},
            "market_environment": self.market_environment,
            "vix_value": self.vix_value,
            "recovery_trade": self.recovery_tracking[strategy_name]["in_recovery"],
            **trade_details
        }
        
        # Add to trade history
        self.strategy_trade_history[strategy_name].append(trade)
        
        # Update strategy metrics
        self._update_strategy_metrics(strategy_name, trade)
        
        # Check for trigger conditions
        adaptations = []
        if trade_result.lower() == 'loss':
            adaptations = self._check_loss_triggers(strategy_name, trade)
        elif trade_result.lower() == 'win' and self.recovery_tracking[strategy_name]["in_recovery"]:
            self._check_recovery_progress(strategy_name, trade)
        
        # Update cooldown manager if linked and adaptations were triggered
        if self.cooldown_manager and adaptations:
            self._notify_cooldown_manager(strategy_name, adaptations)
        
        logger.info(f"Registered {trade_result} trade for {strategy_name}, {len(adaptations)} adaptations triggered")
        
        return {
            "strategy_status": self.get_strategy_status(strategy_name),
            "adaptations": adaptations,
            "trade_id": len(self.strategy_trade_history[strategy_name])
        }
    
    def _update_strategy_metrics(self, strategy_name: str, trade: Dict[str, Any]) -> None:
        """
        Update strategy performance metrics based on new trade.
        
        Args:
            strategy_name: Name of the strategy
            trade: Trade details dictionary
        """
        # Get strategy triggers tracking data
        triggers = self.strategy_triggers[strategy_name]
        
        # Update consecutive wins/losses
        if trade["result"] == "win":
            triggers["consecutive_losses"] = 0
        else:
            triggers["consecutive_losses"] += 1
        
        # Keep recent trades for rolling calculations (last 20)
        triggers["recent_trades"].append(trade)
        if len(triggers["recent_trades"]) > 20:
            triggers["recent_trades"].pop(0)
        
        # Calculate current win rate
        recent_results = [t["result"] == "win" for t in triggers["recent_trades"]]
        if recent_results:
            triggers["current_win_rate"] = sum(recent_results) / len(recent_results)
        
        # Calculate average return
        recent_returns = [t["r_multiple"] for t in triggers["recent_trades"]]
        if recent_returns:
            triggers["current_avg_return"] = sum(recent_returns) / len(recent_returns)
        
        # Calculate average R:R ratio (only for winning trades)
        winning_returns = [t["r_multiple"] for t in triggers["recent_trades"] if t["result"] == "win"]
        losing_returns = [abs(t["r_multiple"]) for t in triggers["recent_trades"] if t["result"] == "loss"]
        
        if winning_returns and losing_returns:
            avg_win = sum(winning_returns) / len(winning_returns)
            avg_loss = sum(losing_returns) / len(losing_returns)
            if avg_loss > 0:
                triggers["current_avg_rr_ratio"] = avg_win / avg_loss
        
        # Update specific setup quality metrics if available
        if "setup_notes" in trade and "breakout_failure" in trade["setup_notes"]:
            triggers["failed_breakouts"] += 1
            
        if "setup_notes" in trade and "volume_confirmation" in trade["setup_notes"]:
            if not trade["setup_notes"]["volume_confirmation"]:
                triggers["volume_confirmation_failures"] += 1
    
    def _check_loss_triggers(self, strategy_name: str, trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if the loss trade triggers any strategy adaptations.
        
        Args:
            strategy_name: Name of the strategy
            trade: Trade details dictionary
            
        Returns:
            List of triggered adaptations
        """
        triggered_adaptations = []
        strategy_config = self.config["strategies"].get(strategy_name, {})
        triggers = self.strategy_triggers[strategy_name]
        
        # Check consecutive losses trigger
        if triggers["consecutive_losses"] >= 5:
            adaptation = self._apply_strategy_adaptation(
                strategy_name,
                "consecutive_losses",
                "5 consecutive losing trades",
                "Flag for review, reduce size by 50%",
                {"position_size_reduction": 0.5}
            )
            triggered_adaptations.append(adaptation)
        
        # Check average return trigger
        if len(triggers["recent_trades"]) >= 10:
            last_10_returns = [t["r_multiple"] for t in triggers["recent_trades"][-10:]]
            avg_return = sum(last_10_returns) / 10
            
            if avg_return < -0.02:  # -2%
                adaptation = self._apply_strategy_adaptation(
                    strategy_name,
                    "negative_return",
                    "average return < -2% over 10 trades",
                    "Reduce size by 50%, require higher quality setups",
                    {"position_size_reduction": 0.5, "minimum_setup_quality": "A"}
                )
                triggered_adaptations.append(adaptation)
        
        # Check win rate deterioration
        historical_win_rate = strategy_config.get("historical_win_rate", 0.5)
        if triggers["current_win_rate"] < (historical_win_rate * 0.8):  # 20% below historical
            adaptation = self._apply_strategy_adaptation(
                strategy_name,
                "win_rate_drop",
                "success rate drops >20% below historical average",
                "Temporary suspension, strategy review required",
                {"suspend": True, "review_required": True}
            )
            triggered_adaptations.append(adaptation)
            
        # Check R:R ratio deterioration
        historical_rr = strategy_config.get("historical_avg_rr_ratio", 1.5)
        if triggers["current_avg_rr_ratio"] < 1.5 and historical_rr > 1.5:
            adaptation = self._apply_strategy_adaptation(
                strategy_name,
                "rr_deterioration",
                "risk:reward ratio deteriorates below 1:1.5",
                "Increase entry criteria stringency, require 2:1 minimum targets",
                {"minimum_reward_risk": 2.0, "position_size_reduction": 0.75}
            )
            triggered_adaptations.append(adaptation)
        
        # Strategy-specific checks
        if strategy_name == "breakout_swing" and trade.get("setup_notes", {}).get("breakout_failure", False):
            # Track breakout-specific issues
            triggers["failed_breakouts"] += 1
            
            if triggers["failed_breakouts"] >= 3:
                adaptation = self._apply_strategy_adaptation(
                    strategy_name,
                    "breakout_failures",
                    "breakouts failing to hold above breakout level",
                    "Require 1.5x normal volume confirmation, tighten stops",
                    {"volume_requirement_multiplier": 1.5, "tighten_stops": True}
                )
                triggered_adaptations.append(adaptation)
        
        # Volume confirmation issues
        elif strategy_name == "vwap_bounce" and trade.get("setup_notes", {}).get("volume_confirmation", True) == False:
            triggers["volume_confirmation_failures"] += 1
            
            if triggers["volume_confirmation_failures"] >= 3:
                adaptation = self._apply_strategy_adaptation(
                    strategy_name,
                    "volume_failures",
                    "Volume fails confirmation on entry 3x",
                    "Limit to morning session only, increase volume requirements",
                    {"time_restriction": "morning_only", "volume_requirement_multiplier": 1.25}
                )
                triggered_adaptations.append(adaptation)
        
        # If adaptations were triggered, set strategy to recovery mode
        if triggered_adaptations and self.strategy_status[strategy_name] != StrategyStatus.SUSPENDED:
            self._set_recovery_mode(strategy_name)
        
        return triggered_adaptations
    
    def _apply_strategy_adaptation(self, 
                                  strategy_name: str, 
                                  trigger_id: str, 
                                  trigger_description: str,
                                  action_description: str,
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply an adaptation to a strategy based on trigger.
        
        Args:
            strategy_name: Name of the strategy
            trigger_id: Identifier for the trigger type
            trigger_description: Human-readable trigger description
            action_description: Description of adaptation action
            parameters: Adaptation parameters
            
        Returns:
            Adaptation details dictionary
        """
        # Create adaptation record
        adaptation = {
            "id": f"{strategy_name}_{trigger_id}_{len(self.active_adaptations.get(strategy_name, []))}",
            "strategy": strategy_name,
            "trigger_id": trigger_id,
            "trigger_description": trigger_description,
            "action": action_description,
            "parameters": parameters,
            "start_time": datetime.datetime.now(),
            "status": "active"
        }
        
        # Store in active adaptations
        if strategy_name not in self.active_adaptations:
            self.active_adaptations[strategy_name] = []
        
        self.active_adaptations[strategy_name].append(adaptation)
        
        # Update strategy status based on adaptation
        if parameters.get("suspend", False):
            self.strategy_status[strategy_name] = StrategyStatus.SUSPENDED
        elif self.strategy_status[strategy_name] == StrategyStatus.ACTIVE:
            if parameters.get("position_size_reduction", 0) > 0:
                self.strategy_status[strategy_name] = StrategyStatus.REDUCED_SIZE
            if parameters.get("review_required", False):
                self.strategy_status[strategy_name] = StrategyStatus.UNDER_REVIEW
        
        # Add to triggered conditions
        self.strategy_triggers[strategy_name]["triggered_conditions"].add(trigger_description)
        
        logger.info(f"Applied adaptation to {strategy_name}: {trigger_description} -> {action_description}")
        
        return adaptation
    
    def _set_recovery_mode(self, strategy_name: str) -> None:
        """
        Put a strategy into recovery mode with specific criteria to return to normal.
        
        Args:
            strategy_name: Name of the strategy
        """
        # Only set recovery if not already in recovery
        if not self.recovery_tracking[strategy_name]["in_recovery"]:
            self.strategy_status[strategy_name] = StrategyStatus.RECOVERY
            
            recovery = self.recovery_tracking[strategy_name]
            recovery["in_recovery"] = True
            recovery["recovery_start_date"] = datetime.datetime.now()
            recovery["profitable_trades_in_recovery"] = 0
            recovery["trades_in_recovery"] = 0
            
            # Set recovery conditions
            recovery["recovery_conditions_met"] = set()
            recovery["remaining_conditions"] = {
                "consecutive_wins": "3 consecutive profitable trades",
                "win_rate_recovery": "win rate returns to within 10% of historical average",
                "volume_confirmation": "Confirmation volume exceeds requirements for 3 entries"
            }
            
            logger.info(f"Strategy {strategy_name} placed in recovery mode")
    
    def _check_recovery_progress(self, strategy_name: str, trade: Dict[str, Any]) -> None:
        """
        Check if a strategy in recovery mode has met criteria to return to normal.
        
        Args:
            strategy_name: Name of the strategy
            trade: Trade details dictionary
        """
        recovery = self.recovery_tracking[strategy_name]
        
        # Skip if not in recovery
        if not recovery["in_recovery"]:
            return
        
        # Update recovery counters
        recovery["trades_in_recovery"] += 1
        if trade["result"] == "win":
            recovery["profitable_trades_in_recovery"] += 1
        
        # Check consecutive wins
        if (recovery["profitable_trades_in_recovery"] >= 3 and 
            "consecutive_wins" in recovery["remaining_conditions"]):
            recovery["recovery_conditions_met"].add("consecutive_wins")
            recovery["remaining_conditions"].pop("consecutive_wins")
            logger.info(f"Strategy {strategy_name} met recovery condition: 3 consecutive wins")
        
        # Check win rate recovery
        historical_win_rate = self.config["strategies"][strategy_name].get("historical_win_rate", 0.5)
        current_win_rate = self.strategy_triggers[strategy_name]["current_win_rate"]
        
        if (current_win_rate >= historical_win_rate * 0.9 and  # within 10% of historical
            "win_rate_recovery" in recovery["remaining_conditions"]):
            recovery["recovery_conditions_met"].add("win_rate_recovery")
            recovery["remaining_conditions"].pop("win_rate_recovery")
            logger.info(f"Strategy {strategy_name} met recovery condition: win rate recovery")
        
        # Check volume confirmation if applicable
        if (strategy_name in ["breakout_swing", "vwap_bounce"] and
            trade.get("setup_notes", {}).get("sufficient_volume", False) and
            "volume_confirmation" in recovery["remaining_conditions"]):
            
            # Check if we have 3 consecutive good volume trades
            recent_trades = self.strategy_trade_history[strategy_name][-3:]
            volume_confirmed = all(t.get("setup_notes", {}).get("sufficient_volume", False) for t in recent_trades)
            
            if volume_confirmed and len(recent_trades) >= 3:
                recovery["recovery_conditions_met"].add("volume_confirmation")
                recovery["remaining_conditions"].pop("volume_confirmation")
                logger.info(f"Strategy {strategy_name} met recovery condition: volume confirmation")
        
        # Check if all conditions met
        if not recovery["remaining_conditions"]:
            self._exit_recovery_mode(strategy_name)
    
    def _exit_recovery_mode(self, strategy_name: str) -> None:
        """
        Return a strategy to normal status after successful recovery.
        
        Args:
            strategy_name: Name of the strategy
        """
        recovery = self.recovery_tracking[strategy_name]
        
        if not recovery["in_recovery"]:
            return
            
        # Reset recovery tracking
        recovery["in_recovery"] = False
        recovery["recovery_start_date"] = None
        recovery["profitable_trades_in_recovery"] = 0
        recovery["trades_in_recovery"] = 0
        recovery["recovery_conditions_met"] = set()
        recovery["remaining_conditions"] = set()
        
        # Remove active adaptations
        if strategy_name in self.active_adaptations:
            self.active_adaptations[strategy_name] = []
        
        # Reset strategy status
        self.strategy_status[strategy_name] = StrategyStatus.ACTIVE
        
        # Reset triggered conditions
        self.strategy_triggers[strategy_name]["triggered_conditions"] = set()
        self.strategy_triggers[strategy_name]["failed_breakouts"] = 0
        self.strategy_triggers[strategy_name]["volume_confirmation_failures"] = 0
        
        logger.info(f"Strategy {strategy_name} successfully exited recovery mode")
    
    def update_market_environment(self, 
                                 environment: Optional[MarketEnvironment] = None,
                                 vix_value: Optional[float] = None,
                                 adx_value: Optional[float] = None,
                                 market_breadth: Optional[float] = None,
                                 sector_correlations: Optional[Dict[str, float]] = None,
                                 is_uptrend: Optional[bool] = None,
                                 volatility_regime: Optional[str] = None,
                                 liquidity_regime: Optional[str] = None,
                                 **market_metrics) -> Dict[str, Any]:
        """
        Update market environment classification and strategy suitability.
        
        Args:
            environment: Overall market environment classification
            vix_value: Current VIX value
            adx_value: Current ADX value (trend strength)
            market_breadth: Current market breadth
            sector_correlations: Dictionary of sector correlation values
            is_uptrend: Whether market is in an uptrend
            volatility_regime: Volatility classification (low, normal, high)
            liquidity_regime: Liquidity classification (low, normal, high)
            **market_metrics: Additional market metrics
            
        Returns:
            Dictionary with strategy suitability assessments
        """
        # Update environment if provided
        if environment:
            self.market_environment = environment
        
        # Update market metrics
        if vix_value is not None:
            self.vix_value = vix_value
        if adx_value is not None:
            self.adx_value = adx_value
        if market_breadth is not None:
            self.market_breadth = market_breadth
        if sector_correlations:
            self.sector_correlations = sector_correlations
            
        # Update micro regimes
        if volatility_regime:
            self.current_volatility = volatility_regime
        elif vix_value is not None:
            self.current_volatility = "high" if vix_value > 25 else "low" if vix_value < 15 else "normal"
            
        if liquidity_regime:
            self.current_liquidity = liquidity_regime
            
        if is_uptrend is not None:
            self.current_trend = "bullish" if is_uptrend else "bearish"
        
        # Determine overall environment if not provided
        if not environment:
            self._classify_market_environment()
        
        # Check strategy suitability
        suitability = {}
        for strategy_name in self.strategy_status:
            suitability[strategy_name] = self._check_environment_suitability(strategy_name)
        
        logger.info(f"Updated market environment to {self.market_environment}, "
                   f"volatility: {self.current_volatility}, "
                   f"trend: {self.current_trend}")
        
        return suitability
    
    def _classify_market_environment(self) -> None:
        """Determine overall market environment based on available metrics."""
        # Simple classification logic
        if self.adx_value >= 25:
            # Strong trend
            if self.current_trend == "bullish":
                self.market_environment = MarketEnvironment.TRENDING_BULL
            elif self.current_trend == "bearish":
                self.market_environment = MarketEnvironment.TRENDING_BEAR
            else:
                self.market_environment = MarketEnvironment.TRANSITION
        elif self.vix_value >= 30:
            self.market_environment = MarketEnvironment.HIGH_VOLATILITY
        elif self.vix_value < 15:
            self.market_environment = MarketEnvironment.LOW_VOLATILITY
        elif self.market_breadth < 0.3:
            # Poor breadth suggests transition
            self.market_environment = MarketEnvironment.TRANSITION
        else:
            self.market_environment = MarketEnvironment.RANGE_BOUND
    
    def _check_environment_suitability(self, strategy_name: str) -> Dict[str, Any]:
        """
        Check if current environment is suitable for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with suitability assessment
        """
        strategy_config = self.config["strategies"].get(strategy_name, {})
        ideal_conditions = strategy_config.get("ideal_conditions", [])
        adverse_conditions = strategy_config.get("adverse_conditions", [])
        
        # Check for ideal conditions
        present_ideal_conditions = []
        for condition in ideal_conditions:
            if self._check_condition(condition):
                present_ideal_conditions.append(condition)
        
        # Check for adverse conditions
        present_adverse_conditions = []
        for condition in adverse_conditions:
            if self._check_condition(condition):
                present_adverse_conditions.append(condition)
        
        # Calculate suitability score (-100 to 100)
        favorable_weight = len(present_ideal_conditions) / max(1, len(ideal_conditions)) * 100
        unfavorable_weight = len(present_adverse_conditions) / max(1, len(adverse_conditions)) * 100
        
        suitability_score = favorable_weight - unfavorable_weight
        
        # Determine suitability category
        if suitability_score >= 50:
            category = "highly_favorable"
        elif suitability_score >= 20:
            category = "favorable"
        elif suitability_score >= -20:
            category = "neutral"
        elif suitability_score >= -50:
            category = "unfavorable"
        else:
            category = "highly_unfavorable"
        
        return {
            "score": suitability_score,
            "category": category,
            "present_ideal_conditions": present_ideal_conditions,
            "present_adverse_conditions": present_adverse_conditions,
            "adaptation_needed": category in ["unfavorable", "highly_unfavorable"],
            "recommended_adjustments": self._get_conditional_adjustments(strategy_name)
        }
    
    def _check_condition(self, condition: str) -> bool:
        """
        Check if a specific market condition is currently true.
        
        Args:
            condition: Condition string to evaluate
            
        Returns:
            Boolean indicating if condition is present
        """
        # ADX conditions
        if "ADX > 25" in condition and self.adx_value > 25:
            return True
        if "ADX < 20" in condition and self.adx_value < 20:
            return True
            
        # VIX conditions
        if "low VIX" in condition and self.vix_value < 15:
            return True
        if "high VIX" in condition and self.vix_value > 25:
            return True
            
        # Trend conditions
        if "established uptrend" in condition and self.current_trend == "bullish":
            return True
        if "established downtrend" in condition and self.current_trend == "bearish":
            return True
            
        # Market breadth
        if "declining overall market breadth" in condition and self.market_breadth < 0.4:
            return True
        
        # Other specific conditions can be implemented based on available metrics
        
        return False
    
    def _get_conditional_adjustments(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get strategy adjustments based on current market conditions.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with recommended adjustments
        """
        adjustments = {}
        
        # Volatility-based adjustments
        if self.current_volatility == "high":
            if strategy_name == "breakout_swing":
                adjustments["stops"] = "Widen stops by 15%"
                adjustments["position_size"] = "Reduce size by 25%"
                adjustments["min_reward_risk"] = 2.0
            elif strategy_name == "theta_spread":
                adjustments["strategy_modification"] = "Convert to iron condors"
                adjustments["width"] = "Increase spread width by 20%"
            elif strategy_name == "vwap_bounce":
                adjustments["time_restriction"] = "Avoid afternoon session"
        
        # Liquidity-based adjustments
        if self.current_liquidity == "low":
            if strategy_name == "breakout_swing":
                adjustments["position_size"] = "Reduce size by 50%"
                adjustments["focus"] = "Only major breakout levels"
            elif strategy_name == "vwap_bounce":
                adjustments["volume_requirement"] = "2x normal minimum volume"
        
        # Trend-based adjustments
        if strategy_name == "breakout_swing" and self.current_trend != "bullish":
            adjustments["confirmation"] = "Require additional confirmation"
            adjustments["stop_placement"] = "Tighter stops (just below breakout level)"
        
        return adjustments
    
    def _notify_cooldown_manager(self, strategy_name: str, adaptations: List[Dict[str, Any]]) -> None:
        """
        Notify cooldown manager of strategy adaptations.
        
        Args:
            strategy_name: Name of the strategy
            adaptations: List of adaptation dictionaries
        """
        if not self.cooldown_manager or not hasattr(self.cooldown_manager, '_activate_cooldown'):
            return
            
        try:
            for adaptation in adaptations:
                # Map strategy adaptation to cooldown
                if "position_size_reduction" in adaptation["parameters"]:
                    position_size_pct = int(100 * (1 - adaptation["parameters"]["position_size_reduction"]))
                    
                    # Create cooldown
                    self.cooldown_manager._activate_cooldown(
                        f"strategy_monitor.{strategy_name}.{adaptation['trigger_id']}",
                        "MODERATE",  # Use appropriate level based on severity
                        expiration=None,  # Let strategy monitor handle expiration
                        position_size_pct=position_size_pct,
                        strategy_name=strategy_name,
                        trigger_description=adaptation["trigger_description"],
                        allow_trading=not adaptation["parameters"].get("suspend", False)
                    )
                    
                    logger.info(f"Notified cooldown manager of {strategy_name} adaptation: {adaptation['trigger_description']}")
        except Exception as e:
            logger.error(f"Error notifying cooldown manager: {str(e)}")
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for strategy usage based on current market conditions.
        
        Returns:
            Dictionary with strategy recommendations
        """
        recommendations = {
            "optimal_strategies": [],
            "avoid_strategies": [],
            "current_environment": self.market_environment,
            "environment_details": {
                "volatility": self.current_volatility,
                "trend": self.current_trend,
                "liquidity": self.current_liquidity,
                "vix": self.vix_value,
                "adx": self.adx_value,
                "market_breadth": self.market_breadth
            },
            "strategy_suitability": {}
        }
        
        # Check each strategy for suitability
        for strategy_name in self.strategy_status:
            # Skip suspended strategies
            if self.strategy_status[strategy_name] == StrategyStatus.SUSPENDED:
                continue
                
            suitability = self._check_environment_suitability(strategy_name)
            recommendations["strategy_suitability"][strategy_name] = suitability
            
            # Add to appropriate lists
            if suitability["category"] in ["highly_favorable", "favorable"]:
                recommendations["optimal_strategies"].append(strategy_name)
            elif suitability["category"] in ["unfavorable", "highly_unfavorable"]:
                recommendations["avoid_strategies"].append(strategy_name)
        
        # Environment-specific recommendations
        if self.market_environment == MarketEnvironment.TRENDING_BULL:
            recommendations["general_position_sizing"] = "Can increase to 1.25x baseline"
            recommendations["holding_period"] = "Extend target holding times by 20%"
            recommendations["entry_tactics"] = "Favor buying pullbacks to rising moving averages"
        elif self.market_environment == MarketEnvironment.HIGH_VOLATILITY:
            recommendations["general_position_sizing"] = "Reduce to 0.75x baseline"
            recommendations["stop_management"] = "Widen stops by 15-20%"
            recommendations["position_management"] = "Use smaller positions with scaling in/out approach"
        
        return recommendations 