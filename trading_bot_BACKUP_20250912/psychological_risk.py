"""
Psychological Risk Management Module

This module provides tools to assess and manage the psychological aspects of trading,
identifying emotional biases and preventing destructive trading behaviors.
"""

import logging
import json
import os
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

class PsychologicalState(str, Enum):
    """Psychological states that can affect trading decisions"""
    OPTIMAL = "optimal"                # Clear thinking, emotionally balanced
    FEAR = "fear"                      # Hesitant, prone to missing opportunities
    GREED = "greed"                    # Overconfident, prone to overtrading
    REVENGE = "revenge"                # Trading to recover losses, emotional
    BOREDOM = "boredom"                # Trading for stimulation, not opportunity
    ANXIETY = "anxiety"                # Worried about outcomes, hesitant
    OVERCONFIDENCE = "overconfidence"  # Excessive risk-taking, ignoring danger signs
    STRESS = "stress"                  # External pressures affecting judgment
    FATIGUE = "fatigue"                # Mental tiredness, reduced cognitive ability
    FOMO = "fomo"                      # Fear of missing out, impulsive entries
    
class PatternType(str, Enum):
    """Types of psychological patterns to monitor"""
    LOSS_AVERSION = "loss_aversion"          # Avoiding taking losses
    OVERTRADING = "overtrading"              # Excessive trading activity
    REVENGE_TRADING = "revenge_trading"      # Attempting to recover losses quickly
    PREMATURE_EXIT = "premature_exit"        # Closing winning trades too early
    SIZE_INCREASE_AFTER_WIN = "size_increase_after_win"  # Increasing size after winning
    SIZE_DECREASE_AFTER_LOSS = "size_decrease_after_loss"  # Decreasing size after losing
    HOLDING_LOSERS = "holding_losers"        # Holding losing trades too long
    
@dataclass
class ChecklistItem:
    """Data class for psychological checklist items"""
    question: str
    weight: float = 1.0
    indicator: Optional[PsychologicalState] = None
    type: str = "binary"  # binary, scale, or text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "question": self.question,
            "weight": self.weight,
            "indicator": self.indicator.value if self.indicator else None,
            "type": self.type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChecklistItem':
        """Create from dictionary"""
        indicator = PsychologicalState(data["indicator"]) if data.get("indicator") else None
        return cls(
            question=data["question"],
            weight=data.get("weight", 1.0),
            indicator=indicator,
            type=data.get("type", "binary")
        )

@dataclass
class TradingBlock:
    """Trading block criteria and details"""
    reason: str
    duration_minutes: int
    created_at: str
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "reason": self.reason,
            "duration_minutes": self.duration_minutes,
            "created_at": self.created_at,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingBlock':
        """Create from dictionary"""
        return cls(
            reason=data["reason"],
            duration_minutes=data["duration_minutes"],
            created_at=data["created_at"],
            active=data.get("active", True)
        )
    
    def is_expired(self) -> bool:
        """Check if the trading block has expired"""
        created = datetime.datetime.fromisoformat(self.created_at)
        elapsed = datetime.datetime.now() - created
        return elapsed.total_seconds() / 60 >= self.duration_minutes

class PsychologicalRiskManager:
    """
    Manages the psychological aspects of trading, including assessment,
    pattern recognition, and risk adjustments.
    """
    
    def __init__(self, 
                history_file: str = "trading_history.json",
                assessment_file: str = "psychological_checklist.json",
                self_reflection_file: str = "trading_reflections.json"):
        """
        Initialize the psychological risk manager.
        
        Args:
            history_file: Path to trading history file
            assessment_file: Path to psychological assessment checklist
            self_reflection_file: Path to trading reflections file
        """
        self.history_file = history_file
        self.assessment_file = assessment_file
        self.self_reflection_file = self_reflection_file
        
        # Load default checklist and trading history
        self.checklist = self._load_default_checklist()
        self.trading_history = self._load_trading_history()
        self.trading_blocks = []
        
        logger.info("Psychological risk manager initialized")
    
    def _load_default_checklist(self) -> List[ChecklistItem]:
        """
        Load default psychological checklist if none exists.
        
        Returns:
            List of checklist items
        """
        default_checklist = [
            ChecklistItem(
                question="Did you get adequate sleep last night?",
                weight=2.0,
                indicator=PsychologicalState.FATIGUE,
                type="binary"
            ),
            ChecklistItem(
                question="Are you free from distractions in your trading environment?",
                weight=1.5,
                indicator=PsychologicalState.STRESS,
                type="binary"
            ),
            ChecklistItem(
                question="Have you done your pre-market analysis?",
                weight=1.8,
                indicator=PsychologicalState.OVERCONFIDENCE,
                type="binary"
            ),
            ChecklistItem(
                question="How would you rate your current stress level? (1-10)",
                weight=1.8,
                indicator=PsychologicalState.STRESS,
                type="scale"
            ),
            ChecklistItem(
                question="Is this trade part of your trading plan?",
                weight=2.5,
                indicator=PsychologicalState.BOREDOM,
                type="binary"
            ),
            ChecklistItem(
                question="Are you trading to recover recent losses?",
                weight=2.0,
                indicator=PsychologicalState.REVENGE,
                type="binary"
            ),
            ChecklistItem(
                question="How strong is your FOMO on this trade? (1-10)",
                weight=2.0,
                indicator=PsychologicalState.FOMO,
                type="scale"
            ),
            ChecklistItem(
                question="Are your entry and exit criteria clearly defined?",
                weight=1.5,
                indicator=PsychologicalState.ANXIETY,
                type="binary"
            ),
            ChecklistItem(
                question="Are you trading a larger size than your plan allows?",
                weight=2.2,
                indicator=PsychologicalState.GREED,
                type="binary"
            ),
            ChecklistItem(
                question="Are you avoiding this trade out of fear despite valid signals?",
                weight=1.5,
                indicator=PsychologicalState.FEAR,
                type="binary"
            ),
            ChecklistItem(
                question="Have you experienced consecutive losses today?",
                weight=1.7,
                indicator=PsychologicalState.REVENGE,
                type="binary"
            ),
            ChecklistItem(
                question="How would you rate your current confidence level? (1-10)",
                weight=1.0,
                indicator=PsychologicalState.OVERCONFIDENCE,
                type="scale"
            ),
            ChecklistItem(
                question="Have you taken a break in the last 2 hours?",
                weight=1.0,
                indicator=PsychologicalState.FATIGUE,
                type="binary"
            ),
            ChecklistItem(
                question="Are you trading during your optimal trading hours?",
                weight=1.2,
                indicator=None,
                type="binary"
            ),
            ChecklistItem(
                question="Are you feeling pressure to 'make something happen' today?",
                weight=1.8,
                indicator=PsychologicalState.BOREDOM,
                type="binary"
            )
        ]
        
        # If assessment file exists, load it
        if os.path.exists(self.assessment_file):
            try:
                with open(self.assessment_file, 'r') as f:
                    loaded_checklist = json.load(f)
                    checklist = [ChecklistItem.from_dict(item) for item in loaded_checklist]
                    logger.info(f"Loaded {len(checklist)} checklist items from {self.assessment_file}")
                    return checklist
            except Exception as e:
                logger.error(f"Error loading checklist: {e}")
        
        # Save default checklist
        try:
            with open(self.assessment_file, 'w') as f:
                json.dump([item.to_dict() for item in default_checklist], f, indent=2)
                logger.info(f"Created default checklist with {len(default_checklist)} items")
        except Exception as e:
            logger.error(f"Error saving default checklist: {e}")
        
        return default_checklist
    
    def _load_trading_history(self) -> Dict[str, Any]:
        """
        Load trading history from file.
        
        Returns:
            Dictionary containing trading history
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    logger.info(f"Loaded trading history with {len(history.get('trades', []))} trades")
                    return history
            except Exception as e:
                logger.error(f"Error loading trading history: {e}")
        
        # Initialize empty history
        empty_history = {
            "trades": [],
            "assessments": [],
            "statistics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "current_streak_type": None,
                "current_streak_count": 0,
                "pattern_occurrences": {}
            }
        }
        
        # Save empty history
        try:
            with open(self.history_file, 'w') as f:
                json.dump(empty_history, f, indent=2)
                logger.info("Created empty trading history")
        except Exception as e:
            logger.error(f"Error saving empty trading history: {e}")
        
        return empty_history
    
    def _save_trading_history(self) -> None:
        """Save trading history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.trading_history, f, indent=2)
                logger.info("Trading history saved")
        except Exception as e:
            logger.error(f"Error saving trading history: {e}")
    
    def evaluate_psychological_risk(self, 
                                  responses: Dict[str, Any],
                                  trade_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate psychological risk based on questionnaire responses and trading history.
        
        Args:
            responses: Dictionary mapping question text to response
            trade_details: Optional details about the trade being considered
            
        Returns:
            Dictionary with risk assessment results and recommendations
        """
        # Calculate risk score based on responses
        risk_score = 0
        max_score = 0
        score_breakdown = {}
        
        for item in self.checklist:
            # Skip items that aren't in the responses
            if item.question not in responses:
                continue
            
            response = responses[item.question]
            item_score = 0
            
            # Score based on response type
            if item.type == "binary":
                # For positive questions (yes is good), no increases risk
                # For negative questions (no is good), yes increases risk
                is_positive = not item.question.startswith("Are you ") or "free from" in item.question
                
                if (is_positive and response.lower() in ["no", "false", "0"]) or \
                   (not is_positive and response.lower() in ["yes", "true", "1"]):
                    item_score = item.weight
            
            elif item.type == "scale":
                # Higher values on scale questions typically indicate higher risk
                # But reverse for questions about confidence where low is concerning
                if "confidence" in item.question.lower():
                    item_score = item.weight * (1 - float(response) / 10)
                else:
                    item_score = item.weight * float(response) / 10
            
            risk_score += item_score
            max_score += item.weight
            
            if item_score > 0:
                score_breakdown[item.question] = item_score
        
        # Normalize score (0-100 scale)
        if max_score > 0:
            normalized_score = (risk_score / max_score) * 100
        else:
            normalized_score = 0
        
        # Get adjustments based on trading patterns
        pattern_adjustments = self._get_pattern_adjustments()
        
        # Apply pattern adjustments to risk score
        adjusted_score = normalized_score
        for adjustment in pattern_adjustments:
            adjusted_score += adjustment["adjustment"]
            score_breakdown[f"Pattern: {adjustment['pattern']}"] = adjustment["adjustment"]
        
        # Cap adjusted score between 0-100
        adjusted_score = max(0, min(100, adjusted_score))
        
        # Determine risk level and adjustment factor
        risk_level = self._get_risk_level(adjusted_score)
        adjustment_factor = self._get_adjustment_factor(adjusted_score)
        
        # Get specific recommendations
        recommendations = self._get_recommendations(adjusted_score, pattern_adjustments, trade_details)
        
        # Check for trading block conditions
        trading_block = None
        if adjusted_score > 80:
            trading_block = TradingBlock(
                reason="Extremely high psychological risk score",
                duration_minutes=60,
                created_at=datetime.datetime.now().isoformat()
            )
            self.trading_blocks.append(trading_block)
            
        # Store assessment
        self._store_assessment(normalized_score, adjusted_score, risk_level, 
                              adjustment_factor, score_breakdown, recommendations)
        
        return {
            "raw_score": normalized_score,
            "adjusted_score": adjusted_score,
            "risk_level": risk_level,
            "adjustment_factor": adjustment_factor,
            "score_breakdown": score_breakdown,
            "pattern_adjustments": pattern_adjustments,
            "recommendations": recommendations,
            "trading_block": trading_block.to_dict() if trading_block else None
        }
    
    def _get_risk_level(self, score: float) -> str:
        """
        Determine risk level based on score.
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            Risk level string
        """
        if score < 20:
            return "Low"
        elif score < 40:
            return "Moderate"
        elif score < 60:
            return "Elevated"
        elif score < 80:
            return "High"
        else:
            return "Extreme"
    
    def _get_adjustment_factor(self, score: float) -> float:
        """
        Calculate risk adjustment factor based on score.
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            Adjustment factor for position sizing (0.0-1.0)
        """
        # Linear scaling from 1.0 (low risk) to 0.25 (high risk)
        if score <= 20:
            return 1.0
        elif score >= 80:
            return 0.25
        else:
            # Scale between 1.0 and 0.25 based on score
            return 1.0 - (score - 20) / 80 * 0.75
    
    def _get_recommendations(self, 
                            score: float, 
                            pattern_adjustments: List[Dict[str, Any]],
                            trade_details: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate recommendations based on risk assessment.
        
        Args:
            score: Risk score (0-100)
            pattern_adjustments: List of pattern adjustment dictionaries
            trade_details: Optional details about the trade being considered
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # General recommendations based on score
        if score < 20:
            recommendations.append("Psychological state is optimal for trading.")
        elif score < 40:
            recommendations.append("Maintain awareness of psychological factors while trading.")
        elif score < 60:
            recommendations.append("Consider reducing position sizes by 25-50%.")
            recommendations.append("Take a short break before placing new trades.")
        elif score < 80:
            recommendations.append("Significantly reduce position sizes (at least 50%).")
            recommendations.append("Consider only taking high-conviction trades.")
            recommendations.append("Take a longer break (30+ minutes) before resuming trading.")
        else:
            recommendations.append("Stop trading for the remainder of the session.")
            recommendations.append("Review trading journal and reflect on current psychological state.")
        
        # Pattern-specific recommendations
        pattern_types = [adj["pattern"] for adj in pattern_adjustments]
        
        if "overtrading" in pattern_types:
            recommendations.append("Reduce trading frequency. Set a maximum number of trades per day.")
            
        if "revenge_trading" in pattern_types:
            recommendations.append("Implement a mandatory 'cooling off' period after losses.")
            
        if "loss_aversion" in pattern_types:
            recommendations.append("Use automated stops to prevent avoiding necessary losses.")
            
        if "premature_exit" in pattern_types:
            recommendations.append("Use partial profit taking to reduce urge to exit entire positions early.")
        
        if "holding_losers" in pattern_types:
            recommendations.append("Review and strictly enforce stop loss rules.")
            
        # Add streak-specific recommendations
        stats = self.trading_history.get("statistics", {})
        if stats.get("current_streak_type") == "loss" and stats.get("current_streak_count", 0) >= 3:
            recommendations.append(f"You've had {stats['current_streak_count']} consecutive losses. Consider taking a trading break.")
            recommendations.append("Scale down position sizes until positive results return.")
            
        # Add trade-specific recommendations if details provided
        if trade_details:
            # Check for size relative to average
            trade_size = trade_details.get("size", 0)
            avg_size = self._calculate_average_position_size()
            
            if trade_size > avg_size * 1.5:
                recommendations.append(f"Proposed position size is significantly larger than your average. Consider reducing size.")
                
        return recommendations
    
    def _store_assessment(self, 
                         raw_score: float, 
                         adjusted_score: float,
                         risk_level: str,
                         adjustment_factor: float,
                         score_breakdown: Dict[str, float],
                         recommendations: List[str]) -> None:
        """
        Store assessment in trading history.
        
        Args:
            raw_score: Raw risk score
            adjusted_score: Adjusted risk score
            risk_level: Risk level string
            adjustment_factor: Position size adjustment factor
            score_breakdown: Breakdown of score components
            recommendations: List of recommendations
        """
        assessment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_score": raw_score,
            "adjusted_score": adjusted_score,
            "risk_level": risk_level,
            "adjustment_factor": adjustment_factor,
            "score_breakdown": score_breakdown,
            "recommendations": recommendations
        }
        
        if "assessments" not in self.trading_history:
            self.trading_history["assessments"] = []
            
        self.trading_history["assessments"].append(assessment)
        self._save_trading_history()
    
    def update_trading_history(self, trade_result: Dict[str, Any]) -> None:
        """
        Update trading history with a new trade result.
        
        Args:
            trade_result: Trade result data
        """
        # Ensure we have a trades list
        if "trades" not in self.trading_history:
            self.trading_history["trades"] = []
        
        # Add psychological state if available from most recent assessment
        if "assessments" in self.trading_history and self.trading_history["assessments"]:
            latest_assessment = self.trading_history["assessments"][-1]
            trade_result["psychological_state"] = {
                "risk_score": latest_assessment["adjusted_score"],
                "risk_level": latest_assessment["risk_level"],
                "adjustment_factor": latest_assessment["adjustment_factor"]
            }
        
        # Add the trade
        self.trading_history["trades"].append(trade_result)
        
        # Update statistics
        self._update_statistics(trade_result)
        
        # Save trading history
        self._save_trading_history()
        
        # Check for patterns after update
        self._analyze_trading_patterns()
    
    def _update_statistics(self, trade_result: Dict[str, Any]) -> None:
        """
        Update trading statistics based on new trade.
        
        Args:
            trade_result: Trade result data
        """
        if "statistics" not in self.trading_history:
            self.trading_history["statistics"] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "current_streak_type": None,
                "current_streak_count": 0,
                "pattern_occurrences": {}
            }
        
        stats = self.trading_history["statistics"]
        
        # Increment total trades
        stats["total_trades"] = stats.get("total_trades", 0) + 1
        
        # Check if trade was a winner or loser
        is_winner = trade_result.get("profit", 0) > 0
        
        if is_winner:
            stats["winning_trades"] = stats.get("winning_trades", 0) + 1
            
            # Update streak information
            if stats.get("current_streak_type") == "win":
                stats["current_streak_count"] += 1
            else:
                stats["current_streak_type"] = "win"
                stats["current_streak_count"] = 1
                
            # Update max streak if needed
            stats["max_consecutive_wins"] = max(
                stats.get("max_consecutive_wins", 0),
                stats["current_streak_count"]
            )
        else:
            stats["losing_trades"] = stats.get("losing_trades", 0) + 1
            
            # Update streak information
            if stats.get("current_streak_type") == "loss":
                stats["current_streak_count"] += 1
            else:
                stats["current_streak_type"] = "loss"
                stats["current_streak_count"] = 1
                
            # Update max streak if needed
            stats["max_consecutive_losses"] = max(
                stats.get("max_consecutive_losses", 0),
                stats["current_streak_count"]
            )
    
    def _analyze_trading_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze recent trades for psychological patterns.
        
        Returns:
            List of detected patterns
        """
        patterns = []
        trades = self.trading_history.get("trades", [])
        stats = self.trading_history.get("statistics", {})
        
        # Need minimum number of trades for analysis
        if len(trades) < 5:
            return patterns
        
        # Analyze most recent trades (last 10 or all if fewer)
        recent_trades = trades[-min(10, len(trades)):]
        
        # Check for overtrading (more than 5 trades per day)
        trade_dates = {}
        for trade in recent_trades:
            date = trade.get("timestamp", "").split("T")[0]
            trade_dates[date] = trade_dates.get(date, 0) + 1
        
        for date, count in trade_dates.items():
            if count > 5:
                patterns.append({
                    "pattern": PatternType.OVERTRADING,
                    "details": f"Executed {count} trades on {date}"
                })
        
        # Check for revenge trading (increasing size after losses)
        for i in range(1, len(recent_trades)):
            prev_trade = recent_trades[i-1]
            curr_trade = recent_trades[i]
            
            prev_profit = prev_trade.get("profit", 0)
            prev_size = prev_trade.get("position_size", 0)
            curr_size = curr_trade.get("position_size", 0)
            
            if prev_profit < 0 and curr_size > prev_size * 1.3:
                patterns.append({
                    "pattern": PatternType.REVENGE_TRADING,
                    "details": f"Increased position size by {(curr_size/prev_size - 1)*100:.0f}% after a loss"
                })
        
        # Check for premature exits
        for trade in recent_trades:
            target = trade.get("profit_target", 0)
            actual = trade.get("profit", 0)
            
            if actual > 0 and target > 0 and actual < target * 0.5:
                patterns.append({
                    "pattern": PatternType.PREMATURE_EXIT,
                    "details": "Exited winning trade far before target was reached"
                })
        
        # Check for holding losers
        for trade in recent_trades:
            stop_loss = trade.get("stop_loss", 0)
            actual_loss = trade.get("profit", 0)
            
            if actual_loss < 0 and stop_loss < 0 and actual_loss < stop_loss * 1.5:
                patterns.append({
                    "pattern": PatternType.HOLDING_LOSERS,
                    "details": "Loss exceeded planned stop loss by significant amount"
                })
        
        # Update pattern occurrences in statistics
        if "pattern_occurrences" not in stats:
            stats["pattern_occurrences"] = {}
            
        for pattern in patterns:
            pattern_type = pattern["pattern"]
            stats["pattern_occurrences"][pattern_type] = stats["pattern_occurrences"].get(pattern_type, 0) + 1
        
        return patterns
    
    def _get_pattern_adjustments(self) -> List[Dict[str, Any]]:
        """
        Calculate risk adjustments based on trading patterns.
        
        Returns:
            List of adjustment dictionaries
        """
        adjustments = []
        
        # Get recent patterns
        patterns = self._analyze_trading_patterns()
        
        for pattern in patterns:
            pattern_type = pattern["pattern"]
            
            # Different adjustments based on pattern type
            if pattern_type == PatternType.OVERTRADING:
                adjustments.append({
                    "pattern": pattern_type,
                    "adjustment": 15.0,
                    "explanation": "Overtrading indicates potential emotional trading"
                })
            elif pattern_type == PatternType.REVENGE_TRADING:
                adjustments.append({
                    "pattern": pattern_type,
                    "adjustment": 20.0,
                    "explanation": "Revenge trading is highly destructive to trading discipline"
                })
            elif pattern_type == PatternType.PREMATURE_EXIT:
                adjustments.append({
                    "pattern": pattern_type,
                    "adjustment": 10.0,
                    "explanation": "Premature exits indicate fear-based decision making"
                })
            elif pattern_type == PatternType.HOLDING_LOSERS:
                adjustments.append({
                    "pattern": pattern_type,
                    "adjustment": 15.0,
                    "explanation": "Holding losers beyond stops indicates loss aversion bias"
                })
        
        # Add streak-based adjustments
        stats = self.trading_history.get("statistics", {})
        streak_type = stats.get("current_streak_type")
        streak_count = stats.get("current_streak_count", 0)
        
        if streak_type == "loss" and streak_count >= 3:
            adjustment_value = min(25.0, streak_count * 5.0)
            adjustments.append({
                "pattern": "losing_streak",
                "adjustment": adjustment_value,
                "explanation": f"{streak_count} consecutive losses increases psychological pressure"
            })
        
        return adjustments
    
    def check_trading_blocks(self) -> Dict[str, Any]:
        """
        Check if trading is currently blocked.
        
        Returns:
            Dictionary with block status information
        """
        active_blocks = []
        
        for block in self.trading_blocks:
            if block.active and not block.is_expired():
                active_blocks.append(block.to_dict())
            elif block.active:
                # Mark as inactive if expired
                block.active = False
        
        if active_blocks:
            return {
                "is_blocked": True,
                "blocks": active_blocks
            }
        else:
            return {
                "is_blocked": False,
                "blocks": []
            }
    
    def add_trading_block(self, reason: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Add a manual trading block.
        
        Args:
            reason: Reason for the trading block
            duration_minutes: Duration of block in minutes
            
        Returns:
            Block information
        """
        block = TradingBlock(
            reason=reason,
            duration_minutes=duration_minutes,
            created_at=datetime.datetime.now().isoformat()
        )
        
        self.trading_blocks.append(block)
        
        return block.to_dict()
    
    def add_reflection(self, reflection: Dict[str, Any]) -> None:
        """
        Add a trading reflection.
        
        Args:
            reflection: Reflection data
        """
        if not os.path.exists(self.self_reflection_file):
            reflections = []
        else:
            try:
                with open(self.self_reflection_file, 'r') as f:
                    reflections = json.load(f)
            except Exception as e:
                logger.error(f"Error loading reflections: {e}")
                reflections = []
        
        # Add timestamp if not provided
        if "timestamp" not in reflection:
            reflection["timestamp"] = datetime.datetime.now().isoformat()
            
        reflections.append(reflection)
        
        try:
            with open(self.self_reflection_file, 'w') as f:
                json.dump(reflections, f, indent=2)
                logger.info("Added trading reflection")
        except Exception as e:
            logger.error(f"Error saving reflection: {e}")
    
    def get_reflections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trading reflections.
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of reflection dictionaries
        """
        if not os.path.exists(self.self_reflection_file):
            return []
            
        try:
            with open(self.self_reflection_file, 'r') as f:
                reflections = json.load(f)
                return reflections[-limit:]
        except Exception as e:
            logger.error(f"Error loading reflections: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get trading performance statistics related to psychological factors.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = self.trading_history.get("statistics", {})
        trades = self.trading_history.get("trades", [])
        
        # Return basic stats if insufficient data
        if len(trades) < 5:
            return stats
        
        # Add psychological performance metrics
        psych_stats = {}
        
        # Win rate by psychological risk level
        risk_level_performance = {}
        risk_level_counts = {}
        
        for trade in trades:
            if "psychological_state" in trade:
                risk_level = trade["psychological_state"].get("risk_level")
                if risk_level:
                    if risk_level not in risk_level_performance:
                        risk_level_performance[risk_level] = 0
                        risk_level_counts[risk_level] = 0
                    
                    risk_level_counts[risk_level] += 1
                    if trade.get("profit", 0) > 0:
                        risk_level_performance[risk_level] += 1
        
        # Calculate win rates by risk level
        win_rates = {}
        for level, wins in risk_level_performance.items():
            if risk_level_counts[level] > 0:
                win_rates[level] = wins / risk_level_counts[level] * 100
        
        psych_stats["win_rates_by_risk_level"] = win_rates
        
        # Calculate average profit by risk level
        avg_profit_by_risk = {}
        total_profit_by_risk = {}
        
        for trade in trades:
            if "psychological_state" in trade:
                risk_level = trade["psychological_state"].get("risk_level")
                if risk_level:
                    if risk_level not in total_profit_by_risk:
                        total_profit_by_risk[risk_level] = 0
                    
                    total_profit_by_risk[risk_level] += trade.get("profit", 0)
        
        for level, total in total_profit_by_risk.items():
            if risk_level_counts.get(level, 0) > 0:
                avg_profit_by_risk[level] = total / risk_level_counts[level]
        
        psych_stats["avg_profit_by_risk_level"] = avg_profit_by_risk
        
        # Pattern statistics
        psych_stats["pattern_occurrences"] = stats.get("pattern_occurrences", {})
        
        # Calculate optimal trading times and days based on performance
        time_performance = {}
        day_performance = {}
        
        for trade in trades:
            timestamp = trade.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    hour = dt.hour
                    day = dt.strftime("%A")
                    
                    # Initialize counters if needed
                    if hour not in time_performance:
                        time_performance[hour] = {"count": 0, "wins": 0, "profit": 0}
                    if day not in day_performance:
                        day_performance[day] = {"count": 0, "wins": 0, "profit": 0}
                    
                    # Update statistics
                    time_performance[hour]["count"] += 1
                    day_performance[day]["count"] += 1
                    
                    profit = trade.get("profit", 0)
                    time_performance[hour]["profit"] += profit
                    day_performance[day]["profit"] += profit
                    
                    if profit > 0:
                        time_performance[hour]["wins"] += 1
                        day_performance[day]["wins"] += 1
                        
                except Exception:
                    pass
        
        # Calculate win rates and average profits
        for hour, data in time_performance.items():
            if data["count"] > 0:
                data["win_rate"] = data["wins"] / data["count"] * 100
                data["avg_profit"] = data["profit"] / data["count"]
        
        for day, data in day_performance.items():
            if data["count"] > 0:
                data["win_rate"] = data["wins"] / data["count"] * 100
                data["avg_profit"] = data["profit"] / data["count"]
        
        psych_stats["performance_by_hour"] = time_performance
        psych_stats["performance_by_day"] = day_performance
        
        # Find best and worst times
        best_hour = max(time_performance.items(), key=lambda x: x[1]["avg_profit"] if x[1]["count"] >= 3 else -9999)
        worst_hour = min(time_performance.items(), key=lambda x: x[1]["avg_profit"] if x[1]["count"] >= 3 else 9999)
        
        psych_stats["optimal_trading_hours"] = {
            "best_hour": int(best_hour[0]) if best_hour[1]["count"] >= 3 else None,
            "worst_hour": int(worst_hour[0]) if worst_hour[1]["count"] >= 3 else None
        }
        
        # Combine with existing stats
        combined_stats = {**stats, "psychological_stats": psych_stats}
        return combined_stats
    
    def _calculate_average_position_size(self) -> float:
        """
        Calculate average position size from trading history.
        
        Returns:
            Average position size
        """
        trades = self.trading_history.get("trades", [])
        if not trades:
            return 0
            
        sizes = [trade.get("position_size", 0) for trade in trades]
        if not sizes:
            return 0
            
        return sum(sizes) / len(sizes)
    
    def get_insights(self) -> Dict[str, Any]:
        """
        Generate insights based on trading history and patterns.
        
        Returns:
            Dictionary with insights
        """
        insights = {
            "general": [],
            "psychological_patterns": [],
            "recommendations": []
        }
        
        stats = self.trading_history.get("statistics", {})
        psych_stats = self.get_performance_stats().get("psychological_stats", {})
        
        # Add general insights
        win_rate = 0
        if stats.get("total_trades", 0) > 0:
            win_rate = stats.get("winning_trades", 0) / stats.get("total_trades", 0) * 100
            insights["general"].append(f"Overall win rate: {win_rate:.1f}%")
        
        # Add risk level insights
        win_rates_by_risk = psych_stats.get("win_rates_by_risk_level", {})
        if win_rates_by_risk:
            best_risk_level = max(win_rates_by_risk.items(), key=lambda x: x[1])
            worst_risk_level = min(win_rates_by_risk.items(), key=lambda x: x[1])
            
            insights["psychological_patterns"].append(
                f"Your win rate is highest ({best_risk_level[1]:.1f}%) when trading at {best_risk_level[0]} psychological risk"
            )
            insights["psychological_patterns"].append(
                f"Your win rate drops to {worst_risk_level[1]:.1f}% when trading at {worst_risk_level[0]} psychological risk"
            )
        
        # Add pattern insights
        pattern_occurrences = psych_stats.get("pattern_occurrences", {})
        for pattern, count in pattern_occurrences.items():
            if count > 2:
                insights["psychological_patterns"].append(
                    f"You show a pattern of {pattern} ({count} occurrences)"
                )
        
        # Add time-based insights
        performance_by_hour = psych_stats.get("performance_by_hour", {})
        if performance_by_hour:
            best_times = []
            worst_times = []
            
            for hour, data in performance_by_hour.items():
                if data.get("count", 0) >= 3:
                    if data.get("win_rate", 0) > win_rate + 10:
                        best_times.append((hour, data["win_rate"]))
                    elif data.get("win_rate", 0) < win_rate - 10:
                        worst_times.append((hour, data["win_rate"]))
            
            if best_times:
                best_times_str = ", ".join([f"{int(h)}:00 ({r:.1f}%)" for h, r in sorted(best_times)])
                insights["recommendations"].append(f"Consider trading more during your best hours: {best_times_str}")
            
            if worst_times:
                worst_times_str = ", ".join([f"{int(h)}:00 ({r:.1f}%)" for h, r in sorted(worst_times)])
                insights["recommendations"].append(f"Consider avoiding trading during your worst hours: {worst_times_str}")
        
        # Add risk management recommendations
        if "HOLDING_LOSERS" in pattern_occurrences:
            insights["recommendations"].append(
                "Implement stricter stop loss discipline to address pattern of holding losing trades too long"
            )
        
        if "REVENGE_TRADING" in pattern_occurrences:
            insights["recommendations"].append(
                "Implement a mandatory cooling-off period after losses to prevent revenge trading"
            )
        
        if "OVERTRADING" in pattern_occurrences:
            insights["recommendations"].append(
                "Set a daily maximum trade limit to prevent overtrading"
            )
        
        return insights
    
    def create_checklist_item(self, 
                            question: str,
                            weight: float = 1.0,
                            indicator: Optional[str] = None,
                            type: str = "binary") -> Dict[str, Any]:
        """
        Create a new checklist item.
        
        Args:
            question: Question text
            weight: Weight factor for scoring
            indicator: Psychological state indicator
            type: Question type (binary, scale, text)
            
        Returns:
            Newly created checklist item
        """
        # Convert string indicator to enum if provided
        indicator_enum = None
        if indicator:
            try:
                indicator_enum = PsychologicalState(indicator)
            except ValueError:
                logger.warning(f"Invalid psychological state indicator: {indicator}")
        
        # Create checklist item
        item = ChecklistItem(
            question=question,
            weight=weight,
            indicator=indicator_enum,
            type=type
        )
        
        # Add to checklist
        self.checklist.append(item)
        
        # Save updated checklist
        try:
            with open(self.assessment_file, 'w') as f:
                json.dump([i.to_dict() for i in self.checklist], f, indent=2)
                logger.info(f"Added new checklist item: {question}")
        except Exception as e:
            logger.error(f"Error saving updated checklist: {e}")
        
        return item.to_dict()
    
    def delete_checklist_item(self, question: str) -> bool:
        """
        Delete a checklist item by question text.
        
        Args:
            question: Question text to delete
            
        Returns:
            True if item was deleted, False otherwise
        """
        original_length = len(self.checklist)
        
        # Remove matching items
        self.checklist = [item for item in self.checklist if item.question != question]
        
        # Check if any items were removed
        if len(self.checklist) < original_length:
            # Save updated checklist
            try:
                with open(self.assessment_file, 'w') as f:
                    json.dump([i.to_dict() for i in self.checklist], f, indent=2)
                    logger.info(f"Deleted checklist item: {question}")
            except Exception as e:
                logger.error(f"Error saving updated checklist: {e}")
            
            return True
        else:
            logger.warning(f"Checklist item not found: {question}")
            return False
    
    def get_checklist(self) -> List[Dict[str, Any]]:
        """
        Get the current psychological assessment checklist.
        
        Returns:
            List of checklist items
        """
        return [item.to_dict() for item in self.checklist] 