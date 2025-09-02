#!/usr/bin/env python3
"""
Anomaly Risk Handler

This module manages the risk response to market anomalies detected by the
MarketAnomalyDetector. It interprets anomaly scores, applies appropriate
risk adjustments, and manages cooldown periods after anomalies.
"""

import os
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class AnomalyRiskHandler:
    """
    Handles risk adjustments based on market anomaly detection.
    
    This class:
    1. Interprets anomaly scores from MarketAnomalyDetector
    2. Applies appropriate risk adjustments based on configuration
    3. Manages cooldown periods after anomalies are detected
    4. Provides interfaces for trade executors to check if trading should be restricted
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the anomaly risk handler.
        
        Args:
            config_path: Path to anomaly risk configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", "anomaly_risk_rules.yaml"
        )
        self.config = self._load_config()
        
        # Current state
        self.current_anomaly_score = 0.0
        self.current_risk_level = "minimal"
        self.active_cooldowns: Dict[str, datetime] = {}
        self.last_anomaly_time: Optional[datetime] = None
        self.recovery_counter = 0
        self.anomaly_types_detected: List[str] = []
        
        logger.info("Anomaly Risk Handler initialized with config from %s", self.config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Loaded anomaly risk configuration successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load anomaly risk configuration: {e}")
            # Return default configuration
            return {
                "thresholds": {
                    "minimal": 0.3,
                    "moderate": 0.5, 
                    "high": 0.7,
                    "critical": 0.85
                },
                "risk_adjustments": {
                    "minimal": {"risk_mode": "normal", "position_size_modifier": 1.0, "cooldown_minutes": 0},
                    "moderate": {"risk_mode": "cautious", "position_size_modifier": 0.7, "cooldown_minutes": 10},
                    "high": {"risk_mode": "defensive", "position_size_modifier": 0.4, "cooldown_minutes": 20},
                    "critical": {"risk_mode": "lockdown", "position_size_modifier": 0.0, "cooldown_minutes": 30}
                },
                "recovery": {
                    "monitor_periods": 5,
                    "gradual": True
                }
            }
    
    def process_anomaly_result(self, anomaly_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process anomaly detection results and determine appropriate risk actions.
        
        Args:
            anomaly_result: Dictionary containing anomaly detection results from MarketAnomalyDetector
            
        Returns:
            Dictionary with risk adjustment actions to take
        """
        # Extract relevant information from the anomaly result
        anomaly_score = anomaly_result.get("latest_score", 0.0)
        self.current_anomaly_score = anomaly_score
        
        # Determine anomaly types if available
        self.anomaly_types_detected = []
        if "feature_contributions" in anomaly_result:
            for feature_type, values in anomaly_result["feature_contributions"].items():
                if any(values):
                    # Convert feature names to anomaly types
                    if "price" in feature_type and "volatility" in feature_type:
                        self.anomaly_types_detected.append("price_spike")
                    elif "volume" in feature_type:
                        self.anomaly_types_detected.append("volume_spike")
                    elif "spread" in feature_type:
                        self.anomaly_types_detected.append("spread_widening")
                    elif "flash" in feature_type or ("price" in feature_type and "drop" in feature_type):
                        self.anomaly_types_detected.append("flash_crash")
        
        # Update risk level based on anomaly score
        previous_risk_level = self.current_risk_level
        self.current_risk_level = self._determine_risk_level(anomaly_score)
        
        # If risk level increased, reset recovery counter
        if self._risk_level_value(self.current_risk_level) > self._risk_level_value(previous_risk_level):
            self.recovery_counter = 0
            self.last_anomaly_time = datetime.now()
        
        # If score is low but we were previously in a higher risk state, handle recovery
        elif anomaly_score < self.config["thresholds"]["moderate"]:
            self._handle_recovery()
        
        # Get the risk adjustments for the current level
        risk_actions = self._get_risk_adjustments()
        
        # Update active cooldowns
        self._update_cooldowns(risk_actions)
        
        # Log risk adjustment
        logger.info(
            "Anomaly score: %.3f, Risk level: %s, Actions: %s", 
            anomaly_score, self.current_risk_level, 
            {k: v for k, v in risk_actions.items() if k != "cooldown_end_time"}
        )
        
        return risk_actions
    
    def _determine_risk_level(self, anomaly_score: float) -> str:
        """Determine the appropriate risk level based on anomaly score."""
        thresholds = self.config["thresholds"]
        
        # Check for specific anomaly types that might override the score-based level
        for anomaly_type in self.anomaly_types_detected:
            if anomaly_type in self.config.get("anomaly_type_overrides", {}):
                override = self.config["anomaly_type_overrides"][anomaly_type]
                if "risk_mode" in override and override.get("risk_mode") == "lockdown":
                    return "critical"
        
        # Determine level based on score thresholds
        if anomaly_score >= thresholds["critical"]:
            return "critical"
        elif anomaly_score >= thresholds["high"]:
            return "high"
        elif anomaly_score >= thresholds["moderate"]:
            return "moderate"
        else:
            return "minimal"
    
    def _risk_level_value(self, level: str) -> int:
        """Convert risk level string to numeric value for comparison."""
        levels = {"minimal": 0, "moderate": 1, "high": 2, "critical": 3}
        return levels.get(level, 0)
    
    def _handle_recovery(self):
        """Handle recovery from higher risk states."""
        if not self.config["recovery"]["gradual"]:
            # Immediate recovery
            self.current_risk_level = "minimal"
            self.recovery_counter = 0
            return
        
        # Gradual recovery
        self.recovery_counter += 1
        
        # If we've had enough normal periods, step down the risk level
        if self.recovery_counter >= self.config["recovery"]["monitor_periods"]:
            current_value = self._risk_level_value(self.current_risk_level)
            if current_value > 0:  # If not already at minimal
                levels = ["minimal", "moderate", "high", "critical"]
                self.current_risk_level = levels[current_value - 1]  # Step down one level
            self.recovery_counter = 0  # Reset for next potential step down
    
    def _get_risk_adjustments(self) -> Dict[str, Any]:
        """Get risk adjustment parameters for the current risk level."""
        # Start with the standard adjustments for this level
        adjustments = self.config["risk_adjustments"][self.current_risk_level].copy()
        
        # Apply any overrides based on specific anomaly types
        for anomaly_type in self.anomaly_types_detected:
            if anomaly_type in self.config.get("anomaly_type_overrides", {}):
                override = self.config["anomaly_type_overrides"][anomaly_type]
                # Update adjustments with any override values
                for key, value in override.items():
                    if key in adjustments:
                        # Use the more conservative value
                        if key == "position_size_modifier":
                            adjustments[key] = min(adjustments[key], value)
                        elif key == "cooldown_minutes":
                            adjustments[key] = max(adjustments[key], value)
                        elif key == "stop_loss_modifier":
                            adjustments[key] = min(adjustments[key], value)
                        else:
                            adjustments[key] = value
        
        # Calculate cooldown end time if applicable
        if adjustments.get("cooldown_minutes", 0) > 0:
            cooldown_end = datetime.now() + timedelta(minutes=adjustments["cooldown_minutes"])
            adjustments["cooldown_end_time"] = cooldown_end
        
        return adjustments
    
    def _update_cooldowns(self, risk_actions: Dict[str, Any]):
        """Update active cooldowns based on risk actions."""
        if "cooldown_end_time" in risk_actions:
            self.active_cooldowns[self.current_risk_level] = risk_actions["cooldown_end_time"]
        
        # Remove expired cooldowns
        now = datetime.now()
        expired = [level for level, end_time in self.active_cooldowns.items() if end_time <= now]
        for level in expired:
            del self.active_cooldowns[level]
    
    def get_active_cooldown(self) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """
        Check if there are any active cooldowns.
        
        Returns:
            Tuple containing:
            - Boolean indicating if a cooldown is active
            - End time of the cooldown (or None)
            - Reason for the cooldown (risk level)
        """
        if not self.active_cooldowns:
            return False, None, None
        
        # Find the most restrictive (highest level) active cooldown
        highest_level = max(self.active_cooldowns.keys(), key=self._risk_level_value)
        end_time = self.active_cooldowns[highest_level]
        
        # Check if it's still active
        if end_time > datetime.now():
            return True, end_time, highest_level
        else:
            # Clean up expired cooldown
            del self.active_cooldowns[highest_level]
            return self.get_active_cooldown()  # Recursively check for other active cooldowns
    
    def is_trading_restricted(self) -> bool:
        """Check if trading should be restricted due to anomalies."""
        # First check if we're in a risk level that disallows trading
        if self.current_risk_level in self.config["risk_adjustments"]:
            if not self.config["risk_adjustments"][self.current_risk_level].get("new_trades_allowed", True):
                return True
        
        # Then check if we're in an active cooldown
        is_in_cooldown, _, _ = self.get_active_cooldown()
        return is_in_cooldown
    
    def get_position_size_modifier(self) -> float:
        """Get the current position size modifier based on anomaly risk."""
        if self.current_risk_level in self.config["risk_adjustments"]:
            return self.config["risk_adjustments"][self.current_risk_level].get("position_size_modifier", 1.0)
        return 1.0
    
    def get_stop_loss_modifier(self) -> float:
        """Get the current stop loss modifier based on anomaly risk."""
        if self.current_risk_level in self.config["risk_adjustments"]:
            return self.config["risk_adjustments"][self.current_risk_level].get("stop_loss_modifier", 1.0)
        return 1.0
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get the current risk status information."""
        is_in_cooldown, cooldown_end, cooldown_reason = self.get_active_cooldown()
        
        return {
            "anomaly_score": self.current_anomaly_score,
            "risk_level": self.current_risk_level,
            "in_cooldown": is_in_cooldown,
            "cooldown_end": cooldown_end,
            "cooldown_reason": cooldown_reason,
            "position_size_modifier": self.get_position_size_modifier(),
            "stop_loss_modifier": self.get_stop_loss_modifier(),
            "trading_restricted": self.is_trading_restricted(),
            "anomaly_types": self.anomaly_types_detected,
            "last_anomaly_time": self.last_anomaly_time
        }


# Example usage
if __name__ == "__main__":
    # Set up logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example anomaly result
    example_result = {
        "latest_score": 0.82,
        "feature_contributions": {
            "price_volatility": [0.4, 0.6, 0.2],
            "volume_spike": [0.0, 0.0, 0.0]
        }
    }
    
    # Create handler and process the result
    handler = AnomalyRiskHandler()
    actions = handler.process_anomaly_result(example_result)
    
    print(f"Actions to take: {actions}")
    print(f"Current risk status: {handler.get_current_risk_status()}")
    print(f"Is trading restricted? {handler.is_trading_restricted()}")
    print(f"Position size modifier: {handler.get_position_size_modifier()}") 