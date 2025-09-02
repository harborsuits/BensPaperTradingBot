import logging
import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.strategy_management.interfaces import MarketRegimeClassifier
from trading_bot.strategy_management.market_context import TradingMarketContext

logger = logging.getLogger(__name__)

class RuleBasedRegimeClassifier(MarketRegimeClassifier):
    """
    A rule-based market regime classifier that uses predefined rules to
    categorize market conditions based on technical indicators and market data.
    """
    
    # Define standard market regimes
    REGIME_BULL = "bull"
    REGIME_BEAR = "bear"
    REGIME_SIDEWAYS = "sideways"
    REGIME_VOLATILE = "volatile"
    REGIME_RECOVERY = "recovery"
    REGIME_CRASH = "crash"
    REGIME_UNKNOWN = "unknown"
    
    def __init__(self):
        self._rules = self._get_default_rules()
        self._performance_data = {}
        self._last_regime = self.REGIME_UNKNOWN
        self._confidence = 0.5
        self._regime_history = []  # List of (timestamp, regime) tuples
        
    def _get_default_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the default rules for classifying market regimes
        """
        return {
            self.REGIME_BULL: {
                "description": "Strong upward trend with sustained momentum",
                "conditions": [
                    {"indicator": "market_trend", "operator": "==", "value": "up"},
                    {"indicator": "market_volatility", "operator": "<", "value": 20},
                    {"indicator": "indicator_rsi", "operator": ">", "value": 50}
                ],
                "weight": 0.25
            },
            self.REGIME_BEAR: {
                "description": "Downward trend with negative sentiment",
                "conditions": [
                    {"indicator": "market_trend", "operator": "==", "value": "down"},
                    {"indicator": "indicator_rsi", "operator": "<", "value": 50},
                    {"indicator": "market_sentiment", "operator": "<", "value": 0}
                ],
                "weight": 0.25
            },
            self.REGIME_SIDEWAYS: {
                "description": "Low volatility range-bound market",
                "conditions": [
                    {"indicator": "market_trend", "operator": "==", "value": "sideways"},
                    {"indicator": "market_volatility", "operator": "<", "value": 15}
                ],
                "weight": 0.2
            },
            self.REGIME_VOLATILE: {
                "description": "High volatility with unclear direction",
                "conditions": [
                    {"indicator": "market_volatility", "operator": ">", "value": 25}
                ],
                "weight": 0.15
            },
            self.REGIME_RECOVERY: {
                "description": "Rebound after significant decline",
                "conditions": [
                    {"indicator": "market_trend", "operator": "==", "value": "up"},
                    {"indicator": "market_prev_trend", "operator": "==", "value": "down"},
                    {"indicator": "indicator_rsi", "operator": ">", "value": 40}
                ],
                "weight": 0.1
            },
            self.REGIME_CRASH: {
                "description": "Rapid decline with high volatility",
                "conditions": [
                    {"indicator": "market_trend", "operator": "==", "value": "down"},
                    {"indicator": "market_volatility", "operator": ">", "value": 30},
                    {"indicator": "market_drawdown", "operator": ">", "value": 5}
                ],
                "weight": 0.05
            }
        }
        
    def classify_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Classify the current market regime based on provided market data
        """
        if not market_data:
            logger.warning("No market data provided for regime classification")
            return self.REGIME_UNKNOWN
            
        # Calculate scores for each regime
        regime_scores = {}
        
        for regime, rule_data in self._rules.items():
            score = self._evaluate_regime_rule(regime, rule_data, market_data)
            regime_scores[regime] = score
            
        # Find the regime with the highest score
        if not regime_scores:
            return self.REGIME_UNKNOWN
            
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        
        # Only update if we have reasonable confidence
        if best_regime[1] > 0.3:
            self._last_regime = best_regime[0]
            self._confidence = best_regime[1]
            
            # Record this classification
            self._regime_history.append((time.time(), self._last_regime))
            if len(self._regime_history) > 100:  # Limit history size
                self._regime_history.pop(0)
                
        return self._last_regime
        
    def _evaluate_regime_rule(self, regime: str, rule_data: Dict[str, Any], 
                              market_data: Dict[str, Any]) -> float:
        """
        Evaluate a single regime rule against market data
        Returns a score between 0 and 1
        """
        if "conditions" not in rule_data:
            return 0.0
            
        conditions = rule_data["conditions"]
        if not conditions:
            return 0.0
            
        # Count how many conditions are satisfied
        satisfied = 0
        total = len(conditions)
        
        for condition in conditions:
            indicator = condition.get("indicator")
            operator = condition.get("operator")
            expected_value = condition.get("value")
            
            if not all([indicator, operator, expected_value is not None]):
                continue
                
            # Get actual value from market data
            actual_value = market_data.get(indicator)
            if actual_value is None:
                continue
                
            # Evaluate condition
            if self._evaluate_condition(actual_value, operator, expected_value):
                satisfied += 1
                
        # Calculate score based on satisfied conditions
        score = satisfied / total if total > 0 else 0
        
        # Apply regime weight
        weight = rule_data.get("weight", 1.0)
        return score * weight
        
    def _evaluate_condition(self, actual, operator, expected) -> bool:
        """
        Evaluate a condition with the given operator
        """
        try:
            if operator == "==":
                return actual == expected
            elif operator == "!=":
                return actual != expected
            elif operator == ">":
                return float(actual) > float(expected)
            elif operator == ">=":
                return float(actual) >= float(expected)
            elif operator == "<":
                return float(actual) < float(expected)
            elif operator == "<=":
                return float(actual) <= float(expected)
            elif operator == "in":
                return actual in expected
            elif operator == "not_in":
                return actual not in expected
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            logger.error(f"Error evaluating condition: {str(e)}")
            return False
            
    def get_regime_confidence(self) -> float:
        """
        Get confidence level for the current regime classification
        """
        return self._confidence
        
    def get_regime_description(self, regime: Optional[str] = None) -> str:
        """
        Get the description for a specific regime or the current regime
        """
        if regime is None:
            regime = self._last_regime
            
        rule_data = self._rules.get(regime, {})
        return rule_data.get("description", "Unknown market regime")
        
    def update_performance(self, regime: str, metrics: Dict[str, float]) -> None:
        """
        Update the performance metrics for a specific regime
        """
        if regime not in self._performance_data:
            self._performance_data[regime] = []
            
        metrics["timestamp"] = time.time()
        self._performance_data[regime].append(metrics)
        
        # Limit history size
        if len(self._performance_data[regime]) > 20:
            self._performance_data[regime].pop(0)
            
    def get_regime_performance(self, regime: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific regime or all regimes
        """
        if regime is not None:
            return self._performance_data.get(regime, [])
        return self._performance_data
            
    def adjust_rules(self, learning_rate: float = 0.1) -> None:
        """
        Adjust classification rules based on performance data
        """
        # Only adjust if we have sufficient performance data
        min_samples = 5
        regimes_to_adjust = []
        
        for regime, metrics_list in self._performance_data.items():
            if len(metrics_list) >= min_samples and regime in self._rules:
                regimes_to_adjust.append(regime)
                
        if not regimes_to_adjust:
            logger.info("Not enough performance data to adjust rules")
            return
            
        # Adjust rules for each regime
        for regime in regimes_to_adjust:
            metrics_list = self._performance_data[regime]
            avg_performance = sum(m.get("accuracy", 0) for m in metrics_list) / len(metrics_list)
            
            # Only adjust if performance is below threshold
            if avg_performance < 0.7:
                self._adjust_regime_rules(regime, metrics_list, learning_rate)
                
    def _adjust_regime_rules(self, regime: str, metrics_list: List[Dict[str, float]], 
                            learning_rate: float) -> None:
        """
        Adjust rules for a specific regime based on performance metrics
        """
        # Extract the rule
        rule_data = self._rules.get(regime, {})
        if not rule_data or "conditions" not in rule_data:
            return
            
        # Calculate avg metrics
        avg_metrics = {}
        for metric in metrics_list[0].keys():
            if metric != "timestamp":
                avg_metrics[metric] = sum(m.get(metric, 0) for m in metrics_list) / len(metrics_list)
                
        # Adjust condition values
        for condition in rule_data["conditions"]:
            indicator = condition.get("indicator")
            operator = condition.get("operator")
            current_value = condition.get("value")
            
            # Skip if can't adjust (string-based values)
            if isinstance(current_value, str):
                continue
                
            # Skip if no performance data for this indicator
            if indicator not in avg_metrics:
                continue
                
            # Calculate new value based on performance
            adjustment = (avg_metrics[indicator] - current_value) * learning_rate
            new_value = current_value + adjustment
            
            # Update the condition
            condition["value"] = new_value
            logger.info(f"Adjusted {regime} rule: {indicator} {operator} {new_value}")
                
    def save_rules(self, filepath: str) -> bool:
        """
        Save classification rules to a file
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    "rules": self._rules,
                    "performance": self._performance_data,
                    "last_updated": time.time()
                }, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving rules to file: {str(e)}")
            return False
            
    def load_rules(self, filepath: str) -> bool:
        """
        Load classification rules from a file
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Rules file not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self._rules = data.get("rules", self._rules)
            self._performance_data = data.get("performance", {})
            return True
        except Exception as e:
            logger.error(f"Error loading rules from file: {str(e)}")
            return False 