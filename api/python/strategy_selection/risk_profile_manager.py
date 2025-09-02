#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Profile Manager

This module handles risk profiling and customized risk management based on 
trader profiles and preferences.
"""

import logging
import json
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RiskToleranceLevel(Enum):
    """Risk tolerance levels from most conservative to most aggressive."""
    VERY_CONSERVATIVE = 1
    CONSERVATIVE = 2
    MODERATE = 3
    AGGRESSIVE = 4
    VERY_AGGRESSIVE = 5


@dataclass
class RiskProfile:
    """Data class to store a trader's risk profile information."""
    tolerance_level: RiskToleranceLevel
    max_drawdown_percent: float
    max_risk_per_trade_percent: float
    max_portfolio_risk_percent: float
    position_sizing_method: str
    diversification_preference: float  # 0-1 scale
    timeframe_preferences: List[str]
    preferred_strategy_types: List[str]
    favorite_currency_pairs: List[str]
    trade_frequency_preference: str
    created_date: str
    last_updated: str
    additional_settings: Dict[str, Any]


class RiskProfileManager:
    """
    Manages trader risk profiles and provides risk parameters
    for strategy selection and position sizing.
    """
    
    # Default risk profile settings by tolerance level
    DEFAULT_PROFILES = {
        RiskToleranceLevel.VERY_CONSERVATIVE: {
            "max_drawdown_percent": 5.0,
            "max_risk_per_trade_percent": 0.5,
            "max_portfolio_risk_percent": 2.0,
            "position_sizing_method": "fixed_percent",
            "diversification_preference": 0.8,
            "trade_frequency_preference": "low"
        },
        RiskToleranceLevel.CONSERVATIVE: {
            "max_drawdown_percent": 10.0,
            "max_risk_per_trade_percent": 1.0,
            "max_portfolio_risk_percent": 4.0,
            "position_sizing_method": "fixed_percent",
            "diversification_preference": 0.7,
            "trade_frequency_preference": "low_to_medium"
        },
        RiskToleranceLevel.MODERATE: {
            "max_drawdown_percent": 15.0,
            "max_risk_per_trade_percent": 1.5,
            "max_portfolio_risk_percent": 8.0,
            "position_sizing_method": "fixed_percent",
            "diversification_preference": 0.5,
            "trade_frequency_preference": "medium"
        },
        RiskToleranceLevel.AGGRESSIVE: {
            "max_drawdown_percent": 25.0,
            "max_risk_per_trade_percent": 2.5,
            "max_portfolio_risk_percent": 15.0,
            "position_sizing_method": "kelly_criterion",
            "diversification_preference": 0.3,
            "trade_frequency_preference": "medium_to_high"
        },
        RiskToleranceLevel.VERY_AGGRESSIVE: {
            "max_drawdown_percent": 35.0,
            "max_risk_per_trade_percent": 4.0,
            "max_portfolio_risk_percent": 25.0,
            "position_sizing_method": "kelly_criterion",
            "diversification_preference": 0.1,
            "trade_frequency_preference": "high"
        }
    }
    
    def __init__(self, profiles_directory: str = None):
        """
        Initialize the risk profile manager.
        
        Args:
            profiles_directory: Directory to store risk profiles
        """
        self.profiles_directory = profiles_directory or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "risk_profiles"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(self.profiles_directory, exist_ok=True)
        
        # Currently loaded profile
        self.current_profile: Optional[RiskProfile] = None
        self.profile_name: Optional[str] = None
        
        logger.info(f"Risk Profile Manager initialized with profile directory: {self.profiles_directory}")
    
    def create_profile(self, name: str, tolerance_level: Union[RiskToleranceLevel, str], 
                       timeframe_preferences: List[str] = None,
                       preferred_strategy_types: List[str] = None,
                       favorite_currency_pairs: List[str] = None,
                       custom_settings: Dict[str, Any] = None) -> RiskProfile:
        """
        Create a new risk profile.
        
        Args:
            name: Profile name
            tolerance_level: Risk tolerance level
            timeframe_preferences: Preferred timeframes
            preferred_strategy_types: Preferred strategy types
            favorite_currency_pairs: Favorite currency pairs
            custom_settings: Additional custom settings
            
        Returns:
            Created risk profile
        """
        # Convert string to enum if needed
        if isinstance(tolerance_level, str):
            tolerance_level = RiskToleranceLevel[tolerance_level.upper()]
        
        # Get default settings for this tolerance level
        defaults = self.DEFAULT_PROFILES[tolerance_level].copy()
        
        # Apply custom settings if provided
        settings = defaults.copy()
        if custom_settings:
            settings.update(custom_settings)
        
        # Set defaults for preferences if not provided
        timeframe_preferences = timeframe_preferences or ["H1", "H4", "D1"]
        preferred_strategy_types = preferred_strategy_types or ["trend_following", "swing_trading"]
        favorite_currency_pairs = favorite_currency_pairs or ["EURUSD", "GBPUSD", "USDJPY"]
        
        # Create timestamps
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Create the profile
        profile = RiskProfile(
            tolerance_level=tolerance_level,
            max_drawdown_percent=settings["max_drawdown_percent"],
            max_risk_per_trade_percent=settings["max_risk_per_trade_percent"],
            max_portfolio_risk_percent=settings["max_portfolio_risk_percent"],
            position_sizing_method=settings["position_sizing_method"],
            diversification_preference=settings["diversification_preference"],
            timeframe_preferences=timeframe_preferences,
            preferred_strategy_types=preferred_strategy_types,
            favorite_currency_pairs=favorite_currency_pairs,
            trade_frequency_preference=settings["trade_frequency_preference"],
            created_date=timestamp,
            last_updated=timestamp,
            additional_settings=custom_settings or {}
        )
        
        # Save the profile
        self._save_profile(name, profile)
        
        # Set as current profile
        self.current_profile = profile
        self.profile_name = name
        
        logger.info(f"Created new risk profile: {name}")
        
        return profile
    
    def load_profile(self, name: str) -> Optional[RiskProfile]:
        """
        Load a saved risk profile.
        
        Args:
            name: Profile name
            
        Returns:
            Loaded risk profile or None if not found
        """
        profile_path = os.path.join(self.profiles_directory, f"{name}.json")
        
        if not os.path.exists(profile_path):
            logger.warning(f"Risk profile not found: {name}")
            return None
        
        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)
            
            # Convert tolerance level from string to enum
            data["tolerance_level"] = RiskToleranceLevel[data["tolerance_level"]]
            
            # Create RiskProfile from data
            profile = RiskProfile(**data)
            
            # Set as current profile
            self.current_profile = profile
            self.profile_name = name
            
            logger.info(f"Loaded risk profile: {name}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error loading risk profile {name}: {str(e)}")
            return None
    
    def _save_profile(self, name: str, profile: RiskProfile) -> bool:
        """
        Save a risk profile to disk.
        
        Args:
            name: Profile name
            profile: Risk profile to save
            
        Returns:
            True if successful, False otherwise
        """
        profile_path = os.path.join(self.profiles_directory, f"{name}.json")
        
        try:
            # Convert to dictionary
            profile_dict = {k: v if not isinstance(v, Enum) else v.name 
                           for k, v in profile.__dict__.items()}
            
            with open(profile_path, 'w') as f:
                json.dump(profile_dict, f, indent=2)
            
            logger.info(f"Saved risk profile: {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving risk profile {name}: {str(e)}")
            return False
    
    def get_position_size(self, account_balance: float, entry_price: float, 
                         stop_loss: float, symbol: str) -> float:
        """
        Calculate position size based on risk profile.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            
        Returns:
            Position size in base currency
        """
        if not self.current_profile:
            logger.warning("No risk profile loaded, using default risk parameters")
            risk_percent = 1.0  # Default 1% risk
        else:
            risk_percent = self.current_profile.max_risk_per_trade_percent
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100.0)
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            logger.warning("Stop loss is same as entry price, can't calculate position size")
            return 0
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Additional position size adjustments based on profile
        if self.current_profile and hasattr(self.current_profile, 'additional_settings'):
            # Apply any symbol-specific adjustments
            if 'symbol_position_adjustments' in self.current_profile.additional_settings:
                adjustments = self.current_profile.additional_settings['symbol_position_adjustments']
                if symbol in adjustments:
                    position_size *= adjustments[symbol]
        
        return position_size
    
    def adjust_strategy_parameters(self, strategy_type: str, base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on risk profile.
        
        Args:
            strategy_type: Type of strategy
            base_parameters: Base strategy parameters
            
        Returns:
            Adjusted parameters
        """
        if not self.current_profile:
            logger.warning("No risk profile loaded, using default strategy parameters")
            return base_parameters
        
        # Create a copy of the parameters
        adjusted_parameters = base_parameters.copy()
        
        # Get risk tolerance level
        risk_level = self.current_profile.tolerance_level
        
        # Common parameters to adjust based on risk profile
        if 'stop_loss_atr_multiple' in adjusted_parameters:
            # More conservative = wider stops (counterintuitive but reduces chance of being stopped out)
            if risk_level == RiskToleranceLevel.VERY_CONSERVATIVE:
                adjusted_parameters['stop_loss_atr_multiple'] *= 1.5
            elif risk_level == RiskToleranceLevel.CONSERVATIVE:
                adjusted_parameters['stop_loss_atr_multiple'] *= 1.25
            elif risk_level == RiskToleranceLevel.AGGRESSIVE:
                adjusted_parameters['stop_loss_atr_multiple'] *= 0.8
            elif risk_level == RiskToleranceLevel.VERY_AGGRESSIVE:
                adjusted_parameters['stop_loss_atr_multiple'] *= 0.65
        
        if 'take_profit_atr_multiple' in adjusted_parameters:
            # More aggressive = higher targets
            if risk_level == RiskToleranceLevel.VERY_CONSERVATIVE:
                adjusted_parameters['take_profit_atr_multiple'] *= 0.7
            elif risk_level == RiskToleranceLevel.CONSERVATIVE:
                adjusted_parameters['take_profit_atr_multiple'] *= 0.85
            elif risk_level == RiskToleranceLevel.AGGRESSIVE:
                adjusted_parameters['take_profit_atr_multiple'] *= 1.2
            elif risk_level == RiskToleranceLevel.VERY_AGGRESSIVE:
                adjusted_parameters['take_profit_atr_multiple'] *= 1.5
        
        # Strategy-specific adjustments
        if strategy_type == "trend_following":
            # Adjust trend sensitivity
            if 'trend_threshold' in adjusted_parameters:
                if risk_level in [RiskToleranceLevel.VERY_CONSERVATIVE, RiskToleranceLevel.CONSERVATIVE]:
                    # More conservative = stronger trend requirement
                    adjusted_parameters['trend_threshold'] *= 1.25
                elif risk_level in [RiskToleranceLevel.AGGRESSIVE, RiskToleranceLevel.VERY_AGGRESSIVE]:
                    # More aggressive = enter trends earlier
                    adjusted_parameters['trend_threshold'] *= 0.8
        
        elif strategy_type == "breakout":
            # Adjust breakout confirmation
            if 'breakout_confirmation_bars' in adjusted_parameters:
                if risk_level in [RiskToleranceLevel.VERY_CONSERVATIVE, RiskToleranceLevel.CONSERVATIVE]:
                    # More conservative = more confirmation
                    adjusted_parameters['breakout_confirmation_bars'] += 1
                elif risk_level in [RiskToleranceLevel.AGGRESSIVE, RiskToleranceLevel.VERY_AGGRESSIVE]:
                    # More aggressive = faster entry
                    adjusted_parameters['breakout_confirmation_bars'] = max(1, 
                                                                          adjusted_parameters['breakout_confirmation_bars'] - 1)
        
        elif strategy_type == "mean_reversion" or strategy_type == "range_trading":
            # Adjust overbought/oversold thresholds
            if 'overbought' in adjusted_parameters and 'oversold' in adjusted_parameters:
                if risk_level in [RiskToleranceLevel.VERY_CONSERVATIVE, RiskToleranceLevel.CONSERVATIVE]:
                    # More conservative = more extreme levels
                    adjusted_parameters['overbought'] = min(85, adjusted_parameters['overbought'] + 5)
                    adjusted_parameters['oversold'] = max(15, adjusted_parameters['oversold'] - 5)
                elif risk_level in [RiskToleranceLevel.AGGRESSIVE, RiskToleranceLevel.VERY_AGGRESSIVE]:
                    # More aggressive = enter earlier
                    adjusted_parameters['overbought'] = max(65, adjusted_parameters['overbought'] - 5)
                    adjusted_parameters['oversold'] = min(35, adjusted_parameters['oversold'] + 5)
        
        # Handle partial exits based on risk profile
        if 'partial_exits' in adjusted_parameters:
            if risk_level in [RiskToleranceLevel.VERY_CONSERVATIVE, RiskToleranceLevel.CONSERVATIVE]:
                # Conservative = more partial exits, taking profits earlier
                adjusted_parameters['partial_exits'] = True
                if 'partial_exit_levels' in adjusted_parameters and 'partial_exit_sizes' in adjusted_parameters:
                    # Adjust to take profits earlier
                    adjusted_parameters['partial_exit_levels'] = [level * 0.8 for level in adjusted_parameters['partial_exit_levels']]
                    # Take larger portions at early exits
                    total_size = sum(adjusted_parameters['partial_exit_sizes'])
                    if len(adjusted_parameters['partial_exit_sizes']) > 1:
                        adjusted_parameters['partial_exit_sizes'][0] = min(0.5, adjusted_parameters['partial_exit_sizes'][0] * 1.5)
                        # Redistribute remaining size
                        remaining = total_size - adjusted_parameters['partial_exit_sizes'][0]
                        for i in range(1, len(adjusted_parameters['partial_exit_sizes'])):
                            adjusted_parameters['partial_exit_sizes'][i] = remaining / (len(adjusted_parameters['partial_exit_sizes']) - 1)
            
            elif risk_level in [RiskToleranceLevel.AGGRESSIVE, RiskToleranceLevel.VERY_AGGRESSIVE]:
                # Aggressive = fewer partial exits, holding for bigger moves
                if 'partial_exit_levels' in adjusted_parameters and 'partial_exit_sizes' in adjusted_parameters:
                    # Adjust to take profits later
                    adjusted_parameters['partial_exit_levels'] = [min(1.0, level * 1.2) for level in adjusted_parameters['partial_exit_levels']]
                    # Take smaller portions at early exits
                    if len(adjusted_parameters['partial_exit_sizes']) > 1:
                        adjusted_parameters['partial_exit_sizes'][0] = max(0.1, adjusted_parameters['partial_exit_sizes'][0] * 0.7)
                        # Increase later exit sizes
                        for i in range(1, len(adjusted_parameters['partial_exit_sizes'])):
                            factor = 1 + (0.3 * (i / (len(adjusted_parameters['partial_exit_sizes']) - 1)))
                            adjusted_parameters['partial_exit_sizes'][i] *= factor
                        # Normalize to ensure sum is 1.0
                        total = sum(adjusted_parameters['partial_exit_sizes'])
                        adjusted_parameters['partial_exit_sizes'] = [size / total for size in adjusted_parameters['partial_exit_sizes']]
        
        return adjusted_parameters
    
    def get_strategy_compatibility(self, strategy_type: str) -> float:
        """
        Calculate compatibility score between a strategy and the risk profile.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Compatibility score (0-1)
        """
        if not self.current_profile:
            # Default moderate compatibility
            return 0.5
        
        # Base compatibility based on preferred strategies
        base_score = 0.5
        if strategy_type in self.current_profile.preferred_strategy_types:
            base_score = 0.8
        
        # Adjust based on risk tolerance and strategy type
        risk_level = self.current_profile.tolerance_level
        
        # Strategy-specific adjustments
        if strategy_type in ["trend_following", "position_trading"]:
            # Trend and position strategies work better with patient, moderate to aggressive profiles
            if risk_level in [RiskToleranceLevel.MODERATE, RiskToleranceLevel.AGGRESSIVE]:
                base_score += 0.15
            elif risk_level == RiskToleranceLevel.VERY_AGGRESSIVE:
                base_score -= 0.1  # Too aggressive may overtrade
        
        elif strategy_type in ["scalping", "day_trading"]:
            # Day trading and scalping better for more active, aggressive traders
            if risk_level in [RiskToleranceLevel.AGGRESSIVE, RiskToleranceLevel.VERY_AGGRESSIVE]:
                base_score += 0.2
            elif risk_level in [RiskToleranceLevel.VERY_CONSERVATIVE, RiskToleranceLevel.CONSERVATIVE]:
                base_score -= 0.2  # Too conservative for quick trading
        
        elif strategy_type in ["mean_reversion", "range_trading", "counter_trend"]:
            # Mean reversion strategies require discipline and moderate risk
            if risk_level == RiskToleranceLevel.MODERATE:
                base_score += 0.15
            elif risk_level == RiskToleranceLevel.VERY_AGGRESSIVE:
                base_score -= 0.15  # Too aggressive may exit too early
        
        elif strategy_type in ["breakout", "momentum"]:
            # Breakout strategies work well with moderate to aggressive profiles
            if risk_level in [RiskToleranceLevel.MODERATE, RiskToleranceLevel.AGGRESSIVE]:
                base_score += 0.1
        
        elif strategy_type in ["carry_trade"]:
            # Carry trades are longer-term and better for patient profiles
            if risk_level in [RiskToleranceLevel.CONSERVATIVE, RiskToleranceLevel.MODERATE]:
                base_score += 0.15
            elif risk_level == RiskToleranceLevel.VERY_AGGRESSIVE:
                base_score -= 0.2  # Too aggressive for long-term holds
        
        # Cap the score between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    def generate_risk_questionnaire(self) -> List[Dict[str, Any]]:
        """
        Generate a risk profiling questionnaire for users.
        
        Returns:
            List of questions with options
        """
        return [
            {
                "id": "investment_horizon",
                "question": "How long do you typically hold a trading position?",
                "options": [
                    {"value": "minutes", "text": "Minutes to hours", "points": 5},
                    {"value": "day", "text": "Intraday to 1 day", "points": 4},
                    {"value": "days", "text": "Several days", "points": 3},
                    {"value": "weeks", "text": "Weeks", "points": 2},
                    {"value": "months", "text": "Months or longer", "points": 1}
                ]
            },
            {
                "id": "drawdown_tolerance",
                "question": "What's the maximum drawdown you could comfortably tolerate?",
                "options": [
                    {"value": "5_percent", "text": "5% or less", "points": 1},
                    {"value": "10_percent", "text": "Up to 10%", "points": 2},
                    {"value": "15_percent", "text": "Up to 15%", "points": 3},
                    {"value": "25_percent", "text": "Up to 25%", "points": 4},
                    {"value": "35_percent", "text": "Up to 35% or more", "points": 5}
                ]
            },
            {
                "id": "frequency",
                "question": "How frequently do you prefer to trade?",
                "options": [
                    {"value": "very_low", "text": "A few times per month", "points": 1},
                    {"value": "low", "text": "A few times per week", "points": 2},
                    {"value": "medium", "text": "Daily", "points": 3},
                    {"value": "high", "text": "Multiple times per day", "points": 4},
                    {"value": "very_high", "text": "Many times throughout the day", "points": 5}
                ]
            },
            {
                "id": "strategy_preference",
                "question": "Which trading approach appeals to you most?",
                "options": [
                    {"value": "value", "text": "Value-based, longer-term positions", "points": 1},
                    {"value": "trend", "text": "Following established trends", "points": 2},
                    {"value": "swing", "text": "Capturing medium-term market swings", "points": 3},
                    {"value": "momentum", "text": "Riding strong market momentum", "points": 4},
                    {"value": "scalping", "text": "Quick in-and-out trades for small profits", "points": 5}
                ]
            },
            {
                "id": "loss_reaction",
                "question": "When a trade moves against you, your typical reaction is to:",
                "options": [
                    {"value": "exit_immediately", "text": "Exit immediately to protect capital", "points": 1},
                    {"value": "reduce_position", "text": "Reduce position size but maintain exposure", "points": 2},
                    {"value": "wait_reassess", "text": "Wait and reassess based on technical levels", "points": 3},
                    {"value": "add_position", "text": "Consider adding to the position if conviction remains", "points": 4},
                    {"value": "double_down", "text": "Double down to improve average entry price", "points": 5}
                ]
            }
        ]
    
    def evaluate_risk_questionnaire(self, answers: Dict[str, str]) -> RiskToleranceLevel:
        """
        Evaluate questionnaire answers to determine risk tolerance level.
        
        Args:
            answers: Dictionary of question IDs to answer values
            
        Returns:
            Calculated risk tolerance level
        """
        questionnaire = self.generate_risk_questionnaire()
        
        # Build mapping of option values to points
        points_map = {}
        for question in questionnaire:
            question_id = question["id"]
            for option in question["options"]:
                key = f"{question_id}_{option['value']}"
                points_map[key] = option["points"]
        
        # Calculate total points
        total_points = 0
        for question_id, answer_value in answers.items():
            key = f"{question_id}_{answer_value}"
            if key in points_map:
                total_points += points_map[key]
        
        # Determine risk level based on total points
        max_possible = len(questionnaire) * 5  # 5 points per question is max
        
        # Calculate percentage of maximum
        percentage = total_points / max_possible
        
        # Map to risk tolerance levels
        if percentage < 0.2:
            return RiskToleranceLevel.VERY_CONSERVATIVE
        elif percentage < 0.4:
            return RiskToleranceLevel.CONSERVATIVE
        elif percentage < 0.6:
            return RiskToleranceLevel.MODERATE
        elif percentage < 0.8:
            return RiskToleranceLevel.AGGRESSIVE
        else:
            return RiskToleranceLevel.VERY_AGGRESSIVE
