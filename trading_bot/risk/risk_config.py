#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Configuration

This module provides configuration settings for the risk management system,
including weights for different risk factors and thresholds for alerts.
These settings can be adjusted to fine-tune the system's sensitivity to
different types of risk.
"""
from typing import Dict, Any

# Default risk thresholds
DEFAULT_RISK_THRESHOLDS = {
    "max_portfolio_risk": 0.05,        # 5% maximum portfolio risk
    "correlation_threshold": 0.7,       # Alert on correlations above 0.7
    "max_position_size": 0.2,           # No position can be > 20% of portfolio
    "drawdown_threshold": 0.1,          # Alert on 10% drawdowns
    "risk_per_trade": 0.01,             # Risk 1% per trade
    "sector_concentration_limit": 0.3,  # Maximum 30% in any sector
    "factor_exposure_limit": 0.4,       # Maximum 40% exposure to any factor
    "liquidity_threshold": 0.6          # Minimum liquidity score (0-1)
}

# Risk weights for strategy rotation (conservative profile)
CONSERVATIVE_RISK_WEIGHTS = {
    "market_beta": 0.25,       # Market exposure risk
    "volatility": 0.25,        # Price volatility risk
    "correlation": 0.15,       # Asset correlation risk
    "sector_exposure": 0.15,   # Sector concentration risk
    "liquidity": 0.10,         # Liquidity risk
    "factor_tilt": 0.10        # Factor exposure risk
}

# Risk weights for strategy rotation (balanced profile)
BALANCED_RISK_WEIGHTS = {
    "market_beta": 0.20,       # Market exposure risk
    "volatility": 0.20,        # Price volatility risk
    "correlation": 0.15,       # Asset correlation risk
    "sector_exposure": 0.15,   # Sector concentration risk
    "liquidity": 0.10,         # Liquidity risk
    "factor_tilt": 0.20        # Factor exposure risk
}

# Risk weights for strategy rotation (aggressive profile)
AGGRESSIVE_RISK_WEIGHTS = {
    "market_beta": 0.15,       # Market exposure risk
    "volatility": 0.15,        # Price volatility risk
    "correlation": 0.10,       # Asset correlation risk
    "sector_exposure": 0.15,   # Sector concentration risk
    "liquidity": 0.10,         # Liquidity risk
    "factor_tilt": 0.35        # Factor exposure risk
}

# Market regime compatibility scores (0-1) for different strategy types
# Higher means better suited for that regime
REGIME_COMPATIBILITY = {
    "trending": {
        "momentum": 0.9,
        "breakout": 0.8,
        "trend_following": 0.9,
        "value": 0.5,
        "mean_reversion": 0.3,
        "low_volatility": 0.4,
        "multi_factor": 0.6
    },
    "volatile": {
        "momentum": 0.3,
        "breakout": 0.5,
        "trend_following": 0.3,
        "value": 0.5,
        "mean_reversion": 0.7,
        "low_volatility": 0.9,
        "multi_factor": 0.6
    },
    "mean_reverting": {
        "momentum": 0.2,
        "breakout": 0.3,
        "trend_following": 0.2,
        "value": 0.6,
        "mean_reversion": 0.9,
        "low_volatility": 0.7,
        "multi_factor": 0.6
    },
    "normal": {
        "momentum": 0.6,
        "breakout": 0.6,
        "trend_following": 0.6,
        "value": 0.7,
        "mean_reversion": 0.6,
        "low_volatility": 0.6,
        "multi_factor": 0.8
    }
}

# Defensive rotation settings for high-risk environments
DEFENSIVE_ROTATION = {
    "max_strategies": 3,          # Reduce to fewer strategies
    "beta_target": 0.3,           # Target lower beta
    "volatility_target": 0.3,     # Target lower volatility
    "correlation_target": 0.4,    # Target lower correlation
    "preferred_strategies": [     # Strategy types to prioritize
        "low_volatility",
        "defensive_value",
        "market_neutral"
    ]
}

# Factor exposure limits
FACTOR_EXPOSURE_LIMITS = {
    "value": 0.4,
    "momentum": 0.4,
    "size": 0.3,
    "quality": 0.4,
    "volatility": 0.3,
    "growth": 0.3
}

class RiskConfigManager:
    """Risk Configuration Manager for fine-tuning risk weights and thresholds.
    
    Provides methods for loading, saving, and managing risk configurations.
    Supports different risk profiles (conservative, balanced, aggressive, high_growth).
    """
    def __init__(self, profile="balanced"):
        """Initialize the risk configuration manager.
        
        Args:
            profile (str): Risk profile to use (conservative, balanced, aggressive, high_growth)
        """
        self.profile = profile
        self.config = self._load_profile(profile)
        
    def _load_profile(self, profile):
        """Load a risk profile configuration.
        
        Args:
            profile (str): Risk profile to load
            
        Returns:
            dict: Risk configuration
        """
        profiles = self.get_profile_presets()
        
        if profile in profiles:
            profile_data = profiles[profile]
            
            # Convert from profiles format to internal config format
            return {
                "risk_weights": profile_data["risk_weights"],
                "thresholds": {
                    "max_drawdown": profile_data["drawdown_threshold"],
                    "correlation_alert": profile_data["correlation_threshold"],
                    "max_position_size": profile_data["max_position_size"],
                    "max_sector_exposure": profile_data["sector_limits"]["max_sector_exposure"],
                    "min_liquidity_score": profile_data["liquidity_requirements"]["min_liquidity_score"],
                    "max_portfolio_risk": profile_data["max_portfolio_risk"],
                }
            }
        else:
            # Default to balanced if profile not found
            return self._load_profile("balanced")
        
    def get_risk_weights(self) -> Dict[str, float]:
        """
        Get risk factor weights based on the current profile.
        
        Returns:
            Dictionary of risk factor weights
        """
        if self.profile == "conservative":
            weights = CONSERVATIVE_RISK_WEIGHTS.copy()
        elif self.profile == "aggressive":
            weights = AGGRESSIVE_RISK_WEIGHTS.copy()
        else:  # balanced is the default
            weights = BALANCED_RISK_WEIGHTS.copy()
            
        # Apply any custom settings
        for factor, weight in self.config.get("custom_settings", {}).get("risk_weights", {}).items():
            if factor in weights:
                weights[factor] = weight
                
        return weights
    
    def get_risk_thresholds(self) -> Dict[str, float]:
        """
        Get risk thresholds based on the current profile.
        
        Returns:
            Dictionary of risk thresholds
        """
        thresholds = self.config["thresholds"].copy()
        
        # Adjust thresholds based on profile
        if self.profile == "conservative":
            thresholds["max_portfolio_risk"] = 0.04      # 4%
            thresholds["correlation_threshold"] = 0.65    # 65%
            thresholds["max_position_size"] = 0.15       # 15%
            thresholds["drawdown_threshold"] = 0.08      # 8%
            thresholds["risk_per_trade"] = 0.005         # 0.5%
        elif self.profile == "aggressive":
            thresholds["max_portfolio_risk"] = 0.07      # 7%
            thresholds["correlation_threshold"] = 0.75    # 75%
            thresholds["max_position_size"] = 0.25       # 25%
            thresholds["drawdown_threshold"] = 0.12      # 12%
            thresholds["risk_per_trade"] = 0.015         # 1.5%
            
        # Apply any custom settings
        for threshold, value in self.config.get("custom_settings", {}).get("risk_thresholds", {}).items():
            if threshold in thresholds:
                thresholds[threshold] = value
                
        return thresholds
    
    def get_regime_compatibility(self, regime: str) -> Dict[str, float]:
        """
        Get strategy compatibility scores for a specific market regime.
        
        Args:
            regime: Market regime name
            
        Returns:
            Dictionary of strategy type to compatibility score
        """
        if regime in REGIME_COMPATIBILITY:
            return REGIME_COMPATIBILITY[regime].copy()
        return REGIME_COMPATIBILITY["normal"].copy()
    
    def get_defensive_settings(self) -> Dict[str, Any]:
        """
        Get settings for defensive rotation during high-risk periods.
        
        Returns:
            Dictionary of defensive rotation settings
        """
        settings = DEFENSIVE_ROTATION.copy()
        
        # Adjust settings based on profile
        if self.profile == "conservative":
            settings["max_strategies"] = 2
            settings["beta_target"] = 0.25
            settings["volatility_target"] = 0.25
        elif self.profile == "aggressive":
            settings["max_strategies"] = 4
            settings["beta_target"] = 0.4
            settings["volatility_target"] = 0.4
            
        # Apply any custom settings
        for setting, value in self.config.get("custom_settings", {}).get("defensive_settings", {}).items():
            if setting in settings:
                settings[setting] = value
                
        return settings
    
    def get_factor_exposure_limits(self) -> Dict[str, float]:
        """
        Get factor exposure limits based on the current profile.
        
        Returns:
            Dictionary of factor exposure limits
        """
        limits = FACTOR_EXPOSURE_LIMITS.copy()
        
        # Apply any custom settings
        for factor, limit in self.config.get("custom_settings", {}).get("factor_exposure_limits", {}).items():
            if factor in limits:
                limits[factor] = limit
                
        return limits
    
    def set_custom_setting(self, category: str, setting: str, value: Any) -> None:
        """
        Set a custom risk setting.
        
        Args:
            category: Setting category (e.g., "risk_weights", "risk_thresholds")
            setting: Setting name
            value: Setting value
        """
        if category not in self.config.get("custom_settings", {}):
            self.config["custom_settings"][category] = {}
            
        self.config["custom_settings"][category][setting] = value
        
    def get_risk_config(self) -> Dict[str, Any]:
        """
        Get the complete risk configuration.
        
        Returns:
            Complete risk configuration dictionary
        """
        return self.config
        
    def update_risk_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the risk configuration.
        
        Args:
            new_config: New risk configuration
        """
        self.config = new_config
        
    def set_profile(self, profile: str) -> None:
        """
        Set the risk profile.
        
        Args:
            profile: Risk profile to use
        """
        self.profile = profile
        self.config = self._load_profile(profile)
        
    def get_profile_presets(self) -> Dict[str, Any]:
        """
        Get a dictionary of preset risk profiles.
        
        Returns:
            Dictionary of preset risk profiles with descriptions
        """
        return {
            "conservative": {
                "name": "Conservative",
                "description": "Lower returns but minimal drawdowns, prioritizes capital preservation",
                "max_portfolio_risk": 0.08,  # 8% max portfolio risk
                "correlation_threshold": 0.75,
                "max_position_size": 0.05,  # 5% max position size
                "drawdown_threshold": 0.05,  # 5% drawdown protection
                "risk_per_trade": 0.005,  # 0.5% risk per trade
                "risk_weights": {
                    "market_beta": 0.25,
                    "volatility": 0.25,
                    "correlation": 0.20,
                    "liquidity": 0.15,
                    "sector_exposure": 0.10,
                    "factor_tilt": 0.05,
                },
                "sector_limits": {
                    "max_sector_exposure": 0.25,  # 25% max sector exposure
                    "max_sectors_above_threshold": 2
                },
                "liquidity_requirements": {
                    "min_liquidity_score": 0.7,
                    "max_days_to_liquidate": 2.0
                },
                "factor_tilt_limits": {
                    "max_factor_exposure": 0.3,  # 30% max factor exposure
                    "max_factors_above_threshold": 1
                }
            },
            "balanced": {
                "name": "Balanced",
                "description": "Moderate risk/return profile, suitable for most investors",
                "max_portfolio_risk": 0.12,  # 12% max portfolio risk
                "correlation_threshold": 0.70,
                "max_position_size": 0.08,  # 8% max position size
                "drawdown_threshold": 0.08,  # 8% drawdown protection
                "risk_per_trade": 0.01,  # 1% risk per trade
                "risk_weights": {
                    "market_beta": 0.20,
                    "volatility": 0.20,
                    "correlation": 0.20,
                    "liquidity": 0.15,
                    "sector_exposure": 0.15,
                    "factor_tilt": 0.10,
                },
                "sector_limits": {
                    "max_sector_exposure": 0.30,  # 30% max sector exposure
                    "max_sectors_above_threshold": 3
                },
                "liquidity_requirements": {
                    "min_liquidity_score": 0.6,
                    "max_days_to_liquidate": 3.0
                },
                "factor_tilt_limits": {
                    "max_factor_exposure": 0.4,  # 40% max factor exposure
                    "max_factors_above_threshold": 2
                }
            },
            "aggressive": {
                "name": "Aggressive",
                "description": "Higher return potential with increased risk, suitable for growth-oriented investors",
                "max_portfolio_risk": 0.18,  # 18% max portfolio risk
                "correlation_threshold": 0.65,
                "max_position_size": 0.12,  # 12% max position size
                "drawdown_threshold": 0.12,  # 12% drawdown protection
                "risk_per_trade": 0.02,  # 2% risk per trade
                "risk_weights": {
                    "market_beta": 0.15,
                    "volatility": 0.15,
                    "correlation": 0.15,
                    "liquidity": 0.15,
                    "sector_exposure": 0.20,
                    "factor_tilt": 0.20,
                },
                "sector_limits": {
                    "max_sector_exposure": 0.40,  # 40% max sector exposure
                    "max_sectors_above_threshold": 4
                },
                "liquidity_requirements": {
                    "min_liquidity_score": 0.5,
                    "max_days_to_liquidate": 5.0
                },
                "factor_tilt_limits": {
                    "max_factor_exposure": 0.5,  # 50% max factor exposure
                    "max_factors_above_threshold": 3
                }
            },
            "high_growth": {
                "name": "High Growth",
                "description": "Maximizes growth potential with substantial risk, suitable for long-term investors with high risk tolerance",
                "max_portfolio_risk": 0.25,  # 25% max portfolio risk
                "correlation_threshold": 0.60,
                "max_position_size": 0.15,  # 15% max position size
                "drawdown_threshold": 0.15,  # 15% drawdown protection
                "risk_per_trade": 0.03,  # 3% risk per trade
                "risk_weights": {
                    "market_beta": 0.10,
                    "volatility": 0.10,
                    "correlation": 0.10,
                    "liquidity": 0.10,
                    "sector_exposure": 0.30,
                    "factor_tilt": 0.30,
                },
                "sector_limits": {
                    "max_sector_exposure": 0.50,  # 50% max sector exposure
                    "max_sectors_above_threshold": 5
                },
                "liquidity_requirements": {
                    "min_liquidity_score": 0.4,
                    "max_days_to_liquidate": 7.0
                },
                "factor_tilt_limits": {
                    "max_factor_exposure": 0.6,  # 60% max factor exposure
                    "max_factors_above_threshold": 4
                }
            }
        }
