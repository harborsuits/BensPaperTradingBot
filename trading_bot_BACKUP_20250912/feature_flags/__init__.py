"""
Trading Bot Feature Flags Package

This package provides a robust feature flag system for the trading bot,
allowing for selective enabling/disabling of trading strategies or risk
features without requiring a full deployment.

The system includes:
- Feature flag management with categories
- Remote management via Telegram
- Persistence across restarts
- Automatic rollback of unstable features
- Flag usage metrics
"""

from .service import (
    get_feature_flag_service,
    FeatureFlag,
    FeatureFlagService,
    FlagCategory,
    FlagChangeCallback,
    FlagChangeEvent,
)

__all__ = [
    'get_feature_flag_service',
    'FeatureFlag',
    'FeatureFlagService',
    'FlagCategory',
    'FlagChangeCallback',
    'FlagChangeEvent',
] 