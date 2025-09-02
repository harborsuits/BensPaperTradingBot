"""
Policy types for the trading bot's decision router.

This module defines the core data structures for policy-driven
trading decisions with strict compliance gates and risk limits.
"""

from enum import Enum
from typing import Dict, List, Optional, TypedDict, Literal


class Instrument(str, Enum):
    """Trading instrument types supported by the system."""
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    OPTIONS = "OPTIONS"


class Regime(str, Enum):
    """Market regime classifications."""
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class OptionsRiskLimits(TypedDict):
    """Risk limits specific to options trading."""
    max_iv_pctile: float  # Maximum implied volatility percentile (0-100)
    max_dte_days: int     # Maximum days to expiration
    greek_caps: Dict[str, float]  # Caps for delta, gamma, vega, etc.


class CryptoRiskLimits(TypedDict):
    """Risk limits specific to crypto trading."""
    max_slippage_bps: float  # Maximum allowed slippage in basis points
    max_dwell_ms: int        # Maximum time to keep orders open in milliseconds
    venue_whitelist: List[str]  # Allowed trading venues


class ETFRiskLimits(TypedDict):
    """Risk limits specific to ETF trading."""
    min_adv_usd: float  # Minimum average daily volume in USD
    max_spread_bps: float  # Maximum bid-ask spread in basis points


class RiskLimits(TypedDict):
    """Comprehensive risk limits across all instrument types."""
    max_daily_loss_pct: float  # Maximum daily loss as percentage of portfolio
    max_gross_exposure_pct: float  # Maximum gross exposure as percentage of portfolio
    per_instrument_max_pct: Dict[str, float]  # Maximum allocation per instrument type
    options: OptionsRiskLimits
    crypto: CryptoRiskLimits
    etf: ETFRiskLimits


class Compliance(TypedDict):
    """Compliance rules and restrictions."""
    insider_trade_sources: Literal["forbidden"]
    public_disclosures: Literal["allowed_lagged", "forbidden"]
    venue_whitelist: Dict[str, List[str]]  # Allowed venues per instrument
    jurisdictions: List[str]  # Allowed trading jurisdictions


class ScoringWeights(TypedDict):
    """Weights for the opportunity scoring algorithm."""
    alpha: float
    regime: float
    sentiment: float
    cost: float
    risk: float


class PlaybookToggles(TypedDict):
    """Feature toggles for trading playbooks."""
    enable_crypto_burst: bool
    enable_options_events: bool
    enable_etf_trend: bool


class Policy(TypedDict):
    """Complete trading policy configuration."""
    version: str
    risk: RiskLimits
    compliance: Compliance
    weights: ScoringWeights
    toggles: PlaybookToggles
    stale_after_ms: int  # Drop stale opportunities after this time
