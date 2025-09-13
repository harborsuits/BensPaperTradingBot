"""
Compliance gates for trading decisions.

This module implements the compliance checks that must pass before
any trading opportunity can be considered. It enforces legal, regulatory,
and policy-based restrictions.
"""

from typing import Dict, List, Any, TypedDict, Optional

from trading_bot.policy.types import Policy, Instrument


class GateFail(TypedDict):
    """Represents a failed compliance gate with reason code and details."""
    code: str
    detail: Optional[str]


class OpportunityMetadata(TypedDict, total=False):
    """Metadata about a trading opportunity for compliance checks."""
    venue: Optional[str]
    jurisdiction: Optional[str]
    uses_public_disclosure: Optional[bool]
    disclosure_lag_ms: Optional[int]
    adv_usd: Optional[float]
    spread_bps: Optional[float]
    iv_pctile: Optional[float]
    dte_days: Optional[int]


def compliance_gates(instrument: str, meta: OpportunityMetadata, policy: Policy) -> List[GateFail]:
    """
    Check if an opportunity passes all compliance gates.
    
    Args:
        instrument: The instrument type (ETF, CRYPTO, OPTIONS)
        meta: Metadata about the opportunity
        policy: The current trading policy
        
    Returns:
        List of failed gates (empty if all gates pass)
    """
    fails: List[GateFail] = []

    # Jurisdiction / Venue checks
    if meta.get("jurisdiction") and meta["jurisdiction"] not in policy["compliance"]["jurisdictions"]:
        fails.append({"code": "jurisdiction_block", "detail": meta["jurisdiction"]})
    
    if meta.get("venue") and meta["venue"] not in policy["compliance"]["venue_whitelist"].get(instrument, []):
        fails.append({"code": "venue_not_whitelisted", "detail": meta["venue"]})

    # "Politician mode" legality - public disclosures must be lagged
    if meta.get("uses_public_disclosure"):
        if (policy["compliance"]["public_disclosures"] != "allowed_lagged" or 
            (meta.get("disclosure_lag_ms", 0) < 24 * 60 * 60 * 1000)):  # 24 hours in ms
            fails.append({"code": "disclosure_not_lagged", "detail": None})

    # Instrument-specific checks
    if instrument == Instrument.ETF.value:
        if (meta.get("adv_usd", 0) < policy["risk"]["etf"]["min_adv_usd"]):
            fails.append({"code": "etf_low_adv", "detail": None})
        if (meta.get("spread_bps", 999) > policy["risk"]["etf"]["max_spread_bps"]):
            fails.append({"code": "etf_wide_spread", "detail": None})
    
    elif instrument == Instrument.OPTIONS.value:
        if (meta.get("iv_pctile", 101) > policy["risk"]["options"]["max_iv_pctile"]):
            fails.append({"code": "options_iv_too_high", "detail": None})
        if (meta.get("dte_days", 999) > policy["risk"]["options"]["max_dte_days"]):
            fails.append({"code": "options_dte_too_long", "detail": None})
    
    elif instrument == Instrument.CRYPTO.value:
        if meta.get("venue") and meta["venue"] not in policy["risk"]["crypto"]["venue_whitelist"]:
            fails.append({"code": "crypto_venue_restricted", "detail": meta["venue"]})

    return fails
