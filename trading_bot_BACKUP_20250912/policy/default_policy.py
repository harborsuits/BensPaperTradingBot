"""
Default trading policy configuration.

This module provides the default policy settings for the trading router,
including risk limits, compliance rules, and scoring weights.
"""

from trading_bot.policy.types import Policy, Instrument

# Default policy configuration
default_policy: Policy = {
    "version": "v1.0.0",
    "risk": {
        "max_daily_loss_pct": 2.0,
        "max_gross_exposure_pct": 80.0,
        "per_instrument_max_pct": {
            Instrument.ETF.value: 60.0,
            Instrument.CRYPTO.value: 20.0,
            Instrument.OPTIONS.value: 10.0
        },
        "options": {
            "max_iv_pctile": 90.0,
            "max_dte_days": 21,
            "greek_caps": {
                "net_delta": 0.4,
                "net_gamma": 0.2,
                "net_vega": 0.5
            }
        },
        "crypto": {
            "max_slippage_bps": 15.0,
            "max_dwell_ms": 2500,
            "venue_whitelist": ["Coinbase", "Kraken"]
        },
        "etf": {
            "min_adv_usd": 10_000_000,
            "max_spread_bps": 5.0
        }
    },
    "compliance": {
        "insider_trade_sources": "forbidden",
        "public_disclosures": "allowed_lagged",
        "venue_whitelist": {
            Instrument.ETF.value: ["NYSE", "NASDAQ"],
            Instrument.CRYPTO.value: ["Coinbase", "Kraken"],
            Instrument.OPTIONS.value: ["OCC"]
        },
        "jurisdictions": ["US"]
    },
    "weights": {
        "alpha": 0.5,
        "regime": 0.2,
        "sentiment": 0.2,
        "cost": 0.05,
        "risk": 0.05
    },
    "toggles": {
        "enable_crypto_burst": True,
        "enable_options_events": True,
        "enable_etf_trend": True
    },
    "stale_after_ms": 15_000
}
