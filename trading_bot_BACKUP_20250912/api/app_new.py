#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot API - FastAPI application for the trading bot
providing UI and API endpoints for monitoring and control.
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
import asyncio
from starlette.concurrency import run_in_threadpool
from trading_bot.core.alerts.telegram import notify
from trading_bot.core.safety.pause_state import set_paused, is_paused
from trading_bot.core.safety.guardrails import evaluate_order as _evaluate_order
from trading_bot.core.safety.context import RiskPolicy
from trading_bot.core.market.earnings_blackout import is_earnings_blackout as _earnings_blackout

# Import all routers
from trading_bot.api.routers.portfolio import router as portfolio_router
from trading_bot.api.routers.trades import router as trades_router
from trading_bot.api.routers.strategies import router as strategies_router
from trading_bot.api.routers.news import router as news_router
from trading_bot.api.routers.alerts import router as alerts_router
from trading_bot.api.websocket_endpoint import router as ws_router
from trading_bot.api.routers.safety import router as safety_router
from trading_bot.api.routers.metrics import router as metrics_router
from trading_bot.api.routers.compat import router as compat_router
from trading_bot.api.routers.compat_logs import router as compat_logs_router
from trading_bot.api.routers.compat_context import router as compat_context_router
from trading_bot.api.routers.compat_metrics import router as compat_metrics_router
from trading_bot.metrics import metrics_router
from trading_bot.api.routers.decisions import router as decisions_router

# Import specific modules if available
try:
    from trading_bot.auth.api import router as auth_router
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False
    
try:
    from trading_bot.api.event_endpoints import event_api
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    
try:
    from trading_bot.api.backtest_endpoints import router as backtest_api
    HAS_BACKTEST = True
except ImportError:
    HAS_BACKTEST = False

# Initialize logging
logger = logging.getLogger("TradingBotAPI")

# Initialize FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="API for monitoring and controlling the trading bot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with proper prefixes
app.include_router(portfolio_router, prefix="/api", tags=["portfolio"])
app.include_router(trades_router, prefix="/api", tags=["trades"])
app.include_router(strategies_router, prefix="/api", tags=["strategies"])
app.include_router(news_router, prefix="/api", tags=["news"])
app.include_router(alerts_router, prefix="/api", tags=["alerts"])
app.include_router(ws_router)
app.include_router(metrics_router)
app.include_router(safety_router, prefix="/api", tags=["safety"])
app.include_router(compat_router)
app.include_router(compat_logs_router)
app.include_router(compat_context_router)
app.include_router(compat_metrics_router)
app.include_router(metrics_router)
app.include_router(decisions_router)

# Include optional routers if available
if HAS_AUTH:
    app.include_router(auth_router, tags=["auth"])
    
if HAS_EVENTS:
    app.include_router(event_api, tags=["events"])
    
if HAS_BACKTEST:
    app.include_router(backtest_api, tags=["backtest"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API info."""
    return {
        "message": "Trading Bot API",
        "version": "1.0.0",
        "endpoints": [
            "/",
            "/health",
            "/metrics",
            "/api/portfolio",
            "/api/trades",
            "/api/strategies",
            "/api/news",
            "/api/alerts",
            "/ws"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Import here to avoid heavy import at module load
    try:
        from trading_bot.monitor.freshness import get_freshness
        freshness = get_freshness()
    except Exception:
        freshness = {}
    import os
    flags = {
        "AI_SCORING_ENABLED": os.environ.get("AI_SCORING_ENABLED", "false"),
        "OPTIONS_ENABLED": os.environ.get("OPTIONS_ENABLED", "false"),
        "WS_LIGHT_MODE": os.environ.get("WS_LIGHT_MODE", "true"),
        "TRADING_MODE": os.environ.get("TRADING_MODE", "paper"),
    }
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "flags": flags,
        "freshness": freshness,
        "entries_paused": is_paused()
    }

# ------------------ Reconcile Scheduler ------------------
logger = logging.getLogger("reconcile")
RECONCILE_INTERVAL_SEC = int(os.environ.get("RECONCILE_INTERVAL_SEC", "30"))

def _get_trade_executor():
    # Return shared executor if initialized elsewhere
    return getattr(app.state, "trade_executor", None)


def initialize_shared_executor() -> None:
    """(Re)initialize the shared executor based on TRADING_MODE env.
    Paper: PaperTradeAdapter
    Live: Tradier TradeExecutor (tiny caps)
    """
    mode = os.environ.get("TRADING_MODE", "paper").lower()
    try:
        if mode == "paper":
            from trading_bot.brokers.paper.adapter import PaperTradeAdapter, PaperTradeConfig
            paper = PaperTradeAdapter()
            paper.connect(PaperTradeConfig().to_dict())
            app.state.trade_executor = paper
            logger.info("Shared executor set to PaperTradeAdapter")
        else:
            from trading_bot.brokers.tradier_client import TradierClient
            from trading_bot.brokers.trade_executor import TradeExecutor
            token = os.environ.get("TRADIER_TOKEN")
            account = os.environ.get("TRADIER_ACCOUNT_ID")
            sandbox_env = os.environ.get("TRADIER_SANDBOX", "true").lower() in ("1", "true", "yes")
            if not token or not account:
                raise RuntimeError("TRADIER_TOKEN and TRADIER_ACCOUNT_ID are required for live mode")
            client = TradierClient(api_key=token, account_id=account, sandbox=sandbox_env)
            # Tiny caps for first week
            app.state.trade_executor = TradeExecutor(
                tradier_client=client,
                max_position_pct=0.01,
                max_risk_pct=0.002,
            )
            logger.info("Shared executor set to TradeExecutor (Tradier) with tiny caps")
    except Exception as e:
        logger.error(f"Executor initialization failed: {e}")
        raise

async def _reconcile_loop():
    await asyncio.sleep(2)
    logger.info(f"Reconcile loop started (every {RECONCILE_INTERVAL_SEC}s)")
    while True:
        try:
            exec_ = _get_trade_executor()
            if exec_ is not None and hasattr(exec_, "reconcile_positions"):
                diffs = await run_in_threadpool(exec_.reconcile_positions)
                if diffs:
                    set_paused(True)
                    logger.warning(f"RECONCILE_MISMATCH {diffs}")
                    try:
                        notify("⚠️ Reconcile mismatch detected — entries paused. POST /safety/ack-reconcile to resume.")
                    except Exception:
                        pass
                # increment counters on app state for /metrics
                app.state.reconcile_runs = getattr(app.state, "reconcile_runs", 0) + 1
                if diffs:
                    app.state.reconcile_mismatches_total = getattr(app.state, "reconcile_mismatches_total", 0) + 1
        except Exception as e:
            logger.error(f"Reconcile loop error: {e}")
        await asyncio.sleep(RECONCILE_INTERVAL_SEC)

@app.on_event("startup")
async def _startup_tasks():
    asyncio.create_task(_reconcile_loop())
    # Bind a shared evaluate function using the current app context
    policy = RiskPolicy()

    def _is_emergency_stop() -> bool:
        return getattr(app.state, "emergency_stop", False)

    def _get_last_price(sym: str):
        try:
            exec_ = _get_trade_executor()
            if exec_ and hasattr(exec_, "get_quote"):
                q = exec_.get_quote(sym) or {}
                quote = q.get("quotes", {}).get("quote", q)
                if isinstance(quote, list) and quote:
                    quote = quote[0]
                return float(quote.get("last")) if quote.get("last") is not None else None
        except Exception:
            return None

    def _is_earnings_blackout(sym: str, now_et):
        # now_et may include tz; convert to UTC naive for the loader
        try:
            now_utc = datetime.utcnow()
        except Exception:
            now_utc = now_et
        return _earnings_blackout(sym, now_utc)

    app.state.evaluate_order = lambda ctx: _evaluate_order(
        ctx, policy, _is_emergency_stop, _get_last_price, _is_earnings_blackout
    )

    # Ensure a shared executor exists; default to Paper if missing
    if not hasattr(app.state, "trade_executor") or app.state.trade_executor is None:
        try:
            initialize_shared_executor()
        except Exception as e:
            logger.warning(f"Executor init skipped: {e}")

# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the API server
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    logger.info(f"Starting Trading Bot API server on {host}:{port}")
    uvicorn.run(
        "trading_bot.api.app_new:app", 
        host=host,
        port=port,
        reload=True
    )
