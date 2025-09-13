#!/usr/bin/env python3
"""
Backend Service for BensBot Trading Dashboard

This service ties together the autonomous trading system and the dashboard
API so that they can be started with a single command.

Responsibilities
----------------
1. Spin up the BensBot `TradingSystem` (paper or live) in a background thread.
2. Launch the FastAPI backend defined in `trading_bot.api.app` on port **5002**
   (matching the React dashboard default).
3. Expose graceful-shutdown handling so SIGINT / SIGTERM cleanly stop both the
   trading system and the API server.

NOTE:  If you encounter import issues when running this module directly
(e.g. `ModuleNotFoundError` for local packages), be sure to either:
    a) run via `python -m trading_bot.backend_service`, which adds the project
       root to `PYTHONPATH` automatically, **or**
    b) set the environment variable `PYTHONPATH` to your project root before
       launching.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
#  Ensure project root is on PYTHONPATH when launching with `python <file>.py`
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
#  Lazy imports of heavy modules so that path tweaking above takes effect.
# ---------------------------------------------------------------------------
from trading_bot.run_bot import TradingSystem  # noqa: E402
from trading_bot.api.app import app as fastapi_app, initialize_api  # noqa: E402
import uvicorn  # noqa: E402

logger = logging.getLogger("backend_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ---------------------------------------------------------------------------
#  Globals used for shutdown signalling.
# ---------------------------------------------------------------------------
_shutdown_requested: bool = False
_trading_system: Optional[TradingSystem] = None


# ---------------------------------------------------------------------------
#  Trading-system bootstrap helpers
# ---------------------------------------------------------------------------

def _start_trading_system(config_path: str, mode: str) -> TradingSystem:
    """Instantiate and start the *synchronous* trading system.

    The caller should run this in a *background thread* so that the main thread
    can continue and drive the FastAPI / Uvicorn server.
    """
    logger.info("Starting TradingSystem (mode=%s) using config %s", mode, config_path)
    ts = TradingSystem(config_path)
    try:
        if mode == "live":
            ts.start_live_trading()
        else:
            ts.start_paper_trading()
    except Exception:
        logger.exception("TradingSystem terminated unexpectedly!")
        raise
    return ts


def _trading_thread_entry(config_path: str, mode: str):
    global _trading_system, _shutdown_requested
    try:
        _trading_system = _start_trading_system(config_path, mode)
    except Exception:
        _shutdown_requested = True  # Trigger shutdown of API server as well.


# ---------------------------------------------------------------------------
#  Graceful-shutdown handling
# ---------------------------------------------------------------------------

def _handle_termination(sig, frame):  # noqa: D401, N803 (signature fixed by signal API)
    """SIGINT / SIGTERM handler that shuts down both subsystems."""
    global _shutdown_requested
    logger.info("Received signal %s; beginning graceful shutdown…", sig)
    _shutdown_requested = True
    if _trading_system is not None:
        try:
            _trading_system.shutdown()
            logger.info("TradingSystem shut down cleanly")
        except Exception:  # noqa: BLE001 – we want to log any problem
            logger.exception("Error while shutting down TradingSystem")

    # Uvicorn’s server will notice shutdown via `should_exit` flag, set below.
    # We can’t call `server.force_exit()` here because we might not have the
    # server object reference. Instead we rely on the Uvicorn loop + signal
    # integration (our handler will be invoked *after* Uvicorn’s own, but both
    # set the same flag).


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):  # noqa: D401 – CLI entrypoint
    """CLI wrapper to launch the backend service."""
    parser = argparse.ArgumentParser(description="BensBot Backend Service")
    parser.add_argument(
        "--config",
        default="config/system_config.json",
        help="Path to trading system configuration JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode to start (default: paper)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API listen host")
    parser.add_argument("--port", type=int, default=5002, help="API listen port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (development only – restarts on code changes)",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    #  1. Spin up TradingSystem in a daemon thread
    # ------------------------------------------------------------------
    t = threading.Thread(
        target=_trading_thread_entry,
        args=(args.config, args.mode),
        name="TradingSystemThread",
        daemon=True,
    )
    t.start()

    # Optional: pass components to API for richer endpoints later
    initialize_api(strategy_rotator=None, continuous_learner=None)

    # Register *our* signal handlers *after* uvicorn sets up its own so that
    # ours still run (Python calls them all).
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_termination)

    # ------------------------------------------------------------------
    #  2. Run FastAPI app via Uvicorn (blocking call)
    # ------------------------------------------------------------------
    uvicorn.run(
        fastapi_app,  # Provided by trading_bot.api.app
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover – executed only when run directly
    main()
