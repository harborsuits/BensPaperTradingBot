#!/usr/bin/env bash
set -euo pipefail

# Detect repo dir robustly (even with special chars in name)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/BenBotLogs"
mkdir -p "$LOG_DIR"

# Load optional env overrides
for f in "$REPO_DIR/.env" "$REPO_DIR/.env.local"; do
  if [[ -f "$f" ]]; then set -a; source "$f"; set +a; fi
done

# Defaults (safe paper mode)
export TRADING_MODE="${TRADING_MODE:-paper}"
export BROKER="${BROKER:-tradier}"
export TRADIER_BASE_URL="${TRADIER_BASE_URL:-https://sandbox.tradier.com/v1}"
export RECONCILE_INTERVAL_SEC="${RECONCILE_INTERVAL_SEC:-30}"
export PROMOTE_KEY="${PROMOTE_KEY:-change_me}"
export JOURNAL_DIR="${JOURNAL_DIR:-$REPO_DIR/journal}"

# Frontend -> backend URLs
export VITE_API_URL="${VITE_API_URL:-http://localhost:8000}"
export VITE_WS_URL="${VITE_WS_URL:-ws://localhost:8000/ws}"

echo ">>> Repo: $REPO_DIR"
echo ">>> Logs: $LOG_DIR"

# ===== Backend =====
cd "$REPO_DIR"
python3 -m venv .venv >/dev/null 2>&1 || true
source .venv/bin/activate || true
python -m pip install --upgrade pip >/dev/null 2>&1 || true
if [[ -f requirements.txt ]]; then pip install -r requirements.txt >/dev/null 2>&1 || true; fi

echo ">>> Starting backend on :8000"
( uvicorn trading_bot.api.app_new:app --port 8000 --reload \
    >"$LOG_DIR/backend.out" 2>"$LOG_DIR/backend.err" & echo $! > "$LOG_DIR/backend.pid" )

# ===== Frontend: packaged app if present, else dev =====
APP_A="$REPO_DIR/new-trading-dashboard/dist/mac/BensBot Pro.app"
APP_B="$REPO_DIR/new-trading-dashboard/dist/mac-arm64/BensBot Pro.app"

if [[ -d "$APP_A" || -d "$APP_B" ]]; then
  APP_PATH=""
  [[ -d "$APP_A" ]] && APP_PATH="$APP_A"
  [[ -z "$APP_PATH" && -d "$APP_B" ]] && APP_PATH="$APP_B"
  echo ">>> Launching packaged desktop app: $APP_PATH"
  open "$APP_PATH"
else
  echo ">>> No packaged app found; running Vite dev UI on :3003"
  pushd "$REPO_DIR/new-trading-dashboard" >/dev/null
  printf "VITE_API_URL=%s\nVITE_WS_URL=%s\n" "$VITE_API_URL" "$VITE_WS_URL" > .env.local
  if command -v npm >/dev/null 2>&1; then
    npm install >/dev/null 2>&1 || true
    ( npm run dev >"$LOG_DIR/vite.out" 2>"$LOG_DIR/vite.err" & echo $! > "$LOG_DIR/vite.pid" )
    sleep 2
    open "http://localhost:3003" || true
  else
    echo "npm not found. Install Node.js to run the dashboard." | tee -a "$LOG_DIR/vite.err"
  fi
  popd >/dev/null
fi

sleep 2
open "http://localhost:8000/docs" >/dev/null 2>&1 || true
echo ">>> Backend PID: $(cat "$LOG_DIR/backend.pid" 2>/dev/null || echo -n '-')"
echo ">>> Done. Check logs in $LOG_DIR if anything looks off."


