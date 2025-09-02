#!/usr/bin/env bash
set -euo pipefail

# ===== EDIT ONLY IF YOUR PATH CHANGES =====
REPO_DIR="/Users/bendickinson/Desktop/Trading:BenBot"
LOG_DIR="$HOME/BenBotLogs"; mkdir -p "$LOG_DIR"
APP_NAME="BensBot Pro.app"
APP_DIR="$REPO_DIR/new-trading-dashboard/dist/mac"
APP_PATH="$APP_DIR/$APP_NAME"

echo "== $(date) :: build-and-launch starting ==" | tee "$LOG_DIR/launcher.out"

# ---- ENV (paper defaults) ----
if [[ -f "$REPO_DIR/.env" ]]; then set -a; source "$REPO_DIR/.env"; set +a; fi
if [[ -f "$REPO_DIR/.env.local" ]]; then set -a; source "$REPO_DIR/.env.local"; set +a; fi
export TRADING_MODE="${TRADING_MODE:-paper}"
export BROKER="${BROKER:-tradier}"
export TRADIER_BASE_URL="${TRADIER_BASE_URL:-https://sandbox.tradier.com/v1}"
export RECONCILE_INTERVAL_SEC="${RECONCILE_INTERVAL_SEC:-30}"
export PROMOTE_KEY="${PROMOTE_KEY:-change_me}"
export JOURNAL_DIR="${JOURNAL_DIR:-$REPO_DIR/journal}"
export VITE_API_URL="${VITE_API_URL:-http://localhost:8000}"
export VITE_WS_URL="${VITE_WS_URL:-ws://localhost:8000/ws}"

# ---- Backend ----
cd "$REPO_DIR"
python3 -m venv .venv >/dev/null 2>&1 || true
source .venv/bin/activate
pip install -r requirements.txt >"$LOG_DIR/pip.out" 2>"$LOG_DIR/pip.err" || true

echo ">>> Starting backend on :8000"
( uvicorn trading_bot.api.app_new:app --port 8000 --reload \
  >"$LOG_DIR/backend.out" 2>"$LOG_DIR/backend.err" & echo $! > "$LOG_DIR/backend.pid" )

# ---- Desktop app: build if missing, else launch ----
cd "$REPO_DIR/new-trading-dashboard"
if [[ ! -d "dist" || ! -d "$APP_DIR" ]]; then
  echo ">>> Building Vite UI for Electron..." | tee -a "$LOG_DIR/launcher.out"
  rm -rf dist
  BUILD_TARGET_ELECTRON=1 \
  VITE_API_URL="$VITE_API_URL" \
  VITE_WS_URL="$VITE_WS_URL" \
  npm install >"$LOG_DIR/npm_install.out" 2>"$LOG_DIR/npm_install.err"
  npm run build >"$LOG_DIR/vite_build.out" 2>"$LOG_DIR/vite_build.err"
  echo ">>> Packaging Electron app..." | tee -a "$LOG_DIR/launcher.out"
  npm run build:electron >"$LOG_DIR/electron_build.out" 2>"$LOG_DIR/electron_build.err" || true
fi

if [[ -d "$APP_DIR" && -d "$APP_PATH/Contents" ]]; then
  echo ">>> Launching packaged app" | tee -a "$LOG_DIR/launcher.out"
  open "$APP_PATH"
else
  echo ">>> Packaged app not found; running dev UI on :3003" | tee -a "$LOG_DIR/launcher.out"
  printf "VITE_API_URL=%s\nVITE_WS_URL=%s\n" "$VITE_API_URL" "$VITE_WS_URL" > .env.local
  ( npm run dev >"$LOG_DIR/vite.out" 2>"$LOG_DIR/vite.err" & echo $! > "$LOG_DIR/vite.pid" )
  sleep 2; open http://localhost:3003 || true
fi

sleep 2
open http://localhost:8000/docs || true
echo "Logs live in: $LOG_DIR" | tee -a "$LOG_DIR/launcher.out"


