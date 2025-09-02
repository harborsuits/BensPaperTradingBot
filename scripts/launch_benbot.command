#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

osascript <<'APPLESCRIPT'
tell application "Terminal"
    do script "cd '$ROOT_DIR'; python3 -m uvicorn trading_bot.api.app_new:app --reload --port 8000"
    delay 1
    do script "cd '$ROOT_DIR/new-trading-dashboard'; if [ ! -f .env.local ]; then echo 'VITE_API_URL=http://localhost:8000' > .env.local; fi; npm i; npm run dev"
end tell
APPLESCRIPT

open http://localhost:8000/docs || true
open http://localhost:5173 || true

echo "Launched backend (uvicorn) and frontend (Vite) in Terminal windows."
echo "Docs:   http://localhost:8000/docs"
echo "UI:     http://localhost:5173"

