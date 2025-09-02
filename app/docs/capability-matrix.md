# Capability Matrix

Modules, endpoints, freshness targets, and UI wiring status.

- Live Market Data
  - Endpoints: /api/quotes?symbols=..., /api/bars?symbol=...&timeframe=...
  - Freshness: quotes ≤60s; bars as requested
  - UI: LiveMarketData, MarketQuote, MarketChart wired

- Market Context
  - Endpoints: /api/context, /api/context/regime, /api/context/sentiment
  - Freshness: ≤5m
  - UI: Context widgets wired

- Portfolio
  - Endpoints: /api/portfolio, /api/portfolio/:mode/history, /portfolio/:mode/orders (POST)
  - Freshness: ≤60s
  - UI: Summary + lists partial

- Strategies
  - Endpoints: /api/strategies, /strategies/:id/(enable|disable) (POST)
  - Freshness: ≤60s
  - UI: List wired; actions partial

- Trade Decisions
  - Endpoints: /api/decisions, /decisions/latest
  - Freshness: ≤60s
  - UI: List basic

- Safety
  - Endpoints: /api/safety/status, /safety/* (POST/PUT)
  - Freshness: immediate
  - UI: Controls wired

- Ingestion/Health
  - Endpoints: /metrics, /data/status
  - Freshness: ≤60s
  - UI: Status widgets wired

- Logs & Alerts
  - Endpoints: /logs, /alerts
  - Freshness: live/poll
  - UI: Basic

- EvoTester
  - Endpoints: /evotester/*
  - Freshness: live/poll
  - UI: Dashboard wired

Guardrails
- Every endpoint returns asOf ISO time; UI shows it and computes stale badges.
- UI never renders zeros for failures; shows "No data"/"Disconnected".
- Dev: MSW disabled by default (VITE_USE_MSW=false); enable explicitly to simulate.

Next actions
- Add per-card "as of" subtitle and source tooltip.
- Wire portfolio positions and orders tables.
- Add strategy start/stop buttons and routing.
