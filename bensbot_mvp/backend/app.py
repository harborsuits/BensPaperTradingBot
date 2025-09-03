import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from .models import *
from .deps import require_api_key, require_jwt
from .auth import router as auth_router, verify_jwt
from .jobs import router as jobs_router
from . import broker as br
from .risk import compute_portfolio_heat, concentration_flag
from .persistence import insert_event

# Simple .env loader (no external dependencies)
def load_env_file():
    env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if key and value:
                        os.environ[key] = value

load_env_file()

app = FastAPI(title="BensBot API", version="1.0.0")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS","http://localhost:8788").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router)
app.include_router(jobs_router)

def _auth(token:str=Depends(require_jwt)):
    verify_jwt(token); return True

# ---- Account ----
@app.get("/api/v1/account/balance", response_model=AccountBalance)
def get_balance(_:bool=Depends(_auth)):
    j = br.account_balance()
    return AccountBalance(**j)

# ---- Positions / Orders ----
@app.get("/api/v1/positions", response_model=list[Position])
def get_positions(_:bool=Depends(_auth)):
    pos = br.open_positions()
    return [Position(**p) for p in pos]

@app.get("/api/v1/orders/open", response_model=list[Order])
def get_orders(_:bool=Depends(_auth)):
    arr = br.open_orders()
    return [Order(**o) for o in arr]

@app.get("/api/v1/orders/recent", response_model=list[Order])
def recent_orders(_:bool=Depends(_auth)):
    # For MVP, reuse open orders as placeholder (Tradier doesn't expose recent easily)
    arr = br.open_orders()
    return [Order(**o) for o in arr]

@app.delete("/api/v1/orders/{order_id}")
def cancel_order(order_id:str, _:bool=Depends(_auth), __=Depends(require_api_key)):
    ok = br.cancel_order(order_id)
    insert_event("INFO","orders","cancel", "", {"order_id": order_id, "ok": ok})
    return {"ok": bool(ok)}

# ---- Strategies (MVP static sample with toggles in memory) ----
STRATS: dict[str,StrategyCard] = {
    "s1": StrategyCard(id="s1", name="Momentum", active=True, exposure_pct=0.12),
    "s2": StrategyCard(id="s2", name="MeanReversion", active=False, exposure_pct=0.00),
    "s3": StrategyCard(id="s3", name="TrendFollow", active=True, exposure_pct=0.18),
}

@app.get("/api/v1/strategies", response_model=list[StrategyCard])
def list_strategies(_:bool=Depends(_auth)):
    return list(STRATS.values())

@app.post("/api/v1/strategies/{sid}/activate")
def activate_strategy(sid:str, _:bool=Depends(_auth), __=Depends(require_api_key)):
    if sid in STRATS:
        STRATS[sid].active = True
        insert_event("INFO","strategy","activated","",{"id":sid})
        return {"ok":True}
    raise HTTPException(404,"strategy_not_found")

@app.post("/api/v1/strategies/{sid}/deactivate")
def deactivate_strategy(sid:str, _:bool=Depends(_auth), __=Depends(require_api_key)):
    if sid in STRATS:
        STRATS[sid].active = False
        insert_event("INFO","strategy","deactivated","",{"id":sid})
        return {"ok":True}
    raise HTTPException(404,"strategy_not_found")

# ---- Signals feed (MVP sample) ----
SIGNALS: list[LiveSignal] = []
@app.get("/api/v1/signals/live", response_model=list[LiveSignal])
def live_signals(_:bool=Depends(_auth)):
    return SIGNALS[-100:]

# ---- Risk / Health ----
@app.get("/api/v1/risk/status", response_model=RiskStatus)
def risk_status(_:bool=Depends(_auth)):
    acct = br.account_balance()
    pos = br.open_positions()
    heat = compute_portfolio_heat(pos, acct["equity"])
    conc = concentration_flag(pos)
    # dd not tracked yet → 0.0
    return RiskStatus(portfolio_heat=round(heat*100,2), dd_pct=0.0, concentration_flag=conc, blocks=[])

@app.get("/api/v1/health", response_model=Health)
def health(_:bool=Depends(_auth)):
    # Broker UP if account call works
    try:
        br.account_balance()
        broker="UP"
    except Exception:
        broker="DOWN"
    return Health(broker=broker, data="UP", last_heartbeat=datetime.utcnow())

# ---- Place order (MVP no-op to keep UI flow; guard with API key) ----
@app.post("/api/v1/orders", response_model=PlaceOrderResponse)
def place_order(req: PlaceOrderRequest, _:bool=Depends(_auth), __=Depends(require_api_key)):
    oid = br.place_order(req.symbol, req.side, req.qty, req.type, req.limit_price)
    insert_event("INFO","orders","placed","", {"order_id": oid, **req.model_dump()})
    return PlaceOrderResponse(order_id=oid)
