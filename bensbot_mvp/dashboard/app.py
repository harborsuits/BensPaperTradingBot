import os, time, requests
import streamlit as st
from datetime import datetime

API = os.getenv("API_URL","http://localhost:8787")
CRED = {"username": os.getenv("ADMIN_USERNAME","admin"), "password": os.getenv("ADMIN_PASSWORD","changeme")}
API_KEY = os.getenv("API_KEY_PRIMARY","localdev-123456")

st.set_page_config(page_title="BensBot", layout="wide")

# ---- Auth once ----
if "token" not in st.session_state:
    try:
        r = requests.post(f"{API}/auth/login", json=CRED, timeout=5)
        r.raise_for_status()
        st.session_state["token"] = r.json()["token"]
    except Exception as e:
        st.error(f"Auth failed: {e}")
        st.stop()

HEADERS = {"Authorization": f"Bearer {st.session_state['token']}"}
HEADERS_MUT = {**HEADERS, "X-API-Key": API_KEY}

def fetch(path, default=None):
    try:
        r = requests.get(f"{API}{path}", headers=HEADERS, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Fetch {path} failed: {e}")
        return default

# ---- Top bar
top1, top2, top3 = st.columns([2,2,2])
acct = fetch("/api/v1/account/balance", {"equity":0,"cash":0,"day_pl_dollar":0,"day_pl_pct":0})
health = fetch("/api/v1/health", {"broker":"DOWN","data":"DOWN","last_heartbeat":str(datetime.utcnow())})

with top1:
    st.metric("Equity", f"${acct['equity']:,.2f}", f"{acct['day_pl_dollar']:,.2f}")
    st.caption(f"Cash: ${acct['cash']:,.2f} · Day P/L: {acct['day_pl_pct']:.2f}%")
with top2:
    status = "🟢 Paper" if os.getenv("BROKER","tradier") else "⚪"
    st.metric("Session", f"{status}", f"Broker: {health['broker']}")
with top3:
    st.metric("Health", f"{health['broker']}/{health['data']}", datetime.fromisoformat(health["last_heartbeat"]).strftime("%H:%M:%S"))

st.divider()

# ---- Left: Positions / Orders
left, right = st.columns([2,2])

with left:
    st.subheader("Positions")
    pos = fetch("/api/v1/positions", [])
    if pos:
        st.dataframe([{**p, "value": p["qty"]*p["last"]} for p in pos], use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

    st.subheader("Open Orders")
    orders = fetch("/api/v1/orders/open", [])
    if orders:
        st.dataframe(orders, use_container_width=True, hide_index=True)
        # Cancel buttons in Open Orders table
        for o in orders:
            if st.button(f"Cancel {o['id']}", key=f"cancel_{o['id']}"):
                try:
                    r = requests.delete(f"{API}/api/v1/orders/{o['id']}", headers=HEADERS_MUT, timeout=6)
                    r.raise_for_status()
                    st.success("Canceled")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Cancel failed: {e}")
    else:
        st.info("No open orders")

    # Quick test order controls (paper)
    st.subheader("Quick Trade (Paper)")
    colA, colB, colC, colD, colE = st.columns([2,1,1,1,1])
    sym = colA.text_input("Symbol", "AAPL")
    qty = colB.number_input("Qty", 1, step=1)
    otype = colC.selectbox("Type", ["market","limit"])
    lpx = colD.number_input("Limit", 0.0, step=0.01, format="%.2f", disabled=(otype!="limit"))
    side = colE.selectbox("Side", ["buy","sell"])
    if st.button("Place Order"):
        try:
            r = requests.post(f"{API}/api/v1/orders", headers=HEADERS_MUT, json={
                "symbol": sym, "side": side, "qty": qty, "type": otype,
                "limit_price": (lpx if otype=="limit" else None)
            }, timeout=6)
            r.raise_for_status()
            st.success(f"Order placed: {r.json()['order_id']}")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Order failed: {e}")

# ---- Right: Strategies / Signals
with right:
    st.subheader("Strategies")
    strats = fetch("/api/v1/strategies", [])
    for s in strats:
        cols = st.columns([4,1,2,2,2])
        with cols[0]:
            st.write(f"**{s['name']}**")
            st.caption(f"Exposure: {s['exposure_pct']*100:.1f}%")
        with cols[1]:
            st.write("🟢" if s["active"] else "⚪")
        with cols[2]:
            if st.button("Activate" if not s["active"] else "Deactivate", key=f"t{s['id']}"):
                path = f"/api/v1/strategies/{s['id']}/activate" if not s["active"] else f"/api/v1/strategies/{s['id']}/deactivate"
                try:
                    r = requests.post(API+path, headers=HEADERS_MUT, timeout=5)
                    r.raise_for_status()
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Toggle failed: {e}")
        with cols[3]:
            st.caption(f"Last signal: {s.get('last_signal_time') or '-'}")
        with cols[4]:
            st.caption(f"30d P/L: {s.get('p_l_30d',0):.2f}")

    st.subheader("Live Signals")
    sigs = fetch("/api/v1/signals/live", [])
    if sigs:
        st.dataframe(sigs[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No signals yet")

st.divider()

# ---- Tabs
tabs = st.tabs(["Risk & Health", "Jobs", "Logs (soon)"])
with tabs[0]:
    risk = fetch("/api/v1/risk/status", {"portfolio_heat":0,"dd_pct":0,"concentration_flag":False,"blocks":[]})
    c1,c2,c3 = st.columns(3)
    c1.metric("Portfolio Heat", f"{risk['portfolio_heat']:.1f}%")
    c2.metric("Drawdown", f"{risk['dd_pct']:.1f}%")
    c3.metric("Concentration", "⚠️" if risk["concentration_flag"] else "OK")
    if risk["blocks"]:
        st.error("\n".join(risk["blocks"]))

with tabs[1]:
    st.write("Backtest (non-blocking demo)")
    if st.button("Start Backtest"):
        try:
            r = requests.post(f"{API}/jobs/backtests", headers=HEADERS, timeout=5)
            r.raise_for_status()
            st.session_state["last_job_id"] = r.json()["job_id"]
        except Exception as e:
            st.error(f"Backtest start failed: {e}")
    job_id = st.session_state.get("last_job_id")
    if job_id:
        r = fetch(f"/jobs/{job_id}", {"status":"QUEUED","progress":0})
        st.progress(r["progress"])
        st.caption(f"Status: {r['status']}")
        if r["status"] in ("QUEUED","RUNNING"):
            time.sleep(1.0)
            st.experimental_rerun()
