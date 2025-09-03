
import json
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import threading
import time
import os
import logging
from datetime import datetime, timezone, timedelta
import random
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot Engine Class
class BotEngine:
    def __init__(self):
        self.mode = "OFF"  # OFF | PAPER | LIVE
        self.running = False
        self.last_heartbeat = None
        self.symbols = []
        self.max_positions = int(os.getenv("MAX_POSITIONS", "5"))
        self.max_pos_pct = float(os.getenv("MAX_POSITION_PCT", "0.05"))
        self.cooldown_sec = int(os.getenv("ORDER_COOLDOWN_SEC", "5"))
        self._interval = None
        self._last_order_ts = 0
        self._error_streak = 0
        self._dedupe = set()
        self.positions = {}  # Mock positions for demo
        self.logger = logger

    def state(self):
        return {
            "mode": self.mode,
            "running": self.running,
            "last_heartbeat": self.last_heartbeat,
            "symbols": self.symbols,
            "max_positions": self.max_positions,
        }

    def start(self, mode="PAPER", symbols=[], max_positions=None):
        if self.running:
            return
        self.mode = mode.upper()
        self.symbols = [s.upper() for s in symbols]
        if max_positions:
            self.max_positions = max_positions
        self.running = True
        self._error_streak = 0
        self.logger.info(f"[engine] start {self.mode} with symbols {self.symbols}")
        self._interval = threading.Thread(target=self._run_loop, daemon=True)
        self._interval.start()

    def pause(self):
        self.mode = "OFF"
        self.logger.info("[engine] paused")

    def stop(self):
        if self._interval:
            self.running = False
            self._interval.join(timeout=2)
            self._interval = None
        self.mode = "OFF"
        self.logger.info("[engine] stopped")

    def _cooldown_ok(self):
        return (time.time() - self._last_order_ts) >= self.cooldown_sec

    def _hash_sig(self, sig):
        return f"{sig['strategy']}:{sig['symbol']}:{sig['action']}:{int(sig['size'])}"

    def _dedupe_signals(self, sigs):
        out = []
        for s in sigs:
            h = self._hash_sig(s)
            if h not in self._dedupe:
                self._dedupe.add(h)
                out.append(s)
        if len(self._dedupe) > 5000:
            self._dedupe = set(list(self._dedupe)[-1000:])
        return out

    def _is_us_equity_open(self, now=None):
        if now is None:
            now = datetime.now(timezone.utc)
        # Naive ET: Mon-Fri 9:30-16:00 ET
        day = now.weekday()  # 0=Mon, 6=Sun
        if day > 4:  # Weekend
            return False
        et_offset = timedelta(hours=-4)  # ET is UTC-4
        et_now = now + et_offset
        open_time = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= et_now <= close_time

    def _tick(self):
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()

        if not self.running or self.mode == "OFF":
            return

        if not self._is_us_equity_open():
            return

        try:
            # Mock account balance
            acct = {
                "equity": 100000,
                "cash": 25000,
                "day_pl_dollar": 0,
                "day_pl_pct": 0
            }

            # Generate demo signals
            sigs = []
            for sym in self.symbols:
                if sym not in self.positions:
                    sigs.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "strategy": "Demo",
                        "symbol": sym,
                        "action": "buy",
                        "size": 1,
                        "reason": "enter"
                    })
                elif self.positions[sym].get("pl_pct", 0) > 1:
                    qty = min(1, abs(int(self.positions[sym].get("qty", 0))))
                    if qty > 0:
                        sigs.append({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "strategy": "Demo",
                            "symbol": sym,
                            "action": "sell",
                            "size": qty,
                            "reason": "take_profit"
                        })

            # Dedupe and risk check
            deduped = self._dedupe_signals(sigs)
            for s in deduped:
                if not self._cooldown_ok():
                    continue
                # Max positions check
                if s["action"] == "buy" and len(self.positions) >= self.max_positions:
                    continue
                # Position size cap
                last_price = 100 + random.uniform(-10, 10)  # Mock price
                cap = self.max_pos_pct * acct["equity"]
                order_value = last_price * s["size"]
                if s["action"] == "buy" and cap and order_value > cap:
                    continue

                # Mock place order
                if self.mode in ["PAPER", "LIVE"]:
                    self._place_order(s)
                    self._last_order_ts = time.time()

            self._error_streak = 0

        except Exception as e:
            self._error_streak += 1
            self.logger.error(f"[engine] tick error: {e}")
            if self._error_streak >= 5:
                self.logger.error("[engine] breaker tripped (5+ errors), pausing")
                self.pause()

    def _place_order(self, signal):
        """Place order via Tradier or mock"""
        self.logger.info(f"[engine] {self.mode} order: {signal['action']} {signal['size']} {signal['symbol']}")

        sym = signal["symbol"]
        action = signal["action"]
        qty = signal["size"]

        if self.mode == "LIVE":
            # Real Tradier integration would go here
            self.logger.info(f"[engine] LIVE order would be placed: {action} {qty} {sym}")
            # For now, simulate as if order was placed
            pass
        elif self.mode == "PAPER":
            # Mock paper trading
            self.logger.info(f"[engine] PAPER order: {action} {qty} {sym}")

        # Update positions (mock for both PAPER and LIVE for now)
        if action == "buy":
            self.positions[sym] = {
                "symbol": sym,
                "qty": qty,
                "avg_price": 100 + random.uniform(-5, 5),
                "last": 100 + random.uniform(-5, 5),
                "pl_dollar": random.uniform(-50, 50),
                "pl_pct": random.uniform(-2, 2)
            }
        elif action == "sell" and sym in self.positions:
            del self.positions[sym]

    def _run_loop(self):
        while self.running:
            self._tick()
            time.sleep(1)  # Check every second

# Global bot engine instance
BOT_ENGINE = BotEngine()

class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {'message': 'Simple API Server', 'status': 'running'}
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/portfolio':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                'totalValue': 100000,
                'dailyChange': 1500,
                'dailyChangePercent': 1.5,
                'cash': 25000,
                'invested': 75000
            }
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/strategies':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = [
                {
                    'id': 'strat-1',
                    'name': 'Momentum',
                    'active': True,
                    'exposure_pct': 0.12,
                    'p_l_30d': 1250.50
                }
            ]
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/bot':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = BOT_ENGINE.state()
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/positions':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = list(BOT_ENGINE.positions.values())
            self.wfile.write(json.dumps(response).encode())

        # V1 API endpoints for dashboard compatibility
        elif path == '/api/v1/account/balance':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            response = {
                'equity': 100000,
                'cash': 25000,
                'day_pl_dollar': 1500,
                'day_pl_pct': 1.5
            }
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/positions':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = list(BOT_ENGINE.positions.values())
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/orders/open':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = []  # Mock empty orders for now
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/orders/recent':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = []  # Mock empty recent orders
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/strategies':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = [
                {
                    'id': 's1',
                    'name': 'Momentum',
                    'active': True,
                    'exposure_pct': 0.12,
                    'p_l_30d': 1250.50
                },
                {
                    'id': 's2',
                    'name': 'MeanReversion',
                    'active': False,
                    'exposure_pct': 0.00,
                    'p_l_30d': -250.25
                }
            ]
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/signals/live':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = []  # Mock empty signals
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/risk/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                'portfolio_heat': 25.0,
                'dd_pct': 0.5,
                'concentration_flag': False,
                'blocks': []
            }
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/v1/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            response = {
                'broker': 'UP',  # Set to UP to show connected
                'data': 'UP',
                'last_heartbeat': datetime.now(timezone.utc).isoformat()
            }
            self.wfile.write(json.dumps(response).encode())

        elif path.startswith('/jobs/'):
            # Handle job status requests like /jobs/job_12345
            job_id = path.split('/')[-1]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                'job_id': job_id,
                'status': 'DONE',  # Mock as completed
                'progress': 100,
                'result_ref': f'report:{job_id}'
            }
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Minimal auth to satisfy React login flow
        if path in ['/auth/login', '/auth/token']:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            # Accept any creds and return a dummy token for local dev
            self.wfile.write(json.dumps({
                'access_token': 'dev-token',  # React expects 'access_token'
                'token': 'dev-token',
                'token_type': 'bearer',
                'expires_in': 60*60*8
            }).encode())
            return

        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'

        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except:
            body = {}

        if path == '/api/bot/start':
            # API key check
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            mode = body.get('mode', 'PAPER')
            symbols = body.get('symbols', [])
            max_positions = body.get('max_positions')

            BOT_ENGINE.start(mode, symbols, max_positions)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())

        elif path == '/api/bot/pause':
            # API key check
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            BOT_ENGINE.pause()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())

        elif path == '/api/bot/stop':
            # API key check
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            BOT_ENGINE.stop()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())

        # V1 POST endpoints for dashboard compatibility
        elif path == '/api/v1/orders':
            # API key check for order placement
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            # Mock order placement
            order_id = f"order_{int(time.time())}"
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'order_id': order_id}).encode())

        elif path.startswith('/api/v1/strategies/') and path.endswith('/activate'):
            # API key check for strategy activation
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())

        elif path.startswith('/api/v1/strategies/') and path.endswith('/deactivate'):
            # API key check for strategy deactivation
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())

        elif path == '/jobs/backtests':
            # Mock job creation for backtests
            job_id = f"job_{int(time.time())}"
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
            self.end_headers()
            self.wfile.write(json.dumps({'job_id': job_id}).encode())

        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_DELETE(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Handle order cancellation
        if path.startswith('/api/v1/orders/'):
            # API key check for order cancellation
            api_key = self.headers.get('X-API-Key', '')
            if api_key != os.getenv('API_KEY_PRIMARY', 'localdev-123456'):
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'invalid_api_key'}).encode())
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode())
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
        self.end_headers()

    def log_message(self, format, *args):
        return  # Suppress log messages

if __name__ == '__main__':
    with socketserver.TCPServer(('', 3000), SimpleAPIHandler) as httpd:
        print('Simple API server running on port 3000')
        httpd.serve_forever()

