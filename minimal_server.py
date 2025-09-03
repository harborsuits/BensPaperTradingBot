import http.server
import socketserver
import json
from datetime import datetime, timezone
import random
import os
import time

# Minimal server that only implements the essential endpoints needed by the React dashboard
class MinimalHandler(http.server.BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, DELETE')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        path = self.path
        
        # Handle auth/token endpoint
        if path == '/auth/token' or path == '/auth/login':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {
                'access_token': 'dev-token',
                'token': 'dev-token',
                'token_type': 'bearer',
                'expires_in': 28800
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle health endpoint
        elif path == '/api/v1/health' or path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {
                'broker': 'UP',
                'data': 'UP',
                'last_heartbeat': datetime.now(timezone.utc).isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle account balance endpoint
        elif path == '/api/v1/account/balance':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {
                'equity': 100000,
                'cash': 25000,
                'day_pl_dollar': 1500,
                'day_pl_pct': 1.5
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle positions endpoint
        elif path == '/api/v1/positions':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = [
                {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'avg_price': 175.25,
                    'last': 180.50,
                    'pl_dollar': 525.00,
                    'pl_pct': 3.0
                },
                {
                    'symbol': 'MSFT',
                    'qty': 50,
                    'avg_price': 320.75,
                    'last': 330.25,
                    'pl_dollar': 475.00,
                    'pl_pct': 2.96
                }
            ]
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle orders endpoints
        elif path == '/api/v1/orders/open' or path == '/api/v1/orders/recent':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = [
                {
                    'id': 'order-1',
                    'symbol': 'NVDA',
                    'side': 'BUY',
                    'qty': 20,
                    'type': 'limit',
                    'limit_price': 110.50,
                    'status': 'open',
                    'ts': datetime.now(timezone.utc).isoformat()
                }
            ]
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle strategies endpoint
        elif path == '/api/v1/strategies':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
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
                    'name': 'Mean Reversion',
                    'active': False,
                    'exposure_pct': 0.0,
                    'p_l_30d': 850.25
                },
                {
                    'id': 's3',
                    'name': 'Trend Following',
                    'active': True,
                    'exposure_pct': 0.18,
                    'p_l_30d': 1750.75
                }
            ]
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle signals endpoint
        elif path == '/api/v1/signals/live':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = [
                {
                    'ts': datetime.now(timezone.utc).isoformat(),
                    'strategy': 'Momentum',
                    'symbol': 'TSLA',
                    'action': 'buy',
                    'size': 10,
                    'reason': 'Strong uptrend detected'
                }
            ]
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle risk status endpoint
        elif path == '/api/v1/risk/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {
                'portfolio_heat': 25.0,
                'dd_pct': 0.5,
                'concentration_flag': False,
                'blocks': []
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle job status endpoint
        elif path.startswith('/jobs/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            job_id = path.split('/')[-1]
            response = {
                'job_id': job_id,
                'status': 'DONE',
                'progress': 100,
                'result_ref': f'report:{job_id}'
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle root endpoint
        elif path == '/' or path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {'message': 'Minimal API Server', 'status': 'running'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle all other endpoints
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not Found'}).encode())
            return
    
    def do_POST(self):
        path = self.path
        
        # Handle auth endpoints
        if path == '/auth/login' or path == '/auth/token':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {
                'access_token': 'dev-token',
                'token': 'dev-token',
                'token_type': 'bearer',
                'expires_in': 28800
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle strategy activation/deactivation
        elif path.startswith('/api/v1/strategies/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {'ok': True}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle order placement
        elif path == '/api/v1/orders':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {'order_id': f'order-{random.randint(1000, 9999)}'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle backtest job
        elif path == '/jobs/backtests':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {'job_id': f'job-{random.randint(1000, 9999)}'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle all other POST requests
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not Found'}).encode())
            return
    
    def do_DELETE(self):
        path = self.path
        
        # Handle order cancellation
        if path.startswith('/api/v1/orders/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            response = {'ok': True}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Handle all other DELETE requests
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not Found'}).encode())
            return
    
    def log_message(self, format, *args):
        # Suppress log messages to keep console clean
        return

if __name__ == '__main__':
    PORT = 3000
    print(f"Starting minimal API server on port {PORT}")
    
    try:
        with socketserver.TCPServer(('', PORT), MinimalHandler) as httpd:
            print(f"Server running at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error: {e}")
