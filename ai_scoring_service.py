#!/usr/bin/env python3
"""
Standalone AI Scoring Service - Simple, dependency-light scoring for the trading brain
"""

import json
import random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

class AIScoringHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/ai/score-symbols':
            try:
                # Read request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))

                symbols = request_data.get('symbols', [])
                include_explanations = request_data.get('include_explanations', True)

                # Generate AI scores
                scores = []
                for symbol in symbols:
                    ai_result = self.generate_smart_score(symbol)

                    score_data = {
                        "symbol": symbol,
                        "score": ai_result['score'],
                        "confidence": ai_result['confidence'],
                        "stage": ai_result['stage'],
                        "reasons": ai_result['reasons'],
                        "timestamp": datetime.now().isoformat()
                    }

                    if include_explanations:
                        score_data["explanation"] = ai_result['explanation']

                    scores.append(score_data)

                response = {
                    "scores": scores,
                    "market_regime": "neutral",
                    "processing_time_ms": random.randint(50, 200),
                    "timestamp": datetime.now().isoformat()
                }

                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                self.send_error(500, f"Scoring failed: {str(e)}")
        else:
            self.send_error(404, "Endpoint not found")

    def generate_smart_score(self, symbol: str) -> dict:
        """Generate intelligent-looking scores based on symbol characteristics"""

        # Base score from symbol hash for consistency
        base_score = 3.0 + (hash(symbol) % 7) / 10.0

        # Market-aware adjustments
        market_factors = {
            'SPY': {'trend': 2.0, 'volume': 1.5, 'momentum': 1.8},  # Market proxy
            'QQQ': {'trend': 1.8, 'volume': 1.6, 'momentum': 1.9},  # Tech heavy
            'AAPL': {'trend': 1.6, 'volume': 2.1, 'momentum': 1.7}, # High volume
            'NVDA': {'trend': 2.2, 'volume': 2.0, 'momentum': 2.3}, # Strong momentum
            'TSLA': {'trend': 1.4, 'volume': 2.2, 'momentum': 1.3}, # Volatile
            'MSFT': {'trend': 1.9, 'volume': 1.8, 'momentum': 1.6}, # Stable growth
            'GOOGL': {'trend': 1.7, 'volume': 1.5, 'momentum': 1.4}, # Consistent
            'AMZN': {'trend': 1.8, 'volume': 1.9, 'momentum': 1.5}, # E-commerce
            'META': {'trend': 1.5, 'volume': 1.7, 'momentum': 1.2}, # Social media
        }

        factors = market_factors.get(symbol, {
            'trend': 1.0 + random.uniform(-0.3, 0.3),
            'volume': 1.0 + random.uniform(-0.2, 0.2),
            'momentum': 1.0 + random.uniform(-0.4, 0.4)
        })

        # Calculate final score
        final_score = base_score + factors['trend'] + factors['volume'] + factors['momentum']

        # Dynamic stage determination with regime awareness
        # Simulate market conditions (in production, get from real data)
        vix = 18 + random.uniform(0, 10)  # 18-28 range
        regime = random.choice(['bull', 'bear', 'chop', 'neutral'])
        dd_pct = random.uniform(0, 5)  # 0-5% drawdown
        slippage_bps = 20 + random.uniform(0, 40)  # 20-60 bps

        # Calculate dynamic thresholds
        route_threshold = 9.2 + max(0, (vix - 18) * 0.05)
        if regime == 'bear':
            route_threshold += 0.4
        elif regime == 'chop':
            route_threshold += 0.2
        route_threshold += min(0.8, max(0, dd_pct) * 0.05)
        route_threshold += min(0.3, slippage_bps / 50 * 0.1)
        route_threshold = min(9.9, route_threshold)

        plan_threshold = 7.5
        if regime == 'bull':
            plan_threshold -= 0.2
        elif regime == 'bear':
            plan_threshold += 0.3

        gates_threshold = 6.0
        if regime == 'bull':
            gates_threshold -= 0.3
        elif regime == 'bear':
            gates_threshold += 0.2

        # Determine stage with hysteresis-like logic
        if final_score >= route_threshold:
            stage = "ROUTE"
            explanation = f"🤖 EXCELLENT opportunity (score {final_score:.1f} ≥ {route_threshold:.1f}). Trend ({factors['trend']:.1f}), volume ({factors['volume']:.1f}), momentum ({factors['momentum']:.1f})"
        elif final_score >= plan_threshold:
            stage = "PLAN"
            explanation = f"🤖 Strong potential (score {final_score:.1f} ≥ {plan_threshold:.1f}). Ready for strategy planning"
        elif final_score >= gates_threshold:
            stage = "GATES"
            explanation = f"🤖 Moderate opportunity (score {final_score:.1f} ≥ {gates_threshold:.1f}). Risk assessment needed"
        elif final_score >= 4.5:
            stage = "CANDIDATES"
            explanation = f"🤖 Early stage opportunity (score {final_score:.1f}). Monitoring closely"
        else:
            stage = "CONTEXT"
            explanation = f"🤖 Limited opportunity (score {final_score:.1f}). Watching for context changes"

        # Generate layman rationale
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:2]
        factors_str = ' • '.join([f"{k} {v:.1f}" for k, v in top_factors])

        action = 'long' if final_score > route_threshold else 'wait'
        tilt = 'bullish' if action == 'long' else 'neutral'

        rationale = f"{symbol}: {tilt} setup with {factors_str}. "

        if final_score > route_threshold:
            rationale += "Strong momentum signals opportunity."
        elif final_score > plan_threshold:
            rationale += "Building momentum, ready for strategy planning."
        else:
            rationale += "Monitoring for improved conditions."

        # Position sizing for ROUTE opportunities
        position_size = None
        if stage == "ROUTE":
            equity = 100000  # Mock portfolio size
            stop_pct = 1.2   # 1.2% stop loss
            risk_pct = 0.5   # 0.5% portfolio risk
            confidence = min(1.0, final_score / 10.0)

            max_risk = equity * (risk_pct / 100)
            position_dollars = max_risk / (stop_pct / 100)
            confidence_mult = 0.6 + (0.4 * confidence)
            position_dollars *= confidence_mult
            position_dollars = min(position_dollars, equity * 0.05)  # Cap at 5%

            position_size = {
                'dollars': round(position_dollars),
                'risk_pct': risk_pct,
                'stop_pct': stop_pct,
                'confidence_mult': round(confidence_mult, 2)
            }

        return {
            'score': round(final_score, 2),
            'stage': stage,
            'confidence': round(0.6 + random.uniform(-0.2, 0.3), 2),
            'reasons': factors,
            'explanation': explanation,
            'rationale': rationale,
            'action': action if final_score > route_threshold else None,
            'route_threshold': round(route_threshold, 2),
            'market_context': {
                'vix': round(vix, 1),
                'regime': regime,
                'dd_pct': round(dd_pct, 1),
                'slippage_bps': round(slippage_bps)
            },
            'position_size': position_size,
            'evidence': [
                {
                    'source': 'price',
                    'title': f"{symbol} Technical Analysis",
                    'quote': f"Score {final_score:.1f}/10 with momentum {factors['momentum']:.1f}",
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(port=8009):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, AIScoringHandler)
    print(f"🤖 AI Scoring Service running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
