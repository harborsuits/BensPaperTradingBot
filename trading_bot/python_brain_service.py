#!/usr/bin/env python3
"""
Python Brain Service

HTTP API service that exposes the Python DecisionEngine for Node.js integration.
Provides REST endpoints for decision-making, health checks, and status monitoring.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

# Import existing components
try:
    from trading_bot.core.decision_scoring import DecisionScorer
    from trading_bot.engine.decision_engine import DecisionEngine
    from trading_bot.policy.service import PolicyService
    from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
except ImportError as e:
    print(f"Warning: Could not import trading_bot modules: {e}")
    print("Running in standalone mode with mock implementations")

    class DecisionScorer:
        def __init__(self): pass
        def record_decision(self, *args): return "mock_decision_id"

    class DecisionEngine:
        def __init__(self, policy_service=None): pass
        def process_opportunities(self, *args): return []

    class PolicyService:
        def __init__(self): pass

    class StrategyIntelligenceRecorder:
        def __init__(self): pass

logger = logging.getLogger(__name__)

class PythonBrainHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Python brain service."""

    def __init__(self, *args, decision_engine=None, decision_scorer=None, **kwargs):
        self.decision_engine = decision_engine
        self.decision_scorer = decision_scorer
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/status':
            self._handle_status()
        elif self.path.startswith('/decisions/'):
            self._handle_get_decision()
        else:
            self._send_error(404, "Endpoint not found")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/decide':
            self._handle_decide()
        elif self.path == '/api/learn':
            self._handle_learn()
        elif self.path == '/api/record-decision':
            self._handle_record_decision()
        else:
            self._send_error(404, "Endpoint not found")

    def _handle_health(self):
        """Health check endpoint."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "decision_engine": "active" if self.decision_engine else "inactive",
                "decision_scorer": "active" if self.decision_scorer else "inactive",
                "policy_service": "active" if hasattr(self.decision_engine, 'policy_service') else "inactive"
            }
        }

        self._send_json_response(200, health_status)

    def _handle_status(self):
        """Status endpoint with detailed information."""
        try:
            status_info = {
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
                "decisions_processed": getattr(self, '_decisions_processed', 0),
                "decisions_recorded": getattr(self, '_decisions_recorded', 0),
                "last_decision_time": getattr(self, '_last_decision_time', None),
                "engine_status": self.decision_engine.get_status() if hasattr(self.decision_engine, 'get_status') else "unknown"
            }
            self._send_json_response(200, status_info)
        except Exception as e:
            self._send_error(500, f"Status error: {str(e)}")

    def _handle_decide(self):
        """Main decision-making endpoint."""
        try:
            # Parse request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            opportunities = request_data.get('opportunities', [])
            current_regime = request_data.get('current_regime', 'neutral')

            if not opportunities:
                self._send_error(400, "No opportunities provided")
                return

            # Process opportunities
            start_time = time.time()
            results = self._process_opportunities(opportunities, current_regime)
            processing_time = time.time() - start_time

            # Track metrics
            self._decisions_processed = getattr(self, '_decisions_processed', 0) + len(opportunities)
            self._last_decision_time = datetime.now().isoformat()

            response = {
                "decisions": results,
                "processing_time_ms": round(processing_time * 1000, 2),
                "timestamp": datetime.now().isoformat(),
                "regime_used": current_regime
            }

            self._send_json_response(200, response)

        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
        except Exception as e:
            logger.error(f"Decision error: {e}")
            self._send_error(500, f"Decision processing failed: {str(e)}")

    def _process_opportunities(self, opportunities: List[Dict], current_regime: str):
        """Process trading opportunities using the decision engine."""
        results = []

        for opp in opportunities:
            try:
                # Enhance opportunity with regime information
                enhanced_opp = {
                    "id": opp.get("id", str(uuid.uuid4())),
                    "instrument": opp.get("instrument", "equity"),
                    "symbol": opp.get("symbol", "UNKNOWN"),
                    "ts": int(time.time() * 1000),
                    "alpha": opp.get("alpha", 0.5),
                    "regime_align": self._calculate_regime_alignment(opp, current_regime),
                    "sentiment_boost": opp.get("sentiment_boost", 0.0),
                    "est_cost_bps": opp.get("est_cost_bps", 20),
                    "risk_penalty": opp.get("risk_penalty", 0.0),
                    "meta": opp.get("meta", {}),
                    "ctx": opp.get("ctx", {
                        "px": opp.get("price", 100.0),
                        "vol": opp.get("volume", 100000),
                        "size_budget_usd": opp.get("size_budget_usd", 1000.0)
                    })
                }

                # Get decision from engine
                if self.decision_engine:
                    try:
                        engine_decisions = self.decision_engine.process_opportunities([enhanced_opp])
                        if engine_decisions:
                            decision = self._format_engine_decision(engine_decisions[0])
                        else:
                            decision = self._create_default_decision(enhanced_opp)
                    except Exception as e:
                        logger.warning(f"Engine decision failed for {enhanced_opp['symbol']}: {e}")
                        decision = self._create_default_decision(enhanced_opp)
                else:
                    decision = self._create_default_decision(enhanced_opp)

                # Record decision for learning
                if self.decision_scorer:
                    try:
                        decision_id = self.decision_scorer.record_decision(
                            signal_id=decision["id"],
                            decision_data={
                                "strategy_id": "python_brain",
                                "symbol": decision["symbol"],
                                "direction": decision["action"],
                                "confidence": decision["confidence"],
                                "expected_profit": decision.get("expected_profit", 0),
                                "expected_risk": decision.get("expected_risk", 0.02),
                                "rationale": decision.get("rationale", {}),
                                "market_context": {
                                    "regime": current_regime
                                }
                            }
                        )
                        self._decisions_recorded = getattr(self, '_decisions_recorded', 0) + 1
                    except Exception as e:
                        logger.warning(f"Decision recording failed: {e}")

                results.append(decision)

            except Exception as e:
                logger.error(f"Error processing opportunity {opp.get('symbol', 'unknown')}: {e}")
                results.append(self._create_error_decision(opp, str(e)))

        return results

    def _calculate_regime_alignment(self, opp: Dict, current_regime: str) -> float:
        """Calculate how well an opportunity aligns with current market regime."""
        instrument = opp.get("instrument", "equity")
        alpha = opp.get("alpha", 0.5)

        # Simple regime alignment logic
        if current_regime == "bull":
            if instrument == "equity":
                return min(1.0, alpha + 0.2)  # Equities align well with bull markets
            elif instrument == "options":
                return max(0.0, alpha - 0.1)  # Options less aligned with bull markets
        elif current_regime == "bear":
            if instrument == "equity":
                return max(0.0, alpha - 0.2)  # Equities don't align well with bear markets
            elif instrument == "options":
                return min(1.0, alpha + 0.3)  # Options align better with bear markets
        elif current_regime == "high_volatility":
            if instrument == "options":
                return min(1.0, alpha + 0.4)  # Options love volatility
            else:
                return max(0.0, alpha - 0.2)  # Other instruments less aligned

        return alpha  # Neutral regime, use alpha as-is

    def _format_engine_decision(self, engine_decision: Dict) -> Dict:
        """Format decision engine output into standardized format."""
        return {
            "id": engine_decision.get("id", str(uuid.uuid4())),
            "symbol": engine_decision.get("symbol", "UNKNOWN"),
            "instrument": engine_decision.get("instrument", "equity"),
            "action": engine_decision.get("action", "no_trade"),
            "confidence": engine_decision.get("score", 0.5),
            "expected_profit": engine_decision.get("expected_profit", 0),
            "expected_risk": engine_decision.get("expected_risk", 0.02),
            "rationale": {
                "reason": engine_decision.get("reason", "Engine decision"),
                "factors": engine_decision.get("factors", [])
            },
            "orders": engine_decision.get("orders", []),
            "timestamp": datetime.now().isoformat()
        }

    def _create_default_decision(self, opp: Dict) -> Dict:
        """Create a default decision when engine is unavailable."""
        confidence = opp.get("alpha", 0.5)

        # Simple rule-based decision
        if confidence > 0.7:
            action = "buy"
        elif confidence < 0.3:
            action = "sell"
        else:
            action = "no_trade"

        return {
            "id": opp.get("id", str(uuid.uuid4())),
            "symbol": opp.get("symbol", "UNKNOWN"),
            "instrument": opp.get("instrument", "equity"),
            "action": action,
            "confidence": confidence,
            "expected_profit": confidence * 0.05,  # Rough estimate
            "expected_risk": 0.02,
            "rationale": {
                "reason": "Default rule-based decision (engine unavailable)",
                "factors": ["alpha_threshold", "confidence_based"]
            },
            "orders": [],
            "timestamp": datetime.now().isoformat()
        }

    def _create_error_decision(self, opp: Dict, error: str) -> Dict:
        """Create error decision."""
        return {
            "id": opp.get("id", str(uuid.uuid4())),
            "symbol": opp.get("symbol", "UNKNOWN"),
            "action": "no_trade",
            "confidence": 0.0,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

    def _handle_learn(self):
        """Handle learning from decision outcomes."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            attribution_id = request_data.get('attribution_id')
            outcome = request_data.get('outcome')

            if not attribution_id or not outcome:
                self._send_error(400, "Missing attribution_id or outcome")
                return

            # Update decision scorer with outcome
            if self.decision_scorer:
                try:
                    self.decision_scorer.close_and_score_decision(attribution_id, outcome)
                    self._send_json_response(200, {"status": "learned"})
                except Exception as e:
                    self._send_error(500, f"Learning failed: {str(e)}")
            else:
                self._send_json_response(200, {"status": "learning_disabled"})

        except Exception as e:
            self._send_error(500, f"Learn request failed: {str(e)}")

    def _handle_record_decision(self):
        """Handle decision recording."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            if self.decision_scorer:
                decision_id = self.decision_scorer.record_decision(
                    signal_id=request_data.get('signal_id', str(uuid.uuid4())),
                    decision_data=request_data
                )
                self._send_json_response(200, {"decision_id": decision_id})
            else:
                self._send_json_response(200, {"status": "recording_disabled"})

        except Exception as e:
            self._send_error(500, f"Record request failed: {str(e)}")

    def _handle_get_decision(self):
        """Handle getting recorded decision."""
        try:
            decision_id = self.path.split('/')[-1]

            if self.decision_scorer:
                # This would need to be implemented in DecisionScorer
                decision = {"id": decision_id, "status": "not_implemented"}
                self._send_json_response(200, decision)
            else:
                self._send_error(404, "Decision recorder not available")

        except Exception as e:
            self._send_error(500, f"Get decision failed: {str(e)}")

    def _send_json_response(self, status_code: int, data: Dict):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_json_response(status_code, {
            "error": message,
            "timestamp": datetime.now().isoformat()
        })

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class PythonBrainService:
    """Main Python Brain Service class."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "host": "localhost",
            "port": 8001,
            "enable_decision_engine": True,
            "enable_decision_scorer": True
        }

        self.decision_engine = None
        self.decision_scorer = None
        self.server = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize service components."""
        try:
            # Initialize decision engine
            if self.config.get("enable_decision_engine", True):
                policy_service = PolicyService()
                self.decision_engine = DecisionEngine(policy_service)
                logger.info("Decision Engine initialized")

            # Initialize decision scorer
            if self.config.get("enable_decision_scorer", True):
                self.decision_scorer = DecisionScorer()
                logger.info("Decision Scorer initialized")

        except Exception as e:
            logger.warning(f"Component initialization failed: {e}")

    def start(self):
        """Start the HTTP service."""
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 8001)

        # Create custom handler with components
        def handler_class(*args, **kwargs):
            return PythonBrainHandler(*args,
                                    decision_engine=self.decision_engine,
                                    decision_scorer=self.decision_scorer,
                                    **kwargs)

        try:
            self.server = HTTPServer((host, port), handler_class)
            print(f"🐍 Python Brain Service running on http://{host}:{port}")

            # Set start time for uptime tracking
            setattr(PythonBrainHandler, '_start_time', time.time())

            self.server.serve_forever()

        except KeyboardInterrupt:
            print("\n🐍 Python Brain Service shutting down...")
        except Exception as e:
            print(f"🐍 Python Brain Service error: {e}")
        finally:
            if self.server:
                self.server.shutdown()

    def stop(self):
        """Stop the service."""
        if self.server:
            self.server.shutdown()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Python Brain Service')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to listen on')
    parser.add_argument('--no-engine', action='store_true', help='Disable decision engine')
    parser.add_argument('--no-scorer', action='store_true', help='Disable decision scorer')

    args = parser.parse_args()

    config = {
        "host": args.host,
        "port": args.port,
        "enable_decision_engine": not args.no_engine,
        "enable_decision_scorer": not args.no_scorer
    }

    service = PythonBrainService(config)
    service.start()


if __name__ == '__main__':
    main()
