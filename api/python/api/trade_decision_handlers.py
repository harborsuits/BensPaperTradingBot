"""
WebSocket handlers for trade decisions and candidates.
These handlers manage real-time updates for trade candidates, execution decisions,
and other trading decision-related information.
"""

import asyncio
import json
import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from pydantic import BaseModel

from trading_bot.api.websocket_handlers import ConnectionManager, WebSocketMessage

logger = logging.getLogger(__name__)

class TradeCandidateUpdate(BaseModel):
    """Model for trade candidate updates sent via WebSocket"""
    id: str
    symbol: str
    strategy_id: str
    strategy_name: str
    direction: str 
    score: float
    entry_price: float
    target_price: float
    stop_loss: float
    potential_profit_pct: float
    risk_reward_ratio: float
    confidence: float
    time_validity: str
    timeframe: str
    created_at: datetime
    status: str
    executed: bool
    reason: Optional[str] = None
    tags: List[str]
    entry_conditions: List[str]
    indicators: List[Dict[str, Any]]


class TradeDecisionHandler:
    """Handler for trade decision WebSocket messages and updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.candidate_update_task = None
        self.trade_candidates = []
        self._initialize_mock_candidates()
    
    def _initialize_mock_candidates(self):
        """Initialize mock trade candidates for development/testing"""
        # Sample symbols
        symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "INTC"]
        
        # Sample strategies
        strategies = [
            {"id": "st_001", "name": "Trend Following"},
            {"id": "st_002", "name": "Mean Reversion"},
            {"id": "st_003", "name": "Breakout"},
            {"id": "st_004", "name": "Gap Trading"},
            {"id": "st_005", "name": "Volatility Expansion"}
        ]
        
        # Sample entry conditions
        entry_condition_sets = [
            ["Price crossed above 20 EMA", "RSI > 60", "Volume surge > 150%"],
            ["Bullish engulfing pattern", "Support level test", "Market in uptrend"],
            ["50 SMA crosses above 200 SMA", "MACD signal line crossover", "Stochastic oversold"],
            ["Gap up on high volume", "Previous resistance broken", "Sector strength"],
            ["Double bottom formation", "Volume confirmation", "Positive divergence on RSI"]
        ]
        
        # Generate random sample candidates
        self.trade_candidates = []
        for i in range(8):  # Generate 8 sample candidates
            # Random parameters
            symbol = random.choice(symbols)
            strategy = random.choice(strategies)
            direction = random.choice(["buy", "sell"])
            score = round(random.uniform(0.3, 0.95), 2)
            entry_price = round(random.uniform(50, 500), 2)
            
            # Derived values
            target_mult = random.uniform(1.5, 3.0) if direction == "buy" else random.uniform(0.3, 0.7)
            stop_mult = random.uniform(0.7, 0.95) if direction == "buy" else random.uniform(1.05, 1.3)
            target_price = round(entry_price * target_mult, 2) if direction == "buy" else round(entry_price * target_mult, 2)
            stop_loss = round(entry_price * stop_mult, 2) if direction == "buy" else round(entry_price * stop_mult, 2)
            
            # Risk calculations
            risk_per_share = abs(entry_price - stop_loss)
            reward_per_share = abs(target_price - entry_price)
            risk_reward_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0
            potential_profit_pct = round((target_price - entry_price) / entry_price * 100, 2) if direction == "buy" else round((entry_price - target_price) / entry_price * 100, 2)
            
            # Status
            executed = random.random() > 0.5
            status = random.choice(["executed", "watching", "pending", "ready", "expired", "skipped"]) if not executed else "executed"
            
            # Indicators (3-5 random technical indicators)
            indicators = []
            indicator_names = ["RSI", "MACD", "ATR", "OBV", "ADX", "CCI", "Stochastic", "MFI", "Bollinger"]
            signal_options = ["bullish", "bearish", "neutral"]
            num_indicators = random.randint(3, 5)
            
            for _ in range(num_indicators):
                ind_name = random.choice(indicator_names)
                indicators.append({
                    "name": ind_name,
                    "value": round(random.uniform(0, 100), 2),
                    "signal": random.choices(signal_options, weights=[0.6, 0.3, 0.1] if direction == "buy" else [0.3, 0.6, 0.1])[0]
                })
                indicator_names.remove(ind_name)  # Prevent duplicates
            
            # Create candidate
            candidate = {
                "id": f"tc_{i+1:03d}",
                "symbol": symbol,
                "strategy_id": strategy["id"],
                "strategy_name": strategy["name"],
                "direction": direction,
                "score": score,
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "potential_profit_pct": potential_profit_pct,
                "risk_reward_ratio": risk_reward_ratio,
                "confidence": round(score * random.uniform(0.8, 1.2), 2),
                "time_validity": f"{random.randint(1, 4)}h",
                "timeframe": random.choice(["5m", "15m", "1h", "4h", "D"]),
                "created_at": datetime.now().isoformat(),
                "status": status,
                "executed": executed,
                "reason": f"High probability setup with strong {direction} signal and confirming volume patterns" if executed 
                          else "Signal not strong enough to exceed execution threshold",
                "tags": random.sample(["momentum", "technical", "reversal", "support", "resistance", "volume"], 2),
                "entry_conditions": random.choice(entry_condition_sets),
                "indicators": indicators
            }
            
            self.trade_candidates.append(candidate)
    
    async def start_candidate_updates(self):
        """Start the background task for periodic trade candidate updates"""
        if self.candidate_update_task is None or self.candidate_update_task.done():
            self.candidate_update_task = asyncio.create_task(self._send_periodic_candidate_updates())
            logger.info("Started trade candidate update background task")
    
    async def stop_candidate_updates(self):
        """Stop the background task for trade candidate updates"""
        if self.candidate_update_task and not self.candidate_update_task.done():
            self.candidate_update_task.cancel()
            try:
                await self.candidate_update_task
            except asyncio.CancelledError:
                pass
            self.candidate_update_task = None
            logger.info("Stopped trade candidate update background task")
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming WebSocket messages related to trade decisions"""
        if message.get("type") == "decisions_request":
            request_type = message.get("request")
            
            if request_type == "trade_candidates":
                return {
                    "type": "trade_candidates",
                    "data": {
                        "candidates": self.trade_candidates
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            elif request_type == "executed_trades":
                # Return only executed trade candidates
                executed_trades = [c for c in self.trade_candidates if c.get("executed", False)]
                return {
                    "type": "executed_trades",
                    "data": {
                        "trades": executed_trades
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            elif request_type == "decision_metrics":
                # This would normally fetch real metrics data from your decision engine
                return {
                    "type": "decision_metrics",
                    "data": {
                        "total_candidates": len(self.trade_candidates),
                        "executed_count": sum(1 for c in self.trade_candidates if c.get("executed", False)),
                        "avg_score": sum(c.get("score", 0) for c in self.trade_candidates) / len(self.trade_candidates) if self.trade_candidates else 0,
                        "execution_threshold": 0.65
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        return None
    
    async def _send_periodic_candidate_updates(self):
        """Send periodic trade candidate updates to all connected clients"""
        try:
            # Initial broadcast of existing candidates
            await self.broadcast_candidates()
            
            while True:
                # Wait for next update interval (30-60 seconds)
                await asyncio.sleep(random.randint(30, 60))
                
                # Randomly decide whether to generate new candidates this cycle (40% chance)
                if random.random() < 0.4:
                    await self._generate_new_candidates()
                    await self.broadcast_candidates()
                
                # Randomly update status of existing candidates (30% chance)
                elif random.random() < 0.3:
                    await self._update_candidate_statuses()
                    await self.broadcast_candidates()
                    
        except asyncio.CancelledError:
            logger.info("Trade candidate update task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in trade candidate update task: {e}")
    
    async def _generate_new_candidates(self):
        """Generate new mock trade candidates"""
        # This simulates the generation of new trade candidates from your decision engine
        # In a real implementation, this would fetch actual candidates from your trading algorithms
        
        # Keep only recent candidates (discard older ones to keep the list manageable)
        if len(self.trade_candidates) > 10:
            self.trade_candidates = self.trade_candidates[-5:]  # Keep the 5 most recent
        
        # Add 1-3 new candidates
        self._initialize_mock_candidates()
        
        logger.info(f"Generated new trade candidates, now tracking {len(self.trade_candidates)} candidates")
    
    async def _update_candidate_statuses(self):
        """Update statuses of existing trade candidates"""
        for candidate in self.trade_candidates:
            # Only update non-final statuses
            if candidate["status"] not in ["executed", "expired", "skipped"]:
                # Random status progression
                status_progression = {
                    "pending": ["watching", "ready", "skipped"],
                    "watching": ["ready", "executed", "skipped"],
                    "ready": ["executed", "skipped"]
                }
                
                if candidate["status"] in status_progression:
                    # Choose next status with weighted probabilities
                    weights = [0.6, 0.3, 0.1] if len(status_progression[candidate["status"]]) == 3 else [0.7, 0.3]
                    new_status = random.choices(
                        status_progression[candidate["status"]], 
                        weights=weights[:len(status_progression[candidate["status"]])]
                    )[0]
                    
                    candidate["status"] = new_status
                    
                    # Update executed flag and reason if status changed to executed
                    if new_status == "executed":
                        candidate["executed"] = True
                        candidate["reason"] = f"Signal confirmed with increasing momentum and strong market conditions"
                    
                    # Update timestamp
                    candidate["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Updated candidate statuses")
    
    async def broadcast_candidates(self):
        """Broadcast the current trade candidates to all connected clients"""
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="cycle_decision",
                channel="decisions",
                data={"candidates": self.trade_candidates},
                timestamp=datetime.now().isoformat()
            )
        )
    
    async def broadcast_execution_decision(self, decision_data: Dict[str, Any]):
        """Broadcast a trade execution decision to all connected clients"""
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="trade_execution",
                channel="decisions",
                data=decision_data,
                timestamp=datetime.now().isoformat()
            )
        )
    
    async def broadcast_decision_metrics(self, metrics_data: Dict[str, Any]):
        """Broadcast updated decision metrics to all connected clients"""
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="decision_metrics",
                channel="decisions",
                data=metrics_data,
                timestamp=datetime.now().isoformat()
            )
        )
