"""AI Chat API handlers for the Trading Dashboard.

This module provides API endpoints for the AI assistant chat functionality
to be integrated with the new React-based trading dashboard.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body, Query, WebSocket, WebSocketDisconnect
import json

# Import the BenBot assistant
try:
    from trading_bot.assistant.benbot_assistant import BenBotAssistant
    BENBOT_AVAILABLE = True
except ImportError:
    BENBOT_AVAILABLE = False
    logging.warning("BenBotAssistant not available. Falling back to mock implementation.")

# Import orchestrator and memory for context
try:
    from trading_bot.core.orchestrator import MainOrchestrator
    from trading_bot.memory.memory_manager import MemoryManager
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    logging.warning("MainOrchestrator not available. AI chat will operate with limited context.")

# Create router
router = APIRouter(prefix="/ai", tags=["AI Assistant"])

# Global state
chat_history = []
connected_clients = []
assistant = None

def get_assistant():
    """Get or initialize the BenBot assistant."""
    global assistant
    if assistant is None:
        if BENBOT_AVAILABLE:
            try:
                logging.info("Initializing BenBot assistant")
                assistant = BenBotAssistant()
                if ORCHESTRATOR_AVAILABLE:
                    from trading_bot.core.orchestrator import get_orchestrator
                    orchestrator = get_orchestrator()
                    assistant.set_orchestrator(orchestrator)
                    logging.info("Connected BenBot assistant to orchestrator")
            except Exception as e:
                logging.error(f"Error initializing BenBot assistant: {e}")
                assistant = MockAssistant()
        else:
            assistant = MockAssistant()
            
    return assistant

class MockAssistant:
    """Mock implementation of BenBot assistant for development/fallback."""
    
    def __init__(self):
        self.conversation_history = []
        
    def process_query(self, query, context=None):
        """Process a user query and return a response."""
        if not context:
            context = {}
            
        # Basic context-aware responses
        if "help" in query.lower():
            return "I'm your BenBot trading assistant. I can help with market analysis, trading strategies, portfolio monitoring, and more. What would you like to know?"
            
        if "market" in query.lower() and "analysis" in query.lower():
            return f"Based on current market conditions, I'm seeing {'bullish' if context.get('market', {}).get('sentiment', 0) > 0 else 'bearish'} momentum. Major indices are showing moderate volatility."
            
        if "strategy" in query.lower():
            return "I recommend considering momentum-based strategies in the current market regime. Our best performing strategies recently have been trend-following with proper risk management."
            
        if "portfolio" in query.lower():
            return f"Your portfolio is currently {'up' if context.get('portfolioValue', 0) > 0 else 'down'} today. I'm monitoring open positions and will alert you to any significant changes."
            
        # Default response
        return f"I understand you're asking about '{query}'. I'm analyzing the current trading context to provide you with the most relevant information. Can you provide more specific details about what you'd like to know?"
        
    def get_conversation_history(self):
        """Return the conversation history."""
        return self.conversation_history


@router.post("/chat/send")
async def send_message(
    message: str = Body(..., embed=True),
    context: Dict[str, Any] = Body({}, embed=True)
):
    """Send a message to the AI assistant."""
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Add user message to history
    user_message = {
        "id": message_id,
        "role": "user",
        "content": message,
        "timestamp": timestamp
    }
    chat_history.append(user_message)
    
    # Get assistant and process query
    assistant = get_assistant()
    
    # Add context to BenBot if available
    context_obj = {
        "current_tab": context.get("currentTab", "Dashboard"),
        "symbol": context.get("symbol", None),
        "portfolio_value": context.get("portfolioValue", 0),
        "market_regime": context.get("market", {}).get("regime", "unknown"),
        "opportunities": context.get("opportunities", []),
        "open_positions": context.get("openPositions", [])
    }
    
    # Process the query
    response = assistant.process_query(message, context=context_obj)
    
    # Add assistant response to history
    assistant_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    }
    chat_history.append(assistant_message)
    
    # Broadcast to WebSocket clients
    for client in connected_clients:
        try:
            await client.send_json({
                "type": "ai_chat_message",
                "data": {
                    "messageId": assistant_message["id"],
                    "type": "new",
                    "message": assistant_message
                }
            })
        except Exception as e:
            logging.error(f"Error sending WebSocket message: {e}")
    
    return {"messageId": message_id}


@router.get("/chat/history")
async def get_chat_history(
    limit: int = Query(50, description="Maximum number of messages to return"),
    before: Optional[str] = Query(None, description="Timestamp to retrieve messages before")
):
    """Get chat message history."""
    filtered_history = chat_history
    
    if before:
        try:
            before_dt = datetime.fromisoformat(before)
            filtered_history = [msg for msg in filtered_history 
                              if datetime.fromisoformat(msg["timestamp"]) < before_dt]
        except (ValueError, TypeError):
            pass
    
    # Return most recent messages first, limited by the 'limit' parameter
    messages = filtered_history[-limit:][::-1]
    
    return {
        "messages": messages,
        "hasMore": len(filtered_history) > limit
    }


@router.post("/chat/clear")
async def clear_history():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return {"success": True}


@router.get("/context")
async def get_trading_context():
    """Get current trading context for AI assistant."""
    # In a real implementation, this would fetch data from the orchestrator
    assistant = get_assistant()
    
    if ORCHESTRATOR_AVAILABLE and hasattr(assistant, "orchestrator") and assistant.orchestrator:
        # Extract real context from orchestrator
        orchestrator = assistant.orchestrator
        
        # Get market context
        try:
            market_context = orchestrator.get_market_context()
            positions = orchestrator.get_positions()
            strategies = orchestrator.get_active_strategies()
            opportunities = orchestrator.get_trading_opportunities()
            
            return {
                "currentTab": "Dashboard",  # Default tab
                "assetClasses": ["stocks", "options", "crypto", "forex"],
                "market": {
                    "regime": market_context.get("regime", "unknown"),
                    "volatility": market_context.get("volatility", 0),
                    "sentiment": market_context.get("sentiment", 0)
                },
                "portfolioValue": positions.get("total_value", 0),
                "openPositions": positions.get("positions", []),
                "opportunities": opportunities,
                "strategies": [s.get("name") for s in strategies]
            }
        except Exception as e:
            logging.error(f"Error getting context from orchestrator: {e}")
            
    # Fallback mock context
    return {
        "currentTab": "Dashboard",
        "assetClasses": ["stocks", "options", "crypto", "forex"],
        "market": {
            "regime": "neutral",
            "volatility": 0.15,
            "sentiment": 0.2
        },
        "portfolioValue": 10000,
        "openPositions": [],
        "opportunities": [],
        "strategies": ["TrendFollowing", "MeanReversion"]
    }


@router.get("/orchestrator/context")
async def get_full_orchestrator_context():
    """Get complete orchestrator context for AI assistant."""
    # This would fetch complete data from the orchestrator
    # For now, return a mock response
    return {
        "market": {
            "regime": "neutral",
            "volatility": 0.15,
            "sentiment": 0.2,
            "keyLevels": {
                "SPY": [410, 420, 430],
                "QQQ": [350, 360, 370]
            },
            "majorIndices": {
                "SPY": {"price": 425.67, "change": 0.45},
                "QQQ": {"price": 362.89, "change": 0.78},
                "DIA": {"price": 345.12, "change": 0.12}
            }
        },
        "portfolio": {
            "value": 10000,
            "cashBalance": 2500,
            "openPositions": [],
            "recentTrades": [],
            "performance": {
                "daily": 0.45,
                "weekly": 1.32,
                "monthly": 2.78,
                "ytd": 8.95
            }
        },
        "strategies": {
            "active": ["TrendFollowing", "MeanReversion"],
            "paused": ["OptionsPremium"],
            "performance": {
                "TrendFollowing": {
                    "winRate": 0.65,
                    "profitFactor": 1.8,
                    "expectancy": 0.45,
                    "sharpe": 1.2
                },
                "MeanReversion": {
                    "winRate": 0.58,
                    "profitFactor": 1.5,
                    "expectancy": 0.32,
                    "sharpe": 0.95
                }
            },
            "activeSessions": []
        },
        "signals": {
            "opportunities": [],
            "alerts": []
        },
        "system": {
            "health": {
                "memoryUsage": 0.45,
                "cpuUsage": 0.32,
                "apiLatency": 120,
                "errorRate": 0.01
            },
            "dataQuality": {
                "marketDataFreshness": 15,
                "missingDataPoints": 0,
                "dataSources": ["AlphaVantage", "Polygon", "Alpaca"]
            }
        }
    }


@router.get("/context/symbol/{symbol}")
async def get_symbol_context(symbol: str):
    """Get context for a specific symbol."""
    # This would fetch symbol-specific data from the orchestrator
    # For now, return a mock response
    return {
        "symbol": symbol,
        "price": 156.78,
        "change": 0.67,
        "volume": 5432100,
        "indicators": {
            "rsi": 58.6,
            "macd": 0.45,
            "ema20": 155.23,
            "ema50": 152.89,
            "ema200": 148.45,
            "bb_width": 0.12
        },
        "sentiment": 0.65,
        "newsCount": 3,
        "analystRatings": {
            "buy": 15,
            "hold": 8,
            "sell": 2
        }
    }


@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat updates."""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)
