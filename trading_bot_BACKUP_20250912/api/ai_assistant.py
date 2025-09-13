#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot AI Assistant

This module provides AI assistant functionality for the trading bot,
enabling natural language interaction with trading data and strategies.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import from trading bot components
from trading_bot.config.unified_config import get_config

# Initialize logging
logger = logging.getLogger("TradingBotAI")

class AIAssistant:
    """
    AI Assistant for the trading bot that integrates with various
    language model providers (OpenAI, Claude, etc.)
    """
    
    def __init__(self):
        self.config = get_config().get("ai_assistant", {})
        self.provider = self.config.get("provider", "simulated")
        self.api_key = self.config.get("api_key", "")
        self.model = self.config.get("model", "")
        
        # History cache
        self.conversation_history = {}
        
        logger.info(f"AI Assistant initialized with provider: {self.provider}")
    
    def get_response(self, message: str, context: str = "trading", 
                     conversation_id: str = "default") -> Dict[str, Any]:
        """
        Get a response from the AI assistant.
        
        Args:
            message: User message to respond to
            context: Context for the conversation (trading, portfolio, etc.)
            conversation_id: ID of the conversation to continue
            
        Returns:
            Response dict with content and metadata
        """
        # Get or create conversation history
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        history = self.conversation_history[conversation_id]
        history.append({"role": "user", "content": message})
        
        # Generate response based on provider
        if self.provider == "openai":
            response = self._openai_response(message, history, context)
        elif self.provider == "claude":
            response = self._claude_response(message, history, context)
        elif self.provider == "mistral":
            response = self._mistral_response(message, history, context)
        else:
            # Simulated mode with pre-defined responses
            response = self._simulated_response(message, context)
        
        # Add to history
        history.append({"role": "assistant", "content": response["content"]})
        
        # Return response with metadata
        return {
            "content": response["content"],
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "provider": self.provider,
            "context": context
        }
    
    def _openai_response(self, message: str, history: List[Dict[str, str]], 
                         context: str) -> Dict[str, str]:
        """Get response from OpenAI API"""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a trading assistant specializing in {context}."},
                    *history
                ]
            )
            
            return {"content": response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {str(e)}")
            return {"content": f"Sorry, I'm having trouble connecting to my AI backend. Error: {str(e)}"}
    
    def _claude_response(self, message: str, history: List[Dict[str, str]], 
                         context: str) -> Dict[str, str]:
        """Get response from Anthropic Claude API"""
        try:
            import anthropic
            client = anthropic.Client(api_key=self.api_key)
            
            # Convert history to Claude format
            claude_messages = []
            for msg in history:
                if msg["role"] == "user":
                    claude_messages.append({"role": "user", "content": msg["content"]})
                else:
                    claude_messages.append({"role": "assistant", "content": msg["content"]})
            
            response = client.messages.create(
                model=self.model or "claude-2",
                system=f"You are a trading assistant specializing in {context}.",
                messages=claude_messages
            )
            
            return {"content": response.content[0].text}
        except Exception as e:
            logger.error(f"Error getting Claude response: {str(e)}")
            return {"content": f"Sorry, I'm having trouble connecting to my AI backend. Error: {str(e)}"}
    
    def _mistral_response(self, message: str, history: List[Dict[str, str]], 
                          context: str) -> Dict[str, str]:
        """Get response from Mistral AI API"""
        try:
            import mistralai.client
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            
            client = MistralClient(api_key=self.api_key)
            
            # Convert history to Mistral format
            mistral_messages = [
                ChatMessage(role="system", content=f"You are a trading assistant specializing in {context}.")
            ]
            
            for msg in history:
                mistral_messages.append(
                    ChatMessage(role=msg["role"], content=msg["content"])
                )
            
            chat_response = client.chat(
                model=self.model or "mistral-medium",
                messages=mistral_messages
            )
            
            return {"content": chat_response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Error getting Mistral response: {str(e)}")
            return {"content": f"Sorry, I'm having trouble connecting to my AI backend. Error: {str(e)}"}
    
    def _simulated_response(self, message: str, context: str) -> Dict[str, str]:
        """Generate simulated responses for development and testing"""
        query = message.lower()
        
        if context == "trading":
            if "portfolio" in query or "holdings" in query:
                return {"content": "Your portfolio is currently up 1.49% today ($12,483.57). Tech stocks are your strongest performers - AAPL (+2.3%), MSFT (+1.8%), and NVDA (+3.2%) following positive analyst coverage on AI chip demand. Your financial sector positions are underperforming with JPM (-0.4%) and BAC (-0.8%)."}
            elif "strategy" in query or "strategies" in query:
                return {"content": "Trading Strategy Performance (Last 30 Days):\n\n1. Momentum Alpha: +4.2% | Win Rate: 68% | Sharpe: 1.82 ✓ PERFORMING WELL\n2. Mean Reversion: +1.8% | Win Rate: 52% | Sharpe: 1.15 ✓ MEETING TARGETS\n3. Volatility Edge: +6.3% | Win Rate: 73% | Sharpe: 2.14 ✓ OUTPERFORMING\n4. Sector Rotation: -2.1% | Win Rate: 42% | Sharpe: 0.74 ✗ UNDERPERFORMING\n5. Pairs Trading: +0.9% | Win Rate: 51% | Sharpe: 0.96 ~ NEUTRAL\n\nOverall strategy blend is delivering alpha of +2.1% compared to benchmark. Would you like me to explain any of these strategies in more detail?"}
            elif "market" in query or "stocks" in query:
                return {"content": "The market is showing strength today with the S&P 500 up 0.8% and Nasdaq up 1.2%. Tech and healthcare sectors are leading gains while energy stocks are showing weakness due to declining oil prices. Trading volume is 12% above the 30-day average, suggesting strong institutional participation.\n\nKey market events today:\n- Fed meeting minutes released (dovish tone)\n- Tech earnings beating expectations (82% of reports)\n- Treasury yields down 5bp across the curve"}
            elif "crypto" in query or "bitcoin" in query or "ethereum" in query:
                return {"content": "Crypto Market Update:\n\n- Bitcoin: $78,245 (+1.6% today)\n- Ethereum: $4,321 (+3.2% today)\n- Your crypto allocation: 25% of portfolio\n- Top holdings: BTC (12%), ETH (8%), SOL (3%)\n\nKey Developments: ETH showing strength ahead of the protocol upgrade scheduled for next week. Institutional inflows to crypto have increased 18% this month."}
            elif "risk" in query or "exposure" in query:
                return {"content": "Risk Analysis:\n- Portfolio Beta: 1.2 (slightly above your 1.0 target)\n- VaR (95%): $24,720\n- Sector concentration risk: High in Technology (42% allocation)\n- Currency exposure: 9% international\n\nRecommendation: Consider adding utilities or consumer staples positions to balance technology exposure and reduce overall portfolio volatility."}
            elif "recommend" in query or "suggest" in query:
                return {"content": "Based on your current portfolio allocation and market conditions, I recommend considering these moves:\n\n1. Increase exposure to semiconductor stocks (AMD, AVGO, TSM) - recent supply chain improvements and AI demand create favorable conditions\n\n2. Consider taking profits on NVDA which is now 12% above our target price\n\n3. Add to your financial sector positions on weakness to improve sector balance"}
            elif "hello" in query or "hi" in query or "hey" in query:
                return {"content": "Hello! I'm your trading assistant. How can I help with your trading today? Would you like to review your portfolio, check market conditions, or get trading recommendations?"}
            else:
                return {"content": "I can provide you with insights on multiple aspects of your trading and investments:\n\n- Portfolio analysis and performance\n- Market conditions and sector trends\n- Trading recommendations\n- Risk assessment\n- News impact analysis\n- Economic outlook\n\nWhat specific information would you like today?"}
        else:
            return {"content": "I'm not familiar with that context. I can help with trading, portfolio analysis, market insights, and investment recommendations."}


# Create a singleton instance
_assistant = None

def get_assistant() -> AIAssistant:
    """Get the singleton AI Assistant instance"""
    global _assistant
    if _assistant is None:
        _assistant = AIAssistant()
    return _assistant
