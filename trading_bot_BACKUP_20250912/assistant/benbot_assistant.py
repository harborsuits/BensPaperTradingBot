#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BenBot Assistant

This module implements a natural language interface for interacting with
the trading bot. It processes user queries, generates appropriate responses,
and interfaces with the main orchestrator to execute trading commands.
"""

import logging
import re
from typing import Dict, List, Optional, Union, Any
import json
import os
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Import NLP and AI libraries if available
try:
    import numpy as np
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False
    
# Try to import OpenAI, but don't fail if it's not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

# Import AI service libraries
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available. Install with: pip install anthropic")
    
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("YAML library not available. Install with: pip install pyyaml")

class BenBotAssistant:
    """
    Natural language assistant for the trading bot system.
    
    This class provides a conversational interface to control the trading bot,
    retrieve information, and perform trading operations using natural language.
    """
    
    def __init__(self, orchestrator=None, config=None):
        """
        Initialize the BenBot Assistant.
        
        Args:
            orchestrator: Optional reference to the MainOrchestrator instance
            config: Configuration dictionary for the assistant
        """
        self.orchestrator = orchestrator
        self.config = config or {}
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Intent patterns - basic regex patterns for detecting user intents
        self.intent_patterns = {
            "run_strategy": r"(?i)run\s+(?:the\s+)?(?:trading\s+)?(?:strategy|strategies)(?:\s+for\s+(.+))?",
            "market_analysis": r"(?i)(?:what\s+is\s+)?(?:the\s+)?(?:current\s+)?market\s+(?:regime|status|analysis)",
            "trading_opportunities": r"(?i)(?:show|get|find)(?:\s+me)?\s+(?:the\s+)?(?:trading\s+)?opportunities",
            "portfolio_status": r"(?i)(?:what\s+is\s+)?(?:the\s+)?(?:current\s+)?portfolio\s+(?:status|value|holdings)",
            "help": r"(?i)(?:help|assist|what\s+can\s+you\s+do|commands)",
            "greeting": r"(?i)(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))",
        }
        
        # Initialize AI service clients
        self.ai_config = self._load_ai_config()
        self.ai_service = self.ai_config.get('provider', 'local')
        self._initialize_ai_service()
        
        logger.info(f"BenBot Assistant initialized with {self.ai_service} AI service")
    
    def set_orchestrator(self, orchestrator):
        """Set the main orchestrator reference."""
        self.orchestrator = orchestrator
        logger.info("Orchestrator reference set in BenBot Assistant")
    
    def _load_ai_config(self):
        """Load AI configuration from config file."""
        if not YAML_AVAILABLE:
            logger.warning("YAML library not available, using default AI config")
            return {'provider': 'local'}
            
        # Try to load from config file
        try:
            config_path = os.path.join(os.path.dirname(__file__), '../config/ai_config.yaml')
            keys_path = os.path.join(os.path.dirname(__file__), '../config/ai_keys.yaml')
            
            # Load main AI configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file).get('ai_assistant', {})
                    logger.info(f"Loaded AI config from {config_path}")
            else:
                logger.warning(f"AI config file not found at {config_path}, using defaults")
                config = {'provider': 'local'}
            
            # Load API keys if available
            if os.path.exists(keys_path):
                with open(keys_path, 'r') as file:
                    keys = yaml.safe_load(file)
                    logger.info(f"Loaded AI keys from {keys_path}")
                    
                    # Merge keys into config
                    if keys and isinstance(keys, dict):
                        for service, service_config in keys.items():
                            if service in config and isinstance(service_config, dict):
                                config[service].update(service_config)
                            else:
                                config[service] = service_config
            else:
                logger.warning(f"AI keys file not found at {keys_path}")
                
            return config
        except Exception as e:
            logger.error(f"Error loading AI config: {e}")
            return {'provider': 'local'}
    
    def _initialize_ai_service(self):
        """Initialize AI service based on configuration."""
        provider = self.ai_config.get('provider', 'local')
        
        if provider == 'openai' and OPENAI_AVAILABLE:
            # Handle both nested and flat key structures
            api_key = self.ai_config.get('openai', '')
            if isinstance(api_key, dict):
                api_key = api_key.get('api_key', '')
            else:
                api_key = api_key
                
            logger.info(f"OpenAI API key available: {bool(api_key)}")
            if api_key:
                openai.api_key = api_key
                logger.info("OpenAI service initialized")
            else:
                logger.warning("OpenAI API key not provided, using local mode")
                self.ai_config['provider'] = 'local'
                
        elif provider == 'anthropic' and ANTHROPIC_AVAILABLE:
            # Handle both nested and flat key structures
            api_key = self.ai_config.get('anthropic', '')
            if isinstance(api_key, dict):
                api_key = api_key.get('api_key', '')
            else:
                api_key = api_key
                
            logger.info(f"Anthropic API key available: {bool(api_key)}")
            if api_key:
                # Store the API key for later use with Anthropic client
                self.anthropic_api_key = api_key
                logger.info("Anthropic service initialized")
            else:
                logger.warning("Anthropic API key not provided, using local mode")
                self.ai_config['provider'] = 'local'
        else:
            if provider != 'local':
                logger.warning(f"{provider} service not available, falling back to local processing")
            self.ai_service = 'local'
    
    def process_query(self, query: str):
        """
        Process a natural language query and return a response.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Response string with results or action confirmation
        """
        # Clean up the input
        query = query.strip()
        
        # Log the query
        logger.info(f"Processing query: {query}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        response = None
        try:
            # Try AI service if configured and API key is available
            if self.ai_service != 'local':
                logger.info(f"Using AI service: {self.ai_service}")
                response = self._get_ai_response(query)
                if response:
                    logger.info("Got response from AI service")
                    logger.info(f"AI response (first 30 chars): {response[:30]}...")
                else:
                    logger.warning("AI service returned empty response, falling back to intent matching")
        except Exception as e:
            logger.error(f"Error using AI service: {e}")
            # Fall back to intent matching
            response = None
        
        # If AI didn't work, use intent matching
        if not response:
            logger.info("Using intent matching for response")
            response = self._match_intent(query)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    def _match_intent(self, query: str) -> str:
        """Match the query against known intent patterns and handle accordingly."""
        
        # Try to match against known patterns
        for intent, pattern in self.intent_patterns.items():
            match = re.search(pattern, query)
            if match:
                logger.info(f"Matched intent: {intent}")
                
                # Call the appropriate handler based on intent
                if intent == "run_strategy":
                    # Extract strategy name if present
                    strategy_name = match.group(1) if match.groups() else None
                    return self._handle_run_strategy(strategy_name)
                
                elif intent == "market_analysis":
                    return self._handle_market_analysis()
                
                elif intent == "trading_opportunities":
                    return self._handle_trading_opportunities()
                
                elif intent == "portfolio_status":
                    return self._handle_portfolio_status()
                
                elif intent == "help":
                    return self._handle_help()
                
                elif intent == "greeting":
                    return self._handle_greeting()
        
        # No intent matched, try generic response
        return self._handle_unknown_intent(query)
    
    # Intent handlers
    
    def _handle_run_strategy(self, strategy_name=None):
        """Handle request to run a trading strategy."""
        try:
            # Call the orchestrator's run_pipeline method
            result = self.orchestrator.run_pipeline(strategy_name)
            
            if strategy_name:
                return f"I've executed the {strategy_name} strategy. {result}"
            else:
                return f"I've executed all active trading strategies. {result}"
                
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return f"I couldn't run the strategy. Error: {str(e)}"
    
    def _handle_market_analysis(self):
        """Handle request for current market analysis."""
        try:
            # Get market regime from orchestrator
            market_regime = self.orchestrator.get_market_regime()
            
            if market_regime:
                regime = market_regime.get("regime", "Unknown")
                confidence = market_regime.get("confidence", 0)
                trend_strength = market_regime.get("trend_strength", "Unknown")
                
                return (f"Current market regime: {regime} (confidence: {confidence:.0%}). "
                        f"Trend strength is {trend_strength}.")
            else:
                return "I couldn't retrieve the current market analysis."
                
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            return f"I encountered an error while retrieving market analysis: {str(e)}"
    
    def _handle_trading_opportunities(self):
        """Handle request for current trading opportunities."""
        try:
            # Get opportunities from orchestrator
            opportunities = self.orchestrator.get_approved_opportunities()
            
            if opportunities and len(opportunities) > 0:
                response = "Here are the current trading opportunities:\n"
                for idx, opp in enumerate(opportunities, 1):
                    symbol = opp.get("symbol", "Unknown")
                    strategy = opp.get("strategy", "Unknown")
                    confidence = opp.get("confidence", 0)
                    response += f"{idx}. {symbol} ({strategy}, confidence: {confidence:.0%})\n"
                return response
            else:
                return "There are no trading opportunities identified at the moment."
                
        except Exception as e:
            logger.error(f"Error getting trading opportunities: {e}")
            return f"I encountered an error while retrieving trading opportunities: {str(e)}"
    
    def _handle_portfolio_status(self):
        """Handle request for portfolio status."""
        # This would typically call a portfolio manager service
        # For now, return a placeholder response
        return "Portfolio status functionality is not yet implemented."
    
    def _handle_help(self):
        """Handle help request."""
        help_text = (
            "Here's what you can ask me to do:\n"
            "1. Run trading strategies (e.g., 'Run the MomentumStrategy')\n"
            "2. Get market analysis (e.g., 'What's the current market regime?')\n"
            "3. Show trading opportunities (e.g., 'Show me the current opportunities')\n"
            "4. Check portfolio status (e.g., 'What's my portfolio value?')\n"
        )
        return help_text
    
    def _handle_greeting(self):
        """Handle greeting."""
        return "Hello! I'm BenBot, your trading assistant. How can I help you today?"
    
    def _handle_unknown_intent(self, query):
        """Handle queries that don't match any known intent."""
        # This would typically use more advanced NLP for fallback handling
        # For now, return a simple response
        return "I'm not sure how to help with that. Try asking about running strategies, market analysis, or trading opportunities."
    
    def _get_ai_response(self, query: str) -> str:
        """Get response from AI service."""
        # Prepare system prompt specific to trading context
        system_prompt = self.ai_config.get('system_prompt', """You are BenBot, an AI assistant for a trading bot system. 
        You help users analyze market conditions, review trading strategies, and manage their portfolio. 
        Be concise, accurate, and focus on providing actionable trading insights.""")
        
        # Get conversation history in AI-friendly format
        # Only include the last 10 messages to avoid context length issues
        history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        # Call the appropriate AI service
        try:
            if self.ai_service == 'openai' and OPENAI_AVAILABLE and openai.api_key:
                logger.info("Using OpenAI for response generation")
                return self._get_openai_response(system_prompt, history, query)
            elif self.ai_service == 'anthropic' and ANTHROPIC_AVAILABLE:
                logger.info("Using Anthropic for response generation")
                return self._get_anthropic_response(system_prompt, history, query)
            else:
                logger.warning(f"AI service {self.ai_service} not properly configured")
                return None
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return None
    
    def _get_openai_response(self, system_prompt, history, query):
        """Get response from OpenAI service."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query if not already included
        if not (len(messages) > 1 and messages[-1]["role"] == "user"):
            messages.append({"role": "user", "content": query})
        
        # Get model from config
        model = self.ai_config.get('model', 'gpt-4-turbo')
        
        try:
            logger.info(f"Calling OpenAI API with model: {model}")
            
            # For newer client version
            if hasattr(openai, 'chat') and hasattr(openai.chat, 'completions'):
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                return response.choices[0].message.content
            # For older client version
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _get_anthropic_response(self, system_prompt, history, query):
        """Get response from Anthropic Claude service."""
        try:
            # Create Claude client
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # Prepare messages
            messages = []
            for msg in history:
                if msg.get("role") == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add current query if not already included
            if not (len(messages) > 0 and messages[-1]["role"] == "user"):
                messages.append({"role": "user", "content": query})
            
            # Get model from config
            model = self.ai_config.get('model', 'claude-3-opus-20240229')
            
            # Call Claude API
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                max_tokens=800
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
    
    def get_conversation_history(self):
        """Return the conversation history."""
        return self.conversation_history 