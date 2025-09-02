#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnhancedStrategyPrioritizer - Advanced AI-driven strategy prioritization system 
that leverages language models to evaluate market conditions and optimize 
strategy selection with contextual awareness, explainability, and risk management.
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import requests
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedStrategyPrioritizer")

# Import existing components
from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
from trading_bot.ai_scoring.regime_aware_strategy_prioritizer import RegimeAwareStrategyPrioritizer, RegimeClassifier
from trading_bot.utils.market_context_fetcher import MarketContextFetcher


class ContextualMemory:
    """
    Maintains a history of decisions, market contexts, and outcomes to provide
    temporal context for the language model.
    """
    
    def __init__(self, max_entries: int = 10, file_path: Optional[str] = None):
        """
        Initialize the contextual memory.
        
        Args:
            max_entries: Maximum number of historical entries to maintain
            file_path: Path to save/load memory (None for in-memory only)
        """
        self.max_entries = max_entries
        self.file_path = file_path
        self.memory = []
        
        # Load memory from file if available
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.memory = json.load(f)
                    
                # Ensure we don't exceed max entries
                if len(self.memory) > self.max_entries:
                    self.memory = self.memory[-self.max_entries:]
                    
                logger.info(f"Loaded {len(self.memory)} memory entries from {self.file_path}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self.memory = []
    
    def add_entry(self, entry: Dict[str, Any]) -> None:
        """
        Add a new entry to the memory.
        
        Args:
            entry: Dictionary containing decision context and outcome
        """
        # Add timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
            
        # Add to memory
        self.memory.append(entry)
        
        # Prune if exceeding max entries
        if len(self.memory) > self.max_entries:
            self.memory.pop(0)
            
        # Save to file if path provided
        if self.file_path:
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                with open(self.file_path, 'w') as f:
                    json.dump(self.memory, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving memory: {e}")
    
    def get_recent_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the most recent memory entries.
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            List of memory entries
        """
        if limit is None or limit >= len(self.memory):
            return self.memory.copy()
        
        return self.memory[-limit:]
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.memory = []
        if self.file_path and os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except Exception as e:
                logger.error(f"Error deleting memory file: {e}")


class RiskGuardrails:
    """
    Implements risk management constraints and circuit breakers to ensure 
    language model recommendations stay within acceptable bounds.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the risk guardrails.
        
        Args:
            config: Risk management configuration
        """
        self.config = config or self._get_default_config()
        
        # Current risk state
        self.risk_warnings = []
        self.current_risk_level = "normal"  # normal, elevated, high, critical
        
        logger.info("RiskGuardrails initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk management configuration."""
        return {
            "strategy_constraints": {
                "default": {
                    "min_allocation": 0.0,
                    "max_allocation": 40.0,
                    "max_change": 15.0
                },
                # Strategy-specific constraints override defaults
                "momentum": {
                    "min_allocation": 5.0,
                    "max_allocation": 35.0,
                },
                "volatility_breakout": {
                    "max_allocation": 25.0
                }
            },
            "volatility_thresholds": {
                "elevated": 25.0,  # VIX threshold for elevated risk
                "high": 35.0,      # VIX threshold for high risk
                "critical": 45.0   # VIX threshold for critical risk
            },
            "drawdown_thresholds": {
                "elevated": -5.0,   # Portfolio drawdown (%) for elevated risk
                "high": -10.0,      # Portfolio drawdown (%) for high risk
                "critical": -15.0   # Portfolio drawdown (%) for critical risk
            },
            "exposure_scaling": {
                "elevated": 0.75,   # Scale exposure to 75% in elevated risk
                "high": 0.5,        # Scale exposure to 50% in high risk
                "critical": 0.25    # Scale exposure to 25% in critical risk
            }
        }
    
    def check_market_risk(self, market_data: Dict[str, Any]) -> str:
        """
        Check market risk level based on market data.
        
        Args:
            market_data: Dictionary of market metrics
            
        Returns:
            Risk level string: "normal", "elevated", "high", or "critical"
        """
        # Clear existing warnings
        self.risk_warnings = []
        
        # Extract key risk metrics
        vix = market_data.get("vix", {}).get("value", 15.0)
        drawdown = market_data.get("drawdown", {}).get("value", 0.0)
        
        # Check volatility thresholds
        if vix >= self.config["volatility_thresholds"]["critical"]:
            self.risk_warnings.append(f"VIX at critical level: {vix:.1f}")
            risk_level = "critical"
        elif vix >= self.config["volatility_thresholds"]["high"]:
            self.risk_warnings.append(f"VIX at high level: {vix:.1f}")
            risk_level = "high"
        elif vix >= self.config["volatility_thresholds"]["elevated"]:
            self.risk_warnings.append(f"VIX at elevated level: {vix:.1f}")
            risk_level = "elevated"
        else:
            risk_level = "normal"
        
        # Check drawdown thresholds if worse than volatility indicator
        if drawdown <= self.config["drawdown_thresholds"]["critical"]:
            self.risk_warnings.append(f"Portfolio in critical drawdown: {drawdown:.1f}%")
            risk_level = "critical"  # Override with critical
        elif drawdown <= self.config["drawdown_thresholds"]["high"] and risk_level != "critical":
            self.risk_warnings.append(f"Portfolio in high drawdown: {drawdown:.1f}%")
            risk_level = "high"  # Override if not already critical
        elif drawdown <= self.config["drawdown_thresholds"]["elevated"] and risk_level not in ["high", "critical"]:
            self.risk_warnings.append(f"Portfolio in elevated drawdown: {drawdown:.1f}%")
            risk_level = "elevated"  # Override if not already high or critical
        
        # Update current risk level
        self.current_risk_level = risk_level
        
        return risk_level
    
    def check_allocation_constraints(self, 
                                   allocations: Dict[str, float],
                                   previous_allocations: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Check if allocations meet risk constraints.
        
        Args:
            allocations: Strategy allocations (percentages)
            previous_allocations: Previous allocations for change checking
            
        Returns:
            Dictionary with validation results
        """
        warnings = []
        adjusted_allocations = allocations.copy()
        
        # Check each strategy's allocation
        for strategy, allocation in allocations.items():
            # Get constraints for this strategy or default
            strategy_constraints = self.config["strategy_constraints"].get(
                strategy, self.config["strategy_constraints"]["default"]
            )
            
            # Check minimum allocation
            min_allocation = strategy_constraints.get("min_allocation", 0.0)
            if allocation < min_allocation:
                warnings.append(f"{strategy} allocation ({allocation:.1f}%) below minimum ({min_allocation:.1f}%)")
                adjusted_allocations[strategy] = min_allocation
            
            # Check maximum allocation
            max_allocation = strategy_constraints.get("max_allocation", 100.0)
            if allocation > max_allocation:
                warnings.append(f"{strategy} allocation ({allocation:.1f}%) above maximum ({max_allocation:.1f}%)")
                adjusted_allocations[strategy] = max_allocation
                
            # Check maximum change if previous allocations provided
            if previous_allocations and strategy in previous_allocations:
                max_change = strategy_constraints.get("max_change", 100.0)
                previous = previous_allocations[strategy]
                change = abs(allocation - previous)
                
                if change > max_change:
                    warnings.append(f"{strategy} allocation change ({change:.1f}%) exceeds maximum ({max_change:.1f}%)")
                    # Limit the change while preserving direction
                    direction = 1 if allocation > previous else -1
                    adjusted_allocations[strategy] = previous + (direction * max_change)
        
        # Normalize adjusted allocations to sum to 100%
        total = sum(adjusted_allocations.values())
        if total > 0:
            adjusted_allocations = {k: (v / total) * 100 for k, v in adjusted_allocations.items()}
        
        return {
            "warnings": warnings,
            "adjusted_allocations": adjusted_allocations,
            "passed": len(warnings) == 0
        }
    
    def apply_risk_scaling(self, 
                         allocations: Dict[str, float], 
                         risk_level: str) -> Dict[str, float]:
        """
        Apply risk-based scaling to allocations.
        
        Args:
            allocations: Strategy allocations (percentages)
            risk_level: Current risk level
            
        Returns:
            Risk-adjusted allocations
        """
        # If normal risk level, return original allocations
        if risk_level == "normal":
            return allocations.copy()
        
        # Get exposure scaling for risk level
        exposure_scale = self.config["exposure_scaling"].get(risk_level, 1.0)
        
        # Scale risky strategies and increase cash allocation
        adjusted_allocations = {}
        
        for strategy, allocation in allocations.items():
            # Apply scaling
            adjusted_allocations[strategy] = allocation * exposure_scale
        
        # Add cash allocation to make total 100%
        cash_allocation = 100.0 - sum(adjusted_allocations.values())
        adjusted_allocations["cash"] = cash_allocation
        
        return adjusted_allocations


class EnhancedStrategyPrioritizer:
    """
    Advanced AI-driven strategy prioritization system that leverages language models
    to evaluate market conditions and optimize strategy selection with contextual
    awareness, explainability, and risk management.
    """
    
    def __init__(
        self,
        strategies: List[str],
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_model: str = "gpt-4",
        use_mock: bool = False,
        cache_duration: int = 60,
        cache_dir: Optional[str] = None,
        memory_file: Optional[str] = None,
        risk_config: Optional[Dict[str, Any]] = None,
        enable_sentiment_data: bool = True,
        enable_macro_data: bool = True,
        enable_alt_data: bool = False,
        feedback_handlers: Optional[List[Callable]] = None
    ):
        """
        Initialize the EnhancedStrategyPrioritizer.
        
        Args:
            strategies: List of strategy names
            api_key: API key for language model service
            api_base_url: Base URL for API calls
            api_model: Model name to use for API calls
            use_mock: Whether to use mock responses
            cache_duration: How long to cache results (in minutes)
            cache_dir: Directory for disk cache
            memory_file: Path to save/load memory
            risk_config: Risk management configuration
            enable_sentiment_data: Whether to include sentiment data
            enable_macro_data: Whether to include macroeconomic data
            enable_alt_data: Whether to include alternative data
            feedback_handlers: List of callables for handling feedback
        """
        self.strategies = strategies
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base_url = api_base_url or "https://api.openai.com/v1/chat/completions"
        self.api_model = api_model
        self.use_mock = use_mock
        self.cache_duration = cache_duration
        self.cache_dir = cache_dir
        self.enable_sentiment_data = enable_sentiment_data
        self.enable_macro_data = enable_macro_data
        self.enable_alt_data = enable_alt_data
        self.feedback_handlers = feedback_handlers or []
        
        # Set up paths
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            memory_path = memory_file or os.path.join(self.cache_dir, "strategy_memory.json")
        else:
            memory_path = memory_file
        
        # Initialize components
        self.memory = ContextualMemory(max_entries=10, file_path=memory_path)
        self.risk_guardrails = RiskGuardrails(config=risk_config)
        self.market_context_fetcher = MarketContextFetcher()
        self.regime_classifier = RegimeClassifier(lookback_days=60)
        
        # Cache for API responses
        self.cache = {}
        self.cache_timestamp = {}
        
        logger.info(f"EnhancedStrategyPrioritizer initialized with {len(strategies)} strategies")
        logger.info(f"Using {'mock' if use_mock else 'API'} mode with model: {api_model}")
        if self.enable_sentiment_data:
            logger.info("Sentiment data enabled")
        if self.enable_macro_data:
            logger.info("Macroeconomic data enabled")
        if self.enable_alt_data:
            logger.info("Alternative data enabled")
    
    def get_strategy_allocation(
        self, 
        market_context: Optional[Dict[str, Any]] = None,
        previous_allocations: Optional[Dict[str, float]] = None,
        force_refresh: bool = False,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Get recommended strategy allocations with explanations.
        
        Args:
            market_context: Optional market context (fetched if not provided)
            previous_allocations: Previous allocations for change tracking
            force_refresh: Whether to bypass cache
            explain: Whether to include explanation
            
        Returns:
            Dictionary with allocations, explanation, and risk warnings
        """
        # Fetch market context if not provided
        if market_context is None:
            market_context = self._fetch_enriched_market_context()
        
        # Generate a cache key based on key market indicators
        regime = market_context.get("regime", {}).get("primary_regime", "unknown")
        vix = market_context.get("vix", {}).get("value", 0)
        trend = market_context.get("trend_strength", {}).get("value", 0)
        cache_key = f"alloc_{regime}_{int(vix)}_{trend:.2f}"
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Check risk level based on market data
        risk_level = self.risk_guardrails.check_market_risk(market_context)
        
        try:
            # If using mock or API call fails, use mock response
            if self.use_mock:
                result = self._get_mock_allocation(market_context, risk_level, explain)
            else:
                # Try to get allocations from language model
                result = self._get_api_allocation(market_context, risk_level, previous_allocations, explain)
                
                # If API call fails, fall back to mock
                if not result:
                    logger.warning("Failed to get API allocations, falling back to mock")
                    result = self._get_mock_allocation(market_context, risk_level, explain)
        except Exception as e:
            logger.error(f"Error getting strategy allocations: {str(e)}")
            logger.error(traceback.format_exc())
            result = self._get_mock_allocation(market_context, risk_level, explain)
            
        # Apply risk constraints
        allocations = result.get("allocations", {})
        constraint_check = self.risk_guardrails.check_allocation_constraints(
            allocations, previous_allocations
        )
        
        if constraint_check["warnings"]:
            logger.warning(f"Allocation constraints violated: {constraint_check['warnings']}")
            allocations = constraint_check["adjusted_allocations"]
            
            # Add constraint warnings to the result
            result["risk_warnings"] = result.get("risk_warnings", []) + constraint_check["warnings"]
            result["allocations"] = allocations
        
        # Apply risk-based scaling if needed
        if risk_level != "normal":
            original_allocations = allocations.copy()
            risk_scaled_allocations = self.risk_guardrails.apply_risk_scaling(
                allocations, risk_level
            )
            
            risk_warning = f"Applied {risk_level} risk scaling: {self.risk_guardrails.config['exposure_scaling'].get(risk_level, 1.0):.0%} exposure"
            logger.warning(risk_warning)
            
            result["risk_warnings"] = result.get("risk_warnings", []) + [risk_warning]
            result["allocations"] = risk_scaled_allocations
            result["original_allocations"] = original_allocations
        
        # Add risk level to result
        result["risk_level"] = risk_level
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        # Add to memory
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "market_context": {
                "regime": market_context.get("regime", {}).get("primary_regime", "unknown"),
                "vix": market_context.get("vix", {}).get("value", 0),
                "trend": market_context.get("trend_strength", {}).get("value", 0),
                "sentiment": market_context.get("sentiment", {}).get("value", "neutral")
            },
            "allocations": result["allocations"],
            "risk_level": risk_level,
            "explanation": result.get("explanation", "")
        }
        self.memory.add_entry(memory_entry)
        
        return result
    
    def _fetch_enriched_market_context(self) -> Dict[str, Any]:
        """
        Fetch and enrich market context with additional data sources.
        
        Returns:
            Enriched market context dictionary
        """
        # Get base market context
        context = self.market_context_fetcher.get_market_context()
        
        # Add market regime classification
        regime_data = self.regime_classifier.classify_regime()
        context["regime"] = regime_data
        
        # Add multi-timeframe analysis
        context["timeframes"] = {
            "daily": context.get("trend_direction", {}).copy(),
            "weekly": self.market_context_fetcher.get_trend_data(timeframe="weekly"),
            "monthly": self.market_context_fetcher.get_trend_data(timeframe="monthly")
        }
        
        # Add sentiment data if enabled
        if self.enable_sentiment_data:
            context["sentiment"] = self._get_sentiment_data()
            
        # Add macroeconomic data if enabled
        if self.enable_macro_data:
            context["macro"] = self._get_macro_data()
            
        # Add alternative data if enabled
        if self.enable_alt_data:
            context["alt_data"] = self._get_alternative_data()
            
        return context
    
    def _get_sentiment_data(self) -> Dict[str, Any]:
        """
        Get market sentiment data from news and social media.
        
        Returns:
            Dictionary of sentiment metrics
        """
        # This would integrate with sentiment APIs
        # For now, return mock data
        return {
            "value": random.choice(["bearish", "neutral", "bullish"]),
            "score": random.uniform(-1.0, 1.0),
            "news_sentiment": random.uniform(-1.0, 1.0),
            "social_sentiment": random.uniform(-1.0, 1.0),
            "source": "mock"
        }
    
    def _get_macro_data(self) -> Dict[str, Any]:
        """
        Get macroeconomic data metrics.
        
        Returns:
            Dictionary of macroeconomic indicators
        """
        # This would integrate with economic data APIs
        # For now, return mock data
        return {
            "inflation": random.uniform(2.0, 8.0),
            "unemployment": random.uniform(3.0, 6.0),
            "gdp_growth": random.uniform(-2.0, 5.0),
            "interest_rates": random.uniform(0.0, 5.0),
            "source": "mock"
        }
    
    def _get_alternative_data(self) -> Dict[str, Any]:
        """
        Get alternative data metrics.
        
        Returns:
            Dictionary of alternative data indicators
        """
        # This would integrate with alternative data providers
        # For now, return mock data
        return {
            "retail_foot_traffic": random.uniform(-10.0, 10.0),
            "credit_card_spending": random.uniform(-5.0, 15.0),
            "shipping_activity": random.uniform(-20.0, 20.0),
            "source": "mock"
        }
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if we have a valid cached response.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            Cached result or None if not found/valid
        """
        # Try disk cache first if enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    # Check file modification time
                    mtime = os.path.getmtime(cache_file)
                    cache_time = datetime.fromtimestamp(mtime)
                    cache_age = (datetime.now() - cache_time).total_seconds() / 60
                    
                    if cache_age < self.cache_duration:
                        logger.debug(f"Using disk-cached allocations ({cache_age:.1f} min old)")
                        with open(cache_file, 'r') as f:
                            return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}")
        
        # Fall back to in-memory cache
        if cache_key in self.cache:
            cache_time = self.cache_timestamp.get(cache_key)
            if cache_time:
                cache_age = (datetime.now() - cache_time).total_seconds() / 60
                if cache_age < self.cache_duration:
                    logger.debug(f"Using in-memory cached allocations ({cache_age:.1f} min old)")
                    return self.cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Cache a result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Cache in memory
        self.cache[cache_key] = result
        self.cache_timestamp[cache_key] = datetime.now()
        
        # Cache to disk if enabled
        if self.cache_dir:
            try:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.debug(f"Saved allocations to disk cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Error writing to disk cache: {e}")
    
    def _get_api_allocation(
        self, 
        market_context: Dict[str, Any],
        risk_level: str,
        previous_allocations: Optional[Dict[str, float]],
        explain: bool
    ) -> Dict[str, Any]:
        """
        Get strategy allocations from language model API based on market context.
        
        Args:
            market_context: Market context data
            risk_level: Current risk level
            previous_allocations: Previous allocations if available
            explain: Whether to include explanation
            
        Returns:
            Dictionary with allocations and explanation
        """
        try:
            # Create model prompt with all relevant context
            prompt = self._create_prompt(market_context, risk_level, previous_allocations)
            
            # Get recent memory entries
            memory_entries = self.memory.get_recent_entries(limit=3)
            
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": prompt}
            ]
            
            # Add memory entries as context
            if memory_entries:
                memory_content = "Previous allocations and market contexts:\n"
                for entry in memory_entries:
                    date_str = entry.get("timestamp", "").split("T")[0] if "timestamp" in entry else "unknown date"
                    regime = entry.get("market_context", {}).get("regime", "unknown")
                    vix = entry.get("market_context", {}).get("vix", "unknown")
                    allocations_str = ", ".join([f"{k}: {v:.1f}%" for k, v in entry.get("allocations", {}).items()])
                    memory_content += f"- Date: {date_str}, Regime: {regime}, VIX: {vix}\n  Allocations: {allocations_str}\n"
                
                messages.append({"role": "system", "content": memory_content})
            
            # Request message
            messages.append({
                "role": "user", 
                "content": "Based on the current market context and historical data, provide optimal strategy allocations."
            })
            
            # Call API
            logger.info(f"Calling API with model: {self.api_model}")
            response = self._call_llm_api(messages)
            
            if not response:
                logger.error("No response from API")
                return {}
            
            # Parse the response
            parsed_result = self._parse_api_response(response, explain)
            
            if not parsed_result:
                logger.error("Failed to parse API response")
                return {}
                
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _create_prompt(
        self, 
        market_context: Dict[str, Any],
        risk_level: str,
        previous_allocations: Optional[Dict[str, float]]
    ) -> str:
        """
        Create a detailed prompt for the language model.
        
        Args:
            market_context: Market context data
            risk_level: Current risk level
            previous_allocations: Previous allocations if available
            
        Returns:
            Prompt string for the language model
        """
        # Extract key market data
        regime = market_context.get("regime", {}).get("primary_regime", "unknown")
        vix = market_context.get("vix", {}).get("value", 0)
        trend = market_context.get("trend_strength", {}).get("value", 0)
        sentiment = market_context.get("sentiment", {}).get("value", "neutral") if self.enable_sentiment_data else "unknown"
        
        prompt = f"""You are an advanced trading strategy allocator for a sophisticated algorithmic trading system. 
Your task is to analyze market conditions and recommend optimal allocation percentages for different trading strategies.

Available strategies:
{', '.join(self.strategies)}

CURRENT MARKET CONTEXT:
- Primary Market Regime: {regime}
- Risk Level: {risk_level}
- VIX: {vix}
- Trend Strength: {trend}
- Market Sentiment: {sentiment}
"""

        # Add secondary characteristics
        if "secondary_characteristics" in market_context.get("regime", {}):
            traits = market_context["regime"]["secondary_characteristics"]
            prompt += f"- Market Traits: {', '.join(traits)}\n"
            
        # Add multi-timeframe context if available
        if "timeframes" in market_context:
            timeframes = market_context["timeframes"]
            prompt += "\nMULTI-TIMEFRAME ANALYSIS:\n"
            for timeframe, data in timeframes.items():
                direction = data.get("value", 0)
                direction_str = "bullish" if direction > 0.3 else "bearish" if direction < -0.3 else "neutral"
                prompt += f"- {timeframe.capitalize()} Trend: {direction_str} (value: {direction:.2f})\n"
                
        # Add macroeconomic data if available
        if "macro" in market_context:
            macro = market_context["macro"]
            prompt += "\nMACROECONOMIC INDICATORS:\n"
            for indicator, value in macro.items():
                if indicator != "source":
                    prompt += f"- {indicator.replace('_', ' ').title()}: {value:.1f}\n"
        
        # Add previous allocations if available
        if previous_allocations:
            prompt += "\nPREVIOUS ALLOCATIONS:\n"
            for strategy, allocation in previous_allocations.items():
                prompt += f"- {strategy}: {allocation:.1f}%\n"
        
        # Add additional instructions
        prompt += f"""
RISK CONTEXT:
Current risk level is {risk_level.upper()}, with the following risk warnings:
{chr(10).join([f"- {warning}" for warning in self.risk_guardrails.risk_warnings])}

YOUR TASK:
1. Analyze the market context and recommend optimal strategy allocations.
2. Provide percentages for each strategy that sum to 100%.
3. Include a clear chain-of-thought explanation for your allocation decisions.
4. Consider the current market regime, risk level, and previous allocations in your decision.

RESPONSE FORMAT:
Return your response as a valid JSON object with the following structure:
{{
  "allocations": {{
    "strategy1": percentage1,
    "strategy2": percentage2,
    ...
  }},
  "explanation": "Your detailed explanation here",
  "reasoning": [
    "Point 1 about why you made these allocation decisions",
    "Point 2 about specific market conditions influencing your decision",
    ...
  ]
}}

All percentages should be numbers (not strings) between 0 and 100, and they should sum to exactly 100.
Only include the strategies listed above in your allocations.
Make your explanation concise but informative, focusing on the key factors that influenced your decision.
"""

        return prompt
    
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Call the language model API.
        
        Args:
            messages: List of message objects
            
        Returns:
            API response text or None if failed
        """
        if not self.api_key:
            logger.error("No API key provided")
            return None
            
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.api_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.api_base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content")
            
            return content
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return None
    
    def _parse_api_response(self, response: str, explain: bool) -> Dict[str, Any]:
        """
        Parse the API response to extract allocations and explanation.
        
        Args:
            response: API response text
            explain: Whether explanation was requested
            
        Returns:
            Dictionary with allocations and explanation
        """
        try:
            # Try to extract JSON from response
            json_match = response
            
            # Check if response contains markdown code block
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0].strip()
                
            # Load JSON data
            data = json.loads(json_match)
            
            # Extract allocations
            allocations = data.get("allocations", {})
            
            # Ensure all values are floats
            allocations = {k: float(v) for k, v in allocations.items()}
            
            # Ensure all strategies are included
            for strategy in self.strategies:
                if strategy not in allocations:
                    allocations[strategy] = 0.0
                    
            # Normalize to ensure sum is 100%
            total = sum(allocations.values())
            if abs(total - 100.0) > 0.01:  # Allow small rounding errors
                allocations = {k: (v / total) * 100 for k, v in allocations.items()}
            
            # Create result
            result = {
                "allocations": allocations,
                "risk_warnings": self.risk_guardrails.risk_warnings.copy()
            }
            
            # Add explanation if requested
            if explain:
                explanation = data.get("explanation", "")
                reasoning = data.get("reasoning", [])
                
                if explanation:
                    result["explanation"] = explanation
                    
                if reasoning:
                    result["reasoning"] = reasoning
                    
            return result
            
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            logger.error(f"Response: {response}")
            return {}
    
    def _get_mock_allocation(
        self, 
        market_context: Dict[str, Any],
        risk_level: str,
        explain: bool
    ) -> Dict[str, Any]:
        """
        Generate mock strategy allocations based on market context.
        
        Args:
            market_context: Market context data
            risk_level: Current risk level
            explain: Whether to include explanation
            
        Returns:
            Dictionary with allocations and explanation
        """
        # Extract regime
        regime = market_context.get("regime", {}).get("primary_regime", "unknown")
        
        # Mock allocations based on regime
        if regime == "bullish" or regime == "moderately_bullish":
            weights = {
                "momentum": 35,
                "trend_following": 30,
                "breakout_swing": 20,
                "mean_reversion": 5,
                "volatility_breakout": 5,
                "option_spreads": 5
            }
        elif regime == "bearish" or regime == "moderately_bearish":
            weights = {
                "momentum": 5,
                "trend_following": 25,
                "breakout_swing": 10,
                "mean_reversion": 15,
                "volatility_breakout": 20,
                "option_spreads": 25
            }
        elif regime == "volatile":
            weights = {
                "momentum": 5,
                "trend_following": 5,
                "breakout_swing": 10,
                "mean_reversion": 20,
                "volatility_breakout": 35,
                "option_spreads": 25
            }
        else:  # sideways, neutral, unknown
            weights = {
                "momentum": 10,
                "trend_following": 10,
                "breakout_swing": 15,
                "mean_reversion": 40,
                "volatility_breakout": 10,
                "option_spreads": 15
            }
        
        # Add some randomness to weights (±5%)
        for strategy in weights:
            noise = random.uniform(-5, 5)
            weights[strategy] = max(0.5, weights[strategy] + noise)
        
        # Normalize weights to sum to 100
        total_weight = sum(weights.values())
        weights = {s: (w / total_weight) * 100 for s, w in weights.items()}
        
        # Ensure all strategies are included
        allocations = {strategy: 0.0 for strategy in self.strategies}
        for strategy, weight in weights.items():
            if strategy in allocations:
                allocations[strategy] = weight
        
        # Normalize again to ensure sum is 100%
        total = sum(allocations.values())
        if total > 0:
            allocations = {s: (w / total) * 100 for s, w in allocations.items()}
        
        # Create result
        result = {
            "allocations": allocations,
            "risk_warnings": self.risk_guardrails.risk_warnings.copy()
        }
        
        # Add explanation if requested
        if explain:
            explanation = f"Allocations based on {regime} market regime with {risk_level} risk level. "
            
            if regime == "bullish" or regime == "moderately_bullish":
                explanation += "Favoring momentum and trend following strategies in bullish conditions."
            elif regime == "bearish" or regime == "moderately_bearish":
                explanation += "Reducing exposure to momentum strategies and increasing defensive allocations in bearish conditions."
            elif regime == "volatile":
                explanation += "Prioritizing volatility-based strategies and option spreads to capitalize on increased volatility."
            else:
                explanation += "Emphasizing mean reversion strategies in sideways or neutral market conditions."
                
            # Add risk context
            if risk_level != "normal":
                explanation += f" Adjusting allocations for {risk_level} risk conditions."
                
            result["explanation"] = explanation
            
            # Add reasoning points
            reasoning = [
                f"Current market regime is {regime}",
                f"Risk level is {risk_level}",
                f"VIX is at {market_context.get('vix', {}).get('value', 0)}"
            ]
            
            if "secondary_characteristics" in market_context.get("regime", {}):
                traits = market_context["regime"]["secondary_characteristics"]
                if traits:
                    reasoning.append(f"Market showing traits: {', '.join(traits)}")
                    
            result["reasoning"] = reasoning
            
        return result
    
    def record_performance_feedback(
        self, 
        allocations: Dict[str, float],
        performance_metrics: Dict[str, float],
        market_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record performance feedback for continuous learning.
        
        Args:
            allocations: Strategy allocations
            performance_metrics: Performance metrics (returns, sharpe, etc.)
            market_context: Market context when allocations were made
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "allocations": allocations,
            "performance": performance_metrics,
            "market_context": market_context
        }
        
        # Save feedback to memory
        self.memory.add_entry(feedback_entry)
        
        # Call feedback handlers
        for handler in self.feedback_handlers:
            try:
                handler(feedback_entry)
            except Exception as e:
                logger.error(f"Error in feedback handler: {str(e)}")
        
        logger.info("Recorded performance feedback")
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self.cache = {}
        self.cache_timestamp = {}
        
        # Clear disk cache if enabled
        if self.cache_dir:
            for file in os.listdir(self.cache_dir):
                if file.startswith("alloc_") and file.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        logger.warning(f"Error deleting cache file {file}: {e}")
        
        logger.info("Cleared allocation cache")


# Command-line testing
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Available strategies
    strategies = [
        "momentum",
        "trend_following",
        "breakout_swing",
        "mean_reversion",
        "volatility_breakout",
        "option_spreads"
    ]
    
    # Initialize prioritizer
    use_mock = os.getenv("OPENAI_API_KEY") is None
    prioritizer = EnhancedStrategyPrioritizer(
        strategies=strategies,
        use_mock=use_mock,
        enable_sentiment_data=True,
        enable_macro_data=True,
        enable_alt_data=False
    )
    
    print("\nTesting Enhanced Strategy Prioritizer")
    print(f"Using mock data: {use_mock}")
    
    try:
        # Test with current market conditions
        print("\nPrioritizing strategies for current market conditions...")
        result = prioritizer.get_strategy_allocation()
        
        print("\nStrategy Allocations:")
        for strategy, allocation in result["allocations"].items():
            print(f"- {strategy}: {allocation:.1f}%")
        
        if "explanation" in result:
            print("\nExplanation:")
            print(result["explanation"])
            
        if "reasoning" in result:
            print("\nReasoning:")
            for point in result["reasoning"]:
                print(f"- {point}")
                
        if "risk_warnings" in result and result["risk_warnings"]:
            print("\nRisk Warnings:")
            for warning in result["risk_warnings"]:
                print(f"- {warning}")
        
    except Exception as e:
        print(f"Error testing strategy prioritizer: {str(e)}")
        print(traceback.format_exc()) 