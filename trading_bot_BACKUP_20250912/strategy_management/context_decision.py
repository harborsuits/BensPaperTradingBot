import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from trading_bot.market_data.market_context import MarketContext
from trading_bot.core.event_system import EventListener, Event
from trading_bot.strategy_management.strategy_base import Strategy
from trading_bot.decision_framework.decision import Decision, DecisionType, DecisionConfidence

logger = logging.getLogger(__name__)

class ContextualWeightCalculator:
    """Calculates weights for different decision sources based on context"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_weights = config.get("regime_weights", {
            "bull": {"technical": 0.3, "fundamental": 0.4, "sentiment": 0.3},
            "bear": {"technical": 0.4, "fundamental": 0.4, "sentiment": 0.2},
            "volatile": {"technical": 0.5, "fundamental": 0.3, "sentiment": 0.2},
            "stable": {"technical": 0.2, "fundamental": 0.5, "sentiment": 0.3},
            "recovery": {"technical": 0.35, "fundamental": 0.35, "sentiment": 0.3}
        })
        self.volatility_scaling = config.get("volatility_scaling", True)
        self.momentum_scaling = config.get("momentum_scaling", True)
        self.performance_feedback = config.get("performance_feedback", True)
        
        # Track historical performance of different signal types
        self.signal_performance = {
            "technical": [],
            "fundamental": [],
            "sentiment": []
        }
    
    def calculate_weights(self, market_context: MarketContext) -> Dict[str, float]:
        """Calculate weights for different signal types based on market context"""
        regime = market_context.current_regime
        base_weights = self.regime_weights.get(regime, 
                                              {"technical": 0.33, "fundamental": 0.33, "sentiment": 0.34})
        
        weights = base_weights.copy()
        
        # Apply volatility scaling if enabled
        if self.volatility_scaling:
            volatility = market_context.get_metric("volatility", 0.0)
            if volatility > 0.3:  # High volatility
                weights["technical"] = weights.get("technical", 0) * 1.2
                weights["sentiment"] = weights.get("sentiment", 0) * 0.8
            elif volatility < 0.1:  # Low volatility
                weights["technical"] = weights.get("technical", 0) * 0.9
                weights["fundamental"] = weights.get("fundamental", 0) * 1.1
        
        # Apply momentum scaling if enabled
        if self.momentum_scaling:
            momentum = market_context.get_metric("momentum", 0.0)
            if abs(momentum) > 0.7:  # Strong momentum (either direction)
                weights["technical"] = weights.get("technical", 0) * 1.2
            
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
            
        return weights
    
    def update_signal_performance(self, signal_type: str, performance: float):
        """Update the performance tracking for a signal type"""
        if signal_type in self.signal_performance:
            self.signal_performance[signal_type].append(performance)
            # Keep only the last 10 performance metrics
            if len(self.signal_performance[signal_type]) > 10:
                self.signal_performance[signal_type] = self.signal_performance[signal_type][-10:]


class ContextDecisionIntegration(EventListener):
    """System for integrating decisions from multiple strategies based on market context"""
    
    def __init__(self, core_context, config: Dict[str, Any]):
        """
        Initialize the context decision integration system
        
        Args:
            core_context: Core context containing references to other systems
            config: Configuration dictionary with settings for the integration
        """
        self.core_context = core_context
        self.config = config
        self.weight_calculator = ContextualWeightCalculator(config.get("weight_calculator", {}))
        
        # Decision confidence thresholds
        self.min_confidence = config.get("min_confidence", 0.4)
        self.strong_confidence = config.get("strong_confidence", 0.75)
        
        # Agreement thresholds
        self.strong_agreement_threshold = config.get("strong_agreement_threshold", 0.8)
        self.moderate_agreement_threshold = config.get("moderate_agreement_threshold", 0.6)
        
        # Minimum required decisions for integration
        self.min_decisions_required = config.get("min_decisions_required", 2)
        
        # Register for relevant events
        self.register_event_handlers()
        
        self.last_integrated_decision = None
        self.decision_history = []
    
    def register_event_handlers(self):
        """Register handlers for relevant events"""
        event_system = self.core_context.event_system
        
        # Register for events from strategies and market context
        event_system.register_handler("strategy_decision", self.handle_strategy_decision)
        event_system.register_handler("market_context_updated", self.handle_market_context_update)
    
    def handle_strategy_decision(self, event: Event):
        """Handle a new decision from a strategy"""
        # Decisions are processed in batches, just store them for now
        pass
    
    def handle_market_context_update(self, event: Event):
        """Handle an update to the market context"""
        # When market context is updated, process all pending decisions
        self.integrate_decisions()
    
    def integrate_decisions(self) -> Optional[Decision]:
        """
        Integrate decisions from multiple strategies based on current market context
        
        Returns:
            Integrated decision or None if no decision could be made
        """
        # Get current market context
        market_context = self.core_context.market_context
        
        # Get current weights for different signal types
        signal_weights = self.weight_calculator.calculate_weights(market_context)
        
        # Get all pending decisions from strategies
        strategy_manager = self.core_context.strategy_manager
        all_strategies = strategy_manager.get_active_strategies()
        
        # Collect decisions by signal type
        decisions_by_type = {
            "technical": [],
            "fundamental": [],
            "sentiment": []
        }
        
        # Collect all decisions from active strategies
        for strategy in all_strategies:
            latest_decision = strategy.get_latest_decision()
            if latest_decision and latest_decision.confidence >= self.min_confidence:
                signal_type = strategy.signal_type
                decisions_by_type[signal_type].append(latest_decision)
        
        # Check if we have enough decisions
        total_decisions = sum(len(decisions) for decisions in decisions_by_type.values())
        if total_decisions < self.min_decisions_required:
            logger.info(f"Not enough decisions to integrate (have {total_decisions}, need {self.min_decisions_required})")
            return None
        
        # Calculate weighted decision by type
        decision_type_scores = {}
        for decision_type in DecisionType:
            scores = {}
            for signal_type, decisions in decisions_by_type.items():
                # Calculate weighted score for this signal type and decision type
                type_decisions = [d for d in decisions if d.decision_type == decision_type]
                if type_decisions:
                    avg_confidence = sum(d.confidence for d in type_decisions) / len(type_decisions)
                    scores[signal_type] = avg_confidence * signal_weights.get(signal_type, 0)
            
            # Sum the scores across signal types
            if scores:
                decision_type_scores[decision_type] = sum(scores.values())
        
        # Determine the final decision type
        if not decision_type_scores:
            return None
            
        final_decision_type = max(decision_type_scores.items(), key=lambda x: x[1])[0]
        final_confidence = decision_type_scores[final_decision_type]
        
        # Check for strong disagreement which might reduce confidence
        opposing_types = self._get_opposing_decision_types(final_decision_type)
        opposing_score = sum(decision_type_scores.get(t, 0) for t in opposing_types)
        
        # If there's significant opposing signals, reduce confidence
        if opposing_score > final_confidence * 0.7:
            final_confidence *= 0.8
        
        # Create the integrated decision
        integrated_decision = Decision(
            decision_type=final_decision_type,
            confidence=final_confidence,
            timestamp=datetime.now(),
            source="context_integration",
            metadata={
                "signal_weights": signal_weights,
                "decision_type_scores": decision_type_scores,
                "total_decisions_integrated": total_decisions
            }
        )
        
        # Store the decision in history
        self.last_integrated_decision = integrated_decision
        self.decision_history.append(integrated_decision)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        # Emit an event about the integrated decision
        self.core_context.event_system.emit_event(
            Event("integrated_decision", {"decision": integrated_decision})
        )
        
        logger.info(f"Integrated decision: {final_decision_type} with confidence {final_confidence:.2f}")
        return integrated_decision
    
    def _get_opposing_decision_types(self, decision_type: DecisionType) -> List[DecisionType]:
        """Get the decision types that would oppose the given type"""
        if decision_type == DecisionType.BUY:
            return [DecisionType.SELL, DecisionType.STRONG_SELL]
        elif decision_type == DecisionType.STRONG_BUY:
            return [DecisionType.SELL, DecisionType.STRONG_SELL]
        elif decision_type == DecisionType.SELL:
            return [DecisionType.BUY, DecisionType.STRONG_BUY]
        elif decision_type == DecisionType.STRONG_SELL:
            return [DecisionType.BUY, DecisionType.STRONG_BUY]
        else:
            return []
    
    def get_decision_agreement_level(self, decision_type: DecisionType) -> float:
        """
        Calculate the level of agreement among strategies for a given decision type
        
        Args:
            decision_type: The decision type to check agreement for
            
        Returns:
            Agreement level from 0.0 to 1.0
        """
        # Get all recent decisions
        strategy_manager = self.core_context.strategy_manager
        all_strategies = strategy_manager.get_active_strategies()
        all_decisions = [s.get_latest_decision() for s in all_strategies if s.get_latest_decision()]
        
        if not all_decisions:
            return 0.0
        
        # Count decisions of the given type
        matching_decisions = [d for d in all_decisions if d.decision_type == decision_type]
        
        # Calculate agreement level
        agreement = len(matching_decisions) / len(all_decisions)
        return agreement
    
    def get_recent_decision_history(self, limit: int = 10) -> List[Decision]:
        """Get the most recent integrated decisions"""
        return self.decision_history[-limit:]
    
    def get_decision_stability(self, window: int = 5) -> float:
        """
        Calculate decision stability over recent history
        
        Args:
            window: Number of recent decisions to consider
            
        Returns:
            Stability score from 0.0 (unstable) to 1.0 (stable)
        """
        recent_decisions = self.get_recent_decision_history(window)
        if len(recent_decisions) < 2:
            return 1.0  # Not enough history to determine instability
        
        # Count changes in decision direction
        changes = 0
        for i in range(1, len(recent_decisions)):
            prev = recent_decisions[i-1].decision_type
            curr = recent_decisions[i].decision_type
            
            if self._is_direction_change(prev, curr):
                changes += 1
        
        # Calculate stability score
        max_possible_changes = len(recent_decisions) - 1
        stability = 1.0 - (changes / max_possible_changes) if max_possible_changes > 0 else 1.0
        
        return stability
    
    def _is_direction_change(self, prev_type: DecisionType, curr_type: DecisionType) -> bool:
        """Check if there's a directional change between decision types"""
        buy_types = [DecisionType.BUY, DecisionType.STRONG_BUY]
        sell_types = [DecisionType.SELL, DecisionType.STRONG_SELL]
        
        # Check if we moved from buy to sell or vice versa
        if prev_type in buy_types and curr_type in sell_types:
            return True
        if prev_type in sell_types and curr_type in buy_types:
            return True
            
        return False 

class DecisionWeight:
    """Represents a weight for a specific decision factor"""
    
    def __init__(self, name: str, value: float, confidence: float = 1.0):
        self.name = name
        self.value = value          # -1.0 to 1.0 (negative = bearish, positive = bullish)
        self.confidence = confidence  # 0.0 to 1.0
        
    def __repr__(self):
        return f"DecisionWeight({self.name}, value={self.value:.2f}, confidence={self.confidence:.2f})"
        
    def adjust(self, adjustment: float):
        """Adjust the weight value"""
        self.value += adjustment
        self.value = max(-1.0, min(1.0, self.value))
        
    def adjust_confidence(self, adjustment: float):
        """Adjust the confidence value"""
        self.confidence += adjustment
        self.confidence = max(0.0, min(1.0, self.confidence))


class ContextualDecision:
    """Represents a trading decision with contextual information"""
    
    def __init__(self, strategy_id: str, symbol: str, action: str = None):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.action = action  # buy, sell, hold
        self.weights: Dict[str, DecisionWeight] = {}
        self.timestamp = datetime.now()
        self.final_score = 0.0
        self.confidence = 0.0
        self.metadata = {}
        
    def add_weight(self, name: str, value: float, confidence: float = 1.0):
        """Add a decision weight factor"""
        self.weights[name] = DecisionWeight(name, value, confidence)
        
    def calculate_final_score(self):
        """Calculate the final decision score based on weights"""
        if not self.weights:
            self.final_score = 0.0
            self.confidence = 0.0
            return 0.0
            
        # Calculate weighted average of all factors
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for name, weight in self.weights.items():
            adjusted_weight = weight.value * weight.confidence
            weighted_sum += adjusted_weight
            total_weight += weight.confidence
            confidence_sum += weight.confidence
            
        # Normalize score to -1.0 to 1.0 range
        if total_weight > 0:
            self.final_score = weighted_sum / total_weight
        else:
            self.final_score = 0.0
            
        # Calculate overall confidence (average of all confidence values)
        if self.weights:
            self.confidence = confidence_sum / len(self.weights)
        else:
            self.confidence = 0.0
            
        return self.final_score
        
    def determine_action(self, threshold: float = 0.3):
        """Determine action based on final score and threshold"""
        score = self.calculate_final_score()
        
        if score > threshold:
            self.action = "buy"
        elif score < -threshold:
            self.action = "sell"
        else:
            self.action = "hold"
            
        return self.action
        
    def to_dict(self) -> Dict:
        """Convert decision to dictionary format"""
        weights_dict = {name: {"value": w.value, "confidence": w.confidence} 
                        for name, w in self.weights.items()}
        
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "action": self.action,
            "final_score": self.final_score,
            "confidence": self.confidence,
            "weights": weights_dict,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContextualDecision':
        """Create decision from dictionary format"""
        decision = cls(
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            action=data["action"]
        )
        
        decision.final_score = data["final_score"]
        decision.confidence = data["confidence"]
        
        for name, weight_data in data.get("weights", {}).items():
            decision.add_weight(
                name=name,
                value=weight_data["value"],
                confidence=weight_data["confidence"]
            )
            
        decision.timestamp = datetime.fromisoformat(data["timestamp"])
        decision.metadata = data.get("metadata", {})
        
        return decision


class ContextDecisionIntegrator(EventListener):
    """
    Integrates market context and strategy signals to produce enhanced trading decisions
    """
    
    def __init__(self, core_context, config: Dict[str, Any]):
        """
        Initialize the context decision integrator
        
        Args:
            core_context: Core context containing references to other systems
            config: Configuration dictionary with settings
        """
        self.core_context = core_context
        self.config = config
        
        # Configuration parameters
        self.decision_threshold = config.get("decision_threshold", 0.3)
        self.context_weight = config.get("context_weight", 0.5)
        self.max_history = config.get("max_history", 100)
        
        # Track decision history
        self.decision_history: Dict[str, List[ContextualDecision]] = {}
        
        # Track market regime adapters
        self.regime_adapters: Dict[str, Dict[str, float]] = {
            "bull": {"technical": 0.8, "fundamental": 0.5, "sentiment": 0.7},
            "bear": {"technical": 0.6, "fundamental": 0.7, "sentiment": 0.5},
            "volatile": {"technical": 0.5, "fundamental": 0.4, "sentiment": 0.8},
            "stable": {"technical": 0.7, "fundamental": 0.6, "sentiment": 0.5},
            "recovery": {"technical": 0.6, "fundamental": 0.5, "sentiment": 0.6}
        }
        
        # Register for events
        self.register_event_handlers()
        
    def register_event_handlers(self):
        """Register handlers for relevant events"""
        event_system = self.core_context.event_system
        
        # Register for strategy signal events
        event_system.register_handler("strategy_signal", self.handle_strategy_signal)
        event_system.register_handler("market_regime_change", self.handle_regime_change)
        
    def handle_strategy_signal(self, event: Event):
        """
        Handle strategy signal event and integrate with market context
        
        Args:
            event: Strategy signal event with data
        """
        signal_data = event.data
        strategy_id = signal_data.get("strategy_id")
        symbol = signal_data.get("symbol")
        action = signal_data.get("action")
        
        # Skip if missing essential data
        if not all([strategy_id, symbol]):
            logger.warning(f"Skipping incomplete strategy signal: {signal_data}")
            return
            
        # Create contextualized decision
        decision = self.create_contextualized_decision(
            strategy_id, symbol, action, signal_data
        )
        
        # Determine final action
        final_action = decision.determine_action(self.decision_threshold)
        
        # Store in history
        self._store_decision(decision)
        
        # Emit enhanced decision event
        self.core_context.event_system.emit_event(
            Event("enhanced_decision", decision.to_dict())
        )
        
        logger.info(f"Enhanced decision for {strategy_id} on {symbol}: "
                   f"{final_action} (score: {decision.final_score:.2f}, "
                   f"confidence: {decision.confidence:.2f})")
        
    def handle_regime_change(self, event: Event):
        """Handle market regime change event"""
        regime_data = event.data
        new_regime = regime_data.get("new_regime")
        
        logger.info(f"Market regime changed to {new_regime}")
        
        # Adjust weights based on new regime
        if new_regime in self.regime_adapters:
            logger.info(f"Adjusting decision weights for {new_regime} regime")
            
    def create_contextualized_decision(self, 
                                     strategy_id: str, 
                                     symbol: str, 
                                     action: str, 
                                     signal_data: Dict[str, Any]) -> ContextualDecision:
        """
        Create a contextualized decision by integrating strategy signal with market context
        
        Args:
            strategy_id: ID of the strategy
            symbol: Symbol the decision is for
            action: Original action (buy, sell, hold)
            signal_data: Additional signal data
            
        Returns:
            Enhanced decision with context integration
        """
        # Create base decision
        decision = ContextualDecision(strategy_id, symbol, action)
        
        # Add original strategy signal as a weight
        signal_strength = self._convert_action_to_weight(action)
        decision.add_weight("strategy_signal", signal_strength, 1.0)
        
        # Add strategy-specific weights if available
        if "weights" in signal_data:
            for name, value in signal_data["weights"].items():
                if isinstance(value, dict) and "value" in value and "confidence" in value:
                    decision.add_weight(name, value["value"], value["confidence"])
                else:
                    decision.add_weight(name, value, 0.8)  # Default confidence
                    
        # Add metadata
        if "metadata" in signal_data:
            decision.metadata = signal_data["metadata"]
            
        # Get market context
        market_context = self.core_context.market_context
        
        # Add market context weights
        self._add_market_context_weights(decision, market_context, symbol)
        
        # Integrate with historical decisions
        self._integrate_historical_context(decision)
        
        # Apply regime-specific adjustments
        self._apply_regime_adjustments(decision, market_context)
        
        return decision
        
    def _convert_action_to_weight(self, action: str) -> float:
        """Convert action string to numerical weight"""
        action_map = {
            "buy": 1.0,
            "strong_buy": 1.0,
            "weak_buy": 0.5,
            "sell": -1.0,
            "strong_sell": -1.0,
            "weak_sell": -0.5,
            "hold": 0.0,
            None: 0.0
        }
        
        return action_map.get(action.lower() if action else None, 0.0)
        
    def _add_market_context_weights(self, 
                                  decision: ContextualDecision, 
                                  market_context: MarketContext, 
                                  symbol: str):
        """
        Add market context weights to decision
        
        Args:
            decision: Decision to enhance
            market_context: Market context manager
            symbol: Symbol the decision is for
        """
        # Add general market trend
        market_trend = market_context.get_metric("market_trend", 0.0)
        decision.add_weight("market_trend", market_trend, 0.7)
        
        # Add sentiment if available
        if market_context.has_sentiment(symbol):
            sentiment = market_context.get_sentiment(symbol)
            decision.add_weight("market_sentiment", sentiment, 0.6)
            
        # Add volatility as a confidence modifier
        volatility = market_context.get_metric("volatility", 0.5)
        # Higher volatility should reduce confidence in trend following
        if abs(decision.weights.get("strategy_signal", DecisionWeight("", 0)).value) > 0.5:
            # For strong signals, reduce confidence in high volatility
            volatility_impact = -0.3 * (volatility - 0.5)
            decision.weights["strategy_signal"].adjust_confidence(volatility_impact)
            
        # Add sector performance if available
        sector = market_context.get_sector(symbol)
        if sector:
            sector_performance = market_context.get_sector_performance(sector)
            decision.add_weight("sector_trend", sector_performance, 0.6)
            
        # Add trading volume change as a confidence factor
        volume_change = market_context.get_volume_change(symbol)
        if volume_change is not None:
            # High volume increases confidence, low volume decreases it
            normalized_volume = min(1.0, max(-1.0, volume_change / 100))  # Normalize around 0
            for weight_name in decision.weights:
                decision.weights[weight_name].adjust_confidence(normalized_volume * 0.2)
        
    def _integrate_historical_context(self, decision: ContextualDecision):
        """
        Integrate historical decisions to provide stability
        
        Args:
            decision: Current decision to enhance
        """
        symbol = decision.symbol
        strategy_id = decision.strategy_id
        
        # Create key for history lookup
        key = f"{strategy_id}_{symbol}"
        
        if key not in self.decision_history or not self.decision_history[key]:
            return
            
        # Get recent decisions
        recent_decisions = self.decision_history[key][-5:]  # Last 5 decisions
        
        if not recent_decisions:
            return
            
        # Calculate average score from recent decisions
        recent_scores = [d.final_score for d in recent_decisions]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Add historical context as a weight
        # Use lower weight for history to favor new signals
        decision.add_weight("historical_context", avg_score, 0.4)
        
        # Check for trend reversal
        if len(recent_scores) >= 3:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            # If we have a strong trend in recent decisions, factor it in
            if abs(trend) > 0.1:
                decision.add_weight("recent_trend", trend * 2, 0.3)  # Amplify trend signal
        
    def _apply_regime_adjustments(self, 
                                decision: ContextualDecision, 
                                market_context: MarketContext):
        """
        Apply regime-specific adjustments to decision weights
        
        Args:
            decision: Decision to enhance
            market_context: Market context manager
        """
        current_regime = market_context.current_regime
        
        if current_regime not in self.regime_adapters:
            return
            
        regime_weights = self.regime_adapters[current_regime]
        
        # Apply regime-specific weight adjustments
        for weight_name, weight in decision.weights.items():
            # Categorize the weight
            category = self._categorize_weight(weight_name)
            
            if category in regime_weights:
                # Adjust confidence based on how well this category performs in current regime
                adjustment = (regime_weights[category] - 0.5) * 0.4
                weight.adjust_confidence(adjustment)
                
    def _categorize_weight(self, weight_name: str) -> str:
        """Categorize a weight name into technical, fundamental, or sentiment"""
        technical_indicators = ["rsi", "macd", "trend", "momentum", "volume", "breakout", 
                               "support", "resistance", "strategy_signal"]
        fundamental_indicators = ["earnings", "growth", "pe_ratio", "dividend", "sector"]
        sentiment_indicators = ["sentiment", "news", "social"]
        
        weight_lower = weight_name.lower()
        
        for indicator in technical_indicators:
            if indicator in weight_lower:
                return "technical"
                
        for indicator in fundamental_indicators:
            if indicator in weight_lower:
                return "fundamental"
                
        for indicator in sentiment_indicators:
            if indicator in weight_lower:
                return "sentiment"
                
        return "other"
        
    def _store_decision(self, decision: ContextualDecision):
        """Store decision in history"""
        key = f"{decision.strategy_id}_{decision.symbol}"
        
        if key not in self.decision_history:
            self.decision_history[key] = []
            
        self.decision_history[key].append(decision)
        
        # Trim history if needed
        if len(self.decision_history[key]) > self.max_history:
            self.decision_history[key] = self.decision_history[key][-self.max_history:]
            
    def get_decision_history(self, strategy_id: str, symbol: str) -> List[Dict]:
        """
        Get decision history for a strategy and symbol
        
        Args:
            strategy_id: ID of the strategy
            symbol: Symbol to get history for
            
        Returns:
            List of decisions in dictionary format
        """
        key = f"{strategy_id}_{symbol}"
        
        if key not in self.decision_history:
            return []
            
        return [decision.to_dict() for decision in self.decision_history[key]]
        
    def calculate_decision_accuracy(self, 
                                  strategy_id: str, 
                                  symbol: str, 
                                  lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate accuracy of past decisions based on subsequent price movement
        
        Args:
            strategy_id: ID of the strategy
            symbol: Symbol to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with accuracy metrics
        """
        key = f"{strategy_id}_{symbol}"
        
        if key not in self.decision_history:
            return {"accuracy": 0.0, "count": 0}
            
        # Get market data service
        market_data = self.core_context.market_data
        
        # Filter decisions by time
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        decisions = [d for d in self.decision_history[key] 
                    if d.timestamp >= cutoff_time]
        
        if not decisions:
            return {"accuracy": 0.0, "count": 0}
            
        # Analyze each decision
        correct_count = 0
        analyzed_count = 0
        
        for decision in decisions:
            # Skip very recent decisions (need time to evaluate)
            if datetime.now() - decision.timestamp < timedelta(days=1):
                continue
                
            # Get price at decision time and after
            try:
                price_at_decision = market_data.get_price_at_time(
                    symbol, decision.timestamp
                )
                
                # Get price 1 day after
                price_after = market_data.get_price_at_time(
                    symbol, decision.timestamp + timedelta(days=1)
                )
                
                if not price_at_decision or not price_after:
                    continue
                    
                # Calculate returns
                returns = (price_after - price_at_decision) / price_at_decision
                
                # Check if decision was correct
                if (decision.action == "buy" and returns > 0.002) or \
                   (decision.action == "sell" and returns < -0.002) or \
                   (decision.action == "hold" and abs(returns) < 0.005):
                    correct_count += 1
                    
                analyzed_count += 1
                
            except Exception as e:
                logger.error(f"Error analyzing decision accuracy: {e}")
                continue
                
        if analyzed_count == 0:
            return {"accuracy": 0.0, "count": 0}
            
        accuracy = correct_count / analyzed_count
        
        return {
            "accuracy": accuracy,
            "count": analyzed_count,
            "correct": correct_count
        } 