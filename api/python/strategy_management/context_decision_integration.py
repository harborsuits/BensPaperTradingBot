import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set, Callable
from datetime import datetime, timedelta
import json
import os
import time

from trading_bot.core.event_system import EventListener, Event

logger = logging.getLogger(__name__)


class DecisionContext:
    """
    Contains all contextual information needed for making a trading decision,
    including market data, technical indicators, sentiment, etc.
    """
    
    def __init__(self):
        # Market data
        self.market_data = {
            "current_price": None,
            "volume": None,
            "open": None,
            "high": None,
            "low": None,
            "previous_close": None
        }
        
        # Technical indicators
        self.technical_indicators = {}
        
        # Sentiment data
        self.sentiment = {
            "market_sentiment": None,
            "news_sentiment": None,
            "social_sentiment": None
        }
        
        # Risk metrics
        self.risk_metrics = {
            "volatility": None,
            "var": None,  # Value at Risk
            "portfolio_heat": None,
            "market_stress_index": None
        }
        
        # Market regime
        self.market_regime = "normal"
        
        # Trading session info
        self.session_info = {
            "time_of_day": None,
            "day_of_week": None,
            "is_market_open": True,
            "minutes_to_close": None,
            "is_holiday": False
        }
        
        # Position data
        self.positions = {}
        
        # Account data
        self.account = {
            "equity": None,
            "buying_power": None,
            "cash": None
        }
        
        # Recent orders
        self.recent_orders = []
        
        # Strategy state
        self.strategy_state = {}
        
        # Timestamp when context was last updated
        self.last_updated = datetime.now()
        
    def update_market_data(self, data: Dict[str, Any]):
        """Update market data"""
        self.market_data.update(data)
        self.last_updated = datetime.now()
        
    def update_technical_indicators(self, indicators: Dict[str, Any]):
        """Update technical indicators"""
        self.technical_indicators.update(indicators)
        self.last_updated = datetime.now()
        
    def update_sentiment(self, sentiment_data: Dict[str, Any]):
        """Update sentiment data"""
        self.sentiment.update(sentiment_data)
        self.last_updated = datetime.now()
        
    def update_risk_metrics(self, risk_data: Dict[str, Any]):
        """Update risk metrics"""
        self.risk_metrics.update(risk_data)
        self.last_updated = datetime.now()
        
    def update_market_regime(self, regime: str):
        """Update market regime"""
        self.market_regime = regime
        self.last_updated = datetime.now()
        
    def update_session_info(self, session_data: Dict[str, Any]):
        """Update trading session info"""
        self.session_info.update(session_data)
        self.last_updated = datetime.now()
        
    def update_positions(self, positions_data: Dict[str, Any]):
        """Update position data"""
        self.positions = positions_data
        self.last_updated = datetime.now()
        
    def update_account(self, account_data: Dict[str, Any]):
        """Update account data"""
        self.account.update(account_data)
        self.last_updated = datetime.now()
        
    def update_orders(self, orders: List[Dict[str, Any]]):
        """Update recent orders"""
        self.recent_orders = orders
        self.last_updated = datetime.now()
        
    def update_strategy_state(self, state_data: Dict[str, Any]):
        """Update strategy state"""
        self.strategy_state.update(state_data)
        self.last_updated = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "market_data": self.market_data,
            "technical_indicators": self.technical_indicators,
            "sentiment": self.sentiment,
            "risk_metrics": self.risk_metrics,
            "market_regime": self.market_regime,
            "session_info": self.session_info,
            "positions": self.positions,
            "account": self.account,
            "recent_orders": self.recent_orders,
            "strategy_state": self.strategy_state,
            "last_updated": self.last_updated.isoformat()
        }


class DecisionRule:
    """
    Represents a rule for making trading decisions
    """
    
    def __init__(self, 
                rule_id: str,
                name: str,
                description: str,
                condition_fn: Callable[[DecisionContext], bool],
                weight: float = 1.0,
                is_active: bool = True,
                category: str = "general"):
        """
        Initialize a decision rule
        
        Args:
            rule_id: Unique identifier
            name: Human-readable name
            description: Rule description
            condition_fn: Function that evaluates if rule condition is met
            weight: Rule weight in decision making
            is_active: Whether rule is active
            category: Rule category (entry, exit, risk, etc.)
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.condition_fn = condition_fn
        self.weight = weight
        self.is_active = is_active
        self.category = category
        
        # Stats
        self.trigger_count = 0
        self.last_triggered = None
        
    def evaluate(self, context: DecisionContext) -> bool:
        """
        Evaluate if rule condition is met
        
        Args:
            context: Decision context
            
        Returns:
            True if condition is met, False otherwise
        """
        if not self.is_active:
            return False
            
        try:
            result = self.condition_fn(context)
            
            if result:
                self.trigger_count += 1
                self.last_triggered = datetime.now()
                
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating rule {self.rule_id}: {e}")
            return False
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without condition function)"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "is_active": self.is_active,
            "category": self.category,
            "trigger_count": self.trigger_count,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None
        }


class DecisionOutput:
    """
    Represents the output of a decision making process
    """
    
    def __init__(self, 
                 decision_id: str,
                 action: str,
                 confidence: float,
                 timestamp: datetime = None,
                 ticker: str = None,
                 quantity: int = None,
                 price: float = None,
                 triggered_rules: List[str] = None,
                 context_snapshot: Dict[str, Any] = None):
        """
        Initialize decision output
        
        Args:
            decision_id: Unique decision ID
            action: Decision action (buy, sell, hold)
            confidence: Confidence score (0-1)
            timestamp: When decision was made
            ticker: Stock ticker
            quantity: Trade quantity
            price: Target price
            triggered_rules: IDs of rules that triggered
            context_snapshot: Snapshot of decision context
        """
        self.decision_id = decision_id
        self.action = action
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.triggered_rules = triggered_rules or []
        self.context_snapshot = context_snapshot or {}
        
        # Result tracking
        self.executed = False
        self.execution_time = None
        self.execution_result = None
        self.performance_impact = None
        
    def mark_executed(self, 
                     execution_time: datetime,
                     execution_result: Dict[str, Any] = None):
        """
        Mark decision as executed
        
        Args:
            execution_time: When decision was executed
            execution_result: Result of execution
        """
        self.executed = True
        self.execution_time = execution_time
        self.execution_result = execution_result
        
    def set_performance_impact(self, impact: Dict[str, Any]):
        """
        Set performance impact
        
        Args:
            impact: Performance impact metrics
        """
        self.performance_impact = impact
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision_id": self.decision_id,
            "action": self.action,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "quantity": self.quantity,
            "price": self.price,
            "triggered_rules": self.triggered_rules,
            "context_snapshot": self.context_snapshot,
            "executed": self.executed,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            "execution_result": self.execution_result,
            "performance_impact": self.performance_impact
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionOutput':
        """Create from dictionary"""
        decision = cls(
            decision_id=data["decision_id"],
            action=data["action"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ticker=data.get("ticker"),
            quantity=data.get("quantity"),
            price=data.get("price"),
            triggered_rules=data.get("triggered_rules", []),
            context_snapshot=data.get("context_snapshot", {})
        )
        
        decision.executed = data.get("executed", False)
        
        if data.get("execution_time"):
            decision.execution_time = datetime.fromisoformat(data["execution_time"])
            
        decision.execution_result = data.get("execution_result")
        decision.performance_impact = data.get("performance_impact")
        
        return decision


class ContextDecisionIntegration(EventListener):
    """
    System for integrating contextual information into trading decisions
    using rule-based and ML-based approaches
    """
    
    def __init__(self, core_context, config: Dict[str, Any]):
        """
        Initialize context decision integration
        
        Args:
            core_context: Core context containing references to other systems
            config: Configuration dictionary
        """
        self.core_context = core_context
        self.config = config
        
        # Configuration
        self.data_path = config.get("data_path", "data/decision_context")
        self.confidence_threshold = config.get("confidence_threshold", 0.65)
        self.enable_ml_models = config.get("enable_ml_models", True)
        self.decision_history_limit = config.get("decision_history_limit", 1000)
        
        # Internal state
        self.decision_context = DecisionContext()
        self.decision_rules: Dict[str, DecisionRule] = {}
        self.decision_history: List[DecisionOutput] = []
        self.ml_models = {}
        
        # Initialize
        os.makedirs(self.data_path, exist_ok=True)
        
        # Register event handlers
        self.register_event_handlers()
        
        logger.info("Context Decision Integration initialized")
        
    def register_event_handlers(self):
        """Register event handlers"""
        event_system = self.core_context.event_system
        
        event_system.register_handler("market_data_update", self.handle_market_data_update)
        event_system.register_handler("indicator_update", self.handle_indicator_update)
        event_system.register_handler("sentiment_update", self.handle_sentiment_update)
        event_system.register_handler("risk_metrics_update", self.handle_risk_metrics_update)
        event_system.register_handler("market_regime_change", self.handle_market_regime_change)
        event_system.register_handler("session_update", self.handle_session_update)
        event_system.register_handler("positions_update", self.handle_positions_update)
        event_system.register_handler("account_update", self.handle_account_update)
        event_system.register_handler("orders_update", self.handle_orders_update)
        event_system.register_handler("strategy_state_update", self.handle_strategy_state_update)
        event_system.register_handler("decision_executed", self.handle_decision_executed)
        
    def add_decision_rule(self, rule: DecisionRule):
        """
        Add a decision rule
        
        Args:
            rule: Decision rule to add
        """
        self.decision_rules[rule.rule_id] = rule
        logger.info(f"Added decision rule: {rule.rule_id}")
        
    def remove_decision_rule(self, rule_id: str):
        """
        Remove a decision rule
        
        Args:
            rule_id: Rule ID to remove
        """
        if rule_id in self.decision_rules:
            del self.decision_rules[rule_id]
            logger.info(f"Removed decision rule: {rule_id}")
            
    def handle_market_data_update(self, event: Event):
        """
        Handle market data update event
        
        Args:
            event: Market data update event
        """
        self.decision_context.update_market_data(event.data)
        
    def handle_indicator_update(self, event: Event):
        """
        Handle indicator update event
        
        Args:
            event: Indicator update event
        """
        self.decision_context.update_technical_indicators(event.data)
        
    def handle_sentiment_update(self, event: Event):
        """
        Handle sentiment update event
        
        Args:
            event: Sentiment update event
        """
        self.decision_context.update_sentiment(event.data)
        
    def handle_risk_metrics_update(self, event: Event):
        """
        Handle risk metrics update event
        
        Args:
            event: Risk metrics update event
        """
        self.decision_context.update_risk_metrics(event.data)
        
    def handle_market_regime_change(self, event: Event):
        """
        Handle market regime change event
        
        Args:
            event: Market regime change event
        """
        regime = event.data.get("regime")
        if regime:
            self.decision_context.update_market_regime(regime)
        
    def handle_session_update(self, event: Event):
        """
        Handle session update event
        
        Args:
            event: Session update event
        """
        self.decision_context.update_session_info(event.data)
        
    def handle_positions_update(self, event: Event):
        """
        Handle positions update event
        
        Args:
            event: Positions update event
        """
        self.decision_context.update_positions(event.data)
        
    def handle_account_update(self, event: Event):
        """
        Handle account update event
        
        Args:
            event: Account update event
        """
        self.decision_context.update_account(event.data)
        
    def handle_orders_update(self, event: Event):
        """
        Handle orders update event
        
        Args:
            event: Orders update event
        """
        self.decision_context.update_orders(event.data)
        
    def handle_strategy_state_update(self, event: Event):
        """
        Handle strategy state update event
        
        Args:
            event: Strategy state update event
        """
        self.decision_context.update_strategy_state(event.data)
        
    def handle_decision_executed(self, event: Event):
        """
        Handle decision executed event
        
        Args:
            event: Decision executed event
        """
        execution_data = event.data
        decision_id = execution_data.get("decision_id")
        
        if not decision_id:
            return
            
        # Find decision in history
        for decision in self.decision_history:
            if decision.decision_id == decision_id:
                # Mark as executed
                decision.mark_executed(
                    execution_time=datetime.fromisoformat(execution_data.get("execution_time", datetime.now().isoformat())),
                    execution_result=execution_data.get("execution_result")
                )
                
                # Set performance impact if available
                if "performance_impact" in execution_data:
                    decision.set_performance_impact(execution_data["performance_impact"])
                    
                break
        
    def make_decision(self, 
                     ticker: str, 
                     context_override: Dict[str, Any] = None) -> DecisionOutput:
        """
        Make a trading decision for a specific ticker
        
        Args:
            ticker: Stock ticker
            context_override: Override values for decision context
            
        Returns:
            Decision output
        """
        # Create a copy of the current context
        context = DecisionContext()
        
        # Update with current values
        context.__dict__.update(self.decision_context.__dict__)
        
        # Apply overrides
        if context_override:
            for key, value in context_override.items():
                if hasattr(context, key):
                    setattr(context, key, value)
                    
        # Set ticker in context
        context.ticker = ticker
        
        # Evaluate rules
        triggered_rules = []
        rule_confidence = 0.0
        rule_count = 0
        
        for rule_id, rule in self.decision_rules.items():
            if rule.evaluate(context):
                triggered_rules.append(rule_id)
                rule_confidence += rule.weight
                rule_count += 1
                
        # Calculate rule-based confidence
        if rule_count > 0:
            rule_confidence /= sum(rule.weight for rule in self.decision_rules.values() if rule.is_active)
        else:
            rule_confidence = 0.0
            
        # Get ML model confidence if enabled
        ml_confidence = 0.0
        
        if self.enable_ml_models and self.ml_models:
            ml_confidence = self._get_ml_confidence(ticker, context)
            
        # Combine confidences
        if self.enable_ml_models and self.ml_models:
            # 70% rule-based, 30% ML-based
            confidence = rule_confidence * 0.7 + ml_confidence * 0.3
        else:
            # 100% rule-based
            confidence = rule_confidence
            
        # Determine action
        action = "hold"  # Default
        
        if confidence >= self.confidence_threshold:
            # Check if we already have a position
            has_position = ticker in context.positions
            
            if has_position:
                # Potential sell decision
                if confidence > 0.8:  # Higher threshold for selling
                    action = "sell"
            else:
                # Potential buy decision
                action = "buy"
                
        # Create decision output
        decision_id = f"decision_{int(time.time())}_{ticker}"
        
        decision = DecisionOutput(
            decision_id=decision_id,
            action=action,
            confidence=confidence,
            ticker=ticker,
            triggered_rules=triggered_rules,
            context_snapshot=context.to_dict()
        )
        
        # Add to history
        self.decision_history.append(decision)
        
        # Trim history if needed
        if len(self.decision_history) > self.decision_history_limit:
            self.decision_history = self.decision_history[-self.decision_history_limit:]
            
        # Emit decision event
        self.core_context.event_system.emit_event(
            Event("trading_decision_made", decision.to_dict())
        )
        
        return decision
        
    def _get_ml_confidence(self, ticker: str, context: DecisionContext) -> float:
        """
        Get confidence from ML models
        
        Args:
            ticker: Stock ticker
            context: Decision context
            
        Returns:
            ML confidence score
        """
        # If no ML models available
        if not self.ml_models:
            return 0.0
            
        try:
            # Get relevant model for ticker or use default
            model = self.ml_models.get(ticker, self.ml_models.get("default"))
            
            if not model:
                return 0.0
                
            # Prepare features from context
            features = self._extract_features_from_context(context)
            
            # Get prediction
            prediction = model.predict(features)
            
            # Convert to confidence
            if hasattr(model, "predict_proba"):
                # Get probability of positive class
                confidence = model.predict_proba(features)[0][1]
            else:
                # Convert prediction to confidence
                confidence = max(0.0, min(1.0, (prediction[0] + 1) / 2))
                
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting ML confidence: {e}")
            return 0.0
            
    def _extract_features_from_context(self, context: DecisionContext) -> pd.DataFrame:
        """
        Extract features from context for ML models
        
        Args:
            context: Decision context
            
        Returns:
            DataFrame with features
        """
        features = {}
        
        # Market data
        for key, value in context.market_data.items():
            if isinstance(value, (int, float)) and value is not None:
                features[f"market_{key}"] = value
                
        # Technical indicators
        for key, value in context.technical_indicators.items():
            if isinstance(value, (int, float)) and value is not None:
                features[f"indicator_{key}"] = value
                
        # Sentiment
        for key, value in context.sentiment.items():
            if isinstance(value, (int, float)) and value is not None:
                features[f"sentiment_{key}"] = value
                
        # Risk metrics
        for key, value in context.risk_metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                features[f"risk_{key}"] = value
                
        # Market regime (one-hot encoded)
        regime_mapping = {
            "normal": 0,
            "volatile": 1,
            "trending_up": 2,
            "trending_down": 3,
            "crisis": 4
        }
        
        for regime, idx in regime_mapping.items():
            features[f"regime_{regime}"] = 1 if context.market_regime == regime else 0
            
        # Convert to DataFrame
        return pd.DataFrame([features])
        
    def get_decision_history(self, 
                            ticker: str = None,
                            start_time: datetime = None,
                            end_time: datetime = None,
                            action: str = None,
                            executed_only: bool = False) -> List[DecisionOutput]:
        """
        Get decision history with optional filters
        
        Args:
            ticker: Filter by ticker
            start_time: Filter by start time
            end_time: Filter by end time
            action: Filter by action
            executed_only: Only return executed decisions
            
        Returns:
            Filtered decision history
        """
        filtered = self.decision_history.copy()
        
        # Apply filters
        if ticker:
            filtered = [d for d in filtered if d.ticker == ticker]
            
        if start_time:
            filtered = [d for d in filtered if d.timestamp >= start_time]
            
        if end_time:
            filtered = [d for d in filtered if d.timestamp <= end_time]
            
        if action:
            filtered = [d for d in filtered if d.action == action]
            
        if executed_only:
            filtered = [d for d in filtered if d.executed]
            
        return filtered
        
    def save_decision_history(self):
        """Save decision history to disk"""
        history_path = os.path.join(self.data_path, "decision_history.json")
        
        try:
            # Convert to dicts
            history_data = [decision.to_dict() for decision in self.decision_history]
            
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logger.info(f"Saved {len(history_data)} decisions to {history_path}")
                
        except Exception as e:
            logger.error(f"Error saving decision history: {e}")
            
    def load_decision_history(self):
        """Load decision history from disk"""
        history_path = os.path.join(self.data_path, "decision_history.json")
        
        if not os.path.exists(history_path):
            logger.info("No decision history found on disk")
            return
            
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
                
            # Convert to objects
            self.decision_history = [
                DecisionOutput.from_dict(data) for data in history_data
            ]
            
            logger.info(f"Loaded {len(self.decision_history)} decisions from {history_path}")
                
        except Exception as e:
            logger.error(f"Error loading decision history: {e}")
            
    def evaluate_decision_performance(self, 
                                    start_time: datetime = None,
                                    end_time: datetime = None) -> Dict[str, Any]:
        """
        Evaluate performance of past decisions
        
        Args:
            start_time: Start time for evaluation
            end_time: End time for evaluation
            
        Returns:
            Performance metrics
        """
        # Get executed decisions within timeframe
        decisions = self.get_decision_history(
            start_time=start_time,
            end_time=end_time,
            executed_only=True
        )
        
        if not decisions:
            return {
                "decision_count": 0,
                "success_rate": None,
                "avg_profit": None,
                "total_profit": None
            }
            
        # Calculate metrics
        success_count = 0
        total_profit = 0.0
        
        for decision in decisions:
            impact = decision.performance_impact or {}
            
            if impact.get("profitable", False):
                success_count += 1
                
            total_profit += impact.get("profit_amount", 0.0)
            
        # Calculate metrics
        success_rate = success_count / len(decisions) if decisions else 0
        avg_profit = total_profit / len(decisions) if decisions else 0
        
        return {
            "decision_count": len(decisions),
            "success_rate": success_rate,
            "avg_profit": avg_profit,
            "total_profit": total_profit,
            "buy_count": sum(1 for d in decisions if d.action == "buy"),
            "sell_count": sum(1 for d in decisions if d.action == "sell"),
            "hold_count": sum(1 for d in decisions if d.action == "hold"),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "evaluation_time": datetime.now().isoformat()
        }
        
    def generate_decision_report(self):
        """Generate a report of decision performance"""
        # Today's decisions
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_perf = self.evaluate_decision_performance(start_time=today)
        
        # This week's decisions
        week_start = today - timedelta(days=today.weekday())
        week_perf = self.evaluate_decision_performance(start_time=week_start)
        
        # This month's decisions
        month_start = today.replace(day=1)
        month_perf = self.evaluate_decision_performance(start_time=month_start)
        
        # All time
        all_time_perf = self.evaluate_decision_performance()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "daily": today_perf,
            "weekly": week_perf,
            "monthly": month_perf,
            "all_time": all_time_perf,
            "active_rules": len([r for r in self.decision_rules.values() if r.is_active]),
            "total_rules": len(self.decision_rules),
            "current_confidence_threshold": self.confidence_threshold
        }
        
        # Write report to disk
        report_path = os.path.join(self.data_path, "decision_report.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved decision report to {report_path}")
                
        except Exception as e:
            logger.error(f"Error saving decision report: {e}")
            
        # Emit report event
        self.core_context.event_system.emit_event(
            Event("decision_report_generated", report)
        )
        
        return report 