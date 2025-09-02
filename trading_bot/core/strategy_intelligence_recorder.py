"""
Strategy Intelligence Recorder

This module captures decision points and rationale from various components
of the trading system and persists them for use by the Strategy Intelligence
dashboard.

Features:
- Decision capture and recording
- Feedback loops for decision quality scoring
- Pattern recognition for successful decisions
- Decision explanation generation
- Learning from historical decisions

It follows the event-driven architecture pattern established in the system,
subscribing to relevant events and storing the associated data.
"""
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from collections import defaultdict, Counter

from trading_bot.data.persistence import PersistenceManager
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)

# Define a new event type for decision quality measurement
EventType.DECISION_QUALITY_MEASURED = "decision_quality_measured"

# Decision outcome classification
class DecisionOutcome:
    """Classification of decision outcomes for scoring purposes."""
    EXCELLENT = "excellent"  # Significantly better than expected
    GOOD = "good"          # Better than expected
    ACCEPTABLE = "acceptable"  # Meets expectations
    POOR = "poor"          # Below expectations
    VERY_POOR = "very_poor"    # Significantly below expectations

class StrategyIntelligenceRecorder:
    """Records decisions, rationale, and performance metrics for the Strategy Intelligence dashboard."""
    
    def __init__(self, persistence_manager: PersistenceManager, event_bus: EventBus):
        """
        Initialize the Strategy Intelligence Recorder.
        
        Args:
            persistence_manager: The persistence manager for storing data
            event_bus: The event bus for subscribing to events
        """
        self.persistence = persistence_manager
        self.event_bus = event_bus
        self.is_initialized = False
        
        # Decision tracking and scoring
        self.decision_history = {}
        self.strategy_performance = defaultdict(list)  # Track strategy performance over time
        self.pattern_database = defaultdict(Counter)  # Market condition -> decision patterns
        self.decision_explanations = {}  # Track human-readable explanations for decisions
        
        # Configure scoring thresholds
        self.scoring_config = {
            'excellent_threshold': 1.5,    # 50% better than expected
            'good_threshold': 1.1,         # 10% better than expected
            'acceptable_threshold': 0.9,    # Within 10% of expected
            'poor_threshold': 0.7          # More than 30% worse than expected
        }
        
        self.register_event_handlers()
        logger.info("Enhanced Strategy Intelligence Recorder initialized")
    
    def register_event_handlers(self):
        """Register handlers for all relevant events."""
        # Market regime events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self.handle_market_regime_change)
        self.event_bus.subscribe(EventType.MARKET_REGIME_DETECTED, self.handle_market_regime_detection)
        
        # Asset selection events
        self.event_bus.subscribe(EventType.ASSET_CLASS_SELECTED, self.handle_asset_class_selection)
        self.event_bus.subscribe(EventType.SYMBOL_SELECTED, self.handle_symbol_selection)
        self.event_bus.subscribe(EventType.SYMBOL_RANKED, self.handle_symbol_ranking)
        
        # Strategy selection events
        self.event_bus.subscribe(EventType.STRATEGY_SELECTED, self.handle_strategy_selection)
        self.event_bus.subscribe(EventType.STRATEGY_COMPATIBILITY_UPDATED, self.handle_strategy_compatibility_update)
        
        # Performance attribution events
        self.event_bus.subscribe(EventType.PERFORMANCE_ATTRIBUTED, self.handle_performance_attribution)
        self.event_bus.subscribe(EventType.EXECUTION_QUALITY_MEASURED, self.handle_execution_quality)
        
        # Strategy lifecycle events
        self.event_bus.subscribe(EventType.STRATEGY_PARAMETER_ADJUSTED, self.handle_strategy_adaptation)
        self.event_bus.subscribe(EventType.STRATEGY_PROMOTED, self.handle_strategy_adaptation)
        self.event_bus.subscribe(EventType.STRATEGY_RETIRED, self.handle_strategy_adaptation)
        
        # Correlation events
        self.event_bus.subscribe(EventType.CORRELATION_MATRIX_UPDATED, self.handle_correlation_update)
        
        # Risk management events
        self.event_bus.subscribe(EventType.RISK_ALLOCATION_CHANGED, self.handle_risk_allocation)
        self.event_bus.subscribe(EventType.PORTFOLIO_EXPOSURE_UPDATED, self.handle_portfolio_exposure)
        self.event_bus.subscribe(EventType.CORRELATION_RISK_ALERT, self.handle_correlation_risk)
        self.event_bus.subscribe(EventType.DRAWDOWN_THRESHOLD_EXCEEDED, self.handle_drawdown_threshold)
        self.event_bus.subscribe(EventType.RISK_ATTRIBUTION_CALCULATED, self.handle_risk_attribution)
        
        # Decision feedback loop events
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self.handle_signal_generated)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self.handle_order_filled)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.handle_trade_executed)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        
        # Indicator-Sentiment integration events
        self.event_bus.subscribe(EventType.INDICATOR_SENTIMENT_INTEGRATED, self.handle_indicator_sentiment_integrated)
        
        logger.info("Event handlers registered for Strategy Intelligence Recorder")
    
    def handle_market_regime_change(self, event: Event):
        """Handle market regime change events."""
        logger.info(f"Recording market regime change: {event.data}")
        
        # Extract data from event
        current_regime = event.data.get("current_regime")
        confidence = event.data.get("confidence")
        detected_at = event.data.get("timestamp", datetime.now())
        trigger = event.data.get("trigger")
        
        # Load existing regime data
        regime_data = self.persistence.load_strategy_state("market_regime_detector") or {}
        
        # Update current regime
        regime_data["current_regime"] = current_regime
        regime_data["confidence"] = confidence
        regime_data["detected_at"] = str(detected_at)
        
        # Add to history if it doesn't exist
        if "history" not in regime_data:
            regime_data["history"] = []
        
        # Add new regime change to history
        regime_data["history"].append({
            "timestamp": detected_at,
            "regime": current_regime,
            "confidence": confidence,
            "trigger": trigger,
            "duration_days": event.data.get("duration_days", 0)
        })
        
        # Keep only the most recent 20 regime changes
        regime_data["history"] = sorted(regime_data["history"], 
                                       key=lambda x: x["timestamp"] if isinstance(x["timestamp"], datetime) 
                                       else datetime.fromisoformat(str(x["timestamp"])),
                                       reverse=True)[:20]
        
        # Save updated regime data
        self.persistence.save_strategy_state("market_regime_detector", regime_data)
        logger.debug(f"Saved market regime data: {current_regime} with confidence {confidence}")
    
    def handle_market_regime_detection(self, event: Event):
        """Handle market regime detection events with detailed metrics."""
        logger.info(f"Recording market regime detection metrics: {event.data}")
        
        # This event contains more detailed metrics about how the regime was detected
        detection_metrics = event.data.get("detection_metrics", {})
        
        # Save detection metrics for intelligence purposes
        self.persistence.save_strategy_state("market_regime_detection_metrics", detection_metrics)
    
    def handle_asset_class_selection(self, event: Event):
        """Handle asset class selection events."""
        logger.info(f"Recording asset class selection: {event.data}")
        
        asset_classes = event.data.get("asset_classes", [])
        
        if not asset_classes:
            logger.warning("Empty asset classes data received")
            return
        
        # Save asset class selection data
        self.persistence.save_strategy_state("market_analysis", {"asset_classes": asset_classes})
    
    def handle_symbol_selection(self, event: Event):
        """Handle symbol selection events."""
        logger.info(f"Recording symbol selection: {event.data}")
        
        symbols = event.data.get("symbols", [])
        selection_reason = event.data.get("selection_reason", {})
        
        if not symbols:
            logger.warning("Empty symbols data received")
            return
        
        # Save symbol historical performance data
        if "historical_performance" in event.data:
            historical_perf = event.data.get("historical_performance", [])
            self.persistence.save_strategy_state("symbol_historical_performance", 
                                               {"performance": historical_perf})
    
    def handle_symbol_ranking(self, event: Event):
        """Handle symbol ranking events."""
        logger.info(f"Recording symbol ranking data: {event.data}")
        
        rankings = event.data.get("rankings", [])
        
        if not rankings:
            logger.warning("Empty symbol rankings received")
            return
        
        # Save symbol ranking data
        self.persistence.save_strategy_state("symbol_selection", {"rankings": rankings})
    
    def handle_strategy_selection(self, event: Event):
        """Handle strategy selection events."""
        logger.info(f"Recording strategy selection: {event.data}")
        
        strategy_id = event.data.get("strategy_id")
        selection_reason = event.data.get("selection_reason", {})
        regime = event.data.get("market_regime")
        
        if not strategy_id:
            logger.warning("No strategy ID in selection event")
            return
        
        # Load existing strategy selection history
        selection_history = self.persistence.load_strategy_state("strategy_selection_history") or {"history": []}
        
        # Add new selection event
        selection_history["history"].append({
            "timestamp": datetime.now(),
            "strategy_id": strategy_id,
            "market_regime": regime,
            "selection_reason": selection_reason
        })
        
        # Keep only the most recent 50 selections
        selection_history["history"] = selection_history["history"][-50:]
        
        # Save updated history
        self.persistence.save_strategy_state("strategy_selection_history", selection_history)
        
        # If benchmarks are provided, save them too
        if "benchmarks" in event.data:
            self.persistence.save_strategy_state("strategy_benchmarks", 
                                               {"benchmarks": event.data["benchmarks"]})
    
    def handle_strategy_compatibility_update(self, event: Event):
        """Handle strategy compatibility matrix updates."""
        logger.info("Recording strategy compatibility matrix")
        
        compatibility_matrix = event.data.get("matrix")
        
        if compatibility_matrix is None:
            logger.warning("No compatibility matrix in event")
            return
        
        # Save compatibility matrix
        self.persistence.save_strategy_state("strategy_compatibility", {"matrix": compatibility_matrix})
    
    def handle_performance_attribution(self, event: Event):
        """Handle performance attribution events."""
        logger.info(f"Recording performance attribution: {event.data}")
        
        factors = event.data.get("factors", [])
        
        if not factors:
            logger.warning("Empty performance attribution factors")
            return
        
        # Save performance attribution factors
        self.persistence.save_strategy_state("performance_attribution", {"factors": factors})
    
    def handle_execution_quality(self, event: Event):
        """Handle execution quality measurement events."""
        logger.info(f"Recording execution quality metrics: {event.data}")
        
        metrics = event.data.get("metrics", [])
        
        if not metrics:
            logger.warning("Empty execution quality metrics")
            return
        
        # Save execution quality metrics
        self.persistence.save_strategy_state("execution_quality", {"metrics": metrics})
    
    def handle_strategy_adaptation(self, event: Event):
        """Handle strategy adaptation events."""
        logger.info(f"Recording strategy adaptation: {event.data}")
        
        strategy = event.data.get("strategy")
        event_type = event.data.get("event_type")
        description = event.data.get("description")
        impact = event.data.get("impact")
        
        if not strategy or not event_type:
            logger.warning("Incomplete strategy adaptation data")
            return
        
        # Load existing adaptation events
        adaptation_data = self.persistence.load_strategy_state("strategy_adaptation") or {"events": []}
        
        # Create timestamps for timeline visualization
        now = datetime.now()
        start_date = event.data.get("start_date", now)
        end_date = event.data.get("end_date", now + pd.Timedelta(days=event.data.get("duration_days", 7)))
        
        # Add new adaptation event
        adaptation_data["events"].append({
            "timestamp": now,
            "strategy": strategy,
            "event_type": event_type,
            "description": description,
            "impact": impact,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Save updated adaptation events
        self.persistence.save_strategy_state("strategy_adaptation", adaptation_data)
    
    def handle_correlation_update(self, event: Event):
        """Handle correlation matrix update events."""
        logger.info("Recording correlation matrix update")
        
        matrix = event.data.get("matrix")
        
        if matrix is None:
            logger.warning("No correlation matrix in event")
            return
        
        # Save correlation matrix
        self.persistence.save_strategy_state("correlation_matrix", {"matrix": matrix})
    
    def handle_risk_allocation(self, event: Event):
        """Handle risk allocation change events."""
        logger.info(f"Recording risk allocation change: {event.data}")
        
        # Extract data from event
        symbol = event.data.get("symbol")
        position = event.data.get("position")
        risk_score = event.data.get("risk_score")
        action = event.data.get("action")
        reason = event.data.get("reason")
        timestamp = event.data.get("timestamp", datetime.now().isoformat())
        
        # Load existing risk allocation data
        risk_data = self.persistence.load_strategy_state("risk_allocation_history") or {"changes": []}
        
        # Add to history
        risk_data["changes"].append({
            "timestamp": timestamp,
            "symbol": symbol,
            "position": position,
            "risk_score": risk_score,
            "action": action,
            "reason": reason
        })
        
        # Ensure we don't keep too many records
        if len(risk_data["changes"]) > 100:
            risk_data["changes"] = risk_data["changes"][-100:]
        
        # Save to persistence
        self.persistence.save_strategy_state("risk_allocation_history", risk_data)
    
    def handle_portfolio_exposure(self, event: Event):
        """Handle portfolio exposure update events."""
        logger.info(f"Recording portfolio exposure update: {event.data}")
        
        # Extract data from event
        total_risk = event.data.get("total_risk")
        max_risk = event.data.get("max_risk")
        action = event.data.get("action")
        timestamp = event.data.get("timestamp", datetime.now().isoformat())
        
        # Load existing exposure data
        exposure_data = self.persistence.load_strategy_state("portfolio_exposure_history") or {"updates": []}
        
        # Update current exposure
        exposure_data["current_exposure"] = total_risk
        exposure_data["max_allowed_exposure"] = max_risk
        
        # Add to history
        exposure_data["updates"].append({
            "timestamp": timestamp,
            "total_risk": total_risk,
            "max_risk": max_risk,
            "action": action
        })
        
        # Ensure we don't keep too many records
        if len(exposure_data["updates"]) > 100:
            exposure_data["updates"] = exposure_data["updates"][-100:]
        
        # Save to persistence
        self.persistence.save_strategy_state("portfolio_exposure_history", exposure_data)
    
    def handle_correlation_risk(self, event: Event):
        """Handle correlation risk alert events."""
        logger.info(f"Recording correlation risk alert: {event.data}")
        
        # Extract data from event
        symbols = event.data.get("symbols", [])
        correlation = event.data.get("correlation")
        threshold = event.data.get("threshold")
        action = event.data.get("action")
        timestamp = event.data.get("timestamp", datetime.now().isoformat())
        
        # Load existing correlation risk data
        correlation_risk_data = self.persistence.load_strategy_state("correlation_risk_alerts") or {"alerts": []}
        
        # Add to alerts
        correlation_risk_data["alerts"].append({
            "timestamp": timestamp,
            "symbols": symbols,
            "correlation": correlation,
            "threshold": threshold,
            "action": action
        })
        
        # Ensure we don't keep too many records
        if len(correlation_risk_data["alerts"]) > 50:
            correlation_risk_data["alerts"] = correlation_risk_data["alerts"][-50:]
        
        # Save to persistence
        self.persistence.save_strategy_state("correlation_risk_alerts", correlation_risk_data)
    
    def handle_drawdown_threshold(self, event: Event):
        """Handle drawdown threshold exceeded events."""
        logger.info(f"Recording drawdown threshold exceeded: {event.data}")
        
        # Extract data from event
        current_drawdown = event.data.get("current_drawdown")
        threshold = event.data.get("threshold")
        severity = event.data.get("severity")
        action = event.data.get("action")
        timestamp = event.data.get("timestamp", datetime.now().isoformat())
        
        # Load existing drawdown data
        drawdown_data = self.persistence.load_strategy_state("drawdown_history") or {"events": []}
        
        # Update current drawdown
        drawdown_data["current_drawdown"] = current_drawdown
        drawdown_data["threshold"] = threshold
        
        # Add to history
        drawdown_data["events"].append({
            "timestamp": timestamp,
            "drawdown": current_drawdown,
            "threshold": threshold,
            "severity": severity,
            "action": action
        })
        
        # Ensure we don't keep too many records
        if len(drawdown_data["events"]) > 50:
            drawdown_data["events"] = drawdown_data["events"][-50:]
        
    def handle_risk_attribution(self, event: Event):
        """Handle risk attribution calculation events."""
        logger.info(f"Recording risk attribution: {event.data}")
        
        # Update risk attribution state
        if 'attribution' in event.data:
            self.persistence.save_strategy_state("risk_attribution", event.data)
            logger.info("Risk attribution data saved")

    #----------------------------------------------------------------------
    # Enhanced decision scoring and feedback loop methods
    #----------------------------------------------------------------------
    
    def handle_signal_generated(self, event: Event):
        """Record when a trading signal is generated for later evaluation."""
        signal_id = event.data.get('signal_id')
        if not signal_id:
            logger.warning("Signal event missing signal_id, cannot track for scoring")
            return
    # Save updated adaptation events
    self.persistence.save_strategy_state("strategy_adaptation", adaptation_data)
    
def handle_correlation_update(self, event: Event):
    """Handle correlation matrix update events."""
    logger.info("Recording correlation matrix update")
    
    matrix = event.data.get("matrix")
    
    if matrix is None:
        logger.warning("No correlation matrix in event")
        return
    
    # Save correlation matrix
    self.persistence.save_strategy_state("correlation_matrix", {"matrix": matrix})
    
def handle_risk_allocation(self, event: Event):
    """Handle risk allocation change events."""
    logger.info(f"Recording risk allocation change: {event.data}")
    
    # Extract data from event
    symbol = event.data.get("symbol")
    position = event.data.get("position")
    risk_score = event.data.get("risk_score")
    action = event.data.get("action")
    reason = event.data.get("reason")
    timestamp = event.data.get("timestamp", datetime.now().isoformat())
    
    # Load existing risk allocation data
    risk_data = self.persistence.load_strategy_state("risk_allocation_history") or {"changes": []}
    
    # Add to history
    risk_data["changes"].append({
        "timestamp": timestamp,
        "symbol": symbol,
        "position": position,
        "risk_score": risk_score,
        "action": action,
        "reason": reason
    })
    
    # Ensure we don't keep too many records
    if len(risk_data["changes"]) > 100:
        risk_data["changes"] = risk_data["changes"][-100:]
    
    # Save to persistence
    self.persistence.save_strategy_state("risk_allocation_history", risk_data)
    
def handle_portfolio_exposure(self, event: Event):
    """Handle portfolio exposure update events."""
    logger.info(f"Recording portfolio exposure update: {event.data}")
    
    # Extract data from event
    total_risk = event.data.get("total_risk")
    max_risk = event.data.get("max_risk")
    action = event.data.get("action")
    timestamp = event.data.get("timestamp", datetime.now().isoformat())
    
    # Load existing exposure data
    exposure_data = self.persistence.load_strategy_state("portfolio_exposure_history") or {"updates": []}
    
    # Update current exposure
    exposure_data["current_exposure"] = total_risk
    exposure_data["max_allowed_exposure"] = max_risk
    
    # Add to history
    exposure_data["updates"].append({
        "timestamp": timestamp,
        "total_risk": total_risk,
        "max_risk": max_risk,
        "action": action
    })
    
    # Ensure we don't keep too many records
    if len(exposure_data["updates"]) > 100:
        exposure_data["updates"] = exposure_data["updates"][-100:]
    
    # Save to persistence
    self.persistence.save_strategy_state("portfolio_exposure_history", exposure_data)
    
def handle_correlation_risk(self, event: Event):
    """Handle correlation risk alert events."""
    logger.info(f"Recording correlation risk alert: {event.data}")
    
    # Extract data from event
    symbols = event.data.get("symbols", [])
    correlation = event.data.get("correlation")
    threshold = event.data.get("threshold")
    action = event.data.get("action")
    timestamp = event.data.get("timestamp", datetime.now().isoformat())
    
    # Load existing correlation risk data
    correlation_risk_data = self.persistence.load_strategy_state("correlation_risk_alerts") or {"alerts": []}
    
    # Add to alerts
    correlation_risk_data["alerts"].append({
        "timestamp": timestamp,
        "symbols": symbols,
        "correlation": correlation,
        "threshold": threshold,
        "action": action
    })
    
    # Ensure we don't keep too many records
    if len(correlation_risk_data["alerts"]) > 50:
        correlation_risk_data["alerts"] = correlation_risk_data["alerts"][-50:]
    
    # Save to persistence
    self.persistence.save_strategy_state("correlation_risk_alerts", correlation_risk_data)
    
def handle_drawdown_threshold(self, event: Event):
    """Handle drawdown threshold exceeded events."""
    logger.info(f"Recording drawdown threshold exceeded: {event.data}")
    
    # Extract data from event
    current_drawdown = event.data.get("current_drawdown")
    threshold = event.data.get("threshold")
    severity = event.data.get("severity")
    action = event.data.get("action")
    timestamp = event.data.get("timestamp", datetime.now().isoformat())
    
    # Load existing drawdown data
    drawdown_data = self.persistence.load_strategy_state("drawdown_history") or {"events": []}
    
    # Update current drawdown
    drawdown_data["current_drawdown"] = current_drawdown
    drawdown_data["threshold"] = threshold
    
    # Add to history
    drawdown_data["events"].append({
        "timestamp": timestamp,
        "drawdown": current_drawdown,
        "threshold": threshold,
        "severity": severity,
        "action": action
    })
    
    # Ensure we don't keep too many records
    if len(drawdown_data["events"]) > 50:
        drawdown_data["events"] = drawdown_data["events"][-50:]
    
    # Save to persistence
    self.persistence.save_strategy_state("drawdown_history", drawdown_data)
    
def handle_risk_attribution(self, event: Event):
    """Handle risk attribution calculation events."""
    logger.info(f"Recording risk attribution data: {event.data}")
    
    risk_data = event.data.get("risk_attribution", {})
    timestamp = event.data.get("timestamp", datetime.now())
    
    # Store risk attribution in persistence
    self.persistence.save_strategy_state(
        "risk_attribution", 
        {
            "timestamp": timestamp,
            "risk_attribution": risk_data
        }
    )
    
    # Add to history
    attribution_data = self.persistence.load_strategy_state("risk_attribution") or {"history": []}
    attribution_data["history"].append({
        "timestamp": timestamp,
        "risk_attribution": risk_data
    })
    
    # Cap history size at 50 entries
    attribution_data["history"] = attribution_data["history"][-50:]
    
    # Save updated data
    self.persistence.save_strategy_state("risk_attribution", attribution_data)
    
def handle_indicator_sentiment_integrated(self, event: Event):
    """Handle integrated indicator and sentiment analysis events.
    
    This records the combined technical and sentiment analysis data
    for use in the dashboard and strategy evaluation.
    """
    data = event.data
    symbol = data.get('symbol')
    timestamp = data.get('timestamp', datetime.now())
    
    if not symbol:
        logger.warning("Received indicator-sentiment integration event without symbol")
        return
            
    logger.info(f"Recording integrated indicator-sentiment data for {symbol}")
    
    # Extract the integrated analysis data
    integrated_data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "integrated_score": data.get('integrated_score', 0),
        "indicator_contribution": data.get('indicator_contribution', 0),
        "sentiment_contribution": data.get('sentiment_contribution', 0),
        "confidence": data.get('confidence', 0.5),
        "bias": "bullish" if data.get('integrated_score', 0) > 0.2 else 
               "bearish" if data.get('integrated_score', 0) < -0.2 else "neutral",
        "data_sources": {
            "indicators": data.get('indicator_metadata', {}).get('indicators', []),
            "sentiment": data.get('sentiment_metadata', {}).get('source_types', [])
        },
        "indicator_weight": data.get('weights', {}).get('indicator_weight', 0.6),
        "sentiment_weight": data.get('weights', {}).get('sentiment_weight', 0.4),
    }
    
    # Add detailed indicator and sentiment data if available
    if 'normalized_indicators' in data:
        integrated_data['indicator_details'] = data['normalized_indicators']
            
    if 'normalized_sentiment' in data:
        integrated_data['sentiment_details'] = data['normalized_sentiment']
        """
        logger.info("Initializing mock Strategy Intelligence data")
        
        # Check if we've already initialized
        if self.is_initialized:
            return
        
        # Check if we already have real data
        existing_data = self.persistence.load_strategy_state("market_regime_detector")
        if existing_data:
            logger.info("Real data exists, skipping mock data initialization")
            self.is_initialized = True
            return
        
        # Generate and save mock data for all components
        self._initialize_mock_market_regime()
        self._initialize_mock_asset_selection()
        self._initialize_mock_symbol_selection()
        self._initialize_mock_strategy_compatibility()
        self._initialize_mock_performance_attribution()
        self._initialize_mock_execution_quality()
        self._initialize_mock_strategy_adaptation()
        self._initialize_mock_correlation_matrix()
        
        self.is_initialized = True
        logger.info("Mock Strategy Intelligence data initialized")
    
    def _initialize_mock_market_regime(self):
        """Initialize mock market regime data."""
        regimes = ["trending", "ranging", "volatile", "low_volatility"]
        current_regime = np.random.choice(regimes)
        confidence = round(np.random.uniform(0.70, 0.95), 2)
        
        now = datetime.now()
        history = []
        for i in range(5):
            regime = np.random.choice(regimes)
            duration_days = np.random.randint(3, 15)
            end_date = now - pd.Timedelta(days=i*duration_days)
            
            history.append({
                "timestamp": end_date,
                "regime": regime,
                "confidence": round(np.random.uniform(0.6, 0.95), 2),
                "duration_days": duration_days,
                "trigger": np.random.choice(["volatility_spike", "trend_reversal", "consolidation", "breakout"])
            })
        
        regime_data = {
            "current_regime": current_regime,
            "confidence": confidence,
            "detected_at": str(now),
            "history": history
        }
        
        self.persistence.save_strategy_state("market_regime_detector", regime_data)
    
    def _initialize_mock_asset_selection(self):
        """Initialize mock asset selection data."""
        mock_data = [
            {"asset_class": "forex", "opportunity_score": 82, "selection_reason": "High volatility across major pairs with current market regime"}, 
            {"asset_class": "crypto", "opportunity_score": 68, "selection_reason": "Increasing institutional adoption creating new trading opportunities"},
            {"asset_class": "stocks", "opportunity_score": 59, "selection_reason": "Earnings season shows potential for mid-cap opportunities"},
            {"asset_class": "commodities", "opportunity_score": 45, "selection_reason": "Oil market consolidation limits short-term trading potential"},
            {"asset_class": "bonds", "opportunity_score": 31, "selection_reason": "Low volatility environment with minimal short-term opportunities"}
        ]
        
        self.persistence.save_strategy_state("market_analysis", {"asset_classes": mock_data})
    
    def _initialize_mock_symbol_selection(self):
        """Initialize mock symbol selection and ranking data."""
        # Symbol rankings
        mock_rankings = [
            {"symbol": "EUR/USD", "liquidity": 95, "volatility": 72, "spread": 88, "trend_strength": 65, "regime_fit": 80, "total_score": 82, "rank": 1},
            {"symbol": "GBP/USD", "liquidity": 92, "volatility": 78, "spread": 85, "trend_strength": 70, "regime_fit": 75, "total_score": 79, "rank": 2},
            {"symbol": "USD/JPY", "liquidity": 90, "volatility": 65, "spread": 87, "trend_strength": 62, "regime_fit": 72, "total_score": 75, "rank": 3},
            {"symbol": "AUD/USD", "liquidity": 85, "volatility": 80, "spread": 83, "trend_strength": 58, "regime_fit": 68, "total_score": 72, "rank": 4},
            {"symbol": "USD/CAD", "liquidity": 82, "volatility": 68, "spread": 80, "trend_strength": 60, "regime_fit": 65, "total_score": 70, "rank": 5},
            {"symbol": "EUR/JPY", "liquidity": 80, "volatility": 85, "spread": 78, "trend_strength": 75, "regime_fit": 58, "total_score": 69, "rank": 6},
            {"symbol": "USD/CHF", "liquidity": 78, "volatility": 60, "spread": 76, "trend_strength": 55, "regime_fit": 60, "total_score": 65, "rank": 7},
            {"symbol": "NZD/USD", "liquidity": 72, "volatility": 75, "spread": 70, "trend_strength": 50, "regime_fit": 55, "total_score": 62, "rank": 8}
        ]
        
        self.persistence.save_strategy_state("symbol_selection", {"rankings": mock_rankings})
        
        # Historical performance
        market_regimes = ["trending", "ranging", "volatile", "low_volatility"]
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
        
        mock_performance = []
        for symbol in symbols:
            for regime in market_regimes:
                perf = (np.random.random() * 20) - 5
                if regime == "trending" and symbol in ["EUR/USD", "GBP/USD"]:
                    perf += 5
                if regime == "ranging" and symbol in ["USD/JPY", "USD/CAD"]:
                    perf += 3
                
                mock_performance.append({
                    "symbol": symbol,
                    "market_regime": regime,
                    "performance": round(perf, 2)
                })
        
        self.persistence.save_strategy_state("symbol_historical_performance", {"performance": mock_performance})
    
    def _initialize_mock_strategy_compatibility(self):
        """Initialize mock strategy compatibility matrix."""
        strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
        regimes = ["trending", "ranging", "volatile", "low_volatility"]
        
        # Create compatibility matrix with sensible values
        matrix = {strategy: {} for strategy in strategies}
        
        # Trend following works well in trending markets
        matrix["TrendFollowing"] = {
            "trending": 95,
            "ranging": 30,
            "volatile": 45,
            "low_volatility": 50
        }
        
        # Mean reversion works well in ranging markets
        matrix["MeanReversion"] = {
            "trending": 35,
            "ranging": 90,
            "volatile": 40,
            "low_volatility": 60
        }
        
        # Breakout works well in volatile markets
        matrix["Breakout"] = {
            "trending": 60,
            "ranging": 40,
            "volatile": 88,
            "low_volatility": 20
        }
        
        # Momentum works well in trending markets 
        matrix["Momentum"] = {
            "trending": 90,
            "ranging": 35,
            "volatile": 60,
            "low_volatility": 40
        }
        
        # Scalping works across the board but best in volatile markets
        matrix["Scalping"] = {
            "trending": 65,
            "ranging": 70,
            "volatile": 85,
            "low_volatility": 55
        }
        
        self.persistence.save_strategy_state("strategy_compatibility", {"matrix": matrix})
        
        # Performance benchmarks
        mock_data = []
        for strategy in strategies:
            win_rate = np.random.uniform(0.35, 0.65)
            profit_factor = np.random.uniform(1.0, 2.5)
            sharpe = np.random.uniform(0.5, 2.5)
            max_dd = np.random.uniform(0.05, 0.25)
            avg_trade = np.random.uniform(-0.5, 1.5)
            
            mock_data.append({
                "strategy": strategy,
                "regime": np.random.choice(regimes),
                "win_rate": round(win_rate * 100, 1),
                "profit_factor": round(profit_factor, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_dd * 100, 1),
                "avg_trade_pct": round(avg_trade, 2),
                "sample_size": np.random.randint(50, 500)
            })
        
        self.persistence.save_strategy_state("strategy_benchmarks", {"benchmarks": mock_data})
    
    def _initialize_mock_performance_attribution(self):
        """Initialize mock performance attribution data."""
        mock_data = [
            {"factor": "Market Regime Detection", "contribution": 42.5},
            {"factor": "Asset Selection", "contribution": 18.7},
            {"factor": "Position Sizing", "contribution": 15.3},
            {"factor": "Entry Timing", "contribution": 12.2},
            {"factor": "Exit Timing", "contribution": 8.6},
            {"factor": "Execution Quality", "contribution": -2.8},
            {"factor": "Fees & Slippage", "contribution": -5.5},
            {"factor": "Other Factors", "contribution": 11.0}
        ]
        
        self.persistence.save_strategy_state("performance_attribution", {"factors": mock_data})
    
    def _initialize_mock_execution_quality(self):
        """Initialize mock execution quality data."""
        metrics = ["Slippage (pips)", "Spread (pips)", "Latency (ms)", "Rejection Rate (%)", "Fill Ratio (%)"]
        
        mock_data = []
        for metric in metrics:
            if metric == "Slippage (pips)":
                expected = 1.2
                actual = round(expected * (1 + np.random.uniform(-0.3, 0.5)), 1)
            elif metric == "Spread (pips)":
                expected = 1.5
                actual = round(expected * (1 + np.random.uniform(-0.1, 0.2)), 1)
            elif metric == "Latency (ms)":
                expected = 250
                actual = int(expected * (1 + np.random.uniform(-0.1, 0.3)))
            elif metric == "Rejection Rate (%)":
                expected = 0.8
                actual = round(expected * (1 + np.random.uniform(0, 0.5)), 1)
            elif metric == "Fill Ratio (%)":
                expected = 99.2
                actual = round(min(100, expected * (1 + np.random.uniform(-0.02, 0))), 1)
            
            mock_data.append({
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "difference": round(((actual - expected) / expected) * 100, 1) if expected != 0 else 0
            })
        
        self.persistence.save_strategy_state("execution_quality", {"metrics": mock_data})
    
    def _initialize_mock_strategy_adaptation(self):
        """Initialize mock strategy adaptation data."""
        now = datetime.now()
        strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
        event_types = ["Parameter Adjustment", "Regime Change Response", "Risk Adjustment", "New Feature", "Filter Added"]
        impacts = ["Positive", "Neutral", "Negative"]
        
        mock_data = []
        for i in range(15):
            days_ago = np.random.randint(1, 60)
            timestamp = now - pd.Timedelta(days=days_ago)
            strategy = np.random.choice(strategies)
            event_type = np.random.choice(event_types)
            
            if event_type == "Parameter Adjustment":
                desc = f"Adjusted {strategy} lookback period from {np.random.randint(10, 30)} to {np.random.randint(10, 30)} days"
            elif event_type == "Regime Change Response":
                desc = f"Switched {strategy} to {np.random.choice(['conservative', 'aggressive', 'neutral'])} mode due to regime change"
            elif event_type == "Risk Adjustment":
                desc = f"Modified {strategy} risk per trade from {np.random.uniform(0.5, 2.0):.1f}% to {np.random.uniform(0.5, 2.0):.1f}%"
            elif event_type == "New Feature":
                desc = f"Added {np.random.choice(['volatility filter', 'volume confirmation', 'news filter', 'correlation check'])} to {strategy}"
            elif event_type == "Filter Added":
                desc = f"Implemented {np.random.choice(['ADX filter', 'RSI filter', 'Bollinger filter', 'MA crossover filter'])} to {strategy}"
            
            impact = np.random.choice(impacts, p=[0.6, 0.3, 0.1])
            impact_value = np.random.uniform(-5, 15) if impact == "Positive" else \
                          np.random.uniform(-2, 2) if impact == "Neutral" else \
                          np.random.uniform(-12, -1)
            
            start_date = timestamp
            end_date = timestamp + pd.Timedelta(days=np.random.randint(1, 15))
            
            mock_data.append({
                "timestamp": timestamp,
                "strategy": strategy,
                "event_type": event_type,
                "description": desc,
                "impact": f"{impact} ({impact_value:.1f}%)",
                "start_date": start_date,
                "end_date": end_date
            })
        
        self.persistence.save_strategy_state("strategy_adaptation", {"events": mock_data})
    
    def _initialize_mock_correlation_matrix(self):
        """Initialize mock correlation matrix data."""
        assets = ["EUR/USD", "BTC/USD", "ETH/USD", "USD/JPY", "GBP/USD"]
        mock_matrix = {}
        
        for i, asset1 in enumerate(assets):
            mock_matrix[asset1] = {}
            for j, asset2 in enumerate(assets):
                if i == j:
                    mock_matrix[asset1][asset2] = 1.0
                else:
                    # Generate a correlation value between -1 and 1
                    # Make it symmetric for validity
                    if asset2 in mock_matrix and asset1 in mock_matrix[asset2]:
                        mock_matrix[asset1][asset2] = mock_matrix[asset2][asset1]
                    else:
                        mock_matrix[asset1][asset2] = round(np.random.uniform(-0.9, 0.9), 2)
        
        self.persistence.save_strategy_state("correlation_matrix", {"matrix": mock_matrix})
