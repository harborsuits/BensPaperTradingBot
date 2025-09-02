"""
Macro Event Definitions Module

This module defines the structure and characteristics of macro economic events
that influence market behavior and trading decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import datetime
import logging

logger = logging.getLogger(__name__)

class EventImportance(Enum):
    """Importance level of economic events."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10

class MarketImpact(Enum):
    """Typical market volatility impact of an event."""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

class EventType(Enum):
    """Types of economic events."""
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    MONETARY_POLICY = "monetary_policy"
    GROWTH = "growth"
    VOLATILITY = "volatility"
    YIELD_CURVE = "yield_curve"
    EARNINGS = "earnings"
    OTHER = "other"

class TradingBias(Enum):
    """Trading bias based on event outcome."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class SectorImpact:
    """Impact of event on market sectors."""
    highest_impact: List[str] = field(default_factory=list)
    medium_impact: List[str] = field(default_factory=list)
    lowest_impact: List[str] = field(default_factory=list)

@dataclass
class ReleaseSchedule:
    """Schedule information for recurring economic events."""
    frequency: str
    typical_time: Optional[str] = None
    typical_day: Optional[str] = None
    advance_notice: Optional[str] = None
    data_period: Optional[str] = None

@dataclass
class KeyMetric:
    """Individual metrics to watch within an economic report."""
    name: str
    description: str
    market_weight: int  # 1-10 scale of importance

@dataclass
class KeyMetrics:
    """Key metrics to focus on within an economic event."""
    primary: List[KeyMetric] = field(default_factory=list)
    secondary: List[KeyMetric] = field(default_factory=list)

@dataclass
class MarketResponse:
    """Typical market response to an event outcome."""
    equities: str
    bonds: Optional[str] = None
    dollar: Optional[str] = None
    commodities: Optional[str] = None
    volatility: Optional[str] = None
    duration_of_effect: Optional[str] = None

@dataclass
class MarketScenario:
    """Possible market scenario based on event outcome."""
    name: str
    primary_catalyst: Any
    typical_market_response: MarketResponse
    description: Optional[str] = None
    ideal_readings: Optional[Dict[str, str]] = None

@dataclass
class StrategyRecommendation:
    """Strategy recommendation for a particular scenario."""
    strategy: str
    implementation: str
    timeframe: Optional[str] = None
    exit_plan: Optional[str] = None
    benefit: Optional[str] = None
    risk_adjustment: Optional[str] = None
    strikes: Optional[str] = None
    sizing: Optional[str] = None

@dataclass
class StrategyApproach:
    """Overall approach for a strategy category."""
    primary_approach: str
    recommended_strategies: List[StrategyRecommendation] = field(default_factory=list)
    key_caution: Optional[str] = None
    volatility_considerations: Optional[str] = None

@dataclass
class PositionManagement:
    """Position management guidance."""
    primary_action: str
    position_management: List[str] = field(default_factory=list)
    sector_adjustments: List[str] = field(default_factory=list)
    target_sectors: Optional[List[str]] = None
    timing: Optional[str] = None

@dataclass
class StrategyAdjustment:
    """Strategy adjustments for different event phases and outcomes."""
    equity_strategy: Optional[PositionManagement] = None
    options_strategy: Optional[StrategyApproach] = None
    futures_strategy: Optional[Dict[str, str]] = None
    fixed_income_strategy: Optional[Dict[str, str]] = None
    bond_strategy: Optional[Dict[str, str]] = None
    timeframe_adjustment: Optional[str] = None
    risk_management: Optional[Dict[str, str]] = None

@dataclass
class StrategySuite:
    """Complete set of strategy adjustments for an event."""
    pre_event: Optional[StrategyAdjustment] = None
    post_event_bullish: Optional[StrategyAdjustment] = None
    post_event_bearish: Optional[StrategyAdjustment] = None
    post_event_neutral: Optional[StrategyAdjustment] = None
    post_event_mixed: Optional[StrategyAdjustment] = None

@dataclass
class HistoricalInstance:
    """Historical instance of an economic event."""
    date: str
    actual: str
    forecast: Optional[str] = None
    market_reaction: str
    notes: Optional[str] = None
    report_type: Optional[str] = None
    decision: Optional[str] = None
    dot_plot: Optional[str] = None

@dataclass
class HistoricalAnalysis:
    """Historical analysis of an economic event."""
    recent_market_reactions: List[HistoricalInstance] = field(default_factory=list)
    pattern_recognition: Optional[Dict[str, List[str]]] = None
    sector_performance_patterns: Optional[Dict[str, List[str]]] = None
    vix_spike_patterns: Optional[List[Dict[str, str]]] = None
    notable_vix_events: Optional[List[Dict[str, Any]]] = None

@dataclass
class RelatedIndicator:
    """Indicator related to the primary economic event."""
    indicator: str
    correlation: float
    lead_time: Optional[str] = None
    predictive_value: Optional[str] = None
    interpretation: Optional[str] = None

@dataclass
class SectorReaction:
    """How a specific sector reacts to the economic event."""
    correlation_to_surprise: Optional[float] = None
    sensitivity_to_inversion: Optional[str] = None 
    outperformance_condition: Optional[str] = None
    underperformance_condition: Optional[str] = None
    specific_factor: Optional[str] = None
    early_warning_signs: Optional[str] = None

@dataclass
class CorrelationAnalysis:
    """Analysis of correlations between the event and other indicators."""
    related_indicators: List[RelatedIndicator] = field(default_factory=list)
    sector_specific_reactions: Dict[str, SectorReaction] = field(default_factory=dict)
    sector_specific_sensitivities: Optional[Dict[str, SectorReaction]] = None

@dataclass
class AlertTiming:
    """When to send alerts for an economic event."""
    pre_event: str
    post_event: Optional[str] = None
    during_event: Optional[str] = None

@dataclass
class PositionSizingAdjustment:
    """How to adjust position sizing around an economic event."""
    pre_event: str
    post_event_high_conviction: str
    post_event_medium_conviction: str
    post_event_low_conviction: str

@dataclass
class AlgorithmicResponse:
    """How the trading bot should respond algorithmically to the event."""
    data_parsing: str
    initial_reaction: Optional[str] = None
    scenario_identification: Optional[str] = None
    trade_execution: str
    confirmation_period: Optional[str] = None
    confirmation_metrics: Optional[str] = None
    adaptive_positioning: Optional[str] = None
    statement_analysis: Optional[str] = None
    press_conference_analysis: Optional[str] = None
    component_importance: Optional[str] = None
    execution_strategy: Optional[str] = None
    cross_validation: Optional[str] = None

@dataclass
class RiskManagement:
    """Risk management adjustments during economic events."""
    stop_loss_adjustment: str
    profit_taking: Optional[str] = None
    correlation_monitoring: Optional[str] = None
    vega_exposure: Optional[str] = None
    correlation_breakdown: Optional[str] = None
    contradictory_signals: Optional[str] = None

@dataclass
class TradingBotImplementation:
    """How the trading bot should implement strategy for this event."""
    alert_timing: AlertTiming
    position_sizing_adjustment: PositionSizingAdjustment
    algorithmic_response: AlgorithmicResponse
    risk_management: RiskManagement
    alert_system: Optional[Dict[str, str]] = None
    automated_adjustments: Optional[Dict[str, str]] = None
    recovery_mode_triggers: Optional[Dict[str, str]] = None
    monitoring_parameters: Optional[Dict[str, str]] = None
    adjustment_timeline: Optional[Dict[str, str]] = None
    automated_screening: Optional[Dict[str, str]] = None
    calendar_management: Optional[Dict[str, str]] = None
    opportunity_scanning: Optional[Dict[str, Any]] = None

@dataclass
class MacroEvent:
    """Base class for macro economic events."""
    event_id: str
    description: str
    importance_level: int  # 1-10 scale
    event_type: EventType
    market_impact: Dict[str, Any]
    key_metrics: Optional[KeyMetrics] = None
    market_reactions: Dict[str, MarketScenario] = field(default_factory=dict)
    strategy_adjustments: StrategySuite = field(default_factory=StrategySuite)
    historical_analysis: Optional[HistoricalAnalysis] = None
    correlation_analysis: Optional[CorrelationAnalysis] = None
    trading_bot_implementation: Optional[TradingBotImplementation] = None
    notes: Optional[str] = None
    
    # Event-specific fields
    release_schedule: Optional[ReleaseSchedule] = None
    trigger_conditions: Optional[Dict[str, Any]] = None
    key_relationships: Optional[Dict[str, Any]] = None
    vix_trading_signals: Optional[Dict[str, List[Dict[str, str]]]] = None
    regime_types: Optional[Dict[str, Dict[str, Any]]] = None
    regime_identification: Optional[Dict[str, Any]] = None
    regime_based_strategy_framework: Optional[Dict[str, Dict[str, Any]]] = None

    def get_current_scenario(self, event_outcome: Dict[str, Any]) -> Tuple[MarketScenario, TradingBias]:
        """
        Determine the current market scenario based on event outcome.
        
        Args:
            event_outcome: Dictionary of actual event metrics and data
            
        Returns:
            Tuple of (market scenario, trading bias)
        """
        # Default implementation - would be overridden in specific event classes
        return (list(self.market_reactions.values())[0], TradingBias.NEUTRAL)
    
    def get_strategy_adjustment(self, phase: str, scenario: MarketScenario) -> StrategyAdjustment:
        """
        Get the appropriate strategy adjustment for the given phase and scenario.
        
        Args:
            phase: Phase of the event (pre_event, post_event)
            scenario: Market scenario
            
        Returns:
            Strategy adjustment recommendations
        """
        if phase == "pre_event":
            return self.strategy_adjustments.pre_event
        
        # Map the scenario to the appropriate post-event strategy
        if scenario.name.lower() == "bullish":
            return self.strategy_adjustments.post_event_bullish
        elif scenario.name.lower() == "bearish":
            return self.strategy_adjustments.post_event_bearish
        elif scenario.name.lower() == "neutral":
            return self.strategy_adjustments.post_event_neutral
        elif scenario.name.lower() == "mixed":
            return self.strategy_adjustments.post_event_mixed
        
        # Default to bullish if no match (this shouldn't happen with proper setup)
        return self.strategy_adjustments.post_event_bullish

# Example of defining a concrete event (just the structure, not complete implementation)
class CPIReport(MacroEvent):
    """Consumer Price Index Report implementation."""
    
    def __init__(self):
        super().__init__(
            event_id="CPI001",
            description="Consumer Price Index - Measures changes in the price level of a weighted average market basket of consumer goods and services",
            importance_level=10,
            event_type=EventType.INFLATION,
            market_impact={
                "typical_volatility": "high",
                "typical_volume_increase": "+65% vs 20-day average",
                "average_spy_move": "±0.8% in first 30 minutes",
                "sector_sensitivity": SectorImpact(
                    highest_impact=["technology", "consumer_discretionary", "real_estate"],
                    medium_impact=["financials", "industrials", "materials"],
                    lowest_impact=["utilities", "consumer_staples", "healthcare"]
                ),
                "after_hours_futures_reaction": "strong and typically maintained into regular session"
            }
        )
        # Additional initialization would go here
    
    def get_current_scenario(self, event_outcome: Dict[str, Any]) -> Tuple[MarketScenario, TradingBias]:
        """
        Determine the current CPI scenario based on actual data vs expectations.
        
        Args:
            event_outcome: Dictionary with actual CPI data and forecasts
            
        Returns:
            Tuple of (market scenario, trading bias)
        """
        # Implementation would analyze the CPI data vs expectations
        # and return the appropriate scenario and bias
        # This is a placeholder - real implementation would have more complex logic
        
        if event_outcome.get("headline_cpi", 0) < event_outcome.get("forecast_headline", 0):
            return (self.market_reactions["bullish_scenario"], TradingBias.BULLISH)
        elif event_outcome.get("headline_cpi", 0) > event_outcome.get("forecast_headline", 0):
            return (self.market_reactions["bearish_scenario"], TradingBias.BEARISH)
        else:
            return (self.market_reactions["neutral_scenario"], TradingBias.NEUTRAL)

# Factory to create and return specific event types
def create_event(event_type: str) -> MacroEvent:
    """
    Factory function to create specific event instances.
    
    Args:
        event_type: Type of economic event to create
        
    Returns:
        Instance of the specified event
    """
    try:
        # Import here to avoid circular imports
        from .event_definitions import get_event
        return get_event(event_type)
    except (ImportError, ValueError):
        # Fallback to basic implementation
        event_map = {
            "cpi": CPIReport,
            # Other event types would be added here
        }
        
        event_class = event_map.get(event_type.lower())
        if not event_class:
            raise ValueError(f"Unknown event type: {event_type}")
        
        return event_class() 