"""
CPI Report - Macro Event Definition

Detailed implementation of the CPI (Consumer Price Index) report macro event
with market impacts, strategy adjustments, and analysis framework.
"""

from ..macro_event_definitions import (
    MacroEvent, EventType, MarketImpact, EventImportance, TradingBias,
    ReleaseSchedule, SectorImpact, KeyMetric, KeyMetrics, MarketResponse,
    MarketScenario, StrategyRecommendation, StrategyApproach, PositionManagement,
    StrategyAdjustment, StrategySuite, HistoricalInstance, HistoricalAnalysis,
    RelatedIndicator, SectorReaction, CorrelationAnalysis, AlertTiming,
    PositionSizingAdjustment, AlgorithmicResponse, RiskManagement,
    TradingBotImplementation
)
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class CPIReport(MacroEvent):
    """
    Consumer Price Index Report implementation with comprehensive
    market impact analysis and strategy recommendations.
    """
    
    def __init__(self):
        """Initialize the CPI Report event definition."""
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
        
        # Add release schedule
        self.release_schedule = ReleaseSchedule(
            frequency="monthly",
            typical_time="8:30 AM ET",
            typical_day="second Thursday",
            advance_notice="scheduled on economic calendar 3+ months ahead",
            data_period="previous month"
        )
        
        # Add key metrics to watch
        self.key_metrics = KeyMetrics(
            primary=[
                KeyMetric("headline_cpi", "Year-over-year change in overall CPI", 8),
                KeyMetric("core_cpi", "Year-over-year excluding food and energy", 9),
                KeyMetric("month_over_month", "Sequential change in CPI", 7)
            ],
            secondary=[
                KeyMetric("shelter_component", "Housing costs (biggest component)", 6),
                KeyMetric("service_inflation", "Service sector prices (stickier)", 7),
                KeyMetric("goods_inflation", "Merchandise prices (more volatile)", 5)
            ]
        )
        
        # Define market scenarios
        self.market_reactions = {
            "bullish_scenario": MarketScenario(
                name="bullish",
                primary_catalyst="CPI below forecast, especially core CPI",
                typical_market_response=MarketResponse(
                    equities="rally 0.5-1.5%, technology leadership",
                    bonds="yields drop 5-15 basis points",
                    dollar="weakens",
                    commodities="gold rises, oil mixed",
                    volatility="VIX drops 5-10%",
                    duration_of_effect="typically sustained for 2-3 trading days if significant beat"
                )
            ),
            "bearish_scenario": MarketScenario(
                name="bearish",
                primary_catalyst="CPI above forecast, especially if core exceeds by >0.1%",
                typical_market_response=MarketResponse(
                    equities="sell off 1-2%, defensive sectors outperform",
                    bonds="yields surge 5-20 basis points",
                    dollar="strengthens",
                    commodities="gold drops, oil mixed",
                    volatility="VIX spikes 8-15%",
                    duration_of_effect="can persist for 3-5 trading days if significant miss"
                )
            ),
            "neutral_scenario": MarketScenario(
                name="neutral",
                primary_catalyst="CPI in-line with expectations",
                typical_market_response=MarketResponse(
                    equities="initial volatility then reversion to prior trend",
                    bonds="minimal movement after initial reaction",
                    dollar="muted response",
                    volatility="initial spike then rapid mean reversion",
                    duration_of_effect="typically returns to trend within same session"
                )
            )
        }
        
        # Define strategy adjustments
        self.strategy_adjustments = StrategySuite(
            pre_event=StrategyAdjustment(
                equity_strategy=PositionManagement(
                    primary_action="reduce overall exposure by 15-25% 1-2 days before report",
                    position_management=[
                        "trim positions with highest beta or rate sensitivity",
                        "avoid initiating new large positions 48 hours before report",
                        "consider partial hedges using index positions"
                    ],
                    sector_adjustments=[
                        "reduce technology exposure if inflation has been trending higher",
                        "reduce small caps if concerned about wage inflation component"
                    ]
                ),
                options_strategy=StrategyApproach(
                    primary_approach="favor volatility-based strategies to capitalize on IV expansion",
                    recommended_strategies=[
                        StrategyRecommendation(
                            strategy="long_straddle",
                            implementation="ATM straddles on SPY or sector ETFs 2-3 days before report",
                            timeframe="7-10 DTE to capture event but minimize theta decay",
                            exit_plan="close position before report release or immediately after initial move"
                        ),
                        StrategyRecommendation(
                            strategy="long_put_protection",
                            implementation="purchase OTM puts on key holdings as insurance",
                            timeframe="14-21 DTE to provide adequate coverage",
                            sizing="protect 30-50% of portfolio value"
                        ),
                        StrategyRecommendation(
                            strategy="calendar_spread",
                            implementation="sell front-week, buy back-week at same strike",
                            benefit="capitalize on front-week IV crush post-announcement"
                        )
                    ],
                    volatility_considerations="purchase protection when IV is relatively low compared to prior reports"
                ),
                futures_strategy={
                    "primary_action": "reduce leveraged exposure by 30-40% ahead of release",
                    "position_sizing": "decrease contract count by at least 1/3",
                    "hedging": "consider e-mini or micro contracts as overnight hedges"
                }
            ),
            post_event_bullish=StrategyAdjustment(
                equity_strategy=PositionManagement(
                    primary_action="increase exposure in rate-sensitive sectors",
                    target_sectors=["technology", "consumer_discretionary", "real_estate"],
                    position_management=[
                        "add to high-quality growth positions",
                        "focus on companies with strong pricing power",
                        "avoid defensive sectors likely to underperform in rally"
                    ],
                    timing="initial positions in first 30-60 minutes, add more if rally confirms into afternoon"
                ),
                options_strategy=StrategyApproach(
                    primary_approach="capitalize on falling volatility and directional move",
                    recommended_strategies=[
                        StrategyRecommendation(
                            strategy="short_put_verticals",
                            implementation="sell put spreads on index ETFs or strong sectors",
                            timeframe="21-30 DTE to capture IV contraction",
                            strikes="sell 30-delta puts, buy 15-delta puts"
                        ),
                        StrategyRecommendation(
                            strategy="long_call_verticals",
                            implementation="buy call spreads on strongest sectors",
                            timeframe="14-21 DTE for directional exposure",
                            strikes="buy 40-delta calls, sell 20-delta calls"
                        ),
                        StrategyRecommendation(
                            strategy="ratio_spreads",
                            implementation="for advanced traders, 1:2 or 1:3 call ratio spreads",
                            benefit="capitalize on both direction and volatility collapse"
                        )
                    ]
                ),
                timeframe_adjustment="lengthen swing trade duration from 2-3 days to 5-7 days"
            ),
            post_event_bearish=StrategyAdjustment(
                equity_strategy=PositionManagement(
                    primary_action="defensive positioning and targeted reduction in beta",
                    target_sectors=["utilities", "consumer_staples", "healthcare", "quality_dividend_stocks"],
                    position_management=[
                        "reduce growth stock exposure, especially unprofitable companies",
                        "apply trailing stops to remaining positions",
                        "consider partial short positions in weakest sectors"
                    ],
                    timing="implement defensive moves quickly, within first hour post-report"
                ),
                options_strategy=StrategyApproach(
                    primary_approach="capture downside momentum while managing risk",
                    recommended_strategies=[
                        StrategyRecommendation(
                            strategy="long_put_verticals",
                            implementation="buy put spreads on index ETFs or weak sectors",
                            timeframe="14-21 DTE to capture momentum",
                            strikes="buy 40-delta puts, sell 20-delta puts"
                        ),
                        StrategyRecommendation(
                            strategy="bear_call_spreads",
                            implementation="collect premium while defining risk",
                            timeframe="21-30 DTE to capitalize on elevated IV",
                            strikes="sell 30-delta calls, buy 15-delta calls"
                        ),
                        StrategyRecommendation(
                            strategy="long_vix_calls",
                            implementation="direct volatility exposure if expecting sustained selling",
                            timeframe="7-14 DTE to capture VIX spike"
                        )
                    ]
                ),
                risk_management="reduce position sizes by 25-30% compared to bullish scenarios"
            )
        )
        
        # Add historical analysis
        self.historical_analysis = HistoricalAnalysis(
            recent_market_reactions=[
                HistoricalInstance(
                    date="2024-03-12",
                    actual="3.2% YoY, 0.4% MoM",
                    forecast="3.1% YoY, 0.3% MoM",
                    market_reaction="SPY -1.3%, 10Y yield +8bp, VIX +12%",
                    notes="Core CPI remained sticky, prompting rate cut concerns"
                ),
                HistoricalInstance(
                    date="2024-02-13",
                    actual="3.1% YoY, 0.3% MoM",
                    forecast="2.9% YoY, 0.2% MoM",
                    market_reaction="Initial -1.5% drop, recovered to -0.6%",
                    notes="Higher than expected but market found support at key technical levels"
                ),
                HistoricalInstance(
                    date="2024-01-11",
                    actual="3.4% YoY, 0.3% MoM",
                    forecast="3.2% YoY, 0.2% MoM",
                    market_reaction="SPY -0.5%, growth stocks -1.2%",
                    notes="Smaller reaction due to mixed components in the report"
                )
            ],
            pattern_recognition={
                "bullish_setups": [
                    "Core CPI declining for 3+ consecutive months",
                    "CPI below 3% with services inflation moderating",
                    "Significant beat (>0.2% below forecast) typically leads to multi-day rally"
                ],
                "bearish_setups": [
                    "Core CPI acceleration after period of decline",
                    "Services inflation rising while goods inflation falls",
                    "Sequential MoM acceleration more impactful than YoY miss"
                ]
            }
        )
        
        # Add correlation analysis
        self.correlation_analysis = CorrelationAnalysis(
            related_indicators=[
                RelatedIndicator("PPI", 0.82, lead_time="+1 day", predictive_value="moderate"),
                RelatedIndicator("PCE", 0.91, lead_time="-7 to -10 days", predictive_value="high"),
                RelatedIndicator("Average Hourly Earnings", 0.68, lead_time="+3 to +7 days", predictive_value="moderate")
            ],
            sector_specific_reactions={
                "technology": SectorReaction(
                    correlation_to_surprise=-0.78,
                    outperformance_condition="CPI below 3.0%, core below forecast",
                    underperformance_condition="Core services inflation rising"
                ),
                "financials": SectorReaction(
                    correlation_to_surprise=0.34,
                    outperformance_condition="Higher than expected CPI with steepening yield curve",
                    underperformance_condition="Lower than expected CPI with yield curve flattening"
                ),
                "utilities": SectorReaction(
                    correlation_to_surprise=-0.62,
                    outperformance_condition="CPI significantly above expectations (flight to safety)",
                    underperformance_condition="CPI below expectations (rotation to growth)"
                )
            }
        )
        
        # Add trading bot implementation
        self.trading_bot_implementation = TradingBotImplementation(
            alert_timing=AlertTiming(
                pre_event="48 hours, 24 hours, and 2 hours before release",
                post_event="Immediate analysis within 5 minutes of release"
            ),
            position_sizing_adjustment=PositionSizingAdjustment(
                pre_event="reduce standard position size by 30-40%",
                post_event_high_conviction="increase to 110% of standard size",
                post_event_medium_conviction="standard position size",
                post_event_low_conviction="70% of standard position size"
            ),
            algorithmic_response=AlgorithmicResponse(
                data_parsing="scan news releases for key metrics within 1 second of release",
                initial_reaction="compare to forecast and determine scenario within 5 seconds",
                trade_execution="staggered execution over first 2-5 minutes to avoid slippage",
                confirmation_period="validate direction at 15-minute mark before adding to positions"
            ),
            risk_management=RiskManagement(
                stop_loss_adjustment="widen by 15-20% during first 30 minutes post-report",
                profit_taking="more aggressive scaling out (take 25% at 50% of target)",
                correlation_monitoring="check if sector reactions align with historical patterns"
            )
        )
        
        # Additional notes
        self.notes = "Watch for sector rotation post-CPI with technology leading if inflation falls and staples outperforming if inflation remains elevated. Pay special attention to the shelter component which has been a sticky point in recent reports. Fed Funds futures reaction provides insight into rate implications."
    
    def get_current_scenario(self, event_outcome: Dict[str, Any]) -> Tuple[MarketScenario, TradingBias]:
        """
        Determine the current CPI scenario based on actual data vs expectations.
        
        Args:
            event_outcome: Dictionary with actual CPI data and forecasts
            
        Returns:
            Tuple of (market scenario, trading bias)
        """
        logger.info(f"Analyzing CPI event outcome: {event_outcome}")
        
        # Extract actual values and forecasts
        headline_cpi = event_outcome.get("headline_cpi", 0)
        core_cpi = event_outcome.get("core_cpi", 0)
        mom_cpi = event_outcome.get("month_over_month", 0)
        
        forecast_headline = event_outcome.get("forecast_headline_cpi", 0)
        forecast_core = event_outcome.get("forecast_core_cpi", 0)
        forecast_mom = event_outcome.get("forecast_month_over_month", 0)
        
        # Calculate surprises (actual - forecast)
        headline_surprise = headline_cpi - forecast_headline
        core_surprise = core_cpi - forecast_core
        mom_surprise = mom_cpi - forecast_mom
        
        # Weight the different components (core is most important)
        weighted_surprise = 0.3 * headline_surprise + 0.5 * core_surprise + 0.2 * mom_surprise
        
        # Determine scenario based on weighted surprise
        if weighted_surprise <= -0.1:
            # Significantly below expectations (bullish)
            return (self.market_reactions["bullish_scenario"], TradingBias.BULLISH)
        elif weighted_surprise >= 0.1:
            # Significantly above expectations (bearish)
            return (self.market_reactions["bearish_scenario"], TradingBias.BEARISH)
        else:
            # Close to expectations
            if abs(core_surprise) > abs(headline_surprise) and abs(core_surprise) >= 0.1:
                # Core surprise is more significant than headline
                if core_surprise > 0:
                    return (self.market_reactions["bearish_scenario"], TradingBias.BEARISH)
                else:
                    return (self.market_reactions["bullish_scenario"], TradingBias.BULLISH)
            else:
                # In-line with expectations
                return (self.market_reactions["neutral_scenario"], TradingBias.NEUTRAL)
    
    def analyze_cpi_components(self, components: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze detailed components of the CPI report.
        
        Args:
            components: Dictionary of CPI components and their values
            
        Returns:
            Dict with component analysis
        """
        # Extract key components
        shelter = components.get("shelter", 0)
        services_less_shelter = components.get("services_less_shelter", 0)
        goods = components.get("goods", 0)
        food = components.get("food", 0)
        energy = components.get("energy", 0)
        
        # Analyze stickiness
        sticky_components = []
        if shelter > 0.3:
            sticky_components.append("shelter")
        if services_less_shelter > 0.3:
            sticky_components.append("services_less_shelter")
        
        # Analyze volatile components
        if abs(energy) > 1.0:
            volatile_note = f"Energy showing significant {'increase' if energy > 0 else 'decrease'} at {energy:.1f}%"
        else:
            volatile_note = "Volatile components relatively stable"
        
        # Determine core drivers
        primary_drivers = []
        for component, value in components.items():
            if value > 0.4:
                primary_drivers.append(f"{component} ({value:.1f}%)")
        
        return {
            "sticky_components": sticky_components,
            "volatile_note": volatile_note,
            "primary_drivers": primary_drivers,
            "shelter_trend": "accelerating" if shelter > 0.4 else "moderating" if shelter < 0.2 else "stable",
            "services_vs_goods": "services-driven" if services_less_shelter > goods else "goods-driven",
            "concern_level": "high" if shelter > 0.4 and services_less_shelter > 0.3 else "moderate" if shelter > 0.3 else "low"
        } 