#!/usr/bin/env python3
"""
EvoTester Strategy Configuration
Curated set of high-quality strategy families for evolutionary optimization
"""

STRATEGY_FAMILIES = {
    # Core families with theoretical edge
    "options_income": {
        "name": "Options Income Strategies",
        "description": "Exploit volatility risk premium through premium collection",
        "strategies": [
            "covered_call",
            "cash_secured_put",
            "collar_strategy",
            "poor_mans_covered_call"
        ],
        "theoretical_edge": "Volatility risk premium (options sellers earn theta decay)",
        "cost_sensitivity": "Low (monthly options, low turnover)",
        "implementation_priority": "HIGH"
    },

    "statistical_arbitrage": {
        "name": "Statistical Arbitrage",
        "description": "Exploit pricing inefficiencies between related instruments",
        "strategies": [
            "pairs_trading",
            "basket_arbitrage",
            "cointegration_trading",
            "triangular_arbitrage"
        ],
        "theoretical_edge": "Market inefficiencies and mean reversion",
        "cost_sensitivity": "Medium (requires tight spreads)",
        "implementation_priority": "HIGH"
    },

    "execution_optimization": {
        "name": "Execution Algorithms",
        "description": "Reduce transaction costs through smart order execution",
        "strategies": [
            "vwap_execution",
            "twap_execution",
            "pov_orders",
            "iceberg_orders",
            "time_of_day_patterns"
        ],
        "theoretical_edge": "Reduce market impact and slippage costs",
        "cost_sensitivity": "High (execution quality matters)",
        "implementation_priority": "HIGH"
    },

    "event_driven": {
        "name": "Event-Driven Strategies",
        "description": "Exploit information asymmetries around events",
        "strategies": [
            "earnings_momentum",
            "fed_meeting_trades",
            "economic_data_reaction",
            "merger_arbitrage"
        ],
        "theoretical_edge": "Information processing inefficiencies",
        "cost_sensitivity": "Medium (timing critical)",
        "implementation_priority": "MEDIUM"
    },

    "momentum_multi_timeframe": {
        "name": "Multi-Timeframe Momentum",
        "description": "Momentum across different holding periods",
        "strategies": [
            "time_series_momentum",
            "cross_sectional_momentum",
            "absolute_momentum",
            "dual_momentum"
        ],
        "theoretical_edge": "Behavioral biases and information diffusion",
        "cost_sensitivity": "Low (monthly rebalancing)",
        "implementation_priority": "MEDIUM"
    },

    "volatility_management": {
        "name": "Volatility Management",
        "description": "Dynamic exposure based on volatility regimes",
        "strategies": [
            "vol_targeting",
            "vol_parity",
            "regime_switching",
            "vol_surface_arbitrage"
        ],
        "theoretical_edge": "Volatility risk premium and regime persistence",
        "cost_sensitivity": "Low (infrequent rebalancing)",
        "implementation_priority": "MEDIUM"
    },

    # Flexible framework for experimental strategies
    "experimental_framework": {
        "name": "Experimental Framework",
        "description": "Flexible templates for testing new ideas",
        "strategies": [
            "custom_indicator_combination",
            "ml_signal_enhancement",
            "sentiment_overlay",
            "macro_factor_integration"
        ],
        "theoretical_edge": "Innovation and alpha discovery",
        "cost_sensitivity": "Variable",
        "implementation_priority": "LOW"
    }
}

# Strategy validation checklist
VALIDATION_REQUIREMENTS = {
    "theoretical_foundation": "Must have economic intuition, not just technical patterns",
    "cost_analysis": "Must survive 0.04%+ transaction costs",
    "out_of_sample": "Must perform on unseen data periods",
    "risk_management": "Must have proper position sizing and stop losses",
    "implementation_cost": "Must be feasible to implement at scale"
}

def get_strategy_families():
    """Get all configured strategy families"""
    return STRATEGY_FAMILIES

def get_high_priority_strategies():
    """Get strategies marked as high priority"""
    return [family for family in STRATEGY_FAMILIES.values()
            if family["implementation_priority"] == "HIGH"]

def validate_strategy_family(family_name):
    """Validate if a strategy family meets our criteria"""
    if family_name not in STRATEGY_FAMILIES:
        return False

    family = STRATEGY_FAMILIES[family_name]

    # Check theoretical edge
    if not family.get("theoretical_edge"):
        return False

    # Check cost sensitivity is acceptable
    acceptable_costs = ["Low", "Medium"]
    if family.get("cost_sensitivity") not in acceptable_costs:
        return False

    return True

def get_evolutionary_parameters():
    """Get parameters for evolutionary optimization"""
    return {
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "selection_pressure": 2.0,
        "fitness_function": "sharpe_ratio_after_costs",
        "validation_periods": ["2020-2024", "2015-2019", "2010-2014"]
    }

if __name__ == "__main__":
    print("🎯 EvoTester Strategy Configuration")
    print("=" * 50)

    print(f"Total strategy families: {len(STRATEGY_FAMILIES)}")
    print(f"High priority families: {len(get_high_priority_strategies())}")

    print("\n📊 Strategy Family Summary:")
    for name, family in STRATEGY_FAMILIES.items():
        priority = family["implementation_priority"]
        edge = family["theoretical_edge"][:50] + "..."
        cost = family["cost_sensitivity"]
        print("2")

    print("\n✅ VALIDATION REQUIREMENTS:")
    for req, desc in VALIDATION_REQUIREMENTS.items():
        print(f"   • {req}: {desc}")

    print("\n🎯 FOCUS: Quality over quantity")
    print("   • 7 curated strategy families")
    print("   • Each with theoretical foundation")
    print("   • Rigorous validation requirements")
    print("   • Evolutionary optimization ready")
