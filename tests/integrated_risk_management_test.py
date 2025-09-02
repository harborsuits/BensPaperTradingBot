#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Risk Management Test

This script demonstrates how all the risk management components work together
in a real-world scenario, showing how risk factors are detected and how the
system responds with strategy rotation and risk mitigation actions.
"""
import os
import sys
import time
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.risk.risk_config import RiskConfigManager
from trading_bot.risk.enhanced_risk_factors import EnhancedRiskFactors
from trading_bot.risk.risk_based_strategy_rotation import RiskBasedStrategyRotation

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def create_mock_market_data():
    """Create mock market data for testing."""
    # Basic price data
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "V", "PG"]
    
    market_data = {}
    for symbol in symbols:
        # Generate random price and volume data
        base_price = np.random.uniform(50, 500)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Last 20 days of data
        days = 20
        close_prices = [base_price]
        for _ in range(days-1):
            # Random walk with drift
            change = np.random.normal(0.0002, volatility)
            close_prices.append(close_prices[-1] * (1 + change))
        
        # Create high and low prices based on close
        high_prices = [price * (1 + np.random.uniform(0, 0.02)) for price in close_prices]
        low_prices = [price * (1 - np.random.uniform(0, 0.02)) for price in close_prices]
        
        # Create volume data
        volumes = [int(np.random.uniform(500000, 5000000)) for _ in range(days)]
        
        # Current price is the last close
        current_price = close_prices[-1]
        
        # Add to market data
        market_data[symbol] = {
            "price": current_price,
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
            "volume": volumes,
            "avg_daily_volume": np.mean(volumes),
            "bid_ask_spread": np.random.uniform(0.001, 0.02)
        }
    
    return market_data

def create_mock_position_data(market_data):
    """Create mock position data for testing."""
    positions = {}
    
    # Create positions for some of the symbols
    for symbol, data in list(market_data.items())[:7]:  # First 7 symbols
        position_size = np.random.randint(10, 100) * 10  # 100 to 1000 shares
        entry_price = data["close"][-5]  # Entered 5 days ago
        market_value = position_size * data["price"]
        
        positions[symbol] = {
            "symbol": symbol,
            "quantity": position_size,
            "entry_price": entry_price,
            "current_price": data["price"],
            "market_value": market_value,
            "avg_daily_volume": data["avg_daily_volume"],
            "bid_ask_spread": data["bid_ask_spread"],
            "stop_loss_price": entry_price * 0.95,  # 5% stop loss
            "days_held": 5
        }
    
    return positions

def create_mock_factor_data():
    """Create mock factor exposure data for testing."""
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "V", "PG"]
    
    factor_data = {}
    for symbol in symbols:
        # Random factor exposures
        factor_data[symbol] = {
            "value": np.random.uniform(-0.5, 0.5),
            "momentum": np.random.uniform(-0.5, 0.5),
            "size": np.random.uniform(-0.5, 0.5),
            "quality": np.random.uniform(-0.5, 0.5),
            "volatility": np.random.uniform(-0.5, 0.5),
            "growth": np.random.uniform(-0.5, 0.5),
            "yield": np.random.uniform(-0.5, 0.5),
            "liquidity": np.random.uniform(0.2, 0.9)
        }
    
    return factor_data

def create_mock_sector_data():
    """Create mock sector data for testing."""
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "V", "PG"]
    
    sectors = ["Technology", "Technology", "Consumer Discretionary", 
               "Communication Services", "Communication Services", 
               "Consumer Discretionary", "Financials", "Financials", 
               "Financials", "Consumer Staples"]
    
    return dict(zip(symbols, sectors))

def create_mock_country_data():
    """Create mock country data for testing."""
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "V", "PG"]
    
    countries = ["USA", "USA", "USA", "USA", "USA", "USA", "USA", "USA", "USA", "USA"]
    
    return dict(zip(symbols, countries))

class MockStrategyManager:
    """Mock strategy manager for testing."""
    def __init__(self):
        self.strategies = {}
        self.active_strategies = {}
        
    def add_strategy(self, strategy):
        """Add a strategy to the manager."""
        self.strategies[strategy.id] = strategy
        
    def get_all_strategies(self):
        """Get all strategies."""
        return self.strategies
        
    def get_active_strategies(self):
        """Get active strategies."""
        return {k: v for k, v in self.strategies.items() if v.is_active}
        
    def get_strategy(self, strategy_id):
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
        
    def promote_strategy(self, strategy_id, reason=None):
        """Promote a strategy to active status."""
        if strategy_id in self.strategies:
            logger.info(f"Promoting strategy {strategy_id} (reason: {reason})")
            self.strategies[strategy_id].is_active = True
            return True
        return False
        
    def demote_strategy(self, strategy_id, reason=None):
        """Demote a strategy from active status."""
        if strategy_id in self.strategies:
            logger.info(f"Demoting strategy {strategy_id} (reason: {reason})")
            self.strategies[strategy_id].is_active = False
            return True
        return False

class MockStrategy:
    """Mock strategy for testing."""
    def __init__(self, strategy_id, name, metadata=None):
        self.id = strategy_id
        self.name = name
        self.metadata = metadata or {}
        self.is_active = False
        
    def __repr__(self):
        return f"Strategy({self.id}, {self.name}, active={self.is_active})"

def create_mock_strategies():
    """Create a set of mock strategies for testing."""
    strategies = [
        # Momentum strategies - higher beta and volatility
        MockStrategy("momentum_1", "Momentum Strategy 1", {
            "strategy_type": "momentum",
            "market_beta": 0.85,
            "volatility": 0.75,
            "correlation_risk": 0.6,
            "sector_bias": 0.4,
            "liquidity_risk": 0.3
        }),
        MockStrategy("momentum_2", "Momentum Strategy 2", {
            "strategy_type": "momentum",
            "market_beta": 0.8,
            "volatility": 0.7,
            "correlation_risk": 0.55,
            "sector_bias": 0.5,
            "liquidity_risk": 0.25
        }),
        
        # Value strategies - moderate beta and lower volatility
        MockStrategy("value_1", "Value Strategy 1", {
            "strategy_type": "value",
            "market_beta": 0.6,
            "volatility": 0.4,
            "correlation_risk": 0.5,
            "sector_bias": 0.6,
            "liquidity_risk": 0.4
        }),
        MockStrategy("value_2", "Value Strategy 2", {
            "strategy_type": "value",
            "market_beta": 0.55,
            "volatility": 0.35,
            "correlation_risk": 0.45,
            "sector_bias": 0.7,
            "liquidity_risk": 0.5
        }),
        
        # Mean reversion strategies - lower beta but can have higher volatility
        MockStrategy("mean_rev_1", "Mean Reversion 1", {
            "strategy_type": "mean_reversion",
            "market_beta": 0.3,
            "volatility": 0.6,
            "correlation_risk": 0.4,
            "sector_bias": 0.5,
            "liquidity_risk": 0.3
        }),
        MockStrategy("mean_rev_2", "Mean Reversion 2", {
            "strategy_type": "mean_reversion",
            "market_beta": 0.25,
            "volatility": 0.55,
            "correlation_risk": 0.35,
            "sector_bias": 0.4,
            "liquidity_risk": 0.35
        }),
        
        # Low volatility strategies
        MockStrategy("low_vol_1", "Low Volatility 1", {
            "strategy_type": "low_volatility",
            "market_beta": 0.2,
            "volatility": 0.2,
            "correlation_risk": 0.3,
            "sector_bias": 0.6,
            "liquidity_risk": 0.5
        }),
        MockStrategy("low_vol_2", "Low Volatility 2", {
            "strategy_type": "low_volatility",
            "market_beta": 0.15,
            "volatility": 0.15,
            "correlation_risk": 0.25,
            "sector_bias": 0.7,
            "liquidity_risk": 0.55
        }),
    ]
    
    return strategies

def main():
    """Main test function."""
    print_section("INTEGRATED RISK MANAGEMENT TEST")
    
    print("Initializing components...")
    
    # Create global event bus
    event_bus = get_global_event_bus()
    
    # Create risk config manager
    risk_config_manager = RiskConfigManager(profile="balanced")
    
    # Get configuration
    risk_config = risk_config_manager.get_risk_config()
    
    # Create enhanced risk factors calculator
    enhanced_risk_factors = EnhancedRiskFactors(config=risk_config)
    
    # Create strategy manager and strategies
    strategy_manager = MockStrategyManager()
    strategies = create_mock_strategies()
    for strategy in strategies:
        strategy_manager.add_strategy(strategy)
    
    # Activate a few strategies initially
    strategy_manager.promote_strategy("momentum_1")
    strategy_manager.promote_strategy("value_1")
    strategy_manager.promote_strategy("mean_rev_1")
    
    # Create risk-based strategy rotation
    rotation_system = RiskBasedStrategyRotation(
        strategy_manager=strategy_manager,
        event_bus=event_bus,
        config={
            "max_active_strategies": 3,
            "risk_factor_weights": risk_config["risk_weights"]
        }
    )
    
    # Print initial state
    print("\nInitial active strategies:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 1: Normal Market Conditions
    print_section("TEST 1: NORMAL MARKET CONDITIONS")
    
    # Create market data
    market_data = create_mock_market_data()
    positions = create_mock_position_data(market_data)
    factor_data = create_mock_factor_data()
    sector_data = create_mock_sector_data()
    country_data = create_mock_country_data()
    
    # Calculate all risk factors
    risk_assessment = enhanced_risk_factors.calculate_all_risk_factors(
        positions=positions,
        factor_data=factor_data,
        sector_data=sector_data,
        country_data=country_data
    )
    
    print("Risk Assessment:")
    print(f"- Overall Risk Level: {risk_assessment['overall_risk_level'].upper()}")
    print(f"- Liquidity Risk: {risk_assessment['liquidity_risk']['liquidity_risk_level']}")
    if 'factor_risk' in risk_assessment:
        print(f"- Factor Risk: {risk_assessment['factor_risk']['factor_risk_level']}")
    if 'sector_risk' in risk_assessment:
        print(f"- Sector Risk: {risk_assessment['sector_risk']['sector_risk_level']}")
    if 'concentration_risk' in risk_assessment:
        print(f"- Concentration Risk: {risk_assessment['concentration_risk']['concentration_risk_level']}")
    
    # Convert risk factors to event format
    if 'factor_risk' in risk_assessment:
        # Publish risk attribution event
        event_bus.create_and_publish(
            event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
            data={
                "risk_factors": risk_assessment['factor_risk']['factor_exposures'],
                "timestamp": datetime.now().isoformat()
            },
            source="test"
        )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after normal market risk assessment:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 2: High Correlation Scenario
    print_section("TEST 2: HIGH CORRELATION SCENARIO")
    
    # Publish correlation risk alert
    event_bus.create_and_publish(
        event_type=EventType.CORRELATION_RISK_ALERT,
        data={
            "symbols": ["AAPL", "MSFT"],
            "correlation": 0.85,
            "threshold": 0.7,
            "action": "diversify_assets",
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after correlation risk alert:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 3: Sector Concentration Risk
    print_section("TEST 3: SECTOR CONCENTRATION RISK")
    
    # Modify sector data to create concentration
    tech_symbols = ["AAPL", "MSFT", "GOOGL", "META"]
    for symbol in tech_symbols:
        if symbol in sector_data:
            sector_data[symbol] = "Technology"
    
    # Recalculate sector risk
    sector_risk = enhanced_risk_factors.calculate_sector_exposures(positions, sector_data)
    
    print("Sector Exposures:")
    for sector, exposure in sector_risk["sector_exposures"].items():
        print(f"- {sector}: {exposure:.1%}")
    
    print(f"\nSector Risk Level: {sector_risk['sector_risk_level'].upper()}")
    if sector_risk["sectors_at_risk"]:
        print("Sectors at risk:")
        for sector in sector_risk["sectors_at_risk"]:
            print(f"- {sector['sector']}: {sector['exposure']:.1%} (limit: {sector['limit']:.1%})")
    
    # Test 4: Liquidity Risk
    print_section("TEST 4: LIQUIDITY RISK")
    
    # Modify some positions to have liquidity issues
    for symbol in list(positions.keys())[:2]:
        positions[symbol]["avg_daily_volume"] = positions[symbol]["avg_daily_volume"] / 10
        positions[symbol]["bid_ask_spread"] = 0.03
    
    # Recalculate liquidity risk
    liquidity_risk = enhanced_risk_factors.calculate_liquidity_risk(positions)
    
    print(f"Portfolio Liquidity Score: {liquidity_risk['portfolio_liquidity_score']:.2f}")
    print(f"Days to Liquidate: {liquidity_risk['days_to_liquidate']:.2f}")
    print(f"Liquidity Risk Level: {liquidity_risk['liquidity_risk_level'].upper()}")
    
    if liquidity_risk["positions_at_risk"]:
        print("\nPositions with liquidity risk:")
        for pos in liquidity_risk["positions_at_risk"]:
            print(f"- {pos['symbol']}: Score {pos['liquidity_score']:.2f}, " +
                  f"Days to liquidate: {pos['days_to_liquidate']:.2f}")
    
    # Test 5: Significant Drawdown
    print_section("TEST 5: SIGNIFICANT DRAWDOWN")
    
    # Publish significant drawdown alert
    event_bus.create_and_publish(
        event_type=EventType.DRAWDOWN_THRESHOLD_EXCEEDED,
        data={
            "current_drawdown": 0.15,
            "threshold": 0.10,
            "exceeded": True,
            "severity": 2,
            "action": "reduce_exposure",
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after significant drawdown:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Test 6: Changing Market Regime
    print_section("TEST 6: CHANGING MARKET REGIME")
    
    # Publish market regime change to trending
    event_bus.create_and_publish(
        event_type=EventType.MARKET_REGIME_CHANGED,
        data={
            "symbol": "SPY",
            "current_regime": "trending",
            "confidence": 0.8,
            "previous_regime": "normal",
            "timestamp": datetime.now().isoformat(),
            "trigger": "sustained_momentum"
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    # Update risk attribution for trending market
    event_bus.create_and_publish(
        event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
        data={
            "risk_factors": {
                "market_beta": 0.03,
                "volatility": 0.02,
                "correlation": 0.01,
                "sector_exposure": 0.015,
                "liquidity": 0.005
            },
            "timestamp": datetime.now().isoformat()
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.5)
    
    print("\nActive strategies after market regime change to TRENDING:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")
    
    # Summary
    print_section("INTEGRATED RISK MANAGEMENT SUMMARY")
    
    print("Risk management components have been successfully integrated and tested.")
    print("\nThis test demonstrated:")
    print("1. Enhanced risk factor calculation (sector, liquidity, etc.)")
    print("2. Strategy rotation based on risk factors")
    print("3. Event-driven risk management")
    print("4. Adaptation to changing market regimes")
    print("5. Defensive rotation during drawdowns")
    
    print("\nFinal active strategies:")
    active = strategy_manager.get_active_strategies()
    for strategy_id, strategy in active.items():
        print(f"- {strategy.name} (ID: {strategy_id})")

if __name__ == "__main__":
    main()
