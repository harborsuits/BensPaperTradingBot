#!/usr/bin/env python3
"""
Example script showing how to use the Strategy Management System
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any

from .interfaces import CoreContext, MarketContext, Strategy, StrategyPrioritizer
from .factory import create_strategy_management_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("strategy_example")

# Sample strategy implementation
class SampleStrategy(Strategy):
    """Sample strategy implementation for demonstration purposes"""
    
    def __init__(self, name, allocation=0.0):
        self.name = name
        self._allocation = allocation
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
    
    def get_allocation(self) -> float:
        return self._allocation
    
    def set_allocation(self, allocation: float) -> None:
        self._allocation = allocation
        logger.info(f"Strategy {self.name} allocation set to {allocation:.2f}%")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "threshold": 0.5,
            "window_size": 20,
            "take_profit": 0.1,
            "stop_loss": 0.05
        }
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        for param, value in params.items():
            setattr(self, param, value)
            logger.info(f"Strategy {self.name} parameter {param} updated to {value}")
    
    def analyze_parameter_sensitivity(self) -> Dict[str, Any]:
        return {
            "threshold": {
                "bullish": {"optimal_value": 0.6},
                "bearish": {"optimal_value": 0.4},
                "volatile": {"optimal_value": 0.7},
                "sideways": {"optimal_value": 0.5},
                "unknown": {"optimal_value": 0.5}
            },
            "window_size": {
                "bullish": {"optimal_value": 15},
                "bearish": {"optimal_value": 25},
                "volatile": {"optimal_value": 10},
                "sideways": {"optimal_value": 30},
                "unknown": {"optimal_value": 20}
            }
        }

# Sample strategy prioritizer
class SampleStrategyPrioritizer(StrategyPrioritizer):
    """Sample strategy prioritizer implementation"""
    
    def prioritize_strategies(self, market_context: MarketContext, include_reasoning: bool = False) -> Dict[str, Any]:
        # Get current regime
        regime = getattr(market_context, "regime", "unknown")
        
        # Simplified allocation logic based on regime
        allocations = {}
        reasoning = {}
        
        if regime == "bullish":
            allocations = {"MomentumStrategy": 40, "TrendFollowingStrategy": 35, "MeanReversionStrategy": 25}
            reasoning = {
                "MomentumStrategy": "Strong performance in bullish markets, capitalizing on trend continuation",
                "TrendFollowingStrategy": "Effective in established bullish trends for capturing longer moves",
                "MeanReversionStrategy": "Reduced allocation during sustained trends, but useful for short-term corrections"
            }
        elif regime == "bearish":
            allocations = {"MomentumStrategy": 25, "TrendFollowingStrategy": 30, "MeanReversionStrategy": 45}
            reasoning = {
                "MomentumStrategy": "Reduced allocation in bearish markets to limit downside exposure",
                "TrendFollowingStrategy": "Moderate allocation to capture downside momentum with proper risk controls",
                "MeanReversionStrategy": "Increased allocation to capitalize on oversold conditions and bounces"
            }
        elif regime == "volatile":
            allocations = {"MomentumStrategy": 20, "TrendFollowingStrategy": 20, "MeanReversionStrategy": 60}
            reasoning = {
                "MomentumStrategy": "Minimal allocation due to false signals in choppy conditions",
                "TrendFollowingStrategy": "Reduced allocation to avoid whipsaws in volatile periods",
                "MeanReversionStrategy": "Maximum allocation to exploit short-term price swings"
            }
        elif regime == "sideways":
            allocations = {"MomentumStrategy": 15, "TrendFollowingStrategy": 15, "MeanReversionStrategy": 70}
            reasoning = {
                "MomentumStrategy": "Minimal allocation in range-bound markets with no clear direction",
                "TrendFollowingStrategy": "Minimal allocation to avoid losses from failed breakouts",
                "MeanReversionStrategy": "Maximal allocation to capitalize on range-bound conditions"
            }
        else:
            # Unknown or default regime - equal allocation
            allocations = {"MomentumStrategy": 33, "TrendFollowingStrategy": 33, "MeanReversionStrategy": 34}
            reasoning = {
                "MomentumStrategy": "Equal allocation due to uncertain market regime",
                "TrendFollowingStrategy": "Equal allocation due to uncertain market regime",
                "MeanReversionStrategy": "Equal allocation due to uncertain market regime"
            }
        
        # Include market summary for context
        market_summary = f"Market in {regime} regime with signal_bias: {getattr(market_context, 'signal_bias', 'neutral')}"
        
        result = {"allocations": allocations}
        
        # Include reasoning if requested
        if include_reasoning:
            result["reasoning"] = reasoning
            result["market_summary"] = market_summary
        
        return result

# Sample unified context manager (simplified)
class SampleUnifiedContextManager:
    """Sample unified context manager for demonstration"""
    
    def __init__(self):
        self.context_data = {
            "market_bias": "neutral",
            "topics": [],
            "signals": {}
        }
    
    def get_context_summary(self) -> Dict[str, Any]:
        return {
            "market_bias": self.context_data["market_bias"],
            "topics": self.context_data["topics"],
            "confidence": 0.7
        }
    
    def get_active_unified_signals(self, min_confidence=0.5) -> Dict[str, Any]:
        return {
            signal_id: signal 
            for signal_id, signal in self.context_data["signals"].items()
            if signal.get("confidence", 0) >= min_confidence
        }
    
    def update_news_impact(self, impact_data: Dict[str, Any]) -> None:
        self.context_data.update({
            "market_bias": impact_data.get("impact", "neutral"),
            "topics": impact_data.get("topics", [])
        })
        logger.info(f"Updated unified context with news impact: {impact_data.get('impact', 'neutral')}")

async def main():
    """Main example function"""
    logger.info("Starting Strategy Management System example")
    
    # Create core context
    core_context = CoreContext()
    
    # Add sample strategies
    strategies = {
        "MomentumStrategy": SampleStrategy("MomentumStrategy", allocation=33.3),
        "TrendFollowingStrategy": SampleStrategy("TrendFollowingStrategy", allocation=33.3),
        "MeanReversionStrategy": SampleStrategy("MeanReversionStrategy", allocation=33.4)
    }
    
    for name, strategy in strategies.items():
        core_context.add_strategy(name, strategy)
    
    # Create unified context and prioritizer
    unified_context = SampleUnifiedContextManager()
    prioritizer = SampleStrategyPrioritizer()
    
    # Set initial market context
    core_context.update_market_context({
        "regime": "bullish",
        "volatility": "low",
        "signal_bias": "moderately bullish"
    })
    
    # Create configuration directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    # Create example config file
    config = {
        "rotator": {
            "rotation_frequency_days": 7,
            "min_change_threshold": 5.0,
            "force_on_regime_change": True,
            "max_allocation_change": 15.0
        },
        "integration": {
            "auto_rotation_enabled": True,
            "auto_rotation_interval_days": 7,
            "respond_to_unified_signals": True
        },
        "learning": {
            "learning_frequency_days": 14,
            "min_data_points": 30,
            "learning_rate": 0.2
        }
    }
    
    with open("config/strategy_management.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create the strategy management system
    system = create_strategy_management_system(
        core_context=core_context,
        strategy_prioritizer=prioritizer,
        unified_context_manager=unified_context,
        config_path="config/strategy_management.json"
    )
    
    # Extract components
    rotator = system["rotator"]
    integration = system["integration"]
    learning = system["learning"]
    
    # Start the integration module
    await integration.start()
    
    # Simulate some market changes and observe system behavior
    logger.info("=== Initial State ===")
    current_allocations = rotator.get_current_allocations()
    logger.info(f"Current allocations: {current_allocations}")
    
    # Trigger an immediate rotation
    logger.info("=== Triggering Initial Rotation ===")
    rotation_result = rotator.rotate_strategies(force=True)
    logger.info(f"Rotation result: {rotation_result}")
    
    # Change market regime and observe auto-rotation
    logger.info("=== Changing Market Regime ===")
    core_context.update_market_context({"regime": "volatile"})
    
    # Wait a moment for async events to process
    await asyncio.sleep(1)
    
    # Check current allocations after regime change
    current_allocations = rotator.get_current_allocations()
    logger.info(f"Allocations after regime change: {current_allocations}")
    
    # Simulate news impact
    logger.info("=== Processing News Impact ===")
    news_items = [
        {
            "title": "Fed Signals Interest Rate Cut",
            "summary": "Federal Reserve indicates potential rate cuts in upcoming meeting.",
            "sentiment": "Positive",
            "source": "Financial News"
        },
        {
            "title": "Economic Growth Exceeds Expectations",
            "summary": "Q2 GDP growth rate higher than analyst predictions.",
            "sentiment": "Positive",
            "source": "Market Watch"
        }
    ]
    
    impact_result = integration.process_news_impact(news_items)
    logger.info(f"News impact result: {impact_result}")
    
    # Wait a moment for async events to process
    await asyncio.sleep(1)
    
    # Force a learning cycle
    logger.info("=== Running Learning Cycle ===")
    
    # Add some mock performance data
    mock_strategy_data = {
        "MomentumStrategy": {"return": 1.2, "sharpe": 1.5},
        "TrendFollowingStrategy": {"return": 0.8, "sharpe": 1.1},
        "MeanReversionStrategy": {"return": 1.5, "sharpe": 1.8}
    }
    
    core_context._notify_listeners("strategy_performance_updated", mock_strategy_data)
    
    # Run learning cycle
    learning_result = learning.force_learning_run()
    logger.info(f"Learning result: {learning_result}")
    
    # Get learning report
    report = learning.get_learning_report()
    logger.info(f"Learning report: {report}")
    
    # Stop the integration module
    await integration.stop()
    
    logger.info("Example completed")

if __name__ == "__main__":
    asyncio.run(main()) 