#!/usr/bin/env python3
"""
Advanced Hybrid Strategy Combiner - Combines multiple advanced strategy types into robust hybrid strategies
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
import datetime
import logging
from collections import defaultdict

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from advanced_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    VolumeProfileStrategy, 
    VolatilityBreakoutStrategy,
    backtest_strategy
)
from strategy_registry import StrategyRegistry
from synthetic_market_generator import SyntheticMarketGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/hybrid_combiner_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('advanced_hybrid_combiner')


class AdvancedHybridStrategy:
    """
    Combines multiple trading strategies into a single hybrid strategy.
    
    The hybrid can use various methods to combine signals:
    - Weighted: Weighted average of signals based on specified weights
    - Consensus: Require a certain number of strategies to agree
    - Confidence: Weight signals by the confidence reported by each strategy
    - Adaptive: Dynamically adjust weights based on recent performance
    """
    
    def __init__(
        self, 
        strategies: List[Dict[str, Any]], 
        combination_method: str = "weighted",
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.6,
        performance_window: int = 20
    ):
        """
        Initialize the hybrid strategy.
        
        Args:
            strategies: List of strategy configurations (type and parameters)
            combination_method: Method to combine signals (weighted, consensus, confidence, adaptive)
            weights: Weights for each strategy (if using weighted method)
            consensus_threshold: Threshold for consensus (0-1)
            performance_window: Window for calculating recent performance (adaptive method)
        """
        self.strategy_name = "AdvancedHybrid"
        self.combination_method = combination_method
        self.consensus_threshold = consensus_threshold
        self.performance_window = performance_window
        
        # Load strategies
        self.strategies = []
        self.strategy_objs = []
        
        for strategy_config in strategies:
            strategy_type = strategy_config["strategy_type"]
            parameters = strategy_config["parameters"]
            
            # Store strategy configuration
            self.strategies.append({
                "type": strategy_type,
                "parameters": parameters
            })
            
            # Create strategy object
            strategy_obj = self._create_strategy(strategy_type, parameters)
            self.strategy_objs.append(strategy_obj)
        
        # Initialize weights
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # Performance tracking for adaptive method
        self.strategy_performance = [[] for _ in range(len(strategies))]
        
        self.parameters = {
            "strategy_count": len(strategies),
            "combination_method": combination_method,
            "weights": self.weights.copy(),
            "consensus_threshold": consensus_threshold,
            "performance_window": performance_window
        }
    
    def _create_strategy(self, strategy_type: str, parameters: Dict[str, Any]) -> Any:
        """
        Create a strategy object of the specified type with the given parameters.
        
        Args:
            strategy_type: Type of strategy
            parameters: Strategy parameters
            
        Returns:
            Strategy object
        """
        if strategy_type == "MeanReversion":
            return MeanReversionStrategy(**parameters)
        elif strategy_type == "Momentum":
            return MomentumStrategy(**parameters)
        elif strategy_type == "VolumeProfile":
            return VolumeProfileStrategy(**parameters)
        elif strategy_type == "VolatilityBreakout":
            return VolatilityBreakoutStrategy(**parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal by combining signals from multiple strategies.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        # Calculate signals from each strategy
        strategy_signals = []
        
        for i, strategy in enumerate(self.strategy_objs):
            signal = strategy.calculate_signal(market_data)
            strategy_signals.append(signal)
            
            # For adaptive method, track performance
            if self.combination_method == "adaptive":
                self._track_strategy_performance(i, signal, market_data)
        
        # Update weights for adaptive method
        if self.combination_method == "adaptive":
            self._update_adaptive_weights()
        
        # Combine signals using the selected method
        if self.combination_method == "weighted" or self.combination_method == "adaptive":
            return self._weighted_signal_combination(strategy_signals)
        elif self.combination_method == "consensus":
            return self._consensus_signal_combination(strategy_signals)
        elif self.combination_method == "confidence":
            return self._confidence_weighted_combination(strategy_signals)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _weighted_signal_combination(self, strategy_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals using weighted average.
        
        Args:
            strategy_signals: List of signal dictionaries from each strategy
            
        Returns:
            Combined signal dictionary
        """
        # Initialize signal values
        buy_weight = 0.0
        sell_weight = 0.0
        confidence = 0.0
        
        # Combine weighted signals
        for i, signal in enumerate(strategy_signals):
            weight = self.weights[i]
            
            if signal["signal"] == "buy":
                buy_weight += weight
                confidence += weight * signal.get("confidence", 1.0)
            elif signal["signal"] == "sell":
                sell_weight += weight
                confidence += weight * signal.get("confidence", 1.0)
        
        # Determine final signal
        if buy_weight > sell_weight:
            return {
                "signal": "buy",
                "confidence": confidence,
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
        elif sell_weight > buy_weight:
            return {
                "signal": "sell",
                "confidence": confidence,
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
        else:
            return {
                "signal": "none",
                "confidence": 0,
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
    
    def _consensus_signal_combination(self, strategy_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals using consensus approach.
        
        Args:
            strategy_signals: List of signal dictionaries from each strategy
            
        Returns:
            Combined signal dictionary
        """
        # Count signal types
        signal_counts = {"buy": 0, "sell": 0, "none": 0}
        
        for signal in strategy_signals:
            signal_type = signal["signal"]
            signal_counts[signal_type] += 1
        
        # Calculate consensus percentages
        total = len(strategy_signals)
        buy_percent = signal_counts["buy"] / total
        sell_percent = signal_counts["sell"] / total
        
        # Determine if consensus threshold is met
        if buy_percent >= self.consensus_threshold:
            # Calculate average confidence of buy signals
            confidence = sum(
                s.get("confidence", 1.0) for s in strategy_signals if s["signal"] == "buy"
            ) / signal_counts["buy"] if signal_counts["buy"] > 0 else 0
            
            return {
                "signal": "buy",
                "confidence": confidence,
                "buy_percent": buy_percent,
                "sell_percent": sell_percent,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
        elif sell_percent >= self.consensus_threshold:
            # Calculate average confidence of sell signals
            confidence = sum(
                s.get("confidence", 1.0) for s in strategy_signals if s["signal"] == "sell"
            ) / signal_counts["sell"] if signal_counts["sell"] > 0 else 0
            
            return {
                "signal": "sell",
                "confidence": confidence,
                "buy_percent": buy_percent,
                "sell_percent": sell_percent,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
        else:
            return {
                "signal": "none",
                "confidence": 0,
                "buy_percent": buy_percent,
                "sell_percent": sell_percent,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
    
    def _confidence_weighted_combination(self, strategy_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals with weights based on reported confidence.
        
        Args:
            strategy_signals: List of signal dictionaries from each strategy
            
        Returns:
            Combined signal dictionary
        """
        # Initialize signal values
        buy_weight = 0.0
        sell_weight = 0.0
        
        # Combine signals weighted by confidence
        for i, signal in enumerate(strategy_signals):
            base_weight = self.weights[i]
            signal_confidence = signal.get("confidence", 0.5)  # Default to 0.5 if not provided
            
            if signal["signal"] == "buy":
                buy_weight += base_weight * signal_confidence
            elif signal["signal"] == "sell":
                sell_weight += base_weight * signal_confidence
        
        # Normalize weights
        total_weight = buy_weight + sell_weight
        
        if total_weight > 0:
            confidence = total_weight
            buy_weight = buy_weight / total_weight
            sell_weight = sell_weight / total_weight
        else:
            confidence = 0
            buy_weight = 0
            sell_weight = 0
        
        # Determine final signal
        if buy_weight > sell_weight:
            return {
                "signal": "buy",
                "confidence": confidence,
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
        elif sell_weight > buy_weight:
            return {
                "signal": "sell",
                "confidence": confidence,
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
        else:
            return {
                "signal": "none",
                "confidence": 0,
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "strategy_signals": [s["signal"] for s in strategy_signals]
            }
    
    def _track_strategy_performance(self, strategy_idx: int, signal: Dict[str, Any], market_data: pd.DataFrame):
        """
        Track performance of an individual strategy for adaptive weighting.
        
        Args:
            strategy_idx: Index of the strategy
            signal: Signal from the strategy
            market_data: Market data
        """
        if len(market_data) < 2:
            return
        
        # Get current and next day's close prices
        current_close = market_data["Close"].iloc[-1] if "Close" in market_data.columns else market_data["close"].iloc[-1]
        
        # Store signal and price
        self.strategy_performance[strategy_idx].append({
            "signal": signal["signal"],
            "price": current_close,
            "timestamp": market_data.index[-1]
        })
        
        # Limit history to performance window
        if len(self.strategy_performance[strategy_idx]) > self.performance_window:
            self.strategy_performance[strategy_idx].pop(0)
    
    def _update_adaptive_weights(self):
        """Update strategy weights based on recent performance."""
        # Need sufficient history to update weights
        if any(len(perf) < self.performance_window for perf in self.strategy_performance):
            return
        
        # Calculate return for each strategy
        strategy_returns = []
        
        for strategy_perf in self.strategy_performance:
            # Calculate returns from signals
            returns = []
            
            for i in range(len(strategy_perf) - 1):
                current = strategy_perf[i]
                next_day = strategy_perf[i + 1]
                
                price_change_pct = (next_day["price"] - current["price"]) / current["price"]
                
                if current["signal"] == "buy":
                    returns.append(price_change_pct)
                elif current["signal"] == "sell":
                    returns.append(-price_change_pct)
                else:
                    returns.append(0)
            
            # Calculate average return
            avg_return = np.mean(returns) if returns else 0
            strategy_returns.append(max(0, avg_return))  # Use positive returns only
        
        # Update weights based on relative performance
        total_return = sum(strategy_returns)
        
        if total_return > 0:
            self.weights = [r / total_return for r in strategy_returns]
        else:
            # Revert to equal weights if no positive returns
            self.weights = [1.0 / len(self.strategy_objs)] * len(self.strategy_objs)


def create_hybrid_from_top_performers(registry: StrategyRegistry, 
                                     scenario: str = None, 
                                     strategy_types: List[str] = None,
                                     top_n: int = 1,
                                     combination_method: str = "weighted") -> AdvancedHybridStrategy:
    """
    Create a hybrid strategy from top performers in the registry.
    
    Args:
        registry: Strategy registry
        scenario: Specific scenario to filter by (optional)
        strategy_types: List of strategy types to include (optional)
        top_n: Number of top performers of each type to include
        combination_method: Method to combine signals
        
    Returns:
        Hybrid strategy
    """
    # Get all strategies from registry
    all_strategies = registry.get_all_strategies()
    
    if not all_strategies:
        raise ValueError("No strategies found in registry")
    
    # Filter by scenario if specified
    if scenario:
        all_strategies = [s for s in all_strategies if s.get("scenario") == scenario]
    
    # Determine available strategy types
    available_types = set(s.get("strategy_type") for s in all_strategies)
    
    if not strategy_types:
        strategy_types = list(available_types)
    else:
        # Validate requested types
        for t in strategy_types:
            if t not in available_types:
                logger.warning(f"Strategy type {t} not found in registry")
    
    # Group by strategy type
    strategies_by_type = defaultdict(list)
    
    for strategy in all_strategies:
        strategy_type = strategy.get("strategy_type")
        
        if strategy_type in strategy_types:
            strategies_by_type[strategy_type].append(strategy)
    
    # Sort each group by performance
    for strategy_type, strategies in strategies_by_type.items():
        # Sort by fitness if available, otherwise by total return
        if "fitness" in strategies[0]:
            strategies_by_type[strategy_type] = sorted(
                strategies, 
                key=lambda s: s.get("fitness", 0), 
                reverse=True
            )
        else:
            strategies_by_type[strategy_type] = sorted(
                strategies, 
                key=lambda s: s.get("performance", {}).get("total_return_pct", 0), 
                reverse=True
            )
    
    # Select top N performers of each type
    selected_strategies = []
    
    for strategy_type, strategies in strategies_by_type.items():
        for i, strategy in enumerate(strategies[:top_n]):
            selected_strategies.append({
                "strategy_type": strategy["strategy_type"],
                "parameters": strategy["parameters"]
            })
    
    logger.info(f"Creating hybrid strategy with {len(selected_strategies)} components")
    
    # Create hybrid strategy
    return AdvancedHybridStrategy(
        strategies=selected_strategies,
        combination_method=combination_method
    )


def backtest_hybrid_strategies(
    hybrid_strategies: List[AdvancedHybridStrategy],
    market_data: pd.DataFrame,
    output_dir: str = None
):
    """
    Backtest multiple hybrid strategies and compare results.
    
    Args:
        hybrid_strategies: List of hybrid strategies
        market_data: Market data for backtesting
        output_dir: Directory for output charts
    """
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"backtest_results/hybrid_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store backtest results
    results = {}
    
    # Backtest each strategy
    for i, hybrid in enumerate(hybrid_strategies):
        method = hybrid.combination_method
        backtest_result = backtest_strategy(hybrid, market_data)
        
        strategy_id = f"Hybrid_{method}_{i}"
        results[strategy_id] = backtest_result
        
        logger.info(f"{strategy_id} - Return: {backtest_result['total_return_pct']:.2f}%, "
                  f"Drawdown: {backtest_result['max_drawdown']:.2f}%, "
                  f"Win Rate: {backtest_result['win_rate']:.1f}%")
    
    # Plot equity curves
    plt.figure(figsize=(12, 8))
    
    for strategy_id, result in results.items():
        plt.plot(result["equity_curve"], label=strategy_id)
    
    plt.title("Hybrid Strategy Comparison")
    plt.xlabel("Trading Days")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "hybrid_comparison.png"))
    
    # Create performance summary table
    summary_data = []
    
    for strategy_id, result in results.items():
        summary_data.append({
            "Strategy": strategy_id,
            "Return (%)": result["total_return_pct"],
            "Max Drawdown (%)": result["max_drawdown"],
            "Win Rate (%)": result["win_rate"],
            "Trade Count": result["trade_count"],
            "Avg Win (%)": result["avg_win"],
            "Avg Loss (%)": result["avg_loss"],
            "Profit Factor": result["profit_factor"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "hybrid_summary.csv"), index=False)
    
    return results


def create_hybrid_strategies_from_registry(registry_path: str):
    """
    Create and test hybrid strategies from registry strategies.
    
    Args:
        registry_path: Path to strategy registry
    """
    # Initialize registry
    registry = StrategyRegistry(registry_path)
    
    # Generate test market data
    market_generator = SyntheticMarketGenerator()
    test_scenarios = {}
    
    test_scenarios["bull_market"] = market_generator.generate_bull_market()
    test_scenarios["bear_market"] = market_generator.generate_bear_market()
    test_scenarios["volatile_market"] = market_generator.generate_volatile_market()
    test_scenarios["flash_crash"] = market_generator.generate_flash_crash()
    
    # Create timestamp for output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # For each scenario, create hybrid strategies with different methods
    for scenario_name, market_data in test_scenarios.items():
        logger.info(f"Creating hybrid strategies for {scenario_name}")
        
        # Create hybrid strategies with different combination methods
        hybrid_weighted = create_hybrid_from_top_performers(
            registry=registry,
            scenario=None,  # Use all scenarios
            top_n=1,
            combination_method="weighted"
        )
        
        hybrid_consensus = create_hybrid_from_top_performers(
            registry=registry,
            scenario=None,  # Use all scenarios
            top_n=1,
            combination_method="consensus"
        )
        
        hybrid_confidence = create_hybrid_from_top_performers(
            registry=registry,
            scenario=None,  # Use all scenarios
            top_n=1,
            combination_method="confidence"
        )
        
        hybrid_adaptive = create_hybrid_from_top_performers(
            registry=registry,
            scenario=None,  # Use all scenarios
            top_n=1,
            combination_method="adaptive"
        )
        
        # Backtest hybrid strategies
        output_dir = f"backtest_results/hybrid_{timestamp}/{scenario_name}"
        
        backtest_hybrid_strategies(
            hybrid_strategies=[
                hybrid_weighted,
                hybrid_consensus,
                hybrid_confidence,
                hybrid_adaptive
            ],
            market_data=market_data,
            output_dir=output_dir
        )


def create_custom_hybrid_strategies():
    """Create and test custom hybrid strategies with manually specified components."""
    # Create hybrid strategies
    hybrid_strategies = []
    
    # 1. Create a hybrid with one of each strategy type
    hybrid_all_types = AdvancedHybridStrategy(
        strategies=[
            {
                "strategy_type": "MeanReversion",
                "parameters": {
                    "lookback_period": 20,
                    "entry_std": 2.0,
                    "exit_std": 0.5,
                    "smoothing": 3
                }
            },
            {
                "strategy_type": "Momentum",
                "parameters": {
                    "short_period": 14,
                    "medium_period": 30,
                    "long_period": 90,
                    "threshold": 0.02,
                    "smoothing": 3
                }
            },
            {
                "strategy_type": "VolumeProfile",
                "parameters": {
                    "lookback_period": 20,
                    "volume_threshold": 1.5,
                    "price_levels": 20,
                    "smoothing": 3
                }
            },
            {
                "strategy_type": "VolatilityBreakout",
                "parameters": {
                    "atr_period": 14,
                    "breakout_multiple": 1.5,
                    "lookback_period": 5,
                    "filter_threshold": 0.2
                }
            }
        ],
        combination_method="weighted"
    )
    
    hybrid_strategies.append(hybrid_all_types)
    
    # 2. Create a mean reversion + volatility breakout hybrid
    hybrid_mr_vb = AdvancedHybridStrategy(
        strategies=[
            {
                "strategy_type": "MeanReversion",
                "parameters": {
                    "lookback_period": 20,
                    "entry_std": 2.0,
                    "exit_std": 0.5,
                    "smoothing": 3
                }
            },
            {
                "strategy_type": "VolatilityBreakout",
                "parameters": {
                    "atr_period": 14,
                    "breakout_multiple": 1.5,
                    "lookback_period": 5,
                    "filter_threshold": 0.2
                }
            }
        ],
        combination_method="consensus",
        consensus_threshold=0.5
    )
    
    hybrid_strategies.append(hybrid_mr_vb)
    
    # 3. Create a momentum + volume profile hybrid
    hybrid_mom_vol = AdvancedHybridStrategy(
        strategies=[
            {
                "strategy_type": "Momentum",
                "parameters": {
                    "short_period": 14,
                    "medium_period": 30,
                    "long_period": 90,
                    "threshold": 0.02,
                    "smoothing": 3
                }
            },
            {
                "strategy_type": "VolumeProfile",
                "parameters": {
                    "lookback_period": 20,
                    "volume_threshold": 1.5,
                    "price_levels": 20,
                    "smoothing": 3
                }
            }
        ],
        combination_method="confidence"
    )
    
    hybrid_strategies.append(hybrid_mom_vol)
    
    # Generate test market data
    market_generator = SyntheticMarketGenerator()
    test_market = market_generator.generate_volatile_market()
    
    # Create timestamp for output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"backtest_results/custom_hybrid_{timestamp}"
    
    # Backtest hybrid strategies
    backtest_hybrid_strategies(
        hybrid_strategies=hybrid_strategies,
        market_data=test_market,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Create and test hybrid trading strategies")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--custom", 
        action="store_true",
        help="Create custom hybrid strategies"
    )
    
    args = parser.parse_args()
    
    if args.custom:
        create_custom_hybrid_strategies()
    else:
        create_hybrid_strategies_from_registry(args.registry)
