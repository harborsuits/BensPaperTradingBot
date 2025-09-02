#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IntegratedStrategyRotator - Advanced multi-strategy trading system that
dynamically rotates between different strategies based on market regimes
and performance metrics.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegratedStrategyRotator")

class MarketRegime(Enum):
    """Market regime classifications"""
    UNKNOWN = 0
    BULL = 1      # Rising prices, low volatility
    BEAR = 2      # Falling prices, high volatility
    SIDEWAYS = 3  # Range-bound, moderate volatility
    HIGH_VOL = 4  # Extremely high volatility regardless of direction
    LOW_VOL = 5   # Extremely low volatility regardless of direction
    CRISIS = 6    # Extreme volatility, sharp declines, liquidity issues

class IntegratedStrategyRotator:
    """
    Central engine for a multi-strategy trading framework that:
    - Diversifies risk across various trading approaches
    - Adapts to changing market regimes 
    - Dynamically adjusts allocations based on performance
    - Provides comprehensive risk management
    
    The rotator combines multiple strategies into a cohesive system:
    - Momentum: Captures continued price movement
    - Trend-Following: Identifies and follows prevailing trends
    - Mean Reversion: Exploits tendency of prices to revert to mean
    - Breakout Swing: Profits from significant breakouts from ranges
    - Volatility Breakout: Focuses on sudden volatility spikes
    - Option Spreads: Uses options for income and hedging
    """
    
    def __init__(
        self,
        strategies: List[str] = None,
        initial_allocations: Dict[str, float] = None,
        config_path: str = None,
        data_dir: str = None,
        portfolio_value: float = 100000.0,
        market_data_provider: Any = None
    ):
        """
        Initialize the integrated strategy rotator.
        
        Args:
            strategies: List of strategy names to include (None = use all available)
            initial_allocations: Starting allocation percentages by strategy
            config_path: Path to configuration file
            data_dir: Directory for data storage and state persistence
            portfolio_value: Total portfolio value for allocation calculations
            market_data_provider: Provider for market data (prices, indicators)
        """
        # Set up strategy list (default to all strategies if None)
        self.default_strategies = [
            "momentum", "trend_following", "mean_reversion", 
            "breakout_swing", "volatility_breakout", "option_spreads"
        ]
        
        self.strategies = strategies or self.default_strategies
        self.portfolio_value = portfolio_value
        
        # Set up directories
        self._setup_directories(data_dir, config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Set up initial allocations (equal if not provided)
        if initial_allocations is None:
            equal_allocation = 100.0 / len(self.strategies)
            self.current_allocations = {s: equal_allocation for s in self.strategies}
        else:
            self.current_allocations = initial_allocations
            self._validate_allocations()
        
        # Strategy performance tracking
        self.performance_history = {}
        self.rotation_history = []
        
        # Market regime tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        
        # Set up market data provider
        self.market_data_provider = market_data_provider
        
        # Setup rotation schedule
        self.last_rotation_date = None
        self.rotation_frequency = self.config.get("rotation_frequency", "weekly")
        
        logger.info(f"IntegratedStrategyRotator initialized with {len(self.strategies)} strategies")
        logger.info(f"Initial allocations: {self.current_allocations}")
    
    def _setup_directories(self, data_dir: str = None, config_path: str = None) -> None:
        """Set up necessary directories and file paths."""
        # Set default paths if not provided
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if data_dir is None:
            self.data_dir = os.path.join(base_dir, "data", "strategy_rotator")
        else:
            self.data_dir = data_dir
            
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set config path
        if config_path is None:
            self.config_path = os.path.join(self.data_dir, "config.json")
        else:
            self.config_path = config_path
            
        # Set other paths
        self.state_path = os.path.join(self.data_dir, "state.json")
        self.performance_path = os.path.join(self.data_dir, "performance.json")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        
        # Default configuration
        default_config = {
            "rotation_frequency": "weekly",
            "max_allocation_change": 20.0,  # Max % change per rotation
            "min_allocation": 5.0,          # Minimum allocation %
            "max_allocation": 40.0,         # Maximum allocation %
            "risk_weight": 0.6,             # Weight of risk metrics in allocation decisions
            "performance_lookback_days": 90,
            "regime_update_frequency": "daily",
            "emergency_stop_drawdown": 25.0,  # Stop trading if drawdown exceeds this %
            "strategy_configs": {
                # Strategy-specific configurations
            }
        }
        
        # Write default config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error writing default config: {str(e)}")
        
        return default_config
    
    def _validate_allocations(self) -> None:
        """Ensure allocations include all strategies and sum to 100%."""
        # Add missing strategies with 0 allocation
        for strategy in self.strategies:
            if strategy not in self.current_allocations:
                self.current_allocations[strategy] = 0.0
        
        # Remove strategies not in self.strategies
        self.current_allocations = {
            k: v for k, v in self.current_allocations.items() 
            if k in self.strategies
        }
        
        # Normalize to 100%
        total = sum(self.current_allocations.values())
        if abs(total - 100.0) > 0.01:  # Allow small rounding errors
            logger.warning(f"Allocations sum to {total}, normalizing to 100%")
            for strategy in self.current_allocations:
                self.current_allocations[strategy] = (
                    self.current_allocations[strategy] / total * 100.0
                )
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime based on market data.
        
        Args:
            market_data: DataFrame with market price/indicator data
            
        Returns:
            Current market regime
        """
        logger.info("Detecting market regime")
        
        if market_data.empty or len(market_data) < 50:
            logger.warning("Insufficient data for regime detection")
            return MarketRegime.UNKNOWN
        
        # Get configuration values
        lookback_period = self.config.get("regime_detection", {}).get("lookback_period", 120)
        vol_window = self.config.get("regime_detection", {}).get("volatility_window", 20)
        trend_window = self.config.get("regime_detection", {}).get("trend_window", 50)
        vol_threshold_high = self.config.get("regime_detection", {}).get("volatility_threshold_high", 1.5)
        vol_threshold_low = self.config.get("regime_detection", {}).get("volatility_threshold_low", 0.5)
        trend_threshold = self.config.get("regime_detection", {}).get("trend_threshold", 0.1)
        
        # Ensure we have enough data
        lookback_period = min(lookback_period, len(market_data) - 1)
        
        # Get relevant data (most recent lookback_period days)
        prices = market_data.iloc[-lookback_period:]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate volatility (annualized)
        current_volatility = returns.iloc[-vol_window:].std() * np.sqrt(252)
        
        # Calculate historical volatility as baseline
        historical_volatility = returns.iloc[:-vol_window].std() * np.sqrt(252) if len(returns) > vol_window*2 else current_volatility
        
        # Calculate volatility ratio
        volatility_ratio = current_volatility / historical_volatility if historical_volatility.any() > 0 else 1.0
        
        # Calculate trend indicators (we'll use moving averages)
        ma_short = prices.iloc[-trend_window:].mean()
        ma_long = prices.mean()
        
        # Normalized trend strength (as percentage)
        trend_strength = (ma_short / ma_long - 1)
        
        # Determine market regime based on volatility and trend
        regime = MarketRegime.UNKNOWN
        
        # Check for crisis conditions first (extreme volatility)
        if volatility_ratio.mean() > vol_threshold_high * 1.5:
            regime = MarketRegime.CRISIS
        
        # Check for high/low volatility regimes
        elif volatility_ratio.mean() > vol_threshold_high:
            regime = MarketRegime.HIGH_VOL
        
        elif volatility_ratio.mean() < vol_threshold_low:
            regime = MarketRegime.LOW_VOL
        
        # Check trend direction for normal volatility
        else:
            if trend_strength.mean() > trend_threshold:
                regime = MarketRegime.BULL
            elif trend_strength.mean() < -trend_threshold:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS
        
        # Record regime change if different
        if regime != self.current_regime:
            self.regime_history.append({
                "date": market_data.index[-1],
                "old_regime": self.current_regime.name,
                "new_regime": regime.name,
                "volatility_ratio": float(volatility_ratio.mean()),
                "trend_strength": float(trend_strength.mean())
            })
            logger.info(f"Market regime changed from {self.current_regime.name} to {regime.name}")
            self.current_regime = regime
        
        return regime
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for each strategy.
        
        Returns:
            Dict of performance metrics by strategy
        """
        logger.info("Getting strategy performance metrics")
        
        # Try to load cached performance data
        if os.path.exists(self.performance_path):
            try:
                with open(self.performance_path, 'r') as f:
                    cached_performance = json.load(f)
                
                # Check if data is recent enough (within 24 hours)
                for strategy_data in cached_performance.values():
                    if "last_updated" in strategy_data:
                        last_updated = datetime.fromisoformat(strategy_data["last_updated"])
                        age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                        
                        if age_hours < 24:
                            logger.info(f"Using cached performance data (age: {age_hours:.1f} hours)")
                            return cached_performance
            
            except Exception as e:
                logger.error(f"Error loading cached performance data: {str(e)}")
        
        # Generate mock performance data for demonstration purposes
        # In a real implementation, this would query actual strategy performance
        # from backtest results or live trading data
        performance_data = self._generate_mock_performance_data()
        
        # Save performance data for future use
        try:
            with open(self.performance_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            logger.info(f"Saved performance data to {self.performance_path}")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
        
        return performance_data
    
    def _generate_mock_performance_data(self) -> Dict[str, Dict[str, float]]:
        """
        Generate mock performance data for testing.
        In a real implementation, this would be replaced with actual performance metrics.
        
        Returns:
            Dict of mock performance metrics by strategy
        """
        # Define strategy characteristics for simulation
        strategy_profiles = {
            "momentum": {
                "base_return": 0.12,  # 12% annual return
                "volatility": 0.15,   # 15% volatility
                "win_rate": 0.55,     # 55% win rate
                "drawdown": -0.18     # 18% max drawdown
            },
            "trend_following": {
                "base_return": 0.10,  # 10% annual return
                "volatility": 0.12,   # 12% volatility 
                "win_rate": 0.48,     # 48% win rate
                "drawdown": -0.15     # 15% max drawdown
            },
            "mean_reversion": {
                "base_return": 0.09,  # 9% annual return
                "volatility": 0.09,   # 9% volatility
                "win_rate": 0.65,     # 65% win rate
                "drawdown": -0.12     # 12% max drawdown
            },
            "breakout_swing": {
                "base_return": 0.14,  # 14% annual return
                "volatility": 0.18,   # 18% volatility
                "win_rate": 0.45,     # 45% win rate
                "drawdown": -0.22     # 22% max drawdown
            },
            "volatility_breakout": {
                "base_return": 0.16,  # 16% annual return
                "volatility": 0.22,   # 22% volatility
                "win_rate": 0.42,     # 42% win rate
                "drawdown": -0.25     # 25% max drawdown
            },
            "option_spreads": {
                "base_return": 0.08,  # 8% annual return
                "volatility": 0.08,   # 8% volatility
                "win_rate": 0.70,     # 70% win rate
                "drawdown": -0.10     # 10% max drawdown
            }
        }
        
        # Apply regime effects to performance
        regime_performance_modifiers = {
            MarketRegime.BULL: {
                "momentum": 1.5,           # 50% better in bull market
                "trend_following": 1.3,    # 30% better
                "mean_reversion": 0.8,     # 20% worse
                "breakout_swing": 1.2,     # 20% better
                "volatility_breakout": 0.9, # 10% worse
                "option_spreads": 0.9      # 10% worse
            },
            MarketRegime.BEAR: {
                "momentum": 0.5,           # 50% worse in bear market
                "trend_following": 1.1,    # 10% better
                "mean_reversion": 1.2,     # 20% better
                "breakout_swing": 0.8,     # 20% worse
                "volatility_breakout": 1.3, # 30% better
                "option_spreads": 1.1      # 10% better
            },
            MarketRegime.SIDEWAYS: {
                "momentum": 0.7,           # 30% worse in sideways
                "trend_following": 0.6,    # 40% worse
                "mean_reversion": 1.5,     # 50% better
                "breakout_swing": 0.8,     # 20% worse
                "volatility_breakout": 0.7, # 30% worse
                "option_spreads": 1.3      # 30% better
            },
            MarketRegime.HIGH_VOL: {
                "momentum": 0.7,           # 30% worse in high vol
                "trend_following": 0.8,    # 20% worse
                "mean_reversion": 0.6,     # 40% worse
                "breakout_swing": 1.0,     # Neutral
                "volatility_breakout": 1.8, # 80% better
                "option_spreads": 1.3      # 30% better
            },
            MarketRegime.LOW_VOL: {
                "momentum": 1.2,           # 20% better in low vol
                "trend_following": 0.8,    # 20% worse
                "mean_reversion": 1.4,     # 40% better
                "breakout_swing": 0.6,     # 40% worse
                "volatility_breakout": 0.5, # 50% worse
                "option_spreads": 1.0      # Neutral
            },
            MarketRegime.CRISIS: {
                "momentum": 0.2,           # 80% worse in crisis
                "trend_following": 0.5,    # 50% worse
                "mean_reversion": 0.3,     # 70% worse
                "breakout_swing": 0.4,     # 60% worse
                "volatility_breakout": 1.5, # 50% better
                "option_spreads": 1.6      # 60% better
            },
            MarketRegime.UNKNOWN: {
                "momentum": 1.0,           # Neutral in unknown
                "trend_following": 1.0,
                "mean_reversion": 1.0,
                "breakout_swing": 1.0,
                "volatility_breakout": 1.0,
                "option_spreads": 1.0
            }
        }
        
        # Generate performance data based on strategy profiles and current regime
        performance_data = {}
        current_regime_modifiers = regime_performance_modifiers.get(
            self.current_regime, regime_performance_modifiers[MarketRegime.UNKNOWN]
        )
        
        for strategy in self.strategies:
            if strategy in strategy_profiles:
                profile = strategy_profiles[strategy]
                modifier = current_regime_modifiers.get(strategy, 1.0)
                
                # Add some randomness to make it realistic
                random_factor = 0.9 + np.random.random() * 0.2  # 0.9 to 1.1
                
                # Calculate metrics with regime effects and randomness
                annual_return = profile["base_return"] * modifier * random_factor
                volatility = profile["volatility"] * (1.0 / (modifier**0.5))  # Inverse relationship with performance
                win_rate = min(0.95, profile["win_rate"] * (modifier**0.3))  # Less affected by regime
                drawdown = profile["drawdown"] * (1.0 / modifier)  # Better performance = smaller drawdown
                
                # Calculate derived metrics
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                sortino_ratio = annual_return / (volatility * 1.5) if volatility > 0 else 0  # Simple approximation
                calmar_ratio = annual_return / abs(drawdown) if drawdown != 0 else 0
                
                performance_data[strategy] = {
                    "annualized_return": annual_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio,
                    "win_rate": win_rate,
                    "max_drawdown": drawdown,
                    "last_updated": datetime.now().isoformat()
                }
        
        return performance_data
    
    def is_rotation_due(self) -> bool:
        """
        Check if strategy rotation is due based on schedule.
        
        Returns:
            True if rotation is due, False otherwise
        """
        # If no last rotation date, rotation is due
        if self.last_rotation_date is None:
            return True
        
        current_date = datetime.now()
        days_since_last_rotation = (current_date - self.last_rotation_date).days
        
        # Check based on rotation frequency
        if self.rotation_frequency == "daily":
            return days_since_last_rotation >= 1
        elif self.rotation_frequency == "weekly":
            return days_since_last_rotation >= 7
        elif self.rotation_frequency == "monthly":
            return days_since_last_rotation >= 30
        elif self.rotation_frequency == "quarterly":
            return days_since_last_rotation >= 90
        else:
            # Default to weekly if unknown frequency
            return days_since_last_rotation >= 7
    
    def rotate_strategies(self, 
                         market_data: pd.DataFrame = None,
                         force_rotation: bool = False) -> Dict[str, Any]:
        """
        Perform strategy rotation based on market regime and performance.
        
        Args:
            market_data: Market data for regime detection (fetched if None)
            force_rotation: Whether to force rotation regardless of schedule
            
        Returns:
            Dict containing rotation results
        """
        logger.info("Initiating strategy rotation")
        
        # Check if rotation is due unless forcing
        if not force_rotation and not self.is_rotation_due():
            logger.info("Strategy rotation not due yet")
            return {
                "rotated": False,
                "message": "Strategy rotation not due yet",
                "current_allocations": self.current_allocations
            }
        
        # Detect market regime
        if market_data is not None:
            regime = self.detect_market_regime(market_data)
        else:
            logger.warning("No market data provided for regime detection, using current regime")
            regime = self.current_regime
        
        # Get strategy performance data
        performance_data = self.get_strategy_performance()
        
        # 1. Calculate regime-based allocations
        regime_allocations = self._calculate_regime_allocations(regime)
        
        # 2. Calculate performance-based allocations
        performance_allocations = self._calculate_performance_allocations(performance_data)
        
        # 3. Blend allocations with weights from config
        regime_weight = self.config.get("rotation", {}).get("regime_weight", 0.5)
        performance_weight = 1.0 - regime_weight
        
        target_allocations = {}
        for strategy in self.strategies:
            target_allocations[strategy] = (
                regime_allocations.get(strategy, 0.0) * regime_weight +
                performance_allocations.get(strategy, 0.0) * performance_weight
            )
        
        # 4. Apply constraints
        constrained_allocations = self._apply_allocation_constraints(target_allocations)
        
        # Calculate allocation changes
        allocation_changes = {}
        for strategy in self.strategies:
            current = float(self.current_allocations.get(strategy, 0.0))
            target = float(constrained_allocations.get(strategy, 0.0))
            allocation_changes[strategy] = target - current
        
        # Update current allocations
        self.current_allocations = constrained_allocations
        
        # Update last rotation date
        self.last_rotation_date = datetime.now()
        
        # Record rotation in history
        rotation_entry = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime.name,
            "previous_allocations": self.current_allocations.copy(),
            "new_allocations": constrained_allocations.copy(),
            "allocation_changes": allocation_changes
        }
        
        self.rotation_history.append(rotation_entry)
        
        # Save state
        self.save_state()
        
        logger.info(f"Completed strategy rotation in {regime.name} regime")
        logger.info(f"New allocations: {constrained_allocations}")
        
        return {
            "rotated": True,
            "previous_allocations": self.current_allocations,
            "new_allocations": constrained_allocations,
            "market_regime": regime.name,
            "allocation_changes": allocation_changes,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_regime_allocations(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Calculate target allocations based on current market regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dict of allocations by strategy
        """
        # Define regime-specific weights based on strategy types
        regime_weights = {
            MarketRegime.BULL: {
                "momentum": 0.3,
                "trend_following": 0.3,
                "mean_reversion": 0.05,
                "breakout_swing": 0.15,
                "volatility_breakout": 0.1,
                "option_spreads": 0.1
            },
            MarketRegime.BEAR: {
                "momentum": 0.05,
                "trend_following": 0.25,
                "mean_reversion": 0.15,
                "breakout_swing": 0.1,
                "volatility_breakout": 0.2,
                "option_spreads": 0.25
            },
            MarketRegime.SIDEWAYS: {
                "momentum": 0.1,
                "trend_following": 0.05,
                "mean_reversion": 0.4,
                "breakout_swing": 0.1,
                "volatility_breakout": 0.1,
                "option_spreads": 0.25
            },
            MarketRegime.HIGH_VOL: {
                "momentum": 0.05,
                "trend_following": 0.15,
                "mean_reversion": 0.15,
                "breakout_swing": 0.1,
                "volatility_breakout": 0.35,
                "option_spreads": 0.2
            },
            MarketRegime.LOW_VOL: {
                "momentum": 0.2,
                "trend_following": 0.1,
                "mean_reversion": 0.35,
                "breakout_swing": 0.1,
                "volatility_breakout": 0.05,
                "option_spreads": 0.2
            },
            MarketRegime.CRISIS: {
                "momentum": 0.0,
                "trend_following": 0.2,
                "mean_reversion": 0.0,
                "breakout_swing": 0.0,
                "volatility_breakout": 0.3,
                "option_spreads": 0.5
            },
            MarketRegime.UNKNOWN: {
                # Even allocation for unknown regime
                "momentum": 1.0/6,
                "trend_following": 1.0/6,
                "mean_reversion": 1.0/6,
                "breakout_swing": 1.0/6,
                "volatility_breakout": 1.0/6,
                "option_spreads": 1.0/6
            }
        }
        
        # Get weights for current regime
        weights = regime_weights.get(regime, regime_weights[MarketRegime.UNKNOWN])
        
        # Normalize weights for available strategies
        allocations = {}
        total_weight = 0.0
        
        # Filter for our active strategies
        for strategy in self.strategies:
            if strategy in weights:
                allocations[strategy] = weights[strategy] * 100.0  # Convert to percentage
                total_weight += weights[strategy]
        
        # Normalize if weights don't sum to 1
        if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
            for strategy in allocations:
                allocations[strategy] = (allocations[strategy] / total_weight) * 100.0
        
        return allocations
    
    def _calculate_performance_allocations(self, performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate target allocations based on strategy performance.
        
        Args:
            performance_data: Dict of performance metrics by strategy
            
        Returns:
            Dict of allocations by strategy
        """
        # Define weights for different performance metrics
        metric_weights = {
            "sharpe_ratio": 0.4,
            "win_rate": 0.2,
            "annualized_return": 0.3,
            "max_drawdown": 0.1  # Will be converted to positive value
        }
        
        # Calculate scores for each strategy
        scores = {}
        has_performance_data = False
        
        for strategy in self.strategies:
            if strategy in performance_data and performance_data[strategy]:
                metrics = performance_data[strategy]
                score = 0.0
                
                # Calculate weighted score from available metrics
                for metric, weight in metric_weights.items():
                    if metric in metrics and metrics[metric] is not None:
                        # For drawdown, convert to positive score (smaller is better)
                        if metric == "max_drawdown":
                            # Convert drawdown to positive value between 0-1
                            drawdown_score = 1.0 - min(abs(metrics[metric]), 1.0)
                            score += drawdown_score * weight
                        else:
                            # Normalize metrics to reasonable ranges
                            if metric == "sharpe_ratio":
                                normalized = min(max(metrics[metric], 0), 3) / 3
                            elif metric == "win_rate":
                                normalized = min(max(metrics[metric], 0), 1)
                            elif metric == "annualized_return":
                                normalized = min(max(metrics[metric], -1), 1) * 0.5 + 0.5  # Convert to 0-1
                            else:
                                normalized = metrics[metric]
                            
                            score += normalized * weight
                
                scores[strategy] = score
                has_performance_data = True
            else:
                # Default score for strategies without performance data
                scores[strategy] = 0.5
        
        # If no performance data available, default to equal weights
        if not has_performance_data:
            return {strategy: 100.0 / len(self.strategies) for strategy in self.strategies}
        
        # Calculate allocations based on scores
        allocations = {}
        total_score = sum(scores.values())
        
        if total_score > 0:
            for strategy in scores:
                allocations[strategy] = (scores[strategy] / total_score) * 100.0
        else:
            # If all scores are 0, equal allocation
            for strategy in scores:
                allocations[strategy] = 100.0 / len(scores)
        
        return allocations
    
    def _apply_allocation_constraints(self, target_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Apply constraints to target allocations.
        
        Args:
            target_allocations: Target strategy allocations
            
        Returns:
            Constrained allocations
        """
        # Get constraint parameters
        min_allocation = self.config.get("min_allocation", 5.0)
        max_allocation = self.config.get("max_allocation", 40.0)
        max_allocation_change = self.config.get("max_allocation_change", 20.0)
        
        # Apply constraints
        constrained_allocations = {}
        
        # 1. Apply min/max constraints and limit changes
        for strategy in self.strategies:
            target = target_allocations.get(strategy, 0.0)
            current = self.current_allocations.get(strategy, 0.0)
            
            # Apply min allocation constraint
            target = max(target, min_allocation)
            
            # Apply max allocation constraint
            target = min(target, max_allocation)
            
            # Limit change magnitude if not forcing
            max_change = max_allocation_change
            if current > 0:
                if target > current:
                    # Limit increase
                    target = min(target, current + max_change)
                else:
                    # Limit decrease
                    target = max(target, current - max_change)
            
            constrained_allocations[strategy] = target
        
        # 2. Normalize to ensure sum is 100%
        total = sum(constrained_allocations.values())
        if abs(total - 100.0) > 0.01:  # Allow small rounding errors
            scale_factor = 100.0 / total
            for strategy in constrained_allocations:
                constrained_allocations[strategy] *= scale_factor
        
        return constrained_allocations
    
    def get_current_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations.
        
        Returns:
            Dict of current allocations by strategy
        """
        return self.current_allocations.copy()
    
    def save_state(self) -> None:
        """Save current state to file."""
        state = {
            "current_allocations": self.current_allocations,
            "last_rotation_date": self.last_rotation_date.isoformat() 
                if self.last_rotation_date else None,
            "current_regime": self.current_regime.name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved state to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def load_state(self) -> bool:
        """
        Load state from file.
        
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if not os.path.exists(self.state_path):
            logger.info("No state file found")
            return False
        
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            self.current_allocations = state.get("current_allocations", {})
            
            last_rotation_date = state.get("last_rotation_date")
            if last_rotation_date:
                self.last_rotation_date = datetime.fromisoformat(last_rotation_date)
            
            regime_name = state.get("current_regime", "UNKNOWN")
            self.current_regime = MarketRegime[regime_name]
            
            logger.info(f"Loaded state from {self.state_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False 