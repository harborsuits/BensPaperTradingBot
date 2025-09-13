#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContinuousLearner - Module for implementing continuous learning and
feedback loops for trading strategies.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import threading
import time
import json
import pickle

# Setup logging
logger = logging.getLogger("ContinuousLearner")

class ContinuousLearner:
    """
    Component that implements continuous learning and feedback loops
    for trading strategies.
    """
    
    def __init__(
        self,
        strategy_rotator,
        config_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        evaluation_interval: int = 86400,  # 1 day in seconds
        auto_retrain: bool = True,
        performance_threshold: float = -0.02,  # -2% performance threshold
    ):
        """
        Initialize the continuous learner.
        
        Args:
            strategy_rotator: Reference to the StrategyRotator instance
            config_path: Path to configuration file
            data_dir: Directory for data storage
            evaluation_interval: Interval for evaluating strategies in seconds
            auto_retrain: Whether to automatically retrain underperforming strategies
            performance_threshold: Threshold for triggering retraining
        """
        self.strategy_rotator = strategy_rotator
        self.evaluation_interval = evaluation_interval
        self.auto_retrain = auto_retrain
        self.performance_threshold = performance_threshold
        
        # Setup data paths
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".trading_bot", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Path for saving/loading model performance metrics
        self.metrics_path = os.path.join(self.data_dir, "performance_metrics.json")
        
        # Performance history
        self.performance_history = self._load_performance_history()
        
        # Model versions tracking
        self.model_versions = {}
        
        # Internal state
        self._running = False
        self._thread = None
    
    def _load_performance_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load performance history from disk."""
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
        
        return {}
    
    def _save_performance_history(self) -> None:
        """Save performance history to disk."""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.performance_history, f)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def start(self) -> None:
        """Start the continuous learning process."""
        if self._running:
            logger.warning("Continuous learner is already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("Continuous learner started")
    
    def stop(self) -> None:
        """Stop the continuous learning process."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        
        logger.info("Continuous learner stopped")
    
    def _run_loop(self) -> None:
        """Main loop for continuous learning."""
        while self._running:
            try:
                # Evaluate strategy performance
                self._evaluate_strategies()
                
                # Check for retraining needs
                if self.auto_retrain:
                    self._check_retraining_needs()
                
                # Save current state
                self._save_performance_history()
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
            
            # Sleep until next evaluation
            time.sleep(self.evaluation_interval)
    
    def _evaluate_strategies(self) -> None:
        """Evaluate the performance of all strategies."""
        # Get performance metrics from strategy rotator
        metrics = self.strategy_rotator.get_performance_metrics()
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        
        # Update performance history
        for strategy_name, strategy_metrics in metrics.items():
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            # Add latest metrics
            self.performance_history[strategy_name].append({
                "timestamp": timestamp,
                "average_performance": strategy_metrics["average_performance"],
                "model_version": self.model_versions.get(strategy_name, "initial")
            })
            
            # Limit history size
            if len(self.performance_history[strategy_name]) > 100:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-100:]
        
        logger.debug("Evaluated strategy performance")
    
    def _check_retraining_needs(self) -> None:
        """Check if any strategies need retraining."""
        # Get market data for retraining
        market_data = self._get_market_data_for_training()
        
        if market_data is None or len(market_data) < 1000:
            logger.warning("Insufficient market data for retraining")
            return
        
        strategies_to_retrain = []
        
        # Check each strategy's recent performance
        for strategy_name, history in self.performance_history.items():
            if not history or len(history) < 5:
                continue
            
            # Calculate average recent performance
            recent_performance = [h["average_performance"] for h in history[-5:]]
            avg_recent_performance = np.mean(recent_performance)
            
            # If performance is below threshold, mark for retraining
            if avg_recent_performance < self.performance_threshold:
                strategies_to_retrain.append(strategy_name)
                logger.info(f"Strategy {strategy_name} marked for retraining (performance: {avg_recent_performance:.4f})")
        
        # Retrain marked strategies
        if strategies_to_retrain:
            self._retrain_strategies(strategies_to_retrain, market_data)
    
    def _get_market_data_for_training(self) -> Optional[pd.DataFrame]:
        """Get market data for strategy training."""
        # This would typically fetch data from your data store
        # For now, we'll return None as a placeholder
        # In a real implementation, this would load historical market data
        return None
    
    def _retrain_strategies(self, strategy_names: List[str], market_data: pd.DataFrame) -> None:
        """Retrain specific strategies with new market data."""
        # Filter to just the strategies that need retraining
        retrain_strategies = {
            name: self.strategy_rotator.strategies_by_name.get(name)
            for name in strategy_names
            if name in self.strategy_rotator.strategies_by_name
        }
        
        # Count successfully retrained strategies
        retrained_count = 0
        
        # Retrain each strategy
        for name, strategy in retrain_strategies.items():
            try:
                # Special handling for RL strategies
                if hasattr(strategy, 'train'):
                    logger.info(f"Retraining strategy: {name}")
                    
                    # Perform training
                    rewards = strategy.train(market_data)
                    
                    # Update version
                    current_version = self.model_versions.get(name, "initial")
                    if current_version == "initial":
                        new_version = "v1.1"
                    else:
                        # Increment version (v1.1 -> v1.2, etc.)
                        base, num = current_version.split('.')
                        new_version = f"{base}.{int(num) + 1}"
                    
                    self.model_versions[name] = new_version
                    
                    # Record retraining event
                    if name not in self.performance_history:
                        self.performance_history[name] = []
                    
                    self.performance_history[name].append({
                        "timestamp": datetime.now().isoformat(),
                        "event": "retrained",
                        "model_version": new_version
                    })
                    
                    retrained_count += 1
                    logger.info(f"Successfully retrained {name} to version {new_version}")
                else:
                    logger.warning(f"Strategy {name} does not support training")
            
            except Exception as e:
                logger.error(f"Error retraining strategy {name}: {e}")
        
        logger.info(f"Retrained {retrained_count} strategies")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report for all strategies.
        
        Returns:
            Dict with performance metrics and training history
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "strategies": {},
            "overall_performance": 0.0
        }
        
        overall_perfs = []
        
        # Generate report for each strategy
        for strategy_name, history in self.performance_history.items():
            if not history:
                continue
            
            # Calculate metrics
            recent_history = history[-10:]
            performances = [h.get("average_performance", 0) for h in recent_history 
                           if "average_performance" in h]
            
            if not performances:
                continue
            
            avg_performance = np.mean(performances)
            overall_perfs.append(avg_performance)
            
            # Get current version
            current_version = self.model_versions.get(strategy_name, "initial")
            
            # Count retraining events
            retraining_events = sum(1 for h in history if h.get("event") == "retrained")
            
            # Add to report
            report["strategies"][strategy_name] = {
                "average_performance": avg_performance,
                "current_version": current_version,
                "retraining_count": retraining_events,
                "performance_trend": performances
            }
        
        # Calculate overall performance
        if overall_perfs:
            report["overall_performance"] = np.mean(overall_perfs)
        
        return report
    
    def manually_trigger_retraining(self, strategy_names: List[str]) -> bool:
        """
        Manually trigger retraining for specific strategies.
        
        Args:
            strategy_names: List of strategy names to retrain
            
        Returns:
            bool: True if retraining was triggered, False otherwise
        """
        # Validate strategy names
        valid_names = [name for name in strategy_names 
                      if name in self.strategy_rotator.strategies_by_name]
        
        if not valid_names:
            logger.warning("No valid strategy names provided for retraining")
            return False
        
        # Get market data
        market_data = self._get_market_data_for_training()
        
        if market_data is None or len(market_data) < 1000:
            logger.warning("Insufficient market data for retraining")
            return False
        
        # Retrain strategies
        self._retrain_strategies(valid_names, market_data)
        return True


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # This would normally be imported
    from trading_bot.strategy.strategy_rotator import StrategyRotator
    
    # Create strategy rotator
    rotator = StrategyRotator()
    
    # Create continuous learner
    learner = ContinuousLearner(
        strategy_rotator=rotator,
        evaluation_interval=3600  # 1 hour for testing
    )
    
    # Start continuous learning
    learner.start()
    
    # Let it run for a while
    time.sleep(10)
    
    # Stop continuous learning
    learner.stop()
    
    # Get performance report
    report = learner.get_performance_report()
    print(json.dumps(report, indent=2)) 