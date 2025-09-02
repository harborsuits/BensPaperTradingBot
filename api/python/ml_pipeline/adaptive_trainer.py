"""
Adaptive ML Training Pipeline
This module provides an end-to-end machine learning pipeline that continuously
trains and adapts models based on market conditions, backtesting results, and
real-world performance.
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components from our existing code
from trading_bot.market_context.market_context import get_market_context
from trading_bot.symbolranker.symbol_ranker import get_symbol_ranker

class AdaptiveTrainer:
    """
    End-to-end ML pipeline that continuously trains and adapts models
    based on market data, backtesting results, and real-world performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the adaptive trainer with configuration.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("AdaptiveTrainer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Access singletons for market context and symbol ranker
        self.market_context = get_market_context()
        self.symbol_ranker = get_symbol_ranker()
        
        # Initialize model states
        self.models = {
            "market_regime": None,
            "strategy_selector": None,
            "symbol_ranker": None
        }
        
        # Performance tracking
        self.performance_history = {
            "strategies": {},
            "symbols": {},
            "pairs": {}
        }
        
        # Store recent predictions for evaluation
        self.recent_predictions = deque(maxlen=100)
        
        # Metadata about training runs
        self.training_metadata = {
            "last_run": None,
            "total_runs": 0,
            "training_duration": 0
        }
        
        # Training state
        self._training_lock = threading.RLock()
        self._is_training = False
        
        # Load any saved state
        self._load_saved_state()
        
        self.logger.info("AdaptiveTrainer initialized")
    
    def train_all_models(self, force=False):
        """
        Train or update all models in the pipeline.
        
        Args:
            force: Whether to force training even if not needed
            
        Returns:
            Dictionary with training results
        """
        # Prevent multiple training runs at the same time
        if self._is_training and not force:
            self.logger.info("Training already in progress, skipping")
            return {"status": "skipped", "reason": "training_in_progress"}
        
        with self._training_lock:
            self._is_training = True
            start_time = time.time()
            
            try:
                self.logger.info("Starting training run for all models")
                
                # Load the latest market context
                context = self.market_context.get_market_context()
                
                # Step 1: Train market regime classifier
                market_regime_results = self._train_market_regime_model(context)
                
                # Step 2: Train strategy selector model
                strategy_selector_results = self._train_strategy_selector_model(context)
                
                # Step 3: Train symbol ranker models
                symbol_ranker_results = self._train_symbol_ranker_models(context)
                
                # Step 4: Evaluate performance and adapt
                evaluation_results = self._evaluate_and_adapt()
                
                # Update metadata
                duration = time.time() - start_time
                self.training_metadata["last_run"] = datetime.datetime.now().isoformat()
                self.training_metadata["total_runs"] += 1
                self.training_metadata["training_duration"] = duration
                
                # Save state
                self._save_current_state()
                
                # Combine all results
                results = {
                    "status": "success",
                    "duration": duration,
                    "market_regime": market_regime_results,
                    "strategy_selector": strategy_selector_results,
                    "symbol_ranker": symbol_ranker_results,
                    "evaluation": evaluation_results,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                self.logger.info(f"Training completed in {duration:.2f} seconds")
                return results
                
            except Exception as e:
                self.logger.error(f"Error during training: {str(e)}")
                return {"status": "error", "error": str(e)}
                
            finally:
                self._is_training = False
    
    def _train_market_regime_model(self, context):
        """
        Train the market regime classifier model.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training market regime model")
        
        try:
            # In a real implementation, this would train a classifier using historical data
            # For this example, we'll simulate training
            
            # Extract features from context
            market_indicators = context.get("market", {}).get("indicators", {})
            
            # Features we might use in a real model
            features = {
                "vix": market_indicators.get("vix", 15),
                "market_direction": 1 if market_indicators.get("market_direction", "neutral") == "bullish" else
                                   -1 if market_indicators.get("market_direction", "neutral") == "bearish" else 0,
                "treasury_10y": market_indicators.get("treasury_10y", 3.0),
                "sector_rotation": self._calculate_sector_rotation(context)
            }
            
            # Simulate model updates - in a real implementation, this would be actual model training
            mock_accuracy = 0.85 + (np.random.random() * 0.1)
            mock_f1_score = 0.82 + (np.random.random() * 0.1)
            
            # Store model state (would be actual model in real implementation)
            self.models["market_regime"] = {
                "version": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "features": features,
                "trained_at": datetime.datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "metrics": {
                    "accuracy": mock_accuracy,
                    "f1_score": mock_f1_score,
                    "regimes_identified": 8
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training market regime model: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _train_strategy_selector_model(self, context):
        """
        Train the strategy selector model.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training strategy selector model")
        
        try:
            # In a real implementation, this would train a model to select strategies
            # For this example, we'll simulate training
            
            # Extract current regime and performance data
            current_regime = context.get("market", {}).get("regime", "unknown")
            
            # Simulate model updates
            mock_accuracy = 0.78 + (np.random.random() * 0.1)
            mock_sharpe = 1.2 + (np.random.random() * 0.5)
            
            # Store model state
            self.models["strategy_selector"] = {
                "version": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "current_regime": current_regime,
                "trained_at": datetime.datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "metrics": {
                    "accuracy": mock_accuracy,
                    "sharpe_ratio": mock_sharpe,
                    "strategies_evaluated": 5
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training strategy selector model: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _train_symbol_ranker_models(self, context):
        """
        Train the symbol ranker models.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training symbol ranker models")
        
        try:
            # In a real implementation, this would train models to rank symbols
            # For this example, we'll simulate training
            
            # Get strategies and symbols from context
            strategies = [s["id"] for s in context.get("strategies", {}).get("ranked", [])]
            symbols = list(context.get("symbols", {}).keys())
            
            # Simulate model updates
            mock_accuracy = 0.75 + (np.random.random() * 0.15)
            mock_precision = 0.72 + (np.random.random() * 0.15)
            
            # Store model state
            self.models["symbol_ranker"] = {
                "version": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "strategies": strategies,
                "symbol_count": len(symbols),
                "trained_at": datetime.datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "metrics": {
                    "accuracy": mock_accuracy,
                    "precision": mock_precision,
                    "strategies_covered": len(strategies),
                    "symbols_analyzed": len(symbols)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training symbol ranker models: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _evaluate_and_adapt(self):
        """
        Evaluate recent performance and adapt models accordingly.
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating performance and adapting models")
        
        try:
            # In a real implementation, this would analyze performance and adapt
            # For this example, we'll simulate the process
            
            # Simulate adapting weights based on performance
            adaption_score = 0.7 + (np.random.random() * 0.3)
            
            return {
                "status": "success",
                "metrics": {
                    "adaptation_score": adaption_score,
                    "predictions_evaluated": len(self.recent_predictions),
                    "improvements_made": 3
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during evaluation and adaptation: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def run_inference(self, context=None):
        """
        Run inference using the trained models to generate predictions.
        
        Args:
            context: Market context to use or None to get latest
            
        Returns:
            Dictionary with predictions
        """
        self.logger.info("Running inference with trained models")
        
        try:
            # Get latest context if not provided
            if context is None:
                context = self.market_context.get_market_context()
            
            # 1. Predict market regime
            market_regime = self._predict_market_regime(context)
            
            # 2. Select best strategies for current regime
            strategies = self._select_strategies(market_regime, context)
            
            # 3. Rank symbols for selected strategies
            symbol_rankings = {}
            for strategy in strategies:
                strategy_id = strategy["id"]
                ranked_symbols = self.symbol_ranker.rank_symbols_for_strategy(
                    strategy_id, limit=5
                )
                symbol_rankings[strategy_id] = ranked_symbols
            
            # 4. Find best symbol-strategy pairs
            pairs = self.symbol_ranker.find_best_symbol_strategy_pairs(limit=10)
            
            # Store prediction for later evaluation
            prediction = {
                "timestamp": datetime.datetime.now().isoformat(),
                "market_regime": market_regime,
                "strategies": strategies,
                "top_pairs": pairs[:3] if pairs else []
            }
            self.recent_predictions.append(prediction)
            
            return {
                "status": "success",
                "market_regime": market_regime,
                "recommended_strategies": strategies,
                "symbol_rankings": symbol_rankings,
                "best_pairs": pairs,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _predict_market_regime(self, context):
        """
        Predict current market regime using trained model.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with regime prediction
        """
        # In a real implementation, this would use a trained model
        # For now, we'll use the regime from context
        regime = context.get("market", {}).get("regime", "unknown")
        confidence = 0.8 + (np.random.random() * 0.2)
        
        return {
            "regime": regime,
            "confidence": confidence,
            "alternatives": [
                {"regime": "stable", "probability": 0.15},
                {"regime": "unsettled", "probability": 0.05}
            ]
        }
    
    def _select_strategies(self, market_regime, context):
        """
        Select best strategies for current market regime.
        
        Args:
            market_regime: Predicted market regime
            context: Current market context
            
        Returns:
            List of recommended strategies with scores
        """
        # Get all strategies
        all_strategies = context.get("strategies", {}).get("ranked", [])
        
        # Filter by current regime
        regime_name = market_regime.get("regime", "unknown")
        matching_strategies = [
            strategy for strategy in all_strategies
            if regime_name in strategy.get("suitable_regimes", [])
        ]
        
        # If no matching strategies, use top 3
        if not matching_strategies and all_strategies:
            matching_strategies = all_strategies[:3]
        
        return matching_strategies
    
    def _calculate_sector_rotation(self, context):
        """
        Calculate sector rotation metric from context.
        
        Args:
            context: Market context
            
        Returns:
            Float representing sector rotation
        """
        # Extract sector performance
        sectors = context.get("market", {}).get("sectors", {})
        
        # If no sectors, return neutral value
        if not sectors:
            return 0.0
        
        # Calculate standard deviation of sector performance
        # Higher values indicate higher rotation/dispersion
        performances = list(sectors.values())
        if performances:
            return np.std(performances)
        
        return 0.0
    
    def _save_current_state(self):
        """Save the current state of all models and metadata."""
        save_dir = self._config.get("save_dir", "data/ml_models")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state = {
            "models": self.models,
            "metadata": self.training_metadata,
            "timestamp": timestamp
        }
        
        filepath = os.path.join(save_dir, f"model_state_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Saved model state to {filepath}")
    
    def _load_saved_state(self):
        """Load the most recent saved state if available."""
        save_dir = self._config.get("save_dir", "data/ml_models")
        
        # Check if directory exists
        if not os.path.exists(save_dir):
            self.logger.info(f"Save directory {save_dir} does not exist. Starting fresh.")
            return
        
        # Find most recent state file
        files = [f for f in os.listdir(save_dir) if f.startswith("model_state_") and f.endswith(".json")]
        if not files:
            self.logger.info("No saved state found. Starting fresh.")
            return
        
        # Sort by timestamp (newest first)
        files.sort(reverse=True)
        latest_file = os.path.join(save_dir, files[0])
        
        try:
            with open(latest_file, 'r') as f:
                state = json.load(f)
            
            # Load state
            self.models = state.get("models", self.models)
            self.training_metadata = state.get("metadata", self.training_metadata)
            
            self.logger.info(f"Loaded saved state from {latest_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading saved state: {str(e)}")


class TrainingScheduler:
    """
    Scheduler for managing periodic training of models.
    """
    
    def __init__(self, config=None):
        """
        Initialize the training scheduler.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Default training intervals (in seconds)
        self.intervals = self._config.get("intervals", {
            "market_regime": 3600,  # 1 hour
            "strategy_selector": 7200,  # 2 hours
            "symbol_ranker": 14400,  # 4 hours
            "full_retrain": 86400  # 24 hours
        })
        
        # Set up logging
        self.logger = logging.getLogger("TrainingScheduler")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Get trainer instance
        self.trainer = get_adaptive_trainer()
        
        # Last training times
        self.last_training = {
            "market_regime": 0,
            "strategy_selector": 0,
            "symbol_ranker": 0,
            "full_retrain": 0
        }
        
        # Thread for running scheduler
        self._scheduler_thread = None
        self._running = False
        self._stop_event = threading.Event()
        
        self.logger.info("TrainingScheduler initialized")
    
    def start(self):
        """Start the training scheduler."""
        if self._running:
            self.logger.info("Scheduler already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        
        self.logger.info("Training scheduler started")
    
    def stop(self):
        """Stop the training scheduler."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        self.logger.info("Training scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        self.logger.info("Scheduler loop started")
        
        while not self._stop_event.is_set():
            current_time = time.time()
            
            # Check if any training is due
            for model_type, interval in self.intervals.items():
                if current_time - self.last_training.get(model_type, 0) >= interval:
                    self.logger.info(f"Training due for {model_type}")
                    
                    if model_type == "full_retrain":
                        # Run full training
                        result = self.trainer.train_all_models()
                        if result.get("status") == "success":
                            # Update all last training times
                            for key in self.last_training.keys():
                                self.last_training[key] = current_time
                    else:
                        # In a real implementation, this would train specific models
                        pass
            
            # Sleep for a bit to avoid busy-waiting
            time.sleep(60)  # Check every minute


# Create singleton instances
_adaptive_trainer = None
_training_scheduler = None

def get_adaptive_trainer(config=None):
    """
    Get the singleton AdaptiveTrainer instance.
    
    Args:
        config: Optional configuration for the trainer
    
    Returns:
        AdaptiveTrainer instance
    """
    global _adaptive_trainer
    if _adaptive_trainer is None:
        _adaptive_trainer = AdaptiveTrainer(config)
    return _adaptive_trainer

def get_training_scheduler(config=None):
    """
    Get the singleton TrainingScheduler instance.
    
    Args:
        config: Optional configuration for the scheduler
    
    Returns:
        TrainingScheduler instance
    """
    global _training_scheduler
    if _training_scheduler is None:
        _training_scheduler = TrainingScheduler(config)
    return _training_scheduler
