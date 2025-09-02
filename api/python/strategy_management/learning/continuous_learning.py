import os
import json
import logging
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from filelock import FileLock

# Import interfaces from parent modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from strategy_management.interfaces import CoreContext

logger = logging.getLogger("continuous_learning")

class ContinuousLearningSystem:
    """
    Implements a continuous learning system that analyzes strategy performance
    and adapts allocation and parameters based on results.
    """
    
    def __init__(self, core_context: CoreContext, config: Dict[str, Any]):
        self.core_context = core_context
        self.config = config
        
        # Set up configuration parameters
        self.learning_frequency_days = config.get("learning_frequency_days", 7)
        self.min_data_points = config.get("min_data_points", 30)
        self.max_history_days = config.get("max_history_days", 365)
        self.learning_rate = config.get("learning_rate", 0.2)
        self.regime_weight = config.get("regime_weight", 0.7)
        
        # File paths for persistence
        self.data_dir = config.get("data_dir", "data/learning")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.performance_file = os.path.join(self.data_dir, "performance_history.json")
        self.strategy_profiles_file = os.path.join(self.data_dir, "strategy_profiles.json")
        self.learning_results_file = os.path.join(self.data_dir, "learning_results.json")
        
        # Initialize file locks for thread safety
        self.performance_lock = FileLock(f"{self.performance_file}.lock")
        self.profiles_lock = FileLock(f"{self.strategy_profiles_file}.lock")
        self.learning_lock = FileLock(f"{self.learning_results_file}.lock")
        
        # Performance tracking
        self.performance_history = self._load_performance_history()
        self.strategy_profiles = self._load_strategy_profiles()
        self.learning_results = self._load_learning_results()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Last learning timestamp
        self.last_learning_run = self.learning_results.get("last_run", 
                                 (datetime.now() - timedelta(days=self.learning_frequency_days)).isoformat())
        
        # Register for portfolio updates
        self.core_context.add_event_listener("portfolio_updated", self._on_portfolio_update)
        self.core_context.add_event_listener("strategy_performance_updated", self._on_strategy_performance_update)
    
    def _load_performance_history(self) -> Dict:
        """Load performance history from file or initialize if not exists"""
        try:
            with self.performance_lock:
                if os.path.exists(self.performance_file):
                    with open(self.performance_file, 'r') as f:
                        return json.load(f)
                else:
                    return {
                        "daily_returns": {},
                        "strategy_returns": {},
                        "market_regimes": {}
                    }
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
            return {
                "daily_returns": {},
                "strategy_returns": {},
                "market_regimes": {}
            }
    
    def _load_strategy_profiles(self) -> Dict:
        """Load strategy profiles from file or initialize if not exists"""
        try:
            with self.profiles_lock:
                if os.path.exists(self.strategy_profiles_file):
                    with open(self.strategy_profiles_file, 'r') as f:
                        return json.load(f)
                else:
                    return {
                        "regime_performance": {},
                        "parameter_sensitivity": {},
                        "adaptive_parameters": {}
                    }
        except Exception as e:
            logger.error(f"Error loading strategy profiles: {str(e)}")
            return {
                "regime_performance": {},
                "parameter_sensitivity": {},
                "adaptive_parameters": {}
            }
    
    def _load_learning_results(self) -> Dict:
        """Load learning results from file or initialize if not exists"""
        try:
            with self.learning_lock:
                if os.path.exists(self.learning_results_file):
                    with open(self.learning_results_file, 'r') as f:
                        return json.load(f)
                else:
                    return {
                        "last_run": (datetime.now() - timedelta(days=self.learning_frequency_days)).isoformat(),
                        "regime_weights": {},
                        "parameter_updates": {},
                        "allocation_adjustments": {},
                        "learning_metrics": {
                            "iterations": 0,
                            "improvement": 0.0,
                            "convergence": False
                        }
                    }
        except Exception as e:
            logger.error(f"Error loading learning results: {str(e)}")
            return {
                "last_run": (datetime.now() - timedelta(days=self.learning_frequency_days)).isoformat(),
                "regime_weights": {},
                "parameter_updates": {},
                "allocation_adjustments": {},
                "learning_metrics": {
                    "iterations": 0,
                    "improvement": 0.0,
                    "convergence": False
                }
            }
    
    def _save_performance_history(self):
        """Save performance history to file"""
        try:
            with self.performance_lock:
                with open(self.performance_file, 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")
    
    def _save_strategy_profiles(self):
        """Save strategy profiles to file"""
        try:
            with self.profiles_lock:
                with open(self.strategy_profiles_file, 'w') as f:
                    json.dump(self.strategy_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving strategy profiles: {str(e)}")
    
    def _save_learning_results(self):
        """Save learning results to file"""
        try:
            with self.learning_lock:
                with open(self.learning_results_file, 'w') as f:
                    json.dump(self.learning_results, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning results: {str(e)}")
    
    def _on_portfolio_update(self, portfolio_data):
        """Handle portfolio updates and record performance"""
        try:
            with self.lock:
                date_str = datetime.now().strftime('%Y-%m-%d')
                
                # Extract daily return
                if "performance_metrics" in portfolio_data:
                    daily_return = portfolio_data["performance_metrics"].get("daily_return", 0.0)
                    self.performance_history["daily_returns"][date_str] = daily_return
                
                # Record market regime if available
                if "market_regime" in portfolio_data:
                    regime = portfolio_data["market_regime"]
                    self.performance_history["market_regimes"][date_str] = regime
                
                # Trim history to max days
                self._trim_history()
                
                # Save updated history
                self._save_performance_history()
                
                # Check if we should run the learning loop
                self._check_learning_schedule()
        except Exception as e:
            logger.error(f"Error handling portfolio update: {str(e)}")
    
    def _on_strategy_performance_update(self, strategy_data):
        """Handle strategy performance updates"""
        try:
            with self.lock:
                date_str = datetime.now().strftime('%Y-%m-%d')
                
                # Create date entry if it doesn't exist
                if date_str not in self.performance_history["strategy_returns"]:
                    self.performance_history["strategy_returns"][date_str] = {}
                
                # Record each strategy's return
                for strategy, metrics in strategy_data.items():
                    if "return" in metrics:
                        self.performance_history["strategy_returns"][date_str][strategy] = metrics["return"]
                
                # Save updated history
                self._save_performance_history()
        except Exception as e:
            logger.error(f"Error handling strategy performance update: {str(e)}")
    
    def _trim_history(self):
        """Trim history to keep only max_history_days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=self.max_history_days)).strftime('%Y-%m-%d')
            
            # Trim daily returns
            self.performance_history["daily_returns"] = {
                date: value for date, value in self.performance_history["daily_returns"].items()
                if date >= cutoff_date
            }
            
            # Trim strategy returns
            self.performance_history["strategy_returns"] = {
                date: value for date, value in self.performance_history["strategy_returns"].items()
                if date >= cutoff_date
            }
            
            # Trim market regimes
            self.performance_history["market_regimes"] = {
                date: value for date, value in self.performance_history["market_regimes"].items()
                if date >= cutoff_date
            }
        except Exception as e:
            logger.error(f"Error trimming history: {str(e)}")
    
    def _check_learning_schedule(self):
        """Check if it's time to run the learning loop"""
        try:
            # Parse last run timestamp
            last_run = datetime.fromisoformat(self.last_learning_run)
            days_since_last_run = (datetime.now() - last_run).total_seconds() / 86400
            
            # Run if it's time
            if days_since_last_run >= self.learning_frequency_days:
                if self._has_sufficient_data():
                    logger.info(f"Starting scheduled learning loop after {days_since_last_run:.1f} days")
                    self.run_learning_loop()
                else:
                    logger.info("Not enough data points for learning loop")
        except Exception as e:
            logger.error(f"Error checking learning schedule: {str(e)}")
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data points for learning"""
        try:
            # Check daily returns history
            if len(self.performance_history["daily_returns"]) < self.min_data_points:
                return False
            
            # Check strategy returns history
            if len(self.performance_history["strategy_returns"]) < self.min_data_points:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking data sufficiency: {str(e)}")
            return False
    
    def run_learning_loop(self) -> Dict:
        """
        Run the complete learning loop to update strategies and parameters
        
        Returns:
            Dict containing the learning results and metrics
        """
        try:
            with self.lock:
                logger.info("Starting continuous learning loop")
                
                # Create containers for results
                results = {
                    "regime_weights": {},
                    "parameter_updates": {},
                    "allocation_adjustments": {},
                    "learning_metrics": {
                        "iterations": 0,
                        "improvement": 0.0,
                        "convergence": False
                    }
                }
                
                # Step 1: Calculate strategy performance in different market regimes
                regime_performance = self._analyze_regime_performance()
                results["regime_weights"] = regime_performance
                
                # Step 2: Analyze parameter sensitivity and optimize
                parameter_updates = self._analyze_parameter_sensitivity()
                results["parameter_updates"] = parameter_updates
                
                # Step 3: Update strategy profiles
                self._update_strategy_profiles(regime_performance, parameter_updates)
                
                # Step 4: Generate allocation adjustments
                allocation_adjustments = self._generate_allocation_adjustments(regime_performance)
                results["allocation_adjustments"] = allocation_adjustments
                
                # Step 5: Apply the adjustments to strategies
                improvement = self._apply_adjustments(allocation_adjustments, parameter_updates)
                
                # Update learning metrics
                results["learning_metrics"] = {
                    "iterations": self.learning_results.get("learning_metrics", {}).get("iterations", 0) + 1,
                    "improvement": improvement,
                    "convergence": improvement < 0.005  # Consider converged if improvement is small
                }
                
                # Update learning results
                self.learning_results = {
                    "last_run": datetime.now().isoformat(),
                    "regime_weights": regime_performance,
                    "parameter_updates": parameter_updates,
                    "allocation_adjustments": allocation_adjustments,
                    "learning_metrics": results["learning_metrics"]
                }
                
                # Save results
                self._save_learning_results()
                self._save_strategy_profiles()
                
                logger.info(f"Completed learning loop with improvement: {improvement:.4f}")
                
                return results
        except Exception as e:
            logger.error(f"Error running learning loop: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _analyze_regime_performance(self) -> Dict:
        """
        Analyze how each strategy performs in different market regimes
        
        Returns:
            Dict mapping strategies to regime performance weights
        """
        try:
            # Get all unique regimes in the history
            regimes = set(self.performance_history["market_regimes"].values())
            strategy_names = set()
            
            # Get all unique strategy names
            for date_data in self.performance_history["strategy_returns"].values():
                strategy_names.update(date_data.keys())
            
            # Create DataFrame from performance history
            performance_data = []
            for date, regime in self.performance_history["market_regimes"].items():
                if date in self.performance_history["strategy_returns"]:
                    strategy_returns = self.performance_history["strategy_returns"][date]
                    row = {"date": date, "regime": regime}
                    row.update(strategy_returns)
                    performance_data.append(row)
            
            if not performance_data:
                logger.warning("No performance data available for regime analysis")
                return {}
            
            # Create DataFrame
            performance_df = pd.DataFrame(performance_data)
            performance_df["date"] = pd.to_datetime(performance_df["date"])
            performance_df = performance_df.set_index("date")
            
            # Calculate regime performance for each strategy
            regime_weights = {}
            
            for strategy in strategy_names:
                if strategy not in performance_df.columns:
                    continue
                    
                regime_perf = {}
                
                # Calculate average return in each regime
                for regime in regimes:
                    regime_data = performance_df[performance_df["regime"] == regime]
                    if len(regime_data) > 0 and strategy in regime_data.columns:
                        avg_return = regime_data[strategy].mean()
                        sharpe = 0.0
                        
                        # Calculate Sharpe if enough data points
                        if len(regime_data) > 5:
                            std = regime_data[strategy].std()
                            if std > 0:
                                sharpe = avg_return / std
                        
                        regime_perf[regime] = {
                            "avg_return": float(avg_return),
                            "sharpe": float(sharpe),
                            "sample_size": len(regime_data)
                        }
                
                # Calculate relative strength of strategy in each regime
                regime_weights[strategy] = regime_perf
            
            return regime_weights
        except Exception as e:
            logger.error(f"Error analyzing regime performance: {str(e)}")
            return {}
    
    def _analyze_parameter_sensitivity(self) -> Dict:
        """
        Analyze strategy parameter sensitivity and find optimal values
        
        Returns:
            Dict of parameter updates for each strategy
        """
        try:
            # Get all strategies with parameter data
            parameter_updates = {}
            
            # Get current market regime
            current_regime = getattr(self.core_context.market_context, "regime", "unknown")
            
            # For each strategy, check parameter sensitivity
            for strategy_name, strategy in self.core_context.strategies.items():
                # Skip if the strategy doesn't support parameter optimization
                if not hasattr(strategy, "get_parameters") or not hasattr(strategy, "analyze_parameter_sensitivity"):
                    continue
                
                # Get current parameters
                current_params = strategy.get_parameters()
                
                # Get parameter sensitivity analysis from the strategy
                sensitivity = strategy.analyze_parameter_sensitivity()
                
                # Use the sensitivity to calculate optimal parameters for current regime
                optimal_params = {}
                
                for param_name, sensitivity_data in sensitivity.items():
                    if param_name in current_params and current_regime in sensitivity_data:
                        # Get current value
                        current_value = current_params[param_name]
                        
                        # Get optimal value for this regime
                        optimal_value = sensitivity_data[current_regime].get("optimal_value", current_value)
                        
                        # Calculate the update, applying learning rate
                        direction = 1 if optimal_value > current_value else -1
                        magnitude = abs(optimal_value - current_value) * self.learning_rate
                        update = current_value + (direction * magnitude)
                        
                        # Add to optimal parameters
                        optimal_params[param_name] = update
                
                # Only add to updates if we have parameter changes
                if optimal_params:
                    parameter_updates[strategy_name] = optimal_params
            
            return parameter_updates
        except Exception as e:
            logger.error(f"Error analyzing parameter sensitivity: {str(e)}")
            return {}
    
    def _update_strategy_profiles(self, regime_weights: Dict, parameter_updates: Dict):
        """Update strategy profiles with the learning results"""
        try:
            # Update regime performance
            if "regime_performance" not in self.strategy_profiles:
                self.strategy_profiles["regime_performance"] = {}
            
            # Update with new regime weights, but preserve history
            for strategy, regime_data in regime_weights.items():
                if strategy not in self.strategy_profiles["regime_performance"]:
                    self.strategy_profiles["regime_performance"][strategy] = {}
                
                # Merge the new data with old data
                for regime, metrics in regime_data.items():
                    self.strategy_profiles["regime_performance"][strategy][regime] = metrics
            
            # Update parameter sensitivity
            if "adaptive_parameters" not in self.strategy_profiles:
                self.strategy_profiles["adaptive_parameters"] = {}
                
            # Record parameter updates
            for strategy, params in parameter_updates.items():
                if strategy not in self.strategy_profiles["adaptive_parameters"]:
                    self.strategy_profiles["adaptive_parameters"][strategy] = {}
                
                # Record the parameter updates with timestamp
                update_record = {
                    "timestamp": datetime.now().isoformat(),
                    "updates": params
                }
                
                # Add to history
                if "history" not in self.strategy_profiles["adaptive_parameters"][strategy]:
                    self.strategy_profiles["adaptive_parameters"][strategy]["history"] = []
                
                self.strategy_profiles["adaptive_parameters"][strategy]["history"].append(update_record)
                
                # Limit history size
                if len(self.strategy_profiles["adaptive_parameters"][strategy]["history"]) > 20:
                    self.strategy_profiles["adaptive_parameters"][strategy]["history"] = \
                        self.strategy_profiles["adaptive_parameters"][strategy]["history"][-20:]
        except Exception as e:
            logger.error(f"Error updating strategy profiles: {str(e)}")
    
    def _generate_allocation_adjustments(self, regime_weights: Dict) -> Dict:
        """
        Generate allocation adjustments based on regime performance
        
        Returns:
            Dict of allocation adjustments for each strategy
        """
        try:
            # Get current market regime
            current_regime = getattr(self.core_context.market_context, "regime", "unknown")
            
            # Default if no regime is known
            if not current_regime:
                logger.warning("No current market regime known, using 'unknown'")
                current_regime = "unknown"
            
            # Get current allocations
            current_allocations = {}
            for strategy_name, strategy in self.core_context.strategies.items():
                if hasattr(strategy, "allocation"):
                    current_allocations[strategy_name] = strategy.allocation
            
            # If no allocation data, can't adjust
            if not current_allocations:
                logger.warning("No current allocation data available")
                return {}
            
            # Calculate adjustment factors based on regime performance
            adjustment_factors = {}
            
            for strategy_name, regime_data in regime_weights.items():
                # Skip if not in current allocations
                if strategy_name not in current_allocations:
                    continue
                
                # Get regime specific performance if available
                if current_regime in regime_data:
                    regime_perf = regime_data[current_regime]
                    
                    # Calculate adjustment factor based on sharpe or return
                    if regime_perf.get("sample_size", 0) > 5:
                        # Prefer Sharpe ratio if available
                        adjustment = regime_perf.get("sharpe", 0) * 2
                    else:
                        # Otherwise use average return
                        adjustment = regime_perf.get("avg_return", 0) * 10
                    
                    # Limit the adjustment factor to a reasonable range
                    adjustment = max(-0.5, min(0.5, adjustment))
                    
                    adjustment_factors[strategy_name] = 1 + adjustment
                else:
                    # No data for this regime, use neutral factor
                    adjustment_factors[strategy_name] = 1.0
            
            # Calculate new allocations
            new_allocations = {}
            total_weighted = 0
            
            for strategy_name, current_alloc in current_allocations.items():
                factor = adjustment_factors.get(strategy_name, 1.0)
                weighted_alloc = current_alloc * factor
                new_allocations[strategy_name] = weighted_alloc
                total_weighted += weighted_alloc
            
            # Normalize to 100%
            if total_weighted > 0:
                for strategy_name in new_allocations:
                    new_allocations[strategy_name] = (new_allocations[strategy_name] / total_weighted) * 100.0
            
            # Calculate adjustment amounts
            allocation_adjustments = {}
            for strategy_name, new_alloc in new_allocations.items():
                current = current_allocations.get(strategy_name, 0.0)
                adjustment = new_alloc - current
                
                # Only include non-zero adjustments
                if abs(adjustment) > 0.1:
                    allocation_adjustments[strategy_name] = {
                        "current": current,
                        "new": new_alloc,
                        "adjustment": adjustment,
                        "factor": adjustment_factors.get(strategy_name, 1.0)
                    }
            
            return allocation_adjustments
        except Exception as e:
            logger.error(f"Error generating allocation adjustments: {str(e)}")
            return {}
    
    def _apply_adjustments(self, allocation_adjustments: Dict, parameter_updates: Dict) -> float:
        """
        Apply allocation and parameter adjustments to strategies
        
        Returns:
            float representing the magnitude of improvement
        """
        try:
            improvement_score = 0.0
            
            # Apply allocation adjustments
            for strategy_name, adjustment in allocation_adjustments.items():
                # Update allocation in the core context
                new_allocation = adjustment["new"]
                self.core_context.update_strategy(strategy_name, {"allocation": new_allocation})
                
                # Add to improvement score
                improvement_score += abs(adjustment["adjustment"]) / 100.0
            
            # Apply parameter updates
            for strategy_name, params in parameter_updates.items():
                # Get strategy
                strategy = self.core_context.strategies.get(strategy_name)
                
                if strategy and hasattr(strategy, "update_parameters"):
                    # Update the parameters
                    strategy.update_parameters(params)
                    
                    # Add to improvement score based on parameter changes
                    for param, value in params.items():
                        # Get current value
                        current = getattr(strategy, param, None)
                        
                        # Calculate relative change
                        if current is not None and current != 0:
                            relative_change = abs((value - current) / current)
                            improvement_score += relative_change * 0.1
            
            return improvement_score
        except Exception as e:
            logger.error(f"Error applying adjustments: {str(e)}")
            return 0.0
    
    def get_learning_report(self) -> Dict:
        """
        Generate a report of recent learning activity
        
        Returns:
            Dict containing learning metrics and insights
        """
        try:
            with self.lock:
                # Get last run info
                last_run_date = datetime.fromisoformat(self.last_learning_run)
                days_ago = (datetime.now() - last_run_date).total_seconds() / 86400
                
                # Compile report
                report = {
                    "last_run": self.last_learning_run,
                    "days_since_last_run": round(days_ago, 1),
                    "next_scheduled_run": (last_run_date + timedelta(days=self.learning_frequency_days)).isoformat(),
                    "learning_metrics": self.learning_results.get("learning_metrics", {}),
                    "regime_insights": {},
                    "allocation_changes": self.learning_results.get("allocation_adjustments", {}),
                    "parameter_changes": self.learning_results.get("parameter_updates", {})
                }
                
                # Add regime insights
                for strategy, regime_data in self.strategy_profiles.get("regime_performance", {}).items():
                    # Find best and worst regimes
                    best_regime = max(regime_data.items(), key=lambda x: x[1].get("sharpe", 0)) if regime_data else None
                    worst_regime = min(regime_data.items(), key=lambda x: x[1].get("sharpe", 0)) if regime_data else None
                    
                    if best_regime and worst_regime:
                        report["regime_insights"][strategy] = {
                            "best_regime": {
                                "name": best_regime[0],
                                "sharpe": best_regime[1].get("sharpe", 0),
                                "avg_return": best_regime[1].get("avg_return", 0)
                            },
                            "worst_regime": {
                                "name": worst_regime[0],
                                "sharpe": worst_regime[1].get("sharpe", 0),
                                "avg_return": worst_regime[1].get("avg_return", 0)
                            }
                        }
                
                return report
        except Exception as e:
            logger.error(f"Error generating learning report: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def force_learning_run(self) -> Dict:
        """Force a learning run even if not scheduled"""
        logger.info("Forcing learning run")
        return self.run_learning_loop() 