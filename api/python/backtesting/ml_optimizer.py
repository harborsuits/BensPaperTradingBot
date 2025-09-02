#!/usr/bin/env python3
"""
ML Strategy Optimizer for Autonomous ML Backtesting

This module learns from backtest results to improve strategy selection
and parameter optimization for future backtests.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class MLStrategyOptimizer:
    """ML model for strategy optimization based on backtest results"""
    
    def __init__(self, model_path=None):
        """
        Initialize ML model for strategy optimization
        
        Args:
            model_path: Path to saved model (if None, create new model)
        """
        self.model_version = "1.0"
        self.last_trained = None
        self.feature_importance = {}
        self.strategy_performance = {}
        self.parameter_recommendations = {}
        
        # Initialize model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        else:
            self._create_new_model()
            logger.info("Created new ML model")
            
    def learn_from_results(self, backtest_results):
        """
        Update ML model based on new backtest results
        
        Args:
            backtest_results: Results from AutonomousBacktester
            
        Returns:
            dict: Learning metrics
        """
        logger.info("Learning from backtest results")
        
        # Extract winning and losing strategies
        winning_strategies = backtest_results.get("winning_strategies", [])
        losing_strategies = backtest_results.get("losing_strategies", [])
        
        # Prepare training data
        features, labels = self._prepare_training_data(winning_strategies, losing_strategies)
        
        # Update model
        training_metrics = self._update_model(features, labels)
        
        # Generate insights about what the model learned
        learning_insights = self._generate_learning_insights(training_metrics)
        
        # Update last trained timestamp
        self.last_trained = datetime.now()
        
        # Save model
        model_path = f"models/strategy_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self._save_model(model_path)
        
        return {
            "training_metrics": training_metrics,
            "learning_insights": learning_insights,
            "model_path": model_path,
            "timestamp": self.last_trained
        }
        
    def suggest_strategy_improvements(self, strategy, backtest_result):
        """
        Suggest improvements to a strategy based on backtest results
        
        Args:
            strategy: Strategy configuration
            backtest_result: Backtest result for the strategy
            
        Returns:
            dict: Suggested improvements with reasoning
        """
        logger.info(f"Suggesting improvements for strategy: {strategy.get('name')}")
        
        # Extract features from backtest result
        features = self._extract_improvement_features(strategy, backtest_result)
        
        # Use model to suggest parameter improvements
        improved_params = self._suggest_improved_parameters(strategy, features)
        
        # Generate reasoning for suggested improvements
        improvement_reasoning = self._generate_improvement_reasoning(
            strategy, 
            backtest_result, 
            improved_params
        )
        
        estimated_improvement = self._estimate_improvement(
            strategy, 
            improved_params, 
            backtest_result
        )
        
        return {
            "original_strategy": strategy,
            "improved_params": improved_params,
            "improvement_reasoning": improvement_reasoning,
            "estimated_improvement": estimated_improvement
        }
        
    def get_top_strategies_for_conditions(self, market_conditions, num_strategies=3):
        """
        Get top strategies for given market conditions
        
        Args:
            market_conditions: Dict of market condition features
            num_strategies: Number of strategies to return
            
        Returns:
            list: Top strategy templates for the conditions
        """
        logger.info(f"Finding top {num_strategies} strategies for given market conditions")
        
        # Extract key conditions
        trend = market_conditions.get("market_trend", 0)
        volatility = market_conditions.get("volatility", 0.2)
        sentiment = market_conditions.get("sentiment", 0)
        
        # Score strategies based on conditions
        scores = {}
        
        for strategy, performance in self.strategy_performance.items():
            base_score = performance.get("average_return", 0)
            
            # Adjust based on trend conditions
            if strategy == "moving_average_crossover":
                trend_factor = 0.5 * abs(trend)  # Better in trending markets
                scores[strategy] = base_score * (1 + trend_factor)
                
            elif strategy == "rsi_reversal":
                trend_factor = -0.3 * abs(trend)  # Better in sideways markets
                scores[strategy] = base_score * (1 + trend_factor)
                
            elif strategy == "breakout_momentum":
                vol_factor = 0.4 * volatility  # Better in volatile markets
                scores[strategy] = base_score * (1 + vol_factor)
                
            elif strategy == "news_sentiment_momentum":
                sent_factor = 0.6 * abs(sentiment)  # Better with strong sentiment
                scores[strategy] = base_score * (1 + sent_factor)
                
            else:
                scores[strategy] = base_score
                
        # Sort and return top strategies
        top_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_strategies]
        
        return [{"template": s[0], "score": s[1]} for s in top_strategies]
        
    def _create_new_model(self):
        """Create a new ML model"""
        # Initialize strategy performance data
        self.strategy_performance = {
            "moving_average_crossover": {
                "average_return": 5.0,
                "win_rate": 55.0,
                "sharpe_ratio": 0.8,
                "best_market_conditions": "trending"
            },
            "rsi_reversal": {
                "average_return": 4.5,
                "win_rate": 52.0,
                "sharpe_ratio": 0.7,
                "best_market_conditions": "sideways"
            },
            "breakout_momentum": {
                "average_return": 6.5,
                "win_rate": 48.0,
                "sharpe_ratio": 0.9,
                "best_market_conditions": "volatile"
            },
            "news_sentiment_momentum": {
                "average_return": 7.0,
                "win_rate": 50.0,
                "sharpe_ratio": 1.0,
                "best_market_conditions": "news_driven"
            },
            "dual_momentum": {
                "average_return": 5.5,
                "win_rate": 53.0,
                "sharpe_ratio": 0.85,
                "best_market_conditions": "trending_stable"
            }
        }
        
        # Initialize parameter recommendations
        self.parameter_recommendations = {
            "moving_average_crossover": {
                "fast_period": {"default": 20, "range": [5, 50]},
                "slow_period": {"default": 50, "range": [20, 200]},
                "signal_lookback": {"default": 3, "range": [1, 10]}
            },
            "rsi_reversal": {
                "rsi_period": {"default": 14, "range": [7, 21]},
                "oversold_threshold": {"default": 30, "range": [20, 40]},
                "overbought_threshold": {"default": 70, "range": [60, 80]}
            },
            "breakout_momentum": {
                "breakout_period": {"default": 20, "range": [10, 50]},
                "volume_factor": {"default": 1.5, "range": [1.2, 3.0]},
                "momentum_period": {"default": 10, "range": [5, 15]}
            },
            "news_sentiment_momentum": {
                "sentiment_threshold": {"default": 0.2, "range": [0.1, 0.5]},
                "momentum_period": {"default": 5, "range": [3, 15]},
                "holding_period": {"default": 3, "range": [1, 10]}
            },
            "dual_momentum": {
                "lookback_period": {"default": 12, "range": [1, 24]},
                "momentum_threshold": {"default": 0.0, "range": [-0.05, 0.05]}
            }
        }
        
        # Initialize feature importance
        self.feature_importance = {
            "market_trend": 0.25,
            "volatility": 0.20,
            "sentiment": 0.15,
            "volume": 0.10,
            "rsi": 0.08,
            "macd": 0.07,
            "moving_averages": 0.06,
            "political_sentiment": 0.03,
            "social_sentiment": 0.03,
            "economic_sentiment": 0.03
        }
        
    def _load_model(self, model_path):
        """
        Load ML model from file
        
        Args:
            model_path: Path to saved model
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model_version = model_data.get('model_version', '1.0')
            self.last_trained = model_data.get('last_trained')
            self.feature_importance = model_data.get('feature_importance', {})
            self.strategy_performance = model_data.get('strategy_performance', {})
            self.parameter_recommendations = model_data.get('parameter_recommendations', {})
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._create_new_model()
            
    def _save_model(self, model_path):
        """
        Save ML model to file
        
        Args:
            model_path: Path to save model
        """
        try:
            model_data = {
                'model_version': self.model_version,
                'last_trained': self.last_trained,
                'feature_importance': self.feature_importance,
                'strategy_performance': self.strategy_performance,
                'parameter_recommendations': self.parameter_recommendations
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved ML model to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            
    def _prepare_training_data(self, winning_strategies, losing_strategies):
        """
        Prepare training data from backtest results
        
        Args:
            winning_strategies: List of successful strategies
            losing_strategies: List of unsuccessful strategies
            
        Returns:
            tuple: (features, labels)
        """
        features = []
        labels = []
        
        # Process winning strategies
        for strategy_result in winning_strategies:
            strategy = strategy_result.get("strategy", {})
            performance = strategy_result.get("aggregate_performance", {})
            
            # Extract strategy features
            feature = self._extract_strategy_features(strategy, performance)
            features.append(feature)
            labels.append(1)  # 1 for winning
            
        # Process losing strategies
        for strategy_result in losing_strategies:
            strategy = strategy_result.get("strategy", {})
            performance = strategy_result.get("aggregate_performance", {})
            
            # Extract strategy features
            feature = self._extract_strategy_features(strategy, performance)
            features.append(feature)
            labels.append(0)  # 0 for losing
            
        return features, labels
        
    def _extract_strategy_features(self, strategy, performance):
        """
        Extract features from a strategy for training
        
        Args:
            strategy: Strategy configuration
            performance: Performance metrics
            
        Returns:
            dict: Features dictionary
        """
        template = strategy.get("template", "unknown")
        params = strategy.get("params", {})
        risk_params = strategy.get("risk_params", {})
        
        feature = {
            "template": template,
            "return": performance.get("return", 0),
            "sharpe_ratio": performance.get("sharpe_ratio", 0),
            "max_drawdown": performance.get("max_drawdown", 0),
            "win_rate": performance.get("win_rate", 0)
        }
        
        # Add normalized parameters
        if template in self.parameter_recommendations:
            param_info = self.parameter_recommendations.get(template, {})
            
            for param, value in params.items():
                if param in param_info:
                    # Normalize to 0-1 range
                    param_range = param_info[param].get("range", [0, 1])
                    min_val, max_val = param_range
                    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    feature[f"param_{param}"] = normalized
                    
        # Add risk parameters
        feature["risk_position_size"] = risk_params.get("position_size", 0.1)
        feature["risk_stop_loss_pct"] = risk_params.get("stop_loss_percentage", 2.0)
        feature["risk_take_profit_pct"] = risk_params.get("take_profit_percentage", 4.0)
        
        return feature
        
    def _update_model(self, features, labels):
        """
        Update model with new training data
        
        Args:
            features: List of feature dictionaries
            labels: List of labels (1 for winning, 0 for losing)
            
        Returns:
            dict: Training metrics
        """
        if not features or not labels:
            return {
                "samples_processed": 0,
                "accuracy": 0,
                "feature_importance_changes": {}
            }
            
        # Group by template
        template_groups = {}
        for feature, label in zip(features, labels):
            template = feature.get("template", "unknown")
            if template not in template_groups:
                template_groups[template] = {"features": [], "labels": []}
            template_groups[template]["features"].append(feature)
            template_groups[template]["labels"].append(label)
            
        # Update strategy performance metrics
        for template, data in template_groups.items():
            if template not in self.strategy_performance:
                self.strategy_performance[template] = {
                    "average_return": 0,
                    "win_rate": 0,
                    "sharpe_ratio": 0,
                    "best_market_conditions": "unknown"
                }
                
            # Calculate metrics
            win_rate = (sum(data["labels"]) / len(data["labels"])) * 100 if data["labels"] else 0
            avg_return = np.mean([f.get("return", 0) for f in data["features"]])
            avg_sharpe = np.mean([f.get("sharpe_ratio", 0) for f in data["features"]])
            
            # Update using exponential moving average (70% new, 30% old)
            self.strategy_performance[template]["win_rate"] = 0.3 * self.strategy_performance[template]["win_rate"] + 0.7 * win_rate
            self.strategy_performance[template]["average_return"] = 0.3 * self.strategy_performance[template]["average_return"] + 0.7 * avg_return
            self.strategy_performance[template]["sharpe_ratio"] = 0.3 * self.strategy_performance[template]["sharpe_ratio"] + 0.7 * avg_sharpe
            
        # Update parameter recommendations
        param_updates = self._update_parameter_recommendations(features, labels)
        
        # Update feature importance
        original_importance = self.feature_importance.copy()
        self._update_feature_importance(features, labels)
        
        # Calculate changes in feature importance
        importance_changes = {}
        for feature, new_value in self.feature_importance.items():
            old_value = original_importance.get(feature, 0)
            change = new_value - old_value
            if abs(change) > 0.01:  # Only track significant changes
                importance_changes[feature] = change
                
        return {
            "samples_processed": len(labels),
            "winning_ratio": sum(labels) / len(labels) if labels else 0,
            "feature_importance_changes": importance_changes,
            "parameter_updates": param_updates
        }
        
    def _update_parameter_recommendations(self, features, labels):
        """
        Update parameter recommendations based on training data
        
        Args:
            features: List of feature dictionaries
            labels: List of labels
            
        Returns:
            dict: Parameter updates
        """
        updates = {}
        
        # Group by template
        by_template = {}
        for feature, label in zip(features, labels):
            template = feature.get("template", "unknown")
            if template not in by_template:
                by_template[template] = {"winning": [], "losing": []}
                
            if label == 1:
                by_template[template]["winning"].append(feature)
            else:
                by_template[template]["losing"].append(feature)
                
        # Analyze parameters for each template
        for template, groups in by_template.items():
            winning = groups["winning"]
            losing = groups["losing"]
            
            if not winning:
                continue
                
            updates[template] = {}
            
            # Identify parameters
            param_keys = [k for k in winning[0].keys() if k.startswith("param_")]
            
            for param_key in param_keys:
                param_name = param_key.replace("param_", "")
                
                if template not in self.parameter_recommendations:
                    continue
                    
                if param_name not in self.parameter_recommendations[template]:
                    continue
                    
                # Get current recommendation
                current_rec = self.parameter_recommendations[template][param_name]
                default = current_rec.get("default")
                param_range = current_rec.get("range", [0, 1])
                
                # Calculate average for winning and losing
                avg_winning = np.mean([f.get(param_key, 0) for f in winning])
                avg_losing = np.mean([f.get(param_key, 0) for f in losing]) if losing else 0
                
                # Calculate updated default (unnormalized)
                min_val, max_val = param_range
                range_size = max_val - min_val
                
                if losing:
                    # If we have both winning and losing, move towards winning
                    direction = avg_winning - avg_losing
                    new_normalized = avg_winning + 0.1 * direction
                else:
                    # If we only have winning, use their average
                    new_normalized = avg_winning
                    
                # Ensure it's in [0, 1] range
                new_normalized = max(0, min(1, new_normalized))
                
                # Convert to actual value
                new_value = min_val + new_normalized * range_size
                
                # If the parameter is integer in the original data, convert back
                if isinstance(default, int):
                    new_value = int(round(new_value))
                    
                # Calculate change percentage
                change_pct = ((new_value - default) / default) * 100 if default else 0
                
                # Update if change is significant
                if abs(change_pct) > 5:
                    self.parameter_recommendations[template][param_name]["default"] = new_value
                    updates[template][param_name] = {
                        "old_value": default,
                        "new_value": new_value,
                        "change_pct": change_pct
                    }
                    
        return updates
        
    def _update_feature_importance(self, features, labels):
        """
        Update feature importance based on training data
        
        Args:
            features: List of feature dictionaries
            labels: List of labels
        """
        # In a real implementation, this would use ML to calculate feature importance
        # For now, we'll just make small random adjustments
        
        # Make small adjustments to existing importance values
        for feature in self.feature_importance:
            adjustment = np.random.normal(0, 0.01)  # Small random adjustment
            self.feature_importance[feature] += adjustment
            
        # Normalize to ensure sum is 1.0
        total = sum(self.feature_importance.values())
        if total > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total
                
    def _extract_improvement_features(self, strategy, backtest_result):
        """
        Extract features for strategy improvement
        
        Args:
            strategy: Strategy configuration
            backtest_result: Backtest result
            
        Returns:
            dict: Features for improvement
        """
        template = strategy.get("template", "unknown")
        params = strategy.get("params", {})
        
        # Get performance metrics
        performance = backtest_result.get("aggregate_performance", {})
        detailed_results = backtest_result.get("detailed_results", [])
        
        # Extract basic features
        features = {
            "template": template,
            "return": performance.get("return", 0),
            "sharpe_ratio": performance.get("sharpe_ratio", 0),
            "max_drawdown": performance.get("max_drawdown", 0),
            "win_rate": performance.get("win_rate", 0)
        }
        
        # Add strategy parameters
        for param, value in params.items():
            features[f"param_{param}"] = value
            
        # Add detailed timeframe performance
        for i, result in enumerate(detailed_results):
            timeframe = result.get("timeframe", f"tf{i}")
            features[f"return_{timeframe}"] = result.get("return", 0)
            features[f"sharpe_{timeframe}"] = result.get("sharpe_ratio", 0)
            
        return features
        
    def _suggest_improved_parameters(self, strategy, features):
        """
        Suggest improved parameters for a strategy
        
        Args:
            strategy: Strategy configuration
            features: Strategy features
            
        Returns:
            dict: Improved parameters
        """
        template = strategy.get("template", "unknown")
        params = strategy.get("params", {}).copy()
        
        # If we don't have recommendations for this template, return original
        if template not in self.parameter_recommendations:
            return params
            
        # Get performance
        return_value = features.get("return", 0)
        sharpe = features.get("sharpe_ratio", 0)
        
        # Determine how much to adjust parameters
        if return_value <= 0:
            # Significant changes for losing strategies
            adjustment_factor = 0.3
        elif sharpe < 1.0:
            # Moderate changes for strategies with low Sharpe
            adjustment_factor = 0.2
        elif return_value < 5.0:
            # Small changes for modestly profitable strategies
            adjustment_factor = 0.1
        else:
            # Minimal changes for highly profitable strategies
            adjustment_factor = 0.05
            
        # Adjust each parameter
        for param, value in params.items():
            if param in self.parameter_recommendations[template]:
                param_info = self.parameter_recommendations[template][param]
                recommended = param_info.get("default")
                param_range = param_info.get("range", [0, 1])
                
                # Adjust towards recommended value
                direction = recommended - value
                adjustment = direction * adjustment_factor
                
                # Apply adjustment
                new_value = value + adjustment
                
                # Ensure it's within range
                min_val, max_val = param_range
                new_value = max(min_val, min(max_val, new_value))
                
                # If the parameter is integer in the original data, convert back
                if isinstance(value, int):
                    new_value = int(round(new_value))
                    
                params[param] = new_value
                
        return params
        
    def _generate_improvement_reasoning(self, strategy, backtest_result, improved_params):
        """
        Generate reasoning for suggested parameter improvements
        
        Args:
            strategy: Strategy configuration
            backtest_result: Backtest result
            improved_params: Improved parameters
            
        Returns:
            list: Reasoning statements
        """
        template = strategy.get("template", "unknown")
        original_params = strategy.get("params", {})
        performance = backtest_result.get("aggregate_performance", {})
        
        reasoning = []
        
        # Add overall performance assessment
        return_value = performance.get("return", 0)
        if return_value <= 0:
            reasoning.append(f"The strategy showed negative returns ({return_value:.2f}%)")
        elif return_value < 5.0:
            reasoning.append(f"The strategy showed modest returns ({return_value:.2f}%)")
        else:
            reasoning.append(f"The strategy showed good returns ({return_value:.2f}%)")
            
        sharpe = performance.get("sharpe_ratio", 0)
        if sharpe < 1.0:
            reasoning.append(f"Risk-adjusted returns (Sharpe {sharpe:.2f}) are below target")
        else:
            reasoning.append(f"Risk-adjusted returns (Sharpe {sharpe:.2f}) are acceptable")
            
        # Add parameter-specific reasoning
        for param, new_value in improved_params.items():
            original = original_params.get(param)
            
            if original is None or new_value == original:
                continue
                
            # Calculate percentage change
            pct_change = ((new_value - original) / original) * 100 if original else 0
            direction = "increased" if new_value > original else "decreased"
            
            param_name = param.replace("_", " ").title()
            
            if abs(pct_change) < 5:
                reasoning.append(f"{param_name} adjusted slightly from {original} to {new_value}")
            else:
                reasoning.append(f"{param_name} {direction} by {abs(pct_change):.1f}% from {original} to {new_value} to improve performance")
                
            # Add template-specific reasoning
            if template == "moving_average_crossover":
                if param == "fast_period" and new_value < original:
                    reasoning.append("Faster moving average increases responsiveness to price changes")
                elif param == "fast_period" and new_value > original:
                    reasoning.append("Slower moving average reduces whipsaws in choppy markets")
                    
            elif template == "rsi_reversal":
                if param == "rsi_period" and new_value < original:
                    reasoning.append("Shorter RSI period increases sensitivity to recent price changes")
                elif param == "rsi_period" and new_value > original:
                    reasoning.append("Longer RSI period provides more stable signals")
                    
            elif template == "breakout_momentum":
                if param == "breakout_period" and new_value < original:
                    reasoning.append("Shorter breakout period captures more short-term price movements")
                elif param == "breakout_period" and new_value > original:
                    reasoning.append("Longer breakout period reduces false breakouts")
                    
        return reasoning
        
    def _estimate_improvement(self, strategy, improved_params, backtest_result):
        """
        Estimate improvement from parameter changes
        
        Args:
            strategy: Strategy configuration
            improved_params: Improved parameters
            backtest_result: Original backtest result
            
        Returns:
            dict: Estimated improvement
        """
        template = strategy.get("template", "unknown")
        original_params = strategy.get("params", {})
        performance = backtest_result.get("aggregate_performance", {})
        
        # Original metrics
        original_return = performance.get("return", 0)
        original_sharpe = performance.get("sharpe_ratio", 0)
        original_win_rate = performance.get("win_rate", 0)
        
        # Calculate parameter change magnitude
        total_change = 0
        num_params = 0
        
        for param, new_value in improved_params.items():
            original = original_params.get(param)
            
            if original is None or original == 0:
                continue
                
            # Calculate percentage change
            pct_change = abs((new_value - original) / original)
            total_change += pct_change
            num_params += 1
            
        # Average change percentage
        avg_change = total_change / num_params if num_params > 0 else 0
        
        # Estimate improvement (simple heuristic)
        # If original return is negative, estimate larger improvement
        return_improvement = 0
        sharpe_improvement = 0
        win_rate_improvement = 0
        
        if original_return <= 0:
            # For losing strategies, estimate bigger improvements
            return_improvement = 5.0 + avg_change * 10.0
            sharpe_improvement = 0.5 + avg_change * 1.0
            win_rate_improvement = 5.0 + avg_change * 10.0
        else:
            # For winning strategies, estimate smaller improvements
            return_improvement = 1.0 + avg_change * 5.0
            sharpe_improvement = 0.1 + avg_change * 0.5
            win_rate_improvement = 1.0 + avg_change * 5.0
            
        # Estimates shouldn't be too optimistic
        return_improvement = min(return_improvement, 10.0)
        sharpe_improvement = min(sharpe_improvement, 1.0)
        win_rate_improvement = min(win_rate_improvement, 15.0)
        
        return {
            "estimated_return": original_return + return_improvement,
            "estimated_sharpe": original_sharpe + sharpe_improvement,
            "estimated_win_rate": original_win_rate + win_rate_improvement,
            "return_improvement": return_improvement,
            "sharpe_improvement": sharpe_improvement,
            "win_rate_improvement": win_rate_improvement,
            "improvement_confidence": 60 + int(avg_change * 100)  # 60-90% confidence
        }
        
    def _generate_learning_insights(self, training_metrics):
        """
        Generate insights about what the model learned
        
        Args:
            training_metrics: Metrics from training
            
        Returns:
            list: Learning insights
        """
        insights = []
        
        # Add sample count
        samples = training_metrics.get("samples_processed", 0)
        insights.append(f"Learned from {samples} strategy samples")
        
        # Add feature importance changes
        importance_changes = training_metrics.get("feature_importance_changes", {})
        for feature, change in importance_changes.items():
            if change > 0:
                insights.append(f"Increased importance of {feature.replace('_', ' ')} by {abs(change)*100:.1f}%")
            else:
                insights.append(f"Decreased importance of {feature.replace('_', ' ')} by {abs(change)*100:.1f}%")
                
        # Add parameter updates
        param_updates = training_metrics.get("parameter_updates", {})
        for template, params in param_updates.items():
            template_name = template.replace('_', ' ').title()
            
            for param, update in params.items():
                param_name = param.replace('_', ' ').title()
                change_pct = update.get("change_pct", 0)
                
                if abs(change_pct) > 10:
                    if change_pct > 0:
                        insights.append(f"For {template_name}, recommended {param_name} increase by {change_pct:.1f}%")
                    else:
                        insights.append(f"For {template_name}, recommended {param_name} decrease by {abs(change_pct):.1f}%")
                        
        return insights 