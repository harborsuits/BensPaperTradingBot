#!/usr/bin/env python3
"""
Integration Test: Full Strategy Optimization Feedback Loop

This integration test validates the complete closed-loop system for strategy optimization:
1. Initial Strategy Optimization (multi-objective) using synthetic market data
2. Approval Workflow for strategy promotion
3. Performance Tracking in simulated "real-world" conditions
4. Verification against synthetic predictions
5. Automatic adjustment of synthetic market parameters based on verification
6. Re-optimization with improved synthetic parameters
7. Measurement of accuracy improvements over multiple iterations

The key innovation demonstrated is the ability of the system to self-correct
and improve its optimization process by learning from real-world performance.
"""

import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Local imports for synthetic market generation and optimization
from trading_bot.autonomous.synthetic_market_generator import (
    SyntheticMarketGenerator, MarketRegimeType
)

# Import the multi-objective optimizer from our practical example
from practical_optimization_example import (
    ParameterSpace, MultiObjectiveOptimizer, generate_synthetic_market_data,
    TradingStrategy, evaluate_strategy_in_regime
)

# Import approval workflow components
from trading_bot.autonomous.approval_workflow import (
    ApprovalStatus, ApprovalRequest, get_approval_workflow_manager
)

# Import performance verification
from trading_bot.autonomous.performance_verification import (
    PerformanceVerifier, StrategyPerformanceRecord, get_performance_verifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_test")

# Integration test output file
INTEGRATION_TEST_REPORT = "integration_verification_report.txt"


class FeedbackLoopIntegrationTest:
    """
    Integration test demonstrating the complete strategy optimization feedback loop.
    """
    
    def __init__(self, test_iterations=3, output_file=INTEGRATION_TEST_REPORT):
        """
        Initialize the integration test.

        Args:
            test_iterations: Number of feedback loop iterations to run
            output_file: Path to output report file
        """
        self.test_iterations = test_iterations
        self.output_file = output_file
        
        # Initialize components
        self.market_generator = SyntheticMarketGenerator()
        self.approval_manager = get_approval_workflow_manager()
        self.performance_verifier = get_performance_verifier()
        
        # Test tracking
        self.strategies = []
        self.optimization_results = []
        self.verification_results = []
        self.optimization_parameters = []
        self.verification_accuracy = []
        
        # Initialize parameter space for our trading strategy
        self.parameter_space = ParameterSpace([
            ('trend_lookback', 'integer', 5, 50),
            ('volatility_lookback', 'integer', 5, 50),
            ('entry_threshold', 'real', 0.0, 2.0),
            ('exit_threshold', 'real', -2.0, 0.0),
            ('stop_loss_pct', 'real', 0.5, 5.0),
            ('position_sizing', 'real', 0.01, 0.5),
            ('trail_stop', 'boolean', None, None),
            ('filter_type', 'categorical', ['sma', 'ema', 'none'], None),
            ('indicator_weight', 'real', 0.1, 0.9)
        ])
        
        # Initial synthetic market generation parameters
        # These will be refined through the feedback loop
        self.synthetic_market_params = {
            'bullish': {
                'trend_strength': 0.6,
                'volatility': 0.015,
                'mean_reversion': 0.2
            },
            'bearish': {
                'trend_strength': -0.4,
                'volatility': 0.025,
                'mean_reversion': 0.15
            },
            'sideways': {
                'trend_strength': 0.05,
                'volatility': 0.01,
                'mean_reversion': 0.7
            },
            'volatile': {
                'trend_strength': 0.1,
                'volatility': 0.035,
                'mean_reversion': 0.1
            }
        }
        
        # Store the current optimization iteration
        self.current_iteration = 0
        
        logger.info("Initialized FeedbackLoopIntegrationTest")
    
    def run_test(self):
        """
        Run the complete integration test with multiple feedback loop iterations.
        """
        logger.info(f"Starting integration test with {self.test_iterations} iterations")
        
        for iteration in range(self.test_iterations):
            self.current_iteration = iteration + 1
            logger.info(f"=== Iteration {self.current_iteration}/{self.test_iterations} ===")
            
            # Step 1: Optimize strategy parameters using current synthetic parameters
            strategy_id, version_id, parameters = self.optimize_strategy()
            
            # Step 2: Submit for approval and auto-approve
            request_id = self.submit_for_approval(strategy_id, version_id)
            
            # Step 3: Simulate real-world performance (we're creating simulated real-world data)
            real_performance = self.simulate_real_world_performance(strategy_id, version_id, parameters)
            
            # Step 4: Verify against synthetic predictions
            verification_result = self.verify_performance(strategy_id, version_id, request_id, real_performance)
            
            # Step 5: Update synthetic market parameters based on verification
            self.update_synthetic_parameters(verification_result)
            
            # Wait a bit to let logs catch up and for clarity
            time.sleep(1)
        
        # Generate final report
        self.generate_report()
        
        logger.info(f"Integration test completed. Report saved to {self.output_file}")
    
    def optimize_strategy(self) -> Tuple[str, str, Dict[str, Any]]:
        """
        Optimize strategy parameters using the current synthetic market parameters.
        
        Returns:
            Tuple of (strategy_id, version_id, optimized_parameters)
        """
        logger.info("Optimizing strategy parameters using multi-objective optimization...")
        
        # Generate synthetic market data with current parameters
        market_data = {}
        for regime, params in self.synthetic_market_params.items():
            market_data[regime] = generate_synthetic_market_data(
                days=252,
                trend_strength=params['trend_strength'],
                volatility=params['volatility'],
                mean_reversion=params['mean_reversion']
            )
        
        # Track these parameters for analysis
        self.optimization_parameters.append(self.synthetic_market_params.copy())
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            parameter_space=self.parameter_space,
            population_size=30,
            crossover_rate=0.7,
            mutation_rate=0.2
        )
        
        # Define multi-objective fitness functions
        def bull_objective(params):
            strategy = TradingStrategy(params)
            results = evaluate_strategy_in_regime(strategy, market_data['bullish'])
            return results['return_pct']  # Higher is better
            
        def bear_objective(params):
            strategy = TradingStrategy(params)
            results = evaluate_strategy_in_regime(strategy, market_data['bearish'])
            return results['return_pct']  # Higher is better
        
        def drawdown_objective(params):
            # Evaluate across all regimes and return worst drawdown
            drawdowns = []
            for regime in market_data:
                strategy = TradingStrategy(params)
                results = evaluate_strategy_in_regime(strategy, market_data[regime])
                drawdowns.append(results['max_drawdown'])
            return -max(drawdowns)  # Higher (less negative) is better
        
        # Run optimization
        objectives = [
            ("bull_return", bull_objective, True),  # Maximize
            ("bear_return", bear_objective, True),  # Maximize
            ("drawdown", drawdown_objective, True),  # Maximize (minimize drawdown)
        ]
        
        pareto_front, objective_values = optimizer.optimize(
            objectives=objectives,
            generations=15,
            parallelism=1,
            verbose=True
        )
        
        # Select a balanced solution from the pareto front
        best_solution = self._select_balanced_solution(pareto_front, objective_values)
        
        # Generate IDs
        strategy_id = f"strategy_{self.current_iteration}"
        version_id = f"v{self.current_iteration}_{int(time.time())}"
        
        # Log results
        logger.info(f"Optimization complete. Selected parameters:")
        for param, value in best_solution.items():
            logger.info(f"  {param}: {value}")
        
        # Save for tracking
        self.strategies.append({
            'strategy_id': strategy_id,
            'version_id': version_id,
            'parameters': best_solution,
            'iteration': self.current_iteration
        })
        
        # Build synthetic performance predictions
        predictions = {}
        expected_performance = {}
        
        for regime in market_data:
            strategy = TradingStrategy(best_solution)
            results = evaluate_strategy_in_regime(strategy, market_data[regime])
            
            predictions[regime] = {
                'return_pct': results['return_pct'],
                'sharpe': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate']
            }
            
            logger.info(f"  Predicted {regime} performance: Return={results['return_pct']:.2f}%, "
                        f"Sharpe={results['sharpe_ratio']:.2f}, Drawdown={results['max_drawdown']:.2f}%")
        
        self.optimization_results.append({
            'strategy_id': strategy_id,
            'version_id': version_id,
            'predictions': predictions,
            'iteration': self.current_iteration
        })
        
        return strategy_id, version_id, best_solution
    
    def _select_balanced_solution(self, pareto_front, objective_values):
        """
        Select a balanced solution from the Pareto front that performs well across all objectives.
        
        Args:
            pareto_front: List of Pareto-optimal parameter sets
            objective_values: Corresponding objective values
            
        Returns:
            Selected parameter set
        """
        # Normalize objective values to 0-1 range
        normalized_objectives = []
        
        for i, obj in enumerate(objective_values):
            obj_values = [values[i] for values in objective_values]
            min_val = min(obj_values)
            max_val = max(obj_values)
            range_val = max_val - min_val if max_val > min_val else 1.0
            
            normalized = [(v - min_val) / range_val for v in obj_values]
            normalized_objectives.append(normalized)
        
        # Calculate distance from ideal point (1,1,1)
        distances = []
        for i in range(len(pareto_front)):
            norm_vals = [normalized_objectives[j][i] for j in range(len(objective_values[0]))]
            distance = sum((1 - v) ** 2 for v in norm_vals) ** 0.5
            distances.append(distance)
        
        # Select solution with minimum distance (most balanced)
        best_idx = distances.index(min(distances))
        return pareto_front[best_idx]
    
    def submit_for_approval(self, strategy_id, version_id) -> str:
        """
        Submit the optimized strategy for approval.
        
        Args:
            strategy_id: ID of the strategy
            version_id: Version ID of the strategy
        
        Returns:
            Request ID
        """
        logger.info(f"Submitting strategy {strategy_id} version {version_id} for approval...")
        
        # Create approval request
        test_id = f"test_{self.current_iteration}_{int(time.time())}"
        request = self.approval_manager.create_request(
            test_id=test_id,
            strategy_id=strategy_id,
            version_id=version_id,
            requester="optimization_integration_test"
        )
        
        logger.info(f"Created approval request {request.request_id}")
        
        # Auto-approve the request (in a real system, this would be done by a human)
        self.approval_manager.approve_request(
            request_id=request.request_id,
            reviewer="integration_test_automation",
            comments=f"Auto-approved for integration test iteration {self.current_iteration}"
        )
        
        logger.info(f"Request {request.request_id} approved")
        
        return request.request_id
    
    def simulate_real_world_performance(self, strategy_id, version_id, parameters) -> Dict[str, Any]:
        """
        Simulate real-world performance of the approved strategy.
        
        In a real system, this would be actual trading performance data.
        For testing, we simulate "real world" data that has some divergence
        from our synthetic predictions.
        
        Args:
            strategy_id: Strategy ID
            version_id: Version ID
            parameters: Strategy parameters
            
        Returns:
            Performance metrics
        """
        logger.info(f"Simulating real-world performance for {strategy_id} {version_id}...")
        
        # Create a trading strategy with the optimized parameters
        strategy = TradingStrategy(parameters)
        
        # Determine current market regime (in reality, this would be the actual market)
        # For testing, we randomly select a regime with some bias
        regimes = list(self.synthetic_market_params.keys())
        weights = [0.4, 0.3, 0.2, 0.1]  # Slight bias toward bullish and bearish
        current_regime = random.choices(regimes, weights=weights, k=1)[0]
        
        logger.info(f"Current real-world market regime: {current_regime}")
        
        # Generate "real world" market data with some randomness
        # In reality, this would be actual market data
        real_params = self.synthetic_market_params[current_regime].copy()
        
        # Add some random variation to represent the gap between our model and reality
        real_params['trend_strength'] *= random.uniform(0.8, 1.2)
        real_params['volatility'] *= random.uniform(0.9, 1.1)
        real_params['mean_reversion'] *= random.uniform(0.9, 1.1)
        
        real_market_data = generate_synthetic_market_data(
            days=252,
            trend_strength=real_params['trend_strength'],
            volatility=real_params['volatility'],
            mean_reversion=real_params['mean_reversion']
        )
        
        # Evaluate strategy on "real" data
        results = evaluate_strategy_in_regime(strategy, real_market_data)
        
        # Create performance record
        performance = {
            'strategy_id': strategy_id,
            'version_id': version_id,
            'regime': current_regime,
            'return_pct': results['return_pct'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'trade_count': results['trade_count'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Real-world performance: Return={results['return_pct']:.2f}%, "
                    f"Sharpe={results['sharpe_ratio']:.2f}, Drawdown={results['max_drawdown']:.2f}%")
        
        return performance
    
    def verify_performance(self, strategy_id, version_id, request_id, real_performance) -> Dict[str, Any]:
        """
        Verify real-world performance against synthetic predictions.
        
        Args:
            strategy_id: Strategy ID
            version_id: Version ID
            request_id: Approval request ID
            real_performance: Real-world performance data
            
        Returns:
            Verification results
        """
        logger.info(f"Verifying performance for {strategy_id} {version_id}...")
        
        # Find the synthetic predictions for this strategy
        predictions = None
        for result in self.optimization_results:
            if result['strategy_id'] == strategy_id and result['version_id'] == version_id:
                predictions = result['predictions']
                break
        
        if not predictions:
            logger.error(f"No predictions found for {strategy_id} {version_id}")
            return {}
        
        # Get the actual regime that occurred
        actual_regime = real_performance['regime']
        
        # Get predicted performance for that regime
        predicted = predictions[actual_regime]
        
        # Calculate accuracy metrics
        return_accuracy = 1.0 - min(1.0, abs(
            predicted['return_pct'] - real_performance['return_pct']) / max(1.0, abs(predicted['return_pct'])))
        sharpe_accuracy = 1.0 - min(1.0, abs(
            predicted['sharpe'] - real_performance['sharpe_ratio']) / max(1.0, abs(predicted['sharpe'])))
        drawdown_accuracy = 1.0 - min(1.0, abs(
            predicted['max_drawdown'] - real_performance['max_drawdown']) / max(1.0, abs(predicted['max_drawdown'])))
        
        overall_accuracy = (return_accuracy + sharpe_accuracy + drawdown_accuracy) / 3.0
        
        # Store verification results
        verification_result = {
            'strategy_id': strategy_id,
            'version_id': version_id,
            'request_id': request_id,
            'actual_regime': actual_regime,
            'predicted': predicted,
            'actual': {
                'return_pct': real_performance['return_pct'],
                'sharpe': real_performance['sharpe_ratio'],
                'max_drawdown': real_performance['max_drawdown'],
                'win_rate': real_performance['win_rate']
            },
            'accuracy': {
                'return': return_accuracy,
                'sharpe': sharpe_accuracy,
                'drawdown': drawdown_accuracy,
                'overall': overall_accuracy
            },
            'iteration': self.current_iteration
        }
        
        self.verification_results.append(verification_result)
        self.verification_accuracy.append(overall_accuracy)
        
        logger.info(f"Verification results:")
        logger.info(f"  Overall accuracy: {overall_accuracy:.2%}")
        logger.info(f"  Return accuracy: {return_accuracy:.2%}")
        logger.info(f"  Sharpe accuracy: {sharpe_accuracy:.2%}")
        logger.info(f"  Drawdown accuracy: {drawdown_accuracy:.2%}")
        
        return verification_result
    
    def update_synthetic_parameters(self, verification_result):
        """
        Update synthetic market parameters based on verification results.
        
        This implements the feedback loop where our synthetic market parameters
        are adjusted to better match real-world behavior.
        
        Args:
            verification_result: Results from verification
        """
        if not verification_result:
            logger.warning("No verification result available, skipping parameter update")
            return
        
        logger.info("Updating synthetic market parameters based on verification...")
        
        # Extract current parameters for the observed regime
        regime = verification_result['actual_regime']
        current_params = self.synthetic_market_params[regime]
        
        # Calculate ratios between actual and predicted metrics
        predicted = verification_result['predicted']
        actual = verification_result['actual']
        
        return_ratio = actual['return_pct'] / predicted['return_pct'] if predicted['return_pct'] != 0 else 1.0
        volatility_ratio = actual['max_drawdown'] / predicted['max_drawdown'] if predicted['max_drawdown'] != 0 else 1.0
        
        # Adjust parameters using a learning rate to prevent over-correction
        learning_rate = 0.2
        
        # Update trend strength based on return ratio
        # If actual returns are higher than predicted, increase trend strength
        current_params['trend_strength'] *= (1.0 + (return_ratio - 1.0) * learning_rate)
        
        # Update volatility based on drawdown ratio
        # If actual drawdowns are higher than predicted, increase volatility
        current_params['volatility'] *= (1.0 + (volatility_ratio - 1.0) * learning_rate)
        
        # Clamp parameters to reasonable ranges
        current_params['trend_strength'] = max(-0.8, min(0.8, current_params['trend_strength']))
        current_params['volatility'] = max(0.005, min(0.05, current_params['volatility']))
        
        logger.info(f"Updated parameters for {regime} regime:")
        logger.info(f"  Trend strength: {current_params['trend_strength']:.4f}")
        logger.info(f"  Volatility: {current_params['volatility']:.4f}")
        logger.info(f"  Mean reversion: {current_params['mean_reversion']:.4f}")
    
    def generate_report(self):
        """
        Generate a comprehensive report of the integration test results.
        """
        logger.info("Generating integration test report...")
        
        # Calculate average accuracy per iteration
        iteration_accuracy = {}
        for result in self.verification_results:
            iteration = result['iteration']
            accuracy = result['accuracy']['overall']
            
            if iteration not in iteration_accuracy:
                iteration_accuracy[iteration] = []
            
            iteration_accuracy[iteration].append(accuracy)
        
        # Calculate averages
        avg_accuracy_by_iteration = {
            iteration: sum(accuracies) / len(accuracies)
            for iteration, accuracies in iteration_accuracy.items()
        }
        
        # Generate report text
        report_lines = [
            "===============================================================",
            "     INTEGRATION TEST: STRATEGY OPTIMIZATION FEEDBACK LOOP     ",
            "===============================================================",
            "",
            f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of iterations: {self.test_iterations}",
            "",
            "===============================================================",
            "                  VERIFICATION ACCURACY                        ",
            "===============================================================",
            ""
        ]
        
        # Add accuracy by iteration
        report_lines.append("Verification Accuracy by Iteration:")
        report_lines.append("------------------------------------")
        for iteration, accuracy in sorted(avg_accuracy_by_iteration.items()):
            report_lines.append(f"Iteration {iteration}: {accuracy:.2%}")
        
        report_lines.extend(["", ""])
        
        # Add individual verification results
        report_lines.append("===============================================================")
        report_lines.append("              DETAILED VERIFICATION RESULTS                   ")
        report_lines.append("===============================================================")
        report_lines.append("")
        
        for result in self.verification_results:
            report_lines.append(f"Strategy: {result['strategy_id']} - {result['version_id']}")
            report_lines.append(f"Iteration: {result['iteration']}")
            report_lines.append(f"Actual Market Regime: {result['actual_regime']}")
            report_lines.append("")
            
            report_lines.append("Predicted vs Actual Performance:")
            report_lines.append("-------------------------------")
            report_lines.append(f"Return:   Predicted={result['predicted']['return_pct']:.2f}%  "
                               f"Actual={result['actual']['return_pct']:.2f}%  "
                               f"Accuracy={result['accuracy']['return']:.2%}")
            report_lines.append(f"Sharpe:   Predicted={result['predicted']['sharpe']:.2f}  "
                               f"Actual={result['actual']['sharpe']:.2f}  "
                               f"Accuracy={result['accuracy']['sharpe']:.2%}")
            report_lines.append(f"Drawdown: Predicted={result['predicted']['max_drawdown']:.2f}%  "
                               f"Actual={result['actual']['max_drawdown']:.2f}%  "
                               f"Accuracy={result['accuracy']['drawdown']:.2%}")
            report_lines.append("")
            report_lines.append(f"Overall Accuracy: {result['accuracy']['overall']:.2%}")
            report_lines.append("")
            report_lines.append("-" * 65)
            report_lines.append("")
        
        # Add summary and recommendations
        report_lines.append("")
        report_lines.append("===============================================================")
        report_lines.append("                 SUMMARY AND CONCLUSIONS                      ")
        report_lines.append("===============================================================")
        report_lines.append("")
        
        # Calculate improvement
        if self.test_iterations > 1:
            first_accuracy = avg_accuracy_by_iteration.get(1, 0)
            last_accuracy = avg_accuracy_by_iteration.get(self.test_iterations, 0)
            improvement = (last_accuracy - first_accuracy) / first_accuracy if first_accuracy > 0 else 0
            
            report_lines.append(f"Initial verification accuracy: {first_accuracy:.2%}")
            report_lines.append(f"Final verification accuracy:   {last_accuracy:.2%}")
            report_lines.append(f"Improvement:                   {improvement:.2%}")
            report_lines.append("")
            
            if improvement > 0:
                report_lines.append("CONCLUSION: The feedback loop successfully improved optimization accuracy.")
            else:
                report_lines.append("CONCLUSION: The feedback loop did not improve optimization accuracy.")
        
        # Write report to file
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report written to {self.output_file}")


def main():
    """Run the integration test."""
    print("=" * 80)
    print("INTEGRATION TEST: STRATEGY OPTIMIZATION FEEDBACK LOOP")
    print("=" * 80)
    print("\nThis test validates the complete closed-loop system for strategy optimization:")
    print("1. Initial Strategy Optimization using synthetic market data")
    print("2. Approval Workflow for strategy promotion")
    print("3. Performance Tracking in simulated 'real-world' conditions")
    print("4. Verification against synthetic predictions")
    print("5. Automatic adjustment of synthetic market parameters")
    print("6. Re-optimization with improved synthetic parameters")
    print("\nThe test will run through multiple iterations of this loop to demonstrate")
    print("how the system learns and improves its optimization process over time.")
    print("\n" + "=" * 80)
    
    # Run the test with 3 iterations
    test = FeedbackLoopIntegrationTest(test_iterations=3)
    test.run_test()
    
    # Display the report path
    print("\n" + "=" * 80)
    print(f"Test complete! Detailed report available at: {test.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
