#!/usr/bin/env python3
"""
BenBot Data Formatter - Formats strategy evaluation data for BenBot analysis system

This module handles the standardization and recording of strategy performance data
to be compatible with the BenBot analysis system.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import datetime
import logging
from typing import Dict, List, Any, Optional, Union

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


class BenBotDataFormatter:
    """
    Handles formatting and storing of strategy evaluation data for the BenBot system.
    
    Provides standardized formats for:
    - Strategy performance metrics
    - Signal history and accuracy
    - Market condition correlations
    - Evolution tracking
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the BenBot data formatter.
        
        Args:
            output_dir: Directory to store BenBot data files
        """
        # Create timestamp for files
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = f"benbot_data_{self.timestamp}"
        
        # Create directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, "benbot_formatter.log"))
            ]
        )
        
        self.logger = logging.getLogger('benbot_formatter')
        
        # Create data store for all strategies
        self.strategy_records = {}
    
    def format_strategy_record(
        self,
        strategy_id: str,
        strategy_type: str,
        parameters: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        scenario_results: Optional[Dict[str, Any]] = None,
        signal_history: Optional[List[Dict[str, Any]]] = None,
        generation: Optional[int] = None,
        parent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Format a strategy record for BenBot.
        
        Args:
            strategy_id: Unique identifier for strategy
            strategy_type: Type of strategy
            parameters: Strategy parameters
            performance_metrics: Key performance metrics
            scenario_results: Results across different scenarios
            signal_history: History of signals and performance
            generation: Evolution generation number
            parent_ids: IDs of parent strategies
            
        Returns:
            Formatted BenBot record
        """
        # Basic record structure
        record = {
            "strategy_id": strategy_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_type": strategy_type,
            "parameters": parameters,
            "performance": {
                "return_pct": performance_metrics.get("return_pct", 0),
                "drawdown_pct": performance_metrics.get("drawdown_pct", 0),
                "win_rate_pct": performance_metrics.get("win_rate_pct", 0),
                "trade_count": performance_metrics.get("trade_count", 0),
                "profit_factor": performance_metrics.get("profit_factor", 0),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
                "consistency_score": performance_metrics.get("consistency_score", 1),
                "overall_score": performance_metrics.get("overall_score", 0),
                "meets_threshold": performance_metrics.get("meets_thresholds", False)
            }
        }
        
        # Add evolutionary data if available
        if generation is not None:
            record["evolution"] = {
                "generation": generation,
                "parent_ids": parent_ids if parent_ids else []
            }
        
        # Add scenario results if available
        if scenario_results:
            record["scenario_results"] = {}
            
            for scenario, results in scenario_results.items():
                # Get key metrics for each scenario
                record["scenario_results"][scenario] = {
                    "return_pct": results.get("metrics", {}).get("return_pct", 0),
                    "drawdown_pct": results.get("metrics", {}).get("drawdown_pct", 0),
                    "win_rate_pct": results.get("metrics", {}).get("win_rate_pct", 0),
                    "score": results.get("score", 0),
                    "meets_thresholds": results.get("meets_thresholds", False)
                }
            
            # Add best and worst scenarios
            best_scenario = max(
                scenario_results.keys(),
                key=lambda s: scenario_results[s].get("score", 0)
            )
            
            worst_scenario = min(
                scenario_results.keys(),
                key=lambda s: scenario_results[s].get("score", 0)
            )
            
            record["best_scenario"] = best_scenario
            record["worst_scenario"] = worst_scenario
        
        # Add signal history if available
        if signal_history:
            # Calculate signal accuracy
            correct_signals = sum(1 for s in signal_history if s.get("correct", False))
            total_signals = len(signal_history)
            
            record["signal_metrics"] = {
                "signal_count": total_signals,
                "accuracy": correct_signals / total_signals if total_signals else 0,
                "buy_signals": sum(1 for s in signal_history if s.get("signal") == "buy"),
                "sell_signals": sum(1 for s in signal_history if s.get("signal") == "sell"),
                "avg_confidence": np.mean([s.get("confidence", 0) for s in signal_history])
            }
            
            # Store recent signals (limit to last 50)
            record["recent_signals"] = signal_history[-50:] if len(signal_history) > 50 else signal_history
        
        return record
    
    def add_strategy_record(self, strategy_record: Dict[str, Any]):
        """
        Add a strategy record to the data store.
        
        Args:
            strategy_record: Formatted strategy record
        """
        strategy_id = strategy_record["strategy_id"]
        self.strategy_records[strategy_id] = strategy_record
        
        # Log addition
        self.logger.info(f"Added strategy record for {strategy_id}")
    
    def save_all_records(self):
        """Save all strategy records to file."""
        # Save full record file
        all_records_path = os.path.join(self.output_dir, f"all_strategy_records_{self.timestamp}.json")
        
        with open(all_records_path, "w") as f:
            json.dump(self.strategy_records, f, indent=2)
        
        # Save summary file
        summary_records = {}
        
        for strategy_id, record in self.strategy_records.items():
            # Create condensed summary with key information
            summary_records[strategy_id] = {
                "strategy_type": record.get("strategy_type"),
                "overall_score": record.get("performance", {}).get("overall_score", 0),
                "return_pct": record.get("performance", {}).get("return_pct", 0),
                "win_rate_pct": record.get("performance", {}).get("win_rate_pct", 0),
                "best_scenario": record.get("best_scenario", "N/A"),
                "worst_scenario": record.get("worst_scenario", "N/A"),
                "meets_threshold": record.get("performance", {}).get("meets_threshold", False)
            }
        
        summary_path = os.path.join(self.output_dir, f"strategy_summary_{self.timestamp}.json")
        
        with open(summary_path, "w") as f:
            json.dump(summary_records, f, indent=2)
        
        self.logger.info(f"Saved {len(self.strategy_records)} strategy records to {all_records_path}")
        self.logger.info(f"Saved strategy summary to {summary_path}")
        
        return all_records_path, summary_path
    
    def save_single_record(self, strategy_record: Dict[str, Any]) -> str:
        """
        Save a single strategy record to file.
        
        Args:
            strategy_record: Formatted strategy record
            
        Returns:
            Path to saved file
        """
        strategy_id = strategy_record["strategy_id"]
        filename = f"{strategy_id}_{self.timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(strategy_record, f, indent=2)
        
        self.logger.info(f"Saved strategy record for {strategy_id} to {filepath}")
        
        return filepath
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of all strategies.
        
        Returns:
            Summary report dictionary
        """
        if not self.strategy_records:
            return {"error": "No strategy records available"}
        
        # Count strategies by type
        strategy_types = {}
        for record in self.strategy_records.values():
            strategy_type = record.get("strategy_type", "Unknown")
            
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = 0
            
            strategy_types[strategy_type] += 1
        
        # Count passing strategies
        passing_strategies = sum(
            1 for r in self.strategy_records.values()
            if r.get("performance", {}).get("meets_threshold", False)
        )
        
        # Calculate average performance
        avg_return = np.mean([
            r.get("performance", {}).get("return_pct", 0)
            for r in self.strategy_records.values()
        ])
        
        avg_win_rate = np.mean([
            r.get("performance", {}).get("win_rate_pct", 0)
            for r in self.strategy_records.values()
        ])
        
        avg_score = np.mean([
            r.get("performance", {}).get("overall_score", 0)
            for r in self.strategy_records.values()
        ])
        
        # Get top strategies
        top_strategies = sorted(
            self.strategy_records.values(),
            key=lambda r: r.get("performance", {}).get("overall_score", 0),
            reverse=True
        )[:10]  # Top 10
        
        # Create summary report
        summary = {
            "total_strategies": len(self.strategy_records),
            "passing_strategies": passing_strategies,
            "pass_rate": passing_strategies / len(self.strategy_records) if self.strategy_records else 0,
            "strategy_types": strategy_types,
            "average_performance": {
                "return_pct": avg_return,
                "win_rate_pct": avg_win_rate,
                "overall_score": avg_score
            },
            "top_strategies": [
                {
                    "strategy_id": r.get("strategy_id"),
                    "strategy_type": r.get("strategy_type"),
                    "overall_score": r.get("performance", {}).get("overall_score", 0),
                    "return_pct": r.get("performance", {}).get("return_pct", 0),
                    "win_rate_pct": r.get("performance", {}).get("win_rate_pct", 0),
                    "best_scenario": r.get("best_scenario", "N/A")
                }
                for r in top_strategies
            ]
        }
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, f"performance_summary_{self.timestamp}.json")
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Generated performance summary saved to {summary_path}")
        
        return summary


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process BenBot data")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input evaluation file or directory"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory for BenBot data"
    )
    
    args = parser.parse_args()
    
    # Initialize formatter
    formatter = BenBotDataFormatter(args.output)
    
    # Process input
    if os.path.isfile(args.input):
        # Single evaluation file
        with open(args.input, "r") as f:
            evaluation = json.load(f)
        
        # Extract data and format record
        strategy_id = f"Strategy-{hash(str(evaluation)) % 100000}"
        strategy_type = evaluation.get("strategy_name", "Unknown")
        parameters = evaluation.get("strategy_parameters", {})
        
        if "metrics" in evaluation:
            performance_metrics = evaluation["metrics"]
        else:
            performance_metrics = {
                "return_pct": evaluation.get("total_return_pct", 0),
                "drawdown_pct": evaluation.get("max_drawdown", 0),
                "win_rate_pct": evaluation.get("win_rate", 0),
                "trade_count": evaluation.get("trade_count", 0),
                "profit_factor": evaluation.get("profit_factor", 0),
                "overall_score": evaluation.get("score", 0),
                "meets_thresholds": evaluation.get("meets_thresholds", False)
            }
        
        # Format and save record
        record = formatter.format_strategy_record(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            parameters=parameters,
            performance_metrics=performance_metrics,
            scenario_results=evaluation.get("scenario_results")
        )
        
        formatter.add_strategy_record(record)
        formatter.save_single_record(record)
        
    elif os.path.isdir(args.input):
        # Directory of evaluation files
        for filename in os.listdir(args.input):
            if filename.endswith(".json"):
                filepath = os.path.join(args.input, filename)
                
                try:
                    with open(filepath, "r") as f:
                        evaluation = json.load(f)
                    
                    # Extract data and format record
                    strategy_id = f"Strategy-{hash(str(evaluation)) % 100000}"
                    strategy_type = evaluation.get("strategy_name", "Unknown")
                    parameters = evaluation.get("strategy_parameters", {})
                    
                    if "metrics" in evaluation:
                        performance_metrics = evaluation["metrics"]
                    else:
                        performance_metrics = {
                            "return_pct": evaluation.get("total_return_pct", 0),
                            "drawdown_pct": evaluation.get("max_drawdown", 0),
                            "win_rate_pct": evaluation.get("win_rate", 0),
                            "trade_count": evaluation.get("trade_count", 0),
                            "profit_factor": evaluation.get("profit_factor", 0),
                            "overall_score": evaluation.get("score", 0),
                            "meets_thresholds": evaluation.get("meets_thresholds", False)
                        }
                    
                    # Format record
                    record = formatter.format_strategy_record(
                        strategy_id=strategy_id,
                        strategy_type=strategy_type,
                        parameters=parameters,
                        performance_metrics=performance_metrics,
                        scenario_results=evaluation.get("scenario_results")
                    )
                    
                    formatter.add_strategy_record(record)
                
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
        
        # Save all records
        formatter.save_all_records()
        
        # Generate summary report
        summary = formatter.generate_summary_report()
        print(f"Processed {len(formatter.strategy_records)} strategy records")
        print(f"Passing strategies: {summary['passing_strategies']} ({summary['pass_rate']:.1%})")
    
    else:
        print(f"Input not found: {args.input}")
