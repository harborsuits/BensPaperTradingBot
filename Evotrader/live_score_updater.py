#!/usr/bin/env python3
"""
Live Score Updater

This module updates strategy scores based on live/paper trading performance,
creating a feedback loop between real-world performance and the evolution process.
"""

import os
import sys
import logging
import datetime
import json
import argparse
import yaml
from typing import Dict, List, Any, Optional, Tuple

# Import required modules
from benbot_api import BenBotAPI
from prop_strategy_registry import PropStrategyRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/live_score_updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('live_score_updater')

class LiveScoreUpdater:
    """
    Updates strategy scores based on live performance data from BenBot.
    Creates a feedback loop for the evolution process.
    """
    
    def __init__(self, 
                 config_path: str = "forex_evotrader_config.yaml",
                 registry_path: str = "./prop_strategy_registry",
                 risk_profile_path: str = "prop_risk_profile.yaml"):
        """
        Initialize the live score updater.
        
        Args:
            config_path: Path to EvoTrader configuration file
            registry_path: Path to strategy registry
            risk_profile_path: Path to risk profile
        """
        self.config_path = config_path
        self.registry_path = registry_path
        self.risk_profile_path = risk_profile_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize API and registry
        benbot_url = self.config.get('benbot', {}).get('api_url', 'http://localhost:8080/benbot/api')
        self.api = BenBotAPI(api_url=benbot_url)
        self.registry = PropStrategyRegistry(registry_path, risk_profile_path)
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load EvoTrader configuration."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        logger.warning(f"Config file {self.config_path} not found, using defaults")
        return {
            'benbot': {
                'api_url': 'http://localhost:8080/benbot/api'
            }
        }
        
    def update_all_strategies(self) -> Dict[str, int]:
        """
        Update all active strategies with live performance data.
        
        Returns:
            Statistics about the update process
        """
        # Get all active strategies from BenBot
        benbot_strategies = self.api.get_all_active_strategies()
        
        # Get all active strategies from registry
        registry_strategies = self.registry.get_active_strategies()
        
        # Strategies that exist in both systems
        benbot_ids = set(s.get('strategy_id') for s in benbot_strategies)
        common_strategies = benbot_ids.intersection(set(registry_strategies))
        
        logger.info(f"Found {len(common_strategies)} active strategies in both systems")
        
        # Update each strategy
        updated = 0
        for strategy_id in common_strategies:
            success = self.update_strategy_score(strategy_id)
            if success:
                updated += 1
        
        # Identify promotion candidates
        promotion_candidates = self.registry.get_promotion_candidates()
        if promotion_candidates:
            logger.info(f"Found {len(promotion_candidates)} promotion candidates")
            for candidate in promotion_candidates:
                logger.info(f"Promotion candidate: {candidate['id']} (Confidence: {candidate['confidence_score']:.2f})")
        
        # Identify demotion candidates
        demotion_candidates = self.registry.get_demotion_candidates()
        if demotion_candidates:
            logger.info(f"Found {len(demotion_candidates)} demotion candidates")
            for candidate in demotion_candidates:
                logger.info(f"Demotion candidate: {candidate['id']} (Confidence: {candidate['confidence_score']:.2f})")
        
        # Generate recommendations report
        report_path = self.generate_recommendations_report(promotion_candidates, demotion_candidates)
        
        # Generate evolution bias configuration
        if updated > 0:
            bias_config = self.apply_feedback_to_evolution()
            logger.info(f"Generated evolution bias configuration with {len(bias_config.get('successful_patterns', {}).get('indicators', {}))} successful indicators")
        
        return {
            'updated_count': updated,
            'total_strategies': len(common_strategies),
            'promotion_candidates': len(promotion_candidates),
            'demotion_candidates': len(demotion_candidates),
            'report_path': report_path
        }
        
    def update_strategy_score(self, strategy_id: str) -> bool:
        """
        Update score for a specific strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Success status
        """
        # Fetch strategy performance from BenBot
        performance = self.api.get_strategy_performance(strategy_id)
        
        if not performance:
            logger.error(f"Failed to get performance data for strategy {strategy_id}")
            return False
        
        # Update registry with live metrics
        success = self.registry.update_live_metrics(strategy_id, performance)
        
        if success:
            logger.info(f"Updated metrics for strategy {strategy_id}")
            
            # Get updated delta metrics
            delta = self.registry.get_performance_delta(strategy_id)
            
            if delta:
                logger.info(f"Strategy {strategy_id} confidence score: {delta.get('confidence_score', 0):.2f}")
                logger.info(f"Recommendation: {delta.get('recommendation', 'unknown')}")
            
            return True
        else:
            logger.error(f"Failed to update metrics for strategy {strategy_id}")
            return False
            
    def generate_recommendations_report(self, promotion_candidates, demotion_candidates) -> str:
        """
        Generate a report of promotion and demotion recommendations.
        
        Args:
            promotion_candidates: List of promotion candidates
            demotion_candidates: List of demotion candidates
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = f"reports/recommendations_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Strategy Recommendations Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write promotion candidates
            f.write("## Promotion Candidates\n\n")
            if promotion_candidates:
                f.write("| Strategy ID | Confidence | Trade Count | Live Sharpe | Days in Paper |\n")
                f.write("|------------|------------|-------------|-------------|---------------|\n")
                
                for candidate in promotion_candidates:
                    f.write(f"| {candidate.get('id')} | {candidate.get('confidence_score', 0):.2f} | ")
                    f.write(f"{candidate.get('trade_count', 0)} | {candidate.get('sharpe_ratio', 0):.2f} | ")
                    f.write(f"{candidate.get('days_in_paper', 0)} |\n")
            else:
                f.write("No promotion candidates at this time.\n")
            
            f.write("\n")
            
            # Write demotion candidates
            f.write("## Demotion Candidates\n\n")
            if demotion_candidates:
                f.write("| Strategy ID | Confidence | Trade Count | Live Sharpe | Current Status | Recommendation |\n")
                f.write("|------------|------------|-------------|-------------|----------------|----------------|\n")
                
                for candidate in demotion_candidates:
                    recommendation = self.registry.get_performance_delta(candidate.get('id', '')).get('recommendation', 'unknown')
                    f.write(f"| {candidate.get('id')} | {candidate.get('confidence_score', 0):.2f} | ")
                    f.write(f"{candidate.get('trade_count', 0)} | {candidate.get('sharpe_ratio', 0):.2f} | ")
                    f.write(f"{candidate.get('status', 'unknown')} | {recommendation} |\n")
            else:
                f.write("No demotion candidates at this time.\n")
        
        logger.info(f"Generated recommendations report: {report_file}")
        return report_file
        
    def apply_feedback_to_evolution(self) -> Dict[str, Any]:
        """
        Apply feedback from live performance to evolution parameters.
        
        Returns:
            Evolution bias configuration
        """
        # Get strategy performance data
        strategies = {}
        for strategy_id in self.registry.get_all_strategy_ids():
            strategy = self.registry.get_strategy(strategy_id)
            delta = self.registry.get_performance_delta(strategy_id)
            live = self.registry.get_live_metrics(strategy_id)
            
            if strategy and delta and live:
                strategies[strategy_id] = {
                    'strategy': strategy,
                    'delta': delta,
                    'live': live
                }
        
        # Separate into successful and unsuccessful strategies
        successful = []
        unsuccessful = []
        
        for strategy_id, data in strategies.items():
            # Only consider strategies with enough live trades
            if data['live'].get('trade_count', 0) < 20:
                continue
            
            confidence = data['delta'].get('confidence_score', 0)
            
            if confidence > 0.7:
                successful.append(data['strategy'])
            elif confidence < 0.3:
                unsuccessful.append(data['strategy'])
        
        logger.info(f"Found {len(successful)} successful strategies and {len(unsuccessful)} unsuccessful strategies")
        
        # Extract patterns from successful strategies
        successful_patterns = self._extract_strategy_patterns(successful)
        unsuccessful_patterns = self._extract_strategy_patterns(unsuccessful)
        
        # Create bias configuration
        bias_config = {
            'successful_patterns': successful_patterns,
            'unsuccessful_patterns': unsuccessful_patterns,
            'generated': datetime.datetime.now().isoformat()
        }
        
        # Save bias configuration
        os.makedirs("config", exist_ok=True)
        with open('config/evolution_bias.json', 'w') as f:
            json.dump(bias_config, f, indent=2)
        
        logger.info(f"Generated evolution bias configuration")
        return bias_config
        
    def _extract_strategy_patterns(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract common patterns from a list of strategies.
        
        Args:
            strategies: List of strategy dictionaries
            
        Returns:
            Dictionary of pattern information
        """
        if not strategies:
            return {}
        
        # Initialize pattern counters
        indicators = {}
        parameters = {}
        timeframes = {}
        
        # Extract patterns from each strategy
        for strategy in strategies:
            # Get parameters and parse
            params = {}
            if isinstance(strategy.get('parameters'), str):
                try:
                    params = json.loads(strategy.get('parameters', '{}'))
                except:
                    params = {}
            else:
                params = strategy.get('parameters', {})
            
            # Count indicators (if available in parameters)
            for indicator in params.get('indicators', []):
                indicators[indicator] = indicators.get(indicator, 0) + 1
            
            # Track parameter ranges (numeric parameters only)
            for param, value in params.items():
                if isinstance(value, (int, float)) and param != 'indicators':
                    if param not in parameters:
                        parameters[param] = {'min': value, 'max': value, 'sum': value, 'count': 1}
                    else:
                        parameters[param]['min'] = min(parameters[param]['min'], value)
                        parameters[param]['max'] = max(parameters[param]['max'], value)
                        parameters[param]['sum'] += value
                        parameters[param]['count'] += 1
            
            # Count timeframes
            timeframe = params.get('timeframe')
            if timeframe:
                timeframes[timeframe] = timeframes.get(timeframe, 0) + 1
        
        # Calculate average for each parameter
        for param in parameters:
            parameters[param]['avg'] = parameters[param]['sum'] / parameters[param]['count']
        
        # Convert counts to ratios
        total_strategies = len(strategies)
        indicator_ratios = {i: count / total_strategies for i, count in indicators.items()}
        timeframe_ratios = {t: count / total_strategies for t, count in timeframes.items()}
        
        return {
            'indicators': indicator_ratios,
            'parameters': parameters,
            'timeframes': timeframe_ratios
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Update strategy scores based on live performance')
    
    parser.add_argument('--strategy-id', type=str, help='Update a specific strategy')
    parser.add_argument('--generate-bias', action='store_true', help='Generate evolution bias configuration')
    parser.add_argument('--config', type=str, default='forex_evotrader_config.yaml', help='Path to EvoTrader configuration')
    parser.add_argument('--registry', type=str, default='./prop_strategy_registry', help='Path to strategy registry')
    parser.add_argument('--risk-profile', type=str, default='prop_risk_profile.yaml', help='Path to risk profile')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = LiveScoreUpdater(
        config_path=args.config,
        registry_path=args.registry,
        risk_profile_path=args.risk_profile
    )
    
    if args.strategy_id:
        # Update specific strategy
        success = updater.update_strategy_score(args.strategy_id)
        if success:
            print(f"Updated strategy {args.strategy_id}")
            delta = updater.registry.get_performance_delta(args.strategy_id)
            if delta:
                print(f"Confidence score: {delta.get('confidence_score', 0):.2f}")
                print(f"Recommendation: {delta.get('recommendation', 'unknown')}")
        else:
            print(f"Failed to update strategy {args.strategy_id}")
    else:
        # Update all strategies
        result = updater.update_all_strategies()
        print(f"Updated {result['updated_count']} of {result['total_strategies']} strategies")
        print(f"Found {result['promotion_candidates']} promotion candidates")
        print(f"Found {result['demotion_candidates']} demotion candidates")
        print(f"Generated report: {result['report_path']}")
    
    if args.generate_bias or not args.strategy_id:
        # Generate evolution bias configuration
        bias_config = updater.apply_feedback_to_evolution()
        print(f"Generated evolution bias configuration")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    main()
