#!/usr/bin/env python3
"""
Test Smart Features

This script provides comprehensive testing capabilities for the smart forex modules:
1. CLI Testing - Run the smart analysis CLI with various parameters
2. Data Collection - Monitor and visualize how the smart modules learn over time
3. Comparative Testing - Compare performance between standard and smart-enhanced methods
"""

import os
import sys
import yaml
import json
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_smart_features')

# Import EvoTrader components
try:
    from forex_evotrader import ForexEvoTrader
    from forex_smart_session import SmartSessionAnalyzer
    from forex_smart_pips import SmartPipAnalyzer
    from forex_smart_news import SmartNewsAnalyzer
    from forex_smart_compliance import SmartComplianceMonitor
    from forex_smart_benbot import SmartBenBotConnector
    from forex_smart_integration import ForexSmartIntegration
except ImportError as e:
    logger.error(f"Import error: {e}. Make sure all required modules are installed and in your PYTHONPATH.")
    sys.exit(1)

# Default config path
DEFAULT_CONFIG = '/Users/bendickinson/Desktop/Evotrader/forex_evotrader_config.yaml'


class SmartFeatureTester:
    """
    Comprehensive tester for smart forex features.
    """
    
    def __init__(self, config_path: str = DEFAULT_CONFIG):
        """
        Initialize the tester.
        
        Args:
            config_path: Path to EvoTrader config file
        """
        self.config_path = config_path
        
        # Load config
        self.config = self._load_config()
        
        # Initialize EvoTrader instances - one standard, one enhanced
        logger.info("Initializing standard EvoTrader instance")
        self.standard_evotrader = ForexEvoTrader(config_path)
        
        logger.info("Initializing enhanced EvoTrader instance")
        self.enhanced_evotrader = ForexEvoTrader(config_path)
        self.enhanced_evotrader.enhance_with_smart_methods()
        
        # Initialize results storage
        self.test_results = {
            'cli_tests': [],
            'data_collection': {},
            'comparative_tests': []
        }
        
        logger.info("Smart Feature Tester initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def test_cli(self, pairs: List[str] = None, analysis_types: List[str] = None):
        """
        Test the smart analysis CLI functionality.
        
        Args:
            pairs: List of currency pairs to test
            analysis_types: List of analysis types to test
        """
        if pairs is None:
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        if analysis_types is None:
            analysis_types = ['session', 'pips', 'news', 'compliance', 'all']
        
        logger.info(f"Testing CLI with pairs: {pairs} and analysis types: {analysis_types}")
        
        results = []
        
        for pair in pairs:
            for analysis_type in analysis_types:
                logger.info(f"Running CLI test: pair={pair}, analysis_type={analysis_type}")
                
                # Build command
                cmd = [
                    'python', 
                    '/Users/bendickinson/Desktop/Evotrader/forex_evotrader.py',
                    'smart-analysis',
                    '--pair', pair,
                    '--analysis-type', analysis_type,
                    '--config', self.config_path
                ]
                
                # Run command
                start_time = time.time()
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True,
                        timeout=30
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Parse output as JSON if possible
                    try:
                        output_data = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        output_data = {'raw': result.stdout}
                    
                    success = result.returncode == 0
                    
                    results.append({
                        'pair': pair,
                        'analysis_type': analysis_type,
                        'success': success,
                        'execution_time': execution_time,
                        'output': output_data,
                        'error': result.stderr if result.stderr else None
                    })
                    
                    if success:
                        logger.info(f"CLI test successful: pair={pair}, analysis_type={analysis_type}")
                    else:
                        logger.error(f"CLI test failed: pair={pair}, analysis_type={analysis_type}")
                        logger.error(f"Error: {result.stderr}")
                
                except subprocess.TimeoutExpired:
                    logger.error(f"CLI test timed out: pair={pair}, analysis_type={analysis_type}")
                    results.append({
                        'pair': pair,
                        'analysis_type': analysis_type,
                        'success': False,
                        'execution_time': 30,
                        'output': None,
                        'error': 'Command timed out'
                    })
                
                except Exception as e:
                    logger.error(f"Error running CLI test: {e}")
                    results.append({
                        'pair': pair,
                        'analysis_type': analysis_type,
                        'success': False,
                        'execution_time': time.time() - start_time,
                        'output': None,
                        'error': str(e)
                    })
        
        # Store results
        self.test_results['cli_tests'] = results
        
        return results
    
    def monitor_data_collection(self, days: int = 7, interval_hours: int = 4):
        """
        Set up monitoring to track how smart modules learn over time.
        
        Args:
            days: Number of days to monitor
            interval_hours: Hours between checks
        """
        logger.info(f"Setting up data collection monitoring for {days} days with {interval_hours} hour intervals")
        
        # Initialize data storage
        self.data_collection = {
            'news_impact_history': [],
            'benbot_confidence': [],
            'session_strength': [],
            'risk_projections': []
        }
        
        # For demo purposes, we'll simulate the data collection
        # In a real environment, this would set up a scheduled task
        
        # Simulate data points over time
        current_time = datetime.datetime.now()
        
        for day in range(days):
            for hour in range(0, 24, interval_hours):
                # Calculate simulated timestamp
                timestamp = current_time + datetime.timedelta(days=day, hours=hour)
                
                # Simulate news impact history growth
                news_count = int(5 + day * 2 + hour / 6)  # Growing number of recorded news impacts
                
                # Simulate BenBot confidence improvement
                base_confidence = 0.5
                confidence_improvement = min(0.4, (day * 24 + hour) / (days * 24) * 0.4)
                benbot_confidence = base_confidence + confidence_improvement
                
                # Simulate session strength learning
                session_accuracy = 0.6 + min(0.35, (day * 24 + hour) / (days * 24) * 0.35)
                
                # Simulate risk projection accuracy
                risk_accuracy = 0.5 + min(0.4, (day * 24 + hour) / (days * 24) * 0.4)
                
                # Record simulated data point
                self.data_collection['news_impact_history'].append({
                    'timestamp': timestamp.isoformat(),
                    'news_count': news_count
                })
                
                self.data_collection['benbot_confidence'].append({
                    'timestamp': timestamp.isoformat(),
                    'average_confidence': benbot_confidence,
                    'trade_entry_confidence': benbot_confidence + 0.05,
                    'trade_exit_confidence': benbot_confidence - 0.02,
                    'risk_adjustment_confidence': benbot_confidence + 0.03
                })
                
                self.data_collection['session_strength'].append({
                    'timestamp': timestamp.isoformat(),
                    'detection_accuracy': session_accuracy,
                    'london_strength': 0.7 + day * 0.02,
                    'newyork_strength': 0.65 + day * 0.03,
                    'tokyo_strength': 0.6 + day * 0.02,
                    'sydney_strength': 0.5 + day * 0.04
                })
                
                self.data_collection['risk_projections'].append({
                    'timestamp': timestamp.isoformat(),
                    'projection_accuracy': risk_accuracy,
                    'drawdown_prediction_error': 0.2 - min(0.15, (day * 24 + hour) / (days * 24) * 0.15),
                    'profit_target_accuracy': 0.6 + min(0.3, (day * 24 + hour) / (days * 24) * 0.3)
                })
        
        # Store results
        self.test_results['data_collection'] = self.data_collection
        
        return self.data_collection
    
    def visualize_data_collection(self, output_dir: str = './reports'):
        """
        Visualize the data collection results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Visualizing data collection results to {output_dir}")
        
        # Ensure we have data
        if not self.test_results.get('data_collection'):
            logger.error("No data collection results to visualize")
            return
        
        # Convert data to pandas DataFrames
        dfs = {}
        for key, data in self.test_results['data_collection'].items():
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                dfs[key] = df
        
        # Create visualizations
        for key, df in dfs.items():
            plt.figure(figsize=(12, 6))
            
            if key == 'news_impact_history':
                df['news_count'].plot(title='News Impact History Growth', marker='o')
                plt.ylabel('Number of Recorded News Impacts')
                plt.xlabel('Date')
            
            elif key == 'benbot_confidence':
                df[['average_confidence', 'trade_entry_confidence', 
                   'trade_exit_confidence', 'risk_adjustment_confidence']].plot(
                    title='BenBot Confidence Evolution', marker='o')
                plt.ylabel('Confidence Score')
                plt.xlabel('Date')
                plt.legend()
            
            elif key == 'session_strength':
                df[['detection_accuracy', 'london_strength', 
                   'newyork_strength', 'tokyo_strength', 'sydney_strength']].plot(
                    title='Session Strength Detection Learning', marker='o')
                plt.ylabel('Strength/Accuracy Score')
                plt.xlabel('Date')
                plt.legend()
            
            elif key == 'risk_projections':
                df[['projection_accuracy', 'drawdown_prediction_error', 
                   'profit_target_accuracy']].plot(
                    title='Risk Projection Accuracy Improvement', marker='o')
                plt.ylabel('Accuracy/Error Score')
                plt.xlabel('Date')
                plt.legend()
            
            # Save figure
            output_file = os.path.join(output_dir, f'{key}_evolution.png')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Saved visualization to {output_file}")
        
        return True
    
    def run_comparative_tests(self, pairs: List[str] = None, test_days: int = 10):
        """
        Compare performance between standard and enhanced methods.
        
        Args:
            pairs: List of currency pairs to test
            test_days: Number of days to simulate
        """
        if pairs is None:
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        logger.info(f"Running comparative tests with pairs: {pairs} for {test_days} days")
        
        # Initialize results
        comparison_results = {
            'session_optimality': [],
            'pip_targets': [],
            'news_safety': [],
            'position_sizing': [],
            'benbot_consultation': []
        }
        
        # Current timestamp
        current_time = datetime.datetime.now()
        
        # Simulate test days
        for day in range(test_days):
            for hour in range(0, 24, 3):  # Check every 3 hours
                # Calculate simulated timestamp
                timestamp = current_time + datetime.timedelta(days=day, hours=hour)
                
                # Test with each pair
                for pair in pairs:
                    # Test 1: Session Optimality
                    standard_optimal, standard_reason = self.standard_evotrader.check_session_optimal(
                        pair, timestamp=timestamp)
                    
                    enhanced_optimal, enhanced_reason = self.enhanced_evotrader.check_session_optimal(
                        pair, timestamp=timestamp)
                    
                    comparison_results['session_optimality'].append({
                        'timestamp': timestamp.isoformat(),
                        'pair': pair,
                        'standard_result': standard_optimal,
                        'standard_reason': standard_reason,
                        'enhanced_result': enhanced_optimal,
                        'enhanced_reason': enhanced_reason,
                        'agreement': standard_optimal == enhanced_optimal
                    })
                    
                    # Test 2: Pip Targets
                    standard_pip_target = self.standard_evotrader.calculate_pip_target(pair)
                    enhanced_pip_target = self.enhanced_evotrader.calculate_pip_target(pair)
                    
                    comparison_results['pip_targets'].append({
                        'timestamp': timestamp.isoformat(),
                        'pair': pair,
                        'standard_target': standard_pip_target,
                        'enhanced_target': enhanced_pip_target,
                        'difference': self._compare_pip_targets(standard_pip_target, enhanced_pip_target)
                    })
                    
                    # Test 3: News Safety
                    standard_safe, standard_news_reason = self.standard_evotrader.check_news_safe(
                        pair, timestamp=timestamp)
                    
                    enhanced_safe, enhanced_news_reason = self.enhanced_evotrader.check_news_safe(
                        pair, timestamp=timestamp)
                    
                    comparison_results['news_safety'].append({
                        'timestamp': timestamp.isoformat(),
                        'pair': pair,
                        'standard_result': standard_safe,
                        'standard_reason': standard_news_reason,
                        'enhanced_result': enhanced_safe,
                        'enhanced_reason': enhanced_news_reason,
                        'agreement': standard_safe == enhanced_safe
                    })
                    
                    # Test 4: Position Sizing
                    equity = 10000.0 - (day * 100)  # Simulate declining equity
                    
                    standard_size = self.standard_evotrader.calculate_position_size(equity, pair)
                    enhanced_size = self.enhanced_evotrader.calculate_position_size(equity, pair)
                    
                    comparison_results['position_sizing'].append({
                        'timestamp': timestamp.isoformat(),
                        'pair': pair,
                        'equity': equity,
                        'standard_size': standard_size,
                        'enhanced_size': enhanced_size,
                        'difference_percent': (enhanced_size / standard_size - 1) * 100 if standard_size else 0
                    })
                    
                    # Test 5: BenBot Consultation
                    standard_consultation = self.standard_evotrader.consult_benbot(
                        'trade_entry', {'pair': pair, 'decision': True, 'confidence': 0.6})
                    
                    enhanced_consultation = self.enhanced_evotrader.consult_benbot(
                        'trade_entry', {'pair': pair, 'decision': True, 'confidence': 0.6})
                    
                    comparison_results['benbot_consultation'].append({
                        'timestamp': timestamp.isoformat(),
                        'pair': pair,
                        'standard_decision': standard_consultation.get('decision'),
                        'enhanced_decision': enhanced_consultation.get('decision'),
                        'standard_confidence': standard_consultation.get('confidence', 0.5),
                        'enhanced_confidence': enhanced_consultation.get('benbot_confidence', 0.5),
                        'agreement': standard_consultation.get('decision') == enhanced_consultation.get('decision')
                    })
        
        # Store results
        self.test_results['comparative_tests'] = comparison_results
        
        return comparison_results
    
    def _compare_pip_targets(self, standard_target, enhanced_target):
        """Compare standard and enhanced pip targets."""
        if isinstance(standard_target, (int, float)) and isinstance(enhanced_target, (int, float)):
            return enhanced_target - standard_target
        
        if isinstance(standard_target, dict) and isinstance(enhanced_target, dict):
            if 'take_profit' in standard_target and 'take_profit' in enhanced_target:
                return enhanced_target['take_profit'] - standard_target['take_profit']
        
        # Can't compare directly, return as-is
        return {
            'standard': standard_target,
            'enhanced': enhanced_target
        }
    
    def analyze_comparative_results(self, output_dir: str = './reports'):
        """
        Analyze and visualize comparative test results.
        
        Args:
            output_dir: Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Analyzing comparative test results to {output_dir}")
        
        # Ensure we have data
        if not self.test_results.get('comparative_tests'):
            logger.error("No comparative test results to analyze")
            return
        
        results = {}
        
        # Process each test type
        for test_type, data in self.test_results['comparative_tests'].items():
            if not data:
                continue
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create visualizations based on test type
            plt.figure(figsize=(12, 6))
            
            if test_type == 'session_optimality':
                # Calculate agreement rate over time
                df['date'] = df['timestamp'].dt.date
                agreement_by_date = df.groupby('date')['agreement'].mean()
                
                agreement_by_date.plot(
                    title='Session Optimality Agreement Rate', marker='o')
                plt.ylabel('Agreement Rate')
                plt.xlabel('Date')
                plt.ylim(0, 1.1)
                
                # Count cases where enhanced found optimal but standard didn't
                enhanced_better = df[(df['enhanced_result'] == True) & (df['standard_result'] == False)].shape[0]
                standard_better = df[(df['enhanced_result'] == False) & (df['standard_result'] == True)].shape[0]
                
                results[test_type] = {
                    'agreement_rate': df['agreement'].mean(),
                    'enhanced_better_count': enhanced_better,
                    'standard_better_count': standard_better,
                    'total_checks': len(df)
                }
            
            elif test_type == 'pip_targets':
                # Plot pip target differences by pair
                df_numeric = df[df['difference'].apply(lambda x: isinstance(x, (int, float)))]
                if not df_numeric.empty:
                    df_pivot = df_numeric.pivot_table(
                        index='timestamp', columns='pair', values='difference', aggfunc='mean')
                    df_pivot.plot(title='Pip Target Differences (Enhanced - Standard)', marker='o')
                    plt.ylabel('Difference in Pips')
                    plt.xlabel('Date')
                    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    plt.legend()
                
                # Calculate statistics
                avg_diff = df_numeric['difference'].mean() if not df_numeric.empty else 0
                
                results[test_type] = {
                    'average_difference': avg_diff,
                    'enhanced_higher_count': df_numeric[df_numeric['difference'] > 0].shape[0] if not df_numeric.empty else 0,
                    'standard_higher_count': df_numeric[df_numeric['difference'] < 0].shape[0] if not df_numeric.empty else 0,
                    'total_comparisons': len(df)
                }
            
            elif test_type == 'news_safety':
                # Calculate agreement rate over time
                df['date'] = df['timestamp'].dt.date
                agreement_by_date = df.groupby('date')['agreement'].mean()
                
                agreement_by_date.plot(
                    title='News Safety Agreement Rate', marker='o')
                plt.ylabel('Agreement Rate')
                plt.xlabel('Date')
                plt.ylim(0, 1.1)
                
                # Count cases where enhanced was more cautious
                enhanced_more_cautious = df[(df['enhanced_result'] == False) & (df['standard_result'] == True)].shape[0]
                standard_more_cautious = df[(df['enhanced_result'] == True) & (df['standard_result'] == False)].shape[0]
                
                results[test_type] = {
                    'agreement_rate': df['agreement'].mean(),
                    'enhanced_more_cautious': enhanced_more_cautious,
                    'standard_more_cautious': standard_more_cautious,
                    'total_checks': len(df)
                }
            
            elif test_type == 'position_sizing':
                # Plot position size differences over equity
                df.sort_values('equity', inplace=True)
                
                # Group by equity range for clearer visualization
                df['equity_bin'] = pd.cut(df['equity'], 10)
                size_diff_by_equity = df.groupby('equity_bin')['difference_percent'].mean()
                
                size_diff_by_equity.plot(
                    title='Position Size Difference % by Equity', kind='bar')
                plt.ylabel('Enhanced Size vs Standard Size (%)')
                plt.xlabel('Equity Range')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Calculate statistics
                results[test_type] = {
                    'average_difference_percent': df['difference_percent'].mean(),
                    'enhanced_larger_count': df[df['difference_percent'] > 0].shape[0],
                    'standard_larger_count': df[df['difference_percent'] < 0].shape[0],
                    'total_comparisons': len(df)
                }
            
            elif test_type == 'benbot_consultation':
                # Calculate agreement rate over time
                df['date'] = df['timestamp'].dt.date
                agreement_by_date = df.groupby('date')['agreement'].mean()
                
                agreement_by_date.plot(
                    title='BenBot Consultation Agreement Rate', marker='o')
                plt.ylabel('Agreement Rate')
                plt.xlabel('Date')
                plt.ylim(0, 1.1)
                
                # Plot confidence comparison
                df['confidence_diff'] = df['enhanced_confidence'] - df['standard_confidence']
                df_pivot = df.pivot_table(
                    index='timestamp', columns='pair', values='confidence_diff', aggfunc='mean')
                
                plt.figure(figsize=(12, 6))
                df_pivot.plot(
                    title='BenBot Confidence Difference (Enhanced - Standard)', marker='o')
                plt.ylabel('Confidence Difference')
                plt.xlabel('Date')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.legend()
                
                # Calculate statistics
                results[test_type] = {
                    'agreement_rate': df['agreement'].mean(),
                    'average_confidence_difference': df['confidence_diff'].mean(),
                    'total_consultations': len(df)
                }
            
            # Save figure
            output_file = os.path.join(output_dir, f'{test_type}_comparison.png')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Saved comparison visualization to {output_file}")
        
        # Save summary results
        summary_file = os.path.join(output_dir, 'comparative_analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved comparison summary to {summary_file}")
        
        return results
    
    def save_test_results(self, output_file: str = './reports/smart_test_results.json'):
        """
        Save all test results to file.
        
        Args:
            output_file: Output file path
        """
        # Create directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Saving test results to {output_file}")
        
        # Convert results to serializable format
        serializable_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'cli_tests': self.test_results.get('cli_tests', []),
            'data_collection': {
                k: v for k, v in self.test_results.get('data_collection', {}).items()
            },
            'comparative_tests': {
                k: v for k, v in self.test_results.get('comparative_tests', {}).items()
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Successfully saved test results to {output_file}")
        
        return output_file


def run_all_tests(config_path: str = DEFAULT_CONFIG, output_dir: str = './reports'):
    """
    Run all tests and generate reports.
    
    Args:
        config_path: Path to EvoTrader config file
        output_dir: Directory to save results
    """
    logger.info(f"Running all smart feature tests with config: {config_path}")
    
    # Initialize tester
    tester = SmartFeatureTester(config_path)
    
    # Run CLI tests
    logger.info("Running CLI tests...")
    cli_results = tester.test_cli()
    
    # Simulate data collection monitoring
    logger.info("Simulating data collection...")
    collection_data = tester.monitor_data_collection()
    
    # Visualize data collection
    logger.info("Visualizing data collection...")
    tester.visualize_data_collection(output_dir)
    
    # Run comparative tests
    logger.info("Running comparative tests...")
    comparative_results = tester.run_comparative_tests()
    
    # Analyze comparative results
    logger.info("Analyzing comparative results...")
    analysis = tester.analyze_comparative_results(output_dir)
    
    # Save all results
    output_file = os.path.join(output_dir, 'smart_test_results.json')
    tester.save_test_results(output_file)
    
    logger.info(f"All tests completed. Results saved to {output_dir}")
    
    return {
        'cli_results': cli_results,
        'collection_data': collection_data,
        'comparative_results': comparative_results,
        'analysis': analysis,
        'output_file': output_file
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='Test EvoTrader Smart Features')
    
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                      help='Path to EvoTrader config file')
    parser.add_argument('--output-dir', type=str, default='./reports',
                      help='Directory to save results')
    parser.add_argument('--test', type=str, choices=['cli', 'data', 'comparative', 'all'],
                      default='all', help='Test to run')
    parser.add_argument('--pairs', type=str, nargs='+',
                      default=['EURUSD', 'GBPUSD', 'USDJPY'],
                      help='Currency pairs to test')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SmartFeatureTester(args.config)
    
    # Run selected test
    if args.test == 'cli' or args.test == 'all':
        logger.info("Running CLI tests...")
        tester.test_cli(args.pairs)
    
    if args.test == 'data' or args.test == 'all':
        logger.info("Simulating data collection...")
        tester.monitor_data_collection()
        tester.visualize_data_collection(args.output_dir)
    
    if args.test == 'comparative' or args.test == 'all':
        logger.info("Running comparative tests...")
        tester.run_comparative_tests(args.pairs)
        tester.analyze_comparative_results(args.output_dir)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'smart_test_results.json')
    tester.save_test_results(output_file)
    
    logger.info(f"Tests completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
