#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Structure Validator

This script validates that your implemented strategies follow the correct structure
and contain all necessary components for integration with your trading pipeline.
It examines the strategy files directly rather than trying to import them.
"""

import os
import re
import glob
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyValidator:
    """Validates strategy implementations."""
    
    def __init__(self):
        """Initialize the validator."""
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.strategies_dir = os.path.join(self.project_root, 'trading_bot', 'strategies')
        
        # Define required components for strategies
        self.required_methods = [
            'define_universe',
            'generate_signals',
            'on_exit_signal',
        ]
        
        # Additional important methods
        self.important_methods = [
            '_adjust_signal_confidence',
            '_track_position',
            'get_positions',
        ]
        
        # Exit strategies that should be included
        self.exit_strategies = [
            'profit_target', 
            'stop_loss',
            'time_stop',
            'iv_decrease',
            'gamma_risk',
            'event',
            'drawdown',
        ]
        
        # Recently implemented strategies to focus on
        self.target_strategies = [
            'covered_call_strategy_new.py',
            'iron_condor_strategy_new.py',
            'straddle_strangle_strategy.py',
        ]
        
        # Result tracking
        self.results = {}
    
    def find_strategy_files(self):
        """Find all strategy files in the project."""
        logger.info("Searching for strategy files...")
        
        all_strategy_files = []
        
        # Walk through the strategies directory
        for root, _, files in os.walk(self.strategies_dir):
            for file in files:
                if file.endswith('.py') and 'test' not in file.lower() and '__init__' not in file:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.project_root)
                    all_strategy_files.append((file, rel_path, full_path))
        
        logger.info(f"Found {len(all_strategy_files)} potential strategy files")
        return all_strategy_files
    
    def validate_strategy_file(self, file_info):
        """Validate a single strategy file."""
        filename, rel_path, full_path = file_info
        
        logger.info(f"Validating {rel_path}")
        
        # Read the file
        try:
            with open(full_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {rel_path}: {e}")
            return {
                'filename': filename,
                'path': rel_path,
                'error': str(e),
                'valid': False
            }
        
        # Check if this is actually a strategy class
        class_match = re.search(r'class\s+(\w+)\(.*Strategy\):', content)
        if not class_match:
            logger.info(f"{rel_path} does not contain a Strategy class, skipping")
            return None
        
        class_name = class_match.group(1)
        
        # Check for strategy registration
        registered = '@register_strategy' in content
        
        # Check for required methods
        methods_found = []
        missing_methods = []
        
        for method in self.required_methods:
            if re.search(rf'def\s+{method}\(', content):
                methods_found.append(method)
            else:
                missing_methods.append(method)
        
        # Check for important methods
        important_found = []
        for method in self.important_methods:
            if re.search(rf'def\s+{method}\(', content):
                important_found.append(method)
        
        # Check for exit strategies
        exit_strategies_found = []
        for strategy in self.exit_strategies:
            if strategy in content.lower():
                exit_strategies_found.append(strategy)
        
        # Check if the strategy follows the pipeline pattern
        follows_pattern = (
            'generate_signals' in content and
            'on_exit_signal' in content and
            'define_universe' in content
        )
        
        # Check for position tracking logic
        has_position_tracking = (
            'position' in content.lower() and
            ('track' in content.lower() or 'monitor' in content.lower())
        )
        
        # Check for risk management
        has_risk_management = (
            'risk' in content.lower() and
            ('max_position' in content.lower() or 'position_size' in content.lower())
        )
        
        # Verify parameter structure
        has_parameters = 'DEFAULT_PARAMS' in content
        
        # Calculate strategy score (0-100%)
        total_checks = (
            len(self.required_methods) + 
            len(self.important_methods) + 
            len(self.exit_strategies) + 
            4  # Pattern, tracking, risk, parameters
        )
        
        passed_checks = (
            len(methods_found) + 
            len(important_found) + 
            len(exit_strategies_found) +
            (1 if follows_pattern else 0) +
            (1 if has_position_tracking else 0) +
            (1 if has_risk_management else 0) +
            (1 if has_parameters else 0)
        )
        
        score = int((passed_checks / total_checks) * 100)
        
        # Create result 
        result = {
            'filename': filename,
            'path': rel_path,
            'class_name': class_name,
            'registered': registered,
            'methods_found': methods_found,
            'missing_methods': missing_methods,
            'important_found': important_found,
            'exit_strategies': exit_strategies_found,
            'follows_pattern': follows_pattern,
            'has_position_tracking': has_position_tracking,
            'has_risk_management': has_risk_management,
            'has_parameters': has_parameters,
            'score': score,
            'valid': score >= 80  # At least 80% score to be considered valid
        }
        
        return result
    
    def validate_against_tradier(self):
        """Basic validation against Tradier paper trading API."""
        logger.info("Validating Tradier paper trading API connection...")
        
        # Tradier paper trading API credentials
        api_key = 'KU2iUnOZIUFre0wypgyOn8TgmGxI'
        account_id = 'VA1201776'
        
        # Test API connection
        import requests
        
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            }
            response = requests.get(
                'https://sandbox.tradier.com/v1/user/profile',
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully connected to Tradier paper API")
                logger.info(f"Account status: {data.get('profile', {}).get('account', {}).get('status', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to Tradier: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Tradier: {e}")
            return False
    
    def validate_against_alpaca(self):
        """Basic validation against Alpaca paper trading API."""
        logger.info("Validating Alpaca paper trading API connection...")
        
        # Alpaca paper trading API credentials
        api_key = 'PKYBHCCT1DIMGZX6P64A'
        api_secret = 'ssidJ2cJU0EGBOhdHrXJd7HegoaPaAMQqs0AU2PO'
        
        # Test API connection
        import requests
        
        try:
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': api_secret
            }
            response = requests.get(
                'https://paper-api.alpaca.markets/v2/account',
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully connected to Alpaca paper API")
                logger.info(f"Account status: {data.get('status', 'unknown')}")
                logger.info(f"Buying power: ${data.get('buying_power', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to Alpaca: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            return False
    
    def run_validation(self):
        """Run the full validation."""
        logger.info("Running strategy validation...")
        
        # Find all strategy files
        strategy_files = self.find_strategy_files()
        
        # Validate each file
        all_results = []
        target_results = []
        
        for file_info in strategy_files:
            filename = file_info[0]
            result = self.validate_strategy_file(file_info)
            
            if result:
                all_results.append(result)
                
                # Track specifically the target strategies
                if filename in self.target_strategies:
                    target_results.append(result)
                    logger.info(f"Target strategy {filename}: Score {result['score']}%")
        
        # Calculate overall statistics
        valid_count = sum(1 for r in all_results if r['valid'])
        average_score = sum(r['score'] for r in all_results) / len(all_results) if all_results else 0
        
        target_valid = sum(1 for r in target_results if r['valid'])
        target_average = sum(r['score'] for r in target_results) / len(target_results) if target_results else 0
        
        # Store overall results
        self.results = {
            'total_strategies': len(all_results),
            'valid_strategies': valid_count,
            'average_score': average_score,
            'target_strategies': len(target_results),
            'target_valid': target_valid,
            'target_average': target_average,
            'strategy_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Display summary
        logger.info("=== Validation Summary ===")
        logger.info(f"Total strategies found: {len(all_results)}")
        logger.info(f"Valid strategies: {valid_count} ({valid_count/len(all_results)*100:.1f}%)")
        logger.info(f"Average score: {average_score:.1f}%")
        logger.info("")
        logger.info(f"Target strategies found: {len(target_results)}/{len(self.target_strategies)}")
        logger.info(f"Valid target strategies: {target_valid} ({target_valid/len(target_results)*100 if target_results else 0:.1f}%)")
        logger.info(f"Target average score: {target_average:.1f}%")
        
        # Validate broker connectivity
        logger.info("\n=== Broker API Validation ===")
        tradier_valid = self.validate_against_tradier()
        alpaca_valid = self.validate_against_alpaca()
        
        # Final validation result
        is_valid = (
            target_valid == len(target_results) and  # All target strategies are valid
            target_average >= 90 and                 # Target strategies average score is >= 90%
            tradier_valid and                        # Tradier API connection works
            alpaca_valid                             # Alpaca API connection works
        )
        
        if is_valid:
            logger.info("\n✅ OVERALL VALIDATION PASSED!")
            logger.info("The implemented strategies are ready for incorporation into your trading pipeline.")
        else:
            logger.warning("\n⚠️ VALIDATION INCOMPLETE")
            if not target_valid == len(target_results):
                logger.warning("- Not all target strategies passed validation")
            if not target_average >= 90:
                logger.warning("- Target strategies average score below 90%")
            if not tradier_valid:
                logger.warning("- Tradier API validation failed")
            if not alpaca_valid:
                logger.warning("- Alpaca API validation failed")
        
        # Save results to file
        self.save_results()
        
        return is_valid
    
    def save_results(self):
        """Save validation results to a file."""
        output_file = os.path.join(self.project_root, 'strategy_validation_results.json')
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Validation results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point."""
    logger.info("=== STRATEGY STRUCTURE VALIDATOR ===")
    validator = StrategyValidator()
    validator.run_validation()
    logger.info("=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
