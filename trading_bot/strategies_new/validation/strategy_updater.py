#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Updater

This module provides utility functions to update existing strategies to match
the standardized best practices from our Iron Condor implementation. It applies
fixes for error handling, validation logic, event subscriptions, and adds
adjustment methods from the strategy_adjustments module.
"""

import logging
import inspect
import os
import re
import ast
from typing import Dict, Any, List, Optional
import importlib
import sys
from pathlib import Path

from trading_bot.strategies_new.validation.strategy_validator import StrategyValidator

# Configure logging
logger = logging.getLogger(__name__)

class StrategyUpdater:
    """
    Updates existing strategies to conform to standardized best practices.
    """
    
    @staticmethod
    def update_strategy(strategy_path: str) -> Dict[str, Any]:
        """
        Update a strategy implementation to follow best practices.
        
        Args:
            strategy_path: Path to the strategy file
            
        Returns:
            Dictionary with update results
        """
        results = {
            "strategy_path": strategy_path,
            "updates_applied": [],
            "updates_skipped": [],
            "errors": []
        }
        
        try:
            # Read the file
            with open(strategy_path, 'r') as f:
                content = f.read()
            
            # Backup the original file
            backup_path = strategy_path + '.bak'
            with open(backup_path, 'w') as f:
                f.write(content)
            results["updates_applied"].append(f"Created backup at {backup_path}")
            
            # Apply updates to the content
            updated_content = content
            
            # Add missing imports
            if "strategy_adjustments" not in updated_content:
                updated_content = StrategyUpdater._add_strategy_adjustments_import(updated_content)
                results["updates_applied"].append("Added strategy_adjustments import")
            
            # Add error handling to critical methods
            updated_content, error_handling_updates = StrategyUpdater._add_error_handling(updated_content)
            results["updates_applied"].extend(error_handling_updates)
            
            # Add event subscriptions
            updated_content, event_updates = StrategyUpdater._add_event_subscriptions(updated_content)
            results["updates_applied"].extend(event_updates)
            
            # Add validation checks
            updated_content, validation_updates = StrategyUpdater._add_validation_checks(updated_content)
            results["updates_applied"].extend(validation_updates)
            
            # Add inheritance from StrategyAdjustments
            updated_content, inheritance_update = StrategyUpdater._add_strategy_adjustments_inheritance(updated_content)
            if inheritance_update:
                results["updates_applied"].append("Added StrategyAdjustments inheritance")
            
            # Write the updated content back to the file
            with open(strategy_path, 'w') as f:
                f.write(updated_content)
            
            if content == updated_content:
                results["updates_skipped"].append("No changes needed or could be applied automatically")
            else:
                results["updates_applied"].append("Updated strategy file with standardized best practices")
                
        except Exception as e:
            results["errors"].append(f"Error updating strategy: {str(e)}")
            
        return results
    
    @staticmethod
    def update_all_strategies(strategies_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Update all strategies in the given directory.
        
        Args:
            strategies_dir: Directory containing strategy modules
            
        Returns:
            Dictionary mapping strategy paths to update results
        """
        results = {}
        
        # Get all Python files in the directory and subdirectories
        for dirpath, dirnames, filenames in os.walk(strategies_dir):
            for filename in [f for f in filenames if f.endswith('_strategy.py') and not f.startswith('__')]:
                strategy_path = os.path.join(dirpath, filename)
                
                # Update the strategy
                results[strategy_path] = StrategyUpdater.update_strategy(strategy_path)
                
        return results
    
    @staticmethod
    def _add_strategy_adjustments_import(content: str) -> str:
        """Add import for strategy_adjustments if needed."""
        import_lines = []
        
        # Find all import statements
        for line in content.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
        
        # Check if we already have the import
        if not any('strategy_adjustments' in line for line in import_lines):
            # Find the last import line
            last_import_idx = 0
            for i, line in enumerate(content.split('\n')):
                if line.startswith('import ') or line.startswith('from '):
                    last_import_idx = i
            
            # Insert our import after the last import
            content_lines = content.split('\n')
            content_lines.insert(last_import_idx + 1, 'from trading_bot.strategies_new.options.base.strategy_adjustments import StrategyAdjustments')
            content = '\n'.join(content_lines)
            
        return content
    
    @staticmethod
    def _add_error_handling(content: str) -> (str, List[str]):
        """Add error handling to critical methods."""
        updates = []
        
        # Define critical methods that should have error handling
        critical_methods = [
            'on_event',
            '_execute_signals',
            'filter_option_chains',
            'construct_iron_condor',
            'construct_butterfly',
            'construct_straddle',
            'construct_strangle',
            'construct_calendar_spread'
        ]
        
        # Pattern to match method definitions
        method_pattern = r'def\s+({}).*?\):'
        
        for method in critical_methods:
            pattern = method_pattern.format(method)
            match = re.search(pattern, content)
            
            if match:
                # Check if the method already has error handling
                method_start = match.start()
                next_def = content.find("def ", method_start + 1)
                if next_def == -1:
                    next_def = len(content)
                    
                method_content = content[method_start:next_def]
                
                if "try:" not in method_content or "except" not in method_content:
                    # Find the body indentation
                    lines = method_content.split('\n')
                    body_start = 1  # Skip the 'def' line
                    
                    # Find the first non-empty line in the body
                    while body_start < len(lines) and (not lines[body_start].strip() or lines[body_start].strip().startswith('#')):
                        body_start += 1
                        
                    if body_start < len(lines):
                        # Compute the indentation
                        indentation = ''
                        for char in lines[body_start]:
                            if char in [' ', '\t']:
                                indentation += char
                            else:
                                break
                        
                        # Add error handling
                        try_block = f"{indentation}try:\n"
                        except_block = f"\n{indentation}except Exception as e:\n{indentation}    logger.error(f\"Error in {method}: {{str(e)}}\")"
                        
                        # Insert the try-except block
                        new_method_content = lines[0] + '\n'  # The 'def' line
                        
                        # Add comment lines and docstring
                        for i in range(1, body_start):
                            new_method_content += lines[i] + '\n'
                        
                        # Add try block
                        new_method_content += try_block
                        
                        # Add indented body
                        for i in range(body_start, len(lines)):
                            new_method_content += f"{indentation}    {lines[i].strip()}\n"
                        
                        # Add except block
                        new_method_content += except_block
                        
                        # Replace the old method with the new one
                        content = content.replace(method_content, new_method_content)
                        updates.append(f"Added error handling to {method} method")
        
        return content, updates
    
    @staticmethod
    def _add_event_subscriptions(content: str) -> (str, List[str]):
        """Add required event subscriptions to register_events method."""
        updates = []
        
        # Find the register_events method
        register_pattern = r'def\s+register_events.*?\):'
        match = re.search(register_pattern, content)
        
        if match:
            # Check the method body
            method_start = match.start()
            next_def = content.find("def ", method_start + 1)
            if next_def == -1:
                next_def = len(content)
                
            method_content = content[method_start:next_def]
            
            # Required event subscriptions
            required_events = [
                "EventType.VOLATILITY_SPIKE",
                "EventType.MARKET_REGIME_CHANGE"
            ]
            
            # Check which events are missing
            missing_events = []
            for event in required_events:
                if event not in method_content:
                    missing_events.append(event)
            
            if missing_events:
                # Find the end of the method body
                lines = method_content.split('\n')
                
                # Find the indentation
                body_start = 1  # Skip the 'def' line
                while body_start < len(lines) and (not lines[body_start].strip() or lines[body_start].strip().startswith('#')):
                    body_start += 1
                    
                if body_start < len(lines):
                    indentation = ''
                    for char in lines[body_start]:
                        if char in [' ', '\t']:
                            indentation += char
                        else:
                            break
                    
                    # Add missing event subscriptions
                    subscription_lines = []
                    for event in missing_events:
                        subscription_lines.append(f"{indentation}EventBus.subscribe({event}, self.on_event)")
                    
                    # Insert at the end of the method
                    new_method_content = method_content
                    if subscription_lines:
                        if not method_content.strip().endswith('pass'):
                            new_method_content += '\n'
                        new_method_content += '\n'.join(subscription_lines)
                    
                    # Replace the old method with the new one
                    content = content.replace(method_content, new_method_content)
                    
                    for event in missing_events:
                        updates.append(f"Added subscription to {event}")
        
        return content, updates
    
    @staticmethod
    def _add_validation_checks(content: str) -> (str, List[str]):
        """Add validation checks to position construction methods."""
        updates = []
        
        # Construction methods to check
        construction_methods = [
            'construct_iron_condor',
            'construct_butterfly', 
            'construct_straddle', 
            'construct_strangle',
            'construct_calendar_spread',
            'construct_bull_call_spread',
            'construct_bear_put_spread',
            'construct_bull_put_spread'
        ]
        
        # Validation checks to add
        validation_code = """
        # Calculate expected profit metrics
        profit_potential = net_credit
        loss_potential = max_loss - net_credit
        risk_reward_ratio = loss_potential / profit_potential if profit_potential > 0 else float('inf')
        
        # Check if risk/reward ratio is acceptable
        max_risk_reward = self.parameters.get('max_risk_reward_ratio', 3.0)
        if risk_reward_ratio > max_risk_reward:
            logger.info(f"Position rejected: risk/reward ratio {risk_reward_ratio:.2f} exceeds maximum {max_risk_reward:.2f}")
            return None
        """
        
        # Method pattern
        method_pattern = r'def\s+({}).*?\):'
        
        for method in construction_methods:
            pattern = method_pattern.format(method)
            match = re.search(pattern, content)
            
            if match:
                # Find the return statement
                method_start = match.start()
                next_def = content.find("def ", method_start + 1)
                if next_def == -1:
                    next_def = len(content)
                    
                method_content = content[method_start:next_def]
                
                # Check if the method already has validation
                if "risk_reward_ratio" not in method_content:
                    # Find the return statement
                    return_match = re.search(r'return\s+\w+', method_content)
                    
                    if return_match:
                        # Get the indentation
                        lines = method_content.split('\n')
                        for line in lines:
                            if line.strip().startswith('return'):
                                indentation = ''
                                for char in line:
                                    if char in [' ', '\t']:
                                        indentation += char
                                    else:
                                        break
                                break
                        
                        # Format the validation code with proper indentation
                        formatted_validation = '\n'.join([f"{indentation}{line}" for line in validation_code.strip().split('\n')])
                        
                        # Insert before the return statement
                        insert_pos = method_content.rfind('\n', 0, return_match.start())
                        if insert_pos == -1:
                            insert_pos = 0
                            
                        new_method_content = method_content[:insert_pos] + '\n' + formatted_validation + method_content[insert_pos:]
                        
                        # Replace the old method with the new one
                        content = content.replace(method_content, new_method_content)
                        updates.append(f"Added validation checks to {method} method")
        
        return content, updates
    
    @staticmethod
    def _add_strategy_adjustments_inheritance(content: str) -> (str, bool):
        """Add StrategyAdjustments to the inheritance hierarchy."""
        # Find the class definition
        class_pattern = r'class\s+(\w+)\s*\(([^)]*)\)\s*:'
        match = re.search(class_pattern, content)
        
        if match:
            class_name = match.group(1)
            base_classes = match.group(2).strip()
            
            # Check if StrategyAdjustments is already in the base classes
            if 'StrategyAdjustments' not in base_classes:
                # Add StrategyAdjustments to the base classes
                new_base_classes = base_classes
                if new_base_classes:
                    new_base_classes += ', StrategyAdjustments'
                else:
                    new_base_classes = 'StrategyAdjustments'
                
                # Replace the class definition
                new_class_def = f'class {class_name}({new_base_classes}):'
                content = content.replace(f'class {class_name}({base_classes}):', new_class_def)
                return content, True
        
        return content, False


if __name__ == "__main__":
    # Simple command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Update strategies to match best practices')
    parser.add_argument('--path', type=str, required=True, help='Path to strategy file or directory')
    parser.add_argument('--all', action='store_true', help='Update all strategies in directory')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.all:
        results = StrategyUpdater.update_all_strategies(args.path)
        for path, result in results.items():
            if result["errors"]:
                logger.error(f"Errors updating {path}:")
                for error in result["errors"]:
                    logger.error(f"  {error}")
            else:
                logger.info(f"Successfully updated {path}")
                for update in result["updates_applied"]:
                    logger.info(f"  {update}")
    else:
        result = StrategyUpdater.update_strategy(args.path)
        if result["errors"]:
            logger.error("Errors:")
            for error in result["errors"]:
                logger.error(f"  {error}")
        else:
            logger.info("Updates applied:")
            for update in result["updates_applied"]:
                logger.info(f"  {update}")
            
            if result["updates_skipped"]:
                logger.info("Updates skipped:")
                for update in result["updates_skipped"]:
                    logger.info(f"  {update}")
