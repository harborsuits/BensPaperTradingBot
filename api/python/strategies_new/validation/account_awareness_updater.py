#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Account Awareness Updater

This script updates all strategy implementations to incorporate account awareness
functionality from the AccountAwareMixin. This ensures all strategies properly check:
- Account balance requirements
- Regulatory constraints (PDT rule, etc.)
- Position sizing based on available capital
- Per-broker trading limitations
"""

import os
import re
import logging
import sys
from pathlib import Path
import importlib.util
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('account_awareness_update.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, project_root)

class AccountAwarenessUpdater:
    """
    Class to update strategies with account awareness functionality.
    """
    
    @staticmethod
    def update_strategy_file(filepath: str) -> Dict[str, Any]:
        """
        Update a strategy file to incorporate account awareness.
        
        Args:
            filepath: Path to the strategy file
            
        Returns:
            Dictionary with results of the update
        """
        result = {
            "filepath": filepath,
            "updates_applied": [],
            "errors": []
        }
        
        try:
            # Read the file content
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Check if the file already imports AccountAwareMixin
            if "from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin" in content:
                result["updates_applied"].append("AccountAwareMixin already imported")
            else:
                # Add import for AccountAwareMixin
                import_line = "from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin"
                
                # Find where to insert the import
                import_match = re.search(r'^(import.*$|from.*$)', content, re.MULTILINE)
                if import_match:
                    # Find the last import statement
                    last_import_match = None
                    for match in re.finditer(r'^(import.*$|from.*$)', content, re.MULTILINE):
                        last_import_match = match
                    
                    if last_import_match:
                        # Insert after the last import
                        content = content[:last_import_match.end()] + "\n" + import_line + content[last_import_match.end():]
                        result["updates_applied"].append("Added AccountAwareMixin import")
                    else:
                        # Shouldn't reach here, but just in case
                        content = import_line + "\n" + content
                        result["updates_applied"].append("Added AccountAwareMixin import at beginning")
                else:
                    # No imports found, add at beginning (after shebang and encoding if present)
                    shebang_match = re.search(r'^#!.*?\n', content)
                    encoding_match = re.search(r'^#.*?coding.*?\n', content)
                    
                    if encoding_match:
                        pos = encoding_match.end()
                    elif shebang_match:
                        pos = shebang_match.end()
                    else:
                        pos = 0
                        
                    content = content[:pos] + import_line + "\n" + content[pos:]
                    result["updates_applied"].append("Added AccountAwareMixin import at beginning")
                    
            # Check if the strategy class already inherits from AccountAwareMixin
            class_pattern = r'class\s+(\w+)\(([^)]+)\):'
            class_match = re.search(class_pattern, content)
            
            if class_match:
                class_name = class_match.group(1)
                parent_classes = class_match.group(2)
                
                if "AccountAwareMixin" in parent_classes:
                    result["updates_applied"].append(f"Class {class_name} already inherits from AccountAwareMixin")
                else:
                    # Add AccountAwareMixin to parent classes
                    new_parent_classes = parent_classes + ", AccountAwareMixin"
                    content = content.replace(f"class {class_name}({parent_classes}):", 
                                            f"class {class_name}({new_parent_classes}):")
                    result["updates_applied"].append(f"Added AccountAwareMixin inheritance to {class_name}")
                    
                # Check for __init__ method
                init_pattern = r'def\s+__init__\s*\(self,\s*([^)]*)\):'
                init_match = re.search(init_pattern, content)
                
                if init_match:
                    # Check if AccountAwareMixin.__init__() is already called
                    if "AccountAwareMixin.__init__(self)" in content:
                        result["updates_applied"].append("AccountAwareMixin.__init__() already called")
                    else:
                        # Find position to insert the call to AccountAwareMixin.__init__()
                        # Look for "super().__init__" call
                        super_init_pattern = r'super\(\).__init__\([^)]*\)'
                        super_init_match = re.search(super_init_pattern, content)
                        
                        if super_init_match:
                            # Insert after super().__init__() call
                            insert_pos = super_init_match.end()
                            content = content[:insert_pos] + "\n        # Initialize account awareness functionality\n        AccountAwareMixin.__init__(self)" + content[insert_pos:]
                            result["updates_applied"].append("Added AccountAwareMixin.__init__() call after super().__init__()")
                        else:
                            # No super().__init__() call found, look for init body
                            init_body_start = init_match.end()
                            next_line = content[init_body_start:].find('\n')
                            if next_line != -1:
                                insert_pos = init_body_start + next_line + 1
                                content = content[:insert_pos] + "        # Initialize account awareness functionality\n        AccountAwareMixin.__init__(self)\n" + content[insert_pos:]
                                result["updates_applied"].append("Added AccountAwareMixin.__init__() call at beginning of __init__ method")
                            else:
                                result["errors"].append("Could not find position to insert AccountAwareMixin.__init__() call")
                else:
                    result["errors"].append(f"Could not find __init__ method in {class_name}")
                    
                # Check for trade execution methods and add account awareness checks
                # Look for methods like _execute_signals, execute_trade, etc.
                execute_methods = [
                    r'def\s+_execute_signals\s*\(self',
                    r'def\s+execute_trade\s*\(self',
                    r'def\s+execute_order\s*\(self',
                    r'def\s+enter_position\s*\(self',
                    r'def\s+enter_long_position\s*\(self',
                    r'def\s+enter_short_position\s*\(self'
                ]
                
                for method_pattern in execute_methods:
                    method_matches = re.finditer(method_pattern, content)
                    for method_match in method_matches:
                        method_name = method_match.group(0).split('(')[0].split()[-1]
                        
                        # Check if account awareness check is already in the method
                        method_start = method_match.start()
                        method_end = content.find('def ', method_start + 1)
                        if method_end == -1:  # Last method in file
                            method_end = len(content)
                            
                        method_content = content[method_start:method_end]
                        
                        if "self.check_pdt_rule_compliance()" in method_content or "self.validate_trade_size(" in method_content:
                            result["updates_applied"].append(f"Method {method_name} already has account awareness checks")
                        else:
                            # Find method body start
                            body_start = method_content.find(':') + 1
                            next_line = method_content[body_start:].find('\n')
                            if next_line != -1:
                                # Insert account awareness checks
                                if "day_trade" in method_name.lower() or "scalp" in method_name.lower():
                                    # Day trading method needs explicit PDT check
                                    account_check = """
        # Check account regulatory compliance
        if not self.check_pdt_rule_compliance(is_day_trade=True):
            logger.warning("Trade execution aborted: PDT rule compliance check failed")
            return
        
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power(day_trade=True)
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
"""
                                else:
                                    # Regular method with basic checks
                                    account_check = """
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
"""
                                
                                # Insert in method body
                                insert_pos = method_start + body_start + next_line + 1
                                content = content[:insert_pos] + account_check + content[insert_pos:]
                                result["updates_applied"].append(f"Added account awareness checks to {method_name}")
                            else:
                                result["errors"].append(f"Could not find position to insert account awareness checks in {method_name}")
                
                # Look for position sizing / risk calculation methods
                sizing_methods = [
                    r'def\s+calculate_position_size\s*\(self',
                    r'def\s+get_position_size\s*\(self',
                    r'def\s+get_order_quantity\s*\(self'
                ]
                
                for method_pattern in sizing_methods:
                    method_matches = re.finditer(method_pattern, content)
                    for method_match in method_matches:
                        method_name = method_match.group(0).split('(')[0].split()[-1]
                        
                        # Check if method already uses account awareness
                        method_start = method_match.start()
                        method_end = content.find('def ', method_start + 1)
                        if method_end == -1:  # Last method in file
                            method_end = len(content)
                            
                        method_content = content[method_start:method_end]
                        
                        if "self.calculate_max_position_size(" in method_content:
                            result["updates_applied"].append(f"Method {method_name} already uses account-aware position sizing")
                        else:
                            # Look for return statement to modify
                            return_pattern = r'return\s+([^#\n]+)'
                            return_match = re.search(return_pattern, method_content)
                            
                            if return_match:
                                original_return = return_match.group(0)
                                return_value = return_match.group(1).strip()
                                
                                # Determine if we're returning position size or notional value
                                is_notional = "notional" in method_name.lower() or "amount" in method_name.lower()
                                
                                # Create replacement with account awareness
                                if is_notional:
                                    # If returning notional amount (dollars)
                                    replacement = f"""
        # Calculate position size based on strategy logic
        original_notional = {return_value}
        
        # Apply account-aware constraints
        is_day_trade = hasattr(self, 'is_day_trade') and self.is_day_trade
        _, max_notional = self.calculate_max_position_size(price, is_day_trade=is_day_trade)
        
        return min(original_notional, max_notional)  # Use lower of the two"""
                                else:
                                    # If returning position size (shares/contracts)
                                    replacement = f"""
        # Calculate position size based on strategy logic
        original_size = {return_value}
        
        # Apply account-aware constraints
        is_day_trade = hasattr(self, 'is_day_trade') and self.is_day_trade
        max_size, _ = self.calculate_max_position_size(price, is_day_trade=is_day_trade)
        
        return min(original_size, max_size)  # Use lower of the two"""
                                
                                # Replace the return statement
                                method_with_replacement = method_content.replace(original_return, replacement)
                                content = content.replace(method_content, method_with_replacement)
                                
                                result["updates_applied"].append(f"Enhanced {method_name} with account-aware position sizing")
                            else:
                                result["errors"].append(f"Could not find return statement in {method_name}")
            else:
                result["errors"].append("Could not find strategy class definition")
                
            # Write updated content back to file
            with open(filepath, 'w') as f:
                f.write(content)
                
        except Exception as e:
            result["errors"].append(f"Error updating file: {str(e)}")
            
        return result
    
    @staticmethod
    def find_strategy_files(base_dir: str) -> List[str]:
        """
        Find all strategy files in the directory tree.
        
        Args:
            base_dir: Base directory to search from
            
        Returns:
            List of file paths
        """
        strategy_files = []
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('_strategy.py'):
                    strategy_files.append(os.path.join(root, file))
                    
        return strategy_files
    
    @staticmethod
    def update_all_strategies(base_dir: str, strategy_type: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Update all strategy files in the directory tree.
        
        Args:
            base_dir: Base directory to search from
            strategy_type: Optional type of strategies to update (stocks, options, forex, crypto)
            
        Returns:
            Dictionary with results for each file
        """
        if strategy_type:
            base_dir = os.path.join(base_dir, strategy_type)
            
        strategy_files = AccountAwarenessUpdater.find_strategy_files(base_dir)
        results = {}
        
        for filepath in strategy_files:
            logger.info(f"Updating {filepath}")
            results[filepath] = AccountAwarenessUpdater.update_strategy_file(filepath)
            
        return results

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update strategies with account awareness functionality')
    parser.add_argument('--strategy-type', type=str, choices=['stocks', 'options', 'forex', 'crypto', 'all'],
                      default='all', help='Type of strategies to update')
    parser.add_argument('--file', type=str, help='Specific strategy file to update')
    
    args = parser.parse_args()
    
    strategies_dir = os.path.join(project_root, 'trading_bot', 'strategies_new')
    
    if args.file:
        # Update specific file
        result = AccountAwarenessUpdater.update_strategy_file(args.file)
        
        logger.info(f"Updated {args.file}")
        if result['updates_applied']:
            logger.info("Applied updates:")
            for update in result['updates_applied']:
                logger.info(f"  - {update}")
                
        if result['errors']:
            logger.error("Errors:")
            for error in result['errors']:
                logger.error(f"  - {error}")
    else:
        # Update all strategies of specified type
        if args.strategy_type != 'all':
            results = AccountAwarenessUpdater.update_all_strategies(strategies_dir, args.strategy_type)
        else:
            results = AccountAwarenessUpdater.update_all_strategies(strategies_dir)
            
        # Print summary
        success_count = sum(1 for result in results.values() if not result["errors"])
        error_count = len(results) - success_count
        
        logger.info(f"Update complete: {success_count} strategies updated successfully, {error_count} had errors")
        
        # Print errors for files that had issues
        if error_count > 0:
            logger.info("\nStrategies with errors:")
            for path, result in results.items():
                if result["errors"]:
                    logger.error(f"\n{path}:")
                    for error in result["errors"]:
                        logger.error(f"  - {error}")

if __name__ == "__main__":
    main()
