#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unregistered Strategy Finder

This script focuses on strategies in the new 'strategies_new' directory
and identifies which ones still need to be registered with the system.
"""

import os
import re
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def find_all_strategy_files(directory):
    """Find all strategy implementation files in the directory."""
    strategy_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and ('strategy' in file.lower() or 'base' in file.lower()):
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, project_root)
                strategy_files.append(relative_path)
    
    return strategy_files

def analyze_strategy_file(filepath):
    """Check if a strategy file contains class definitions and registration."""
    full_path = os.path.join(project_root, filepath)
    results = []
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Join lines to get full content for regex pattern matching
            content = ''.join(lines)
            
            # Find all strategy class definitions
            class_pattern = re.compile(r'class\s+(\w+)(?:Strategy|BaseStrategy)\b')
            class_matches = list(class_pattern.finditer(content))
            
            for match in class_matches:
                class_name = match.group(1)
                if 'BaseStrategy' in match.group(0):
                    class_name += 'BaseStrategy'
                else:
                    class_name += 'Strategy'
                
                # Find the line number of this class definition
                class_pos = match.start()
                class_line_num = content[:class_pos].count('\n')
                
                # Check if this class is registered by looking for @register_strategy
                # before the class definition but after any previous class
                is_registered = False
                
                # Determine the starting point - either after the previous class or at the beginning
                start_line = 0
                if class_matches.index(match) > 0:
                    prev_match = class_matches[class_matches.index(match) - 1]
                    prev_class_pos = prev_match.start()
                    start_line = content[:prev_class_pos].count('\n')  # Line number of previous class
                
                # Check for decorator in the lines between previous class and current class
                for line_num in range(start_line, class_line_num):
                    if line_num < len(lines):
                        line = lines[line_num]
                        # Look for @register_strategy decorator
                        if '@register_strategy' in line:
                            is_registered = True
                            break
                
                # Determine if it's a base class
                is_base = 'BaseStrategy' in class_name
                
                results.append({
                    'file': filepath,
                    'class_name': class_name,
                    'is_registered': is_registered,
                    'is_base': is_base
                })
    except Exception as e:
        print(f"Error analyzing {filepath}: {str(e)}")
    
    return results

def main():
    """Main function."""
    print("Finding unregistered strategies in the 'strategies_new' directory...")
    
    # Find all strategy files in the new directory
    strategies_new_dir = os.path.join(project_root, 'trading_bot', 'strategies_new')
    strategy_files = find_all_strategy_files(strategies_new_dir)
    
    print(f"Found {len(strategy_files)} potential strategy files")
    
    # Analyze each file
    all_strategies = []
    for filepath in strategy_files:
        strategies = analyze_strategy_file(filepath)
        all_strategies.extend(strategies)
    
    # Filter base classes and count statistics
    standard_strategies = [s for s in all_strategies if not s['is_base']]
    registered_strategies = [s for s in standard_strategies if s['is_registered']]
    unregistered_strategies = [s for s in standard_strategies if not s['is_registered']]
    
    base_strategies = [s for s in all_strategies if s['is_base']]
    
    # Print statistics
    print(f"\nTotal strategy classes found: {len(all_strategies)}")
    print(f"Base strategy classes: {len(base_strategies)}")
    print(f"Standard strategy classes: {len(standard_strategies)}")
    print(f"Registered with system: {len(registered_strategies)} ({len(registered_strategies)/len(standard_strategies)*100:.1f}% of standard strategies)")
    print(f"Unregistered strategies: {len(unregistered_strategies)}")
    
    # Print unregistered strategies grouped by directory
    if unregistered_strategies:
        print("\nUnregistered strategies that should be registered:")
        
        # Group by directory
        by_directory = {}
        for strategy in unregistered_strategies:
            directory = os.path.dirname(strategy['file'])
            if directory not in by_directory:
                by_directory[directory] = []
            by_directory[directory].append(strategy)
        
        for directory, strategies in sorted(by_directory.items()):
            print(f"\n{directory}:")
            for strategy in sorted(strategies, key=lambda s: s['class_name']):
                print(f"  - {strategy['class_name']} ({os.path.basename(strategy['file'])})")
    else:
        print("\nCongratulations! All standard strategies are registered.")

if __name__ == "__main__":
    main()
