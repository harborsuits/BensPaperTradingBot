#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Finder

This script performs a filesystem scan to identify all strategy implementations
in the codebase, without requiring dependencies to be installed.
"""

import os
import re
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def scan_for_pattern(directory, pattern, ignore_dirs=None):
    """Scan directory recursively for files matching a pattern."""
    if ignore_dirs is None:
        ignore_dirs = []
    
    found_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if pattern.search(content):
                            rel_path = os.path.relpath(filepath, project_root)
                            found_files.append(rel_path)
                except UnicodeDecodeError:
                    # Skip binary files or files with encoding issues
                    pass
    
    return found_files

def find_strategy_classes():
    """Find all strategy class implementations in the codebase."""
    strategy_pattern = re.compile(r'class\s+\w+(?:Strategy|BaseStrategy)\b')
    strategy_files = scan_for_pattern(
        project_root, 
        strategy_pattern,
        ignore_dirs=['venv', '__pycache__', 'node_modules', '.git']
    )
    return strategy_files

def find_strategy_registrations():
    """Find all strategy registrations via the decorator."""
    registration_pattern = re.compile(r'@register_strategy')
    registered_files = scan_for_pattern(
        project_root, 
        registration_pattern,
        ignore_dirs=['venv', '__pycache__', 'node_modules', '.git']
    )
    return registered_files

def analyze_strategy_file(filepath):
    """Extract information about a strategy file."""
    full_path = os.path.join(project_root, filepath)
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract class name
            class_match = re.search(r'class\s+(\w+)(?:Strategy|BaseStrategy)\b', content)
            class_name = class_match.group(1) + "Strategy" if class_match else "Unknown"
            
            # Check if it's a base class
            is_base = bool(re.search(r'class\s+\w+BaseStrategy\b', content))
            
            # Check if it's registered
            is_registered = '@register_strategy' in content
            
            # Determine strategy type from path
            path_parts = filepath.split(os.path.sep)
            if 'stocks' in path_parts:
                strategy_type = 'stocks'
            elif 'forex' in path_parts:
                strategy_type = 'forex'
            elif 'crypto' in path_parts:
                strategy_type = 'crypto'
            elif 'options' in path_parts:
                strategy_type = 'options'
            else:
                strategy_type = 'unknown'
                
            # Find trading approach
            if 'trend' in path_parts:
                approach = 'trend'
            elif 'momentum' in path_parts:
                approach = 'momentum'
            elif 'breakout' in path_parts:
                approach = 'breakout'
            elif 'mean_reversion' in path_parts or 'range' in path_parts:
                approach = 'mean_reversion'
            elif 'gap' in path_parts:
                approach = 'gap'
            elif 'volume' in path_parts:
                approach = 'volume'
            elif 'event' in path_parts or 'news' in path_parts or 'earnings' in path_parts:
                approach = 'event_driven'
            elif 'sector' in path_parts:
                approach = 'sector'
            elif 'short' in path_parts:
                approach = 'short'
            else:
                approach = 'unknown'
            
            return {
                'file': filepath,
                'class_name': class_name,
                'is_base': is_base,
                'is_registered': is_registered,
                'market_type': strategy_type,
                'approach': approach
            }
    except Exception as e:
        return {
            'file': filepath,
            'error': str(e),
            'class_name': 'Error',
            'is_base': False,
            'is_registered': False,
            'market_type': 'unknown',
            'approach': 'unknown'
        }

def group_strategies_by_type(strategies):
    """Group strategies by market type and approach."""
    grouped = {}
    
    for strategy in strategies:
        market_type = strategy['market_type']
        approach = strategy['approach']
        
        if market_type not in grouped:
            grouped[market_type] = {}
            
        if approach not in grouped[market_type]:
            grouped[market_type][approach] = []
            
        grouped[market_type][approach].append(strategy)
    
    return grouped

def main():
    """Main function to find and analyze strategies."""
    print("Scanning for strategy implementations...")
    
    # Find strategy files
    strategy_files = find_strategy_classes()
    print(f"Found {len(strategy_files)} strategy files")
    
    # Find registered strategy files
    registered_files = find_strategy_registrations()
    print(f"Found {len(registered_files)} registered strategy files")
    
    # Analyze each strategy file
    strategies = []
    for filepath in strategy_files:
        info = analyze_strategy_file(filepath)
        strategies.append(info)
    
    # Group strategies by type
    grouped = group_strategies_by_type(strategies)
    
    # Print summary by market type
    print("\nStrategy Summary by Market Type:")
    print("===============================")
    
    total_registered = sum(1 for s in strategies if s['is_registered'])
    total_base = sum(1 for s in strategies if s['is_base'])
    total_standard = len(strategies) - total_base
    
    for market_type, approaches in sorted(grouped.items()):
        market_strategies = [s for approach_strategies in approaches.values() for s in approach_strategies]
        registered_count = sum(1 for s in market_strategies if s['is_registered'])
        
        print(f"\n{market_type.upper()} Strategies: {len(market_strategies)} total, {registered_count} registered")
        
        for approach, approach_strategies in sorted(approaches.items()):
            approach_registered = sum(1 for s in approach_strategies if s['is_registered'])
            print(f"  - {approach.replace('_', ' ').title()}: {len(approach_strategies)} strategies, {approach_registered} registered")
            
            for strategy in sorted(approach_strategies, key=lambda s: s['class_name']):
                reg_status = "✓" if strategy['is_registered'] else " "
                base_status = "B" if strategy['is_base'] else " "
                print(f"    [{reg_status}][{base_status}] {strategy['class_name']} ({strategy['file']})")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("==================")
    print(f"Total strategy files: {len(strategies)}")
    print(f"Base strategy classes: {total_base}")
    print(f"Standard strategy classes: {total_standard}")
    print(f"Registered with system: {total_registered} ({total_registered/total_standard*100:.1f}% of standard strategies)")
    
    # Print recommendations
    print("\nRecommendations:")
    print("==============")
    if total_registered < total_standard:
        print(f"- {total_standard - total_registered} strategies are not registered with the system")
        print("  Consider adding @register_strategy decorators to these strategies")
    
    # Print list of unregistered strategies
    unregistered = [s for s in strategies if not s['is_base'] and not s['is_registered']]
    if unregistered:
        print("\nUnregistered strategies that should be registered:")
        for strategy in sorted(unregistered, key=lambda s: s['class_name']):
            print(f"- {strategy['class_name']} ({strategy['file']})")

if __name__ == "__main__":
    main()
