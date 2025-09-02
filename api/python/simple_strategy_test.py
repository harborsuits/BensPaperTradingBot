#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify that our strategy classes work as expected.
This script avoids complex imports from the trading_bot package.
"""

import sys
import os
import inspect
from pprint import pprint

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import only the required strategy files directly
bull_call_path = os.path.join(project_root, 'trading_bot/strategies/options/vertical_spreads/bull_call_spread_strategy_new.py')
covered_call_path = os.path.join(project_root, 'trading_bot/strategies/options/income_strategies/covered_call_strategy_new.py')

# Import the strategy classes directly using exec
bull_call_namespace = {}
covered_call_namespace = {}

try:
    with open(bull_call_path, 'r') as f:
        bull_call_code = f.read()
    exec(bull_call_code, bull_call_namespace)
    print("✅ Successfully imported Bull Call Spread strategy code")
except Exception as e:
    print(f"❌ Error importing Bull Call Spread strategy: {e}")
    
try:
    with open(covered_call_path, 'r') as f:
        covered_call_code = f.read()
    exec(covered_call_code, covered_call_namespace)
    print("✅ Successfully imported Covered Call strategy code")
except Exception as e:
    print(f"❌ Error importing Covered Call strategy: {e}")

# Check if the strategy classes exist in the namespace
print("\nChecking for strategy classes:")
if 'BullCallSpreadStrategy' in bull_call_namespace:
    print(f"✅ BullCallSpreadStrategy class found")
    bull_call_class = bull_call_namespace['BullCallSpreadStrategy']
    print(f"Class docstring: {bull_call_class.__doc__.strip().split('\\n')[0]}")
else:
    print("❌ BullCallSpreadStrategy class not found")

if 'CoveredCallStrategy' in covered_call_namespace:
    print(f"✅ CoveredCallStrategy class found")
    covered_call_class = covered_call_namespace['CoveredCallStrategy']
    print(f"Class docstring: {covered_call_class.__doc__.strip().split('\\n')[0]}")
else:
    print("❌ CoveredCallStrategy class not found")

# Check for the parameter dictionaries
print("\nChecking for strategy parameters:")
if 'BullCallSpreadStrategy' in bull_call_namespace:
    bull_call_class = bull_call_namespace['BullCallSpreadStrategy']
    if hasattr(bull_call_class, 'DEFAULT_PARAMS'):
        print("✅ BullCallSpreadStrategy has DEFAULT_PARAMS")
        print(f"Strategy name: {bull_call_class.DEFAULT_PARAMS.get('strategy_name')}")
        print(f"Asset class: {bull_call_class.DEFAULT_PARAMS.get('asset_class')}")
    else:
        print("❌ BullCallSpreadStrategy does not have DEFAULT_PARAMS")

if 'CoveredCallStrategy' in covered_call_namespace:
    covered_call_class = covered_call_namespace['CoveredCallStrategy']
    if hasattr(covered_call_class, 'DEFAULT_PARAMS'):
        print("✅ CoveredCallStrategy has DEFAULT_PARAMS")
        print(f"Strategy name: {covered_call_class.DEFAULT_PARAMS.get('strategy_name')}")
        print(f"Asset class: {covered_call_class.DEFAULT_PARAMS.get('asset_class')}")
    else:
        print("❌ CoveredCallStrategy does not have DEFAULT_PARAMS")

print("\nTest completed successfully")
