#!/usr/bin/env python3
"""
Strategy Adapter Verification Script

This standalone script tests that all registered strategies are:
1. Properly adapted with the StrategyAdapter
2. Expose the standardized interface methods
3. Can be instantiated correctly

No external dependencies required.
"""

import os
import sys
import importlib
import inspect
from typing import List, Dict, Any, Type

# List of strategies to verify
STRATEGIES_TO_TEST = [
    "IronCondorStrategy", 
    "StrangleStrategy",
    "ButterflySpreadStrategy", 
    "CalendarSpreadStrategy"
]

# Expected interface methods
REQUIRED_METHODS = [
    "generate_signals",
    "size_position",
    "manage_open_trades"
]

class VerificationResult:
    """Container for verification results"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []
        self.warnings = []
        
    def add_pass(self, strategy_name, message):
        self.passed.append((strategy_name, message))
        
    def add_fail(self, strategy_name, message):
        self.failed.append((strategy_name, message))
        
    def add_error(self, strategy_name, error):
        self.errors.append((strategy_name, str(error)))
        
    def add_warning(self, strategy_name, message):
        self.warnings.append((strategy_name, message))
        
    def all_passed(self):
        return len(self.failed) == 0 and len(self.errors) == 0
    
    def generate_report(self):
        """Generate a report of the verification results"""
        report = []
        report.append("\n" + "="*70)
        report.append("STRATEGY ADAPTER VERIFICATION REPORT")
        report.append("="*70 + "\n")
        
        # Summary
        report.append(f"Strategies tested: {len(self.passed) + len(self.failed) + len(self.errors)}")
        report.append(f"Passed: {len(self.passed)}")
        report.append(f"Failed: {len(self.failed)}")
        report.append(f"Errors: {len(self.errors)}")
        report.append(f"Warnings: {len(self.warnings)}")
        report.append("")
        
        # Details
        if self.passed:
            report.append("PASSED:")
            for strategy, message in self.passed:
                report.append(f"  ✅ {strategy}: {message}")
            report.append("")
            
        if self.failed:
            report.append("FAILED:")
            for strategy, message in self.failed:
                report.append(f"  ❌ {strategy}: {message}")
            report.append("")
            
        if self.errors:
            report.append("ERRORS:")
            for strategy, error in self.errors:
                report.append(f"  ⚠️ {strategy}: {error}")
            report.append("")
            
        if self.warnings:
            report.append("WARNINGS:")
            for strategy, message in self.warnings:
                report.append(f"  ⚠️ {strategy}: {message}")
            report.append("")
            
        report.append("="*70)
        if self.all_passed():
            report.append("✅ ALL STRATEGIES PASSED VERIFICATION")
        else:
            report.append("❌ SOME STRATEGIES FAILED VERIFICATION")
        report.append("="*70)
        
        return "\n".join(report)


def find_strategy_module(strategy_class_name):
    """Find the module that contains the strategy class"""
    potential_module_names = [
        f"trading_bot.strategies.{strategy_class_name.lower()}", 
        f"trading_bot.strategies.{strategy_class_name.replace('Strategy', '').lower()}_strategy",
        f"trading_bot.strategies.options.{strategy_class_name.lower()}"
    ]
    
    for module_name in potential_module_names:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, strategy_class_name):
                return module
        except (ImportError, ModuleNotFoundError):
            continue
            
    return None


def verify_strategy_adapter(strategy_class_name, result):
    """Verify that a strategy is properly adapted"""
    try:
        # Find the module containing the strategy
        module = find_strategy_module(strategy_class_name)
        if module is None:
            result.add_error(strategy_class_name, f"Could not find module for {strategy_class_name}")
            return
            
        # Get the strategy class
        strategy_class = getattr(module, strategy_class_name)
        
        # Get the adapter class
        adapter_module = importlib.import_module("trading_bot.strategies.strategy_adapter")
        adapter_class = getattr(adapter_module, "StrategyAdapter")
        
        # Create an instance
        try:
            strategy_instance = strategy_class()
            result.add_pass(strategy_class_name, "Successfully instantiated")
        except Exception as e:
            result.add_error(strategy_class_name, f"Failed to instantiate: {str(e)}")
            return
            
        # Check if it's already adapted or needs adaptation
        is_adapter = isinstance(strategy_instance, adapter_class)
        
        # Check for required methods
        missing_methods = []
        for method_name in REQUIRED_METHODS:
            if not hasattr(strategy_instance, method_name):
                missing_methods.append(method_name)
                
        if missing_methods:
            if is_adapter:
                result.add_fail(strategy_class_name, 
                    f"Is an adapter but missing methods: {', '.join(missing_methods)}")
            else:
                result.add_fail(strategy_class_name, 
                    f"Not adapted and missing methods: {', '.join(missing_methods)}")
        else:
            if is_adapter:
                result.add_pass(strategy_class_name, "Already adapted with all required methods")
            else:
                # Check if methods are callable
                for method_name in REQUIRED_METHODS:
                    method = getattr(strategy_instance, method_name)
                    if not callable(method):
                        result.add_fail(strategy_class_name, 
                            f"Method {method_name} exists but is not callable")
                        return
                
                result.add_pass(strategy_class_name, "Implements required interface natively")
                
        # Verify adapter factory works if not already an adapter
        if not is_adapter:
            try:
                create_adapter = getattr(adapter_module, "create_strategy_adapter")
                adapted_strategy = create_adapter(strategy_instance)
                
                # Verify adapted strategy has all methods
                for method_name in REQUIRED_METHODS:
                    if not hasattr(adapted_strategy, method_name):
                        result.add_fail(strategy_class_name, 
                            f"Adapted strategy missing method: {method_name}")
                        return
                
                result.add_pass(strategy_class_name, "Successfully adapted using factory function")
            except Exception as e:
                result.add_error(strategy_class_name, f"Failed to adapt: {str(e)}")
                
    except Exception as e:
        result.add_error(strategy_class_name, f"Unexpected error: {str(e)}")


def verify_component_registry():
    """Verify the component registry handles adaptation correctly"""
    result = VerificationResult()
    
    try:
        # Import the component registry
        registry_module = importlib.import_module("trading_bot.strategies.components.component_registry")
        ComponentRegistry = getattr(registry_module, "ComponentRegistry")
        
        # Create an instance
        registry = ComponentRegistry()
        
        # Check that get_strategy_instance wraps strategies
        result.add_pass("ComponentRegistry", "Successfully instantiated")
        
        # If we can access the registry methods, check for adaptation logic
        if hasattr(registry, "get_strategy_instance"):
            get_strategy_method = getattr(registry, "get_strategy_instance")
            source = inspect.getsource(get_strategy_method)
            
            # Check for adapter usage in the method
            if "create_strategy_adapter" in source or "StrategyAdapter" in source:
                result.add_pass("ComponentRegistry", "Uses strategy adapter in get_strategy_instance")
            else:
                result.add_warning("ComponentRegistry", 
                    "get_strategy_instance does not appear to use the adapter")
        else:
            result.add_warning("ComponentRegistry", "Missing get_strategy_instance method")
            
    except Exception as e:
        result.add_error("ComponentRegistry", f"Verification failed: {str(e)}")
        
    return result


def main():
    print("Verifying strategy adapter integration...")
    
    # Verify individual strategies
    result = VerificationResult()
    for strategy_class_name in STRATEGIES_TO_TEST:
        verify_strategy_adapter(strategy_class_name, result)
        
    # Verify component registry
    registry_result = verify_component_registry()
    
    # Print results
    print(result.generate_report())
    print("\nComponent Registry Verification:")
    for status, items in [
        ("PASSED", registry_result.passed),
        ("FAILED", registry_result.failed),
        ("ERRORS", registry_result.errors),
        ("WARNINGS", registry_result.warnings)
    ]:
        if items:
            print(f"{status}:")
            for item, message in items:
                print(f"  - {item}: {message}")
    
    # Return success or failure
    return 0 if result.all_passed() and registry_result.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
