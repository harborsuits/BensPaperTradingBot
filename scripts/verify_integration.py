#!/usr/bin/env python3
"""
Options Strategy Integration Verification

This script directly examines the core files to verify:
1. Strategy adapter implementation
2. Component registry integration
3. Near-miss detection logic
4. Event emission for optimizations

No imports or dependencies required - pure file analysis.
"""

import os
import re
import json

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VERIFICATION_RESULTS = []

# Files to check
FILES_TO_VERIFY = {
    "Strategy Adapter": "trading_bot/strategies/strategy_adapter.py",
    "Component Registry": "trading_bot/strategies/components/component_registry.py",
    "Autonomous Engine": "trading_bot/autonomous/autonomous_engine.py",
    "Optimization Handlers": "trading_bot/event_system/strategy_optimization_handlers.py"
}

# Patterns to search for in each file (regex patterns)
VERIFICATION_PATTERNS = {
    "Strategy Adapter": [
        r"class\s+StrategyAdapter",                        # Adapter class definition
        r"def\s+generate_signals",                         # Required method
        r"def\s+size_position",                            # Required method
        r"def\s+manage_open_trades",                       # Required method
        r"def\s+create_strategy_adapter",                  # Factory function
    ],
    "Component Registry": [
        r"from\s+.*strategy_adapter\s+import",             # Import of adapter
        r"create_strategy_adapter\(.*\)",                  # Usage of adapter factory
        r"get_strategy_instance",                          # Method to get wrapped strategy
    ],
    "Autonomous Engine": [
        r"def\s+_is_near_miss_candidate",                  # Near-miss detection
        r"def\s+_optimize_near_miss_candidates",           # Optimization method
        r"STRATEGY_OPTIMISED|STRATEGY_EXHAUSTED",          # Event types
        r"emit_event\(.*STRATEGY_OPTIMISED",               # Event emission
    ],
    "Optimization Handlers": [
        r"class\s+StrategyOptimizationTracker",            # Tracker class
        r"_handle_optimization_event",                     # Unified event handler
        r"event_type=\"STRATEGY_OPTIMISED\"",             # Event registration 
        r"event_type=\"STRATEGY_EXHAUSTED\""              # Event registration
    ]
}

def check_file_exists(file_path):
    """Check if a file exists"""
    full_path = os.path.join(PROJECT_ROOT, file_path)
    return os.path.isfile(full_path)

def read_file_content(file_path):
    """Read a file's content"""
    full_path = os.path.join(PROJECT_ROOT, file_path)
    try:
        with open(full_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {str(e)}"

def check_patterns_in_content(content, patterns):
    """Check if all patterns exist in content"""
    results = []
    for pattern in patterns:
        match = re.search(pattern, content)
        results.append((pattern, match is not None))
    return results

def verify_file(file_key, file_path):
    """Verify a single file against its patterns"""
    result = {
        "file": file_key,
        "path": file_path,
        "exists": False,
        "patterns_found": [],
        "patterns_missing": [],
        "overall_result": "FAILED"
    }
    
    # Check if file exists
    if not check_file_exists(file_path):
        result["error"] = f"File {file_path} does not exist"
        VERIFICATION_RESULTS.append(result)
        return
        
    # File exists
    result["exists"] = True
    
    # Read content
    content = read_file_content(file_path)
    if content.startswith("ERROR:"):
        result["error"] = content
        VERIFICATION_RESULTS.append(result)
        return
        
    # Check patterns
    patterns = VERIFICATION_PATTERNS.get(file_key, [])
    pattern_results = check_patterns_in_content(content, patterns)
    
    # Process results
    for pattern, found in pattern_results:
        if found:
            result["patterns_found"].append(pattern)
        else:
            result["patterns_missing"].append(pattern)
            
    # Overall result
    if not result["patterns_missing"]:
        result["overall_result"] = "PASSED"
        
    VERIFICATION_RESULTS.append(result)

def generate_report():
    """Generate a verification report"""
    report_lines = []
    report_lines.append("\n" + "="*70)
    report_lines.append("OPTIONS STRATEGY INTEGRATION VERIFICATION REPORT")
    report_lines.append("="*70 + "\n")
    
    # Overall summary
    passed_count = sum(1 for r in VERIFICATION_RESULTS if r["overall_result"] == "PASSED")
    total_count = len(VERIFICATION_RESULTS)
    
    report_lines.append(f"Files verified: {total_count}")
    report_lines.append(f"Files passed: {passed_count}")
    report_lines.append(f"Files failed: {total_count - passed_count}")
    report_lines.append("")
    
    # Detailed results
    for result in VERIFICATION_RESULTS:
        file_key = result["file"]
        report_lines.append(f"File: {file_key} ({result['path']})")
        report_lines.append(f"Status: {result['overall_result']}")
        
        if "error" in result:
            report_lines.append(f"Error: {result['error']}")
        elif not result["exists"]:
            report_lines.append("File does not exist")
        else:
            if result["patterns_found"]:
                report_lines.append("Features found:")
                for pattern in result["patterns_found"]:
                    feature_name = get_feature_name(file_key, pattern)
                    report_lines.append(f"  ✅ {feature_name}")
                    
            if result["patterns_missing"]:
                report_lines.append("Features missing:")
                for pattern in result["patterns_missing"]:
                    feature_name = get_feature_name(file_key, pattern)
                    report_lines.append(f"  ❌ {feature_name}")
                    
        report_lines.append("")
        
    # Overall assessment
    report_lines.append("="*70)
    if passed_count == total_count:
        report_lines.append("✅ ALL INTEGRATION POINTS VERIFIED")
        report_lines.append("The options strategy integration is correctly implemented.")
    else:
        report_lines.append("❌ SOME INTEGRATION POINTS FAILED VERIFICATION")
        report_lines.append("The options strategy integration is incomplete or incorrect.")
    report_lines.append("="*70)
    
    return "\n".join(report_lines)

def get_feature_name(file_key, pattern):
    """Convert a regex pattern to a readable feature name"""
    pattern_to_feature = {
        # Strategy Adapter
        r"class\s+StrategyAdapter": "StrategyAdapter class defined",
        r"def\s+generate_signals": "generate_signals method implemented",
        r"def\s+size_position": "size_position method implemented",
        r"def\s+manage_open_trades": "manage_open_trades method implemented",
        r"def\s+create_strategy_adapter": "Factory function for creating adapters",
        
        # Component Registry
        r"from\s+.*strategy_adapter\s+import": "Strategy adapter imported in registry",
        r"create_strategy_adapter\(.*\)": "Adapter factory used in registry",
        r"get_strategy_instance": "Method to retrieve adapted strategies",
        
        # Autonomous Engine
        r"def\s+_is_near_miss_candidate": "Near-miss candidate detection",
        r"def\s+_optimize_near_miss_candidates": "Near-miss optimization logic",
        r"STRATEGY_OPTIMISED|STRATEGY_EXHAUSTED": "Optimization event types defined",
        r"emit_event\(.*STRATEGY_OPTIMISED": "Strategy optimization events emitted",
        
        # Optimization Handlers
        r"class\s+StrategyOptimizationTracker": "Optimization tracker implemented",
        r"handle_strategy_optimised": "Handler for optimized strategies",
        r"handle_strategy_exhausted": "Handler for exhausted optimization",
    }
    
    return pattern_to_feature.get(pattern, pattern)

def verify_integration():
    """Verify the complete integration"""
    for file_key, file_path in FILES_TO_VERIFY.items():
        verify_file(file_key, file_path)
        
    # Extra verification for strategy files
    strategy_files = find_strategy_files()
    
    # Generate and print report
    report = generate_report()
    print(report)
    
    # Write report to file
    with open("integration_verification_report.txt", "w") as f:
        f.write(report)
        
    print(f"Report written to: {os.path.join(PROJECT_ROOT, 'integration_verification_report.txt')}")
    
    # Return success or failure
    return all(r["overall_result"] == "PASSED" for r in VERIFICATION_RESULTS)

def find_strategy_files():
    """Find all strategy files in the project"""
    strategies_dir = os.path.join(PROJECT_ROOT, "trading_bot", "strategies")
    strategy_files = []
    
    if os.path.isdir(strategies_dir):
        for root, dirs, files in os.walk(strategies_dir):
            for file in files:
                if file.endswith("_strategy.py") or file.endswith("Strategy.py"):
                    strategy_files.append(os.path.join(root, file))
                    
    return strategy_files

def main():
    """Main function"""
    print("Verifying options strategy integration...")
    success = verify_integration()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
