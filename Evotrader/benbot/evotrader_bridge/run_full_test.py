#!/usr/bin/env python3
"""
Full testing script for EvoTrader Bridge system.

This script orchestrates the complete testing workflow:
1. Runs evolutionary test with the desired strategies
2. Analyzes the results to identify improvements and parameter optimizations
3. Produces a comprehensive test report
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import time
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_test_id() -> str:
    """Create a unique test ID based on timestamp."""
    timestamp = int(time.time())
    return f"test_{timestamp}"


def run_evolution_test(args: argparse.Namespace) -> str:
    """
    Run the evolution test script.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to test results directory
    """
    print("\n" + "="*80)
    print("RUNNING EVOLUTIONARY TEST")
    print("="*80)
    
    # Create test ID if not provided
    test_id = args.test_id if args.test_id else create_test_id()
    print(f"Test ID: {test_id}")
    
    # Create test results directory
    test_dir = f"test_results/{test_id}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python3", 
        os.path.join(SCRIPT_DIR, "run_test_evolution.py"),
        "--strategies", str(args.strategies),
        "--generations", str(args.generations),
        "--test-id", test_id
    ]
    
    # Run the evolution process
    print(f"Running evolution with {args.strategies} strategies for {args.generations} generations...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save command output
    with open(f"{test_dir}/evolution_log.txt", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n")
            f.write(result.stderr)
    
    # Check if successful
    if result.returncode != 0:
        print("Evolution process failed!")
        print(result.stderr)
        exit(1)
    
    print(f"Evolution process completed successfully!")
    print(f"Results saved to: {test_dir}")
    
    return test_dir


def run_analysis(test_dir: str) -> None:
    """
    Run the analysis script on test results.
    
    Args:
        test_dir: Directory containing test results
    """
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    # Build command
    cmd = [
        "python3",
        os.path.join(SCRIPT_DIR, "analyze_evolution.py"),
        "--test-dir", test_dir
    ]
    
    # Run the analysis process
    print(f"Analyzing results in {test_dir}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save command output
    with open(f"{test_dir}/analysis_log.txt", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n")
            f.write(result.stderr)
    
    # Check if successful
    if result.returncode != 0:
        print("Analysis process failed!")
        print(result.stderr)
        exit(1)
    
    print(f"Analysis completed successfully!")
    print(f"Analysis results saved to: {test_dir}/analysis")


def generate_summary_report(test_dir: str, args: argparse.Namespace) -> None:
    """
    Generate a summary report of the test.
    
    Args:
        test_dir: Directory containing test results
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    # Extract test ID from directory
    test_id = os.path.basename(test_dir)
    
    # Create report data
    report = {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "strategies": args.strategies,
            "generations": args.generations,
            "focus_strategies": args.focus or "All"
        },
        "summary": {},
        "performance": {},
        "recommendations": {}
    }
    
    # Try to load performance data
    try:
        # Look for performance analysis data
        analysis_dir = os.path.join(test_dir, "analysis")
        if os.path.exists(analysis_dir):
            # Load parameter recommendations
            param_file = os.path.join(analysis_dir, "parameters", "optimal_parameters.json")
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    param_data = json.load(f)
                    report["recommendations"] = param_data.get("parameter_recommendations", {})
            
            # Look for fitness progression data
            charts_dir = os.path.join(analysis_dir, "charts")
            if os.path.exists(charts_dir):
                # Note chart paths
                report["charts"] = {
                    "fitness_progression": "analysis/charts/fitness_progression.png",
                    "strategy_distribution": "analysis/charts/strategy_distribution.png",
                    "performance_metrics": "analysis/charts/performance_metrics.png"
                }
    except Exception as e:
        print(f"Error loading analysis data: {str(e)}")
    
    # Try to extract overall performance stats
    try:
        # Look for final generation stats
        evolution_dir = os.path.join(test_dir, "evolution", "generation_stats")
        if os.path.exists(evolution_dir):
            gen_files = [f for f in os.listdir(evolution_dir) if f.startswith("generation_") and f.endswith(".json")]
            if gen_files:
                gen_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                final_gen_file = os.path.join(evolution_dir, gen_files[-1])
                
                with open(final_gen_file, 'r') as f:
                    final_stats = json.load(f)
                    
                    # Extract summary metrics
                    report["summary"] = {
                        "total_generations": len(gen_files),
                        "final_avg_fitness": final_stats.get("avg_fitness", 0),
                        "final_max_fitness": final_stats.get("max_fitness", 0),
                        "strategy_distribution": final_stats.get("strategy_distribution", {})
                    }
    except Exception as e:
        print(f"Error loading generation data: {str(e)}")
    
    # Save report as JSON
    report_file = os.path.join(test_dir, "test_summary.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable report
    report_md = os.path.join(test_dir, "TEST_REPORT.md")
    with open(report_md, 'w') as f:
        f.write(f"# EvoTrader Bridge Test Report\n\n")
        f.write(f"**Test ID:** {test_id}  \n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        
        f.write("## Test Configuration\n\n")
        f.write(f"- **Strategy Population:** {args.strategies}\n")
        f.write(f"- **Generations:** {args.generations}\n")
        if args.focus:
            f.write(f"- **Focus Strategies:** {args.focus}\n")
        f.write("\n")
        
        # Add summary section if available
        if report["summary"]:
            f.write("## Summary Results\n\n")
            f.write(f"- **Total Generations:** {report['summary'].get('total_generations', 0)}\n")
            f.write(f"- **Final Average Fitness:** {report['summary'].get('final_avg_fitness', 0):.4f}\n")
            f.write(f"- **Final Maximum Fitness:** {report['summary'].get('final_max_fitness', 0):.4f}\n\n")
            
            # Add strategy distribution if available
            strategy_dist = report["summary"].get("strategy_distribution")
            if strategy_dist:
                f.write("### Strategy Distribution in Final Generation\n\n")
                f.write("| Strategy Type | Count |\n")
                f.write("|--------------|-------|\n")
                for strategy_type, count in strategy_dist.items():
                    f.write(f"| {strategy_type} | {count} |\n")
                f.write("\n")
        
        # Add charts section if available
        if "charts" in report:
            f.write("## Performance Charts\n\n")
            
            # Fitness progression
            f.write("### Fitness Progression\n\n")
            f.write(f"![Fitness Progression]({report['charts'].get('fitness_progression')})\n\n")
            
            # Strategy distribution
            f.write("### Strategy Distribution\n\n")
            f.write(f"![Strategy Distribution]({report['charts'].get('strategy_distribution')})\n\n")
            
            # Performance metrics
            f.write("### Performance Metrics\n\n")
            f.write(f"![Performance Metrics]({report['charts'].get('performance_metrics')})\n\n")
        
        # Add recommendations section if available
        if report["recommendations"]:
            f.write("## Strategy Parameter Recommendations\n\n")
            
            for strategy_type, data in report["recommendations"].items():
                f.write(f"### {strategy_type}\n\n")
                
                if "parameters" in data:
                    f.write("| Parameter | Optimal Value | Recommended Range |\n")
                    f.write("|-----------|--------------|------------------|\n")
                    
                    for param_name, param_data in data["parameters"].items():
                        optimal = param_data.get("optimal_value", 0)
                        range_min = param_data.get("recommended_range", [0, 0])[0]
                        range_max = param_data.get("recommended_range", [0, 0])[1]
                        
                        f.write(f"| {param_name} | {optimal:.2f} | {range_min:.2f} - {range_max:.2f} |\n")
                    
                    f.write("\n")
        
        # Add next steps
        f.write("## Next Steps\n\n")
        f.write("1. **Review the evolved strategies** - Check the best-performing strategies and analyze how they differ from the original implementations.\n")
        f.write("2. **Apply parameter recommendations** - Update your strategies with the recommended parameter values for improved performance.\n")
        f.write("3. **Run A/B tests with your best strategies** - Verify the improvements in a controlled environment before deployment.\n")
        f.write("4. **Promote successful strategies** - Move proven strategies to the production system.\n")
        f.write("5. **Continue evolution** - Set up ongoing evolution to continue improving strategies over time.\n\n")
        
        # Add results location
        f.write("## Results Location\n\n")
        f.write(f"Full test results can be found in: `{test_dir}`\n\n")
        f.write("Specific outputs:\n\n")
        f.write(f"- Evolution logs: `{test_dir}/evolution_log.txt`\n")
        f.write(f"- Analysis logs: `{test_dir}/analysis_log.txt`\n")
        f.write(f"- Performance charts: `{test_dir}/analysis/charts/`\n")
        f.write(f"- Parameter analysis: `{test_dir}/analysis/parameters/`\n")
    
    print(f"Summary report generated: {report_md}")
    print(f"JSON report: {report_file}")


def main():
    """Run the full test process."""
    parser = argparse.ArgumentParser(description='Run full EvoTrader Bridge test')
    
    # Test configuration
    parser.add_argument('--strategies', type=int, default=15, 
                        help='Number of strategies to create (default: 15)')
    parser.add_argument('--generations', type=int, default=5, 
                        help='Number of generations to evolve (default: 5)')
    parser.add_argument('--test-id', type=str, help='Identifier for this test run')
    parser.add_argument('--focus', type=str, help='Focus on specific strategy types, comma-separated')
    
    args = parser.parse_args()
    
    # Print test configuration
    print("\nEVOTRADER BRIDGE FULL TEST")
    print("------------------------")
    print(f"Strategy Population: {args.strategies}")
    print(f"Generations: {args.generations}")
    if args.focus:
        print(f"Focus Strategies: {args.focus}")
    print("------------------------\n")
    
    # Run evolution test
    test_dir = run_evolution_test(args)
    
    # Run analysis
    run_analysis(test_dir)
    
    # Generate summary report
    generate_summary_report(test_dir, args)
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results available in: {test_dir}")
    print(f"Open {test_dir}/TEST_REPORT.md for a complete summary")


if __name__ == "__main__":
    main()
