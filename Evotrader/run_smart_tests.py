#!/usr/bin/env python3
"""
Run Smart Features Tests

Simple wrapper script to run the test_smart_features.py with default parameters.
This creates all necessary reports in the reports directory.
"""

import os
import sys
import subprocess
import datetime

# Make sure reports directory exists
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# Create timestamp subdirectory for this run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(REPORTS_DIR, f'smart_test_{timestamp}')
os.makedirs(output_dir)

print(f"Running smart feature tests with output to: {output_dir}")

# Run the tests
cmd = [
    'python3',
    'test_smart_features.py',
    '--output-dir', output_dir,
    '--test', 'all',
    '--pairs', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'
]

try:
    result = subprocess.run(cmd, check=True)
    print(f"Tests completed successfully. Results in: {output_dir}")
    print(f"Key files to review:")
    print(f" - {os.path.join(output_dir, 'smart_test_results.json')}")
    print(f" - {os.path.join(output_dir, 'comparative_analysis_summary.json')}")
    print(f" - Various visualizations in {output_dir}/*.png")
except subprocess.CalledProcessError as e:
    print(f"Error running tests: {e}")
    sys.exit(1)
