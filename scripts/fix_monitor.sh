#!/bin/bash
# This script fixes the syntax error in the LogMonitor component

# Look for the problematic line and fix the closing bracket
sed -i.bak '
250,285 {
  # Add closing parenthesis for filteredLogs.map
  /filteredLogs.map/ s/)))/)))/
}
' /Users/bendickinson/Desktop/Trading:BenBot/new-trading-dashboard/src/components/developer/LogMonitor.tsx

chmod +x /Users/bendickinson/Desktop/Trading:BenBot/fix_monitor.sh
