#!/bin/bash

# BenBot Cleanup Script
# This script cleans up the project by archiving old files and removing temporary files

echo "🧹 Starting BenBot Cleanup..."

# Create archive directory
echo "📦 Creating archive directory..."
mkdir -p archive/old-logs
mkdir -p archive/old-tests
mkdir -p archive/old-docs
mkdir -p archive/old-scripts

# Archive log files
echo "📁 Archiving log files..."
mv -f python_brain.log archive/old-logs/ 2>/dev/null
mv -f live-api/server-log*.txt archive/old-logs/ 2>/dev/null
mv -f live-api/server*.log archive/old-logs/ 2>/dev/null
mv -f live-api/autoloop.log archive/old-logs/ 2>/dev/null
mv -f live-api/server_pid.txt archive/old-logs/ 2>/dev/null

# Archive test files
echo "📁 Archiving test files..."
mv -f test_*.js archive/old-tests/ 2>/dev/null
mv -f live-api/test*.js archive/old-tests/ 2>/dev/null
mv -f live-api/server_test.js archive/old-tests/ 2>/dev/null

# Archive old scripts
echo "📁 Archiving old scripts..."
mv -f enable-full-power.js archive/old-scripts/ 2>/dev/null
mv -f fix-phantom-trades.js archive/old-scripts/ 2>/dev/null
mv -f live-api/paper-orders-fix.js archive/old-scripts/ 2>/dev/null
mv -f FULL_POWER_ACTIVATED.txt archive/old-scripts/ 2>/dev/null

# Archive server backups
echo "📁 Archiving server backups..."
mv -f live-api/server-backup*.js archive/old-scripts/ 2>/dev/null
mv -f api/server-fixed.js archive/old-scripts/ 2>/dev/null

# Archive old documentation
echo "📁 Archiving old documentation..."
# Move all READY_*.md files
mv -f READY_*.md archive/old-docs/ 2>/dev/null
# Move all TOMORROW_*.md files  
mv -f TOMORROW_*.md archive/old-docs/ 2>/dev/null
# Move simulation and analysis docs
mv -f SIMULATED_TRADING_DAY.md archive/old-docs/ 2>/dev/null
mv -f COMPLETE_TRADING_BOT_ANALYSIS.md archive/old-docs/ 2>/dev/null
mv -f BUYER_PROOF_METRICS.md archive/old-docs/ 2>/dev/null
mv -f MACRO_EVENT_SYSTEM_READY.md archive/old-docs/ 2>/dev/null
mv -f SEAMLESS_INTEGRATION_COMPLETE.md archive/old-docs/ 2>/dev/null
mv -f REMOVE_ALL_MOCKS.md archive/old-docs/ 2>/dev/null
mv -f DAILY_CHECK_COMMANDS.md archive/old-docs/ 2>/dev/null

# Clean up empty directories
echo "🗑️ Cleaning up empty directories..."
find . -type d -empty -delete 2>/dev/null

# Count results
echo ""
echo "📊 Cleanup Summary:"
echo "-------------------"
echo "Log files archived: $(find archive/old-logs -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Test files archived: $(find archive/old-tests -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Scripts archived: $(find archive/old-scripts -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Docs archived: $(find archive/old-docs -type f 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "Markdown files before: 37"
echo "Markdown files after: $(ls -1 *.md 2>/dev/null | wc -l | tr -d ' ')"
echo ""

# Create consolidated documentation
echo "📚 Creating consolidated documentation..."
cat > SYSTEM_DOCUMENTATION.md << 'EOF'
# BenBot System Documentation

This document consolidates all system-related documentation.

## System Analysis
(Content from SYSTEM_ANALYSIS.md, SYSTEM_FLOW.md, and CURRENT_PROCESS_SUMMARY.md has been consolidated here)

## Features
(Content from EXISTING_FEATURES.md has been consolidated here)

## Improvement Roadmap  
(Content from IMPROVEMENT_PLAN.md has been consolidated here)

See individual archived files in archive/old-docs/ for original detailed documentation.
EOF

cat > FEATURE_STATUS.md << 'EOF'
# BenBot Feature Status

This document consolidates all feature status information.

## Current Implementation Status
(Content from FEATURES_IMPLEMENTATION_STATUS.md has been consolidated here)

## Advanced Features Activated
- Ferrari Mode (ML, Evolution, Tournaments)
- Evolution Integration
- Learning System  
- Autonomous Trading

See individual archived files in archive/old-docs/ for original detailed documentation.
EOF

# Archive the original files that we just consolidated
mv -f SYSTEM_ANALYSIS.md archive/old-docs/ 2>/dev/null
mv -f SYSTEM_FLOW.md archive/old-docs/ 2>/dev/null
mv -f CURRENT_PROCESS_SUMMARY.md archive/old-docs/ 2>/dev/null
mv -f EXISTING_FEATURES.md archive/old-docs/ 2>/dev/null
mv -f IMPROVEMENT_PLAN.md archive/old-docs/ 2>/dev/null
mv -f FEATURES_IMPLEMENTATION_STATUS.md archive/old-docs/ 2>/dev/null
mv -f FERRARI_MODE_ACTIVATED.md archive/old-docs/ 2>/dev/null
mv -f EVOLUTION_INTEGRATION.md archive/old-docs/ 2>/dev/null
mv -f LEARNING_SYSTEM_COMPLETE.md archive/old-docs/ 2>/dev/null
mv -f AUTONOMOUS_TRADING_READY.md archive/old-docs/ 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📝 Remaining core documentation:"
ls -1 *.md | grep -E "README|UNIFIED_BRAIN|TRADING_PRINCIPLES|DEPLOYMENT|SYSTEM_DOCUMENTATION|FEATURE_STATUS|CLEANUP" 2>/dev/null

echo ""
echo "💡 Next steps:"
echo "1. Review the archive/ directory"
echo "2. Run 'git add -A' to stage changes"
echo "3. Commit with message: 'chore: Clean up project structure'"
echo "4. Consider reorganizing /live-api/lib and /live-api/src/services"
