#!/bin/bash
# System Integrity Verification Script
# This script verifies that all critical fixes and patterns are intact

echo "🔍 BenBot System Integrity Check"
echo "================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
INTEGRITY_OK=true

# Function to check if a pattern exists in a file
check_pattern() {
    local file=$1
    local pattern=$2
    local description=$3
    
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo -e "${GREEN}✅ $description${NC}"
        return 0
    else
        echo -e "${RED}❌ $description - MISSING!${NC}"
        INTEGRITY_OK=false
        return 1
    fi
}

# Function to check if a file exists
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $description exists${NC}"
        return 0
    else
        echo -e "${RED}❌ $description - MISSING!${NC}"
        INTEGRITY_OK=false
        return 1
    fi
}

# Function to check file hash
check_file_hash() {
    local file=$1
    local expected_patterns=("$@")
    
    echo -e "\n${YELLOW}Checking $file integrity...${NC}"
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ File missing: $file${NC}"
        INTEGRITY_OK=false
        return 1
    fi
    
    # Check each expected pattern
    for ((i=1; i<${#expected_patterns[@]}; i++)); do
        pattern="${expected_patterns[$i]}"
        if ! grep -q "$pattern" "$file" 2>/dev/null; then
            echo -e "${RED}❌ Critical pattern missing in $file: $pattern${NC}"
            INTEGRITY_OK=false
        fi
    done
}

echo -e "\n1. Checking critical files existence..."
echo "======================================="

check_file "minimal_server.js" "Main server file"
check_file "minimal_server.js.fixed-backup" "Server backup file"
check_file "lib/autoLoop.js" "AutoLoop module"
check_file "lib/PaperBroker.js" "PaperBroker module"
check_file "lib/brainIntegrator.js" "BrainIntegrator module"
check_file "config/tradingThresholds.js" "Trading thresholds config"
check_file "services/enhancedPerformanceRecorder.js" "Enhanced recorder"
check_file "start-benbot.sh" "Startup script"
check_file "verify-dashboard-fixes.sh" "Dashboard verification script"

echo -e "\n2. Checking minimal_server.js fixes..."
echo "======================================="

check_pattern "minimal_server.js" "(allStrategies || \[\]).forEach" "Strategy API array iteration fix"
check_pattern "minimal_server.js" "strategy\.instance.*performance.*sharpe" "Strategy instance property access"
check_pattern "minimal_server.js" "quantity: pos\.qty || 0" "Position quantity transformation"
check_pattern "minimal_server.js" "cost_basis: (pos\.avg_price || 0) \* (pos\.qty || 0)" "Position cost_basis calculation"
check_pattern "minimal_server.js" "new TokenBucketLimiter(20, 50)" "Rate limiter fix for quotes"
check_pattern "minimal_server.js" "minConfidence: 0.25" "BrainIntegrator testing mode"
check_pattern "minimal_server.js" "enhancedRecorder: enhancedRecorder" "Enhanced recorder connection"

echo -e "\n3. Checking PaperBroker integration..."
echo "======================================="

check_pattern "lib/PaperBroker.js" "getQuotesCache" "Real quotes integration"
check_pattern "lib/PaperBroker.js" "fillPrice: order.filled_price" "Fill price event property"

echo -e "\n4. Checking AutoLoop parallel evaluation..."
echo "==========================================="

check_pattern "lib/autoLoop.js" "await Promise\.all" "Parallel candidate fetching"
check_pattern "lib/autoLoop.js" "BATCH_SIZE = 20" "Batch processing for parallel evaluation"
check_pattern "lib/autoLoop.js" "PARALLEL VERSION" "Parallel evaluation marker"

echo -e "\n5. Checking trading thresholds..."
echo "=================================="

check_pattern "config/tradingThresholds.js" "buyThreshold: 0.35" "Aggressive buy threshold"
check_pattern "config/tradingThresholds.js" "sellThreshold: 0.30" "Aggressive sell threshold"
check_pattern "config/tradingThresholds.js" "minConfidence: 0.25" "Low confidence threshold"

echo -e "\n6. Checking for dangerous old files..."
echo "======================================="

DANGEROUS_FILES=(
    "server-archive/server.js"
    "server-archive/server-backup-*.js"
    "backup/minimal_server.js"
)

for pattern in "${DANGEROUS_FILES[@]}"; do
    if ls $pattern 2>/dev/null | grep -q .; then
        echo -e "${YELLOW}⚠️  Old file found: $pattern (should not be used)${NC}"
    fi
done

echo -e "\n7. Checking process status..."
echo "=============================="

# Check if backend is running
if lsof -ti:4000 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend server is running on port 4000${NC}"
else
    echo -e "${YELLOW}⚠️  Backend server is not running on port 4000${NC}"
fi

# Check if frontend is running
if lsof -ti:3003 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is running on port 3003${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend is not running on port 3003${NC}"
fi

echo -e "\n8. Checking module exports..."
echo "=============================="

# Check if getQuotesCache is exported
if grep -q "module.exports.getQuotesCache" "minimal_server.js"; then
    echo -e "${GREEN}✅ getQuotesCache is exported${NC}"
else
    echo -e "${RED}❌ getQuotesCache export missing - PaperBroker won't get real prices!${NC}"
    INTEGRITY_OK=false
fi

# Final verdict
echo -e "\n======================================="
if [ "$INTEGRITY_OK" = true ]; then
    echo -e "${GREEN}✅ SYSTEM INTEGRITY CHECK PASSED${NC}"
    echo "All critical fixes and patterns are intact."
    exit 0
else
    echo -e "${RED}❌ SYSTEM INTEGRITY CHECK FAILED${NC}"
    echo "Critical fixes are missing! Run these commands to restore:"
    echo ""
    echo "cd /Users/bendickinson/Desktop/benbot/live-api"
    echo "cp minimal_server.js.fixed-backup minimal_server.js"
    echo "./start-benbot.sh"
    exit 1
fi
