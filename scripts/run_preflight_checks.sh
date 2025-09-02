#!/bin/bash
# Pre-flight Checks for Paper Trading
# This script runs a series of validation tests to ensure the trading bot
# is ready for paper trading

set -e

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

# Output directory for test results
RESULTS_DIR="./test_results"
mkdir -p "$RESULTS_DIR"

echo -e "${BOLD}Running Pre-Paper Trading Validation Tests${RESET}\n"

# Function to check if required environment variables are set
check_environment() {
    echo -e "\n${BOLD}Checking environment variables...${RESET}"
    local missing_vars=0
    
    # List of required environment variables
    required_vars=(
        "TRADING_ALPACA_API_KEY"
        "TRADING_ALPACA_API_SECRET"
        "TRADING_TRADIER_API_KEY"
        "TRADING_MODE"
        "TRADING_LOG_LEVEL"
    )
    
    # Optional but recommended variables
    optional_vars=(
        "TRADING_EMAIL_SERVER"
        "TRADING_EMAIL_PORT"
        "TRADING_EMAIL_USERNAME"
        "TRADING_EMAIL_PASSWORD"
        "TRADING_EMAIL_RECIPIENTS"
    )
    
    # Check required variables
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo -e "  ${RED}✗ $var is not set${RESET}"
            missing_vars=$((missing_vars+1))
        else
            echo -e "  ${GREEN}✓ $var is set${RESET}"
        fi
    done
    
    # Check optional variables
    for var in "${optional_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo -e "  ${YELLOW}⚠ $var is not set (optional)${RESET}"
        else
            echo -e "  ${GREEN}✓ $var is set${RESET}"
        fi
    done
    
    # Ensure trading mode is set to paper
    if [ "$TRADING_MODE" != "paper" ]; then
        echo -e "\n${YELLOW}⚠ Warning: TRADING_MODE is not set to 'paper'. Current value: $TRADING_MODE${RESET}"
    fi
    
    if [ $missing_vars -gt 0 ]; then
        echo -e "\n${RED}${BOLD}✗ $missing_vars required environment variables are missing${RESET}"
        echo -e "Please set all required environment variables before proceeding."
        return 1
    else
        echo -e "\n${GREEN}${BOLD}✓ All required environment variables are set${RESET}"
        return 0
    fi
}

# Function to run a test and report results
run_test() {
    local test_name=$1
    local test_command=$2
    local test_description=$3
    
    echo -e "\n${BOLD}Running $test_name...${RESET}"
    echo -e "  $test_description"
    
    # Create or clear log file
    local log_file="$RESULTS_DIR/${test_name}.log"
    > "$log_file"
    
    # Run the test and capture output and exit code
    eval "$test_command" > "$log_file" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "  ${GREEN}${BOLD}✓ $test_name passed${RESET}"
        return 0
    else
        echo -e "  ${RED}${BOLD}✗ $test_name failed (exit code: $exit_code)${RESET}"
        echo -e "  ${YELLOW}See log file for details: $log_file${RESET}"
        return 1
    fi
}

# Run each test
echo -e "${BOLD}Step 1: Checking environment...${RESET}"
check_environment
env_check=$?

echo -e "\n${BOLD}Step 2: Running data validation tests...${RESET}"
run_test "data_validator_test" "python -m tests.validation.data_validator_test" "Testing data quality checks"
data_test=$?

echo -e "\n${BOLD}Step 3: Running emergency controls tests...${RESET}"
run_test "kill_switch_test" "python -m tests.validation.kill_switch_test" "Testing kill switch functionality"
kill_switch_test=$?

echo -e "\n${BOLD}Step 4: Running end-to-end integration test...${RESET}"
run_test "integration_test" "python -m tests.integration.end_to_end_test" "Testing full system integration"
integration_test=$?

# Run additional checks
echo -e "\n${BOLD}Step 5: Checking for critical code issues...${RESET}"
run_test "code_checks" "python -m scripts.check_hardcoded_credentials" "Checking for hardcoded credentials"
code_check=$?

# Generate preflight report
echo -e "\n${BOLD}Generating pre-flight summary...${RESET}"
SUMMARY_FILE="$RESULTS_DIR/preflight_summary.txt"

{
    echo "BensBot Pre-Paper Trading Validation Summary"
    echo "============================================="
    echo "Date: $(date)"
    echo ""
    echo "Test Results:"
    echo "-------------"
    echo "Environment Check: $([ $env_check -eq 0 ] && echo "PASS" || echo "FAIL")"
    echo "Data Validator Test: $([ $data_test -eq 0 ] && echo "PASS" || echo "FAIL")"
    echo "Kill Switch Test: $([ $kill_switch_test -eq 0 ] && echo "PASS" || echo "FAIL")"
    echo "Integration Test: $([ $integration_test -eq 0 ] && echo "PASS" || echo "FAIL")"
    echo "Code Check: $([ $code_check -eq 0 ] && echo "PASS" || echo "FAIL")"
    echo ""
    
    # Final verdict
    total=$(( $env_check + $data_test + $kill_switch_test + $integration_test + $code_check ))
    if [ $total -eq 0 ]; then
        echo "Final Status: READY FOR PAPER TRADING"
    else
        echo "Final Status: NOT READY - $total checks failed"
        echo "Action Required: Fix the issues reported in the test logs before enabling paper trading."
    fi
} > "$SUMMARY_FILE"

echo -e "Summary written to: $SUMMARY_FILE"

# Print final status
if [ $total -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}✓ All checks passed! System is ready for paper trading.${RESET}"
    exit 0
else
    echo -e "\n${RED}${BOLD}✗ $total checks failed. System is NOT ready for paper trading.${RESET}"
    echo -e "${YELLOW}Please fix the issues reported in the test logs before enabling paper trading.${RESET}"
    exit 1
fi
