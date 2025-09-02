#!/bin/bash
# Setup Paper Trading Environment Variables
# This script sets up all required environment variables for paper trading

# Text formatting
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

echo -e "${BOLD}Setting up Paper Trading Environment Variables${RESET}\n"

# Ask if this is for development or production
echo -e "${YELLOW}Is this for development or production?${RESET}"
select ENV_TYPE in "Development" "Production"; do
    case $ENV_TYPE in
        Development) 
            ENV_SUFFIX="_DEV"
            echo -e "${BLUE}Setting up for DEVELOPMENT${RESET}"
            break
            ;;
        Production) 
            ENV_SUFFIX=""
            echo -e "${BLUE}Setting up for PRODUCTION${RESET}"
            break
            ;;
        *) echo "Invalid option $REPLY";;
    esac
done

# Function to prompt for a value with a default
get_value() {
    local prompt=$1
    local default=$2
    local var_name=$3
    local is_secret=$4
    
    if [ -n "$default" ]; then
        prompt="$prompt [$default]"
    fi
    
    if [ "$is_secret" = true ]; then
        prompt="$prompt (hidden)"
        read -sp "$prompt: " value
        echo ""  # Add newline after hidden input
    else
        read -p "$prompt: " value
    fi
    
    # Use default if no value provided
    if [ -z "$value" ]; then
        value=$default
    fi
    
    # Set the variable in environment
    export $var_name=$value
    
    # Add to .env file
    if [ "$is_secret" = true ]; then
        echo "$var_name=$value" >> .env.paper
        echo "${GREEN}✓${RESET} $var_name: [HIDDEN]"
    else
        echo "$var_name=$value" >> .env.paper
        echo "${GREEN}✓${RESET} $var_name: $value"
    fi
}

# Start with a clean .env file
echo "# Paper Trading Environment Variables" > .env.paper
echo "# Created on $(date)" >> .env.paper
echo "" >> .env.paper

# Required Variables
echo -e "\n${BOLD}Setting Required Variables${RESET}"

# Trading Mode
echo "TRADING_MODE=paper" >> .env.paper
echo "${GREEN}✓${RESET} TRADING_MODE: paper"

# Alpaca API Credentials
echo -e "\n${BOLD}Alpaca API Credentials${RESET}"
get_value "Enter Alpaca API Key" "" "TRADING_ALPACA_API_KEY${ENV_SUFFIX}" true
get_value "Enter Alpaca API Secret" "" "TRADING_ALPACA_API_SECRET${ENV_SUFFIX}" true
get_value "Enter Alpaca Base URL" "https://paper-api.alpaca.markets" "TRADING_ALPACA_BASE_URL${ENV_SUFFIX}" false

# Tradier API Credentials (if used)
echo -e "\n${BOLD}Tradier API Credentials${RESET}"
get_value "Enter Tradier API Key" "" "TRADING_TRADIER_API_KEY${ENV_SUFFIX}" true
get_value "Enter Tradier API URL" "https://sandbox.tradier.com/v1" "TRADING_TRADIER_API_URL${ENV_SUFFIX}" false

# Logging Configuration
echo -e "\n${BOLD}Logging Configuration${RESET}"
get_value "Enter Log Level" "INFO" "TRADING_LOG_LEVEL" false
get_value "Enter Log Directory" "./logs" "TRADING_LOG_DIR" false

# Trading Parameters
echo -e "\n${BOLD}Trading Parameters${RESET}"
get_value "Enter Max Position Size (%)" "10" "TRADING_MAX_POSITION_PCT" false
get_value "Enter Max Daily Loss (%)" "5" "TRADING_MAX_DAILY_LOSS_PCT" false
get_value "Enter Max Drawdown (%)" "10" "TRADING_MAX_DRAWDOWN_PCT" false

# Market Data Configuration
echo -e "\n${BOLD}Market Data Configuration${RESET}"
get_value "Market Data Source" "alpaca" "TRADING_MARKET_DATA_SOURCE" false
get_value "Default Symbols" "SPY,AAPL,MSFT,GOOGL,AMZN" "TRADING_DEFAULT_SYMBOLS" false

# Email Notifications (Optional)
echo -e "\n${BOLD}Email Notifications (Optional)${RESET}"
get_value "Enable Email Notifications? (yes/no)" "no" "TRADING_EMAIL_ENABLED" false

if [ "$TRADING_EMAIL_ENABLED" = "yes" ]; then
    get_value "SMTP Server" "smtp.gmail.com" "TRADING_EMAIL_SERVER" false
    get_value "SMTP Port" "587" "TRADING_EMAIL_PORT" false
    get_value "Email Username" "" "TRADING_EMAIL_USERNAME" false
    get_value "Email Password" "" "TRADING_EMAIL_PASSWORD" true
    get_value "Email Recipients (comma-separated)" "" "TRADING_EMAIL_RECIPIENTS" false
fi

# Dashboard Configuration
echo -e "\n${BOLD}Dashboard Configuration${RESET}"
get_value "Dashboard Port" "8501" "TRADING_DASHBOARD_PORT" false
get_value "Dashboard Title" "BensBot Trading Dashboard" "TRADING_DASHBOARD_TITLE" false

# Add extra variables based on environment type
if [ "$ENV_TYPE" = "Development" ]; then
    echo -e "\n${BOLD}Development-specific Variables${RESET}"
    echo "TRADING_DEV_MODE=true" >> .env.paper
    echo "${GREEN}✓${RESET} TRADING_DEV_MODE: true"
    
    get_value "Enable Mock Trading? (yes/no)" "yes" "TRADING_MOCK_TRADING" false
    
    echo "TRADING_MOCK_DATA_DELAY=0" >> .env.paper
    echo "${GREEN}✓${RESET} TRADING_MOCK_DATA_DELAY: 0"
fi

# Success message
echo -e "\n${GREEN}${BOLD}✅ Environment variables have been set up successfully!${RESET}"
echo -e "Variables are saved in ${BOLD}.env.paper${RESET}"

# Instructions for loading the variables
echo -e "\n${BOLD}To load these variables into your current session, run:${RESET}"
echo -e "${BLUE}source .env.paper${RESET}"

# Instructions for validation
echo -e "\n${BOLD}To validate your setup, run the preflight checks:${RESET}"
echo -e "${BLUE}./scripts/run_preflight_checks.sh${RESET}"
