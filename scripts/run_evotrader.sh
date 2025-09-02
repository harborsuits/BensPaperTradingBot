#!/bin/bash
# EvoTrader Integration Launcher for BensBot
# This script provides easy access to EvoTrader functionality

# Set up colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Set paths
PROJECT_ROOT="$(pwd)"
PYTHON_PATH="$PROJECT_ROOT"

# Check if EvoTrader repository exists
if [ ! -d "$PROJECT_ROOT/Evotrader" ]; then
    echo -e "${RED}EvoTrader repository not found!${NC}"
    echo -e "${YELLOW}Would you like to clone it now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
        echo -e "${BLUE}Cloning EvoTrader repository...${NC}"
        git clone https://github.com/TheClitCommander/Evotrader.git
        echo -e "${GREEN}EvoTrader repository cloned successfully!${NC}"
    else
        echo -e "${RED}EvoTrader repository is required for this integration. Exiting.${NC}"
        exit 1
    fi
fi

# Display welcome message
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}BensBot - EvoTrader Integration Launcher${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Check command line arguments
if [ $# -eq 0 ]; then
    # No arguments, show menu
    echo -e "${BLUE}Select an option:${NC}"
    echo "1. Run EvoTrader Dashboard"
    echo "2. Run Strategy Evolution (Forex)"
    echo "3. Run Strategy Evolution (Crypto)"
    echo "4. List Available Strategies"
    echo "5. Deploy Strategy to Paper Trading"
    echo "6. Help"
    echo "7. Exit"
    echo ""
    echo -e "${YELLOW}Enter your choice (1-7):${NC}"
    read -r choice
    
    case $choice in
        1)
            echo -e "${BLUE}Launching EvoTrader Dashboard...${NC}"
            # Launch the Streamlit dashboard with EvoTrader Lab component
            PYTHONPATH="$PYTHON_PATH" streamlit run "$PROJECT_ROOT/trading_bot/dashboard/app.py" -- --show-evotrader-lab
            ;;
        2)
            echo -e "${BLUE}Running Forex Strategy Evolution...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command evolve --asset forex
            ;;
        3)
            echo -e "${BLUE}Running Crypto Strategy Evolution...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command evolve --asset crypto
            ;;
        4)
            echo -e "${BLUE}Listing Available Strategies...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command list
            ;;
        5)
            echo -e "${YELLOW}Enter strategy ID to deploy:${NC}"
            read -r strategy_id
            echo -e "${BLUE}Deploying Strategy $strategy_id to Paper Trading...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command deploy --strategy-id "$strategy_id"
            ;;
        6)
            echo -e "${BLUE}EvoTrader Integration Help${NC}"
            echo "This launcher provides easy access to EvoTrader functionality within BensBot."
            echo ""
            echo "Available commands:"
            echo "  ./run_evotrader.sh dashboard     - Launch EvoTrader Dashboard"
            echo "  ./run_evotrader.sh evolve forex  - Run Forex Strategy Evolution"
            echo "  ./run_evotrader.sh evolve crypto - Run Crypto Strategy Evolution"
            echo "  ./run_evotrader.sh list          - List Available Strategies"
            echo "  ./run_evotrader.sh deploy <id>   - Deploy Strategy to Paper Trading"
            echo "  ./run_evotrader.sh help          - Show this help message"
            ;;
        7)
            echo -e "${GREEN}Exiting.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            exit 1
            ;;
    esac
else
    # Command line arguments provided
    command="$1"
    case $command in
        dashboard)
            echo -e "${BLUE}Launching EvoTrader Dashboard...${NC}"
            PYTHONPATH="$PYTHON_PATH" streamlit run "$PROJECT_ROOT/trading_bot/dashboard/app.py" -- --show-evotrader-lab
            ;;
        evolve)
            asset="${2:-forex}"
            if [ "$asset" != "forex" ] && [ "$asset" != "crypto" ]; then
                echo -e "${RED}Invalid asset class: $asset. Use 'forex' or 'crypto'.${NC}"
                exit 1
            fi
            echo -e "${BLUE}Running $asset Strategy Evolution...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command evolve --asset "$asset"
            ;;
        list)
            echo -e "${BLUE}Listing Available Strategies...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command list
            ;;
        deploy)
            strategy_id="$2"
            if [ -z "$strategy_id" ]; then
                echo -e "${RED}Strategy ID is required for deployment.${NC}"
                echo -e "${YELLOW}Usage: ./run_evotrader.sh deploy <strategy_id>${NC}"
                exit 1
            fi
            echo -e "${BLUE}Deploying Strategy $strategy_id to Paper Trading...${NC}"
            PYTHONPATH="$PYTHON_PATH" python -m trading_bot.research.evotrader_integration.cli --command deploy --strategy-id "$strategy_id"
            ;;
        help)
            echo -e "${BLUE}EvoTrader Integration Help${NC}"
            echo "This launcher provides easy access to EvoTrader functionality within BensBot."
            echo ""
            echo "Available commands:"
            echo "  ./run_evotrader.sh dashboard     - Launch EvoTrader Dashboard"
            echo "  ./run_evotrader.sh evolve forex  - Run Forex Strategy Evolution"
            echo "  ./run_evotrader.sh evolve crypto - Run Crypto Strategy Evolution"
            echo "  ./run_evotrader.sh list          - List Available Strategies"
            echo "  ./run_evotrader.sh deploy <id>   - Deploy Strategy to Paper Trading"
            echo "  ./run_evotrader.sh help          - Show this help message"
            ;;
        *)
            echo -e "${RED}Unknown command: $command${NC}"
            echo -e "${YELLOW}Run './run_evotrader.sh help' for usage information.${NC}"
            exit 1
            ;;
    esac
fi

exit 0
