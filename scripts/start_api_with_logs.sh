#!/bin/bash
# Start the API with detailed logging to help diagnose connection issues

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${GREEN}Starting BenBot API with REAL OpenAI Integration${NC}"
echo -e "${BLUE}===================================================${NC}"

# Set project root
PROJECT_ROOT="/Users/bendickinson/Desktop/Trading:BenBot"
cd "$PROJECT_ROOT"

# Set up logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_ROOT/api_log_$TIMESTAMP.log"
echo -e "${YELLOW}Logging API output to: $LOG_FILE${NC}"

# Setup Python path
export PYTHONPATH="$PROJECT_ROOT"

# Add debug logging
cat > "$PROJECT_ROOT/debug_logger.py" << EOL
import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("$LOG_FILE"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set OpenAI debug logging
logging.getLogger("openai").setLevel(logging.DEBUG)
EOL

# Start the API with debug logging
echo -e "${GREEN}API server starting at http://localhost:5000${NC}"
echo -e "${BLUE}Dashboard will connect to this API automatically${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo -e "${BLUE}===================================================${NC}"

# Run with debug logging
PYTHONPATH="$PROJECT_ROOT" python3 -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); import debug_logger; from trading_bot.api.app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=5000)"
