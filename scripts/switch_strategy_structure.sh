#!/bin/bash
# Strategy Structure Switch Script
# This script allows switching between the old and new strategy structure

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
BASE_DIR="/Users/bendickinson/Desktop/Trading:BenBot/trading_bot"
STRATEGIES_DIR="${BASE_DIR}/strategies"
STRATEGIES_NEW_DIR="${BASE_DIR}/strategies_new"
STRATEGIES_BACKUP_DIR="${BASE_DIR}/strategies_backup"

# Command to execute based on argument
function switch_to_new() {
    echo -e "${BLUE}Switching to new strategy structure...${NC}"
    
    # Check if strategies_backup already exists
    if [ -d "$STRATEGIES_BACKUP_DIR" ]; then
        echo -e "${RED}Error: Backup directory already exists. Run 'restore' first or remove it manually.${NC}"
        exit 1
    fi
    
    # Create backup of current strategies
    echo "Creating backup of current strategies..."
    mv "$STRATEGIES_DIR" "$STRATEGIES_BACKUP_DIR"
    
    # Move new strategies to production
    echo "Moving new strategies to production..."
    mv "$STRATEGIES_NEW_DIR" "$STRATEGIES_DIR"
    
    echo -e "${GREEN}Successfully switched to new strategy structure!${NC}"
    echo "The old structure is available at $STRATEGIES_BACKUP_DIR"
    echo "To restore, run: $0 restore"
}

function restore_old() {
    echo -e "${BLUE}Restoring original strategy structure...${NC}"
    
    # Check if backup exists
    if [ ! -d "$STRATEGIES_BACKUP_DIR" ]; then
        echo -e "${RED}Error: Backup directory doesn't exist. Nothing to restore.${NC}"
        exit 1
    fi
    
    # Check if there's a strategies_new to move back
    if [ -d "$STRATEGIES_NEW_DIR" ]; then
        echo -e "${RED}Error: strategies_new directory already exists. Remove it first.${NC}"
        exit 1
    fi
    
    # Move current strategies to strategies_new
    echo "Moving current strategies to strategies_new..."
    mv "$STRATEGIES_DIR" "$STRATEGIES_NEW_DIR"
    
    # Restore backup
    echo "Restoring original strategies..."
    mv "$STRATEGIES_BACKUP_DIR" "$STRATEGIES_DIR"
    
    echo -e "${GREEN}Successfully restored original strategy structure!${NC}"
    echo "The new structure is available at $STRATEGIES_NEW_DIR"
    echo "To switch back, run: $0 switch"
}

function check_status() {
    echo -e "${BLUE}Checking strategy structure status...${NC}"
    
    if [ -d "$STRATEGIES_BACKUP_DIR" ]; then
        echo -e "${GREEN}Using new strategy structure${NC}"
        echo "Original structure is backed up at $STRATEGIES_BACKUP_DIR"
    else
        echo -e "${GREEN}Using original strategy structure${NC}"
        if [ -d "$STRATEGIES_NEW_DIR" ]; then
            echo "New structure is available at $STRATEGIES_NEW_DIR"
        else
            echo "New structure not found"
        fi
    fi
}

# Main script logic
case "$1" in
    switch)
        switch_to_new
        ;;
    restore)
        restore_old
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {switch|restore|status}"
        echo "  switch  - Switch to the new strategy structure"
        echo "  restore - Restore the original strategy structure"
        echo "  status  - Check current strategy structure status"
        exit 1
esac

exit 0
