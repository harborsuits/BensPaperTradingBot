#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Rotation Demo

This script demonstrates the AI-powered strategy rotation system
that automatically adjusts capital allocation based on market conditions.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyRotationDemo")

# Load environment variables
load_dotenv()

# Import required modules
from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
from trading_bot.ai_scoring.strategy_rotator import StrategyRotator
from trading_bot.notification_manager.telegram_notifier import TelegramNotifier


def print_colored(text: str, color: str = None):
    """Print colored text to the console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color.lower(), colors["reset"]) if color else colors["reset"]
    print(f"{color_code}{text}{colors['reset']}")


def get_sample_market_conditions():
    """Get sample market conditions for the demo."""
    # Define different market scenarios
    scenarios = {
        "bullish": {
            "market_regime": "bullish",
            "volatility_index": 15.2,
            "trend_strength": 0.75,
            "market_indices": {
                "SPY": {"daily_change_pct": 0.8, "above_200ma": True},
                "QQQ": {"daily_change_pct": 1.2, "above_200ma": True}
            },
            "sector_performance": {
                "technology": 1.5,
                "financials": 0.7,
                "healthcare": 0.3,
                "energy": -0.2,
                "utilities": -0.5
            },
            "recent_news": [
                {
                    "headline": "Fed signals continuation of accommodative policy",
                    "sentiment": "positive",
                    "relevance": 0.85
                }
            ]
        },
        "bearish": {
            "market_regime": "bearish",
            "volatility_index": 32.5,
            "trend_strength": 0.65,
            "market_indices": {
                "SPY": {"daily_change_pct": -1.8, "above_200ma": False},
                "QQQ": {"daily_change_pct": -2.5, "above_200ma": False}
            },
            "sector_performance": {
                "technology": -3.2,
                "financials": -1.5,
                "healthcare": -0.8,
                "energy": -1.2,
                "utilities": 0.3
            },
            "recent_news": [
                {
                    "headline": "Recession fears grow as economic indicators weaken",
                    "sentiment": "negative",
                    "relevance": 0.9
                }
            ]
        },
        "volatile": {
            "market_regime": "volatile",
            "volatility_index": 28.5,
            "trend_strength": 0.3,
            "market_indices": {
                "SPY": {"daily_change_pct": -1.5, "above_200ma": True},
                "QQQ": {"daily_change_pct": -2.1, "above_200ma": False}
            },
            "sector_performance": {
                "technology": -2.3,
                "financials": -1.0,
                "healthcare": 0.8,
                "energy": -1.2,
                "utilities": 1.5
            },
            "recent_news": [
                {
                    "headline": "Market volatility spikes amid mixed economic data",
                    "sentiment": "neutral",
                    "relevance": 0.85
                }
            ]
        },
        "sideways": {
            "market_regime": "sideways",
            "volatility_index": 18.5,
            "trend_strength": 0.2,
            "market_indices": {
                "SPY": {"daily_change_pct": 0.1, "above_200ma": True},
                "QQQ": {"daily_change_pct": -0.2, "above_200ma": True}
            },
            "sector_performance": {
                "technology": 0.3,
                "financials": -0.2,
                "healthcare": 0.5,
                "energy": -0.1,
                "utilities": 0.2
            },
            "recent_news": [
                {
                    "headline": "Markets await direction as earnings season begins",
                    "sentiment": "neutral",
                    "relevance": 0.75
                }
            ]
        }
    }
    
    return scenarios


def main():
    """Run the strategy rotation demo."""
    print_colored("\n=== AI-Powered Strategy Rotation Demo ===", "cyan")
    print("This demo demonstrates how the system automatically adjusts strategy")
    print("allocations based on different market conditions using AI analysis.")
    
    # Check environment variables
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    # Available strategies
    strategies = [
        "breakout_swing",
        "momentum",
        "mean_reversion",
        "trend_following",
        "volatility_breakout",
        "option_spreads"
    ]
    
    # Initialize Telegram notifier if credentials available
    notifier = None
    if telegram_token and telegram_chat_id:
        try:
            notifier = TelegramNotifier(
                bot_token=telegram_token,
                default_chat_id=telegram_chat_id,
                debug=True
            )
            print_colored("✓ Telegram notifier initialized successfully", "green")
        except Exception as e:
            print_colored(f"✗ Error initializing Telegram notifier: {str(e)}", "red")
    else:
        print_colored("! Telegram credentials not found, notifications disabled", "yellow")
    
    # Initialize StrategyPrioritizer
    try:
        prioritizer = StrategyPrioritizer(available_strategies=strategies)
        print_colored("✓ Strategy prioritizer initialized successfully", "green")
    except Exception as e:
        print_colored(f"✗ Error initializing strategy prioritizer: {str(e)}", "red")
        return
    
    # Initialize StrategyRotator
    try:
        rotator = StrategyRotator(
            strategies=strategies,
            strategy_prioritizer=prioritizer,
            portfolio_value=100000.0,
            notifier=notifier,
            max_allocation_change=20.0,  # Allow larger changes for demo
            rotation_frequency_days=1    # Rotation every day for demo
        )
        print_colored("✓ Strategy rotator initialized successfully", "green")
    except Exception as e:
        print_colored(f"✗ Error initializing strategy rotator: {str(e)}", "red")
        return
    
    # Get market scenarios
    scenarios = get_sample_market_conditions()
    
    # Demo loop - run through different market scenarios
    print_colored("\nDemo options:", "cyan")
    print("1. Run all market scenarios sequentially")
    print("2. Test specific market scenario")
    print("3. Test manual allocation adjustment")
    print("0. Exit")
    
    choice = input("\nSelect an option (0-3): ")
    
    if choice == "0":
        print_colored("Exiting demo.", "yellow")
        return
    
    elif choice == "1":
        # Run all scenarios
        print_colored("\nRunning all market scenarios sequentially...", "yellow")
        
        for scenario_name, market_context in scenarios.items():
            print_colored(f"\n=== Testing {scenario_name.upper()} market scenario ===", "cyan")
            print(f"Market regime: {market_context['market_regime']}")
            print(f"VIX: {market_context['volatility_index']}")
            print(f"Trend strength: {market_context['trend_strength']}")
            
            # Get current allocations before rotation
            print_colored("\nCurrent allocations before rotation:", "yellow")
            for strategy, allocation in rotator.get_allocations().items():
                print(f"{strategy}: {allocation:.1f}%")
            
            # Perform rotation
            print_colored("\nPerforming strategy rotation...", "blue")
            result = rotator.rotate_strategies(market_context, force_rotation=True)
            
            # Print result
            if result['status'] == 'success':
                print_colored("✓ Rotation completed successfully", "green")
                
                # Print new allocations
                print_colored("\nNew allocations after rotation:", "green")
                for strategy, allocation in rotator.get_allocations().items():
                    dollar_allocation = (allocation / 100.0) * rotator.portfolio_value
                    print(f"{strategy}: {allocation:.1f}% (${dollar_allocation:,.2f})")
                
                # Print biggest changes
                changes = result['results']['allocation_changes']
                sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)
                print_colored("\nBiggest allocation changes:", "magenta")
                for strategy, change in sorted_changes[:3]:
                    direction = "+" if change > 0 else ""
                    print(f"{strategy}: {direction}{change:.1f}%")
            else:
                print_colored(f"✗ Rotation failed: {result['message']}", "red")
            
            # Pause between scenarios
            if scenario_name != list(scenarios.keys())[-1]:
                print_colored("\nPausing before next scenario...", "yellow")
                time.sleep(3)
        
        print_colored("\nAll scenarios completed!", "green")
        
    elif choice == "2":
        # Test specific scenario
        print_colored("\nAvailable market scenarios:", "cyan")
        for i, scenario in enumerate(scenarios.keys(), 1):
            print(f"{i}. {scenario.title()}")
        
        scenario_choice = input("\nSelect a scenario (1-4): ")
        try:
            scenario_idx = int(scenario_choice) - 1
            if 0 <= scenario_idx < len(scenarios):
                scenario_name = list(scenarios.keys())[scenario_idx]
                market_context = scenarios[scenario_name]
                
                print_colored(f"\n=== Testing {scenario_name.upper()} market scenario ===", "cyan")
                print(f"Market regime: {market_context['market_regime']}")
                print(f"VIX: {market_context['volatility_index']}")
                
                # Get strategy rankings from prioritizer
                print_colored("\nGetting strategy rankings from AI...", "blue")
                priorities = prioritizer.prioritize_strategies(market_context)
                
                print_colored("\nAI Strategy Rankings:", "green")
                for i, strategy in enumerate(priorities['rankings'], 1):
                    print(f"{i}. {strategy}")
                
                print_colored("\nMarket Summary:", "yellow")
                print(priorities['market_summary'])
                
                # Perform rotation
                print_colored("\nPerforming strategy rotation...", "blue")
                result = rotator.rotate_strategies(market_context, force_rotation=True)
                
                # Print result
                if result['status'] == 'success':
                    print_colored("✓ Rotation completed successfully", "green")
                    
                    # Print new allocations
                    print_colored("\nNew allocations after rotation:", "green")
                    for strategy, allocation in rotator.get_allocations().items():
                        dollar_allocation = (allocation / 100.0) * rotator.portfolio_value
                        print(f"{strategy}: {allocation:.1f}% (${dollar_allocation:,.2f})")
            else:
                print_colored("Invalid scenario choice", "red")
        except ValueError:
            print_colored("Invalid input", "red")
            
    elif choice == "3":
        # Test manual allocation adjustment
        print_colored("\n=== Testing Manual Allocation Adjustment ===", "cyan")
        
        # Print current allocations
        print_colored("\nCurrent allocations:", "yellow")
        for strategy, allocation in rotator.get_allocations().items():
            print(f"{strategy}: {allocation:.1f}%")
        
        # Get strategy to adjust
        print_colored("\nAvailable strategies:", "cyan")
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
        
        strategy_choice = input("\nSelect a strategy to adjust (1-6): ")
        try:
            strategy_idx = int(strategy_choice) - 1
            if 0 <= strategy_idx < len(strategies):
                strategy = strategies[strategy_idx]
                current_allocation = rotator.get_allocations().get(strategy, 0.0)
                
                print(f"\nAdjusting allocation for: {strategy}")
                print(f"Current allocation: {current_allocation:.1f}%")
                
                new_allocation = float(input("Enter new allocation percentage: "))
                
                # Perform adjustment
                print_colored("\nPerforming manual adjustment...", "blue")
                result = rotator.manual_adjust_allocation(strategy, new_allocation)
                
                # Print result
                if result['status'] == 'success':
                    print_colored("✓ Adjustment completed successfully", "green")
                    
                    # Print new allocations
                    print_colored("\nNew allocations after adjustment:", "green")
                    for s, allocation in rotator.get_allocations().items():
                        print(f"{s}: {allocation:.1f}%")
                else:
                    print_colored(f"✗ Adjustment failed: {result['message']}", "red")
            else:
                print_colored("Invalid strategy choice", "red")
        except ValueError:
            print_colored("Invalid input", "red")
    
    # Print final summary
    print_colored("\nDemo completed successfully!", "green")
    print("Strategy rotation system has demonstrated how AI can automatically")
    print("adjust capital allocation based on changing market conditions.")


if __name__ == "__main__":
    main() 