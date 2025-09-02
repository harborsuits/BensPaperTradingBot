#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trade Scoring Demo

This script demonstrates the GPT-4 enhanced trade scoring system
with Telegram notifications using the live trade scoring module.
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
logger = logging.getLogger("TradeScoreDemo")

# Load environment variables
load_dotenv()

# Import the LiveTradeScorer
from trading_bot.live_trade_scorer import LiveTradeScorer


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


def main():
    """Run the live trade scoring demo."""
    print_colored("\n=== GPT-4 Live Trade Scoring Demo ===", "cyan")
    print("This demo demonstrates the GPT-4 enhanced trade scoring system with Telegram notifications.")
    
    # Check environment variables
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not telegram_token or not telegram_chat_id:
        print_colored("Error: Telegram credentials not found in environment variables.", "red")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file or environment.")
        return
    
    # Initialize the LiveTradeScorer
    print_colored("\nInitializing Live Trade Scoring System...", "yellow")
    
    use_mock = openai_api_key is None
    if use_mock:
        print_colored("No OpenAI API key found, using mock LLM client.", "yellow")
    
    try:
        # Initialize the live trade scorer
        scorer = LiveTradeScorer(
            use_mock=use_mock,
            telegram_config={
                "debug": True
            },
            notifier_config={
                "include_reasoning": True,
                "include_market_context": True,
                "include_strategy_performance": True
            }
        )
        print_colored("✓ Live Trade Scoring System initialized successfully", "green")
    except Exception as e:
        print_colored(f"✗ Error initializing Live Trade Scoring System: {str(e)}", "red")
        return
    
    # Sample trade setups
    print_colored("\nLoading sample trade setups...", "yellow")
    
    sample_trades = [
        {
            "symbol": "AAPL",
            "strategy_name": "breakout_swing",
            "direction": "long",
            "entry": 186.50,
            "stop": 182.00,
            "target": 195.00,
            "timeframe": "4h",
            "setup_type": "cup_and_handle_breakout",
            "quantity": 10,
            "risk_reward": 2.1
        },
        {
            "symbol": "MSFT",
            "strategy_name": "momentum",
            "direction": "long",
            "entry": 378.25,
            "stop": 370.50,
            "target": 395.00,
            "timeframe": "1d",
            "setup_type": "trend_continuation",
            "quantity": 5,
            "risk_reward": 2.2
        },
        {
            "symbol": "TSLA",
            "strategy_name": "mean_reversion",
            "direction": "long",
            "entry": 215.30,
            "stop": 210.00,
            "target": 230.00,
            "timeframe": "1h",
            "setup_type": "oversold_bounce",
            "quantity": 8,
            "risk_reward": 3.0
        }
    ]
    
    # Sample market context
    market_context = {
        "market_regime": "bullish",
        "sector_performance": {
            "technology": 2.3,
            "finance": -0.5,
            "healthcare": 0.8,
            "energy": 1.2,
            "consumer": 0.3
        },
        "market_indices": {
            "SPY": {
                "daily_change_pct": 0.75,
                "trend": "uptrend",
                "above_200ma": True
            },
            "QQQ": {
                "daily_change_pct": 1.2,
                "trend": "uptrend",
                "above_200ma": True
            }
        },
        "volatility_index": 15.7,
        "trend_strength": 0.82,
        "risk_appetite": "high",
        "liquidity": "abundant",
        "recent_news": [
            {
                "headline": "Fed signals potential rate cut in next meeting",
                "sentiment": "positive",
                "relevance": 0.9,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    
    # Sample strategy performance
    strategy_performance = {
        "breakout_swing": {
            "win_rate": 0.72,
            "profit_factor": 3.2,
            "avg_win": 2.1,
            "avg_loss": -0.8,
            "trades_count": 50,
            "recent_trades": [
                {"symbol": "MSFT", "result": "win", "return_pct": 3.2},
                {"symbol": "AMZN", "result": "loss", "return_pct": -1.5},
                {"symbol": "GOOGL", "result": "win", "return_pct": 4.1}
            ],
            "regime_performance": {
                "bullish": {"win_rate": 0.80, "trades": 25},
                "bearish": {"win_rate": 0.45, "trades": 12}
            }
        },
        "momentum": {
            "win_rate": 0.68,
            "profit_factor": 2.8,
            "avg_win": 2.4,
            "avg_loss": -1.2,
            "trades_count": 38,
            "recent_trades": [
                {"symbol": "TSLA", "result": "win", "return_pct": 5.3},
                {"symbol": "NVDA", "result": "win", "return_pct": 4.2},
                {"symbol": "AMD", "result": "win", "return_pct": 3.1}
            ],
            "regime_performance": {
                "bullish": {"win_rate": 0.75, "trades": 28},
                "bearish": {"win_rate": 0.40, "trades": 10}
            }
        },
        "mean_reversion": {
            "win_rate": 0.65,
            "profit_factor": 2.1,
            "avg_win": 1.5,
            "avg_loss": -0.9,
            "trades_count": 42,
            "recent_trades": [
                {"symbol": "SPY", "result": "win", "return_pct": 1.2},
                {"symbol": "QQQ", "result": "win", "return_pct": 1.8},
                {"symbol": "AAPL", "result": "loss", "return_pct": -1.1}
            ],
            "regime_performance": {
                "bullish": {"win_rate": 0.60, "trades": 20},
                "bearish": {"win_rate": 0.70, "trades": 22}
            }
        }
    }
    
    # Demo options
    print_colored("\nDemo Options:", "cyan")
    print("1. Evaluate a single trade")
    print("2. Evaluate multiple trades")
    print("3. Run both demos")
    print("0. Exit")
    
    choice = input("\nSelect an option (0-3): ")
    
    if choice == "0":
        print_colored("Exiting demo.", "yellow")
        return
    
    if choice == "1" or choice == "3":
        # Evaluate a single trade
        print_colored("\n=== Evaluating Single Trade ===", "cyan")
        print(f"Trade: {sample_trades[0]['symbol']} {sample_trades[0]['direction']} @ ${sample_trades[0]['entry']}")
        print(f"Setup: {sample_trades[0]['setup_type']} on {sample_trades[0]['timeframe']} timeframe")
        print(f"Risk/Reward: {sample_trades[0]['risk_reward']}")
        
        print_colored("\nSending to GPT-4 for evaluation...", "blue")
        
        # Evaluate the trade
        evaluation = scorer.evaluate_trade(
            trade_data=sample_trades[0],
            context_data=market_context,
            strategy_perf=strategy_performance,
            notify=True
        )
        
        # Print evaluation results
        print_colored("\nTrade Evaluation Results:", "green")
        print(f"Confidence Score: {evaluation.get('confidence_score', 0):.1f}/10.0")
        print(f"Bias Alignment: {evaluation.get('bias_alignment', 'N/A')}")
        print(f"Recommendation: {evaluation.get('recommendation', 'N/A')}")
        print_colored(f"\nReasoning:", "magenta")
        print(f"{evaluation.get('reasoning', 'No reasoning provided')}")
        
        # Determine if trade should be executed
        should_execute = evaluation.get('should_execute', False)
        if should_execute:
            print_colored("\n✓ DECISION: Execute the trade", "green")
        else:
            print_colored("\n✗ DECISION: Skip the trade", "red")
        
        print_colored("\nTelegram notification has been sent with these results.", "blue")
        
        # Add a pause if we're also running the multiple trades demo
        if choice == "3":
            print_colored("\nPausing before multiple trades demo...", "yellow")
            time.sleep(5)
    
    if choice == "2" or choice == "3":
        # Evaluate multiple trades
        print_colored("\n=== Evaluating Multiple Trades ===", "cyan")
        print(f"Number of trades: {len(sample_trades)}")
        
        for i, trade in enumerate(sample_trades):
            print(f"Trade {i+1}: {trade['symbol']} {trade['direction']} @ ${trade['entry']} ({trade['setup_type']})")
        
        print_colored("\nSending to GPT-4 for batch evaluation...", "blue")
        print("This may take a minute as we're evaluating multiple trades...")
        
        # Evaluate multiple trades
        results = scorer.evaluate_multiple_trades(
            trades=sample_trades,
            context_data=market_context,
            strategy_perf=strategy_performance,
            delay=2.0
        )
        
        # Print summary
        print_colored("\nBatch Evaluation Complete!", "green")
        
        approved_count = sum(1 for r in results if r['evaluation'].get('should_execute', False))
        rejected_count = len(results) - approved_count
        
        print(f"Trades evaluated: {len(results)}")
        print(f"Approved: {approved_count}")
        print(f"Rejected: {rejected_count}")
        
        # Print results for each trade
        print_colored("\nIndividual Results:", "cyan")
        
        for i, result in enumerate(results):
            trade = result['trade_data']
            evaluation = result['evaluation']
            
            print_colored(f"\nTrade {i+1}: {trade['symbol']} ({trade['direction']})", "yellow")
            print(f"Confidence: {evaluation.get('confidence_score', 0):.1f}/10.0")
            print(f"Recommendation: {evaluation.get('recommendation', 'N/A')}")
            print(f"Execute: {'Yes' if evaluation.get('should_execute', False) else 'No'}")
        
        print_colored("\nTelegram notifications have been sent with these results.", "blue")
        print_colored("A batch summary notification has also been sent.", "blue")
    
    print_colored("\nDemo completed successfully!", "green")
    print("Check your Telegram for the evaluation notifications.")


if __name__ == "__main__":
    main() 