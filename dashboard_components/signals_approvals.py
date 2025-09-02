"""
Signals & Approvals Component for BensBot Dashboard
Displays trading signals and strategy approval interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

#############################
# Data Retrieval Functions
#############################

def get_trading_signals(db, account_type=None):
    """
    Retrieve upcoming and recent trading signals from MongoDB
    Returns a DataFrame with signal details
    """
    if db is None:
        return generate_mock_signals_data(account_type)
    
    try:
        # Query filter based on account_type
        query = {}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get signals from MongoDB
        signals_docs = list(db.trading_signals.find(query).sort("timestamp", -1).limit(50))
        
        if signals_docs:
            # Convert to DataFrame
            df = pd.DataFrame(signals_docs)
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_signals_data(account_type)
    except Exception as e:
        st.error(f"Error retrieving signals data: {e}")
        return generate_mock_signals_data(account_type)

def get_pending_approvals(db, account_type=None):
    """
    Retrieve strategies pending approval from MongoDB
    Returns a DataFrame with strategy details
    """
    if db is None:
        return generate_mock_approvals_data(account_type)
    
    try:
        # Query filter based on account_type
        query = {"status": "pending_approval"}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get pending approvals from MongoDB
        approvals_docs = list(db.strategies.find(query))
        
        if approvals_docs:
            # Convert to DataFrame
            df = pd.DataFrame(approvals_docs)
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_approvals_data(account_type)
    except Exception as e:
        st.error(f"Error retrieving approvals data: {e}")
        return generate_mock_approvals_data(account_type)

def get_signal_history(db, strategy_id, account_type=None):
    """
    Retrieve historical signals for a specific strategy
    Returns a DataFrame with signal history
    """
    if db is None:
        return generate_mock_signal_history(strategy_id, account_type)
    
    try:
        # Query filter
        query = {"strategy_id": strategy_id}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get signal history from MongoDB
        history_docs = list(db.signal_history.find(query).sort("timestamp", 1))
        
        if history_docs:
            # Convert to DataFrame
            df = pd.DataFrame(history_docs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_signal_history(strategy_id, account_type)
    except Exception as e:
        st.error(f"Error retrieving signal history: {e}")
        return generate_mock_signal_history(strategy_id, account_type)

#############################
# Mock Data Generators
#############################

def generate_mock_signals_data(account_type=None):
    """
    Generate synthetic trading signals data for development and testing
    Returns a DataFrame with signal details
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        # If no specific account type, default to all types
        account_types = ["live", "paper"]
    
    # Define asset classes and symbols
    asset_classes = {
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM"],
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "ADA/USD", "DOT/USD"],
        "commodity": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F"]
    }
    
    # Strategy templates
    strategy_templates = [
        {"id": "s001", "name": "TrendFollower"},
        {"id": "s002", "name": "BreakoutHunter"},
        {"id": "s003", "name": "MeanReverter"},
        {"id": "s004", "name": "VolatilityHarvester"},
        {"id": "s005", "name": "MacroEdge"},
        {"id": "s006", "name": "SectorRotator"},
        {"id": "s007", "name": "DividendHarvester"},
        {"id": "s008", "name": "SwingTrader"},
        {"id": "s009", "name": "PatternRecognizer"},
        {"id": "s010", "name": "TechnicalEnsemble"}
    ]
    
    # Number of signals to generate
    num_signals = random.randint(15, 30)
    
    signals = []
    
    # Signal types
    signal_types = ["entry", "exit", "adjust_position", "adjust_stop", "adjust_target"]
    signal_directions = ["long", "short"]
    signal_strengths = ["weak", "moderate", "strong"]
    signal_statuses = ["pending", "triggered", "executed", "expired", "rejected"]
    
    # Generate random signals
    for i in range(num_signals):
        # Select random asset class and symbol
        asset_class = random.choice(list(asset_classes.keys()))
        symbol = random.choice(asset_classes[asset_class])
        
        # Select random strategy
        strategy = random.choice(strategy_templates)
        
        # Determine if this is a future or past signal
        is_future = random.random() < 0.4  # 40% chance of future signal
        
        # Generate timestamp
        now = datetime.datetime.now()
        if is_future:
            # Future signal (upcoming)
            minutes_from_now = random.randint(5, 60 * 24)  # Up to 24 hours in the future
            timestamp = now + datetime.timedelta(minutes=minutes_from_now)
            # Future signals can only be pending
            status = "pending"
        else:
            # Past signal
            minutes_ago = random.randint(1, 60 * 24 * 7)  # Up to 7 days in the past
            timestamp = now - datetime.timedelta(minutes=minutes_ago)
            # Past signals can be any status except pending
            status = random.choice([s for s in signal_statuses if s != "pending"])
        
        # Generate signal details
        signal_type = random.choice(signal_types)
        
        # Price depends on signal type
        current_price = random.uniform(10, 1000) if asset_class == "equity" else \
                        random.uniform(0.5, 2.0) if asset_class == "forex" else \
                        random.uniform(100, 50000) if asset_class == "crypto" else \
                        random.uniform(20, 2000)  # commodity
        
        if signal_type == "entry":
            signal_price = current_price * random.uniform(0.99, 1.01)
            stop_price = signal_price * (random.uniform(0.9, 0.95) if random.choice(signal_directions) == "long" else random.uniform(1.05, 1.1))
            target_price = signal_price * (random.uniform(1.05, 1.15) if random.choice(signal_directions) == "long" else random.uniform(0.85, 0.95))
        elif signal_type == "exit":
            signal_price = current_price * random.uniform(0.99, 1.01)
            stop_price = None
            target_price = None
        elif signal_type == "adjust_position":
            signal_price = current_price * random.uniform(0.99, 1.01)
            stop_price = signal_price * (random.uniform(0.9, 0.95) if random.choice(signal_directions) == "long" else random.uniform(1.05, 1.1))
            target_price = signal_price * (random.uniform(1.05, 1.15) if random.choice(signal_directions) == "long" else random.uniform(0.85, 0.95))
        elif signal_type == "adjust_stop":
            signal_price = None
            stop_price = current_price * (random.uniform(0.9, 0.95) if random.choice(signal_directions) == "long" else random.uniform(1.05, 1.1))
            target_price = None
        else:  # adjust_target
            signal_price = None
            stop_price = None
            target_price = current_price * (random.uniform(1.05, 1.15) if random.choice(signal_directions) == "long" else random.uniform(0.85, 0.95))
        
        # Select account type for this signal
        if isinstance(account_type, list):
            signal_account_type = random.choice(account_type)
        else:
            signal_account_type = account_type or random.choice(["live", "paper"])
        
        # Create signal object
        signal = {
            "signal_id": f"sig_{random.randint(10000, 99999)}",
            "symbol": symbol,
            "asset_class": asset_class,
            "strategy_id": strategy["id"],
            "strategy_name": strategy["name"],
            "signal_type": signal_type,
            "direction": random.choice(signal_directions),
            "strength": random.choice(signal_strengths),
            "signal_price": None if signal_price is None else round(signal_price, 2),
            "stop_price": None if stop_price is None else round(stop_price, 2),
            "target_price": None if target_price is None else round(target_price, 2),
            "current_price": round(current_price, 2),
            "quantity": random.randint(10, 200) if asset_class == "equity" else \
                       random.randint(10000, 100000) if asset_class == "forex" else \
                       round(random.uniform(0.1, 10), 4) if asset_class == "crypto" else \
                       random.randint(1, 10),  # commodity
            "timestamp": timestamp,
            "status": status,
            "account_type": signal_account_type,
            "risk_reward": round(random.uniform(1.5, 3.0), 2),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "notes": f"Signal generated based on {random.choice(['price action', 'momentum', 'breakout', 'reversion', 'volatility'])} analysis"
        }
        
        signals.append(signal)
    
    # Sort by timestamp (newest first)
    signals = sorted(signals, key=lambda x: x["timestamp"], reverse=True)
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)
    
    return signals_df

def generate_mock_approvals_data(account_type=None):
    """
    Generate synthetic strategy approval data for development and testing
    Returns a DataFrame with strategy details
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        # If no specific account type, randomly choose one
        account_type = random.choice(["live", "paper"])
    
    # Strategy templates with consistent IDs
    strategy_templates = [
        {"id": "s001", "name": "TrendFollower", "category": "momentum", 
         "description": "Classic trend following strategy using SMA crossovers"},
        {"id": "s002", "name": "BreakoutHunter", "category": "breakout", 
         "description": "Detects and trades price breakouts with ATR-based risk management"},
        {"id": "s003", "name": "MeanReverter", "category": "mean_reversion", 
         "description": "Mean reversion using Bollinger Bands and RSI"},
        {"id": "s004", "name": "VolatilityHarvester", "category": "volatility", 
         "description": "Options-based volatility harvesting strategy"},
        {"id": "s005", "name": "MacroEdge", "category": "macro", 
         "description": "Macro factors-driven strategy using economic indicators"},
        {"id": "s006", "name": "SectorRotator", "category": "rotation", 
         "description": "Sector rotation based on market cycle analysis"},
        {"id": "s007", "name": "DividendHarvester", "category": "income", 
         "description": "Focuses on stable dividend income with low volatility"},
        {"id": "s008", "name": "SwingTrader", "category": "swing", 
         "description": "Multi-day swing trading using momentum and sentiment"},
        {"id": "s009", "name": "PatternRecognizer", "category": "pattern", 
         "description": "Trades chart patterns with machine learning validation"},
        {"id": "s010", "name": "TechnicalEnsemble", "category": "ensemble", 
         "description": "Ensemble of technical indicators with adaptive weighting"}
    ]
    
    # Number of pending approvals to generate
    num_approvals = random.randint(3, 7)
    
    approvals = []
    
    # Generate random approvals
    for i in range(num_approvals):
        # Select random strategy
        template = random.choice(strategy_templates)
        
        # Create a copy with the status set to pending_approval
        strategy = template.copy()
        strategy["status"] = "pending_approval"
        strategy["approved"] = False
        strategy["account_type"] = account_type
        strategy["strategy_id"] = f"{template['id']}_{account_type}"
        
        # Generate random performance metrics
        strategy["win_rate"] = round(random.uniform(52, 68), 1)
        strategy["profit_factor"] = round(random.uniform(1.2, 2.2), 2)
        strategy["sharpe_ratio"] = round(random.uniform(1.0, 2.0), 2)
        strategy["max_drawdown"] = round(random.uniform(-3, -10), 1)
        strategy["total_trades"] = random.randint(50, 300)
        strategy["annual_return"] = round(random.uniform(8, 25), 1)
        
        # Generate approval request timestamp
        days_ago = random.randint(1, 10)
        strategy["request_date"] = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Add some parameters
        strategy["parameters"] = {
            "fast_ma": random.randint(5, 20),
            "slow_ma": random.randint(20, 50),
            "stop_loss_pct": round(random.uniform(2, 5), 1),
            "take_profit_pct": round(random.uniform(5, 15), 1),
            "max_position_size": round(random.uniform(2, 10), 1)
        }
        
        approvals.append(strategy)
    
    # Convert to DataFrame
    approvals_df = pd.DataFrame(approvals)
    
    return approvals_df

def generate_mock_signal_history(strategy_id, account_type=None):
    """
    Generate synthetic signal history for a specific strategy
    Returns a DataFrame with signal history
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        account_type = random.choice(["live", "paper"])
    
    # Determine signal count and timeframe
    signal_count = random.randint(15, 30)
    days_back = 90  # 3 months of history
    
    # Extract base strategy_id and name
    base_strategy_id = strategy_id.split('_')[0] if '_' in strategy_id else strategy_id
    
    # Map strategy ID to name
    strategy_names = {
        "s001": "TrendFollower",
        "s002": "BreakoutHunter",
        "s003": "MeanReverter",
        "s004": "VolatilityHarvester",
        "s005": "MacroEdge",
        "s006": "SectorRotator",
        "s007": "DividendHarvester",
        "s008": "SwingTrader",
        "s009": "PatternRecognizer",
        "s010": "TechnicalEnsemble"
    }
    
    strategy_name = strategy_names.get(base_strategy_id, "Unknown Strategy")
    
    # Define asset classes and symbols
    asset_classes = {
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM"],
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "ADA/USD", "DOT/USD"],
        "commodity": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F"]
    }
    
    # Signal types and statuses
    signal_types = ["entry", "exit"]
    signal_directions = ["long", "short"]
    signal_statuses = ["triggered", "executed", "expired", "rejected"]
    
    # Generate signal timestamps (oldest to newest)
    now = datetime.datetime.now()
    timestamps = []
    
    for i in range(signal_count):
        days_ago = random.randint(0, days_back)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        timestamp = now - datetime.timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        timestamps.append(timestamp)
    
    # Sort timestamps (oldest first)
    timestamps.sort()
    
    # For consistency, pick one asset class and a few symbols for this strategy
    primary_asset_class = random.choice(list(asset_classes.keys()))
    strategy_symbols = random.sample(asset_classes[primary_asset_class], min(5, len(asset_classes[primary_asset_class])))
    
    # Determine base win rate for this strategy
    base_win_rate = random.uniform(0.4, 0.7)
    
    # Generate signal history
    history = []
    
    for i in range(signal_count):
        # Select symbol from strategy's symbols
        symbol = random.choice(strategy_symbols)
        
        # Generate signal type and direction
        signal_type = random.choice(signal_types)
        direction = random.choice(signal_directions)
        
        # Generate prices based on asset class
        if primary_asset_class == "equity":
            current_price = round(random.uniform(50, 500), 2)
        elif primary_asset_class == "forex":
            current_price = round(random.uniform(0.5, 2.0), 5)
        elif primary_asset_class == "crypto":
            current_price = round(random.uniform(100, 50000), 2)
        else:  # commodity
            current_price = round(random.uniform(20, 2000), 2)
        
        signal_price = round(current_price * random.uniform(0.99, 1.01), 2)
        
        # Determine if this signal was a winner (based on base win rate)
        is_winner = random.random() < base_win_rate
        
        # Calculate P&L based on whether it was a winner
        if is_winner:
            if direction == "long":
                exit_price = round(signal_price * random.uniform(1.01, 1.1), 2)
            else:  # short
                exit_price = round(signal_price * random.uniform(0.9, 0.99), 2)
            
            pnl = round((exit_price - signal_price) if direction == "long" else (signal_price - exit_price), 2)
            pnl_pct = round(((exit_price / signal_price) - 1) * 100 if direction == "long" else ((signal_price / exit_price) - 1) * 100, 2)
        else:
            if direction == "long":
                exit_price = round(signal_price * random.uniform(0.9, 0.99), 2)
            else:  # short
                exit_price = round(signal_price * random.uniform(1.01, 1.1), 2)
            
            pnl = round((exit_price - signal_price) if direction == "long" else (signal_price - exit_price), 2)
            pnl_pct = round(((exit_price / signal_price) - 1) * 100 if direction == "long" else ((signal_price / exit_price) - 1) * 100, 2)
        
        # Generate quantity based on asset class
        if primary_asset_class == "equity":
            quantity = random.randint(10, 200)
        elif primary_asset_class == "forex":
            quantity = random.randint(10000, 100000)
        elif primary_asset_class == "crypto":
            quantity = round(random.uniform(0.1, 10), 4)
        else:  # commodity
            quantity = random.randint(1, 10)
        
        # Calculate value
        value = quantity * abs(signal_price)
        
        # Create signal history record
        record = {
            "signal_id": f"sig_{random.randint(10000, 99999)}",
            "symbol": symbol,
            "asset_class": primary_asset_class,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "signal_type": signal_type,
            "direction": direction,
            "signal_price": signal_price,
            "exit_price": exit_price if signal_type == "entry" else None,
            "quantity": quantity,
            "value": round(value, 2),
            "pnl": pnl if signal_type == "entry" else None,
            "pnl_pct": pnl_pct if signal_type == "entry" else None,
            "timestamp": timestamps[i],
            "status": random.choice(signal_statuses),
            "account_type": account_type,
            "is_winner": is_winner if signal_type == "entry" else None
        }
        
        history.append(record)
    
    # Convert to DataFrame
    history_df = pd.DataFrame(history)
    
    return history_df
