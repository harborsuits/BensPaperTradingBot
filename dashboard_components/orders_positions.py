"""
Orders & Positions Component for BensBot Dashboard
Displays current positions and open orders with management controls
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

def get_positions(db, account_type=None):
    """
    Retrieve current positions from MongoDB for specified account type
    Returns a DataFrame with position details
    """
    if db is None:
        return generate_mock_positions_data(account_type)
    
    try:
        # Query filter based on account_type
        query = {}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get positions from MongoDB
        positions_docs = list(db.positions.find(query))
        
        if positions_docs:
            # Convert to DataFrame
            df = pd.DataFrame(positions_docs)
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_positions_data(account_type)
    except Exception as e:
        st.error(f"Error retrieving positions data: {e}")
        return generate_mock_positions_data(account_type)

def get_open_orders(db, account_type=None):
    """
    Retrieve open/pending orders from MongoDB for specified account type
    Returns a DataFrame with order details
    """
    if db is None:
        return generate_mock_orders_data(account_type)
    
    try:
        # Query filter based on account_type
        query = {"status": {"$in": ["open", "pending", "partial_fill"]}}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get orders from MongoDB
        orders_docs = list(db.orders.find(query))
        
        if orders_docs:
            # Convert to DataFrame
            df = pd.DataFrame(orders_docs)
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_orders_data(account_type)
    except Exception as e:
        st.error(f"Error retrieving orders data: {e}")
        return generate_mock_orders_data(account_type)

def get_position_history(db, symbol, account_type=None):
    """
    Retrieve historical data for a specific position
    Returns a DataFrame with position value over time
    """
    if db is None:
        return generate_mock_position_history(symbol, account_type)
    
    try:
        # Query filter
        query = {"symbol": symbol}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get position history from MongoDB
        history_docs = list(db.position_history.find(query).sort("timestamp", 1))
        
        if history_docs:
            # Convert to DataFrame
            df = pd.DataFrame(history_docs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_position_history(symbol, account_type)
    except Exception as e:
        st.error(f"Error retrieving position history: {e}")
        return generate_mock_position_history(symbol, account_type)

def get_execution_history(db, account_type=None, limit=50):
    """
    Retrieve recent trade executions from MongoDB
    Returns a DataFrame with execution details
    """
    if db is None:
        return generate_mock_executions_data(account_type)
    
    try:
        # Query filter based on account_type
        query = {}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get executions from MongoDB
        executions_docs = list(db.executions.find(query).sort("timestamp", -1).limit(limit))
        
        if executions_docs:
            # Convert to DataFrame
            df = pd.DataFrame(executions_docs)
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_executions_data(account_type)
    except Exception as e:
        st.error(f"Error retrieving execution history: {e}")
        return generate_mock_executions_data(account_type)

#############################
# Mock Data Generators
#############################

def generate_mock_positions_data(account_type=None):
    """
    Generate synthetic positions data for development and testing
    Returns a DataFrame with position details
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        # If no specific account type, randomly choose one
        account_type = random.choice(["live", "paper"])
    
    # Define asset classes and symbols
    asset_classes = {
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM"],
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "ADA/USD", "DOT/USD"],
        "commodity": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F"]
    }
    
    # Number of positions to generate (fewer for live, more for paper)
    if account_type == "live":
        num_positions = random.randint(3, 8)
        base_equity = 100000.0
    else:  # paper
        num_positions = random.randint(5, 15)
        base_equity = 100000.0
    
    positions = []
    
    # Generate random positions
    for i in range(num_positions):
        # Select random asset class and symbol
        asset_class = random.choice(list(asset_classes.keys()))
        symbol = random.choice(asset_classes[asset_class])
        
        # Determine if long or short
        side = random.choice(["long", "short"])
        
        # Generate position size based on asset class
        if asset_class == "equity":
            quantity = random.randint(10, 200)
            entry_price = round(random.uniform(50, 500), 2)
            current_price = round(entry_price * random.uniform(0.9, 1.1), 2)
        elif asset_class == "forex":
            quantity = random.randint(10000, 100000)
            entry_price = round(random.uniform(0.5, 2.0), 5)
            current_price = round(entry_price * random.uniform(0.99, 1.01), 5)
        elif asset_class == "crypto":
            quantity = round(random.uniform(0.1, 10), 4)
            entry_price = round(random.uniform(100, 50000), 2)
            current_price = round(entry_price * random.uniform(0.8, 1.2), 2)
        else:  # commodity
            quantity = random.randint(1, 10)
            entry_price = round(random.uniform(20, 2000), 2)
            current_price = round(entry_price * random.uniform(0.9, 1.1), 2)
        
        # Calculate position value
        position_value = quantity * current_price
        
        # Calculate unrealized P&L
        if side == "long":
            unrealized_pnl = quantity * (current_price - entry_price)
            unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
        else:  # short
            unrealized_pnl = quantity * (entry_price - current_price)
            unrealized_pnl_pct = ((entry_price / current_price) - 1) * 100
        
        # Set stop loss and take profit levels
        if side == "long":
            stop_loss = round(entry_price * random.uniform(0.85, 0.95), 2)
            take_profit = round(entry_price * random.uniform(1.1, 1.3), 2)
        else:  # short
            stop_loss = round(entry_price * random.uniform(1.05, 1.15), 2)
            take_profit = round(entry_price * random.uniform(0.7, 0.9), 2)
        
        # Calculate risk metrics
        risk_per_share = abs(entry_price - stop_loss)
        potential_loss = risk_per_share * quantity
        risk_reward = abs((take_profit - entry_price) / (entry_price - stop_loss))
        
        # Calculate exposure as percentage of account
        exposure_pct = (position_value / base_equity) * 100
        
        # Generate entry timestamp
        days_ago = random.randint(1, 30)
        entry_timestamp = (datetime.datetime.now() - datetime.timedelta(days=days_ago))
        
        # Create position object
        position = {
            "symbol": symbol,
            "asset_class": asset_class,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "position_value": round(position_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_per_share": round(risk_per_share, 2),
            "potential_loss": round(potential_loss, 2),
            "risk_reward": round(risk_reward, 2),
            "exposure_pct": round(exposure_pct, 2),
            "entry_timestamp": entry_timestamp,
            "days_held": days_ago,
            "account_type": account_type,
            "strategy_id": f"strat_{random.randint(1, 10)}",
            "order_id": f"ord_{random.randint(10000, 99999)}"
        }
        
        positions.append(position)
    
    # Create DataFrame
    positions_df = pd.DataFrame(positions)
    
    return positions_df

def generate_mock_orders_data(account_type=None):
    """
    Generate synthetic open/pending orders data for development and testing
    Returns a DataFrame with order details
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        # If no specific account type, randomly choose one
        account_type = random.choice(["live", "paper"])
    
    # Define asset classes and symbols
    asset_classes = {
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM"],
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "ADA/USD", "DOT/USD"],
        "commodity": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F"]
    }
    
    # Number of orders to generate (fewer for live, more for paper)
    if account_type == "live":
        num_orders = random.randint(2, 5)
    else:  # paper
        num_orders = random.randint(3, 8)
    
    orders = []
    
    # Order types and statuses
    order_types = ["market", "limit", "stop", "stop_limit"]
    order_statuses = ["open", "pending", "partial_fill"]
    time_in_force = ["day", "gtc", "ioc", "fok"]
    
    # Generate random orders
    for i in range(num_orders):
        # Select random asset class and symbol
        asset_class = random.choice(list(asset_classes.keys()))
        symbol = random.choice(asset_classes[asset_class])
        
        # Determine if buy or sell
        side = random.choice(["buy", "sell"])
        
        # Generate order details based on asset class
        if asset_class == "equity":
            quantity = random.randint(10, 200)
            current_price = round(random.uniform(50, 500), 2)
        elif asset_class == "forex":
            quantity = random.randint(10000, 100000)
            current_price = round(random.uniform(0.5, 2.0), 5)
        elif asset_class == "crypto":
            quantity = round(random.uniform(0.1, 10), 4)
            current_price = round(random.uniform(100, 50000), 2)
        else:  # commodity
            quantity = random.randint(1, 10)
            current_price = round(random.uniform(20, 2000), 2)
        
        # Select order type and generate prices accordingly
        order_type = random.choice(order_types)
        
        if order_type == "market":
            limit_price = None
            stop_price = None
        elif order_type == "limit":
            if side == "buy":
                # Buy limit is below current price
                limit_price = round(current_price * random.uniform(0.95, 0.99), 2)
            else:  # sell
                # Sell limit is above current price
                limit_price = round(current_price * random.uniform(1.01, 1.05), 2)
            stop_price = None
        elif order_type == "stop":
            if side == "buy":
                # Buy stop is above current price
                stop_price = round(current_price * random.uniform(1.01, 1.05), 2)
            else:  # sell
                # Sell stop is below current price
                stop_price = round(current_price * random.uniform(0.95, 0.99), 2)
            limit_price = None
        else:  # stop_limit
            if side == "buy":
                # Buy stop is above current price
                stop_price = round(current_price * random.uniform(1.01, 1.05), 2)
                # Buy stop-limit is slightly higher than stop
                limit_price = round(stop_price * random.uniform(1.001, 1.01), 2)
            else:  # sell
                # Sell stop is below current price
                stop_price = round(current_price * random.uniform(0.95, 0.99), 2)
                # Sell stop-limit is slightly lower than stop
                limit_price = round(stop_price * random.uniform(0.99, 0.999), 2)
        
        # Calculate order value
        order_value = quantity * (limit_price or stop_price or current_price)
        
        # Generate timestamps
        minutes_ago = random.randint(1, 1440)  # Up to 24 hours ago
        submit_timestamp = (datetime.datetime.now() - datetime.timedelta(minutes=minutes_ago))
        
        # Create order object
        order = {
            "order_id": f"ord_{random.randint(10000, 99999)}",
            "symbol": symbol,
            "asset_class": asset_class,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "current_price": current_price,
            "order_value": round(order_value, 2),
            "status": random.choice(order_statuses),
            "time_in_force": random.choice(time_in_force),
            "submit_timestamp": submit_timestamp,
            "minutes_active": minutes_ago,
            "account_type": account_type,
            "strategy_id": f"strat_{random.randint(1, 10)}"
        }
        
        orders.append(order)
    
    # Create DataFrame
    orders_df = pd.DataFrame(orders)
    
    return orders_df

def generate_mock_position_history(symbol, account_type=None):
    """
    Generate synthetic historical data for a specific position
    Returns a DataFrame with position value over time
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        account_type = random.choice(["live", "paper"])
    
    # Determine history length (days)
    days = random.randint(7, 30)
    
    # Generate timestamps
    now = datetime.datetime.now()
    timestamps = [now - datetime.timedelta(days=i) for i in range(days, 0, -1)]
    
    # Determine asset class and starting price based on symbol
    if any(currency in symbol for currency in ["/", "JPY", "GBP", "EUR", "AUD", "NZD", "CHF", "CAD"]):
        asset_class = "forex"
        starting_price = random.uniform(0.5, 2.0)
        volatility = 0.002  # 0.2% daily volatility
    elif any(crypto in symbol for crypto in ["BTC", "ETH", "XRP", "LTC", "ADA", "DOT"]):
        asset_class = "crypto"
        starting_price = random.uniform(100, 50000)
        volatility = 0.03  # 3% daily volatility
    elif any(commodity in symbol for commodity in ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F"]):
        asset_class = "commodity"
        starting_price = random.uniform(20, 2000)
        volatility = 0.01  # 1% daily volatility
    else:
        asset_class = "equity"
        starting_price = random.uniform(50, 500)
        volatility = 0.015  # 1.5% daily volatility
    
    # Determine side
    side = random.choice(["long", "short"])
    
    # Determine quantity
    if asset_class == "equity":
        quantity = random.randint(10, 200)
    elif asset_class == "forex":
        quantity = random.randint(10000, 100000)
    elif asset_class == "crypto":
        quantity = round(random.uniform(0.1, 10), 4)
    else:  # commodity
        quantity = random.randint(1, 10)
    
    # Generate price time series with random walk
    prices = [starting_price]
    for i in range(1, days):
        # Random walk with slight drift based on side
        drift = 0.0005 if side == "long" else -0.0005
        change = np.random.normal(drift, volatility) * prices[-1]
        new_price = max(0.01, prices[-1] + change)
        prices.append(new_price)
    
    # Calculate position values and P&L
    position_values = [price * quantity for price in prices]
    
    # Calculate entry price (first price)
    entry_price = prices[0]
    
    # Calculate unrealized P&L
    if side == "long":
        unrealized_pnls = [quantity * (price - entry_price) for price in prices]
        unrealized_pnl_pcts = [((price / entry_price) - 1) * 100 for price in prices]
    else:  # short
        unrealized_pnls = [quantity * (entry_price - price) for price in prices]
        unrealized_pnl_pcts = [((entry_price / price) - 1) * 100 for price in prices]
    
    # Create history records
    history = []
    
    for i in range(days):
        record = {
            "symbol": symbol,
            "asset_class": asset_class,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "price": prices[i],
            "position_value": position_values[i],
            "unrealized_pnl": unrealized_pnls[i],
            "unrealized_pnl_pct": unrealized_pnl_pcts[i],
            "timestamp": timestamps[i],
            "account_type": account_type
        }
        
        history.append(record)
    
    # Create DataFrame
    history_df = pd.DataFrame(history)
    
    # Format numeric columns
    for col in ["entry_price", "price", "position_value", "unrealized_pnl", "unrealized_pnl_pct"]:
        if col in history_df.columns:
            history_df[col] = history_df[col].apply(lambda x: round(x, 2))
    
    return history_df

def generate_mock_executions_data(account_type=None, limit=50):
    """
    Generate synthetic trade execution history
    Returns a DataFrame with execution details
    """
    # Map account_type to internal format
    if account_type and account_type != "All":
        account_type = account_type.lower().replace(" trading", "").strip()
    else:
        account_type = random.choice(["live", "paper"])
    
    # Define asset classes and symbols
    asset_classes = {
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM"],
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "ADA/USD", "DOT/USD"],
        "commodity": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F"]
    }
    
    # Number of executions to generate (up to the limit)
    num_executions = min(limit, random.randint(20, 100))
    
    executions = []
    
    # Action types
    action_types = ["buy", "sell", "buy_to_cover", "sell_short"]
    
    # Generate random executions
    for i in range(num_executions):
        # Select random asset class and symbol
        asset_class = random.choice(list(asset_classes.keys()))
        symbol = random.choice(asset_classes[asset_class])
        
        # Determine action type
        action = random.choice(action_types)
        
        # Generate execution details based on asset class
        if asset_class == "equity":
            quantity = random.randint(10, 200)
            price = round(random.uniform(50, 500), 2)
        elif asset_class == "forex":
            quantity = random.randint(10000, 100000)
            price = round(random.uniform(0.5, 2.0), 5)
        elif asset_class == "crypto":
            quantity = round(random.uniform(0.1, 10), 4)
            price = round(random.uniform(100, 50000), 2)
        else:  # commodity
            quantity = random.randint(1, 10)
            price = round(random.uniform(20, 2000), 2)
        
        # Calculate value
        value = quantity * price
        
        # Generate timestamps
        minutes_ago = random.randint(1, 10080)  # Up to 7 days ago
        timestamp = (datetime.datetime.now() - datetime.timedelta(minutes=minutes_ago))
        
        # Create execution record
        execution = {
            "execution_id": f"ex_{random.randint(10000, 99999)}",
            "order_id": f"ord_{random.randint(10000, 99999)}",
            "symbol": symbol,
            "asset_class": asset_class,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": round(value, 2),
            "timestamp": timestamp,
            "account_type": account_type,
            "strategy_id": f"strat_{random.randint(1, 10)}",
            "venue": random.choice(["NASDAQ", "NYSE", "ARCA", "BATS", "IEX", "Binance", "Coinbase", "Kraken", "FXCM", "Oanda"])
        }
        
        executions.append(execution)
    
    # Sort by timestamp (newest first)
    executions = sorted(executions, key=lambda x: x["timestamp"], reverse=True)
    
    # Create DataFrame
    executions_df = pd.DataFrame(executions)
    
    return executions_df
