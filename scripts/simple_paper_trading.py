#!/usr/bin/env python3
"""
Simplified Paper Trading Script for BensBot.

This standalone script implements a basic paper trading system using
MongoDB for persistence and minimal dependencies.
"""

import os
import sys
import time
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pymongo
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/simple_paper_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("paper_trading")

# Load environment variables
load_dotenv()

# Configuration with environment variable fallbacks
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/bensbot_trading")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "bensbot_trading")
INITIAL_BALANCE = float(os.getenv("PAPER_INITIAL_BALANCE", "100000"))
COMMISSION_RATE = float(os.getenv("PAPER_COMMISSION_RATE", "0.0005"))  # 0.05%
SLIPPAGE_FACTOR = float(os.getenv("PAPER_SLIPPAGE_FACTOR", "0.0001"))  # 0.01%
SYMBOLS = os.getenv("TRADING_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.05"))  # 5% of account per position
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL_SECONDS", "60"))  # Sync to DB every minute
USE_PERSISTENCE = os.getenv("USE_PERSISTENCE", "true").lower() == "true"


class PaperTradingEngine:
    """Simple paper trading engine with MongoDB persistence"""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        """Initialize the paper trading engine"""
        self.balance = initial_balance
        self.positions = {}  # symbol -> {quantity, avg_price, market_value, unrealized_pnl}
        self.orders = {}     # order_id -> order details
        self.trades = []     # list of executed trades
        self.next_order_id = 1
        self.start_time = datetime.now()
        
        # Market data cache
        self.market_data = {}
        self.last_market_data_update = {}
        
        # MongoDB connection if persistence is enabled
        if USE_PERSISTENCE:
            try:
                self.mongo_client = pymongo.MongoClient(MONGODB_URI)
                self.db = self.mongo_client[MONGODB_DATABASE]
                self.positions_collection = self.db["paper_positions"]
                self.orders_collection = self.db["paper_orders"]
                self.trades_collection = self.db["paper_trades"]
                self.account_collection = self.db["paper_account"]
                logger.info("Connected to MongoDB successfully")
                
                # Try to load state
                self._load_state()
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                logger.warning("Running without persistence")
                self.mongo_client = None
                self.db = None
        else:
            self.mongo_client = None
            self.db = None
    
    def _load_state(self):
        """Load state from MongoDB"""
        if not self.db:
            return
            
        try:
            # Load account balance
            account_doc = self.account_collection.find_one({"account_type": "paper"})
            if account_doc:
                self.balance = account_doc.get("balance", self.balance)
                self.start_time = account_doc.get("start_time", self.start_time)
                logger.info(f"Loaded account balance: ${self.balance:.2f}")
            
            # Load positions
            position_docs = self.positions_collection.find({"account_type": "paper"})
            for doc in position_docs:
                symbol = doc.get("symbol")
                if symbol:
                    self.positions[symbol] = {
                        "quantity": doc.get("quantity", 0),
                        "avg_price": doc.get("avg_price", 0),
                        "market_value": doc.get("market_value", 0),
                        "unrealized_pnl": doc.get("unrealized_pnl", 0),
                    }
            logger.info(f"Loaded {len(self.positions)} positions")
            
            # Load orders - only open ones
            order_docs = self.orders_collection.find({"account_type": "paper", "status": "open"})
            for doc in order_docs:
                order_id = doc.get("order_id")
                if order_id:
                    self.orders[order_id] = doc
                    # Update next_order_id to be higher than any existing orders
                    try:
                        order_id_int = int(order_id.replace("paper_", ""))
                        self.next_order_id = max(self.next_order_id, order_id_int + 1)
                    except:
                        pass
            logger.info(f"Loaded {len(self.orders)} open orders")
            
        except Exception as e:
            logger.error(f"Error loading state from MongoDB: {e}")
    
    def _save_state(self):
        """Save state to MongoDB"""
        if not self.db:
            return
            
        try:
            # Save account balance
            self.account_collection.update_one(
                {"account_type": "paper"},
                {"$set": {
                    "balance": self.balance,
                    "start_time": self.start_time,
                    "equity": self.get_total_equity(),
                    "last_updated": datetime.now()
                }},
                upsert=True
            )
            
            # Save positions
            for symbol, position in self.positions.items():
                self.positions_collection.update_one(
                    {"account_type": "paper", "symbol": symbol},
                    {"$set": {
                        "quantity": position["quantity"],
                        "avg_price": position["avg_price"],
                        "market_value": position["market_value"],
                        "unrealized_pnl": position["unrealized_pnl"],
                        "last_updated": datetime.now()
                    }},
                    upsert=True
                )
            
            # Save orders - store all for history
            for order_id, order in self.orders.items():
                order_doc = order.copy()
                order_doc["last_updated"] = datetime.now()
                self.orders_collection.update_one(
                    {"account_type": "paper", "order_id": order_id},
                    {"$set": order_doc},
                    upsert=True
                )
                
            logger.debug("State saved to MongoDB")
            
        except Exception as e:
            logger.error(f"Error saving state to MongoDB: {e}")
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol using yfinance"""
        now = datetime.now()
        
        # Return cached data if less than 30 seconds old
        if symbol in self.last_market_data_update:
            last_update = self.last_market_data_update[symbol]
            if (now - last_update).total_seconds() < 30 and symbol in self.market_data:
                return self.market_data[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if len(data) > 0:
                # Use Close for most recent price
                price = data.iloc[-1]['Close']
                self.market_data[symbol] = price
                self.last_market_data_update[symbol] = now
                return price
            else:
                logger.warning(f"No market data found for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def get_total_equity(self) -> float:
        """Calculate total account equity (cash + position market values)"""
        total = self.balance
        
        # Update position market values
        for symbol, position in list(self.positions.items()):
            price = self.get_market_price(symbol)
            if price and position["quantity"] != 0:
                market_value = position["quantity"] * price
                unrealized_pnl = market_value - (position["quantity"] * position["avg_price"])
                position["market_value"] = market_value
                position["unrealized_pnl"] = unrealized_pnl
                total += market_value
            elif position["quantity"] == 0:
                # Remove zero quantity positions
                del self.positions[symbol]
        
        return total
    
    def place_order(self, symbol: str, quantity: float, side: str, order_type: str = "market", 
                   limit_price: Optional[float] = None) -> Dict[str, Any]:
        """Place a paper trade order"""
        # Generate order ID
        order_id = f"paper_{self.next_order_id}"
        self.next_order_id += 1
        
        # Get current price
        current_price = self.get_market_price(symbol)
        if not current_price:
            logger.error(f"Could not get price for {symbol}")
            return {"status": "rejected", "reason": "No market data available", "order_id": order_id}
        
        # For limit orders, check if price is acceptable
        execution_price = current_price
        if order_type == "limit":
            if not limit_price:
                return {"status": "rejected", "reason": "Limit order requires a price", "order_id": order_id}
            
            if side == "buy" and current_price > limit_price:
                # Price is above limit for a buy order
                order = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "order_type": order_type,
                    "limit_price": limit_price,
                    "status": "open",
                    "created_at": datetime.now(),
                    "account_type": "paper"
                }
                self.orders[order_id] = order
                logger.info(f"Placed limit buy order {order_id} for {quantity} {symbol} @ ${limit_price:.2f}")
                return {"status": "accepted", "order_id": order_id}
                
            elif side == "sell" and current_price < limit_price:
                # Price is below limit for a sell order
                order = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "order_type": order_type,
                    "limit_price": limit_price,
                    "status": "open",
                    "created_at": datetime.now(),
                    "account_type": "paper"
                }
                self.orders[order_id] = order
                logger.info(f"Placed limit sell order {order_id} for {quantity} {symbol} @ ${limit_price:.2f}")
                return {"status": "accepted", "order_id": order_id}
            
            # For limit orders within execution range, use the limit price
            execution_price = limit_price
        
        # Apply slippage
        slippage = execution_price * SLIPPAGE_FACTOR
        if side == "buy":
            execution_price += slippage  # Pay more when buying
        else:
            execution_price -= slippage  # Get less when selling
        
        # Calculate commission
        commission = execution_price * quantity * COMMISSION_RATE
        total_cost = (execution_price * quantity) + commission if side == "buy" else (execution_price * quantity) - commission
        
        # Check if we have enough funds for buy orders
        if side == "buy" and total_cost > self.balance:
            return {"status": "rejected", "reason": "Insufficient funds", "order_id": order_id}
        
        # Check if we have enough shares for sell orders
        if side == "sell":
            position = self.positions.get(symbol, {"quantity": 0})
            if position["quantity"] < quantity:
                return {"status": "rejected", "reason": "Insufficient shares", "order_id": order_id}
        
        # Execute the trade
        if side == "buy":
            # Update balance
            self.balance -= total_cost
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": execution_price,
                    "market_value": execution_price * quantity,
                    "unrealized_pnl": 0
                }
            else:
                # Calculate new average price
                current_qty = self.positions[symbol]["quantity"]
                current_avg = self.positions[symbol]["avg_price"]
                new_qty = current_qty + quantity
                new_avg = ((current_qty * current_avg) + (quantity * execution_price)) / new_qty
                
                self.positions[symbol]["quantity"] = new_qty
                self.positions[symbol]["avg_price"] = new_avg
                self.positions[symbol]["market_value"] = new_qty * execution_price
                self.positions[symbol]["unrealized_pnl"] = 0  # Reset PNL on new purchases
        else:
            # Sell order
            # Calculate realized profit/loss
            avg_price = self.positions[symbol]["avg_price"]
            realized_pnl = (execution_price - avg_price) * quantity
            
            # Update balance with sale proceeds
            self.balance += total_cost
            
            # Update position
            self.positions[symbol]["quantity"] -= quantity
            # Keep average price the same
            self.positions[symbol]["market_value"] = self.positions[symbol]["quantity"] * execution_price
            # Recalculate unrealized PNL
            self.positions[symbol]["unrealized_pnl"] = (execution_price - avg_price) * self.positions[symbol]["quantity"]
        
        # Record the trade
        trade = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "execution_price": execution_price,
            "commission": commission,
            "timestamp": datetime.now(),
            "account_type": "paper",
            "realized_pnl": realized_pnl if side == "sell" else 0
        }
        self.trades.append(trade)
        
        # If using MongoDB, save the trade
        if self.db:
            try:
                self.trades_collection.insert_one(trade)
            except Exception as e:
                logger.error(f"Error saving trade to MongoDB: {e}")
        
        # Create filled order record
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "execution_price": execution_price,
            "status": "filled",
            "commission": commission,
            "created_at": datetime.now(),
            "filled_at": datetime.now(),
            "account_type": "paper"
        }
        self.orders[order_id] = order
        
        logger.info(f"Executed {side} order for {quantity} {symbol} @ ${execution_price:.2f}")
        return {"status": "filled", "order_id": order_id, "execution_price": execution_price}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order"""
        if order_id not in self.orders:
            return {"status": "rejected", "reason": "Order not found"}
            
        order = self.orders[order_id]
        if order["status"] != "open":
            return {"status": "rejected", "reason": f"Order is already {order['status']}"}
        
        # Update order status
        order["status"] = "cancelled"
        order["cancelled_at"] = datetime.now()
        
        logger.info(f"Cancelled order {order_id}")
        return {"status": "cancelled", "order_id": order_id}
    
    def process_open_orders(self):
        """Process any open limit orders"""
        for order_id, order in list(self.orders.items()):
            if order["status"] != "open":
                continue
                
            symbol = order["symbol"]
            current_price = self.get_market_price(symbol)
            if not current_price:
                continue
                
            # Check if limit order can be executed
            if order["order_type"] == "limit":
                limit_price = order["limit_price"]
                if order["side"] == "buy" and current_price <= limit_price:
                    # Execute buy limit order
                    self.execute_limit_order(order_id)
                elif order["side"] == "sell" and current_price >= limit_price:
                    # Execute sell limit order
                    self.execute_limit_order(order_id)
    
    def execute_limit_order(self, order_id: str):
        """Execute a limit order that has reached its price"""
        order = self.orders[order_id]
        symbol = order["symbol"]
        quantity = order["quantity"]
        side = order["side"]
        limit_price = order["limit_price"]
        
        # Apply slippage (usually less for limit orders, but still some)
        slippage = limit_price * (SLIPPAGE_FACTOR / 2)  # Half the slippage of market orders
        execution_price = limit_price - slippage if side == "sell" else limit_price + slippage
        
        # Calculate commission
        commission = execution_price * quantity * COMMISSION_RATE
        total_cost = (execution_price * quantity) + commission if side == "buy" else (execution_price * quantity) - commission
        
        # Check sufficient funds/shares
        if side == "buy" and total_cost > self.balance:
            logger.warning(f"Insufficient funds to execute limit buy order {order_id}")
            return
            
        if side == "sell":
            position = self.positions.get(symbol, {"quantity": 0})
            if position["quantity"] < quantity:
                logger.warning(f"Insufficient shares to execute limit sell order {order_id}")
                return
        
        # Execute the trade
        realized_pnl = 0
        if side == "buy":
            # Update balance
            self.balance -= total_cost
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": execution_price,
                    "market_value": execution_price * quantity,
                    "unrealized_pnl": 0
                }
            else:
                # Calculate new average price
                current_qty = self.positions[symbol]["quantity"]
                current_avg = self.positions[symbol]["avg_price"]
                new_qty = current_qty + quantity
                new_avg = ((current_qty * current_avg) + (quantity * execution_price)) / new_qty
                
                self.positions[symbol]["quantity"] = new_qty
                self.positions[symbol]["avg_price"] = new_avg
                self.positions[symbol]["market_value"] = new_qty * execution_price
        else:
            # Sell order
            # Calculate realized profit/loss
            avg_price = self.positions[symbol]["avg_price"]
            realized_pnl = (execution_price - avg_price) * quantity
            
            # Update balance with sale proceeds
            self.balance += total_cost
            
            # Update position
            self.positions[symbol]["quantity"] -= quantity
            # Keep average price the same
            self.positions[symbol]["market_value"] = self.positions[symbol]["quantity"] * execution_price
            # Recalculate unrealized PNL
            self.positions[symbol]["unrealized_pnl"] = (execution_price - avg_price) * self.positions[symbol]["quantity"]
        
        # Update order status
        order["status"] = "filled"
        order["execution_price"] = execution_price
        order["commission"] = commission
        order["filled_at"] = datetime.now()
        
        # Record the trade
        trade = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "execution_price": execution_price,
            "commission": commission,
            "timestamp": datetime.now(),
            "account_type": "paper",
            "realized_pnl": realized_pnl
        }
        self.trades.append(trade)
        
        # If using MongoDB, save the trade
        if self.db:
            try:
                self.trades_collection.insert_one(trade)
            except Exception as e:
                logger.error(f"Error saving trade to MongoDB: {e}")
        
        logger.info(f"Executed limit {side} order {order_id} for {quantity} {symbol} @ ${execution_price:.2f}")
    
    def update_positions(self):
        """Update position market values and unrealized PNL"""
        total_value = 0
        
        for symbol, position in list(self.positions.items()):
            if position["quantity"] == 0:
                # Remove zero positions
                del self.positions[symbol]
                continue
                
            price = self.get_market_price(symbol)
            if price:
                market_value = position["quantity"] * price
                unrealized_pnl = market_value - (position["quantity"] * position["avg_price"])
                position["market_value"] = market_value
                position["unrealized_pnl"] = unrealized_pnl
                position["current_price"] = price
                total_value += market_value
        
        return total_value
    
    def generate_order_book_report(self) -> str:
        """Generate a simple report of filled orders"""
        if not self.trades:
            return "No trades executed yet."
            
        report = "Order Book:\n"
        report += "-" * 80 + "\n"
        report += f"{'Order ID':<15} {'Symbol':<8} {'Side':<5} {'Quantity':<10} {'Price':<10} {'Status':<10} {'Timestamp':<20}\n"
        report += "-" * 80 + "\n"
        
        # Sort by timestamp (most recent first)
        sorted_orders = sorted(
            self.orders.values(), 
            key=lambda x: x.get("filled_at", x.get("cancelled_at", x.get("created_at"))),
            reverse=True
        )
        
        for order in sorted_orders[:15]:  # Show 15 most recent
            order_id = order["order_id"]
            symbol = order["symbol"]
            side = order["side"]
            quantity = order["quantity"]
            price = order.get("execution_price", order.get("limit_price", 0))
            status = order["status"]
            timestamp = order.get("filled_at", order.get("cancelled_at", order.get("created_at")))
            
            report += f"{order_id:<15} {symbol:<8} {side:<5} {quantity:<10.2f} ${price:<9.2f} {status:<10} {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        report += "-" * 80 + "\n"
        return report
    
    def generate_portfolio_report(self) -> str:
        """Generate a simple portfolio report"""
        # Update positions first
        self.update_positions()
        
        # Calculate total equity
        cash = self.balance
        securities_value = sum(pos["market_value"] for pos in self.positions.values())
        total_equity = cash + securities_value
        
        # Calculate performance
        start_balance = INITIAL_BALANCE
        total_pnl = total_equity - start_balance
        pnl_pct = (total_pnl / start_balance) * 100
        
        report = "Portfolio Summary:\n"
        report += "-" * 80 + "\n"
        report += f"Cash Balance: ${cash:.2f}\n"
        report += f"Securities Value: ${securities_value:.2f}\n"
        report += f"Total Equity: ${total_equity:.2f}\n"
        report += f"Total P&L: ${total_pnl:.2f} ({pnl_pct:.2f}%)\n"
        report += "-" * 80 + "\n"
        report += "Positions:\n"
        report += f"{'Symbol':<8} {'Quantity':<10} {'Avg Price':<12} {'Current':<10} {'Market Value':<15} {'Unrealized P&L':<15}\n"
        report += "-" * 80 + "\n"
        
        for symbol, position in self.positions.items():
            quantity = position["quantity"]
            avg_price = position["avg_price"]
            current_price = position.get("current_price", self.get_market_price(symbol) or 0)
            market_value = position["market_value"]
            unrealized_pnl = position["unrealized_pnl"]
            
            report += f"{symbol:<8} {quantity:<10.2f} ${avg_price:<11.2f} ${current_price:<9.2f} ${market_value:<14.2f} ${unrealized_pnl:<14.2f}\n"
        
        report += "-" * 80 + "\n"
        return report


def run_paper_trading_simulation():
    """Run a paper trading simulation"""
    # Create the paper trading engine
    engine = PaperTradingEngine()
    logger.info(f"Started paper trading with ${INITIAL_BALANCE:.2f}")
    
    # Track the last sync time for DB persistence
    last_sync_time = datetime.now()
    last_report_time = datetime.now()
    
    try:
        while True:
            # Process any open orders
            engine.process_open_orders()
            
            # Update position values
            engine.update_positions()
            
            # Simple simulation - randomly place orders
            if random.random() < 0.05:  # 5% chance each loop
                # Pick a random symbol
                symbol = random.choice(SYMBOLS)
                
                # Decide buy or sell
                side = random.choice(["buy", "sell"])
                
                # For sells, we need to own the stock
                if side == "sell" and (symbol not in engine.positions or engine.positions[symbol]["quantity"] <= 0):
                    # Skip this iteration
                    pass
                else:
                    # Calculate quantity
                    if side == "buy":
                        # Get current price
                        price = engine.get_market_price(symbol)
                        if price:
                            # Risk maximum 5% of account on any trade
                            max_trade_value = engine.balance * MAX_POSITION_SIZE_PCT
                            # Calculate max shares we can buy
                            max_shares = max_trade_value / price
                            # Use a random fraction of max shares
                            quantity = max_shares * random.uniform(0.3, 1.0)
                            # Round to 2 decimals
                            quantity = round(quantity, 2)
                            
                            # Place the order
                            if quantity > 0:
                                logger.info(f"Placing {side} order for {quantity} shares of {symbol}")
                                result = engine.place_order(symbol, quantity, side)
                                logger.info(f"Order result: {result}")
                    else:
                        # For sells, use a random fraction of current position
                        current_quantity = engine.positions[symbol]["quantity"]
                        quantity = current_quantity * random.uniform(0.2, 0.7)
                        quantity = round(quantity, 2)
                        
                        if quantity > 0:
                            logger.info(f"Placing {side} order for {quantity} shares of {symbol}")
                            result = engine.place_order(symbol, quantity, side)
                            logger.info(f"Order result: {result}")
            
            # Sync to database if interval has passed
            now = datetime.now()
            if (now - last_sync_time).total_seconds() >= SYNC_INTERVAL and USE_PERSISTENCE:
                engine._save_state()
                last_sync_time = now
            
            # Show a report every 5 minutes
            if (now - last_report_time).total_seconds() >= 300:
                logger.info(engine.generate_portfolio_report())
                logger.info(engine.generate_order_book_report())
                last_report_time = now
            
            # Sleep to avoid excessive API calls and CPU usage
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Paper trading simulation stopped by user")
    except Exception as e:
        logger.exception(f"Error in paper trading simulation: {e}")
    finally:
        # Final sync to database
        if USE_PERSISTENCE:
            engine._save_state()
        
        # Print final reports
        logger.info("Final portfolio:")
        logger.info(engine.generate_portfolio_report())
        logger.info(engine.generate_order_book_report())


if __name__ == "__main__":
    # Make sure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/trades", exist_ok=True)
    
    # Start the simulation
    run_paper_trading_simulation()
