from flask import Flask, request, jsonify, g, make_response
import logging
import json
import os
import uuid
import datetime
import functools
from typing import Dict, Any, List, Optional, Union

# Import executor components
from trading_bot.strategy_loader import StrategyLoader
from trading_bot.trade_executor import TradeExecutor, TradeResult, TradeType, OrderSide, OrderType
from trading_bot.risk.psychological_risk import PsychologicalRiskManager, PositionSizer
from trading_bot.cooldown_manager import CooldownManager
from trading_bot.strategy_monitor import StrategyMonitor

# Add import for TradeJournal
from trade_journal import TradeJournal, TradeEntry

# Import trading optimizers
from trading_bot.trading_optimizers import (
    StrategyRotator,
    ConfidenceScorer,
    TradeReportGenerator,
    ReplayEngine
)

# Import market context analyzer 
from trading_bot.market_context.context_analyzer import MarketContextAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("WebhookReceiver")

# Initialize the app
app = Flask(__name__)

# API Authentication configuration
API_TOKENS = {}  # Will store token: username pairs
API_AUTH_ENABLED = os.environ.get("API_AUTH_ENABLED", "false").lower() == "true"

# Create market context analyzer
market_context = MarketContextAnalyzer({
    "MARKETAUX_API_KEY": os.environ.get("MARKETAUX_API_KEY", "7PgROm6BE4m6ejBW8unmZnnYS6kIygu5lwzpfd9K"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "CACHE_EXPIRY_MINUTES": 30
})

# Load API tokens from environment or file
def load_api_tokens():
    """Load API tokens from environment variable or tokens file"""
    global API_TOKENS
    
    # Try to load from environment variable
    token_str = os.environ.get("API_TOKENS", "")
    if token_str:
        try:
            # Format should be "token1:user1,token2:user2"
            pairs = token_str.split(",")
            for pair in pairs:
                if ":" in pair:
                    token, user = pair.split(":", 1)
                    API_TOKENS[token.strip()] = user.strip()
            logger.info(f"Loaded {len(API_TOKENS)} API tokens from environment")
            return
        except Exception as e:
            logger.error(f"Failed to parse API_TOKENS environment variable: {e}")
    
    # Try to load from file
    tokens_file = os.environ.get("API_TOKENS_FILE", "api_tokens.json")
    if os.path.exists(tokens_file):
        try:
            with open(tokens_file, 'r') as f:
                API_TOKENS = json.load(f)
            logger.info(f"Loaded {len(API_TOKENS)} API tokens from {tokens_file}")
        except Exception as e:
            logger.error(f"Failed to load API tokens from file {tokens_file}: {e}")
    
    # If we still don't have any tokens and auth is enabled, create a default one
    if API_AUTH_ENABLED and not API_TOKENS:
        default_token = str(uuid.uuid4())
        API_TOKENS[default_token] = "default_user"
        logger.warning(f"No API tokens found, created default token: {default_token}")
        logger.warning("This token will be regenerated if the server restarts.")
        logger.warning("Set API_TOKENS environment variable or create api_tokens.json file for persistent tokens.")

# Authentication decorator
def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not API_AUTH_ENABLED:
            return f(*args, **kwargs)
        
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Missing or invalid Authorization header"}), 401
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        if token not in API_TOKENS:
            return jsonify({"status": "error", "message": "Invalid API token"}), 401
        
        # Store username in Flask's g object for logging
        g.username = API_TOKENS[token]
        return f(*args, **kwargs)
    return decorated

# Default account size
DEFAULT_ACCOUNT_SIZE = 100000.0

# Initialize components
loader = StrategyLoader()
loader.load_all()

# Create trade executor
executor = TradeExecutor(
    loader=loader,
    account_id="main_account",
    api_key=os.environ.get("BROKER_API_KEY", "demo"),
    api_secret=os.environ.get("BROKER_API_SECRET", "demo"),
    paper_trading=(os.environ.get("PAPER_TRADING", "true").lower() == "true")
)

# Create psychological risk manager
psych_risk_manager = PsychologicalRiskManager(
    consecutive_loss_threshold=3,
    max_trades_per_lookback=5,
    lookback_days=7,
    journal_dir="journal"
)

# Create position sizer
position_sizer = PositionSizer(
    account_size=DEFAULT_ACCOUNT_SIZE,
    max_risk_percent=1.0,
    min_risk_percent=0.25,
    max_position_percent=5.0
)

# Initialize risk management systems
cooldown_manager = CooldownManager(config_path="cooldown_triggers.yaml")
cooldown_manager.link_position_sizer(position_sizer)
cooldown_manager.integrate_with_position_sizer()

# Initialize strategy monitoring system
strategy_monitor = StrategyMonitor(config_path="strategy_config.json")
strategy_monitor.link_cooldown_manager(cooldown_manager)

# Initialize trade journal after other components
trade_journal = TradeJournal(journal_dir="journal")

# Initialize optimizer components
strategy_rotator = StrategyRotator(
    strategies=["trend_following", "mean_reversion", "breakout", "momentum", "volatility"],
    initial_capital=100000.0,
    minimum_allocation=0.05
)

confidence_scorer = ConfidenceScorer()
report_generator = TradeReportGenerator(output_dir="reports")
replay_engine = ReplayEngine()

# Load API tokens on startup
load_api_tokens()

# Test data
trades = []
open_trades = {}

# Enable CORS for development and dashboard use
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to enable cross-origin requests from the dashboard"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

@app.route('/webhook', methods=['POST'])
@require_auth
def webhook_handler():
    """
    Handle incoming webhook requests with trade signals.
    
    Expected JSON format:
    {
        "action": "entry" or "exit",
        "symbol": "AAPL",
        "strategy": "MeanReversion",
        "strategy_id": "mr_v1",
        "trade_type": "equity",
        "entry_price": 150.25,        # For entry actions
        "stop_price": 148.50,         # For entry actions
        "target_price": 155.00,       # Optional for entry actions
        "risk_percent": 1.0,          # Optional for entry actions
        "order_type": "market",       # Optional, defaults to market
        "order_details": {},          # Optional additional order details
        "trade_id": "123456"          # Required for exit actions
    }
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    
    data = request.json
    logger.info(f"Received webhook: {data}")
    
    # Validate required fields
    if "action" not in data:
        return jsonify({"status": "error", "message": "Missing required field: action"}), 400
    
    # Process based on action type
    if data["action"] == "entry":
        return process_entry_signal(data)
    elif data["action"] == "exit":
        return process_exit_signal(data)
    else:
        return jsonify({
            "status": "error", 
            "message": f"Invalid action: {data['action']}"
        }), 400

def process_entry_signal(data: Dict[str, Any]):
    """Process an entry signal and execute trade if appropriate."""
    # Validate required fields
    required_fields = ["symbol", "strategy", "entry_price", "stop_price"]
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({
            "status": "error", 
            "message": f"Missing required fields: {', '.join(missing_fields)}"
        }), 400
    
    try:
        # Generate trade ID if not provided
        trade_id = data.get("trade_id", str(uuid.uuid4()))
        
        # Get optional fields with defaults
        symbol = data["symbol"]
        strategy_id = data.get("strategy_id", data["strategy"])
        trade_type = data.get("trade_type", "equity")
        order_type = data.get("order_type", "market")
        risk_percent = data.get("risk_percent")
        target_price = data.get("target_price")
        manual_responses = data.get("psychological_checklist", {})
        
        # ========== Get market context for this symbol ==========
        try:
            logger.info(f"Getting market context for {symbol}")
            symbol_context = market_context.get_market_context(focus_symbols=[symbol])
            
            # Log context info
            logger.info(f"Market bias for {symbol}: {symbol_context.get('bias', 'unknown')} "
                       f"(confidence: {symbol_context.get('confidence', 0):.2f})")
            
            # Check if strongly bearish context
            if symbol_context.get('bias') == 'bearish' and symbol_context.get('confidence', 0) > 0.7:
                logger.warning(f"Considering skipping {symbol} entry due to strongly bearish market context")
                
                # Reduce risk if bearish but don't skip completely
                if risk_percent:
                    risk_percent = risk_percent * 0.5
                    logger.info(f"Reduced risk from {data.get('risk_percent')} to {risk_percent} due to bearish market")
                
                # Add context warning
                data["context_warning"] = "Trade taken in bearish market conditions"
            
            # Add market context to the trade data
            data["market_context"] = {
                "bias": symbol_context.get('bias', 'unknown'),
                "confidence": symbol_context.get('confidence', 0),
                "summary": symbol_context.get('reasoning', ''),
                "triggers": symbol_context.get('triggers', [])
            }
            
        except Exception as context_error:
            logger.error(f"Error getting market context for {symbol}: {str(context_error)}")
            # Continue with the trade even if context analysis fails
        # ========================================================
        
        # Assess psychological risk
        psych_assessment = psych_risk_manager.evaluate_psych_risk({
            "symbol": data["symbol"],
            "strategy": data["strategy"],
            "manual_responses": manual_responses
        })
        
        logger.info(f"Psychological assessment for {data['symbol']}: "
                   f"{psych_assessment['risk_level']} risk "
                   f"(score: {psych_assessment['risk_score']:.1f})")
        
        # Calculate position size based on risk parameters
        position_sizing = position_sizer.calculate_position_size(
            entry_price=float(data["entry_price"]),
            stop_price=float(data["stop_price"]),
            risk_percent=risk_percent
        )
        
        # Adjust position size based on psychological assessment if needed
        if psych_assessment["risk_level"] in ["moderate", "high", "extreme"]:
            position_sizing = position_sizer.adjust_for_psych_risk(
                position_sizing,
                psych_assessment["risk_score"]
            )
            logger.info(f"Adjusted position size from {position_sizing.get('original_size')} to "
                       f"{position_sizing['position_size']} due to {psych_assessment['risk_level']} risk")
        
        # Further adjust position size based on market context if available
        if "market_context" in data:
            context = data["market_context"]
            if context.get("bias") == "bearish" and context.get("confidence", 0) > 0.5:
                original_size = position_sizing["position_size"]
                context_adjustment = max(0.7, 1.0 - context.get("confidence", 0))  # 0.7 to 0.3 reduction based on confidence
                position_sizing["position_size"] = position_sizing["position_size"] * context_adjustment
                position_sizing["context_adjusted"] = True
                logger.info(f"Adjusted position size from {original_size} to "
                           f"{position_sizing['position_size']} due to bearish market context")
            elif context.get("bias") == "bullish" and context.get("confidence", 0) > 0.7:
                # Optionally increase position size in strongly bullish context
                pass
        
        # Skip trade if extreme psychological risk
        if psych_assessment["recommendation"] == "avoid":
            logger.warning(f"Skipping trade due to extreme psychological risk: {psych_assessment['factors']}")
            return jsonify({
                "status": "skipped",
                "trade_id": trade_id,
                "message": "Trade skipped due to extreme psychological risk",
                "psychological_assessment": psych_assessment,
                "market_context": data.get("market_context")
            }), 200
        
        # Execute trade if we have a valid position size
        if position_sizing["position_size"] <= 0:
            return jsonify({
                "status": "error", 
                "message": f"Invalid position size: {position_sizing.get('error', 'unknown error')}"
            }), 400
        
        # Prepare order details
        order_details = {
            "trade_id": trade_id,
            "symbol": data["symbol"],
            "strategy": data["strategy"],
            "strategy_id": strategy_id,
            "quantity": position_sizing["position_size"],
            "order_type": OrderType(order_type),
            "side": OrderSide.BUY,
            "price": float(data["entry_price"]),
            "stop_price": float(data["stop_price"]),
            "target_price": float(target_price) if target_price else None,
            "trade_type": TradeType(trade_type)
        }
        
        # Add any additional order details
        if "order_details" in data and isinstance(data["order_details"], dict):
            order_details.update(data["order_details"])
        
        # Execute the trade
        result = executor.execute_trade(order_details)
        
        # Record trade in psychological risk manager's history
        trade_record = {
            "trade_id": trade_id,
            "symbol": data["symbol"],
            "strategy": data["strategy"],
            "position_size": position_sizing["position_size"],
            "entry_price": float(data["entry_price"]),
            "stop_price": float(data["stop_price"]),
            "target_price": float(target_price) if target_price else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "psychological_assessment": psych_assessment,
            "market_context": data.get("market_context"),
            "status": "open"
        }
        
        psych_risk_manager.update_trading_history(trade_record)
        
        return jsonify({
            "status": "success" if result.success else "error",
            "trade_id": trade_id,
            "message": "Trade executed successfully" if result.success else f"Trade failed: {result.error_message}",
            "execution_details": result.to_dict(),
            "position_sizing": position_sizing,
            "psychological_assessment": psych_assessment,
            "market_context": data.get("market_context")
        }), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Error processing entry signal: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error", 
            "message": f"Error processing entry signal: {str(e)}"
        }), 500

def process_exit_signal(data: Dict[str, Any]):
    """Process an exit signal and execute the exit if appropriate."""
    # Validate required fields
    if "trade_id" not in data:
        return jsonify({
            "status": "error",
            "message": "Missing required field: trade_id"
        }), 400
    
    if "symbol" not in data:
        return jsonify({
            "status": "error", 
            "message": "Missing required field: symbol"
        }), 400
    
    try:
        trade_id = data["trade_id"]
        symbol = data["symbol"]
        exit_price = data.get("exit_price")
        
        # Get order type (default to market)
        order_type = data.get("order_type", "market")
        
        # Prepare exit order
        exit_order = {
            "trade_id": trade_id,
            "symbol": symbol,
            "order_type": OrderType(order_type),
            "side": OrderSide.SELL,
            "price": float(exit_price) if exit_price else None
        }
        
        # Add any additional order details
        if "order_details" in data and isinstance(data["order_details"], dict):
            exit_order.update(data["order_details"])
        
        # Execute the exit
        result = executor.exit_trade(exit_order)
        
        # If successful, update the trade in psychological risk manager's history
        if result.success:
            # Calculate profit/loss
            try:
                # Try to get the trade from history
                trade_history = [t for t in psych_risk_manager.trading_history 
                                if t.get("trade_id") == trade_id]
                
                if trade_history:
                    trade = trade_history[0]
                    entry_price = trade.get("entry_price", 0)
                    position_size = trade.get("position_size", 0)
                    
                    # Calculate profit/loss
                    if entry_price and position_size and exit_price:
                        profit_loss = (float(exit_price) - entry_price) * position_size
                        
                        # Update trade record
                        trade_update = trade.copy()
                        trade_update.update({
                            "exit_price": float(exit_price),
                            "exit_timestamp": datetime.datetime.now().isoformat(),
                            "profit_loss": profit_loss,
                            "result": "win" if profit_loss > 0 else "loss",
                            "status": "closed"
                        })
                        
                        psych_risk_manager.update_trading_history(trade_update)
                        
                        logger.info(f"Updated trade {trade_id} with P/L: ${profit_loss:.2f}")
            except Exception as calc_error:
                logger.error(f"Error calculating P/L for trade {trade_id}: {str(calc_error)}")
        
        return jsonify({
            "status": "success" if result.success else "error",
            "trade_id": trade_id,
            "message": "Exit executed successfully" if result.success else f"Exit failed: {result.error_message}",
            "execution_details": result.to_dict()
        }), 200 if result.success else 400
        
    except Exception as e:
        logger.error(f"Error processing exit signal: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error processing exit signal: {str(e)}"
        }), 500

@app.route('/psychological-assessment', methods=['POST'])
def get_psychological_assessment():
    """Get psychological risk assessment."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    
    data = request.json
    
    # Check required fields
    if "symbol" not in data:
        return jsonify({"status": "error", "message": "Missing required field: symbol"}), 400
    
    try:
        assessment = psych_risk_manager.evaluate_psych_risk(data)
        return jsonify({
            "status": "success",
            "assessment": assessment
        }), 200
        
    except Exception as e:
        logger.error(f"Error in psychological assessment: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error", 
            "message": f"Error in psychological assessment: {str(e)}"
        }), 500

@app.route('/checklist-questions', methods=['GET'])
def get_checklist_questions():
    """Get psychological checklist questions."""
    try:
        questions = psych_risk_manager.get_checklist_questions()
        return jsonify({
            "status": "success",
            "questions": questions
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting checklist questions: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error getting checklist questions: {str(e)}"
        }), 500

@app.route('/open-trades', methods=['GET'])
@require_auth
def get_open_trades():
    """Get list of currently open trades with enhanced information."""
    try:
        open_trades_list = get_open_trades_list()
        
        return jsonify({
            "status": "success",
            "open_trades": open_trades_list,
            "count": len(open_trades_list),
            "timestamp": datetime.datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting open trades: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error getting open trades: {str(e)}"
        }), 500

@app.route('/update-account-size', methods=['POST'])
def update_account_size():
    """Update account size for position sizing."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    
    data = request.json
    
    # Check required fields
    if "account_size" not in data:
        return jsonify({"status": "error", "message": "Missing required field: account_size"}), 400
    
    try:
        new_size = float(data["account_size"])
        if new_size <= 0:
            return jsonify({"status": "error", "message": "Account size must be positive"}), 400
        
        position_sizer.update_account_size(new_size)
        
        return jsonify({
            "status": "success",
            "message": f"Account size updated to ${new_size:,.2f}"
        }), 200
        
    except ValueError:
        return jsonify({"status": "error", "message": "Account size must be a number"}), 400
    except Exception as e:
        logger.error(f"Error updating account size: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error updating account size: {str(e)}"
        }), 500

@app.route('/trade-pattern-analysis', methods=['GET'])
def get_trade_pattern_analysis():
    """Get analysis of trading patterns."""
    try:
        analysis = psych_risk_manager.analyze_trade_pattern()
        return jsonify({
            "status": "success",
            "analysis": analysis
        }), 200
        
    except Exception as e:
        logger.error(f"Error in trade pattern analysis: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error", 
            "message": f"Error in trade pattern analysis: {str(e)}"
        }), 500

@app.route('/recent-statistics', methods=['GET'])
def get_recent_statistics():
    """Get recent trading statistics."""
    try:
        stats = psych_risk_manager.get_recent_statistics()
        return jsonify({
            "status": "success",
            "statistics": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting recent statistics: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error getting recent statistics: {str(e)}"
        }), 500

@app.route('/cooldown_status', methods=['GET'])
def get_cooldown_status():
    """Get current cooldown status and restrictions."""
    try:
        restrictions = cooldown_manager.get_trade_restrictions()
        return jsonify({
            "status": "success",
            "trading_allowed": cooldown_manager.is_trading_allowed(),
            "cooldown_level": str(cooldown_manager.current_level),
            "active_cooldowns": len(cooldown_manager.active_cooldowns),
            "position_sizing": cooldown_manager.get_position_size_adjustment(),
            "restrictions": restrictions
        })
    except Exception as e:
        logger.error(f"Error retrieving cooldown status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/strategy_status', methods=['GET'])
@require_auth
def get_all_strategies():
    """Get status of all trading strategies with enhanced information."""
    try:
        strategies = {}
        for strategy_name in strategy_monitor.strategy_status:
            strategy_info = strategy_monitor.get_strategy_status(strategy_name)
            
            # Get historical performance for the strategy
            strategy_trades = trade_journal.get_strategy_trades(strategy_name)
            
            # Calculate performance metrics
            wins = [t for t in strategy_trades if t.get('result') == 'win']
            win_rate = len(wins) / len(strategy_trades) if strategy_trades else 0
            total_pnl = sum([t.get('pnl_dollars', 0) for t in strategy_trades])
            
            # Calculate performance in current market environment
            current_env = str(strategy_monitor.market_environment)
            env_trades = [t for t in strategy_trades if t.get('market_regime') == current_env]
            env_win_rate = len([t for t in env_trades if t.get('result') == 'win']) / len(env_trades) if env_trades else 0
            
            # Add enhanced information
            strategy_info.update({
                "historical_performance": {
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "trade_count": len(strategy_trades)
                },
                "current_environment_performance": {
                    "win_rate": env_win_rate,
                    "trade_count": len(env_trades)
                },
                "allocation": strategy_rotator.get_allocation(strategy_name),
                "confidence_score": confidence_scorer.calculate_strategy_confidence(
                    strategy_name, 
                    win_rate, 
                    env_win_rate,
                    strategy_info.get('status')
                )
            })
            
            strategies[strategy_name] = strategy_info
            
        return jsonify({
            "status": "success",
            "strategies": strategies,
            "market_environment": str(strategy_monitor.market_environment),
            "optimal_strategies": strategy_monitor.get_strategy_recommendations()["optimal_strategies"],
            "capital_allocation": strategy_rotator.get_all_allocations(),
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving strategy status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/account/summary', methods=['GET'])
@require_auth
def get_account_summary():
    """Get account summary with balance, equity, and key metrics."""
    try:
        # In a real implementation, this would fetch data from the broker API
        # Here, we'll simulate it with data from our position sizer, etc.
        
        # Get trade history for calculating metrics
        trades = trade_journal.get_recent_trades(days=30)
        todays_trades = [t for t in trades if 
                         datetime.datetime.fromisoformat(t['timestamp']).date() == 
                         datetime.datetime.now().date()]
        
        # Calculate daily P&L
        daily_pnl = sum([t.get('pnl_dollars', 0) for t in todays_trades])
        
        # Get current positions value
        open_position_value = sum([
            t.get('quantity', 0) * t.get('current_price', 0) 
            for t in get_open_trades_list()
        ])
        
        # Calculate basic metrics
        wins = [t for t in trades if t.get('result') == 'win']
        win_rate = len(wins) / len(trades) if trades else 0
        
        # Build account summary
        account_summary = {
            "balance": position_sizer.account_size,
            "equity": position_sizer.account_size + open_position_value,
            "open_position_value": open_position_value,
            "daily_pnl": daily_pnl,
            "monthly_pnl": sum([t.get('pnl_dollars', 0) for t in trades]),
            "win_rate": win_rate,
            "trade_count": len(trades),
            "drawdown": position_sizer.current_drawdown_pct if hasattr(position_sizer, 'current_drawdown_pct') else 0,
            "metrics": {
                "profit_factor": sum([t.get('pnl_dollars', 0) for t in wins]) / 
                                abs(sum([t.get('pnl_dollars', 0) for t in [t for t in trades if t.get('result') == 'loss']])) 
                                if sum([t.get('pnl_dollars', 0) for t in [t for t in trades if t.get('result') == 'loss']]) != 0 else float('inf'),
                "avg_win": sum([t.get('pnl_dollars', 0) for t in wins]) / len(wins) if wins else 0,
                "avg_loss": abs(sum([t.get('pnl_dollars', 0) for t in [t for t in trades if t.get('result') == 'loss']])) / 
                         len([t for t in trades if t.get('result') == 'loss']) if len([t for t in trades if t.get('result') == 'loss']) > 0 else 0,
                "largest_win": max([t.get('pnl_dollars', 0) for t in wins]) if wins else 0,
                "largest_loss": min([t.get('pnl_dollars', 0) for t in trades]) if trades else 0
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify({
            "status": "success",
            "account": account_summary
        })
        
    except Exception as e:
        logger.error(f"Error retrieving account summary: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error retrieving account summary: {str(e)}"
        }), 500

def get_open_trades_list():
    """Helper function to get list of open trades for various endpoints."""
    try:
        open_trades_list = [
            t for t in psych_risk_manager.trading_history 
            if t.get("status") == "open"
        ]
        
        # Enhance with current prices (would be fetched from broker API in a real implementation)
        for trade in open_trades_list:
            if 'current_price' not in trade:
                # Just use a small random adjustment to entry price for demo purposes
                import random
                entry_price = trade.get('entry_price', 0)
                trade['current_price'] = entry_price * (1 + random.uniform(-0.02, 0.05))
                
                # Calculate unrealized P&L
                trade['unrealized_pnl'] = (trade['current_price'] - entry_price) * trade.get('position_size', 0)
                trade['unrealized_pnl_pct'] = ((trade['current_price'] / entry_price) - 1) * 100 if entry_price else 0
        
        return open_trades_list
    except Exception as e:
        logger.error(f"Error getting open trades list: {str(e)}")
        return []

@app.route('/dashboard/data', methods=['GET'])
@require_auth
def get_dashboard_data():
    """
    Get comprehensive data bundle for dashboard initialization.
    This endpoint is optimized to return all key data in one request.
    """
    try:
        # Get all required data
        account_data = get_account_summary().json
        open_trades_data = get_open_trades().json
        strategy_data = get_all_strategies().json
        recent_trades = trade_journal.get_recent_trades(days=5)
        recommendations_data = get_trade_recommendations().json
        cooldown_data = get_cooldown_status().json
        
        # Get market context for dashboard
        try:
            context_data = market_context.get_market_context()
            
            # Prepare context summary
            market_context_summary = {
                "bias": context_data.get("bias", "neutral"),
                "confidence": context_data.get("confidence", 0),
                "triggers": context_data.get("triggers", [])[:3],  # Top 3 triggers
                "suggested_strategies": context_data.get("suggested_strategies", []),
                "reasoning": context_data.get("reasoning", ""),
                "timestamp": context_data.get("timestamp")
            }
        except Exception as context_error:
            logger.error(f"Error getting market context for dashboard: {str(context_error)}")
            market_context_summary = {
                "bias": "neutral",
                "confidence": 0,
                "triggers": ["Context analysis unavailable"],
                "suggested_strategies": [],
                "reasoning": f"Error: {str(context_error)}",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Bundle everything for dashboard
        dashboard_data = {
            "account": account_data["account"],
            "open_trades": open_trades_data["open_trades"],
            "strategies": strategy_data["strategies"],
            "market_environment": strategy_data["market_environment"],
            "recent_trades": recent_trades,
            "recommendations": recommendations_data["recommendations"],
            "cooldown_status": cooldown_data,
            "market_context": market_context_summary,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify({
            "status": "success",
            "data": dashboard_data
        })
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error getting dashboard data: {str(e)}"
        }), 500

# Add utility function for JSON date handling in file export/import
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# Add endpoint to export data for dashboard charts
@app.route('/dashboard/chart-data', methods=['GET'])
@require_auth
def get_chart_data():
    """Get data specifically formatted for dashboard charts."""
    try:
        chart_type = request.args.get('type', 'equity_curve')
        days = int(request.args.get('days', 30))
        
        # Get trade data
        all_trades = trade_journal.get_all_trades()
        trades = trade_journal.get_recent_trades(days=days)
        
        result = {}
        
        if chart_type == 'equity_curve':
            # Generate equity curve data
            equity_curve = []
            starting_balance = position_sizer.account_size
            current_equity = starting_balance
            
            # Sort trades by date
            sorted_trades = sorted(all_trades, key=lambda t: t.get('timestamp', ''))
            
            for trade in sorted_trades:
                pnl = trade.get('pnl_dollars', 0)
                current_equity += pnl
                equity_curve.append({
                    'date': trade.get('timestamp', ''),
                    'equity': current_equity,
                    'trade_id': trade.get('trade_id', ''),
                    'pnl': pnl
                })
            
            result['equity_curve'] = equity_curve
            
        elif chart_type == 'win_loss':
            # Generate win/loss data
            wins = [t for t in trades if t.get('result') == 'win']
            losses = [t for t in trades if t.get('result') == 'loss']
            
            result['win_loss'] = {
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) if trades else 0
            }
            
        elif chart_type == 'strategy_performance':
            # Generate strategy performance data
            strategies = {}
            for trade in trades:
                strat = trade.get('strategy_name', 'Unknown')
                if strat not in strategies:
                    strategies[strat] = {
                        'trades': 0,
                        'wins': 0,
                        'pnl': 0
                    }
                    
                strategies[strat]['trades'] += 1
                strategies[strat]['pnl'] += trade.get('pnl_dollars', 0)
                if trade.get('result') == 'win':
                    strategies[strat]['wins'] += 1
            
            # Calculate win rates and other metrics
            for strat in strategies:
                strategies[strat]['win_rate'] = strategies[strat]['wins'] / strategies[strat]['trades'] if strategies[strat]['trades'] > 0 else 0
            
            result['strategy_performance'] = strategies
            
        elif chart_type == 'pnl_distribution':
            # Generate P&L distribution data
            pnl_values = [t.get('pnl_dollars', 0) for t in trades]
            
            result['pnl_distribution'] = pnl_values
            
        elif chart_type == 'day_of_week':
            # Generate day of week performance data
            days_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            day_performance = {day: {'trades': 0, 'wins': 0, 'pnl': 0} for day in days_of_week.values()}
            
            for trade in trades:
                try:
                    trade_date = datetime.datetime.fromisoformat(trade.get('timestamp', ''))
                    day = days_of_week[trade_date.weekday()]
                    day_performance[day]['trades'] += 1
                    day_performance[day]['pnl'] += trade.get('pnl_dollars', 0)
                    if trade.get('result') == 'win':
                        day_performance[day]['wins'] += 1
                except (ValueError, KeyError):
                    continue
                    
            for day in day_performance:
                if day_performance[day]['trades'] > 0:
                    day_performance[day]['win_rate'] = day_performance[day]['wins'] / day_performance[day]['trades']
                else:
                    day_performance[day]['win_rate'] = 0
                    
            result['day_of_week'] = day_performance
            
        elif chart_type == 'drawdown':
            # This would require tracking drawdown over time
            # For now, provide a placeholder with current drawdown
            result['drawdown'] = {
                'current': position_sizer.current_drawdown_pct if hasattr(position_sizer, 'current_drawdown_pct') else 0,
                'max': position_sizer.max_drawdown_pct if hasattr(position_sizer, 'max_drawdown_pct') else 0
            }
        
        elif chart_type == 'all':
            # Return data for all charts
            return jsonify({
                "status": "success",
                "chart_data": {
                    # Call this function recursively for each chart type
                    **json.loads(get_chart_data(type='equity_curve').data)['chart_data'],
                    **json.loads(get_chart_data(type='win_loss').data)['chart_data'],
                    **json.loads(get_chart_data(type='strategy_performance').data)['chart_data'],
                    **json.loads(get_chart_data(type='pnl_distribution').data)['chart_data'],
                    **json.loads(get_chart_data(type='day_of_week').data)['chart_data'],
                    **json.loads(get_chart_data(type='drawdown').data)['chart_data']
                }
            })
        
        return jsonify({
            "status": "success",
            "chart_data": result
        })
        
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error getting chart data: {str(e)}"
        }), 500

# Return the API token list for reference
@app.route('/api/tokens', methods=['GET'])
def get_api_tokens():
    """Get list of valid API tokens (only works locally)"""
    # Only allow this endpoint to be called locally
    if request.remote_addr != '127.0.0.1':
        return jsonify({"status": "error", "message": "This endpoint is only accessible locally"}), 403
        
    return jsonify({
        "status": "success",
        "auth_enabled": API_AUTH_ENABLED,
        "tokens": list(API_TOKENS.keys()),
        "user_count": len(API_TOKENS)
    })

@app.route('/api/status', methods=['GET'])
@require_auth
def api_status():
    """Get overall API and trading system status"""
    try:
        # Get basic system stats
        return jsonify({
            "status": "success",
            "service": "Trading Bot API",
            "version": "1.0.0",
            "uptime": "12 hours", # This would be determined dynamically in production
            "trading_allowed": cooldown_manager.is_trading_allowed(),
            "cooldown_level": str(cooldown_manager.current_level),
            "active_cooldowns": len(cooldown_manager.active_cooldowns),
            "strategies_enabled": strategy_monitor.get_active_strategy_count(),
            "paper_trading": executor.paper_trading,
            "account_size": position_sizer.account_size,
            "market_environment": str(strategy_monitor.market_environment),
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving API status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/journal/metrics', methods=['GET'])
@require_auth
def get_journal_metrics():
    """Get advanced performance metrics from journal."""
    try:
        # Get filters
        period = request.args.get('period', 'all')  # all, day, week, month, year
        strategy = request.args.get('strategy')
        
        # Get trade data
        trades = trade_journal.get_all_trades()
        
        # Filter by date if specified
        if period != 'all':
            now = datetime.datetime.now()
            if period == 'day':
                cutoff = now - datetime.timedelta(days=1)
            elif period == 'week':
                cutoff = now - datetime.timedelta(weeks=1)
            elif period == 'month':
                cutoff = now - datetime.timedelta(days=30)
            elif period == 'year':
                cutoff = now - datetime.timedelta(days=365)
                
            trades = [t for t in trades if datetime.datetime.fromisoformat(t['timestamp']) >= cutoff]
        
        # Filter by strategy if specified
        if strategy:
            trades = [t for t in trades if t['strategy_name'] == strategy]
        
        # Calculate top-level metrics
        wins = [t for t in trades if t['result'] == 'win']
        losses = [t for t in trades if t['result'] == 'loss']
        
        win_rate = len(wins) / len(trades) if trades else 0
        
        total_profit = sum([t['pnl_dollars'] for t in wins]) if wins else 0
        total_loss = abs(sum([t['pnl_dollars'] for t in losses])) if losses else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
        
        # Advanced metrics
        r_multiples = [t.get('r_multiple', 0) for t in trades if t.get('r_multiple') is not None]
        expectancy = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        
        # Calculate metrics by strategy
        strategy_metrics = {}
        strategies = set(t['strategy_name'] for t in trades if t['strategy_name'])
        
        for strat in strategies:
            strat_trades = [t for t in trades if t['strategy_name'] == strat]
            strat_wins = [t for t in strat_trades if t['result'] == 'win']
            
            strat_win_rate = len(strat_wins) / len(strat_trades) if strat_trades else 0
            strat_profit = sum([t['pnl_dollars'] for t in strat_trades]) if strat_trades else 0
            
            strategy_metrics[strat] = {
                "name": strat,
                "trades": len(strat_trades),
                "win_rate": strat_win_rate,
                "profit": strat_profit,
                "avg_win": sum([t['pnl_dollars'] for t in strat_wins]) / len(strat_wins) if strat_wins else 0,
                "avg_loss": abs(sum([t['pnl_dollars'] for t in [t for t in strat_trades if t['result'] == 'loss']])) / 
                           len([t for t in strat_trades if t['result'] == 'loss']) if len([t for t in strat_trades if t['result'] == 'loss']) > 0 else 0
            }
        
        # Sort strategies by profit for top performers
        top_performers = sorted(
            [s for s, m in strategy_metrics.items() if m['trades'] >= 5],
            key=lambda s: strategy_metrics[s]['profit'],
            reverse=True
        )[:5]
        
        # Find underperformers (win rate < 50% with at least 5 trades)
        underperformers = [
            s for s, m in strategy_metrics.items() 
            if m['trades'] >= 5 and m['win_rate'] < 0.5
        ]
        
        # Calculate performance by day of week
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_performance = {day: {"trades": 0, "wins": 0, "profit": 0} for day in days_of_week}
        
        for trade in trades:
            try:
                trade_date = datetime.datetime.fromisoformat(trade['timestamp'])
                day = days_of_week[trade_date.weekday()]
                day_performance[day]["trades"] += 1
                day_performance[day]["profit"] += trade['pnl_dollars']
                if trade['result'] == 'win':
                    day_performance[day]["wins"] += 1
            except (ValueError, IndexError):
                continue
                
        # Calculate win rate by day
        for day in day_performance:
            day_performance[day]["win_rate"] = (
                day_performance[day]["wins"] / day_performance[day]["trades"] 
                if day_performance[day]["trades"] > 0 else 0
            )
        
        # Build response
        metrics = {
            "overall": {
                "trades": len(trades),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "win_loss_ratio": win_loss_ratio,
                "expectancy": expectancy,
                "total_profit": total_profit - total_loss,
                "avg_win": avg_win,
                "avg_loss": avg_loss
            },
            "top_performers": {
                strategy: strategy_metrics[strategy] for strategy in top_performers
            },
            "underperformers": {
                strategy: {
                    **strategy_metrics[strategy],
                    "improvement_focus": [
                        "stricter entry criteria", 
                        "faster exits" if strategy_metrics[strategy]["avg_loss"] > 20 else "better stop placement"
                    ]
                } for strategy in underperformers
            },
            "day_performance": day_performance,
            "period": period
        }
        
        return jsonify({
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error retrieving journal metrics: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error retrieving journal metrics: {str(e)}"
        }), 500

@app.route('/journal/recommendations', methods=['GET'])
@require_auth
def get_trade_recommendations():
    """Get AI or rules-based recommendations for trade adjustments."""
    try:
        # Get filters
        min_confidence = float(request.args.get('min_confidence', 0.7))
        
        # Get recent trades for analysis
        trades = trade_journal.get_recent_trades(days=30)
        
        # Placeholder for actual AI/rule-based analysis
        # In a real implementation, this would use the ConfidenceScorer 
        # and other components to generate meaningful recommendations
        
        # Generate example recommendations
        recommendations = []
        
        # Generate strategy adjustment recommendations
        strategy_metrics = {}
        strategies = set(t['strategy_name'] for t in trades if t['strategy_name'])
        
        for strat in strategies:
            strat_trades = [t for t in trades if t['strategy_name'] == strat]
            if len(strat_trades) < 3:
                continue
                
            strat_wins = [t for t in strat_trades if t['result'] == 'win']
            strat_win_rate = len(strat_wins) / len(strat_trades) if strat_trades else 0
            
            # Strategy is underperforming
            if strat_win_rate < 0.4 and len(strat_trades) >= 5:
                recommendations.append({
                    "suggestion": f"Reduce position size for {strat} by 25%",
                    "focus": "risk management",
                    "confidence": 0.85,
                    "reasoning": "Consistent underperformance with sufficient sample size"
                })
            
            # Strategy is performing well
            if strat_win_rate > 0.6 and len(strat_trades) >= 5:
                recommendations.append({
                    "suggestion": f"Consider increasing position size for {strat} by 10%",
                    "focus": "opportunity",
                    "confidence": 0.75,
                    "reasoning": "Strategy showing consistent profitability"
                })
        
        # Check market regime match with strategies
        market_env = str(strategy_monitor.market_environment)
        recommendations.append({
            "suggestion": f"Current market regime is {market_env}",
            "focus": "information",
            "confidence": 0.95,
            "reasoning": "Based on current VIX and market breadth readings"
        })
        
        if market_env == "HIGH_VOLATILITY":
            recommendations.append({
                "suggestion": "Prioritize mean reversion strategies",
                "focus": "strategy selection",
                "confidence": 0.8,
                "reasoning": "High volatility environments favor mean reversion tactics"
            })
        elif market_env == "LOW_VOLATILITY_TRENDING":
            recommendations.append({
                "suggestion": "Prioritize trend following strategies",
                "focus": "strategy selection",
                "confidence": 0.8,
                "reasoning": "Trending low-volatility environments favor trend following"
            })
            
        # Check recent losses for psychological guidance
        recent_trades = trades[:10] if len(trades) >= 10 else trades
        recent_losses = [t for t in recent_trades if t['result'] == 'loss']
        
        if len(recent_losses) >= 3:
            recommendations.append({
                "suggestion": "Consider taking a short break to reset mentally",
                "focus": "psychology",
                "confidence": 0.7,
                "reasoning": "Multiple recent losses may impact decision making"
            })
            
        # Filter by confidence threshold
        recommendations = [r for r in recommendations if r["confidence"] >= min_confidence]
            
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error generating recommendations: {str(e)}"
        }), 500

# Add new API routes for market context
@app.route('/api/market-context', methods=['GET'])
@require_auth
def get_market_context():
    """Get current market context and sentiment"""
    try:
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        symbols = request.args.get('symbols')
        
        # Parse symbols if provided
        focus_symbols = None
        if symbols:
            focus_symbols = [s.strip().upper() for s in symbols.split(',')]
        
        # Get context
        context = market_context.get_market_context(
            force_refresh=refresh,
            focus_symbols=focus_symbols
        )
        
        return jsonify({
            'status': 'success',
            'data': context
        })
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/strategy-recommendation', methods=['GET'])
@require_auth
def get_strategy_recommendation():
    """Get recommended trading strategies based on current market"""
    try:
        # Get query parameters
        symbols = request.args.get('symbols')
        
        # Parse symbols if provided
        focus_symbols = None
        if symbols:
            focus_symbols = [s.strip().upper() for s in symbols.split(',')]
        
        # Get context
        context = market_context.get_market_context(focus_symbols=focus_symbols)
        
        # Extract strategies and reasoning
        strategies = context.get('suggested_strategies', [])
        triggers = context.get('triggers', [])
        reasoning = context.get('reasoning', '')
        
        return jsonify({
            'status': 'success',
            'data': {
                'market_bias': context.get('bias'),
                'confidence': context.get('confidence'),
                'recommended_strategies': strategies,
                'market_drivers': triggers,
                'analysis': reasoning,
                'timestamp': context.get('timestamp')
            }
        })
    except Exception as e:
        logger.error(f"Error getting strategy recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load API tokens on startup
    load_api_tokens()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 