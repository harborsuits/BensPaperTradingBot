#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Trading Orchestrator

This module implements the complete AI-driven trading decision flow, connecting
all AI components (indicator-sentiment integrator, LLM evaluator, position sizing,
adaptive risk management, and strategy pattern discovery) into a cohesive system 
with full contextual awareness.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable

from trading_bot.core.event_bus import EventBus, Event, EventType
from trading_bot.core.persistence import PersistenceManager
from trading_bot.ai_analysis.indicator_sentiment_integrator import IndicatorSentimentIntegrator
from trading_bot.ai_analysis.llm_trade_evaluator import LLMTradeEvaluator
from trading_bot.strategies.forex.adaptive_risk_manager import AdaptiveRiskManager
from trading_bot.strategies.forex.confidence_adjusted_position_sizing import ConfidenceAdjustedPositionSizing
from trading_bot.ai_analysis.strategy_pattern_discovery import StrategyPatternDiscovery

# Configure logging
logger = logging.getLogger(__name__)

class AITradingOrchestrator:
    """
    AI-Driven Trading Orchestrator
    
    Coordinates all AI components into a cohesive system with full 
    contextual awareness and end-to-end control of the trading process.
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 persistence_manager: Optional[PersistenceManager] = None,
                 state_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI Trading Orchestrator.
        
        Args:
            event_bus: Event bus for system communication
            persistence_manager: Optional persistence manager for state
            state_dir: Directory for state storage (created if not provided)
            config: Configuration for all AI components
        """
        self.event_bus = event_bus
        
        # Set up state directory
        if not state_dir:
            state_dir = os.path.join(os.path.dirname(__file__), '../../state')
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        
        # Set up persistence manager
        if not persistence_manager:
            # Create persistence manager if not provided
            persistence_manager = PersistenceManager(state_dir)
        self.persistence = persistence_manager
        
        # Load configuration
        self.config = self._load_config(config)
        logger.info(f"Loaded AI configuration with {len(self.config)} sections")
        
        # Initialize AI components
        self._initialize_components()
        
        # Track active symbols and their state
        self.active_symbols = {}
        self.trade_in_progress = {}
        self.last_check_time = {}
        
        # Track context and state
        self.last_account_balance = 0
        self.system_ready = False
        
        # Live trading control and approval tracking
        self.paper_trading_mode = True  # Default to paper trading
        self.pending_approvals = {}  # Tracks trades awaiting approval
        self.approval_timeout_seconds = 300  # 5 minutes for approval response
        self.emergency_sell_enabled = True  # Allow emergency sells
        
        logger.info("AI Trading Orchestrator initialized")
    
    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load and merge configuration from file and provided config."""
        default_config = {
            'integrator': {
                'indicator_weight': 0.6,
                'sentiment_weight': 0.4,
                'news_sentiment_weight': 0.4,
                'social_sentiment_weight': 0.3,
                'market_sentiment_weight': 0.3,
                'stale_data_seconds': 3600,
                'integration_interval_seconds': 5.0,
                'min_data_points': 3,
                'max_cache_size': 1000,
                'cache_expiry_seconds': 3600
            },
            'llm_evaluator': {
                'model': 'gpt-4',
                'use_mock': True,  # Set to False in production with API key
                'cache_enabled': True
            },
            'position_sizing': {
                'max_risk_per_trade_percent': 1.0,
                'min_position_size': 0.01,
                'max_position_size': 5.0,
                'use_confidence_adjustment': True,
                'min_confidence_threshold': 0.4,
                'high_confidence_threshold': 0.7,
                'signal_agreement_bonus': 0.3,
                'signal_disagreement_penalty': 0.5
            },
            'risk_manager': {
                'base_portfolio_size': 10000,
                'risk_scale_factor': 0.8,
                'min_risk_percent': 0.2,
                'max_risk_percent': 2.0,
                'performance_window_days': 30,
                'min_trades_for_adjustment': 20,
                'profit_factor_threshold': 1.3,
                'drawdown_threshold': 10.0,
                'auto_optimization': True
            },
            'pattern_discovery': {
                'pattern_library': 'default',
                'pattern_matching_interval_seconds': 60,
                'max_pattern_matches': 5
            },
            'orchestrator': {
                'min_time_between_trades_seconds': 300,  # 5 minutes
                'max_concurrent_trades_per_symbol': 1,
                'enable_llm_evaluation': True,
                'min_llm_confidence_to_trade': 0.6,
                'health_check_interval_seconds': 300,
                'paper_trading': True,  # Default to paper trading
                'paper_trading_end_date': '2025-05-30',  # One month from now
                'require_approval_for_live': True,  # Require human approval for live trades
                'high_confidence_threshold': 0.95,  # Threshold for automatic execution
                'market_crash_protection': True,  # Enable market crash protection
                'crash_detection_threshold': -5.0,  # % market drop to trigger protection
                'emergency_sell_enabled': True  # Allow emergency sells
            }
        }
        
        # Load config from file if exists
        config_path = os.path.join(self.state_dir, 'ai_config.json')
        file_config = {}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                logger.info(f"Loaded AI configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Merge configurations with precedence: provided > file > default
        merged_config = default_config.copy()
        
        # Update with file config
        for section, settings in file_config.items():
            if section in merged_config:
                merged_config[section].update(settings)
            else:
                merged_config[section] = settings
                
        # Update with provided config
        if config:
            for section, settings in config.items():
                if section in merged_config:
                    merged_config[section].update(settings)
                else:
                    merged_config[section] = settings
                    
        return merged_config
                
    def _initialize_components(self):
        """Initialize all AI components with proper configuration."""
        # Initialize indicator-sentiment integrator
        integrator_config = self.config.get('integrator', {})
        self.integrator = IndicatorSentimentIntegrator(
            event_bus=self.event_bus,
            llm_evaluator=None,  # Will set this after LLM is initialized
            indicator_weight=integrator_config.get('indicator_weight', 0.6),
            sentiment_weight=integrator_config.get('sentiment_weight', 0.4),
            config=integrator_config,
            max_cache_size=integrator_config.get('max_cache_size', 1000),
            cache_expiry_seconds=integrator_config.get('cache_expiry_seconds', 3600)
        )
        
        # Initialize LLM trade evaluator
        llm_config = self.config.get('llm_evaluator', {})
        cache_dir = os.path.join(self.state_dir, 'llm_cache') if llm_config.get('cache_enabled', True) else None
        os.makedirs(cache_dir, exist_ok=True) if cache_dir else None
        
        self.llm_evaluator = LLMTradeEvaluator(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model=llm_config.get('model', 'gpt-4'),
            use_mock=llm_config.get('use_mock', True),
            cache_dir=cache_dir,
            integrator=self.integrator  # Give LLM direct access to integrator
        )
        
        # Set LLM evaluator in integrator (circular reference for direct access)
        self.integrator.llm_evaluator = self.llm_evaluator
        
        # Initialize confidence-adjusted position sizing
        position_config = self.config.get('position_sizing', {})
        self.position_sizer = ConfidenceAdjustedPositionSizing(position_config)
        
        # Initialize adaptive risk manager
        risk_config = self.config.get('risk_manager', {})
        self.risk_manager = AdaptiveRiskManager(
            position_sizer=self.position_sizer,
            state_dir=os.path.join(self.state_dir, 'risk_manager'),
            parameters=risk_config
        )
        
        # Initialize strategy pattern discovery
        pattern_config = self.config.get('pattern_discovery', {})
        self.pattern_discovery = StrategyPatternDiscovery(
            event_bus=self.event_bus,
            pattern_library=pattern_config.get('pattern_library', 'default'),
            pattern_matching_interval_seconds=pattern_config.get('pattern_matching_interval_seconds', 60),
            max_pattern_matches=pattern_config.get('max_pattern_matches', 5)
        )
        
        logger.info("All AI components initialized")
        
    def register_event_handlers(self):
        """Register all event handlers for full contextual awareness."""
        # Market data and indicator handling
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self.handle_market_data)
        self.event_bus.subscribe(EventType.TECHNICAL_INDICATORS_UPDATED, self.handle_indicator_update)
        
        # Sentiment data handling
        self.event_bus.subscribe(EventType.NEWS_SENTIMENT_UPDATED, self.handle_sentiment_update)
        self.event_bus.subscribe(EventType.SOCIAL_SENTIMENT_UPDATED, self.handle_sentiment_update)
        self.event_bus.subscribe(EventType.MARKET_SENTIMENT_UPDATED, self.handle_sentiment_update)
        
        # Trade signal handling
        self.event_bus.subscribe(EventType.TRADE_SIGNAL_RECEIVED, self.handle_trade_signal)
        
        # Trade execution handling
        self.event_bus.subscribe(EventType.ORDER_FILLED, self.handle_order_filled)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        
        # Account and portfolio tracking
        self.event_bus.subscribe(EventType.ACCOUNT_BALANCE_UPDATED, self.handle_account_update)
        
        # System events
        self.event_bus.subscribe(EventType.SYSTEM_START, self.handle_system_start)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self.handle_system_shutdown)
        self.event_bus.subscribe(EventType.SYSTEM_STATUS_REQUEST, self.handle_status_request)
        
        # Human approval events
        self.event_bus.subscribe(EventType.TRADE_APPROVAL_RESPONSE, self.handle_approval_response)
        self.event_bus.subscribe(EventType.MARKET_CRASH_ALERT, self.handle_market_crash_alert)
        
        logger.info("Event handlers registered for AI Trading Orchestrator")
        
        # Set up periodic health checks
        health_interval = self.config.get('orchestrator', {}).get('health_check_interval_seconds', 300)
        self.event_bus.register_interval(self.publish_health_check, health_interval)
        
        # Check for pending approvals timeout
        self.event_bus.register_interval(self.check_pending_approvals, 60)  # Check every minute
    
    def handle_order_filled(self, event: Event):
        """Handle order filled events to track active trades."""
        data = event.data
        symbol = data.get('symbol')
        
        if not symbol:
            return
            
        # Mark symbol as having an active trade
        self.active_symbols[symbol] = True
        
        # Track applied patterns for this trade
        applied_pattern_ids = data.get('ai_metadata', {}).get('applied_pattern_ids', [])
        if applied_pattern_ids:
            # Add pattern IDs to the order data for later tracking
            # when the trade is closed
            data['applied_pattern_ids'] = applied_pattern_ids
        
        logger.info(f"Order filled for {symbol}")
    
    def handle_trade_signal(self, event: Event):
        """
        Handle incoming trade signals, enriching with AI analysis 
        and controlling execution with confidence-based position sizing.
        """
        data = event.data
        symbol = data.get('symbol')
        direction = data.get('signal_type', 'UNKNOWN')
        entry_price = data.get('price', 0.0)
        strategy = data.get('strategy', 'Unknown')
        
        # Skip if system not ready
        if not self.system_ready:
            logger.warning(f"Ignoring trade signal for {symbol} - system not ready")
            return
            
        # Skip if we already have a trade in progress for this symbol
        if symbol in self.trade_in_progress and self.trade_in_progress[symbol]:
            logger.info(f"Ignoring trade signal for {symbol} - trade already in progress")
            return
            
        # Skip if we've checked this symbol too recently
        min_interval = self.config.get('orchestrator', {}).get('min_time_between_trades_seconds', 300)
        if symbol in self.last_check_time:
            elapsed = (datetime.now() - self.last_check_time[symbol]).total_seconds()
            if elapsed < min_interval:
                logger.info(f"Ignoring trade signal for {symbol} - checked too recently ({elapsed:.0f}s < {min_interval}s)")
                return
                
        logger.info(f"Processing trade signal for {symbol}: {direction} at {entry_price}, strategy: {strategy}")
        
        # Get latest account balance
        account_balance = self.last_account_balance
        if account_balance <= 0:
            logger.warning(f"Cannot process trade signal - invalid account balance: {account_balance}")
            return
            
        # Mark symbol as being checked
        self.last_check_time[symbol] = datetime.now()
        
        # Extract stop loss in pips (convert from price to pips if needed)
        stop_loss_price = data.get('stop_loss', 0.0)
        if not stop_loss_price or stop_loss_price == entry_price:
            logger.warning(f"Invalid stop loss for {symbol}: {stop_loss_price} - skipping trade")
            return
            
        # Calculate stop loss in pips
        if 'stop_loss_pips' in data:
            stop_loss_pips = data['stop_loss_pips']
        else:
            # Approximate conversion based on symbol
            stop_loss_pips = self._calculate_pips(symbol, entry_price, stop_loss_price)
            
        # Get or request integrated analysis
        integrated_data = None
        if symbol in self.active_symbols and self.active_symbols[symbol].get('integrated_data'):
            integrated_data = self.active_symbols[symbol]['integrated_data']
        else:
            # Request on-demand integration
            integrated_data = self.integrator.get_integrated_analysis(symbol)
            
            # Store for future use
            if symbol in self.active_symbols:
                self.active_symbols[symbol]['integrated_data'] = integrated_data
                
        if not integrated_data:
            logger.warning(f"No integrated data available for {symbol} - proceeding with limited analysis")
            
        # Optionally enhance with LLM evaluation
        llm_evaluation = None
        if self.config.get('orchestrator', {}).get('enable_llm_evaluation', True) and integrated_data:
            try:
                llm_evaluation = self.llm_evaluator.evaluate_trade(
                    symbol=symbol,
                    direction=direction,
                    strategy=strategy,
                    price=entry_price,
                    stop_loss=stop_loss_price,
                    take_profit=data.get('take_profit'),
                    use_integrated_data=True  # Use the integrated data we already have
                )
                
                logger.info(f"LLM evaluation for {symbol}: confidence={llm_evaluation.get('confidence_score', 0):.2f}, "
                            f"recommendation='{llm_evaluation.get('recommendation', 'unknown')}'")
                            
                # Check if LLM confidence is sufficient
                min_llm_confidence = self.config.get('orchestrator', {}).get('min_llm_confidence_to_trade', 0.6)
                if llm_evaluation.get('confidence_score', 0) < min_llm_confidence:
                    logger.info(f"Skipping trade for {symbol} - LLM confidence ({llm_evaluation.get('confidence_score', 0):.2f}) "
                                f"below threshold ({min_llm_confidence})")
                    
                    # Publish evaluation result so dashboard knows why we skipped
                    self.event_bus.publish(
                        EventType.TRADE_EVALUATION_COMPLETED,
                        {
                            'symbol': symbol,
                            'evaluation': llm_evaluation,
                            'skipped': True,
                            'reason': 'Low LLM confidence',
                            'timestamp': datetime.now()
                        }
                    )
                    return
                    
            except Exception as e:
                logger.error(f"Error during LLM evaluation for {symbol}: {str(e)}")
                # Continue without LLM evaluation if it fails
        
        # Update risk management parameters based on performance history
        optimized_params = self.risk_manager.get_optimized_parameters(account_balance)
        self.position_sizer.confidence_params.update(optimized_params)
        
        # Calculate position size with confidence adjustment
        position_size, adjustment_details = self.position_sizer.calculate_position_size_with_confidence(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_pips=stop_loss_pips,
            account_balance=account_balance,
            integrated_data=integrated_data or {}  # Empty dict if None
        )
        
        # Skip if position size is zero (below confidence threshold)
        if position_size <= 0:
            logger.info(f"Skipping trade for {symbol} - position size is zero (likely below confidence threshold)")
            
            # Publish decision so dashboard knows why we skipped
            self.event_bus.publish(
                EventType.TRADE_DECISION,
                {
                    'symbol': symbol,
                    'decision': 'SKIP',
                    'reason': 'Zero position size',
                    'confidence': integrated_data.get('confidence', 0) if integrated_data else 0,
                    'adjustment_details': adjustment_details,
                    'timestamp': datetime.now()
                }
            )
            return
            
        # Calculate risk amount for logging
        risk_amount = self.position_sizer.calculate_risk_amount(
            position_size=position_size,
            stop_loss_pips=stop_loss_pips,
            symbol=symbol,
            entry_price=entry_price
        )
        
        # Log the decision details
        logger.info(f"Trade decision for {symbol}: {direction} {position_size:.2f} lots at {entry_price}")
        logger.info(f"Risk amount: ${risk_amount:.2f} ({(risk_amount/account_balance)*100:.2f}% of account)")
        logger.info(f"Confidence: {integrated_data.get('confidence', 0):.2f} if integrated_data else 'N/A'}, "
                    f"Adjustment factor: {adjustment_details.get('adjustment_factor', 1.0):.2f}")
                    
        # Get applicable specialized patterns for this symbol and conditions
        current_market_conditions = self._get_current_market_conditions(symbol)
        applicable_patterns = self.pattern_discovery.get_applicable_patterns(
            symbol=symbol,
            timeframe=data.get('timeframe', '1h'),  # Default to 1h if not specified
            current_market_conditions=current_market_conditions
        )
        
        # Prepare trade evaluation data
        eval_data = {
            'symbol': symbol,
            'direction': direction,
            'strategy': strategy,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'integrated_data': integrated_data,
            'risk_amount': risk_amount,
            'applicable_patterns': applicable_patterns
        }
        
        # Create order parameters
        order_params = {
            'symbol': symbol,
            'direction': direction,
            'position_size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': data.get('take_profit'),
            'strategy': strategy,
            'timestamp': datetime.now()
        }
        
        # Add AI metadata to order
        confidence = integrated_data.get('confidence', 0) if integrated_data else 0
        llm_confidence = llm_evaluation.get('confidence_score', 0) if llm_evaluation else 0
        
        # Extract applicable pattern IDs if any
        applied_patterns = []
        pattern_boost = 0.0
        applicable_patterns = eval_data.get('applicable_patterns', [])
        
        if applicable_patterns:
            # Get high-confidence patterns
            high_conf_patterns = [p for p in applicable_patterns if p.get('confidence', 0) > 0.7]
            
            # Apply confidence boost if we have high confidence patterns
            if high_conf_patterns:
                pattern_boost = 0.1  # 10% confidence boost when specialized patterns apply
                pattern_names = [p.get('pattern_name') for p in high_conf_patterns]
                logger.info(f"Applying specialized patterns to {symbol}: {', '.join(pattern_names)}")
                
            # Track pattern IDs for performance tracking
            applied_patterns = [p.get('pattern_id') for p in applicable_patterns]
        
        ai_metadata = {
            'confidence': min(1.0, confidence + pattern_boost),  # Apply pattern boost but cap at 1.0
            'integrated_score': integrated_data.get('integrated_score', 0) if integrated_data else 0,
            'risk_amount': risk_amount,
            'risk_percent': (risk_amount/account_balance)*100,
            'adjustment_factor': adjustment_details.get('adjustment_factor', 1.0),
            'llm_confidence': llm_confidence,
            'applied_pattern_ids': applied_patterns,
            'pattern_boost': pattern_boost if pattern_boost > 0 else None
        }
        order_params['ai_metadata'] = ai_metadata
        
        # Determine if trade needs approval for live trading
        needs_approval = False
        
        if not self.paper_trading_mode:  # Only check approval for live trading
            orchestrator_config = self.config.get('orchestrator', {})
            require_approval = orchestrator_config.get('require_approval_for_live', True)
            high_confidence_threshold = orchestrator_config.get('high_confidence_threshold', 0.95)
            
            # High confidence can bypass approval if configured that way
            high_confidence_bypass = (confidence >= high_confidence_threshold or 
                                     llm_confidence >= high_confidence_threshold)
            
            # Emergency sells can bypass approval
            emergency_sell = (direction.upper() == 'SELL' and 
                             self.emergency_sell_enabled and
                             symbol in self.trade_in_progress)
            
            # Determine if we need approval
            needs_approval = require_approval and not (high_confidence_bypass or emergency_sell)
            
            if high_confidence_bypass:
                logger.info(f"High confidence trade ({max(confidence, llm_confidence):.2f} >= {high_confidence_threshold}) - bypassing approval")
            elif emergency_sell:
                logger.info(f"Emergency sell order for {symbol} - bypassing approval")
        
        if needs_approval:
            # Store pending approval
            approval_id = f"{symbol}_{int(time.time())}"
            self.pending_approvals[approval_id] = {
                'order_params': order_params,
                'timestamp': datetime.now(),
                'expiry': datetime.now() + timedelta(seconds=self.approval_timeout_seconds)
            }
            
            # Request approval
            self.event_bus.publish(
                EventType.TRADE_APPROVAL_REQUEST,
                {
                    'approval_id': approval_id,
                    'symbol': symbol,
                    'direction': direction,
                    'position_size': position_size,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': data.get('take_profit'),
                    'confidence': confidence,
                    'llm_confidence': llm_confidence,
                    'risk_amount': risk_amount,
                    'risk_percent': (risk_amount/account_balance)*100,
                    'timestamp': datetime.now(),
                    'expires_at': self.pending_approvals[approval_id]['expiry']
                }
            )
            logger.info(f"Requesting approval for live trade on {symbol} (ID: {approval_id})")
            
            # Mark trade as waiting approval but not fully in progress
            self.trade_in_progress[symbol] = "pending_approval"
        else:
            # No approval needed (paper trading or high confidence), execute immediately
            self.event_bus.publish(
                EventType.ORDER_REQUEST,
                order_params
            )
            
            # Mark trade in progress
            self.trade_in_progress[symbol] = True
            
            logger.info(f"Published {'paper ' if self.paper_trading_mode else ''}order request for {symbol}")
    
    def handle_order_filled(self, event: Event):
        """Handle order filled events to track active trades."""
        data = event.data
        symbol = data.get('symbol')
        
        if not symbol:
            return
            
        # Mark symbol as having an active trade
        self.active_symbols[symbol] = True
        
        # Track applied patterns for this trade
        applied_pattern_ids = data.get('ai_metadata', {}).get('applied_pattern_ids', [])
        if applied_pattern_ids:
            # Add pattern IDs to the order data for later tracking
            # when the trade is closed
            data['applied_pattern_ids'] = applied_pattern_ids
        
        logger.info(f"Order filled for {symbol}")
    
    def handle_trade_closed(self, event: Event):
        """
        Handle trade closed events to record results and
        feed back to the AI learning system.
        """
        data = event.data
        symbol = data.get('symbol')
        profit_loss = data.get('profit_loss', 0)
        
        logger.info(f"Trade closed for {symbol}: P/L {profit_loss}")
        
        # Mark trade no longer in progress
        if symbol in self.trade_in_progress:
            self.trade_in_progress[symbol] = False
            
        # Record trade result for AI learning
        try:
            # Get confidence data if available
            confidence_data = None
            if symbol in self.active_symbols and self.active_symbols[symbol].get('integrated_data'):
                confidence_data = self.active_symbols[symbol]['integrated_data']
                
            # Record in risk manager for learning
            self.risk_manager.record_trade_result(
                trade_data={
                    'symbol': symbol,
                    'entry_price': data.get('entry_price'), 
                    'exit_price': data.get('exit_price'),
                    'profit_loss': profit_loss,
                    'trade_id': data.get('trade_id'),
                    'timestamp': data.get('timestamp', datetime.now().isoformat())
                },
                account_balance=self.last_account_balance,
                confidence_data=confidence_data
            )
            
            logger.info(f"Recorded trade result for {symbol} in risk manager for AI learning")
            
        except Exception as e:
            logger.error(f"Error recording trade result: {str(e)}")
            
    def handle_account_update(self, event: Event):
        """Handle account balance updates to track portfolio growth."""
        data = event.data
        balance = data.get('balance', 0)
        
        if balance > 0:
            self.last_account_balance = balance
            logger.debug(f"Updated account balance: {balance}")
            
    def handle_system_start(self, event: Event):
        """Handle system start event to initialize AI components."""
        logger.info("System start event received")
        self.system_ready = True
        
        # Set trading mode from config
        orchestrator_config = self.config.get('orchestrator', {})
        self.paper_trading_mode = orchestrator_config.get('paper_trading', True)
        
        # Check paper trading end date
        paper_end_date = orchestrator_config.get('paper_trading_end_date')
        if paper_end_date:
            try:
                end_date = datetime.fromisoformat(paper_end_date)
                if datetime.now() > end_date:
                    logger.info(f"Paper trading end date ({paper_end_date}) has passed")
                    # Still keep paper trading mode unless explicitly changed
            except ValueError:
                logger.warning(f"Invalid paper trading end date format: {paper_end_date}")
                
        trading_mode = "PAPER" if self.paper_trading_mode else "LIVE"
        logger.info(f"AI Trading Orchestrator initialized in {trading_mode} trading mode")
        
        # Publish AI system status
        self.event_bus.publish(
            EventType.AI_SYSTEM_READY,
            {
                'timestamp': datetime.now(),
                'components': [
                    'indicator_sentiment_integrator',
                    'llm_trade_evaluator',
                    'confidence_position_sizing',
                    'adaptive_risk_manager',
                    'strategy_pattern_discovery'
                ]
            }
        )
        
    def handle_system_shutdown(self, event: Event):
        """Handle system shutdown to gracefully close AI components."""
        logger.info("System shutdown event received")
        self.system_ready = False
        
        # Perform any cleanup needed
        
    def handle_status_request(self, event: Event):
        """Handle system status request to report AI component health."""
        logger.debug("Status request received")
        
        # Gather status from all components
        status = {
            'timestamp': datetime.now(),
            'system_ready': self.system_ready,
            'active_symbols': len(self.active_symbols),
            'trades_in_progress': sum(1 for v in self.trade_in_progress.values() if v),
            'components': {
                'integrator': self.integrator.handle_status_request(event).data if hasattr(self.integrator, 'handle_status_request') else {'status': 'unknown'},
                'risk_manager': self.risk_manager.get_performance_summary(),
                'position_sizer': {
                    'status': 'active',
                    'current_parameters': self.position_sizer.confidence_params
                },
                'llm_evaluator': {
                    'status': 'active', 
                    'model': self.llm_evaluator.model,
                    'recent_evaluations': len(self.llm_evaluator.evaluations)
                },
                'pattern_discovery': {
                    'status': 'active',
                    'pattern_library': self.pattern_discovery.pattern_library,
                    'recent_matches': len(self.pattern_discovery.recent_matches)
                }
            }
        }
        
        # Publish AI system status
        self.event_bus.publish(
            EventType.AI_SYSTEM_STATUS,
            status
        )
        
    def publish_health_check(self):
        """Publish periodic health check to ensure AI system is functioning."""
        if not self.system_ready:
            return
            
        # Create health check data
        health_data = {
            'timestamp': datetime.now(),
            'system_ready': self.system_ready,
            'components_healthy': True,
            'active_symbols': len(self.active_symbols),
            'trades_in_progress': sum(1 for v in self.trade_in_progress.values() if v)
        }
        
        # Check integrator health
        try:
            integrator_status = self.integrator.handle_status_request(None)
            health_data['integrator_healthy'] = integrator_status.data.get('status') == 'operational'
        except Exception:
            health_data['integrator_healthy'] = False
            health_data['components_healthy'] = False
            
        # Publish health check
        self.event_bus.publish(
            EventType.AI_HEALTH_CHECK,
            health_data
        )
        
    def _calculate_pips(self, symbol: str, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate the distance in pips between entry and stop loss.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Distance in pips
        """
        price_diff = abs(entry_price - stop_loss_price)
        
        # For JPY pairs, 1 pip = 0.01, for others 1 pip = 0.0001
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        # Convert to pips
        pips = price_diff / pip_value
        
        return pips
        
    def start(self):
        """Start the AI trading orchestrator."""
        if self.system_ready:
            logger.warning("AI Trading Orchestrator already started")
            return
            
        # Register event handlers
        self.register_event_handlers()
        
        # Start AI components
        self.indicator_sentiment_integrator.register_event_handlers()
        self.llm_trade_evaluator.register_event_handlers()
        self.adaptive_risk_manager.register_event_handlers()
        self.position_sizer.register_event_handlers()
        self.pattern_discovery.register_event_handlers()
        
        # Mark system as ready
        self.system_ready = True
        
        logger.info("AI Trading Orchestrator started")
        
    def handle_approval_response(self, event: Event):
        """Handle human approval response for live trades."""
        data = event.data
        approval_id = data.get('approval_id')
        approved = data.get('approved', False)
        
        if approval_id not in self.pending_approvals:
            logger.warning(f"Received approval response for unknown ID: {approval_id}")
            return
            
        pending_info = self.pending_approvals[approval_id]
        order_params = pending_info['order_params']
        symbol = order_params['symbol']
        
        logger.info(f"Received {'approval' if approved else 'rejection'} for trade ID {approval_id} ({symbol})")
        
        if approved:
            # Execute the approved trade
            self.event_bus.publish(
                EventType.ORDER_REQUEST,
                order_params
            )
            
            # Mark trade in progress
            self.trade_in_progress[symbol] = True
            
            logger.info(f"Executing approved live trade for {symbol}")
        else:
            # Clear pending status
            if symbol in self.trade_in_progress and self.trade_in_progress[symbol] == "pending_approval":
                self.trade_in_progress[symbol] = False
                
            logger.info(f"Live trade rejected for {symbol}")
            
        # Remove from pending approvals
        del self.pending_approvals[approval_id]
    
    def check_pending_approvals(self):
        """Check for expired pending approvals and cancel them."""
        now = datetime.now()
        expired_ids = []
        
        for approval_id, info in self.pending_approvals.items():
            if now > info['expiry']:
                expired_ids.append(approval_id)
                symbol = info['order_params']['symbol']
                logger.info(f"Approval timeout for trade ID {approval_id} ({symbol})")
                
                # Clear pending status
                if symbol in self.trade_in_progress and self.trade_in_progress[symbol] == "pending_approval":
                    self.trade_in_progress[symbol] = False
                    
                # Publish timeout event
                self.event_bus.publish(
                    EventType.TRADE_APPROVAL_TIMEOUT,
                    {
                        'approval_id': approval_id,
                        'symbol': symbol,
                        'timestamp': now
                    }
                )
                
        # Remove expired approvals
        for expired_id in expired_ids:
            del self.pending_approvals[expired_id]
    
    def handle_market_crash_alert(self, event: Event):
        """Handle market crash alerts for emergency action."""
        data = event.data
        severity = data.get('severity', 0.0)
        affected_symbols = data.get('affected_symbols', [])
        
        orchestrator_config = self.config.get('orchestrator', {})
        crash_threshold = orchestrator_config.get('crash_detection_threshold', -5.0)
        
        if severity <= crash_threshold and orchestrator_config.get('market_crash_protection', True):
            logger.warning(f"Market crash protection activated (severity: {severity:.2f}%)")
            
            # Enable emergency sells
            self.emergency_sell_enabled = True
            
            # For severe crashes, we might want to close all positions
            if severity <= crash_threshold * 2:  # Double the threshold for extreme crashes
                logger.warning("Severe market crash detected - initiating emergency position closures")
                
                for symbol in affected_symbols:
                    if symbol in self.trade_in_progress and self.trade_in_progress[symbol] == True:
                        # Create emergency sell order
                        self.event_bus.publish(
                            EventType.EMERGENCY_CLOSE_POSITION,
                            {
                                'symbol': symbol,
                                'timestamp': datetime.now(),
                                'reason': f"Emergency market crash protection (severity: {severity:.2f}%)"
                            }
                        )
                        logger.info(f"Emergency position closure requested for {symbol}")
    
    def set_trading_mode(self, paper_trading: bool = True):
        """Set trading mode between paper and live."""
        if paper_trading != self.paper_trading_mode:
            old_mode = "PAPER" if self.paper_trading_mode else "LIVE"
            new_mode = "PAPER" if paper_trading else "LIVE"
            
            logger.warning(f"Changing trading mode from {old_mode} to {new_mode}")
            self.paper_trading_mode = paper_trading
            
            # Publish mode change event
            self.event_bus.publish(
                EventType.TRADING_MODE_CHANGED,
                {
                    'paper_trading': paper_trading,
                    'timestamp': datetime.now()
                }
            )
    
    def _get_current_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions for pattern matching."""
        # This would collect various market conditions that might be relevant for pattern matching
        # For now, return a simple placeholder
        return {
            'symbol': symbol,
            'volatility': self.active_symbols.get(f"{symbol}_volatility", 'normal'),
            'trend': self.active_symbols.get(f"{symbol}_trend", 'neutral'),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'active_events': []  # Would be populated with active economic events
        }
    
    def stop(self):
        """Stop the AI trading orchestrator."""
        if not self.system_ready:
            logger.warning("AI Trading Orchestrator already stopped")
            return
            
        # Mark system as not ready
        self.system_ready = False
        
        logger.info("AI Trading Orchestrator stopped")
