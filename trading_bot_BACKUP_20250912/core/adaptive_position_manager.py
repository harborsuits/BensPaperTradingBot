"""
Adaptive Position Manager

This module connects market regime detection with position sizing to ensure 
that position sizes are properly adapted to current market conditions.

It acts as a bridge between the market regime detector, portfolio risk manager,
and the position sizing components.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.data.persistence import PersistenceManager
from trading_bot.core.event_bus import EventBus, Event, EventType
from trading_bot.strategies.forex.base.pip_based_position_sizing import PipBasedPositionSizing

logger = logging.getLogger(__name__)

class AdaptivePositionManager:
    """
    Manages position sizing based on market regime and portfolio state.
    
    This class:
    1. Monitors market regime changes
    2. Tracks correlation between assets
    3. Calculates adaptive position sizes
    4. Manages overall portfolio risk limits
    """
    
    def __init__(self, 
                 persistence_manager: PersistenceManager,
                 event_bus: EventBus,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive position manager.
        
        Args:
            persistence_manager: For loading market state and strategy data
            event_bus: For publishing and subscribing to events
            config: Configuration parameters
        """
        self.persistence = persistence_manager
        self.event_bus = event_bus
        self.config = config or {}
        
        # Initialize position sizers for different asset classes
        self.position_sizers = {
            'forex': PipBasedPositionSizing(self.config.get('forex_position_sizing', {})),
            # Add other asset class position sizers here
        }
        
        # Track current market state
        self.market_state = {
            'regime': 'unknown',
            'volatility': 'medium',
            'trend': 'neutral',
            'last_update': datetime.now()
        }
        
        # Track portfolio state
        self.portfolio_state = {
            'current_positions': [],
            'correlation_matrix': {},
            'current_drawdown': 0.0,
            'recent_trades': []
        }
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Adaptive Position Manager initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        # Market regime events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self.handle_market_regime_change)
        self.event_bus.subscribe(EventType.MARKET_REGIME_DETECTED, self.handle_market_regime_detection)
        
        # Portfolio events
        self.event_bus.subscribe(EventType.CORRELATION_MATRIX_UPDATED, self.handle_correlation_update)
        self.event_bus.subscribe(EventType.PORTFOLIO_EXPOSURE_UPDATED, self.handle_portfolio_update)
        self.event_bus.subscribe(EventType.DRAWDOWN_THRESHOLD_EXCEEDED, self.handle_drawdown_update)
        
        # Trade events
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        
        logger.info("Event handlers registered for Adaptive Position Manager")
    
    def handle_market_regime_change(self, event: Event):
        """Handle market regime change events."""
        logger.info(f"Market regime changed: {event.data}")
        
        regime = event.data.get('regime')
        if regime:
            self.market_state['regime'] = regime
            self.market_state['last_update'] = datetime.now()
            
            # Update volatility and trend if available
            if 'volatility' in event.data:
                self.market_state['volatility'] = event.data['volatility']
            if 'trend' in event.data:
                self.market_state['trend'] = event.data['trend']
            
            # Save to persistence
            self.persistence.save_strategy_state("adaptive_position_manager", {
                'market_state': self.market_state
            })
    
    def handle_market_regime_detection(self, event: Event):
        """Handle market regime detection events with detailed metrics."""
        # Similar to regime change but with more details
        if 'regime' in event.data:
            self.handle_market_regime_change(event)
    
    def handle_correlation_update(self, event: Event):
        """Handle correlation matrix update events."""
        logger.info("Correlation matrix updated")
        
        correlation_matrix = event.data.get('correlation_matrix')
        if correlation_matrix:
            self.portfolio_state['correlation_matrix'] = correlation_matrix
            
            # Save to persistence
            self._update_portfolio_state()
    
    def handle_portfolio_update(self, event: Event):
        """Handle portfolio exposure update events."""
        logger.info("Portfolio exposure updated")
        
        positions = event.data.get('positions')
        if positions:
            self.portfolio_state['current_positions'] = positions
            
            # Save to persistence
            self._update_portfolio_state()
    
    def handle_drawdown_update(self, event: Event):
        """Handle drawdown events."""
        logger.info(f"Drawdown update: {event.data}")
        
        current_drawdown = event.data.get('current_drawdown')
        if current_drawdown is not None:
            self.portfolio_state['current_drawdown'] = current_drawdown
            
            # Save to persistence
            self._update_portfolio_state()
    
    def handle_trade_closed(self, event: Event):
        """Handle trade closed events to update recent trades."""
        logger.debug("Trade closed, updating recent trades")
        
        # Extract trade data
        trade_data = {
            'symbol': event.data.get('symbol'),
            'strategy_id': event.data.get('strategy_id'),
            'direction': event.data.get('direction'),
            'pnl': event.data.get('pnl', 0),
            'pnl_percent': event.data.get('pnl_percent', 0),
            'timestamp': event.data.get('timestamp', datetime.now()),
            'exit_price': event.data.get('exit_price'),
            'entry_price': event.data.get('entry_price')
        }
        
        # Add to recent trades
        self.portfolio_state['recent_trades'].append(trade_data)
        
        # Limit to last 50 trades
        self.portfolio_state['recent_trades'] = self.portfolio_state['recent_trades'][-50:]
        
        # Save to persistence
        self._update_portfolio_state()
    
    def _update_portfolio_state(self):
        """Update portfolio state in persistence."""
        self.persistence.save_strategy_state("adaptive_position_manager", {
            'market_state': self.market_state,
            'portfolio_state': self.portfolio_state
        })
    
    def _get_recent_trades_for_symbol(self, symbol: str, max_trades: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent trades for a specific symbol.
        
        Args:
            symbol: Symbol to filter trades for
            max_trades: Maximum number of trades to return
            
        Returns:
            List of recent trades for the symbol
        """
        # Filter to symbol and sort by timestamp (newest first)
        filtered_trades = [
            trade for trade in self.portfolio_state['recent_trades']
            if trade.get('symbol') == symbol
        ]
        
        # Sort by timestamp (newest first)
        filtered_trades.sort(key=lambda t: t.get('timestamp', datetime.min), reverse=True)
        
        # Return limited number
        return filtered_trades[:max_trades]
    
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss_pips: float, 
                              account_balance: float,
                              strategy_id: Optional[str] = None,
                              asset_class: str = 'forex') -> Dict[str, Any]:
        """
        Calculate adaptive position size based on current market conditions.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            account_balance: Account balance
            strategy_id: Strategy ID for tracking
            asset_class: Asset class (forex, stocks, etc.)
            
        Returns:
            Position sizing information with explanation
        """
        # Get appropriate position sizer for asset class
        if asset_class not in self.position_sizers:
            logger.warning(f"No position sizer for {asset_class}, using forex as fallback")
            asset_class = 'forex'
            
        position_sizer = self.position_sizers[asset_class]
        
        # Get recent trades for this symbol
        recent_trades = self._get_recent_trades_for_symbol(symbol)
        
        # Calculate position size with full context
        position_size = position_sizer.calculate_adaptive_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_pips=stop_loss_pips,
            account_balance=account_balance,
            recent_trades=recent_trades,
            market_regime=self.market_state['regime'],
            volatility_state=self.market_state['volatility'],
            current_positions=self.portfolio_state['current_positions'],
            correlation_matrix=self.portfolio_state['correlation_matrix'],
            current_drawdown=self.portfolio_state['current_drawdown']
        )
        
        # Generate explanation for the position size
        explanation = self._generate_position_size_explanation(
            symbol, entry_price, stop_loss_pips, account_balance, position_size, asset_class
        )
        
        # Create response with full information
        response = {
            'position_size': position_size,
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss_pips': stop_loss_pips,
            'account_balance': account_balance,
            'market_regime': self.market_state['regime'],
            'volatility_state': self.market_state['volatility'],
            'current_drawdown': self.portfolio_state['current_drawdown'],
            'explanation': explanation,
            'timestamp': datetime.now()
        }
        
        # Publish position size event
        self.event_bus.create_and_publish(
            EventType.POSITION_SIZE_CALCULATED,
            {
                'symbol': symbol,
                'position_size': position_size,
                'stop_loss_pips': stop_loss_pips,
                'strategy_id': strategy_id,
                'market_regime': self.market_state['regime']
            }
        )
        
        logger.info(f"Calculated position size for {symbol}: {position_size} lots")
        return response
    
    def _generate_position_size_explanation(self,
                                          symbol: str,
                                          entry_price: float,
                                          stop_loss_pips: float,
                                          account_balance: float,
                                          position_size: float,
                                          asset_class: str) -> str:
        """
        Generate human-readable explanation for position size calculation.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            account_balance: Account balance
            position_size: Calculated position size
            asset_class: Asset class
            
        Returns:
            Explanation text
        """
        # Get base risk percentage
        base_risk = self.position_sizers[asset_class].parameters['max_risk_per_trade_percent']
        
        # Calculate risk amount
        pip_value = 10.0  # Approximation for standard lot
        risk_amount = position_size * stop_loss_pips * pip_value
        risk_percent = (risk_amount / account_balance) * 100
        
        # Format explanation
        parts = [
            f"Position size for {symbol}: {position_size:.2f} lots",
            f"Market regime: {self.market_state['regime']} with {self.market_state['volatility']} volatility",
            f"Risk amount: ${risk_amount:.2f} ({risk_percent:.2f}% of ${account_balance:.2f})",
            f"Stop loss: {stop_loss_pips} pips from entry at {entry_price}"
        ]
        
        # Add adjustments explanation
        if self.market_state['regime'] != 'stable':
            if self.market_state['regime'] in ['volatile', 'ranging']:
                parts.append(f"Position reduced due to {self.market_state['regime']} market conditions")
            elif self.market_state['regime'] in ['trending', 'bull_trend', 'bear_trend']:
                parts.append(f"Position increased due to favorable {self.market_state['regime']} market conditions")
        
        # Add drawdown explanation if applicable
        if self.portfolio_state['current_drawdown'] > 5.0:
            parts.append(f"Position reduced due to current drawdown of {self.portfolio_state['current_drawdown']:.2f}%")
        
        # Add correlation explanation if applicable
        has_correlated_positions = any(
            p.get('symbol') != symbol and p.get('correlation', 0) > 0.5
            for p in self.portfolio_state['current_positions']
        )
        if has_correlated_positions:
            parts.append("Position reduced due to correlation with existing positions")
        
        return "\n".join(parts)
    
    def load_state(self):
        """Load state from persistence."""
        state = self.persistence.load_strategy_state("adaptive_position_manager") or {}
        
        if 'market_state' in state:
            self.market_state = state['market_state']
            
        if 'portfolio_state' in state:
            self.portfolio_state = state['portfolio_state']
            
        logger.info("Loaded state for Adaptive Position Manager")
    
    def get_market_state(self) -> Dict[str, Any]:
        """
        Get current market state.
        
        Returns:
            Current market state information
        """
        return self.market_state.copy()
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Current portfolio state information
        """
        return self.portfolio_state.copy()
