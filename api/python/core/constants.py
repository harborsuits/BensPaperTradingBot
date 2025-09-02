"""
Core constants for the trading bot system, including event types,
trading modes, and other configuration constants.
"""
from enum import Enum, auto

class EventType(str, Enum):
    """Event types used in the event-driven architecture."""
    
    # Market data events
    MARKET_DATA_RECEIVED = "market_data_received"
    TICK_RECEIVED = "tick_received"
    BAR_CLOSED = "bar_closed"
    
    # Order events
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Strategy events
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"
    PATTERN_DISCOVERED = "pattern_discovered"
    PATTERN_CONFIRMED = "pattern_confirmed"
    PATTERN_FAILED = "pattern_failed"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    ERROR_OCCURRED = "error_occurred"
    LOG_MESSAGE = "log_message"
    END_OF_DAY = "end_of_day"
    
    # Backtesting events
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_PROGRESS = "backtest_progress"
    
    # Health and monitoring events
    HEALTH_CHECK = "health_check"
    HEALTH_STATUS_CHANGED = "health_status_changed"
    COMPONENT_RESTARTED = "component_restarted"
    
    # Strategy Intelligence events
    MARKET_REGIME_CHANGED = "market_regime_changed"
    MARKET_REGIME_DETECTED = "market_regime_detected"
    ASSET_CLASS_SELECTED = "asset_class_selected"
    SYMBOL_SELECTED = "symbol_selected"
    SYMBOL_RANKED = "symbol_ranked"
    STRATEGY_SELECTED = "strategy_selected"
    STRATEGY_COMPATIBILITY_UPDATED = "strategy_compatibility_updated"
    PERFORMANCE_ATTRIBUTED = "performance_attributed"
    EXECUTION_QUALITY_MEASURED = "execution_quality_measured"
    STRATEGY_PARAMETER_ADJUSTED = "strategy_parameter_adjusted"
    STRATEGY_PROMOTED = "strategy_promoted"
    STRATEGY_RETIRED = "strategy_retired"
    CORRELATION_MATRIX_UPDATED = "correlation_matrix_updated"
    
    # Risk management events
    RISK_LIMIT_REACHED = "risk_limit_reached"
    DRAWDOWN_ALERT = "drawdown_alert"
    CAPITAL_ADJUSTED = "capital_adjusted"
    POSITION_SIZE_CALCULATED = "position_size_calculated"
    RISK_ALLOCATION_CHANGED = "risk_allocation_changed"
    PORTFOLIO_EXPOSURE_UPDATED = "portfolio_exposure_updated"
    CORRELATION_RISK_ALERT = "correlation_risk_alert"
    DRAWDOWN_THRESHOLD_EXCEEDED = "drawdown_threshold_exceeded"
    RISK_ATTRIBUTION_CALCULATED = "risk_attribution_calculated"
    
    # Mode and approval events
    MODE_CHANGED = "mode_changed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    
    # Workflow and paper/live transition events
    WORKFLOW_EVENT = "workflow_event"  # Generic workflow event
    APPROVAL_EVENT = "approval_event"  # Generic approval event
    STRATEGY_REGISTERED = "strategy_registered"  # New strategy registered
    ORDER_PLACED = "order_placed"  # Order placed by strategy
    POSITION_UPDATE = "position_update"  # Position updated
    PAPER_TO_LIVE_TRANSITION = "paper_to_live_transition"  # Strategy transitioning from paper to live
    POSITION_TRANSITION = "position_transition"  # Position handling during transition

class TradingMode(str, Enum):
    """Trading modes for the system."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    STOPPED = "stopped"

class HealthStatus(str, Enum):
    """Health status indicators for system components."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    UNKNOWN = "unknown"

class ApprovalStatus(str, Enum):
    """Approval status for live trading."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NOT_REQUESTED = "not_requested"

# Configuration constants
DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_MONGODB_DATABASE = "bensbot"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_API_BASE_URL = "http://localhost:8000"
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 5
HEALTH_CHECK_INTERVAL_SECONDS = 60
WATCHDOG_CHECK_INTERVAL_SECONDS = 30
