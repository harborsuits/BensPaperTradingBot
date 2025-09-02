"""
Strategy Enhancement Module for LLM-enhanced Trading

This module integrates LLM insights with ML models to enhance trading strategies,
providing dynamic parameter adjustments, risk management, and strategy learning.
"""

import os
import json
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import internal components
from .financial_llm_engine import FinancialLLMEngine
from .memory_system import MemorySystem, MemoryType
from .reasoning_engine import ReasoningEngine, ReasoningTask, ReasoningResult

# Initialize logger
logger = logging.getLogger("strategy_enhancement")

class EnhancementType(Enum):
    """Types of strategy enhancements"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    RISK_MODIFICATION = "risk_modification"
    SIGNAL_FILTER = "signal_filter"
    POSITION_SIZING = "position_sizing"
    ENTRY_TIMING = "entry_timing"
    EXIT_TIMING = "exit_timing"
    TRADING_PAUSE = "trading_pause"
    
@dataclass
class StrategyAdjustment:
    """Represents a specific adjustment to a trading strategy"""
    adjustment_type: EnhancementType
    parameter_name: str
    original_value: Any
    new_value: Any
    confidence: float  # 0.0 to 1.0
    explanation: str
    source: str  # "llm", "ml", "consensus", "human"
    timestamp: float
    expiration: Optional[float] = None  # When this adjustment expires
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "adjustment_type": self.adjustment_type.value,
            "parameter_name": self.parameter_name,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "source": self.source,
            "timestamp": self.timestamp,
            "expiration": self.expiration,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        date_str = datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M")
        return (f"StrategyAdjustment({self.adjustment_type.value}, {self.parameter_name}, "
                f"{self.original_value} → {self.new_value}, conf={self.confidence:.2f}, "
                f"source={self.source}, time={date_str})")
    
    @property
    def is_expired(self) -> bool:
        """Check if adjustment has expired"""
        if not self.expiration:
            return False
        return datetime.datetime.now().timestamp() > self.expiration

class StrategyEnhancer:
    """
    Enhances trading strategies with LLM insights and ML model integration
    
    Features:
    - Dynamic parameter adjustments based on market regime
    - Risk management enhancements
    - Anomaly detection and response
    - Strategy reflection and learning
    """
    
    def __init__(
        self,
        reasoning_engine: Optional[ReasoningEngine] = None,
        memory_system: Optional[MemorySystem] = None,
        ml_regime_detector = None,  # Will connect to MLRegimeDetector
        portfolio_optimizer = None,  # Will connect to PortfolioOptimizer
        regime_change_notifier = None,  # Will connect to RegimeChangeNotifier
        confidence_threshold: float = 0.7,
        max_adjustment_history: int = 100,
        adjustment_cooldown: int = 3600,  # Seconds between similar adjustments
        debug: bool = False
    ):
        """
        Initialize the strategy enhancer
        
        Args:
            reasoning_engine: Engine for LLM reasoning
            memory_system: Memory system for context
            ml_regime_detector: ML-based regime detector
            portfolio_optimizer: Portfolio optimization system
            regime_change_notifier: Notification system for regime changes
            confidence_threshold: Minimum confidence for adjustments
            max_adjustment_history: Maximum adjustments to track
            adjustment_cooldown: Cooldown period between similar adjustments (seconds)
            debug: Enable debug logging
        """
        # Set up logging
        logging_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=logging_level)
        self.debug = debug
        
        # Store components
        self.reasoning_engine = reasoning_engine
        self.memory_system = memory_system
        self.ml_regime_detector = ml_regime_detector
        self.portfolio_optimizer = portfolio_optimizer
        self.regime_change_notifier = regime_change_notifier
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.max_adjustment_history = max_adjustment_history
        self.adjustment_cooldown = adjustment_cooldown
        
        # Initialize state
        self.active_adjustments = {}  # key: parameter_name, value: StrategyAdjustment
        self.adjustment_history = []  # List of all adjustments
        self.anomaly_detected = False
        self.trading_paused = False
        self.pause_reason = None
        self.current_regime = "unknown"
        self.regime_confidence = 0.0
        
        # Parameter bounds to ensure safe operation
        self.parameter_bounds = {
            # Default bounds for common parameters
            "stop_loss_pct": (0.005, 0.15),      # 0.5% to 15%
            "take_profit_pct": (0.01, 0.5),      # 1% to 50%
            "max_position_size": (0.01, 0.3),    # 1% to 30% of portfolio
            "entry_threshold": (0.6, 0.95),      # Signal strength threshold
            "exit_threshold": (0.4, 0.9),        # Exit signal threshold
            # Add more parameter bounds as needed
        }
        
        # Register for regime change events if notifier exists
        if self.regime_change_notifier:
            try:
                self.regime_change_notifier.register_callback(self._on_regime_change)
                logger.info("Registered for regime change notifications")
            except Exception as e:
                logger.error(f"Failed to register for regime change notifications: {e}")
                
        logger.info("Strategy enhancer initialized")

    def update_regime_state(self):
        """Update the current regime state from the detector"""
        if not self.ml_regime_detector:
            return
            
        try:
            self.current_regime = self.ml_regime_detector.current_regime
            self.regime_confidence = self.ml_regime_detector.regime_confidence
            logger.info(f"Updated regime state: {self.current_regime} (conf: {self.regime_confidence:.2f})")
        except Exception as e:
            logger.error(f"Error updating regime state: {e}")

    def _on_regime_change(self, old_regime: str, new_regime: str, confidence: float):
        """Handle regime change notification"""
        logger.info(f"Regime change detected: {old_regime} → {new_regime} (conf: {confidence:.2f})")
        
        self.current_regime = new_regime
        self.regime_confidence = confidence
        
        # Clear expired adjustments
        self._clear_expired_adjustments()
        
        # Generate new regime-specific adjustments
        if confidence >= self.confidence_threshold:
            self._generate_regime_adjustments(old_regime, new_regime, confidence)
    
    def _clear_expired_adjustments(self):
        """Remove expired adjustments"""
        expired_params = []
        for param_name, adjustment in self.active_adjustments.items():
            if adjustment.is_expired:
                expired_params.append(param_name)
                logger.info(f"Adjustment expired: {adjustment}")
                
        for param in expired_params:
            del self.active_adjustments[param]
    
    def _generate_regime_adjustments(self, old_regime: str, new_regime: str, confidence: float):
        """Generate strategy adjustments for a new regime"""
        if not self.reasoning_engine:
            logger.warning("Cannot generate regime adjustments: reasoning engine not available")
            return
            
        # Prepare context with regime information
        context = {
            "old_regime": old_regime,
            "new_regime": new_regime,
            "regime_confidence": confidence,
            "market_volatility": self._get_market_volatility()
        }
        
        # Get ML signals
        ml_signals = {
            "regime": new_regime,
            "confidence": confidence,
            "direction": "unknown"  # Will be replaced with actual ML direction
        }
        
        # Add regime characteristics if available
        if self.ml_regime_detector and hasattr(self.ml_regime_detector, "get_regime_characteristics"):
            try:
                ml_signals["regime_characteristics"] = self.ml_regime_detector.get_regime_characteristics(new_regime)
            except Exception as e:
                logger.error(f"Error getting regime characteristics: {e}")
        
        # Get reasoning about strategy adjustments
        try:
            result = self.reasoning_engine.reason_sync(
                task=ReasoningTask.STRATEGY_ADJUSTMENT,
                ml_signals=ml_signals,
                context=context
            )
            
            # Process the result to extract parameter adjustments
            self._process_adjustment_reasoning(result)
            
        except Exception as e:
            logger.error(f"Error generating regime adjustments: {e}")
    
    def _get_market_volatility(self) -> float:
        """Get current market volatility estimate"""
        # Placeholder - would connect to your market data system
        return 0.15  # Default moderate volatility
    
    def _process_adjustment_reasoning(self, result: ReasoningResult):
        """Process reasoning result for strategy adjustments"""
        # Skip if confidence is too low
        if result.confidence < self.confidence_threshold:
            logger.info(f"Skipping adjustments due to low confidence: {result.confidence:.2f}")
            return
            
        # Extract adjustment recommendations from LLM signals
        adjustments = []
        
        # Look for specific parameter adjustments in the explanation
        import re
        
        # Pattern for parameter adjustments: Parameter: value → new_value (reason)
        pattern = r"([A-Za-z_]+):\s*([0-9.]+)\s*→\s*([0-9.]+)\s*\(([^)]+)\)"
        matches = re.findall(pattern, result.explanation)
        
        for match in matches:
            param_name, old_value, new_value, reason = match
            param_name = param_name.lower().strip()
            
            try:
                old_val = float(old_value)
                new_val = float(new_value)
                
                # Check bounds if defined for this parameter
                if param_name in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param_name]
                    new_val = max(min_val, min(max_val, new_val))
                
                # Create adjustment
                adjustment = StrategyAdjustment(
                    adjustment_type=self._get_adjustment_type(param_name),
                    parameter_name=param_name,
                    original_value=old_val,
                    new_value=new_val,
                    confidence=result.confidence,
                    explanation=reason.strip(),
                    source="consensus" if result.consensus_score > 0.5 else "llm",
                    timestamp=datetime.datetime.now().timestamp(),
                    expiration=datetime.datetime.now().timestamp() + 86400,  # 24 hour default
                    metadata={
                        "regime": self.current_regime,
                        "consensus_score": result.consensus_score
                    }
                )
                
                adjustments.append(adjustment)
                
            except ValueError:
                logger.warning(f"Invalid parameter values: {old_value} → {new_value}")
        
        # Apply the adjustments
        for adjustment in adjustments:
            self._apply_adjustment(adjustment)
            
        logger.info(f"Processed {len(adjustments)} parameter adjustments")
    
    def _get_adjustment_type(self, param_name: str) -> EnhancementType:
        """Map parameter name to adjustment type"""
        param_name = param_name.lower()
        
        if "stop" in param_name or "sl" in param_name:
            return EnhancementType.RISK_MODIFICATION
        elif "take_profit" in param_name or "tp" in param_name:
            return EnhancementType.RISK_MODIFICATION
        elif "size" in param_name or "position" in param_name:
            return EnhancementType.POSITION_SIZING
        elif "entry" in param_name:
            return EnhancementType.ENTRY_TIMING
        elif "exit" in param_name:
            return EnhancementType.EXIT_TIMING
        else:
            return EnhancementType.PARAMETER_ADJUSTMENT
    
    def _apply_adjustment(self, adjustment: StrategyAdjustment):
        """Apply a strategy adjustment"""
        # Check for cooldown period
        if adjustment.parameter_name in self.active_adjustments:
            last_adjustment = self.active_adjustments[adjustment.parameter_name]
            time_since_last = adjustment.timestamp - last_adjustment.timestamp
            
            if time_since_last < self.adjustment_cooldown:
                logger.info(f"Skipping adjustment due to cooldown: {adjustment.parameter_name}")
                return
        
        # Store the adjustment
        self.active_adjustments[adjustment.parameter_name] = adjustment
        
        # Add to history
        self.adjustment_history.append(adjustment)
        if len(self.adjustment_history) > self.max_adjustment_history:
            self.adjustment_history.pop(0)
            
        # Log the adjustment
        logger.info(f"Applied adjustment: {adjustment}")
        
        # Save to memory if available
        if self.memory_system:
            content = (f"Strategy Adjustment: {adjustment.parameter_name} changed from "
                     f"{adjustment.original_value} to {adjustment.new_value}\n"
                     f"Reason: {adjustment.explanation}")
                     
            self.memory_system.add_memory(
                content=content,
                memory_type=MemoryType.SHORT_TERM,
                importance=0.7,
                tags=["strategy_adjustment", self.current_regime, adjustment.adjustment_type.value],
                source="strategy_enhancer",
                metadata=adjustment.to_dict()
            )
    
    def get_enhanced_parameters(
        self,
        strategy_parameters: Dict[str, Any],
        strategy_type: str,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get enhanced strategy parameters with LLM adjustments
        
        Args:
            strategy_parameters: Original parameters
            strategy_type: Type of strategy
            symbol: Trading symbol
            timeframe: Trading timeframe
            
        Returns:
            Enhanced parameters
        """
        # Update regime state
        self.update_regime_state()
        
        # Check if trading is paused
        if self.trading_paused:
            logger.warning(f"Trading is paused: {self.pause_reason}")
            return {**strategy_parameters, "trading_enabled": False}
        
        # Clear expired adjustments
        self._clear_expired_adjustments()
        
        # Create a copy of original parameters
        enhanced_params = strategy_parameters.copy()
        
        # Apply active adjustments
        adjustments_applied = []
        
        for param_name, adjustment in self.active_adjustments.items():
            if param_name in enhanced_params:
                # Store original for logging
                original = enhanced_params[param_name]
                
                # Apply the adjustment
                enhanced_params[param_name] = adjustment.new_value
                
                # Record for logging
                adjustments_applied.append(f"{param_name}: {original} → {adjustment.new_value}")
        
        if adjustments_applied:
            logger.info(f"Enhanced parameters for {strategy_type}/{symbol}/{timeframe}: {', '.join(adjustments_applied)}")
        
        return enhanced_params
    
    def check_for_anomalies(
        self,
        market_data: Dict[str, Any],
        strategy_signals: Dict[str, Any],
        symbol: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for market anomalies that might require strategy adjustments
        
        Args:
            market_data: Current market data
            strategy_signals: Strategy signals
            symbol: Trading symbol
            
        Returns:
            (anomaly_detected, explanation) tuple
        """
        if not self.reasoning_engine:
            return False, None
            
        # Prepare context
        context = {
            "symbol": symbol,
            "current_regime": self.current_regime,
            "market_data": json.dumps(market_data),
            "strategy_signals": json.dumps(strategy_signals)
        }
        
        # ML signals for anomaly detection
        ml_signals = {
            "symbol": symbol,
            "regime": self.current_regime,
            "confidence": self.regime_confidence
        }
        
        # Use reasoning engine to detect anomalies
        try:
            result = self.reasoning_engine.reason_sync(
                task=ReasoningTask.ANOMALY_DETECTION,
                ml_signals=ml_signals,
                context=context
            )
            
            # Check if anomaly was detected with high confidence
            anomaly_detected = "anomaly" in result.conclusion.lower() and result.confidence > 0.8
            
            if anomaly_detected:
                logger.warning(f"Market anomaly detected: {result.conclusion}")
                self.anomaly_detected = True
                
                # Consider pausing trading if severe anomaly
                if "severe" in result.conclusion.lower() or "extreme" in result.conclusion.lower():
                    self._pause_trading(result.conclusion)
                
                return True, result.conclusion
            
            # Reset anomaly state if no anomaly detected
            self.anomaly_detected = False
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking for anomalies: {e}")
            return False, None
    
    def _pause_trading(self, reason: str):
        """Pause trading due to unusual conditions"""
        self.trading_paused = True
        self.pause_reason = reason
        logger.warning(f"Trading paused: {reason}")
        
        # Notify if available
        if self.regime_change_notifier:
            try:
                self.regime_change_notifier.send_notification(
                    title="TRADING PAUSED - ANOMALY DETECTED",
                    message=f"Trading has been automatically paused: {reason}",
                    priority="high"
                )
            except Exception as e:
                logger.error(f"Error sending pause notification: {e}")
    
    def resume_trading(self):
        """Resume trading after pause"""
        self.trading_paused = False
        self.pause_reason = None
        logger.info("Trading resumed")
        
        # Notify if available
        if self.regime_change_notifier:
            try:
                self.regime_change_notifier.send_notification(
                    title="Trading Resumed",
                    message="Automated trading has been resumed",
                    priority="medium"
                )
            except Exception as e:
                logger.error(f"Error sending resume notification: {e}")
    
    def analyze_trade_result(
        self,
        symbol: str,
        entry_time: datetime.datetime,
        exit_time: datetime.datetime,
        entry_price: float,
        exit_price: float,
        position_size: float,
        profit_loss: float,
        trade_metadata: Dict[str, Any]
    ):
        """
        Analyze completed trade for learning and future improvement
        
        Args:
            symbol: Trading symbol
            entry_time: Trade entry time
            exit_time: Trade exit time
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            profit_loss: Profit/loss amount
            trade_metadata: Additional trade metadata
        """
        if not self.reasoning_engine or not self.memory_system:
            return
            
        # Prepare context
        context = {
            "symbol": symbol,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "profit_loss": profit_loss,
            "percent_return": (exit_price - entry_price) / entry_price * 100,
            "trade_duration_hours": (exit_time - entry_time).total_seconds() / 3600,
            "regime": trade_metadata.get("regime", "unknown")
        }
        
        # ML signals
        ml_signals = {
            "symbol": symbol,
            "profit_loss": profit_loss,
            "success": profit_loss > 0
        }
        
        # Get active adjustments at time of trade
        active_adjustments = []
        for adj in self.adjustment_history:
            if adj.timestamp <= entry_time.timestamp() and (not adj.expiration or adj.expiration >= exit_time.timestamp()):
                active_adjustments.append(adj.to_dict())
                
        context["active_adjustments"] = active_adjustments
        
        # Use reasoning engine for trade analysis
        try:
            result = self.reasoning_engine.reason_sync(
                task=ReasoningTask.TRADING_REFLECTION,
                ml_signals=ml_signals,
                context=context
            )
            
            # Store the reflection in memory
            self.memory_system.add_memory(
                content=result.explanation,
                memory_type=MemoryType.MEDIUM_TERM,
                importance=0.8 if abs(profit_loss) > 100 else 0.6,
                tags=["trade_reflection", symbol, "success" if profit_loss > 0 else "failure"],
                source="trade_analysis",
                metadata={
                    "symbol": symbol,
                    "entry_time": entry_time.isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "profit_loss": profit_loss,
                    "conclusion": result.conclusion
                }
            )
            
            logger.info(f"Stored trade reflection for {symbol}: {result.conclusion}")
            
        except Exception as e:
            logger.error(f"Error analyzing trade result: {e}")
    
    def suggest_new_strategies(self) -> List[Dict[str, Any]]:
        """
        Suggest new trading strategies based on recent market conditions
        and memory of successful trades
        
        Returns:
            List of strategy suggestions
        """
        if not self.reasoning_engine or not self.memory_system:
            return []
            
        # Get successful trade reflections from memory
        success_memories = self.memory_system.query_memories(
            tags=["trade_reflection", "success"],
            min_importance=0.7,
            limit=10
        )
        
        if not success_memories:
            logger.info("No successful trade memories found for strategy suggestions")
            return []
            
        # Prepare context with successful patterns
        success_patterns = "\n\n".join([
            f"Symbol: {mem.metadata.get('symbol')}\n"
            f"Date: {mem.metadata.get('entry_time')}\n"
            f"Reflection: {mem.content}"
            for mem in success_memories
        ])
        
        context = {
            "current_regime": self.current_regime,
            "successful_patterns": success_patterns
        }
        
        # ML signals
        ml_signals = {
            "regime": self.current_regime,
            "confidence": self.regime_confidence
        }
        
        # Use reasoning engine to suggest strategies
        try:
            result = self.reasoning_engine.reason_sync(
                task=ReasoningTask.MARKET_HYPOTHESIS,
                ml_signals=ml_signals,
                context=context
            )
            
            # Parse strategy suggestions
            suggestions = []
            
            # Extract strategy suggestions using regex
            import re
            pattern = r"Strategy\s+(\d+):\s+([^\n]+)\n+Parameters:\s+([^\n]+)\n+Rationale:\s+([^\n]+)"
            matches = re.findall(pattern, result.explanation, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                strategy_num, name, params, rationale = match
                
                suggestion = {
                    "name": name.strip(),
                    "parameters": params.strip(),
                    "rationale": rationale.strip(),
                    "confidence": result.confidence,
                    "regime": self.current_regime
                }
                
                suggestions.append(suggestion)
            
            logger.info(f"Generated {len(suggestions)} strategy suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating strategy suggestions: {e}")
            return []
    
    def get_adjustment_history(
        self,
        parameter_name: Optional[str] = None,
        adjustment_type: Optional[EnhancementType] = None,
        limit: int = 20
    ) -> List[StrategyAdjustment]:
        """
        Get history of strategy adjustments
        
        Args:
            parameter_name: Filter by parameter name
            adjustment_type: Filter by adjustment type
            limit: Maximum results to return
            
        Returns:
            List of adjustments
        """
        filtered = self.adjustment_history.copy()
        
        if parameter_name:
            filtered = [adj for adj in filtered if adj.parameter_name == parameter_name]
            
        if adjustment_type:
            filtered = [adj for adj in filtered if adj.adjustment_type == adjustment_type]
            
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda adj: adj.timestamp, reverse=True)
        
        return filtered[:limit]
