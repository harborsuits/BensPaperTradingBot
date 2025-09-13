import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import interfaces from parent modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from strategy_management.interfaces import CoreContext, MarketContext

logger = logging.getLogger("context_decision_integration")

class ContextDecisionIntegration:
    """
    Integrates unified context with decision-making systems to ensure
    all trading decisions incorporate both stats and sentiment data.
    """
    
    def __init__(self, core_context, unified_context_manager, strategy_rotator, config: Dict[str, Any]):
        self.core_context = core_context
        self.unified_context = unified_context_manager
        self.strategy_rotator = strategy_rotator
        self.config = config
        
        # Automatic rotation settings
        self.auto_rotation_enabled = config.get("auto_rotation_enabled", True)
        self.auto_rotation_interval_days = config.get("auto_rotation_interval_days", 7)
        
        # Signal response settings
        self.respond_to_unified_signals = config.get("respond_to_unified_signals", True)
        self.unified_signal_threshold = config.get("unified_signal_threshold", 0.7)
        
        # Risk-based rotation settings
        self.risk_triggered_rotation = config.get("risk_triggered_rotation", True)
        self.risk_threshold = config.get("risk_threshold", "elevated")
        
        # Background tasks
        self.tasks = []
        self.running = False
        
        # Register for signal and risk updates
        self.core_context.add_event_listener("signal_added", self._on_signal_update)
        self.core_context.add_event_listener("risk_updated", self._on_risk_update)
    
    async def start(self):
        """Start the integration module"""
        if self.running:
            return
        
        self.running = True
        
        # Start auto-rotation task if enabled
        if self.auto_rotation_enabled:
            task = asyncio.create_task(self._auto_rotation_loop())
            self.tasks.append(task)
            
        logger.info("Started context-decision integration module")
    
    async def stop(self):
        """Stop the integration module"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks = []
        
        logger.info("Stopped context-decision integration module")
    
    async def _auto_rotation_loop(self):
        """Background task for automatic strategy rotation"""
        try:
            while self.running:
                # Check if rotation is needed
                if self.strategy_rotator.should_rotate():
                    logger.info("Performing scheduled strategy rotation")
                    # Get latest unified context
                    context_summary = self.unified_context.get_context_summary()
                    
                    # Use unified context to inform rotation
                    market_bias = context_summary.get("market_bias", "neutral")
                    
                    # Update market context with unified signal bias
                    self.core_context.update_market_context({
                        "signal_bias": market_bias
                    })
                    
                    # Perform the rotation
                    rotation_result = self.strategy_rotator.rotate_strategies(force=True)
                    
                    if rotation_result.get("rotated", False):
                        logger.info("Auto-rotation completed successfully")
                    else:
                        logger.warning(f"Auto-rotation failed: {rotation_result.get('message', 'Unknown error')}")
                
                # Wait until next check
                await asyncio.sleep(3600)  # Check hourly
                
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            logger.info("Auto-rotation task cancelled")
        except Exception as e:
            logger.error(f"Error in auto-rotation task: {str(e)}")
    
    def _on_signal_update(self, signal):
        """Handle new or updated unified signals"""
        if not self.respond_to_unified_signals:
            return
        
        try:
            # Get active unified signals
            active_signals = self.unified_context.get_active_unified_signals(
                min_confidence=self.unified_signal_threshold
            )
            
            # Check for significant signal changes
            signal_count = len(active_signals)
            
            if signal_count >= 5:
                # Many active signals - might indicate a significant market event
                logger.info(f"Large number of active unified signals detected ({signal_count})")
                
                # Get the signal direction bias
                long_count = sum(1 for s in active_signals.values() if s["direction"] == "long")
                short_count = sum(1 for s in active_signals.values() if s["direction"] == "short")
                
                if long_count > short_count * 2:
                    bias = "strongly bullish"
                elif long_count > short_count:
                    bias = "moderately bullish"
                elif short_count > long_count * 2:
                    bias = "strongly bearish"
                elif short_count > long_count:
                    bias = "moderately bearish"
                else:
                    bias = "neutral"
                
                # If significant bias, consider triggering rotation
                if bias in ["strongly bullish", "strongly bearish"]:
                    logger.info(f"Significant signal bias detected: {bias}")
                    
                    # Update market context
                    self.core_context.update_market_context({
                        "signal_bias": bias
                    })
                    
                    # Consider triggering rotation if bias is strong
                    # This will depend on time since last rotation
                    days_since_rotation = (datetime.now() - self.strategy_rotator.last_rotation).total_seconds() / 86400
                    
                    if days_since_rotation > 2:  # At least 2 days since last rotation
                        logger.info("Triggering signal-based rotation due to significant signal bias")
                        self.strategy_rotator.rotate_strategies(force=True)
        except Exception as e:
            logger.error(f"Error handling signal update: {str(e)}")
    
    def _on_risk_update(self, risk_state):
        """Handle risk state updates"""
        if not self.risk_triggered_rotation:
            return
        
        try:
            risk_level = risk_state.overall_risk_level
            
            # Define risk level hierarchy
            risk_levels = ["low", "normal", "elevated", "high"]
            
            # Check if risk exceeds threshold
            if risk_levels.index(risk_level) >= risk_levels.index(self.risk_threshold):
                logger.info(f"Risk level {risk_level} exceeds threshold {self.risk_threshold}")
                
                # See if we need to rotate based on risk
                days_since_rotation = (datetime.now() - self.strategy_rotator.last_rotation).total_seconds() / 86400
                
                if days_since_rotation > 1:  # At least 1 day since last rotation
                    logger.info("Triggering risk-based rotation due to elevated risk level")
                    self.strategy_rotator.rotate_strategies(force=True)
        except Exception as e:
            logger.error(f"Error handling risk update: {str(e)}")
    
    def manual_process_current_context(self):
        """
        Manually process the current unified context to make immediate decisions.
        This is useful for testing or one-off decision making.
        """
        try:
            # Get context summary
            context_summary = self.unified_context.get_context_summary()
            
            # Update market context
            self.core_context.update_market_context({
                "signal_bias": context_summary.get("market_bias", "neutral")
            })
            
            # Perform a rotation based on current context
            rotation_result = self.strategy_rotator.rotate_strategies(force=True)
            
            return {
                "context_summary": context_summary,
                "rotation_result": rotation_result
            }
        except Exception as e:
            logger.error(f"Error in manual context processing: {str(e)}")
            return {
                "error": str(e)
            }
    
    def get_unified_context_summary(self):
        """Get the latest unified context summary"""
        try:
            return self.unified_context.get_context_summary()
        except Exception as e:
            logger.error(f"Error getting unified context summary: {str(e)}")
            return {
                "error": str(e),
                "market_bias": "neutral",
                "confidence": 0.0
            }
    
    def process_news_impact(self, news_items):
        """
        Process news items to extract market impact and update context
        
        Args:
            news_items: List of news item dictionaries
            
        Returns:
            Dict with processed impact information
        """
        try:
            if not news_items:
                return {"processed": 0, "impact": "none"}
            
            # Extract sentiment and topics from news
            sentiments = []
            topics = set()
            
            for item in news_items:
                # Extract sentiment
                sentiment = item.get("sentiment", "neutral")
                if sentiment == "Positive":
                    sentiments.append(1.0)
                elif sentiment == "Negative":
                    sentiments.append(-1.0)
                else:
                    sentiments.append(0.0)
                
                # Extract topics
                title = item.get("title", "")
                summary = item.get("summary", "")
                content = f"{title} {summary}".lower()
                
                # Check for major topics
                if any(word in content for word in ["fed", "interest rate", "federal reserve", "monetary"]):
                    topics.add("monetary_policy")
                if any(word in content for word in ["inflation", "cpi", "price index", "deflation"]):
                    topics.add("inflation")
                if any(word in content for word in ["recession", "gdp", "growth", "economy"]):
                    topics.add("economic_growth")
                if any(word in content for word in ["earnings", "revenue", "profit", "quarterly"]):
                    topics.add("earnings")
                if any(word in content for word in ["war", "conflict", "geopolitical", "tension"]):
                    topics.add("geopolitical")
                
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Determine overall impact
            impact = "none"
            if avg_sentiment > 0.5:
                impact = "positive"
            elif avg_sentiment < -0.5:
                impact = "negative"
            elif avg_sentiment > 0.2:
                impact = "slightly_positive"
            elif avg_sentiment < -0.2:
                impact = "slightly_negative"
            else:
                impact = "neutral"
            
            # Update unified context with news impact
            if hasattr(self.unified_context, "update_news_impact"):
                self.unified_context.update_news_impact({
                    "sentiment": avg_sentiment,
                    "impact": impact,
                    "topics": list(topics)
                })
            
            return {
                "processed": len(news_items),
                "impact": impact,
                "avg_sentiment": avg_sentiment,
                "topics": list(topics)
            }
        except Exception as e:
            logger.error(f"Error processing news impact: {str(e)}")
            return {"error": str(e), "processed": 0, "impact": "none"} 