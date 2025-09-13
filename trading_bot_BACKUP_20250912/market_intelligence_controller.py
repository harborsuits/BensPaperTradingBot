"""
Market Intelligence Controller - Integration Hub
This module serves as the central integration point for the Market Intelligence Center,
providing a unified interface for all components and coordinating data flow between them.
"""

import os
import sys
import time
import json
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
from trading_bot.market_context.market_context import get_market_context
from trading_bot.symbolranker.symbol_ranker import get_symbol_ranker
from trading_bot.ml_pipeline.adaptive_trainer import get_adaptive_trainer, get_training_scheduler

class MarketIntelligenceController:
    """
    Central controller that integrates all components of the Market Intelligence Center
    and coordinates data flow between them.
    """
    
    def __init__(self, config=None):
        """
        Initialize the controller with configuration.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("MarketIntelligenceController")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Access our singletons
        self.market_context = get_market_context()
        self.symbol_ranker = get_symbol_ranker()
        self.ml_trainer = get_adaptive_trainer()
        self.training_scheduler = get_training_scheduler()
        
        # Whether the system is initialized
        self._initialized = False
        
        # Thread for background processing
        self._background_thread = None
        self._running = False
        self._stop_event = threading.Event()
        
        # Metadata about the controller
        self.metadata = {
            "start_time": None,
            "last_update": None,
            "status": "initializing"
        }
        
        self.logger.info("MarketIntelligenceController initialized")
    
    def initialize(self, symbols=None):
        """
        Initialize the Market Intelligence Center with initial data.
        
        Args:
            symbols: List of stock symbols to initialize with or None for defaults
            
        Returns:
            Boolean indicating success
        """
        self.logger.info("Initializing Market Intelligence Center")
        
        try:
            # Default symbols if none provided
            if symbols is None:
                symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "JPM", "SPY", "QQQ"]
            
            # Update market context with initial data
            self.market_context.update_market_data()
            self.market_context.update_symbol_data(symbols)
            self.market_context.update_strategy_rankings()
            self.market_context.match_symbols_to_strategies()
            
            # Save initial state
            self.market_context.save_to_file("data/market_context/latest_context.json")
            
            # Initialize ML pipeline
            self.ml_trainer.train_all_models()
            
            # Start the training scheduler
            self.training_scheduler.start()
            
            # Start background processing
            self._start_background_processing()
            
            # Update metadata
            now = datetime.datetime.now()
            self.metadata["start_time"] = now.isoformat()
            self.metadata["last_update"] = now.isoformat()
            self.metadata["status"] = "running"
            
            self._initialized = True
            self.logger.info("Market Intelligence Center initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Market Intelligence Center: {str(e)}")
            self.metadata["status"] = f"initialization_failed: {str(e)}"
            return False
    
    def update(self, symbols=None, force=False):
        """
        Update all data in the Market Intelligence Center.
        
        Args:
            symbols: List of stock symbols to update or None for all
            force: Whether to force updates even if not needed
            
        Returns:
            Dictionary with update results
        """
        self.logger.info(f"Updating Market Intelligence Center (force={force})")
        
        try:
            # Initialize if not already
            if not self._initialized:
                success = self.initialize(symbols)
                if not success:
                    return {"status": "error", "message": "Failed to initialize"}
            
            # Get existing symbols if none provided
            if symbols is None:
                context = self.market_context.get_market_context()
                symbols = list(context.get("symbols", {}).keys())
            
            # Update market context
            self.market_context.update_market_data()
            self.market_context.update_symbol_data(symbols)
            self.market_context.update_strategy_rankings()
            self.market_context.match_symbols_to_strategies()
            
            # Save updated state
            self.market_context.save_to_file("data/market_context/latest_context.json")
            
            # Run inference to generate predictions
            predictions = self.ml_trainer.run_inference()
            
            # Update metadata
            self.metadata["last_update"] = datetime.datetime.now().isoformat()
            
            return {
                "status": "success",
                "timestamp": datetime.datetime.now().isoformat(),
                "symbols_updated": len(symbols),
                "predictions": predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error updating Market Intelligence Center: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_top_symbol_strategy_pairs(self, limit=5):
        """
        Get the top symbol-strategy pairs.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of symbol-strategy pairs
        """
        # Initialize if not already
        if not self._initialized:
            self.initialize()
        
        # Get pairs from context
        context = self.market_context.get_market_context()
        pairs = context.get("symbol_strategy_pairs", [])
        
        # Return limited results
        return pairs[:limit] if limit > 0 else pairs
    
    def get_market_summary(self):
        """
        Get a summary of current market conditions.
        
        Returns:
            Dictionary with market summary
        """
        # Initialize if not already
        if not self._initialized:
            self.initialize()
        
        # Get context
        context = self.market_context.get_market_context()
        
        # Extract relevant information
        market = context.get("market", {})
        
        return {
            "regime": market.get("regime", "unknown"),
            "indicators": market.get("indicators", {}),
            "volatility": market.get("volatility", {}),
            "sectors": market.get("sectors", {}),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_symbol_analysis(self, symbol):
        """
        Get detailed analysis for a specific symbol.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary with symbol analysis
        """
        # Initialize if not already
        if not self._initialized:
            self.initialize()
        
        # Get context
        context = self.market_context.get_market_context()
        
        # Check if symbol exists in context
        if symbol not in context.get("symbols", {}):
            # Update symbol data
            self.market_context.update_symbol_data([symbol])
            
            # Get updated context
            context = self.market_context.get_market_context()
        
        # Extract symbol data
        symbol_data = context.get("symbols", {}).get(symbol, {})
        
        # Get news for symbol
        news = context.get("news", {}).get("symbols", {}).get(symbol, [])
        
        # Find strategies this symbol is paired with
        paired_strategies = []
        for pair in context.get("symbol_strategy_pairs", []):
            if pair.get("symbol") == symbol:
                paired_strategies.append({
                    "strategy": pair.get("strategy"),
                    "score": pair.get("score")
                })
        
        return {
            "symbol": symbol,
            "price": symbol_data.get("price", {}),
            "technicals": symbol_data.get("technicals", {}),
            "news": news,
            "strategies": paired_strategies,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_strategy_recommendations(self, market_regime=None):
        """
        Get strategy recommendations for a specific market regime.
        
        Args:
            market_regime: Market regime to get recommendations for or None for current
            
        Returns:
            Dictionary with strategy recommendations
        """
        # Initialize if not already
        if not self._initialized:
            self.initialize()
        
        # Get context
        context = self.market_context.get_market_context()
        
        # Use current regime if none specified
        if market_regime is None:
            market_regime = context.get("market", {}).get("regime", "unknown")
        
        # Get strategies
        all_strategies = context.get("strategies", {}).get("ranked", [])
        
        # Filter by regime
        matching_strategies = [
            strategy for strategy in all_strategies
            if market_regime in strategy.get("suitable_regimes", [])
        ]
        
        # If no matching strategies, use top 3
        if not matching_strategies and all_strategies:
            matching_strategies = all_strategies[:3]
        
        return {
            "market_regime": market_regime,
            "strategies": matching_strategies,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _start_background_processing(self):
        """Start background processing thread."""
        if self._running:
            self.logger.info("Background processing already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._background_thread = threading.Thread(target=self._background_loop)
        self._background_thread.daemon = True
        self._background_thread.start()
        
        self.logger.info("Background processing started")
    
    def _stop_background_processing(self):
        """Stop background processing thread."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._background_thread:
            self._background_thread.join(timeout=5)
        
        # Also stop the training scheduler
        self.training_scheduler.stop()
        
        self.logger.info("Background processing stopped")
    
    def _background_loop(self):
        """Background processing loop."""
        self.logger.info("Background loop started")
        
        while not self._stop_event.is_set():
            try:
                # Periodic updates - every 30 minutes
                if (time.time() - datetime.datetime.fromisoformat(self.metadata["last_update"]).timestamp()) >= 1800:
                    self.update()
                
                # Save market context periodically
                context = self.market_context.get_market_context()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.market_context.save_to_file(f"data/market_context/context_{timestamp}.json")
                
            except Exception as e:
                self.logger.error(f"Error in background loop: {str(e)}")
            
            # Sleep for a bit
            time.sleep(60)  # Check every minute
    
    def shutdown(self):
        """Shut down the Market Intelligence Center."""
        self.logger.info("Shutting down Market Intelligence Center")
        
        # Stop background processing
        self._stop_background_processing()
        
        # Final state save
        try:
            self.market_context.save_to_file("data/market_context/final_context.json")
        except Exception as e:
            self.logger.error(f"Error saving final state: {str(e)}")
        
        # Update metadata
        self.metadata["status"] = "shutdown"
        
        self.logger.info("Market Intelligence Center shut down")


# Create singleton instance
_market_intelligence_controller = None

def get_market_intelligence_controller(config=None):
    """
    Get the singleton MarketIntelligenceController instance.
    
    Args:
        config: Optional configuration for the controller
    
    Returns:
        MarketIntelligenceController instance
    """
    global _market_intelligence_controller
    if _market_intelligence_controller is None:
        _market_intelligence_controller = MarketIntelligenceController(config)
    return _market_intelligence_controller
