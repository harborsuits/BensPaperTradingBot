"""
LLM Integration Script for Autonomous Trading System

This script connects all LLM enhancement components with the existing trading system,
providing a complete integration of ML models with LLM-powered reasoning.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LLM integration components
from trading_bot.llm_integration.financial_llm_engine import FinancialLLMEngine, LLMProvider
from trading_bot.llm_integration.memory_system import MemorySystem, MemoryType
from trading_bot.llm_integration.prompt_engineering import PromptManager, PromptTemplate
from trading_bot.llm_integration.reasoning_engine import ReasoningEngine, ReasoningTask
from trading_bot.llm_integration.strategy_enhancement import StrategyEnhancer, EnhancementType

# Import existing trading system components
from trading_bot.ml_pipeline.ml_regime_detector import MLRegimeDetector
from trading_bot.ml_pipeline.portfolio_optimizer import PortfolioOptimizer
from trading_bot.strategies.regime_specific_strategy import RegimeSpecificStrategy
from trading_bot.triggers.regime_change_notifier import RegimeChangeNotifier
from trading_bot.automated_trader import AutomatedTrader

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llm_integration")

class LLMTradingIntegration:
    """
    Integrates LLM capabilities with the trading system
    
    This class initializes all components and provides methods to use them together.
    """
    
    def __init__(
        self,
        config_path: str,
        enable_memory: bool = True,
        memory_db_path: Optional[str] = None,
        enable_caching: bool = True,
        default_llm_provider: LLMProvider = LLMProvider.OPENAI,
        debug: bool = False
    ):
        """
        Initialize the LLM trading integration
        
        Args:
            config_path: Path to configuration file with API keys
            enable_memory: Whether to enable the memory system
            memory_db_path: Path to memory database file
            enable_caching: Whether to enable LLM response caching
            default_llm_provider: Default LLM provider
            debug: Enable debug logging
        """
        self.debug = debug
        self.config_path = config_path
        
        # Step 1: Initialize the LLM engine
        logger.info("Initializing LLM engine...")
        self.llm_engine = FinancialLLMEngine(
            config_path=config_path,
            default_provider=default_llm_provider,
            enable_caching=enable_caching,
            debug=debug
        )
        
        # Step 2: Initialize the memory system if enabled
        self.memory_system = None
        if enable_memory:
            logger.info("Initializing memory system...")
            self.memory_system = MemorySystem(
                db_path=memory_db_path,
                enable_embeddings=True,
                debug=debug
            )
        
        # Step 3: Initialize the prompt manager
        logger.info("Initializing prompt manager...")
        self.prompt_manager = PromptManager(
            memory_system=self.memory_system,
            debug=debug
        )
        
        # Wait to initialize other components until existing system components are provided
        self.reasoning_engine = None
        self.strategy_enhancer = None
        self.ml_regime_detector = None
        self.portfolio_optimizer = None
        self.regime_change_notifier = None
        self.automated_trader = None
        
        logger.info("LLM trading integration base initialization complete")
        
    def connect_existing_components(
        self,
        ml_regime_detector: Optional[MLRegimeDetector] = None,
        portfolio_optimizer: Optional[PortfolioOptimizer] = None,
        regime_change_notifier: Optional[RegimeChangeNotifier] = None,
        automated_trader: Optional[AutomatedTrader] = None
    ):
        """
        Connect existing trading system components
        
        Args:
            ml_regime_detector: ML-based regime detector
            portfolio_optimizer: Portfolio optimizer
            regime_change_notifier: Regime change notification system
            automated_trader: Automated trading engine
        """
        logger.info("Connecting existing trading system components...")
        
        # Store references
        self.ml_regime_detector = ml_regime_detector
        self.portfolio_optimizer = portfolio_optimizer
        self.regime_change_notifier = regime_change_notifier
        self.automated_trader = automated_trader
        
        # Step 4: Initialize the reasoning engine
        logger.info("Initializing reasoning engine...")
        self.reasoning_engine = ReasoningEngine(
            llm_engine=self.llm_engine,
            prompt_manager=self.prompt_manager,
            memory_system=self.memory_system,
            ml_regime_detector=ml_regime_detector,
            debug=self.debug
        )
        
        # Step 5: Initialize the strategy enhancer
        logger.info("Initializing strategy enhancer...")
        self.strategy_enhancer = StrategyEnhancer(
            reasoning_engine=self.reasoning_engine,
            memory_system=self.memory_system,
            ml_regime_detector=ml_regime_detector,
            portfolio_optimizer=portfolio_optimizer,
            regime_change_notifier=regime_change_notifier,
            debug=self.debug
        )
        
        logger.info("LLM trading integration complete")
        
    def enhance_strategy(
        self,
        strategy: RegimeSpecificStrategy,
        symbol: str,
        timeframe: str
    ):
        """
        Enhance a strategy with LLM-powered adjustments
        
        Args:
            strategy: The strategy to enhance
            symbol: Trading symbol
            timeframe: Trading timeframe
        """
        if not self.strategy_enhancer:
            logger.warning("Strategy enhancer not initialized")
            return
            
        logger.info(f"Enhancing strategy for {symbol}/{timeframe}...")
        
        # Get the original parameters
        original_params = strategy.get_parameters()
        
        # Get enhanced parameters
        enhanced_params = self.strategy_enhancer.get_enhanced_parameters(
            strategy_parameters=original_params,
            strategy_type=strategy.__class__.__name__,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Update the strategy parameters
        strategy.update_parameters(enhanced_params)
        
        logger.info(f"Strategy enhanced for {symbol}/{timeframe}")
        
    async def analyze_market_conditions(
        self,
        symbols: List[str],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform LLM-enhanced market analysis
        
        Args:
            symbols: List of symbols to analyze
            market_data: Market data
            
        Returns:
            Analysis results
        """
        if not self.reasoning_engine:
            logger.warning("Reasoning engine not initialized")
            return {"error": "Reasoning engine not initialized"}
            
        logger.info(f"Analyzing market conditions for {len(symbols)} symbols...")
        
        # Prepare context
        context = {
            "symbols": symbols,
            "market_data": market_data,
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        # Get ML signals from regime detector if available
        ml_signals = {}
        if self.ml_regime_detector:
            try:
                ml_signals["regime"] = self.ml_regime_detector.current_regime
                ml_signals["regime_confidence"] = self.ml_regime_detector.regime_confidence
                ml_signals["regime_features"] = self.ml_regime_detector.get_top_regime_features(5)
            except Exception as e:
                logger.error(f"Error getting ML signals: {e}")
        
        # Perform reasoning
        result = await self.reasoning_engine.reason(
            task=ReasoningTask.MARKET_ANALYSIS,
            ml_signals=ml_signals,
            context=context
        )
        
        return {
            "analysis": result.explanation,
            "conclusion": result.conclusion,
            "confidence": result.confidence,
            "consensus_score": result.consensus_score,
            "llm_signals": result.llm_signals,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    def analyze_market_conditions_sync(
        self,
        symbols: List[str],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous version of analyze_market_conditions"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.analyze_market_conditions(symbols, market_data)
            )
        finally:
            loop.close()
            
    async def analyze_news_impact(
        self,
        symbol: str,
        news_headlines: List[str],
        news_content: Optional[str] = None,
        price_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the potential impact of news on a symbol
        
        Args:
            symbol: Trading symbol
            news_headlines: List of news headlines
            news_content: Full news content if available
            price_data: Recent price data if available
            
        Returns:
            News impact analysis
        """
        if not self.reasoning_engine:
            logger.warning("Reasoning engine not initialized")
            return {"error": "Reasoning engine not initialized"}
            
        logger.info(f"Analyzing news impact for {symbol}...")
        
        # Prepare context
        context = {
            "symbol": symbol,
            "news_headlines": "\n".join(news_headlines),
            "news_content": news_content,
            "price_data": price_data,
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        # Get ML signals
        ml_signals = {
            "symbol": symbol
        }
        
        # Add regime information if available
        if self.ml_regime_detector:
            ml_signals["regime"] = self.ml_regime_detector.current_regime
            
        # Perform reasoning
        result = await self.reasoning_engine.reason(
            task=ReasoningTask.NEWS_IMPACT,
            ml_signals=ml_signals,
            context=context
        )
        
        return {
            "analysis": result.explanation,
            "impact": result.conclusion,
            "confidence": result.confidence,
            "consensus_score": result.consensus_score,
            "llm_signals": result.llm_signals,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def analyze_news_impact_sync(
        self,
        symbol: str,
        news_headlines: List[str],
        news_content: Optional[str] = None,
        price_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of analyze_news_impact"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.analyze_news_impact(symbol, news_headlines, news_content, price_data)
            )
        finally:
            loop.close()
    
    def record_trade_completion(
        self,
        symbol: str,
        entry_time: datetime.datetime,
        exit_time: datetime.datetime,
        entry_price: float,
        exit_price: float,
        position_size: float,
        profit_loss: float,
        strategy_name: str,
        trade_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a completed trade for learning
        
        Args:
            symbol: Trading symbol
            entry_time: Trade entry time
            exit_time: Trade exit time
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            profit_loss: Profit/loss amount
            strategy_name: Name of the strategy used
            trade_metadata: Additional trade metadata
        """
        if not self.strategy_enhancer:
            logger.warning("Strategy enhancer not initialized")
            return
            
        logger.info(f"Recording trade completion for {symbol}...")
        
        # Add strategy name to metadata
        if trade_metadata is None:
            trade_metadata = {}
        trade_metadata["strategy_name"] = strategy_name
        
        # Record the trade
        self.strategy_enhancer.analyze_trade_result(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            profit_loss=profit_loss,
            trade_metadata=trade_metadata
        )
        
        logger.info(f"Trade recorded for {symbol}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the memory system
        
        Returns:
            Memory system summary
        """
        if not self.memory_system:
            return {"error": "Memory system not enabled"}
            
        # Get counts by memory type
        counts = self.memory_system.count_memories()
        
        # Get a sample of recent memories
        short_term = self.memory_system.get_memories_by_type(MemoryType.SHORT_TERM, limit=5)
        medium_term = self.memory_system.get_memories_by_type(MemoryType.MEDIUM_TERM, limit=5)
        long_term = self.memory_system.get_memories_by_type(MemoryType.LONG_TERM, limit=5)
        
        # Format for display
        short_term_items = [
            {"content": mem.content[:200] + "..." if len(mem.content) > 200 else mem.content,
             "importance": mem.importance,
             "created": datetime.datetime.fromtimestamp(mem.created_at).isoformat(),
             "tags": mem.tags}
            for mem in short_term
        ]
        
        medium_term_items = [
            {"content": mem.content[:200] + "..." if len(mem.content) > 200 else mem.content,
             "importance": mem.importance,
             "created": datetime.datetime.fromtimestamp(mem.created_at).isoformat(),
             "tags": mem.tags}
            for mem in medium_term
        ]
        
        long_term_items = [
            {"content": mem.content[:200] + "..." if len(mem.content) > 200 else mem.content,
             "importance": mem.importance,
             "created": datetime.datetime.fromtimestamp(mem.created_at).isoformat(),
             "tags": mem.tags}
            for mem in long_term
        ]
        
        return {
            "counts": counts,
            "short_term_sample": short_term_items,
            "medium_term_sample": medium_term_items,
            "long_term_sample": long_term_items,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_llm_insights_for_ui(
        self,
        symbols: List[str],
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get LLM insights formatted for UI display
        
        Args:
            symbols: List of symbols
            timeframe: Trading timeframe
            
        Returns:
            Insights for UI display
        """
        if not self.reasoning_engine:
            return {"error": "Reasoning engine not initialized"}
            
        insights = {}
        
        # Get market analysis
        try:
            market_analysis = self.analyze_market_conditions_sync(
                symbols=symbols,
                market_data={"timeframe": timeframe}
            )
            insights["market_analysis"] = market_analysis
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            insights["market_analysis"] = {"error": str(e)}
        
        # Get regime information
        if self.ml_regime_detector:
            try:
                insights["regime"] = {
                    "current": self.ml_regime_detector.current_regime,
                    "confidence": self.ml_regime_detector.regime_confidence,
                    "detected_at": self.ml_regime_detector.last_update_time.isoformat() 
                              if hasattr(self.ml_regime_detector, "last_update_time") else None
                }
            except Exception as e:
                logger.error(f"Error getting regime info: {e}")
                insights["regime"] = {"error": str(e)}
        
        # Get strategy adjustments
        if self.strategy_enhancer:
            try:
                # Get recent adjustments
                adjustments = self.strategy_enhancer.get_adjustment_history(limit=10)
                
                insights["strategy_adjustments"] = [
                    {
                        "parameter": adj.parameter_name,
                        "old_value": adj.original_value,
                        "new_value": adj.new_value,
                        "type": adj.adjustment_type.value,
                        "confidence": adj.confidence,
                        "explanation": adj.explanation,
                        "timestamp": datetime.datetime.fromtimestamp(adj.timestamp).isoformat()
                    }
                    for adj in adjustments
                ]
            except Exception as e:
                logger.error(f"Error getting strategy adjustments: {e}")
                insights["strategy_adjustments"] = {"error": str(e)}
        
        # Get memory stats
        if self.memory_system:
            try:
                insights["memory_stats"] = {
                    "total_memories": sum(self.memory_system.count_memories().values()),
                    "counts_by_type": self.memory_system.count_memories()
                }
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
                insights["memory_stats"] = {"error": str(e)}
        
        return insights
            
    def shutdown(self):
        """Shutdown and clean up resources"""
        if self.memory_system:
            try:
                self.memory_system.close()
                logger.info("Memory system closed")
            except Exception as e:
                logger.error(f"Error closing memory system: {e}")
        
        logger.info("LLM trading integration shutdown complete")


# Example usage
if __name__ == "__main__":
    # Path to configuration file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
    
    # Initialize integration
    llm_trading = LLMTradingIntegration(
        config_path=config_path,
        enable_memory=True,
        debug=True
    )
    
    print("LLM trading integration initialized.")
    print("To use, connect your existing trading components:")
    print("  llm_trading.connect_existing_components(")
    print("      ml_regime_detector=your_regime_detector,")
    print("      portfolio_optimizer=your_portfolio_optimizer,")
    print("      regime_change_notifier=your_regime_notifier,")
    print("      automated_trader=your_automated_trader")
    print("  )")
