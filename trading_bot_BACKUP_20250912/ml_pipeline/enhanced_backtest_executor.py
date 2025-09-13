"""
Enhanced Backtest Executor
Runs parameter-driven backtests using the variant generator and promotes successful strategies.
"""

import os
import sys
import json
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our components
from trading_bot.ml_pipeline.backtest_feedback_loop import get_backtest_executor, get_backtest_feedback_system
from trading_bot.ml_pipeline.variant_generator import get_variant_generator
from trading_bot.ml_pipeline.backtest_scorer import get_backtest_scorer
from trading_bot.ml_pipeline.strategy_promoter import get_strategy_promoter
from trading_bot.market_context.market_context import get_market_context


class EnhancedBacktestExecutor:
    """
    Enhanced backtest executor that leverages variant generation and strategy promotion.
    """
    
    def __init__(self):
        """Initialize the enhanced backtest executor."""
        self.logger = logging.getLogger("EnhancedBacktestExecutor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Get component instances
        self.base_executor = get_backtest_executor()
        self.feedback_system = get_backtest_feedback_system()
        self.variant_generator = get_variant_generator()
        self.backtest_scorer = get_backtest_scorer()
        self.strategy_promoter = get_strategy_promoter()
        self.market_context = get_market_context()
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        self.logger.info("EnhancedBacktestExecutor initialized")
    
    def get_current_market_regime(self) -> str:
        """
        Get the current market regime from market context.
        
        Returns:
            String representing current market regime
        """
        try:
            context = self.market_context.get_current_context()
            regime = context.get("market_regime", "unknown")
            return regime
        except Exception as e:
            self.logger.error(f"Error getting market regime: {str(e)}")
            return "unknown"
    
    def backtest_strategy_variants(self, symbol: str, strategy: str, 
                                   max_variants: int = 10, 
                                   start_date: str = None, 
                                   end_date: str = None,
                                   parallel: bool = False) -> Dict:
        """
        Backtest multiple variants of a strategy for a symbol.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy identifier
            max_variants: Maximum number of variants to test
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            parallel: Whether to run variants in parallel
            
        Returns:
            Dictionary with backtest results and summary stats
        """
        self.logger.info(f"Testing strategy variants for {symbol} with {strategy}")
        
        # Get current market regime
        market_regime = self.get_current_market_regime()
        self.logger.info(f"Current market regime: {market_regime}")
        
        # Get variants for this strategy
        variants = self.variant_generator.get_strategy_variants(strategy, limit=max_variants)
        
        if not variants:
            self.logger.warning(f"No variants found for strategy: {strategy}")
            return {
                "status": "error",
                "message": f"No variants found for strategy: {strategy}",
                "results": [],
                "summary": {
                    "count": 0,
                    "pass_count": 0,
                    "pass_rate": 0,
                    "promotions": 0
                }
            }
        
        self.logger.info(f"Testing {len(variants)} variants")
        
        # Store results
        all_results = []
        promoted_count = 0
        
        # Function to process a single variant
        def process_variant(params):
            try:
                # Run backtest with these parameters
                result = self.base_executor.backtest_pair(
                    symbol=symbol,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    params=params
                )
                
                # Score the result
                score = self.backtest_scorer.score_result(
                    result=result,
                    strategy_id=strategy,
                    current_regime=market_regime
                )
                
                # Try to promote if it passed scoring
                promotion_result = None
                if score.get("passed", False):
                    promotion_result = self.strategy_promoter.evaluate_and_promote(
                        symbol=symbol,
                        strategy=strategy,
                        result=result,
                        params=params,
                        score=score,
                        market_regime=market_regime
                    )
                
                # Create combined result
                return {
                    "params": params,
                    "result": result,
                    "score": score,
                    "promotion": promotion_result
                }
            
            except Exception as e:
                self.logger.error(f"Error processing variant: {str(e)}")
                return {
                    "params": params,
                    "error": str(e)
                }
        
        # Process variants (parallel or sequential)
        if parallel and len(variants) > 1:
            with ThreadPoolExecutor(max_workers=min(len(variants), 5)) as executor:
                futures = {executor.submit(process_variant, params): params for params in variants}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                        if result.get("promotion", {}).get("promoted", False):
                            promoted_count += 1
                    except Exception as e:
                        self.logger.error(f"Error in parallel variant processing: {str(e)}")
        else:
            for params in variants:
                result = process_variant(params)
                all_results.append(result)
                if result.get("promotion", {}).get("promoted", False):
                    promoted_count += 1
        
        # Sort results by score (descending)
        all_results.sort(key=lambda x: x.get("score", {}).get("score", 0), reverse=True)
        
        # Get summary statistics
        passed_results = [r for r in all_results if r.get("score", {}).get("passed", False)]
        pass_count = len(passed_results)
        pass_rate = pass_count / len(all_results) if all_results else 0
        
        # Get optimal parameter grid
        optimal_params = {}
        if passed_results:
            best_result = passed_results[0]
            optimal_params = best_result.get("params", {})
        
        # Create summary
        summary = {
            "count": len(all_results),
            "pass_count": pass_count,
            "pass_rate": round(pass_rate, 2),
            "promotions": promoted_count,
            "optimal_params": optimal_params,
            "market_regime": market_regime
        }
        
        self.logger.info(f"Completed testing {len(variants)} variants with {pass_count} passes and {promoted_count} promotions")
        
        return {
            "status": "success",
            "symbol": symbol,
            "strategy": strategy,
            "results": all_results,
            "summary": summary
        }
    
    def backtest_multiple_strategies(self, symbol: str, strategies: List[str], 
                                    variants_per_strategy: int = 5) -> Dict:
        """
        Backtest multiple strategies for a symbol.
        
        Args:
            symbol: Stock symbol
            strategies: List of strategy identifiers
            variants_per_strategy: Max variants to test per strategy
            
        Returns:
            Dictionary with results for each strategy
        """
        self.logger.info(f"Testing multiple strategies for {symbol}: {strategies}")
        
        results = {}
        
        for strategy in strategies:
            self.logger.info(f"Testing strategy: {strategy}")
            strategy_results = self.backtest_strategy_variants(
                symbol=symbol,
                strategy=strategy,
                max_variants=variants_per_strategy
            )
            results[strategy] = strategy_results
        
        # Determine overall best strategy
        best_strategy = None
        best_score = -float('inf')
        
        for strategy, result in results.items():
            strategy_results = result.get("results", [])
            if not strategy_results:
                continue
            
            # Get best score for this strategy
            best_result = max(strategy_results, key=lambda x: x.get("score", {}).get("score", 0), default=None)
            if best_result:
                score = best_result.get("score", {}).get("score", 0)
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        summary = {
            "tested_strategies": len(strategies),
            "best_strategy": best_strategy,
            "best_score": round(best_score, 2) if best_score > -float('inf') else 0
        }
        
        self.logger.info(f"Completed testing multiple strategies. Best: {best_strategy} with score {best_score:.2f}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "strategies": strategies,
            "results": results,
            "summary": summary
        }
    
    def optimize_ml_portfolio(self, symbols: List[str], strategies: List[str], 
                             variants_per_combination: int = 3) -> Dict:
        """
        Run a full optimization across multiple symbols and strategies.
        
        Args:
            symbols: List of stock symbols
            strategies: List of strategy identifiers
            variants_per_combination: Max variants per symbol-strategy combination
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Optimizing ML portfolio for {len(symbols)} symbols and {len(strategies)} strategies")
        
        # Get current market regime
        market_regime = self.get_current_market_regime()
        
        # Store results
        all_results = {}
        promotions = []
        
        # Process each symbol-strategy combination
        for symbol in symbols:
            symbol_results = {}
            
            for strategy in strategies:
                # Check if strategy is suitable for current regime
                regime_compatible = self.variant_generator.is_regime_compatible(strategy, market_regime)
                
                if not regime_compatible:
                    self.logger.info(f"Skipping {strategy} for {symbol} - not compatible with {market_regime} regime")
                    continue
                
                # Run backtest with variants
                result = self.backtest_strategy_variants(
                    symbol=symbol,
                    strategy=strategy,
                    max_variants=variants_per_combination
                )
                
                symbol_results[strategy] = result
                
                # Track promotions
                promoted_results = [
                    r for r in result.get("results", [])
                    if r.get("promotion", {}).get("promoted", False)
                ]
                
                for r in promoted_results:
                    promotions.append({
                        "symbol": symbol,
                        "strategy": strategy,
                        "score": r.get("score", {}).get("score", 0),
                        "params": r.get("params", {})
                    })
            
            all_results[symbol] = symbol_results
        
        # Sort promotions by score
        promotions.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Create summary
        summary = {
            "symbols_tested": len(symbols),
            "strategies_tested": len(strategies),
            "combinations_tested": len(symbols) * len(strategies),
            "promotions": len(promotions),
            "market_regime": market_regime
        }
        
        self.logger.info(f"Completed portfolio optimization with {len(promotions)} new promotions")
        
        return {
            "status": "success",
            "results": all_results,
            "promotions": promotions,
            "summary": summary
        }


# Create singleton instance
_enhanced_backtest_executor = None

def get_enhanced_backtest_executor():
    """
    Get the singleton EnhancedBacktestExecutor instance.
    
    Returns:
        EnhancedBacktestExecutor instance
    """
    global _enhanced_backtest_executor
    if _enhanced_backtest_executor is None:
        _enhanced_backtest_executor = EnhancedBacktestExecutor()
    return _enhanced_backtest_executor
