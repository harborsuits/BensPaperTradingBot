"""
Strategy Promotion System
Evaluates backtest results and automatically promotes high-performing
strategy variants to paper/live trading based on configurable criteria.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our components
from trading_bot.ml_pipeline.variant_generator import get_variant_generator
from trading_bot.ml_pipeline.backtest_scorer import get_backtest_scorer


class StrategyPromoter:
    """
    Evaluates and promotes high-performing strategies to paper/live trading.
    Acts as a gatekeeper between research and execution.
    """
    
    def __init__(self):
        """Initialize the strategy promoter."""
        self.logger = logging.getLogger("StrategyPromoter")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.variant_generator = get_variant_generator()
        self.backtest_scorer = get_backtest_scorer()
        
        # Promotion storage
        self.promotions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "strategy_promotions.json"
        )
        os.makedirs(os.path.dirname(self.promotions_path), exist_ok=True)
        
        # Load existing promotions
        self.promotions = self._load_promotions()
        
        # Default promotion thresholds (can be overridden by strategy profiles)
        self.default_thresholds = {
            "score": 0.85,
            "sharpe_ratio": 1.5,
            "total_return": 10.0,
            "max_drawdown": -8.0,
            "win_rate": 0.55
        }
        
        self.logger.info("StrategyPromoter initialized")
    
    def _load_promotions(self) -> Dict:
        """
        Load existing strategy promotions from disk.
        
        Returns:
            Dictionary with promotion history
        """
        if os.path.exists(self.promotions_path):
            try:
                with open(self.promotions_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading promotions: {str(e)}")
                return {"promotions": [], "meta": {"last_updated": datetime.datetime.now().isoformat()}}
        else:
            return {"promotions": [], "meta": {"last_updated": datetime.datetime.now().isoformat()}}
    
    def _save_promotions(self):
        """Save promotions to disk."""
        try:
            with open(self.promotions_path, 'w') as f:
                json.dump(self.promotions, f, indent=2)
            self.logger.debug("Promotions saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving promotions: {str(e)}")
    
    def get_strategy_thresholds(self, strategy_id: str) -> Dict:
        """
        Get promotion thresholds for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Dictionary with promotion thresholds
        """
        # Get optimization goals from strategy profile
        optimization = self.variant_generator.get_optimization_goals(strategy_id)
        
        if not optimization:
            self.logger.warning(f"No optimization goals found for {strategy_id}, using defaults")
            return self.default_thresholds
        
        # Convert optimization goals to promotion thresholds
        thresholds = {
            "score": optimization.get("promotion_threshold", self.default_thresholds["score"]),
            "sharpe_ratio": optimization.get("min_sharpe", self.default_thresholds["sharpe_ratio"]),
            "total_return": optimization.get("min_return", self.default_thresholds["total_return"]),
            "max_drawdown": -abs(optimization.get("max_drawdown", abs(self.default_thresholds["max_drawdown"]))),
            "win_rate": optimization.get("min_win_rate", self.default_thresholds["win_rate"])
        }
        
        return thresholds
    
    def evaluate_promotion_criteria(self, result: Dict, score: Dict, strategy_id: str) -> Dict:
        """
        Evaluate if a backtest result meets promotion criteria.
        
        Args:
            result: Dictionary with backtest results
            score: Dictionary with score information from backtest_scorer
            strategy_id: Strategy identifier
        
        Returns:
            Dictionary with evaluation results and reason
        """
        # Get thresholds for this strategy
        thresholds = self.get_strategy_thresholds(strategy_id)
        
        # Get values from result
        score_value = score.get("score", 0)
        sharpe = result.get("sharpe_ratio", 0)
        total_return = result.get("total_return", 0)
        max_drawdown = result.get("max_drawdown", 0)
        win_rate = result.get("win_rate", 0)
        
        # Ensure max_drawdown is negative for comparison
        if max_drawdown > 0:
            max_drawdown = -max_drawdown
        
        # Check each criterion
        meets_score = score_value >= thresholds["score"]
        meets_sharpe = sharpe >= thresholds["sharpe_ratio"]
        meets_return = total_return >= thresholds["total_return"]
        meets_drawdown = max_drawdown >= thresholds["max_drawdown"]  # Less negative is better
        meets_win_rate = win_rate >= thresholds["win_rate"]
        
        # Combined result
        promoted = meets_score and meets_sharpe and meets_return and meets_drawdown and meets_win_rate
        
        # Generate reason
        reason = "Promoted successfully" if promoted else "Failed criteria: "
        
        if not meets_score:
            reason += f"Score ({score_value:.2f} < {thresholds['score']:.2f}) "
        if not meets_sharpe:
            reason += f"Sharpe ({sharpe:.2f} < {thresholds['sharpe_ratio']:.2f}) "
        if not meets_return:
            reason += f"Return ({total_return:.2f}% < {thresholds['total_return']:.2f}%) "
        if not meets_drawdown:
            reason += f"Drawdown ({max_drawdown:.2f}% < {thresholds['max_drawdown']:.2f}%) "
        if not meets_win_rate:
            reason += f"Win Rate ({win_rate:.2f} < {thresholds['win_rate']:.2f}) "
        
        return {
            "promoted": promoted,
            "reason": reason,
            "threshold_score": thresholds["score"],
            "threshold_sharpe": thresholds["sharpe_ratio"],
            "threshold_return": thresholds["total_return"],
            "threshold_drawdown": thresholds["max_drawdown"],
            "threshold_win_rate": thresholds["win_rate"]
        }
    
    def evaluate_and_promote(self, symbol: str, strategy: str, result: Dict, 
                             params: Dict, score: Dict, market_regime: str = None) -> Dict:
        """
        Evaluate backtest results and promote the strategy if it meets criteria.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy identifier
            result: Dictionary with backtest results
            params: Strategy parameters used
            score: Score information from backtest_scorer
            market_regime: Current market regime
        
        Returns:
            Dictionary with promotion result and details
        """
        evaluation = self.evaluate_promotion_criteria(result, score, strategy)
        
        if not evaluation["promoted"]:
            self.logger.info(f"Strategy {strategy} for {symbol} not promoted: {evaluation['reason']}")
            return {
                "promoted": False,
                "reason": evaluation["reason"],
                "evaluation": evaluation
            }
        
        # Create promotion record
        promotion = {
            "symbol": symbol,
            "strategy": strategy,
            "params": params,
            "score": score.get("score", 0),
            "sharpe_ratio": result.get("sharpe_ratio", 0),
            "total_return": result.get("total_return", 0),
            "win_rate": result.get("win_rate", 0),
            "max_drawdown": result.get("max_drawdown", 0),
            "timestamp": datetime.datetime.now().isoformat(),
            "regime": market_regime or "unknown"
        }
        
        # Add to promotions list
        self.promotions["promotions"].append(promotion)
        self.promotions["meta"]["last_updated"] = datetime.datetime.now().isoformat()
        self.promotions["meta"]["count"] = len(self.promotions["promotions"])
        
        # Save promotions
        self._save_promotions()
        
        # Queue for paper trading
        self._queue_for_paper_trading(promotion)
        
        self.logger.info(f"Strategy {strategy} for {symbol} promoted successfully")
        
        return {
            "promoted": True,
            "reason": evaluation["reason"],
            "evaluation": evaluation,
            "promotion": promotion
        }
    
    def _queue_for_paper_trading(self, promotion: Dict):
        """
        Queue a promoted strategy for paper trading.
        
        Args:
            promotion: Dictionary with promotion details
        """
        try:
            # In a real implementation, this would send to paper trading engine
            # For now, we'll just save to a queue file
            queue_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "paper_trade_queue.json"
            )
            
            # Load existing queue
            queue = []
            if os.path.exists(queue_path):
                try:
                    with open(queue_path, 'r') as f:
                        queue = json.load(f)
                except:
                    queue = []
            
            # Add to queue
            queue.append({
                "promotion": promotion,
                "status": "queued",
                "queued_at": datetime.datetime.now().isoformat()
            })
            
            # Save queue
            with open(queue_path, 'w') as f:
                json.dump(queue, f, indent=2)
            
            self.logger.info(f"Queued {promotion['strategy']} for {promotion['symbol']} for paper trading")
        
        except Exception as e:
            self.logger.error(f"Error queueing for paper trading: {str(e)}")
    
    def get_top_promotions(self, limit: int = 10, symbol: str = None, 
                           strategy: str = None, regime: str = None) -> List[Dict]:
        """
        Get top promoted strategies, optionally filtered.
        
        Args:
            limit: Maximum number of promotions to return
            symbol: Optional filter by symbol
            strategy: Optional filter by strategy
            regime: Optional filter by market regime
        
        Returns:
            List of promotion records
        """
        promotions = self.promotions.get("promotions", [])
        
        # Apply filters
        if symbol:
            promotions = [p for p in promotions if p.get("symbol") == symbol]
        if strategy:
            promotions = [p for p in promotions if p.get("strategy") == strategy]
        if regime:
            promotions = [p for p in promotions if p.get("regime") == regime]
        
        # Sort by score (descending)
        promotions = sorted(promotions, key=lambda p: p.get("score", 0), reverse=True)
        
        # Limit results
        return promotions[:limit]
    
    def get_best_promotion_for_symbol(self, symbol: str, regime: str = None) -> Optional[Dict]:
        """
        Get the best promoted strategy for a specific symbol and regime.
        
        Args:
            symbol: Stock symbol
            regime: Optional market regime
        
        Returns:
            Best promotion record or None if not found
        """
        top = self.get_top_promotions(limit=1, symbol=symbol, regime=regime)
        return top[0] if top else None
    
    def clear_expired_promotions(self, days_threshold: int = 60):
        """
        Clear promotions older than a specified threshold.
        
        Args:
            days_threshold: Age threshold in days
        """
        now = datetime.datetime.now()
        threshold = now - datetime.timedelta(days=days_threshold)
        
        # Filter promotions
        old_count = len(self.promotions.get("promotions", []))
        self.promotions["promotions"] = [
            p for p in self.promotions.get("promotions", [])
            if datetime.datetime.fromisoformat(p.get("timestamp", now.isoformat())) > threshold
        ]
        new_count = len(self.promotions["promotions"])
        
        # Update metadata
        self.promotions["meta"]["last_updated"] = now.isoformat()
        self.promotions["meta"]["count"] = new_count
        
        # Save promotions
        self._save_promotions()
        
        self.logger.info(f"Cleared {old_count - new_count} expired promotions")


# Create singleton instance
_strategy_promoter = None

def get_strategy_promoter():
    """
    Get the singleton StrategyPromoter instance.
    
    Returns:
        StrategyPromoter instance
    """
    global _strategy_promoter
    if _strategy_promoter is None:
        _strategy_promoter = StrategyPromoter()
    return _strategy_promoter


if __name__ == "__main__":
    # Example usage
    promoter = get_strategy_promoter()
    
    # Example promotion test
    result = {
        "sharpe_ratio": 1.8,
        "total_return": 15.0,
        "win_rate": 0.65,
        "max_drawdown": -5.0
    }
    
    score = {"score": 0.9}
    
    evaluation = promoter.evaluate_and_promote(
        symbol="AAPL",
        strategy="momentum_breakout",
        result=result,
        params={"ma_short": 10, "ma_long": 50},
        score=score,
        market_regime="trending"
    )
    
    print(f"Promotion result: {evaluation['promoted']}")
    print(f"Reason: {evaluation['reason']}")
    
    # Get top promotions
    top = promoter.get_top_promotions(limit=3)
    print(f"Top promotions: {len(top)}")
    for p in top:
        print(f"{p['symbol']} - {p['strategy']} - Score: {p['score']:.2f} - Sharpe: {p['sharpe_ratio']:.2f}")
