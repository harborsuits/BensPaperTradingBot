"""
Orchestrator Context Endpoints

This module provides FastAPI endpoints to expose the orchestrator context
to the frontend for comprehensive trading context awareness.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Initialize logger
logger = logging.getLogger("api.orchestrator_context")

# Create router
router = APIRouter(prefix="/ai/orchestrator", tags=["OrchestratorContext"])

# Real data extraction from trading system components

# Import required system components
from trading_bot.components.market_analyzer import MarketAnalyzer
from trading_bot.components.portfolio_manager import PortfolioManager
from trading_bot.components.strategy_manager import StrategyManager
from trading_bot.components.position_manager import PositionManager
from trading_bot.components.system_monitor import SystemMonitor
from trading_bot.components.anomaly_detector import AnomalyDetector
from trading_bot.performance.performance_tracker import PerformanceTracker

# Global references to system components
_market_analyzer = None
_portfolio_manager = None
_strategy_manager = None
_position_manager = None
_system_monitor = None
_anomaly_detector = None
_performance_tracker = None

# Logger
logger = get_logger("orchestrator_context_endpoints")

def register_anomaly_detector(anomaly_detector: AnomalyDetector):
    """Register the anomaly detector instance"""
    global _anomaly_detector
    _anomaly_detector = anomaly_detector
    logger.info("Anomaly detector registered with orchestrator endpoints")
    return _anomaly_detector

def initialize_dependencies(market_analyzer=None, portfolio_manager=None, strategy_manager=None,
                          position_manager=None, system_monitor=None, anomaly_detector=None,
                          performance_tracker=None):
    """Initialize system component dependencies for the orchestrator context endpoints"""
    global _market_analyzer, _portfolio_manager, _strategy_manager, _position_manager
    global _system_monitor, _anomaly_detector, _performance_tracker
    
    _market_analyzer = market_analyzer
    _portfolio_manager = portfolio_manager
    _strategy_manager = strategy_manager
    _position_manager = position_manager
    _system_monitor = system_monitor
    _anomaly_detector = anomaly_detector
    _performance_tracker = performance_tracker
    
    logger.info("Orchestrator context dependencies initialized")

def get_orchestrator_context_data():
    """Get real orchestrator context data from system components"""
    context = {
        "market": get_market_context(),
        "portfolio": get_portfolio_context(),
        "strategies": get_strategies_context(),
        "trades": get_trades_context(),
        "system": get_system_context()
    }
    
    return context

def get_market_context():
    """Get real market context data from MarketAnalyzer"""
    if not _market_analyzer:
        logger.warning("MarketAnalyzer not initialized, returning minimal market context")
        return {
            "regime": "unknown",
            "regimeConfidence": 0.0,
            "lastUpdated": datetime.now().isoformat()
        }
    
    try:
        # Get real market regime data from market analyzer
        market_state = _market_analyzer.get_current_market_state()
        regime_info = _market_analyzer.get_regime_info()
        sentiment_info = _market_analyzer.get_market_sentiment()
        volatility_info = _market_analyzer.get_volatility_metrics()
        
        # Get major indices data
        indices_data = {}
        for index in ["SPY", "QQQ", "IWM", "DIA"]:
            if index in _market_analyzer.tracked_symbols:
                index_data = _market_analyzer.get_symbol_data(index)
                if index_data:
                    indices_data[index] = {
                        "price": index_data.get("last_price", 0.0),
                        "change": index_data.get("change_percent", 0.0),
                        "volume": index_data.get("volume", 0)
                    }
        
        # Format positive and negative sentiment factors
        sentiment_factors = {
            "positive": [],
            "negative": []
        }
        
        if sentiment_info and "factors" in sentiment_info:
            for factor in sentiment_info["factors"]:
                if factor.get("impact", 0) > 0:
                    sentiment_factors["positive"].append({
                        "factor": factor.get("name", "Unknown factor"),
                        "impact": factor.get("impact", 0.0)
                    })
                else:
                    sentiment_factors["negative"].append({
                        "factor": factor.get("name", "Unknown factor"),
                        "impact": abs(factor.get("impact", 0.0))
                    })
        
        # Build complete market context
        market_context = {
            "regime": regime_info.get("current_regime", "unknown"),
            "regimeConfidence": regime_info.get("confidence", 0.0),
            "volatility": volatility_info.get("current_level", 0.0),
            "volatilityTrend": volatility_info.get("trend", "stable"),
            "sentiment": sentiment_info.get("overall_score", 0.0),
            "sentimentFactors": sentiment_factors,
            "majorIndices": indices_data,
            "lastUpdated": market_state.get("last_updated", datetime.now().isoformat())
        }
        
        return market_context
        
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}")
        return {
            "regime": "error",
            "error": str(e),
            "lastUpdated": datetime.now().isoformat()
        }

def get_portfolio_context():
    """Get real portfolio context data from PortfolioManager"""
    if not _portfolio_manager or not _position_manager:
        logger.warning("Portfolio/Position managers not initialized, returning minimal portfolio context")
        return {
            "value": 0.0,
            "cashBalance": 0.0,
            "lastUpdated": datetime.now().isoformat()
        }
    
    try:
        # Get real portfolio data
        portfolio_summary = _portfolio_manager.get_portfolio_summary()
        positions = _position_manager.get_all_positions()
        
        # Format positions data
        formatted_positions = []
        for position in positions:
            formatted_positions.append({
                "symbol": position.get("symbol"),
                "assetClass": position.get("asset_class", "stock"),
                "quantity": position.get("quantity", 0),
                "entryPrice": position.get("entry_price", 0.0),
                "currentPrice": position.get("current_price", 0.0),
                "pnl": position.get("unrealized_pnl", 0.0),
                "pnlPercent": position.get("unrealized_pnl_percent", 0.0),
                "strategy": position.get("strategy", "unknown")
            })
        
        # Calculate allocation if available
        allocation = {}
        if "allocation" in portfolio_summary:
            allocation = portfolio_summary["allocation"]
        elif "asset_allocation" in portfolio_summary:
            allocation = portfolio_summary["asset_allocation"]
        else:
            # Derive allocation from positions if not directly available
            total_value = portfolio_summary.get("total_value", 0.0)
            if total_value > 0:
                asset_classes = {}
                for position in positions:
                    asset_class = position.get("asset_class", "stock")
                    position_value = position.get("market_value", 0.0)
                    asset_classes[asset_class] = asset_classes.get(asset_class, 0.0) + position_value
                
                cash_value = portfolio_summary.get("cash_balance", 0.0)
                asset_classes["cash"] = cash_value
                
                # Calculate percentages
                for asset_class, value in asset_classes.items():
                    allocation[asset_class] = value / total_value
        
        return {
            "value": portfolio_summary.get("total_value", 0.0),
            "cashBalance": portfolio_summary.get("cash_balance", 0.0),
            "allocation": allocation,
            "openPositions": formatted_positions,
            "lastUpdated": portfolio_summary.get("last_updated", datetime.now().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio context: {str(e)}")
        return {
            "value": 0.0,
            "cashBalance": 0.0,
            "error": str(e),
            "lastUpdated": datetime.now().isoformat()
        }

def get_strategies_context():
    """Get real strategies context data from StrategyManager"""
    if not _strategy_manager:
        logger.warning("StrategyManager not initialized, returning minimal strategies context")
        return {
            "activeStrategies": [],
            "lastUpdated": datetime.now().isoformat()
        }
    
    try:
        # Get strategies data from strategy manager
        strategies = _strategy_manager.get_all_strategies()
        active_strategies = _strategy_manager.get_active_strategies()
        strategy_performance = {}
        
        # Get performance data if performance tracker is available
        if _performance_tracker:
            strategy_performance = _performance_tracker.get_strategy_performance()
        
        # Format strategies data
        formatted_strategies = []
        for strategy in strategies:
            strategy_id = strategy.get("id") or strategy.get("name")
            
            # Get performance data for this strategy
            performance_data = {}
            if strategy_id in strategy_performance:
                performance_data = strategy_performance[strategy_id]
            
            formatted_strategy = {
                "id": strategy_id,
                "name": strategy.get("name", "Unknown Strategy"),
                "assetClass": strategy.get("asset_class", "stocks"),
                "isActive": strategy_id in active_strategies,
                "confidence": strategy.get("confidence", 0.0),
                "lastSignal": strategy.get("last_signal", None),
                "lastSignalTime": strategy.get("last_signal_time"),
                "performance": {
                    "winRate": performance_data.get("win_rate", 0.0),
                    "avgReturn": performance_data.get("avg_return", 0.0),
                    "sharpe": performance_data.get("sharpe", 0.0)
                }
            }
            
            formatted_strategies.append(formatted_strategy)
        
        return {
            "activeStrategies": formatted_strategies,
            "lastUpdated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting strategies context: {str(e)}")
        return {
            "activeStrategies": [],
            "error": str(e),
            "lastUpdated": datetime.now().isoformat()
        }

def get_trades_context():
    """Get real trades context data from PositionManager"""
    if not _position_manager:
        logger.warning("PositionManager not initialized, returning minimal trades context")
        return {
            "recentTrades": [],
            "lastUpdated": datetime.now().isoformat()
        }
    
    try:
        # Get recent trades data
        recent_trades = _position_manager.get_recent_trades(limit=10)
        
        # Format trades data
        formatted_trades = []
        for trade in recent_trades:
            formatted_trades.append({
                "symbol": trade.get("symbol"),
                "direction": trade.get("direction", "long"),
                "quantity": trade.get("quantity", 0),
                "price": trade.get("execution_price", 0.0),
                "timestamp": trade.get("execution_time"),
                "strategy": trade.get("strategy", "unknown"),
                "pnl": trade.get("realized_pnl", 0.0),
                "status": trade.get("status", "completed")
            })
        
        return {
            "recentTrades": formatted_trades,
            "lastUpdated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trades context: {str(e)}")
        return {
            "recentTrades": [],
            "error": str(e),
            "lastUpdated": datetime.now().isoformat()
        }

def get_system_context():
    """Get real system context data from SystemMonitor"""
    if not _system_monitor:
        logger.warning("SystemMonitor not initialized, returning minimal system context")
        return {
            "health": {
                "status": "unknown",
                "lastUpdated": datetime.now().isoformat()
            }
        }
    
    try:
        # Get system health data
        system_health = _system_monitor.get_system_health()
        api_latency = _system_monitor.get_api_latency()
        error_rates = _system_monitor.get_error_rates()
        data_quality = _system_monitor.get_data_quality_metrics()
        
        return {
            "health": {
                "status": system_health.get("status", "unknown"),
                "memoryUsage": system_health.get("memory_usage", 0.0),
                "cpuUsage": system_health.get("cpu_usage", 0.0),
                "apiLatency": api_latency.get("average_ms", 0),
                "errorRate": error_rates.get("overall_rate", 0.0),
                "dataQuality": data_quality.get("overall_score", 0.0),
                "lastUpdated": system_health.get("last_updated", datetime.now().isoformat())
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system context: {str(e)}")
        return {
            "health": {
                "status": "error",
                "error": str(e),
                "lastUpdated": datetime.now().isoformat()
            }
        }

def get_real_strategy_rankings():
    """Get real strategy ranking data from the strategy manager and market analyzer"""
    if not _strategy_manager or not _market_analyzer:
        logger.warning("Strategy manager or market analyzer not initialized, returning minimal strategy rankings")
        return {
            "marketRegime": "unknown",
            "regimeConfidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "rankedStrategies": []
        }
    
    try:
        # Get current market regime from market analyzer
        regime_info = _market_analyzer.get_regime_info()
        current_regime = regime_info.get("current_regime", "unknown")
        regime_confidence = regime_info.get("confidence", 0.0)
        
        # Get strategy rankings from strategy manager
        strategy_rankings = _strategy_manager.get_strategy_rankings()
        
        # Format ranked strategies
        ranked_strategies = []
        for strategy in strategy_rankings:
            # Get strategy metadata
            strategy_id = strategy.get("id") or strategy.get("name")
            strategy_metadata = _strategy_manager.get_strategy_metadata(strategy_id)
            
            # Build strategy entry with detailed information
            strategy_entry = {
                "strategy": strategy.get("name"),
                "assetClass": strategy.get("asset_class", "stocks"),
                "suitabilityScore": strategy.get("suitability_score", 0.0),
                "confidenceScore": strategy.get("confidence_score", 0.0),
                "rationale": strategy.get("rationale", "No rationale provided"),
                "expectedPerformance": strategy.get("expected_performance", "unknown")
            }
            
            # Add regime compatibility if available in metadata
            if strategy_metadata and "regime_compatibility" in strategy_metadata:
                strategy_entry["regimeCompatibility"] = strategy_metadata["regime_compatibility"]
            
            ranked_strategies.append(strategy_entry)
        
        # Sort strategies by suitability score
        ranked_strategies.sort(key=lambda x: x.get("suitabilityScore", 0), reverse=True)
        
        return {
            "marketRegime": current_regime,
            "regimeConfidence": regime_confidence,
            "timestamp": datetime.now().isoformat(),
            "rankedStrategies": ranked_strategies
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy rankings: {str(e)}")
        return {
            "marketRegime": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "rankedStrategies": []
        }

# Endpoint for the full orchestrator context
@router.get("/context")
async def get_orchestrator_context():
    """Get the complete orchestrator context with data from all parts of the system"""
    # Get real data from the orchestrator context data function
    # This pulls data from various system components:
    # - Market analyzer for regime, sentiment, etc.
    # - Portfolio manager for positions, value, etc.
    # - Strategy manager for active strategies
    # - System monitor for health metrics
    return get_orchestrator_context_data()

# Asset-specific context endpoint
@router.get("/context/{asset_class}")
async def get_asset_specific_context(asset_class: Literal["stocks", "options", "crypto", "forex"]):
    """Get context data filtered for a specific asset class"""
    # Get the full context from the real data
    full_context = get_orchestrator_context_data()
    
    # Create an asset-specific view of the context
    asset_context = {
        "market": full_context.get("market", {}),  # Include full market data
        "portfolio": {
            "value": full_context.get("portfolio", {}).get("value", 0.0),
            "cashBalance": full_context.get("portfolio", {}).get("cashBalance", 0.0),
            # Filter positions by asset class
            "openPositions": [
                position for position in full_context.get("portfolio", {}).get("openPositions", [])
                if position.get("assetClass", "").lower() == asset_class.lower()
            ],
            # Include asset allocation
            "allocation": full_context.get("portfolio", {}).get("allocation", {})
        },
        "system": full_context.get("system", {}),  # Include system health
        "assetClass": asset_class  # Add asset class identifier
    }
    
    # Add asset-specific data based on the asset class
    if asset_class == "stocks" and _market_analyzer:
        try:
            # Get real stock-specific data
            sector_data = _market_analyzer.get_sector_performance()
            market_cap_data = _market_analyzer.get_market_cap_performance()
            style_data = _market_analyzer.get_investment_style_performance()
            
            asset_context["stockSpecific"] = {
                "sectorPerformance": sector_data,
                "marketCap": market_cap_data,
                "style": style_data
            }
        except Exception as e:
            logger.error(f"Error getting stock-specific data: {str(e)}")
            # Provide minimal data in case of error
            asset_context["stockSpecific"] = {
                "sectorPerformance": {},
                "marketCap": {},
                "style": {},
                "error": str(e)
            }
    
    elif asset_class == "options" and _market_analyzer:
        try:
            # Get real options-specific data
            options_data = _market_analyzer.get_options_market_data()
            
            asset_context["optionsSpecific"] = {
                "impliedVolatility": options_data.get("implied_volatility", {}),
                "putCallRatio": options_data.get("put_call_ratio", 0.0),
                "optionVolume": options_data.get("volume", {}),
                "expirationClusters": options_data.get("expiration_clusters", [])
            }
        except Exception as e:
            logger.error(f"Error getting options-specific data: {str(e)}")
            # Provide minimal data in case of error
            asset_context["optionsSpecific"] = {
                "impliedVolatility": {},
                "putCallRatio": 0.0,
                "optionVolume": {},
                "expirationClusters": [],
                "error": str(e)
            }
    
    elif asset_class == "crypto" and _market_analyzer:
        try:
            # Get real crypto-specific data
            crypto_data = _market_analyzer.get_crypto_market_data()
            
            asset_context["cryptoSpecific"] = {
                "bitcoinDominance": crypto_data.get("bitcoin_dominance", 0.0),
                "exchangeFlows": crypto_data.get("exchange_flows", {}),
                "defiTVL": crypto_data.get("defi_tvl", 0.0),
                "networkHealth": crypto_data.get("network_health", {})
            }
        except Exception as e:
            logger.error(f"Error getting crypto-specific data: {str(e)}")
            # Provide minimal data in case of error
            asset_context["cryptoSpecific"] = {
                "bitcoinDominance": 0.0,
                "exchangeFlows": {},
                "defiTVL": 0.0,
                "networkHealth": {},
                "error": str(e)
            }
    
    elif asset_class == "forex" and _market_analyzer:
        try:
            # Get real forex-specific data
            forex_data = _market_analyzer.get_forex_market_data()
            
            asset_context["forexSpecific"] = {
                "interestRates": forex_data.get("interest_rates", {}),
                "economicIndicators": forex_data.get("economic_indicators", {}),
                "centralBankStance": forex_data.get("central_bank_stance", {})
            }
        except Exception as e:
            logger.error(f"Error getting forex-specific data: {str(e)}")
            # Provide minimal data in case of error
            asset_context["forexSpecific"] = {
                "interestRates": {},
                "economicIndicators": {},
                "centralBankStance": {},
                "error": str(e)
            }
    
    return asset_context

# Strategy prioritization endpoint
@router.get("/strategies/ranking")
async def get_strategy_rankings(asset_class: Optional[str] = None):
    """Get the current ranking of strategies based on market conditions
    
    Parameters:
    - asset_class: Optional filter for strategies of a specific asset class
    """
    # Get the real strategy rankings
    rankings = get_real_strategy_rankings()
    
    # Filter by asset class if requested
    if asset_class:
        rankings["rankedStrategies"] = [
            strategy for strategy in rankings["rankedStrategies"]
            if strategy.get("assetClass", "").lower() == asset_class.lower()
        ]
    
    return rankings


def get_real_trade_candidates():
    """Get real trade candidates with evaluation scores and rationale from the strategy manager"""
    if not _strategy_manager:
        logger.warning("Strategy manager not initialized, returning minimal trade candidates")
        return {
            "timestamp": datetime.now().isoformat(),
            "candidates": []
        }
    
    try:
        # Get current trade candidates from strategy manager
        trade_candidates = _strategy_manager.get_trade_candidates()
        
        # Format candidates data
        formatted_candidates = []
        for candidate in trade_candidates:
            # Format the rationale with factors analysis
            rationale = {
                "primary": candidate.get("rationale", "No rationale provided")
            }
            
            # Add technical factors if available
            if "technical_factors" in candidate:
                rationale["technicalFactors"] = [
                    {
                        "factor": factor.get("name", "Unknown factor"),
                        "impact": factor.get("impact", 0.0)
                    } for factor in candidate["technical_factors"]
                ]
            
            # Add fundamental factors if available
            if "fundamental_factors" in candidate:
                rationale["fundamentalFactors"] = [
                    {
                        "factor": factor.get("name", "Unknown factor"),
                        "impact": factor.get("impact", 0.0)
                    } for factor in candidate["fundamental_factors"]
                ]
            
            # Add options-specific factors if available
            if "options_factors" in candidate:
                rationale["optionsSpecificFactors"] = [
                    {
                        "factor": factor.get("name", "Unknown factor"),
                        "impact": factor.get("impact", 0.0)
                    } for factor in candidate["options_factors"]
                ]
            
            # Add risk assessment if available
            if "risk_assessment" in candidate:
                risk_data = candidate["risk_assessment"]
                rationale["riskAssessment"] = {
                    "stopLossLevel": risk_data.get("stop_loss_level", 0.0),
                    "riskRewardRatio": risk_data.get("risk_reward_ratio", 0.0),
                    "potentialDownside": risk_data.get("potential_downside", 0.0),
                    "potentialUpside": risk_data.get("potential_upside", 0.0)
                }
                
                # Add options-specific risk metrics if available
                if "max_loss" in risk_data:
                    rationale["riskAssessment"].update({
                        "maxLoss": risk_data.get("max_loss", 0.0),
                        "maxGain": risk_data.get("max_gain", 0.0),
                        "probabilityOfProfit": risk_data.get("probability_of_profit", 0.0)
                    })
            
            # Format the estimated position if available
            estimated_position = None
            if "estimated_position" in candidate:
                position_data = candidate["estimated_position"]
                estimated_position = {
                    "quantity": position_data.get("quantity", 0),
                    "entryPrice": position_data.get("entry_price", 0.0),
                    "stopLoss": position_data.get("stop_loss", 0.0),
                    "takeProfit": position_data.get("take_profit", 0.0),
                    "riskPercentage": position_data.get("risk_percentage", 0.0)
                }
                
                # Add options-specific position data if available
                if "entry_debit" in position_data:
                    estimated_position.update({
                        "entryDebit": position_data.get("entry_debit", 0.0),
                        "maxRiskPercentage": position_data.get("max_risk_percentage", 0.0)
                    })
            
            # Build the complete candidate entry
            candidate_entry = {
                "symbol": candidate.get("symbol", "Unknown"),
                "assetClass": candidate.get("asset_class", "stocks"),
                "direction": candidate.get("direction", "long"),
                "strategy": candidate.get("strategy", "Unknown"),
                "score": candidate.get("score", 0.0),
                "confidenceLevel": candidate.get("confidence_level", "low"),
                "status": candidate.get("status", "rejected"),
                "rationale": rationale
            }
            
            # Add estimated position if available
            if estimated_position:
                candidate_entry["estimatedPosition"] = estimated_position
            
            formatted_candidates.append(candidate_entry)
        
        # Sort candidates by score
        formatted_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "candidates": formatted_candidates
        }
        
    except Exception as e:
        logger.error(f"Error getting trade candidates: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "candidates": []
        }

# Trade candidates endpoint
@router.get("/decisions/candidates")
async def get_trade_candidates(
    asset_class: Optional[str] = None,
    status: Optional[Literal["selected", "watch", "rejected"]] = None
):
    """Get trade candidates with their scores and selection rationale
    
    Parameters:
    - asset_class: Optional filter for candidates of a specific asset class
    - status: Optional filter for candidates with a specific status (selected, watch, rejected)
    """
    # Get the real trade candidates
    candidates_data = get_real_trade_candidates()
    
    # Apply filters if specified
    filtered_candidates = candidates_data["candidates"]
    
    if asset_class:
        filtered_candidates = [
            candidate for candidate in filtered_candidates
            if candidate.get("assetClass", "").lower() == asset_class.lower()
        ]
    
    if status:
        filtered_candidates = [
            candidate for candidate in filtered_candidates
            if candidate.get("status", "").lower() == status.lower()
        ]
    
    # Update the candidates list with filtered results
    candidates_data["candidates"] = filtered_candidates
    
    return candidates_data


def get_real_market_anomalies():
    """Get real market anomalies with severity ratings and potential impacts from the anomaly detector"""
    if not _anomaly_detector:
        logger.warning("Anomaly detector not initialized, returning minimal anomalies data")
        return {
            "timestamp": datetime.now().isoformat(),
            "anomalyCount": 0,
            "anomalies": []
        }
    
    try:
        # Get current anomalies from the anomaly detector
        anomalies = _anomaly_detector.get_active_anomalies()
        
        # Format anomalies data
        formatted_anomalies = []
        for anomaly in anomalies:
            # Format metrics data
            metrics = {}
            if "metrics" in anomaly:
                metrics = anomaly["metrics"]
            
            # Format potential impact data
            potential_impact = {
                "description": anomaly.get("impact_description", "No impact description provided")
            }
            
            # Add outlooks if available
            if "short_term_outlook" in anomaly:
                potential_impact["shortTermOutlook"] = anomaly["short_term_outlook"]
            if "medium_term_outlook" in anomaly:
                potential_impact["mediumTermOutlook"] = anomaly["medium_term_outlook"]
            
            # Add recommended actions if available
            if "recommended_actions" in anomaly:
                potential_impact["recommendedActions"] = anomaly["recommended_actions"]
            
            # Map severity to a label
            severity = anomaly.get("severity", 0.0)
            severity_label = "low"
            if severity >= 0.7:
                severity_label = "high"
            elif severity >= 0.4:
                severity_label = "medium"
            
            # Build the complete anomaly entry
            anomaly_entry = {
                "id": anomaly.get("id", f"anomaly-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                "type": anomaly.get("type", "unknown"),
                "description": anomaly.get("description", "No description provided"),
                "severity": severity,
                "severityLabel": severity_label,
                "detectionTime": anomaly.get("detection_time", datetime.now().isoformat()),
                "affectedAssets": anomaly.get("affected_assets", []),
                "metrics": metrics,
                "potentialImpact": potential_impact,
                "status": anomaly.get("status", "active")
            }
            
            formatted_anomalies.append(anomaly_entry)
        
        # Sort anomalies by severity
        formatted_anomalies.sort(key=lambda x: x.get("severity", 0), reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "anomalyCount": len(formatted_anomalies),
            "anomalies": formatted_anomalies
        }
        
    except Exception as e:
        logger.error(f"Error getting market anomalies: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "anomalyCount": 0,
            "anomalies": []
        }

# Anomaly detection endpoint
@router.get("/anomalies")
async def get_market_anomalies(
    severity: Optional[Literal["low", "medium", "high"]] = None,
    status: Optional[Literal["active", "resolved", "monitoring"]] = None,
    type: Optional[str] = None
):
    """Get detected market anomalies with severity ratings and potential impacts
    
    Parameters:
    - severity: Optional filter for anomalies of a specific severity level
    - status: Optional filter for anomalies with a specific status
    - type: Optional filter for anomalies of a specific type
    """
    # Get the real market anomalies
    anomalies_data = get_real_market_anomalies()
    
    # Apply filters if specified
    filtered_anomalies = anomalies_data["anomalies"]
    
    if severity:
        filtered_anomalies = [
            anomaly for anomaly in filtered_anomalies
            if anomaly.get("severityLabel", "").lower() == severity.lower()
        ]
    
    if status:
        filtered_anomalies = [
            anomaly for anomaly in filtered_anomalies
            if anomaly.get("status", "").lower() == status.lower()
        ]
    
    if type:
        filtered_anomalies = [
            anomaly for anomaly in filtered_anomalies
            if anomaly.get("type", "").lower() == type.lower()
        ]
    
    # Update the anomalies list with filtered results
    anomalies_data["anomalies"] = filtered_anomalies
    anomalies_data["anomalyCount"] = len(filtered_anomalies)
    
    return anomalies_data
