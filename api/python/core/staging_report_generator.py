"""
Staging Report Generator for comprehensive reports on strategy performance during testing.
"""
import os
import logging
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from trading_bot.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class StagingReport:
    """
    Generates comprehensive reports on strategy performance during the staging phase.
    
    This helps evaluate strategies before promoting them to live trading,
    identifying potential issues and ensuring they meet all validation criteria.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the staging report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.service_registry = ServiceRegistry.get_instance()
        self.service_registry.register_service("staging_report", self)
        
        self.config = config or {}
        self.reports_directory = self.config.get("reports_directory", "./reports/staging")
        self.report_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Ensure reports directory exists
        os.makedirs(self.reports_directory, exist_ok=True)
        
        # Validation checkpoints from config
        self.validation_checkpoints = self.config.get("validation_checkpoints", {
            "min_sharpe_ratio": 0.8,
            "max_drawdown_pct": -10.0,
            "min_win_rate": 0.4,
            "max_daily_loss_pct": -3.0,
            "resource_utilization_threshold": 80,
            "min_profit_factor": 1.2,
            "min_trades": 30,
            "max_error_rate": 0.01
        })
        
        logger.info("Staging report generator initialized")
    
    def generate_strategy_report(self, strategy_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a single strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict: Strategy report
        """
        # Get required services
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        performance_tracker = self.service_registry.get_service("performance_tracker")
        risk_detector = self.service_registry.get_service("risk_violation_detector")
        health_monitor = self.service_registry.get_service("system_health_monitor")
        
        if not workflow or not performance_tracker:
            logger.error("Required services not available")
            return {"error": "Required services not available"}
        
        # Get strategy info
        strategy_info = workflow.get_strategy_info(strategy_id)
        if not strategy_info:
            return {"error": f"Strategy {strategy_id} not found"}
        
        # Get performance metrics
        performance = performance_tracker.get_strategy_metrics(strategy_id)
        
        # Get trades
        trades = performance_tracker.get_recent_trades(strategy_id, limit=100)
        
        # Get risk violations if available
        violations = []
        if risk_detector:
            violations = risk_detector.get_violations(strategy_id).get(strategy_id, [])
        
        # Get system health data if available
        system_health = {}
        if health_monitor:
            system_health = health_monitor.get_latest_metrics()
        
        # Evaluate strategy against validation checkpoints
        validation_results = self._evaluate_validation_checkpoints(strategy_id, performance, trades, violations, system_health)
        
        # Create report
        report = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_info.get("name", "Unknown"),
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": performance,
            "validation_results": validation_results,
            "risk_violations": violations,
            "trade_summary": self._generate_trade_summary(trades),
            "system_health": system_health,
            "overall_status": "PASS" if validation_results["passed_all"] else "FAIL",
            "recommendation": self._generate_recommendation(validation_results, violations)
        }
        
        # Add to report history
        if strategy_id not in self.report_history:
            self.report_history[strategy_id] = []
        
        self.report_history[strategy_id].append(report)
        
        # Save report to file
        self._save_report_to_file(strategy_id, report)
        
        return report
    
    def _evaluate_validation_checkpoints(
        self, 
        strategy_id: str, 
        performance: Dict[str, Any], 
        trades: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        system_health: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy against all validation checkpoints.
        
        Args:
            strategy_id: Strategy identifier
            performance: Performance metrics
            trades: Recent trades
            violations: Risk violations
            system_health: System health metrics
            
        Returns:
            Dict: Validation results
        """
        results = {}
        
        # Check Sharpe ratio
        sharpe = performance.get("sharpe_ratio", 0)
        min_sharpe = self.validation_checkpoints.get("min_sharpe_ratio", 0.8)
        results["sharpe_ratio"] = {
            "value": sharpe,
            "threshold": min_sharpe,
            "passed": sharpe >= min_sharpe
        }
        
        # Check drawdown
        drawdown = performance.get("max_drawdown_pct", 0)
        max_drawdown = self.validation_checkpoints.get("max_drawdown_pct", -10.0)
        results["max_drawdown"] = {
            "value": drawdown,
            "threshold": max_drawdown,
            "passed": drawdown >= max_drawdown  # drawdown is negative, so >= means better
        }
        
        # Check win rate
        win_rate = performance.get("win_rate", 0)
        min_win_rate = self.validation_checkpoints.get("min_win_rate", 0.4)
        results["win_rate"] = {
            "value": win_rate,
            "threshold": min_win_rate,
            "passed": win_rate >= min_win_rate
        }
        
        # Check profit factor
        profit_factor = performance.get("profit_factor", 0)
        min_profit_factor = self.validation_checkpoints.get("min_profit_factor", 1.2)
        results["profit_factor"] = {
            "value": profit_factor,
            "threshold": min_profit_factor,
            "passed": profit_factor >= min_profit_factor
        }
        
        # Check number of trades
        num_trades = len(trades)
        min_trades = self.validation_checkpoints.get("min_trades", 30)
        results["trade_count"] = {
            "value": num_trades,
            "threshold": min_trades,
            "passed": num_trades >= min_trades
        }
        
        # Check for risk violations
        has_violations = len(violations) > 0
        results["risk_violations"] = {
            "value": len(violations),
            "threshold": 0,
            "passed": not has_violations
        }
        
        # Check system health (if available)
        if system_health:
            error_rate = system_health.get("error_rate", 0)
            max_error_rate = self.validation_checkpoints.get("max_error_rate", 0.01)
            results["error_rate"] = {
                "value": error_rate,
                "threshold": max_error_rate,
                "passed": error_rate <= max_error_rate
            }
        
        # Overall result
        passed_all = all(result["passed"] for result in results.values())
        
        return {
            "checkpoints": results,
            "passed_all": passed_all,
            "pass_count": sum(1 for result in results.values() if result["passed"]),
            "total_count": len(results)
        }
    
    def _generate_trade_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of trade performance.
        
        Args:
            trades: List of trades
            
        Returns:
            Dict: Trade summary
        """
        if not trades:
            return {"total_trades": 0}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        
        # Handle case where DataFrame might be empty or missing columns
        if df.empty or "pnl" not in df.columns:
            return {"total_trades": 0}
        
        # Basic stats
        total_trades = len(df)
        winning_trades = df[df["pnl"] > 0].shape[0]
        losing_trades = df[df["pnl"] < 0].shape[0]
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = df[df["pnl"] > 0]["pnl"].sum() if not df[df["pnl"] > 0].empty else 0
        total_loss = df[df["pnl"] < 0]["pnl"].sum() if not df[df["pnl"] < 0].empty else 0
        
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # By symbol
        by_symbol = df.groupby("symbol").agg({
            "pnl": ["sum", "mean", "count"]
        }).reset_index()
        
        by_symbol.columns = ["symbol", "total_pnl", "avg_pnl", "trade_count"]
        
        symbol_performance = by_symbol.to_dict(orient="records")
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "by_symbol": symbol_performance
        }
    
    def _generate_recommendation(
        self, 
        validation_results: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate recommendations based on validation results.
        
        Args:
            validation_results: Validation results
            violations: Risk violations
            
        Returns:
            Dict: Recommendation
        """
        passed_all = validation_results["passed_all"]
        checkpoints = validation_results["checkpoints"]
        
        if passed_all and not violations:
            return {
                "status": "READY_FOR_LIVE",
                "message": "Strategy has passed all validation checkpoints and has no risk violations. Ready for live trading.",
                "actions": ["Submit for final approval", "Promote to live trading"]
            }
        
        # Find failing checkpoints
        failing_checkpoints = [
            name for name, result in checkpoints.items() 
            if not result["passed"]
        ]
        
        # Generate specific recommendations
        recommendations = []
        
        for checkpoint in failing_checkpoints:
            if checkpoint == "sharpe_ratio":
                recommendations.append("Improve risk-adjusted returns by refining entry/exit criteria")
            elif checkpoint == "max_drawdown":
                recommendations.append("Reduce maximum drawdown by implementing tighter stop losses")
            elif checkpoint == "win_rate":
                recommendations.append("Increase win rate by improving signal quality or filtering criteria")
            elif checkpoint == "profit_factor":
                recommendations.append("Improve profit factor by increasing average win size or reducing average loss size")
            elif checkpoint == "trade_count":
                recommendations.append("Continue testing to accumulate more trades for statistical significance")
            elif checkpoint == "risk_violations":
                recommendations.append("Address risk violations by adjusting position sizing or risk parameters")
            elif checkpoint == "error_rate":
                recommendations.append("Investigate and fix system errors to improve reliability")
        
        # Overall recommendation
        if len(failing_checkpoints) <= 2 and validation_results["pass_count"] / validation_results["total_count"] >= 0.7:
            status = "NEEDS_IMPROVEMENT"
            message = "Strategy shows promise but needs improvements in specific areas before live trading."
        else:
            status = "SIGNIFICANT_ISSUES"
            message = "Strategy has significant issues that need to be addressed before considering live trading."
        
        return {
            "status": status,
            "message": message,
            "failing_checkpoints": failing_checkpoints,
            "recommendations": recommendations,
            "actions": ["Continue paper trading", "Adjust strategy parameters", "Address specific issues"]
        }
    
    def _save_report_to_file(self, strategy_id: str, report: Dict[str, Any]) -> None:
        """
        Save a report to a JSON file.
        
        Args:
            strategy_id: Strategy identifier
            report: Report data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.reports_directory}/staging_report_{strategy_id}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved staging report to {filename}")
        except Exception as e:
            logger.error(f"Error saving report to file: {str(e)}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report for all strategies in staging.
        
        Returns:
            Dict: Comprehensive report
        """
        # Get staging manager
        staging_manager = self.service_registry.get_service("staging_mode_manager")
        if not staging_manager:
            return {"error": "Staging manager not available"}
        
        # Get all strategies in staging
        strategies = []
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        if workflow:
            strategies = workflow.get_all_strategies()
        
        # Generate individual reports
        strategy_reports = {}
        for strategy in strategies:
            strategy_id = strategy.get("id")
            if strategy_id:
                report = self.generate_strategy_report(strategy_id)
                strategy_reports[strategy_id] = report
        
        # Get system health
        system_health = {}
        health_monitor = self.service_registry.get_service("system_health_monitor")
        if health_monitor:
            system_health = health_monitor.generate_health_report()
        
        # Get risk report
        risk_report = {}
        risk_detector = self.service_registry.get_service("risk_violation_detector")
        if risk_detector:
            risk_report = risk_detector.generate_risk_report()
        
        # Staging duration
        staging_duration = staging_manager.get_staging_duration()
        
        # Overall staging status
        strategies_ready = sum(1 for report in strategy_reports.values() 
                              if report.get("overall_status") == "PASS")
        
        total_strategies = len(strategy_reports)
        has_met_duration = staging_manager.has_met_minimum_duration()
        
        if has_met_duration and strategies_ready == total_strategies and total_strategies > 0:
            overall_status = "READY_FOR_PRODUCTION"
            recommendation = "All strategies have passed validation criteria and the minimum staging duration has been met. The system is ready for live trading."
        elif has_met_duration and strategies_ready > 0:
            overall_status = "PARTIALLY_READY"
            recommendation = f"{strategies_ready} out of {total_strategies} strategies are ready for live trading. Consider promoting ready strategies while continuing to test others."
        elif strategies_ready > 0:
            overall_status = "PROMISING_BUT_INSUFFICIENT_DURATION"
            recommendation = "Some strategies show promise but haven't met the minimum staging duration. Continue testing until the minimum duration is reached."
        else:
            overall_status = "NEEDS_IMPROVEMENT"
            recommendation = "No strategies have passed all validation criteria. Address the specific issues highlighted in the individual strategy reports."
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "staging_duration": {
                "days": staging_duration.days,
                "hours": staging_duration.seconds // 3600,
                "total_seconds": staging_duration.total_seconds()
            },
            "has_met_minimum_duration": has_met_duration,
            "strategies_count": {
                "total": total_strategies,
                "ready": strategies_ready,
                "needs_improvement": total_strategies - strategies_ready
            },
            "overall_status": overall_status,
            "recommendation": recommendation,
            "system_health_summary": system_health,
            "risk_summary": risk_report,
            "strategy_reports": strategy_reports
        }
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.reports_directory}/comprehensive_staging_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved comprehensive staging report to {filename}")
        except Exception as e:
            logger.error(f"Error saving comprehensive report to file: {str(e)}")
        
        return report
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate a daily summary report for strategies in staging.
        
        Returns:
            Dict: Daily report
        """
        # Similar to comprehensive report but with daily focus
        # and less detail
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        if not workflow:
            return {"error": "Workflow service not available"}
        
        # Get all strategies
        strategies = workflow.get_all_strategies()
        
        # Get performance for each strategy
        performance_tracker = self.service_registry.get_service("performance_tracker")
        daily_performance = {}
        
        if performance_tracker:
            for strategy in strategies:
                strategy_id = strategy.get("id")
                if strategy_id:
                    # Get daily metrics
                    metrics = performance_tracker.get_strategy_metrics(strategy_id)
                    daily_return = metrics.get("daily_return_pct", 0)
                    
                    daily_performance[strategy_id] = {
                        "name": strategy.get("name", "Unknown"),
                        "daily_return_pct": daily_return,
                        "daily_trades": metrics.get("daily_trades", 0),
                        "daily_win_rate": metrics.get("daily_win_rate", 0)
                    }
        
        # Get system health
        system_health = {}
        health_monitor = self.service_registry.get_service("system_health_monitor")
        if health_monitor:
            system_health = health_monitor.get_latest_metrics()
        
        # Get risk violations for the day
        risk_violations = {}
        risk_detector = self.service_registry.get_service("risk_violation_detector")
        if risk_detector:
            all_violations = risk_detector.get_violations()
            
            # Filter to today's violations
            today = datetime.now().date()
            for strategy_id, violations in all_violations.items():
                today_violations = [
                    v for v in violations 
                    if datetime.fromisoformat(v["timestamp"]).date() == today
                ]
                if today_violations:
                    risk_violations[strategy_id] = today_violations
        
        report = {
            "date": datetime.now().date().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "strategies_count": len(strategies),
            "daily_performance": daily_performance,
            "system_health": system_health,
            "risk_violations": risk_violations,
            "summary": {
                "avg_daily_return": sum(p["daily_return_pct"] for p in daily_performance.values()) / len(daily_performance) if daily_performance else 0,
                "strategies_with_positive_return": sum(1 for p in daily_performance.values() if p["daily_return_pct"] > 0),
                "strategies_with_violations": len(risk_violations)
            }
        }
        
        # Save daily report
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{self.reports_directory}/daily_staging_report_{date_str}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved daily staging report to {filename}")
        except Exception as e:
            logger.error(f"Error saving daily report to file: {str(e)}")
        
        return report
