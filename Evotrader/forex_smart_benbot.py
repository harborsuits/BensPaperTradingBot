#!/usr/bin/env python3
"""
Forex Smart BenBot Connector - Enhanced BenBot Integration

This module provides intelligent BenBot communication capabilities:
- Confidence-weighted decision making
- Evidence-based BenBot consultation
- Structured data feedback loop
- Performance tracking of BenBot decisions
"""

import os
import sys
import yaml
import json
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import math
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_smart_benbot')


class SmartBenBotConnector:
    """
    Enhanced BenBot integration with confidence-weighted decision making.
    """
    
    def __init__(self, benbot_endpoint: str, config: Dict[str, Any] = None):
        """
        Initialize the BenBot connector.
        
        Args:
            benbot_endpoint: URL endpoint for BenBot API
            config: Configuration dictionary
        """
        self.benbot_endpoint = benbot_endpoint
        self.config = config or {}
        
        # Decision history for performance tracking
        self.decision_history = []
        
        # Confidence history by action type
        self.confidence_history = {
            'trade_entry': [],
            'trade_exit': [],
            'risk_adjustment': [],
            'session_evaluation': [],
            'news_evaluation': []
        }
        
        # API configuration
        self.api_timeout = self.config.get('api_timeout', 10)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)
        
        # Decision tracking DB (optional)
        self.decision_db_path = self.config.get('decision_db_path', None)
        self.use_db = self.decision_db_path is not None
        
        # Load API credentials if available
        self.api_key = self.config.get('benbot_api_key', None)
        self.api_headers = {}
        if self.api_key:
            self.api_headers['Authorization'] = f"Bearer {self.api_key}"
        
        logger.info(f"Smart BenBot Connector initialized to endpoint: {benbot_endpoint}")
        
        # Initialize database if needed
        if self.use_db:
            self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the decision tracking database."""
        if not self.use_db:
            return
        
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.decision_db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS benbot_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                action_type TEXT,
                decision TEXT,
                confidence REAL,
                evidence TEXT,
                evotrader_decision TEXT,
                evotrader_confidence REAL,
                final_decision TEXT,
                outcome TEXT,
                outcome_score REAL
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"BenBot decision database initialized at {self.decision_db_path}")
        except Exception as e:
            logger.error(f"Error initializing decision database: {e}")
            self.use_db = False
    
    def consult_benbot_with_confidence(self, 
                                     action: str, 
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consult BenBot with confidence metrics and supporting evidence.
        
        Args:
            action: Action type ('trade_entry', 'trade_exit', 'risk_adjustment', etc.)
            data: Data for the consultation
            
        Returns:
            Response with decision and metadata
        """
        # Calculate our own confidence in the proposed action
        signal_confidence = self._calculate_signal_confidence(action, data)
        
        # Gather supporting evidence for BenBot
        supporting_evidence = self._gather_supporting_evidence(action, data)
        
        # Enhance the data with confidence and evidence
        enhanced_data = {
            **data,
            'confidence': signal_confidence,
            'supporting_evidence': supporting_evidence
        }
        
        # Add original EvoTrader decision if available
        original_decision = data.get('decision', None)
        enhanced_data['original_decision'] = original_decision
        
        # Make BenBot request
        benbot_response = self._make_benbot_request(action, enhanced_data)
        
        # If BenBot response fails, fall back to our decision
        if not benbot_response or 'error' in benbot_response:
            logger.warning(f"BenBot consultation failed, using EvoTrader decision: {original_decision}")
            return {
                'decision': original_decision,
                'confidence': signal_confidence,
                'source': 'evotrader',
                'reason': 'BenBot API error or timeout',
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        # Get BenBot confidence and decision
        benbot_confidence = benbot_response.get('confidence', 0.5)
        benbot_decision = benbot_response.get('decision', original_decision)
        
        # Record confidence for this action type
        if action in self.confidence_history:
            self.confidence_history[action].append(benbot_confidence)
            # Keep history manageable
            if len(self.confidence_history[action]) > 100:
                self.confidence_history[action] = self.confidence_history[action][-100:]
        
        # Make final decision based on weighted confidence
        final_decision = self._weighted_decision(
            benbot_decision, benbot_confidence,
            original_decision, signal_confidence,
            action
        )
        
        # Prepare response
        response = {
            'decision': final_decision,
            'benbot_decision': benbot_decision,
            'benbot_confidence': benbot_confidence,
            'evotrader_decision': original_decision,
            'evotrader_confidence': signal_confidence,
            'reasoning': benbot_response.get('reasoning', ''),
            'source': 'benbot' if final_decision == benbot_decision else 'evotrader',
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Record the decision for tracking
        self._record_decision(action, response)
        
        return response
    
    def _weighted_decision(self, 
                         benbot_decision: Any, 
                         benbot_confidence: float,
                         evotrader_decision: Any, 
                         evotrader_confidence: float,
                         action_type: str) -> Any:
        """
        Make weighted decision based on confidence comparison.
        
        Args:
            benbot_decision: BenBot's decision
            benbot_confidence: BenBot's confidence (0-1)
            evotrader_decision: EvoTrader's original decision
            evotrader_confidence: EvoTrader's confidence (0-1)
            action_type: Type of action
            
        Returns:
            Final decision
        """
        # Check if decisions are already the same
        if benbot_decision == evotrader_decision:
            return benbot_decision
        
        # Get historical performance weight for BenBot
        benbot_weight = self._get_historical_performance_weight(action_type)
        
        # Calculate weighted confidences
        benbot_weighted = benbot_confidence * benbot_weight
        evotrader_weighted = evotrader_confidence * (1.0 - benbot_weight)
        
        # Choose decision with higher weighted confidence
        if benbot_weighted > evotrader_weighted:
            logger.info(f"Using BenBot decision ({benbot_confidence:.2f} * {benbot_weight:.2f} = {benbot_weighted:.2f}) "
                       f"over EvoTrader ({evotrader_confidence:.2f} * {1-benbot_weight:.2f} = {evotrader_weighted:.2f})")
            return benbot_decision
        else:
            logger.info(f"Using EvoTrader decision ({evotrader_confidence:.2f} * {1-benbot_weight:.2f} = {evotrader_weighted:.2f}) "
                       f"over BenBot ({benbot_confidence:.2f} * {benbot_weight:.2f} = {benbot_weighted:.2f})")
            return evotrader_decision
    
    def _get_historical_performance_weight(self, action_type: str) -> float:
        """
        Calculate weight for BenBot based on historical performance.
        
        Args:
            action_type: Type of action
            
        Returns:
            Weight (0-1) with higher values giving more weight to BenBot
        """
        # Default weight is 0.7 (favoring BenBot slightly)
        default_weight = 0.7
        
        # If we have outcomes recorded, adjust based on performance
        if action_type in self.confidence_history and len(self.confidence_history[action_type]) > 10:
            # This is a simplified approach - in production this would 
            # analyze actual outcome success rates
            # For now, we'll just get the average confidence as a proxy
            avg_confidence = sum(self.confidence_history[action_type]) / len(self.confidence_history[action_type])
            
            # Scale to 0.4-0.9 range based on confidence
            # Low confidence -> less weight (0.4), high confidence -> more weight (0.9)
            return 0.4 + (0.5 * avg_confidence)
        
        return default_weight
    
    def _calculate_signal_confidence(self, action: str, data: Dict[str, Any]) -> float:
        """
        Calculate confidence in EvoTrader's signal or action.
        
        Args:
            action: Action type
            data: Signal data
            
        Returns:
            Confidence score (0-1)
        """
        # Default to medium confidence
        confidence = 0.5
        
        # Adjust based on action type and available data
        if action == 'trade_entry':
            # For trade entries, consider signal strength and session quality
            signal_strength = data.get('signal_strength', 0.5)
            session_optimal = data.get('session_optimal', False)
            strategy_win_rate = data.get('strategy_win_rate', 0.5)
            
            # Combine factors
            confidence = (signal_strength * 0.4) + (0.3 if session_optimal else 0.0) + (strategy_win_rate * 0.3)
            confidence = min(max(confidence, 0.1), 0.9)  # Keep between 0.1-0.9
        
        elif action == 'trade_exit':
            # For exits, consider profit/loss and time in trade
            profit_target_ratio = data.get('profit_target_ratio', 0.0)
            stop_loss_ratio = data.get('stop_loss_ratio', 0.0)
            time_in_trade_ratio = data.get('time_in_trade_ratio', 0.5)
            
            if profit_target_ratio >= 1.0:
                # At or beyond profit target, high confidence
                confidence = 0.9
            elif stop_loss_ratio >= 1.0:
                # At or beyond stop loss, high confidence
                confidence = 0.9
            else:
                # Partial profit/loss, scale confidence
                confidence = 0.5 + (0.4 * max(profit_target_ratio, stop_loss_ratio))
        
        elif action == 'risk_adjustment':
            # For risk adjustment, consider drawdown utilization
            drawdown = data.get('current_drawdown', 0.0)
            max_drawdown = data.get('max_drawdown', 5.0)
            drawdown_ratio = drawdown / max_drawdown if max_drawdown > 0 else 0.0
            
            if drawdown_ratio > 0.8:
                # Near max drawdown, high confidence in reducing risk
                confidence = 0.9
            else:
                # Scale with drawdown ratio
                confidence = 0.5 + (0.4 * drawdown_ratio)
        
        elif action == 'session_evaluation':
            # For session checks, look at historical performance
            session = data.get('session', '')
            historical_optimality = data.get('historical_optimal_score', 0.5)
            pair = data.get('pair', '')
            
            if historical_optimality > 0.7:
                confidence = 0.8
            elif historical_optimality > 0.5:
                confidence = 0.6
            else:
                confidence = 0.4
        
        return confidence
    
    def _gather_supporting_evidence(self, action: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gather supporting evidence for BenBot consultation.
        
        Args:
            action: Action type
            data: Signal data
            
        Returns:
            List of evidence items
        """
        evidence = []
        
        # Add relevant evidence based on action type
        if action == 'trade_entry':
            # Technical indicators
            for indicator, value in data.items():
                if indicator in ['rsi', 'macd', 'ema_diff', 'atr', 'adx']:
                    evidence.append({
                        'type': 'technical_indicator',
                        'name': indicator,
                        'value': value
                    })
            
            # Session data
            if 'session' in data:
                evidence.append({
                    'type': 'session_info',
                    'session': data.get('session', ''),
                    'is_optimal': data.get('session_optimal', False)
                })
            
            # Strategy performance
            if 'strategy_id' in data:
                evidence.append({
                    'type': 'strategy_performance',
                    'strategy_id': data.get('strategy_id', ''),
                    'win_rate': data.get('strategy_win_rate', 0.0),
                    'avg_pips': data.get('strategy_avg_pips', 0.0)
                })
            
            # Market condition
            if 'market_condition' in data:
                evidence.append({
                    'type': 'market_condition',
                    'condition': data.get('market_condition', ''),
                    'volatility': data.get('volatility', 0.0)
                })
        
        elif action == 'trade_exit':
            # Current trade metrics
            evidence.append({
                'type': 'trade_metrics',
                'profit_loss_pips': data.get('profit_loss_pips', 0.0),
                'time_in_trade': data.get('time_in_trade', ''),
                'profit_target_ratio': data.get('profit_target_ratio', 0.0),
                'stop_loss_ratio': data.get('stop_loss_ratio', 0.0)
            })
            
            # Exit reason
            if 'exit_reason' in data:
                evidence.append({
                    'type': 'exit_reason',
                    'reason': data.get('exit_reason', '')
                })
        
        elif action == 'risk_adjustment':
            # Account metrics
            evidence.append({
                'type': 'account_metrics',
                'current_drawdown': data.get('current_drawdown', 0.0),
                'daily_pnl': data.get('daily_pnl', 0.0),
                'open_risk': data.get('open_risk', 0.0)
            })
            
            # Adjustment recommendation
            if 'recommended_adjustment' in data:
                evidence.append({
                    'type': 'adjustment_recommendation',
                    'adjustment': data.get('recommended_adjustment', '')
                })
        
        return evidence
    
    def _make_benbot_request(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request to BenBot.
        
        Args:
            action: Action type
            data: Request data
            
        Returns:
            BenBot response
        """
        endpoint = f"{self.benbot_endpoint}/{action}"
        headers = {
            'Content-Type': 'application/json',
            **self.api_headers
        }
        
        # Prepare request data
        request_data = {
            'action': action,
            'timestamp': datetime.datetime.now().isoformat(),
            'data': data
        }
        
        # Add retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=request_data,
                    timeout=self.api_timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"BenBot API error: {response.status_code} - {response.text}")
                    # Maybe more serious error - wait longer
                    time.sleep(self.retry_delay * 2)
            
            except requests.RequestException as e:
                logger.warning(f"BenBot API request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        # If we get here, all attempts failed
        logger.error(f"All BenBot API attempts failed for action: {action}")
        return {'error': 'API request failed', 'action': action}
    
    def _record_decision(self, action: str, response: Dict[str, Any]) -> None:
        """
        Record decision for tracking and analysis.
        
        Args:
            action: Action type
            response: Decision response
        """
        # Add to in-memory history
        self.decision_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            **response
        })
        
        # Keep history manageable
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        # Record to database if enabled
        if self.use_db:
            self._save_decision_to_db(action, response)
    
    def _save_decision_to_db(self, action: str, response: Dict[str, Any]) -> None:
        """
        Save decision to database.
        
        Args:
            action: Action type
            response: Decision response
        """
        if not self.use_db:
            return
        
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.decision_db_path)
            cursor = conn.cursor()
            
            # Convert evidence to JSON string
            evidence_str = json.dumps(response.get('supporting_evidence', []))
            
            cursor.execute('''
            INSERT INTO benbot_decisions
            (timestamp, action_type, decision, confidence, evidence, 
             evotrader_decision, evotrader_confidence, final_decision, outcome, outcome_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                action,
                response.get('benbot_decision', ''),
                response.get('benbot_confidence', 0.0),
                evidence_str,
                response.get('evotrader_decision', ''),
                response.get('evotrader_confidence', 0.0),
                response.get('decision', ''),
                'pending',  # Outcome will be updated later
                0.0  # Outcome score will be updated later
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving decision to database: {e}")
    
    def update_decision_outcome(self, decision_id: str, outcome: str, score: float) -> bool:
        """
        Update outcome of a previous decision.
        
        Args:
            decision_id: Decision identifier
            outcome: Outcome descriptor ('success', 'failure', etc.)
            score: Numeric score for the outcome (-1 to 1)
            
        Returns:
            Success status
        """
        if not self.use_db:
            logger.warning("Decision tracking database not enabled, can't update outcome")
            return False
        
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.decision_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE benbot_decisions
            SET outcome = ?, outcome_score = ?
            WHERE id = ?
            ''', (outcome, score, decision_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated decision {decision_id} with outcome: {outcome}, score: {score}")
            return True
        except Exception as e:
            logger.error(f"Error updating decision outcome: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for BenBot decisions.
        
        Returns:
            Performance statistics
        """
        if not self.use_db:
            logger.warning("Decision tracking database not enabled, can't get performance stats")
            return {}
        
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.decision_db_path)
            cursor = conn.cursor()
            
            # Get overall stats
            cursor.execute('''
            SELECT 
                action_type,
                COUNT(*) as total_decisions,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                AVG(outcome_score) as avg_score,
                AVG(benbot_confidence) as avg_confidence
            FROM benbot_decisions
            WHERE outcome != 'pending'
            GROUP BY action_type
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            # Format results
            stats = {
                'overall': {
                    'total_decisions': 0,
                    'success_rate': 0.0,
                    'avg_score': 0.0,
                    'avg_confidence': 0.0
                },
                'by_action': {}
            }
            
            total_decisions = 0
            total_successes = 0
            total_score = 0.0
            total_confidence = 0.0
            
            for row in results:
                action, count, successes, avg_score, avg_confidence = row
                success_rate = successes / count if count > 0 else 0.0
                
                stats['by_action'][action] = {
                    'total_decisions': count,
                    'success_rate': success_rate,
                    'avg_score': avg_score,
                    'avg_confidence': avg_confidence
                }
                
                total_decisions += count
                total_successes += successes
                total_score += avg_score * count
                total_confidence += avg_confidence * count
            
            # Calculate overall stats
            if total_decisions > 0:
                stats['overall'] = {
                    'total_decisions': total_decisions,
                    'success_rate': total_successes / total_decisions,
                    'avg_score': total_score / total_decisions,
                    'avg_confidence': total_confidence / total_decisions
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}


# Mock BenBot server for testing
class MockBenBotServer:
    """Simple mock server for testing BenBot integration."""
    
    def handle_request(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a mock BenBot request."""
        # Create response based on action type
        if action == 'trade_entry':
            original_decision = data.get('original_decision', False)
            pair = data.get('pair', 'EURUSD')
            
            # Mock BenBot logic (usually agrees with 70% confidence)
            if random.random() < 0.7:
                return {
                    'decision': original_decision,
                    'confidence': random.uniform(0.7, 0.9),
                    'reasoning': f"BenBot approves entry for {pair} based on technical alignment"
                }
            else:
                return {
                    'decision': not original_decision,
                    'confidence': random.uniform(0.6, 0.8),
                    'reasoning': f"BenBot rejects entry for {pair} due to unfavorable conditions"
                }
        
        elif action == 'trade_exit':
            original_decision = data.get('original_decision', False)
            profit_target_ratio = data.get('profit_target_ratio', 0.0)
            
            if profit_target_ratio > 0.8:
                # High agreement on taking profit
                return {
                    'decision': True,
                    'confidence': 0.9,
                    'reasoning': "BenBot confirms exit due to profit target proximity"
                }
            else:
                # More mixed on early exits
                return {
                    'decision': original_decision,
                    'confidence': 0.6,
                    'reasoning': "BenBot provides moderate confidence assessment on exit"
                }
        
        # Default response for other actions
        return {
            'decision': data.get('original_decision', None),
            'confidence': 0.5,
            'reasoning': "BenBot provides neutral assessment"
        }


# Test function
def test_smart_benbot():
    """Test the BenBot connector functionality."""
    # Create mock server for testing
    mock_server = MockBenBotServer()
    
    # Create connector with a function that redirects to mock server
    connector = SmartBenBotConnector(
        benbot_endpoint="http://mock-benbot-server",
        config={'api_timeout': 1}
    )
    
    # Override the _make_benbot_request method to use mock server
    connector._make_benbot_request = lambda action, data: mock_server.handle_request(action, data)
    
    # Test trade entry consultation
    entry_response = connector.consult_benbot_with_confidence(
        'trade_entry',
        {
            'pair': 'EURUSD',
            'decision': True,
            'signal_strength': 0.75,
            'session_optimal': True,
            'strategy_win_rate': 0.6,
            'rsi': 70,
            'macd': 0.0025
        }
    )
    
    print("\nTrade Entry Consultation:")
    print(f"Decision: {entry_response['decision']}")
    print(f"Source: {entry_response['source']}")
    print(f"BenBot confidence: {entry_response.get('benbot_confidence', 0):.2f}")
    print(f"EvoTrader confidence: {entry_response.get('evotrader_confidence', 0):.2f}")
    
    # Test trade exit consultation
    exit_response = connector.consult_benbot_with_confidence(
        'trade_exit',
        {
            'pair': 'GBPUSD',
            'decision': True,
            'profit_loss_pips': 15.5,
            'time_in_trade': '2h 15m',
            'profit_target_ratio': 0.85,
            'stop_loss_ratio': 0.0,
            'exit_reason': 'approaching_profit_target'
        }
    )
    
    print("\nTrade Exit Consultation:")
    print(f"Decision: {exit_response['decision']}")
    print(f"Source: {exit_response['source']}")
    print(f"Reasoning: {exit_response.get('reasoning', '')}")
    
    return "Smart BenBot tests completed"


if __name__ == "__main__":
    test_smart_benbot()
