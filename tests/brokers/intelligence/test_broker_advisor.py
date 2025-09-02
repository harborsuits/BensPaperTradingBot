#!/usr/bin/env python3
"""
Unit Tests for Broker Advisor

Tests the broker intelligence advisor to ensure proper scoring,
recommendation generation, and circuit breaker functionality.
"""

import os
import sys
import unittest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from trading_bot.brokers.metrics.base import MetricType, MetricOperation, MetricPeriod
from trading_bot.brokers.intelligence.broker_advisor import (
    BrokerAdvisor,
    BrokerSelectionAdvice,
    BrokerSelectionFactor
)


class TestBrokerAdvisor(unittest.TestCase):
    """Test suite for the BrokerAdvisor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock metrics manager
        self.metrics_manager = Mock()
        
        # Create broker advisor with test configuration
        self.config = {
            "factor_weights": {
                "latency": 0.20,
                "reliability": 0.40,
                "execution_quality": 0.25,
                "cost": 0.15
            },
            "circuit_breaker_thresholds": {
                "error_count": 5,
                "error_rate": 0.3,
                "availability_min": 90.0,
                "reset_after_seconds": 5  # Short time for testing
            },
            "asset_class_weights": {
                "forex": {
                    "latency": 0.30,
                    "reliability": 0.40,
                    "execution_quality": 0.20,
                    "cost": 0.10
                }
            }
        }
        
        self.advisor = BrokerAdvisor(
            metrics_manager=self.metrics_manager,
            config=self.config
        )
        
        # Set up sample broker capabilities
        self.advisor.register_broker_capability(
            broker_id="broker_a",
            asset_classes=["equities", "forex"],
            operation_types=["order", "quote", "data"]
        )
        
        self.advisor.register_broker_capability(
            broker_id="broker_b",
            asset_classes=["equities", "options"],
            operation_types=["order", "quote"]
        )
        
        self.advisor.register_broker_capability(
            broker_id="broker_c",
            asset_classes=["forex", "crypto"],
            operation_types=["order", "quote", "data"]
        )
    
    def test_broker_capability_registration(self):
        """Test broker capability registration"""
        # Check if brokers were registered correctly
        self.assertIn("broker_a", self.advisor.broker_capabilities)
        self.assertIn("broker_b", self.advisor.broker_capabilities)
        self.assertIn("broker_c", self.advisor.broker_capabilities)
        
        # Check capabilities for broker_a
        capabilities = self.advisor.broker_capabilities["broker_a"]
        self.assertIn("equities", capabilities["asset_classes"])
        self.assertIn("forex", capabilities["asset_classes"])
        self.assertIn("order", capabilities["operation_types"])
        self.assertIn("quote", capabilities["operation_types"])
        self.assertIn("data", capabilities["operation_types"])
    
    def test_circuit_breaker_trip_and_reset(self):
        """Test circuit breaker tripping and automatic reset"""
        # Initially, no circuit breaker should be active
        self.assertFalse(self.advisor.is_circuit_breaker_active("broker_a"))
        
        # Trip circuit breaker
        self.advisor.trip_circuit_breaker(
            broker_id="broker_a",
            reason="Test reason",
            reset_after_seconds=1  # Short time for testing
        )
        
        # Check if circuit breaker is active
        self.assertTrue(self.advisor.is_circuit_breaker_active("broker_a"))
        
        # Check circuit breaker state
        self.assertIn("broker_a", self.advisor.circuit_breakers)
        breaker = self.advisor.circuit_breakers["broker_a"]
        self.assertTrue(breaker["active"])
        self.assertEqual(breaker["reason"], "Test reason")
        
        # Wait for automatic reset
        time.sleep(1.5)  # Give a little extra time for the check
        
        # Check if circuit breaker was automatically reset
        self.assertFalse(self.advisor.is_circuit_breaker_active("broker_a"))
    
    def test_manual_circuit_breaker_reset(self):
        """Test manual circuit breaker reset"""
        # Trip circuit breaker
        self.advisor.trip_circuit_breaker(
            broker_id="broker_a",
            reason="Test reason",
            reset_after_seconds=60  # Long time
        )
        
        # Verify it's active
        self.assertTrue(self.advisor.is_circuit_breaker_active("broker_a"))
        
        # Manually reset
        self.advisor.reset_circuit_breaker("broker_a")
        
        # Verify it's inactive
        self.assertFalse(self.advisor.is_circuit_breaker_active("broker_a"))
    
    @patch('trading_bot.brokers.intelligence.broker_advisor.BrokerAdvisor.get_broker_performance_score')
    def test_get_selection_advice(self, mock_get_score):
        """Test broker selection advice generation"""
        # Mock the get_broker_performance_score method
        mock_get_score.side_effect = lambda broker_id, asset_class, operation_type: {
            "broker_a": {
                "broker_id": "broker_a",
                "overall_score": 85.5,
                "factor_scores": {
                    "latency": 90.0,
                    "reliability": 85.0,
                    "execution_quality": 83.0,
                    "cost": 80.0,
                    "circuit_breaker": 100
                }
            },
            "broker_b": {
                "broker_id": "broker_b",
                "overall_score": 75.0,
                "factor_scores": {
                    "latency": 70.0,
                    "reliability": 80.0,
                    "execution_quality": 75.0,
                    "cost": 70.0,
                    "circuit_breaker": 100
                }
            },
            "broker_c": {
                "broker_id": "broker_c",
                "overall_score": 60.0,
                "factor_scores": {
                    "latency": 60.0,
                    "reliability": 65.0,
                    "execution_quality": 55.0,
                    "cost": 50.0,
                    "circuit_breaker": 100
                }
            }
        }[broker_id]
        
        # Get selection advice for equities/order
        advice = self.advisor.get_selection_advice(
            asset_class="equities",
            operation_type="order"
        )
        
        # Verify the advice
        self.assertEqual(advice.asset_class, "equities")
        self.assertEqual(advice.operation_type, "order")
        self.assertEqual(advice.primary_broker_id, "broker_a")
        self.assertIn("broker_b", advice.backup_broker_ids)
        self.assertEqual(len(advice.blacklisted_broker_ids), 0)
        
        # Verify the scores are passed through
        self.assertEqual(advice.priority_scores["broker_a"], 85.5)
        self.assertEqual(advice.priority_scores["broker_b"], 75.0)
        
        # Verify broker_c not included (doesn't support equities)
        self.assertNotIn("broker_c", advice.priority_scores)
    
    def test_broker_blacklisting(self):
        """Test that brokers with active circuit breakers are blacklisted"""
        # Trip circuit breaker for broker_a
        self.advisor.trip_circuit_breaker(
            broker_id="broker_a",
            reason="Test reason"
        )
        
        # Mock the performance scoring
        with patch('trading_bot.brokers.intelligence.broker_advisor.BrokerAdvisor.get_broker_performance_score') as mock_get_score:
            mock_get_score.side_effect = lambda broker_id, asset_class, operation_type: {
                "broker_a": {
                    "broker_id": "broker_a",
                    "overall_score": 85.5,
                    "factor_scores": {
                        "latency": 90.0,
                        "reliability": 85.0,
                        "execution_quality": 83.0,
                        "cost": 80.0,
                        "circuit_breaker": 0  # Circuit breaker active
                    }
                },
                "broker_b": {
                    "broker_id": "broker_b",
                    "overall_score": 75.0,
                    "factor_scores": {
                        "latency": 70.0,
                        "reliability": 80.0,
                        "execution_quality": 75.0,
                        "cost": 70.0,
                        "circuit_breaker": 100
                    }
                }
            }[broker_id]
            
            # Get selection advice for equities/order
            advice = self.advisor.get_selection_advice(
                asset_class="equities",
                operation_type="order"
            )
            
            # Verify broker_a is blacklisted
            self.assertIn("broker_a", advice.blacklisted_broker_ids)
            
            # Verify broker_b is primary (since broker_a is blacklisted)
            self.assertEqual(advice.primary_broker_id, "broker_b")
    
    def test_factor_score_calculation(self):
        """Test calculation of factor scores from metrics"""
        # Mock metrics data
        metrics = {
            "latency": {
                "mean_ms": 200  # 200ms average latency
            },
            "reliability": {
                "availability": 99.5,  # 99.5% availability
                "errors": 2  # 2 errors
            },
            "execution_quality": {
                "avg_slippage_pct": 0.05  # 0.05% slippage
            },
            "costs": {
                "avg_commission": 2.5  # $2.50 average commission
            }
        }
        
        # Set up metrics manager mock
        self.metrics_manager.get_broker_metrics.return_value = metrics
        
        # Calculate performance score
        performance = self.advisor.get_broker_performance_score(
            broker_id="broker_a",
            asset_class="equities",
            operation_type="order"
        )
        
        # Verify factor scores
        # Latency: 100 - (200/10) = 80
        self.assertAlmostEqual(performance["factor_scores"]["latency"], 80, delta=1)
        
        # Reliability: (99.5 * 0.75) + ((100 - (2 * 10)) * 0.25) = 74.625 + 20 = 94.625
        self.assertAlmostEqual(performance["factor_scores"]["reliability"], 94.625, delta=1)
        
        # Execution quality: 100 - (0.05 * 100) = 95
        self.assertAlmostEqual(performance["factor_scores"]["execution_quality"], 95, delta=1)
        
        # Cost: 100 - (2.5 * 10) = 75
        self.assertAlmostEqual(performance["factor_scores"]["cost"], 75, delta=1)
        
        # Overall: (80 * 0.2) + (94.625 * 0.4) + (95 * 0.25) + (75 * 0.15)
        # = 16 + 37.85 + 23.75 + 11.25 = 88.85
        self.assertAlmostEqual(performance["overall_score"], 88.85, delta=1)
    
    def test_asset_class_specific_weights(self):
        """Test asset class specific factor weights"""
        # Mock metrics data (simplified)
        metrics = {
            "latency": {"mean_ms": 100},
            "reliability": {"availability": 99, "errors": 1},
            "execution_quality": {"avg_slippage_pct": 0.1},
            "costs": {"avg_commission": 1.0}
        }
        
        # Set up metrics manager mock
        self.metrics_manager.get_broker_metrics.return_value = metrics
        
        # Calculate performance score for equities (standard weights)
        equities_performance = self.advisor.get_broker_performance_score(
            broker_id="broker_a",
            asset_class="equities",
            operation_type="order"
        )
        
        # Calculate performance score for forex (custom weights)
        forex_performance = self.advisor.get_broker_performance_score(
            broker_id="broker_a",
            asset_class="forex",
            operation_type="order"
        )
        
        # Verify different weights were used
        # Forex has higher latency weight (0.3 vs 0.2) and lower execution_quality weight (0.2 vs 0.25)
        self.assertNotEqual(equities_performance["overall_score"], forex_performance["overall_score"])
    
    def test_failover_recommendation(self):
        """Test failover recommendation logic"""
        # Mock the performance scoring
        with patch('trading_bot.brokers.intelligence.broker_advisor.BrokerAdvisor.get_broker_performance_score') as mock_get_score:
            mock_get_score.side_effect = lambda broker_id, asset_class, operation_type: {
                "broker_a": {
                    "broker_id": "broker_a",
                    "overall_score": 95.0,
                    "factor_scores": {
                        "latency": 95.0,
                        "reliability": 95.0,
                        "execution_quality": 95.0,
                        "cost": 95.0,
                        "circuit_breaker": 100
                    }
                },
                "broker_b": {
                    "broker_id": "broker_b",
                    "overall_score": 70.0,
                    "factor_scores": {
                        "latency": 70.0,
                        "reliability": 70.0,
                        "execution_quality": 70.0,
                        "cost": 70.0,
                        "circuit_breaker": 100
                    }
                }
            }[broker_id]
            
            # Get selection advice, specifying broker_b as current broker
            advice = self.advisor.get_selection_advice(
                asset_class="equities",
                operation_type="order",
                current_broker_id="broker_b"
            )
            
            # Verify failover is recommended (difference > 20)
            self.assertTrue(advice.is_failover_recommended)
            
            # Now test with a smaller difference
            mock_get_score.side_effect = lambda broker_id, asset_class, operation_type: {
                "broker_a": {
                    "broker_id": "broker_a",
                    "overall_score": 75.0,
                    "factor_scores": {
                        "latency": 75.0,
                        "reliability": 75.0,
                        "execution_quality": 75.0,
                        "cost": 75.0,
                        "circuit_breaker": 100
                    }
                },
                "broker_b": {
                    "broker_id": "broker_b",
                    "overall_score": 70.0,
                    "factor_scores": {
                        "latency": 70.0,
                        "reliability": 70.0,
                        "execution_quality": 70.0,
                        "cost": 70.0,
                        "circuit_breaker": 100
                    }
                }
            }[broker_id]
            
            # Get selection advice again
            advice = self.advisor.get_selection_advice(
                asset_class="equities",
                operation_type="order",
                current_broker_id="broker_b"
            )
            
            # Verify failover is not recommended (difference < 20)
            self.assertFalse(advice.is_failover_recommended)
    
    def test_selection_advice_serialization(self):
        """Test serialization and deserialization of selection advice"""
        # Create a sample selection advice
        advice = BrokerSelectionAdvice(
            asset_class="equities",
            operation_type="order",
            primary_broker_id="broker_a",
            backup_broker_ids=["broker_b", "broker_c"],
            blacklisted_broker_ids=["broker_d"],
            recommendation_factors={
                "broker_a": {
                    "latency": 90.0,
                    "reliability": 85.0
                }
            },
            priority_scores={
                "broker_a": 85.0,
                "broker_b": 75.0,
                "broker_c": 65.0
            },
            is_failover_recommended=True,
            advisory_notes=["Note 1", "Note 2"]
        )
        
        # Convert to dictionary
        advice_dict = advice.to_dict()
        
        # Create new advice from dictionary
        new_advice = BrokerSelectionAdvice.from_dict(advice_dict)
        
        # Verify values match
        self.assertEqual(new_advice.asset_class, "equities")
        self.assertEqual(new_advice.operation_type, "order")
        self.assertEqual(new_advice.primary_broker_id, "broker_a")
        self.assertEqual(new_advice.backup_broker_ids, ["broker_b", "broker_c"])
        self.assertEqual(new_advice.blacklisted_broker_ids, ["broker_d"])
        self.assertEqual(new_advice.priority_scores["broker_a"], 85.0)
        self.assertTrue(new_advice.is_failover_recommended)
        self.assertEqual(new_advice.advisory_notes, ["Note 1", "Note 2"])


if __name__ == '__main__':
    unittest.main()
