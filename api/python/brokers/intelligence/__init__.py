"""
Broker Intelligence Package

This package provides intelligent advisory systems that analyze broker 
performance metrics and provide recommendations to the orchestrator
without interfering with its decision-making authority.
"""

from trading_bot.brokers.intelligence.broker_advisor import (
    BrokerAdvisor,
    BrokerSelectionAdvice,
    BrokerSelectionFactor
)

__all__ = [
    'BrokerAdvisor',
    'BrokerSelectionAdvice',
    'BrokerSelectionFactor'
]
