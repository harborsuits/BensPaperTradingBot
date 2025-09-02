# BensBot Trading System

Welcome to the BensBot Trading System documentation. This comprehensive documentation covers all aspects of the trading system, from configuration to development guidelines.

## Overview

BensBot is a professional-grade algorithmic trading system designed for:

- **Automated Strategy Execution** - Run trading strategies with customizable risk parameters
- **Backtesting & Optimization** - Test strategies against historical data
- **Real-time Market Intelligence** - Access market data, news, and AI-powered analysis
- **Risk Management** - Enforce strict risk controls across all trading activities

## Key Features

- **Robust Typed Configuration** - Type-safe, validated settings with environment variable integration
- **Multi-broker Support** - Seamless operation with Tradier, Alpaca, and other brokers
- **Strategy Rotation** - Dynamic allocation across multiple trading strategies
- **Real-time News Integration** - Live feeds from multiple financial news sources
- **API Access** - RESTful API for external access to all system capabilities
- **AI-Powered Analysis** - Integration with multiple AI model providers

## Getting Started

1. [Configuration](configuration.md) - Set up your config files and environment variables
2. [API Keys](api-keys.md) - Configure access to market data, news, and AI services 
3. [Risk Management](risk-management.md) - Understand and configure risk parameters
4. [Strategy System](strategy-system.md) - Learn about available strategies and customization

## Architecture

The system is built on a modular architecture with the following core components:

- **Typed Settings System** - Centralized configuration with validation
- **Core Orchestrator** - Coordinates all trading operations
- **Strategy Framework** - Pluggable strategies with standard interfaces
- **Risk Management** - Pre and post-trade risk enforcement
- **Data Management** - Market data acquisition and caching
- **Broker Adapters** - Communication with trading platforms
- **API Layer** - External access and integrations

## Demo

To run a quick demo of the backtester:

```bash
python -m trading_bot.backtesting.unified_backtester --strategy=momentum --symbol=AAPL --days=90
```
