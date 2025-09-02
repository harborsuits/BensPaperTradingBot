# Enhanced Strategy Prioritizer

## Overview

The Enhanced Strategy Prioritizer is an advanced AI-driven system that leverages language models to evaluate market conditions and optimize strategy selection. It provides contextual awareness, explainability, and robust risk management to improve trading decisions.

## Key Components

### 1. EnhancedStrategyPrioritizer

The core component that uses language models to analyze market data and recommend strategy allocations. It includes:

- **Data Fusion**: Integrates multiple data sources including technical indicators, sentiment analysis, macroeconomic data, and alternative data.
- **Multi-timeframe Analysis**: Considers short-term, medium-term, and long-term market trends.
- **Contextual Memory**: Maintains a history of past decisions and outcomes to provide context for future recommendations.
- **Explainability**: Provides detailed reasoning for allocation decisions through chain-of-thought explanations.
- **Risk Guardrails**: Implements circuit breakers and constraints to ensure recommendations stay within safe bounds.
- **Performance Feedback Loop**: Tracks outcomes of allocation decisions to improve future recommendations.

### 2. PrioritizerIntegration

A utility class that simplifies integration between the Enhanced Strategy Prioritizer and other components of the trading system.

### 3. Demo Application

The `demo_enhanced_prioritizer.py` script provides a live demonstration of the Enhanced Strategy Prioritizer in action, with:

- Real-time visualization of strategy allocations
- Explanations and reasoning for decisions
- Risk level monitoring and warnings
- Simulated performance metrics

## Usage

### Basic Usage

```python
from trading_bot.ai_scoring.enhanced_strategy_prioritizer import EnhancedStrategyPrioritizer

# Define strategies
strategies = ["momentum", "trend_following", "mean_reversion", "volatility_breakout"]

# Create prioritizer
prioritizer = EnhancedStrategyPrioritizer(
    strategies=strategies,
    api_key="your_api_key",  # Optional, defaults to OPENAI_API_KEY env var
    use_mock=False,          # Set to True to use mock responses
    enable_sentiment_data=True,
    enable_macro_data=True
)

# Get strategy allocations with explanation
result = prioritizer.get_strategy_allocation()

# Access allocations and explanation
allocations = result["allocations"]
explanation = result.get("explanation", "")
reasoning = result.get("reasoning", [])
risk_warnings = result.get("risk_warnings", [])

# Record performance feedback
prioritizer.record_performance_feedback(
    allocations=allocations,
    performance_metrics={
        "return": 0.05,
        "sharpe_ratio": 1.2,
        "drawdown": -0.02,
        "win_rate": 65
    }
)
```

### Integration with Strategy Rotator

```python
from trading_bot.ai_scoring.prioritizer_integration import PrioritizerIntegration
from trading_bot.ai_scoring.strategy_rotator import StrategyRotator

# Create strategy rotator
rotator = StrategyRotator()

# Create integration
integration = PrioritizerIntegration(
    strategies=["momentum", "trend_following", "mean_reversion"],
    strategy_rotator=rotator,
    use_mock=False
)

# Get allocations and apply to rotator
allocations = integration.get_allocations(apply_to_rotator=True)

# Get annotated allocations with explanation
annotated = integration.get_annotated_allocations()
```

### Running the Demo

```bash
# Run with mock responses (no API key needed)
python demo_enhanced_prioritizer.py --use-mock

# Run with live API (requires OPENAI_API_KEY env var)
python demo_enhanced_prioritizer.py

# Customize update interval
python demo_enhanced_prioritizer.py --update-interval 30

# Disable market simulation
python demo_enhanced_prioritizer.py --no-simulation
```

## Configuration

The Enhanced Strategy Prioritizer can be configured with:

- **API Settings**: API key, base URL, model name
- **Data Sources**: Enable/disable sentiment data, macroeconomic data, alternative data
- **Risk Parameters**: Volatility thresholds, drawdown thresholds, allocation constraints
- **Memory Settings**: Max entries, file path for persistence
- **Cache Settings**: Cache duration, cache directory

## Requirements

- Python 3.8+
- OpenAI API key or compatible provider
- Required packages: `numpy`, `pandas`, `requests`, `rich` (for demo visualization)

## Advanced Features

### Creating Custom Risk Guardrails

```python
from trading_bot.ai_scoring.enhanced_strategy_prioritizer import RiskGuardrails

# Create custom risk guardrails
custom_risk_config = {
    "volatility_thresholds": {
        "elevated": 20.0,  # Custom VIX threshold for elevated risk
        "high": 30.0,      # Custom VIX threshold for high risk
        "critical": 40.0   # Custom VIX threshold for critical risk
    },
    "strategy_constraints": {
        "momentum": {
            "min_allocation": 5.0,
            "max_allocation": 30.0,
        }
    }
}

# Initialize prioritizer with custom risk config
prioritizer = EnhancedStrategyPrioritizer(
    strategies=strategies,
    risk_config=custom_risk_config
)
```

### Custom Feedback Handlers

```python
def log_allocation_performance(feedback_entry):
    """Custom handler for allocation performance feedback."""
    print(f"Feedback received at {feedback_entry['timestamp']}")
    print(f"Allocations: {feedback_entry['allocations']}")
    print(f"Performance: {feedback_entry['performance']}")

# Initialize prioritizer with custom feedback handler
prioritizer = EnhancedStrategyPrioritizer(
    strategies=strategies,
    feedback_handlers=[log_allocation_performance]
)
``` 