"""
Prompt Engineering Module for LLM-Enhanced Trading

This module manages prompt templates and generation for different trading scenarios,
optimizing LLM interactions for financial analysis and decision-making.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

# Initialize logger
logger = logging.getLogger("prompt_engineering")

class PromptTemplate(Enum):
    """Types of prompt templates for different trading scenarios"""
    MARKET_ANALYSIS = "market_analysis"
    REGIME_DETECTION = "regime_detection"
    NEWS_ANALYSIS = "news_analysis"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    EARNINGS_ANALYSIS = "earnings_analysis"
    ECONOMIC_CALENDAR = "economic_calendar"
    TECHNICAL_PATTERN = "technical_pattern"
    TRADING_REFLECTION = "trading_reflection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    COMPETITOR_ANALYSIS = "competitor_analysis"

class PromptManager:
    """
    Manages prompt templates and generation for financial LLM interactions
    
    Features:
    - Template management with variables
    - Context-aware prompt construction
    - Prompt optimization for different LLM providers
    - Memory integration
    """
    
    def __init__(
        self,
        templates_dir: Optional[str] = None,
        memory_system = None,
        enable_few_shot: bool = True,
        max_prompt_tokens: int = 4000,
        debug: bool = False
    ):
        """
        Initialize the prompt manager
        
        Args:
            templates_dir: Directory containing prompt templates
            memory_system: Optional MemorySystem for context retrieval
            enable_few_shot: Whether to include examples in prompts
            max_prompt_tokens: Maximum tokens for prompts
            debug: Enable debug logging
        """
        # Set up logging
        logging_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=logging_level)
        self.debug = debug
        
        # Configuration
        self.memory_system = memory_system
        self.enable_few_shot = enable_few_shot
        self.max_prompt_tokens = max_prompt_tokens
        
        # Load templates
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "llm_integration", "models", "prompts"
        )
        os.makedirs(self.templates_dir, exist_ok=True)
        
        self.templates = self._load_templates()
        self._create_default_templates()
        logger.info(f"Prompt manager initialized with {len(self.templates)} templates")
        
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load templates from directory"""
        templates = {}
        
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory {self.templates_dir} does not exist")
            return templates
            
        for template_type in PromptTemplate:
            template_path = os.path.join(self.templates_dir, f"{template_type.value}.json")
            if os.path.exists(template_path):
                try:
                    with open(template_path, "r") as f:
                        templates[template_type.value] = json.load(f)
                    logger.info(f"Loaded template: {template_type.value}")
                except Exception as e:
                    logger.error(f"Error loading template {template_type.value}: {e}")
        
        return templates
        
    def _save_template(self, template_type: PromptTemplate, template: Dict[str, Any]):
        """Save template to file"""
        template_path = os.path.join(self.templates_dir, f"{template_type.value}.json")
        try:
            with open(template_path, "w") as f:
                json.dump(template, f, indent=2)
            logger.info(f"Saved template: {template_type.value}")
        except Exception as e:
            logger.error(f"Error saving template {template_type.value}: {e}")
            
    def _create_default_templates(self):
        """Create default templates if they don't exist"""
        # Create default market analysis template
        if PromptTemplate.MARKET_ANALYSIS.value not in self.templates:
            self.templates[PromptTemplate.MARKET_ANALYSIS.value] = {
                "template": """
You are a professional financial analyst with expertise in market analysis.

Current date: {current_date}
Relevant Market Data:
{market_data}

Technical Indicators:
{technical_indicators}

Recent News Headlines:
{news_headlines}

Memory Context:
{memory_context}

Analyze the current market conditions and provide:
1. Overall market sentiment
2. Key drivers and catalysts
3. Risk factors
4. Sector rotation insights
5. Expected market direction in the short term
6. Actionable trading insights

Format your response in a professional, concise manner suitable for institutional traders.
                """,
                "variables": ["current_date", "market_data", "technical_indicators", 
                            "news_headlines", "memory_context"],
                "examples": [
                    {
                        "input": {
                            "current_date": "2025-04-24",
                            "market_data": "S&P 500: 4,731.28 (+0.45%)\nNASDAQ: 16,798.31 (+0.73%)\nDJI: 38,150.30 (+0.25%)",
                            "technical_indicators": "VIX: 15.23 (-0.89)\nS&P 500 RSI: 62.3 (Neutral)\nMACD: Positive crossover",
                            "news_headlines": "Fed maintains rates, signals caution on inflation\nTech earnings beat expectations\nOil prices stabilize amid Middle East tensions",
                            "memory_context": "Market experienced high volatility last week with rotation from growth to value"
                        },
                        "output": """
# Market Analysis - April 24, 2025

## Current Assessment
- **Overall Sentiment**: Moderately Bullish
- **Market Position**: Mid-cycle expansion phase

## Key Observations
- Broad market strength with growth sectors leading advances
- Risk appetite improving as VIX declines
- Healthy technical setup with positive MACD crossover

## Primary Drivers
1. Fed's accommodative stance maintaining liquidity
2. Strong tech earnings exceeding Wall Street expectations
3. Declining volatility encouraging institutional inflows

## Risk Factors
- Inflation concerns remain despite Fed's cautious outlook
- Geopolitical tensions could disrupt energy markets
- Valuations stretched in select technology names

## Sector Intelligence
- Technology: Momentum continues with earnings outperformance
- Energy: Stabilizing on supply concerns, monitor geopolitical developments
- Financials: Benefits from steepening yield curve

## Actionable Insights
- Maintain bullish positioning while implementing trailing stops
- Consider selective rotation into quality cyclicals
- Monitor Fed commentary for inflation signaling
"""
                    }
                ]
            }
            self._save_template(PromptTemplate.MARKET_ANALYSIS, 
                              self.templates[PromptTemplate.MARKET_ANALYSIS.value])
                              
        # Create default regime detection template
        if PromptTemplate.REGIME_DETECTION.value not in self.templates:
            self.templates[PromptTemplate.REGIME_DETECTION.value] = {
                "template": """
You are an expert market regime analyst specializing in identifying market regimes and regime shifts.

Current date: {current_date}

Market Indicators:
{market_indicators}

Volatility Metrics:
{volatility_metrics}

Correlation Data:
{correlation_data}

Sentiment Indicators:
{sentiment_indicators}

Macroeconomic Factors:
{macro_factors}

Historical Context:
{historical_context}

Given this information, analyze the current market regime. Specifically:
1. Identify the current market regime (e.g., bull market, bear market, sideways/range-bound, high volatility, low volatility)
2. Assess the strength and stability of the current regime
3. Identify early warning signs of potential regime shifts
4. Determine appropriate trading strategies for the current regime
5. Estimate confidence level in your regime assessment (1-10)

Respond in a professional, evidence-based format suitable for institutional traders.
                """,
                "variables": ["current_date", "market_indicators", "volatility_metrics", 
                            "correlation_data", "sentiment_indicators", "macro_factors",
                            "historical_context"],
                "examples": [
                    {
                        "input": {
                            "current_date": "2025-04-24",
                            "market_indicators": "S&P 500 200-day SMA: Positive slope\n50-day vs 200-day: 50-day above\nAdvance-Decline Line: Healthy breadth\nNew Highs vs New Lows: Positive ratio",
                            "volatility_metrics": "VIX: 14.75 (Low)\nVIX Term Structure: Contango\nRealised Volatility (20d): 10.3%\nImplied vs Realized Spread: +4.45%",
                            "correlation_data": "S&P Sectors Correlation: 0.42 (Below average)\nGlobal Market Correlation: 0.38 (Below average)\nStyle Factor Dispersion: High",
                            "sentiment_indicators": "AAII Bull-Bear: +15% (Bullish)\nPut-Call Ratio: 0.72 (Bullish sentiment)\nCNN Fear & Greed: 72 (Greed)",
                            "macro_factors": "Fed Funds Rate: 3.5%\nYield Curve (10Y-2Y): +0.45% (Positive)\nInflation (CPI): 2.8% YoY\nUnemployment: 3.9%",
                            "historical_context": "Market recovered from correction 2 months ago\nSector leadership has shifted to cyclicals\nValue has outperformed growth over last 60 days"
                        },
                        "output": """
# Market Regime Analysis - April 24, 2025

## Current Regime Assessment
**PRIMARY REGIME: Bull Market - Middle Stage** 
**Confidence Level: 8/10**

## Regime Characteristics
- **Momentum**: Strong positive with healthy breadth
- **Volatility**: Low and stabilizing (VIX in contango)
- **Correlations**: Below average, indicating healthy market dispersion
- **Sentiment**: Bullish but not at extreme levels
- **Macro Environment**: Supportive with positive yield curve and moderate inflation

## Regime Stability Analysis
The current bull market regime shows high stability with multiple confirming factors:
- Broad market participation (positive A/D line)
- Low cross-asset correlations enabling alpha generation
- Volatility term structure in contango suggesting market expects continued stability
- Fundamental backdrop remains supportive with manageable inflation

## Early Warning Indicators
Despite regime stability, these factors warrant monitoring:
- Sentiment approaching elevated levels (Fear & Greed at 72)
- Cyclical sector outperformance may be maturing
- Some momentum divergences in small caps (not yet critical)

## Recommended Trading Strategies
1. **Position Sizing**: Maintain full allocations with trailing stops
2. **Sector Focus**: Overweight cyclicals and financials
3. **Factor Exposure**: Tilt toward quality, momentum, and value
4. **Volatility Strategy**: Consider selling options premium (high implied vs realized spread)
5. **Risk Management**: Normal stop loss levels, avoid excessive leverage

## Regime Shift Probability
- Near-term regime shift (30 days): LOW (15%)
- Medium-term shift (90 days): MODERATE (35%)
- Most likely next regime: Late-stage bull or volatility expansion

## Key Monitoring Points
- Deterioration in market breadth would be first warning sign
- VIX term structure inversion would indicate volatility regime shift
- Fed policy change could accelerate transition to late-cycle dynamics
"""
                    }
                ]
            }
            self._save_template(PromptTemplate.REGIME_DETECTION, 
                              self.templates[PromptTemplate.REGIME_DETECTION.value])
    
        # Add other default templates as needed
        # For brevity, I'm not including all templates in this implementation
            
    def get_template(self, template_type: PromptTemplate) -> Optional[Dict[str, Any]]:
        """Get a template by type"""
        return self.templates.get(template_type.value)
        
    def add_template(self, template_type: PromptTemplate, template: str, 
                   variables: List[str], examples: Optional[List[Dict[str, Any]]] = None):
        """Add or update a template"""
        self.templates[template_type.value] = {
            "template": template,
            "variables": variables,
            "examples": examples or []
        }
        self._save_template(template_type, self.templates[template_type.value])
        
    def _get_relevant_memories(self, template_type: PromptTemplate, 
                             context: Dict[str, Any], limit: int = 3) -> str:
        """Get relevant memories for context"""
        if not self.memory_system:
            return ""
            
        # Define tags based on template type and context
        tags = [template_type.value]
        
        # Add symbol tags if available
        if "symbol" in context:
            tags.append(context["symbol"])
        if "symbols" in context:
            tags.extend(context["symbols"])
            
        # Add sector tags if available
        if "sector" in context:
            tags.append(context["sector"])
            
        # Query the memory system
        try:
            # First try with tags
            memories = self.memory_system.query_memories(
                tags=tags,
                limit=limit
            )
            
            # If no results, try a more general query
            if not memories and "query_text" in context:
                memories = self.memory_system.query_memories(
                    text_query=context["query_text"],
                    limit=limit
                )
                
            if not memories:
                return ""
                
            # Format memories as context
            memory_texts = []
            for memory in memories:
                timestamp = datetime.fromtimestamp(memory.created_at).strftime("%Y-%m-%d")
                memory_texts.append(f"[{timestamp}] {memory.content}")
                
            return "\n\n".join(memory_texts)
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return ""
            
    def _format_few_shot_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format few-shot examples for the prompt"""
        if not examples:
            return ""
            
        formatted_examples = []
        for i, example in enumerate(examples):
            input_section = "\n".join(f"{k}: {v}" for k, v in example["input"].items())
            formatted_examples.append(f"Example {i+1}:\n\nInput:\n{input_section}\n\nOutput:\n{example['output']}\n")
            
        return "\n".join(formatted_examples)
        
    def generate_prompt(
        self,
        template_type: PromptTemplate,
        context: Dict[str, Any],
        include_examples: Optional[bool] = None,
    ) -> str:
        """
        Generate a prompt from a template and context
        
        Args:
            template_type: Type of prompt template to use
            context: Dictionary with context variables
            include_examples: Whether to include few-shot examples
            
        Returns:
            Formatted prompt string
        """
        # Get template
        template_data = self.get_template(template_type)
        if not template_data:
            raise ValueError(f"Template {template_type.value} not found")
            
        template = template_data["template"]
        variables = template_data["variables"]
        examples = template_data["examples"]
        
        # Check if context has all required variables
        missing_vars = [var for var in variables if var not in context]
        
        # Add memory context if available and not provided
        if "memory_context" in variables and "memory_context" not in context and self.memory_system:
            context["memory_context"] = self._get_relevant_memories(template_type, context)
            
        # Fill in missing variables with defaults
        for var in missing_vars:
            if var == "current_date":
                context[var] = datetime.now().strftime("%Y-%m-%d")
            else:
                context[var] = f"[No {var} data available]"
                
        # Format the template with context
        prompt = template.format(**context)
        
        # Add few-shot examples if enabled
        include_examples = include_examples if include_examples is not None else self.enable_few_shot
        if include_examples and examples:
            examples_text = self._format_few_shot_examples(examples)
            prompt = f"{prompt}\n\nExamples of similar analyses:\n\n{examples_text}\n\nNow provide your analysis:"
            
        # Log prompt if debug
        if self.debug:
            logger.debug(f"Generated prompt for {template_type.value}:\n{prompt}")
            
        return prompt
        
    def register_prompt_result(
        self,
        template_type: PromptTemplate,
        context: Dict[str, Any],
        prompt: str,
        completion: str,
        quality_score: Optional[float] = None
    ):
        """
        Register a successful prompt completion for future optimization
        
        Args:
            template_type: Type of prompt used
            context: Context variables used
            prompt: The full prompt sent to the LLM
            completion: The LLM completion
            quality_score: Optional quality score (0-1)
        """
        if template_type.value not in self.templates:
            logger.warning(f"Cannot register result for unknown template: {template_type.value}")
            return
            
        # Update examples if quality is high
        if quality_score and quality_score > 0.8:
            template = self.templates[template_type.value]
            
            # Create a new example
            example = {
                "input": context,
                "output": completion
            }
            
            # Add to examples (limit to 5)
            template["examples"] = template["examples"][:4] + [example]
            
            # Save template
            self._save_template(template_type, template)
            logger.info(f"Added high-quality example to {template_type.value} template")

    def create_specialized_prompt(
        self, 
        base_template: PromptTemplate,
        specialization: str,
        additional_context: Dict[str, str]
    ) -> str:
        """
        Create a specialized version of a base prompt for specific scenarios
        
        Args:
            base_template: The base template to specialize
            specialization: Description of the specialization
            additional_context: Additional context to add
            
        Returns:
            Specialized prompt string
        """
        template_data = self.get_template(base_template)
        if not template_data:
            raise ValueError(f"Base template {base_template.value} not found")
            
        # Create context with current date
        context = {"current_date": datetime.now().strftime("%Y-%m-%d")}
        
        # Add additional context
        context.update(additional_context)
        
        # Get memory context
        if self.memory_system:
            context["memory_context"] = self._get_relevant_memories(base_template, context)
        
        # Generate base prompt
        base_prompt = self.generate_prompt(base_template, context, include_examples=False)
        
        # Add specialization
        specialized_prompt = f"{base_prompt}\n\nAdditional Focus:\n{specialization}"
        
        return specialized_prompt
