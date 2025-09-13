"""
Reasoning Engine for LLM-Enhanced Trading

This module connects LLM insights with ML models to make trading decisions,
providing regime-aware reasoning, consensus mechanisms, and decision explanations.
"""

import os
import json
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import internal modules
from .financial_llm_engine import FinancialLLMEngine, LLMProvider, LLMResponse
from .prompt_engineering import PromptManager, PromptTemplate
from .memory_system import MemorySystem, MemoryType

# Initialize logger
logger = logging.getLogger("reasoning_engine")

class ReasoningTask(Enum):
    """Types of reasoning tasks"""
    REGIME_CONFIRMATION = "regime_confirmation"
    SIGNAL_VALIDATION = "signal_validation"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    RISK_ASSESSMENT = "risk_assessment"
    NEWS_IMPACT = "news_impact"
    ANOMALY_DETECTION = "anomaly_detection"
    MARKET_HYPOTHESIS = "market_hypothesis"
    
@dataclass
class ReasoningResult:
    """Result of a reasoning operation"""
    task: ReasoningTask
    conclusion: str
    confidence: float  # 0.0 to 1.0
    explanation: str
    ml_signals: Dict[str, Any]
    llm_signals: Dict[str, Any]
    consensus_score: float  # -1.0 to 1.0 (disagreement to agreement)
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task": self.task.value,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "ml_signals": self.ml_signals,
            "llm_signals": self.llm_signals,
            "consensus_score": self.consensus_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        date_str = datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M")
        return (f"ReasoningResult({self.task.value}, conclusion='{self.conclusion}', "
                f"confidence={self.confidence:.2f}, consensus={self.consensus_score:.2f}, "
                f"time={date_str})")

class ReasoningEngine:
    """
    Engine for LLM-enhanced reasoning about market conditions and trading decisions
    
    Features:
    - Integration between ML signals and LLM analysis
    - Consensus mechanisms for decision validation
    - Regime-specific reasoning approaches
    - Explanation generation for decisions
    """
    
    def __init__(
        self,
        llm_engine: Optional[FinancialLLMEngine] = None,
        prompt_manager: Optional[PromptManager] = None,
        memory_system: Optional[MemorySystem] = None,
        ml_regime_detector = None,  # Will be the MLRegimeDetector
        confidence_threshold: float = 0.65,
        reasoning_history_size: int = 100,
        debug: bool = False
    ):
        """
        Initialize the reasoning engine
        
        Args:
            llm_engine: LLM engine for generating reasoning
            prompt_manager: Prompt manager for LLM interactions
            memory_system: Memory system for context
            ml_regime_detector: ML-based market regime detector
            confidence_threshold: Minimum confidence to act on a decision
            reasoning_history_size: Size of reasoning history to maintain
            debug: Enable debug logging
        """
        # Set up logging
        logging_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=logging_level)
        self.debug = debug
        
        # Save components
        self.llm_engine = llm_engine
        self.prompt_manager = prompt_manager
        self.memory_system = memory_system
        self.ml_regime_detector = ml_regime_detector
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.reasoning_history_size = reasoning_history_size
        
        # Initialize history
        self.reasoning_history = []
        
        logger.info("Reasoning engine initialized")
        
    def _get_current_regime(self) -> Tuple[str, float]:
        """Get current market regime from detector"""
        if not self.ml_regime_detector:
            return "unknown", 0.0
            
        try:
            # Call regime detector - need to adjust based on actual implementation
            regime = self.ml_regime_detector.current_regime
            confidence = self.ml_regime_detector.regime_confidence
            return regime, confidence
        except Exception as e:
            logger.error(f"Error getting current regime: {e}")
            return "unknown", 0.0
            
    def _extract_llm_signals(self, llm_response: str, task: ReasoningTask) -> Dict[str, Any]:
        """Extract structured signals from LLM response"""
        signals = {}
        
        try:
            # Extract sentiment if present
            sentiment_match = re.search(r"sentiment[:\s]+([a-zA-Z]+)", llm_response, re.IGNORECASE)
            if sentiment_match:
                signals["sentiment"] = sentiment_match.group(1).lower()
                
            # Extract confidence
            confidence_match = re.search(r"confidence[:\s]+(\d+(/\d+)?|[0-9]*\.[0-9]+)", 
                                       llm_response, re.IGNORECASE)
            if confidence_match:
                conf_str = confidence_match.group(1)
                if "/" in conf_str:  # Handle "8/10" format
                    num, denom = conf_str.split("/")
                    signals["stated_confidence"] = float(num) / float(denom)
                else:
                    signals["stated_confidence"] = float(conf_str)
                    
            # Extract regime if present
            regime_match = re.search(r"regime[:\s]+([a-zA-Z\s]+)", llm_response, re.IGNORECASE)
            if regime_match:
                signals["regime"] = regime_match.group(1).strip().lower()
                
            # Task-specific extraction
            if task == ReasoningTask.RISK_ASSESSMENT:
                risk_match = re.search(r"risk[:\s]+([a-zA-Z]+)", llm_response, re.IGNORECASE)
                if risk_match:
                    signals["risk_level"] = risk_match.group(1).lower()
            
            elif task == ReasoningTask.NEWS_IMPACT:
                impact_match = re.search(r"impact[:\s]+([a-zA-Z]+)", llm_response, re.IGNORECASE)
                if impact_match:
                    signals["impact"] = impact_match.group(1).lower()
                    
            # Extract any numeric predictions
            predictions = {}
            pred_matches = re.findall(r"([a-zA-Z\s]+):\s*([+-]?[0-9]*\.?[0-9]+%?)", llm_response)
            for label, value in pred_matches:
                clean_label = label.strip().lower().replace(" ", "_")
                clean_value = value.strip()
                if "%" in clean_value:
                    clean_value = float(clean_value.replace("%", "")) / 100.0
                else:
                    try:
                        clean_value = float(clean_value)
                    except ValueError:
                        continue
                predictions[clean_label] = clean_value
                
            if predictions:
                signals["predictions"] = predictions
                
        except Exception as e:
            logger.error(f"Error extracting LLM signals: {e}")
            
        return signals
        
    def _calculate_consensus(
        self, 
        ml_signals: Dict[str, Any], 
        llm_signals: Dict[str, Any],
        task: ReasoningTask
    ) -> float:
        """
        Calculate consensus score between ML and LLM signals
        
        Returns:
            float: -1.0 to 1.0 (complete disagreement to complete agreement)
        """
        # Default neutral consensus
        consensus = 0.0
        
        try:
            # Different calculation methods based on task
            if task == ReasoningTask.REGIME_CONFIRMATION:
                # Check if regime names match or are similar
                ml_regime = ml_signals.get("regime", "").lower()
                llm_regime = llm_signals.get("regime", "").lower()
                
                if ml_regime and llm_regime:
                    if ml_regime == llm_regime:
                        consensus = 1.0
                    elif (("bull" in ml_regime and "bull" in llm_regime) or
                          ("bear" in ml_regime and "bear" in llm_regime) or
                          ("sideways" in ml_regime and "sideways" in llm_regime) or
                          ("volatile" in ml_regime and "volatile" in llm_regime)):
                        consensus = 0.5
                    else:
                        consensus = -0.5
                        
            elif task == ReasoningTask.SIGNAL_VALIDATION:
                # Compare directional signals
                ml_direction = ml_signals.get("direction", "").lower()
                llm_sentiment = llm_signals.get("sentiment", "").lower()
                
                direction_map = {
                    "buy": ["bullish", "positive"],
                    "sell": ["bearish", "negative"],
                    "hold": ["neutral", "mixed"]
                }
                
                for ml_dir, llm_sentiments in direction_map.items():
                    if ml_direction == ml_dir and llm_sentiment in llm_sentiments:
                        consensus = 1.0
                        break
                    elif ml_direction == ml_dir and any(s in llm_sentiment for s in llm_sentiments):
                        consensus = 0.5
                        break
                    elif ml_direction != ml_dir and llm_sentiment in llm_sentiments:
                        consensus = -0.7
                        break
                
            elif task == ReasoningTask.RISK_ASSESSMENT:
                # Compare risk levels
                ml_risk = ml_signals.get("risk_level", "").lower()
                llm_risk = llm_signals.get("risk_level", "").lower()
                
                if ml_risk and llm_risk:
                    risk_levels = ["low", "moderate", "high", "extreme"]
                    if ml_risk == llm_risk:
                        consensus = 1.0
                    else:
                        ml_idx = risk_levels.index(ml_risk) if ml_risk in risk_levels else -1
                        llm_idx = risk_levels.index(llm_risk) if llm_risk in risk_levels else -1
                        
                        if ml_idx >= 0 and llm_idx >= 0:
                            # Calculate distance between risk assessments
                            distance = abs(ml_idx - llm_idx) / len(risk_levels)
                            consensus = 1.0 - (2.0 * distance)  # Convert to -1 to 1 scale
            
            # Add more task-specific consensus calculations as needed
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            
        return consensus
            
    def _generate_explanation(
        self, 
        llm_response: str, 
        ml_signals: Dict[str, Any],
        llm_signals: Dict[str, Any],
        consensus_score: float,
        task: ReasoningTask
    ) -> str:
        """Generate a concise explanation of the reasoning process"""
        try:
            # Extract a concise explanation from the LLM response
            # First look for a conclusion or summary section
            conclusion_match = re.search(r"(?:conclusion|summary):(.*?)(?:\n\n|\n#|\Z)", 
                                       llm_response, re.IGNORECASE | re.DOTALL)
            
            if conclusion_match:
                explanation = conclusion_match.group(1).strip()
            else:
                # Take the first paragraph as a fallback
                paragraphs = [p for p in llm_response.split("\n\n") if p.strip()]
                explanation = paragraphs[0] if paragraphs else "No explanation available."
                
            # Add consensus information
            if consensus_score > 0.7:
                consensus_text = "ML models and LLM analysis strongly agree."
            elif consensus_score > 0.3:
                consensus_text = "ML models and LLM analysis generally agree."
            elif consensus_score > -0.3:
                consensus_text = "Mixed signals between ML models and LLM analysis."
            elif consensus_score > -0.7:
                consensus_text = "Some disagreement between ML models and LLM analysis."
            else:
                consensus_text = "Strong disagreement between ML models and LLM analysis."
                
            # Combine and format
            explanation = f"{explanation}\n\nConsensus: {consensus_text}"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Unable to generate explanation due to an error."
            
    def _save_to_memory(self, result: ReasoningResult):
        """Save reasoning result to memory system"""
        if not self.memory_system:
            return
            
        try:
            # Determine memory type based on importance
            if result.confidence > 0.8 and abs(result.consensus_score) > 0.8:
                memory_type = MemoryType.MEDIUM_TERM
                importance = 0.8
            else:
                memory_type = MemoryType.SHORT_TERM
                importance = 0.6
                
            # Create content with conclusion and explanation
            content = f"Task: {result.task.value}\nConclusion: {result.conclusion}\n\n{result.explanation}"
            
            # Create tags
            tags = [result.task.value, "reasoning"]
            
            # Add symbols if present
            if "symbols" in result.metadata:
                tags.extend(result.metadata["symbols"])
            if "symbol" in result.metadata:
                tags.append(result.metadata["symbol"])
                
            # Add to memory
            self.memory_system.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                source="reasoning_engine",
                metadata=result.to_dict()
            )
            
            logger.info(f"Saved reasoning result to memory: {result.task.value}")
            
        except Exception as e:
            logger.error(f"Error saving to memory: {e}")
    
    async def reason(
        self,
        task: ReasoningTask,
        ml_signals: Dict[str, Any],
        context: Dict[str, Any],
        save_to_memory: bool = True
    ) -> ReasoningResult:
        """
        Perform reasoning task combining ML signals with LLM analysis
        
        Args:
            task: Type of reasoning task to perform
            ml_signals: Signals and data from ML models
            context: Additional context for the reasoning
            save_to_memory: Whether to save result to memory
            
        Returns:
            ReasoningResult object
        """
        if not self.llm_engine or not self.prompt_manager:
            raise ValueError("LLM engine and prompt manager required for reasoning")
            
        # Map reasoning task to prompt template
        task_to_template = {
            ReasoningTask.REGIME_CONFIRMATION: PromptTemplate.REGIME_DETECTION,
            ReasoningTask.SIGNAL_VALIDATION: PromptTemplate.MARKET_ANALYSIS,
            ReasoningTask.STRATEGY_ADJUSTMENT: PromptTemplate.STRATEGY_ADJUSTMENT,
            ReasoningTask.RISK_ASSESSMENT: PromptTemplate.RISK_ASSESSMENT,
            ReasoningTask.NEWS_IMPACT: PromptTemplate.NEWS_ANALYSIS,
            ReasoningTask.ANOMALY_DETECTION: PromptTemplate.MARKET_ANALYSIS,
            ReasoningTask.MARKET_HYPOTHESIS: PromptTemplate.MARKET_ANALYSIS
        }
        
        template = task_to_template.get(task)
        if not template:
            raise ValueError(f"No template mapping for task {task.value}")
            
        # Get current market regime
        current_regime, regime_confidence = self._get_current_regime()
        
        # Add regime and ML signals to context
        prompt_context = context.copy()
        prompt_context["current_regime"] = current_regime
        prompt_context["regime_confidence"] = f"{regime_confidence:.2f}"
        prompt_context["ml_signals"] = json.dumps(ml_signals, indent=2)
        
        # Format appropriate prompt
        prompt = self.prompt_manager.generate_prompt(template, prompt_context)
        
        # Generate LLM response
        llm_response = await self.llm_engine.generate(
            prompt=prompt,
            system_message=f"You are analyzing {task.value} for trading decisions."
        )
        
        # Extract signals from LLM response
        llm_signals = self._extract_llm_signals(llm_response.text, task)
        
        # Calculate consensus
        consensus_score = self._calculate_consensus(ml_signals, llm_signals, task)
        
        # Determine confidence based on LLM confidence and consensus
        llm_confidence = llm_signals.get("stated_confidence", 0.5)
        ml_confidence = ml_signals.get("confidence", 0.5)
        
        # Weight confidence based on consensus - if high agreement, average confidences
        # If low agreement, reduce confidence
        if consensus_score > 0.3:
            confidence = (llm_confidence + ml_confidence) / 2.0
        else:
            confidence = min(llm_confidence, ml_confidence) * (1.0 + consensus_score)
            confidence = max(0.1, confidence)  # Ensure minimum confidence
            
        # Extract primary conclusion
        conclusion = llm_signals.get("conclusion", "No clear conclusion")
        
        # Generate explanation
        explanation = self._generate_explanation(
            llm_response.text,
            ml_signals,
            llm_signals,
            consensus_score,
            task
        )
        
        # Create result
        result = ReasoningResult(
            task=task,
            conclusion=conclusion,
            confidence=confidence,
            explanation=explanation,
            ml_signals=ml_signals,
            llm_signals=llm_signals,
            consensus_score=consensus_score,
            timestamp=datetime.datetime.now().timestamp(),
            metadata={
                "context": context,
                "regime": current_regime,
                "llm_model": llm_response.model,
                "llm_provider": llm_response.provider.value
            }
        )
        
        # Add to history
        self.reasoning_history.append(result)
        if len(self.reasoning_history) > self.reasoning_history_size:
            self.reasoning_history.pop(0)
            
        # Save to memory if enabled
        if save_to_memory:
            self._save_to_memory(result)
            
        logger.info(f"Completed reasoning task: {task.value} (confidence: {confidence:.2f})")
        return result
        
    def reason_sync(
        self,
        task: ReasoningTask,
        ml_signals: Dict[str, Any],
        context: Dict[str, Any],
        save_to_memory: bool = True
    ) -> ReasoningResult:
        """Synchronous version of reason"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.reason(
                    task=task,
                    ml_signals=ml_signals,
                    context=context,
                    save_to_memory=save_to_memory
                )
            )
        finally:
            loop.close()
            
    def get_recent_reasoning(
        self,
        task: Optional[ReasoningTask] = None,
        limit: int = 10
    ) -> List[ReasoningResult]:
        """Get recent reasoning results, optionally filtered by task"""
        if task:
            results = [r for r in self.reasoning_history if r.task == task]
        else:
            results = self.reasoning_history.copy()
            
        # Sort by timestamp (newest first)
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]
        
    def get_reasoning_trends(
        self,
        task: ReasoningTask,
        window: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze trends in reasoning results for a specific task
        
        Args:
            task: Task to analyze trends for
            window: Number of recent results to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        recent = self.get_recent_reasoning(task, window)
        if not recent:
            return {"error": "No reasoning results available"}
            
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in recent) / len(recent)
        
        # Calculate average consensus
        avg_consensus = sum(r.consensus_score for r in recent) / len(recent)
        
        # Track conclusion changes
        conclusions = [r.conclusion for r in recent]
        conclusion_stable = len(set(conclusions)) == 1
        
        # Create trend report
        return {
            "task": task.value,
            "sample_size": len(recent),
            "avg_confidence": avg_confidence,
            "avg_consensus": avg_consensus,
            "confidence_trend": "stable" if abs(recent[0].confidence - avg_confidence) < 0.1 else 
                               ("increasing" if recent[0].confidence > avg_confidence else "decreasing"),
            "consensus_trend": "stable" if abs(recent[0].consensus_score - avg_consensus) < 0.1 else
                              ("increasing" if recent[0].consensus_score > avg_consensus else "decreasing"),
            "conclusion_stable": conclusion_stable,
            "latest_conclusion": recent[0].conclusion if recent else None,
            "timestamp": datetime.datetime.now().timestamp()
        }
        
    def override_decision(
        self,
        task: ReasoningTask,
        original_result: ReasoningResult,
        override_reason: str,
        new_conclusion: str
    ) -> ReasoningResult:
        """
        Create an overridden version of a reasoning result
        
        Args:
            task: Reasoning task
            original_result: Original result to override
            override_reason: Reason for the override
            new_conclusion: New conclusion
            
        Returns:
            New reasoning result with override information
        """
        # Create new result based on original
        new_result = ReasoningResult(
            task=original_result.task,
            conclusion=new_conclusion,
            confidence=0.9,  # High confidence for manual override
            explanation=f"MANUAL OVERRIDE: {override_reason}\n\nOriginal: {original_result.explanation}",
            ml_signals=original_result.ml_signals,
            llm_signals=original_result.llm_signals,
            consensus_score=original_result.consensus_score,
            timestamp=datetime.datetime.now().timestamp(),
            metadata={
                **original_result.metadata,
                "override": True,
                "override_reason": override_reason,
                "original_conclusion": original_result.conclusion
            }
        )
        
        # Add to history
        self.reasoning_history.append(new_result)
        if len(self.reasoning_history) > self.reasoning_history_size:
            self.reasoning_history.pop(0)
            
        # Save to memory if available
        if self.memory_system:
            content = f"Override Applied: {override_reason}\nNew Conclusion: {new_conclusion}\nOriginal Conclusion: {original_result.conclusion}"
            
            self.memory_system.add_memory(
                content=content,
                memory_type=MemoryType.MEDIUM_TERM,  # Higher importance for human overrides
                importance=0.9,
                tags=[task.value, "override", "human_insight"],
                source="human_override",
                metadata=new_result.to_dict()
            )
            
        logger.info(f"Created decision override for {task.value}: {new_conclusion}")
        return new_result

# Import error for type hints only
try:
    import re
except ImportError:
    pass
