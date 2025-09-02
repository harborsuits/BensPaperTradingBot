"""
GPT-4 Trade Scorer for evaluating trading signals.

This module provides the core functionality for evaluating trade signals using GPT-4.
It takes trade data, market context, and strategy performance, then asks GPT-4
to evaluate whether the trade should be executed.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List

# Import utilities for LLM communication
from trading_bot.utils.llm_client import LLMClient

# Import for notification integration (if available)
try:
    from trading_bot.ai_scoring.trade_evaluation_notifier import TradeEvaluationNotifier
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False


class TradeScorer:
    """
    Evaluates trading signals using GPT-4 to provide a confidence score
    and recommendation based on current market conditions.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        notifier: Optional['TradeEvaluationNotifier'] = None
    ):
        """
        Initialize the trade scorer.
        
        Args:
            llm_client: Client for GPT-4 API calls
            prompt_path: Path to the prompt template file
            config: Configuration parameters
            notifier: Optional TradeEvaluationNotifier for sending notifications
        """
        self.llm = llm_client
        self.logger = logging.getLogger("TradeScorer")
        self.notifier = notifier
        
        # Default configuration
        self.config = {
            'confidence_threshold': 6.0,
            'temperature': 0.4,
            'max_tokens': 800,
            'log_prompts': True,
            'log_responses': True,
            'cache_results': True,
            'auto_notify': False  # Whether to automatically send notifications
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        # Load the prompt template
        self.prompt_template = self._load_prompt_template(prompt_path)
        
        # Initialize cache if enabled
        self.cache = {} if self.config['cache_results'] else None
        
        self.logger.info("Trade Scorer initialized")
    
    def _load_prompt_template(self, prompt_path: Optional[str] = None) -> str:
        """
        Load the prompt template from file.
        
        Args:
            prompt_path: Path to the prompt template file
            
        Returns:
            Prompt template string
        """
        # Use default path if not provided
        if not prompt_path:
            prompt_path = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "trade_eval_prompt.txt"
            )
        
        try:
            with open(prompt_path, 'r') as f:
                template = f.read()
                self.logger.info(f"Loaded prompt template from {prompt_path}")
                return template
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {str(e)}")
            # Provide a minimal default template
            return """
            You are a professional trading analyst. Evaluate this trade based on:
            - Current market conditions
            - Recent news
            - Strategy performance history

            ### TRADE:
            {trade}

            ### MARKET CONTEXT:
            {context}

            ### STRATEGY PERFORMANCE:
            {strategy}

            Return a JSON response with:
            {
                "confidence_score": float,
                "bias_alignment": "None | Weak | Moderate | Strong",
                "reasoning": "...",
                "recommendation": "Proceed with trade | Reduce size | Skip"
            }
            """
    
    def score_trade(
        self,
        trade_data: Dict[str, Any],
        context_data: Dict[str, Any],
        strategy_perf: Dict[str, Any],
        notify: bool = None
    ) -> Dict[str, Any]:
        """
        Evaluates a trade using GPT-4 and returns a structured recommendation.
        
        Args:
            trade_data: Details of the trade signal
            context_data: Current market context and conditions
            strategy_perf: Historical performance of the strategy
            notify: Whether to send notification (overrides auto_notify config)
            
        Returns:
            Dictionary with evaluation results
        """
        # Check cache first if enabled
        if self.cache is not None:
            cache_key = self._generate_cache_key(trade_data, context_data, strategy_perf)
            if cache_key in self.cache:
                self.logger.info(f"Using cached result for {trade_data.get('symbol', 'unknown')}")
                evaluation = self.cache[cache_key]
                # Even if using cached result, still send notification if requested
                self._handle_notification(evaluation, trade_data, context_data, strategy_perf, notify)
                return evaluation
        
        # Format the prompt
        prompt = self.prompt_template.format(
            trade=json.dumps(trade_data, indent=2),
            context=json.dumps(context_data, indent=2),
            strategy=json.dumps(strategy_perf, indent=2)
        )
        
        # Log the prompt if enabled
        if self.config['log_prompts']:
            self.logger.debug(f"Prompt for {trade_data.get('symbol', 'unknown')}:\n{prompt}")
        
        try:
            # Query the LLM
            response = self.llm.query(
                prompt=prompt,
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            
            # Log the response if enabled
            if self.config['log_responses']:
                self.logger.debug(f"Response for {trade_data.get('symbol', 'unknown')}:\n{response}")
            
            # Parse the response as JSON
            try:
                evaluation = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse LLM response as JSON: {response}")
                # Extract JSON from response text if possible
                import re
                json_match = re.search(r'({.*})', response.replace('\n', ''), re.DOTALL)
                if json_match:
                    try:
                        evaluation = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        self.logger.error("Failed to extract JSON from response")
                        evaluation = self._create_error_evaluation("Invalid response format")
                else:
                    evaluation = self._create_error_evaluation("Invalid response format")
            
            # Add 'should_execute' based on confidence threshold
            confidence = evaluation.get('confidence_score', 0)
            evaluation['should_execute'] = confidence >= self.config['confidence_threshold']
            
            # Add metadata
            evaluation['timestamp'] = self.llm.last_query_time or "unknown"
            evaluation['model'] = getattr(self.llm, 'model', "unknown")
            
            # Cache the result if enabled
            if self.cache is not None:
                self.cache[cache_key] = evaluation
            
            # Send notification if configured
            self._handle_notification(evaluation, trade_data, context_data, strategy_perf, notify)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error scoring trade: {str(e)}")
            return self._create_error_evaluation(str(e))
    
    def _generate_cache_key(
        self,
        trade_data: Dict[str, Any],
        context_data: Dict[str, Any],
        strategy_perf: Dict[str, Any]
    ) -> str:
        """Generate a cache key for the trade evaluation."""
        # Create a simplified representation of the input data
        key_data = {
            'symbol': trade_data.get('symbol'),
            'strategy': trade_data.get('strategy_name'),
            'entry': trade_data.get('entry'),
            'stop': trade_data.get('stop'),
            'target': trade_data.get('target'),
            'direction': trade_data.get('direction'),
            'market_regime': context_data.get('market_regime'),
            'timestamp': context_data.get('timestamp', '')
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _create_error_evaluation(self, error_message: str) -> Dict[str, Any]:
        """Create an evaluation result for error cases."""
        return {
            'confidence_score': 0.0,
            'bias_alignment': 'None',
            'reasoning': f"Error: {error_message}",
            'recommendation': 'Skip',
            'should_execute': False,
            'error': True,
            'error_message': error_message
        }
    
    def should_execute_trade(self, evaluation: Dict[str, Any]) -> bool:
        """
        Determine if a trade should be executed based on the evaluation.
        
        Args:
            evaluation: The evaluation result from score_trade
            
        Returns:
            Boolean indicating if the trade should be executed
        """
        # Check for errors
        if evaluation.get('error', False):
            return False
        
        # Check recommendation
        recommendation = evaluation.get('recommendation', '').lower()
        if 'skip' in recommendation:
            return False
        
        # Check confidence against threshold
        confidence = evaluation.get('confidence_score', 0.0)
        if confidence < self.config['confidence_threshold']:
            return False
        
        # If all checks pass, return True (or the precalculated value)
        return evaluation.get('should_execute', True)
    
    def log_evaluation(
        self,
        evaluation: Dict[str, Any],
        should_execute: bool,
        file_path: Optional[str] = None
    ) -> None:
        """
        Log the evaluation result to a file for record keeping.
        
        Args:
            evaluation: The evaluation result
            should_execute: Whether the trade should be executed
            file_path: Path to the log file (optional)
        """
        if not file_path:
            # Use default path in user's home directory
            home_dir = os.path.expanduser("~")
            log_dir = os.path.join(home_dir, ".trading_bot", "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, "trade_evaluations.jsonl")
        
        # Add execution decision to evaluation
        evaluation_copy = evaluation.copy()
        evaluation_copy['execution_decision'] = should_execute
        
        # Append to log file
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(evaluation_copy) + '\n')
            self.logger.debug(f"Evaluation logged to {file_path}")
        except Exception as e:
            self.logger.error(f"Error logging evaluation: {str(e)}")
    
    def set_notifier(self, notifier: 'TradeEvaluationNotifier') -> None:
        """
        Set or update the notifier for sending trade evaluations.
        
        Args:
            notifier: TradeEvaluationNotifier instance
        """
        self.notifier = notifier
        self.logger.info("Trade evaluation notifier set")
    
    def _handle_notification(
        self,
        evaluation: Dict[str, Any],
        trade_data: Dict[str, Any],
        context_data: Dict[str, Any],
        strategy_perf: Dict[str, Any],
        notify: Optional[bool] = None
    ) -> None:
        """
        Handle sending notifications for trade evaluations.
        
        Args:
            evaluation: The evaluation result
            trade_data: Trade data used for evaluation
            context_data: Market context data
            strategy_perf: Strategy performance data
            notify: Override for notification setting
        """
        # Determine if notification should be sent
        should_notify = notify if notify is not None else self.config['auto_notify']
        
        if should_notify and self.notifier and NOTIFIER_AVAILABLE:
            try:
                self.logger.debug(f"Sending notification for {trade_data.get('symbol', 'unknown')}")
                self.notifier.send_evaluation_notification(
                    evaluation=evaluation,
                    trade_data=trade_data,
                    context_data=context_data,
                    strategy_perf=strategy_perf
                )
                
                # If trade should be executed, also send execution notification
                if self.should_execute_trade(evaluation):
                    self.notifier.send_trade_execution_notification(
                        trade_data=trade_data,
                        evaluation=evaluation
                    )
            except Exception as e:
                self.logger.error(f"Error sending notification: {str(e)}") 